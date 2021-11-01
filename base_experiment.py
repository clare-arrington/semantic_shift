 
#%%
from temp.predictions import make_sense_prediction, assess_sense_prediction, assess_standard_prediction
from temp.shift import s4_m3, threshold_crossvalidation
from temp.wordvectors import VectorVariations, WordVectors, load_wordvectors, intersection, set_subtraction, extend_normal_with_sense
from temp.printers import print_sense_output, print_shift_info
from temp.alignment import align

from scipy.spatial.distance import cosine

from typing import Tuple, NamedTuple, List, Dict
from collections import namedtuple
import pathlib
import pickle
import numpy as np
import re

# Vector_Info = namedtuple('Vector_Info', ['corpus_name', 'description', 'type'])
Target_Info = namedtuple('Target', ['word', 'shifted_word', 'is_shifted'])

class Train_Method_Info:
    def __init__(self, name, params, threshold=None): 
        self.name = name
        self.params = params
        self.threshold = threshold

    def __repr__(self):
        return f"Train_Method_Info('{self.name}', {self.params}, {self.threshold})"

## Parse through the targets to match them up 
def make_word_pairs(
    vector_type: str, 
    vocab: List[str], 
    targets: NamedTuple):

    word_pairs = []
    if vector_type in ['original', 'new', 'both_sense']:
        for target in targets:
            word_pairs.append((target.word, target.shifted_word))

    else:
        sense_words = [word for word in vocab if '.' in word]
        print(f'{len(sense_words)} filtered senses\n')

        for target in targets:
            r = re.compile(f'{target.word}.[0-9]')
            for sense in filter(r.match, sense_words):
                word_pairs.append((sense, target.shifted_word))

    return word_pairs

## TODO: this modifies long term; fix that issue up
def remove_targets(
    targets: List[NamedTuple], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations):

    if align_wv.type != 'both_sense':
        remove_targets = []
        for index, target in enumerate(targets):
            if align_wv.type == 'sense':
                if f'{target.word}.0' not in align_wv.normal_vec.words:
                    print(f'{target.word}.0 missing from {align_wv.corpus_name}')
                    remove_targets.append(index)
            else:
                if target.word not in align_wv.normal_vec.words:
                    print(f'{target.word} missing from {align_wv.corpus_name}')
                    remove_targets.append(index)

            if target.shifted_word not in anchor_wv.normal_vec.words:
                print(f'{target.shifted_word} ({target.word}) missing from {anchor_wv.corpus_name}')
                remove_targets.append(index)

        for index in sorted(remove_targets, reverse=True):
            # print(f'Deleting {targets[index].word}')
            del targets[index]

    else:
        sense_targets = []
        all_vocab = set(align_wv.normal_vec.words + anchor_wv.normal_vec.words)
        sense_words = sorted([word for word in all_vocab if '.' in word])
        print("Removing words that aren't in one or both embeddings")
        for word in sense_words:
            if word not in align_wv.normal_vec.words:
                print(f'\t{word} : {align_wv.corpus_name}')

            elif word not in anchor_wv.normal_vec.words:
                print(f'\t{word} : {anchor_wv.corpus_name}')
            
            else:
                # TODO: label
                sense_targets.append(Target_Info(word, word, 0))

        targets = sense_targets

    print(f'\nRunning test on {len(targets)} targets')

    return targets

def align_vectors(
    align_method: Train_Method_Info, 
    word_pairs: Tuple[str, str], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations, 
    modify_wv='all'):

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    print("Size of common vocab:", len(wv1))

    ## Get landmarks
    print(f'Starting {align_method.name} aligning')
    if align_method.name == 'global':
        landmarks = list(wv1.word_id.values())

    elif align_method.name == 's4':
        extended_wv1, extended_wv2 = extend_normal_with_sense(
            wv1, wv2, align_wv, anchor_wv, word_pairs)
        print(f"Size of WV after senses added: {len(wv1)} -> {len(extended_wv1)}" )

        ## Align with subset of landmarks
        landmarks, non_landmarks, Q = s4_m3(
            wv1, wv2, extended_wv1, extended_wv2, **align_method.params)
        print('Done with S4 aligning')
        print(f"Check for any unwanted mutations: {len(wv1)}, {len(extended_wv1)}" )

        sense_landmarks = [extended_wv1.words[i] for i in landmarks if '.' in extended_wv1.words[i]]
        print(f"Senses in landmarks: {', '.join(sense_landmarks)}")
        ## Just reset to make sure these weren't modified by S4
        # wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
        # wv1, wv2 = extend_normal_with_sense(wv1, wv2, align_wv, anchor_wv, word_pairs)

        wv1 = extended_wv1
        wv2 = extended_wv2

    ## Align
    wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
    print("Size of aligned vocab:", len(wv1_))
    align_wv.partial_align_vec = wv1_
    anchor_wv.partial_align_vec = wv2_
    
    ## Align the original so that it matches wv1_ but has its full vocab
    align_wv.post_align_vec = WordVectors(
        words=align_wv.normal_vec.words, 
        vectors=np.dot(align_wv.normal_vec.vectors, Q))
    anchor_wv.post_align_vec = anchor_wv.normal_vec

    return landmarks, align_wv, anchor_wv

def get_target_distances(
        classify_methods: List[Train_Method_Info],
        word_pairs: Tuple[str, str], 
        landmarks,
        align_wv: VectorVariations, 
        anchor_wv: VectorVariations    
        ):

    ## If not sense, dists will be 1 to 1 for each target
    ## If there are senses it will be 1 to many for each target 
    target_dists = {}

    for method in classify_methods:

        if method.name == 'cosine':
            print('Starting cosine predicting')

            #if method.threshold == 0:
            method.threshold = threshold_crossvalidation(
                align_wv.partial_align_vec, anchor_wv.partial_align_vec, 
                **method.params, landmarks=landmarks)

            dists = []
            for align_word, anchor_word in word_pairs:
                dist = cosine(align_wv.post_align_vec[align_word], 
                              anchor_wv.post_align_vec[anchor_word])
                dists.append(dist)

            target_dists['cosine'] = np.array(dists) 

        if method.name == 's4':
            print('Starting S4 predicting')
            model = s4_m3(align_wv.partial_align_vec, 
                       align_wv.partial_align_vec, 
                       landmarks=landmarks, 
                       verbose=0, 
                       **method.params, 
                       update_landmarks=False)

            # Concatenate vectors of target words for prediction
            target_vectors = []
            for align_word, anchor_word in word_pairs:
                x = (align_wv.post_align_vec[align_word], 
                     anchor_wv.post_align_vec[anchor_word])
                target_vectors.append(np.concatenate(x))

            target_vectors = np.array(target_vectors)
            dists = model.predict(target_vectors).flatten()
            # print(f'Target vector size {dists.shape}')

            target_dists['s4'] = dists

    return target_dists, classify_methods

# %%
def predict_target_shift(
        align_method: Train_Method_Info, 
        classify_methods: List[Train_Method_Info],  
        dataset_name: str, targets: List[NamedTuple],
        align_wv: VectorVariations, anchor_wv: VectorVariations,
        output_path: str, num_loops):

    print('\n================================')
    print(f'Starting run for {align_method.name}, {align_wv.type}')

    ## Pull and prep vector data
    align_wv, anchor_wv = load_wordvectors(dataset_name, align_wv, anchor_wv)
    targets = remove_targets(targets, align_wv, anchor_wv)
    word_pairs = make_word_pairs(align_wv.type, align_wv.normal_vec.words, targets)

    # TODO: could import this if I wanted to not overwrite a previous best
    best_accuracies = {clf_method.name : 0 for clf_method in classify_methods}
    all_accuracies = {clf_method.name : [] for clf_method in classify_methods}

    for i in range(num_loops):
        print(f'{i+1} / {num_loops}')

        landmarks, align_wv, anchor_wv = align_vectors(
                                            align_method, word_pairs, 
                                            align_wv, anchor_wv)

        dists, classify_methods = get_target_distances(
                                    classify_methods, 
                                    word_pairs, landmarks, 
                                    align_wv, anchor_wv)

        for method in classify_methods:

            if align_wv.type in ['original', 'new']: 
                accuracies, results = assess_standard_prediction(
                    dists[method.name], method.threshold, targets)
            
            elif 'sense' in align_wv.type:
                prediction_info, shift_data, ratio_data = make_sense_prediction(
                    dists[method.name], method.threshold, 
                    align_wv.model, anchor_wv.model, word_pairs)

                accuracies, results = assess_sense_prediction(
                    shift_data, ratio_data, targets)
            
            all_accuracies[method.name].append(accuracies)
            max_acc = max(accuracies.values())

            if max_acc > best_accuracies[method.name]:
                print(f'Found new best accuracy {max_acc:.2f} for {method.name}, so printing info')
                best_accuracies[method.name] = max_acc

                path_out = f"{output_path}/{align_method.name}_{method.name}"
                pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

                if 'sense' in align_wv.type :
                    print_sense_output(
                        prediction_info, results, 
                        method.threshold, path_out, 
                        align_wv, anchor_wv)
                
                print_shift_info(
                    accuracies, results, 
                    path_out, align_method.name, 
                    method.name, method.threshold, 
                    align_wv, anchor_wv)
                    
                results.to_csv(f"{path_out}/labels.csv", index=False)

                with open(f'{output_path}/{align_method.name}_landmarks.dat' , 'wb') as pf:
                    landmarks = [align_wv.normal_vec.words[w] for w in landmarks]
                    pickle.dump(landmarks, pf)

    print('Tests complete!')
    for classify_method, accuracy in best_accuracies.items():
        print(f'Best accuracy for {classify_method} was {accuracy:.2f}')

    return all_accuracies

#%%
## TODO: add a sweep dict to pass the other params

## If we expect the embeddings to be similar, the number of targets would be lower.
## 
def align_param_sweep(
    dataset_name, targets, num_loops, output_path,
    align_wv, anchor_wv, align_method,
    classify_params, classify_method_thresholds,
    n_targets=[50, 100], # possible landmarks; amount shift is simulated on
    n_negatives=[50, 100], 
    rates=[.1, .25], # how much a word is shifted when simulated
    ):

        parameter_sweep = {}
        for n_target in n_targets:
            for n_negs in n_negatives:
                for rate in rates:

                    align_params = {"n_targets": n_target,
                                    "n_negatives": n_negs,
                                    "rate": rate
                                    }

                    print(f'\n\nRunning {n_target} targets, {n_negs} negatives, {rate}', end='')

                    parameter_sweep[(n_target, n_negs, rate)] = \
                            predict_target_shift(align_method, align_params, 
                                classify_method_thresholds, classify_params,
                                dataset_name, targets,
                                align_wv, anchor_wv, 
                                output_path, num_loops)

        return parameter_sweep

## TODO: these could be combined to one b/c the params are the same
## TODO: class sweeps could be done on the same alignment :/ maybe break it up so it can do that then sweep?
def classify_param_sweep(
    dataset_name, targets, num_loops, output_path,
    align_wv, anchor_wv, align_method,
    align_params, classify_method_thresholds,
    n_targets=[100, 250, 500, 750],
    n_negatives=[100, 250, 500, 750],
    rates=[.1, .25],
    ):

        parameter_sweep = {}
        for n_target in n_targets:
            for n_negs in n_negatives:
                for rate in rates:

                    classify_params = { "n_targets": n_target,
                                        "n_negatives": n_negs,
                                        "rate": rate
                                    }

                    print(f'\n\nRunning {n_target} targets, {n_negs} negatives, {rate}', end='')

                    parameter_sweep[(n_target, n_negs, rate)] = \
                            predict_target_shift(align_method, align_params, 
                                classify_method_thresholds, classify_params,
                                dataset_name, targets,
                                align_wv, anchor_wv, 
                                output_path, num_loops)

        return parameter_sweep

## align_info should be (name, desc)
def main(
    dataset_name:str, 
    targets: List[Target_Info],
    align_info: Tuple[str, str], 
    anchor_info: Tuple[str, str], 
    vector_types: List[str],
    align_methods: List[Train_Method_Info]=None, 
    classify_methods: List[Train_Method_Info]=None, 
    num_loops: int = 1):

    for vector_type in vector_types:
        for align_method in align_methods:
            if vector_type in ['original', 'new', 'both_sense']:
                align_type = vector_type
                anchor_type = vector_type
            
            elif vector_type == 'sense':
                align_type = vector_type
                anchor_type = 'new'

            align_wv = VectorVariations(corpus_name = align_info[0],
                                    desc = align_info[1], 
                                    type = align_type)
            anchor_wv = VectorVariations(corpus_name = anchor_info[0],
                                        desc = anchor_info[1], 
                                        type = anchor_type)

            output_path = f'/home/clare/Data/align_results/{dataset_name}/align_{align_info[0]}/{vector_type}'
            pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

            # if (align_params is None) and (align_method == 's4'):
            #     all_accs = align_param_sweep(
            #                 dataset_name, targets, num_loops, output_path,
            #                 align_wv, anchor_wv, align_method,
            #                 classify_params, classify_method_thresholds)
            #     save_file_name = f'{align_method}_align_param_sweep'

            # elif (classify_params is None) and ('s4' in classify_method_thresholds):
            #     all_accs = classify_param_sweep(
            #                 dataset_name, targets, num_loops, output_path,
            #                 align_wv, anchor_wv, align_method,
            #                 align_params, classify_method_thresholds)
            #     save_file_name = f'{align_method}_classify_param_sweep'

            # else:
            all_accs = predict_target_shift(
                            align_method, classify_methods,
                            dataset_name, targets,
                            align_wv, anchor_wv, 
                            output_path, num_loops)
            save_file_name = align_method.name
            
            print(f'Results will be saved to {output_path}')
            with open(f'{output_path}/{save_file_name}.dat' , 'wb') as pf:
                pickle.dump(all_accs, pf)


#%%
# Test stuff

# def test_stuff():
#     align_method = align_methods[0]
#     classify_method = classify_methods[0]
#     vector_type = 'sense'
#     align_type = vector_type
#     anchor_type = 'new'

#     align_wv = VectorVariations(corpus_name = align_info[0],
#                             desc = align_info[1], 
#                             type = align_type)
#     anchor_wv = VectorVariations(corpus_name = anchor_info[0],
#                                 desc = anchor_info[1], 
#                                 type = anchor_type)

#     output_path = f'/home/clare/Data/align_results/{dataset_name}/align_{align_info[0]}/{vector_type}'
#     print(output_path)
    
#     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

#     print('\n================================')
#     print(f'Starting run for {align_method.name}, {align_wv.type}')

#     ## Pull and prep vector data
#     align_wv, anchor_wv = load_wordvectors(dataset_name, align_wv, anchor_wv)
#     targets = remove_targets(targets, align_wv, anchor_wv)
#     word_pairs = make_word_pairs(align_wv.type, align_wv.normal_vec.words, targets)

#     ## Align Stuff
#     wv1 = align_wv.normal_vec
#     wv1, wv2 = intersection(wv1, anchor_wv.normal_vec)
#     print("Size of common vocab:", len(wv1))

#     # wv1, wv2 = extend_normal_with_sense(wv1, wv2, align_wv, anchor_wv, word_pairs)
#     # print("Size of extended vocab:", len(wv1))

#     s4_m3(wv1, wv2)