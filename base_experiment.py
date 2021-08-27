 
#%%
from WordVectors import WordVectors, load_wordvectors, intersection
from s4 import s4, threshold_crossvalidation
from printers import print_sense_output, print_shift_info
from alignment import align

from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score

from typing import Tuple, NamedTuple, List, Dict
from collections import defaultdict, namedtuple
import pathlib
import numpy as np
import pandas as pd
import re

Vector_Info = namedtuple('Vector_Info', ['corpus_name', 'description', 'type'])
Target_Info = namedtuple('Target', ['word', 'shifted_word', 'is_shifted'])
# Prediction_Info = namedtuple('Prediction', )

## Parse through the targets to match them up if 
def make_word_pairs(vector_type: str, vocab: List[str], targets: NamedTuple):
    word_pairs = []
    if vector_type in ['original', 'new']:
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

def get_aligned_dists(align_method: str, classify_methods: Dict[str, float], 
                      word_pairs: Tuple[str, str], wordvec1: WordVectors, wordvec2: WordVectors):
    
    wv1, wv2 = intersection(wordvec1, wordvec2)
    print("Size of common vocab.", len(wv1))

    ## Get landmarks
    print(f'Starting {align_method} aligning')
    if align_method == 'global':
        landmarks = wv1.words
    elif align_method == 's4':
        align_params = {
                "n_targets": 100,
                "n_negatives": 10,
                "rate": .25
            }

        landmarks, non_landmarks, Q = s4(wv1, wv2, cls_model="nn", verbose=0, **align_params)
        print('Done with S4 aligning')

    ## Align
    wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)

    ## Align the original so that it matches wv1_ but has its full vocab
    aligned_wv = WordVectors(words=wordvec1.words, vectors=np.dot(wordvec1.vectors, Q))
    anchored_wv = wordvec2

    ## If not sense, dists will be 1 to 1 for each target
    ## If there are senses it will be 1 to many for each target 
    target_dists = {}
    if 'cosine' in classify_methods:
        print('Starting cosine predicting')
        if classify_methods['cosine'] == 0:
            auto_params = {"rate": 1.5,
                        "n_fold": 1,
                        "n_targets": 50,
                        "n_negatives": 100}
            classify_methods['cosine'] = threshold_crossvalidation(wv1_, wv2_, iters=2, **auto_params, landmarks=landmarks)

        dists = []
        for align_word, anchor_word in word_pairs:
            dist = cosine(aligned_wv[align_word], anchored_wv[anchor_word])
            dists.append(dist)

        target_dists['cosine'] = np.array(dists) 

    if 's4' in classify_methods:
        print('Starting S4 predicting')
        cls_params = {'n_targets':1000, 'n_negatives':1000, 'rate':0.25}
        model = s4(wv1_, wv2_, landmarks=landmarks, verbose=0, **cls_params, update_landmarks=False)

        # Concatenate vectors of target words for prediction
        target_vectors = []
        for align_word, anchor_word in word_pairs:
            # x = np.array([np.concatenate((aligned_wv[align_word], anchored_wv[anchor_word]))])
            x = (aligned_wv[align_word], anchored_wv[anchor_word])
            target_vectors.append(np.concatenate(x))

        target_vectors = np.array(target_vectors)
        dists = model.predict(target_vectors).flatten()
        print(f'Target vector size {dists.shape}')

        target_dists['s4'] = dists

    return target_dists, classify_methods

def assess_standard_prediction(predictions, threshold: float, targets: List[NamedTuple]):
    target_words, _, true_labels = list(zip(*targets))    

    predictions = predictions > threshold
    accuracy = accuracy_score(true_labels, predictions)
    accuracies = {'Predicted' : accuracy}

    results = pd.DataFrame(data={'Words' : target_words, 'True Label' : true_labels, 'Predicted' : predictions})
    
    for label in ['True Label', 'Predicted']:
        results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

def make_sense_prediction(predictions, threshold, model, word_pairs):
    ## Go through each sense from the sense embedding (word.#) and find its distance to the word 
    ## connected to it in the anchored embedding
    ## This could be the same word, a translated word, etc
    prediction_info = defaultdict(list)
    for word_pair, dist in zip(word_pairs, predictions):
        sense_word, anchor_word = word_pair
        shift_prediction = int(dist > threshold)
        word_count = model.wv.get_vecattr(sense_word, "count")
        target = sense_word.split('.')[0]

        info = (sense_word, dist, word_count, shift_prediction)
        prediction_info[target].append(info)

    # TODO: this could be passed in maybe? eh
    shift_labels = ['Majority', 'Main', 'Weighted']
    shift_data = {shift : [] for shift in shift_labels}

    ratios = [.05, .1, .15, .2]
    ratio_data = defaultdict(list)

    for target, sense_predictions in prediction_info.items():
        
        shifted_count = 0
        unshifted_count = 0
        weighted_dist = 0
        shifts = []

        for sense, dist, count, shift_prediction in sense_predictions:
            weighted_dist += (dist * count)

            shifts.append(shift_prediction)
            if shift_prediction:
                shifted_count += count
            else:
                unshifted_count += count

        majority_cutoff = len(shifts) // 2
        is_shifted = sum(shifts) > majority_cutoff
        shift_data['Majority'].append(is_shifted)

        # TODO: may have to convert to bool?
        biggest_sense = max(sense_predictions, key=lambda t: t[2])
        is_shifted = biggest_sense[3]
        shift_data['Main'].append(is_shifted)

        weighted_dist /= (shifted_count + unshifted_count)
        is_shifted = weighted_dist > threshold
        shift_data['Weighted'].append(is_shifted)

        ratio = shifted_count / (shifted_count + unshifted_count)
        for ratio_cutoff in ratios:
            is_shifted = ratio >= ratio_cutoff
            ratio_cutoff = f'{int(ratio_cutoff*100)}%'
            ratio_data[ratio_cutoff].append(is_shifted)

    return prediction_info, shift_data, ratio_data

def assess_sense_prediction(shift_data, ratio_data, targets: List[NamedTuple]):
    
    target_words, _, true_labels = list(zip(*targets))    

    ## Assess predictions
    accuracies = {}
    for method, shift_pred in shift_data.items():
        accuracy = accuracy_score(true_labels, shift_pred)
        print(f'{method} accuracy: {accuracy:.2f}')
        accuracies[method] = accuracy

    for ratio_cutoff, shift_pred in ratio_data.items():
        accuracy = accuracy_score(true_labels, shift_pred)
        print(f'{ratio_cutoff} ratio accuracy: {accuracy:.2f}')
        accuracies[ratio_cutoff] = accuracy

    data = {'Words' : target_words, 'True Label' : true_labels}
    data.update(shift_data)
    data.update(ratio_data)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

# %%
## TODO: rename this
def temp(align_method: str, classify_methods: Dict[str, float], dataset_name: str, 
         align_vector: NamedTuple, anchor_vector: NamedTuple, targets: List[NamedTuple],
         num_loops: int = 1):

    wordvec1, wordvec2, model1 = load_wordvectors(dataset_name, align_vector, anchor_vector)

    remove_targets = []
    for index, target in enumerate(targets):
        if align_vector.type == 'sense':
            if f'{target.word}.0' not in wordvec1.words:
                print(f'{target.word}.0 missing from {align_vector.corpus_name}')
                remove_targets.append(index)
        else:
            if target.word not in wordvec1.words:
                print(f'{target.word} missing from {align_vector.corpus_name}')
                remove_targets.append(index)

        if target.shifted_word not in wordvec2.words:
            print(f'{target.shifted_word} ({target.word}) missing from {anchor_vector.corpus_name}')
            remove_targets.append(index)

    for index in sorted(remove_targets, reverse=True):
        # print(f'Deleting {targets[index].word}')
        del targets[index]

    print(f'Running test on {len(targets)} targets')

    word_pairs = make_word_pairs(align_vector.type, wordvec1.words, targets)

    # TODO: could import this
    best_accuracies = {clf_method : 0 for clf_method in classify_methods}
    all_accuracies = {clf_method : [] for clf_method in classify_methods}

    for i in range(num_loops):

        print(f'{i+1} / {num_loops}')

        dists, classify_methods = get_aligned_dists(align_method, classify_methods, 
                                                          word_pairs, wordvec1, wordvec2)

        for classify_method, threshold in classify_methods.items():
            if align_vector.type in ['original', 'new']: 
                accuracies, results = assess_standard_prediction(dists[classify_method], threshold, targets)

            elif align_vector.type == 'sense':
                prediction_info, shift_data, ratio_data = make_sense_prediction(dists[classify_method], threshold, model1, word_pairs)
                accuracies, results = assess_sense_prediction(shift_data, ratio_data, targets)
            
            all_accuracies[classify_method].append(accuracies)
            max_acc = max(accuracies.values())

            if max_acc > best_accuracies[classify_method]:
                print(f'Found new best accuracy {max_acc:.2f} for {classify_method}, so printing info')
                best_accuracies[classify_method] = max_acc

                path_out = f"results/{dataset_name}/{align_vector.type}/{align_method}_{classify_method}"
                pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)

                if align_vector.type == 'sense':
                    print_sense_output(prediction_info, results, threshold, path_out, align_vector, anchor_vector)
                
                print_shift_info(accuracies, results, path_out, align_method, classify_method, threshold, align_vector, anchor_vector)
                results.to_csv(f"{path_out}/labels.csv", index=False)

    print('Tests complete!')
    for classify_method, accuracy in best_accuracies.items():
        print(f'Best accuracy for {classify_method} was {accuracy:.2f}')

    return all_accuracies

if __name__=="__main__":
    print('')
