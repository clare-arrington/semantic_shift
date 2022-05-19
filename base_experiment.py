 
#%%
from shift_steps.predictions import get_prediction_info, assess_standard_prediction, \
                                    make_normal_prediction, assess_normal_prediction, \
                                    make_shared_prediction, assess_shared_prediction, \
                                    make_sense_prediction, assess_sense_prediction 
from shift_steps.target_prep import Target_Info, filter_targets, make_word_pairs
from shift_steps.wordvectors import VectorVariations, load_wordvectors
from shift_steps.printers import print_single_sense_output, print_both_sense_output, \
                                print_shared_sense_output, \
                                print_shift_info, save_landmark_info
from shift_steps.train_steps import Train_Method_Info, align_vectors, get_target_distances

from typing import Tuple, NamedTuple, List
import itertools
import pathlib
import json

# %%
def classify_dists(dists, classify_method, align_wv, anchor_wv, word_pairs):
    if align_wv.type in ['original', 'normal']: 
        accuracies, results = assess_standard_prediction(
            dists[classify_method.name], classify_method.threshold, 
            word_pairs.target_labels)
        
        return accuracies, results, None
    
    elif 'sense' in align_wv.type:
        prediction_info = get_prediction_info(
            dists[classify_method.name], classify_method.threshold, 
            align_wv.model, anchor_wv.model, word_pairs.target_wps)

        if anchor_wv.type == 'normal':
            shift_data, ratio_data = make_normal_prediction(
                prediction_info, classify_method.threshold)

            accuracies, results = assess_normal_prediction(
                shift_data, ratio_data, word_pairs.target_labels)

            return accuracies, results, prediction_info

        elif anchor_wv.type == 'shared_sense':
            pred_labels, sense_matched_data = make_shared_prediction(
                prediction_info, classify_method.threshold)   

            accuracies, results = assess_shared_prediction(
                pred_labels, word_pairs.target_labels)
            
        else:
            pred_labels, sense_matched_data = make_sense_prediction(
                prediction_info, classify_method.threshold)   

            accuracies, results = assess_sense_prediction(
                pred_labels, word_pairs.target_labels)

            return accuracies, results, sense_matched_data

def save_results(
        accuracies, results, other_data, path_out, 
        align_method, classify_method, align_wv, anchor_wv
        ):
    if 'sense' in align_wv.type:
        if anchor_wv.type == 'sense':
            sense_matched_data = other_data
            print_both_sense_output(
                results, sense_matched_data,
                classify_method.threshold, path_out, 
                align_wv, anchor_wv)
        elif anchor_wv.type == 'shared_sense':
            # other_data
            print_shared_sense_output(
                results, sense_matched_data,
                classify_method.threshold, path_out, 
                align_wv, anchor_wv)
        else:
            prediction_info = other_data
            print_single_sense_output(
                prediction_info, results, 
                classify_method.threshold, path_out, 
                align_wv, anchor_wv)
    
    print_shift_info(
        accuracies, results, 
        path_out, align_method.name, 
        classify_method.name, classify_method.threshold, 
        align_wv, anchor_wv)
        
    results.to_csv(f"{path_out}/labels.csv", index=False)

def predict_target_shift(
        align_method: Train_Method_Info, 
        classify_methods: List[Train_Method_Info],  
        word_pairs,
        align_wv, anchor_wv,
        output_path: str, 
        num_loops: int,
        verbose: bool):

    # TODO: get labels here instead of later at the class step?

    # TODO: could import this if I wanted to not overwrite a previous best
    best_accuracies = {clf_method.name : 0 for clf_method in classify_methods}
    all_accuracies = {clf_method.name : [] for clf_method in classify_methods}

    for i in range(num_loops):
        print(f'{i+1} / {num_loops}')

        landmarks, landmark_pairs, align_wv, anchor_wv = align_vectors(
                                            align_method, word_pairs.all_wps, 
                                            align_wv, anchor_wv, verbose)

        dists, classify_methods = get_target_distances(
                                    classify_methods, 
                                    word_pairs.target_wps, landmarks, 
                                    align_wv, anchor_wv)

        for classify_method in classify_methods:

            accuracies, results, other_data = classify_dists(
                dists, classify_method, align_wv, anchor_wv, word_pairs)

            all_accuracies[classify_method.name].append(accuracies)
            max_acc = max(accuracies.values())

            if max_acc > best_accuracies[classify_method.name]:
                print(f'Saving new best accuracy {max_acc:.2f} for {classify_method.name}')
                best_accuracies[classify_method.name] = max_acc

                path_out = f"{output_path}/{align_method.name}_{classify_method.name}"
                save_results(
                    accuracies, results, other_data, path_out, 
                    align_method, classify_method, align_wv, anchor_wv)
                save_landmark_info(path_out, landmark_pairs, landmarks)

    print('Tests complete!')
    for classify_method, accuracy in best_accuracies.items():
        print(f'Best accuracy for {classify_method} was {accuracy:.2f}')

    return all_accuracies

def setup_wv(sense_method, align_info, anchor_info, slice_path, vector_path):
    if sense_method in ['original', 'normal']:
        align_type = sense_method
        anchor_type = sense_method        
    elif sense_method == 'SSA':
        align_type = 'sense'
        anchor_type = 'normal'
    elif sense_method == 'BSA':
        align_type = 'sense'
        anchor_type = 'sense'
    elif sense_method == 'TSA':
        align_type = 'shared_sense'
        anchor_type = 'shared_sense'

    align_wv = VectorVariations(corpus_name = align_info[0],
                            desc = align_info[1], 
                            type = align_type)
    anchor_wv = VectorVariations(corpus_name = anchor_info[0],
                                desc = anchor_info[1], 
                                type = anchor_type)

    align_wv, anchor_wv = load_wordvectors(
        align_wv, anchor_wv, slice_path, vector_path, normalize=True)

    return align_wv, anchor_wv

#%%
## align_info should be (name, desc)
def main(
    dataset_name:str,
    clust_together: bool, 
    data_path: str,
    targets: List[Target_Info],
    align_info: Tuple[str, str], 
    anchor_info: Tuple[str, str], 
    sense_methods: List[str],
    align_methods: List[Train_Method_Info]=None, 
    classify_methods: List[Train_Method_Info]=None, 
    num_loops: int = 1, 
    slice_num: int = None,
    verbose: bool = True
    ):

    # slice_num = None
    if slice_num is not None:
        slice_path = f'/slice_{slice_num}'
    else:
        slice_path = ''
    if clust_together:
        suffix = '_both'
        clust_label = 'both'
    else:
        suffix = ''
        clust_label = 'single'

    for sense_method, align_method in itertools.product(sense_methods, align_methods):
    
        print('\n================================================================')
        print(f'Starting {sense_method} method with {align_method.name} align')
        print(f'\t and {", ".join([method.name for method in classify_methods])} classify method(s) ')

        ## TODO: do these things not need to be copies?
        align_wv, anchor_wv = setup_wv(
            sense_method, align_info, anchor_info, slice_path, 
            f'{data_path}/word_vectors/{dataset_name}/{clust_label}')
        targets = filter_targets(targets, align_wv, anchor_wv)
        word_pairs = make_word_pairs(targets, align_wv, anchor_wv)
        print(word_pairs.target_labels, word_pairs.target_wps)

        output_path = (
            f"{data_path}/align_results/{dataset_name}{suffix}/"
            f"anchor_{anchor_info[0]}/align_{align_info[0]}"
            f"/{sense_method}{slice_path}"
        )
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
        for classify_method in classify_methods:
            result_path = f"{output_path}/{align_method.name}_{classify_method.name}"
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

        # print(f'Starting run for {align_method.name}, {align_wv.type} {slice_path[1:]}\n')
        all_accs = predict_target_shift(
                        align_method, classify_methods,
                        targets, word_pairs,
                        align_wv.copy(), anchor_wv.copy(), 
                        output_path, num_loops, 
                        clust_together, verbose)
                
        # return all_accs
        save_file_name = align_method.name
        
        if verbose: print(f'Results will be saved to {output_path}')
        with open(f'{output_path}/{save_file_name}_run_results.json' , 'w') as pf:
            json.dump(all_accs, pf)
