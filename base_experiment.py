 
#%%
from shift_steps.predictions import get_prediction_info, make_sense_prediction, assess_sense_prediction, assess_standard_prediction
from shift_steps.target_prep import Target_Info, filter_targets, make_word_pairs
from shift_steps.wordvectors import VectorVariations, load_wordvectors
from shift_steps.printers import print_single_sense_output, print_both_sense_output, print_shift_info, save_landmark_info
from shift_steps.train_steps import Train_Method_Info, align_vectors, get_target_distances

from typing import Tuple, NamedTuple, List
import itertools
import pathlib
import pickle

# %%
def predict_target_shift(
        align_method: Train_Method_Info, 
        classify_methods: List[Train_Method_Info],  
        dataset_name: str, 
        targets: List[NamedTuple],
        align_wv: VectorVariations, anchor_wv: VectorVariations,
        output_path: str, data_path: str, slice_path: str,
        num_loops: int):

    # print(f'Starting run for {align_method.name}, {align_wv.type} {slice_path[1:]}\n')

    ## Pull and prep vector data
    align_wv, anchor_wv = load_wordvectors(align_wv, anchor_wv, slice_path,
        f'{data_path}/word_vectors/{dataset_name}', normalize=True)
    targets = filter_targets(targets, align_wv, anchor_wv)
    all_word_pairs, target_word_pairs = make_word_pairs(targets, align_wv, anchor_wv)

    # TODO: could import this if I wanted to not overwrite a previous best
    best_accuracies = {clf_method.name : 0 for clf_method in classify_methods}
    all_accuracies = {clf_method.name : [] for clf_method in classify_methods}

    for i in range(num_loops):
        print(f'{i+1} / {num_loops}')

        landmarks, landmark_terms, align_wv, anchor_wv = align_vectors(
                                            align_method, all_word_pairs, 
                                            align_wv, anchor_wv)

        dists, classify_methods = get_target_distances(
                                    classify_methods, 
                                    target_word_pairs, landmarks, 
                                    align_wv, anchor_wv)

        for classify_method in classify_methods:

            if align_wv.type in ['original', 'normal']: 
                accuracies, results = assess_standard_prediction(
                    dists[classify_method.name], classify_method.threshold, targets)
            
            elif 'sense' in align_wv.type:
                prediction_info = get_prediction_info(
                    dists[classify_method.name], classify_method.threshold, 
                    align_wv.model, anchor_wv.model, target_word_pairs)

                shift_data, ratio_data = make_sense_prediction(prediction_info, classify_method.threshold)

                accuracies, results = assess_sense_prediction(
                    shift_data, ratio_data, targets)
            
            all_accuracies[classify_method.name].append(accuracies)
            max_acc = max(accuracies.values())

            if max_acc > best_accuracies[classify_method.name]:
                print(f'Found new best accuracy {max_acc:.2f} for {classify_method.name}, so printing info')
                best_accuracies[classify_method.name] = max_acc
                
                path_out = f"{output_path}/{align_method.name}_{classify_method.name}"
                pathlib.Path(path_out).mkdir(parents=True, exist_ok=True)
                
                if align_wv.type == 'sense':
                    if anchor_wv.type == 'sense':
                        print_both_sense_output(
                            prediction_info, results, 
                            classify_method.threshold, path_out, 
                            align_wv, anchor_wv)
                    else:
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

                save_landmark_info(path_out, landmark_terms, landmarks)

    print('Tests complete!')
    for classify_method, accuracy in best_accuracies.items():
        print(f'Best accuracy for {classify_method} was {accuracy:.2f}')

    return all_accuracies

#%%
## align_info should be (name, desc)
def main(
    dataset_name:str, 
    data_path: str,
    targets: List[Target_Info],
    align_info: Tuple[str, str], 
    anchor_info: Tuple[str, str], 
    sense_methods: List[str],
    align_methods: List[Train_Method_Info]=None, 
    classify_methods: List[Train_Method_Info]=None, 
    num_loops: int = 1, slice_num: int = None):

    for sense_method, align_method in itertools.product(sense_methods, align_methods):
    
        print('\n================================================================')
        print(f'Starting {sense_method} method with {align_method.name} align')
        print(f'\t and {", ".join([method.name for method in classify_methods])} classify method(s) ')

        if sense_method in ['original', 'normal']:
            align_type = sense_method
            anchor_type = sense_method        
        elif sense_method == 'SSA':
            align_type = 'sense'
            anchor_type = 'normal'
        elif sense_method == 'BSA':
            align_type = 'sense'
            anchor_type = 'sense'

        align_wv = VectorVariations(corpus_name = align_info[0],
                                desc = align_info[1], 
                                type = align_type)
        anchor_wv = VectorVariations(corpus_name = anchor_info[0],
                                    desc = anchor_info[1], 
                                    type = anchor_type)
        # slice_num = None
        if slice_num is not None:
            slice_path = f'/slice_{slice_num}'
        else:
            slice_path = ''

        output_path = f'{data_path}/align_results/{dataset_name}/align_{align_info[0]}/{sense_method}{slice_path}'
        pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

        ## TODO: should this targets be a copy?
        all_accs = predict_target_shift(
                        align_method, classify_methods,
                        dataset_name, targets,
                        align_wv, anchor_wv, 
                        output_path, data_path, slice_path,
                        num_loops)
        save_file_name = align_method.name
        
        ## TODO: save in JSON instead?
        print(f'Results will be saved to {output_path}')
        with open(f'{output_path}/{save_file_name}_run_results.pkl' , 'wb') as pf:
            pickle.dump(all_accs, pf)
