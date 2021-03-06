from typing import Dict, Tuple, NamedTuple
from tabulate import tabulate
import pandas as pd
import pickle

def print_single_sense_output(sense_dists: Dict[str, Tuple], results: pd.DataFrame, 
                       threshold: float, path_out: str, 
                       align_vector: NamedTuple, anchor_vector: NamedTuple):

    with open(f'{path_out}/senses.txt', 'w') as fout:

        print('Sense breakdown for each target word with:', file=fout)
        print(f'\t - cosine distance of sense in aligned vector {align_vector.desc} \
                \n\t to target word in unchanged vector {anchor_vector.desc}', file=fout)
        print(f'\t - count of sense in aligned vector {align_vector.desc}\n', file=fout)
        print(f'Distance threshold = {threshold:.2f}\n\n', file=fout)

        print('Format for each target is...', file=fout)
        print('target : true label', file=fout)
        print('\tSense # : cos. dist, count, threshold label', file=fout)

        is_shifted = {  0: 'not shifted', 
                            1: 'shifted', 
                            'No Shift': 'not shifted', 
                            'Shifted': 'shifted'}

        for target, sense_info in sense_dists.items():
            label = results[results.Words == target]['True Label'].iloc[0]
            print(f'\n{target}', file=fout)
            print(f'\tLabel: {is_shifted[label]}', file=fout)

            for anchor_term, info in sense_info.items():    
                sense_sorted = sorted(info, key=lambda tup: tup[0])
                count_printed = False
                for sense, count, anchor_count, dist, shift_prediction in sense_sorted:
                    if not count_printed:
                        print(f'\tNum occurences: {anchor_count}\n', file=fout)
                        count_printed = True
                    print(f'\tSense {sense[-1]} : {dist:.2f}, {count} occurences \t {is_shifted[shift_prediction]}', file=fout)

def print_both_sense_output(results: pd.DataFrame, sense_matched_data,
                       threshold: float, path_out: str, 
                       align_vector: NamedTuple, anchor_vector: NamedTuple):

    with open(f'{path_out}/senses.txt', 'w') as fout:

        print('Sense breakdown for each target word with:', file=fout)
        print(f'\t - cosine distance of sense in aligned vector {align_vector.desc} \
                \n\t to target word in sense vector {anchor_vector.desc}', file=fout)
        print(f'\t - count of senses in both vector {align_vector.desc}\n', file=fout)
        print(f'Distance threshold = {threshold:.2f}\n', file=fout)
        print('=====================================================\n', file=fout)

        print('Format for each target is...', file=fout)
        print('target : true label', file=fout)
        print('\tAlign Sense # : count', file=fout)
        print('\t\tAnchor Sense # : cos. dist, count, label based on threshold', file=fout)
        print('\n=====================================================', file=fout)

        is_shifted = {  0: 'not shifted', 
                            1: 'shifted', 
                            'No Shift': 'not shifted', 
                            'Shifted': 'shifted'}

        vector_names = [
            (anchor_vector.corpus_name, align_vector.corpus_name),
            (align_vector.corpus_name, anchor_vector.corpus_name)]
        for target, match_info in sense_matched_data.items():
            use_other_count = False
            label = results[results.Words == target]['True Label'].iloc[0]
            print(f'\n{target} : {is_shifted[label]}', file=fout)
                
            for corpus_names, matches in zip(vector_names, match_info):
                main_corpus, other_corpus = corpus_names
                print(f'\tMatches for {main_corpus} senses', file=fout)
                for sense in sorted(matches):
                    match, count, other_count, dist, label = matches[sense]
                    if use_other_count:
                        count, other_count = other_count, count
                    print(f'\t\t{main_corpus}.{sense[-1]}  ({count})', end='', file=fout)
                    print(f' -> {other_corpus}.{match[-1]} ({other_count})', file=fout)
                    print(f'\t\t\t{is_shifted[label].capitalize()}, {dist:.2f}\n', file=fout)
                use_other_count = True

def print_shift_info(accuracies: Dict[str,float], results: pd.DataFrame, path_out: str, 
                     align_method: str, classify_method: str, threshold: float,
                     align_vector: NamedTuple, anchor_vector: NamedTuple) -> None :

    with open(f'{path_out}/accuracies.txt', 'w') as fout:
        if align_vector.type == 'sense':
            print(f'Sense induction on {align_vector.desc}', file=fout)
        else:
            print(f'{align_vector.type.capitalize()} {align_vector.desc}', file=fout)

        print(f'Aligned to {anchor_vector.type} {anchor_vector.desc}\n', file=fout)
        print(f'{align_method.capitalize()} alignment; {classify_method} classification\n', file=fout)
        
        if classify_method == 'cosine':
            print(f'Cosine distance threshold = {threshold:.2f}', file=fout)

        ## Get the highest accuracy and print all methods that achieved it
        max_acc = max(accuracies.values())
        print(f'\nMethods with highest accuracy of {max_acc:.2f}', file=fout)
        for label, accuracy in accuracies.items():
            ## Fix this in the earlier part          
            if accuracy == max_acc:
                print(f'\t{label} Shifted', file=fout)

        ## Go through each method, giving accuracy and which terms were incorrectly labeled
        for label, accuracy in accuracies.items():
            incorrect = results[results['True Label'] != results[label]]
            unshifted = incorrect[incorrect['True Label'] == 'No Shift'].Words    
            shifted = incorrect[incorrect['True Label'] == 'Shifted'].Words    

            print('\n=========================================\n', file=fout)
            if type(label) == float:
                label = f'{int(label*100)}%'
            print(f'{label} Shifted\n', file=fout)
            
            print(f'Accuracy : {accuracy:.2f}', file=fout)
            print(f'Incorrect predictions: {len(incorrect)} / {len(results)}', file=fout)

            print(f'\n{len(unshifted)} predicted shifted; correct label is unshifted', file=fout)
            for word in unshifted:
                print(f'\t{word}', file=fout)

            print(f'\n{len(shifted)} predicted unshifted; correct label is shifted', file=fout)
            for word in shifted:
                print(f'\t{word}', file=fout)

def save_landmark_info(path_out, landmark_pairs, landmarks):
    with open(f'{path_out}/landmarks.txt' , 'w') as f:
        for l1, l2 in landmark_pairs:
            f.write(f'({l1}, {l2})\n')

    ## I save both to be safe; you can verify the models are 
    ## the same if these two sets pull the same information
    with open(f'{path_out}/landmark_pairs.pkl' , 'wb') as pf:
        pickle.dump(landmark_pairs, pf)

    with open(f'{path_out}/landmark_ids.pkl' , 'wb') as pf:
        pickle.dump(landmarks, pf)