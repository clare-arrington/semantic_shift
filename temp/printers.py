from typing import Dict, Tuple, NamedTuple
import pandas as pd

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

        for target, info in sense_dists.items():        
            label = results[results.Words == target]['True Label'].iloc[0]
            print(f'\n{target} : {is_shifted[label]}', file=fout)
            
            sense_sorted = sorted(info, key=lambda tup: tup[0])
            for sense, dist, count, label in sense_sorted:
                print(f'\tSense {sense[-1]} : {dist:.2f}, {count}, {is_shifted[label]}', file=fout)

def print_both_sense_output(sense_dists: Dict[str, Tuple], results: pd.DataFrame, 
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

        for target, info in sense_dists.items():        
            label = results[results.Words == target]['True Label'].iloc[0]
            print(f'\n{target} : {is_shifted[label]}', file=fout)
            
            sense_sorted = sorted(info, key=lambda tup: tup[:2])
            printed = []
            for sense, anchor, count, anchor_wc, dist, label in sense_sorted:
                if sense not in printed:
                    if '0' not in sense:
                        print('', file=fout)
                    print(f'\tAlign Sense {sense[-1]} : {count}', file=fout)
                    printed.append(sense)

                print(f'\t\tAnchor Sense {anchor[-1]} : {dist:.2f}, {anchor_wc}, {is_shifted[label]}', file=fout)


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
