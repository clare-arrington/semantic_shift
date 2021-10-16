from sklearn.metrics import accuracy_score
import pandas as pd

from typing import NamedTuple, List
from collections import defaultdict

def assess_standard_prediction(predictions, threshold: float, targets: List[NamedTuple]):
    target_words, _, true_labels = list(zip(*targets))    

    predictions = predictions > threshold
    accuracy = accuracy_score(true_labels, predictions)
    accuracies = {'Predicted' : accuracy}

    results = pd.DataFrame(data={'Words' : target_words, 'True Label' : true_labels, 'Predicted' : predictions})
    
    for label in ['True Label', 'Predicted']:
        results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

## Go through each sense from the sense embedding (word.#) and find its distance to the word 
## connected to it in the anchored embedding
## This could be the same word, a translated word, etc
def make_sense_prediction(predictions, threshold, model, word_pairs):
    prediction_info = defaultdict(list)
    for word_pair, dist in zip(word_pairs, predictions):
        sense_word, anchor_word = word_pair
        shift_prediction = int(dist > threshold)
        word_count = model.wv.get_vecattr(sense_word, "count")
        target = sense_word.split('.')[0]

        info = (sense_word, dist, word_count, shift_prediction)
        prediction_info[target].append(info)

    # TODO: this could be passed in if we only wanted to do a subset
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
    
    # target_words, _, true_labels = list(zip(*targets))   
    with open('/home/clare/Data/corpus_data/semeval/truth/binary.txt') as fin:
        og_targets = fin.read().strip().split('\n')
        true_labels = []
        target_words = []
        for target in sorted(og_targets):
            target, label = target.split('\t')
            label = bool(int(label))
            word, pos = target.split('_')
            true_labels.append(label)
            target_words.append(word)

    accuracies = {}
    for method, shift_pred in shift_data.items():
        accuracy = accuracy_score(true_labels, shift_pred)
        #print(f'{method} accuracy: {accuracy:.2f}')
        accuracies[method] = accuracy

    for ratio_cutoff, shift_pred in ratio_data.items():
        accuracy = accuracy_score(true_labels, shift_pred)
        #print(f'{ratio_cutoff} ratio accuracy: {accuracy:.2f}')
        accuracies[ratio_cutoff] = accuracy

    data = {'Words' : target_words, 'True Label' : true_labels}
    data.update(shift_data)
    data.update(ratio_data)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

# for target, info in prediction_info.items():
#     print(target.capitalize())
#     for sense in info:
#         print(f'\t{sense[0]} : {sense[1]:.2f}')