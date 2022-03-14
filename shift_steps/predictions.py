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
def get_prediction_info(predictions, threshold, align_model, anchor_model, word_pairs):
    prediction_info = defaultdict(lambda: defaultdict(list))
    for word_pair, dist in zip(word_pairs, predictions):
        sense_word, anchor_word = word_pair
        shift_prediction = int(dist > threshold)

        word_count = align_model.wv.get_vecattr(sense_word, "count")
        anchor_wc = anchor_model.wv.get_vecattr(anchor_word, "count")
        
        info = (anchor_word, word_count, anchor_wc, dist, shift_prediction)
        target = sense_word.split('.')[0]

        prediction_info[target][sense_word].append(info)

    return prediction_info

def make_sense_prediction(prediction_info, threshold):

    ratios = [.05, .1, .15, .2]
    ratio_data = defaultdict(list)
    shift_data = {shift : [] for shift in ['Majority', 'Main', 'Weighted']}

    target_words = []
    true_labels = []

    for target, sense_predictions in prediction_info.items():
        for sense, predictions in sense_predictions.items():

            target_words.append(sense)
            true_labels.append(True)

            shifted_count = 0
            unshifted_count = 0
            weighted_dist = 0
            shifts = []

            for anchor, count, anch_c, dist, shift_prediction in predictions:
                weighted_dist += (dist * count)

                shifts.append(shift_prediction)
                if shift_prediction:
                    shifted_count += count
                else:
                    unshifted_count += count

            majority_cutoff = len(shifts) // 2
            is_shifted = sum(shifts) > majority_cutoff
            shift_data['Majority'].append(is_shifted)

            biggest_sense = max(predictions, key=lambda t: t[1])
            is_shifted = biggest_sense[4]
            shift_data['Main'].append(is_shifted)

            weighted_dist /= (shifted_count + unshifted_count)
            is_shifted = weighted_dist > threshold
            shift_data['Weighted'].append(is_shifted)

            ratio = shifted_count / (shifted_count + unshifted_count)
            for ratio_cutoff in ratios:
                is_shifted = ratio >= ratio_cutoff
                ratio_cutoff = f'{int(ratio_cutoff*100)}%'
                ratio_data[ratio_cutoff].append(is_shifted)

    return shift_data, ratio_data, target_words, true_labels

def assess_sense_prediction(shift_data, ratio_data, target_words, true_labels):
    
    # target_words, _, true_labels = list(zip(*target_word_pairs)) 

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
