from sklearn.metrics import accuracy_score
import pandas as pd

from typing import NamedTuple, List
from collections import defaultdict

def assess_standard_prediction(predictions, threshold: float, target_labels):
    true_labels = list(target_labels.values())
    target_words = [target for target, anchor in target_labels.keys()]

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
        
        info = (sense_word, word_count, anchor_wc, dist, shift_prediction)
        target = sense_word.split('.')[0]

        prediction_info[target][anchor_word].append(info)

    return prediction_info

def get_weighted_shift(predictions, threshold, use_other_count=False):
    shifted_count = 0
    unshifted_count = 0
    weighted_dist = 0
    for sense, count, other_count, dist, shift_prediction in predictions:
        if use_other_count:
            count = other_count
        weighted_dist += (dist * count)
        if shift_prediction:
            shifted_count += count
        else:
            unshifted_count += count

    weighted_dist /= (shifted_count + unshifted_count)
    return weighted_dist > threshold

def make_normal_prediction(prediction_info, threshold):
    # ratios = [.05, .1, .15, .2]
    # ratio_data = defaultdict(dict)
    shift_data = {shift : {} for shift in [ 'Majority', 'Main', 'Weighted']}

    for target, sense_predictions in prediction_info.items():
        for anchor, predictions in sense_predictions.items():
            pair = (target, anchor)
            shifts = []

            majority_cutoff = len(shifts) // 2
            is_shifted = sum(shifts) > majority_cutoff
            shift_data['Majority'][pair] = is_shifted

            biggest_sense = max(predictions, key=lambda t: t[1])
            is_shifted = biggest_sense[4]
            shift_data['Main'][pair] = is_shifted

            is_shifted = get_weighted_shift(predictions, threshold)
            shift_data['Weighted'][pair] = is_shifted

            # ratio = shifted_count / (shifted_count + unshifted_count)
            # for ratio_cutoff in ratios:
            #     is_shifted = ratio >= ratio_cutoff
            #     ratio_cutoff = f'{int(ratio_cutoff*100)}%'
            #     ratio_data[ratio_cutoff][pair] = is_shifted

    return shift_data, None #, ratio_data

def assess_normal_prediction(shift_data, ratio_data, target_labels):
    true_labels = list(target_labels.values())
    target_words = [target for target, anchor in target_labels.keys()]

    accuracies = {}
    method_preds = {}
    for method, shift_pred in shift_data.items():
        pred_labels = list(shift_pred.values())
        accuracy = accuracy_score(true_labels, pred_labels)
        #print(f'{method} accuracy: {accuracy:.2f}')
        accuracies[method] = accuracy
        method_preds[method] = pred_labels

    # ratio_preds = {}
    # for ratio_cutoff, shift_pred in ratio_data.items():
    #     pred_labels = list(shift_pred.values())
    #     accuracy = accuracy_score(true_labels, pred_labels)
    #     #print(f'{ratio_cutoff} ratio accuracy: {accuracy:.2f}')
    #     accuracies[ratio_cutoff] = accuracy
    #     ratio_preds[ratio_cutoff] = pred_labels

    data = {'Words' : target_words, 'True Label' : true_labels}
    data.update(method_preds)
    # data.update(ratio_preds)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

def make_sense_prediction(prediction_info, threshold):
    pred_labels = {shift : {} for shift in ['Majority', 'Main', 'Weighted']}

    sense_matched_data = {}

    for target, sense_predictions in prediction_info.items():
        # target = 'face'
        # sense_predictions = prediction_info[target]
        # TODO: this is bad for US / UK
        pair = (target, target)
        align_matches = {}
        anchor_matches = {}
        for anchor_sense in sorted(sense_predictions):     
            predictions = sorted(sense_predictions[anchor_sense])
            closest_match = min(predictions, key = lambda t: t[3])
            anchor_matches[anchor_sense] = closest_match

            for align_tuple in predictions:
                align_sense, *tuple_info = align_tuple
                anchor_tuple = tuple([anchor_sense] + tuple_info)
                if align_sense not in align_matches:
                    closest_match = anchor_tuple
                else:
                    closest_match = min(
                        align_matches[align_sense], anchor_tuple, key = lambda t: t[3])
                align_matches[align_sense] = closest_match

        sense_matched_data[target] = (anchor_matches, align_matches)

        preds = [anchor_matches.values(), align_matches.values()]

        ## For each of these below, assume not shifted until we find a case.
        ## Get majority shift
        is_shifted = False
        for predictions in preds:
            shifts = [m[-1] for m in predictions]
            majority_cutoff = len(shifts) // 2
            is_shifted = is_shifted or (sum(shifts) > majority_cutoff)
        pred_labels['Majority'][pair] = is_shifted

        ## Get main shift
        is_shifted = False
        for predictions in preds:
            biggest_sense = max(predictions, key=lambda t: t[1])
            is_shifted = is_shifted or biggest_sense[4]        
        pred_labels['Main'][pair] = is_shifted

        ## Get weighted shift
        anchor_shift = get_weighted_shift(
            anchor_matches.values(), threshold)
        align_shift = get_weighted_shift(
            align_matches.values(), threshold, use_other_count=True)

        is_shifted = align_shift or anchor_shift
        pred_labels['Weighted'][pair] = is_shifted

    return pred_labels, sense_matched_data

def assess_sense_prediction(pred_labels, target_labels):
    true_labels = list(target_labels.values())
    target_words = [target for target, anchor in target_labels.keys()]

    accuracies = {}
    method_preds = {}
    for method, shift_pred in pred_labels.items():
        pred_labels = list(shift_pred.values())
        accuracy = accuracy_score(true_labels, pred_labels)
        accuracies[method] = accuracy
        method_preds[method] = pred_labels

    data = {'Words' : target_words, 'True Label' : true_labels}
    data.update(method_preds)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results

def make_shared_prediction(prediction_info, threshold):
    shift_data = {shift : {} for shift in [ 'Majority', 'Main', 'Weighted']}

    for target, sense_predictions in prediction_info.items():
        for anchor, predictions in sense_predictions.items():
            pair = (target, anchor)
            shifts = []

            majority_cutoff = len(shifts) // 2
            is_shifted = sum(shifts) > majority_cutoff
            shift_data['Majority'][pair] = is_shifted

            biggest_sense = max(predictions, key=lambda t: t[1])
            is_shifted = biggest_sense[4]
            shift_data['Main'][pair] = is_shifted

            is_shifted = get_weighted_shift(predictions, threshold)
            shift_data['Weighted'][pair] = is_shifted

    return shift_data

def assess_shared_prediction(shift_data, target_labels):
    true_labels = list(target_labels.values())
    target_words = [target for target, anchor in target_labels.keys()]

    accuracies = {}
    method_preds = {}
    for method, shift_pred in shift_data.items():
        pred_labels = list(shift_pred.values())
        accuracy = accuracy_score(true_labels, pred_labels)
        accuracies[method] = accuracy
        method_preds[method] = pred_labels

    data = {'Words' : target_words, 'True Label' : true_labels}
    data.update(method_preds)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].astype(int).map({0:'No Shift', 1:'Shifted'})

    return accuracies, results
