#%%
from WordVectors import WordVectors, intersection
from alignment import align
from s4 import s4, threshold_crossvalidation

from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import re

def get_feature_cdf(x):
    """
    Estimate a CDF for feature distribution x
    One way this can be done is via sorting arguments according to values,
    getting a sorted array of positions (low to high)
    then normalize this by len(x)
    Arguments:
        x       - feature vector
    Returns:
        p       - CDF values (percentile) for input feature vector
                i.e.: p[i] is the probability that X <= x[i]
    """
    y = np.argsort(x)
    p = np.zeros(len(x))
    for i, v in enumerate(y):
        p[v] = i+1  # i+1 is the position of element v in the CDF
    p = p/len(x)  # normalize for cumulative probabilities
    return p

def vote(x, hard=False):
    """
    Cast vote to decide whether there is semantic shift of a word or not.
    Arguments:
            x       - N x d array of N words and d features with columns as CDFs
            hard    - use hard voting, all features cast a binary vote, decision is averaged
                      if False, then votes are average, then binary the decision is made
    Returns:
            r       - Binary array of N elements (decision)
    """
    x = x.reshape(-1, 1)
    r = np.zeros((len(x)), dtype=float)
    for i, p in enumerate(x):
        if hard:
            p_vote = np.mean([float(pi > 0.5) for pi in p])
            r[i] = p_vote
        else:
            avg = np.mean(p)
            r[i] = avg
    return r

def plot_cm(cm, save_image=''):
    plt.imshow(cm, cmap='Reds')
    classNames = ['Negative','Positive']
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, f'{s[i][j]} = {cm[i][j]:.2f}')
    if save_image != '':
        plt.savefig(save_image)
    else:
        plt.show()

def load_wordvectors(run):
    if run == 'original':
        wv1 = WordVectors(input_file="wordvectors/semeval/original_corpus2.vec")
        wv2 = WordVectors(input_file="wordvectors/semeval/original_corpus1.vec")
        
        return wv1, wv2, _, _

    else:
        m1 = Word2Vec.load(f'wordvectors/semeval/{run}_corpus2.vec')
        v1 = list(m1.wv.index_to_key)
        # vectors=m1.wv.vectors
        wv1 = WordVectors(words=v1, vectors=m1.wv.get_normed_vectors())

        if run == 'target':
            m2 = Word2Vec.load('wordvectors/semeval/new_corpus1.vec')
        else:
            m2 = Word2Vec.load(f'wordvectors/semeval/{run}_corpus1.vec')
        wv2 = WordVectors(words=list(m2.wv.index_to_key), vectors=m2.wv.get_normed_vectors())

    return wv1, wv2, v1, m1

def print_shift_info(run, accuracies, results, align_method, classifier, accuracy_output):
    with open(accuracy_output, 'w') as fout:
        if run == 'target':
            print(f'Sense induction on corpus 1960 - 2010', file=fout)
            print(f'Aligned to corpus 1810-1860\n', file=fout)
        else:
            print(f'{run.capitalize()} corpus 1960 - 2010', file=fout)
            print(f'Aligned to {run} corpus 1810-1860\n', file=fout)

        print(f'{align_method.capitalize()} alignment; {classifier.capitalize()} classification\n', file=fout)
        
        max_acc = max(accuracies.values())
        print(f'Methods with highest accuracy of {max_acc:.2f}', file=fout)
        for label, accuracy in accuracies.items():
            if type(label) == float:
                label = f'{int(label*100)}%'            
            if accuracy == max_acc:
                print(f'\t{label} Shifted', file=fout)

        for label, accuracy in accuracies.items():
            incorrect = results[results['True Label'] != results[label]]
            unshifted = incorrect[incorrect['True Label'] == 'Not'].Words    
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

#%%
run = 'target'
align_method = 's4'
classifier = 'cosine'

path_out = f"results/semeval/{run}_{align_method}_{classifier}"

accuracy_output=f"{path_out}_accuracy.txt"
senses_output=f"{path_out}_senses.txt"
labels_output=f"{path_out}_labels.csv"
# image_output=f"{path_out}_cm.png"

## Pull data
with open(f'../data/semeval/truth/binary.txt') as fin:
    data = map(lambda s: s.strip().split("\t"), fin.readlines())
    targets, true_class = zip(*data)
    y_true = np.array(true_class, dtype=int)

target_words = [target.split('_')[0] for target in targets]
#%%
## Run a few times for best results
best_accuracy = (0, [])
for i in range(1):
    wv1, wv2, vocab, _ = load_wordvectors(run)

    wv1, wv2 = intersection(wv1, wv2)
    # print("Size of common vocab.", len(wv1))

    ## Get landmarks
    if align_method == 'global':
        landmarks = wv1.words

    elif align_method == 's4':
        align_params = {
                "n_targets": 100,
                "n_negatives": 50,
                "rate": 1,
                "iters": 100
            }
        landmarks, non_landmarks, Q = s4(wv1, wv2, cls_model="nn", verbose=0, **align_params)

    ## Align
    wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)

    ## Get predictions 
    if classifier == 'cosine':
        auto_params = {
            "rate": 1.5,
            "n_fold": 1,
            "n_targets": 50,
            "n_negatives": 100
            }
        t_cos = threshold_crossvalidation(wv1_, wv2_, iters=2, **auto_params, landmarks=landmarks)

        if run == 'original' or run == 'new':
            dists = np.array([cosine(wv1_[w], wv2_[w]) for w in wv1.words])
            # # TODO: ?
            # dists = get_feature_cdf(dists)
            y_pred = np.array([dists[wv1.word_id[i.lower()]] for i in targets])
            # y_pred = vote(y_pred)

            y_bin = (y_pred>t_cos)

            accuracy = accuracy_score(y_true, y_bin)
            print(f'Accuracy: {accuracy:.2f}')
            if accuracy > best_accuracy[0]:
                best_accuracy = (accuracy, y_bin)

print(f'Cos. dist : {t_cos:.2f}')

if run == 'original' or run == 'new': 
    accuracy, y_bin = best_accuracy
    y_bin = [int(y) for y in y_bin]
    #prediction = (y_bin == y_true)
    accuracies = {'Predicted' : accuracy}
    
    results = pd.DataFrame(data={'Words' : target_words, 'True Label' : y_true, 'Predicted' : y_bin})
    for label in ['True Label', 'Predicted']:
        results[label] = results[label].map({0:'Not', 1:'Shifted'})

elif run == 'target':
    wv1, wv2, vocab, model = load_wordvectors(run)
    wv1_ = WordVectors(words=wv1.words, vectors=np.dot(wv1.vectors, Q))

    filtered = [word for word in vocab if '.' in word]
    print(f'{len(filtered)} filtered senses\n')

    #%%
    sense_dists = {}
    for target_pos in targets:
        # Compute mean vector
        target = target_pos.split('_')[0] 
        sense_dists[target] = []

        r = re.compile(f'{target}.[0-9]')

        for sense in filter(r.match, filtered):
            dist = cosine(wv1_[sense], wv2[target_pos])        
            word_count = model.wv.get_vecattr(sense, "count")
            is_shifted = int(dist > t_cos)

            info = (sense, dist, word_count, is_shifted)
            sense_dists[target].append(info)
    
    shift_labels = ['Majority', 'Main', 'Weighted']
    shift_data = {shift : [] for shift in shift_labels}
    ratio_data = {rate:[] for rate in [.05, .1, .15, .2]}

    for target, senses in sense_dists.items():
        shifted_count = 0
        unshifted_count = 0
        weighted_dist = 0
        shifts = []
        for sense, dist, count, is_shifted in senses:
            weighted_dist += (dist * count)

            shifts.append(is_shifted)
            if is_shifted:
                shifted_count += count
            else:
                unshifted_count += count

        majority_cutoff = len(shifts) // 2
        is_shifted = int(sum(shifts) > majority_cutoff)
        shift_data['Majority'].append(is_shifted)

        biggest_sense = max(senses, key=lambda t: t[2])
        is_shifted = biggest_sense[3]
        shift_data['Main'].append(is_shifted)

        weighted_dist /= (shifted_count + unshifted_count)
        is_shifted = int(weighted_dist > t_cos)
        shift_data['Weighted'].append(is_shifted)

        # TODO: I changed to include equal to
        ratio = shifted_count / (shifted_count + unshifted_count)
        for ratio_cutoff in ratio_data.keys():
            is_shifted = int(ratio >= ratio_cutoff)
            ratio_data[ratio_cutoff].append(is_shifted)

    ## Assess predictions
    accuracies = {}
    for label, shift in shift_data.items():
        accuracy = accuracy_score(y_true, shift)
        print(f'{label} accuracy: {accuracy:.2f}')
        accuracies[label] = accuracy

    for ratio_cutoff, shift in ratio_data.items():
        accuracy = accuracy_score(y_true, shift)
        print(f'{ratio_cutoff:.2f} ratio accuracy: {accuracy:.2f}')
        accuracies[ratio_cutoff] = accuracy

    data = {'Words' : target_words, 'True Label' : y_true}
    data.update(shift_data)
    data.update(ratio_data)
    results = pd.DataFrame(data)

    for label in results.columns:
        if label != 'Words':
            results[label] = results[label].map({0:'Not', 1:'Shifted'})

#%%
print_shift_info(run, accuracies, results, align_method, classifier, accuracy_output)

results.to_csv(labels_output, index=False)
#%%
with open(senses_output, 'w') as fout:
    print('Sense breakdown for each target word with:', file=fout)
    print('\t- cosine distance of sense in aligned vector (1960-2010) to target word in unchanged vector (1810-1860)', file=fout)
    print('\t- count of sense in aligned vector (1960-2010)\n', file=fout)

    print(f'Cosine distance threshold = {t_cos:.2f}', file=fout)

    for target, info in sense_dists.items():        
        is_shifted = results[results.Words == target]['True Label'].iloc[0]
        if is_shifted == 'Not':
            print(f'\n{target} : not shifted', file=fout)
        else:
            print(f'\n{target} : shifted', file=fout)
        
        sense_sorted = sorted(info, key=lambda tup: tup[0])
        for sense, dist, count, _ in sense_sorted:
            print(f'\tSense {sense[-1]} : {dist:.2f}, {count}', file=fout)

# %%
