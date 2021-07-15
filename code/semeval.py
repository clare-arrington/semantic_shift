#%%
"""
Run tests on SemEval2020 Task 1 data on the subtasks of:
    1 - binary classification
    2 - ranking
"""
from WordVectors import WordVectors, intersection
from alignment import align
from s4 import s4, threshold_crossvalidation

from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import numpy as np

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

#%%
np.random.seed(1)

## Pull data
with open("data/semeval/truth/binary.txt") as fin:
    data = map(lambda s: s.strip().split("\t"), fin.readlines())
    targets, true_class = zip(*data)
    y_true = np.array(true_class, dtype=int)

with open("data/semeval/truth/graded.txt") as fin:
    data = map(lambda s: s.strip().split("\t"), fin.readlines())
    _, true_ranking = zip(*data)
    true_ranking = np.array(true_ranking, dtype=float)

# TODO: will have to train these myself at some point
wv1 = WordVectors(input_file="wordvectors/semeval/english-corpus1.vec")
wv2 = WordVectors(input_file="wordvectors/semeval/english-corpus2.vec")

#%%
align_method = 'global'
classifier = 'cosine'

file_output=f"results/semeval/{align_method}_{classifier}_results.txt"
image_output=f"results/semeval/{align_method}_{classifier}_cm.png"
#%%
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

#%%
## Get predictions 
if classifier == 'cosine':
    auto_params = {
        "rate": 1.5,
        "n_fold": 1,
        "n_targets": 50,
        "n_negatives": 100
        }
    t_cos = threshold_crossvalidation(wv1_, wv2_, iters=1, **auto_params, landmarks=landmarks)

    dists = np.array([cosine(wv1_[w], wv2_[w]) for w in wv1.words])
    # TODO: ?
    x = get_feature_cdf(dists)
    y_pred = np.array([x[wv1.word_id[i.lower()]] for i in targets])
    #y_pred = vote(target_dists)

    y_bin = (y_pred>t_cos)

elif classifier == "s4":
    cls_params = {
        "n_targets": 100,
        "n_negatives": 50,
        "rate": 1,
        "iters": 500
    }
    model = s4(wv1_, wv2_, landmarks=landmarks, verbose=0, **cls_params, update_landmarks=False)
    ## Concatenate vectors of target words for prediction
    x = np.array([np.concatenate((wv1_[t.lower()], wv2_[t.lower()])) for t in targets])
    y_pred = model.predict(x)
    y_bin = y_pred > 0.5

    #rho, pvalue = spearmanr(true_ranking, y_pred)

#%%
## Assess predictions
accuracy = accuracy_score(y_true, y_bin)

targets = np.array(targets)
prediction = (y_bin == y_true)

incorrect = targets[~prediction]
changed = y_true.astype('bool')
incorr_pred = {}
incorr_pred['changed'] = np.intersect1d(incorrect, targets[changed])
incorr_pred['no change'] = np.intersect1d(incorrect, targets[~changed])

with open(file_output, 'w') as fout:
    print(f'{align_method.capitalize()} alignment; {classifier.capitalize()} classification', file=fout)
    print(f'Accuracy : {accuracy:.2f}\n', file=fout)

    print(f'**{len(incorrect)} incorrect predictions**', file=fout)

    for label, words in incorr_pred.items():
        print(f'{len(words)} {label}', file=fout)
        for word in words:
            print(f'\t{word} : ', file=fout)
        print('', file=fout) 


    # predictions = {'correct': [], 'incorrect': []}
    # predictions["correct"] = targets[prediction]
    # predictions["incorrect"]= targets[~prediction]

    # for label, words in predictions.items():
    #     print(f'{len(words)} {label} predictions', file=fout)
    #     for i in range(0, len(words), 5):
    #         print(f'\t{", ".join(words[i:i+5])}', file=fout)
    #     print('', file=fout) 

#%%
cm = confusion_matrix(y_true, y_bin, normalize="all")
plot_cm(cm, save_image=image_output)

# %%
