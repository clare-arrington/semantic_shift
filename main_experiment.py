#%%
"""
Runs semantic change experiment on arxiv data.
Saves a file un results/arxiv/ containing the semantic shift scores for
each alignment method (Global, Noise-Aware, S4) in order to compare them.
"""

from gensim.models import Word2Vec
from WordVectors import WordVectors, intersection
from alignment import align
from s4 import s4

from scipy.spatial.distance import cosine
import numpy as np
import os
import pickle, random

from nltk.corpus import stopwords


def distribution_of_change(*wvs, metric="euclidean"):
    """
    Gets distribution of change per word across input WordVectors list wvs.
    Assumes the WordVectors in wvs have been previously aligned to the same reference point
    (E.g.: align all to wvs[0]).
    Arguments:
            wvs - list of WordVectors objects
    Returns:
            d   - array of N elements with the mean cosine distance across the aligned WordVectors
                    (N is the size of the common vocabulary)
    """

    d = np.zeros((len(wvs[0])))
    for i, w in enumerate(wvs[0].words):
        # Compute mean vector
        v_mean = np.mean([wv[w] for wv in wvs], axis=0)
        # Compute distances to the mean
        if metric == "euclidean":
            distances = [np.linalg.norm(v_mean-wv[w])**2 for wv in wvs]
        elif metric == "cosine":
            distances = [cosine(v_mean, wv[w]) for wv in wvs]
        # distances = [cosine(v_mean, wv[w]) for wv in wvs]
        mean_d = np.mean(distances)
        d[i] = mean_d
    return d

def distribution_of_senses(*wvs, metric="euclidean"):
    """
    Gets distribution of change per word across input WordVectors list wvs.
    Assumes the WordVectors in wvs have been previously aligned to the same reference point
    (E.g.: align all to wvs[0]).
    Arguments:
            wvs - list of WordVectors objects
    Returns:
            d   - array of N elements with the mean cosine distance across the aligned WordVectors
                    (N is the size of the common vocabulary)
    """

    d = np.zeros((len(wvs[0])))
    for i, w in enumerate(wvs[0].words):
        # Compute mean vector
        v_mean = np.mean([wv[w] for wv in wvs], axis=0)
        # Compute distances to the mean
        if metric == "euclidean":
            distances = [np.linalg.norm(v_mean-wv[w])**2 for wv in wvs]
        elif metric == "cosine":
            distances = [cosine(v_mean, wv[w]) for wv in wvs]
        # distances = [cosine(v_mean, wv[w]) for wv in wvs]
        mean_d = np.mean(distances)
        d[i] = mean_d
    return d

def print_table(d, words, n=20, f=None, include_stable=False):
    """
    Prints table of stable and unstable words in the following format:
    <stable words> | <unstable words>
    Arguments:
                d       - distance distribution
                words   - list of words - indices of d and words must match
                n       - number of rows in the table
    """
    print("-"*20, file=f)
    indices = np.argsort(d)

    stable_results = {}
    unstable_results = {}

    if include_stable:
        print('Stable', file=f)
        for index in indices[:n]:
            print(f"{d[index]:1.2f} : {words[index]}", file=f)
            stable_results[words[index]] = d[index]

    print('\n\nUnstable', file=f)
    for index in reversed(indices[-n:]):
        print(f"{d[index]:1.2f} : {words[index]}", file=f)
        unstable_results[words[index]] = d[index]

    print("-"*20, file=f)

    return stable_results, unstable_results

#%%
num = 2
model = Word2Vec.load(f'wordvectors/senses/semeval_corpus{num}.vec')

sense_words = ['circle', 'edge', 'head', 'land', 
                'lass', 'rag', 'thump', 'tip',
                'multitude', 'risk', 'twist', 'savage']

vocab = list(model.wv.index_to_key)
print(len(vocab))

filtered = [word for word in vocab if '.' in word]
print(len(filtered))

path_out = "results/usuk/"

#%%
##
#wva = WordVectors(input_file="wordvectors/ukus/coca.vec")
wva = WordVectors(words=vocab, vectors=model.wv.vectors)
wvb = WordVectors(input_file="wordvectors/ukus/bnc.vec")
wva, wvb = intersection(wva, wvb)

## Q is the np matrix to 
## wva is changed, wvb isn't
wva, wvb, Q = align(wva, wvb)

## Set order of words for both wva and wvb after aligning
words = wva.words

print("-- Common vocab", len(words))
# each column of this matrix will store a set of results for a method
out_grid = np.zeros((len(words), 5))

d = distribution_of_change(wva, wvb)

# print("====== GLOBAL", file=fout)
# print("=> landmarks", len(wva.words))
# print_table(d, wva.words)
# out_grid[:, 0] = d  # add first column

with open(os.path.join(path_out, f'sense_results.txt'), "w") as fout:
    fout.write("====== GLOBAL ======\n")
    fout.write(f"=> {len(wva.words)} landmarks\n")
    stable_results, unstable_results = print_table(d, wva.words, n=10, f=fout, include_stable=True)

# print("===== SELF")
# landmarks, nonl, Q = s4(wva, wvb, iters=100, verbose=1)
# wva, wvb, Q = align(wva, wvb, anchor_words=landmarks)
# d = distribution_of_change(wva, wvb)
# print_table(d, wva.words)
# out_grid[:, 2] = d  # last column

#%%
wva = WordVectors(words=vocab, vectors=model.wv.vectors)
wvb = WordVectors(input_file="wordvectors/ukus/bnc.vec")

r = WordVectors(words=wva.words, vectors=np.dot(wva.vectors, Q))

# TODO: why did I get stable results?
#d = distribution_of_change(r, wvb)

sense_words = ['supplements', 'variance', 'flats', 'meters', 'mixer', 'web', 'receiver']
filtered = sorted(filtered)

d = np.zeros((len(filtered)))
for i, w in enumerate(filtered):
    # Compute mean vector
    word = w.split('_')[0]
    v_mean = np.mean([r[w], wvb[word]], axis=0)
    # Compute distances to the mean
    distances = [np.linalg.norm(v_mean-wv)**2 for wv in [r[w], wvb[word]]]
    d[i] = np.mean(distances)


with open(os.path.join(path_out, f'sense_differences.txt'), "w") as fout:
    target = ''
    for dist, w in zip(d, filtered):
        word, sense = w.split('_')
        if target != word:
            target = word
            fout.write(f'\n{word}\n')
        fout.write(f'{sense} : {dist}\n')


