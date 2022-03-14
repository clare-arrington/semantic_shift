 
from shift_steps.wordvectors import WordVectors
from scipy.linalg import orthogonal_procrustes
import numpy as np

# Word alignment module
def align(wv1, wv2, anchor_indices=None, anchor_words=None, 
        anchor_top=None, anchor_bot=None, 
        anchor_random=None, anchor_pairs=None,
        exclude={},
        method="procrustes"):
    """
    Implement OP alignment for a given set of landmarks.
    If no landmark is given, performs global alignment.
    Arguments:
        wv1 - WordVectors object to align to wv2
        wv2 - Target WordVectors. Will align wv1 to it.
        anchor_indices - (optional) uses word indices as landmarks
        anchor_words - (optional) uses words as landmarks
        exclude - set of words to exclude from alignment
        method - Alignment objective. Currently only supports orthogonal procrustes.
    """
    if anchor_top is not None:
        v1 = [wv1.vectors[i] for i in range(anchor_top) if wv1.words[i] not in exclude]
        v2 = [wv2.vectors[i] for i in range(anchor_top) if wv2.words[i] not in exclude]
    elif anchor_bot is not None:
        v1 = [wv1.vectors[-i] for i in range(anchor_bot) if wv1.words[i] not in exclude]
        v2 = [wv2.vectors[-i] for i in range(anchor_bot) if wv2.words[i] not in exclude]
    elif anchor_random is not None:
        anchors = np.random.choice(range(len(wv1.vectors)), anchor_random)
        v1 = [wv1.vectors[i] for i in anchors if wv1.words[i] not in exclude]
        v2 = [wv2.vectors[i] for i in anchors if wv2.words[i] not in exclude]
    elif anchor_indices is not None:
        v1 = [wv1.vectors[i] for i in anchor_indices if wv1.words[i] not in exclude]
        v2 = [wv2.vectors[i] for i in anchor_indices if wv2.words[i] not in exclude]
    elif anchor_words is not None:
        v1 = [wv1[w] for w in anchor_words if w not in exclude]
        v2 = [wv2[w] for w in anchor_words if w not in exclude]
    elif anchor_pairs is not None:
        v1 = []
        v2 = []
        for w1, w2 in anchor_pairs:
            if (w1 not in exclude) and (w2 not in exclude) and \
                (w1 in wv1) and (w2) in wv2:
                v1.append(wv1[w1])
                v2.append(wv2[w2])
    else:  # just use all words
        v1 = [wv1[w] for w in wv1.words if w not in exclude]
        v2 = [wv2[w] for w in wv2.words if w not in exclude]

    v1 = np.array(v1)
    v2 = np.array(v2)
    if method=="procrustes":  # align with OP
        Q, _ = orthogonal_procrustes(v1, v2)

    wv1_ = WordVectors(words=wv1.words, vectors=np.dot(wv1.vectors, Q))

    return wv1_, wv2, Q
