#%%
"""Implements self supervised semantic shift functions
It uses poisoning attacks to learn landmarks in a self-supervised way
At each iteration, generate perturbation on the data, generating positive
and negative samples
Learn the separation between them (using any classifier)
Apply the classifier to the original (non-perturbated) data
Negatives -> landmarks
Positives -> semantically changed
We can begin by aligning on all words, and then learn better landmarks from
there. Alternatively, one can start from random landmarks."""


# Third party modules
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, log_loss
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow import keras

# Local modules
from temp.wordvectors import WordVectors, intersection
from temp.alignment import align

# Initialize random seeds
np.random.seed(1)
tf.random.set_seed(1)

## Does this remove the NUMA node warning?
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def plot_s4_results(iter, num_words, debug, 
                    hist, align_hist, cos_hist):
        iter += 1  # add one to iter for plotting
        plt.plot(range(iter), hist["num_landmarks"], label="landmarks")
        plt.hlines(num_words, 0, iter, colors="red")
        plt.ylabel("No. of landmarks")
        plt.xlabel("Iteration")
        plt.show()
        plt.plot(range(iter), hist["loss"], c="red", label="loss")
        plt.ylabel("Loss (binary crossentropy)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.show()
        plt.plot(range(iter), align_hist["cml_loss"], label="in (landmarks)")
        plt.plot(range(iter), align_hist["cml_out"], label="out")
        plt.plot(range(iter), align_hist["all"], label="all")
        plt.ylabel("Alignment loss (MSE)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.show()

        if debug:
            plt.plot(range(iter), cos_hist["cml_in"], label="cos in")
            plt.plot(range(iter), cos_hist["cml_out"], label="cos out")
            plt.legend()
            plt.show()

        plt.plot(range(iter), hist["cml_overlap"], label="overlap")

        plt.ylabel("Jaccard Index", fontsize=16)
        plt.xlabel("Iteration", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # plt.legend()
        plt.tight_layout()
        plt.savefig("overlap.pdf", format="pdf")
        #plt.show()

def inject_change_single(wv, w, words, v_a, alpha, replace=False,
                         max_tries=50):
    """
    Injects change to word w in wv by randomly selecting a word t in wv
    and injecting the sense of t in to w.
    The modified vector of w must have a higher cosine distance to v_a
    than its original version. This is done by sampling t while the cosine of
    w is not greater than that of v_a and wv(w) or until a max_tries.
    v_a is the vector of word w in the parallel corpus (not wv).

    Arguments:
            wv      -   WordVectors of the corpus to be modified
            w       -   (str) Word to be modified
            words   -   (list) Pool of words to sample from, injecting sense
            v_a     -   (np.ndarray) word vector of w in the source parallel to wv
            alpha   -   (float) Rate of injected change
            replace -   (bool) Whether to replace w with t instead of 'moving' w towards t
    Returns:
            x       -   (np.ndarray) modified vector of w
    """
    cos_t = cosine(v_a, wv[w])  # cosine distance threshold we want to surpass

    c = 0
    tries = 0
    w_id = wv.word_id[w]
    v_b = np.copy(wv.vectors[w_id])
    while c < cos_t and tries < max_tries:
        tries += 1
        selected = np.random.choice(words)  # select word with new sense
        if not replace:
            b = wv[w] + alpha*wv[selected]
            v_b = b
        else:
            v_b = wv[selected]

        c = cosine(v_a, v_b)

    return v_b


def inject_change_batch(wv, changes, alpha, replace=True):
    """
    Given a WordVectors object and a list of words, perform fast injection
    of semantic change by using the update rule from Word2Vec
    wv - WordVectors (input)
    changes - list of n tuples (a, b) that drives the change such that b->a
          i.e.: simulates using b in the contexts of a
    alpha - degree in which to inject the change
              if scalar: apply same alpha to every pair
              if array-like: requires size n, specifies individual alpha values
                              for each pair
    replace  - (bool) if True, words are replaced instead of moved
                e.g.: if pair is (dog, car), then v_car <- v_dog
    Returns a WordVectors object with the change
    """
    wv_new = WordVectors(words=wv.words, vectors=np.copy(wv.vectors))
    for i, pair in enumerate(changes):
        t, w = pair
        t_i = wv.word_id[t]  # target word
        w_i = wv.word_id[w]  # modified word
        # Update vector with alpha and score
        # Higher score means vectors are already close, thus apply less change
        # Alpha controls the rate of change
        if not replace:
            b = wv_new[w] + alpha*(1)*wv[t]
            wv_new.vectors[w_i] = b
        else:
            wv_new.vectors[w_i] = wv[t]
        # print("norm b", np.linalg.norm(b))
    return wv_new


def get_features(x, names=["cos"]):
    """
    Compute features given input training data (concatenated vectors)
    Default features is cosine. Accepted features: cosine (cos).
    Attributes:
            x   - size n input training data as concatenated word vectors
            names - size d list of features to compute
    Returns:
            n x d feature matrix (floats)
    """
    x_out = np.zeros((len(x), len(names)), dtype=float)
    for i, p in enumerate(x):
        for j, feat in enumerate(names):
            if feat == "cos":
                x_ = cosine(p[:len(p)//2], p[len(p)//2:])
                x_out[i][j] = x_
    return x_out

def get_loss(v1, v2):
    vector_diff = np.linalg.norm(v1 - v2)
    return vector_diff**2 / len(v1)

def build_sklearn_model():
    """
    Build SVM using sklearn model
    The model uses an RBF kernel and the features are given by difference
    between input vectors u-v.
    Return: sklearn SVC
    """
    model = SVC(random_state=0, probability=True)
    return model


def build_keras_model(dim):
    """
    Builds the keras model to be used in self-supervision.
    Return: Keras-Tensorflow2 model
    """
    h1_dim = 100
    h2_dim = 100
    model = keras.Sequential([
                             keras.layers.InputLayer(input_shape=(dim)),
                             keras.layers.Dense(h1_dim, activation="relu",
                                                activity_regularizer=keras.regularizers.l2(1e-2)),
                             # keras.layers.Dense(h2_dim, activation="relu",
                             #                    activity_regularizer=keras.regularizers.l2(1e-2)),
                             keras.layers.Dense(1, activation="sigmoid")
                            ])
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

## TODO: this feels pretty redundant to things in s4
def threshold_crossvalidation(wv1, wv2, iters=100,
                                        n_fold=1,
                                        n_targets=100,
                                        n_negatives=100,
                                        fast=True,
                                        rate=0.5,
                                        t=0.5,
                                        landmarks=None,
                                        t_overlap=1,
                                        debug=False):
    """
    Runs crossvalidation over self-supervised samples, carrying out a model
    selection to determine the best cosine threshold to use in the final
    prediction.

    Arguments:
        wv1, wv2    - input WordVectors - required to be intersected and ALIGNED before call
        plot        - 1: plot functions in the end 0: do not plot
        iters       - max no. of iterations
        n_fold      - n-fold crossvalidation (1 - leave one out, 10 - 10-fold cv, etc.)
        n_targets   - number of positive samples to generate
        n_negatives - number of negative samples
        fast        - use fast semantic change simulation
        rate        - rate of semantic change injection
        t           - classificaiton threshold (0.5)
        t_overlap   - overlap threshold for (stop criterion)
        landmarks   - list of words to use as landmarks (classification only)
        debug       - toggles debugging mode on/off. Provides reports on several metrics. Slower.
    Returns:
        t - selected cosine threshold t
    """

    wv2_original = WordVectors(words=wv2.words, vectors=wv2.vectors.copy())
    landmark_set = set(landmarks)
    non_landmarks = [w for w in wv1.words if w not in landmark_set]

    for iter in range(iters):

        replace = dict()  # replacement dictionary
        pos_samples = list()
        pos_vectors = dict()

        # Randomly sample words to inject change to
        # If no word is flagged as non_landmarks, sample from all words
        # In practice, this should never occur when selecting landmarks
        # but only for classification when aligning on all words
        if len(non_landmarks) > 0:
            targets = np.random.choice(non_landmarks, n_targets)
            # Make targets deterministic
            #targets = non_landmarks
        else:
            targets = np.random.choice(wv1.words, n_targets)

        for target in targets:

            # Simulate semantic change in target word
            v = inject_change_single(wv2_original, target, wv1.words,
                                     wv1[target], rate)

            pos_vectors[target] = v

            pos_samples.append(target)
        # Convert to numpy array
        pos_samples = np.array(pos_samples)
        # Get negative samples from landmarks

        neg_samples = np.random.choice(landmarks, n_negatives, p=None)
        neg_vectors = {w: wv2_original[w] for w in neg_samples}
        # Create dictionary of supervision samples (positive and negative)
        # Mapping word -> vector
        sup_vectors = {**neg_vectors, **pos_vectors}

        # Prepare training data
        train_words = np.concatenate((pos_samples, neg_samples))
        # assign labels to positive and negative samples
        y_train = [1] * len(pos_samples) + [0] * len(neg_samples)

        # Stack columns to shuffle data and labels together
        train = np.column_stack((train_words, y_train))
        # Shuffle batch
        np.random.shuffle(train)
        # Detach data and labels
        train_words = train[:, 0]
        y_train = train[:, -1].astype(int)

        # Calculate cosine distance of training samples
        x_train = np.array([cosine(wv1[w], sup_vectors[w]) for w in train_words])

        # t_pool = [0.2, 0.7]
        t_pool = np.arange(0.2, 1, 0.1)

        best_acc = 0
        best_t = 0
        for t_ in t_pool:
            acc = 0
            for i in range(0, len(x_train), n_fold):
                x_cv = x_train[i:i+n_fold]
                y_true = y_train[i:i+n_fold]
                y_hat = x_cv > t_
                acc += sum(y_hat == y_true)/len(x_cv)
            acc = acc/(len(x_train)//n_fold)
            if acc > best_acc:
                best_acc = acc
                best_t = t_
                #print(f"- New best t: {t_:.2f}, {int(acc*100)}%")

    return best_t

#%%

def get_initial_landmarks(wv1, wv2):
    wv1, wv2, Q = align(wv1, wv2)  # start form global alignment
    
    ## TODO: here we do a direct matching; is this where the subroutine would happen?
    landmark_dists = [euclidean(u, v) for u, v in zip(wv1.vectors, wv2.vectors)]
    
    landmark_args = np.argsort(landmark_dists)
    cutoff = int(len(wv1.words) * 0.5)
    landmarks = [wv1.words[i] for i in landmark_args[:cutoff]]

    return landmarks

def s4(wv1, wv2, verbose=False, plot=False, 
                            cls_model="nn",
                            iters=100,
                            n_targets=10,
                            n_negatives=10,
                            rate=0,
                            threshold=0.5,
                            t_overlap=1,
                            landmarks=None,
                            update_landmarks=True,
                            return_model=False,
                            debug=False):
    """
    Performs self-supervised learning of semantic change.
    Generates negative samples by sampling from landmarks.
    Generates positive samples via simulation of semantic change on random non-landmark words.
    Trains a classifier, fine-tune it across multiple iterations.
    If update_landmarks is True, then it learns landmarks from that step. In this case,
    the returned values are landmarks, non_landmarks, Q (transform matrix)
    Otherwise, landmarks are fixed from a starting set and the returned value
    is the learned classifier - landmarks must be passed.
    Arguments:
        wv1, wv2    - input WordVectors - required to be intersected before call
        verbose     - 1: display log, 0: quiet
        plot        - 1: plot functions in the end 0: do not plot
        cls_model   - classification model to use {"nn", "svm_auto", "svm_features"}
        iters       - max no. of iterations
        n_targets   - number of positive samples to generate
        n_negatives - number of negative samples
        rate        - rate of semantic change injection
        threshold           - classificaiton threshold (0.5)
        t_overlap   - overlap threshold for (stop criterion)
        landmarks   - list of words to use as landmarks (classification only)
        update_landmarks - if True, learns landmarks. Otherwise, learns classification model.
        debug       - toggles debugging mode on/off. Provides reports on several metrics. Slower.
    Returns:
        if update_landmarks is True:
            landmarks - list of landmark words
            non_landmarks - list of non_landmark words
            Q           - transformation matrix for procrustes alignment
        if update_landmarks is False:
            model       - binary classifier
    """

    plot=False
    iters=100
    n_targets=100
    n_negatives=50
    rate=1
    threshold=0.5
    t_overlap=1
    update_landmarks=True
    debug=False

    # Define verbose prints
    if verbose:
        def verbose_print(*s, end="\n"):
            print(*s, end=end)
    else:
        def verbose_print(*s, end="\n"):
            return None

    wv2_original = WordVectors(words=wv2.words, vectors=wv2.vectors.copy())

    avg_window = 0  # number of iterations to use in running average

    # Begin alignment
    if update_landmarks:
        landmarks = get_initial_landmarks(wv1, wv2)

    landmark_set = set(landmarks)
    non_landmarks = [w for w in wv1.words if w not in landmark_set]

    ## Align with subset of landmarks
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)

    if cls_model == "nn":
        model = build_keras_model(wv1.dimension*2)
    elif cls_model == "svm_auto" or cls_model == "svm_features":
        model = build_sklearn_model()  # get SVC

    # General set of histories
    hist = {
        "num_landmarks"  : [],   # store no. of landmark history
        "loss"   : [],   # store self-supervision loss history
        "overlap"   : [],   # store landmark overlap history
        "cml_overlap" : [] # mean overlap history
    }

    # Histories specific to alignment
    align_hist = {
        "loss": [],     # store landmark alignment loss (landmarks)
        "cml_loss" : [], # store cumulative loss alignment
        "out" : [],     # store alignment loss outside of landmarks (non-landmarks)
        "cml_out" : [],
        "all" : []
    }

    # Histories specific to cosine
    cos_hist = {
        "in"  : [],
        "cml_in"   : [], #cumulative
        "out" : [],
        "cml_out"  : []
    }

    for iter in range(iters):

        prev_landmarks = set(landmarks)

        # Randomly sample words to inject change to
        # If no word is flagged as non_landmarks, sample from all words
        # In practice, the second case should never occur when selecting landmarks;
        # only for classification when aligning on all words
        if len(non_landmarks) > 0:
            pos_samples = np.random.choice(non_landmarks, n_targets)
            # Make targets deterministic
            #pos_samples = non_landmarks
        else:
            pos_samples = np.random.choice(wv1.words, n_targets)

        pos_vectors = dict()
        for target in pos_samples:
            # Simulate semantic change in target word
            pos_vectors[target] = inject_change_single(wv2_original, target, 
                                        wv1.words, wv1[target], rate)

        # Get negative samples from landmarks
        ## TODO: need to sample without replace otherwise count mismatch b/n both
        neg_samples = np.random.choice(landmarks, n_negatives, p=None, replace=False)
        neg_vectors = {w: wv2_original[w] for w in neg_samples}

        # Prepare training data
        train_words = np.concatenate((pos_samples, neg_samples))
        # Assign labels to positive and negative samples
        y_train = [1] * len(pos_samples) + [0] * len(neg_samples)

        # Shuffle data and labels together, then split
        train = np.column_stack((train_words, y_train))
        np.random.shuffle(train)

        train_words = train[:, 0]
        y_train = train[:, -1].astype(int)

        # Create dictionary of supervision samples (positive and negative)
        # Mapping word -> vector
        sup_vectors = {**neg_vectors, **pos_vectors}
        x_train = np.array([np.append(wv1[w], sup_vectors[w]) for w in train_words])

        # Append history
        hist["num_landmarks"].append(len(landmarks))
        v1_lm_vectors = np.array([wv1[w] for w in landmarks])
        v2_lm_vectors = np.array([wv2_original[w] for w in landmarks])

        alignment_loss = get_loss(v1_lm_vectors, v2_lm_vectors)
        align_hist["loss"].append(alignment_loss)

        cml_align_loss = np.mean(align_hist["loss"][-avg_window:])
        align_hist["cml_loss"].append(cml_align_loss)

        # out loss
        if len(non_landmarks) == 0:
            align_out_loss = 0
        else:
            v1_non_lm_vectors = np.array([wv1[w] for w in non_landmarks])
            v2_non_lm_vectors = np.array([wv2_original[w] for w in non_landmarks])
            align_out_loss = get_loss(v1_non_lm_vectors, v2_non_lm_vectors)
        
        align_hist["out"].append(align_out_loss)

        cml_out_loss = np.mean(align_hist["out"][-avg_window:])
        align_hist["cml_out"].append(cml_out_loss)

        # all loss
        align_all_loss = get_loss(wv1.vectors, wv2_original.vectors)
        align_hist["all"].append(align_all_loss)

        if debug:
            # cosine loss
            cos_in = np.mean([cosine(u, v) for u, v in zip (v1_lm_vectors, v2_lm_vectors)])
            cos_out = np.mean([cosine(u, v) for u, v in zip(v1_non_lm_vectors, v2_non_lm_vectors)])
            cos_hist["in"].append(cos_in)
            cos_hist["out"].append(cos_out)
            cos_hist["cml_in"].append(np.mean(cos_hist["in"]))
            cos_hist["cml_out"].append(np.mean(cos_hist["out"]))

        # Begin training of neural network
        if cls_model == "nn":
            loss, accuracy = model.train_on_batch(x_train, y_train, reset_metrics=False)
        elif cls_model == "svm_auto":
            model.fit(x_train, y_train)
            pred_train = model.predict_proba(x_train)
            history = [log_loss(y_train, pred_train)]
        elif cls_model == "svm_features":
            x_train_ = get_features(x_train)  # retrieve manual features
            model.fit(x_train_, y_train)
            pred_train = model.predict_proba(x_train_)
            y_hat_t = (pred_train[:, 0] > 0.5)
            acc_t = accuracy_score(y_train, y_hat_t)
            history = [log_loss(y_train, pred_train), acc_t]

        hist["loss"].append(loss)

        # Apply model on original data to select landmarks
        x_real = np.array([np.append(u, v) for u, v
                            in zip(wv1.vectors, wv2_original.vectors)])
        if cls_model == "nn":
            predict_real = model.predict(x_real)
        elif cls_model == "svm_auto":
            predict_real = model.predict_proba(x_real)
            predict_real = predict_real[:, 1]
        elif cls_model == "svm_features":
            x_real_ = get_features(x_real)
            predict_real = model.predict_proba(x_real_)
            predict_real = predict_real[:, 1]

        if update_landmarks:
            landmarks = []
            non_landmarks = []
            for word, prediction in zip(wv1.words, predict_real):
                if prediction < threshold:
                    landmarks.append(word)
                else:
                    non_landmarks.append(word)

        # Update landmark overlap using Jaccard Index
        isect_ab = set.intersection(prev_landmarks, set(landmarks))
        union_ab = set.union(prev_landmarks, set(landmarks))
        j_index = len(isect_ab)/len(union_ab)
        hist["overlap"].append(j_index)

        cml_overlap = np.mean(hist["overlap"][-avg_window:])
        hist["cml_overlap"].append(cml_overlap)  

        verbose_print("> %3d | L %4d | l(in): %.2f | l(out): %.2f | loss: %.2f | overlap %.2f | acc: %.2f" %
                        (iter, len(landmarks), align_hist["cml_loss"][-1],
                         align_hist["cml_out"][-1], loss, hist["cml_overlap"][-1], accuracy),
                         end="\r")

        wv1, wv2_original, Q = align(wv1, wv2_original, anchor_words=landmarks)

        # Check if overlap difference is below threhsold
        if np.mean(hist["overlap"]) > t_overlap:
            break

    if plot:
        plot_s4_results(iter, len(wv1.words), debug, hist, align_hist, cos_hist)
    
    if not update_landmarks:
        return model
    elif return_model:
        return landmarks, non_landmarks, Q, model
    else:
        return landmarks, non_landmarks, Q 

def main():
    """
    Runs main experiments using self supervised alignment.
    """
    # wv_source = "wordvectors/latin/corpus1/0.vec"
    # wv_target = "wordvectors/latin/corpus2/0.vec"
    # wv_source = "wordvectors/source/theguardianuk.vec"
    # wv_target = "wordvectors/source/thenewyorktimes_1.vec"
    wv_source = "wordvectors/semeval/latin-corpus1.vec"
    wv_target = "wordvectors/semeval/latin-corpus2.vec"
    # wv_source = "wordvectors/usuk/bnc.vec"
    # wv_target = "wordvectors/usuk/coca_mag.vec"
    # wv_source = "wordvectors/artificial/NYT-0.vec"
    # wv_target = "wordvectors/artificial/NYT-500_random.vec"
    plt.style.use("seaborn")

    # Read WordVectors
    normalized = False
    wv1 = WordVectors(input_file=wv_source, normalized=normalized)
    wv2 = WordVectors(input_file=wv_target, normalized=normalized)

    wv1, wv2 = intersection(wv1, wv2)

    landmarks, non_landmarks, Q = s4(wv1, wv2,
                                                            cls_model="nn",
                                                            n_targets=100,
                                                            n_negatives=100,
                                                            rate=1,
                                                            t=0.5,
                                                            iters=100,
                                                            verbose=1,
                                                            plot=1)
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)
    d_l = [cosine(wv1[w], wv2[w]) for w in landmarks]
    d_n = [cosine(wv1[w], wv2[w]) for w in non_landmarks]
    sns.distplot(d_l, color="blue")
    sns.distplot(d_n, color="red")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
