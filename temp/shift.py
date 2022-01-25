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
# Local modules
from temp.wordvectors import WordVectors, extend_normal_with_sense
from temp.alignment import align

# Third party modules
from scipy.spatial.distance import cosine, euclidean
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

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

## TODO: rename alt_vec
def inject_change_single(wv, target_vec, alt_vec, 
                         rate, max_tries=50):

    dist_threshold = cosine(target_vec, alt_vec)

    cos_dist = dist_threshold 
    tries = 0
    while cos_dist <= dist_threshold and tries < max_tries:
        tries += 1
        word = np.random.choice(wv.words)  
        new_vec = alt_vec + (rate * wv[word])

        cos_dist = cosine(target_vec, new_vec)

    return new_vec

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

def run_iteration(
    wv1, wv2_original, 
    landmarks, non_landmarks,
    n_targets, n_negatives, 
    rate, use_cosine=False):

    # Randomly sample words to inject change to
    # If no word is flagged as non_landmarks, sample from all words
    # In practice, this should never occur when selecting landmarks
    # but only for classification when aligning on all words
    if len(non_landmarks) > 0:
        pos_samples = np.random.choice(non_landmarks, n_targets)
        # Make targets deterministic
        # pos_samples = non_landmarks
    else:
        nl = list(wv1.word_id.values())
        pos_samples = np.random.choice(nl, n_targets)

    pos_vectors = dict()
    for target in pos_samples:
        # Simulate semantic change in target word
        pos_vectors[target] = inject_change_single(
            wv2_original, wv1[target], wv2_original[target], rate)

    # Get negative samples from landmarks
    neg_samples = np.random.choice(list(landmarks), n_negatives, p=None)
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
    
    if use_cosine:
        # Calculate cosine distance of training samples
        x_train = np.array([cosine(wv1[w], sup_vectors[w]) for w in train_words])
    else:
        # Append the two samples together
        x_train = np.array([np.append(wv1[w], sup_vectors[w]) for w in train_words])
    
    return x_train, y_train

def threshold_crossvalidation(wv1, wv2, iters=100,
                                        n_fold=1,
                                        n_targets=100,
                                        n_negatives=100,
                                        rate=0.5,
                                        landmarks=None):
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
    # landmarks = [id for w, id in wv1.word_id.items() if w in landmarks]
    non_landmarks = [id for id in wv1.word_id.values() if id not in landmarks]

    for iter in tqdm(range(iters)):
        x_train, y_train = run_iteration(
            wv1, wv2_original, landmarks, non_landmarks,
            n_targets, n_negatives, rate, use_cosine=True)

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
    wv1_, wv2_, Q = align(wv1, wv2)  # start from intersected alignment
    
    landmark_dists = [euclidean(u, v) for u, v in zip(
        wv1_.vectors, wv2_.vectors)]
    landmark_args = np.argsort(landmark_dists)
    cutoff = int(len(wv1_.words) * 0.5)
    # landmarks = [wv1.words[i] for i in landmark_args[:cutoff]]

    return landmark_args[:cutoff]

def s4(wv1, wv2, extended_wv1=None, extended_wv2=None,
          verbose=False, plot=False, 
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
    # verbose=False
    # plot=False
    # iters=100
    # n_targets=100
    # n_negatives=50
    # rate=1
    # threshold=0.5
    # t_overlap=1
    # update_landmarks=True
    # return_model=False
    # debug=False

    # Define verbose prints
    if verbose:
        def verbose_print(*s, end="\n"):
            print(*s, end=end)
    else:
        def verbose_print(*s, end="\n"):
            return None

    # Begin alignment
    if landmarks is None:
        landmarks = get_initial_landmarks(wv1, wv2)
     
    ## TODO: might have to add a word to ID conversion if they pass in landmarks
    avg_window = 0  # number of iterations to use in running average
    # landmark_set = set(landmarks)

    if extended_wv1 is not None:
        wv1 = extended_wv1
        wv2 = extended_wv2

    wv2_original = WordVectors(words=wv2.words, vectors=wv2.vectors.copy())
    non_landmarks = [id for id in wv1.word_id.values() if id not in landmarks]
    
    wv1, wv2, Q = align(wv1, wv2, anchor_words=landmarks)

    model = build_keras_model(wv1.dimension*2)

    # General set of histories (iterative data)
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

    for iter in tqdm(range(iters)):
        x_train, y_train = run_iteration(
            wv1, wv2_original, landmarks,
            non_landmarks, n_targets, n_negatives, rate)

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
        loss, accuracy = model.train_on_batch(x_train, y_train, reset_metrics=False)
        hist["loss"].append(loss)

        # Apply model on original data to select landmarks
        x_real = np.array([np.append(u, v) for u, v
                            in zip(wv1.vectors, wv2_original.vectors)])
        predict_real = model.predict(x_real)

        # Update landmark overlap using Jaccard Index
        prev_landmarks = set(landmarks)
        if update_landmarks:
            landmarks = []
            non_landmarks = []
            for id, prediction in zip(wv1.word_id.values(), predict_real):
                if prediction < threshold:
                    landmarks.append(id)
                else:
                    non_landmarks.append(id)

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


# %%
