from temp.wordvectors import VectorVariations, WordVectors, intersection, extend_normal_with_sense
from temp.alignment import align
from temp.shift import s4, threshold_crossvalidation
from scipy.spatial.distance import cosine
from typing import Tuple, List
import numpy as np

class Train_Method_Info:
    def __init__(self, name, params, threshold=None): 
        self.name = name
        self.params = params
        self.threshold = threshold

    def __repr__(self):
        return f"Train_Method_Info('{self.name}', {self.params}, {self.threshold})"

## TODO: reconsider the files and naming conventions 
## Align Step
def align_vectors(
    align_method: Train_Method_Info, 
    word_pairs: Tuple[str, str], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations, 
    add_senses_in_alignment=True):

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    print("Size of common vocab:", len(wv1))

    ## Get landmarks
    print(f'Starting {align_method.name} aligning')
    if align_method.name == 'global':
        landmarks = list(wv1.word_id.values())

    elif align_method.name == 's4':
        if add_senses_in_alignment:
            align_wv.extended_vec, anchor_wv.extended_vec = extend_normal_with_sense(
                wv1, wv2, align_wv, anchor_wv, word_pairs)

            print(f"Size of align WV after senses added: {len(wv1)} -> {len(align_wv.extended_vec)}" )
            print(f"\nShould be {len(wv1)} + {len(word_pairs)} = {len(wv1) + len(word_pairs)}")
            print(f"Size of anchor WV after senses added: {len(wv2)} -> {len(anchor_wv.extended_vec)}" )
            print(f"\nShould be {len(wv2)} + {len(word_pairs)} = {len(wv2) + len(word_pairs)}")

            ## Align with subset of landmarks
            ## TODO: I need to get the landmark pair
            landmarks, _, landmark_pairs, Q = s4(
                wv1, wv2, align_wv.extended_vec, anchor_wv.extended_vec, **align_method.params)
            print('Done with S4 aligning')
            print(f"Check for any unwanted mutations: {len(wv1)}, {len(align_wv.extended_vec)}" )
            print(f"Check for any unwanted mutations: {len(wv2)}, {len(anchor_wv.extended_vec)}" )

            print(landmarks[:3])

            # sense_landmarks = [extended_wv1.words[i] for i in landmarks if '.' in extended_wv1.words[i]]
            # print(f"Senses in landmarks: {', '.join(sense_landmarks)}")
        
            wv1 = align_wv.extended_vec
            wv2 = anchor_wv.extended_vec
        else:
            ## Align with subset of landmarks
            landmarks, _, Q = s4(
                wv1, wv2, **align_method.params)
            print('Done with S4 aligning')
            print(f"Check for any unwanted mutations: {len(wv1)}" )

    ## Align
    print(f'{len(landmarks)} landmarks selected')
    wv1_, wv2_, Q = align(wv1, wv2, anchor_words=landmarks)
    print("Size of aligned vocab:", len(wv1_))
    align_wv.partial_align_vec = wv1_
    anchor_wv.partial_align_vec = wv2_

    landmark_terms = [wv1_.words[w] for w in landmarks]
    
    ## Align the original so that it matches wv1_ but has its full vocab
    align_wv.post_align_vec = WordVectors(
        words=align_wv.normal_vec.words, 
        vectors=np.dot(align_wv.normal_vec.vectors, Q))
    anchor_wv.post_align_vec = anchor_wv.normal_vec

    return landmarks, landmark_terms, align_wv, anchor_wv

## Classify Step
def get_target_distances(
        classify_methods: List[Train_Method_Info],
        word_pairs: Tuple[str, str], 
        landmarks: List[int],
        align_wv: VectorVariations, 
        anchor_wv: VectorVariations    
        ):

    ## If not sense, dists will be 1 to 1 for each target
    ## If there are senses it will be 1 to many for each target 
    target_dists = {}

    for classify_method in classify_methods:
        if classify_method.name == 'cosine':
            print('Starting cosine predicting')

            #if classify_method.threshold == 0:
            classify_method.threshold = threshold_crossvalidation(
                align_wv.partial_align_vec, anchor_wv.partial_align_vec, 
                **classify_method.params, landmarks=landmarks)

            dists = []
            for align_word, anchor_word in word_pairs:
                dist = cosine(align_wv.post_align_vec[align_word], 
                              anchor_wv.post_align_vec[anchor_word])
                dists.append(dist)

            target_dists['cosine'] = np.array(dists) 

        if classify_method.name == 's4':
            print('Starting S4 predicting')
            model = s4(align_wv.partial_align_vec, 
                       anchor_wv.partial_align_vec, 
                       landmarks=landmarks, 
                       **classify_method.params, 
                       update_landmarks=False)

            # Concatenate vectors of target words for prediction
            target_vectors = []
            for align_word, anchor_word in word_pairs:
                x = (align_wv.post_align_vec[align_word], 
                     anchor_wv.post_align_vec[anchor_word])
                target_vectors.append(np.concatenate(x))

            target_vectors = np.array(target_vectors)
            dists = model.predict(target_vectors).flatten()
            # print(f'Target vector size {dists.shape}')

            target_dists['s4'] = dists

    return target_dists, classify_methods
