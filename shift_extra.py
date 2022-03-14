#%%
from collections import defaultdict
from email.policy import default
from base_experiment import filter_targets, make_word_pairs, Target_Info
from shift_configs.us_uk import get_us_uk_targets
from shift_steps.alignment import align
from shift_steps.wordvectors import WordVectors, VectorVariations, load_wordvectors, intersection, extend_normal_with_sense
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import pickle

# TODO: why ID?
def find_neighbors( id, main_wv, other_wv, 
                    wv_names):

    categories = [f'Target from {wv_names[0]}']
    target = main_wv.words[id]
    words = [target]
    
    ## TODO: should this be run once?
    _, indices = perform_mapping(main_wv, main_wv, 
                                n_neighbors=20) 
    neighbors = indices[id]
    for i in neighbors:
        word = main_wv.words[i]
        if word != target:
            words.append(word)
            categories.append(wv_names[0])
        if len(words) == 11:
            break

    _, indices = perform_mapping(main_wv, other_wv)
    neighbors = indices[id]
    words += [other_wv.words[i] for i in neighbors]
    categories += [wv_names[1]] * len(neighbors)

    return words, categories 
    
def prep_vectors(
    align_wv, anchor_wv, dataset_name, 
    data_path, output_path, targets, slice_path='',
    norm=False):

    align_wv, anchor_wv = load_wordvectors(align_wv, anchor_wv, slice_path,
        f'{data_path}/word_vectors/{dataset_name}')

    if norm:
        align_wv.normal_vec.normalize()
        anchor_wv.normal_vec.normalize()

    targets = filter_targets(targets, align_wv, anchor_wv)
    all_word_pairs, target_word_pairs = make_word_pairs(targets,
        align_wv, anchor_wv)

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    print("Size of common vocab:", len(wv1))

    extended_wv1, extended_wv2 = extend_normal_with_sense(
        wv1, wv2, align_wv, anchor_wv, all_word_pairs)
    print(f"Size of WV after senses added: {len(wv1)} -> {len(extended_wv1)}" )

    with open(f'{output_path}/landmark_pairs.pkl' , 'rb') as pf:
        landmark_pairs = pickle.load(pf)
        print(len(landmark_pairs))
        print(landmark_pairs[-3:])

    # sense_landmarks = [lm for lm in landmark_terms if '.' in lm]

    ## Align with subset of landmarks
    wv1_, wv2_, Q = align(extended_wv1, extended_wv2, anchor_pairs=landmark_pairs)
    print("Size of aligned vocab:", len(wv1_))
    align_wv.partial_align_vec = wv1_
    anchor_wv.partial_align_vec = wv2_

    align_wv.post_align_vec = WordVectors(
        words=align_wv.normal_vec.words, 
        vectors=np.dot(align_wv.normal_vec.vectors, Q))
    anchor_wv.post_align_vec = anchor_wv.normal_vec

    return target_word_pairs
    # return align_wv, anchor_wv

def perform_mapping(wva, wvb, n_neighbors=10, metric="cosine"):
    """
    Given aligned wv_a and wv_b, performs mapping (translation) of words in a to those in b
    Returns (distances, indices) as n-sized lists of distances and the indices of the top neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=12, metric=metric).fit(wvb.vectors)
    distances, indices = nbrs.kneighbors(wva.vectors)

    return distances, indices

def get_neighbor_coordinates(x):
    """
    Apply decomposition to an input matrix and returns a 2d set of points.
    """
    return list(PCA(n_components=2).fit_transform(x))

def generic_run(
    data_path, dataset_name, 
    targets, align_wv, anchor_wv,
    result_dict,
    print_results=False
    ):

    output_path = f'{data_path}/align_results/{dataset_name}/anchor_{anchor_wv.corpus_name}/align_{align_wv.corpus_name}/BSA/s4_cosine'
    print(f'Going to save neighbors to {output_path}')

    target_word_pairs = prep_vectors(
        align_wv, anchor_wv, dataset_name, data_path,
        output_path, targets, norm=True) 

    wv1 = align_wv.post_align_vec
    wv2 = anchor_wv.post_align_vec
    sense_wv_names = [align_wv.desc, anchor_wv.desc]
    target_wv_names = [anchor_wv.desc, align_wv.desc]

    # get_neighbors_older(wv1, wv2, target_word_pairs, sense_wv_names, 
    #                 output_path, print_results, result_dict)
    # get_neighbors_older(wv2, wv1, target_word_pairs, target_wv_names,
    #                 output_path, print_results, result_dict)

    get_neighbors(wv2, wv1, target_word_pairs, target_wv_names, result_dict)

    print('Done')
    if not print_results:
        return result_dict

def get_neighbors(main_wv, other_wv, target_word_pairs, wv_names, result_dict):
    targets = set()
    for _, sense in tqdm(target_word_pairs):
        if sense in targets:
            continue
        targets.add(sense)
        id = main_wv.word_id[sense]
        neighbors, categories = find_neighbors(id, main_wv, other_wv, wv_names)

        for neighbor, category in zip(neighbors[1:], categories[1:]):
            if neighbor not in result_dict[sense][category]:
                result_dict[sense][category].append(neighbor)

def get_neighbors_older(main_wv, other_wv, target_word_pairs, wv_names, output_path, print_results, result_dict):
    if print_results:
        with open(output_path+'/neighbors.txt', 'w') as f:
            print(f'=========== Targets from {wv_names[0]} ===========', file=f)
    
    targets = set()
    for _, sense in tqdm(target_word_pairs):
        if sense in targets:
            continue
        targets.add(sense)
        id = main_wv.word_id[sense]
        neighbors, categories = find_neighbors(id, main_wv, other_wv, wv_names)

    if print_results:
        print_neighbors(neighbors, categories, output_path+'/neighbors.txt')
    else:
        result_dict['_'.join(wv_names)] = (neighbors, categories)

def print_neighbors(neighbors, categories, path):
    with open(path, 'a') as f:
        print(f'{neighbors[0]} ', file=f, end='')
        category = ''
        for neighbor, c in zip(neighbors[1:], categories[1:]):
            if c != category:
                category = c
                print(f'\n\n\t=== {category} neighbors ===\n\t\t', file=f, end='')
            print(neighbor, file=f, end=', ')
        print('\n\n', file=f)

def print_results(result_dict, output_path, anchor_name, source_names):
    with open(output_path, 'w') as f:
        print(f'=========== Targets from {anchor_name} ===========', file=f)
    
        for target in sorted(list(result_dict.keys())):
            print(f'{target} ', file=f, end='')

            source_neighbors = result_dict[target]

            print(f'\n\n\t=== Target neighbors from {anchor_name} ===\n\t\t', file=f, end='')
            print(', '.join(source_neighbors[anchor_name]), file=f)

            for source in source_names:
                print(f'\n\t=== {source} neighbors ===\n\t\t', file=f, end='')
                print(', '.join(source_neighbors[source]), file=f)

            print('\n', file=f)

def prep_cs(align_name=None, anchor_name=None):
    
    ### Set align wv
    if align_name == '1800s':
        align_wv = VectorVariations(corpus_name = '1800s',
                    desc = '1810 - 1860', 
                    type = 'sense')
    elif align_name == '2000s':
        align_wv = VectorVariations(corpus_name = '2000s',
            desc = '1960 - 2010', 
            type = 'sense')
    elif align_name == 'coca':
        align_wv = VectorVariations(corpus_name = 'coca',
            desc = '1990 - 2010', 
            type = 'sense')
    else:
        align_wv = VectorVariations(corpus_name = 'ai',
            desc = 'ArXiv AI', 
            type = 'sense')

    ### Set align wv
    if anchor_name == 'ai':
        anchor_wv = VectorVariations(corpus_name = 'ai',
            desc = 'ArXiv AI', 
            type = 'sense')
    else:
        anchor_wv = VectorVariations(corpus_name = 'coca',
            desc = '1990 - 2010', 
            type = 'sense')

    return align_wv, anchor_wv

#%%
# def run_all_combos(anchor:str, others:List[str]):

dataset_name = 'time'
data_path = '/home/clare/Data'
anchor = 'coca'
others = ['1800s', '2000s']

## TODO: from some insane reason sorting these targets mess up filtering??
with open(f'{data_path}/corpus_data/time/targets.txt') as f:
    targets = [Target_Info(t, t, True) for t in f.read().split()]

source_names = []
result_dict = defaultdict(lambda: defaultdict(list))
for align_name in others:
    align_wv, anchor_wv = prep_cs(align_name=align_name, anchor_name=anchor)
    source_names.append(align_wv.desc)

    result_dict = generic_run(data_path, dataset_name, 
        targets, align_wv, anchor_wv, result_dict)

output_path = f'{data_path}/align_results/{dataset_name}/anchor_{anchor_wv.corpus_name}/neighbors.txt'
print_results(result_dict, output_path, anchor_wv.desc, source_names)

# %%
