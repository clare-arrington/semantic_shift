#%%
from collections import defaultdict
from base_experiment import filter_targets, make_word_pairs, Target_Info
from shift_steps.alignment import align
from shift_steps.wordvectors import WordVectors, VectorVariations, load_wordvectors, intersection, extend_normal_with_sense
from plotting.neighbors import get_neighbor_coordinates, get_neighbors
from plotting.output_results import make_plot, print_results
import numpy as np
from dotenv import dotenv_values
import pickle, json
from pathlib import Path

#%%
def prep_vectors(
    align_wv, anchor_wv, dataset_name, 
    data_path, output_path, targets, slice_path='',
    norm=False):

    align_wv, anchor_wv = load_wordvectors(
        align_wv, anchor_wv, slice_path,
        f'{data_path}/word_vectors/{dataset_name}', normalize=norm)

    targets = filter_targets(targets, align_wv, anchor_wv)
    word_pairs = make_word_pairs(targets, align_wv, anchor_wv)

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    # print("Size of common vocab:", len(wv1))

    extended_wv1, extended_wv2 = extend_normal_with_sense(
        wv1, wv2, align_wv, anchor_wv, word_pairs.all_wps)
    # print(f"Size of WV after senses added: {len(wv1)} -> {len(extended_wv1)}" )

    with open(f'{output_path}/landmark_pairs.pkl' , 'rb') as pf:
        landmark_pairs = pickle.load(pf)
        # print(f'{len(landmark_pairs)} landmark pairs\nSome with senses:')
        # print(landmark_pairs[-3:])

    ## Align with subset of landmarks
    wv1_, wv2_, Q = align(extended_wv1, extended_wv2, anchor_pairs=landmark_pairs)
    # print("Size of aligned vocab:", len(wv1_))
    align_wv.partial_align_vec = wv1_
    anchor_wv.partial_align_vec = wv2_

    align_wv.post_align_vec = WordVectors(
        words=align_wv.normal_vec.words, 
        vectors=np.dot(align_wv.normal_vec.vectors, Q))
    anchor_wv.post_align_vec = anchor_wv.normal_vec

def handle_results(results, result_dict, sense_dict, wv_descs, 
    path, plot_results):
    Path(path).mkdir(parents=True, exist_ok=True)
    
    for sense, data in results.items():
        neighbors, categories, vectors = data
        for neighbor, category in zip(neighbors[1:], categories[1:]):
            # TODO: why was I checking first?
            # if neighbor not in result_dict[sense][category]:
            result_dict[sense][category].append(neighbor)

        if plot_results:
            x = get_neighbor_coordinates(vectors)
            make_plot(x, neighbors, categories, wv_descs, 
                    f'{path}/{sense}.html')

        # First 6 will be target + it's neighbors
        root = sense.split('_')[0]
        sense_label_key = (sense, wv_descs[0])
        neighbor_pairs = zip(neighbors[:6], vectors[:6])
        sense_dict[root][sense_label_key] = list(neighbor_pairs)
    
    return result_dict, sense_dict

def unpack_target_dict(target_dict):
    categories = []
    neighbors = []
    vectors = []
    for (sense, category), neighbor_pairs in target_dict.items():
        neighbors.append(sense)
        vectors.append(neighbor_pairs[0][1])
        categories.append(f'Target from {category}')
        
        for neighbor, vector in neighbor_pairs[1:]:
            neighbors.append(neighbor)
            vectors.append(vector)
            categories.append(category)

    return neighbors, categories, vectors

def run(
    anchor_wv, align_wv,
    anchor_result_dict,
    path,
    plot_results=True,
    ):

    wv_descs = [anchor_wv.desc, align_wv.desc]
    every_sense_dict = defaultdict(lambda: defaultdict(list))
    align_result_dict = defaultdict(lambda: defaultdict(list))

    # TODO: target is in some WV still ?
    results = get_neighbors(anchor_wv, align_wv, wv_descs)
    # here i want the senses and 
    anchor_result_dict, every_sense_dict = handle_results(
        results, anchor_result_dict, every_sense_dict, wv_descs, 
        f'{path}/plots/{anchor_wv.corpus_name}', 
        plot_results)

    # Swap it for the plots of anchor to align
    wv_descs.reverse()
    results = get_neighbors(align_wv, anchor_wv, wv_descs)
    align_result_dict, every_sense_dict = handle_results(
        results, align_result_dict, every_sense_dict, wv_descs, 
        f'{path}/plots/{align_wv.corpus_name}', 
        plot_results)

    if plot_results:
        plot_path = f'{path}/plots/all_senses' 
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        for root in every_sense_dict:
            nbors, cats, vecs = unpack_target_dict(
                every_sense_dict[root])
            x = get_neighbor_coordinates(vecs)
            # TODO: color printing will need to be diff
            make_plot(x, nbors, cats, 
                    wv_descs, f'{plot_path}/{root}.html')

    print('Done')
    return anchor_result_dict, align_result_dict

#%%
dataset_name = 'semeval'
data_path = dotenv_values(".env")['data_path']

with open(f"plotting/{dataset_name}.json", "r") as read_file:
    config = json.load(read_file)

with open(f"{data_path}/corpus_data/{dataset_name}/truth/{config['target_file']}") as f:
    targets = []
    for target in f.read().strip().split('\n'):
        target, label = target.split('\t')
        targets.append(Target_Info(target, target, label))

descs = config['corpora_desc']
anchor_name = config['anchor']
anchor_wv = VectorVariations(corpus_name = anchor_name,
            desc = descs[anchor_name], 
            type = 'sense')

output_path = f'{data_path}/align_results/{dataset_name}/anchor_{anchor_name}'

#%%            
source_names = []
anchor_result_dict = defaultdict(lambda: defaultdict(list))
for align_name in config['others']:
    align_wv = VectorVariations(corpus_name = align_name,
            desc = descs[align_name], 
            type = 'sense')

    source_names.append(align_wv.desc)
    align_path = f'{output_path}/align_{align_wv.corpus_name}'

    prep_vectors(align_wv, anchor_wv, dataset_name, data_path,
        f'{align_path}/BSA/s4_cosine', targets, norm=True) 

    anchor_result_dict, align_result_dict = run(
        anchor_wv, align_wv, anchor_result_dict, align_path,
        plot_results=False)

    print_results(align_result_dict, align_path, align_wv.desc, 
    [anchor_wv.desc])

#%%
print_results(anchor_result_dict, output_path, anchor_wv.desc, source_names)

# %%
