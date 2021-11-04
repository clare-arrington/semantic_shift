#%%
from base_experiment import make_word_pairs, Target_Info
from temp.alignment import align
from temp.wordvectors import WordVectors, VectorVariations, load_wordvectors, intersection, extend_normal_with_sense
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import os

# os.chdir('..')
def make_plot(x, words, categories, 
              wv_names, path, flip=False):
    data = []
    for x_i, word, category in zip(x, words, categories):
        data.append([
            x_i[0], x_i[1], word, category, 12
        ])

    df = pd.DataFrame(data, columns=['x','y','name','label','size'])

    layout = {
        "paper_bgcolor": "#FAFAFA",
        "plot_bgcolor": "#DDDDDD",
        "dragmode": "pan",
        'font': {
            'family': "Courier New, monospace",
            'size': 13
        },
        'margin': {
            'l': 60,
            'r': 40,
            'b': 40,
            't': 40,
            'pad': 4
        },
        'xaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'yaxis': {
            "showgrid": True,
            "zeroline": False,
            "visible": True,
            "title": ''
        },
        'legend': {
            "title":''
        },
        'title': {
        }
    }

    symbols = ["square", "circle"]
    colors = ["#84b000", "#764e80"]
    if flip:
        symbols.reverse()
        colors.reverse()

    colors.append("#7b2514")
    symbols.append(symbols[0])

    fig = px.scatter(df, x='x', y='y', 
        color='label', symbol='label', text='name',
        hover_name="name", size="size",
        hover_data={"label":False,
                    "name":False,
                    "x":False, 
                    "y":False},
        symbol_map={wv_names[0]: symbols[0], 
                    wv_names[1]: symbols[1],
                    f'Target from {wv_names[0]}': symbols[2]},
        color_discrete_map={wv_names[0]: colors[0], 
                            wv_names[1]: colors[1],
                            f'Target from {wv_names[0]}': colors[2]
                            }
    )
    fig.update_layout(**layout)
    fig.update_traces(textposition='top center',
    textfont={'family': "Raleway, sans-serif" }
    )
    # fig.show()
    fig.write_html(path)

def plot_alignment( id, main_wv, other_wv, 
                    wv_names, path, 
                    flip=False, count=1):

    vecs, words = [], []
    categories = [f'Target from {wv_names[0]}']
    
    _, indices = perform_mapping(main_wv, main_wv, 
                                n_neighbors=10+count) 
    neighbors = indices[id][count-1:]
    vecs += [main_wv[i] for i in neighbors]
    words += [main_wv.words[i] for i in neighbors]
    categories += [wv_names[0]] * (len(neighbors) - 1)
    # importance += ['non_target'] * len(neighbors) - 1

    _, indices = perform_mapping(main_wv, other_wv)
    neighbors = indices[id]
    vecs += [other_wv[i] for i in neighbors]
    words += [other_wv.words[i] for i in neighbors]
    categories += [wv_names[1]] * len(neighbors)
    # importance += ['non_target'] * len(neighbors) 

    x = get_neighbor_coordinates(vecs)
    make_plot(x, words, categories, wv_names, path, flip)

def prep_vectors(align_wv, anchor_wv, data_path, targets):
    align_wv, anchor_wv = load_wordvectors(align_wv, anchor_wv,
        f'{data_path}/word_vectors/{dataset_name}')
    all_word_pairs, target_word_pairs = make_word_pairs(
        align_wv.type, align_wv.normal_vec.words, targets)

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    print("Size of common vocab:", len(wv1))

    extended_wv1, extended_wv2 = extend_normal_with_sense(
        wv1, wv2, align_wv, anchor_wv, all_word_pairs)
    print(f"Size of WV after senses added: {len(wv1)} -> {len(extended_wv1)}" )

    output_path = f'{data_path}/align_results/semeval/align_ccoha1/sense/s4_cosine/'
    with open(f'{output_path}/landmarks.pkl' , 'rb') as pf:
        landmark_terms = pickle.load(pf)
        print(len(landmark_terms))

    landmarks = [extended_wv1.word_id[word] for word in landmark_terms]
    # sense_landmarks = [lm for lm in landmark_terms if '.' in lm]

    ## Align with subset of landmarks
    wv1_, wv2_, Q = align(extended_wv1, extended_wv2, anchor_words=landmarks)
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

#%%
data_path = '/home/clare/Data'
dataset_name = 'semeval'
targets = []
with open(f'{data_path}/corpus_data/semeval/truth/binary.txt') as fin:
    og_targets = fin.read().strip().split('\n')
    for target in og_targets:
        target, label = target.split('\t')
        label = bool(int(label))
        word, pos = target.split('_')

        target = Target_Info(word=word, shifted_word=target, is_shifted=label)
        targets.append(target)

align_wv = VectorVariations(corpus_name = 'ccoha1',
                        desc = '1810 - 1860', 
                        type = 'sense')
anchor_wv = VectorVariations(corpus_name = 'ccoha2',
                            desc = '1960 - 2010', 
                            type = 'new')
target_word_pairs = prep_vectors(align_wv, anchor_wv, data_path, targets) 

#%%
## TODO: would be ideal to have this be more than just the intersection
wv1 = align_wv.partial_align_vec
wv2 = anchor_wv.partial_align_vec
wv_names_1800s = [align_wv.desc, anchor_wv.desc]
wv_names_2000s = [anchor_wv.desc, align_wv.desc]
path = f'{data_path}/plots/align_neighbors'

num_senses = defaultdict(int)
for sense, target in target_word_pairs:
    id = wv1.word_id[sense]
    # plot_alignment(id, wv1, wv2, wv_names_1800s, 
    #     f'{path}/{sense}.html')

    num_senses[target] += 1

#%%
for target, count in num_senses.items():
    id = wv2.word_id[target]
    plot_alignment(id, wv2, wv1, wv_names_2000s, 
        f'{path}/{target}.html', flip=True, count=count)
    
#%%
## Sense, its local neighbors, and aligned neighbors
## Target, its local neighbors, and aligned neighbors