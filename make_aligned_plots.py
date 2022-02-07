#%%
from base_experiment import filter_targets, make_word_pairs, Target_Info
from us_uk import get_us_uk_targets
from shift_steps.alignment import align
from shift_steps.wordvectors import WordVectors, VectorVariations, load_wordvectors, intersection, extend_normal_with_sense
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

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
                    flip=False):

    categories = [f'Target from {wv_names[0]}']
    target = main_wv.words[id]
    words = [target]
    vecs = [main_wv[id]]
    
    _, indices = perform_mapping(main_wv, main_wv, 
                                n_neighbors=20) 
    neighbors = indices[id]
    for i in neighbors:
        word = main_wv.words[i]
        if word != target:
            vecs.append(main_wv[i])
            words.append(word)
            categories.append(wv_names[0])
        if len(words) == 11:
            break

    _, indices = perform_mapping(main_wv, other_wv)
    neighbors = indices[id]
    vecs += [other_wv[i] for i in neighbors]
    words += [other_wv.words[i] for i in neighbors]
    categories += [wv_names[1]] * len(neighbors)

    x = get_neighbor_coordinates(vecs)
    make_plot(x, words, categories, wv_names, path, flip)

def prep_vectors(
    align_wv, anchor_wv, dataset_name, 
    data_path, targets, slice_path='',
    norm=False):

    align_wv, anchor_wv = load_wordvectors(align_wv, anchor_wv, slice_path,
        f'{data_path}/word_vectors/{dataset_name}')

    if norm:
        align_wv.normal_vec.normalize()
        anchor_wv.normal_vec.normalize()

    targets = filter_targets(targets, align_wv, anchor_wv)
    all_word_pairs, target_word_pairs = make_word_pairs(
        align_wv.type, align_wv.normal_vec.words, targets)

    wv1, wv2 = intersection(align_wv.normal_vec, anchor_wv.normal_vec)
    print("Size of common vocab:", len(wv1))

    extended_wv1, extended_wv2 = extend_normal_with_sense(
        wv1, wv2, align_wv, anchor_wv, all_word_pairs)
    print(f"Size of WV after senses added: {len(wv1)} -> {len(extended_wv1)}" )

    output_path = f'{data_path}/align_results/{dataset_name}/align_{align_wv.corpus_name}/sense{slice_path}/s4_cosine'
    with open(f'{output_path}/landmarks.pkl' , 'rb') as pf:
        landmark_terms = pickle.load(pf)
        print(len(landmark_terms))

    ## TODO: same problem still with landmarks not all being there at times
    ## Is extended the problem?
    landmarks = [extended_wv1.word_id[word] for word in landmark_terms if word in extended_wv1.word_id]
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

def news_slices():
    for slice_num in range(0, 6):
        data_path = '/data/arrinj'
        dataset_name = 'news'
        targets = []
        with open(f'{data_path}/corpus_data/{dataset_name}/targets.txt') as fin:
            og_targets = fin.read().strip().split('\n')
            for target in og_targets:
                target = Target_Info(word=target, shifted_word=target, is_shifted=True)
                targets.append(target)

        ## Swap
        anchor_wv = VectorVariations(corpus_name = 'mainstream',
                                desc = 'Mainstream news corpus', 
                                type = 'new')
        align_wv = VectorVariations(corpus_name = 'alternative',
                                    desc = 'Pseudoscience health corpus', 
                                    type = 'sense')

        if slice_num is not None:
            slice_path = f'/slice_{slice_num}'
        else:
            slice_path = ''

        target_word_pairs = prep_vectors(
            align_wv, anchor_wv, dataset_name, 
            data_path, targets, slice_path, norm=True) 

        ## TODO: would be ideal to have this be more than just the intersection
        # wv1 = align_wv.partial_align_vec
        # wv2 = anchor_wv.partial_align_vec
        wv1 = align_wv.post_align_vec
        wv2 = anchor_wv.post_align_vec
        sense_wv_names = [align_wv.desc, anchor_wv.desc]
        target_wv_names = [anchor_wv.desc, align_wv.desc]

        path = f'{data_path}/plots/align_{align_wv.corpus_name}{slice_path}'
        Path(path).mkdir(parents=True, exist_ok=True)

        targets = set()
        for sense, target in tqdm(target_word_pairs):
        # sense = 'vaccine.0'
        # target = 'vaccine'
            targets.add(target)
            id = wv1.word_id[sense]
            plot_alignment(id, wv1, wv2, sense_wv_names, 
                f'{path}/{sense}.html')

        for target in tqdm(list(targets)):
            id = wv2.word_id[target]
            plot_alignment(id, wv2, wv1, target_wv_names, 
                f'{path}/{target}.html', flip=True)
        
        print(slice_num, ' Done')

def generic_run(
    data_path, dataset_name, 
    targets, align_wv, anchor_wv
    ):
    target_word_pairs = prep_vectors(
        align_wv, anchor_wv, dataset_name, 
        data_path, targets, norm=True) 

    wv1 = align_wv.post_align_vec
    wv2 = anchor_wv.post_align_vec
    sense_wv_names = [align_wv.desc, anchor_wv.desc]
    target_wv_names = [anchor_wv.desc, align_wv.desc]

    path = f'{data_path}/plots/align_{align_wv.corpus_name}'
    print(f'Going to save plots to {path}')
    Path(path).mkdir(parents=True, exist_ok=True)

    targets = set()
    for sense, target in tqdm(target_word_pairs):
        targets.add(target)
        id = wv1.word_id[sense]
        plot_alignment(id, wv1, wv2, sense_wv_names, 
            f'{path}/{sense}.html')

    for target in tqdm(list(targets)):
        id = wv2.word_id[target]
        plot_alignment(id, wv2, wv1, target_wv_names, 
            f'{path}/{target}.html', flip=True)
        
    print('Done')

def run_us_uk(swap=False):
    data_path = '/data/arrinj'
    dataset_name = 'us_uk'
    targets = get_us_uk_targets(data_path, get_uk=True)

    if swap:
        anchor_wv = VectorVariations(corpus_name = 'bnc',
                                desc = 'UK corpus (BNC)', 
                                type = 'sense')
        align_wv = VectorVariations(corpus_name = 'coca',
                                    desc = 'English corpus (COCA)', 
                                    type = 'new')
    else:
        align_wv = VectorVariations(corpus_name = 'bnc',
                                desc = 'UK corpus (BNC)', 
                                type = 'sense')
        anchor_wv = VectorVariations(corpus_name = 'coca',
                                    desc = 'English corpus (COCA)', 
                                    type = 'new')

    generic_run(data_path, dataset_name, 
                targets, align_wv, anchor_wv
                )

def run_semeval(swap=False):
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

    if swap:
        align_wv = VectorVariations(corpus_name = '1800s',
                                desc = '1810 - 1860', 
                                type = 'sense')
        anchor_wv = VectorVariations(corpus_name = '2000s',
                                    desc = '1960 - 2010', 
                                    type = 'new')
    else:
        align_wv = VectorVariations(corpus_name = '2000s',
                        desc = '1960 - 2010', 
                        type = 'sense')
        anchor_wv = VectorVariations(corpus_name = '1800s',
                                    desc = '1810 - 1860', 
                                    type = 'new')

    generic_run(data_path, dataset_name, 
                targets, align_wv, anchor_wv
                )

def run_news(swap=False):
    data_path = '/data/arrinj'
    dataset_name = 'news'

    targets = []
    with open(f'{data_path}/corpus_data/{dataset_name}/targets.txt') as fin:
        og_targets = fin.read().strip().split('\n')
        for target in og_targets:
            target = Target_Info(word=target, shifted_word=target, is_shifted=True)
            targets.append(target)

    if swap:
        align_wv = VectorVariations(corpus_name = 'mainstream',
                    desc = 'Mainstream news corpus', 
                    type = 'new')
        anchor_wv = VectorVariations(corpus_name = 'conspiracy',
                    desc = 'Political conspiracy corpus', 
                    type = 'sense')
        # anchor_wv = VectorVariations(corpus_name = 'alternative',
        #                             desc = 'Pseudoscience health corpus', 
        #                             type = 'sense')
    else:
        anchor_wv = VectorVariations(corpus_name = 'mainstream',
                    desc = 'Mainstream news corpus', 
                    type = 'new')
        # align_wv = VectorVariations(corpus_name = 'conspiracy',
        #             desc = 'Political conspiracy corpus', 
        #             type = 'sense')
        align_wv = VectorVariations(corpus_name = 'alternative',
                                    desc = 'Pseudoscience health corpus', 
                                    type = 'sense')

    generic_run(data_path, dataset_name, 
                targets, align_wv, anchor_wv
                )

#%%
## Sense, its local neighbors, and aligned neighbors
## Target, its local neighbors, and aligned neighbors

def pairwise_plots():
    for slice_num in range(0, 6):
        data_path = '/data/arrinj'
        dataset_name = 'news'
        targets = []
        with open(f'{data_path}/corpus_data/{dataset_name}/targets.txt') as fin:
            og_targets = fin.read().strip().split('\n')
            for target in og_targets:
                target = Target_Info(word=target, shifted_word=target, is_shifted=True)
                targets.append(target)

        ## Swap
        anchor_wv = VectorVariations(corpus_name = 'mainstream',
                                desc = 'Mainstream news corpus', 
                                type = 'new')
        align_wv = VectorVariations(corpus_name = 'alternative',
                                    desc = 'Pseudoscience health corpus', 
                                    type = 'sense')

        if slice_num is not None:
            slice_path = f'/slice_{slice_num}'
        else:
            slice_path = ''

        target_word_pairs = prep_vectors(
            align_wv, anchor_wv, dataset_name, 
            data_path, slice_path, targets, norm=True) 

        ## TODO: would be ideal to have this be more than just the intersection
        # wv1 = align_wv.partial_align_vec
        # wv2 = anchor_wv.partial_align_vec
        wv1 = align_wv.post_align_vec
        wv2 = anchor_wv.post_align_vec
        sense_wv_names = [align_wv.desc, anchor_wv.desc]
        target_wv_names = [anchor_wv.desc, align_wv.desc]

        path = f'{data_path}/plots/align_{align_wv.corpus_name}{slice_path}'
        Path(path).mkdir(parents=True, exist_ok=True)

        targets = set()
        for sense, target in tqdm(target_word_pairs):
        # sense = 'vaccine.0'
        # target = 'vaccine'
            targets.add(target)
            id = wv1.word_id[sense]
            plot_alignment(id, wv1, wv2, sense_wv_names, 
                f'{path}/{sense}.html')

        for target in tqdm(list(targets)):
            id = wv2.word_id[target]
            plot_alignment(id, wv2, wv1, target_wv_names, 
                f'{path}/{target}.html', flip=True)
        
        print(slice_num, ' Done')

# run_us_uk()
# run_semeval(swap=False)
run_news()