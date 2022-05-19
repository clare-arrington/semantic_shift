from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# TODO: why ID? <- don't recall this question
def find_neighbors(id, main_wv, other_wv, wv_descs, 
    local_neighbor_indices, mapped_neighbor_indices):

    categories = [f'Target from {wv_descs[0]}']
    target = main_wv.words[id]
    words = [target]
    vecs = [main_wv[id]]
     
    neighbors = local_neighbor_indices[id]
    for i in neighbors:
        word = main_wv.words[i]
        if word != target:
            vecs.append(main_wv[i])
            words.append(word)
            categories.append(wv_descs[0])
        if len(words) == 11:
            break

    neighbors = mapped_neighbor_indices[id]
    vecs += [other_wv[i] for i in neighbors]
    words += [other_wv.words[i] for i in neighbors]
    categories += [wv_descs[1]] * len(neighbors)

    return words, categories, vecs

def get_neighbor_coordinates(x):
    """
    Apply decomposition to an input matrix and returns a 2d set of points.
    """
    return list(PCA(n_components=2).fit_transform(x))

def perform_mapping(wva, wvb, n_neighbors=10, metric="cosine"):
    """
    Given aligned wv_a and wv_b, performs mapping (translation) of words in a to those in b
    Returns (distances, indices) as n-sized lists of distances and the indices of the top neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=12, metric=metric).fit(wvb.vectors)
    distances, indices = nbrs.kneighbors(wva.vectors)

    return distances, indices

def get_neighbors(main_wv, other_wv, wv_descs):
    results = {}

    main_vec = main_wv.post_align_vec
    other_vec = other_wv.post_align_vec
    
    _, local_neighbor_indices = perform_mapping(main_vec, main_vec, 
                            n_neighbors=20) 
    _, mapped_neighbor_indices = perform_mapping(main_vec, other_vec)

    for sense in tqdm(main_wv.senses):
        id = main_vec.word_id[sense]
        results[sense] = find_neighbors(id, main_vec, other_vec, wv_descs,
                                local_neighbor_indices, mapped_neighbor_indices)

    return results