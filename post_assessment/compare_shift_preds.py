#%%
import pandas as pd
import itertools

data_path = '/data/arrinj'
align_path = f'{data_path}/align_results/news'
with open(f'{data_path}/corpus_data/news/targets.txt') as fin:
    targets = fin.read().split()

prediction_results = []
corpus_names = ["Alternative"]
# corpus_names = ["Alternative"]
align_methods = ["sense"]
iters = itertools.product(corpus_names, align_methods,range(0, 6))
for corpus, align_method, slice_num in iters:
    full_path = f'{align_path}/align_{corpus.lower()}/{align_method}/slice_{slice_num}/s4_cosine/labels.csv'

    labels = pd.read_csv(full_path)
    labels.set_index('Words', inplace=True)
    if align_method == 'sense':
        agg_predictions = labels.apply(pd.Series.value_counts, axis=1)
        predictions = agg_predictions.fillna(0).idxmax(axis=1)
    else:
        predictions = labels.Predicted

    prediction_results.append(dict(predictions))

#%%
iterables = [ 
    [f'{c} sense embedding aligned to normal' for c in corpus_names], 
    [f"Slice {slice_num}" for slice_num in range(0,6)],
    ["Sense"]
    ]
cols = pd.MultiIndex.from_product(iterables, names=["corpus", "slice", "alignment"])

df = pd.DataFrame(prediction_results).T
df.sort_index(inplace=True)
df.columns = cols

#%%
df.to_csv(f'{align_path}/shift_summary.csv')
# %%
