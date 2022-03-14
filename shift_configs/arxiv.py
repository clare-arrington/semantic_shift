#%%
from base_experiment import align_param_sweep, classify_param_sweep, main, Target_Info

with open('/home/clare/Data/corpus_data/arxiv/targets.txt') as fin:
    og_targets = fin.read().strip().split('\n')
    targets = []
    for target in og_targets:
        target = Target_Info(word=target, shifted_word=target, is_shifted=True)
        targets.append(target)

print(f'{len(targets)} targets loaded')

## Data Info
dataset_name = 'arxiv'
anchor_info = ('ai', 'AI Corpus')
align_info = ('phys', 'Phys Corpus')

# ? semeval: 100, 10, .25
## This is only align for S4, not cosine
# align_params = {"n_targets": 50,
#                 "n_negatives": 25,
#                 "rate": .1
#                 }


# classify_params = {"n_targets": 500,
#                   "n_negatives": 750,
#                   "rate": .25
#                   }

align_params = None
classify_params = None

main(dataset_name, targets,
     align_info, anchor_info,
     align_methods=['global'],
     classify_method_thresholds={'cosine':0},
     vector_types=['new','sense'],
     align_params=align_params,
     num_loops=2,
     classify_params=classify_params)
# %%
