#%%
from base_experiment import classify_param_sweep, main, Target_Info

with open('/home/clare/Data/corpus_data/semeval/truth/binary.txt') as fin:
    og_targets = fin.read().strip().split('\n')
    targets = []
    for target in og_targets:
        target, label = target.split('\t')
        label = bool(int(label))
        word, pos = target.split('_')

        target = Target_Info(word=word, shifted_word=target, is_shifted=label)
        targets.append(target)

print(f'{len(targets)} targets loaded')

## Data Info
dataset_name = 'semeval'
anchor_info = ('ccoha2_both', 'CCOHA 1960 - 2010')
align_info = ('ccoha1_both', 'CCOHA 1810 - 1860')

# Results from paper
# .7 for english
# s4 align - cosine (maybe better if s4 classify)
# align old to new
## see if its okay this way from 

## This is only align for S4, not cosine
align_params = {"n_targets": 100,
                "n_negatives": 50,
                "rate": 1,
                "iters": 100
                }

# classify_param = {"n_targets": 500,
#                   "n_negatives": 750,
#                   "rate": .25
#                   }

## TODO: attach params to methods
## Make methods for both as dicts

#%%
main(dataset_name, targets,
     align_info, anchor_info,
     align_methods=['global', 's4'],
     classify_method_thresholds={'cosine':0},
     vector_types=['both_sense'],
     align_params=align_params,
     num_loops=10,
     classify_params=None)
# %%
