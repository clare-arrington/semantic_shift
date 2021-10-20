#%%
from base_experiment import main, Target_Info, Train_Method_Info

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

#%%
# Results from paper
# .7 for english
# align old to new

## This is only align for S4, not cosine
align_params = {"n_targets": 100,
                "n_negatives": 50,
                "rate": 1,
                "iters": 100
                }

classify_params = {"n_targets": 500,
                  "n_negatives": 750,
                  "rate": .25
                  }

auto_params = { "rate": 1.5,
                "n_fold": 1,
                "n_targets": 50,
                "n_negatives": 100}

align_methods = [
    Train_Method_Info('global', None),
    Train_Method_Info('s4', align_params)
]

classify_methods = [
    Train_Method_Info('cosine', auto_params, 0)
    # Train_Method_Info('s4', classify_params, .5)
]

#%%
main(dataset_name, targets,
     align_info, anchor_info,
     align_methods, classify_methods,
     vector_types=['both_sense'],
     num_loops=10)
# %%
