#%%
from base_experiment import main, Target_Info, Train_Method_Info
import pandas as pd

data_path = '/home/clare/Data'
# target_data = f'{data_path}/masking_results/semeval/ccoha1_extra/target_sense_labels.csv'
# target_data = pd.read_csv(target_data, usecols=['target'])
# all_targets = [t for t in target_data.target.unique()]
# skip = []

targets = []
with open(f'{data_path}/corpus_data/semeval/truth/binary.txt') as fin:
    og_targets = fin.read().strip().split('\n')
    for target in og_targets:
        target, label = target.split('\t')
        label = bool(int(label))
        word, pos = target.split('_')

        target = Target_Info(word=word, shifted_word=target, is_shifted=label)
        targets.append(target)
        # skip.append(word)

# for target in all_targets:
#     if target in skip:
#         continue
#     target = Target_Info(word=target, shifted_word=target, is_shifted=None)
#     targets.append(target)
        
print(f'{len(targets)} targets loaded')

## Data Info
dataset_name = 'semeval'
anchor_info = ('ccoha2', 'CCOHA 1960 - 2010')
align_info  = ('ccoha1', 'CCOHA 1810 - 1860')

#%%
# Results from paper
# .7 for english
# align old (ccoha1) to new (ccoha2)

s4_align_params = {"n_targets": 100,
                "n_negatives": 50,
                "rate": 1,
                "iters": 100
                }

s4_classify_params = {"n_targets": 500,
                  "n_negatives": 750,
                  "rate": .25
                  }

cos_classify_params = { "rate": 1.5,
                "n_fold": 1,
                "n_targets": 50,
                "n_negatives": 100}

align_methods = [
    #Train_Method_Info('global', None),
    Train_Method_Info('s4', s4_align_params)
]

classify_methods = [
    Train_Method_Info('cosine', cos_classify_params, 0)
    #Train_Method_Info('s4', s4_classify_params, .5)
]

#%%
vector_types = ['sense']

main(
    dataset_name, data_path,
    targets,
    align_info, anchor_info,
    vector_types,
    align_methods, classify_methods,
    num_loops=10)

# %%
