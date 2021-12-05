#%%
from base_experiment import main, Target_Info, Train_Method_Info
import pandas as pd

def get_targets(data_path, corpus_name, run, all_targets=False):
    if all_targets:
        target_data = f'{data_path}/masking_results/semeval/{corpus_name}/target_sense_labels.pkl'
        target_data = pd.read_pickle(target_data)
        all_targets = [t for t in target_data.target.unique()]
    else: 
        all_targets = []
    
    targets = []
    with open(f'{data_path}/corpus_data/semeval/truth/binary.txt') as fin:
        og_targets = fin.read().strip().split('\n')
        for target in og_targets:
            target, label = target.split('\t')
            label = bool(int(label))
            if run == 'sense':
                word, pos = target.split('_')
            else:
                word = target
            target = Target_Info(word=word, shifted_word=target, is_shifted=label)
            targets.append(target)
            if word in all_targets:
                all_targets.remove(word)
            
    for target in all_targets:
        target = Target_Info(word=target, shifted_word=target, is_shifted=None)
        targets.append(target)

    print(f'{len(targets)} targets loaded')
    print(targets[:5])
    return targets

## Data Info
data_path = '/home/clare/Data'
dataset_name = 'semeval'
align_info = ('2000s', 'CCOHA 1960 - 2010')
anchor_info = ('1800s', 'CCOHA 1810 - 1860')

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
    Train_Method_Info('global', None),
    Train_Method_Info('s4', s4_align_params)
]

classify_methods = [
    Train_Method_Info('cosine', cos_classify_params, 0)
    #Train_Method_Info('s4', s4_classify_params, .5)
]

#%%
for v_type in ['new','sense']:
    targets = get_targets(data_path, align_info[0], v_type)
    print(f'{len(targets)} targets loaded')

    main(
        dataset_name, data_path,
        targets,
        align_info, anchor_info,
        [v_type],
        align_methods, classify_methods,
        num_loops=10)

# %%
