#%%
from base_experiment import main, Target_Info, Train_Method_Info

def get_targets(path):
    targets = []
    with open(path) as fin:
        for word in fin.read().split():
            target = Target_Info(word=word, shifted_word=word, is_shifted=True)
            targets.append(target)
    print(f'{len(targets)} targets')
    return targets

## Data Info
## One way to the other?
dataset_name = 'news'
anchor_info = ('mainstream', 'Mainstream news corpus')
align_info = ('alternative', 'Pseudoscience health corpus')
# align_info = ('conspiracy', 'Political conspiracy corpus')
data_path = '/data/arrinj'

# targets = get_targets(f'{data_path}/corpus_data/{dataset_name}/targets.txt')

#%%
# TODO: no idea what are good params currently
s4_align_params = { "n_targets": 100,
                    "n_negatives": 100,
                    "rate": .1
                    }

s4_classify_params = {  "n_targets": 1000,
                        "n_negatives": 1000,
                        "rate": .25
                        }

cos_classify_params = { "n_targets": 50,
                        "n_negatives": 100,
                        "rate": 1.5,
                        "n_fold": 1  }

align_methods = [
    #Train_Method_Info('global', None)
    Train_Method_Info('s4', s4_align_params)
]

classify_methods = [
    Train_Method_Info('cosine', cos_classify_params, 0)
    #Train_Method_Info('s4', s4_classify_params, .5)
]

#%%
vector_types = ['sense']

# slice_num = 0
# for slice_num in range(1,6):
targets = get_targets(f'{data_path}/corpus_data/{dataset_name}/targets.txt')

main(
    dataset_name, data_path,
    targets,
    align_info, anchor_info,
    vector_types,
    align_methods, classify_methods,
    num_loops=5)

# %%
