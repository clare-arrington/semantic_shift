#%%
from base_experiment import main, Target_Info, Train_Method_Info

def get_word_pairs(path, get_us, get_uk, shift_label):
    with open(path) as fin:
        targets = []
        for pair in fin.read().strip().split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                target_word = us_word
                shifted_word = uk_word
            elif get_uk:
                target_word = uk_word
                shifted_word = us_word

            target = Target_Info(target_word, shifted_word, is_shifted=shift_label)
            targets.append(target)
        return targets

def get_us_uk_targets(path, get_us=False, get_uk=False):
    print('Pulling targets for US / UK data now')

    targets = []
    path = f'{path}/corpus_data/us_uk/truth'

    ## Get dissimilar
    with open(f'{path}/dissimilar.txt') as fin:
        for word in fin.read().split():
            target = Target_Info(word=word, shifted_word=word, is_shifted=True)
            targets.append(target)
    print(f'{len(targets)} dissimilar')

    ## Get similar
    similar_targets = get_word_pairs(f'{path}/similar.txt', get_us, get_uk, shift_label=False)
    targets.extend(similar_targets)
    print(f'{len(similar_targets)} similar')

    spelling_targets = get_word_pairs(f'{path}/spelling.txt', get_us, get_uk, shift_label=False)
    targets.extend(spelling_targets)
    print(f'{len(spelling_targets)} spelling')

    return targets

## Paper Results
## UK aligned to US
## Global: .38 / .45
##     S4: .44 / .70

#  BNC: 45813
# COCA: 30415

if __name__ == "__main__":
    ## Data Info
    data_path = '/data/arrinj'
    dataset_name = 'us_uk'
    bnc = ('bnc', 'UK corpus (BNC)')
    coca = ('coca', 'English corpus (COCA)')

    align_info = bnc
    anchor_info = coca

    if align_info[0] == 'bnc':
        targets = get_us_uk_targets(data_path, get_uk=True)
    elif align_info[0] == 'coca':
        targets = get_us_uk_targets(data_path, get_us=True)

#%%
    # best us_uk: 100, 100, .1
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
        Train_Method_Info('cosine', cos_classify_params, 0),
        Train_Method_Info('s4', s4_classify_params, .5)
    ]

#%%
    sense_methods = ['SSA']

    main(
        dataset_name, data_path,
        targets,
        align_info, anchor_info,
        sense_methods,
        align_methods, classify_methods,
        num_loops=3)
# %%
