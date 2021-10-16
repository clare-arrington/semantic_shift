#%%
from base_experiment import main, Target_Info
import pickle

def get_us_uk_targets(path, get_us=False, get_uk=False):
    targets = []
    added_targets = set()

    ## Get dissimilar
    num_dissimilar = 0
    with open(f'{path}/dissimilar.txt') as fin:
        for word in fin.read().split():
            target = Target_Info(word=word, shifted_word=word, is_shifted=True)
            targets.append(target)
            added_targets.add(word)
            num_dissimilar += 1

    print(f'{num_dissimilar} dissimilar')

    ## TODO: can't have word in both sets currently :/
    ## Get similar
    num_similar = 0
    with open(f'{path}/similar.txt') as fin:
        for pair in fin.read().strip().split('\n'):
            uk_word, us_word = pair.split()
            if get_us:
                target_word = us_word
                shifted_word = uk_word
            elif get_uk:
                target_word = uk_word
                shifted_word = us_word

            if target_word not in added_targets:
                target = Target_Info(target_word, shifted_word, is_shifted=False)
                targets.append(target)
                added_targets.add(target_word)
                num_similar += 1

    print(f'{num_similar} similar')

    return targets

## Data Info
dataset_name = 'us_uk'
anchor_info = ('coca', 'English corpus (COCA)')
align_info = ('bnc', 'UK corpus (BNC)')
num_loops = 10

target_path = '/home/clare/Data/corpus_data/us_uk/truth'

if align_info[0] == 'bnc':
    targets = get_us_uk_targets(f'{target_path}', get_uk=True)
elif align_info[0] == 'coca':
    targets = get_us_uk_targets(f'{target_path}', get_us=True)


## Paper results
# s4 both
# uk aligned to us anchor: .7

# best us_uk: 100, 100, .1
align_params = {"n_targets": 100,
                "n_negatives": 100,
                "rate": .1
                }

main(dataset_name, targets, num_loops, 
     align_info, anchor_info, align_params)
# %%
