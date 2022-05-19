from dotenv import dotenv_values

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

            target = (target_word, shifted_word, shift_label)
            targets.append(target)
        return targets

def get_us_uk_targets(path, get_us=False, get_uk=False):
    print('Pulling targets for US / UK data now')

    targets = []
    path = f'{path}/corpus_data/us_uk/truth/full_files'

    ## Get dissimilar
    with open(f'{path}/dissimilar.txt') as fin:
        for word in fin.read().split():
            target = (word, word, True)
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

## Data Info
data_path = dotenv_values(".env")['data_path']
align_name = 'bnc'
anchor_name = 'coca'

if align_name == 'bnc':
    targets = get_us_uk_targets(data_path, get_uk=True)
elif align_name == 'coca':
    targets = get_us_uk_targets(data_path, get_us=True)

sense_methods = [
    'normal', 
    'SSA', 
    'BSA',
]

shift_config = {
    "align_name"    : align_name,
    "anchor_name"   : anchor_name,
    "sense_methods" : sense_methods,
    "num_loops"     : 1,
    "targets"       : targets,
    "data_path"     : data_path,
    "dataset_name"  : 'us_uk',
    "corpora_info"  : {
            "bnc"   : 'UK English corpus (BNC)',
            "coca"  : 'US English corpus (COCA)'
    }
}

methods = {
    'align' : [
        'global',
        's4',
    ],
    'classify' : [
        'cosine',
        's4',
    ]
}

# best us_uk: 100, 100, .1
params = {
    "s4_align_params" : {
        "n_targets": 100,
        "n_negatives": 100,
        "rate": .1
        },
    "s4_classify_params" : {
        "n_targets": 1000,
        "n_negatives": 1000,
        "rate": .25
        },
    "cos_classify_params" : { 
        "rate": 1.5,
        "n_fold": 1,
        "n_targets": 50,
        "n_negatives": 100
    }
}
