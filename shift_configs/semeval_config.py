#%%
import pandas as pd

# Results from paper
# .7 for english
# align old (ccoha1) to new (ccoha2)

## TODO: this should be prepped ahead of time for all?
def get_targets(data_path, corpus_name, sense_method, all_targets=False):
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
            target_pos, label = target.split('\t')
            shifted_label = bool(int(label))
            target, pos = target_pos.split('_') 

            if sense_method == 'SSA':
                word = target
                shifted_word = target_pos
            elif sense_method == 'BSA':                             
                word = target
                shifted_word = target
            else:
                word = target_pos
                shifted_word = target_pos
                
            target = (word, shifted_word, shifted_label)
            targets.append(target)
            if word in all_targets:
                all_targets.remove(word)
            
    for target in all_targets:
        target = (target, target, None)
        targets.append(target)

    print(f'{len(targets)} targets loaded')
    print(targets[:5])
    return targets

target_terms =  ['virus', 'bit', 'memory', 'long', 
            'float', 'web', 'worm', 'bug', 'structure',
            'cloud', 'ram', 'apple', 'cookie', 
            'spam',  'intelligence', 'artificial', 
            'time', 'work', 'action', 'goal', 'branch',
            'power', 'result', 'complex', 'root',
            'process', 'child', 'language', 'term',
            'rule', 'law', 'accuracy', 'mean', 
            'scale', 'variable', 'rest', 
            'normal', 'network', 'frame', 'constraint', 
            'subject', 'order', 'set', 'learn', 'machine',
            'problem', 'scale', 'large', 
            'model', 'based', 'theory', 'example', 
            'function', 'field', 'space', 'state', 
            'environment', 'compatible', 'case', 'natural', 
            'agent', 'utility', 'absolute', 'value', 
            'range', 'knowledge', 'symbol', 'true', 
            'class', 'object', 'fuzzy', 'global', 'local', 
            'search', 'traditional', 'noise', 'system']

targets = [(target, target, None) for target in target_terms]

## Data Info
data_path = '/home/clare/Data'

shift_config = {
    "anchor_name"   : "1800s",
    "align_name"    : "2000s",
    "sense_methods" : [],
    "num_loops"     : 3,
    "targets"       : targets,

    "dataset_name"  : "semeval",
    "corpora_info"  : {
            "1800s" : "CCOHA 1810 - 1860",
            "2000s" : "CCOHA 1960 - 2010"
    },

    
    "data_path"     : data_path
}

params = {
    "s4_align_params" : {
        "n_targets": 100,
        "n_negatives": 50,
        "rate": 1,
        "iters": 100
        },
    "s4_classify_params" : {
        "n_targets": 500,
        "n_negatives": 750,
        "rate": .25
        },
    "cos_classify_params" : { 
        "rate": 1.5,
        "n_fold": 1,
        "n_targets": 50,
        "n_negatives": 100
    }
}
