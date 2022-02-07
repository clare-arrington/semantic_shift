import itertools
from temp.wordvectors import VectorVariations
from typing import NamedTuple, List
from collections import namedtuple
import re

Target_Info = namedtuple('Target', ['word', 'shifted_word', 'is_shifted'])

## Parse through the targets to match them up 
def make_word_pairs(
    targets: List[NamedTuple], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations):

    all_wp = []
    target_wp = []
    if align_wv.type in ['original', 'normal', 'both_sense']:
        for target in targets:
            all_wp.append((target.word, target.shifted_word))
            
            ## TODO: double check the logic of this
            if target.is_shifted is not None:
                target_wp.append((target.word, target.shifted_word))

    else:
        ## TODO: verify this is okay for SSA still
        print(f'{len(align_wv.senses)} filtered senses for {align_wv.desc}')
        print(f'{len(anchor_wv.senses)} filtered senses for {anchor_wv.desc}\n')

        for target in targets:
            r = re.compile(f'{target.word}.[0-9]')
            align_sense_matches = filter(r.match, align_wv.senses)

            if anchor_wv.type == 'sense':
                anchor_sense_matches = filter(r.match, anchor_wv.senses)
            else:
                anchor_sense_matches = [target.shifted_word]

            for sense, match in itertools.product(align_sense_matches, anchor_sense_matches):
                all_wp.append((sense, match))

                if target.is_shifted is not None:
                    target_wp.append((sense, match))

    print(f'{len(all_wp)} total word pairs; {len(target_wp)} of those are targets')

    return all_wp, target_wp

def check_wv_target(wv, t_word):
    if wv.type == 'sense' and t_word not in wv.terms_with_sense:
        print(f'{t_word} senses missing from {wv.corpus_name}')
        return True
    elif wv.type == 'normal' and t_word not in wv.normal_vec.words:
        print(f'{t_word} missing from {wv.corpus_name}')
        return True
    else:
        return False

## TODO: this modifies targets long term; fix that issue
## TODO: I edited this for BSA but can't test on SemEval; test on US/UK later
def filter_targets(
    targets: List[NamedTuple], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations):

    remove_targets = []
    for index, target in enumerate(targets):

        ## Check for the align WV
        if check_wv_target(align_wv, target.word):
            remove_targets.append(index)
            continue

        ## Check for the anchor WV
        if check_wv_target(anchor_wv, target.shifted_word):
            remove_targets.append(index)

        if anchor_wv.type == 'sense' and target.shifted_word not in anchor_wv.terms_with_sense:
            print(f'{target.word} senses missing from {anchor_wv.corpus_name}')
            remove_targets.append(index)
            continue
        elif anchor_wv.type == 'normal' and target.shifted_word not in anchor_wv.normal_vec.words:
            print(f'{target.shifted_word} ({target.word}) missing from {anchor_wv.corpus_name}')
            remove_targets.append(index)



    for index in sorted(remove_targets, reverse=True):
        # print(f'Deleting {targets[index].word}')
        del targets[index]

    print(f'\nRunning test on {len(targets)} targets')

    return targets

