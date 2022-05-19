import itertools
from shift_steps.wordvectors import VectorVariations
from typing import NamedTuple, List
from collections import namedtuple
import re

## Parse through the targets to match them up 
def make_word_pairs(
    targets: List[NamedTuple], 
    align_wv: VectorVariations, 
    anchor_wv: VectorVariations):

    all_wp = []
    target_wp = []
    target_labels = {}

    if align_wv.type in ['original', 'normal']:
        for target in targets:
            pair = (target.word, target.shifted_word)
            all_wp.append(pair)
            
            if target.is_shifted is not None:
                target_wp.append(pair)
                target_labels[pair] = target.is_shifted
    else:
        print(f'{len(align_wv.senses)} filtered senses for {align_wv.type} align WV {align_wv.desc}')
        print(f'{len(anchor_wv.senses)} filtered senses for {anchor_wv.type} anchor WV {anchor_wv.desc}\n')

        # TODO: this doesn't seem like it'd work for US / UK
        for target in targets:
            # Find the senses for the current align target
            r = re.compile(f'{target.word}.[0-9]')
            align_sense_matches = filter(r.match, align_wv.senses)

            # Find the senses for the current anchor match if it exists
            if anchor_wv.type == 'normal':
                # Sense paired to plain token
                anchor_sense_matches = [target.shifted_word]
            elif anchor_wv.type == 'shared_sense':
                # Sense paired to itself on other side
                anchor_sense_matches = []
            elif anchor_wv.type == 'sense':
                # Sense paired to all senses
                anchor_sense_matches = filter(r.match, anchor_wv.senses)

            for sense, match in itertools.product(align_sense_matches, anchor_sense_matches):
                pair = (sense, match)
                all_wp.append(pair)

                if target.is_shifted is not None:
                    target_wp.append(pair)

            if target.is_shifted is not None:
                target_labels[(target.word, target.shifted_word)] = target.is_shifted

    print(f'{len(all_wp)} total word pairs; {len(target_wp)} of those are targets')

    return Word_Pairs(all_wp, target_wp, target_labels)

def check_target_in_wv(wv, t_word):
    if wv.type == 'sense' and t_word not in wv.terms_with_sense:
        print(f'Check 1: {t_word} senses missing from {wv.corpus_name}')
        return True
    elif wv.type == 'normal' and t_word not in wv.normal_vec.words:
        print(f'Check 1: {t_word} missing from {wv.corpus_name}')
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
        if check_target_in_wv(align_wv, target.word):
            print(align_wv.corpus_name, align_wv.type, target.word)
            remove_targets.append(index)
            continue

        ## Check for the anchor WV
        if check_target_in_wv(anchor_wv, target.shifted_word):
            remove_targets.append(index)
            continue

        if anchor_wv.type == 'sense' and target.shifted_word not in anchor_wv.terms_with_sense:
            print(f'Check 2: {target.word} senses missing from {anchor_wv.corpus_name}')
            remove_targets.append(index)
        elif anchor_wv.type == 'normal' and target.shifted_word not in anchor_wv.normal_vec.words:
            print(f'Check 2: {target.shifted_word} ({target.word}) missing from {anchor_wv.corpus_name}')
            remove_targets.append(index)

    for index in sorted(remove_targets, reverse=True):
        # print(f'Deleting {targets[index].word}')
        del targets[index]

    print(f'\nRunning test on {len(targets)} targets')

    return targets

Target_Info = namedtuple('Target', ['word', 'shifted_word', 'is_shifted'])

class Word_Pairs:
    def __init__(self, all_wps=None, target_wps=None, target_labels=None):

        self.all_wps = all_wps 
        self.target_wps = target_wps
        self.target_labels = target_labels
 