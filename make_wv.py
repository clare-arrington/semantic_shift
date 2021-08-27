#%%
from gensim.models import Word2Vec
from nltk import corpus
from nltk.corpus import stopwords
from collections import Counter
import pickle
import random
import glob
import re
import tqdm

# TODO: perhaps this should be interfaced through another file for each corpora as well

def load_sentences(main_path, sense_path='', subset=None):
    if '.dat' in main_path:
        with open(main_path, 'rb') as f:
            sentences = pickle.load(f)
    elif '.txt' in main_path:
        with open(main_path, 'r') as f:
            sentences = f.read().splitlines()

    if subset:
        num_samples = min(len(sentences), subset)
        sentences = random.sample(sentences, num_samples)
    else: 
        num_samples = len(sentences)

    print(f'{num_samples} sentences\n')

    # if sense_path != '':
    #     new_targets = []
    #     paths = glob.glob(f'{sense_path}*.dat')
    #     for path in paths:
    #         target = path[len(sense_path):].split('_')[0]
    #         if target not in targets:
    #             continue
    #         with open(path, 'rb') as f:
    #             sense_sentences = pickle.load(f)
    #             print(f'{target} : {len(sense_sentences)} sentences')
    #             sentences.extend(sense_sentences)
    #             new_targets.append(target)

    #     print(f'\n{len(sentences)} sentences total')

    #     return sentences, new_targets

    if sense_path != '':
        paths = glob.glob(f'{sense_path}*.dat')
        targets = []
        for path in tqdm.tqdm(paths):
            with open(path, 'rb') as f:
                sense_sentences = pickle.load(f)
                target = path[len(sense_path):].split('_')[0]
                targets.append(target)

                print(f'{target} : {len(sense_sentences)} sentences')
                sentences.extend(sense_sentences)

        print(f'\n{len(sentences)} sentences total')

        return sentences, targets

    else:
        return sentences

def clean_sentences(sentences, sense_words=[]):
    
    stops = stopwords.words('english')
    found_senses = []
    
    # Reg pattern matches three things: word.#, word_pos, word
    # reg_pattern = re.compile(r'[a-z]+.\d\.\d|[a-z]+_[a-z]{2}|[a-z]+')
    if len(sense_words) > 0:
        reg_pattern = re.compile(r'[a-z]+\.\d|[a-z]+')
    else:
        reg_pattern = re.compile(r'[a-z]+')

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        clean_sent = []
        cleaned = re.findall(reg_pattern, sent)

        for word in cleaned:

            ## 'Target' won't pass but 'target.0' should
            first_word, *etc = word.split('.')
            if first_word in sense_words and len(etc) == 1: 
                found_senses.append(word)
                clean_sent.append(word) 

            elif (word not in stops) and (len(word) >= 2):
                clean_sent.append(word)

        clean_sents.append(clean_sent)

    return clean_sents, found_senses
        
# %%
run = 'sense'
dataset = 'us_uk'
corpus_name = 'coca'
min_count = 49
num_sents = 10500000
# SemEval : min_count=20
# British : min_count=100
# American : min_count=200

# with open(f'../data/semeval/targets.txt') as fin:
#     targets = fin.read().split()

dissimilar = []
with open('../data/us_uk/truth/dissimilar.txt') as fin:
    for word in fin.read().split():
        dissimilar.append(word)

us = []
uk = []
with open('../data/us_uk/truth/similar.txt') as fin:
    for pair in fin.read().strip().split('\n'):
        uk_word, us_word = pair.split()
        if us_word not in dissimilar:
            us.append(us_word)
            uk.append(uk_word)

targets = dissimilar + us

#%%
if run == 'sense':
    # main_path = f'../data/{dataset}/corpora/ccoha{num}_non_target.dat'
    main_path = f'../data/{dataset}/coca_non_target.dat'
    export_path = f'wordvectors/{dataset}/{run}_{corpus_name}.vec'
    sense_path = f'../data/masking_results/{dataset}_old/{corpus_name}/sentences/'
    new_sense_path = f'../data/masking_results/{dataset}/sentences/'

    sents, sense_words = load_sentences(main_path, sense_path, subset=num_sents)
    # sense_words = [word.split('_')[0] for word in targets]
    # sense_words = targets

    clean_sents, found_senses = clean_sentences(sents, targets)
    #Counter(found_senses)

elif run == 'new':
    main_path = f'../data/{dataset}/{corpus_name}.txt'
    export_path = f'wordvectors/{dataset}/{run}_{corpus_name}.vec'

    sents = load_sentences(main_path, subset=num_sents)
    clean_sents, _ = clean_sentences(sents)

    del sents

# TODO: pickle the clean main sents so only the sense ones need to be done
# also pickle new clean


with open(f'{run}_sents_{num_sents}.dat', 'rb') as pf:
    sentences = pickle.load(pf)


with open(f'{run}_sents_{num_sents}.dat', 'wb') as pf:
    pickle.dump(clean_sents, pf)

#%%
# Vocab contains 26064 words
model = Word2Vec(clean_sents, vector_size=300, min_count=min_count, window=10)
model.save(export_path)
# %%


paths = glob.glob(f'{sense_path}*.dat')
new_paths = glob.glob(f'{new_sense_path}*.dat')

sentences = []
targets = []
for path in tqdm.tqdm(new_paths):
    with open(path, 'rb') as f:
        sense_sentences = pickle.load(f)
        target = path[len(new_sense_path):].split('_')[0]
        targets.append(target)

        print(f'{target} : {len(sense_sentences)} sentences')
        sentences.extend(sense_sentences)


for path in tqdm.tqdm(paths):
    with open(path, 'rb') as f:
        target = path[len(sense_path):].split('_')[0]
        if target in targets:
            print('skipping', target)
            continue
        targets.append(target)

        sense_sentences = pickle.load(f)

        print(f'{target} : {len(sense_sentences)} sentences')
        sentences.extend(sense_sentences)



print(f'\n{len(sentences)} sentences total')


# %%
