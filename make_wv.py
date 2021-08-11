#%%
from gensim.models import Word2Vec
from nltk import corpus
from nltk.corpus import stopwords
import pickle
import random
import glob
import re
import tqdm

def load_sentences(main_path, sense_path=''):
    if '.dat' in main_path:
        with open(main_path, 'rb') as f:
            sentences = pickle.load(f)
    elif '.txt' in main_path:
        with open(main_path, 'r') as f:
            sentences = f.read().splitlines()

    num_samples = min(len(sentences), 10000000)
    sentences = random.sample(sentences, num_samples)
    print(f'{num_samples} sentences\n')

    if sense_path != '':
        paths = glob.glob(f'{sense_path}*.dat')
        for path in paths:
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

def clean_sentences(sentences, targets, sense_words=[]):
    
    # untagged_targets = [word.split('_')[0] for word in targets]
    stops = stopwords.words('english')
    
    # Reg pattern matches three things: word.#, word_pos, word
    # reg_pattern = re.compile(r'\w+\.\d|[a-z]+_[a-z]{2}|[a-z]+')
    reg_pattern = re.compile(r'\w+\.\d|[a-z]+')

    clean_sents = []
    for sent in tqdm.tqdm(sentences):
        clean_sent = []

        cleaned = re.findall(reg_pattern, sent)

        for word in cleaned:

            # TODO: check if sense word is added fine
            if word in targets or word in sense_words:
                clean_sent.append(word) 

            # elif word in untagged_targets:
            #     continue

            elif (word not in stops) and (len(word) >= 3):
                clean_sent.append(word)

        # TODO: may want to limit min size of sentence

        clean_sents.append(clean_sent)

    return clean_sents
        
# %%
run = 'new'
dataset = 'us_uk'
corpus_name = 'coca'

# with open(f'../data/semeval/targets.txt') as fin:
#     targets = fin.read().split()

## For target
if run == 'target':
    # main_path = f'../data/{dataset}/corpora/ccoha{num}_non_target.dat'
    main_path = f'../data/{dataset}/coca_non_target.dat'
    export_path = f'wordvectors/{dataset}/{run}_{corpus_name}.vec'
    sense_path = f'../data/masking_results/{dataset}/{corpus_name}/sentences/'
    
    sents, targets = load_sentences(main_path, sense_path)
    # sense_words = [word.split('_')[0] for word in targets]
    # sense_words = targets

    clean_sents = clean_sentences(sents, targets)

elif run == 'new':
    main_path = f'../data/{dataset}/{corpus_name}.txt'
    export_path = f'wordvectors/{dataset}/{run}_{corpus_name}.vec'

    sents = load_sentences(main_path)
    clean_sents = clean_sentences(sents, [])

model = Word2Vec(clean_sents, vector_size=300, min_count=20, window=10)
model.save(export_path)
# %%
