#%%
from base_experiment import temp, Target_Info, Vector_Info
import glob
import pickle

for vector_training_type in ['original']:
    for align_method in ['global']:
        ## Training Info
        # align_method = 'global'
        if align_method == 'global':
            classify_methods = {'cosine' : .3}
            num_runs = 1
            #classify_methods = {'s4': .5} 
        else:
            classify_methods = {'cosine': .3, 's4': .5}

        #vector_training_type = 'sense'

        if vector_training_type in ['original', 'new']:
            align_type = vector_training_type
            anchor_type = vector_training_type
        elif vector_training_type == 'sense':
            align_type = vector_training_type
            anchor_type = 'new'

        ## Data Info
        dataset_name = 'us_uk'
        align_vector = Vector_Info(corpus_name = 'coca', 
                                description = 'English corpus (COCA)', 
                                type = align_type)
        anchor_vector = Vector_Info(corpus_name = 'bnc', 
                                    description = 'UK corpus (BNC)', 
                                    type = anchor_type)

        ## Pull data
        sense_words = []
        sense_path = f"../data/masking_results/us_uk/coca/sentences/"

        ## Since it's only a subset, gotta pull this way
        # paths = glob.glob(f'{sense_path}*.dat')
        # for path in paths:
            # with open(path, 'rb') as f:
            #     target = path[len(sense_path):].split('_')[0]
            #     sense_words.append(target)

        targets = []
        added_targets = set()

        ## Get dissimilar
        num_dissimilar = 0
        with open('../data/us_uk/truth/dissimilar.txt') as fin:
            for word in fin.read().split():
                #if word in sense_words:
                target = Target_Info(word=word, shifted_word=word, is_shifted=True)
                targets.append(target)
                added_targets.add(word)
                num_dissimilar += 1

        print(f'{num_dissimilar} dissimilar')

        ## TODO: can't have dupes currently :/
        ## Get similar
        num_similar = 0
        with open('../data/us_uk/truth/similar.txt') as fin:
            for pair in fin.read().strip().split('\n'):
                uk_word, us_word = pair.split()
                #if us_word in sense_words and us_word not in added_targets:
                if us_word not in added_targets:
                    target = Target_Info(word=us_word, shifted_word=uk_word, is_shifted=False)
                    targets.append(target)
                    added_targets.add(us_word)
                    num_similar += 1

        print(f'{num_similar} similar')
        
        # Could also pass best acc in 
        all_accs = temp(align_method, classify_methods, dataset_name,
                        align_vector, anchor_vector, targets, num_loops=num_runs)


# %%
last_part = 'sense_global'
with open(f'results/us_uk/numerical/{last_part}.dat' , 'rb') as pf:
    results = pickle.load(pf)

# %%
