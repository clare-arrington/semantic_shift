#%%
from base_experiment import main, Target_Info, Train_Method_Info

## Data Info
data_path = "/home/clare/Data"

with open(f'{data_path}/corpus_data/time/targets.txt') as f:
    targets = [Target_Info(t, t, True) for t in  f.read().split()]


s4_align_params = {"n_targets": 100,
                "n_negatives": 50,
                "rate": 1,
                "iters": 100
                }

s4_classify_params = {"n_targets": 500,
                  "n_negatives": 750,
                  "rate": .25
                  }

cos_classify_params = { "rate": 1.5,
                "n_fold": 1,
                "n_targets": 50,
                "n_negatives": 100}

align_methods = [
    # Train_Method_Info("global", None),
    Train_Method_Info("s4", s4_align_params)
]

classify_methods = [
    Train_Method_Info("cosine", cos_classify_params, 0),
    Train_Method_Info("s4", s4_classify_params, .5)
]

old  = ("1800s",  "CCOHA 1810 - 1860")
mid  = ("2000s",  "CCOHA 1960 - 2010")
coca = ( "coca",  "COCA 1990 - 2010")
# ai  = (   "ai",  "AI Corpus")

align_info = mid
anchor_info = coca

main(
    "time", data_path,
    targets,
    align_info, anchor_info,
    ["BSA"],
    align_methods, classify_methods,
    num_loops=5)

print('All done!!')
#%%