#%%
import pickle

output_path = '/home/clare/Data/align_results/semeval/align_ccoha1_both/both_sense'
with open(f'{output_path}/s4_landmarks.dat' , 'rb') as pf:
    landmarks = pickle.load(pf)

print(len(landmarks))
# %%
sense_landmarks = [lm for lm in landmarks if '.' in lm]
for sense_lm in sorted(sense_landmarks):
    print(sense_lm)
# %%
