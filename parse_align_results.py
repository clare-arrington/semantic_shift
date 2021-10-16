#%%
from collections import defaultdict
from tabulate import tabulate
import pickle
import glob

def get_new_table(path, align_types=['global', 's4']): 
    result_summary = []

    for align_type in align_types:

        with open(f'{path}/new/{align_type}.dat', 'rb') as f:
            run_results = pickle.load(f)

        for classify_type, results in run_results.items():
            accs = [result['Predicted'] for result in results]
            max_acc = max(accs)
            avg_acc = sum(accs) / len(accs)

            result_summary.append([align_type, classify_type, avg_acc, max_acc])

    headers = [
    'Alignment\nMethod', 'Classify\nMethod', 
    'Average\nAccuracy', 'Highest\nAccuracy']

    return tabulate(result_summary, headers, floatfmt=".2f")

## TODO: change the tuples
def get_sense_table(path, align_types=['global', 's4']): 
    
    main_summary = []
    method_summary = []
    for align_type in align_types:
        with open(f'{path}/sense/{align_type}.dat', 'rb') as f:
            run_results = pickle.load(f)

        for classify_type, results in run_results.items():
            shift_accs = defaultdict(list)
            for run in results:
                for shift_method, acc in run.items():
                    shift_accs[shift_method].append(acc)

            max_acc = 0
            avg_acc = 0
            method_accs = []
            method_names = []
            for shift_method, accs in shift_accs.items():
                shift_max = max(accs)
                shift_avg = sum(accs) / len(accs)
                method_accs.append((round(shift_avg, 2), round(shift_max, 2)))
                method_names.append(shift_method)

                max_acc = max(max_acc, shift_max)
                avg_acc += shift_avg

            avg_acc /= len(shift_accs)

            main_results = [align_type, classify_type, avg_acc, max_acc]
            method_results = [align_type, classify_type] + method_accs

            main_summary.append(main_results)
            method_summary.append(method_results)

    main_columns = ['Alignment\nMethod', 'Classify\nMethod']
    summary_columns = ['Average\nAccuracy', 'Highest\nAccuracy']

    main_table = tabulate(main_summary, main_columns + summary_columns, floatfmt=".2f")
    methods_table = tabulate(method_summary, main_columns + method_names, floatfmt=".2f")

    return main_table, methods_table

#%%
dataset_name = 'semeval_older'
aligned = 'ccoha2'
path = f'/home/clare/Data/align_results/{dataset_name}/align_{aligned}/'

new_table = get_new_table(path)
sense_table, methods_table = get_sense_table(path)
print(sense_table)
print(methods_table)

with open(f'{path}summary_results.txt', 'w') as f:
    print(f'Aligned to {aligned.upper()}\n', file=f)
    print('===== Normal =====\n', file=f)
    print(new_table, file=f)
    print('\n===== Sense =====\n', file=f)
    print(sense_table, file=f)
    print('', file=f)
    print(methods_table, file=f)

#%%
def dunno(path, embedding_types=['new']):
    results = defaultdict(dict)
    for embedding_type in embedding_types:

        highest_acc = defaultdict(dict)
        best_methods = defaultdict(dict)
        for path in glob.glob(f'{path}/{embedding_type}/*/'):
            params = path.split('/')[-2]
            align, classify = params.split('_') 

            with open(f'{path}accuracies.txt') as f:
                lines = f.read().strip().split('\n')

            line = ''
            while 'highest accuracy' not in line:
                line = lines.pop(0)

            highest_acc = float(line[-4:])

            methods = []
            while '====' not in line:
                if '\t' in line:
                    methods.append(line.strip())
                line = lines.pop(0)

            highest_acc[align][classify].append()

            if embedding_type == 'new':
                results[params] = highest_acc
            else:
                results[embedding_type][params] = (highest_acc, methods)

# %%
# with open('/home/clare/Data/align_results/semeval/align_ccoha1/original/s4_landmarks.dat', 'rb') as f:
#     landmarks = pickle.load(f)

# set(targets) - set(targets).intersection(set(landmarks))