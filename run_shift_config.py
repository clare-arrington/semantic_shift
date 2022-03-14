from base_experiment import main, Target_Info, Train_Method_Info
from shift_configs.semeval_config import shift_config, params

align_methods = [
    Train_Method_Info("global", None),
    Train_Method_Info("s4", params['s4_align_params']) ]

classify_methods = [
    Train_Method_Info("cosine", params['cos_classify_params'], 0),
    Train_Method_Info("s4", params['s4_classify_params'], .5) ]

## 
align_name = shift_config['align_name']
anchor_name = shift_config['anchor_name']

align_info = (align_name, shift_config['corpora_info'])
anchor_info = (anchor_name, shift_config['corpora_info'])

targets = [Target_Info(*t) for t in shift_config['targets']]

main(
    shift_config['dataset_name'], shift_config['data_path'],
    targets,
    align_info, anchor_info,
    shift_config['sense_methods'],
    align_methods, classify_methods,
    num_loops=shift_config['num_loops'])