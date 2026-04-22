import os
import argparse
import pickle
import pandas as pd
# from unlearning_metrics.evaluate.config import parameters
from _old.config import parameters

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, help="Directory containing results")
args = parser.parse_args()


# Get seed from directory name
current_seed = int(args.results_dir.split("seed")[-1])

forget_type = parameters.get('class_to_replace', parameters.get('forgetting_data_amount'))

# Update directory mapping to use seed-specific paths
dir_names_dict = {
    'retrain': f'{args.results_dir}/retrain_{forget_type}',
    'FT': f'{args.results_dir}/ft_{forget_type}',
    'GA': f'{args.results_dir}/ga_{forget_type}',
    'wfisher': f'{args.results_dir}/wfisher_{forget_type}'
}

all_metrics = []

for method, dir_name in dir_names_dict.items():
    if method in parameters['unlearn_methods']:
        # List evaluation files
        eval_files = [
            f for f in os.listdir(dir_name)
            if f.startswith('evaluation_epoch_') and f.endswith('.pkl')
        ]
        if not eval_files:
            print(f"Warning: No evaluation results found for method {method} in {dir_name}")
            continue

        for eval_file in eval_files:
            # Extract epoch number
            epoch_str = eval_file.replace('evaluation_epoch_', '').replace('.pkl', '')
            if epoch_str == 'final':
                epoch = 'final'
            else:
                try:
                    epoch = int(epoch_str)
                except ValueError:
                    print(f"Warning: Could not parse epoch number from filename {eval_file}")
                    continue

            # Load evaluation results
            eval_file_path = os.path.join(dir_name, eval_file)
            with open(eval_file_path, 'rb') as f:
                data = pickle.load(f)
            # Flatten data
            flattened_data = {
                'seed': current_seed,
                'method': method,
                'epoch': epoch,
                'remaining_accuracy': data['accuracy']['retain'],
                'forget_accuracy': data['accuracy']['forget'],
                'testing_accuracy': data['accuracy']['test'],
                'SVC_MIA_forget_correctness': data['SVC_MIA_forget_efficacy']['correctness'],
                'SVC_MIA_forget_confidence': data['SVC_MIA_forget_efficacy']['confidence'],
                'SVC_MIA_forget_entropy': data['SVC_MIA_forget_efficacy']['entropy'],
                'SVC_MIA_forget_m_entropy': data['SVC_MIA_forget_efficacy']['m_entropy'],
                'SVC_MIA_forget_prob': data['SVC_MIA_forget_efficacy']['prob'],
                'unlearn_time_ms': data.get('unlearn_time_ms', None),
                'wasserstein_dist': data.get('wasserstein_dist', None),
                'activation_distance_forget': data.get('activ_dist_forget', None),
                'activation_distance_test': data.get('activ_dist_test', None),
                'wasserstein_dist_retrain': data.get('wasserstein_dist_retrain', None),
                'activation_distance_retrain': data.get('activ_dist_retrain', None),
                # Experiment parameters
                'param_epochs_for_unlearning': parameters['epochs_for_unlearning'].get(method),
                'param_learning_rate_for_unlearning': parameters['learning_rate_for_unlearning'].get(method),
                'param_alpha': parameters.get('alpha', None),
                'param_arch': parameters['arch'],
                'param_dataset': parameters['dataset'],
                'param_lr': parameters['lr'],
                'param_epochs_train': parameters['epochs_train'],
                'param_forgetting_data_amount': parameters.get('forgetting_data_amount'),
                'class_to_forget': parameters.get('class_to_replace')
            }

            all_metrics.append(flattened_data)

df = pd.DataFrame(all_metrics)

def epoch_sort_key(epoch_value):
    """Manual sorting of epoch values because 'final' should be placed at the end"""
    if epoch_value == 'final':
        return float('inf')  # Place 'final' epochs at the end
    else:
        return int(epoch_value)

df['epoch_sort_key'] = df['epoch'].apply(epoch_sort_key)  
df = df.sort_values(by=['method', 'epoch_sort_key']).drop(columns=['epoch_sort_key'])

# Compute gaps relative to retrain
df_retrain = df[df['method'] == 'retrain']
methods = df['method'].unique()
methods = [m for m in methods if m != 'retrain']
merged_metrics = []
max_retrain_epoch = df[df['method'] == 'retrain']['epoch'].max()
#Wfisher does not use epochs, so we replace 'final' with max num of epochs used for retrain
df['epoch'] = df['epoch'].replace('final', max_retrain_epoch) 

for method in methods:
    df_method = df[df['method'] == method]
    df_merged = pd.merge(df_method, df_retrain, on='epoch', suffixes=('', '_retrain'))

    # Compute gaps
    df_merged['ra_gap'] = abs(df_merged['remaining_accuracy'] - df_merged['remaining_accuracy_retrain'])
    df_merged['ua_gap'] = abs(df_merged['forget_accuracy'] - df_merged['forget_accuracy_retrain'])
    df_merged['test_gap'] = abs(df_merged['testing_accuracy'] - df_merged['testing_accuracy_retrain'])
    df_merged['mia_confidence_gap'] = abs(df_merged['SVC_MIA_forget_confidence'] - df_merged['SVC_MIA_forget_confidence_retrain'])
    df_merged['average_gap'] = (df_merged['ra_gap'] + df_merged['ua_gap'] + df_merged['test_gap'] + df_merged['mia_confidence_gap']) / 4
    df_merged['run_time_efficiency'] = df_merged['unlearn_time_ms'] / df_merged['unlearn_time_ms_retrain']

    # Select relevant columns
    cols_to_keep = ['seed', 'method', 'epoch'] + [col for col in df_method.columns if col not in ['seed', 'method', 'epoch']] + ['ra_gap', 'ua_gap', 'test_gap', 'mia_confidence_gap', 'average_gap', 'run_time_efficiency']
    df_merged = df_merged[cols_to_keep]
    merged_metrics.append(df_merged)

# Combine merged data into a single DataFrame
df_final = pd.concat(merged_metrics, ignore_index=True)

df_final.reset_index(drop=True, inplace=True)
df_retrain.reset_index(drop=True, inplace=True)

#For some reason, wasseresstein_dist_retrain is duplicated. 
#TODO: Fix this while creating metrics dataframe 
if 'wasserstein_dist_retrain' in df_final.columns and df_final.columns.duplicated().any():
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

# Optionally include retrain method data
include_retrain = True
if include_retrain:
    df_final = pd.concat([df_final, df_retrain], ignore_index=True)

# Save DataFrame
output_path = os.path.join(args.results_dir, f'final_table_seed{current_seed}_{forget_type}.csv')
df_final.to_csv(output_path, index=False)
print(f"Created table at: {output_path}")
