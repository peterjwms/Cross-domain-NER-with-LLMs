import ast
from glob import glob
import json

import pandas as pd
from tqdm import tqdm
from grid_search import evaluate_results

def replace_label(label: str) -> str:
    new_label = label.upper()
    if new_label == 'ORGANIZATION':
        new_label = 'ORG'
    elif new_label == 'LOCATION':
        new_label = 'LOC'
    return new_label

if __name__ == "__main__":
    # rerun the test for the previous results, accounting for caps and org/organization issues
    fixed_scores = []
    for file in tqdm(glob('results/*.csv')):
        fixed_results = []
        with open(file, 'r') as f:
            results = pd.read_csv(f)
        # print(results)
        for instance in results.to_dict(orient='records'):
            # print(instance['mentions'])
            # print(instance['result'])
            true_mentions = ast.literal_eval(instance['mentions']) if not pd.isna(instance['result']) else []
            result_list = ast.literal_eval(instance['result']) if not pd.isna(instance['result']) else []
            fixed_true_mentions = []
            fixed_result = []
            for mention in result_list:
                fixed_result.append({
                    'text': mention['text'],
                    'label': replace_label(mention['label'])
                })
            for mention in true_mentions:
                fixed_true_mentions.append({
                    'text': mention['text'],
                    'label': replace_label(mention['label'])
                })

            fixed_results.append({
                'text': instance['text'],
                'mentions': fixed_true_mentions,
                'result': fixed_result
            })


        fixed_results_df = pd.DataFrame(fixed_results, columns=['text', 'mentions', 'result'])
        fixed_results_df.to_csv(f'fixed_results/{file.split('\\')[-1]}', index=False)
        # print(results[0])
        f1, precision, recall = evaluate_results(fixed_results, skip_missing=True)
        # print(f"File: {file}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        fixed_scores.append({
            'file': file.split('\\')[-1],
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        })
    fixed_scores_df = pd.DataFrame(fixed_scores, columns=['file', 'F1', 'Precision', 'Recall'])
    fixed_scores_df.to_csv('fixed_scores1.csv', index=False)