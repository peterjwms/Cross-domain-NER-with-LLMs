import ast
from glob import glob
import json

import pandas as pd
from tqdm import tqdm
from grid_search import evaluate_results, Mention
from generate_prompt import get_entity_types
import matplotlib.pyplot as plt

def replace_label(label: str) -> str:
    new_label = label.upper()
    if new_label == 'ORGANIZATION':
        new_label = 'ORG'
    elif new_label == 'LOCATION':
        new_label = 'LOC'
    return new_label

def evaluate_per_label_results(results, domain: str):
    """
    Evaluate the performance of an experiment for each label in the target domain ontology.
    Returns a dictionary where keys are the domain labels
    and values are dictionaries with keys/value pairs for F1, precision, recall, true/false positives, and false negatives for each label
    """

    entity_types = get_entity_types(domain)
    per_label_counts = {}
    per_label_scores = {}
    for entity_type in entity_types:
        per_label_counts[entity_type['label']] = {
            'true_pos': 0,
            'false_pos': 0,
            'false_neg': 0
        }
        per_label_scores[entity_type['label']] = {
            'precision': None,
            'recall': None,
            'f1': None
        }

    
    for instance in results:

        true_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['mentions']])
        if instance['result'] is None:
            predicted_mentions = set([])

        else:
            predicted_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['result']])

        true_positives = true_mentions & predicted_mentions
        false_positives = predicted_mentions - true_positives
        false_negatives = true_mentions - true_positives


        for tp in true_positives:
            per_label_counts[tp.label]['true_pos'] += 1
        for fp in false_positives:
            if fp.label in per_label_counts.keys():
                per_label_counts[fp.label]['false_pos'] += 1
        for fn in false_negatives:
            per_label_counts[fn.label]['false_neg'] += 1
        
    
    for label, counts in per_label_counts.items():
        true_pos = counts['true_pos']
        false_pos = counts['false_pos']
        false_neg = counts['false_neg']

        per_label_scores[label]

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        per_label_scores[label]['precision'] = precision
        per_label_scores[label]['recall'] = recall
        per_label_scores[label]['f1'] = f1
    
    return per_label_scores


def retrieve_results(domain, example, k, selection, split="dev"):
    fixed_results = []
    filepath = f"results/{domain}_{split}_{k}_{example}_{selection}.csv"
    with open(filepath, 'r') as f:
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
    return fixed_results



if __name__ == "__main__":
    dev = 'dev'
    for domain in ['star_wars', 'star_trek', 'red_rising']:
        for k in [0,1,3,5]:
            for example_domain in ['self', 'star_wars']:
                for selection in  ['random','most_dense', 'most_unique']:
                    if domain == example_domain:
                        continue
                    experiment = f'{domain}_{dev}_{k}_{example_domain}_{selection}'
                    results = retrieve_results(domain,example_domain,k,selection)
                    per_label = evaluate_per_label_results(results,domain)
                    per_label_df = pd.DataFrame(per_label).T
                    filepath = f"per_label_results/tables/{domain}_dev_{k}_{example_domain}_{selection}.csv"
                    per_label_df.to_csv(filepath)
                    # with open(filepath,'w') as file:
                    #     file.write(json.dumps(per_label))


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