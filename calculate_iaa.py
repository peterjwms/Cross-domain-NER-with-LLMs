# calculate IAA for each domain

import json
from pathlib import Path

import pandas as pd

from grid_search import Mention
from re_evaluate import replace_label

def get_domain_annotations(domain: str):
    with open(Path(f"{domain}_IAA.json"), 'r') as file:
        st_data = json.load(file)

    annotations = []
    texts = []
    for instance in st_data:
        text = instance['data']['text']
        # anno1 = instance['data']['mentions']
        # for mention in anno1:
        #     mention['label'] = replace_label(mention['label'])

        anno1 = [
            {
            'label': replace_label(mention['label']),
            'start': mention['start'],
            'end': mention['end'],
            'text': mention['text']
        }
        for mention in instance['data']['mentions']
        ]
    
        anno2 = [
            {
            'label': replace_label(item['value']['labels'][0]),
            'start': item['value']['start'],
            'end': item['value']['end'],
            'text': item['value']['text']
            } 
        for item in instance['annotations'][0]['result']
        ]

        texts.append(text)
        annotations.append((anno1, anno2))

    return texts, annotations


def get_f1(annotations):

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for anno1, anno2 in annotations:
        anno1 = set(Mention(mention['text'].strip(' ,.?!\"\''), mention['label']) for mention in anno1)
        anno2 = set(Mention(mention['text'].strip(' ,.?!\"\''), mention['label']) for mention in anno2)
        
        print(anno1)
        print(anno2)
        print("\n")
        tp = (anno1 & anno2)
        fp = (anno1 - anno2)
        fn = (anno2 - anno1)

        print(tp)
        print(fp)
        print(fn)
        print("=="*20)

        overall_tp += len(tp)
        overall_fp += len(fp)
        overall_fn += len(fn)

    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


if __name__ == "__main__":
    domains = ['star_wars', 'star_trek', 'red_rising']

    # texts, st_annotations = get_domain_annotations('star_trek')
    # print(st_annotations)

    # prec, rec, f1 = get_f1(st_annotations)
    # print(f"Precision: {prec}, Recall: {rec}, F1: {f1}")
    results = []
    for domain in domains:
        texts, annotations = get_domain_annotations(domain)
        print(f"Domain: {domain}")

        prec, rec, f1 = get_f1(annotations)
        print(f"Precision: {prec}, Recall: {rec}, F1: {f1}")

        results.append({
            'domain': domain,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('IAA_results.csv', index=False)


# red rising - confusion between rank/profession, planet/loc (from changes to ontology)
# also issues for tech/weapon and rank/per for the annotators less familiar with the data
# star trek - mostly around what is/isn't named tech, and where the boundary is
# star wars - droid/per, org/gpe/norp/species (Jedi)