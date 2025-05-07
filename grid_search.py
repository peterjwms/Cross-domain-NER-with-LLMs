import json
import os
from pprint import pprint
from time import sleep
import warnings
from dotenv import load_dotenv
from google import genai
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from generate_prompt import get_entity_types, create_k_examples, get_train_test_dev_data, load_data_split, stringify_ontology, task_definition_prompt, get_k_examples
from prompt_experiments import gemini_api_post_request
import re
import ast

class Mention(BaseModel):
    text: str
    label: str

    def __init__(self, text: str, label: str):
        super().__init__(text=re.sub(r"'s$", "", text), label=label)


    def __eq__(self, other: "Mention") -> bool:
        if not isinstance(other, Mention):
            raise TypeError(f'Incompatible types: Mention and {type(other)}')
        return self.label == other.label and self.text == other.text
    
    def __hash__(self) -> int:
        return hash((self.label, self.text))
    
    def ispartialmatch(self, other: "Mention") -> bool:
        if not isinstance(other, Mention):
            raise TypeError(f'Incompatible types: Mention and {type(other)}')
        return NotImplementedError

        # TODO: maybe implement this further to check for equality more easily


def run_grid_search(model, client, parameters: dict = None):
    """
    Run a grid search over the parameters provided.
    """

    # TODO: low-priority - could adjust create_k_examples to not return anything and to just create the files
    # create_all_examples(parameters['target_domain'], parameters['example_selection'])

    # TODO: low-priority - potentially add indices to the dev/test/train splits that we can use to better evaluate
    scores_list = []
    # i = 0
    for domain in tqdm(parameters['target_domain'], desc="Domains", unit="domain"):
        # load the test set, ontology, and create task definition prompt for the domain
        dataset = load_data_split(domain, parameters['dataset'])
        type_descriptions = stringify_ontology(domain)
        prompt_definition = task_definition_prompt(domain, type_descriptions)
        
        for example_domain in tqdm(parameters['example_domain'], desc="Example Domains", leave=False):
            if example_domain == domain:
                continue # skip for star_wars where there is no transfer learning
            # TODO: other example selection methods?? - random
            for example_selection in tqdm(parameters['example_selection'], desc="Example Selection Methods", leave=False):
                # get all examples for the domain using the selection method
                examples = get_all_domain_examples(domain, example_domain, example_selection)

                for n_examples in tqdm(parameters['n_examples'], desc="Number of Examples", leave=False):
                    # print(f"Domain: {domain}, n_examples: {n_examples}, example_domain: {example_domain}, example_selection: {example_selection}")
                    # i += 1

                    # fill in {{examples}} slot in prompt with n_examples for this experiment
                    base_prompt = prompt_definition.replace('{{examples}}', get_k_examples(n_examples, examples, example_domain))
                    # print(base_prompt)

                    # run the experiment here on a full dev/test set using the base prompt built above
                    results = run_experiments(base_prompt, dataset, model, client)
                    results_df = pd.DataFrame(results, columns=['text', 'mentions', 'result'])
                    results_df.to_csv(f"results/{domain}_{parameters['dataset']}_{n_examples}_{example_domain}_{example_selection}.csv", index=False)
                    
                    # evaluate results
                    f1, precision, recall = evaluate_results(results)
                    # print(f"F1: {f1}, precision: {precision}, recall: {recall}")
                    
                    # append results to scores dataframe
                    scores_list.append(pd.DataFrame({'domain': domain, 'n_examples': n_examples, 'example_domain': example_domain, 
                                                     'example_selection': example_selection, 'prompt': base_prompt, 
                                                     'F1': f1, 'precision': precision, 'recall': recall}, index=[0]))
    
    # print(i)
    scores_df = pd.concat(scores_list, ignore_index=True)
    scores_df.to_csv(f'{parameters['dataset']}_grid_search_results.csv', index=False)

    return results


def create_all_examples(domains, methods):
    """
    Create all examples for each given domain for each selection method.
    """
    for domain in domains:
        train = load_data_split(domain, 'train')
        for method in methods:
            create_k_examples(train, domain, method, 5)


def get_all_domain_examples(target_domain, example_domain, example_selection):
    # TODO: could add a check to see if the file exists and if not create it - shouldn't be necessary because we create all of them first in the grid search
    if example_domain == 'self':
        example_domain = target_domain
        # print(f"Using {example_domain} as example domain")
    
    with open(f"sorted_examples/{example_domain}_{example_selection}.json", 'r') as file:
        examples = json.load(file)
        # pprint(f"Loaded {len(examples)} examples from {example_domain} using {example_selection} method")
        # pprint(examples)
    return examples


def run_experiments(base_prompt, dataset, model_name, client: genai.Client):
    """
    Run experiment on all instances of the test set, using the prompt provided.
    """
    results = []
    i = 0
    for test_instance in tqdm(dataset, desc="Running experiments", unit="instance", leave=False):
        full_prompt = base_prompt.replace("{{test_instance}}", test_instance['text'])
        # i += 1
        # if i == 5: # for testing purposes
        #     break
        try:
            # TODO: look into batch embed requests
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': list[Mention]
                }
            )
            # print(response.text)
            # sleep(2) # rate limits - 30 requests per minute, this empirically seems to work
            json_result = json.loads(response.text)
            # print(json_result)
            # TODO: fix these results to look like json with ""
            # also need to make sure anything that's empty is an empty list
            results.append({'text': test_instance['text'], 'mentions': test_instance['mentions'], 'result': json_result}),
        except Exception as e:
            print(f"Error: {e}") # most likely a rate limit error
            # print(reply)
            results.append({'text': test_instance['text'], 'mentions': test_instance['mentions'], 'result': None})        
        

    return results


def evaluate_results(results, skip_missing: bool = True):
    """
    Evaluate results of the experiment for exact mention matching.
    Returns the F1, precision, recall.
    """
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    unusable_results = 0

    for instance in results:
        # this gets per-instance results
        true_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['mentions']])
        if instance['result'] is None:
            unusable_results += 1
            if skip_missing:
                continue
            predicted_mentions = set([])

        else:
            predicted_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['result']])

        true_positives = true_mentions & predicted_mentions
        false_positives = predicted_mentions - true_positives
        false_negatives = true_mentions - true_positives

        overall_tp += len(true_positives)
        overall_fp += len(false_positives)
        overall_fn += len(false_negatives)
    
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if skip_missing:
        pass
        with open('results/evaluation.txt', 'a') as file:
            print(f'{unusable_results} of {len(results)} instances could not be evaluated', file=file)
    
    return float(f"{f1*100:0.2f}"), float(f"{precision*100:0.2f}"), float(f"{recall*100:0.2f}")


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
            'flase_neg': 0
        }
        per_label_scores[entity_type['label']] = {
            'precision': None,
            'recall': None,
            'f1': None
        }

    
    for instance in results:

        true_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['mentions']])
        if instance['results'] is None:
            predicted_mentions = set([])

        else:
            predicted_mentions = set([Mention(mention['text'], mention['label']) for mention in instance['results']])

        true_positives = true_mentions & predicted_mentions
        false_positives = predicted_mentions - true_positives
        false_negatives = true_mentions - true_positives


        for tp in true_positives:
            per_label_counts[tp.label]['true_pos'] += 1
        for fp in false_positives:
            per_label_counts[fp.label]['false_pos'] += 1
        for fn in false_negatives:
            per_label_counts[fn.label]['false_neg'] += 1
        
    
    for label, counts in per_label_counts.items():
        true_pos = counts['true_pos']
        false_pos = counts['false_pos']
        false_neg = counts['false_pos']

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
    filepath = f"results/{domain}_{split}_{k}_{example}_{selection}"
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
    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')
    # model_name = 'gemini-2.5-flash-preview-04-17' # RPM 10, TPM 250,000, RPD 500
    model_name = 'gemini-2.0-flash-lite' # RPM 30, TPM 1,000,000, RPD 1500
    client = genai.Client(api_key=api_key)


    

    search_parameters = {
        'dataset': 'dev',
        'n_examples': [0,1,3,5],
        'example_domain': ['self', 'star_wars'],
        'example_selection': ['random','most_dense', 'most_unique'],
        'target_domain': ['star_wars', 'star_trek', 'red_rising']
    }

    # just used in place of search_parameters to test evaluation
    mini_test = {
        'dataset': 'dev',
        'n_examples': [3,5],
        'example_domain': ['self'],
        'example_selection': ['random','most_dense', 'most_unique'],
        'target_domain': ['star_trek']
    }

    # creates train/test/dev splits for each domain if they don't already exist
    for domain in search_parameters['target_domain']:
        _, _, _ = get_train_test_dev_data(domain, n=100)

    # reset the log evaluation file for a new run
    with open('results/evaluation.txt', 'w') as file:
        print(f"Start grid search at {pd.Timestamp.now()}", file=file)
        pass

    create_all_examples(search_parameters['target_domain'], search_parameters['example_selection'])
    # results = run_grid_search(model_name, client, mini_test)
    results = run_grid_search(model_name, client, search_parameters)


    # notes from presentation
    # should run each experiment multiple times (3-5) to get an average/range of scores
    # dense examples sometimes will just get things like lists, which aren't necessarily good/helpful examples
    # might be good to pull cross-domain examples that only have entity types that are also in the target domain
