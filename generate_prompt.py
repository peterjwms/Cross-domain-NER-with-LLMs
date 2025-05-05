import json
import spacy
import csv
import os
import random

HEADER = ""

def preprocess_json(filepath: str) -> list[dict]:
    with open(f"label_json/{filepath}.json", 'r') as file:
        data = json.load(file)
    sentences = []
    for entry in data:
        if entry['total_annotations'] == 0:
            continue  # skip items with cancelled annotations
        sentence = {}
        sentence['text'] = entry['data']['text']
        mentions = []
        for annotation in entry['annotations'][0]['result']:
            mention = annotation['value']
            mentions.append({
                'label': mention['labels'][0],
                'start': mention['start'],
                'end': mention['end'],
                'text': mention['text']
                })
        sentence['mentions'] = mentions
        sentences.append(sentence)
    return sentences


def split_sentences(entries: list[dict]) -> list[dict]:
    nlp = spacy.load("en_core_web_sm")
    all_sentences = []
    for entry in entries:
        doc = nlp(entry['text'])      
        ordered_sentences = [{'text': sent.text, 'mentions': []} for sent in doc.sents]
        i = 0
        for mention in entry['mentions']:
            for i, sent in enumerate(doc.sents):
                if mention["start"] >= sent.start_char and mention["end"] <= sent.end_char:
                    mention["start"] -= sent.start_char
                    mention["end"] -= sent.start_char
                    ordered_sentences[i]["mentions"].append(mention)
        all_sentences += ordered_sentences
    return all_sentences


def domain_name_print(domain: str) -> str:
    match domain:
        case 'star_wars':
            return 'Star Wars'
        case 'star_trek':
            return 'Star Trek'
        case 'red_rising':
            return 'Red Rising'


def task_definition_prompt(target_domain, type_descriptions):
    with open(f'prompts/current_prompt.txt', 'r') as file:
        prompt = file.read()

    prompt = prompt.replace('{{domain_name}}',domain_name_print(target_domain))
    prompt = prompt.replace('{{ontology}}',type_descriptions)

    return prompt


def stringify_ontology(target_domain):
    type_descriptions = '\n'.join([e['label'] + ": " + e['name'] + ' - ' + e['description'] for e in get_entity_types(target_domain)])
    return type_descriptions


def get_k_examples(k: int, examples: list[dict], example_domain: str) -> str:
    examples_string = ''
    if k > 0:
        if example_domain == 'self':
            examples_string += "Use these examples to train your tagging system: \n"
        else:
            examples_string += f"Use these examples of {domain_name_print(example_domain)} data to train your tagging system. This is for cross-domain learning."
            examples_string += "These examples may have different types from those you are asked to tag. Do NOT tag such items, but let the examples inform your tagging decisions.\n"
        for i in range(k):
            examples_string += f"\"{examples[i]['text']}\"" + '\n'
            for mention in examples[i]['mentions']:
                examples_string += json.dumps({'entity': mention['text'], 'label': mention['label']})  + '\n' # f"{mention['label']}: {mention['text']}\n"
    return examples_string


def  get_entity_types(domain):
    """Returns entity types as a list of dicts.
    Use stringify_ontology to get a string representation for a domain's ontology."""
    entity_types = []
    with open(f'ontologies/{domain}.tsv','r') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        for line in tsv_reader:
            entity_types.append({'name': line[0], 'label': line[1], 'description': line[2]})
    return entity_types


def find_most_diverse(sentences, unique_only=False):
    return sorted(sentences, key=lambda x: count_mentions(x,unique_only), reverse=True)


def count_mentions(x,unique_only):
    if not unique_only:
        return len(x['mentions'])
    else:
        return len(set([mention['label'] for mention in x['mentions']]))


def create_k_shot_prompt(train_data, source_name, selection_method, k: int, entity_types) -> None:
    examples = create_k_examples(train_data, source_name, selection_method, k) if k > 0 else None
    return task_definition_prompt(entity_types=entity_types, examples=examples)


def create_k_examples(train_data, source_name, selection_method, k: int) -> None:
    filepath = f'sorted_examples/{source_name}_{selection_method}.json'
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)[:k]
    else:
        
        if selection_method == 'most_dense':
            ordered = find_most_diverse(train_data)
        if selection_method == 'most_unique':
            ordered = find_most_diverse(train_data, unique_only=True)

        with open(f'sorted_examples/{source_name}_{selection_method}.json','w') as file:
            json.dump(ordered, file, indent=4)
            # for item in ordered:
            #     file.write(json.dumps(item) + '\n')
        return ordered[:k]


def load_data_split(name, split): 
    with open(f'data_splits/{name}_{split}.json', 'r') as file:
        return json.load(file)


def save_data_split(name, data, data_split):
    with open(f'data_splits/{name}_{data_split}.json', 'w') as file:
        json.dump(data, file, indent=4)
        # for item in data:
        #     file.write(json.dumps(item) + '\n')


def get_train_test_dev_data(name: str, n: int = 100,):
    if os.path.isfile(f'data_splits/{name}_test.json'):

        test = load_data_split(name, 'test')
        train = load_data_split(name, 'train')
        dev = load_data_split(name, 'dev')

    else:
        data = preprocess_json(name)
        data = split_sentences(data)
        random.seed(42)
        random.shuffle(data)

        test = data[:n]
        dev = data[n:2*n]
        train = data[2*n:]

        save_data_split(name, test, 'test')
        save_data_split(name, train, 'train')
        save_data_split(name, dev, 'dev')

    return train, dev, test