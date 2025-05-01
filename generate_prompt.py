import json
import spacy
import csv

HEADER = "You are now an entity recognition model. Always answers as helpfully as possible."

def preprocess_json(filepath: str) -> list[dict]:
    with open(filepath, 'r') as file:
        data = json.load(file)
    sentences = []
    for entry in data:
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


def task_definition_prompt(entity_types, examples=None):
    type_descriptions = '\n'.join([e['label'] + ": " + e['name'] + ' - ' + e['description'] for e in entity_types])
    prompt = HEADER + '\n' + 'These are the entity types you are tasked to identify:' + '\n'
    prompt += type_descriptions + '\n'
    if examples is not None:
        prompt += "Use these examples to train your tagging system: "
        for example in examples:
            prompt += example['text'] + '\n'
            for mention in example['mentions']:
                prompt += f"{mention['label']}: {mention['text']}\n"
    prompt += "Please label all entities that fit these descriptions. Do NOT give any labels that are not in the above ontology\n"
    return prompt


def test_prompt(sentence):
    prompt = "Tag all named entities in the following sentence: \n"
    prompt += sentence["text"]
    prompt += "Please format your output as follows for each entity in the sentence: \n"
    prompt += "LABEL: text"
    return prompt


def  get_entity_types(ontology):
    entity_types = []
    with open(ontology,'r') as file:
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
