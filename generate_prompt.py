import json
import spacy

HEADER = "You are now and entity recognition model. Always answers as helpfully as possible."

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

def generate_basic_prompt(sentence, entity_types):
    type_descriptions = '\n'.join([e['label'] + ": " + e['description'] for e in entity_types])
    prompt = f"{HEADER} \n"


stuck_sentences = preprocess_json('label_json/project-1-at-2025-04-24-09-56-a64fbdd4.json')
fixed_sentences = split_sentences(stuck_sentences)

print(fixed_sentences)
