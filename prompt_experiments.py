import requests
import os
from generate_prompt import preprocess_json, get_entity_types, task_definition_prompt, test_prompt,split_sentences, find_most_diverse
import random


def gemini_api_post_request(api_key, model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "safetySettings": [ 
              {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              },
              {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
              }
            ]
    }

    params = {
        "key": api_key,
    }

    response = requests.post(url, headers=headers, params=params,json=payload)
    return response.json()

api_key =  'AIzaSyAYRzeokD9WTuuRfYv-HKbnQP9rrBUqCFg' #os.environ.get('GEMINI_API_KEY') not working for me on windows

model_name = 'gemini-2.5-flash-preview-04-17'

rr_json = 'label_json/project-8-at-2025-04-25-10-07-1ab099cf.json'
rr_ontology = 'ontologies/red_rising_ontology.tsv'

stuck_sentences = preprocess_json(rr_json)
fixed_sentences = split_sentences(stuck_sentences)

sorted_sents = find_most_diverse(fixed_sentences,unique_only=True)
ontology = 'ontologies/Star Wars Ontology.tsv'
entity_types = get_entity_types(rr_ontology)
prompt = task_definition_prompt(entity_types,examples=sorted_sents[:5])
print(f"{prompt=}")


print(prompt)
reply = gemini_api_post_request(api_key, model_name, prompt)

print(f"{reply=}")

random.seed = 66
random_indices = random.sample(range(len(fixed_sentences)),10)
for i in random_indices:
    prompt = test_prompt(fixed_sentences[i])
    print('NEW ITEM')
    print(fixed_sentences[i]['text'])
    print(fixed_sentences[i]['mentions'])
    reply = gemini_api_post_request(api_key, model_name, prompt)

    
print(f"{reply=}")
print(f"{reply['candidates'][0]['content']['parts'][0]=}")