import requests
import os
from generate_prompt import  get_entity_types, test_prompt, get_train_test_dev_data, create_k_shot_prompt
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
    



api_key =  os.environ.get('GOOGLE_API_KEY')

model_name = 'gemini-2.5-flash-preview-04-17'



train, test, dev = get_train_test_dev_data('star_wars')
ontology = 'ontologies/star_wars.tsv'
entity_types = get_entity_types(ontology)

prompt = create_k_shot_prompt(train, 'star_wars', 'most_unique', 5, entity_types)

# stuck_sentences = preprocess_json('label_json/star_wars.json')
# fixed_sentences = split_sentences(stuck_sentences)

# sorted_sents = find_most_diverse(fixed_sentences,unique_only=True)


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

print(reply)
print(reply['candidates'][0]['content']['parts'][0])
print(f"{reply=}")





# random.seed = 66
# random_indices = random.sample(range(len(fixed_sentences)),10)
# for i in random_indices:
#     prompt = test_prompt(fixed_sentences[i])
#     print('NEW ITEM')
#     print(fixed_sentences[i]['text'])
#     print(fixed_sentences[i]['mentions'])
#     reply = gemini_api_post_request(api_key, model_name, prompt)

    
print(f"{reply['candidates'][0]['content']['parts'][0]=}")
