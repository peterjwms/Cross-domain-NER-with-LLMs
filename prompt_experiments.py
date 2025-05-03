from pprint import pprint
import requests
import os
from generate_prompt import  get_entity_types, get_train_test_dev_data, create_k_shot_prompt
import random
from dotenv import load_dotenv
load_dotenv()


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
    




if __name__ == "__main__":
    api_key =  os.environ.get('GOOGLE_API_KEY')
    print(f"{api_key=}")

    model_name = 'gemini-2.5-flash-preview-04-17'

    for domain in ["red_rising", "star_wars"]:
        train, test, dev = get_train_test_dev_data(domain)
        # ontology = f'ontologies/{domain}.tsv'
        entity_types = get_entity_types(domain)

        prompt = create_k_shot_prompt(train, 'star_wars', 'most_unique', 5, entity_types)
        pprint(f"{prompt=}")

        pprint(prompt)
        reply = gemini_api_post_request(api_key, model_name, prompt)

        pprint(reply)
        pprint(reply['candidates'][0]['content']['parts'][0])
        pprint(f"{reply=}")







    # random.seed = 66
    # random_indices = random.sample(range(len(fixed_sentences)),10)
    # for i in random_indices:
    #     prompt = test_prompt(fixed_sentences[i])
    #     print('NEW ITEM')
    #     print(fixed_sentences[i]['text'])
    #     print(fixed_sentences[i]['mentions'])
    #     reply = gemini_api_post_request(api_key, model_name, prompt)

        
    print(f"{reply['candidates'][0]['content']['parts'][0]=}")
