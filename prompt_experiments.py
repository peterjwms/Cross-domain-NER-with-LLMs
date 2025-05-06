from pprint import pprint
import requests
import os
from generate_prompt import  get_entity_types, domain_name_print, load_data_split, get_train_test_dev_data
import random
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    

def calculate_label_distributions():
  for domain in ["star_wars","star_trek","red_rising"]:
    entity_types = get_entity_types(domain)
    entity_counter = {}
    for entity_type in entity_types:
      entity_counter[entity_type['label']] = 0
    entity_counts_all = entity_counter
     
    
     
    

    for split in ["train","test","dev"]:
      data_json = load_data_split(domain, split)
      entity_counts_split = entity_counter

      data_json = load_data_split(domain, split)
      for instance in data_json:
          for mention in instance['mentions']:
            entity_counts_split[mention['label']] += 1
            entity_counts_all[mention['label']] += 1
      with open(f"entity_distributions/{domain}_{split}.tsv", 'w') as file:
          for entity_type in entity_types:
            file.write(entity_type['label'] + '\t' + str(entity_counts_split[entity_type['label']]) + '\n')
      draw_pie_chart(domain, entity_counts=entity_counts_split, split=split)

    with open(f"entity_distributions/{domain}_all.tsv", 'w') as file:
      for entity_type in entity_types:
        file.write(entity_type['label'] + '\t' + str(entity_counts_all[entity_type['label'] ]) + '\n')
    draw_pie_chart(domain,entity_counts=entity_counts_split)
          
        
def draw_pie_chart(domain: str, entity_counts: dict, split: str = 'all'):
  if split == 'all':
      split_name = 'Full Corpus'
  else:
      split_name = split.capitalize() + "Split"
      
  labels = entity_counts.keys()
  values = entity_counts.values()

  cmap = cm.get_cmap('nipy_spectral', len(values))
  colors = [cmap(i) for i in range(len(values))]
  plt.pie(values, labels=None, autopct='%1.1f%%', colors=colors.reverse())
  # plt.subplots_adjust(left=0.8)

  plt.legend(labels=[f'{l}, {v}' for l, v in zip(labels, values)],loc="upper right",bbox_to_anchor=(1, 1))
  plt.title(f'{domain_name_print(domain)} Entity Distribution for {split_name}')
  plt.savefig(f"entity_distributions/{domain}_{split}_pie.png")
  plt.clf()



if __name__ == "__main__":
    for domain in ['star_wars','star_trek','red_rising']:
       _, _, _ = get_train_test_dev_data(domain)
    calculate_label_distributions()
    # api_key =  os.environ.get('GOOGLE_API_KEY')
    # print(f"{api_key=}")

    # model_name = 'gemini-2.5-flash-preview-04-17'

    # for domain in ["red_rising"]:
    #     train, test, dev = get_train_test_dev_data(domain)
    #     # ontology = f'ontologies/{domain}.tsv'
    #     entity_types = get_entity_types(domain)

    #     prompt = create_k_shot_prompt(train, 'star_wars', 'most_unique', 5, entity_types)
    #     pprint(f"{prompt=}")

    #     pprint(prompt)
    #     reply = gemini_api_post_request(api_key, model_name, prompt)

    #     pprint(reply)
    #     pprint(reply['candidates'][0]['content']['parts'][0])
    #     pprint(f"{reply=}")







    # random.seed = 66
    # random_indices = random.sample(range(len(fixed_sentences)),10)
    # for i in random_indices:
    #     prompt = test_prompt(fixed_sentences[i])
    #     print('NEW ITEM')
    #     print(fixed_sentences[i]['text'])
    #     print(fixed_sentences[i]['mentions'])
    #     reply = gemini_api_post_request(api_key, model_name, prompt)

        
    #print(f"{reply['candidates'][0]['content']['parts'][0]=}")
