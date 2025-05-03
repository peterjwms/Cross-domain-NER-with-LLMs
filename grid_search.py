import json
import os
from pprint import pprint
from dotenv import load_dotenv
import pandas as pd
from generate_prompt import create_k_examples, load_data_split, task_definition_prompt, get_k_examples
from prompt_experiments import gemini_api_post_request


def run_grid_search(model, api_key, parameters: dict = None):
    """
    Run a grid search over the parameters provided.
    """

    create_all_examples(parameters['target_domain'], parameters['example_selection'])

    # TODO: potentially add indices to the dev/test/train splits that we can use to better evaluate
    scores_df = pd.DataFrame(columns=['domain', 'n_examples', 'example_domain', 'example_selection', 'prompt', 'score'])
    i = 0
    for domain in parameters['target_domain']:
        # TODO: load the test set for the domain here
        dataset = load_data_split(domain, parameters['dataset'])
        for example_domain in parameters['example_domain']:
            for example_selection in parameters['example_selection']:
                # TODO: load the sorted examples for the domain and selection method here
                prompt_definition = task_definition_prompt(target_domain=domain)
                examples = get_all_domain_examples(domain, example_domain, example_selection)

                for n_examples in parameters['n_examples']:
                    print(f"Domain: {domain}, n_examples: {n_examples}, example_domain: {example_domain}, example_selection: {example_selection}")
                    i += 1

                    # TODO: could change this to load the examples once, and then slice to the correct number here 
                    # would be marginally faster by not loading the same file multiple times 
                    
                    base_prompt = prompt_definition.replace('{{examples}}', get_k_examples(n_examples,examples))
                    
                    # TODO: could change this function to also take pieces of the prompt we might want to test 
                    # basically anything that's not the examples or test instance
                    # can use the same format as the {{test_instance}}} in the prompt already for the examples and ontology
                    # allows for better tracking of changes to the prompt
                    
                    

                    # run the experiment here on a full dev/test set using the base prompt built above
                    # TODO: think about what we actually want to return and store from this
                    # could save this full set of results to a file for later analysis
                    # then could also save just the actual score post-evaluation 
                    results = run_experiments(base_prompt, dataset, model, api_key)
                    results_df = pd.DataFrame(results, columns=['text', 'mentions', 'result'])
                    results_df.to_csv(f"results/{domain}_{parameters['dataset']}_{n_examples}_{example_domain}_{example_selection}.csv", index=False)
                    
                    # TODO: evaluate results
                    # score = evaluate_results(results)
                    
                    # TODO: save results to dataframe
                    # scores_df = pd.concat([scores_df, 
                    #                        pd.DataFrame({'domain': domain, 'n_examples': n_examples, 
                    #                                      'example_domain': example_domain, 'example_selection': example_selection,
                    #                                      'prompt': prompt_definition, 'score': score})])
    
    # print(i)
    # TODO: save all results to csv
    # scores_df.to_csv('grid_search_results.csv', index=False)

    return results


def create_all_examples(domains, methods):
    """
    Create all examples for each given domain for each selection method.
    """
    for domain in domains:
        # print(f"Creating examples for {domain}")
        train = load_data_split(domain, 'train')
        for method in methods:
            # TODO: could alter this to actually retrieve the examples here instead of just creating the files if they don't exist
            create_k_examples(train, domain, method, 5)
            # print(f"Created examples for {domain} using {method} method")
    pass


def run_experiments(base_prompt, test_set, model_name, api_key):
    """
    Run experiment on all instances of the test set, using the prompt provided.
    """
    results = []
    for i, test_instance in enumerate(test_set):
        full_prompt = base_prompt.replace("{{test_instance}}", test_instance['text'])
        # print(full_prompt)
        
        reply = gemini_api_post_request(api_key, model_name, full_prompt)
        print(reply)
        result = reply['candidates'][0]['content']['parts'][0]
        # print(result)
        results.append({'text': test_instance['text'], 'mentions': test_instance['mentions'], 'result': result}),
        
        if i == 5: # for testing purposes
            break

    return results


def get_all_domain_examples(target_domain, example_domain, example_selection):
    if example_domain == 'self':
        example_domain = target_domain
    
    with open(f"sorted_examples/{example_domain}_{example_selection}.json", 'r') as file:
        examples = json.load(file)
        pprint(f"Loaded {len(examples)} examples from {example_domain} using {example_selection} method")
        pprint(examples)
    return examples


if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ.get('GOOGLE_API_KEY')
    model_name = 'gemini-2.5-flash-preview-04-17'

    search_parameters = {
        'dataset': 'dev',
        'n_examples': [0,1,3,5],
        'example_domain': ['self', 'star_wars'],
        'example_selection': ['most_dense', 'most_unique'],
        'target_domain': ['star_wars', 'red_rising' ]# , 'star_trek']
    }
    create_all_examples(search_parameters['target_domain'], search_parameters['example_selection'])
    results = run_grid_search(model_name, api_key, search_parameters)