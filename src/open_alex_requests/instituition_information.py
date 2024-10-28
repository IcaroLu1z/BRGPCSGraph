import pandas as pd
import json
import requests
import time

#-----------------------------------------------------------------------------------------------------------------------------
# Load data and preprocess
    
authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-processed.csv')

#------------------------------------------------------------------------------------------------------------------------------

instituition_list = set()
for inst_id in authors_data['institution_id']:
    instituition_list.add(inst_id)
    
print(f'Number of instituitions: {len(instituition_list)}')

select = ",".join((
    'id',
    'display_name',
    'country_code',
    'type',
    'works_count',
    'cited_by_count',
    'geo',
))

print('Starting to fetch instituitions information')
n_inst = 0
calls = 0
instituitions = []  # Use a list to store the work details
id = None

for inst in instituition_list:
    try:
        id = inst
        url = f'https://api.openalex.org/institutions/{id}'
        
        # set cursor value and request page from OpenAlex
        url_1 = f'{url}&select={select}'
        results = requests.get(url_1).json()
        n_inst += 1
        instituitions.append(results)
        calls += 1
        inst_name = results['display_name']
        works = results['works_count']
        cited_count = results['cited_by_count']
        print(f'{id} {n_inst}/{len(instituition_list)}: {inst_name} has {works} works and is cited {cited_count} times')
    except Exception as e:
        print(f'An exception occurred: {str(e)}, calls made: {calls}, authors collected: {len(instituitions)}, last_author: {id}')

print(f'done. made {calls} api requests. collected {len(instituitions)} authors.')
with open('/media/work/icarovasconcelos/mono/data/authors/instituition_information.json', 'w') as f:
    json.dump(instituitions, f)

#-----------------------------------------------------------------------------------------------------------------------------
