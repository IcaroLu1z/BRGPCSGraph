import pandas as pd
import json
import requests
import time

#-----------------------------------------------------------------------------------------------------------------------------
# Load data and preprocess
    
authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')

#------------------------------------------------------------------------------------------------------------------------------

authors_list = []
for author_id in authors_data['author_id']:
    authors_list.append(author_id)
    
print(f'Number of authors: {len(authors_list)}')

select = ",".join((
    'id',
    'display_name',
    'works_count',
    'cited_by_count',
    'summary_stats',
    'last_known_institution',
    'topics',
))

print('Starting to fetch authors information')
n_authors = 0
calls = 0
authors = []  # Use a list to store the work details
id = None

for author in authors_list:
    try:
        id = author
        url = f'https://api.openalex.org/authors/{id}'
        
        # set cursor value and request page from OpenAlex
        url_1 = f'{url}&select={select}'
        results = requests.get(url_1).json()
        n_authors += 1
        authors.append(results)
        calls += 1
        author_name = results['display_name']
        works = results['works_count']
        cited_count = results['cited_by_count']
        print(f'{id} {n_authors}/{len(authors_list)}: {author_name} has {works} works and is cited {cited_count} times')
    except Exception as e:
        print(f'An exception occurred: {str(e)}, calls made: {calls}, authors collected: {len(authors)}, last_author: {id}')

print(f'done. made {calls} api requests. collected {len(authors)} authors.')
with open('/media/work/icarovasconcelos/mono/data/authors/authors_information.json', 'w') as f:
    json.dump(authors, f)

#-----------------------------------------------------------------------------------------------------------------------------
