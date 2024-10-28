import pandas as pd
import json
import requests

#-----------------------------------------------------------------------------------------------------------------------------
# Load data and preprocess

ppg_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')

#-----------------------------------------------------------------------------------------------------------------------------

cursor = '*'

select = ",".join((
    'id',
    'ids',
    'title',
    'display_name',
    'publication_year',
    'publication_date',
    'primary_location',
    'open_access',
    'authorships',
    'cited_by_count',
    'is_retracted',
    'is_paratext',
    'updated_date',
    'created_date',
    'topics',

))

n_works = 0
calls = 0
work_ids = set()  # Use a set to store unique work IDs
works = []  # Use a list to store the work details
try:
    for i, author_id in enumerate(ppg_data['author_id']):
        url = f'https://api.openalex.org/works?filter=author.id:{author_id},from_publication_date:2004-01-01'
        # loop through pages
        cursor = '*'
        author_works_count = 0
        while cursor:
            # set cursor value and request page from OpenAlex
            url_1 = f'{url}&select={select}&cursor={cursor}'
            page_with_results = requests.get(url_1).json()
            results = page_with_results['results']
            author_works_count = page_with_results['meta']['count']
            for result in results:
                work_id = result['id']
                if work_id not in work_ids:  # Check if work ID is already in the set
                    work_ids.add(work_id)  # Add work ID to the set
                    works.append(result)  # Append the work to the list
                    n_works += 1
            # update cursor to meta.next_cursor
            cursor = page_with_results['meta']['next_cursor']
            calls += 1
        print(f'{author_id} {i+1}/{len(ppg_data)}: {author_works_count} author works / {n_works} total works')

    print(f'done. made {calls} api requests. collected {len(works)} works / {len(work_ids)} unique work ids.')

except Exception as e:
    print(f'An exception occurred: {str(e)} / Last Author ID: {author_id}')
    

with open('/media/work/icarovasconcelos/mono/data/raw_works_since_2004.json', 'w') as f:
    json.dump(works, f)
    
#-----------------------------------------------------------------------------------------------------------------------------

