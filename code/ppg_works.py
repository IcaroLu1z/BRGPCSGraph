import pandas as pd
import json
import requests

#-----------------------------------------------------------------------------------------------------------------------------
# Load data and preprocess
ppg_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors.csv')

ppg_data = ppg_data.dropna(subset=['author_id'])
ppg_data.drop_duplicates(subset=['author_id'], inplace=True)
ppg_data = ppg_data.fillna("null")

ppg_data.to_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv', index=False)

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
    for author_id in ppg_data['author_id']:
        url = f'https://api.openalex.org/works?filter=author.id:{author_id},from_publication_date:2004-01-01'
        # loop through pages
        cursor = '*'
        while cursor:
            # set cursor value and request page from OpenAlex
            url_1 = f'{url}&select={select}&cursor={cursor}'
            page_with_results = requests.get(url_1).json()

            results = page_with_results['results']
            for result in results:
                work_id = result['id']
                if work_id not in work_ids:  # Check if work ID is already in the set
                    work_ids.add(work_id)  # Add work ID to the set
                    works.append(result)  # Append the work to the list
                    n_works += 1
            # update cursor to meta.next_cursor
            cursor = page_with_results['meta']['next_cursor']
            calls += 1
            if calls in [5, 10, 20, 50, 100] or calls % 500 == 0:
                print(f'{calls} api requests made so far')

    print(f'done. made {calls} api requests. collected {len(works)} works / {len(work_ids)} unique work ids.')

except Exception as e:
    print(f'An exception occurred: {str(e)}')

with open('/media/work/icarovasconcelos/mono/data/raw_works_since_2004.json', 'w') as f:
    json.dump(works, f)
    
#-----------------------------------------------------------------------------------------------------------------------------

data = []
for work in works:
    for authorship in work['authorships']:
        if authorship:
            author = authorship['author']
            author_id = author['id'].split('/')[-1] if author else None
            author_name = author['display_name'] if author else None
            author_position = authorship['author_position']
            for institution in authorship['institutions']:
                if institution:
                    institution_id = institution['id'].split('/')[-1]
                    institution_name = institution['display_name']
                    institution_country_code = institution['country_code']
                    topic_name = work['topics'][0]['display_name'] if 'topics' in work and work['topics'] else None
                    data.append({
                        'work_id': work['id'].split('/')[-1],
                        'work_title': work['title'],
                        'work_display_name': work['display_name'],
                        'work_publication_year': work['publication_year'],
                        'work_publication_date': work['publication_date'],
                        'author_id': author_id,
                        'author_name': author_name,
                        'author_position': author_position,
                        'institution_id': institution_id,
                        'institution_name': institution_name,
                        'institution_country_code': institution_country_code,
                        'topic_name': topic_name,
                    })
                    
df_works = pd.DataFrame(data)
df_works.to_csv('/media/work/icarovasconcelos/mono/data/organized_works_since_2004.csv', index=False)
print(len(data))