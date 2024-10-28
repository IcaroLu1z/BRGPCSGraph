import pandas as pd
import json
import numpy as np
import os
import requests
import zipfile
import geopandas as gpd
from shapely.geometry import Point
import gdown

authors_csv = "https://docs.google.com/spreadsheets/d/1aDyvwiUHiDZre47Z0AOml0D7gS17mgfFFbqSJ6Svi64/export?format=csv&gid=716386560"

gdown.download(authors_csv, '/media/work/icarovasconcelos/mono/data/authors/author.csv', quiet=False)

data_authors = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/author.csv')

duplicated_names = data_authors[data_authors.duplicated(['author_name'], keep=False)]

with open('/media/work/icarovasconcelos/mono/data/authors/authors_information.json', 'r') as f:
    authors_information = json.load(f)
    
with open('/media/work/icarovasconcelos/mono/data/authors/instituition_information.json', 'r') as f:
    instituition_information = json.load(f)
    
data_authors = data_authors.dropna(subset=['author_id'])
data_authors = data_authors.drop_duplicates(subset=['author_name'])
duplicated_ids = data_authors[data_authors.duplicated(['author_id'], keep=False)]
data_authors = data_authors.drop_duplicates(subset=['author_id'])
data_authors = data_authors.drop(columns=['responsavel_por_preencher', 'autor_confiavel', 'orientador_confiavel'])

print('Repeated IDs and their index:')
print(duplicated_ids[['author_name', 'author_id']])

print('Duplicates names: ', duplicated_names['author_name'].nunique())
print('Duplicates ids: ', duplicated_ids['author_id'].nunique())
print('Authors ids:', data_authors['author_id'].nunique())

#append new columns
data_authors['works_count'] = 0
data_authors['cite_count'] = 0
data_authors['2yr_mean_citedness'] = 0.0
data_authors['h_index'] = 0
data_authors['i10_index'] = 0
data_authors['longitude'] = 0.0
data_authors['latitude'] = 0.0
data_authors['state'] = 'None'
data_authors['region'] = 'None'


# URL to the Natural Earth shapefile zip
url = 'https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip'

# Local path where the zip file will be saved
zip_path = '/tmp/BR_UF_2022.zip'
shapefile_dir = '/tmp/IBGE/'

# Download the file
response = requests.get(url)
with open(zip_path, 'wb') as file:
    file.write(response.content)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(shapefile_dir)

# Path to the extracted shapefile
shapefile_path = os.path.join(shapefile_dir, 'BR_UF_2022.shp')

# Load the shapefile with GeoPandas
brasil = gpd.read_file(shapefile_path)


for index, author_id, inst_id in zip(data_authors.index, data_authors['author_id'], data_authors['institution_id']):
    author_id_url = 'https://openalex.org/' + author_id  # Construct the ID URL
    inst_id_url = 'https://openalex.org/' + inst_id  # Construct the ID URL
    a_result = next((item for item in authors_information if item['id'] == author_id_url), None)
    i_result = next((item for item in instituition_information if item['id'] == inst_id_url), None)
    
    
    # If the result is found, update the corresponding columns
    if a_result:
        data_authors.at[index, 'works_count'] = a_result.get('works_count', 0)
        data_authors.at[index, 'cite_count'] = a_result.get('cited_by_count', 0)
        
        # Extract nested summary stats
        summary_stats = a_result.get('summary_stats', {})
        data_authors.at[index, '2yr_mean_citedness'] = summary_stats.get('2yr_mean_citedness', 0.0)
        data_authors.at[index, 'h_index'] = summary_stats.get('h_index', 0)
        data_authors.at[index, 'i10_index'] = summary_stats.get('i10_index', 0)
        
        
    if i_result:
        geo = i_result.get('geo', {})
        data_authors.at[index, 'longitude'] = geo.get('longitude', 0)
        data_authors.at[index, 'latitude'] = geo.get('latitude', 0)
        
    if inst_id == 'I4210130985':
        data_authors.at[index, 'longitude'] = -44.26167
        data_authors.at[index, 'latitude'] = -21.13556
        print(f'Author {data_authors.at[index, "author_id"]} is in {data_authors.at[index, "state"]} / coordinates: {lon}, {lat}')
        
    lon = data_authors.at[index, 'longitude']
    lat = data_authors.at[index, 'latitude']
    coord = Point(lon, lat)
    # Usar índice espacial para encontrar o estado que contém o ponto
    possible_states = brasil[brasil.contains(coord)]
    
    if not possible_states.empty:
        data_authors.at[index, 'state'] = possible_states.iloc[0]['SIGLA_UF']
        data_authors.at[index, 'region'] = possible_states.iloc[0]['NM_REGIAO']
    else:
        data_authors.at[index, 'state'] = 'None'
        data_authors.at[index, 'region'] = 'None'
            

print(data_authors.head(5))
data_authors.to_csv('/media/work/icarovasconcelos/mono/data/authors/authors-information.csv', index=False)
print('Saved authors data')
