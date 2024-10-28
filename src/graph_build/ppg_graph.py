import pandas as pd
import json
import networkx as nx
from itertools import combinations
import math


# Load data

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-information.csv')

works_file_path = '/media/work/icarovasconcelos/mono/data/raw_works_since_2004.json'
with open(works_file_path, 'r') as f:
    works_data = json.load(f)

#------------------------------------------------------------------------------------------------------------------------------
works_and_authors = {}    

unlimited = math.inf        
limit = 20 # Limit to n authors per work or unlimited

collaboration_graph = nx.Graph()
    
for work in works_data:
    if len(work['authorships']) > limit:
        continue
    authors_list = []
    for authorship1, authorship2 in combinations(work['authorships'], 2):
        if authorship1 and authorship2:
            author1 = authorship1['author']
            author2 = authorship2['author']
            if author1['id'].split('/')[-1] == 'A9999999999' or author2['id'].split('/')[-1] == 'A9999999999':
                continue
            author_id1 = author1['id'].split('/')[-1] if author1 else None
            author_id2 = author2['id'].split('/')[-1] if author2 else None
            if collaboration_graph.has_edge(author_id1, author_id2):
                collaboration_graph[author_id1][author_id2]['weight'] += 1
            else:
                collaboration_graph.add_edge(author_id1, author_id2, weight=1)
            
        
    
#------------------------------------------------------------------------------------------------------------------------------

dict_authors = authors_data.to_dict(orient='records')

print("Number of nodes:", collaboration_graph.number_of_nodes())

# Print the number of edges
print("Number of edges:", collaboration_graph.number_of_edges())

total_weight = 0

total_weight = sum(data['weight'] for _, _, data in collaboration_graph.edges(data=True))

print("Number of colabs:", total_weight)

highest_weight = 0

highest_weight = max(collaboration_graph.edges(data=True), key=lambda edge: edge[2]['weight'])

print("Most colabs:",highest_weight)

isolated_nodes = list(nx.isolates(collaboration_graph))

print("Isolated nodes:",len(isolated_nodes))

authors_set = set()
for author in dict_authors:
    if author['author_id'] in collaboration_graph.nodes:
        authors_set.add(author['author_id'])
        
print('Authors in data:', len(dict_authors))

print("Authors in graph:", len(authors_set))

authors_not_in_graph = set()
for author in dict_authors:
    if author['author_id'] not in collaboration_graph.nodes:
        authors_not_in_graph.add(author['author_id'])

print("Authors not in graph:", len(authors_not_in_graph))
print("Authors not in graph:", authors_not_in_graph)

file_name = f'/media/work/icarovasconcelos/mono/data/collaboration_networkx_reduced.graphml'
nx.write_graphml(collaboration_graph, file_name)
