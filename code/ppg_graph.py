import pandas as pd
import json
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
import math


# Load data

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')

works_file_path = '/media/work/icarovasconcelos/mono/data/raw_works_since_2004.json'
with open(works_file_path, 'r') as f:
    works_data = json.load(f)

#------------------------------------------------------------------------------------------------------------------------------
works_and_authors = {}    

unlimited = math.inf        
limit = unlimited # Limit to n authors per work or unlimited


for work in works_data:
    i = 0
    authors_list = []
    for authorship in work['authorships']:
        if i >= limit:  # Limit to n authors per work
            break
        if authorship:
            author = authorship['author']
            author_id = author['id'].split('/')[-1] if author else None
            author_name = author['display_name'] if author else None
            authors_list.append({
                'author_id': author_id,
                'author_name': author_name,
            })
            i+=1
    works_and_authors[work['id'].split('/')[-1]] = authors_list
    
#------------------------------------------------------------------------------------------------------------------------------

dict_authors = authors_data.to_dict(orient='records')

collaboration_graph = nx.Graph()

for a in dict_authors:
    collaboration_graph.add_node(a['author_id'], label=a['author_name'], type='author', **a)

# Connect authors by work
for work_id, authors in works_and_authors.items():
    for author1, author2 in combinations(authors, 2):
        if 'author_id' in author1 and 'author_id' in author2:
            if collaboration_graph.has_edge(author1['author_id'], author2['author_id']):
                # If the edge already exists, increment its weight by 1
                collaboration_graph[author1['author_id']][author2['author_id']]['weight'] += 1
            else:
                # If the edge does not exist, add it with a weight of 1
                collaboration_graph.add_edge(author1['author_id'], author2['author_id'], weight=1)

collaboration_graph.remove_node('A9999999999')

print("Number of nodes:", collaboration_graph.number_of_nodes())

# Print the number of edges
print("Number of edges:", collaboration_graph.number_of_edges())

total_weight = 0

total_weight = sum(data['weight'] for _, _, data in collaboration_graph.edges(data=True))

print("Number of colabs:", total_weight)

highest_weight = 0

highest_weight = max(collaboration_graph.edges(data=True), key=lambda edge: edge[2]['weight'])

print("Most colabs:",highest_weight)


file_name = f'/media/work/icarovasconcelos/mono/data/{limit}a_per_w_collaboration_network.graphml'

# Unlimited
#nx.write_graphml(collaboration_graph, "/media/work/icarovasconcelos/mono/collaboration_network.graphml")

# Limited to 30 authors per work
nx.write_graphml(collaboration_graph, file_name)