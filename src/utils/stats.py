import networkx as nx
import pandas as pd
from itertools import product

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-information.csv')

ppg_subgraph = []
for author in authors_data['author_id']:
    ppg_subgraph.append(author)

graph_back = nx.read_graphml('/media/work/icarovasconcelos/mono/data/backbones/collabNetx_backbone_a0.24.graphml')
graph = nx.read_graphml('/media/work/icarovasconcelos/mono/data/collaboration_networkx_reduced.graphml')
largest_cc_original = max(nx.connected_components(graph), key=len)
largest_cc_back = max(nx.connected_components(graph_back), key=len)

authors_in_component_back = []
authors_in_component = []
authors_in_graph = []
authors_in_back = []
isolated_authors = []

isolated_nodes = list(nx.isolates(graph))

if isolated_nodes:
    for node in isolated_nodes:
        if node in ppg_subgraph:
            isolated_authors.append(node)
    print(f"Isolated nodes: {isolated_nodes}")
else:
    print("No isolated nodes found.")


for a in ppg_subgraph:
    if a in largest_cc_back:
        authors_in_component_back.append(a)
    if a in largest_cc_original:
        authors_in_component.append(a)
    if a in graph.nodes:
        authors_in_graph.append(a)
    if a in graph_back.nodes:
        authors_in_back.append(a)
        
reduction_percentage_authors = 100 - (len(authors_in_component_back) / len(authors_in_component) * 100)
reduction_percentage_nodes = 100 - (graph_back.number_of_nodes() / graph.number_of_nodes() * 100)
reduction_percentage_edges = 100 - (graph_back.number_of_edges() / graph.number_of_edges() * 100)


print(f'{graph.number_of_nodes()=}')
print(f'{graph_back.number_of_nodes()=}')
print(f'{reduction_percentage_nodes=}')

print(f'{graph.number_of_edges()=}')
print(f'{graph_back.number_of_edges()=}')
print(f'{reduction_percentage_edges=}')

print(f'{len(authors_in_component)=}')
print(f'{len(authors_in_component_back)=}')
print(f'{reduction_percentage_authors=}')


print(f'{len(largest_cc_original)=}')
print(f'{len(largest_cc_back)=}')
print(f'{len(ppg_subgraph)=}')
print(f'{len(authors_in_graph)=}')
print(f'{len(authors_in_back)=}')
print(f'{len(isolated_authors)=}')
print(f'{nx.density(graph)=}')
print(f'{nx.density(graph_back)=}')
