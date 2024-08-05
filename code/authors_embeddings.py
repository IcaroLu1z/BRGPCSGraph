import pandas as pd
import numpy as np
import networkx as nx
import node2vec as n2v

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')

ppgcc_subgraph = []

for author_id in authors_data['author_id']:
    ppgcc_subgraph.append(author_id)

collab_net = nx.read_graphml('/media/work/icarovasconcelos/mono/collaboration_network.graphml')

# Create a Node2Vec object
node2vec = n2v.Node2Vec(collab_net, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Generate embeddings
model = node2vec.fit(window=10, min_count=1)

# Get node embeddings for author nodes
author_embeddings = {node: model.wv[node] for node in ppgcc_subgraph if node in model.wv}    

# Convert embeddings to array
embeddings_array = np.array([author_embeddings[node] for node in ppgcc_subgraph if node in author_embeddings])
np.save('/media/work/icarovasconcelos/mono/data/n2v_authors_embeddings.npy', embeddings_array)
