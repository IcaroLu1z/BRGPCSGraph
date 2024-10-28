import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
from functools import partial

#--------------------------------------------------------LOAD DATA----------------------------------------------------------
authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')
g_path = '/media/work/icarovasconcelos/mono/data/collaboration_networkx_reduced.graphml'

#--------------------------------------------------------EDGES PER 100-------------------------------------------------------
# Calculates the ratio of edges in the backbone to the original graph
def edges_per100(original, backbone):
    return (backbone.number_of_edges() / original.number_of_edges())

#--------------------------------------------------------EXTRACT BACKBONE---------------------------------------------------
# Extracts the backbone of a graph
def extract_backbone(graph, alpha, ignore_nodes, g_strength=None, g_degree=None):
    backbone = nx.Graph()
    edges = []
    sorted_edges = sorted(graph.edges.data(), key=lambda x: x[2]['weight'], reverse=True)
    
    for edge in sorted_edges:
        source, target, weight = edge
        weight = weight['weight']
        s_source = g_strength[source]
        s_target = g_strength[target]
        
        k_source = g_degree[source]
        k_target = g_degree[target]
        
        weight_ratio_source = weight / s_source
        weight_ratio_target = weight / s_target
        
        pij = (1 - weight_ratio_source) ** (k_source - 1)
        pji = (1 - weight_ratio_target) ** (k_target - 1)
        
        p = min(pij, pji)
        
        
        if(source in ignore_nodes and target in ignore_nodes):
            edges.append(edge)
        
        elif(source in ignore_nodes):
            if ignore_nodes[source] <= 20:
                ignore_nodes[source] += 1
                edges.append(edge)
                
        elif(target in ignore_nodes):
            if ignore_nodes[target] <= 20:
                ignore_nodes[target] += 1
                edges.append(edge)
            
        elif(p < alpha):
            edges.append(edge)
            
    backbone.add_edges_from(edges)
    
    return backbone

#--------------------------------------------------------PROCESS ALPHA------------------------------------------------------
# Extracts the backbone of a graph for a given alpha
def process_alpha(alpha, g, ppgcc_subgraph, g_strength, g_degree):
    graph = g.copy()
    ignore_nodes = ppgcc_subgraph.copy()
    g_s = dict(g_strength)
    g_d = dict(g_degree)
    
    backbone = extract_backbone(graph, alpha, ignore_nodes, g_s, g_d)
    cc = max(nx.connected_components(backbone), key=len)
    
    main_nodes_in_cc = set()
    for node in ignore_nodes.keys():
        if node in cc:
            main_nodes_in_cc.add(node)
            
    edges_ratio = edges_per100(graph, backbone)
    
    print(f'Finalized backbone with alpha={alpha:.3f} / L={edges_ratio} / Authors in CC={len(main_nodes_in_cc)}')
    del backbone
    return alpha, edges_ratio, main_nodes_in_cc

# ---------------------------------------------------------------MAIN CODE-------------------------------------------------------
chosen_alpha = 0.0  
best_edge_ratio = float('inf')
best_len_cc = -1
processes = 6

# Load graph and calculate strength and degree
g = nx.read_graphml(g_path)
g_strength = g.degree(weight='weight')
g_degree = g.degree()

# Create a dictionary to store minimum degree of each ppg author in the backbone
ppgcc_subgraph = {}
for author_id in authors_data['author_id']:
    ppgcc_subgraph[author_id] = 0 # Dict author_id -> degree on backbone

# Create a partial function to process alpha
process_alpha_partial = partial(process_alpha, g=g, ppgcc_subgraph=ppgcc_subgraph, g_strength=g_strength, g_degree=g_degree)

# Calculate the backbone for each alpha
alphas = np.linspace(0.01, 1.0, 200)
with mp.Pool(processes=processes) as pool:
    results = pool.map(process_alpha_partial, alphas)
    
# Create a table with the results
alpha_table = pd.DataFrame()
    
for alpha, edge_ratio, main_nodes_in_cc in results:
    #if (edge_ratio < best_edge_ratio) or (edge_ratio == best_edge_ratio and len(main_nodes_in_cc) > best_len_cc):
    # Update the best alpha if the current alpha has a better edge ratio and more authors in the connected component
    if (0.05 < edge_ratio < 0.15 ) and (len(main_nodes_in_cc) > best_len_cc):
        best_edge_ratio = edge_ratio
        best_len_cc = len(main_nodes_in_cc)
        chosen_alpha = alpha
        
    row = {'alpha': alpha, 'edge_ratio': edge_ratio, 'authors_cc': len(main_nodes_in_cc)}
    alpha_table = pd.concat([alpha_table, pd.DataFrame(row, index=[0])], ignore_index=True)
    
alpha_table.to_csv('/media/work/icarovasconcelos/mono/data/alpha_table.csv', index=False)

# ---------------------------------------------------------------FINAL BACKBONE-------------------------------------------------------
# Extract the final backbone
final_graph = g.copy()
final_ignore_nodes = ppgcc_subgraph.copy()
final_g_s = dict(g_strength)
final_g_d = dict(g_degree)

best_backbone = extract_backbone(final_graph, chosen_alpha, final_ignore_nodes, final_g_s, final_g_d)
nx.write_graphml(best_backbone, f'/media/work/icarovasconcelos/mono/data/backbones/collabNetx_backbone_a{chosen_alpha:.2f}.graphml')

# ---------------------------------------------------------------PRINT RESULTS-------------------------------------------------------
nn_before = g.number_of_nodes()
ne_before = g.number_of_edges()
weight_before = sum([g[u][v]['weight'] for u, v in g.edges()])
percent_of_edges_before = 1

nn_after = best_backbone.number_of_nodes()
ne_after = best_backbone.number_of_edges()
weight_after = sum([best_backbone[u][v]['weight'] for u, v in best_backbone.edges()])
percent_of_edges_after = edges_per100(final_graph, best_backbone)

print(f'Number of nodes: {nn_before} vs {nn_after}')
print(f'Number of edges: {ne_before} vs {ne_after}')
print(f'Weight: {weight_before} vs {weight_after}')
print(f'L: {percent_of_edges_before} vs {percent_of_edges_after}')
print(f'Chosen alpha: {chosen_alpha:.3f}')