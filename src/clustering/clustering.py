from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import os
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

# Parameters
random_state = 42

def intra_cluster_distances(X, labels):
    unique_labels = np.unique(labels)
    intra_distances = {}
    
    for label in unique_labels:
        # Get points in the current cluster
        cluster_points = X[labels == label]
        
        # Compute pairwise distances within the cluster
        distances = pairwise_distances(cluster_points)
        
        # Take the mean distance for the current cluster
        intra_distances[label] = np.mean(distances)
    
    return intra_distances

def inter_cluster_distances(X, labels):
    unique_labels = np.unique(labels)
    centroids = []
    
    # Calculate the centroid of each cluster
    for label in unique_labels:
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    # Compute pairwise distances between centroids
    inter_distances_matrix = pairwise_distances(centroids)
    
    # Extract only the upper triangle of the distance matrix (excluding diagonal)
    upper_tri_indices = np.triu_indices(len(unique_labels), k=1)
    inter_distances_values = inter_distances_matrix[upper_tri_indices]
    
    # Calculate the mean of the inter-cluster distances
    mean_inter_cluster_distance = np.mean(inter_distances_values)
    
    return mean_inter_cluster_distance


model_type = 'Ward'
#model_type = 'kmeans'

best_silhouette = -1

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-processed.csv')

stats_table = pd.DataFrame(columns=['model', 'k', 'labels', 'clusters size', 'alpha', 'epochs', 'window', 'dimension', 'walk_length', 'num_walks', 'p', 
                                    'q', 'intra_distances', 'inter_distances',  'silhouette', 'calinski_harabasz', 'davies_bouldin', 'DBCV', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'embedding_path'])

#directory = '/media/work/icarovasconcelos/mono/data/backbone_embeddings'
directory = '/media/work/icarovasconcelos/mono/data/test_embeddings'
#directory = '/media/work/icarovasconcelos/mono/data/embeddings'
k = 2

for k in range(3, 11):
    with os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.endswith('.npy'):
                continue
            
            embeddings_path = entry.path
            
            if model_type == 'Ward':
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            elif model_type == 'kmeans':
                model = KMeans(n_clusters=k, random_state=random_state)
            
            # Load your embeddings
            embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
            
            # Convert dictionary values to a list of embeddings
            embeddings = np.array(list(embeddings_dict.values()))

            params = embeddings_path.split('_')[-8:]
            params[-1] = params[-1].replace('.npy', '')
            #model_name = model_type + f'_{params[-8]}_{params[-5]}_{params[-4]}_{params[-3]}_{params[-2]}_{params[-1]}'
            model_name = model_type + f'_{params[-5]}_{params[-4]}'
            
            # Perform clustering
            model.fit(embeddings)
            
            if model_type == 'GaussianMixture':
                labels = model.predict(embeddings)
            else:
                labels = model.labels_
            
            n_labels = len(set(labels))
            clusters_size = np.bincount(labels)
            
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            
            silhouette = silhouette_score(embeddings_scaled, labels, random_state=random_state)
            c_h_score = calinski_harabasz_score(embeddings_scaled, labels)
            d_b_score = davies_bouldin_score(embeddings_scaled, labels)
            intra_distances = intra_cluster_distances(embeddings_scaled, labels)
            inter_distances = inter_cluster_distances(embeddings_scaled, labels)
            print(f'Intra distances: {intra_distances}')
            print(f'Inter distances: {inter_distances}')
            
            true_labels = [authors_data.loc[authors_data['author_id'] == key, 'gp_score'].values[0] 
                            for key in embeddings_dict.keys()]
            
            nmi = normalized_mutual_info_score(true_labels, labels)
            homogeneity = homogeneity_score(true_labels, labels)
            completeness = completeness_score(true_labels, labels)
            v_measure = v_measure_score(true_labels, labels)
            
            row = {
                'model': model_name,
                'k': k,
                'labels': n_labels,
                'clusters size': str(clusters_size),
                'alpha': params[-8],
                'epochs': params[-6],
                'window': params[-7],
                'dimension': params[-5],
                'walk_length': params[-4],
                'num_walks': params[-3],
                'p': params[-2],
                'q': params[-1],
                'intra_distances': intra_distances,
                'inter_distances': inter_distances,
                'silhouette': silhouette,
                'calinski_harabasz': c_h_score,
                'davies_bouldin': d_b_score,
                'nmi': nmi,
                'homogeneity': homogeneity,
                'completeness': completeness,
                'v_measure': v_measure,
                'embedding_path': embeddings_path
            }
       
            stats_table = pd.concat([stats_table, pd.DataFrame(row, index=[0])], ignore_index=True)
            
            print(f'Silhouette score for {model_type} with k = {k}, {n_labels} clusters, clusters sizes {str(clusters_size)}, {params[-5]} dimenssions and {params[-4]} walk length: {silhouette}')
        

stat_path = directory + f'/.{model_type}_stat_table.csv'
  
stats_table.to_csv(stat_path, index=False)

#print(f'Best silhouette score: {best_silhouette} for {best_model_name} with {best_k} clusters')