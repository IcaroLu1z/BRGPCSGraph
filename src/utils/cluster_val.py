import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# List of embeddings files
embeddings_files = [
    '/media/work/icarovasconcelos/mono/data/embeddings/n2v_authors_embeddings_alpha0.10_3win_50ep_16dim_25wl_8nw_1p_1q.npy',
    '/media/work/icarovasconcelos/mono/data/embeddings/n2v_authors_embeddings_alpha0.10_3win_50ep_16dim_50wl_15nw_1p_1q.npy',
    '/media/work/icarovasconcelos/mono/data/embeddings/n2v_authors_embeddings_alpha0.10_4win_50ep_16dim_75wl_22nw_1p_1q.npy',
    '/media/work/icarovasconcelos/mono/data/embeddings/n2v_authors_embeddings_alpha0.10_5win_50ep_16dim_100wl_30nw_1p_1q.npy',
   
]

kmeans_files = [
    '/media/work/icarovasconcelos/mono/results/clusters_models/kmeans_alpha0.10_16dim_25wl_8nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/kmeans_alpha0.10_16dim_50wl_15nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/kmeans_alpha0.10_16dim_75wl_22nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/kmeans_alpha0.10_16dim_100wl_30nw_1p_1q.pkl',
]



ward_files = [
    '/media/work/icarovasconcelos/mono/results/clusters_models/Ward_alpha0.10_16dim_25wl_8nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/Ward_alpha0.10_16dim_50wl_15nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/Ward_alpha0.10_16dim_75wl_22nw_1p_1q.pkl',
    '/media/work/icarovasconcelos/mono/results/clusters_models/Ward_alpha0.10_16dim_100wl_30nw_1p_1q.pkl',
]


authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors-processed.csv')

model_name, models_path = 'kmeans', kmeans_files
#model_name, models_path = 'Ward', ward_files

row = dict()

try:
    val_table = pd.read_csv('/media/work/icarovasconcelos/mono/results/analysis/val_table.csv')
except:
    val_table = pd.DataFrame(columns=['model', 'dimension', 'walk_length', 'num_walks', 'p', 'q', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'silhouette'])

# Iterate over the list of files
for embeddings_path, model_path in zip(embeddings_files, models_path):
    # Load your embeddings
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    
    # Convert dictionary values to a list of embeddings
    embeddings = np.array(list(embeddings_dict.values()))
    
    # Extract parameters from the file name for the subtitle
    params = embeddings_path.split('_')[-6:]  # Adjust this based on your file name structure
    subtitle = f'{params[-5]}_{params[-4]}_{params[-3]}_{params[-2]}_{params[-1]}'

    # Load the K-Means model
    model = pickle.load(open(model_path, "rb"))
    model_labels = model.labels_
    
    # If you have true labels, load them too (for supervised metrics)
    true_labels = [authors_data.loc[authors_data['author_id'] == key, 'gp_score'].values[0] 
                 for key in embeddings_dict.keys()]

    # Normalize embeddings for Silhouette Score (if needed)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Compute clustering metrics for K-Means
    nmi_score = normalized_mutual_info_score(true_labels, model_labels)
    homogeneity = homogeneity_score(true_labels, model_labels)
    completeness = completeness_score(true_labels, model_labels)
    v_measure = v_measure_score(true_labels, model_labels)
    silhouette = silhouette_score(embeddings_scaled, model_labels)
    
    row = {
        'model': model_name + params[-4],
        'dimension': params[-5],
        'walk_length': params[-4],
        'num_walks': params[-3],
        'p': params[-2],
        'q': params[-1],
        #'window_size': params[],
        #'epochs': params[],
        'nmi': nmi_score,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'silhouette': silhouette
    }
    
    val_table = pd.concat([val_table, pd.DataFrame(row, index=[0])], ignore_index=True)

print(val_table[['model', 'nmi', 'homogeneity', 'completeness', 'v_measure', 'silhouette']])

val_table.to_csv('/media/work/icarovasconcelos/mono/results/analysis/val_table.csv', index=False)
    
'''import matplotlib.pyplot as plt

# List of metrics to plot
metrics = ['nmi', 'homogeneity', 'completeness', 'v_measure', 'silhouette']

# Create a figure with subplots
fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=True)

for i, metric in enumerate(metrics):
    axs[i].bar(val_table['model'], val_table[metric], color='limegreen')
    
    axs[i].set_title(f'{metric.upper()}')
    axs[i].set_xlabel('Model')
    axs[i].set_xticks(range(len(val_table['model'])))
    axs[i].set_xticklabels(val_table['model'], rotation=45, ha='right')
    axs[i].grid(True)

# Set common y-label
axs[0].set_ylabel('Score')

plt.tight_layout()

# Save the combined plot
plt.savefig('/media/work/icarovasconcelos/mono/results/analysis/combined_metrics_bar.png')
plt.close()

print('Combined bar plot saved: combined_metrics_bar.png')

print('Combined plot saved: combined_metrics_comparison.png')'''