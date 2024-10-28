from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-processed.csv')

model_paths = ['/media/work/icarovasconcelos/mono/results/clusters_models/ward_3cluster_a0.24_10win_50ep_2dim_100wl_25nw_1p_1q.npy',
                '/media/work/icarovasconcelos/mono/results/clusters_models/ward_4cluster_a0.24_10win_50ep_2dim_150wl_38nw_1p_1q.npy',
                '/media/work/icarovasconcelos/mono/results/clusters_models/ward_5cluster_a0.24_10win_50ep_2dim_200wl_50nw_1p_1q.npy',
                '/media/work/icarovasconcelos/mono/results/clusters_models/ward_6cluster_a0.24_10win_50ep_2dim_175wl_44nw_1p_1q.npy']

embeddings_path = ['/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_100wl_25nw_1p_1q.npy',
                     '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_150wl_38nw_1p_1q.npy',
                     '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_200wl_50nw_1p_1q.npy',
                     '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_175wl_44nw_1p_1q.npy']


fig, axs = plt.subplots(2, 2, figsize=(16, 8))

for i, paths in enumerate(zip(model_paths, embeddings_path)):
    model = pickle.load(open(paths[0], "rb"))
    embeddings_dict = np.load(paths[1], allow_pickle=True).item()

    # Convert dictionary values to a list of embeddings
    embeddings = np.array(list(embeddings_dict.values()))

    gp_scores = [authors_data.loc[authors_data['author_id'] == key, 'gp_score'].values[0] 
                 for key in embeddings_dict.keys()]

    model_labels = model.labels_
    silhouette = silhouette_score(embeddings, model_labels)
    n_clusters = len(set(model.labels_))
    
    params = paths[1].split('_')[-6:]  # Adjust this based on your file name structure
    subtitle = f'Model with {n_clusters}, Silhouette: {silhouette:.2f}'

    print(f'Ward with {model.n_clusters} clusters: {silhouette}')

    sp_model = sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=model_labels, palette='viridis', ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(subtitle, fontsize=22)
    axs[i//2, i%2].legend(title='Clusters', loc='best', fontsize=16)

plt.suptitle('Ward Clusters', fontsize=24)
plt.tight_layout()
plt.show()
plt.savefig('/media/work/icarovasconcelos/mono/results/figures/ward_clusters.png')

    
    
