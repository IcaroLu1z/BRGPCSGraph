from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-processed.csv')

embeddings_file = '/media/work/icarovasconcelos/mono/data/backbone_embeddings/n2v_authors_embeddings_a0.24_4win_50ep_2dim_75wl_22nw_1p_1q.npy'
titles = []
random_state = 21
visual_random_state = 21
k_clusters = [3,4,5,6]
tsne_paths = []
umap_paths = []
sp_paths = []

model_paths = []

model_name = 'Ward'
#model_name = 'kmeans'

best_silhouette = -1
for k in k_clusters:
    embeddings_dict = np.load(embeddings_file, allow_pickle=True).item()
    
    # Convert dictionary values to a list of embeddings
    embeddings = np.array(list(embeddings_dict.values()))
    
    gp_scores = [authors_data.loc[authors_data['author_id'] == key, 'gp_score'].values[0] 
                 for key in embeddings_dict.keys()]
    
    params = embeddings_file.split('_')[-6:]  # Adjust this based on your file name structure
    subtitle = f'{params[-5]}_{params[-4]}_{params[-3]}_{params[-2]}_{params[-1]}'
    titles.append(subtitle)
    
    if model_name == 'kmeans':
        model = KMeans(n_clusters=k, random_state=random_state)
    elif model_name == 'Ward':
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        
    model.fit(embeddings)
    model_labels = model.labels_
    
    silhouette = silhouette_score(embeddings, model_labels)
    print(f'{model_name} with {k} clusters: {silhouette=}')
    
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_model = model
    
    
    # Save the model
    model_path = '/media/work/icarovasconcelos/mono/results/clusters_models/' + model_name + '_'
    model_path = model_path + f'{params[-5]}_{params[-4]}_{params[-3]}_{params[-2]}_{params[-1]}'
    model_path = model_path.replace('.npy', '.pkl')
    pickle.dump(best_model, open(model_path, "wb"))
    
    fig_tsne, axs_tsne = plt.subplots(1, 2, figsize=(12, 6))
    
    tsne = TSNE(n_components=2, random_state=visual_random_state)
    tsne_result = tsne.fit_transform(embeddings)
    
    # Plot t-SNE with K-Means labels
    scatter1 = axs_tsne[0].scatter(tsne_result[:, 0], tsne_result[:, 1], c=model_labels, cmap='viridis')
    legend1 = axs_tsne[0].legend(*scatter1.legend_elements(), title=f"{model_name} Clusters")
    axs_tsne[0].add_artist(legend1)
    axs_tsne[0].set_title(f't-SNE with {k} {model_name} Clusters\n{subtitle}')
    
    # Plot t-SNE with gp_score labels
    scatter2 = axs_tsne[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=gp_scores, cmap='plasma')
    legend2 = axs_tsne[1].legend(*scatter2.legend_elements(), title="gp_scores")
    axs_tsne[1].add_artist(legend2)
    axs_tsne[1].set_title(f't-SNE with gp_scores\n{subtitle}')
    
    # Save the figure
    tsne_paths.append(f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_{k}clusters_tsne_{subtitle}.png')
    plt.savefig(tsne_paths[-1])
    plt.close()
    
    # Initialize the UMAP plots
    fig_umap, axs_umap = plt.subplots(1, 2, figsize=(12, 6))
    
     # Perform UMAP
    umap_model = umap.UMAP(n_components=2, random_state=visual_random_state)
    umap_result = umap_model.fit_transform(embeddings)

    # Plot UMAP with K-Means labels
    scatter3 = axs_umap[0].scatter(umap_result[:, 0], umap_result[:, 1], c=model_labels, cmap='viridis')
    legend3 = axs_umap[0].legend(*scatter3.legend_elements(), title=f"{model_name} Clusters")
    axs_umap[0].add_artist(legend3)
    axs_umap[0].set_title(f'UMAP with {k} {model_name} Clusters\n{subtitle}')

    # Plot UMAP with gp_score labels
    scatter4 = axs_umap[1].scatter(umap_result[:, 0], umap_result[:, 1], c=gp_scores, cmap='plasma')
    legend4 = axs_umap[1].legend(*scatter4.legend_elements(), title="gp_scores")
    axs_umap[1].add_artist(legend4)
    axs_umap[1].set_title(f'UMAP with gp_scores\n{subtitle}')
    
    # Save the figure
    umap_paths.append(f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_{k}clusters_umap_{subtitle}.png')
    plt.savefig(umap_paths[-1])
    plt.close()
    
    
    if params[-5].replace('dim', '') == '2':
        # Plot a scatterplot
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sp_model = sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=model_labels, palette='viridis', ax=ax[0])
        sp_emb = sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=gp_scores, palette='plasma', ax=ax[1])
        legend5 = sp_model.legend(title=f"{model_name} Clusters")
        legend6 = sp_emb.legend(title="gp_scores")
        sp_model.add_artist(legend5)
        sp_emb.add_artist(legend6)
        
        sp_model.set_title(f'Scatter Plot with {k} {model_name} Clusters\n{subtitle}')
        sp_emb.set_title(f'Scatter Plot with gp_scores\n{subtitle}')
        sp_paths.append(f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_{k}clusters_scatter_{subtitle}.png')
        plt.savefig(sp_paths[-1])
        plt.close()
    
def plot_images(images_paths, save_path, title):
    images = [plt.imread(image_path) for image_path in images_paths]
    
    fig, axs = plt.subplots(2, 2, figsize=(24, 12))  # Adjust figsize to control the plot size
    
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i])
        ax.axis('off')  # Turn off axes
        ax.set_aspect('auto')  # Ensure no scaling of images
    
    fig.suptitle(title, fontsize=16)  # Set the overall title
    plt.tight_layout()  # Adjust layout to fit title
    plt.savefig(save_path)  # Save the plot as a PNG file
    plt.close()  # Close the plot to free memory

saved_tsne_board = f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_tsne_board.png'
saved_umap_board = f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_umap_board.png'
saved_sp_board = f'/media/work/icarovasconcelos/mono/results/figures/{model_name}_scatter_board.png'

# Plot and save the t-SNE images
plot_images(tsne_paths, saved_tsne_board, f'{model_name} w/ t-SNE')

# Plot and save the UMAP images
plot_images(umap_paths, saved_umap_board, f'{model_name} w/ UMAP')

plot_images(sp_paths, saved_sp_board, f'{model_name} w/ Scatter Plot')
