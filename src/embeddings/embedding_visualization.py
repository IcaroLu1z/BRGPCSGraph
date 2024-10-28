from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import numpy as np

dim = 128
w_l = 200
n_w = 60
window = 10   
epochs = 10
w = 4
p = 1
q = 1

visual_random_state = 42
kmeans_random_state = 42

#embeddings_path = f'/media/work/icarovasconcelos/mono/data/n2v_authors_embeddings_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q.npy'
embeddings_path = f'/media/work/icarovasconcelos/mono/data/n2v_authors_embeddings_{window}win_{epochs}ep_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q.npy'

# Load your embeddings
embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()

## Convert dictionary values to a list of embeddings
embeddings = np.array(list(embeddings_dict.values()))

# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=visual_random_state)

# Fit and transform the embeddings
tsne_result = tsne.fit_transform(embeddings)

'''# Plotting and saving
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title(f't-SNE ({dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q)')
plt.savefig(f'/media/work/icarovasconcelos/mono/figures/n2v_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q_tsne_plot.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to free memory
'''

# Define number of clusters
n_clusters = 5  # Change this based on your needs

# Initialize KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state)

# Fit KMeans
kmeans.fit(embeddings)

# Get cluster labels
labels = kmeans.labels_
print(type(labels))
# Plotting and saving
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
plt.title(f't-SNE with K-Means Clustering\n({dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q)')
plt.savefig(f'/media/work/icarovasconcelos/mono/figures/n2v_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q_tsne_kmeans_plot.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to free memory


# Initialize UMAP
umap_model = umap.UMAP(n_components=2, random_state=visual_random_state)

# Fit and transform the embeddings
umap_result = umap_model.fit_transform(embeddings)

'''# Plotting and saving
plt.scatter(umap_result[:, 0], umap_result[:, 1])
plt.title('UMAP')
plt.savefig('umap_plot.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to free memory'''

# Initialize KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state)

# Fit KMeans
kmeans.fit(embeddings)

# Get cluster labels
labels = kmeans.labels_
print(labels)
# Plotting and saving
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=labels, cmap='viridis')
plt.title(f'UMAP with K-Means Clustering\n({dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q)')
plt.savefig(f'/media/work/icarovasconcelos/mono/figures/n2v_{dim}dim_{w_l}wl_{n_w}nw_{p}p_{q}q_umap_kmeans_plot.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to free memory