import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

random_state = 42


model_paths = ['/media/work/icarovasconcelos/mono/results/clusters_models/ward_3cluster_a0.24_10win_50ep_2dim_100wl_25nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_4cluster_a0.24_10win_50ep_2dim_150wl_38nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_5cluster_a0.24_10win_50ep_2dim_200wl_50nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_6cluster_a0.24_10win_50ep_2dim_175wl_44nw_1p_1q.npy']
              
embeddings_path = ['/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_100wl_25nw_1p_1q.npy',
                   '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_150wl_38nw_1p_1q.npy',
                   '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_200wl_50nw_1p_1q.npy',
                   '/media/work/icarovasconcelos/mono/data/test_embeddings/n2v_authors_embeddings_a0.24_10win_50ep_2dim_175wl_44nw_1p_1q.npy']

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-information.csv')

model_3_k = AgglomerativeClustering(n_clusters=3, linkage='ward')
model_4_k = AgglomerativeClustering(n_clusters=4, linkage='ward')
model_5_k = AgglomerativeClustering(n_clusters=5, linkage='ward')
model_6_k = AgglomerativeClustering(n_clusters=6, linkage='ward')

models = [model_3_k, model_4_k, model_5_k, model_6_k]

df_model_3_k = authors_data.copy()
df_model_4_k = authors_data.copy()
df_model_5_k = authors_data.copy()
df_model_6_k = authors_data.copy()

dfs = [df_model_3_k, df_model_4_k, df_model_5_k, df_model_6_k]

print(authors_data.info())
'''
 0   data_year                1489 non-null   int64  
 1   gp_code                  1489 non-null   object 
 2   gp_name                  1489 non-null   object 
 3   gp_score                 1489 non-null   int64  
 4   institution_id           1489 non-null   object 
 5   institution_acr          1489 non-null   object 
 6   author_name              1489 non-null   object 
 7   phd_year                 1489 non-null   int64  
 8   regime_trabalho          1489 non-null   object 
 9   carga_horaria            1489 non-null   int64  
 10  lattes_cv_link           1489 non-null   object 
 11  author_id                1489 non-null   object 
 12  productivity_grant       1489 non-null   bool   
 13  productivity_grant_type  1489 non-null   object 
 14  phd_institution_id       1474 non-null   object 
 15  phd_institution_name     1489 non-null   object 
 16  phd_gp_code              951 non-null    object 
 17  phd_supervisor_id        1460 non-null   object 
 18  phd_supervisor_name      1488 non-null   object 
 19  works_count              1489 non-null   int64  
 20  cite_count               1489 non-null   int64  
 21  2yr_mean_citedness       1489 non-null   float64
 22  h_index                  1489 non-null   int64  
 23  i10_index                1489 non-null   int64  
 24  longitude                1489 non-null   float64
 25  latitude                 1489 non-null   float64

'''

attributes_hist = ['2yr_mean_citedness', 'phd_year', 'productivity_grant_type', 'latitude']
attributes_box = ['2yr_mean_citedness', 'gp_score', 'phd_year', 'productivity_grant_type', 'latitude', 'longitude', 
                    'works_count', 'institution_acr', 'state', 'region']
n_model = 1
for model_path, model, emb, df in zip(model_paths, models, embeddings_path, dfs):
    embeddings_dict = np.load(emb, allow_pickle=True).item()
    embeddings = np.array(list(embeddings_dict.values()))
    try:
        model = pickle.load(open(model_path, "rb"))
    except:
        model.fit(embeddings)
        pickle.dump(model, open(model_path, "wb"))

    labels = model.labels_
    n_labels = len(set(labels))
    clusters_size = np.bincount(labels)
    
    s_score = silhouette_score(embeddings, labels)
    db_score = davies_bouldin_score(embeddings, labels)
    ch_score = calinski_harabasz_score(embeddings, labels)
    
    
    print(f'Model {n_model} - {n_labels} clusters - Silhouette Score: {s_score} - Davies Bouldin Score: {db_score} - Calinski Harabasz Score: {ch_score}')
    
    df['cluster'] = labels

    # Criar uma figura com 4 subplots (2x2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plotar histogramas empilhados para cada atributo
    for ax, attr in zip(axs.flatten(), attributes_hist):
        # Criação de listas para armazenar os dados de cada cluster
        clusters_data = [df[df['cluster'] == i][attr].values for i in range(model.n_clusters)]

        # Plotar histogramas empilhados
        ax.hist(clusters_data, bins=20, stacked=True, label=[f'Cluster {i}' for i in range(model.n_clusters)])
        
        # Personalizar o gráfico
        ax.set_title(f'{attr} - {model.n_clusters} Clusters')
        ax.set_xlabel(attr)
        ax.set_ylabel('Número de Samples')
        ax.legend(title='Clusters')

    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()
    plt.show()
    plt.savefig(f'/media/work/icarovasconcelos/mono/results/analysis/histograms_{model.n_clusters}_clusters.png')
    
    # Plotar boxplots para cada atributo
    fig, axs = plt.subplots(math.ceil(len(attributes_box) / 2), 2, figsize=(32, 24))
        
    fig.suptitle(f'Model {n_model} - Silhouette Score: {s_score}', fontsize=16)
    
    for i, attr in enumerate(attributes_box):
        row = i // 2
        col = i % 2
        ax = axs[row, col]

        # Check if the attribute is numerical or object
        if pd.api.types.is_numeric_dtype(df[attr]):
            # Plot boxplot for numerical attributes
            ax.boxplot([df[df['cluster'] == i][attr].values for i in range(model.n_clusters)], showfliers=False)
            ax.set_title(f'{attr} - {model.n_clusters} Clusters')
            ax.set_xlabel('Clusters')
            ax.set_ylabel(attr)            
        else:
            # Plot count plot (bar chart) for object/categorical attributes
            sns.countplot(x=attr, hue='cluster', data=df, ax=ax)
            ax.set_title(f'{attr} - {model.n_clusters} Clusters')
            ax.set_xlabel(attr)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylabel('Número de Samples')

        ax.grid(True)

        plt.tight_layout()
        plt.show()
        plt.savefig(f'/media/work/icarovasconcelos/mono/results/analysis/boxplots_model{n_model}_clusters.png')

    
    # Calculate mean, median, and standard deviation for each cluster
    mean = df.groupby('cluster').mean()
    median = df.groupby('cluster').median()
    std = df.groupby('cluster').std()

    cluster_sizes = df['cluster'].value_counts().sort_index()
    cluster_size_df = pd.DataFrame(cluster_sizes)

    # Rename the column to 'Cluster Size' for clarity
    cluster_size_df.columns = ['Cluster Size']

    # Combine mean, median, and std into a single DataFrame
    summary_stats = pd.concat([mean, median, std], keys=['Mean', 'Median', 'Std'], axis=1)

    # Concatenate the cluster size with the summary statistics based on the cluster index
    summary_stats = pd.concat([cluster_size_df, summary_stats], axis=1)
    # Reset the display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Print the consolidated summary statistics
    print('\n')
    print(summary_stats)
    print('\n')
    print('-' * 40)
    
    n_model += 1
    
coord, ax_c = plt.subplots(4, 2, figsize=(26, 16))
coord.suptitle('Latitude and Longitude Boxplots', fontsize=32)
workCount_2yrmc, ax_2y = plt.subplots(4, 2, figsize=(26, 16))
workCount_2yrmc.suptitle('Works Count and 2-Year Mean Citedness Boxplots', fontsize=24)
pYear_gpScore, ax_pgp = plt.subplots(4, 2, figsize=(26, 16))
pYear_gpScore.suptitle('PhD Year and GP Score Boxplots', fontsize=24)
state_region, ax_sr = plt.subplots(4, 2, figsize=(36, 24))
state_region.suptitle('State and Region Countplot', fontsize=36)
inst_acr, ax_acr = plt.subplots(4, 1, figsize=(32, 26))
for i, df in enumerate(dfs):
    n_clusters = df['cluster'].nunique()
    
    # Plotting latitude boxplots for each cluster, without outliers
    ax_c[i, 0].boxplot([df[df['cluster'] == cluster_id]['latitude'].values 
                           for cluster_id in range(n_clusters)])
    ax_c[i, 0].set_title(f'Latitude - Model {i+1} - {n_clusters} Clusters', fontsize=28)
    ax_c[i, 0].set_xlabel('Clusters', fontsize=24)
    ax_c[i, 0].set_ylabel('Latitude', fontsize=24)
    ax_c[i, 0].tick_params(axis='both', which='major', labelsize=22)
    ax_c[i, 0].grid()
    # Plotting longitude boxplots for each cluster, without outliers
    ax_c[i, 1].boxplot([df[df['cluster'] == cluster_id]['longitude'].values 
                           for cluster_id in range(n_clusters)])
    ax_c[i, 1].set_title(f'Longitude - Model {i+1} - {n_clusters} Clusters', fontsize=28)
    ax_c[i, 1].set_xlabel('Clusters', fontsize=24)
    ax_c[i, 1].set_ylabel('Longitude', fontsize=24)
    ax_c[i, 1].tick_params(axis='both', which='major', labelsize=22)
    ax_c[i, 1].grid()
    # Plotting works_count boxplots for each cluster, without outliers
    ax_2y[i, 0].boxplot([df[df['cluster'] == cluster_id]['works_count'].values 
                            for cluster_id in range(n_clusters)], showfliers=False)
    ax_2y[i, 0].set_title(f'Works Count - Model {i+1} - {n_clusters} Clusters', fontsize=20)
    ax_2y[i, 0].set_xlabel('Clusters', fontsize=18)
    ax_2y[i, 0].set_ylabel('Works Count', fontsize=18)
    ax_2y[i, 0].tick_params(axis='both', which='major', labelsize=16)
    ax_2y[i, 0].grid()
    # Plotting 2yr_mean_citedness boxplots for each cluster, without outliers
    ax_2y[i, 1].boxplot([df[df['cluster'] == cluster_id]['2yr_mean_citedness'].values 
                            for cluster_id in range(n_clusters)], showfliers=False)
    ax_2y[i, 1].set_title(f'2-Year Mean Citedness - Model {i+1} - {n_clusters} Clusters', fontsize=20)
    ax_2y[i, 1].set_xlabel('Clusters', fontsize=18)
    ax_2y[i, 1].set_ylabel('2-Year Mean Citedness', fontsize=18)
    ax_2y[i, 1].tick_params(axis='both', which='major', labelsize=16)
    ax_2y[i, 1].grid()
    # Plotting gp_score boxplots for each cluster, without outliers
    ax_pgp[i, 0].boxplot([df[df['cluster'] == cluster_id]['gp_score'].values 
                             for cluster_id in range(n_clusters)])
    ax_pgp[i, 0].set_title(f'GP Score - Model {i+1} - {n_clusters} Clusters', fontsize=20)
    ax_pgp[i, 0].set_xlabel('Clusters', fontsize=18)
    ax_pgp[i, 0].set_ylabel('GP Score', fontsize=18)
    ax_pgp[i, 0].tick_params(axis='both', which='major', labelsize=16)
    ax_pgp[i, 0].grid()
    # Plotting phd_year boxplots for each cluster, without outliers
    ax_pgp[i, 1].boxplot([df[df['cluster'] == cluster_id]['phd_year'].values 
                             for cluster_id in range(n_clusters)])
    ax_pgp[i, 1].set_title(f'PhD Year - Model {i+1} - {n_clusters} Clusters', fontsize=20)
    ax_pgp[i, 1].set_xlabel('Clusters', fontsize=18)
    ax_pgp[i, 1].set_ylabel('PhD Year', fontsize=18)
    ax_pgp[i, 1].tick_params(axis='both', which='major', labelsize=16)
    ax_pgp[i, 1].grid()
    
    # Plotting state stacked bar for each cluster (assuming state is numeric)
    state_counts = df.groupby(['cluster', 'state']).size().unstack(fill_value=0)
    top_10_states = state_counts.sum(axis=0).nlargest(7).index
    state_counts_top_10 = state_counts[top_10_states]
    state_percentages_top_10 = state_counts_top_10.div(state_counts_top_10.sum(axis=1), axis=0)
    state_percentages_top_10.plot(
        kind='bar', 
        stacked=True, 
        ax=ax_sr[i, 0], 
        fontsize=22,
        colormap='viridis'  # Você pode ajustar o esquema de cores
    )
    ax_sr[i, 0].set_title(f'State - Model {i+1} - {n_clusters} Clusters', fontsize=32)
    ax_sr[i, 0].set_xlabel('Clusters', fontsize=28)
    ax_sr[i, 0].set_ylabel('Proportion', fontsize=28)  # Label para indicar que são proporções
    ax_sr[i, 0].tick_params(axis='both', which='major', labelsize=22)
    ax_sr[i, 0].legend(title='State', fontsize=18, title_fontsize=20)
    ax_sr[i, 0].grid(axis='y')  # Coloca grid nas linhas horizontais para facilitar a leitura

    # Plotando gráfico de pizza para 'region'
    region_counts = df.groupby(['cluster', 'region']).size().unstack(fill_value=0)
    region_percentages = region_counts.div(region_counts.sum(axis=1), axis=0)
    region_percentages.plot(
        kind='bar', 
        stacked=True, 
        ax=ax_sr[i, 1], 
        fontsize=22,
        colormap='plasma'  # Você pode escolher outro esquema de cores se preferir
    )
    ax_sr[i, 1].set_title(f'Region - Model {i+1} - {n_clusters} Clusters', fontsize=26)
    ax_sr[i, 1].set_xlabel('Clusters', fontsize=24)
    ax_sr[i, 1].set_ylabel('Proportion', fontsize=24)  # Label para indicar que são proporções
    ax_sr[i, 1].tick_params(axis='both', which='major', labelsize=22)
    ax_sr[i, 1].legend(title='Region', fontsize=18, title_fontsize=20)
    ax_sr[i, 1].grid(axis='y')  # Coloca grid apenas nas linhas horizontais

    # Plotando gráfico de pizza para 'institution_acr'
    institution_counts = df.groupby(['cluster', 'institution_acr']).size().unstack(fill_value=0)
    institution_counts.sum(axis=0).plot.pie(
        ax=ax_acr[i],
        autopct='%1.1f%%', 
        startangle=90, 
        counterclock=False, 
        fontsize=18
    )
    ax_acr[i].set_title(f'Institution Acronym - Model {i+1} - {n_clusters} Clusters', fontsize=24)
    ax_acr[i].set_ylabel('')  # Remover o label padrão de ylabel
    ax_acr[i].grid()

coord.tight_layout()
workCount_2yrmc.tight_layout()
pYear_gpScore.tight_layout()
state_region.tight_layout()
inst_acr.tight_layout()

coord.savefig('/media/work/icarovasconcelos/mono/results/figures/coord_boxplot.png')
plt.close(coord)
workCount_2yrmc.savefig(f'/media/work/icarovasconcelos/mono/results/figures/workCount_2yrmc_boxplot.png')
plt.close(workCount_2yrmc)
pYear_gpScore.savefig(f'/media/work/icarovasconcelos/mono/results/figures/pYear_gpScore_boxplot.png')
plt.close(pYear_gpScore)
state_region.savefig(f'/media/work/icarovasconcelos/mono/results/figures/state_region_boxplot.png')
plt.close(state_region)
inst_acr.savefig(f'/media/work/icarovasconcelos/mono/results/figures/inst_boxplot.png')
plt.close(inst_acr)
