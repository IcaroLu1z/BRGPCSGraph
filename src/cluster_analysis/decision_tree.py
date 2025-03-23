import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pickle

authors_data = pd.read_csv('/media/work/icarovasconcelos/mono/data/authors/authors-information.csv')
model_paths = ['/media/work/icarovasconcelos/mono/results/clusters_models/ward_3cluster_a0.24_10win_50ep_2dim_100wl_25nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_4cluster_a0.24_10win_50ep_2dim_150wl_38nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_5cluster_a0.24_10win_50ep_2dim_200wl_50nw_1p_1q.npy',
               '/media/work/icarovasconcelos/mono/results/clusters_models/ward_6cluster_a0.24_10win_50ep_2dim_175wl_44nw_1p_1q.npy']

label_encoder = LabelEncoder()
print(authors_data.head())
print(authors_data.info())

authors_data = authors_data.drop(columns=['author_id', 'gp_code', 'author_name', 'data_year', 'gp_name', 'regime_trabalho', 
                                          'carga_horaria', 'lattes_cv_link', 'institution_acr', 'phd_institution_name',
                                          'phd_supervisor_name', 'phd_supervisor_id', 'phd_gp_code', 'institution_id', 'phd_institution_id',])

authors_data_encoded = authors_data.copy()

for column in authors_data_encoded.select_dtypes(include=['object']).columns:
    authors_data_encoded[column] = label_encoder.fit_transform(authors_data_encoded[column])

print(authors_data_encoded.head())

authors_data_encoded.info()
lin = len(model_paths)
col = 6

fig, axs = plt.subplots(lin, col, figsize=(35, 18))
fig1, axs1 = plt.subplots(2, 2, figsize=(20, 10))
fig2, axs2 = plt.subplots(2, 2, figsize=(20, 10))
fig3, axs3 = plt.subplots(2, 2, figsize=(20, 10))
fig4, axs4 = plt.subplots(2, 2, figsize=(20, 10))
fig5, axs5 = plt.subplots(2, 2, figsize=(20, 10))
fig6, axs6 = plt.subplots(2, 2, figsize=(20, 10))


for i, model_path in enumerate(model_paths):
    model = pickle.load(open(model_path, "rb"))

    # Train a Decision Tree Classifier
    DT = DecisionTreeClassifier(random_state=21)
    RF = RandomForestClassifier(random_state=21)

    DT.fit(authors_data_encoded, model.labels_)
    RF.fit(authors_data_encoded, model.labels_)
    n_clusters = len(set(model.labels_))

    RF_perm_importance = permutation_importance(RF, authors_data_encoded, model.labels_, n_repeats=2*len(authors_data_encoded.columns), random_state=42)
    DT_perm_importance = permutation_importance(DT, authors_data_encoded, model.labels_, n_repeats=2*len(authors_data_encoded.columns), random_state=42)
    DT_importance = DT.feature_importances_
    RF_importance = RF.feature_importances_

    DTIndices = np.argsort(DT_importance)[::-1]  # Sort by importance
    
    axs[i, 0].barh(range(authors_data_encoded.shape[1]), DT_importance[DTIndices])
    axs[i, 0].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 0].set_yticklabels(authors_data_encoded.columns[DTIndices])
    axs[i, 0].set_title(f'Decision Tree Classifier - {n_clusters} clusters')
    axs[i, 0].set_xlabel('Feature Importance')
    axs[i, 0].set_ylabel('Feature')
    
    axs1[i//2, i%2].barh(range(authors_data_encoded.shape[1]), DT_importance[DTIndices])
    axs1[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs1[i//2, i%2].set_yticklabels(authors_data_encoded.columns[DTIndices])
    axs1[i//2, i%2].set_title(f'DT Feature Importance - {n_clusters} clusters', fontsize=22)
    axs1[i//2, i%2].set_xlabel('Feature Importance', fontsize=18)
    axs1[i//2, i%2].set_ylabel('Feature', fontsize=18)
    axs1[i//2, i%2].tick_params(axis='both', which='major', labelsize=16)
    
    RFIndices = np.argsort(RF_importance)[::-1]  # Sort by importance
    
    axs[i, 1].barh(range(authors_data_encoded.shape[1]), RF_importance[RFIndices])
    axs[i, 1].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 1].set_yticklabels(authors_data_encoded.columns[RFIndices])
    axs[i, 1].set_title(f'Random Forest Classifier - {n_clusters} clusters')
    axs[i, 1].set_xlabel('Feature Importance')
    axs[i, 1].set_ylabel('Feature')
    
    axs2[i//2, i%2].barh(range(authors_data_encoded.shape[1]), RF_importance[RFIndices])
    axs2[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs2[i//2, i%2].set_yticklabels(authors_data_encoded.columns[RFIndices])
    axs2[i//2, i%2].set_title(f'RF Feature Importance - {n_clusters} clusters', fontsize=22)
    axs2[i//2, i%2].set_xlabel('Feature Importance', fontsize=18)
    axs2[i//2, i%2].set_ylabel('Feature', fontsize=18)
    axs2[i//2, i%2].tick_params(axis='both', which='major', labelsize=16)
    
    RF_PermScoreIndices = np.argsort(RF_perm_importance.importances_mean)[::-1]  # Sort by importance
    
    axs[i, 3].barh(range(authors_data_encoded.shape[1]), RF_perm_importance.importances_mean[RF_PermScoreIndices], align='center')
    axs[i, 3].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 3].set_yticklabels(authors_data_encoded.columns)
    axs[i, 3].set_title(f'RF Permutation Importance - {n_clusters} clusters')
    axs[i, 3].set_xlabel('Perm Importance')
    axs[i, 3].set_ylabel('Feature')
    
    axs3[i//2, i%2].barh(range(authors_data_encoded.shape[1]), RF_perm_importance.importances_mean[RF_PermScoreIndices], align='center')
    axs3[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs3[i//2, i%2].set_yticklabels(authors_data_encoded.columns)
    axs3[i//2, i%2].set_title(f'RF Permutation Importance - {n_clusters} clusters', fontsize=22)
    axs3[i//2, i%2].set_xlabel('Perm Importance', fontsize=18)
    axs3[i//2, i%2].set_ylabel('Feature', fontsize=18)
    axs3[i//2, i%2].tick_params(axis='both', which='major', labelsize=16)
    
    DT_PermScoreIndices = np.argsort(DT_perm_importance.importances_mean)[::-1]  # Sort by importance
    
    axs[i, 2].barh(range(authors_data_encoded.shape[1]), DT_perm_importance.importances_mean[DT_PermScoreIndices], align='center')
    axs[i, 2].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 2].set_yticklabels(authors_data_encoded.columns)
    axs[i, 2].set_title(f'DT Permutation Importance - {n_clusters} clusters')
    axs[i, 2].set_xlabel('Perm Importance')
    axs[i, 2].set_ylabel('Feature')
    
    axs4[i//2, i%2].barh(range(authors_data_encoded.shape[1]), DT_perm_importance.importances_mean[DT_PermScoreIndices], align='center')
    axs4[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs4[i//2, i%2].set_yticklabels(authors_data_encoded.columns)
    axs4[i//2, i%2].set_title(f'DT Permutation Importance - {n_clusters} clusters', fontsize=22)
    axs4[i//2, i%2].set_xlabel('Perm Importance', fontsize=18)
    axs4[i//2, i%2].set_ylabel('Feature', fontsize=18)
    axs4[i//2, i%2].tick_params(axis='both', which='major', labelsize=16)
    
    explainerRF = shap.TreeExplainer(RF)
    shap_values_rf = explainerRF.shap_values(authors_data_encoded)
    
    mean_shap_values_rf = np.abs(shap_values_rf).mean(axis=0)  # This works for binary/multiclass

    # If shap_values is 3D (for multiclass), sum over the cluster dimension
    if len(mean_shap_values_rf.shape) > 1:
    # Sum or average across the class dimension
        mean_shap_values_rf = mean_shap_values_rf.mean(axis=1)

    RF_SHAPIndices = np.argsort(mean_shap_values_rf)[::-1]  # Sort by importance
        
    explainerDT = shap.TreeExplainer(DT)
    shap_values_DT = explainerDT.shap_values(authors_data_encoded)
    
    mean_shap_values_DT = np.abs(shap_values_DT).mean(axis=0)  # This works for binary/multiclass
    
    # If shap_values is 3D (for multiclass), sum over the cluster dimension
    if len(mean_shap_values_DT.shape) > 1:
    # Sum or average across the class dimension
        mean_shap_values_DT = mean_shap_values_DT.mean(axis=1)

    DT_SHAPIndices = np.argsort(mean_shap_values_DT)[::-1]  # Sort by importance    
    

    # Now the shapes should match for plotting
    axs[i, 4].barh(range(authors_data_encoded.shape[1]), mean_shap_values_rf[RF_SHAPIndices], align='center')
    axs[i, 4].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 4].set_yticklabels(authors_data_encoded.columns)
    axs[i, 4].set_title(f'RF SHAP Values - {n_clusters} clusters')
    axs[i, 4].set_xlabel('SHAP Values')
    axs[i, 4].set_ylabel('Feature')
    
    axs5[i//2, i%2].barh(range(authors_data_encoded.shape[1]), mean_shap_values_rf[RF_SHAPIndices], align='center')
    axs5[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs5[i//2, i%2].set_yticklabels(authors_data_encoded.columns)
    axs5[i//2, i%2].set_title(f'RF SHAP Values - {n_clusters} clusters', fontsize=22)
    axs5[i//2, i%2].set_xlabel('SHAP Values', fontsize=18)
    axs5[i//2, i%2].set_ylabel('Feature', fontsize=18)
    axs5[i//2, i%2].tick_params(axis='both', which='major', labelsize=16)
    
    axs[i, 5].barh(range(authors_data_encoded.shape[1]), mean_shap_values_DT[DT_SHAPIndices], align='center')
    axs[i, 5].set_yticks(range(authors_data_encoded.shape[1]))
    axs[i, 5].set_yticklabels(authors_data_encoded.columns)
    axs[i, 5].set_title(f'DT SHAP Values - {n_clusters} clusters')
    axs[i, 5].set_xlabel('SHAP Values')
    axs[i, 5].set_ylabel('Feature')
    
    axs6[i//2, i%2].barh(range(authors_data_encoded.shape[1]), mean_shap_values_DT[DT_SHAPIndices], align='center')
    axs6[i//2, i%2].set_yticks(range(authors_data_encoded.shape[1]))
    axs6[i//2, i%2].set_yticklabels(authors_data_encoded.columns)
    axs6[i//2, i%2].set_title(f'DT SHAP Values - {n_clusters} clusters', fontsize=22)
    axs6[i//2, i%2].set_xlabel('SHAP Values', fontsize=18)
    axs6[i//2, i%2].set_ylabel('Feature', fontsize=18)
    
fig.tight_layout()
fig.savefig('/media/work/icarovasconcelos/mono/results/feature_importance.png')

fig1.suptitle('Feature Importance - Decision Tree', fontsize=24)
fig1.tight_layout()
fig1.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_dt.png')

fig2.suptitle('Feature Importance - Random Forest', fontsize=24)
fig2.tight_layout()
fig2.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_rf.png')

fig3.suptitle('Permutation Importance - Random Forest', fontsize=24)
fig3.tight_layout()
fig3.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_perm_rf.png')

fig4.suptitle('Permutation Importance - Decision Tree', fontsize=24)
fig4.tight_layout()
fig4.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_perm_dt.png')

fig5.suptitle('SHAP Values - Random Forest', fontsize=24)
fig5.tight_layout()
fig5.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_shap.png')

fig6.suptitle('SHAP Values - Decision Tree', fontsize=24)
fig6.tight_layout()
fig6.savefig('/media/work/icarovasconcelos/mono/results/feature_importance_shap_dt.png')


plt.close(fig)
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)
    
for i, model_path in enumerate(model_paths):
    model = pickle.load(open(model_path, "rb"))

    # Train a Decision Tree Classifier
    DT = DecisionTreeClassifier(random_state=42)
    DT.fit(authors_data_encoded, model.labels_)
    n_clusters = len(set(model.labels_))

    # Plot the Decision Tree
    plt.figure(figsize=(28, 12))  # Smaller figure size to reduce overall space
    plot_tree(
        DT,
        filled=True,
        feature_names=authors_data.columns,  # Use actual column names from the DataFrame
        class_names=[f'Cluster {i}' for i in range(n_clusters)],  # Adjust class names to the number of clusters
        proportion=False,  # Keep the nodes at uniform size instead of proportional
        max_depth=3,  # Limit the depth of the tree for readability
        rounded=True,  # Round node boxes for a cleaner look
        precision=2,
        # Decrease fontsize based on number of clusters 
        fontsize=15 if n_clusters <= 4 else 12,
    )
    
    # Reduce the white space between nodes
    plt.subplots_adjust(left=0.01, right=0.98, top=0.95, bottom=0.01, hspace=0.1)    
    # Set title and save the figure
    plt.title(f'Decision Tree Classifier - {n_clusters} clusters')
    plt.savefig(f'/media/work/icarovasconcelos/mono/results/decision_tree_{n_clusters}_clusters.png')
    plt.show()
    tree_rules = export_text(DT, feature_names=list(authors_data.columns))
    print(tree_rules)
    print(40*'=')
    print('\n\n\n\n')
    
    # Close the figure to release memory
    plt.close()



    