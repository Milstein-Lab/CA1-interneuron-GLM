from scipy.spatial.distance import cdist
from scipy.spatial.distance import cityblock
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ruptures as rpt
import sys
import torch
import slicetca
import pickle
import os
import utils as ut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap
from sklearn.decomposition import PCA
import ruptures as rpt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def get_animal_model_reconstruction_dict_dynamic_k(animal_model, tensor_for_animal, max_clusters=12, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False, carryforward=True):
    per_cell_internals_dict = {}
    reconstruction_full_animal = animal_model.construct().numpy(force=True)

    if x00:
        w1 = animal_model.vectors[0][0].detach().numpy()
        X = np.abs(w1.T)
    else:
        f = animal_model.vectors[2][1].detach()
        f1 = f.permute(1, 0, 2).reshape(f.shape[1], -1)
        X = torch.abs(f1).cpu().numpy()  # standardize var name for both

    umap_model = umap.UMAP(n_components=3, random_state=0)
    X_umap = umap_model.fit_transform(X)

    # Use UMAP for clustering if selected
    model_input = X_umap if use_umap else X

    for cell in range(reconstruction_full_animal.shape[1]):

        print(f"\nProcessing cell {cell}...")

        MSE_dict = {}
        x_umap_dict = {}
        labels_dict = {}
        indices_for_cluster_number = {}
        TCA_reconstructions_dict = {}
        Recon_by_cluster_av_dict = {}
        cluster_trial_mean_dict = {}

        reconstructed_cell = reconstruction_full_animal[:, cell, :]
        real_cell_activity = tensor_for_animal[:, cell, :].detach().numpy()

        for clusters_chosen in range(1, max_clusters):
            print(f"\nOriginal k = {clusters_chosen}")

            if use_breakpoints:
                n_bkps = clusters_chosen - 1
                algo = rpt.Binseg(model="l2", min_size=3).fit(model_input)
                try:
                    bkps = algo.predict(n_bkps=n_bkps)
                except rpt.exceptions.BadSegmentationParameters:
                    print("  Skipping due to bad breakpoint config")
                    continue
                labels = np.zeros(model_input.shape[0], dtype=int)
                start = 0
                for cluster_id, end in enumerate(bkps):
                    labels[start:end] = cluster_id
                    start = end
            else:
                kmeans = KMeans(n_clusters=clusters_chosen, random_state=0, n_init=10)
                labels = kmeans.fit_predict(model_input)

            print("Before reassignment:")
            for cluster_id in range(clusters_chosen):
                count = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} trials")

            if np.any([np.sum(labels == i) < 2 for i in range(clusters_chosen)]):

                if reassign_small_clusters:
                    centroid_space = np.array([
                        model_input[labels == i].mean(axis=0)
                        if np.any(labels == i) else np.full((model_input.shape[1],), np.nan)
                        for i in range(clusters_chosen)
                    ])
                    for cluster_id in range(clusters_chosen):
                        trial_indices = np.where(labels == cluster_id)[0]
                        if len(trial_indices) < 2:
                            for idx in trial_indices:
                                trial = model_input[idx]
                                #                             dists = cdist([trial], centroid_space)[0]
                                dists = [cityblock(trial, centroid) if not np.any(np.isnan(centroid)) else np.inf for centroid in centroid_space]
                                dists[cluster_id] = np.inf  # mask current cluster
                                if np.all(np.isnan(dists)):
                                    continue
                                new_cluster = np.nanargmin(dists)
                                labels[idx] = new_cluster

            # Only keep unique, valid clusters
            unique_clusters = np.unique(labels)
            if -1 in unique_clusters:
                unique_clusters = unique_clusters[unique_clusters != -1]
            new_k = len(unique_clusters)
            print(f"After reassignment → actual k = {new_k}")

            _, labels = np.unique(labels, return_inverse=True)

            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            cluster_trial_indices = {}

            for n in range(new_k):
                trial_indices = np.where(labels == n)[0]
                cluster_trial_indices[n] = trial_indices
                if len(trial_indices) == 0:
                    continue
                cluster_trials = real_cell_activity[trial_indices, :]
                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)
                valid_cluster_indices.append((n, trial_indices))

            empty_cell = np.zeros_like(reconstructed_cell)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :] = valid_cluster_mean_trials_list[i]

            for fill_k in range(clusters_chosen, max_clusters):
                key = f"clusters_chosen_{fill_k}"

                if carryforward:
                    # Copy result forward from last valid cluster (new_k)
                    MSE_dict[key] = np.mean((real_cell_activity - empty_cell) ** 2)
                    x_umap_dict[key] = X_umap
                    labels_dict[key] = labels
                    indices_for_cluster_number[key] = cluster_trial_indices
                    TCA_reconstructions_dict[key] = reconstructed_cell
                    Recon_by_cluster_av_dict[key] = empty_cell
                    cluster_trial_mean_dict[key] = valid_cluster_mean_trials_list
                else:
                    if fill_k == new_k:
                        MSE_dict[key] = np.mean((real_cell_activity - empty_cell) ** 2)
                        x_umap_dict[key] = X_umap
                        labels_dict[key] = labels
                        indices_for_cluster_number[key] = cluster_trial_indices
                        TCA_reconstructions_dict[key] = reconstructed_cell
                        Recon_by_cluster_av_dict[key] = empty_cell
                        cluster_trial_mean_dict[key] = valid_cluster_mean_trials_list
                    else:
                        MSE_dict[key] = np.nan
                        x_umap_dict[key] = None
                        labels_dict[key] = None
                        indices_for_cluster_number[key] = None
                        TCA_reconstructions_dict[key] = None
                        Recon_by_cluster_av_dict[key] = None
                        cluster_trial_mean_dict[key] = None

        per_cell_internals_dict[f"cell_{cell}"] = {
            "MSE_dict": MSE_dict,
            "x_umap_dict": x_umap_dict,
            "labels_dict": labels_dict,
            "indices_for_cluster_number": indices_for_cluster_number,
            "TCA_reconstructions_dict": TCA_reconstructions_dict,
            "Recon_by_cluster_av_dict": Recon_by_cluster_av_dict,
            "cluster_trial_mean_dict": cluster_trial_mean_dict,
        }

    return per_cell_internals_dict


# MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
# Load data
#filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
filename = "EC_GLM"

# cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
# animal_id = int(sys.argv[1])       # Provided via command-line argument
# ranks = int(sys.argv[2])

animal_id = 0
ranks = 40

filepath = os.path.join("datasets", filename + ".mat")
activity_dict, factors_dict = ut.preprocess_data(filepath)

filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)


tensor_list_by_animal_all_SST = []
for animal in residual_activity_dict:
    neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
    neural_data_tensor = torch.tensor(neural_data)
    # Normalize per cell
    for i in range(neural_data_tensor.shape[1]):
        cell = neural_data_tensor[:, i, :]
        min_val = cell.min()
        max_val = cell.max()
        neural_data_tensor[:, i, :] = (cell - min_val) / (max_val - min_val + 1e-8)
    tensor_list_by_animal_all_SST.append(neural_data_tensor)


if __name__ == "__main__":

    tensor_for_animal = tensor_list_by_animal_all_SST[animal_id]
    # tensor_for_cell = tensor_for_animal[:,cell_id,:]
    # cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    # cell_of_interest.requires_grad_()
    components, model = slicetca.decompose(tensor_for_animal,
                                       number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_0,
                                           seed=0)

    get_animal_model_reconstruction_dict_dynamic_k(model, tensor_for_animal, max_clusters=8, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False, carryforward=True)

    save_dir = fr"/scratch/msf157/data/ca1_data2/cell_EC_model_ranks{ranks}_kmean_reassign_x00_carryforward"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"MSE_EC_cell_latent_{ranks}_animal{animal_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump([model, internals_dict], f)

    print(f"Saved model for animal {animal_id} to {save_path}")


