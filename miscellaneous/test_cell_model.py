
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


def get_cell_reconstruction_dict_mod(cell_model, tensor_for_cell, cell_num=None, max_clusters=12, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False, use_dbscan=False):
    per_cell_internals_dict = {}
    reconstructed_cell = cell_model.construct().numpy(force=True)[:, 0, :]
    real_cell_activity = tensor_for_cell.detach().numpy()

    print(f"reconstructed_cell.shape {reconstructed_cell.shape}")
    print(f"tensor_for_cell.shape {tensor_for_cell.shape}")

    if x00:
        w1 = cell_model.vectors[0][0].detach().numpy()
        X = np.abs(w1.T)
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(X)

    else:
        f = cell_model.vectors[2][1].detach()
        f1 = f.permute(1, 0, 2).reshape(f.shape[1], -1)
        f1 = torch.abs(f1).cpu().numpy()
        print(f"f1.shape {f1.shape}")
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)

    cluster_labels_dict = {}
    cluster_pca_dict = {}
    cluster_centroids_dict = {}

    MSE_dict = {}
    x_pca_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    if use_dbscan:
        cluster_range = [None]  # DBSCAN doesn't care about cluster count
    else:
        cluster_range = range(1, max_clusters)

    for clusters_chosen in cluster_range:

        #     for clusters_chosen in range(1, max_clusters):
        if use_breakpoints:
            print(f"Using breakpoint clustering (clusters = {clusters_chosen})")
            model_input = X_umap if use_umap else (X if x00 else f1)
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

            centroids = np.array([
                model_input[labels == i].mean(axis=0)
                for i in range(clusters_chosen)
            ])
            X_pca = PCA(n_components=3).fit_transform(model_input)

        else:
            model_input = X_umap if use_umap else (X if x00 else f1)

            if use_dbscan:
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                labels = dbscan.fit_predict(model_input)

                # Skip if DBSCAN failed (e.g., all -1)
                unique_labels = np.unique(labels)
                if len(unique_labels[unique_labels != -1]) < 2:
                    print(f"⚠️ DBSCAN found too few clusters: {unique_labels}")
                    continue

                # Re-assign labels so they’re 0-indexed (skip noise points)
                cluster_ids = [i for i in unique_labels if i != -1]
                labels_mapped = np.zeros_like(labels)
                for new_id, old_id in enumerate(cluster_ids):
                    labels_mapped[labels == old_id] = new_id
                labels = labels_mapped

                clusters_chosen = len(np.unique(labels))
                centroids = np.array([model_input[labels == i].mean(axis=0) for i in range(clusters_chosen)])

            else:
                kmeans = KMeans(n_clusters=clusters_chosen, random_state=0, n_init=10)
                labels = kmeans.fit_predict(model_input)
                centroids = kmeans.cluster_centers_

            X_pca = PCA(n_components=3).fit_transform(model_input)

        #             kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
        #             if x00:
        #                 labels = kmeans.fit_predict(X)
        #                 model_input = X
        #             else:
        #                 labels = kmeans.fit_predict(f1)
        #                 model_input = f1
        #             centroids = kmeans.cluster_centers_
        #             if use_umap:
        #                 X_pca = PCA(n_components=3).fit_transform(X_umap)
        #             else:
        #                 X_pca = PCA(n_components=3).fit_transform(model_input)

        cluster_labels_dict[clusters_chosen] = labels
        cluster_centroids_dict[clusters_chosen] = centroids
        cluster_pca_dict[clusters_chosen] = X_pca

        print(f"\nclusters_chosen = {clusters_chosen}")
        print("Before reassignment:")
        for cluster_id in range(clusters_chosen):
            count = np.sum(labels == cluster_id)
            print(f"  Cluster {cluster_id}: {count} trials")

        if reassign_small_clusters:
            if use_umap:
                model_input = X_umap
                centroid_space = np.array([
                    model_input[labels == i].mean(axis=0)
                    for i in range(clusters_chosen)])
            else:
                model_input = X if x00 else f1
                centroid_space = centroids
            for cluster_id in range(clusters_chosen):
                trial_indices = np.where(labels == cluster_id)[0]
                if len(trial_indices) < 3:
                    print(f"  Reassigning cluster {cluster_id} (size={len(trial_indices)})...")
                    for idx in trial_indices:
                        trial = model_input[idx]
                        dists = cdist([trial], centroid_space)[0]
                        dists[cluster_id] = np.inf
                        new_cluster = np.argmin(dists)
                        labels[idx] = new_cluster

        print("After reassignment:")
        for cluster_id in range(clusters_chosen):
            count = np.sum(labels == cluster_id)
            print(f"  Cluster {cluster_id}: {count} trials")

        x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = X_pca
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = labels

        valid_cluster_mean_trials_list = []
        valid_cluster_indices = []
        cluster_trial_indices = {}

        for n in range(clusters_chosen):
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

        # plt.imshow(empty_cell, aspect='auto')
        # plt.show()

        key = f"clusters_chosen_{clusters_chosen}"
        MSE_dict[key] = np.mean((real_cell_activity - empty_cell) ** 2)
        Recon_by_cluster_av_dict[key] = empty_cell
        TCA_reconstructions_dict[key] = reconstructed_cell
        cluster_trial_mean_dict[key] = valid_cluster_mean_trials_list
        indices_for_cluster_number[key] = cluster_trial_indices

    per_cell_internals_dict[f"cell_{cell_num}"] = {
        "MSE_dict": MSE_dict,
        "x_pca_dict": x_pca_dict,
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
# filename = "EC_GLM"
filename = "NDNFanalC"

cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
animal_id = int(sys.argv[2])       # Provided via command-line argument
ranks = int(sys.argv[3])

# cell_id = 0
# animal_id = 10
# ranks = 40

filepath = os.path.join("datasets", filename + ".mat")
activity_dict, factors_dict = ut.preprocess_data2(filepath, normalize=True, new_NDNF=True)

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
    print(tensor_for_animal.shape)
    tensor_for_cell = tensor_for_animal[:,cell_id,:]
    cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    cell_of_interest.requires_grad_()
    components, model = slicetca.decompose(cell_of_interest,
                                       number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_000,
                                           seed=0)

    internals_dict = get_cell_reconstruction_dict_mod(model, tensor_for_cell, cell_num=cell_id, max_clusters=8, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False, use_dbscan=False)

    save_dir = fr"/scratch/msf157/data/ca1_data2/testing_cell_super_new_NDNF_model_ranks{ranks}_reassign_regkmean_x00_cell"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"MSE_EC_cell_latent_{ranks}_animal{animal_id}_cell_id{cell_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump([model, internals_dict], f)

    print(f"Saved model for animal {animal_id} cell {cell_id} to {save_path}")




###############################################################################