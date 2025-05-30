from scipy.spatial.distance import cdist
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ruptures as rpt
import re
import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import ruptures as rpt
from scipy.stats import sem
import h5py
import mat73
import utils as ut
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
from collections import OrderedDict
from sklearn.cluster import KMeans
import umap
from sklearn.metrics import silhouette_score
from GLM_regression import *
import warnings


def load_data_regular(name="NDNFanalC", new_NDNF=True):
    file_path = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM"
    filename = name
    filepath = os.path.join(file_path, "datasets", filename + ".mat")

    activity_dict, factors_dict = preprocess_data2(filepath, normalize=True, new_NDNF=new_NDNF)

    filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

    GLM_params, double_predicted_activity_dict_NDNF_new = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
    double_residual_activity_dict_NDNF_new = ut.get_residual_activity_dict(activity_dict, double_predicted_activity_dict_NDNF_new)

    return GLM_params, activity_dict, double_predicted_activity_dict_NDNF_new, factors_dict, filtered_factors_dict, double_residual_activity_dict_NDNF_new


def load_data_double(name="NDNFanalC", double_track=False):
    file_path = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM"
    filename = name
    filepath = os.path.join(file_path, "datasets", filename + ".mat")

    activity_dict, factors_dict = preprocess_data2(filepath, normalize=True, new_NDNF=True)

    filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

    if double_track:

        double_new_NDNF_velocity_dict = get_double_track_length(filtered_factors_dict, activity=False, dual_track_length=True)
        double_new_NDNF_activity_dict = get_double_track_length(activity_dict, activity=True, dual_track_length=True)

    else:

        double_new_NDNF_velocity_dict = get_double_track_length(filtered_factors_dict, activity=False, dual_track_length=False)
        double_new_NDNF_activity_dict = get_double_track_length(activity_dict, activity=True, dual_track_length=False)

    GLM_params, double_predicted_activity_dict_NDNF_new = ut.fit_GLM_population(double_new_NDNF_velocity_dict, double_new_NDNF_activity_dict, quintile=None, regression='linear')
    double_residual_activity_dict_NDNF_new = ut.get_residual_activity_dict(double_new_NDNF_activity_dict, double_predicted_activity_dict_NDNF_new)

    return activity_dict, double_predicted_activity_dict_NDNF_new, filtered_factors_dict, double_new_NDNF_activity_dict, double_new_NDNF_velocity_dict, double_residual_activity_dict_NDNF_new


def get_cell_mean_sem_plotting_cellTCA(cell_model_dict):
    animal_list = []

    for animal in cell_model_dict[20]:
        cell_cluster_mse_list = []

        for cell in cell_model_dict[20][animal]:
            cluster_list = []
            mse_dict = cell_model_dict[20][animal][cell][1][f"cell_{cell}"]["MSE_dict"]

            for cluster_chosen in mse_dict:
                mse = mse_dict[cluster_chosen]
                cluster_list.append(mse)

            cell_cluster_mse_list.append(cluster_list)

        max_len = max(len(lst) for lst in cell_cluster_mse_list)
        padded = np.array([
            lst + [np.nan] * (max_len - len(lst))
            for lst in cell_cluster_mse_list
        ])

        cell_average_array = np.nanmean(padded, axis=0)
        animal_list.append(cell_average_array)

    animal_array = np.array(animal_list)

    animal_mean = np.nanmean(animal_array, axis=0)
    animal_sem = sem(animal_array, axis=0, nan_policy="omit")

    return animal_mean, animal_sem


def get_model_data_per_animal3(mse_dir, cell_type="EC"):
    # Pattern to extract rank and animal_id
    pattern = re.compile(fr"MSE_{cell_type}_animal_latent_(\d+)_animal(\d+)\.pkl")

    # Structure: {rank: {animal_id: model_list}}
    rank_mse_dict = {}

    for fname in os.listdir(mse_dir):
        if fname.endswith(".pkl"):
            match = pattern.match(fname)
            if match:
                rank = int(match.group(1))
                animal_id = int(match.group(2))

                path = os.path.join(mse_dir, fname)
                with open(path, "rb") as f:
                    model_list = pickle.load(f)

                # Proper way to assign value
                rank_mse_dict.setdefault(rank, {})[animal_id] = model_list
            else:
                print(f"[Skipping] Unexpected filename format: {fname}")

    # Summary printout
    print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
    for rank, animal_dict in rank_mse_dict.items():
        print(f"  Rank {rank}: {len(animal_dict)} animals loaded.")

    return rank_mse_dict


def get_cluster_MSE_for_plotting(internals_per_animal_dict, cell_or_animal="cell"):
    per_animal_list = []

    for animal in internals_per_animal_dict:
        cluster_keys = set()
        cell_dicts = internals_per_animal_dict[animal].values()

        for cell_dict in cell_dicts:
            MSE_dict = cell_dict["MSE_dict"] if cell_or_animal == "cell" else internals_per_animal_dict[animal]["MSE_dict"]
            cluster_keys.update(MSE_dict.keys())

        cluster_keys = sorted(cluster_keys)
        key_to_index = {key: i for i, key in enumerate(cluster_keys)}
        num_keys = len(cluster_keys)

        cell_MSE_matrix = []

        for cell in internals_per_animal_dict[animal]:
            if cell_or_animal == "cell":
                MSE_dict = internals_per_animal_dict[animal][cell]["MSE_dict"]
            else:
                MSE_dict = internals_per_animal_dict[animal]["MSE_dict"]

            row = np.full(num_keys, np.nan)
            for key, mse in MSE_dict.items():
                if key in key_to_index:  # just in case
                    idx = key_to_index[key]
                    row[idx] = mse
            cell_MSE_matrix.append(row)

        cell_MSE_matrix = np.array(cell_MSE_matrix)
        if cell_MSE_matrix.shape[1] == 0:
            print(f"⚠️ Skipping animal {animal} — no valid MSEs")
            continue

        mean_cell_activity = np.nanmean(cell_MSE_matrix, axis=0)
        per_animal_list.append(mean_cell_activity)

    if len(per_animal_list) == 0:
        raise ValueError("😭 No animals had valid MSEs to average!")

    per_animal_array = np.vstack(per_animal_list)
    print("Per-animal array shape:", per_animal_array.shape)

    mean_cluster = np.nanmean(per_animal_array, axis=0)
    sem_cluster = sem(per_animal_array, axis=0, nan_policy="omit")

    return mean_cluster, sem_cluster


def fix_reassignment(cell_SST_kmeans_reassign_x00_dict, cell_SST_model_ranks20_kmeans_reassign_x00, carry_forward=False, use_cell=True, use_animal=False, odd_animal=False):
    animal_list = []

    for animal in cell_SST_kmeans_reassign_x00_dict:
        cell_list = []

        for cell in cell_SST_kmeans_reassign_x00_dict[animal]:
            max_clusters = cell_SST_kmeans_reassign_x00_dict[animal][cell]

            if odd_animal:
                MSE_dict = cell_SST_model_ranks20_kmeans_reassign_x00[animal][cell]['MSE_dict']

            if use_animal:
                MSE_dict = cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][1][cell]["MSE_dict"]

            if use_cell:
                MSE_dict = cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][1][f"cell_{cell}"]["MSE_dict"]

            MSE_list = []
            for cluster in sorted(MSE_dict.keys(), key=lambda x: int(x.split("_")[-1])):
                MSE_list.append(MSE_dict[cluster])

            MSE_array = np.array(MSE_list, dtype=float)

            if max_clusters < len(MSE_array):
                if carry_forward:
                    MSE_array[max_clusters:] = MSE_array[max_clusters - 1]
                else:
                    MSE_array[max_clusters:] = np.nan

            cell_list.append(MSE_array)

        cell_array = np.array(cell_list)

        cell_array = np.array(cell_list)

        # Debug check for empty slices:
        for idx, arr in enumerate(cell_array):
            if np.isnan(arr).all():
                print(f"⚠️ Cell {idx} in animal {animal} has all-NaN MSE values!")

        cell_mean = np.nanmean(cell_array, axis=0)
        cell_mean = np.nanmean(cell_array, axis=0)
        animal_list.append(cell_mean)

    animal_array = np.array(animal_list)
    animal_mean = np.nanmean(animal_array, axis=0)
    animal_sem = sem(animal_array, axis=0, nan_policy='omit')

    print(animal_mean.shape)
    print(animal_sem.shape)
    return animal_mean, animal_sem


def get_example_cluster_counts_cell(cell_SST_model_ranks20_kmeans_reassign_x00):
    max_clusters_per_cell = []

    max_clusters_dict = {}

    for animal in cell_SST_model_ranks20_kmeans_reassign_x00[20]:
        max_clusters_cell_dict = {}
        for cell in cell_SST_model_ranks20_kmeans_reassign_x00[20][animal]:
            labels_dict = cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][1][f"cell_{cell}"]["labels_dict"]
            new_max_unique = []
            for clusters_chosen in labels_dict:
                #                 print(f"labels_dict[clusters_chosen].shape {labels_dict[clusters_chosen].shape}")
                labels = labels_dict[clusters_chosen]
                unique_vals, counts = np.unique(labels, return_counts=True)

                num_unique = len(np.unique(labels_dict[clusters_chosen]))
                new_max_unique.append(num_unique)
            new_max_unique = np.array(new_max_unique)
            max_unique = np.max(new_max_unique)
            max_clusters_per_cell.append(max_unique)
            max_clusters_cell_dict[cell] = max_unique

        max_clusters_dict[animal] = max_clusters_cell_dict
        unique_counts = set(max_clusters_cell_dict.values())

    print(len(max_clusters_per_cell))

    return max_clusters_per_cell, max_clusters_dict


def get_example_cluster_counts(animal_SST_model_ranks20_kmean_reassign_x00_fixed, solo=False):
    max_clusters_per_cell = []

    max_clusters_dict = {}

    if solo:
        for animal in animal_SST_model_ranks20_kmean_reassign_x00_fixed:
            max_clusters_cell_dict = {}
            for cell in animal_SST_model_ranks20_kmean_reassign_x00_fixed[animal]:
                labels_dict = animal_SST_model_ranks20_kmean_reassign_x00_fixed[animal][cell]['labels_dict']
                new_max_unique = []
                for clusters_chosen in labels_dict:
                    #                 print(f"labels_dict[clusters_chosen].shape {labels_dict[clusters_chosen].shape}")
                    labels = labels_dict[clusters_chosen]
                    unique_vals, counts = np.unique(labels, return_counts=True)
                    singleton_labels = unique_vals[counts == 1]
                    if len(singleton_labels) > 0:
                        print(f"Warning: {len(singleton_labels)} singleton(s) found in clusters_chosen = {clusters_chosen}")

                    num_unique = len(np.unique(labels_dict[clusters_chosen]))
                    new_max_unique.append(num_unique)
                new_max_unique = np.array(new_max_unique)
                max_unique = np.max(new_max_unique)
                max_clusters_per_cell.append(max_unique)
                max_clusters_cell_dict[cell] = max_unique

            max_clusters_dict[animal] = max_clusters_cell_dict
            unique_counts = set(max_clusters_cell_dict.values())

        print(len(max_clusters_per_cell))

    else:
        for animal in animal_SST_model_ranks20_kmean_reassign_x00_fixed[20]:
            max_clusters_cell_dict = {}
            for cell in animal_SST_model_ranks20_kmean_reassign_x00_fixed[20][animal][1]:
                labels_dict = animal_SST_model_ranks20_kmean_reassign_x00_fixed[20][animal][1][cell]['labels_dict']
                new_max_unique = []
                for clusters_chosen in labels_dict:
                    #                 print(f"labels_dict[clusters_chosen].shape {labels_dict[clusters_chosen].shape}")
                    labels = labels_dict[clusters_chosen]
                    print(labels)
                    unique_vals, counts = np.unique(labels, return_counts=True)
                    singleton_labels = unique_vals[counts == 1]
                    if len(singleton_labels) > 0:
                        print(f"Warning: {len(singleton_labels)} singleton(s) found in clusters_chosen = {clusters_chosen}")

                    num_unique = len(np.unique(labels_dict[clusters_chosen]))
                    new_max_unique.append(num_unique)
                new_max_unique = np.array(new_max_unique)
                max_unique = np.max(new_max_unique)
                max_clusters_per_cell.append(max_unique)

                max_clusters_cell_dict[cell] = max_unique

            max_clusters_dict[animal] = max_clusters_cell_dict

        print(len(max_clusters_per_cell))

    return max_clusters_per_cell, max_clusters_dict


def get_model_data_per_animal(mse_dir, cell_type="EC"):

    # Pattern to extract rank and animal_id
    pattern = re.compile(fr"MSE_{cell_type}_cell_latent_(\d+)_animal(\d+)\.pkl")

    # Structure: {rank: {animal_id: model_list}}
    rank_mse_dict = {}

    for fname in os.listdir(mse_dir):
        if fname.endswith(".pkl"):
            match = pattern.match(fname)
            if match:
                rank = int(match.group(1))
                animal_id = int(match.group(2))

                path = os.path.join(mse_dir, fname)
                with open(path, "rb") as f:
                    model_list = pickle.load(f)

                # Proper way to assign value
                rank_mse_dict.setdefault(rank, {})[animal_id] = model_list
            else:
                print(f"[Skipping] Unexpected filename format: {fname}")

    # Summary printout
    print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
    for rank, animal_dict in rank_mse_dict.items():
        print(f"  Rank {rank}: {len(animal_dict)} animals loaded.")

    return rank_mse_dict


# def get_model_data_per_animal2(mse_dir, cell_type="EC"):
#     import re
#     import os
#     import pickle
#
#     # Pattern to extract rank and animal_id
#     pattern = re.compile(fr"MSE_{cell_type}_animal_latent_(\d+)_animal(\d+)\.pkl")
#
#     # Structure: {rank: {animal_id: model_list}}
#     rank_mse_dict = {}
#
#     for fname in os.listdir(mse_dir):
#         if fname.endswith(".pkl"):
#             match = pattern.match(fname)
#             if match:
#                 rank = int(match.group(1))
#                 animal_id = int(match.group(2))
#
#                 path = os.path.join(mse_dir, fname)
#                 with open(path, "rb") as f:
#                     model_list = pickle.load(f)
#
#                 # Proper way to assign value
#                 rank_mse_dict.setdefault(rank, {})[animal_id] = model_list
#             else:
#                 print(f"[Skipping] Unexpected filename format: {fname}")
#
#     # Summary printout
#     print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
#     for rank, animal_dict in rank_mse_dict.items():
#         print(f"  Rank {rank}: {len(animal_dict)} animals loaded.")
#
#     return rank_mse_dict

def get_model_data_per_animal2(mse_dir, cell_type="EC"):
    import re
    import os
    import pickle
    from collections import OrderedDict

    pattern = re.compile(fr"MSE_{cell_type}_animal_latent_(\d+)_animal(\d+)\.pkl")
    rank_mse_dict = {}

    for fname in os.listdir(mse_dir):
        if fname.endswith(".pkl"):
            match = pattern.match(fname)
            if match:
                rank = int(match.group(1))
                animal_id = int(match.group(2))

                path = os.path.join(mse_dir, fname)
                with open(path, "rb") as f:
                    model_list = pickle.load(f)

                rank_mse_dict.setdefault(rank, {})[animal_id] = model_list
            else:
                print(f"[Skipping] Unexpected filename format: {fname}")

    # Sort the animal IDs in each rank
    for rank in rank_mse_dict:
        sorted_animals = dict(sorted(rank_mse_dict[rank].items()))
        rank_mse_dict[rank] = sorted_animals

    # Summary printout
    print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
    for rank, animal_dict in rank_mse_dict.items():
        print(f"  Rank {rank}: {len(animal_dict)} animals loaded. IDs: {list(animal_dict.keys())}")

    return rank_mse_dict



def preprocess_animal(animal_EC_model_ranks20_contig_x00, residual_activity_dict, num_clusters=8, reassign_clusters=False, x00=True, umap=True, contiguous=True, ranks=20):
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

    internals_per_animal_dict_EC_animal_x00_regkmean = {}
    for animal in animal_EC_model_ranks20_contig_x00[ranks]:
        animal_model = animal_EC_model_ranks20_contig_x00[ranks][animal][0]
        tensor_for_animal = tensor_list_by_animal_all_SST[animal]
        internals_dict = get_animal_model_reconstruction_dict_mod(animal_model, tensor_for_animal, max_clusters=num_clusters, display=False, reassign_small_clusters=reassign_clusters, x00=x00, use_umap=umap, use_breakpoints=contiguous)

        internals_per_animal_dict_EC_animal_x00_regkmean[animal] = internals_dict

    return internals_per_animal_dict_EC_animal_x00_regkmean


def get_animal_model_reconstruction_dict_mod(animal_model, tensor_for_animal, max_clusters=12, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    per_cell_internals_dict = {}
    reconstruction_full_animal = animal_model.construct().numpy(force=True)

    if x00:
        w1 = animal_model.vectors[0][0].detach().numpy()
        X = np.abs(w1.T)
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(X)
    else:
        f = animal_model.vectors[2][1].detach()
        f1 = f.permute(1, 0, 2).reshape(f.shape[1], -1)
        f1 = torch.abs(f1).cpu().numpy()
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)

    cluster_labels_dict = {}
    cluster_pca_dict = {}
    cluster_centroids_dict = {}

    for clusters_chosen in range(1, max_clusters):
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
            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            model_input = X_umap if use_umap else (X if x00 else f1)
            labels = kmeans.fit_predict(model_input)

            centroids = kmeans.cluster_centers_
            if use_umap:
                X_pca = PCA(n_components=3).fit_transform(X_umap)
            else:
                X_pca = PCA(n_components=3).fit_transform(model_input)

        cluster_labels_dict[clusters_chosen] = labels
        cluster_centroids_dict[clusters_chosen] = centroids
        cluster_pca_dict[clusters_chosen] = X_pca

    for cell in range(reconstruction_full_animal.shape[1]):
        #     cell = 0
        print(f"Processing cell {cell}...")

        MSE_dict = {}
        x_pca_dict = {}
        labels_dict = {}
        indices_for_cluster_number = {}
        TCA_reconstructions_dict = {}
        Recon_by_cluster_av_dict = {}
        cluster_trial_mean_dict = {}

        reconstructed_cell = reconstruction_full_animal[:, cell, :]
        real_cell_activity = tensor_for_animal[:, cell, :].detach().numpy()

        for clusters_chosen in range(1, max_clusters):
            labels = cluster_labels_dict[clusters_chosen].copy()
            centroids = cluster_centroids_dict[clusters_chosen]
            X_pca = cluster_pca_dict[clusters_chosen]
            model_input = X_umap if use_umap else (X if x00 else f1)

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
                    if len(trial_indices) < 2:
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

            key = f"clusters_chosen_{clusters_chosen}"
            MSE_dict[key] = np.mean((real_cell_activity - empty_cell) ** 2)
            Recon_by_cluster_av_dict[key] = empty_cell
            TCA_reconstructions_dict[key] = reconstructed_cell
            cluster_trial_mean_dict[key] = valid_cluster_mean_trials_list
            indices_for_cluster_number[key] = cluster_trial_indices

        per_cell_internals_dict[f"cell_{cell}"] = {
            "MSE_dict": MSE_dict,
            "x_pca_dict": x_pca_dict,
            "labels_dict": labels_dict,
            "indices_for_cluster_number": indices_for_cluster_number,
            "TCA_reconstructions_dict": TCA_reconstructions_dict,
            "Recon_by_cluster_av_dict": Recon_by_cluster_av_dict,
            "cluster_trial_mean_dict": cluster_trial_mean_dict,
        }

    return per_cell_internals_dict


def get_model_data_per_cell(mse_dir):
    pattern = re.compile(r"MSE_.*?_cell_latent_(\d+)_animal(\d+)_cell_id(\d+)\.pkl")

    # Temporarily use unordered nested structure
    raw_rank_mse_dict = {}

    for fname in os.listdir(mse_dir):
        if fname.endswith(".pkl"):
            match = pattern.match(fname)
            if match:
                rank = int(match.group(1))
                animal_id = int(match.group(2))
                cell_id = int(match.group(3))

                path = os.path.join(mse_dir, fname)
                with open(path, "rb") as f:
                    model_list = pickle.load(f)

                raw_rank_mse_dict.setdefault(rank, {}).setdefault(animal_id, {})[cell_id] = model_list
            else:
                print(f"[Skipping] Unexpected filename format: {fname}")

    # Sort everything into OrderedDicts
    rank_mse_dict = OrderedDict()
    for rank in sorted(raw_rank_mse_dict.keys()):
        rank_mse_dict[rank] = OrderedDict()
        for animal_id in sorted(raw_rank_mse_dict[rank].keys()):
            cells = raw_rank_mse_dict[rank][animal_id]
            sorted_cells = OrderedDict(sorted(cells.items()))
            rank_mse_dict[rank][animal_id] = sorted_cells

    # Summary printout
    print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
    for rank, animal_dict in rank_mse_dict.items():
        print(f"  Rank {rank}:")
        for animal_id, cell_dict in animal_dict.items():
            print(f"    Animal {animal_id}: {len(cell_dict)} cells")

    return rank_mse_dict


def plot_per_cell_clustering_internals_single_cluster(cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, residual_activity_dict_NDNF, animal_id=1, cell_id=1, num_clusters=4, plot=True):
    tensor_list_by_animal_all_NDNF = []
    for animal in residual_activity_dict_NDNF:
        neural_data = ut.get_animal_neural_tensor(residual_activity_dict_NDNF, animal=animal)
        neural_data_tensor = torch.tensor(neural_data)
        # Normalize per cell
        for i in range(neural_data_tensor.shape[1]):
            cell = neural_data_tensor[:, i, :]
            min_val = cell.min()
            max_val = cell.max()
            neural_data_tensor[:, i, :] = (cell - min_val) / (max_val - min_val + 1e-8)
        tensor_list_by_animal_all_NDNF.append(neural_data_tensor)

    real_activity = tensor_list_by_animal_all_NDNF[animal_id][:, cell_id, :].detach().numpy()

    #     for culsters_chosen in cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["TCA_reconstructions_dict"]:

    TCA_reco = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["TCA_reconstructions_dict"][f"clusters_chosen_{num_clusters}"]

    animal_reco = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["Recon_by_cluster_av_dict"][f"clusters_chosen_{num_clusters}"]

    indices_dict = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["indices_for_cluster_number"][f"clusters_chosen_{num_clusters}"]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        im1 = (axs[0].imshow(real_activity, aspect='auto', vmin=0, vmax=1))
        axs[0].set_title("Raw Data")
        fig.colorbar(im1, ax=axs[0])
        im2 = axs[1].imshow(TCA_reco, aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im2, ax=axs[1])
        im3 = axs[2].imshow(animal_reco, aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im3, ax=axs[2])
        axs[1].set_title("SliceTCA Reconstruction")
        axs[2].set_title("Cluster Average Reconstruction")
        plt.tight_layout()
        plt.show()

    num_clusters = len(indices_dict)
    clusters_list = []

    if plot:
        fig, axs = plt.subplots(2, num_clusters, figsize=(num_clusters * 4, 6), squeeze=False)

    for n in range(num_clusters):
        indices = indices_dict[n]
        cluster = real_activity[indices, :]  # trials x bins
        clusters_list.append(cluster)


        if plot:
            axs[0, n].imshow(cluster, aspect='auto', vmin=0, vmax=1)
            axs[0, n].set_title(f"Cluster {n}")
            axs[0, n].set_xlabel("Position")
            axs[0, n].set_ylabel("Trial")

            # Plot the cluster mean
            axs[1, n].plot(np.mean(cluster, axis=0))
            axs[1, n].set_xlabel("Position")
            axs[1, n].set_ylabel("Mean activity")
            axs[1, n].set_ylim(0, 1)
    if plot:
        plt.tight_layout()
        plt.show()

    return clusters_list


def preprocess_data2(filepath, normalize=True, new_NDNF=False):
    factors_dict = {}
    activity_dict = {}

    if new_NDNF:
        with h5py.File(filepath, 'r') as f:
            animal_group = f['animal']
            shiftR_refs = animal_group['ShiftR'][:]
            shiftRunning_refs = animal_group['ShiftRunning'][:]
            shiftL_refs = animal_group['ShiftL'][:]
            shiftV_refs = animal_group['ShiftV'][:]

            for animal_idx in range(len(shiftR_refs)):
                delta_f = np.array(f[shiftR_refs[animal_idx][0]])
                delta_f = delta_f.swapaxes(0, 2)
                velocity = np.array(f[shiftRunning_refs[animal_idx][0]]).T
                lick_rate = np.array(f[shiftL_refs[animal_idx][0]]).T
                reward_loc = np.array(f[shiftV_refs[animal_idx][0]]).T

                if delta_f.shape[1] > 1:
                    delta_f = delta_f[:, 1:, :]  # remove duplicate neuron

                num_trials = min(delta_f.shape[1], velocity.shape[1], lick_rate.shape[1], reward_loc.shape[1])

                delta_f = delta_f[:, :num_trials, :]
                velocity = velocity[:, :num_trials]
                lick_rate = lick_rate[:, :num_trials]
                reward_loc = reward_loc[:, :num_trials]

                nan_trials = (
                        np.any(np.isnan(lick_rate), axis=0) |
                        np.any(np.isnan(reward_loc), axis=0) |
                        np.any(np.isnan(velocity), axis=0) |
                        np.any(np.isnan(delta_f), axis=(0, 2))
                )

                animal_key = f'animal_{animal_idx + 1}'
                factors_dict[animal_key] = {
                    "Licks": lick_rate[:, ~nan_trials],
                    "Reward_loc": reward_loc[:, ~nan_trials],
                    "Velocity": velocity[:, ~nan_trials]
                }

                if normalize:
                    for var in factors_dict[animal_key]:
                        factors_dict[animal_key][var] = ((factors_dict[animal_key][var] - np.min(factors_dict[animal_key][var])) /
                                                         (np.max(factors_dict[animal_key][var]) - np.min(factors_dict[animal_key][var])))

                activity_dict[animal_key] = {}
                for neuron_idx in range(delta_f.shape[2]):  # loop over neurons
                    neuron_activity = delta_f[:, :, neuron_idx]  # (trial, bin)
                    if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0):
                        continue

                    cleaned_activity = neuron_activity[:, ~nan_trials]
                    if normalize:
                        cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)
                    neuron_key = f'cell_{neuron_idx + 1}'
                    activity_dict[animal_key][neuron_key] = cleaned_activity


    else:
        data_dict = mat73.loadmat(filepath)

        # Setup position variables
        num_spatial_bins = 10
        position_matrix = np.zeros((50, num_spatial_bins))
        bin_size = 50 // num_spatial_bins
        for i in range(num_spatial_bins):
            position_matrix[i * bin_size:(i + 1) * bin_size, i] = 1

        for animal_idx, (delta_f, velocity, lick_rate, reward_loc) in enumerate(
                zip(data_dict['animal']['ShiftR'], data_dict['animal']['ShiftRunning'], data_dict['animal']['ShiftLrate'], data_dict['animal']['ShiftV'])):

            num_trials = min(delta_f.shape[1], lick_rate.shape[1], reward_loc.shape[1], velocity.shape[1])
            delta_f = delta_f[:, :num_trials, :]
            velocity = velocity[:, :num_trials]
            lick_rate = lick_rate[:, :num_trials]
            reward_loc = reward_loc[:, :num_trials]

            nan_trials = (
                    np.any(np.isnan(lick_rate), axis=0) |
                    np.any(np.isnan(reward_loc), axis=0) |
                    np.any(np.isnan(velocity), axis=0) |
                    np.any(np.isnan(delta_f), axis=(0, 2)))

            animal_key = f'animal_{animal_idx + 1}'
            factors_dict[animal_key] = {
                "Licks": lick_rate[:, ~nan_trials],
                "Reward_loc": reward_loc[:, ~nan_trials],
                "Velocity": velocity[:, ~nan_trials]}

            # Add position info
            num_trials = factors_dict[animal_key]["Velocity"].shape[1]
            for bin_idx in range(num_spatial_bins):
                bin_key = f"Position_{bin_idx + 1}"
                factors_dict[animal_key][bin_key] = np.tile(position_matrix[:, bin_idx][:, np.newaxis], num_trials)

            if normalize:
                for var in factors_dict[animal_key]:
                    factors_dict[animal_key][var] = (
                            (factors_dict[animal_key][var] - np.min(factors_dict[animal_key][var])) /
                            (np.max(factors_dict[animal_key][var]) - np.min(factors_dict[animal_key][var])))

            activity_dict[animal_key] = {}
            for neuron_idx in range(delta_f.shape[2]):
                neuron_activity = delta_f[:, :, neuron_idx]
                if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0):
                    continue
                cleaned_activity = neuron_activity[:, ~nan_trials]
                if normalize:
                    cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)
                neuron_key = f'cell_{neuron_idx + 1}'
                activity_dict[animal_key][neuron_key] = cleaned_activity

    return activity_dict, factors_dict


def get_sorted_activity(activity_dict, title="New NDNF Raw"):
    data_per_animal_new_NDNF = []
    sorted_data_new_NDNF = []

    for animal in activity_dict:
        cells_data_list = []
        for cell in activity_dict[animal]:
            cell_activity = activity_dict[animal][cell]
            trial_av = np.mean(cell_activity, axis=1)
            cells_data_list.append(trial_av)
        cells_data_array = np.array(cells_data_list)
        sorted_cells_data_array = np.argsort(np.argmax(cells_data_array, axis=1))
        data_per_animal_new_NDNF.append(cells_data_array)
        sorted_data_new_NDNF.append(sorted_cells_data_array)

    n_animals = len(data_per_animal_new_NDNF)
    first_column_cutoff = 9
    n_rows = max(first_column_cutoff, n_animals - first_column_cutoff)

    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 2.5 * n_rows))
    axs = np.atleast_2d(axs)

    max_value = 0
    min_value = 0
    for data in data_per_animal_new_NDNF:
        maximum = np.max(data)
        minimum = np.min(data)
        if maximum > max_value:
            max_value = maximum

        if minimum < min_value:
            min_value = minimum

    for idx, cells_data_array in enumerate(data_per_animal_new_NDNF):
        sorted_indices = sorted_data_new_NDNF[idx]
        sorted_activity = cells_data_array[sorted_indices, :]

        if idx < first_column_cutoff:
            row = idx
            col = 0
        else:
            row = idx - first_column_cutoff
            col = 1

        ax = axs[row, col]
        im = ax.imshow(sorted_activity, aspect='auto', cmap='viridis')
        ax.set_title(f"{title} Animal{idx + 1}")
        ax.set_ylabel("Cells (sorted)")
        ax.set_xlabel("Position Bin")
        fig.colorbar(im, ax=ax, orientation='vertical')

    # Hide any unused axes (in case total < n_rows * 2)
    for row in range(n_rows):
        for col in range(2):
            idx_equiv = row if col == 0 else row + first_column_cutoff
            if idx_equiv >= n_animals:
                axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()


# def get_and_plot_concatenated_data(activity_dict, double_residual_activity_dict_NDNF_new, double_new_NDNF_activity_dict_100, double_residual_activity_dict_NDNF_new_100):
#     seperated_activity_dict_A = {}
#     seperated_activity_dict_B = {}
#
#     for idx, animal in enumerate(activity_dict):
#         cell_seperated_activity_dict_A = {}
#         cell_seperated_activity_dict_B = {}
#
#         for cell in activity_dict[animal]:
#             if idx < 9:
#                 cell_seperated_activity_dict_A[cell] = activity_dict[animal][cell]
#             else:
#                 cell_seperated_activity_dict_B[cell] = activity_dict[animal][cell]
#
#         if idx < 9:
#             seperated_activity_dict_A[animal] = cell_seperated_activity_dict_A
#         else:
#             seperated_activity_dict_B[animal] = cell_seperated_activity_dict_B
#
#     seperated_animal_residual_dict_blank = {}
#     seperated_animal_residual_dict_fixed = {}
#
#     for animal in double_residual_activity_dict_NDNF_new:
#         seperated_cell_dict_blank = {}
#         seperated_cell_dict_fixed = {}
#
#         for cell in double_residual_activity_dict_NDNF_new[animal]:
#             blank_trials = activity_dict[animal][cell].shape[1]
#
#             blank_data = double_residual_activity_dict_NDNF_new[animal][cell][:, :blank_trials]
#             fixed_data = double_residual_activity_dict_NDNF_new[animal][cell][:, blank_trials:]
#
#             seperated_cell_dict_blank[cell] = blank_data
#             seperated_cell_dict_fixed[cell] = fixed_data
#
#         seperated_animal_residual_dict_blank[animal] = seperated_cell_dict_blank
#         seperated_animal_residual_dict_fixed[animal] = seperated_cell_dict_fixed
#
#     animal_belt_A = []
#     animal_belt_B = []
#     animal_belt_A_residuals = []
#     animal_belt_B_residuals = []
#     for ids, animal in enumerate(seperated_animal_residual_dict_blank):
#         cells_belt_A = []
#         cells_belt_B = []
#         cells_belt_A_residuals = []
#         cells_belt_B_residuals = []
#         for cell in seperated_animal_residual_dict_blank[animal]:
#             cells_belt_A.append(np.mean(seperated_activity_dict_A[f"animal_{ids + 1}"][cell], axis=1))
#             cells_belt_B.append(np.mean(seperated_activity_dict_B[f"animal_{ids + 10}"][cell], axis=1))
#             cells_belt_A_residuals.append(np.mean(seperated_animal_residual_dict_blank[animal][cell], axis=1))
#             cells_belt_B_residuals.append(np.mean(seperated_animal_residual_dict_fixed[animal][cell], axis=1))
#
#         cells_belt_A_array = np.array(cells_belt_A)
#         cells_belt_B_array = np.array(cells_belt_B)
#         cells_belt_A_residuals_array = np.array(cells_belt_A_residuals)
#         cells_belt_B_residuals_array = np.array(cells_belt_B_residuals)
#
#         animal_belt_A.append(np.mean(cells_belt_A_array, axis=0))
#         animal_belt_B.append(np.mean(cells_belt_B_array, axis=0))
#         animal_belt_A_residuals.append(np.mean(cells_belt_A_residuals_array, axis=0))
#         animal_belt_B_residuals.append(np.mean(cells_belt_B_residuals_array, axis=0))
#
#     animal_belt_A_residuals_array = np.array(animal_belt_A_residuals)
#     mean_animal_belt_A_residuals = np.mean(animal_belt_A_residuals, axis=0)
#     sem_animal_belt_A_residuals = sem(animal_belt_A_residuals, axis=0)
#
#     animal_belt_B_residuals_array = np.array(animal_belt_B_residuals)
#     mean_animal_belt_B_residuals = np.mean(animal_belt_B_residuals, axis=0)
#     sem_animal_belt_B_residuals = sem(animal_belt_B_residuals, axis=0)
#
#     animal_belt_A_array = np.array(animal_belt_A)
#     mean_animal_belt_A = np.mean(animal_belt_A, axis=0)
#     sem_animal_belt_A = sem(animal_belt_A, axis=0)
#
#     animal_belt_B_array = np.array(animal_belt_B)
#     mean_animal_belt_B = np.mean(animal_belt_B, axis=0)
#     sem_animal_belt_B = sem(animal_belt_B, axis=0)
#
#     fig, axs = plt.subplots(3, 3, figsize=(20, 12))
#     axs[0, 0].plot(mean_animal_belt_A, color='grey', label='Raw')
#     axs[0, 0].fill_between(range(len(mean_animal_belt_A)), mean_animal_belt_A + sem_animal_belt_A, mean_animal_belt_A - sem_animal_belt_A, alpha=0.3, color='grey')
#     axs[0, 0].set_ylim(-0.25, 0.65)
#     axs[0, 0].plot(mean_animal_belt_A_residuals, color='orange', label='Velocity-Subtracted Residual')
#     axs[0, 0].fill_between(range(len(mean_animal_belt_A_residuals)), mean_animal_belt_A_residuals + sem_animal_belt_A_residuals, mean_animal_belt_A_residuals - sem_animal_belt_A_residuals, alpha=0.3, color='orange')
#     axs[0, 0].set_title("Belt E (Random)")
#     axs[0, 0].set_ylabel("DF/F")
#     axs[0, 0].set_xlabel("Position Bin")
#     axs[0, 0].legend()
#
#     axs[0, 1].plot(mean_animal_belt_B, color='grey', label='Raw')
#     axs[0, 1].fill_between(range(len(mean_animal_belt_B)), mean_animal_belt_B + sem_animal_belt_B, mean_animal_belt_B - sem_animal_belt_B, alpha=0.3, color='grey')
#     axs[0, 1].plot(mean_animal_belt_B_residuals, color='r', label='Velocity-Subtracted Residual')
#     axs[0, 1].fill_between(range(len(mean_animal_belt_B_residuals)), mean_animal_belt_B_residuals + sem_animal_belt_B_residuals, mean_animal_belt_B_residuals - sem_animal_belt_B_residuals, alpha=0.3, color='r')
#     axs[0, 1].set_ylim(-0.25, 0.65)
#     axs[0, 1].set_title("Belt 1A (Cue+Fixed)")
#     axs[0, 1].set_ylabel("DF/F")
#     axs[0, 1].set_xlabel("Position Bin")
#     axs[0, 1].legend()
#
#     for i in range(len(animal_belt_A_array)):
#         axs[1, 0].plot(animal_belt_A_array[i])
#         axs[1, 0].set_title("Belt E (Random) Raw Animal Traces")
#         axs[1, 0].set_ylim(-0.5, 2.0)
#         axs[1, 1].plot(animal_belt_B_array[i])
#         axs[1, 1].set_title("Belt 1A (Cue+Fixed) Raw Animal Traces")
#         axs[1, 1].set_ylim(-0.5, 2.0)
#         axs[2, 0].plot(animal_belt_A_residuals_array[i])
#         axs[2, 0].set_title("Belt E (Random) Residual Animal Traces")
#         axs[2, 0].set_ylim(-0.5, 2.0)
#         axs[2, 1].plot(animal_belt_B_residuals_array[i])
#         axs[2, 1].set_title("Belt 1A (Cue+Fixed) Residual Animal Traces")
#         axs[2, 1].set_ylim(-0.5, 2.0)
#         axs[1, 0].set_ylabel("DF/F")
#         axs[1, 0].set_xlabel("Position Bin")
#         axs[1, 1].set_ylabel("DF/F")
#         axs[1, 1].set_xlabel("Position Bin")
#         axs[2, 0].set_ylabel("DF/F")
#         axs[2, 0].set_xlabel("Position Bin")
#         axs[2, 1].set_ylabel("DF/F")
#         axs[2, 1].set_xlabel("Position Bin")
#
#     animal_data_belt_old = []
#     residual_data_belt_old = []
#     belt_old_list = []
#     belt_old_residuals_list = []
#
#     correlation_dict_resid_raw = {}
#
#     for idx, animal in enumerate(double_new_NDNF_activity_dict_100):
#         cell_list = []
#         residual_list = []
#         correlation_per_cell_resid_raw = {}
#         cells_belt_old = {}
#         cells_belt_old_residuals = {}
#
#         for cell in double_new_NDNF_activity_dict_100[animal]:
#             cell_act = double_new_NDNF_activity_dict_100[animal][cell]
#             residual_act = double_residual_activity_dict_NDNF_new_100[animal][cell]
#             trial_av = np.mean(cell_act, axis=1)
#             trial_av_residuals = np.mean(residual_act, axis=1)
#             #         r_value_resid_raw, _ = pearsonr(trial_av, trial_av_residuals)
#
#             cells_belt_old[cell] = cell_act
#             cells_belt_old_residuals[cell] = residual_act
#
#             #         correlation_per_cell_resid_raw[cell] = r_value_resid_raw
#             cell_list.append(trial_av)
#             residual_list.append(trial_av_residuals)
#         belt_old_list.append(cells_belt_old)
#         belt_old_residuals_list.append(cells_belt_old_residuals)
#
#         correlation_dict_resid_raw[animal] = correlation_per_cell_resid_raw
#         cell_array = np.array(cell_list)
#         residual_array = np.array(residual_list)
#         cell_av = np.mean(cell_array, axis=0)
#         residual_av = np.mean(residual_array, axis=0)
#         animal_data_belt_old.append(cell_av)
#         residual_data_belt_old.append(residual_av)
#
#     array_animal_data_belt_old = np.array(animal_data_belt_old)
#     mean_animal_belt_old = np.mean(array_animal_data_belt_old, axis=0)
#     sem_animal_belt_old = sem(array_animal_data_belt_old, axis=0)
#
#     array_residuals_A = np.array(residual_data_belt_old)
#     mean_residual_belt_old = np.mean(array_residuals_A, axis=0)
#     sem_residual_belt_old = sem(array_residuals_A, axis=0)
#
#     axs[0, 2].plot(mean_animal_belt_old, color='grey', label='Raw')
#     axs[0, 2].fill_between(range(len(mean_animal_belt_old)), mean_animal_belt_old + sem_animal_belt_old, mean_animal_belt_old - sem_animal_belt_old, alpha=0.3, color='grey')
#     axs[0, 2].set_ylim(-0.25, 0.65)
#     axs[0, 2].plot(mean_residual_belt_old, color='orange', label='Velocity-Subtracted Residual')
#     axs[0, 2].fill_between(range(len(mean_residual_belt_old)), mean_residual_belt_old + sem_residual_belt_old, mean_residual_belt_old - sem_residual_belt_old, alpha=0.3, color='orange')
#     axs[0, 2].set_ylim(-0.25, 0.65)
#     axs[0, 2].set_title("New NDNF Concatenated")
#     axs[0, 2].set_ylabel("DF/F")
#     axs[0, 2].set_xlabel("Position Bin")
#     axs[0, 2].legend()
#
#     for i in range(len(animal_data_belt_old)):
#         axs[1, 2].plot(animal_data_belt_old[i])
#         axs[1, 2].set_title("New NDNF Animal Traces Concatenated")
#         axs[1, 2].set_ylim(-0.5, 2.0)
#         axs[1, 2].set_ylabel("DF/F")
#         axs[1, 2].set_xlabel("Position Bin")
#         axs[2, 2].plot(residual_data_belt_old[i])
#         axs[2, 2].set_title("New NDNF Animal Traces Concatenated")
#         axs[2, 2].set_ylim(-0.5, 2.0)
#         axs[2, 2].set_ylabel("DF/F")
#         axs[2, 2].set_xlabel("Position Bin")
#
#     plt.tight_layout()
#     plt.plot()
#

def split_new_NDNF_into_two(activity_dict, factor=False):
    if factor:
        random_reward_new_NDNF_activity_dict = {}
        fixed_reward_new_NDNF_activity_dict = {}

        for idx, animal in enumerate(activity_dict):
            random_reward_cell = {}
            fixed_reward_cell = {}
            for cell in activity_dict[animal]:
                if idx < 9:
                    random_reward_cell[cell] = activity_dict[animal]["Velocity"]
                else:
                    fixed_reward_cell[cell] = activity_dict[animal]["Velocity"]
            if idx < 9:
                random_reward_new_NDNF_activity_dict[animal] = random_reward_cell
            else:
                fixed_reward_new_NDNF_activity_dict[animal] = fixed_reward_cell

        return random_reward_new_NDNF_activity_dict, fixed_reward_new_NDNF_activity_dict

    else:
        random_reward_new_NDNF_activity_dict = {}
        fixed_reward_new_NDNF_activity_dict = {}

        for idx, animal in enumerate(activity_dict):
            random_reward_cell = {}
            fixed_reward_cell = {}
            for cell in activity_dict[animal]:
                if idx < 9:
                    random_reward_cell[cell] = activity_dict[animal][cell]
                else:
                    fixed_reward_cell[cell] = activity_dict[animal][cell]
            if idx < 9:
                random_reward_new_NDNF_activity_dict[animal] = random_reward_cell
            else:
                fixed_reward_new_NDNF_activity_dict[animal] = fixed_reward_cell

        return random_reward_new_NDNF_activity_dict, fixed_reward_new_NDNF_activity_dict


def get_plot_data(activity_dict, residual_activity_dict_NDNF_new, activity_dict_old_NDNF, residual_activity_dict_NDNF_old, animal_data=False, combine_new_old=True):
    combined_dict = {}

    for animal in activity_dict:
        combined_dict_cells = {}
        for cell in activity_dict[animal]:
            combined_dict_cells[cell] = activity_dict[animal][cell]
        combined_dict[f"new_{animal}"] = combined_dict_cells

    for animal in activity_dict_old_NDNF:
        combined_dict_cell = {}
        for cell in activity_dict_old_NDNF[animal]:
            combined_dict_cell[cell] = activity_dict_old_NDNF[animal][cell]
        combined_dict[f"old_{animal}"] = combined_dict_cell


    combined_dict_residuals = {}

    for animal in residual_activity_dict_NDNF_new:
        combined_dict_cells = {}
        for cell in residual_activity_dict_NDNF_new[animal]:
            combined_dict_cells[cell] = residual_activity_dict_NDNF_new[animal][cell]
        combined_dict_residuals[f"new_{animal}"] = combined_dict_cells

    for animal in residual_activity_dict_NDNF_old:
        combined_dict_cell = {}
        for cell in residual_activity_dict_NDNF_old[animal]:
            combined_dict_cell[cell] = residual_activity_dict_NDNF_old[animal][cell]
        combined_dict_residuals[f"old_{animal}"] = combined_dict_cell

    animal_data_belt_B_combined = []
    residual_data_belt_B_combined = []

    animal_data_belt_B_new = []
    residual_data_belt_B_new = []

    animal_data_belt_B_old = []
    residual_data_belt_B_old = []

    for idx, animal in enumerate(combined_dict):
        cell_list = []
        residual_list = []

        for cell in combined_dict[animal]:
            cell_act = combined_dict[animal][cell]
            residual_act = combined_dict_residuals[animal][cell]
            trial_av = np.mean(cell_act, axis=1)
            trial_av_residuals = np.mean(residual_act, axis=1)

            if idx > 8:
                animal_data_belt_B_combined.append(np.mean(cell_act, axis=1))
                residual_data_belt_B_combined.append(np.mean(residual_act, axis=1))

            if 9 <= idx < 18:
                animal_data_belt_B_new.append(np.mean(cell_act, axis=1))
                residual_data_belt_B_new.append(np.mean(residual_act, axis=1))

            if idx >= 18:
                animal_data_belt_B_old.append(np.mean(cell_act, axis=1))
                residual_data_belt_B_old.append(np.mean(residual_act, axis=1))

    array_animal_data_belt_B_combined = np.array(animal_data_belt_B_combined)
    mean_animal_belt_B_combined = np.mean(array_animal_data_belt_B_combined, axis=0)
    sem_animal_belt_B_combined = sem(array_animal_data_belt_B_combined, axis=0)

    array_animal_data_belt_B_combined_residuals = np.array(residual_data_belt_B_combined)
    mean_animal_data_belt_B_combined_residuals = np.mean(array_animal_data_belt_B_combined_residuals, axis=0)
    sem_animal_data_belt_B_combined_residuals = sem(array_animal_data_belt_B_combined_residuals, axis=0)

    array_animal_data_belt_B_new = np.array(animal_data_belt_B_new)
    mean_animal_belt_B_new = np.mean(array_animal_data_belt_B_new, axis=0)
    sem_animal_belt_B_new = sem(array_animal_data_belt_B_new, axis=0)

    array_animal_data_belt_B_new_residuals = np.array(residual_data_belt_B_new)
    mean_animal_data_belt_B_new_residuals = np.mean(array_animal_data_belt_B_new_residuals, axis=0)
    sem_animal_data_belt_B_new_residuals = sem(array_animal_data_belt_B_new_residuals, axis=0)

    array_animal_data_belt_B_old = np.array(animal_data_belt_B_old)
    mean_animal_belt_B_old = np.mean(array_animal_data_belt_B_old, axis=0)
    sem_animal_belt_B_old = sem(array_animal_data_belt_B_old, axis=0)

    array_animal_data_belt_B_old_residuals = np.array(residual_data_belt_B_old)
    mean_animal_data_belt_B_old_residuals = np.mean(array_animal_data_belt_B_old_residuals, axis=0)
    sem_animal_data_belt_B_old_residuals = sem(array_animal_data_belt_B_old_residuals, axis=0)

    all_heatmap_data = np.concatenate([
        array_animal_data_belt_B_combined.flatten(),
        array_animal_data_belt_B_combined_residuals.flatten(),
        array_animal_data_belt_B_new.flatten(),
        array_animal_data_belt_B_new_residuals.flatten(),
        array_animal_data_belt_B_old.flatten(),
        array_animal_data_belt_B_old_residuals.flatten()
    ])

    vmin = np.min(all_heatmap_data)
    vmax = np.max(all_heatmap_data)

    fig, axs = plt.subplots(3, 3, figsize=(20, 12))
    axs[0, 2].plot(mean_animal_belt_B_combined, color='grey', label='Raw')
    axs[0, 2].fill_between(range(len(mean_animal_belt_B_combined)), mean_animal_belt_B_combined + sem_animal_belt_B_combined, mean_animal_belt_B_combined - sem_animal_belt_B_combined, alpha=0.3, color='grey')
    axs[0, 2].plot(mean_animal_data_belt_B_combined_residuals, color='orange', label='Velocity-Subtracted Residual')
    axs[0, 2].fill_between(range(len(mean_animal_data_belt_B_combined_residuals)), mean_animal_data_belt_B_combined_residuals + sem_animal_data_belt_B_combined_residuals, mean_animal_data_belt_B_combined_residuals - sem_animal_data_belt_B_combined_residuals, alpha=0.3, color='orange')
    axs[0, 2].set_title("Velocity Subtraction Per Random Belt Combined New+Original")
    axs[0, 2].set_ylabel("DF/F")
    axs[0, 2].set_xlabel("Position Bin")
    axs[0, 2].legend()

    sorted_indices = np.argsort(np.argmax(array_animal_data_belt_B_combined, axis=1))
    im1 = axs[1, 2].imshow(array_animal_data_belt_B_combined[sorted_indices,:], aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 2].set_title("Individual Cells Trial Averaged Raw")
    axs[1, 2].set_ylabel("DF/F")
    axs[1, 2].set_xlabel("Position Bin")

    im2 = axs[2, 2].imshow(array_animal_data_belt_B_combined_residuals, aspect='auto', vmin=vmin, vmax=vmax)
    axs[2, 2].set_title("Individual Cells Trial Averaged Vel-Sub Resid.")
    axs[2, 2].set_ylabel("DF/F")
    axs[2, 2].set_xlabel("Position Bin")

    axs[0, 1].plot(mean_animal_belt_B_new, color='grey', label='Raw')
    axs[0, 1].fill_between(range(len(mean_animal_belt_B_new)), mean_animal_belt_B_new + sem_animal_belt_B_new, mean_animal_belt_B_new - sem_animal_belt_B_new, alpha=0.3, color='grey')
    axs[0, 1].plot(mean_animal_data_belt_B_new_residuals, color='orange', label='Velocity-Subtracted Residual')
    axs[0, 1].fill_between(range(len(mean_animal_data_belt_B_new_residuals)), mean_animal_data_belt_B_new_residuals + sem_animal_data_belt_B_new_residuals, mean_animal_data_belt_B_new_residuals - sem_animal_data_belt_B_new_residuals, alpha=0.3, color='orange')
    axs[0, 1].set_title("Velocity Subtraction Per Random Belt Combined New")
    axs[0, 1].set_ylabel("DF/F")
    axs[0, 1].set_xlabel("Position Bin")
    axs[0, 1].legend()

    im3 = axs[1, 1].imshow(array_animal_data_belt_B_new, aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title("Individual Cells Trial Averaged Raw")
    axs[1, 1].set_ylabel("DF/F")
    axs[1, 1].set_xlabel("Position Bin")

    im4 = axs[2, 1].imshow(array_animal_data_belt_B_new_residuals, aspect='auto', vmin=vmin, vmax=vmax)
    axs[2, 1].set_title("Individual Cells Trial Averaged Vel-Sub Resid.")
    axs[2, 1].set_ylabel("DF/F")
    axs[2, 1].set_xlabel("Position Bin")

    axs[0, 0].plot(mean_animal_belt_B_old, color='grey', label='Raw')
    axs[0, 0].fill_between(range(len(mean_animal_belt_B_old)), mean_animal_belt_B_old + sem_animal_belt_B_old, mean_animal_belt_B_old - sem_animal_belt_B_old, alpha=0.3, color='grey')
    axs[0, 0].plot(mean_animal_data_belt_B_old_residuals, color='orange', label='Velocity-Subtracted Residual')
    axs[0, 0].fill_between(range(len(mean_animal_data_belt_B_old_residuals)), mean_animal_data_belt_B_old_residuals + sem_animal_data_belt_B_old_residuals, mean_animal_data_belt_B_old_residuals - sem_animal_data_belt_B_old_residuals, alpha=0.3, color='orange')
    axs[0, 0].set_title("Velocity Subtraction Per Random Belt Combined Original")
    axs[0, 0].set_ylabel("DF/F")
    axs[0, 0].set_xlabel("Position Bin")
    axs[0, 0].legend()

    im5 = axs[1, 0].imshow(array_animal_data_belt_B_old, aspect='auto', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("Individual Cells Trial Averaged Raw")
    axs[1, 0].set_ylabel("DF/F")
    axs[1, 0].set_xlabel("Position Bin")

    im6 = axs[2, 0].imshow(array_animal_data_belt_B_old_residuals, aspect='auto', vmin=vmin, vmax=vmax)
    axs[2, 0].set_title("Individual Cells Trial Averaged Vel-Sub Resid.")
    axs[2, 0].set_ylabel("DF/F")
    axs[2, 0].set_xlabel("Position Bin")

    fig.colorbar(im1, ax=axs[1, 2])
    fig.colorbar(im2, ax=axs[2, 2])
    fig.colorbar(im3, ax=axs[1, 1])
    fig.colorbar(im4, ax=axs[2, 1])
    fig.colorbar(im5, ax=axs[1, 0])
    fig.colorbar(im6, ax=axs[2, 0])

    plt.tight_layout()
    plt.plot()


def plot_changepoint_distribution(fraction_animal_first_changepoints_list_new_NDNF, fraction_animal_second_changepoints_list_new_NDNF, cell_type="EC"):

    first = fraction_animal_first_changepoints_list_new_NDNF
    second = fraction_animal_second_changepoints_list_new_NDNF

    x_first = np.random.normal(1, 0.05, size=len(first))  # center at 1
    x_second = np.random.normal(2, 0.05, size=len(second))  # center at 2

    plt.figure(figsize=(6, 6))

    boxprops = dict(facecolor="lightgrey", color='black', alpha=0.1)
    plt.boxplot([first, second], positions=[1, 2], widths=0.3, showfliers=False, patch_artist=True,
                boxprops=boxprops,
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                medianprops=dict(color='black'))

    plt.scatter(x_first, first, color='blue', alpha=0.7, label='First')
    plt.scatter(x_second, second, color='red', alpha=0.7, label='Second')

    plt.xlim(0.5, 2.5)
    plt.ylim(0, 1)
    plt.xticks([1, 2], ['First Changepoint', 'Second Changepoint'])
    plt.ylabel('Fraction of Total Trials')
    plt.title(f'Changepoint Distribution {cell_type}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="EC", title="NDNF New Cell SliceTCA x00"):
    if cell_type=="EC":
        colors = ["k", 'green']
    elif cell_type=="SST":
        colors = ['cyan', 'blue']
    else:
        colors = ['orange', 'red']

    for i in range(2):
        plt.plot(mean_list_NDNF_new_cell_x00[i], label=["Early", "Late"][i], color=colors[i])
        plt.fill_between(range(len(mean_list_NDNF_new_cell_x00[i])),
                            mean_list_NDNF_new_cell_x00[i] + sem_list_NDNF_new_cell_x00[i],
                            mean_list_NDNF_new_cell_x00[i] - sem_list_NDNF_new_cell_x00[i],
                            color=colors[i], alpha=0.1)
        plt.ylim(-0.6, 1)
    plt.title(title)
    plt.xlabel("Position Bins")
    plt.ylabel("DF/F")
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_contig_cluster_av(cell_new_NDNF_model_ranks20_contiguous_x00_cell, newer_activity, animal_TCA=False):
    if animal_TCA:

        animal_clust_1_list = []
        animal_clust_2_list = []
        animal_clust_3_list = []

        for animal in cell_new_NDNF_model_ranks20_contiguous_x00_cell:

            for cell in cell_new_NDNF_model_ranks20_contiguous_x00_cell[animal]:
                mean_list = cell_new_NDNF_model_ranks20_contiguous_x00_cell[animal][cell]["cluster_trial_mean_dict"]["clusters_chosen_3"]

                animal_clust_1_list.append(mean_list[0])
                animal_clust_2_list.append(mean_list[1])
                animal_clust_3_list.append(mean_list[2])

        animal_clust_1_array = np.array(animal_clust_1_list)
        animal_clust_2_array = np.array(animal_clust_2_list)
        animal_clust_3_array = np.array(animal_clust_3_list)

        mean_animal_clust_1 = np.mean(animal_clust_1_array, axis=0)
        mean_animal_clust_2 = np.mean(animal_clust_2_array, axis=0)
        mean_animal_clust_3 = np.mean(animal_clust_3_array, axis=0)

        sem_animal_clust_1 = sem(animal_clust_1_array, axis=0)
        sem_animal_clust_2 = sem(animal_clust_2_array, axis=0)
        sem_animal_clust_3 = sem(animal_clust_3_array, axis=0)

        mean_list = [mean_animal_clust_1, mean_animal_clust_2, mean_animal_clust_3]
        sem_list = [sem_animal_clust_1, sem_animal_clust_2, sem_animal_clust_3]
        raw_cluster_list = [animal_clust_1_array, animal_clust_2_array, animal_clust_3_array]


    else:

        animal_activity_residual = {}
        for idx, animal in enumerate(newer_activity):
            cell_activity = {}
            for ids, cell in enumerate(newer_activity[animal]):
                cell_activity[ids] = newer_activity[animal][cell]

            animal_activity_residual[idx] = cell_activity

        animal_clust_1_list = []
        animal_clust_2_list = []

        for i, animal in enumerate(cell_new_NDNF_model_ranks20_contiguous_x00_cell[20]):

            for j, cell in enumerate(cell_new_NDNF_model_ranks20_contiguous_x00_cell[20][animal]):
                labels_array = cell_new_NDNF_model_ranks20_contiguous_x00_cell[20][animal][cell][1][f"cell_{cell}"]["labels_dict"]["clusters_chosen_3"]
                cutpoints = np.unique(labels_array)
                change_indices = np.where(np.diff(labels_array) != 0)[0] + 1
                early_cp = change_indices[0]
                late_cp = change_indices[1]

                residuals = animal_activity_residual[i][j]

                early_learn_activity = residuals[:, :early_cp]
                mean_early_learn_activity = np.mean(early_learn_activity, axis=1)
                late_learn_activity = residuals[:, late_cp:]
                mean_late_learn_activity = np.mean(late_learn_activity, axis=1)

                animal_clust_1_list.append(mean_early_learn_activity)
                animal_clust_2_list.append(mean_late_learn_activity)

        animal_clust_1_array = np.array(animal_clust_1_list)
        animal_clust_2_array = np.array(animal_clust_2_list)

        mean_animal_clust_1 = np.mean(animal_clust_1_array, axis=0)
        mean_animal_clust_2 = np.mean(animal_clust_2_array, axis=0)

        sem_animal_clust_1 = sem(animal_clust_1_array, axis=0)
        sem_animal_clust_2 = sem(animal_clust_2_array, axis=0)

        mean_list = [mean_animal_clust_1, mean_animal_clust_2]
        sem_list = [sem_animal_clust_1, sem_animal_clust_2]
        raw_cluster_list = [animal_clust_1_array, animal_clust_2_array]

    return mean_list, sem_list, raw_cluster_list


def get_double_track_length(activity_dict, activity=True, dual_track_length=False):
    if activity:

        if dual_track_length:
            trial_nums_list_A = []
            trial_nums_list_B = []

            for ids, animal in enumerate(activity_dict):
                trail_num = activity_dict[animal]["cell_2"].shape[1]
                if ids < 9:
                    trial_nums_list_A.append(trail_num)
                else:
                    trial_nums_list_B.append(trail_num)

            real_trial_length = []
            for i in range(9):
                t_num_A = trial_nums_list_A[i]
                t_num_B = trial_nums_list_B[i]
                nums = [t_num_A, t_num_B]

                min_num = np.min(nums)
                real_trial_length.append(min_num)

            real_trial_length = np.array(real_trial_length)

            double_real_trial_length = np.concatenate([real_trial_length, real_trial_length])

        animal_dict_A = {}
        animal_dict_B = {}

        for idx, animal in enumerate(activity_dict):
            cell_dict_A = {}
            cell_dict_B = {}
            for cell in activity_dict[animal]:

                neuron_activity = activity_dict[animal][cell]

                if dual_track_length:
                    trialss = double_real_trial_length[idx]
                    neuron_activity_truncated = neuron_activity[:, :trialss]

                if idx < 9:
                    if dual_track_length:
                        cell_dict_A[cell] = neuron_activity_truncated
                    else:
                        cell_dict_A[cell] = neuron_activity
                else:
                    if dual_track_length:
                        cell_dict_B[cell] = neuron_activity_truncated
                    else:
                        cell_dict_B[cell] = neuron_activity

            if idx < 9:
                animal_dict_A[animal] = cell_dict_A
            else:
                animal_dict_B[animal] = cell_dict_B

        double_track_activity_dict_new_NDNF = {}

        for ids in range(9):
            combined_cell_activity_dict = {}

            animal_A = animal_dict_A[f"animal_{ids + 1}"]
            animal_B = animal_dict_B[f"animal_{ids + 10}"]
            for cell in animal_A:
                if dual_track_length:
                    double_track_activity = np.concatenate([animal_A[cell], animal_B[cell]], axis=0)
                else:
                    double_track_activity = np.concatenate([animal_A[cell], animal_B[cell]], axis=1)
                combined_cell_activity_dict[cell] = double_track_activity

            double_track_activity_dict_new_NDNF[f"animal_{ids + 1}"] = combined_cell_activity_dict

        return double_track_activity_dict_new_NDNF

    else:

        if dual_track_length:

            trial_nums_list_A = []
            trial_nums_list_B = []

            for ids, animal in enumerate(activity_dict):
                trail_num = activity_dict[animal]["Velocity"].shape[1]
                if ids < 9:
                    trial_nums_list_A.append(trail_num)
                else:
                    trial_nums_list_B.append(trail_num)

            real_trial_length = []
            for i in range(9):
                t_num_A = trial_nums_list_A[i]
                t_num_B = trial_nums_list_B[i]
                nums = [t_num_A, t_num_B]

                min_num = np.min(nums)
                real_trial_length.append(min_num)

            real_trial_length = np.array(real_trial_length)

            double_real_trial_length = np.concatenate([real_trial_length, real_trial_length])

        animal_dict_A = {}
        animal_dict_B = {}

        for idx, animal in enumerate(activity_dict):
            velocity_array = activity_dict[animal]["Velocity"]

            if dual_track_length:
                trialss = double_real_trial_length[idx]
                velocity_array_truncated = velocity_array[:, :trialss]

            if idx < 9:
                if dual_track_length:
                    animal_dict_A[animal] = velocity_array_truncated
                else:
                    animal_dict_A[animal] = velocity_array
            else:
                if dual_track_length:
                    animal_dict_B[animal] = velocity_array_truncated
                else:
                    animal_dict_B[animal] = velocity_array

        double_track_velocity_dict_new_NDNF = {}

        for ids in range(9):
            velocity_A = animal_dict_A[f"animal_{ids + 1}"]
            velocity_B = animal_dict_B[f"animal_{ids + 10}"]

            if dual_track_length:
                combined_velocity = np.concatenate([velocity_A, velocity_B], axis=0)
            else:
                combined_velocity = np.concatenate([velocity_A, velocity_B], axis=1)

            double_track_velocity_dict_new_NDNF[f"animal_{ids + 1}"] = {"Velocity": combined_velocity}

        return double_track_velocity_dict_new_NDNF


def plot_window(activity_dict, factors_dict, residual_activity_dict_NDNF_new):

    window_length = 30
    half_window = window_length // 2

    animal_reward_locked_data = []

    residual_animal_reward_locked_data = []

    for idx, animal in enumerate(activity_dict):

        if idx < 9:

            cell_data_list = []
            residual_cell_data_list = []

            for cell in activity_dict[animal]:

                reward_data_flat = factors_dict[animal]["Reward_loc"].flatten()


                cell_activity = activity_dict[animal][cell]
                cell_activity_flat = cell_activity.flatten()

                residual_activity_flat = residual_activity_dict_NDNF_new[animal][cell].flatten()

                reward_indices = []

                reward_data = []
                residual_reward_data = []

                for bin_idx, i in enumerate(reward_data_flat):
                    if i > 0:
                        start_index = bin_idx - half_window
                        end_index = bin_idx + half_window
                        if start_index < 0 or end_index > len(cell_activity_flat):
                            continue
                        indices_of_interest = np.arange(start_index, end_index)


                        if reward_indices:
                            previous_indices = reward_indices[-1]
                            overlap = np.intersect1d(indices_of_interest, previous_indices)
                            if overlap.size > 0:
                                continue

                        reward_data.append(cell_activity_flat[indices_of_interest])
                        residual_reward_data.append(residual_activity_flat[indices_of_interest])
                        reward_indices.append(indices_of_interest)

                reward_data_array = np.array(reward_data)
                residual_reward_data_array = np.array(residual_reward_data)

                mean_reward_data_array = np.mean(reward_data_array, axis=0)
                mean_residual_reward_data_array = np.mean(residual_reward_data_array, axis=0)

                cell_data_list.append(mean_reward_data_array)
                residual_cell_data_list.append(mean_residual_reward_data_array)

            cell_data_array = np.array(cell_data_list)
            residual_cell_data_array = np.array(residual_cell_data_list)

            animal_reward_locked_data.append(np.mean(cell_data_array, axis=0))
            residual_animal_reward_locked_data.append(np.mean(residual_cell_data_array, axis=0))

    animal_reward_locked_array = np.array(animal_reward_locked_data)
    mean_animal_reward_locked_array = np.mean(animal_reward_locked_array, axis=0)
    sem_animal_reward_locked_array = sem(animal_reward_locked_array, axis=0)

    residual_animal_reward_locked_array = np.array(residual_animal_reward_locked_data)
    residual_mean_animal_reward_locked_array = np.mean(residual_animal_reward_locked_array, axis=0)
    residual_sem_animal_reward_locked_array = sem(residual_animal_reward_locked_array, axis=0)

    x_vals = np.arange(-(0.5*window_length), 0.5*window_length)
    plt.axvline(0, linestyle="--", color="red", label="Random Reward")  # vertical red dashed line at 0
    plt.plot(x_vals, mean_animal_reward_locked_array, color='grey', label='Raw')
    plt.fill_between(x_vals, mean_animal_reward_locked_array+sem_animal_reward_locked_array, mean_animal_reward_locked_array-sem_animal_reward_locked_array, color='grey', alpha=0.3)
    plt.plot(x_vals, residual_mean_animal_reward_locked_array, color='orange', label='Vel-Sub Residual')
    plt.fill_between(x_vals, residual_mean_animal_reward_locked_array+residual_sem_animal_reward_locked_array, residual_mean_animal_reward_locked_array-residual_sem_animal_reward_locked_array, color='orange', alpha=0.3)
    plt.title(f"{window_length} Bins Surrounding Reward Random Track NDNF")
    plt.ylabel("DF/F")
    plt.xlabel("Position Bins")
    plt.xticks(np.arange(-15, 16, 5))  # ticks from -15 to 15
    plt.tight_layout()
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()



def plot_roll(activity_dict, factors_dict, residual_activity_dict_NDNF_new):

    animal_rolled_data = []
    animal_rolled_resid_data = []

    for idx, animal in enumerate(activity_dict):

        if idx < 9:

            cell_list = []
            cell_resid_list = []

            for cell in activity_dict[animal]:

                cell_activity = activity_dict[animal][cell]
                reward_data = factors_dict[animal]["Reward_loc"]

                residual_activity = residual_activity_dict_NDNF_new[animal][cell]

                rolled_trials = []
                rolled_trials_resid_list = []

                for trial_idx in range(reward_data.shape[1]):
                    trial = reward_data[:, trial_idx]
                    trial_activity = cell_activity[:, trial_idx]
                    trial_residuals = residual_activity[:, trial_idx]

                    if np.sum(trial > 0) != 1:
                        continue

                    reward_idx = np.argmax(trial)
                    shift = 25 - reward_idx

                    rolled_trial = np.roll(trial_activity, shift)

                    plt.plot()


                    rolled_trial_residuals = np.roll(trial_residuals, shift)
                    rolled_trials.append(rolled_trial)
                    rolled_trials_resid_list.append(rolled_trial_residuals)

                rolled_trials_array = np.array(rolled_trials)
                rolled_trials_resid_array = np.array(rolled_trials_resid_list)

                rolled_trials_av = np.mean(rolled_trials_array, axis=0)
                cell_list.append(rolled_trials_av)
                rolled_resid_trials_av = np.mean(rolled_trials_resid_array, axis=0)
                cell_resid_list.append(rolled_resid_trials_av)

            cell_array = np.array(cell_list)
            animal_rolled_data.append(np.mean(cell_array, axis=0))

            cell_resid_array = np.array(cell_resid_list)
            animal_rolled_resid_data.append(np.mean(cell_resid_array, axis=0))


    animal_array = np.array(animal_rolled_data)
    mean_animal = np.mean(animal_array, axis=0)
    sem_animal = sem(animal_array, axis=0)

    animal_resid_array = np.array(animal_rolled_resid_data)
    mean_resid_animal = np.mean(animal_resid_array, axis=0)
    sem_resid_animal = sem(animal_resid_array, axis=0)



    plt.plot(mean_animal, color='grey', label='Raw')
    plt.fill_between(range(len(mean_animal)), mean_animal-sem_animal, mean_animal+sem_animal, color='grey', alpha=0.3)
    plt.plot(mean_resid_animal, color='orange', label="Vel.-Sub. Residual")
    plt.fill_between(range(len(mean_resid_animal)), mean_resid_animal-sem_resid_animal, mean_resid_animal+sem_resid_animal, color='orange', alpha=0.3)
    plt.title("Reward-Sorted NDNF Random Reward Belt")
    plt.ylabel("DF/F")
    plt.xlabel("Position Bin")
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.show()


def get_elbow_score_data(new_cell_SST_model_ranks20_kmeans_reassign_x00, animal=0, cell=0):
    MSE_dict = new_cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][1][f"cell_{cell}"]['MSE_dict']

    MSE_list = []
    for clusters_chosen in MSE_dict:
        MSE_list.append(MSE_dict[clusters_chosen])
    MSE_array = np.array(MSE_list)

    elbow_kmeans = find_elbow_point(MSE_array)

    return elbow_kmeans


def get_label_probs(testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell, elbow_kmeans, animal=1, cell=1):
    labels_dict = testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][1][f"cell_{cell}"]["labels_dict"]
    for idx, clusters_chosen in enumerate(labels_dict):
        cluster_of_interest = elbow_kmeans - 1
        if idx == cluster_of_interest:
            proper_labels = labels_dict[clusters_chosen]
            label_probs = sliding_window_probabilities(proper_labels, window_size=10)

    return label_probs


def get_changepoints(cell_new_NDNF_model_ranks20_contiguous_x00_cell, activity_dict, animal_TCA=False):
    animal_length = []

    for animal in activity_dict:
        for cell in activity_dict[animal]:
            animal_length.append(activity_dict[animal][cell].shape[1])

    animal_first_changepoints_list = []
    animal_second_changepoints_list = []

    fraction_first_changepoints_list = []
    fraction_second_changepoints_list = []

    counts = 0

    if animal_TCA:
        for animal in cell_new_NDNF_model_ranks20_contiguous_x00_cell:

            first_changepoints_list = []
            second_changepoints_list = []

            for cell in cell_new_NDNF_model_ranks20_contiguous_x00_cell[animal]:

                trials_num = animal_length[counts]

                labels = cell_new_NDNF_model_ranks20_contiguous_x00_cell[animal][cell]["labels_dict"]["clusters_chosen_3"]

                change_indices = np.where(np.diff(labels) != 0)[0] + 1

                first_changepoints = change_indices[0]
                first_changepoints_list.append(first_changepoints)
                fraction_first_changepoints_list.append(first_changepoints / trials_num)
                second_changepoints = change_indices[1]

                if np.any(second_changepoints > trials_num):
                    print(f"counts {counts} we got problem {second_changepoints} {trials_num}")
                second_changepoints_list.append(second_changepoints)
                fraction_second_changepoints_list.append(second_changepoints / trials_num)
                counts += 1

            animal_first_changepoints_list.append(first_changepoints_list)
            animal_second_changepoints_list.append(second_changepoints_list)


    else:
        for animal in cell_new_NDNF_model_ranks20_contiguous_x00_cell[20]:

            first_changepoints_list = []
            second_changepoints_list = []

            for cell in cell_new_NDNF_model_ranks20_contiguous_x00_cell[20][animal]:

                trials_num = animal_length[counts]

                labels = cell_new_NDNF_model_ranks20_contiguous_x00_cell[20][animal][cell][1][f"cell_{cell}"]["labels_dict"]["clusters_chosen_3"]

                change_indices = np.where(np.diff(labels) != 0)[0] + 1

                first_changepoints = change_indices[0]
                first_changepoints_list.append(first_changepoints)
                fraction_first_changepoints_list.append(first_changepoints / trials_num)
                second_changepoints = change_indices[1]

                if np.any(second_changepoints > trials_num):
                    print(f"counts {counts} we got problem {second_changepoints} {trials_num}")
                second_changepoints_list.append(second_changepoints)
                fraction_second_changepoints_list.append(second_changepoints / trials_num)
                counts += 1

            animal_first_changepoints_list.append(first_changepoints_list)
            animal_second_changepoints_list.append(second_changepoints_list)

    return animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list


def get_plot_umap(new_cell_SST_model_ranks20_kmeans_reassign_x00, k=5, use_3d_umap=False, umap_seed=42, animal=0, cell=0):
    model = new_cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][0]
    w1 = model.vectors[0][0].detach().numpy()
    X = np.abs(w1.T)

    ### 1. KMeans → UMAP (clustering on original high-dim data)
    kmeans1 = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels_kmeans1 = kmeans1.fit_predict(X)

    umap_vis1 = umap.UMAP(n_components=3 if use_3d_umap else 2, random_state=umap_seed).fit_transform(X)

    ### 2. UMAP → KMeans (dimensionality reduction before clustering)
    umap_lowdim = umap.UMAP(n_components=3 if use_3d_umap else 2, random_state=umap_seed).fit_transform(X)
    kmeans2 = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels_kmeans2 = kmeans2.fit_predict(umap_lowdim)

    ### Plotting
    if use_3d_umap:
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        axs = axs.flatten()

        # Row 1: KMeans → UMAP
        axs[0].scatter(umap_vis1[:, 0], umap_vis1[:, 1], c=labels_kmeans1, cmap='tab10', s=50)
        axs[0].set_title(f"KMeans → UMAP: u0 vs u1")
        axs[1].scatter(umap_vis1[:, 0], umap_vis1[:, 2], c=labels_kmeans1, cmap='tab10', s=50)
        axs[1].set_title(f"KMeans → UMAP: u0 vs u2")
        axs[2].scatter(umap_vis1[:, 1], umap_vis1[:, 2], c=labels_kmeans1, cmap='tab10', s=50)
        axs[2].set_title(f"KMeans → UMAP: u1 vs u2")

#         # Row 2: UMAP → KMeans
#         axs[3].scatter(umap_lowdim[:, 0], umap_lowdim[:, 1], c=labels_kmeans2, cmap='tab10', s=50)
#         axs[3].set_title(f"UMAP → KMeans: u0 vs u1")
#         axs[4].scatter(umap_lowdim[:, 0], umap_lowdim[:, 2], c=labels_kmeans2, cmap='tab10', s=50)
#         axs[4].set_title(f"UMAP → KMeans: u0 vs u2")
#         axs[5].scatter(umap_lowdim[:, 1], umap_lowdim[:, 2], c=labels_kmeans2, cmap='tab10', s=50)
#         axs[5].set_title(f"UMAP → KMeans: u1 vs u2")

        for ax in axs:
            ax.set_xlabel("UMAP")
            ax.set_ylabel("UMAP")

        plt.tight_layout()
        plt.show()

    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        axs[0].scatter(umap_vis1[:, 0], umap_vis1[:, 1], c=labels_kmeans1, cmap='tab10', s=50)
        axs[0].set_title(f"KMeans → UMAP (K={k})")
        axs[0].set_xlabel("UMAP 1")
        axs[0].set_ylabel("UMAP 2")
        axs[0].grid(True)

        axs[1].scatter(umap_lowdim[:, 0], umap_lowdim[:, 1], c=labels_kmeans2, cmap='tab10', s=50)
        axs[1].set_title(f"UMAP → KMeans (K={k})")
        axs[1].set_xlabel("UMAP 1")
        axs[1].set_ylabel("UMAP 2")
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


def sliding_window_probabilities(labels, window_size=10):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n = len(labels)

    # Dictionary to hold probability lists per label
    label_probs = {label: [] for label in unique_labels}

    for start in range(n - window_size + 1):
        window = labels[start:start + window_size]
        counts = np.bincount(window, minlength=unique_labels.max() + 1)
        probs = counts / window_size

        for label in unique_labels:
            label_probs[label].append(probs[label])

    # Convert lists to arrays
    for label in label_probs:
        label_probs[label] = np.array(label_probs[label])

    return label_probs


def find_elbow_point(y_vals, min_index=2):
    from scipy.spatial.distance import cdist
    import numpy as np

    x = np.arange(len(y_vals))
    y = y_vals

    # First and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Compute distances to the line
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_p1 = np.vstack((x - p1[0], y - p1[1])).T
    scalar_proj = np.dot(vec_from_p1, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_from_p1 - proj, axis=1)

    # Force elbow to be at least min_index (default = 2)
    elbow_idx = np.argmax(dist_to_line[min_index:]) + min_index
    return int(elbow_idx)


def compare_umap_kmeans(new_cell_SST_model_ranks20_kmeans_reassign_x00, animal=0, cell=0):
    # use_3d_umap=False, umap_seed=42,

    MSE_dict = new_cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][1][f"cell_{cell}"]['MSE_dict']

    model = new_cell_SST_model_ranks20_kmeans_reassign_x00[20][animal][cell][0]

    w1 = model.vectors[0][0].detach().numpy()
    X = np.abs(w1.T)

    silhouette_scores = []
    cluster_range = range(2, 9)  # silhouette score needs at least 2 clusters

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(np.unique(labels)) > 1:  # at least 2 clusters
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(np.nan)  # if somehow all same cluster

    max_silhouette_reg = np.argmax(silhouette_scores) + 2

    MSE_list = []
    for clusters_chosen in MSE_dict:
        MSE_list.append(MSE_dict[clusters_chosen])
    MSE_array = np.array(MSE_list)

    difference_MSE_list = []
    for idx in range(len(MSE_array)):
        if idx < (len(MSE_array) - 1):
            difference = np.abs(MSE_array[idx] - MSE_array[idx + 1])
            difference_MSE_list.append(difference)

    #     MSE_dict_umap = new_cell_SST_model_ranks20_kmeans_reassign_UMAP_x00[20][animal][cell][1][f"cell_{cell}"]['MSE_dict']
    #     model_umap = new_cell_SST_model_ranks20_kmeans_reassign_UMAP_x00[20][animal][cell][0]

    #     w1_umap = model_umap.vectors[0][0].detach().numpy()
    #     X_umap = np.abs(w1_umap.T)

    #     silhouette_scores_umap = []
    #     cluster_range = range(2, 9)  # silhouette score needs at least 2 clusters

    #     for k in cluster_range:
    #         reducer = umap.UMAP(random_state=42)
    #         X_umap = reducer.fit_transform(X)
    #         kmeans_umap = KMeans(n_clusters=k, random_state=0, n_init=10)
    #         labels_umap = kmeans_umap.fit_predict(X_umap)

    #         if len(np.unique(labels_umap)) > 1:  # at least 2 clusters
    #             score_umap = silhouette_score(X_umap, labels_umap)
    #             silhouette_scores_umap.append(score_umap)
    #         else:
    #             silhouette_scores_umap.append(np.nan)  # if somehow all same cluster

    #     max_silhouette_umap = np.argmax(silhouette_scores_umap[1:-1])+3

    #     MSE_list_umap = []
    #     for clusters_chosen in MSE_dict_umap:
    #         MSE_list_umap.append(MSE_dict_umap[clusters_chosen])
    #     MSE_array_umap = np.array(MSE_list_umap)

    #     difference_MSE_list_umap = []
    #     for idx in range(len(MSE_array_umap)):
    #         if idx < (len(MSE_array_umap)-1):
    #             difference_umap = np.abs(MSE_array_umap[idx] - MSE_array_umap[idx+1])
    #             difference_MSE_list_umap.append(difference_umap)

    MSE_array_min_max = (MSE_array - np.min(MSE_array)) / (np.max(MSE_array) - np.min(MSE_array))
    #     MSE_array_min_max_umap = (MSE_array_umap - np.min(MSE_array_umap)) / (np.max(MSE_array_umap) - np.min(MSE_array_umap))

    difference_MSE_array = np.array(difference_MSE_list)
    #     difference_MSE_array_umap = np.array(difference_MSE_list_umap)

    difference_MSE_array_min_max = (difference_MSE_array - np.min(difference_MSE_array)) / (np.max(difference_MSE_array) - np.min(difference_MSE_array))
    #     difference_MSE_array_umap_min_max = (difference_MSE_array_umap - np.min(difference_MSE_array_umap)) / (np.max(difference_MSE_array_umap) - np.min(difference_MSE_array_umap))

    elbow_kmeans = find_elbow_point(MSE_array_min_max)
    #     elbow_umap = find_elbow_point(MSE_array_min_max_umap)

    # return difference_MSE_array_umap_min_max, elbow_umap, silhouette_scores_umap, max_silhouette_umap

    return MSE_array_min_max, difference_MSE_array_min_max, elbow_kmeans, cluster_range, silhouette_scores, max_silhouette_reg


def plot_kmeans_clusters_silhouette(MSE_array_min_max, difference_MSE_array_min_max, elbow_kmeans, cluster_range, silhouette_scores, max_silhouette_reg):
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].plot(MSE_array_min_max, color='b', label="K-Means")
    #     axs[0].plot(MSE_array_min_max_umap, color='r', label="UMAP then K-Means")
    axs[0].axvline(x=elbow_kmeans, color='b', linestyle='--', label=f"KMeans Elbow = {elbow_kmeans + 1}")
    #     axs[0].axvline(x=elbow_umap, color='r', linestyle='--', label=f"UMAP Elbow = {elbow_umap+1}")
    axs[0].set_xticks(np.arange(7), np.arange(1, 8))
    axs[0].set_ylabel("Reconstruction MSE")
    axs[0].set_xlabel("Number K-Means Clusters")
    axs[0].legend()

    axs[1].plot(difference_MSE_array_min_max, color='b', label="K-Means")
    #     axs[1].plot(difference_MSE_array_umap_min_max, color='r', label="UMAP then K-Means")
    axs[1].set_xticks(np.arange(7), np.arange(1, 8))
    axs[1].set_ylabel("delta MSE")
    axs[1].legend()
    axs[1].set_xlabel("Number K-Means Clusters")

    axs[2].plot(cluster_range, silhouette_scores, marker='o', color='b', label="K=Means")
    #     axs[2].plot(cluster_range, silhouette_scores_umap, marker='o', color='r', label="UMAP then K=Means")
    axs[2].set_xlabel("Number of Clusters")
    axs[2].set_ylabel("Silhouette Score")
    axs[2].set_title(f"Silhouette Max={max_silhouette_reg}")
    axs[2].set_xticks(cluster_range)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(data, title='Dominance Across Cell Types', ylabel='Dominance Score'):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    groups = ['SST', 'EC', 'NDNF']

    means = [np.mean(d) for d in data]
    sems = [np.std(d) / np.sqrt(len(d)) for d in data]

    x_pos = np.arange(len(groups))

    plt.figure(figsize=(8, 6))

    # Bar plot of the means
    plt.bar(x_pos, means, yerr=sems, capsize=5, alpha=0.6, color=['blue', 'green', 'orange'])

    # Scatter individual points
    for i, d in enumerate(data):
        x_jitter = np.random.normal(loc=0, scale=0.05, size=len(d))  # add little horizontal jitter
        plt.scatter(np.full_like(d, x_pos[i]) + x_jitter, d, color='black', s=30, alpha=0.7)

    plt.xticks(x_pos, groups)
    plt.ylabel(ylabel)
    plt.title(title)

    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Run Tukey HSD
    tukey = pairwise_tukeyhsd(endog=data[0] + data[1] + data[2],
                              groups=['SST'] * len(data[0]) + ['EC'] * len(data[1]) + ['NDNF'] * len(data[2]),
                              alpha=0.05)

    print(tukey.summary())

    comparisons = {}

    for result in tukey._results_table.data[1:]:
        group1, group2, meandiff, p_adj, lower, upper, reject = result
        comparisons[(group1, group2)] = {'reject': reject, 'p_adj': p_adj}  # Only one direction

    y_max = max([max(d) for d in data])

    def plot_star(start_idx, end_idx, y, height_ratio=0.02, significance='*'):
        x1, x2 = x_pos[start_idx], x_pos[end_idx]
        height = height_ratio * y_max
        plt.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, color='black')
        plt.text((x1 + x2) / 2, y + height + 0.01 * y_max, significance, ha='center', va='bottom', color='black', fontsize=12)

    start_height = y_max * 1.05
    height_increment = y_max * 0.08
    current_height = start_height

    # Helper function to decide significance label
    def get_significance_label(p_val):
        if p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        elif p_val < 0.1:
            return f'p={p_val:.2f}'
        else:
            return None

    def get_comparison_result(comparisons, group1, group2):
        """ Helper to get comparison result regardless of order """
        if (group1, group2) in comparisons:
            return comparisons[(group1, group2)]
        elif (group2, group1) in comparisons:
            return comparisons[(group2, group1)]
        else:
            return {'reject': None, 'p_adj': None}

    for (group1, group2) in [('SST', 'EC'), ('SST', 'NDNF'), ('EC', 'NDNF')]:  # only the 3 you care about
        result = get_comparison_result(comparisons, group1, group2)
        if result['p_adj'] is not None:
            significance = get_significance_label(result['p_adj'])
            if significance:
                idx1 = groups.index(group1)
                idx2 = groups.index(group2)
                plot_star(idx1, idx2, current_height, significance=significance)
                current_height += height_increment

    plt.tight_layout()
    plt.show()


def get_MSE_contig_eml(per_animal_diff_score_SST, use_animal=False):
    animal_ealy_mid_list = []
    animal_ealy_late_list = []
    animal_late_mid_list = []
    animal_overall_list = []

    for animal in per_animal_diff_score_SST:

        ealy_mid_list = []
        ealy_late_list = []
        late_mid_list = []
        overall_list = []

        for cell in per_animal_diff_score_SST[animal]:
            ealy_mid = per_animal_diff_score_SST[animal][cell]['MSE_E_M']
            if use_animal:
                ealy_mid_list.append(ealy_mid)
            else:
                animal_ealy_mid_list.append(ealy_mid)
            ealy_late = per_animal_diff_score_SST[animal][cell]['MSE_E_L']
            if use_animal:
                ealy_late_list.append(ealy_late)
            else:
                animal_ealy_late_list.append(ealy_late)
            late_mid = per_animal_diff_score_SST[animal][cell]['MSE_L_M']
            if use_animal:
                late_mid_list.append(late_mid)
            else:
                animal_late_mid_list.append(late_mid)
            overall = ealy_mid + ealy_late + late_mid
            if use_animal:
                overall_list.append(overall)
            else:
                animal_overall_list.append(overall)
        if use_animal:
            ealy_mid_list_av = np.mean(ealy_mid_list)
            animal_ealy_mid_list.append(ealy_mid_list_av)
            ealy_late_list_av = np.mean(ealy_late_list)
            animal_ealy_late_list.append(ealy_late_list_av)
            late_mid_list_av = np.mean(late_mid_list)
            animal_late_mid_list.append(late_mid_list_av)
            overall_av = np.mean(overall)
            animal_overall_list.append(overall_av)

    return animal_ealy_mid_list, animal_ealy_late_list, animal_late_mid_list, animal_overall_list


def get_diff_score(internals_per_animal_dict_EC_animal_x00_contig, animal=False):
    per_animal_diff_score = {}
    if animal:
        for animal in internals_per_animal_dict_EC_animal_x00_contig:

            diff_dict = {}

            for cell in internals_per_animal_dict_EC_animal_x00_contig[animal]:
                mean_list = internals_per_animal_dict_EC_animal_x00_contig[animal][cell]["cluster_trial_mean_dict"]["clusters_chosen_3"]

                early = mean_list[0]
                middle = mean_list[1]
                late = mean_list[2]

                diff_dict[cell] = {}

                MSE_E_M = np.mean(np.square(early - middle))
                MSE_E_L = np.mean(np.square(early - late))
                MSE_L_M = np.mean(np.square(late - middle))

                diff_dict[cell]["MSE_E_M"] = MSE_E_M
                diff_dict[cell]["MSE_E_L"] = MSE_E_L
                diff_dict[cell]["MSE_L_M"] = MSE_L_M

            per_animal_diff_score[animal] = diff_dict
    else:
        for animal in internals_per_animal_dict_EC_animal_x00_contig[20]:
            diff_dict = {}

            for cell in internals_per_animal_dict_EC_animal_x00_contig[20][animal]:
                mean_list = internals_per_animal_dict_EC_animal_x00_contig[20][animal][cell][1][f"cell_{cell}"]["cluster_trial_mean_dict"]["clusters_chosen_3"]

                early = mean_list[0]
                middle = mean_list[1]
                late = mean_list[2]

                diff_dict[cell] = {}

                MSE_E_M = np.mean(np.square(early - middle))
                MSE_E_L = np.mean(np.square(early - late))
                MSE_L_M = np.mean(np.square(late - middle))

                diff_dict[cell]["MSE_E_M"] = MSE_E_M
                diff_dict[cell]["MSE_E_L"] = MSE_E_L
                diff_dict[cell]["MSE_L_M"] = MSE_L_M

            per_animal_diff_score[animal] = diff_dict

    return per_animal_diff_score


def get_counts_of_first_kmeans_failure(model_dict, rank=20, cluster_range=range(1, 9)):
    cluster_of_first_singletons_dict = {}
    cluster_count_dict = {k: 0 for k in cluster_range}  # initialize 1–8 with 0s

    if rank not in model_dict:
        return cluster_of_first_singletons_dict, cluster_count_dict

    for animal in model_dict[rank]:
        for cell in model_dict[rank][animal]:
            labels_dict = model_dict[rank][animal][cell][1][f"cell_{cell}"]['labels_dict']
            found_singleton = False

            for idx, clusters_chosen in enumerate(labels_dict):
                labels = labels_dict[clusters_chosen]
                unique_vals, counts = np.unique(labels, return_counts=True)
                singleton_labels = unique_vals[counts == 1]

                if len(singleton_labels) > 0 and not found_singleton:
                    cluster_number = int(clusters_chosen.split('_')[-1])
                    print(f"Warning: {len(singleton_labels)} singleton(s) found in {clusters_chosen} for {animal}, cell {cell}")

                    if animal not in cluster_of_first_singletons_dict:
                        cluster_of_first_singletons_dict[animal] = {}
                    cluster_of_first_singletons_dict[animal][cell] = cluster_number

                    if cluster_number in cluster_count_dict:
                        cluster_count_dict[cluster_number] += 1

                    found_singleton = True
                    break

    print(f"✅ Found first singleton cluster for {sum(len(c) for c in cluster_of_first_singletons_dict.values())} cells")
    return cluster_of_first_singletons_dict, cluster_count_dict


def plot_histogram_from_counts(cluster_count_list):
    cluster_count_dict_SST = cluster_count_list[0]
    cluster_count_dict_NDNF = cluster_count_list[1]
    cluster_count_dict_EC = cluster_count_list[2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    plt.suptitle("Number of Clusters with 2 or Less Members")

    cluster_numbers_SST = sorted(cluster_count_dict_SST.keys())
    counts_SST = [cluster_count_dict_SST[k] for k in cluster_numbers_SST]

    axs[0].bar(cluster_numbers_SST, counts_SST, align='center', width=0.6, color='b')
    axs[0].set_xlabel('Cluster number where first singleton appeared')
    axs[0].set_ylabel('Number of cells')
    axs[0].set_title("SST Cell SliceTCA x00")
    axs[0].set_xticks(cluster_numbers_SST)

    cluster_numbers_NDNF = sorted(cluster_count_dict_NDNF.keys())
    counts_NDNF = [cluster_count_dict_NDNF[k] for k in cluster_numbers_NDNF]

    axs[1].bar(cluster_numbers_NDNF, counts_NDNF, align='center', width=0.6, color='orange')
    axs[1].set_xlabel('Cluster number where first singleton appeared')
    axs[1].set_ylabel('Number of cells')
    axs[1].set_title("NDNF Cell SliceTCA x00")
    axs[1].set_xticks(cluster_numbers_NDNF)

    cluster_numbers_EC = sorted(cluster_count_dict_EC.keys())
    counts_EC = [cluster_count_dict_EC[k] for k in cluster_numbers_EC]

    axs[2].bar(cluster_numbers_EC, counts_EC, align='center', width=0.6, color='g')
    axs[2].set_xlabel('Cluster number where first singleton appeared')
    axs[2].set_ylabel('Number of cells')
    axs[2].set_title("EC Cell SliceTCA x00")
    axs[2].set_xticks(cluster_numbers_EC)

    plt.tight_layout()

    plt.show()


def load_data_MSEs():
    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_SST_latent_2_22_animal{i}"
    SST_mean_2_22, SST_sem_2_22 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_SST_latent_22_42_animal{i}"
    SST_mean_22_42, SST_sem_22_42 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_SST_latent_42_62_animal{i}"
    sst_mean_42_64, sst_sem_42_64 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=10)


    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "NDNF_latent_loss_2_22_animal_{i}"
    NDNF_mean_2_22, NDNF_sem_2_22 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=4)

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_NDNF_latent_22_42_animal{i}"
    NDNF_mean_22_42, NDNF_sem_22_42 = get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=4)

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    file_name_template = "loss_dict_NDNF_latent_42_62_animal{i}"
    animal_list = [0, 2, 3]
    NDNF_mean_42_62, NDNF_sem_42_62 = get_data_by_animal_num(base_dir, file_name_template, animal_list=animal_list, num_animals=None)



    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_experiment"
    EC_latent_means_2_22, EC_latent_sems_2_22 = cell_type_get_mean_sem(base_dir, start_num=2, end_num=22, cell_type="EC")

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"
    EC_latent_means_22_42, EC_latent_sems_22_42 = cell_type_get_mean_sem(base_dir, start_num=22, end_num=43, cell_type="EC")

    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_experiment"
    EC_latent_means_42_62, EC_latent_sems_42_62 = cell_type_get_mean_sem(base_dir, start_num=43, end_num=63, cell_type="EC")


    concatenated_EC_means = np.concatenate([EC_latent_means_2_22, EC_latent_means_22_42, EC_latent_means_42_62])
    concatenated_EC_sem = np.concatenate([EC_latent_sems_2_22, EC_latent_sems_22_42, EC_latent_sems_42_62])

    concatenated_SST_means = np.concatenate([SST_mean_2_22, SST_mean_22_42[:-1], sst_mean_42_64[:-1]])
    concatenated_SST_sem = np.concatenate([SST_sem_2_22, SST_sem_22_42[:-1], sst_sem_42_64[:-1]])

    concatenated_NDNF_means = np.concatenate([NDNF_mean_2_22, NDNF_mean_22_42[:-1], NDNF_mean_42_62[:-1]])
    concatenated_NDNF_sem = np.concatenate([NDNF_sem_2_22, NDNF_sem_22_42[:-1], NDNF_sem_42_62[:-1]])



    ##########################################




    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_timebins_experiment"

    # Holds lists of scalar losses for each latent number
    latent_numbers = []

    for i in range(2, 63):
        # Get all matching files
        pattern = os.path.join(base_dir, f"timebins_loss_dict_EC_latent_{i}_{i}_animal*.pkl")
        file_paths = sorted(glob.glob(pattern))

        # Store float values from each file
        loss_floats = []

        for path in file_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
                data = float(data)
                loss_floats.append(data)

        latent_numbers.append(loss_floats)

    EC_latent_means_timebins_62 = []
    EC_latent_sems_timebins_62 = []

    for i in latent_numbers:
        EC_latent_means_timebins_62.append(np.mean(i))
        EC_latent_sems_timebins_62.append(sem(i))

    EC_latent_sems_timebins_62 = np.array(EC_latent_sems_timebins_62)








    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_timebins_experiment"

    # Holds lists of scalar losses for each latent number
    latent_numbers = []

    for i in range(2, 63):
        # Get all matching files
        pattern = os.path.join(base_dir, f"timebins_loss_dict_NDNF_latent_{i}_{i}_animal*.pkl")
        file_paths = sorted(glob.glob(pattern))

        # Store float values from each file
        loss_floats = []

        for path in file_paths:
            with open(path, "rb") as f:
                data = pickle.load(f)
                data = float(data)
                loss_floats.append(data)

        latent_numbers.append(loss_floats)

    NDNF_latent_means_timebins_62 = []
    NDNF_latent_sems_timebins_62 = []

    for i in latent_numbers:
        NDNF_latent_means_timebins_62.append(np.mean(i))
        NDNF_latent_sems_timebins_62.append(sem(i))
    NDNF_latent_sems_timebins_62 = np.array(NDNF_latent_sems_timebins_62)







    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"

    SST_list_2_24_timebins = []


    for i in range(10):  # Assuming animals 0 to

        file_name = f"timebins_loss_dict_SST_latent_2_22_animal{i}.pkl"
        file_path = os.path.join(base_dir, file_name)

        # Load the pickle file
        with open(file_path, "rb") as f:
            animal = pickle.load(f)

        SST_list_2_24_timebins.append(animal)

    SST_mean_2_24_timebins, SST_sem_2_24_timebins = get_mean_sem_loss2(SST_list_2_24_timebins)


    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"

    SST_list_22_42_timebins = []


    for i in range(10):  # Assuming animals 0 to

        file_name = f"timebins_loss_dict_SST_latent_22_42_animal{i}.pkl"
        file_path = os.path.join(base_dir, file_name)

        # Load the pickle file
        with open(file_path, "rb") as f:
            animal = pickle.load(f)

        SST_list_22_42_timebins.append(animal)

    SST_mean_22_42_timebins, SST_sem_22_42_timebins = get_mean_sem_loss2(SST_list_22_42_timebins)


    base_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets"

    SST_list_42_62_timebins = []


    for i in range(10):  # Assuming animals 0 to

        file_name = f"timebins_loss_dict_SST_latent_42_62_animal{i}.pkl"
        file_path = os.path.join(base_dir, file_name)

        # Load the pickle file
        with open(file_path, "rb") as f:
            animal = pickle.load(f)

        SST_list_42_62_timebins.append(animal)

    SST_mean_42_62_timebins, SST_sem_42_62_timebins = get_mean_sem_loss2(SST_list_42_62_timebins)

    SST_latent_mean_timebins_62 = np.concatenate([SST_mean_2_24_timebins, SST_mean_22_42_timebins, SST_mean_42_62_timebins])
    SST_latent_sem_timebins_62 = np.concatenate([SST_sem_2_24_timebins, SST_sem_22_42_timebins, SST_sem_42_62_timebins])


    return concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem, SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62


def plot_MSE_latents(concatenated_EC_means, concatenated_EC_sem, concatenated_SST_means, concatenated_SST_sem, concatenated_NDNF_means, concatenated_NDNF_sem, SST_latent_mean_timebins_62, SST_latent_sem_timebins_62, NDNF_latent_means_timebins_62, NDNF_latent_sems_timebins_62, EC_latent_means_timebins_62, EC_latent_sems_timebins_62):

    ##########################################

    fig, axs = plt.subplots(1,2, figsize=(20,10))

    axs[0].plot(concatenated_EC_means, label="EC", color='g')
    axs[0].fill_between(range(len(concatenated_EC_means)),
                     concatenated_EC_means + concatenated_EC_sem,
                     concatenated_EC_means - concatenated_EC_sem,
                     alpha=0.3, color='g')
    axs[0].plot(concatenated_SST_means, label="SST", color='b')
    axs[0].fill_between(range(len(concatenated_SST_means)),
                     concatenated_SST_means + concatenated_SST_sem,
                     concatenated_SST_means - concatenated_SST_sem,
                     alpha=0.3, color='b')

    axs[0].plot(concatenated_NDNF_means, label="NDNF", color='orange')
    axs[0].fill_between(range(len(concatenated_NDNF_means)),
                     concatenated_NDNF_means + concatenated_NDNF_sem,
                     concatenated_NDNF_means - concatenated_NDNF_sem,
                     alpha=0.3, color='orange')
    axs[0].legend()
    axs[0].set_ylabel("MSE")
    axs[0].set_ylim(0,1)
    axs[0].set_ylabel("MSE")
    axs[0].set_title("Slice TCA per Animal (x,0,0)")
    axs[0].set_xlabel("Number of Latents")


    axs[1].plot(SST_latent_mean_timebins_62, color='b', label='SST')
    axs[1].fill_between(range(len(SST_latent_mean_timebins_62)),
                     SST_latent_mean_timebins_62 + SST_latent_sem_timebins_62,
                     SST_latent_mean_timebins_62 - SST_latent_sem_timebins_62,
                     alpha=0.3, color='b')
    axs[1].plot(NDNF_latent_means_timebins_62, color='orange', label='NDNF')
    axs[1].fill_between(range(len(NDNF_latent_means_timebins_62)),
                     NDNF_latent_means_timebins_62 + NDNF_latent_sems_timebins_62,
                     NDNF_latent_means_timebins_62 - NDNF_latent_sems_timebins_62,
                     alpha=0.3, color='orange')
    axs[1].plot(EC_latent_means_timebins_62, color='g', label='EC')
    axs[1].fill_between(range(len(EC_latent_means_timebins_62)),
                     EC_latent_means_timebins_62 + EC_latent_sems_timebins_62,
                     EC_latent_means_timebins_62 - EC_latent_sems_timebins_62,
                     alpha=0.3, color='g')
    axs[1].legend()
    axs[1].set_ylabel("MSE")
    axs[1].set_title("Slice TCA per Animal (0,0,x)")
    axs[1].set_ylim(0,1)
    axs[1].set_xlabel("Number of Latents")

    plt.show()


def get_data_by_animal_num(base_dir, file_name_template, animal_list=None, num_animals=None):
    SST_list_2_22 = []

    if animal_list is None:
        animal_list = range(num_animals)

    for i in animal_list:
        file_name = file_name_template.format(i=i)
        file_path = os.path.join(base_dir, file_name)

        with open(file_path, "rb") as f:
            animal = pickle.load(f)

        SST_list_2_22.append(animal)

    SST_mean_2_22, SST_sem_2_22 = get_mean_sem_loss2(SST_list_2_22)
    return SST_mean_2_22, SST_sem_2_22


def show_contiguous_internals(cell_models, animal_models, activity_dicts, r2_variable_activity_dict_new_fixed_NDNF, r2_variable_activity_dict_old_NDNF, r2_variable_activity_dict_SST, r2_variable_activity_dict_EC, cell_type=True, seperate_fixed_random=True, random_reward=False, combine_new_old=True):
    if cell_type == "NDNF_new":
        model = cell_models[0]

        if seperate_fixed_random:

            models_blank = {20: {}}
            models_fixed = {20: {}}

            for idx in model[20]:
                if idx < 9:
                    models_blank[20][idx] = model[20][idx]
                else:
                    models_fixed[20][idx] = model[20][idx]

            if random_reward:

                new_activity = activity_dicts[0]
                newer_activity = {}
                for idx, animal in enumerate(new_activity):
                    if idx < 9:
                        newer_activity[animal] = new_activity[animal]

                mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(models_blank, newer_activity, animal_TCA=False)
                plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="NDNF", title="NDNF New Random Reward Cell SliceTCA Contiguous")
                cell_first_changepoints_list_new_NDNF, fraction_cell_first_changepoints_list_new_NDNF, cell_second_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF = get_changepoints(models_blank, newer_activity)
                plot_changepoint_distribution(fraction_cell_first_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF, cell_type='New NDNF Random Reward')
                plot_individual_learning_traces(raw_cluster_list, sup="NDNF New Random Reward")


            else:
                if combine_new_old:
                    new_model = models_fixed  # Start with NDNF_new
                    original = cell_models[1]  # This is NDNF_original

                    for animal in original[20]:
                        new_model[20][f"old_{animal}"] = original[20][animal]

                    new_activity = activity_dicts[0]
                    original_activity = activity_dicts[1]

                    newer_activity = {}
                    for idx, animal in enumerate(new_activity):
                        if idx > 8:
                            newer_activity[animal] = new_activity[animal]

                    for animal in original_activity:
                        newer_activity[f"old_{animal}"] = original_activity[animal]

                    print(f"newer_activity.keys() {newer_activity.keys()}")
                    print(f"new_model[20].keys() {new_model[20].keys()}")

                    mean_list_EC_cell_x00, sem_list_EC_cell_x00, raw_cluster_list = get_contig_cluster_av(new_model, newer_activity, animal_TCA=False)
                    plot_average_activity_contig_kmeans(mean_list_EC_cell_x00, sem_list_EC_cell_x00, cell_type="NDNF", title="NDNF Combined New+Original Fixed Reward Cell SliceTCA Contiguous")
                    cell_first_changepoints_list_EC, fraction_cell_first_changepoints_list_EC, cell_second_changepoints_list_EC, fraction_cell_second_changepoints_list_EC = get_changepoints(new_model, newer_activity)
                    plot_changepoint_distribution(fraction_cell_first_changepoints_list_EC, fraction_cell_second_changepoints_list_EC, cell_type="NDNF Fixed Reward New+Original")
                    plot_individual_learning_traces(raw_cluster_list, sup="NDNF Fixed New+Original")


                else:
                    new_activity = activity_dicts[0]
                    newer_activity = {}
                    for idx, animal in enumerate(new_activity):
                        if idx > 8:
                            newer_activity[animal] = new_activity[animal]

                    mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(models_fixed, newer_activity, animal_TCA=False)
                    plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="NDNF", title="NDNF New Fixed Cell SliceTCA Contiguous")
                    cell_first_changepoints_list_new_NDNF, fraction_cell_first_changepoints_list_new_NDNF, cell_second_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF = get_changepoints(models_fixed, newer_activity)
                    plot_changepoint_distribution(fraction_cell_first_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF, cell_type='New NDNF Fixed Reward')
                    plot_individual_learning_traces(raw_cluster_list, sup="NDNF Fixed New")

        else:
            mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(models)
            plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="NDNF", title="NDNF New Cell SliceTCA Contiguous")
            cell_first_changepoints_list_new_NDNF, fraction_cell_first_changepoints_list_new_NDNF, cell_second_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF = get_changepoints(model, activity_dicts[0])
            plot_changepoint_distribution(fraction_cell_first_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF, cell_type='New NDNF')
            plot_individual_learning_traces(raw_cluster_list, sup="NDNF Fixed New+Original")

    elif cell_type == "NDNF_above_zero":

        model = cell_models[0]
        old_model = cell_models[1]

        #         for idx, animal in enumerate(models[20]):
        #             if idx>8:
        #                 model[20][animalal] = models[20][animal]

        models_blank = {20: {}}
        models_fixed = {20: {}}

        for idx in model[20]:
            if idx < 9:
                models_blank[20][idx] = model[20][idx]
            else:
                models_fixed[20][idx] = model[20][idx]

        combined_r_value_dict_fixed = {}

        for animal in r2_variable_activity_dict_new_fixed_NDNF:
            animal_num = int(animal.split('_')[-1])  # e.g., 10 from 'animal_10'
            if animal_num > 8:
                combined_r_value_dict_fixed[animal] = r2_variable_activity_dict_new_fixed_NDNF[animal]

        combined_activity_dict = {}
        for idx, animal in enumerate(residual_activity_dict_NDNF_new):
            if idx > 8:
                combined_activity_dict[animal] = residual_activity_dict_NDNF_new[animal]

        above_zero_indices = []
        below_zero_indices = []

        above_zero_indices_numerical_list = []
        below_zero_indices_numerical_list = []

        for idx, animal in enumerate(combined_r_value_dict_fixed):
            for ids, cell in enumerate(combined_r_value_dict_fixed[animal]):
                indexing_list = [animal, cell]
                indices = [idx + 9, ids]
                if combined_r_value_dict_fixed[animal][cell] > 0:
                    above_zero_indices.append(indexing_list)
                    above_zero_indices_numerical_list.append(indices)
                else:
                    below_zero_indices.append(indexing_list)
                    below_zero_indices_numerical_list.append(indices)

        above_zero_activity_dict = {}
        for animal_index, cell_index in above_zero_indices:
            above_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        below_zero_activity_dict = {}
        for animal_index, cell_index in below_zero_indices:
            below_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        above_zero_models_dict = {20: {}}
        for animal_index, cell_index in above_zero_indices_numerical_list:
            above_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]

        below_zero_models_dict = {20: {}}
        for animal_index, cell_index in below_zero_indices_numerical_list:
            if (animal_index in model[20] and
                    cell_index in model[20][animal_index]):
                below_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]
            else:
                continue

        print(f"models_fixed[20].keys() {models_fixed[20].keys()}")
        print(f"combined_activity_dict {combined_activity_dict.keys()}")
        print(f"combined_r_value_dict_fixed.keys() {combined_r_value_dict_fixed.keys()}")

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(above_zero_models_dict, above_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="NDNF", title="NDNF Above Zero Velocity Correlation")

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(below_zero_models_dict, below_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="NDNF", title="NDNF Below Zero Velocity Correlation")
        #

        cell_first_changepoints_list_new_NDNF, fraction_cell_first_changepoints_list_new_NDNF, cell_second_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF = get_changepoints(models_fixed, combined_activity_dict)
        plot_changepoint_distribution(fraction_cell_first_changepoints_list_new_NDNF, fraction_cell_second_changepoints_list_new_NDNF, cell_type='NDNF Above Zero Velocity Correlation')
        plot_individual_learning_traces(raw_cluster_list, sup="NDNF Above Zero Velocity Correlation")



    elif cell_type == "NDNF_original":
        model = cell_models[1]
        mean_list_NDNF_old_cell_x00, sem_list_NDNF_old_cell_x00, raw_cluster_list = get_contig_cluster_av(model, activity_dicts[1], animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_old_cell_x00, sem_list_NDNF_old_cell_x00, cell_type="NDNF", title="NDNF Original Cell SliceTCA Contiguous")
        cell_first_changepoints_list_old_NDNF, fraction_cell_first_changepoints_list_old_NDNF, cell_second_changepoints_list_old_NDNF, fraction_cell_second_changepoints_list_old_NDNF = get_changepoints(model, activity_dicts[1])
        plot_changepoint_distribution(fraction_cell_first_changepoints_list_old_NDNF, fraction_cell_second_changepoints_list_old_NDNF, cell_type="Original NDNF")
        plot_individual_learning_traces(raw_cluster_list, sup="NDNF Fixed Original")

    elif cell_type == "SST":
        model = cell_models[2]
        mean_list_SST_cell_x00, sem_list_SST_cell_x00, raw_cluster_list = get_contig_cluster_av(model, activity_dicts[2], animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_SST_cell_x00, sem_list_SST_cell_x00, cell_type="SST", title="SST Cell SliceTCA Contiguous")
        cell_first_changepoints_list_SST, fraction_cell_first_changepoints_list_SST, cell_second_changepoints_list_SST, fraction_cell_second_changepoints_list_SST = get_changepoints(model, activity_dicts[2])
        plot_changepoint_distribution(fraction_cell_first_changepoints_list_SST, fraction_cell_second_changepoints_list_SST, cell_type="SST")
        plot_individual_learning_traces(raw_cluster_list, sup="SST Fixed Original")

    elif cell_type == "SST_above_zero":

        model = cell_models[2]
        #         old_model = cell_models[1]

        #         for animal in old_model[20]:
        #             model[20][animal+18] = old_model[20][animal]

        #         for animal in old_model[20]:
        #             model[20][animal + 18] = old_model[20][animal]

        #         print("After adding old model animals:")
        #         print(sorted(model[20].keys()))

        #         models_blank = {20: {}}
        #         models_fixed = {20: {}}

        #         for idx in model[20]:
        #             if idx < 9:
        #                 models_blank[20][idx] = model[20][idx]
        #             else:
        #                 models_fixed[20][idx] = model[20][idx]

        combined_r_value_dict_fixed = r2_variable_activity_dict_SST

        combined_activity_dict = activity_dicts[2]

        above_zero_indices = []
        below_zero_indices = []

        above_zero_indices_numerical_list = []
        below_zero_indices_numerical_list = []

        for idx, animal in enumerate(combined_r_value_dict_fixed):
            for ids, cell in enumerate(combined_r_value_dict_fixed[animal]):
                indexing_list = [animal, cell]
                indices = [idx, ids]
                if combined_r_value_dict_fixed[animal][cell] > 0:
                    above_zero_indices.append(indexing_list)
                    above_zero_indices_numerical_list.append(indices)
                else:
                    below_zero_indices.append(indexing_list)
                    below_zero_indices_numerical_list.append(indices)

        above_zero_activity_dict = {}
        for animal_index, cell_index in above_zero_indices:
            above_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        below_zero_activity_dict = {}
        for animal_index, cell_index in below_zero_indices:
            below_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        above_zero_models_dict = {20: {}}
        for animal_index, cell_index in above_zero_indices_numerical_list:
            above_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]

        below_zero_models_dict = {20: {}}
        for animal_index, cell_index in below_zero_indices_numerical_list:
            if (animal_index in model[20] and
                    cell_index in model[20][animal_index]):
                below_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]
            else:
                continue

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(above_zero_models_dict, above_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="SST", title="SST Above Zero Velocity Correlation")

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(below_zero_models_dict, below_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="SST", title="SST Below Zero Velocity Correlation")
    #

    elif cell_type == "EC_above_zero":

        model = cell_models[3]

        combined_r_value_dict_fixed = r2_variable_activity_dict_EC

        combined_activity_dict = activity_dicts[3]

        above_zero_indices = []
        below_zero_indices = []

        above_zero_indices_numerical_list = []
        below_zero_indices_numerical_list = []

        for idx, animal in enumerate(combined_r_value_dict_fixed):
            for ids, cell in enumerate(combined_r_value_dict_fixed[animal]):
                indexing_list = [animal, cell]
                indices = [idx, ids]
                if combined_r_value_dict_fixed[animal][cell] > 0:
                    above_zero_indices.append(indexing_list)
                    above_zero_indices_numerical_list.append(indices)
                else:
                    below_zero_indices.append(indexing_list)
                    below_zero_indices_numerical_list.append(indices)

        above_zero_activity_dict = {}
        for animal_index, cell_index in above_zero_indices:
            above_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        below_zero_activity_dict = {}
        for animal_index, cell_index in below_zero_indices:
            below_zero_activity_dict.setdefault(animal_index, {})[cell_index] = combined_activity_dict[animal_index][cell_index]

        above_zero_models_dict = {20: {}}
        for animal_index, cell_index in above_zero_indices_numerical_list:
            above_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]

        below_zero_models_dict = {20: {}}
        for animal_index, cell_index in below_zero_indices_numerical_list:
            if (animal_index in model[20] and
                    cell_index in model[20][animal_index]):
                below_zero_models_dict[20].setdefault(animal_index, {})[cell_index] = model[20][animal_index][cell_index]
            else:
                continue

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(above_zero_models_dict, above_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="EC", title="EC Above Zero Velocity Correlation")

        mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, raw_cluster_list = get_contig_cluster_av(below_zero_models_dict, below_zero_activity_dict, animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_NDNF_new_cell_x00, sem_list_NDNF_new_cell_x00, cell_type="EC", title="EC Below Zero Velocity Correlation")
    #

    elif cell_type == "EC":
        model = cell_models[3]
        mean_list_EC_cell_x00, sem_list_EC_cell_x00, raw_cluster_list = get_contig_cluster_av(model, activity_dicts[3], animal_TCA=False)
        plot_average_activity_contig_kmeans(mean_list_EC_cell_x00, sem_list_EC_cell_x00, cell_type="EC", title="EC Cell SliceTCA Contiguous")
        cell_first_changepoints_list_EC, fraction_cell_first_changepoints_list_EC, cell_second_changepoints_list_EC, fraction_cell_second_changepoints_list_EC = get_changepoints(model, activity_dicts[3])
        plot_changepoint_distribution(fraction_cell_first_changepoints_list_EC, fraction_cell_second_changepoints_list_EC, cell_type="EC")
        plot_individual_learning_traces(raw_cluster_list, sup="EC Fixed Original")

    elif cell_type == "NDNF_total":
        new_model = copy.deepcopy(cell_models[0])  # Start with NDNF_new
        original = cell_models[1]  # This is NDNF_original

        for animal in original[20]:
            new_model[20][f"new_{animal}"] = original[20][animal]

        new_activity = activity_dicts[0]
        original_activity = activity_dicts[1]

        for animal in original_activity:
            new_activity[f"new_{animal}"] = original_activity[animal]

        mean_list_EC_cell_x00, sem_list_EC_cell_x00 = get_contig_cluster_av(new_model)
        plot_average_activity_contig_kmeans(mean_list_EC_cell_x00, sem_list_EC_cell_x00, cell_type="EC", title="EC Cell SliceTCA")
        cell_first_changepoints_list_EC, fraction_cell_first_changepoints_list_EC, cell_second_changepoints_list_EC, fraction_cell_second_changepoints_list_EC = get_changepoints(new_model, new_activity)
        plot_changepoint_distribution(fraction_cell_first_changepoints_list_EC, fraction_cell_second_changepoints_list_EC, cell_type="NDNF")


def get_correlations_per_cell(belt_A_list, belt_B_list, belt_A_residuals_list, belt_B_residuals_list, filtered_factors_dict):
    correlation_vs_velocity_dict = {}
    correlation_vs_velocity_dict_residuals = {}

    for i in range(len(belt_A_list)):
        animal_A = belt_A_list[i]
        animal_B = belt_B_list[i]
        animal_A_residuals = belt_A_residuals_list[i]
        animal_B_residuals = belt_B_residuals_list[i]

        correlation_per_cell_dict = {}
        correlation_per_cell_dict_residuals = {}
        correlation_per_cell_list = []
        correlation_per_cell_list_residuals = []

        for cell in animal_A:
            cell_A_activity = animal_A[cell].flatten()
            cell_A_residual_activity = animal_A_residuals[cell].flatten()
            cell_B_activity = animal_B[cell].flatten()
            cell_B_residual_activity = animal_B_residuals[cell].flatten()

            velocity_early = filtered_factors_dict[f"animal_{i + 1}"]["Velocity"].flatten()
            velocity_late = filtered_factors_dict[f"animal_{i + 10}"]["Velocity"].flatten()

            r_value_vs_vel_A_residuals, _ = pearsonr(cell_A_residual_activity, velocity_early)
            r_value_vs_vel_B_residuals, _ = pearsonr(cell_B_residual_activity, velocity_late)

            r_value_vs_vel_A, _ = pearsonr(cell_A_activity, velocity_early)
            r_value_vs_vel_B, _ = pearsonr(cell_B_activity, velocity_late)

            r_value_list = [r_value_vs_vel_A, r_value_vs_vel_B]
            r_value_list_residuals = [r_value_vs_vel_A_residuals, r_value_vs_vel_B_residuals]

            correlation_per_cell_dict_residuals[cell] = r_value_list_residuals
            correlation_per_cell_dict[cell] = r_value_list

        correlation_vs_velocity_dict[i] = correlation_per_cell_dict
        correlation_vs_velocity_dict_residuals[i] = correlation_per_cell_dict_residuals

    return correlation_vs_velocity_dict, correlation_vs_velocity_dict_residuals


def plot_correlations(correlation_vs_velocity_dict, correlation_vs_velocity_dict_residuals):
    # plt.plot(correlation_vs_velocity_dict[0]["cell_5"])
    length = len(correlation_vs_velocity_dict)
    fig, axs = plt.subplots(1, length, figsize=(30, 4))
    for i in range(length):
        for cell in correlation_vs_velocity_dict[i]:
            axs[i].plot(correlation_vs_velocity_dict[i][cell])
            axs[i].set_ylim(-1, 1)
            axs[i].set_title(f"Animal {i} Raw DF/F Cells")
            axs[i].set_xticks(np.arange(2), ["Random Reward", "Fixed Reward"])
            axs[i].set_ylabel("Correlation to Velocity")

    plt.tight_layout()
    plt.show()

    length = len(correlation_vs_velocity_dict_residuals)
    fig, axs = plt.subplots(1, length, figsize=(30, 4))
    for i in range(length):
        for cell in correlation_vs_velocity_dict_residuals[i]:
            axs[i].plot(correlation_vs_velocity_dict_residuals[i][cell])
            axs[i].set_ylim(-1, 1)
            axs[i].set_title(f"Animal {i} Vel. Sub. Residuals")
            axs[i].set_xticks(np.arange(2), ["Random Reward", "Fixed Reward"])
            axs[i].set_ylabel("Correlation to Velocity")

    plt.tight_layout()
    plt.show()


def plot_individual_learning_traces(raw_cluster_list, sup="NDNF Fixed New"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(sup)

    im = axs[0].imshow(raw_cluster_list[0], aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[0].set_title("Early Learn Trial Av")
    axs[0].set_ylabel("Cell ID")
    axs[0].set_xlabel("Position Bins")

    im2 = axs[1].imshow(raw_cluster_list[1], aspect='auto')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Late Learn Trial Av")
    axs[1].set_ylabel("Cell ID")
    axs[1].set_xlabel("Position Bins")

    plt.tight_layout()
    plt.show()


def example_EC_cell(velocity):
    length = 50
    num_trials = velocity.shape[1]
    x = np.linspace(0, length - 1, length)

    mean1 = 15
    std_dev1 = 5
    original_gaussian1 = norm.pdf(x, mean1, std_dev1)

    mean2 = 35
    std_dev2 = 5
    original_gaussian2 = norm.pdf(x, mean2, std_dev2)

    gaussian_list = []

    for i in range(num_trials):
        appear_1 = np.random.choice([0, 1])
        appear_2 = np.random.choice([0, 1])

        gaussian1 = original_gaussian1 * appear_1
        gaussian2 = original_gaussian2 * appear_2

        combined_gaussian = gaussian1 + gaussian2

        bimodal_gaussian = combined_gaussian / np.max(combined_gaussian) if np.max(combined_gaussian) != 0 else combined_gaussian

        gaussian_list.append(bimodal_gaussian)

    pf = np.stack(gaussian_list)

    return pf


def visualize_occupancy_weighting_problem(velocity_change=True, EC_bump=False, I_dip=False):
    track_length = 200
    total_time = 1000  # ms
    position_bins = 50
    threshold = 0  # baseline threshold

    # 1. Time allocation: double time in middle 10 bins
    time_per_bin = np.ones(position_bins)
    middle_track = position_bins // 2
    reward_bins = list(range(middle_track - 5, middle_track + 5))
    time_per_bin[reward_bins] = 2

    if velocity_change:

        # Normalize to total time
        time_per_bin = time_per_bin / np.sum(time_per_bin) * total_time
        time_per_bin = np.round(time_per_bin).astype(int)

        # Fix rounding error
        diff = total_time - np.sum(time_per_bin)
        time_per_bin[0] += diff

    else:
        time_per_bin = np.full(position_bins, total_time // position_bins)
        diff = total_time - np.sum(time_per_bin)  # Just in case there's a leftover ms
        time_per_bin[0] += diff  # Add it to the first bin to fix rounding

    # 2. Generate E and I over time
    E_rate_list = []
    I_rate_list = []

    for i in range(total_time):
        # Baseline EC rhythm
        if i % 10 == 0:
            E = 0.45
        else:
            E = 0.4

        # Dip inhibition during reward
        if I_dip and 400 < i < 600:
            I = 0.35
        else:
            I = 0.4

        E_rate_list.append(E)
        I_rate_list.append(I)

    # 3. Simulate plateau record based on time allocation
    plateau_record = []
    global_time = 0
    Vm_Record = []

    for bin_idx in range(position_bins):
        ms_in_bin = time_per_bin[bin_idx]

        for _ in range(ms_in_bin):
            E = E_rate_list[global_time]
            I = I_rate_list[global_time]
            Vm = E - I
            Vm_Record.append(Vm)
            plateau_record.append(1 if Vm > threshold else 0)
            global_time += 1

    # Get times (ms) where a plateau occurred
    plateau_times = np.where(np.array(plateau_record) == 1)[0]  # time indices where plateau happened

    # Bin into 20 equal-duration time bins from 0 to 1000 ms
    bins = np.linspace(0, total_time, 21)  # 20 bins = 21 edges

    plt.figure(figsize=(8, 3))
    plt.hist(plateau_times, bins=bins, color="purple", edgecolor="black")
    plt.title("Plateaus per Time Bin (Histogram of Plateau Times)")
    plt.xlabel("Time (ms)")
    plt.ylabel("# of Plateaus")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.plot(plateau_record[:200])
    plt.title("Plateau Record (First 200 ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Plateau (0/1)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Plot plateaus per bin
    plateau_per_bin = []
    start = 0
    for ms in time_per_bin:
        bin_plateaus = plateau_record[start:start + ms]
        plateau_per_bin.append(np.sum(bin_plateaus))
        start += ms

    plt.figure(figsize=(8, 3))
    plt.bar(range(position_bins), plateau_per_bin)
    plt.title("Plateaus per Position Bin")
    plt.xlabel("Position Bin")
    plt.ylabel("# of Plateaus")
    plt.ylim(0, 10)
    plt.tight_layout()
    plt.show()

    plt.plot(E_rate_list, label="EC", color='g')
    plt.plot(I_rate_list, label="Dend_I", color='orange', linewidth=3)
    plt.ylabel("Activity")
    plt.xlabel("Time (ms)")
    plt.ylim(0.3, 0.5)
    plt.legend()
    plt.show()

    plt.plot(Vm_Record, label="Dendrite Vm", color='b')
    plt.ylim(-0.1, 0.1)
    plt.axhline(0, linestyle="--", linewidth=2, label='Threshold', color='r')
    plt.ylabel("Activity")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.bar(range(position_bins), time_per_bin)
    plt.title("Occupancy per Position Bin")
    plt.xlabel("Position Bin")
    plt.ylabel("Time Spent (ms)")
    plt.tight_layout()
    plt.show()


def get_plot_trial_av_kmeans(cell_new_NDNF_model_ranks20_contiguous_x00_cell, activity_dict, residual_activity_dict_NDNF_new, use_residuals=True, k=3):
    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_new_NDNF_model_ranks20_contiguous_x00_cell, activity_dict, animal_TCA=False)

    early_activity_every_cell = []
    late_activity_every_cell = []

    q1_activity_every_cell = []
    q5_activity_every_cell = []

    raw_early_activity_every_cell = []
    raw_late_activity_every_cell = []

    raw_q1_activity_every_cell = []
    raw_q5_activity_every_cell = []

    cell_list = []
    correlation_velcity_list = []
    peak_loc_list = []
    trough_loc_list = []

    trial_av_act_list = []

    for idx, animal in enumerate(residual_activity_dict_NDNF_new):
        if idx > 8:
            first_cp_animal = animal_first_changepoints_list[idx]
            second_cp_animal = animal_second_changepoints_list[idx]
            for ids, cell in enumerate(residual_activity_dict_NDNF_new[animal]):
                early_cp = first_cp_animal[ids]
                late_cp = second_cp_animal[ids]

                if use_residuals:
                    cell_activity = residual_activity_dict_NDNF_new[animal][cell]

                    cell_raw_activity = activity_dict[animal][cell]

                    cell_activity_trials = cell_activity.shape[1]
                    quintile_length = cell_activity_trials // 5

                    cutpoints_list = []
                    counts = 0
                    for i in range(4):
                        cutpoints_list.append(counts + quintile_length)
                        counts += quintile_length

                    q1 = cutpoints_list[0]
                    q5 = cutpoints_list[-1]

                    q1_activity = cell_raw_activity[:, :q1]
                    q5_activity = cell_raw_activity[:, q5:]
                    raw_q1_activity_every_cell.append(np.mean(q1_activity, axis=1))
                    raw_q5_activity_every_cell.append(np.mean(q5_activity, axis=1))

                    early_activity = cell_raw_activity[:, :early_cp]
                    late_activity = cell_raw_activity[:, :late_cp]

                    raw_early_activity_every_cell.append(np.mean(early_activity, axis=1))
                    raw_late_activity_every_cell.append(np.mean(late_activity, axis=1))


                else:
                    cell_activity = activity_dict[animal][cell]

                cell_activity_trials = cell_activity.shape[1]
                quintile_length = cell_activity_trials // 5

                cutpoints_list = []
                counts = 0
                for i in range(4):
                    cutpoints_list.append(counts + quintile_length)
                    counts += quintile_length

                q1 = cutpoints_list[0]
                q5 = cutpoints_list[-1]

                q1_activity = cell_activity[:, :q1]
                q1_activity_every_cell.append(np.mean(q1_activity, axis=1))
                q5_activity = cell_activity[:, q5:]
                q5_activity_every_cell.append(np.mean(q5_activity, axis=1))

                early_activity = cell_activity[:, :early_cp]
                late_activity = cell_activity[:, :late_cp]

                early_activity_every_cell.append(np.mean(early_activity, axis=1))
                late_activity_every_cell.append(np.mean(late_activity, axis=1))

                if use_residuals:
                    trial_av = np.mean(residual_activity_dict_NDNF_new[animal][cell], axis=1)
                    trial_av_act = np.mean(activity_dict[animal][cell], axis=1)
                    trial_av_act_list.append(trial_av_act)
                else:
                    trial_av = np.mean(activity_dict[animal][cell], axis=1)
                cell_list.append(trial_av)
                cell_activity_flat = activity_dict[animal][cell].flatten()
                velocity_flat = filtered_factors_dict[animal]["Velocity"].flatten()
                r_value, _ = pearsonr(cell_activity_flat, velocity_flat)
                peak_loc_list.append(np.argmax(trial_av))
                trough_loc_list.append(np.argmin(trial_av))
                correlation_velcity_list.append(r_value)

    if use_residuals:

        cell_array = np.array(trial_av_act_list)
        residual_array = np.array(cell_list)

        mean_resid = np.mean(residual_array, axis=0)
        sem_resid = sem(residual_array, axis=0)

        mean_cell = np.mean(cell_array, axis=0)
        sem_cell = sem(cell_array, axis=0)

        raw_early_activity_every_cell_array = np.array(raw_early_activity_every_cell)
        mean_raw_early_activity_every_cell_array = np.mean(raw_early_activity_every_cell_array, axis=0)
        sem_raw_early_activity_every_cell_array = sem(raw_early_activity_every_cell_array, axis=0)

        raw_late_activity_every_cell_array = np.array(raw_late_activity_every_cell)
        mean_raw_late_activity_every_cell_array = np.mean(raw_late_activity_every_cell_array, axis=0)
        sem_raw_late_activity_every_cell_array = sem(raw_late_activity_every_cell_array, axis=0)

        early_activity_every_cell_array = np.array(early_activity_every_cell)
        mean_early_activity_every_cell_array = np.mean(early_activity_every_cell_array, axis=0)
        sem_early_activity_every_cell_array = sem(early_activity_every_cell_array, axis=0)

        late_activity_every_cell_array = np.array(late_activity_every_cell)
        mean_late_activity_every_cell_array = np.mean(late_activity_every_cell_array, axis=0)
        sem_late_activity_every_cell_array = sem(late_activity_every_cell_array, axis=0)

        raw_q1_activity_every_cell_array = np.array(raw_q1_activity_every_cell)
        mean_raw_q1_activity_every_cell_array = np.mean(raw_q1_activity_every_cell_array, axis=0)
        sem_raw_q1_activity_every_cell_array = sem(raw_q1_activity_every_cell_array, axis=0)

        raw_q5_activity_every_cell_array = np.array(raw_q5_activity_every_cell)
        mean_raw_q5_activity_every_cell_array = np.mean(raw_q5_activity_every_cell_array, axis=0)
        sem_raw_q5_activity_every_cell_array = sem(raw_q5_activity_every_cell_array, axis=0)

        q1_activity_every_cell_array = np.array(q1_activity_every_cell)
        mean_q1_activity_every_cell_array = np.mean(q1_activity_every_cell_array, axis=0)
        sem_q1_activity_every_cell_array = sem(q1_activity_every_cell_array, axis=0)

        q5_activity_every_cell_array = np.array(q5_activity_every_cell)
        mean_q5_activity_every_cell_array = np.mean(q5_activity_every_cell_array, axis=0)
        sem_q5_activity_every_cell_array = sem(q5_activity_every_cell_array, axis=0)

        mean_resid = np.mean(residual_array, axis=0)
        sem_resid = sem(residual_array, axis=0)

        mean_cell = np.mean(cell_array, axis=0)
        sem_cell = sem(cell_array, axis=0)

        plt.plot(mean_resid, color='orange', label="Vel-Sub. Resid.")
        plt.fill_between(range(len(mean_resid)), mean_resid - sem_resid, mean_resid + sem_resid, alpha=0.1, color="orange")
        plt.plot(mean_cell, color='grey', label="Raw")
        plt.fill_between(range(len(mean_cell)), mean_cell - sem_cell, mean_cell + sem_cell, alpha=0.1, color='grey')
        plt.title("Raw vs Residual New NDNF Data")
        plt.ylabel("Activity")
        plt.xlabel("Position Bin")
        plt.legend()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

        # Plot 1: Residual Early vs Late
        axs[0, 1].plot(mean_early_activity_every_cell_array, color='orange', label='Early')
        axs[0, 1].fill_between(
            range(len(mean_early_activity_every_cell_array)),
            mean_early_activity_every_cell_array + sem_early_activity_every_cell_array,
            mean_early_activity_every_cell_array - sem_early_activity_every_cell_array,
            color='orange', alpha=0.1
        )
        axs[0, 1].plot(mean_late_activity_every_cell_array, color='r', label='Late')
        axs[0, 1].fill_between(
            range(len(mean_late_activity_every_cell_array)),
            mean_late_activity_every_cell_array + sem_late_activity_every_cell_array,
            mean_late_activity_every_cell_array - sem_late_activity_every_cell_array,
            color='r', alpha=0.1
        )
        axs[0, 1].set_title("Residual Early vs Late")
        axs[0, 1].legend()

        # Plot 2: Raw Early vs Late
        axs[0, 0].plot(mean_raw_early_activity_every_cell_array, color='orange', label='Raw Early')
        axs[0, 0].fill_between(
            range(len(mean_raw_early_activity_every_cell_array)),
            mean_raw_early_activity_every_cell_array + sem_raw_early_activity_every_cell_array,
            mean_raw_early_activity_every_cell_array - sem_raw_early_activity_every_cell_array,
            color='orange', alpha=0.1
        )
        axs[0, 0].plot(mean_raw_late_activity_every_cell_array, color='r', label='Raw Late')
        axs[0, 0].fill_between(
            range(len(mean_raw_late_activity_every_cell_array)),
            mean_raw_late_activity_every_cell_array + sem_raw_late_activity_every_cell_array,
            mean_raw_late_activity_every_cell_array - sem_raw_late_activity_every_cell_array,
            color='r', alpha=0.1)
        axs[0, 0].set_title("Raw Early vs Late")
        axs[0, 0].legend()

        # Plot 3: Residual Q1 vs Q5
        axs[1, 1].plot(mean_q1_activity_every_cell_array, color='orange', label='Q1')
        axs[1, 1].fill_between(
            range(len(mean_q1_activity_every_cell_array)),
            mean_q1_activity_every_cell_array + sem_q1_activity_every_cell_array,
            mean_q1_activity_every_cell_array - sem_q1_activity_every_cell_array,
            color='orange', alpha=0.1
        )
        axs[1, 1].plot(mean_q5_activity_every_cell_array, color='r', label='Q5')
        axs[1, 1].fill_between(
            range(len(mean_q5_activity_every_cell_array)),
            mean_q5_activity_every_cell_array + sem_q5_activity_every_cell_array,
            mean_q5_activity_every_cell_array - sem_q5_activity_every_cell_array,
            color='r', alpha=0.1
        )
        axs[1, 1].set_title("Residual Q1 vs Q5")
        axs[1, 1].legend()

        # Plot 4: Raw Q1 vs Q5
        axs[1, 0].plot(mean_raw_q1_activity_every_cell_array, color='orange', label='Raw Q1')
        axs[1, 0].fill_between(
            range(len(mean_raw_q1_activity_every_cell_array)),
            mean_raw_q1_activity_every_cell_array + sem_raw_q1_activity_every_cell_array,
            mean_raw_q1_activity_every_cell_array - sem_raw_q1_activity_every_cell_array,
            color='orange', alpha=0.1
        )
        axs[1, 0].plot(mean_raw_q5_activity_every_cell_array, color='r', label='Raw Q5')
        axs[1, 0].fill_between(
            range(len(mean_raw_q5_activity_every_cell_array)),
            mean_raw_q5_activity_every_cell_array + sem_raw_q5_activity_every_cell_array,
            mean_raw_q5_activity_every_cell_array - sem_raw_q5_activity_every_cell_array,
            color='r', alpha=0.1
        )
        axs[1, 0].set_title("Raw Q1 vs Q5")
        axs[1, 0].legend()

        # Final touches
        for ax in axs.flat:
            ax.set_xlabel("Position Bin")
            ax.set_ylabel("Activity (ΔF/F)")
            ax.set_ylim(-0.5, 0.7)

        plt.tight_layout()
        plt.show()

    cell_array = np.array(cell_list)
    sorted_array = np.argsort(np.argmax(cell_array, axis=1))
    plt.figure(figsize=(6, 10))
    finished_cell_array = cell_array[sorted_array, :]
    plt.imshow(finished_cell_array)
    if use_residuals:
        plt.title("NDNF Residual Activity")
    else:
        plt.title("NDNF Raw Activity")
    plt.xlabel("Position Bins")
    plt.ylabel("Cell ID")

    X = np.column_stack((peak_loc_list, correlation_velcity_list, trough_loc_list))

    n_clusters = k  # e.g. 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
    labels = kmeans.fit_predict(X)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    scatter1 = axs[0].scatter(X[:, 0], X[:, 1], c=labels)
    axs[0].set_xlabel("Peak Location (Bin)")
    axs[0].set_ylabel("Correlation to Velocity")
    axs[0].set_title("Peak vs Velocity")
    axs[0].grid(True)
    fig.colorbar(scatter1, ax=axs[0], label="Cluster ID")

    scatter2 = axs[1].scatter(X[:, 2], X[:, 1], c=labels)
    axs[1].set_xlabel("Trough Location (Bin)")
    axs[1].set_ylabel("Correlation to Velocity")
    axs[1].set_title("Trough vs Velocity")
    axs[1].grid(True)
    fig.colorbar(scatter2, ax=axs[1], label="Cluster ID")

    scatter3 = axs[2].scatter(X[:, 2], X[:, 0], c=labels)
    axs[2].set_xlabel("Trough Location (Bin)")
    axs[2].set_ylabel("Peak Location (Bin)")
    axs[2].set_title("Trough vs Peak")
    axs[2].grid(True)
    fig.colorbar(scatter3, ax=axs[2], label="Cluster ID")

    plt.tight_layout()
    plt.show()

    early_cluster_activity_dict = defaultdict(list)
    late_cluster_activity_dict = defaultdict(list)
    for i, label in enumerate(labels):
        early_cluster_activity_dict[label].append(early_activity_every_cell[i])
        late_cluster_activity_dict[label].append(late_activity_every_cell[i])

    q1_cluster_activity_dict = defaultdict(list)
    q5_cluster_activity_dict = defaultdict(list)
    for i, label in enumerate(labels):
        q1_cluster_activity_dict[label].append(q1_activity_every_cell[i])
        q5_cluster_activity_dict[label].append(q5_activity_every_cell[i])

    cluster_activity_dict = defaultdict(list)

    for i, label in enumerate(labels):
        cluster_activity_dict[label].append(cell_list[i])

    fig, axs = plt.subplots(4, len(cluster_activity_dict), figsize=(15, 15))
    for cluster in cluster_activity_dict:
        activity_list = cluster_activity_dict[cluster]
        activity_array = np.array(activity_list)

        early_cluster_activity_list = early_cluster_activity_dict[cluster]
        late_cluster_activity_list = late_cluster_activity_dict[cluster]

        early_cluster_activity_array = np.array(early_cluster_activity_dict[cluster])
        late_cluster_activity_array = np.array(late_cluster_activity_dict[cluster])

        sem_early = sem(early_cluster_activity_array, axis=0)
        sem_late = sem(late_cluster_activity_array, axis=0)

        mean_early = np.mean(early_cluster_activity_array, axis=0)
        mean_late = np.mean(late_cluster_activity_array, axis=0)

        q1_cluster_activity_list = q1_cluster_activity_dict[cluster]
        q5_cluster_activity_list = q5_cluster_activity_dict[cluster]

        q1_cluster_activity_array = np.array(q1_cluster_activity_list)
        q5_cluster_activity_array = np.array(q5_cluster_activity_list)

        sem_q1 = sem(q1_cluster_activity_array, axis=0)
        sem_q5 = sem(q5_cluster_activity_array, axis=0)

        mean_q1 = np.mean(q1_cluster_activity_array, axis=0)
        mean_q5 = np.mean(q5_cluster_activity_array, axis=0)

        if use_residuals:
            plt.suptitle("Velocity-Subtracted Residual Data")
        else:
            plt.suptitle("Raw Data")

        axs[0, cluster].imshow(activity_array, aspect='auto')
        axs[0, cluster].set_title(f"Cluster {cluster} \n Cells Activity")
        axs[0, cluster].set_ylabel("Cell ID")
        axs[0, cluster].set_xlabel("Position Bin")
        axs[1, cluster].plot(np.mean(activity_array, axis=0))
        axs[1, cluster].set_ylabel("Cell ID")
        axs[1, cluster].set_xlabel("Position Bin")
        axs[1, cluster].set_title(f"Trial Average Activity")
        axs[1, cluster].set_ylim(-0.6, 1.3)

        axs[2, cluster].plot(mean_early, color='orange', label="Early Learn")
        axs[2, cluster].fill_between(range(len(mean_early)), mean_early + sem_early, mean_early - sem_early, color="orange", alpha=0.1)
        axs[2, cluster].plot(mean_late, color='r', label="Late Learn")
        axs[2, cluster].fill_between(range(len(mean_late)), mean_late + sem_late, mean_late - sem_late, color="r", alpha=0.1)
        axs[2, cluster].set_ylabel("Cell ID")
        axs[2, cluster].set_xlabel("Position Bin")
        axs[2, cluster].set_title(f"Contiguous Changepoints")
        axs[2, cluster].set_ylim(-0.6, 1.3)
        axs[2, cluster].legend()

        axs[3, cluster].plot(mean_q1, color='orange', label="Q1")
        axs[3, cluster].fill_between(range(len(mean_q1)), mean_q1 + sem_q1, mean_q1 - sem_q1, color="orange", alpha=0.1)
        axs[3, cluster].plot(mean_q5, color='r', label="Q5")
        axs[3, cluster].fill_between(range(len(mean_q5)), mean_q5 + sem_q5, mean_q5 - sem_q5, color="r", alpha=0.1)
        axs[3, cluster].set_ylabel("Cell ID")
        axs[3, cluster].set_xlabel("Position Bin")
        axs[3, cluster].set_title(f"Quintiles")
        axs[3, cluster].set_ylim(-0.6, 1.3)
        axs[3, cluster].legend()

    plt.tight_layout()
    plt.show()


def plot_probability_curves_EC(n=212):

    num_trials = n

    # Field 1: uniform probability (e.g., 0.5 across all trials)
    prob_field1 = np.full(num_trials, 0.5)

    # Field 2: Gaussian-shaped probability centered in the middle trials
    trial_indices = np.arange(num_trials)
    center = num_trials // 2
    std_for_prob = num_trials / 5
    prob_field2 = norm.pdf(trial_indices, loc=center, scale=std_for_prob)
    prob_field2 = prob_field2 / np.max(prob_field2)  # normalize to [0, 1]

    # Plot
    plt.figure(figsize=(5, 4))
    plt.plot(prob_field1, label="Field 1 Probability (uniform)", linestyle="--")
    plt.plot(prob_field2, label="Field 2 Probability (center-weighted)", color="orange")
    plt.xlabel("Trial Index")
    plt.ylabel("Probability of Appearing")
    plt.title("Probability Curves for Field 1 and Field 2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def get_synthetic_data(activity_dict, velocity, place_field_type='flat', place_field_scale=1, place_field_shift=0, velocity_weight_type='flat', velocity_weight=1, velocity_power=1, noise_scale=1):
    from scipy.stats import norm

    # 1. Make "ground truth" place field
    def get_average_cell_profile(activity_dict):
        all_cells_average = []
        for animal in activity_dict:
            for neuron in activity_dict[animal]:
                cell_trial_average = activity_dict[animal][neuron].mean(axis=1)
                all_cells_average.append(cell_trial_average)
        all_cells_average = np.stack(all_cells_average, axis=0).mean(axis=0)
        return all_cells_average

    place_field_profile = get_average_cell_profile(activity_dict)
    num_trials = velocity.shape[1]
    place_field = np.tile(place_field_profile, (num_trials, 1)).T

    def staircase_vector(start, stop, num_steps, length):
        steps = np.linspace(start, stop, num_steps)  # Generate step levels
        step_counts = np.full(num_steps, length // num_steps)  # Base count per step
        step_counts[:length % num_steps] += 1  # Distribute remainder among first steps
        return np.repeat(steps, step_counts)  # Repeat steps with adjusted counts

    match place_field_type:
        case "flat":
            place_field_scale = np.ones(num_trials)
            place_field *= place_field_scale
        case "positive_ramp":
            place_field_scale = np.linspace(0, place_field_scale, num_trials)
            place_field *= place_field_scale
        case "negative_ramp":
            place_field_scale = np.linspace(place_field_scale, 0, num_trials)
            place_field *= place_field_scale
        case "step":
            place_field_scale = staircase_vector(0, place_field_scale, num_steps=2, length=num_trials)
            place_field *= place_field_scale
        case "BTSP":
            place_field_scale = BTSP_field(num_trials)
            place_field *= place_field_scale
        case "EC":
            place_field = example_EC_cell(velocity)
            place_field = place_field.T

    place_field = np.roll(place_field, shift=place_field_shift, axis=0)

    # 2. Combine the synthetic place field with velocity
    match velocity_weight_type:
        case "flat":
            velocity_weight = velocity_weight * np.ones(num_trials)
        case "positive_ramp":
            velocity_weight = np.linspace(0, velocity_weight, num_trials)
        case "negative_ramp":
            velocity_weight = np.linspace(velocity_weight, 0, num_trials)
        case "step":
            velocity_weight = staircase_vector(0, velocity_weight, num_steps=5, length=num_trials)

    velocity_component = velocity_weight * (velocity ** velocity_power)

    noise = np.random.normal(0, noise_scale, size=(len(place_field_profile), num_trials))
    combined_activity = place_field + velocity_component + noise

    return combined_activity, place_field, velocity_component, noise


def example_EC_cell(velocity):
    length = 50
    num_trials = velocity.shape[1]  # shape should be (position, trials)
    x = np.linspace(0, length - 1, length)

    # Define Gaussians
    mean1 = 15
    std_dev1 = 5
    original_gaussian1 = norm.pdf(x, mean1, std_dev1)

    mean2 = 35
    std_dev2 = 5
    original_gaussian2 = norm.pdf(x, mean2, std_dev2)

    # Define probability curve for field 2
    trial_indices = np.arange(num_trials)
    center = num_trials // 2
    std_for_prob = num_trials / 5
    prob_curve_field2 = norm.pdf(trial_indices, loc=center, scale=std_for_prob)
    prob_curve_field2 = prob_curve_field2 / np.max(prob_curve_field2)  # Normalize to [0, 1]

    gaussian_list = []

    for i in range(num_trials):
        appear_1 = np.random.choice([0, 1])  # Uniform probability for field 1

        # Use the center-weighted probability for field 2
        appear_2 = np.random.rand() < prob_curve_field2[i]

        gaussian1 = original_gaussian1 * appear_1
        gaussian2 = original_gaussian2 * appear_2

        combined_gaussian = gaussian1 + gaussian2
        bimodal_gaussian = combined_gaussian / np.max(combined_gaussian) if np.max(combined_gaussian) != 0 else combined_gaussian

        gaussian_list.append(bimodal_gaussian)

    pf = np.stack(gaussian_list)
    return pf


def model_cell_TCA():
    # Define the Gaussian curve
    position_bins = 50
    trials = 60
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-0.5, 1.5, 50)
    positive_gaussian = np.exp(-x1 ** 2 / (2 * 0.1 ** 2))
    negative_gaussian = -np.exp(-x2 ** 2 / (2 * 0.1 ** 2))

    # Define the weights
    weight_one = np.concatenate([
        np.ones(20),
        np.linspace(1, 0, 20),
        np.zeros(20)
    ])
    weight_two = np.concatenate([
        np.zeros(20),
        np.linspace(0, 1, 20),
        np.ones(20)
    ])

    plt.plot(positive_gaussian, color="orange")
    plt.title("Early Place Field")
    plt.xlabel("Position Bins")
    plt.show()
    plt.plot(negative_gaussian, color="b")
    plt.title("Late Place Field")
    plt.xlabel("Position Bins")
    plt.show()

    plt.plot(weight_one, label="Weight 1", color="orange")
    plt.plot(weight_two, label="Weight 2", color="b")
    plt.ylabel("Weights")
    plt.xlabel("Trials")
    plt.legend()
    plt.show()

    # Make the 60 × 50 matrices
    positive_matrix = np.outer(weight_one, positive_gaussian)
    negative_matrix = np.outer(weight_two, negative_gaussian)

    # Combine them
    combined_matrix = positive_matrix + negative_matrix

    import seaborn as sns
    sns.heatmap(combined_matrix, cmap='inferno', xticklabels=5, yticklabels=5)
    plt.xlabel("Position Bins")
    plt.ylabel("Trials")
    plt.title("Combined Activity Matrix")
    plt.show()

    # combined_matrix_normalized = (combined_matrix / np.max(combined_matrix)) / (np.min(combined_matrix) / np.max(combined_matrix))

    # Convert to PyTorch tensor and reshape to [60, 1, 50]
    tensor_combined = torch.tensor(combined_matrix, dtype=torch.float32).unsqueeze(1)

    # Confirm the shape
    print(tensor_combined.shape)  # should be: torch.Size([60, 1, 50])

    ranks = 2

    components, model_synthetic_SST = slicetca.decompose(tensor_combined,
                                                         number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                                         positive=False,
                                                         learning_rate=1 * 10 ** -3,
                                                         min_std=10 ** -5,  # max_iter=150,
                                                         max_iter=15_000,
                                                         seed=0)

    axes = slicetca.plot(model_synthetic_SST,
                         variables=('trial', 'neuron', 'time'),
                         #               colors=(trial_colors[trial_idx], None, None), # we only want the trials to be colored
                         #               ticks=(None, None, np.linspace(0,150,4)), # we only want to modify the time ticks
                         #               tick_labels=(None, None, np.linspace(-1,0.5,4)),
                         #               sorting_indices=(trial_idx, neuron_sorting_peak_time, None),
                         quantile=0.99)

    X = model_synthetic_SST.vectors[0][0].detach().numpy().T

    print(X.shape)

    # Let's say you want to find 2 changepoints (i.e., 3 segments)
    n_bkps = 2

    # Run change point detection using Binseg (can also try 'pelt' or 'bottomup')
    model = rpt.Binseg(model="l2").fit(X)
    change_points = model.predict(n_bkps=n_bkps)
    change_points = change_points[:-1]
    # change_points includes the *end* of each segment
    print(f"Change points: {change_points}")  # e.g., [18, 40, 60]

    # Plot both latents with vertical lines at changepoints
    plt.figure(figsize=(10, 4))
    plt.plot(X[:, 0], label='Latent 1')
    plt.plot(X[:, 1], label='Latent 2')
    for cp in change_points:  # exclude the final one (always == len)
        plt.axvline(cp, color='red', linestyle='--', label='Changepoint' if cp == change_points[0] else None)

    plt.xlabel("Trial")
    plt.ylabel("Latent Value")
    plt.title("SliceTCA Latents with Changepoints")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.imshow(combined_matrix, aspect='auto')
    plt.ylabel("Trials")
    plt.xlabel("Position Bins")
    plt.title("Synthetic Neuron")
    plt.colorbar()
    plt.show()

    early_cut = combined_matrix[:25, :]
    plt.imshow(early_cut, aspect='auto', vmin=-1, vmax=1)
    plt.ylabel("Trials")
    plt.xlabel("Position Bins")
    plt.title("Synthetic Neuron Early Learn")
    plt.colorbar()
    plt.show()

    plt.plot(np.mean(early_cut, axis=0), color='orange')
    plt.xlabel("Position Bins")
    plt.title("Synthetic Neuron Early Learn Trial Av")
    plt.show()

    late_cut = combined_matrix[30:, :]
    plt.imshow(late_cut, aspect='auto', vmin=-1, vmax=1)
    plt.ylabel("Trials")
    plt.xlabel("Position Bins")
    plt.title("Synthetic Neuron Late Learn")
    plt.colorbar()
    plt.show()

    plt.plot(np.mean(late_cut, axis=0), color='b')
    plt.xlabel("Position Bins")
    plt.title("Synthetic Neuron Late Learn Trial Av")
    plt.show()


def sliceTCA_example_SST_cell():
    # MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
    # Load data
    filename = "SSTindivsomata_GLM"
    # filename = "NDNFindivsomata_GLM"
    # filename = "EC_GLM"
    # filename = "NDNFanalC"

    # cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
    # animal_id = int(sys.argv[2])       # Provided via command-line argument
    # ranks = int(sys.argv[3])

    cell_id = 2
    animal_id = 1
    ranks = 20

    filepath = os.path.join(filename + ".mat")
    activity_dict, factors_dict = ut.preprocess_data2(filepath, normalize=True, new_NDNF=False)

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
        tensor_for_cell = tensor_for_animal[:, cell_id, :]
        cell_of_interest = tensor_for_animal[:, cell_id, :].unsqueeze(1)
        print(f"cell_of_interest.shape {cell_of_interest.shape}")
        cell_of_interest.requires_grad_()
        components, SST_model = slicetca.decompose(cell_of_interest,
                                                   number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                                   positive=True,
                                                   learning_rate=1 * 10 ** -3,
                                                   min_std=10 ** -5,  # max_iter=150,
                                                   max_iter=15_000, seed=0)
        w_SST = SST_model.vectors[0][0].detach().numpy()
        X = w_SST.T

        from scipy.ndimage import gaussian_filter1d

        # Apply Gaussian smoothing along the trials (axis=0)
        X_smooth = gaussian_filter1d(X, sigma=2, axis=0)  # try sigma=1, 2, or 3

        # Re-run change point detection on smoothed data
        model = rpt.Binseg(model="l2").fit(X_smooth)
        change_points = model.predict(n_bkps=n_bkps)[:-1]  # remove final 'len' cp

        # Plot smoothed latents with changepoints
        plt.figure(figsize=(10, 4))
        for i in range(X_smooth.shape[1]):
            plt.plot(X_smooth[:, i])

        for cp in change_points:
            plt.axvline(cp, color='red', linestyle='--', label='Changepoint' if cp == change_points[0] else None)

        plt.xlabel("Trial")
        plt.ylabel("Latent Value")
        plt.title("SliceTCA Latents with Changepoints")
        plt.tight_layout()
        plt.show()

        SST_model, change_points = sliceTCA_example_SST_cell()

        cell_id = 4

        plt.imshow(activity_dict_SST['animal_2'][f"cell_{cell_id}"].T, aspect='auto')
        plt.axhline(change_points[0], linestyle="--", linewidth=3, color='r')
        plt.axhline(change_points[1], linestyle="--", linewidth=3, color='r')
        plt.ylabel("Trials")
        plt.xlabel("Position Bin")
        plt.show()

        cell_ex_activity = activity_dict_SST['animal_2'][f"cell_{cell_id}"]
        early_cp = change_points[0]
        late_cp = change_points[1]

        plt.plot(np.mean(cell_ex_activity[:, :early_cp], axis=1), color='cyan', label='Early Learn')
        plt.plot(np.mean(cell_ex_activity[:, late_cp:], axis=1), color='blue', label='Late Learn')
        plt.ylabel("DF/F")
        plt.xlabel("Position Bins")
        plt.legend()
        plt.show()
