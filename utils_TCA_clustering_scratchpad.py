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


def get_model_data_per_animal2(mse_dir, cell_type="EC"):
    import re
    import os
    import pickle

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
    # More flexible pattern: ignore the cell type
    pattern = re.compile(r"MSE_.*?_cell_latent_(\d+)_animal(\d+)_cell_id(\d+)\.pkl")

    # Structure: {rank: {animal_id: {cell_id: mse_value}}}
    rank_mse_dict = {}

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

                # Initialize nested dicts
                rank_mse_dict.setdefault(rank, {}).setdefault(animal_id, {})[cell_id] = model_list
            else:
                print(f"[Skipping] Unexpected filename format: {fname}")

    # Summary printout
    print(f"Loaded MSEs for {len(rank_mse_dict)} rank(s).")
    for rank, animal_dict in rank_mse_dict.items():
        print(f"  Rank {rank}:")
        for animal_id, cell_dict in animal_dict.items():
            print(f"    Animal {animal_id}: {len(cell_dict)} cells")

    return rank_mse_dict


def plot_per_cell_clustering_internals_single_cluster(cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, residual_activity_dict_NDNF, animal_id=1, cell_id=1, num_clusters=4):
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
    fig, axs = plt.subplots(2, num_clusters, figsize=(num_clusters * 4, 6), squeeze=False)

    for n in range(num_clusters):
        indices = indices_dict[n]
        cluster = real_activity[indices, :]  # trials x bins

        # Plot the cluster heatmap
        axs[0, n].imshow(cluster, aspect='auto', vmin=0, vmax=1)
        axs[0, n].set_title(f"Cluster {n}")
        axs[0, n].set_xlabel("Position")
        axs[0, n].set_ylabel("Trial")

        # Plot the cluster mean
        axs[1, n].plot(np.mean(cluster, axis=0))
        axs[1, n].set_xlabel("Position")
        axs[1, n].set_ylabel("Mean activity")
        axs[1, n].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

