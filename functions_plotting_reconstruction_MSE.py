import os
import glob
import pickle
import numpy as np
from scipy.stats import sem
from GLM_regression import *
from sklearn.decomposition import PCA


import re


def extract_animal_cell(filename):
    basename = os.path.basename(filename)
    match = re.search(r"animal(\d+)_cell_id(\d+)", basename)
    if match:
        animal = int(match.group(1))
        cell = int(match.group(2))
        return (animal, cell)
    else:
        raise ValueError(f"Filename doesn't match expected format: {filename}")


def get_mean_sem_MSE_from_pkl_UMAP_kmean_00x(data_dir, residual_activity_dict, cell_type="EC"):
    animal_MSE_list = []
    internals_dict_every_cell_dict = {}

    # print("All available pickle files in the directory:")
    # print(glob.glob(os.path.join(data_dir, "*.pkl")))

    for animal_idx, animal in enumerate(residual_activity_dict):
        if cell_type == "EC" and animal_idx == len(residual_activity_dict) - 1:
            continue
        pattern = os.path.join(data_dir, f"MSE_{cell_type}_cell_latent_40_animal{animal_idx}_cell_id*.pkl")
        model_paths = sorted(glob.glob(pattern), key=extract_animal_cell)

        mse_list = []

        internals_dict_animal = {}

        for cell_num, path in enumerate(model_paths):
            with open(path, "rb") as f:
                internals_dict = pickle.load(f)
                # print(f"Loaded {path}, keys: {list(internals_dict.keys())}")

            internals_dict_animal[f"cell_{cell_num}"] = internals_dict
            mse = internals_dict.get("MSE_list", [])

            mse_list.append(mse)

        internals_dict_every_cell_dict[animal] = internals_dict_animal

        if mse_list:
            mse_array = np.array(mse_list)
            # print(f"mse_array.shape for animal {animal_idx}: {mse_array.shape}")
            mean_mse = np.mean(mse_array, axis=0)
            animal_MSE_list.append(mean_mse)

    if not animal_MSE_list:
        print("❌ No valid data was found. Check file naming and expected_length.")

    animal_MSE_array = np.array(animal_MSE_list)
    mean_MSE = np.mean(animal_MSE_array, axis=0)
    sem_MSE = sem(animal_MSE_array, axis=0)

    return mean_MSE, sem_MSE, internals_dict_every_cell_dict


plot_latents = False
plot_reconstruction = False
plot_kmeans = False
display = False
plot_mean_cluster = False
max_clusters = 12

import umap
import ruptures as rpt


def get_real_animal_tensor(residual_activity_dict_SST, animal_num=1):
    real_animal_activity = residual_activity_dict_SST[f'animal_{animal_num}']
    real_animal_data_list = []
    for neuron in real_animal_activity:
        cell_activity = real_animal_activity[neuron]
        real_animal_data_list.append(cell_activity)
    animal_tensor = np.array(real_animal_data_list)
    animal_tensor = animal_tensor.transpose(2, 0, 1)

    return animal_tensor


def get_per_animal_MSE_UMAP_contigmeans_00x(animal_model_list, residual_activity_dict_SST, max_clusters=12, display=True):
    MSE_dict = {}
    x_pca_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    for clusters_chosen in range(max_clusters):

        animal_MSE_dict = {}
        animal_X_pca_dict = {}
        animal_labels_dict = {}
        animal_cluster_trial_indices = {}
        animal_mean_cluster_latent_dict = {}
        reconstructed_TCA_tensor_per_animal_dict = {}
        animal_cluster_reconstruction = {}
        animal_cluster_mean_dict = {}

        for idx, animal_model in enumerate(animal_model_list):  #

            idx = idx + 1

            reconstruction_full_animal = animal_model.construct().numpy(force=True)
            reconstructed_TCA_tensor_per_animal_dict[f"animal_{animal_model}"] = reconstruction_full_animal  #

            if plot_latents:
                axes = slicetca.plot(animal_model,
                                     variables=('trial', 'neuron', 'time'),
                                     #   ticks=(None, None, np.linspace(0,50,3)), # we only want to modify the time ticks
                                     #   tick_labels=(None, None, np.linspace(0,50,3)),
                                     #   sorting_indices=(None, neuron_sorting_peak_time, None),
                                     quantile=0.99)

            f = animal_model.vectors[2][1].detach()
            f1 = f.permute(2, 0, 1)  # [5, 40, 212]

            f1 = f1.reshape(-1, f1.shape[-1]).T  # [200, 212]

            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)  # (trials,3)

            #             kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            #             labels = kmeans.fit_predict(f1)

            algo = rpt.Binseg(model="l2", min_size=2).fit(X_umap)
            bkps = algo.predict(n_bkps=max_clusters)

            labels = np.zeros(X_umap.shape[0], dtype=int)
            start = 0
            for cluster_id, end in enumerate(bkps):
                labels[start:end] = cluster_id
                start = end

            #                 pca = PCA(n_components=2)
            #                 X_pca = pca.fit_transform(X)

            #                 animal_X_pca_dict[f"animal_{idx}"] = X_pca
            animal_X_pca_dict[f"animal_{idx}"] = X_umap
            animal_labels_dict[f"animal_{idx}"] = labels

            #### get mean activity in cluster
            mean_cluster_trials = []
            animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, idx)

            ########### get the indicces and the mean latents
            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                #                 if display:
                # print(f"trial_indices {trial_indices}")
                if len(trial_indices) > 2:
                    cluster_trials = reconstruction_full_animal[trial_indices, :, :]
                    #                     if display:
                    # print(f"cluster_trials.shape {cluster_trials.shape}")
                    mean_cluster = cluster_trials.mean(axis=0)
                    valid_cluster_mean_trials_list.append(mean_cluster)
                    valid_cluster_indices.append((n, trial_indices))

                else:
                    print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
            animal_cluster_mean_dict[f"animal_{idx}"] = valid_cluster_mean_trials_list
            cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
            animal_cluster_trial_indices[f"animal_{idx}"] = cluster_trial_indices

            # print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

            ########### reconstruct the tensor with the avrage latent
            empty_cell = np.zeros(reconstruction_full_animal.shape)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :, :] = valid_cluster_mean_trials_list[i]
            animal_cluster_reconstruction[f"animal_{idx}"] = empty_cell

            ########### get the MSE
            neuron_MSE_dict = {}
            for neuron_idx, neuron in enumerate(residual_activity_dict_SST[f"animal_{idx}"]):
                reconstruction = empty_cell[:, neuron_idx, :]
                real_cell_activity = animal_tensor[:, neuron_idx, :]
                MSE = np.mean(np.square(real_cell_activity - reconstruction))
                neuron_MSE_dict[f"neuron_{neuron_idx}"] = MSE
            animal_key = f"animal_{idx}"
            animal_MSE_dict[animal_key] = neuron_MSE_dict

        #             fig, axs = plt.subplots(1,3, figsize=(20,10))

        #             for i in range(empty_cell.shape[1]):
        #                 axs[1].imshow(reconstruction_full_animal[:,i,:], aspect='auto')
        #                 axs[1].set_title("TCA reconstruction")

        #                 axs[0].imshow(animal_tensor[:,i,:], aspect='auto')
        #                 axs[0].set_title("real activity")

        #                 axs[2].imshow(empty_cell[:,i,:], aspect='auto')
        #                 axs[2].set_title("our reconstruction")

        #             plt.show()

        MSE_dict[f"clusters_chosen_{clusters_chosen}"] = animal_MSE_dict
        x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = animal_X_pca_dict
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = animal_labels_dict
        indices_for_cluster_number[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_trial_indices
        Recon_by_cluster_av_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_reconstruction
        TCA_reconstructions_dict[f"clusters_chosen_{clusters_chosen}"] = reconstructed_TCA_tensor_per_animal_dict
        cluster_trial_mean_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_mean_dict

    return MSE_dict, x_pca_dict, labels_dict, indices_for_cluster_number, Recon_by_cluster_av_dict, TCA_reconstructions_dict, cluster_trial_mean_dict


import os
import pickle
import re


# Folder containing your models


# Pattern to extract the animal index
def extract_animal_index(filename):
    match = re.search(r'animal(\d+)00x\.pkl$', filename)
    return int(match.group(1)) if match else -1


def load_model(folder_path, cell_type="EC"):
    # List all matching .pkl files
    model_files = [
        f for f in os.listdir(folder_path)
        if f.startswith(f"model_{cell_type}_animal_40_animal") and f.endswith("00x.pkl")
    ]

    # Sort by extracted animal index
    model_files_sorted = sorted(model_files, key=extract_animal_index)

    # Load all models into a list
    EC_model_list_00x = []
    for fname in model_files_sorted:
        full_path = os.path.join(folder_path, fname)
        with open(full_path, "rb") as f:
            model = pickle.load(f)
            EC_model_list_00x.append(model)

    print(f"✅ Loaded {len(EC_model_list_00x)} {cell_type} models in order.")

    return EC_model_list_00x



def plot_side_by_side_MSE(cells_mean_list, cells_sem_list, animal_mean_list, animal_sem_list, labels, title="SliceTCA UMAP then Contiguous K-Means 00x"):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    colors = ['blue', 'orange', 'green']

    # Plot animal-level MSE
    for i, label in enumerate(labels):
        x = range(len(animal_mean_list[i]))
        mean = np.array(animal_mean_list[i])
        sem = np.array(animal_sem_list[i])
        axs[0].plot(x, mean, label=f"{label} AnimalTCA", color=colors[i])
        axs[0].fill_between(x, mean - sem, mean + sem, alpha=0.3, color=colors[i])

    axs[0].legend()
    axs[0].set_xlabel("Number of K-Means Clusters")
    axs[0].set_xticks(np.arange(0, 10), np.arange(2, 12))
    axs[0].set_ylabel("MSE")
    axs[0].set_ylim(0.7, 1.1)
    axs[0].set_title(f"Animal {title}")

    # Plot cell-level MSE
    for i, label in enumerate(labels):
        x = range(len(cells_mean_list[i]))
        mean = np.array(cells_mean_list[i])
        sem = np.array(cells_sem_list[i])
        axs[1].plot(x, mean, label=f"{label} CellTCA", color=colors[i])
        axs[1].fill_between(x, mean - sem, mean + sem, alpha=0.3, color=colors[i])

    axs[1].legend()
    axs[1].set_xlabel("Number of K-Means Clusters")
    axs[1].set_xticks(np.arange(0, 10), np.arange(2, 12))
    axs[1].set_ylabel("MSE")
    axs[1].set_ylim(0.7, 1.1)
    axs[1].set_title(f"Cell {title}")

    plt.tight_layout()
    plt.show()


def get_per_animal_MSE_regkmeans_00x(animal_model_list, residual_activity_dict_SST, max_clusters=12, display=True):
    MSE_dict = {}
    x_pca_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    for clusters_chosen in range(2, max_clusters):

        animal_MSE_dict = {}
        animal_X_pca_dict = {}
        animal_labels_dict = {}
        animal_cluster_trial_indices = {}
        animal_mean_cluster_latent_dict = {}
        reconstructed_TCA_tensor_per_animal_dict = {}
        animal_cluster_reconstruction = {}
        animal_cluster_mean_dict = {}

        for idx, animal_model in enumerate(animal_model_list):  #

            idx = idx + 1

            reconstruction_full_animal = animal_model.construct().numpy(force=True)
            reconstructed_TCA_tensor_per_animal_dict[f"animal_{animal_model}"] = reconstruction_full_animal  #

            if plot_latents:
                axes = slicetca.plot(animal_model,
                                     variables=('trial', 'neuron', 'time'),
                                     #   ticks=(None, None, np.linspace(0,50,3)), # we only want to modify the time ticks
                                     #   tick_labels=(None, None, np.linspace(0,50,3)),
                                     #   sorting_indices=(None, neuron_sorting_peak_time, None),
                                     quantile=0.99)

            f = animal_model.vectors[2][1].detach()
            f1 = f.permute(2, 0, 1)  # [5, 40, 212]

            f1 = f1.reshape(-1, f1.shape[-1]).T  # [200, 212]

            #             umap_model = umap.UMAP(n_components=3, random_state=0)
            #             X_umap = umap_model.fit_transform(f1)  #(trials,3)

            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            labels = kmeans.fit_predict(f1)

            #             algo = rpt.Binseg(model="l2", min_size=2).fit(X_umap)
            #             bkps = algo.predict(n_bkps=max_clusters)

            #             labels = np.zeros(X_umap.shape[0], dtype=int)
            #             start = 0
            #             for cluster_id, end in enumerate(bkps):
            #                 labels[start:end] = cluster_id
            #                 start = end

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(f1)

            animal_X_pca_dict[f"animal_{idx}"] = X_pca

            animal_labels_dict[f"animal_{idx}"] = labels

            #### get mean activity in cluster
            mean_cluster_trials = []
            animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, idx)

            ########### get the indicces and the mean latents
            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                #                 if display:
                # print(f"trial_indices {trial_indices}")
                if len(trial_indices) > 2:
                    cluster_trials = reconstruction_full_animal[trial_indices, :, :]
                    #                     if display:
                    # print(f"cluster_trials.shape {cluster_trials.shape}")
                    mean_cluster = cluster_trials.mean(axis=0)
                    valid_cluster_mean_trials_list.append(mean_cluster)
                    valid_cluster_indices.append((n, trial_indices))

                else:
                    print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
            animal_cluster_mean_dict[f"animal_{idx}"] = valid_cluster_mean_trials_list
            cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
            animal_cluster_trial_indices[f"animal_{idx}"] = cluster_trial_indices

            # print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

            ########### reconstruct the tensor with the avrage latent
            empty_cell = np.zeros(reconstruction_full_animal.shape)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :, :] = valid_cluster_mean_trials_list[i]
            animal_cluster_reconstruction[f"animal_{idx}"] = empty_cell

            ########### get the MSE
            neuron_MSE_dict = {}
            for neuron_idx, neuron in enumerate(residual_activity_dict_SST[f"animal_{idx}"]):
                reconstruction = empty_cell[:, neuron_idx, :]
                real_cell_activity = animal_tensor[:, neuron_idx, :]
                MSE = np.mean(np.square(real_cell_activity - reconstruction))
                neuron_MSE_dict[f"neuron_{neuron_idx}"] = MSE
            animal_key = f"animal_{idx}"
            animal_MSE_dict[animal_key] = neuron_MSE_dict

        #             fig, axs = plt.subplots(1,3, figsize=(20,10))

        #             for i in range(empty_cell.shape[1]):
        #                 axs[1].imshow(reconstruction_full_animal[:,i,:], aspect='auto')
        #                 axs[1].set_title("TCA reconstruction")

        #                 axs[0].imshow(animal_tensor[:,i,:], aspect='auto')
        #                 axs[0].set_title("real activity")

        #                 axs[2].imshow(empty_cell[:,i,:], aspect='auto')
        #                 axs[2].set_title("our reconstruction")

        #             plt.show()

        MSE_dict[f"clusters_chosen_{clusters_chosen}"] = animal_MSE_dict
        x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = animal_X_pca_dict
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = animal_labels_dict
        indices_for_cluster_number[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_trial_indices
        Recon_by_cluster_av_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_reconstruction
        TCA_reconstructions_dict[f"clusters_chosen_{clusters_chosen}"] = reconstructed_TCA_tensor_per_animal_dict
        cluster_trial_mean_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_mean_dict

    return MSE_dict, x_pca_dict, labels_dict, indices_for_cluster_number, Recon_by_cluster_av_dict, TCA_reconstructions_dict, cluster_trial_mean_dict

def get_per_animal_MSE_contigkmeans_00x(animal_model_list, residual_activity_dict_SST, max_clusters=12, display=True):
    MSE_dict = {}
    x_pca_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    for clusters_chosen in range(max_clusters):

        animal_MSE_dict = {}
        animal_X_pca_dict = {}
        animal_labels_dict = {}
        animal_cluster_trial_indices = {}
        animal_mean_cluster_latent_dict = {}
        reconstructed_TCA_tensor_per_animal_dict = {}
        animal_cluster_reconstruction = {}
        animal_cluster_mean_dict = {}

        for idx, animal_model in enumerate(animal_model_list):  #

            idx = idx + 1

            reconstruction_full_animal = animal_model.construct().numpy(force=True)
            reconstructed_TCA_tensor_per_animal_dict[f"animal_{animal_model}"] = reconstruction_full_animal  #

            if plot_latents:
                axes = slicetca.plot(animal_model,
                                     variables=('trial', 'neuron', 'time'),
                                     #   ticks=(None, None, np.linspace(0,50,3)), # we only want to modify the time ticks
                                     #   tick_labels=(None, None, np.linspace(0,50,3)),
                                     #   sorting_indices=(None, neuron_sorting_peak_time, None),
                                     quantile=0.99)

            f = animal_model.vectors[2][1].detach()
            f1 = f.permute(2, 0, 1)  # [5, 40, 212]

            f1 = f1.reshape(-1, f1.shape[-1]).T  # [200, 212]

            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)  # (trials,3)

            #             kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            #             labels = kmeans.fit_predict(f1)

            algo = rpt.Binseg(model="l2", min_size=2).fit(f1)
            bkps = algo.predict(n_bkps=clusters_chosen)

            labels = np.zeros(X_umap.shape[0], dtype=int)
            start = 0
            for cluster_id, end in enumerate(bkps):
                labels[start:end] = cluster_id
                start = end

            #             pca = PCA(n_components=2)
            #             X_pca = pca.fit_transform(X)

            #             animal_X_pca_dict[f"animal_{idx}"] = X_pca
            animal_X_pca_dict[f"animal_{idx}"] = X_umap
            animal_labels_dict[f"animal_{idx}"] = labels

            #### get mean activity in cluster
            mean_cluster_trials = []
            animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, idx)

            ########### get the indicces and the mean latents
            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                #                 if display:
                # print(f"trial_indices {trial_indices}")
                if len(trial_indices) > 2:
                    cluster_trials = reconstruction_full_animal[trial_indices, :, :]
                    #                     if display:
                    # print(f"cluster_trials.shape {cluster_trials.shape}")
                    mean_cluster = cluster_trials.mean(axis=0)
                    valid_cluster_mean_trials_list.append(mean_cluster)
                    valid_cluster_indices.append((n, trial_indices))

                else:
                    print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
            animal_cluster_mean_dict[f"animal_{idx}"] = valid_cluster_mean_trials_list
            cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
            animal_cluster_trial_indices[f"animal_{idx}"] = cluster_trial_indices

            # print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

            ########### reconstruct the tensor with the avrage latent
            empty_cell = np.zeros(reconstruction_full_animal.shape)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :, :] = valid_cluster_mean_trials_list[i]
            animal_cluster_reconstruction[f"animal_{idx}"] = empty_cell

            ########### get the MSE
            neuron_MSE_dict = {}
            for neuron_idx, neuron in enumerate(residual_activity_dict_SST[f"animal_{idx}"]):
                reconstruction = empty_cell[:, neuron_idx, :]
                real_cell_activity = animal_tensor[:, neuron_idx, :]
                MSE = np.mean(np.square(real_cell_activity - reconstruction))
                neuron_MSE_dict[f"neuron_{neuron_idx}"] = MSE
            animal_key = f"animal_{idx}"
            animal_MSE_dict[animal_key] = neuron_MSE_dict

        #             fig, axs = plt.subplots(1,3, figsize=(20,10))

        #             for i in range(empty_cell.shape[1]):
        #                 axs[1].imshow(reconstruction_full_animal[:,i,:], aspect='auto')
        #                 axs[1].set_title("TCA reconstruction")

        #                 axs[0].imshow(animal_tensor[:,i,:], aspect='auto')
        #                 axs[0].set_title("real activity")

        #                 axs[2].imshow(empty_cell[:,i,:], aspect='auto')
        #                 axs[2].set_title("our reconstruction")

        #             plt.show()

        MSE_dict[f"clusters_chosen_{clusters_chosen}"] = animal_MSE_dict
        x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = animal_X_pca_dict
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = animal_labels_dict
        indices_for_cluster_number[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_trial_indices
        Recon_by_cluster_av_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_reconstruction
        TCA_reconstructions_dict[f"clusters_chosen_{clusters_chosen}"] = reconstructed_TCA_tensor_per_animal_dict
        cluster_trial_mean_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_mean_dict

    return MSE_dict, x_pca_dict, labels_dict, indices_for_cluster_number, Recon_by_cluster_av_dict, TCA_reconstructions_dict, cluster_trial_mean_dict


def get_per_animal_MSE_UMAP_regkmeans_00x(animal_model_list, residual_activity_dict_SST, max_clusters=12, display=True):
    MSE_dict = {}
    x_pca_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    for clusters_chosen in range(2, max_clusters):

        animal_MSE_dict = {}
        animal_X_pca_dict = {}
        animal_labels_dict = {}
        animal_cluster_trial_indices = {}
        animal_mean_cluster_latent_dict = {}
        reconstructed_TCA_tensor_per_animal_dict = {}
        animal_cluster_reconstruction = {}
        animal_cluster_mean_dict = {}

        for idx, animal_model in enumerate(animal_model_list):  #

            idx = idx + 1

            reconstruction_full_animal = animal_model.construct().numpy(force=True)
            reconstructed_TCA_tensor_per_animal_dict[f"animal_{animal_model}"] = reconstruction_full_animal  #

            if plot_latents:
                axes = slicetca.plot(animal_model,
                                     variables=('trial', 'neuron', 'time'),
                                     #   ticks=(None, None, np.linspace(0,50,3)), # we only want to modify the time ticks
                                     #   tick_labels=(None, None, np.linspace(0,50,3)),
                                     #   sorting_indices=(None, neuron_sorting_peak_time, None),
                                     quantile=0.99)

            f = animal_model.vectors[2][1].detach()
            f1 = f.permute(2, 0, 1)  # [5, 40, 212]

            f1 = f1.reshape(-1, f1.shape[-1]).T  # [200, 212]

            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)  # (trials,3)

            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            labels = kmeans.fit_predict(X_umap)

            #             algo = rpt.Binseg(model="l2", min_size=2).fit(X_umap)
            #             bkps = algo.predict(n_bkps=max_clusters)

            #             labels = np.zeros(X_umap.shape[0], dtype=int)
            #             start = 0
            #             for cluster_id, end in enumerate(bkps):
            #                 labels[start:end] = cluster_id
            #                 start = end

            #                 pca = PCA(n_components=2)
            #                 X_pca = pca.fit_transform(X)

            #                 animal_X_pca_dict[f"animal_{idx}"] = X_pca
            animal_X_pca_dict[f"animal_{idx}"] = X_umap
            animal_labels_dict[f"animal_{idx}"] = labels

            #### get mean activity in cluster
            mean_cluster_trials = []
            animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, idx)

            ########### get the indicces and the mean latents
            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                #                 if display:
                # print(f"trial_indices {trial_indices}")
                if len(trial_indices) > 2:
                    cluster_trials = reconstruction_full_animal[trial_indices, :, :]
                    #                     if display:
                    print(f"cluster_trials.shape {cluster_trials.shape}")
                    mean_cluster = cluster_trials.mean(axis=0)
                    valid_cluster_mean_trials_list.append(mean_cluster)
                    valid_cluster_indices.append((n, trial_indices))

                else:
                    print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
            animal_cluster_mean_dict[f"animal_{idx}"] = valid_cluster_mean_trials_list
            cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
            animal_cluster_trial_indices[f"animal_{idx}"] = cluster_trial_indices

            # print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

            ########### reconstruct the tensor with the avrage latent
            empty_cell = np.zeros(reconstruction_full_animal.shape)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :, :] = valid_cluster_mean_trials_list[i]
            animal_cluster_reconstruction[f"animal_{idx}"] = empty_cell

            ########### get the MSE
            neuron_MSE_dict = {}
            for neuron_idx, neuron in enumerate(residual_activity_dict_SST[f"animal_{idx}"]):
                reconstruction = empty_cell[:, neuron_idx, :]
                real_cell_activity = animal_tensor[:, neuron_idx, :]
                MSE = np.mean(np.square(real_cell_activity - reconstruction))
                neuron_MSE_dict[f"neuron_{neuron_idx}"] = MSE
            animal_key = f"animal_{idx}"
            animal_MSE_dict[animal_key] = neuron_MSE_dict

        #             fig, axs = plt.subplots(1,3, figsize=(20,10))

        #             for i in range(empty_cell.shape[1]):
        #                 axs[1].imshow(reconstruction_full_animal[:,i,:], aspect='auto')
        #                 axs[1].set_title("TCA reconstruction")

        #                 axs[0].imshow(animal_tensor[:,i,:], aspect='auto')
        #                 axs[0].set_title("real activity")

        #                 axs[2].imshow(empty_cell[:,i,:], aspect='auto')
        #                 axs[2].set_title("our reconstruction")

        #             plt.show()

        MSE_dict[f"clusters_chosen_{clusters_chosen}"] = animal_MSE_dict
        x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = animal_X_pca_dict
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = animal_labels_dict
        indices_for_cluster_number[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_trial_indices
        Recon_by_cluster_av_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_reconstruction
        TCA_reconstructions_dict[f"clusters_chosen_{clusters_chosen}"] = reconstructed_TCA_tensor_per_animal_dict
        cluster_trial_mean_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_mean_dict

    return MSE_dict, x_pca_dict, labels_dict, indices_for_cluster_number, Recon_by_cluster_av_dict, TCA_reconstructions_dict, cluster_trial_mean_dict

####### cell by cell stuff:

def get_clusters_for_cell(internals_SST_regkmeans_00x, residual_activity_dict_SST, animal_id=1, cell_id=0):

    internals = internals_SST_regkmeans_00x[f"animal_{animal_id}"][f"cell_{cell_id}"]

    actual_activity_list = []
    for key, value in residual_activity_dict_SST[f"animal_{animal_id}"].items():
        actual_activity_list.append(value)
    for cluster in range(len(internals["Recon_by_cluster_av_list"])):
        fig, axs = plt.subplots(1, 4, figsize=(20,10))
        axs[0].imshow(actual_activity_list[cell_id].T, aspect="auto")
        axs[0].set_xlabel("Position Bin")
        axs[0].set_ylabel("Trial")
        axs[0].set_title("Actual Activity")
        TCA_reco = internals["TCA_reconstructions_list"][cluster][:,0,:]
        axs[1].imshow(TCA_reco, aspect="auto")
        axs[1].set_xlabel("Position Bin")
        axs[1].set_ylabel("Trial")
        axs[1].set_title("Cell Slice TCA Reconstruction")
        reco_cell = internals["Recon_by_cluster_av_list"][cluster]
        axs[2].imshow(reco_cell, aspect="auto")
        axs[2].set_xlabel("Position Bin")
        axs[2].set_ylabel("Trial")
        axs[2].set_title("Cluster Reconstruction")

        means = internals["cluster_trial_mean_list"][cluster]

        for i in range(len(means)):
                axs[3].plot(means[i])


        axs[3].set_title("Cluster Means")
        axs[3].set_xlabel("Position Bin")
        axs[3].set_ylabel("DF/F")