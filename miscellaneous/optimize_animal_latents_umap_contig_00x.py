
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
import umap


def get_real_animal_tensor(residual_activity_dict_SST, animal_num=1):
    real_animal_activity = residual_activity_dict_SST[f'animal_{animal_num}']
    real_animal_data_list = []
    for neuron in real_animal_activity:
        cell_activity = real_animal_activity[neuron]
        real_animal_data_list.append(cell_activity)
    animal_tensor = np.array(real_animal_data_list)
    animal_tensor = animal_tensor.transpose(2, 0, 1)

    return animal_tensor


def get_per_animal_sliceTCA_reconstruction_MSE_kmean(animal_model, tensor_for_animal, residual_activity_dict_SST, animal_id, max_clusters=12, display=False):
    MSE_dict = {}
    x_umap_dict = {}
    labels_dict = {}
    indices_for_cluster_number = {}
    TCA_reconstructions_dict = {}
    Recon_by_cluster_av_dict = {}
    cluster_trial_mean_dict = {}

    for clusters_chosen in range(max_clusters):

        animal_MSE_dict = {}
        animal_X_umap_dict = {}
        animal_labels_dict = {}
        animal_cluster_trial_indices = {}
        animal_mean_cluster_latent_dict = {}
        reconstructed_TCA_tensor_per_animal_dict = {}
        animal_cluster_reconstruction = {}
        animal_cluster_mean_dict = {}

        # for idx, animal_model in enumerate(animal_model_list):
        #
        #     idx = idx + 1

        reconstruction_full_animal = animal_model.construct().numpy(force=True)
        reconstructed_TCA_tensor_per_animal_dict[f"animal_{animal_id}"] = reconstruction_full_animal

        f = animal_model.vectors[2][1].detach()
        f1 = f.permute(2, 0, 1)  # [5, 40, 212]

        f1 = f1.reshape(-1, f1.shape[-1]).T  # [200, 212]

        f1 = torch.abs(f1)

        umap_model = umap.UMAP(n_components=3, random_state=0)
        X_umap = umap_model.fit_transform(f1)  # (trials,3)

        algo = rpt.Binseg(model="l2", min_size=3).fit(X_umap)
        bkps = algo.predict(n_bkps=max_clusters)

        labels = np.zeros(X_umap.shape[0], dtype=int)
        start = 0
        for cluster_id, end in enumerate(bkps):
            labels[start:end] = cluster_id
            start = end

        # X_umap_list.append(X_umap)
        # labels_list.append(labels)

        start = 0
        for cluster_id, end in enumerate(bkps):
            labels[start:end] = cluster_id
            start = end

        pca_model = PCA(n_components=3)
        X_pca = pca_model.fit_transform(f1)

        animal_X_umap_dict[f"animal_{animal_id}"] = X_umap
        animal_labels_dict[f"animal_{animal_id}"] = labels

        mean_cluster_trials = []
        # animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, animal_id)

        valid_cluster_mean_trials_list = []
        valid_cluster_indices = []
        for n in range(clusters_chosen):
            trial_indices = np.where(labels == n)[0]
            if display:
                print(f"trial_indices {trial_indices}")
            if len(trial_indices) > 2:
                cluster_trials = reconstruction_full_animal[trial_indices, :, :]
                if display:
                    print(f"cluster_trials.shape {cluster_trials.shape}")
                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)
                valid_cluster_indices.append((n, trial_indices))

            else:
                print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
        animal_cluster_mean_dict[f"animal_{animal_id}"] = valid_cluster_mean_trials_list
        cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
        animal_cluster_trial_indices[f"animal_{animal_id}"] = cluster_trial_indices

        ########### reconstruct the tensor with the avrage latent
        empty_cell = np.zeros(reconstruction_full_animal.shape)
        for i, (n, trials) in enumerate(valid_cluster_indices):
            empty_cell[trials, :, :] = valid_cluster_mean_trials_list[i]
        animal_cluster_reconstruction[f"animal_{animal_id}"] = empty_cell

        ########### get the MSE
        neuron_MSE_dict = {}
        for neuron_idx in range(tensor_for_animal.shape[1]):
            reconstruction = empty_cell[:, neuron_idx, :]
            real_cell_activity = tensor_for_animal[:, neuron_idx, :]
            MSE = np.mean(np.square(real_cell_activity.detach().numpy() - reconstruction))
            neuron_MSE_dict[f"neuron_{neuron_idx}"] = MSE
        animal_key = f"animal_{animal_id}"
        animal_MSE_dict[animal_key] = neuron_MSE_dict



        MSE_dict[f"clusters_chosen_{clusters_chosen}"] = animal_MSE_dict
        x_umap_dict[f"clusters_chosen_{clusters_chosen}"] = animal_X_umap_dict
        labels_dict[f"clusters_chosen_{clusters_chosen}"] = animal_labels_dict
        indices_for_cluster_number[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_trial_indices
        Recon_by_cluster_av_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_reconstruction
        TCA_reconstructions_dict[f"clusters_chosen_{clusters_chosen}"] = reconstructed_TCA_tensor_per_animal_dict
        cluster_trial_mean_dict[f"clusters_chosen_{clusters_chosen}"] = animal_cluster_mean_dict

    internals_dict = {
    "MSE_dict" : MSE_dict,
    "x_umap_dict" : x_umap_dict,
    "labels_dict" : labels_dict,
    "indices_for_cluster_number" : indices_for_cluster_number,
    "TCA_reconstructions_dict" : TCA_reconstructions_dict,
    "Recon_by_cluster_av_dict" : Recon_by_cluster_av_dict,
    "cluster_trial_mean_dict" : cluster_trial_mean_dict,
    }

    return internals_dict


# MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
# Load data
#filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
filename = "EC_GLM"

animal_id = int(sys.argv[1])       # Provided via command-line argument
ranks = int(sys.argv[2])

# animal_id = 1
# ranks = 40

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

    # cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    tensor_for_animal.requires_grad_()
    components, animal_model = slicetca.decompose(tensor_for_animal,
                                       number_components=(0, 0, ranks),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_000,
                                           seed=0)

    internals_dict = get_per_animal_sliceTCA_reconstruction_MSE_kmean(animal_model, tensor_for_animal, residual_activity_dict, animal_id, max_clusters=5, display=False)

    save_dir = fr"/scratch/msf157/data/ca1_data2/animal_EC_model_ranks{ranks}_umap_contig_00x"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"MSE_EC_animal_latent_{ranks}_animal{animal_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump([animal_model, internals_dict], f)

    print(f"Saved model for animal {animal_id} to {save_path}")




###############################################################################