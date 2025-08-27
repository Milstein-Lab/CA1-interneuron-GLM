
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


################## UMAP then contiguous KMEAMS ##############


def get_real_animal_tensor(residual_activity_dict_SST, animal_num=1):
    real_animal_activity = residual_activity_dict_SST[f'animal_{animal_num}']
    real_animal_data_list = []
    for neuron in real_animal_activity:
        cell_activity = real_animal_activity[neuron]
        real_animal_data_list.append(cell_activity)
    animal_tensor = np.array(real_animal_data_list)
    animal_tensor = animal_tensor.transpose(2, 0, 1)

    return animal_tensor


def get_per_cell_sliceTCA_reconstruction_contig_00x(model, tensor_for_animal, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=True):
    MSE_list = []
    #     X_umap_list = []
    labels_list = []
    indices_for_cluster_number_list = []
    TCA_reconstructions_list = []
    Recon_by_cluster_av_list = []
    cluster_trial_mean_list = []

    for clusters_chosen in range(max_clusters):

        reconstruction_cell = model.construct().numpy(force=True)
        TCA_reconstructions_list.append(reconstruction_cell)

        w1 = model.vectors[0][0].detach()
        X = torch.abs(w1.T)

        algo = rpt.Binseg(model="l2", min_size=3).fit(X)
        bkps = algo.predict(n_bkps=clusters_chosen)

        labels = np.zeros(X.shape[0], dtype=int)

        start = 0
        for cluster_id, end in enumerate(bkps):
            labels[start:end] = cluster_id
            start = end

        #         X_umap_list.append(X_umap)
        labels_list.append(labels)

        # animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, animal_id)
        neuron_activity = tensor_for_animal[:, cell_id, :].detach().numpy()

        valid_cluster_mean_trials_list = []
        valid_cluster_indices = []
        for n in range(clusters_chosen):
            trial_indices = np.where(labels == n)[0]

            if len(trial_indices) > 2:

                cluster_trials = reconstruction_cell[trial_indices, 0, :]  # shape (num_trials, time)

                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)

                valid_cluster_indices.append((n, trial_indices))

            else:
                print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
        cluster_trial_mean_list.append(valid_cluster_mean_trials_list)
        cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
        indices_for_cluster_number_list.append(cluster_trial_indices)

        empty_cell = np.zeros(neuron_activity.shape)

        for i, (n, trials) in enumerate(valid_cluster_indices):
            empty_cell[trials, :] = valid_cluster_mean_trials_list[i]
        Recon_by_cluster_av_list.append(empty_cell)

        neuron_MSE = np.mean(np.square(neuron_activity - empty_cell))
        MSE_list.append(neuron_MSE)

    internals_dict = {
        "MSE_list": MSE_list,
        #         "X_umap_list" : X_umap_list,
        "labels_list": labels_list,
        "indices_for_cluster_number_list": indices_for_cluster_number_list,
        "TCA_reconstructions_list": TCA_reconstructions_list,
        "Recon_by_cluster_av_list": Recon_by_cluster_av_list,
        "cluster_trial_mean_list": cluster_trial_mean_list,

    }

    return internals_dict

    # return MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list

# MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
# Load data
#filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
filename = "EC_GLM"

cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
animal_id = int(sys.argv[2])       # Provided via command-line argument
ranks = int(sys.argv[3])

# cell_id = 3
# animal_id = 0
# ranks = 40

filepath = os.path.join("datasets", filename + ".mat")
activity_dict, factors_dict = ut.preprocess_data(filepath)
filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)

# # Convert neural activity to tensors
# tensor_list_by_animal_all_SST = []
# for animal in residual_activity_dict:
#     neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
#     neural_data_tensor = torch.tensor(neural_data / neural_data.std())
#     tensor_list_by_animal_all_SST.append(neural_data_tensor)


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

    cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    cell_of_interest.requires_grad_()
    components, model = slicetca.decompose(cell_of_interest,
                                       number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_000,
                                           seed=0)

    internals_dict = get_per_cell_sliceTCA_reconstruction_contig_00x(model, tensor_for_animal, residual_activity_dict, max_clusters=5, animal_id=animal_id, cell_id=cell_id, display=False)

    save_dir = fr"/scratch/msf157/data/ca1_data2/EC_model_ranks{ranks}_contigkmeans_x00"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"MSE_EC_cell_latent_{ranks}_animal{animal_id}_cell_id{cell_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump([model, internals_dict], f)

    print(f"Saved model for animal {animal_id} cell {cell_id} to {save_path}")




###############################################################################