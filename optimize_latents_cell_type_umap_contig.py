
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


def get_per_cell_sliceTCA_reconstruction_UMAP_contig(model, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=True):
    animal_id = animal_id+1
    MSE_list = []
    X_umap_list = []
    labels_list = []
    indices_for_cluster_number = []
    TCA_reconstructions_list = []
    Recon_by_cluster_av_list = []
    cluster_trial_mean_list = []

    for clusters_chosen in range(max_clusters):

        #         cell_model = SST_every_cell_model_list[cell_id]
        reconstruction_cell = model.construct().numpy(force=True)
        TCA_reconstructions_list.append(reconstruction_cell)

        w1 = model.vectors[0][0].detach()
        X = w1.T

        umap_model = umap.UMAP(n_components=3, random_state=0)
        X_umap = umap_model.fit_transform(X)  # (trials,3)

        #         kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
        #         labels = kmeans.fit_predict(X_umap)

        algo = rpt.Binseg(model="l2", min_size=6).fit(X_umap)
        bkps = algo.predict(n_bkps=max_clusters)

        labels = np.zeros(X_umap.shape[0], dtype=int)
        start = 0
        for cluster_id, end in enumerate(bkps):
            labels[start:end] = cluster_id
            start = end

        X_umap_list.append(X_umap)
        labels_list.append(labels)

        mean_cluster_trials = []
        animal_tensor = get_real_animal_tensor(residual_activity_dict_SST, animal_id)
        neuron_activity = animal_tensor[:, cell_id, :]

        valid_cluster_mean_trials_list = []
        valid_cluster_indices = []
        for n in range(clusters_chosen):
            trial_indices = np.where(labels == n)[0]
            if display:
                print(f"trial_indices {trial_indices}")
            if len(trial_indices) > 2:
                cluster_trials = reconstruction_cell[trial_indices, :]
                if display:
                    print(f"cluster_trials.shape {cluster_trials.shape}")
                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)
                valid_cluster_indices.append((n, trial_indices))

            else:
                print(f"Skipping cluster {n} (only {len(trial_indices)} trials)")
        cluster_trial_mean_list.append(valid_cluster_mean_trials_list)
        cluster_trial_indices = {n: np.where(labels == n)[0] for n in range(clusters_chosen)}
        cluster_trial_mean_list.append(cluster_trial_indices)

        empty_cell = np.zeros(neuron_activity.shape)
        for i, (n, trials) in enumerate(valid_cluster_indices):
            empty_cell[trials, :] = valid_cluster_mean_trials_list[i]
        Recon_by_cluster_av_list.append(empty_cell)

        neuron_MSE = np.mean(np.square(neuron_activity - empty_cell))
        MSE_list.append(neuron_MSE)

        internals_dict = {
            "MSE_list" : MSE_list,
            "x_umap_list" : x_umap_list,
            "labels_list" : labels_list,
            "indices_for_cluster_number" : indices_for_cluster_number,
            "Recon_by_cluster_av_list" : Recon_by_cluster_av_list,
            "TCA_reconstructions_list" : TCA_reconstructions_list,
            "cluster_trial_mean_list" : cluster_trial_mean_list,

        }

    return internals_dict

    #return MSE_list, X_umap_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list


# MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
# Load data
#filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
filename = "EC_GLM"

cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
animal_id = int(sys.argv[2])       # Provided via command-line argument
ranks = int(sys.argv[3])

# cell_id = 3
# animal_id = 2
# ranks = 40

filepath = os.path.join("datasets", filename + ".mat")
activity_dict, factors_dict = ut.preprocess_data(filepath)
filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)

# Convert neural activity to tensors
tensor_list_by_animal_all_SST = []
for animal in residual_activity_dict:
    neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
    neural_data_tensor = torch.tensor(neural_data / neural_data.std())
    tensor_list_by_animal_all_SST.append(neural_data_tensor)

if __name__ == "__main__":

    tensor_for_animal = tensor_list_by_animal_all_SST[animal_id]

    cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    components, model = slicetca.decompose(cell_of_interest,
                                       number_components=(ranks, 0, 0),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_000,
                                           seed=0)

    internals_dict = get_per_cell_sliceTCA_reconstruction_UMAP_contig(model, residual_activity_dict, max_clusters=4, animal_id=animal_id, cell_id=cell_id, display=False)

    save_dir = r"/scratch/msf157/data/CA1-inter/UMAP_contig_EC_reconstruct_MSE_data"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"UMAP_contig_MSE_EC_cell_latent_{ranks}_animal{animal_id}_cell_id{cell_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(internals_dict, f)

    print(f"Saved model for animal {animal_id} cell {cell_id} to {save_path}")




###############################################################################