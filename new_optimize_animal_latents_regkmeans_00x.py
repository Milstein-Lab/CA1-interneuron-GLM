
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
import warnings


def get_animal_model_reconstruction_dict_00x_regkmean(animal_model, tensor_for_animal, animal_id, max_clusters=12, display=False):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    per_cell_internals_dict = {}

    reconstruction_full_animal = animal_model.construct().numpy(force=True)

    for cell in range(reconstruction_full_animal.shape[1]):
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

        f = animal_model.vectors[2][1].detach()  # [20, 78, 192]
        f1 = f.permute(1, 0, 2)  # [78, 20, 192] --> trials × latents × neurons
        f1 = f1.reshape(f1.shape[0], -1)  # [78, 20 * 192] = [78, 3840]

        print("Final f1.shape:", f1.shape)

        for clusters_chosen in range(1, max_clusters):

            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            labels = kmeans.fit_predict(f1)

            from sklearn.decomposition import PCA
            X_pca = PCA(n_components=3).fit_transform(f1)
            x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = X_pca
            labels_dict[f"clusters_chosen_{clusters_chosen}"] = labels

            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            cluster_trial_indices = {}

            plt.figure()
            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                cluster_trial_indices[n] = trial_indices
                if len(trial_indices) > 2:
                    cluster_trials = reconstructed_cell[trial_indices, :]

                    mean_cluster = cluster_trials.mean(axis=0)
                    plt.plot(mean_cluster)
                    valid_cluster_mean_trials_list.append(mean_cluster)
                    valid_cluster_indices.append((n, trial_indices))

            plt.show()

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

    internals_dict = get_animal_model_reconstruction_dict_00x_regkmean(animal_model, tensor_for_animal, animal_id, max_clusters=8, display=False)

    # internals_dict = get_per_animal_sliceTCA_reconstruction_MSE_kmean(animal_model, tensor_for_animal, residual_activity_dict, animal_id, max_clusters=12, display=False)

    # save_dir = fr"/scratch/msf157/data/ca1_data2/new_animal_EC_model_ranks{ranks}_regkmean_00x"
    save_dir = fr"/ocean/projects/bio240068p/mfinch/CA1-inter/data/new_animal_EC_model_ranks{ranks}_regkmean_00x"

    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"MSE_EC_animal_latent_{ranks}_animal{animal_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump([animal_model, internals_dict], f)

    print(f"Saved model for animal {animal_id} to {save_path}")




###############################################################################