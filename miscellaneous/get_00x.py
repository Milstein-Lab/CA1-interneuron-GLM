
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


# MSE_list, x_pca_list, labels_list, indices_for_cluster_number, Recon_by_cluster_av_list, TCA_reconstructions_list, cluster_trial_mean_list = get_per_cell_sliceTCA_reconstruction_MSE(SST_every_cell_model_list, residual_activity_dict_SST, max_clusters=12, animal_id=1, cell_id=2, display=False)
# Load data
#filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
filename = "EC_GLM"

# cell_id = int(sys.argv[1])         # SLURM_ARRAY_TASK_ID
# animal_id = int(sys.argv[2])       # Provided via command-line argument
# ranks = int(sys.argv[3])

cell_id = 3
animal_id = 5
ranks = 40

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
    print(tensor_for_animal.shape)

    cell_of_interest = tensor_for_animal[:,cell_id,:].unsqueeze(1)
    components, model = slicetca.decompose(cell_of_interest,
                                       number_components=(0, 0, ranks),  # (trials, neurons, time bins)
                                       positive=True,
                                       learning_rate=1 * 10 ** -3,
                                       min_std=10 ** -5, #max_iter=150,
                                       max_iter=15_000,
                                           seed=0)

    # internals_dict = get_per_cell_sliceTCA_reconstruction_UMAP_contig(model, residual_activity_dict, max_clusters=4, animal_id=animal_id, cell_id=cell_id, display=False)

    save_dir = r"/scratch/msf157/data/CA1-inter/models_EC_00x"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"00x_model_EC_cell_latent_{ranks}_animal{animal_id}_cell_id{cell_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model for animal {animal_id} cell {cell_id} to {save_path}")




###############################################################################