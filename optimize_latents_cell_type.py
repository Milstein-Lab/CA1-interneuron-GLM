import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca
import pickle

import utils as ut
import plot as pt
plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

# # we define the tensor
# reconstructed_noisy_tensor = torch.relu(torch.tensor(reconstruction_full, device=device)+torch.randn(reconstruction_full.shape, device=device)*0.1)

filename = "SSTindivsomata_GLM"
# filename = "NDNFindivsomata_GLM"
#filename = "EC_GLM"

filepath = os.path.join("datasets", filename + ".mat")
activity_dict, factors_dict = ut.preprocess_data(filepath)
filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)

# neural_data_tensor_all = neural_data_tensor_all.unsqueeze(1)

tensor_list_by_animal_all_SST = []
for animal in residual_activity_dict:
    neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
    neural_data_tensor = torch.tensor(neural_data / neural_data.std())
    tensor_list_by_animal_all_SST.append(neural_data_tensor)

loss_dict = {}

if __name__ == "__main__":
    # for i in range(len(tensor_list_by_animal_all_SST)):
    for i in range(2):
        loss_grid, seed_grid = slicetca.grid_search(tensor_list_by_animal_all_SST[i],
                                                    min_ranks=[2, 0, 0],
                                                    max_ranks=[3, 0, 0],
                                                    seed=0,
                                                    min_std=10 ** -5,
                                                    learning_rate=2 * 10 ** -3,
                                                    max_iter=15_000,
                                                    positive=True)
        print(f"loss_grid {loss_grid}")

        loss_dict[i] = loss_grid

print(f"loss_dict {loss_dict}")

import pickle
import os

# Define directory and ensure it exists
save_dir = r"/scratch/msf157/data/CA1-inter"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Define full file path
save_path = os.path.join(save_dir, "loss_dict_SST_latent_2_32.pkl")

# Save loss_dict as a pickle file
with open(save_path, "wb") as f:
    pickle.dump(loss_dict, f)

print(f"Saved loss_dict to {save_path}")



#
# from mpi4py import MPI
# import os
# import torch
# import slicetca
# import utils as ut
#
# # Initialize MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()  # Process ID
# size = comm.Get_size()  # Total number of processes
#
# # Define filename
# filename = "SSTindivsomata_GLM"
# filepath = os.path.join("datasets", filename + ".mat")
#
# # Only rank 0 loads and distributes data
# if rank == 0:
#     activity_dict, factors_dict = ut.preprocess_data(filepath)
#     filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
#     GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
#     residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)
#
#     # Prepare tensor list and distribute work across MPI workers
#     animal_keys = list(residual_activity_dict.keys())
#     num_animals = len(animal_keys)
# else:
#     residual_activity_dict = None
#     animal_keys = None
#     num_animals = None
#
# # Broadcast metadata to all ranks
# num_animals = comm.bcast(num_animals, root=0)
# animal_keys = comm.bcast(animal_keys, root=0)
#
# # Distribute data among ranks
# # Each worker gets a subset of animals based on its rank
# local_animals = [animal_keys[i] for i in range(rank, num_animals, size)]
#
# # Each worker processes its assigned animals
# local_loss_dict = {}
#
# for animal in local_animals:
#     neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
#
#     # Run grid search on the assigned animal
#     loss_grid, seed_grid = slicetca.grid_search(neural_data,
#                                                 min_ranks=[2, 0, 0],
#                                                 max_ranks=[2, 0, 0],
#                                                 seed=0,
#                                                 min_std=10 ** -5,
#                                                 learning_rate=2 * 10 ** -3,
#                                                 max_iter=15_000,
#                                                 positive=True)
#
#     print(f"Rank {rank} processed animal {animal} with loss_grid {loss_grid}")
#     local_loss_dict[animal] = loss_grid
#
# # Gather results at rank 0
# all_loss_dict = comm.gather(local_loss_dict, root=0)
#
# # Rank 0 merges all results
# if rank == 0:
#     final_loss_dict = {k: v for d in all_loss_dict for k, v in d.items()}
#     print("Final Loss Dictionary:", final_loss_dict)
