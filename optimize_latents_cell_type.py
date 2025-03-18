
import sys
import torch
import slicetca
import pickle
import os
import utils as ut

# Ensure task ID is provided
if len(sys.argv) < 2:
    print("Usage: python optimize_latents_cell_type.py <animal_index>")
    sys.exit(1)

# Get animal index from command-line argument
i = int(sys.argv[1])  # Animal index
print(f"Running optimization for animal index {i}")

# Load data
filename = "SSTindivsomata_GLM"
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

# Ensure valid index
if i >= len(tensor_list_by_animal_all_SST):
    print(f"Error: Animal index {i} is out of range. Maximum index allowed: {len(tensor_list_by_animal_all_SST) - 1}")
    sys.exit(1)

# Run SlicerTCA grid search for this specific animal
loss_grid, seed_grid = slicetca.grid_search(
    tensor_list_by_animal_all_SST[i],
    min_ranks=[2, 0, 0],  # Modify if needed
    max_ranks=[3, 0, 0],  # Modify if needed
    seed=0,
    min_std=10 ** -5,
    learning_rate=2 * 10 ** -3,
    max_iter=15_000,
    positive=True
)

# Save results
save_dir = r"/scratch/msf157/data/CA1-inter"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"loss_dict_SST_latent_2_32_{i}.pkl")
with open(save_path, "wb") as f:
    pickle.dump(loss_grid, f)

print(f"Saved loss_dict for animal {i} to {save_path}")





# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import torch
# import slicetca
# import pickle
# import sys
#
#
# import utils as ut
# import plot as pt
# plt.rcParams.update({'font.size': 12,
#                      'axes.spines.right': False,
#                      'axes.spines.top':   False,
#                      'legend.frameon':    False,})
#
# # # we define the tensor
# # reconstructed_noisy_tensor = torch.relu(torch.tensor(reconstruction_full, device=device)+torch.randn(reconstruction_full.shape, device=device)*0.1)
#
# #filename = "SSTindivsomata_GLM"
# #filename = "NDNFindivsomata_GLM"
# filename = "EC_GLM"
#
# filepath = os.path.join("datasets", filename + ".mat")
# activity_dict, factors_dict = ut.preprocess_data(filepath)
# filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
#
# GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
# residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)
#
# # neural_data_tensor_all = neural_data_tensor_all.unsqueeze(1)
#
# tensor_list_by_animal_all_SST = []
# for animal in residual_activity_dict:
#     neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
#     neural_data_tensor = torch.tensor(neural_data / neural_data.std())
#     tensor_list_by_animal_all_SST.append(neural_data_tensor)
#
# loss_dict = {}
#
# if __name__ == "__main__":
#
#     # Parse command-line arguments
#     i = int(sys.argv[1])  # Animal index
#     # min_rank_value = int(sys.argv[2])  # First value in min_ranks
#     # max_rank_value = int(sys.argv[3])  # First value in max_ranks
#
#     # Ensure the index is within bounds
#     if i >= len(tensor_list_by_animal_all_SST):
#         print(f"Skipping index {i}, out of range.")
#         sys.exit(0)
#
#     # Run grid search with user-specified ranks
#     loss_grid, seed_grid = slicetca.grid_search(
#         tensor_list_by_animal_all_SST[i],
#         min_ranks=[2, 0, 0],  # Controlled manually
#         max_ranks=[3, 0, 0],  # Controlled manually
#         seed=0,
#         min_std=10 ** -5,
#         learning_rate=2 * 10 ** -3,
#         max_iter=15_000,
#         positive=True
#     )
#
#     print(f"Animal {i} | min_rank={min_rank_value} | max_rank={max_rank_value}")
#     loss_dict = {i: loss_grid}
#
#     # Save results
#     save_dir = r"/scratch/msf157/data/CA1-inter"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"loss_dict_SST_latent_{min_rank_value}_{max_rank_value}.pkl")
#
#     with open(save_path, "wb") as f:
#         pickle.dump(loss_dict, f)
#
#     print(f"Saved loss_dict to {save_path}")

#########################################

#     for i in range(len(tensor_list_by_animal_all_SST)):
#     #for i in range(2):
#         loss_grid, seed_grid = slicetca.grid_search(tensor_list_by_animal_all_SST[i],
#                                                     min_ranks=[2, 0, 0],
#                                                     max_ranks=[3, 0, 0],
#                                                     seed=0,
#                                                     min_std=10 ** -5,
#                                                     learning_rate=2 * 10 ** -3,
#                                                     max_iter=15_000,
#                                                     positive=True)
#         print(f"loss_grid {loss_grid}")
#
#         loss_dict[i] = loss_grid
#
# print(f"loss_dict {loss_dict}")
#
# import pickle
# import os
#
# # Define directory and ensure it exists
# save_dir = r"/scratch/msf157/data/CA1-inter"
# os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
#
# # Define full file path
# save_path = os.path.join(save_dir, "loss_dict_SST_latent_2_32.pkl")
#
# # Save loss_dict as a pickle file
# with open(save_path, "wb") as f:
#     pickle.dump(loss_dict, f)
#
# print(f"Saved loss_dict to {save_path}")
#




###########################################






###########################################################

#
# import os
# import sys
# import torch
# import slicetca
# import pickle
# import utils as ut
#
# # Get the animal index from SLURM
# animal_index = int(sys.argv[1])  # Passed as argument
#
# # Load Data
# filename = "SSTindivsomata_GLM"
# filepath = os.path.join("datasets", filename + ".mat")
# activity_dict, factors_dict = ut.preprocess_data(filepath)
# filtered_factors_dict = ut.subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])
# GLM_params, predicted_activity_dict = ut.fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
# residual_activity_dict = ut.get_residual_activity_dict(activity_dict, predicted_activity_dict)
#
# # Get the animal key
# animal_keys = list(residual_activity_dict.keys())
# animal = animal_keys[animal_index]
#
# # Process animal
# neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
# neural_data_tensor = torch.tensor(neural_data / neural_data.std())
#
# loss_grid, seed_grid = slicetca.grid_search(neural_data_tensor,
#                                             min_ranks=[2, 0, 0],
#                                             max_ranks=[3, 0, 0],
#                                             seed=0,
#                                             min_std=10 ** -5,
#                                             learning_rate=2 * 10 ** -3,
#                                             max_iter=15_000,
#                                             positive=True)
#
# # Save results separately
# save_dir = "/scratch/msf157/data/CA1-inter"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, f"loss_dict_SST_latent_2_32_{animal}.pkl")
#
# with open(save_path, "wb") as f:
#     pickle.dump({animal: loss_grid}, f)
#
# print(f"Saved loss_dict for {animal} to {save_path}")

