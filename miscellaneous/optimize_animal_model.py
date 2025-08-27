
import sys
import torch
import slicetca
import pickle
import os
import utils as ut

# Load data
filename = "SSTindivsomata_GLM"
#filename = "NDNFindivsomata_GLM"
#filename = "EC_GLM"
#
# animal_id = int(sys.argv[1])       # Provided via command-line argument
# ranks = int(sys.argv[2])


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

    model_list = []
    #tensor_for_animal = tensor_list_by_animal_all_SST[animal_id]
    for animal_id, tensor_for_animal in enumerate(tensor_list_by_animal_all_SST):

        components, model = slicetca.decompose(tensor_for_animal,
                                               number_components=(0, 0, 40),  # (trials, neurons, time bins)
                                               positive=True,
                                               learning_rate=1 * 10 ** -3,
                                               min_std=10 ** -5,
                                               max_iter=15,
                                               seed=0)

        model_list.append(model)

    save_dir = r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\new_SST_model_single_animal00x"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    save_path = os.path.join(save_dir, f"model_SST_animal_40_animal{animal_id}00x.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(model_list, f)

    print(f"Saved model for animal {animal_id} saved to {save_path}")
