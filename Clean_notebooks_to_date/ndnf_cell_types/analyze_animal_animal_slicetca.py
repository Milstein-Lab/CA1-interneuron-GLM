import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import sem


from plotting_celltypes_new import (fit_GLM_population, get_residual_activity_dict, get_animal_clean_dict_activity)

def get_tensor_per_animal_list(clean_resid_activity_dict_NDNF_newest):
    tensor_per_animal_list = []

    for animal in clean_resid_activity_dict_NDNF_newest:
        cell_list = []
        for cell in clean_resid_activity_dict_NDNF_newest[animal]:
            cell_list.append(clean_resid_activity_dict_NDNF_newest[animal][cell])

        cells_array = np.array(cell_list)
        cells_array = cells_array.transpose(2, 0, 1)
        tensor_per_animal_list.append(cells_array)

    return tensor_per_animal_list
def run():
    filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat'

    animal_clean_dict_activity, animal_vel_dict, animal_trials_original, \
        animal_trials_clean, trials_to_remove_local, animal_lick_dict = get_animal_clean_dict_activity(filepath)

    GLM_params, predicted_activity_dict = fit_GLM_population(
        animal_vel_dict, animal_clean_dict_activity,
        quintile=None, regression='ridge', alphas=None
    )

    residual_activity_dict_NDNF_new = get_residual_activity_dict(
        animal_clean_dict_activity, predicted_activity_dict
    )

    clean_resid_activity_dict_NDNF_newest = {}
    clean_velocity_dict_NDNF_newest = {}
    clean_lick_dict_NDNF_newest = {}

    for idx, animal in enumerate(residual_activity_dict_NDNF_new):
        if 14 < idx < 29:
            clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
            clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
            clean_lick_dict_NDNF_newest[f"animal_{idx+1}"] = animal_lick_dict[animal]

    tensor_per_animal_list = get_tensor_per_animal_list(clean_resid_activity_dict_NDNF_newest)
    # tensor_per_animal_list[animal_idx] has shape (trials, cells, pos)

    filepathdata_dir_ndnf_animal_tca_x00 = "./per_k_pickles_ndnf"

    MSE_an_av_per_latent = []
    MSE_an_sem_per_latent = []
    k_values = []

    # --- gather and sort pickle files by k ---
    all_files = [
        f for f in os.listdir(filepathdata_dir_ndnf_animal_tca_x00)
        if f.endswith(".pkl") and "per_num_latents_k" in f
    ]

    def extract_k(fname):
        # expects format like per_num_latents_k10.pkl
        base = os.path.splitext(fname)[0]
        k_str = base.split("k")[-1]
        return int(k_str)

    all_files_sorted = sorted(all_files, key=extract_k)

    for fname in all_files_sorted:
        k = extract_k(fname)
        k_values.append(k)

        full_path = os.path.join(filepathdata_dir_ndnf_animal_tca_x00, fname)
        with open(full_path, 'rb') as f:
            per_num_latents_dict_k = pickle.load(f)   # {k: [model_animal_0, model_animal_1, ...]}

        model_list_per_animal = per_num_latents_dict_k[k]

        MSE_per_animal_list = []

        for animal_idx, animal_model in enumerate(model_list_per_animal):
            animal_tensor = tensor_per_animal_list[animal_idx]  # (trials, cells, pos)
            reconstruction_full_animal = animal_model.construct().numpy(force=True)

            # sanity check shapes match
            if reconstruction_full_animal.shape != animal_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for k={k}, animal {animal_idx}: "
                    f"tensor {animal_tensor.shape}, recon {reconstruction_full_animal.shape}"
                )

            MSE = np.mean((animal_tensor - reconstruction_full_animal) ** 2)
            MSE_per_animal_list.append(MSE)

        MSE_per_animal_array = np.array(MSE_per_animal_list)
        MSE_an_av_per_latent.append(MSE_per_animal_array.mean())
        MSE_an_sem_per_latent.append(sem(MSE_per_animal_array))

    # --- convert to arrays for plotting ---
    k_values = np.array(k_values)
    print(f"k_values {k_values}")
    MSE_an_av_per_latent = np.array(MSE_an_av_per_latent)
    MSE_an_sem_per_latent = np.array(MSE_an_sem_per_latent)

    # --- plot mean ± SEM ---
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        k_values,
        MSE_an_av_per_latent,
        yerr=MSE_an_sem_per_latent,
        marker="o",
        linestyle="-",
        capsize=3,
    )
    plt.xlabel("Number of trial components (k)")
    plt.ylabel("Reconstruction MSE (mean ± SEM across animals)")
    plt.title("NDNF SliceTCA reconstruction error vs. number of latents")
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    run()
