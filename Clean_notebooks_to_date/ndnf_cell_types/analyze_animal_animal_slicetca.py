import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import sem
import h5py
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
import re

plt.rcParams['axes.titlesize'] = 20       # all titles
plt.rcParams['axes.labelsize'] = 16      # x and y labels
plt.rcParams['xtick.labelsize'] = 16      # tick labels
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["legend.fontsize"] = 14
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.titlepad'] = 12.0


def flatten_data(neuron_dict):
    flattened_data = {}
    for var in neuron_dict:
        flattened_data[var] = neuron_dict[var].flatten()
    return flattened_data

def fit_GLM(animal_factors_dict, neuron_activity, regression='linear', alphas=None):
    neuron_activity_flat = neuron_activity.flatten()
    flattened_data = flatten_data(animal_factors_dict)
    variable_names = [var for var in flattened_data]
    design_matrix_X = np.stack([flattened_data[var] for var in variable_names], axis=1)

    if regression == 'linear':
        model = LinearRegression()
    elif regression == 'lasso':
        model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
    elif regression == 'ridge':
        model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
    elif regression == 'elastic':
        l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000],
                             l1_ratio=l1_ratio, cv=None)

    model.fit(design_matrix_X, neuron_activity_flat)
    neuron_predicted_activity = model.predict(design_matrix_X)

    trialavg_neuron_activity = np.mean(neuron_activity, axis=1)
    trialavg_predicted_activity = np.mean(neuron_predicted_activity.reshape(neuron_activity.shape), axis=1)
    pearson_R = np.corrcoef(trialavg_predicted_activity, trialavg_neuron_activity)[0, 1]
    neuron_GLM_params = {}
    neuron_GLM_params['weights'] = {var: model.coef_[idx] for idx, var in enumerate(variable_names)}
    neuron_GLM_params['intercept'] = model.intercept_
    neuron_GLM_params['alpha'] = model.alpha_ if regression == 'ridge' else None
    neuron_GLM_params['l1_ratio'] = model.l1_ratio_ if regression == 'elastic' else None
    neuron_GLM_params['R2'] = model.score(design_matrix_X, neuron_activity_flat)
    neuron_GLM_params['pearson_R'] = pearson_R
    neuron_GLM_params['model'] = model
    return neuron_GLM_params, neuron_predicted_activity

def has_run_of_n_nans(trial_data, n=4):
    
    nan_mask = np.isnan(trial_data)
    if not np.any(nan_mask):
        return False
    
    conv = np.convolve(nan_mask.astype(int), np.ones(n, dtype=int), mode='valid')
    return np.any(conv >= n)


def interp_nans_1d(trial_data):
    x = np.arange(trial_data.size)
    nan_mask = np.isnan(trial_data)

    if not np.any(nan_mask):
        return trial_data

    if not np.any(~nan_mask):  # all NaNs
        return trial_data

    trial_data[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], trial_data[~nan_mask])
    return trial_data

def get_residual_activity_dict(activity_dict, predicted_activity_dict):
    residual_activity_dict = {}
    for animal in activity_dict:
        residual_activity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            residual_activity_dict[animal][neuron] = activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
    return residual_activity_dict


def fit_GLM_population(factors_dict, activity_dict, quintile=None, regression='ridge', alphas=None):
    GLM_params = {}
    predicted_activity_dict = {}

    for animal in factors_dict:
        GLM_params[animal] = {}
        predicted_activity_dict[animal] = {}
        animal_factors_dict = factors_dict[animal].copy()

        if quintile is not None:
            num_trials = animal_factors_dict['Activity'].shape[1]
            start_idx, end_idx = get_quintile_indices(num_trials, quintile)
            for var in animal_factors_dict:
                animal_factors_dict[var] = animal_factors_dict[var][:, start_idx:end_idx]

        for neuron_idx in activity_dict[animal]:
            neuron_activity = activity_dict[animal][neuron_idx]
            neuron_GLM_params, neuron_predicted_activity = fit_GLM(animal_factors_dict, neuron_activity, regression, alphas)
            GLM_params[animal][neuron_idx] = neuron_GLM_params
            predicted_activity_dict[animal][neuron_idx] = neuron_predicted_activity.reshape(activity_dict[animal][neuron_idx].shape)
                
    return GLM_params, predicted_activity_dict


def get_animal_clean_dict_activity(filepath, use_final=True):
    with h5py.File(filepath, "r") as f:
        if use_final:
            animal_group = f["animals"]
        else:
            animal_group = f["animal"]

        print(f"animal_group.keys() {animal_group.keys()}")

        shiftR_refs = animal_group["ShiftR"][:]
        shiftRunning_refs = animal_group["ShiftRunning"][:]

        if use_final:
            shiftL_refs = animal_group["ShiftL"][:]
        else:
            shiftL_refs = animal_group["ShiftLrate"][:]

        animal_clean_dict_activity = {}
        animal_trials_original = []
        animal_trials_clean = []

        animal_vel_dict = {}
        animal_lick_dict = {}

        trials_to_remove_local = []  # debug tracking for count == 105
        count = 0  # global cell counter (across animals)

        for animal_idx in range(len(shiftR_refs)):
            # ΔF: (cells, trials, time)
            delta_f = np.array(f[shiftR_refs[animal_idx][0]])
            animal_trials_original.append(delta_f.shape[1])

            # velocity & lick: raw (trials, time?) → transpose: (time, trials)
            vel = np.array(f[shiftRunning_refs[animal_idx][0]]).T
            lick = np.array(f[shiftL_refs[animal_idx][0]]).T

            # --- align trial counts across df / vel / lick ---
            n_df_trials = delta_f.shape[1]
            n_vel_trials = vel.shape[1]
            n_lick_trials = lick.shape[1]

            n_trials = min(n_df_trials, n_vel_trials, n_lick_trials)

            if (n_df_trials, n_vel_trials, n_lick_trials) != (n_trials,) * 3:
                delta_f = delta_f[:, :n_trials, :]
                vel = vel[:, :n_trials]
                lick = lick[:, :n_trials]

            # preallocate clean arrays
            vel_clean = np.empty_like(vel)
            lick_clean = np.empty_like(lick)
            delta_f_clean = np.empty_like(delta_f)

            # list of trials to drop for this animal (union across cells)
            trials_to_remove_list = []

            # special case: skip cell 0 for this animal
            if animal_idx == 22:
                valid_cells = range(1, delta_f.shape[0])
            else:
                valid_cells = range(delta_f.shape[0])

            # --- clean per cell / per trial ---
            for cell in valid_cells:
                # cell_data: (time, trials)
                cell_data = delta_f[cell, :, :].T

                for trial in range(cell_data.shape[1]):
                    trial_data = cell_data[:, trial]
                    vel_data_trial = vel[:, trial]
                    lick_data_trial = lick[:, trial]

                    nan_trial = np.any(np.isnan(trial_data))
                    nan_vel = np.any(np.isnan(vel_data_trial))

                    if nan_trial or nan_vel:
                        # decide: drop or interpolate based on runs of >=5 NaNs
                        too_many_nans = (
                            has_run_of_n_nans(trial_data, n=5)
                            or has_run_of_n_nans(vel_data_trial, n=5)
                        )

                        if too_many_nans:
                            # mark for removal (for all cells later)
                            if count == 105:
                                trials_to_remove_local.append(trial)
                            if trial not in trials_to_remove_list:
                                trials_to_remove_list.append(trial)
                        else:
                            # interpolate and keep this trial
                            clean_trial = interp_nans_1d(trial_data.copy())
                            delta_f_clean[cell, trial, :] = clean_trial

                            clean_vel = interp_nans_1d(vel_data_trial.copy())
                            vel_clean[:, trial] = clean_vel

                            clean_lick = interp_nans_1d(lick_data_trial.copy())
                            lick_clean[:, trial] = clean_lick
                    else:
                        # no NaNs anywhere: just copy
                        delta_f_clean[cell, trial, :] = trial_data
                        vel_clean[:, trial] = vel_data_trial
                        lick_clean[:, trial] = lick_data_trial

                count += 1  # increment per cell

            # --- drop bad trials across all cells for this animal ---
            trials_to_remove_array = np.array(trials_to_remove_list, dtype=int)

            if trials_to_remove_array.size > 0:
                mask = np.ones(delta_f_clean.shape[1], dtype=bool)
                mask[trials_to_remove_array] = False

                delta_f_clean = delta_f_clean[:, mask, :]
                vel_clean = vel_clean[:, mask]
                lick_clean = lick_clean[:, mask]

            animal_trials_clean.append(delta_f_clean.shape[1])

            # --- build per-cell dict with z-scoring ---
            cell_dict = {}
            for cell in valid_cells:
                cell_data = delta_f_clean[cell, :, :]  # (trials, time)

                mean = np.mean(cell_data)
                std = np.std(cell_data)

                if std == 0 or not np.isfinite(std):
                    print(" -> zero or bad std for this cell, skipping")
                    continue

                cell_z = (cell_data - mean) / std  # (trials, time)
                cell_dict[f"cell_{cell+1}"] = cell_z.T  # (time, trials)

            animal_clean_dict_activity[f"animal_{animal_idx+1}"] = cell_dict
            animal_vel_dict[f"animal_{animal_idx+1}"] = {"Velocity": vel_clean}
            animal_lick_dict[f"animal_{animal_idx+1}"] = {"Licks": lick_clean}

        return (
            animal_clean_dict_activity,
            animal_vel_dict,
            animal_trials_original,
            animal_trials_clean,
            trials_to_remove_local,
            animal_lick_dict,
        )



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

def get_tensor_per_cell_list(residual_activity_dict_SST):

    tensor_per_animal_list = []

    for animal in sorted(residual_activity_dict_SST.keys()):
        cell_tensor_list = []
        for cell in residual_activity_dict_SST[animal]:
            proper_shape_activity = residual_activity_dict_SST[animal][cell].T  # (pos, trials)
            proper_shape_activity = np.expand_dims(proper_shape_activity, axis=1)
            cell_tensor_list.append(proper_shape_activity)
        tensor_per_animal_list.append(cell_tensor_list)


    return tensor_per_animal_list

# def extract_k(fname):
#     # expects format like per_num_latents_k10.pkl
#     base = os.path.splitext(fname)[0]
#     k_str = base.split("k")[-1]
#     return int(k_str)

def extract_k(fname):
    """
    Extracts the integer K from filenames like:
      per_num_latents_k10.pkl
      per_num_latents_k36_per_cell.pkl
      anything_k5_something_else.pkl
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r'_k(\d+)', base)   # look for "_k" followed by digits
    if not m:
        raise ValueError(f"Could not parse k from filename: {fname}")
    return int(m.group(1))

def get_mse_from_model_filepath(models_dir, tensor_per_animal_list, use_animal=True):

    MSE_an_av_per_latent = []
    MSE_an_sem_per_latent = []
    k_values = []

    all_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl") and "per_num_latents_k" in f]

    all_files_sorted = sorted(all_files, key=extract_k)

    for fname in all_files_sorted:
        k = extract_k(fname)
        k_values.append(k)

        full_path = os.path.join(models_dir, fname)
        with open(full_path, 'rb') as f:
            per_num_latents_dict_k = pickle.load(f)   # {k: [model_animal_0, model_animal_1, ...]}

        model_list_per_animal = per_num_latents_dict_k[k]

        MSE_per_animal_list = []

        if use_animal:

            for animal_idx, animal_model in enumerate(model_list_per_animal):
                animal_tensor = tensor_per_animal_list[animal_idx]  # (trials, cells, pos)
                reconstruction_full_animal = animal_model.construct().numpy(force=True)

                if reconstruction_full_animal.shape != animal_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for k={k}, animal {animal_idx}: "
                        f"tensor {animal_tensor.shape}, recon {reconstruction_full_animal.shape}"
                    )
                
                print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

                MSE = np.mean((animal_tensor - reconstruction_full_animal) ** 2)
                MSE_per_animal_list.append(MSE)

        else:
            # print("made it hereeeeeeeeee")
            for animal_idx, animal_model_list in enumerate(model_list_per_animal):
                MSE_per_cells = []
                for cell in range(len(animal_model_list)):
                    cell_model = animal_model_list[cell]
                    reconstruction_full_cell = cell_model.construct().numpy(force=True)
                    
                    cell_tensor = tensor_per_animal_list[animal_idx][cell]

                    print(f"animal_idx {animal_idx} cell {cell}")

                    if reconstruction_full_cell.shape != cell_tensor.shape:
                        raise ValueError(
                            f"Shape mismatch for k={k}, animal {animal_idx}: "
                            f"tensor {cell_tensor.shape}, recon {reconstruction_full_cell.shape}"
                        )
                    
                    # print(f"reconstruction_full_cell.shape {reconstruction_full_cell.shape}")
                    
                    MSE = np.mean((cell_tensor - reconstruction_full_cell) ** 2)
                    MSE_per_cells.append(MSE)

                MSE_per_animal_list.append(np.mean(MSE_per_cells))


        MSE_per_animal_array = np.array(MSE_per_animal_list)
        MSE_an_av_per_latent.append(MSE_per_animal_array.mean())
        MSE_an_sem_per_latent.append(sem(MSE_per_animal_array))

    return MSE_an_av_per_latent, MSE_an_sem_per_latent, k_values

def plot_one_celltype(MSE_an_av_per_latent, MSE_an_sem_per_latent, k_values, color:str, label:str, ax:str):
    ax.errorbar(
        k_values,
        MSE_an_av_per_latent,
        yerr=MSE_an_sem_per_latent,
        marker="o",
        linestyle="-", color=color, label=label,
        capsize=3)
    ax.legend()





def run():

    animal_clean_dict_activity_NDNF, animal_vel_dict_NDNF, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict_NDNF = get_animal_clean_dict_activity('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat', use_final=True)
    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict_NDNF, animal_clean_dict_activity_NDNF,quintile=None, regression='ridge', alphas=None)
    residual_activity_dict_NDNF = get_residual_activity_dict(animal_clean_dict_activity_NDNF, predicted_activity_dict)

    animal_clean_dict_activity_SST, animal_vel_dict_SST, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict_SST = get_animal_clean_dict_activity('/Users/michaelfinch/CA1-interneuron-GLM/datasets/SSTindivsomata_GLM.mat', use_final=False)
    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict_SST, animal_clean_dict_activity_SST,quintile=None, regression='ridge', alphas=None)
    residual_activity_dict_SST = get_residual_activity_dict(animal_clean_dict_activity_SST, predicted_activity_dict)

    min_num_trials = 100000


    for animal in residual_activity_dict_SST:
        for cell in residual_activity_dict_SST[animal]:
            if residual_activity_dict_SST[animal][cell].shape[1] < min_num_trials:
                min_num_trials = residual_activity_dict_SST[animal][cell].shape[1]


    print(f"min_num_trials {min_num_trials}")


    animal_clean_dict_activity_EC, animal_vel_dict_EC, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict_EC = get_animal_clean_dict_activity('/Users/michaelfinch/CA1-interneuron-GLM/datasets/EC_GLM.mat', use_final=False)
    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict_EC, animal_clean_dict_activity_EC,quintile=None, regression='ridge', alphas=None)
    residual_activity_dict_EC = get_residual_activity_dict(animal_clean_dict_activity_EC, predicted_activity_dict)


    clean_resid_activity_dict_NDNF_newest = {}
    clean_velocity_dict_NDNF_newest = {}
    clean_lick_dict_NDNF_newest = {}

    for idx, animal in enumerate(residual_activity_dict_NDNF):
        if 14 < idx < 29:
            clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF[animal]
            clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict_NDNF[animal]
            clean_lick_dict_NDNF_newest[f"animal_{idx+1}"] = animal_lick_dict_NDNF[animal]


    # # if use_animal:
    # tensor_per_animal_list_NDNF = get_tensor_per_animal_list(clean_resid_activity_dict_NDNF_newest)
    # tensor_per_animal_list_SST = get_tensor_per_animal_list(residual_activity_dict_SST)

    # for animal in range(len(tensor_per_animal_list_SST)):
    #     print(tensor_per_animal_list_SST[animal].shape)
    # else:
    #     tensor_per_cell_list = get_tensor_per_cell_list(residual_activity_dict_SST)


    fig, axs = plt.subplots(1,3)

    tensor_filepath= "./tensor_per_animal_list_ndnf.pkl"
    with open(tensor_filepath, 'rb') as f:
        tensor_per_animal_list_NDNF = pickle.load(f)
    MSE_an_av_per_latent_NDNF_x00_animal, MSE_an_sem_per_latent_NDNF_x00_animal, k_values_NDNF_x00_animal = get_mse_from_model_filepath("./per_k_pickles_ndnf_x00", tensor_per_animal_list_NDNF, use_animal=True)
    
    tensor_filepath= "./tensor_per_animal_list_sst.pkl"
    with open(tensor_filepath, 'rb') as f:
        tensor_per_animal_list_SST = pickle.load(f)
    MSE_an_av_per_latent_SST_x00_animal, MSE_an_sem_per_latent_SST_x00_animal, k_values_SST_x00_animal = get_mse_from_model_filepath("./per_k_pickles_sst_x00", tensor_per_animal_list_SST, use_animal=True)

    tensor_filepath= "./tensor_per_animal_list_ec.pkl"
    with open(tensor_filepath, 'rb') as f:
        tensor_per_animal_list_EC = pickle.load(f)
    MSE_an_av_per_latent_EC_x00_animal, MSE_an_sem_per_latent_EC_x00_animal, k_values_EC_x00_animal = get_mse_from_model_filepath("./per_k_pickles_ec_x00", tensor_per_animal_list_EC, use_animal=True)




    plot_one_celltype(MSE_an_av_per_latent_NDNF_x00_animal, MSE_an_sem_per_latent_NDNF_x00_animal, k_values_NDNF_x00_animal, color='orange', label="NDNF", ax=axs[0])
    plot_one_celltype(MSE_an_av_per_latent_SST_x00_animal, MSE_an_sem_per_latent_SST_x00_animal, k_values_SST_x00_animal, color='blue', label="SST", ax=axs[0])
    plot_one_celltype(MSE_an_av_per_latent_EC_x00_animal, MSE_an_sem_per_latent_EC_x00_animal, k_values_EC_x00_animal, color='green', label="EC", ax=axs[0])

    axs[0].set_xlabel("Number of latents x00 Animal")
    axs[0].set_ylabel("Reconstruction MSE")
    axs[0].set_title("Animal SliceTCA")


    MSE_an_av_per_latent_NDNF_0x0_animal, MSE_an_sem_per_latent_NDNF_0x0_animal, k_values_NDNF_0x0_animal = get_mse_from_model_filepath("./per_k_pickles_ndnf_0x0", tensor_per_animal_list_NDNF, use_animal=True)
    MSE_an_av_per_latent_SST_0x0_animal, MSE_an_sem_per_latent_SST_0x0_animal, k_values_SST_0x0_animal = get_mse_from_model_filepath("./per_k_pickles_sst_0x0", tensor_per_animal_list_SST, use_animal=True)
    MSE_an_av_per_latent_EC_0x0_animal, MSE_an_sem_per_latent_EC_0x0_animal, k_values_EC_0x0_animal = get_mse_from_model_filepath("./per_k_pickles_ec_0x0", tensor_per_animal_list_EC, use_animal=True)

    plot_one_celltype(MSE_an_av_per_latent_NDNF_0x0_animal, MSE_an_sem_per_latent_NDNF_0x0_animal, k_values_NDNF_0x0_animal, color='orange', label="NDNF", ax=axs[1])
    plot_one_celltype(MSE_an_av_per_latent_SST_0x0_animal, MSE_an_sem_per_latent_SST_0x0_animal, k_values_SST_0x0_animal, color='blue', label="SST", ax=axs[1])
    plot_one_celltype(MSE_an_av_per_latent_EC_0x0_animal, MSE_an_sem_per_latent_EC_0x0_animal, k_values_EC_0x0_animal, color='green', label="EC", ax=axs[1])


    axs[1].set_xlabel("Number of latents 0x0 Animal")
    axs[1].set_ylabel("Reconstruction MSE")
    axs[1].set_title("Animal SliceTCA")


    ################# per cell --IF WE WANTED TO RUN IT FROM SCRATCH 


    # tensor_filepath= "./tensor_per_cell_list_ndnf.pkl"
    # with open(tensor_filepath, 'rb') as f:
    #     tensor_per_cell_list_NDNF = pickle.load(f)
    # MSE_an_av_per_latent_NDNF_x00_cell, MSE_an_sem_per_latent_NDNF_x00_cell, k_values_NDNF_x00_cell = get_mse_from_model_filepath("./per_k_pickles_ndnf_per_cell", tensor_per_cell_list_NDNF, use_animal=False)

    # cell_by_cell_tca_ndnf_mse_dict = {"MSE_an_av_per_latent_NDNF_x00_cell":MSE_an_av_per_latent_NDNF_x00_cell,
    #                                    "MSE_an_sem_per_latent_NDNF_x00_cell":MSE_an_sem_per_latent_NDNF_x00_cell,
    #                                      "k_values_NDNF_x00_cell":k_values_NDNF_x00_cell}
    
    # save_path_ndnf = "./cell_by_cell_tca_ndnf_mse_dict.pkl"
    # with open(save_path_ndnf, 'wb') as f:
    #     pickle.dump(cell_by_cell_tca_ndnf_mse_dict, f)
    
    # tensor_filepath= "./tensor_per_cell_list_sst.pkl"
    # with open(tensor_filepath, 'rb') as f:
    #     tensor_per_cell_list_SST = pickle.load(f)
    # MSE_an_av_per_latent_SST_x00_cell, MSE_an_sem_per_latent_SST_x00_cell, k_values_SST_x00_cell = get_mse_from_model_filepath("./per_k_pickles_sst_per_cell", tensor_per_cell_list_SST, use_animal=False)

    # cell_by_cell_tca_sst_mse_dict = {"MSE_an_av_per_latent_SST_x00_cell":MSE_an_av_per_latent_SST_x00_cell,
    #                                    "MSE_an_sem_per_latent_SST_x00_cell":MSE_an_sem_per_latent_SST_x00_cell,
    #                                      "k_values_SST_x00_cell":k_values_SST_x00_cell}

    # save_path_ndnf = "./cell_by_cell_tca_sst_mse_dict.pkl"
    # with open(save_path_ndnf, 'wb') as f:
    #     pickle.dump(cell_by_cell_tca_sst_mse_dict, f)


    # tensor_filepath= "./tensor_per_cell_list_ec.pkl"
    # with open(tensor_filepath, 'rb') as f:
    #     tensor_per_cell_list_EC = pickle.load(f)
    # MSE_an_av_per_latent_EC_x00_cell, MSE_an_sem_per_latent_EC_x00_cell, k_values_EC_x00_cell = get_mse_from_model_filepath("./per_k_pickles_ec_per_cell", tensor_per_cell_list_EC, use_animal=False)


    # cell_by_cell_tca_ec_mse_dict = {"MSE_an_av_per_latent_EC_x00_cell":MSE_an_av_per_latent_EC_x00_cell,
    #                                    "MSE_an_sem_per_latent_EC_x00_cell":MSE_an_sem_per_latent_EC_x00_cell,
    #                                      "k_values_EC_x00_cell":k_values_EC_x00_cell}
    
    # save_path_ec = "./cell_by_cell_tca_ec_mse_dict.pkl"
    # with open(save_path_ec, 'wb') as f:
    #     pickle.dump(cell_by_cell_tca_ec_mse_dict, f)

    #############################################################


    save_path_ndnf = "./cell_by_cell_tca_ndnf_mse_dict.pkl"
    with open(save_path_ndnf, 'rb') as f:
        cell_by_cell_tca_ndnf_mse_dict = pickle.load(f)

    MSE_an_av_per_latent_NDNF_x00_cell = cell_by_cell_tca_ndnf_mse_dict["MSE_an_av_per_latent_NDNF_x00_cell"]
    MSE_an_sem_per_latent_NDNF_x00_cell = cell_by_cell_tca_ndnf_mse_dict["MSE_an_sem_per_latent_NDNF_x00_cell"]
    k_values_NDNF_x00_cell = cell_by_cell_tca_ndnf_mse_dict["k_values_NDNF_x00_cell"]


    save_path_sst = "./cell_by_cell_tca_sst_mse_dict.pkl"
    with open(save_path_sst, 'rb') as f:
        cell_by_cell_tca_sst_mse_dict = pickle.load(f)

    MSE_an_av_per_latent_SST_x00_cell = cell_by_cell_tca_sst_mse_dict["MSE_an_av_per_latent_SST_x00_cell"]
    MSE_an_sem_per_latent_SST_x00_cell = cell_by_cell_tca_sst_mse_dict["MSE_an_sem_per_latent_SST_x00_cell"]
    k_values_SST_x00_cell = cell_by_cell_tca_sst_mse_dict["k_values_SST_x00_cell"]

    save_path_ec = "./cell_by_cell_tca_ec_mse_dict.pkl"
    with open(save_path_ec, 'rb') as f:
        cell_by_cell_tca_ec_mse_dict = pickle.load(f)

    MSE_an_av_per_latent_EC_x00_cell = cell_by_cell_tca_ec_mse_dict["MSE_an_av_per_latent_EC_x00_cell"]
    MSE_an_sem_per_latent_EC_x00_cell = cell_by_cell_tca_ec_mse_dict["MSE_an_sem_per_latent_EC_x00_cell"]
    k_values_EC_x00_cell = cell_by_cell_tca_ec_mse_dict["k_values_EC_x00_cell"]

    plot_one_celltype(MSE_an_av_per_latent_NDNF_x00_cell, MSE_an_sem_per_latent_NDNF_x00_cell, k_values_NDNF_x00_cell, color='orange', label="NDNF", ax=axs[2])
    plot_one_celltype(MSE_an_av_per_latent_SST_x00_cell, MSE_an_sem_per_latent_SST_x00_cell, k_values_SST_x00_cell, color='blue', label="SST", ax=axs[2])
    plot_one_celltype(MSE_an_av_per_latent_EC_x00_cell, MSE_an_sem_per_latent_EC_x00_cell, k_values_EC_x00_cell, color='green', label="EC", ax=axs[2])


    axs[2].set_xlabel("Number of latents x00 Cell")
    axs[2].set_ylabel("Reconstruction MSE")
    axs[2].set_title("Cell SliceTCA")
    axs[2].axvline(20, color='r', linewidth=2, linestyle='--', label='Elbow=20')

    plt.tight_layout()
    plt.show()

if __name__ =="__main__":

    run()
