
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca

import click 

from scipy.stats import ranksums, ks_2samp, ttest_ind, pearsonr


from sklearn.linear_model import LassoCV, Ridge, ElasticNetCV, LinearRegression

from scipy.spatial.distance import mahalanobis

from sklearn.linear_model import LinearRegression

from scipy import stats

from sklearn.decomposition import PCA

plt.rcParams['savefig.dpi'] = 600



# import utils as ut
# import plot as pt
plt.rcParams.update({'font.size': 5,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['axes.titlesize'] = 5
plt.rcParams['axes.labelsize'] = 5      
plt.rcParams['legend.fontsize'] = 4     


import sys
from scipy.stats import sem
import pandas as pd

sys.path.append('/Users/michaelfinch/CA1-interneuron-GLM')

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from modelling_to_date_utils import *
from SliceTCA_example import *

from scipy.stats import ttest_rel

import click

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from spiking_model_utils import load_data_regular, preprocess_data2

import pickle

from matplotlib.lines import Line2D

def plot_LDA1_LDA2_state_space_prepost(title_fs, group_array, title="", ax=None):
    """
    LDA1: spatial axis (pre vs post reward bins).
    LDA2: learning axis (early vs late trials, using full track).
    Scatter colors: early/late × pre/post reward.
    """

    n_samples, n_cells = group_array.shape
    n_pos_bins = 50
    n_trials   = n_samples // n_pos_bins

    pos_idx   = np.tile(np.arange(n_pos_bins), n_trials)
    trial_idx = np.repeat(np.arange(n_trials), n_pos_bins)

    trials_per_quint = n_trials // 5
    trial_quint = trial_idx // trials_per_quint


    early_mask = (trial_quint == 0)
    late_mask  = (trial_quint == 4)
    mid_mask   = ~(early_mask | late_mask)

    pre_mask  = (pos_idx < 25)
    post_mask = (pos_idx >= 25)
    
    space_labels = post_mask.astype(int)  # 0 = pre, 1 = post

    print(f"group_array.shape {group_array.shape}")

    print("Any NaNs?", np.isnan(group_array).any())
    print("Any infs?", np.isinf(group_array).any())

    pre = group_array[space_labels == 0]
    post = group_array[space_labels == 1]

    var_pre  = pre.var(axis=0)
    var_post = post.var(axis=0)

    print("Zero-var features in pre:", np.where(var_pre == 0)[0])
    print("Zero-var features in post:", np.where(var_post == 0)[0])


    lda_space = LinearDiscriminantAnalysis(n_components=1, solver="svd") #eigen
    print("LDA1 solver at runtime:", lda_space.solver)
    LDA1 = lda_space.fit_transform(group_array, space_labels).ravel()
    print(f"{title} | LDA1 explained var (pre vs post):",
          lda_space.explained_variance_ratio_)

    used_mask = np.zeros_like(early_mask, dtype=bool)
    used_mask[early_mask] = True
    used_mask[late_mask] = True
    
    X_used = group_array[used_mask, :]  

    late_for_used = late_mask[used_mask]
    y_used = np.zeros(late_for_used.shape[0], dtype=int)
    y_used[late_for_used] = 1 ###binary array for lda
    lda_learn = LinearDiscriminantAnalysis(n_components=1, solver="svd") #eigen
    print("LDA2 solver at runtime:", lda_learn.solver)
    LDA2_used = lda_learn.fit_transform(X_used, y_used) 
    LDA2_used = LDA2_used[:, 0]  

    n_samples = group_array.shape[0]
    LDA2 = np.full(n_samples, np.nan)
    LDA2[used_mask] = LDA2_used


    mid_valid = mid_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mid_valid], LDA2[mid_valid],
                s=8, alpha=0.15, color="lightgray", label="middle trials")

    mask_early_pre = early_mask & pre_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_early_pre], LDA2[mask_early_pre],
                s=14, alpha=0.9, color='b', label="early pre-reward")

    mask_early_post = early_mask & post_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_early_post], LDA2[mask_early_post],
                s=14, alpha=0.9, color="r", label="early post-reward")

    mask_late_pre = late_mask & pre_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_late_pre], LDA2[mask_late_pre],
                s=14, alpha=0.9, color="green", label="late pre-reward")

    mask_late_post = late_mask & post_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_late_post], LDA2[mask_late_post],
                s=14, alpha=0.9, color="k", label="late post-reward")

    ax.axvline(0, color="k", lw=1, alpha=0.4)
    ax.axhline(0, color="k", lw=1, alpha=0.4)

    ax.set_xlabel("LDA1 (pre vs post reward)")
    ax.set_ylabel("LDA2 (early vs late trials)")
    ax.set_ylim(-5,5)
    ax.set_xlim(-5,5)
    ax.legend(frameon=True, fontsize=title_fs-3)

    def mahal_group_distance(X, maskA, maskB):
        A = X[maskA]; B = X[maskB]
        meanA = A.mean(axis=0)
        meanB = B.mean(axis=0)
        cov = np.cov(np.vstack([A,B]).T)
        inv_cov = np.linalg.inv(cov)
        return mahalanobis(meanA, meanB, inv_cov)
    
    X0 = np.column_stack([LDA1, LDA2])  # shape (n_samples, 2)

    d_pre_0 = mahal_group_distance(X0, mask_early_pre, mask_late_pre)
    d_post_0 = mahal_group_distance(X0, mask_early_post, mask_late_post)

    ax.set_title(f"{title} \n Mahal. Dist. Early vs Late Pre-Reward={d_pre_0:.2f} \n Mahal. Dist. Early vs Late Post-Reward={d_post_0:.2f}")


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


def get_mse_from_model_filepath(models_dir):

    MSE_an_av_per_latent = []
    MSE_an_sem_per_latent = []
    k_values = []

    all_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl") and "per_num_latents_k" in f]

    all_files_sorted = sorted(all_files, key=extract_k)

    for idx, fname in enumerate(all_files_sorted):
        k = extract_k(fname)
        k_values.append(k)

        full_path = os.path.join(models_dir, fname)
        with open(full_path, 'rb') as f:
            per_num_latents_dict_k = pickle.load(f)   # {k: [model_animal_0, model_animal_1, ...]}

        if k == 20:
            latent_model20 = per_num_latents_dict_k[k]

            return latent_model20


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
                    # print(" -> zero or bad std for this cell, skipping")
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


def get_residual_activity_dict(activity_dict, predicted_activity_dict):
    residual_activity_dict = {}
    for animal in activity_dict:
        residual_activity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            residual_activity_dict[animal][neuron] = activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
    return residual_activity_dict


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

def flatten_data(neuron_dict):
    flattened_data = {}
    for var in neuron_dict:
        flattened_data[var] = neuron_dict[var].flatten()
    return flattened_data


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




def plot_LDA1_LDA2_state_space_prepost(group_array, title="", ax=None):
    """
    LDA1: spatial axis (pre vs post reward bins).
    LDA2: learning axis (early vs late trials, using full track).
    Scatter colors: early/late × pre/post reward.
    """

    n_samples, n_cells = group_array.shape
    n_pos_bins = 50
    n_trials   = n_samples // n_pos_bins

    pos_idx   = np.tile(np.arange(n_pos_bins), n_trials)
    trial_idx = np.repeat(np.arange(n_trials), n_pos_bins)

    trials_per_quint = n_trials // 5
    trial_quint = trial_idx // trials_per_quint


    early_mask = (trial_quint == 0)
    late_mask  = (trial_quint == 4)
    mid_mask   = ~(early_mask | late_mask)

    pre_mask  = (pos_idx < 25)
    post_mask = (pos_idx >= 25)
    
    space_labels = post_mask.astype(int)  # 0 = pre, 1 = post

    print(f"group_array.shape {group_array.shape}")

    print("Any NaNs?", np.isnan(group_array).any())
    print("Any infs?", np.isinf(group_array).any())

    pre = group_array[space_labels == 0]
    post = group_array[space_labels == 1]

    var_pre  = pre.var(axis=0)
    var_post = post.var(axis=0)

    print("Zero-var features in pre:", np.where(var_pre == 0)[0])
    print("Zero-var features in post:", np.where(var_post == 0)[0])


    lda_space = LinearDiscriminantAnalysis(n_components=1, solver="svd") #eigen
    print("LDA1 solver at runtime:", lda_space.solver)
    LDA1 = lda_space.fit_transform(group_array, space_labels).ravel()
    print(f"{title} | LDA1 explained var (pre vs post):",
          lda_space.explained_variance_ratio_)

    used_mask = np.zeros_like(early_mask, dtype=bool)
    used_mask[early_mask] = True
    used_mask[late_mask] = True
    
    X_used = group_array[used_mask, :]  

    late_for_used = late_mask[used_mask]
    y_used = np.zeros(late_for_used.shape[0], dtype=int)
    y_used[late_for_used] = 1 ###binary array for lda
    lda_learn = LinearDiscriminantAnalysis(n_components=1, solver="svd") #eigen
    print("LDA2 solver at runtime:", lda_learn.solver)
    LDA2_used = lda_learn.fit_transform(X_used, y_used) 
    LDA2_used = LDA2_used[:, 0]  

    n_samples = group_array.shape[0]
    LDA2 = np.full(n_samples, np.nan)
    LDA2[used_mask] = LDA2_used


    mid_valid = mid_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mid_valid], LDA2[mid_valid],
                s=8, alpha=0.15, color="lightgray", label="middle trials")

    mask_early_pre = early_mask & pre_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_early_pre], LDA2[mask_early_pre],
                s=14, alpha=0.9, color='b', label="early pre-reward")

    mask_early_post = early_mask & post_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_early_post], LDA2[mask_early_post],
                s=14, alpha=0.9, color="r", label="early post-reward")

    mask_late_pre = late_mask & pre_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_late_pre], LDA2[mask_late_pre],
                s=14, alpha=0.9, color="green", label="late pre-reward")

    mask_late_post = late_mask & post_mask & ~np.isnan(LDA2)
    ax.scatter(LDA1[mask_late_post], LDA2[mask_late_post],
                s=14, alpha=0.9, color="k", label="late post-reward")

    ax.axvline(0, color="k", lw=1, alpha=0.4)
    ax.axhline(0, color="k", lw=1, alpha=0.4)

    ax.set_xlabel("LDA1 (pre vs post reward)")
    ax.set_ylabel("LDA2 (early vs late trials)")
    ax.set_ylim(-5,5)
    ax.set_xlim(-5,5)
    ax.legend(frameon=True)

    def mahal_group_distance(X, maskA, maskB):
        A = X[maskA]; B = X[maskB]
        meanA = A.mean(axis=0)
        meanB = B.mean(axis=0)
        cov = np.cov(np.vstack([A,B]).T)
        inv_cov = np.linalg.inv(cov)
        return mahalanobis(meanA, meanB, inv_cov)
    
    X0 = np.column_stack([LDA1, LDA2])  # shape (n_samples, 2)

    d_pre_0 = mahal_group_distance(X0, mask_early_pre, mask_late_pre)
    d_post_0 = mahal_group_distance(X0, mask_early_post, mask_late_post)

    ax.set_title(f"{title} \n Mahal. Dist. Early vs Late Pre-Reward={d_pre_0:.2f} \n Mahal. Dist. Early vs Late Post-Reward={d_post_0:.2f}")

def get_labels_all_different_Ks_single(model_20_NDNF_resid, which_vectors: int):

    w1 = model_20_NDNF_resid.vectors[which_vectors][0]
    f1 = model_20_NDNF_resid.vectors[which_vectors][1]
    F = f1.detach().cpu().numpy()   # (latents, cells, pos) = (20, 115, 50)
    W = w1.detach().cpu().numpy()   # (latents, trials) = (20, 100)

    print(f"F.shape {F.shape}  W.shape {W.shape}")

    # Build X so rows = cells (115)
    if which_vectors == 0:
        # Use latent×pos per cell, flattened: (115, 20*50)
        X = np.moveaxis(F, 1, 0)              # (cells=115, latents=20, pos=50)
        X = X.reshape(X.shape[0], -1)         # (115, 1000)
        print("X shape (latent×pos flat):", X.shape)

    elif which_vectors == 1:
        X = W.T  # (115, 20) mean over pos
        print("X shape (mean over pos):", X.shape)

    else:
        X = np.moveaxis(F, 2, 0)     # -> (cells=115, latents=20, trials=100)
        X = X.reshape(X.shape[0], -1)  # -> (115, 20*100) = (115, 2000)
        print(X.shape)  # (115, 2000)

    Xz = StandardScaler().fit_transform(X)
    labels_cells_dict_all_K = {K: KMeans(n_clusters=K, n_init=100, random_state=42).fit_predict(Xz) for K in range(1, 11)}
    return labels_cells_dict_all_K


def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict,
                      num_clusters=8, reassign_clusters=False,
                      x00=True, umap=True, contiguous=True, ranks=20):

    internals_per_animal_dict_EC_animal_x00_regkmean = {}

    # iterate over matching animal keys
    for idx, animal in enumerate(residual_activity_dict):
        internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}

        for idt, cell in enumerate(residual_activity_dict[animal]):
            cell_data = residual_activity_dict[animal][cell].T
            cell_data = ((cell_data - np.min(cell_data)) /
                         (np.max(cell_data) - np.min(cell_data)))
            cell_data_3d = np.expand_dims(cell_data, axis=1)
            cell_data_3d = torch.from_numpy(cell_data_3d)

            print(f"NDNF_fixed_model_dict[animal].keys() {NDNF_fixed_model_dict[animal].keys()}")   
            print(f"residual_activity_dict[animal].keys() {residual_activity_dict[animal].keys()}")

            # use the same animal_key for the model dict
            cell_model = NDNF_fixed_model_dict[animal][cell]

            internals_dict = get_animal_model_reconstruction_dict_mod(
                cell_model,
                cell_data_3d,
                max_clusters=num_clusters,
                display=False,
                reassign_small_clusters=reassign_clusters,
                x00=x00,
                use_umap=umap,
                use_breakpoints=contiguous,
            )

            internals_per_animal_dict_EC_animal_x00_regkmean_cell[cell] = internals_dict

        internals_per_animal_dict_EC_animal_x00_regkmean[animal] = \
            internals_per_animal_dict_EC_animal_x00_regkmean_cell

    return internals_per_animal_dict_EC_animal_x00_regkmean


def reshape_contig_dict(cued_contig_dict, NDNF_cued_model_dict_clean):
    # Match the old structure: outer key is rank (20)
    cued_contig_final = {20: {}}

    for animal in cued_contig_dict:  # animal index: 0,1,2,...
        cued_contig_final[20][animal] = {}

        for cell in cued_contig_dict[animal]:  # cell index: 0,1,2,...
            # 1) SliceTCA model object
            model_obj = NDNF_cued_model_dict_clean[animal][cell]

            # 2) Internals for this cell – currently under "cell_0"
            internals_cell0 = cued_contig_dict[animal][cell]["cell_0"]

            # 3) Rename "cell_0" → f"cell_{cell}" to match old API
            per_cell_internals = {f"cell_{cell}": internals_cell0}

            # 4) Store as [model_obj, per_cell_internals]
            cued_contig_final[20][animal][cell] = [model_obj, per_cell_internals]

    return cued_contig_final



def get_animal_model_reconstruction_dict_mod(animal_model, tensor_for_animal, max_clusters=12, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    per_cell_internals_dict = {}
    reconstruction_full_animal = animal_model.construct().numpy(force=True)

    if x00:
        w1 = animal_model.vectors[0][0].detach().numpy()
        X = np.abs(w1.T)
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(X)
    else:
        f = animal_model.vectors[2][1].detach()
        f1 = f.permute(1, 0, 2).reshape(f.shape[1], -1)
        f1 = torch.abs(f1).cpu().numpy()
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)

    cluster_labels_dict = {}
    cluster_pca_dict = {}
    cluster_centroids_dict = {}

    for clusters_chosen in range(1, max_clusters):
        if use_breakpoints:
            print(f"Using breakpoint clustering (clusters = {clusters_chosen})")
            model_input = X_umap if use_umap else (X if x00 else f1)
            n_bkps = clusters_chosen - 1
            algo = rpt.Binseg(model="l2", min_size=3).fit(model_input)
            try:
                bkps = algo.predict(n_bkps=n_bkps)
            except rpt.exceptions.BadSegmentationParameters:
                print("  Skipping due to bad breakpoint config")
                continue

            labels = np.zeros(model_input.shape[0], dtype=int)
            start = 0
            for cluster_id, end in enumerate(bkps):
                labels[start:end] = cluster_id
                start = end

            centroids = np.array([
                model_input[labels == i].mean(axis=0)
                for i in range(clusters_chosen)
            ])
            X_pca = PCA(n_components=3).fit_transform(model_input)

        else:
            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            model_input = X_umap if use_umap else (X if x00 else f1)
            labels = kmeans.fit_predict(model_input)

            centroids = kmeans.cluster_centers_
            if use_umap:
                X_pca = PCA(n_components=3).fit_transform(X_umap)
            else:
                X_pca = PCA(n_components=3).fit_transform(model_input)

        cluster_labels_dict[clusters_chosen] = labels
        cluster_centroids_dict[clusters_chosen] = centroids
        cluster_pca_dict[clusters_chosen] = X_pca

    for cell in range(reconstruction_full_animal.shape[1]):
        #     cell = 0
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

        for clusters_chosen in range(1, max_clusters):
            labels = cluster_labels_dict[clusters_chosen].copy()
            centroids = cluster_centroids_dict[clusters_chosen]
            X_pca = cluster_pca_dict[clusters_chosen]
            model_input = X_umap if use_umap else (X if x00 else f1)

            print(f"\nclusters_chosen = {clusters_chosen}")
            print("Before reassignment:")
            for cluster_id in range(clusters_chosen):
                count = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} trials")

            if reassign_small_clusters:
                if use_umap:
                    model_input = X_umap
                    centroid_space = np.array([
                        model_input[labels == i].mean(axis=0)
                        for i in range(clusters_chosen)])
                else:
                    model_input = X if x00 else f1
                    centroid_space = centroids
                for cluster_id in range(clusters_chosen):
                    trial_indices = np.where(labels == cluster_id)[0]
                    if len(trial_indices) < 2:
                        print(f"  Reassigning cluster {cluster_id} (size={len(trial_indices)})...")
                        for idx in trial_indices:
                            trial = model_input[idx]
                            dists = cdist([trial], centroid_space)[0]
                            dists[cluster_id] = np.inf
                            new_cluster = np.argmin(dists)
                            labels[idx] = new_cluster

            print("After reassignment:")
            for cluster_id in range(clusters_chosen):
                count = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} trials")

            x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = X_pca
            labels_dict[f"clusters_chosen_{clusters_chosen}"] = labels

            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            cluster_trial_indices = {}

            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                cluster_trial_indices[n] = trial_indices
                if len(trial_indices) == 0:
                    continue
                cluster_trials = real_cell_activity[trial_indices, :]
                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)
                valid_cluster_indices.append((n, trial_indices))

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


def get_cp_dict(cell_SST_model_ranks20_contig_x00):
    changepoints_dict = {}
    for animal in cell_SST_model_ranks20_contig_x00[20]:
        changepoints_cell_dict = {}
        for cell in cell_SST_model_ranks20_contig_x00[20][animal]:
            labels = cell_SST_model_ranks20_contig_x00[20][animal][cell][1][f'cell_{cell}']["labels_dict"]["clusters_chosen_3"]
            changepoints = np.where(np.diff(labels) != 0)[0]
            changepoints_cell_dict[cell] = changepoints
        changepoints_dict[animal] = changepoints_cell_dict

    return changepoints_dict

def get_predicted_trial_number(components_type_1):
    min_trials = 94

    X_all = components_type_1[0][0].T

    print(f"X_all {X_all.shape}")

    trial_idx = np.arange(min_trials)

    reg = LinearRegression()

    reg.fit(X_all, trial_idx)

    neuron_predicted_activity = reg.predict(X_all)

    return neuron_predicted_activity, reg


def plot_bars_from_predicted(neuron_predicted_trial_num_0, neuron_predicted_trial_num_1, num_trials=94, ax=None, title=None):
    accuracy1_list = []
    accuracy0_list = []

    for i in range(num_trials):
        accuracy1 = np.mean(np.square(i - neuron_predicted_trial_num_1[i]))
        accuracy0 = np.mean(np.square(i - neuron_predicted_trial_num_0[i]))

        accuracy1_list.append(accuracy1)
        accuracy0_list.append(accuracy0)

    data_reg_trial_num_pred = [accuracy0_list, accuracy1_list]

    plot_MSEs(data_reg_trial_num_pred, ax, title)



def plot_MSEs(data, ax, title="MSE per cell with mean ± SEM"):
    means = [np.mean(d) for d in data]
    # standard error of the mean
    errs  = [np.std(d, ddof=1) / np.sqrt(len(d)) for d in data]

    x = np.arange(2)

    colors = ["orange", "blue"]

    # bars with error bars + horizontal caps
    ax.bar(
        x,
        means,
        yerr=errs,
        width=0.6,
        alpha=0.5,
        edgecolor="black",
        capsize=5,              # this gives you the little horizontal caps
        error_kw=dict(capthick=1)
    )

    # overlay individual data points with horizontal jitter
    for i, vals in enumerate(data):
        vals = np.asarray(vals)
        jitter = (np.random.rand(len(vals)) - 0.5) * 0.15
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            color=colors[i],
            zorder=3,
        )

    
    t_statistic, p_value = stats.ttest_ind(data[0], data[1], equal_var=False)

    ax.set_xticks(x)
    ax.set_xticklabels(["0", "1"])
    ax.set_ylabel("MSE per cell")
    ax.set_title(f"{title} \n p={p_value}")



def scatter_with_bestfit(ax, x, y, color, label):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    ax.scatter(x, y, s=6, color=color, alpha=0.5, label=label)

    fit = LinearRegression().fit(x.reshape(-1,1), y)
    m = fit.coef_[0]
    b = fit.intercept_

    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, m*xx + b, color=color, lw=2)

    return m, b



def shuffle_my_trials(data, rng):
    shuff = data.copy()
    n_trials, n_cells, n_pos = data.shape
    for c in range(n_cells):
        perm = rng.permutation(n_trials)
        shuff[:, c, :] = data[perm, c, :]

    return shuff


def run(use_pure, use_all):

    filepath = os.path.join(os.path.expanduser('~'), 'CA1_interneuron_model', 'datasets', 'NDNF_E0A1B1_251107.mat')
    animal_clean_dict_activity, factors_dict_NDNF, _, _, _, animal_lick_dict = get_animal_clean_dict_activity(filepath, use_final=True)
    GLM_params_NDNF, predicted_activity_dict = fit_GLM_population(factors_dict_NDNF, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)
    residual_activity_dict_NDNF = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)

    min_trials = 10000

    for animal in residual_activity_dict_NDNF:
        for cell in residual_activity_dict_NDNF[animal]:
            n_trials = residual_activity_dict_NDNF[animal][cell].shape[1]
            if  n_trials < min_trials:
                min_trials = n_trials





    gospel_dict= {"A1_first_or_only":[17,18,20,21,22],
        "A1_after_B1":[15,16,19,23,24,25,26,27,28],
        "B1_first_or_only":[30,31,32,37,38,39,40,41],
        "B1_after_A1":[29,33,34,35,36]
    }

    if use_pure:
        idx_of_interest = gospel_dict["A1_first_or_only"]
    else:
        idx_of_interest = gospel_dict["A1_after_B1"]

    data_list = []
    data_list_flat = []


    clean_resid_activity_dict_NDNF_newest = {}

    clean_velocity_dict_NDNF_newest = {}

    clean_lick_dict_NDNF_newest = {}

    save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_fixed_model.pkl'
    with open(save_path, 'rb') as f:
        sliceTCA_model = pickle.load(f)


    with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/better_NDNF_fixed_model_dict_clean.pkl', 'rb') as f:
        NDNF_model_dict_clean  = pickle.load(f) 


    labels_dict_raw_new = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

    labels_raw = np.asarray(labels_dict_raw_new[2])


    binary_array = np.zeros(len(labels_raw))

    count = 0

    NDNF_model_dict_clean_clean = {}

    for idx, animal in enumerate(residual_activity_dict_NDNF):
        if not use_all:
            NDNF_model_dict_clean_clean_cell = {}
            if idx in idx_of_interest:
                clean_resid_activity_dict_NDNF_newest[animal] = residual_activity_dict_NDNF[animal]
                clean_velocity_dict_NDNF_newest[animal] = factors_dict_NDNF[animal]
                clean_lick_dict_NDNF_newest [animal] = animal_lick_dict[animal]
                for idt, cell in enumerate(residual_activity_dict_NDNF[animal]):
                    trunc_data = residual_activity_dict_NDNF[animal][cell][:,:min_trials]
                    data_list.append(trunc_data)
                    data_list_flat.append(trunc_data.flatten())

                    NDNF_model_dict_clean_clean_cell[cell] = NDNF_model_dict_clean[animal][cell]

                    binary_array[count] =1 

                    count+=1

            NDNF_model_dict_clean_clean[animal] = NDNF_model_dict_clean_clean_cell

        else:
            NDNF_model_dict_clean_clean_cell = {}
            if 14 < idx < 29:
                clean_resid_activity_dict_NDNF_newest[animal] = residual_activity_dict_NDNF[animal]
                clean_velocity_dict_NDNF_newest[animal] = factors_dict_NDNF[animal]
                clean_lick_dict_NDNF_newest [animal] = animal_lick_dict[animal]
                for cell in residual_activity_dict_NDNF[animal]:
                    trunc_data = residual_activity_dict_NDNF[animal][cell][:,:min_trials]
                    data_list.append(trunc_data)
                    data_list_flat.append(trunc_data.flatten())

                    NDNF_model_dict_clean_clean_cell[cell] = NDNF_model_dict_clean[animal][cell]

                    binary_array[count] =1 

                    count+=1

            NDNF_model_dict_clean_clean[animal] = NDNF_model_dict_clean_clean_cell

    idx_of_interest = np.where(binary_array==1)[0]
    labels = labels_raw[idx_of_interest] 

    cell_type_0 = np.where(labels==0)[0]
    cell_type_1 = np.where(labels==1)[0]



    group_array = np.array(data_list).T


    n_pos = 50
    t = np.arange(min_trials).reshape(-1, 1)   # trials 0…min_trials-1

    slopes = np.zeros(len(data_list))

    for idx in range(len(data_list)):
        
        data = data_list[idx]                           # (n_pos * n_trials,)
        
        av_activity_per_trial = data.mean(axis=0) 
        
        reg = LinearRegression().fit(t, av_activity_per_trial)
        slopes[idx] = reg.coef_[0]    


    slopes_0 = slopes[cell_type_0]
    slopes_1 = slopes[cell_type_1]


    stat, p_rank = ranksums(slopes_0, slopes_1)

    stat, p_ks = ks_2samp(slopes_0, slopes_1)


    fig, axs = plt.subplots(5,5, figsize=(12,12))


    if not use_all:
        if use_pure:
            fig.suptitle("Seen This Track First")
        else:
            fig.suptitle("Seen Other Track First")
        
    else:
        fig.suptitle("All of Fixed Track Sessions")




    fig.subplots_adjust(
    wspace=0.6,
    hspace=0.6)


    save_path = '/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/ndnf_cell_types/models_for_celltypes_x00.pkl'  
    with open(save_path, 'rb') as f:
        models_for_celltypes_x00 = pickle.load(f)

    components_type_0 = models_for_celltypes_x00["components_type_0"]
    components_type_1 = models_for_celltypes_x00["components_type_1"]
    model_type_0 = models_for_celltypes_x00["model_type_0"]
    model_type_1 = models_for_celltypes_x00["model_type_1"]

    neuron_predicted_trial_num_0, reg0 = get_predicted_trial_number(components_type_0)
    neuron_predicted_trial_num_1, reg1 = get_predicted_trial_number(components_type_1)

    trial_idx = np.arange(min_trials)


    m0, b0 = scatter_with_bestfit(axs[0,0], neuron_predicted_trial_num_0, trial_idx, "b", "Cell Type 0")
    m1, b1 = scatter_with_bestfit(axs[0,0], neuron_predicted_trial_num_1, trial_idx, "orange", "Cell Type 1")

    axs[0,0].set_xlabel("Predicted trial index (from model)")
    axs[0,0].set_ylabel("True trial index")
    # axs[0,0].plot(trial_idx, alpha=0.3, linestyle="--", color='g', label="Indentity")
    axs[0,0].legend()


    plot_bars_from_predicted(neuron_predicted_trial_num_0, neuron_predicted_trial_num_1, ax=axs[0,1], title="Prediction Accuracy")


    components_type_0_shuffled = models_for_celltypes_x00["components_type_0_shuffled"]
    components_type_1_shuffled = models_for_celltypes_x00["components_type_1_shuffled"]


    shuffled_neuron_predicted_trial_num_0, reg0_scatter = get_predicted_trial_number(components_type_0_shuffled)
    shuffled_neuron_predicted_trial_num_1, reg1_scatter = get_predicted_trial_number(components_type_1_shuffled)
            
    m0, b0 = scatter_with_bestfit(axs[0,2], shuffled_neuron_predicted_trial_num_0, trial_idx, "b", "Cell Type 0")
    m1, b1 = scatter_with_bestfit(axs[0,2], shuffled_neuron_predicted_trial_num_1, trial_idx, "orange", "Cell Type 1")

    # axs[0,2].plot(shuffled_neuron_predicted_trial_num_0)
    # axs[0,2].plot(shuffled_neuron_predicted_trial_num_1)

    axs[0,2].set_xlabel("Predicted trial index (from model)")
    axs[0,2].set_ylabel("True trial index")
    axs[0,2].legend()

    # axs[0,2].plot(trial_idx, alpha=0.3, linestyle="--", color='g', label="Identity")


    plot_bars_from_predicted(shuffled_neuron_predicted_trial_num_0, shuffled_neuron_predicted_trial_num_1, ax=axs[0,3], title="Shuffled Prediction Accuracy")







    axs[1,0].boxplot([slopes_0, slopes_1], labels=["cluster 0", "cluster 1"])
    axs[1,0].axhline(0, color="k", lw=1, alpha=0.4)
    axs[1,0].set_ylabel("Slope of Mean Activity Across Trials")
    axs[1,0].set_title(f"Line Best Fit Activity Scalar Per Trial \n p={p_rank:.3f}")

    axs[1,1].hist(slopes_0, bins=25, alpha=0.5, label="cluster 0")
    axs[1,1].hist(slopes_1, bins=25, alpha=0.5, label="cluster 1")
    axs[1,1].set_title(f"Learning-related mean drift over trials \n KS test p={p_ks:.3f}")
    axs[1,1].legend()
    

    X_all = np.array(data_list_flat).T #posxtrials, cells

    pos_idx   = np.tile(np.arange(n_pos), min_trials)         # 0..49 repeated per trial
    trial_idx = np.repeat(np.arange(min_trials), n_pos)       # 0..n_trials-1, each repeated 50


    space_labels = (pos_idx >= 25).astype(int)   # 0 = pre (<25), 1 = post (>=25)

    lda_space = LinearDiscriminantAnalysis(n_components=1)
    LDA1 = lda_space.fit_transform(X_all, space_labels).ravel()
    print(f"X_all.shape {X_all.shape} space_labels.shape {space_labels.shape}")
    space_axis = lda_space.coef_[0]        # axis in cell-space or latent-space


    reg_trial = LinearRegression()

    reg_trial.fit(X_all, trial_idx)

    LDA2 = reg_trial.predict(X_all)   # also shape (4700,)




     # axis in the same feature space (cells / latents)
    trial_axis = reg_trial.coef_
    trial_axis /= np.linalg.norm(trial_axis)      # normalize just for convenience


    space_axis_0 = space_axis[cell_type_0]
    space_axis_1 = space_axis[cell_type_1]

    trial_axis_0 = trial_axis[cell_type_0]
    trial_axis_1 = trial_axis[cell_type_1]

    stat, p_learn = ks_2samp(trial_axis_0, trial_axis_1)


    stat, p_ks = ks_2samp(space_axis_0, space_axis_1)

    # ---- Trial axis weights ----
    axs[1,2].hist(trial_axis_0, bins=20, alpha=0.6, label="type 0")
    axs[1,2].hist(trial_axis_1, bins=20, alpha=0.6, label="type 1")
    axs[1,2].set_title(f"Learning axis weights (trial) KS p={p_learn}")
    axs[1,2].set_xlabel("Regression weight per cell")
    axs[1,2].set_ylabel("Number of cells")
    axs[1,2].legend()




    axs[1,3].set_ylabel("Regression Weight")
    axs[1,3].boxplot([trial_axis_0, trial_axis_1],
             tick_labels=["type 0", "type 1"])
    axs[1,3].set_title("Learning axis weights full activity")


    axs[2,0].bar(np.arange(2), [np.var(trial_axis_0),np.var(trial_axis_1)])
    axs[2,0].set_xticks(np.arange(2), ["Cell Type 0", "Cell Type 1"])
    axs[2,0].set_ylabel("Variance of Regressor Weights")



    # with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_fixed_model_dict_clean.pkl', 'rb') as f:
    #     NDNF_model_dict_clean  = pickle.load(f) 

    print(f"NDNF_model_dict_clean_clean.keys() {NDNF_model_dict_clean.keys()}")
    
    # contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)


    contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)



    contig_dict = reshape_contig_dict(contig_dict_all_cell_tca, NDNF_model_dict_clean_clean)


    cp_dict_NDNF = get_cp_dict(contig_dict)

    mse_celltype_0 = []
    mse_celltype_1 = []

    count = 0
    
    for idx, animal in enumerate(clean_resid_activity_dict_NDNF_newest):
        for idt, cell in enumerate(clean_resid_activity_dict_NDNF_newest[animal]):
            cps = cp_dict_NDNF[animal][cell]

            cp_early = cps[0]
            cp_late = cps[1]

            data_early = np.mean(clean_resid_activity_dict_NDNF_newest[animal][cell][:,:cp_early], axis=1)
            data_late = np.mean(clean_resid_activity_dict_NDNF_newest[animal][cell][:,cp_late:], axis=1)

            mse = np.mean(np.square(data_early-data_late))

            if count in cell_type_0:
                mse_celltype_0.append(mse)
            else:
                mse_celltype_1.append(mse)
           
            count += 1




   
    lda_space = LinearDiscriminantAnalysis()
    lda_space.fit(X_all, space_labels)
    space_axis = lda_space.coef_[0]   # shape: (n_cells,)

    reg = LinearRegression()
    reg.fit(X_all, trial_idx)
    trial_axis = reg.coef_              # shape: (n_cells,)

    space_w_0 = space_axis[cell_type_0]
    space_w_1 = space_axis[cell_type_1]

    trial_w_0 = trial_axis[cell_type_0]
    trial_w_1 = trial_axis[cell_type_1]



    data_list_stacked = np.stack(data_list)

    data_list_0 = []

    data_list_1 = []

    print(f"data_list_stacked.shape {data_list_stacked.shape}")

    for cell in range(data_list_stacked.shape[0]):
        if cell in cell_type_0:
            data_list_0.append(data_list_stacked[cell,:,:])
        else:
            data_list_1.append(data_list_stacked[cell,:,:])


    data_array_0 = np.array(data_list_0)
    data_array_1 = np.array(data_list_1)

    data_array_0 = data_array_0[:74,:,:]
    data_array_1 = data_array_1[:74,:,:]

    X_all_0 = data_array_0.reshape(
        data_array_0.shape[0] * data_array_0.shape[1],
        data_array_0.shape[2]
    ).T

    X_all_1 = data_array_1.reshape(
    data_array_1.shape[0] * data_array_1.shape[1],
    data_array_1.shape[2]
).T



    X_all = np.array(data_list_flat).T #posxtrials, cells

    pos_idx   = np.tile(np.arange(n_pos), min_trials)         # 0..49 repeated per trial
    trial_idx = np.repeat(np.arange(min_trials), n_pos)       # 0..n_trials-1, each repeated 50


    space_labels = (pos_idx >= 25).astype(int)   # 0 = pre (<25), 1 = post (>=25)

    lda_space = LinearDiscriminantAnalysis(n_components=1)
    LDA1 = lda_space.fit_transform(X_all, space_labels).ravel()
    print(f"X_all.shape {X_all.shape} space_labels.shape {space_labels.shape}")
    space_axis = lda_space.coef_[0]        # axis in cell-space or latent-space


    reg_trial = LinearRegression()

    reg_trial.fit(X_all, trial_idx)

    LDA2 = reg_trial.predict(X_all)   # also shape (4700,)




    data_mean = np.mean(data_list_stacked, axis=1)  # (cells, trials) = (174, 94)

    X_trial = data_mean.T
    y_trial = np.arange(X_trial.shape[0])

    reg = Ridge(alpha=1.0)
    reg.fit(X_trial, y_trial)

    reg_score = reg.predict(X_trial)   # (94,)  <-- one per trial

    coefficients = reg.coef_ #one per cell 


    lda = LinearDiscriminantAnalysis()
    lda.fit(data_mean, labels)

    lda_score = lda.transform(data_mean).ravel()   # one per cell 

    _, p = ranksums(coefficients[labels==0], coefficients[labels==1])


    def rotz(deg):
        th = np.deg2rad(deg)
        return np.array([[ np.cos(th), -np.sin(th), 0],
                        [ np.sin(th),  np.cos(th), 0],
                        [ 0,           0,          1]])


    def plot_PCA_one_celltype(model_type_1_array, ax0):
        transposed = model_type_1_array.transpose(0, 2, 1) 
        n_trials, n_cells, n_pos = transposed.shape

        X = transposed.reshape(n_trials * n_cells, n_pos)

        pca = PCA(n_components=3).fit(X)
        Z = pca.transform(X)

        Z_reshaped = Z.reshape(n_trials, n_cells, 3) 
        print(Z_reshaped.shape)

        E_euc_dist_list = []
        M_euc_dist_list = []
        L_euc_dist_list = []

        E_trial_data = []
        M_trial_data = []
        L_trial_data = []
        
        cos_angle_per_trial_E = []
        cos_angle_per_trial_M = []
        cos_angle_per_trial_L = []

        for trial in range(Z_reshaped.shape[0]):
            trial_data = Z_reshaped[trial,:,:]

            diffs = np.diff(trial_data, axis=0)  

            diffs_list = []

            for i in range(len(diffs)):
                if i < len(diffs)-1:
                    v1 = diffs[i]
                    v2 = diffs[i+1]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                    diffs_list.append(cos_angle)

            euclid_dist = np.linalg.norm(diffs, axis=1)
            
            if trial <20:
                E_euc_dist_list.append(euclid_dist)
                E_trial_data.append(trial_data)
                cos_angle_per_trial_E.append(diffs_list)
            elif trial>73:
                L_euc_dist_list.append(euclid_dist)
                L_trial_data.append(trial_data)
                cos_angle_per_trial_M.append(diffs_list)
            else:
                M_euc_dist_list.append(euclid_dist)
                M_trial_data.append(trial_data)
                cos_angle_per_trial_L.append(diffs_list)


        R = rotz(67.5)


        E_trial_data_array = np.array(E_trial_data) 
        mean_E_trial_data_array = np.mean(E_trial_data_array, axis=0) @ R.T

        print(f"mean_E_trial_data_array.shape {mean_E_trial_data_array.shape}")

        M_trial_data_array = np.array(M_trial_data)
        mean_M_trial_data_array = np.mean(M_trial_data_array, axis=0) @ R.T

        L_trial_data_array = np.array(L_trial_data) 
        mean_L_trial_data_array = np.mean(L_trial_data_array, axis=0) @ R.T


        print(E_trial_data_array.shape)

        pos = np.arange(50)

        order = np.argsort(pos)  # so lines follow position


        sc = ax0.scatter(mean_E_trial_data_array[:,2],  mean_E_trial_data_array[:,1],  mean_E_trial_data_array[:,0],
                        c=pos, cmap="viridis", s=18, alpha=0.5, marker="o", label="early")
        ax0.scatter(mean_M_trial_data_array[:,2], mean_M_trial_data_array[:,1], mean_M_trial_data_array[:,0],
                c=pos, cmap="viridis", s=18, alpha=0.5, marker="^", label="middle")
        ax0.scatter(mean_L_trial_data_array[:,2],   mean_L_trial_data_array[:,1],   mean_L_trial_data_array[:,0],
                c=pos, cmap="viridis", s=18, alpha=0.5, marker="s", label="late")

        ax0.plot(mean_E_trial_data_array[order,2],  mean_E_trial_data_array[order,1],  mean_E_trial_data_array[order,0],  alpha=0.6, lw=0.5)
        ax0.plot(mean_M_trial_data_array[order,2], mean_M_trial_data_array[order,1], mean_M_trial_data_array[order,0], alpha=0.6, lw=0.5)
        ax0.plot(mean_L_trial_data_array[order,2],   mean_L_trial_data_array[order,1],   mean_L_trial_data_array[order,0],   alpha=0.6, lw=0.5)

        ax0.set_xlabel("PC3"); ax0.set_ylabel("PC2"); ax0.set_zlabel("PC1")
        # ax0.set_xlim(-5,4); ax0.set_ylim(-5,10); ax0.set_zlim(-5,9)
        ax0.legend()
        return sc, E_euc_dist_list, M_euc_dist_list, L_euc_dist_list, cos_angle_per_trial_E, cos_angle_per_trial_M, cos_angle_per_trial_L  # return mappable for colorbar


    gs = axs[0,0].get_gridspec()

    # remove only the axes you’re replacing: (2,0),(3,0),(2,1),(3,1)
    for r in [2, 3]:
        for c in [0, 1]:
            axs[r, c].remove()

    # each new plot spans 2 rows x 1 col
    ax0 = fig.add_subplot(gs[2:4, 0], projection="3d")  # uses (2,0)+(3,0)
    ax1 = fig.add_subplot(gs[2:4, 1], projection="3d")  # uses (2,1)+(3,1)

    save_path = '/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/ndnf_cell_types/models_for_celltypes_x00.pkl'  
    with open(save_path, 'rb') as f:
        models_for_celltypes_x00 = pickle.load(f)

    model_type_0 = models_for_celltypes_x00["model_type_0"]
    model_type_1 = models_for_celltypes_x00["model_type_1"]

    component_type_0 = models_for_celltypes_x00["components_type_0"]
    component_type_1 = models_for_celltypes_x00["components_type_1"]

    model_type_0_array = model_type_0.construct().numpy(force=True)
    model_type_1_array = model_type_1.construct().numpy(force=True)

    if use_all:
        sc, E_euc_dist_list0, M_euc_dist_list0, L_euc_dist_list0, cos_angle_per_trial_E0, cos_angle_per_trial_M0, cos_angle_per_trial_L0 = plot_PCA_one_celltype(model_type_0_array, ax0)
        sc, E_euc_dist_list1, M_euc_dist_list1, L_euc_dist_list1, cos_angle_per_trial_E1, cos_angle_per_trial_M1, cos_angle_per_trial_L1 = plot_PCA_one_celltype(model_type_1_array, ax1)

    else:
        data_celltype0 = []
        data_celltype1 = []

        if use_pure:
            used_sessions = gospel_dict["A1_first_or_only"]
        else:
            used_sessions = gospel_dict["A1_after_B1"]

        print(f"clean_resid_activity_dict_NDNF_newest.keys() {clean_resid_activity_dict_NDNF_newest.keys()}")
        
        count = 0
        for idx, animal in enumerate(clean_resid_activity_dict_NDNF_newest):
            for idt, cell in enumerate(clean_resid_activity_dict_NDNF_newest[animal]):
                if count in cell_type_0:
                    data_celltype0.append(clean_resid_activity_dict_NDNF_newest[animal][cell][:,:94])
                else:
                    data_celltype1.append(clean_resid_activity_dict_NDNF_newest[animal][cell][:,:94])

                count+=1

        data_celltype0_array = np.array(data_celltype0)
        data_celltype1_array = np.array(data_celltype1)

        print(f"data_celltype0_array.shape {data_celltype0_array.shape}")

        type_0_array = data_celltype0_array.transpose(2,0,1)
        type_1_array = data_celltype1_array.transpose(2,0,1)


        sc, E_euc_dist_list0, M_euc_dist_list0, L_euc_dist_list0, cos_angle_per_trial_E0, cos_angle_per_trial_M0, cos_angle_per_trial_L0 = plot_PCA_one_celltype(type_0_array, ax0)
        sc, E_euc_dist_list1, M_euc_dist_list1, L_euc_dist_list1, cos_angle_per_trial_E1, cos_angle_per_trial_M1, cos_angle_per_trial_L1 = plot_PCA_one_celltype(type_1_array, ax1)
    

    def plot_euc_dist(E_euc_dist_list, label=None, ax=None):

        E_euc_dist_array = np.array(E_euc_dist_list)
        mean_E_euc_dist_array = np.mean(E_euc_dist_array, axis=0)
        sem_E_euc_dist_array = sem(E_euc_dist_array, axis=0)

        ax.plot(mean_E_euc_dist_array, label=label)
        ax.fill_between(range(len(mean_E_euc_dist_array)), mean_E_euc_dist_array+sem_E_euc_dist_array, mean_E_euc_dist_array-sem_E_euc_dist_array, alpha=0.3)

    plot_euc_dist(E_euc_dist_list0, label="Early 0", ax=axs[4,0])
    plot_euc_dist(M_euc_dist_list0, label="Middle 0", ax=axs[4,0])
    plot_euc_dist(L_euc_dist_list0, label="Late 0", ax=axs[4,0])

    axs[4,0].legend()

    plot_euc_dist(E_euc_dist_list1, label="Early 1", ax=axs[4,1])
    plot_euc_dist(M_euc_dist_list1, label="Middle 1", ax=axs[4,1])
    plot_euc_dist(L_euc_dist_list1, label="Late 1", ax=axs[4,1])
    
    axs[4,1].legend()

    plot_euc_dist(cos_angle_per_trial_E0, label="Early 0", ax=axs[3,4])
    plot_euc_dist(cos_angle_per_trial_M0, label="Middle 0", ax=axs[3,4])
    plot_euc_dist(cos_angle_per_trial_L0, label="Late 0", ax=axs[3,4])

    plot_euc_dist(cos_angle_per_trial_E1, label="Early 1", ax=axs[4,4])
    plot_euc_dist(cos_angle_per_trial_M1, label="Middle 1", ax=axs[4,4])
    plot_euc_dist(cos_angle_per_trial_L1, label="Late 1", ax=axs[4,4])

    axs[3,4].set_title("Cosine Angle Average per Trial")
    axs[4,4].set_title("Cosine Angle Average per Trial")



    def euclidian_dist_var(E_euc_dist_list):
        var_list = []
        for i in range(len(E_euc_dist_list)):
            dist= E_euc_dist_list[i]
            var = np.var(dist)
            var_list.append(var)

        return var_list
    
    def euclidian_dist_sum(E_euc_dist_list):
        sum_list = []
        for i in range(len(E_euc_dist_list)):
            dist= E_euc_dist_list[i]
            sums = np.sum(dist)
            sum_list.append(sums)

        return sum_list


    def plot_the_lists(var_listE, var_listM, var_listL, ax=None, celltype=None, title=None):
        varE = np.array(var_listE)
        varM = np.array(var_listM)
        varL = np.array(var_listL)

        means = [varE.mean(), varM.mean(), varL.mean()]
        sems  = [sem(varE), sem(varM), sem(varL)]

        x = np.arange(3)
        labels = ["Early", "Middle", "Late"]

        ax.bar(x, means, yerr=sems, capsize=6, width=0.6, alpha=0.8)

        rng = np.random.default_rng(0)
        for i, vals in enumerate([varE, varM, varL]):
            jitter = (rng.random(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), x[i]) + jitter, vals,
                    s=18, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(f"{title} of Euclidean distance")
        ax.set_title(f"{celltype} {title} Per Trial For Learning Block")
        ax.axhline(0, linewidth=0.8)



    var_listE0 = euclidian_dist_var(E_euc_dist_list0)
    var_listM0 = euclidian_dist_var(M_euc_dist_list0)
    var_listL0 = euclidian_dist_var(L_euc_dist_list0)

    var_listE1 = euclidian_dist_var(E_euc_dist_list1)
    var_listM1 = euclidian_dist_var(M_euc_dist_list1)
    var_listL1 = euclidian_dist_var(L_euc_dist_list1)

    plot_the_lists(var_listE0, var_listM0, var_listL0, ax=axs[2,2], celltype="Cell Type 0", title="Variance")
    plot_the_lists(var_listE1, var_listM1, var_listL1, ax=axs[2,3], celltype="Cell Type 1", title="Variance")

    sum_listE0 = euclidian_dist_var(E_euc_dist_list0)
    sum_listM0 = euclidian_dist_var(M_euc_dist_list0)
    sum_listL0 = euclidian_dist_var(L_euc_dist_list0)

    sum_listE1 = euclidian_dist_sum(E_euc_dist_list1)
    sum_listM1 = euclidian_dist_sum(M_euc_dist_list1)
    sum_listL1 = euclidian_dist_sum(L_euc_dist_list1)

    plot_the_lists(sum_listE0, sum_listM0, sum_listL0, ax=axs[3,2], celltype="Cell Type 0", title="Sum Length")
    plot_the_lists(sum_listE1, sum_listM1, sum_listL1, ax=axs[3,3], celltype="Cell Type 1", title="Sum Length")



    def get_behav_list(clean_velocity_dict_NDNF_newest, behav="Velocity"):
        velocity_list = []

        for animal in clean_velocity_dict_NDNF_newest:
            velocity_list.append(clean_velocity_dict_NDNF_newest[animal][behav][:,:94])

        velocity_array = np.array(velocity_list)

        mean_velocity_array = np.mean(velocity_array, axis=0)

        return mean_velocity_array


    def get_r_array(velocity_array, euclid_dist_array):
        r_list = []

        for trial in range(velocity_array.shape[1]):
            velocity_for_trial = velocity_array[:,trial]
            euclid_dist_array_trial = euclid_dist_array[trial,:]
            if not np.any(np.isnan(velocity_for_trial)) and np.std(velocity_for_trial[:-1])!=0 and np.std(euclid_dist_array_trial)!=0:
                r, _ = pearsonr(velocity_for_trial[:-1], euclid_dist_array_trial)
                r_list.append(r)

        r_array = np.array(r_list)
        return r_array


    def plot_corr_euclidian(euclid_dist_array_0, euclid_dist_array_1, clean_lick_dict_NDNF_newest, behav=None, ax=None):

        licks_array = get_behav_list(clean_lick_dict_NDNF_newest, behav=behav)

        r0 = get_r_array(licks_array, euclid_dist_array_0)
        r1 = get_r_array(licks_array, euclid_dist_array_1)

        _,Pval=ttest_ind(r0,r1)

        means = [r0.mean(), r1.mean()]
        sems = [r0.std(ddof=1) / np.sqrt(len(r0)), r1.std(ddof=1) / np.sqrt(len(r1))]

        x = np.array([0, 1])

        ax.bar(x, means, yerr=sems, capsize=6, width=0.6)

        rng = np.random.default_rng(0)
        jitter0 = (rng.random(len(r0)) - 0.5) * 0.18
        jitter1 = (rng.random(len(r1)) - 0.5) * 0.18
        ax.scatter(np.full(len(r0), x[0]) + jitter0, r0, s=14, alpha=0.6)
        ax.scatter(np.full(len(r1), x[1]) + jitter1, r1, s=14, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Cell Type 0 n={len(r0)}", f"Cell Type 1 n={len(r1)}"])
        ax.set_ylabel(f"Pearson r {behav} vs PCA-step distance")
        ax.set_title(f"p = {Pval:.3g}")

        ax.axhline(0, linewidth=0.8)

    def get_euclidian_dist_array(overall_data_array_0):
        overall_data_array_0_2d = overall_data_array_0.reshape(
            overall_data_array_0.shape[0] * overall_data_array_0.shape[1],
            overall_data_array_0.shape[2])
        pca = PCA(n_components=3).fit(overall_data_array_0_2d)
        pca_0 = pca.transform(overall_data_array_0_2d)

        pca_0_3d = pca_0.reshape(overall_data_array_0.shape[0],  # 94
                                overall_data_array_0.shape[1],  # 50
                                3)  

        euclid_dist_list = []
        for trial in range(pca_0_3d.shape[0]):
            data = pca_0_3d[trial,:,:]
            diffs = np.diff(data, axis=0)                  # (n_points-1, 3)
            euclid_dist = np.linalg.norm(diffs, axis=1)
            euclid_dist_list.append(euclid_dist)


        euclid_dist_array = np.array(euclid_dist_list)
        return euclid_dist_array
    
    # labels_celltype = labels_dict_raw_new[2]

    
    early_data = []
    middle_data = []
    late_data = []

    overall_data = []

    for idx, animal in enumerate(clean_resid_activity_dict_NDNF_newest):
        for idt, cell in enumerate(clean_resid_activity_dict_NDNF_newest[animal]):

            model = NDNF_model_dict_clean_clean[animal][cell]
            model_reco = model.construct().numpy(force=True)

            print(f"clean_resid_activity_dict_NDNF_newest[animal][cell].shape {clean_resid_activity_dict_NDNF_newest[animal][cell].shape} NDNF_model_dict_clean_clean[animal][cell].shape {model_reco.shape} ")

            



            early_cp = cp_dict_NDNF[animal][cell][0]
            late_cp = cp_dict_NDNF[animal][cell][1]
            data_early = clean_resid_activity_dict_NDNF_newest[animal][cell][:,:early_cp]
            early_data.append(np.mean(data_early, axis=1))
            data_late = clean_resid_activity_dict_NDNF_newest[animal][cell][:,late_cp:]
            middle_data.append(np.mean(data_late, axis=1))
            data_middle = clean_resid_activity_dict_NDNF_newest[animal][cell][:,early_cp:late_cp]
            late_data.append(np.mean(data_middle, axis=1))
            overall_data.append(clean_resid_activity_dict_NDNF_newest[animal][cell][:,:94])
            

    early_data_array = np.array(early_data)
    early_data_array_0 = early_data_array[cell_type_0,:].T
    early_data_array_1 = early_data_array[cell_type_1,:].T

    middle_data_array = np.array(middle_data)
    middle_data_array_0 = middle_data_array[cell_type_0,:].T
    middle_data_array_1 = middle_data_array[cell_type_1,:].T

    late_data_array = np.array(late_data)
    late_data_array_0 = late_data_array[cell_type_0,:].T
    late_data_array_1 = late_data_array[cell_type_1,:].T

    overall_data_array = np.array(overall_data)
    overall_data_array_0 = overall_data_array[cell_type_0,:].T
    overall_data_array_1 = overall_data_array[cell_type_1,:].T

    euclid_dist_array_0 = get_euclidian_dist_array(overall_data_array_0)
    euclid_dist_array_1 = get_euclidian_dist_array(overall_data_array_1)

    plot_corr_euclidian(euclid_dist_array_0, euclid_dist_array_1, clean_lick_dict_NDNF_newest, behav="Licks", ax=axs[4,3])
    plot_corr_euclidian(euclid_dist_array_0, euclid_dist_array_1, clean_velocity_dict_NDNF_newest, behav="Velocity", ax=axs[4,2])

    print(f"model_type_0_array.shape {model_type_0_array.shape}")



    plt.tight_layout()
    plt.show()


@click.command()
@click.option('--use_pure/--use_mixed', default=True, help="seen just this track or not ")
@click.option('--use_all/--use_subset', default=True, help="to split or not to split")

def cli(use_pure, use_all):
    run(use_pure, use_all)

if __name__ == "__main__":
    cli()
