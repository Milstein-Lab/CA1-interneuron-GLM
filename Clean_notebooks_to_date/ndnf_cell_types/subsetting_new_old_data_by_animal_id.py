import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca

# import utils as ut
# import plot as pt
plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

import sys
from scipy.stats import sem
sys.path.append('/Users/michaelfinch/CA1-interneuron-GLM')

import click

from spiking_model_utils import *

import datetime as dt

from collections import Counter

from matplotlib.lines import Line2D

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from collections import defaultdict

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import ruptures as rpt





def plot_session_contributions_new_vs_old():
    res_new = get_session_contribution_counts(use_new_data=True)
    res_old = get_session_contribution_counts(use_new_data=False)

    cond_labels_short = res_new["cond_labels_short"]  # same for old
    n_cond = 4
    x = np.arange(n_cond)

    cond_session_idx_lists_new = res_new["cond_session_idx_lists"]
    cond_session_idx_lists_old = res_old["cond_session_idx_lists"]

    # union of all sessions across both datasets
    all_sessions_union = sorted(set(res_new["all_sessions"]) | set(res_old["all_sessions"]))

    # shared color map across both
    cmap = plt.get_cmap("tab20")
    n_colors = cmap.N
    color_map = {sess: cmap(i % n_colors) for i, sess in enumerate(all_sessions_union)}

    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharey=True)


    def plot_one_panel(ax, res, title_suffix):
        """
        Plot stacked bars for one dataset (NEW or OLD),
        and print each session name + n inside its stacked segment.
        """
        counts = res["counts"]
        total_n_all = res["total_n_all"]

        # running bottoms for each condition
        bottom = np.zeros(n_cond, dtype=float)

        # stack by session, using union so color assignment is consistent
        for sess in all_sessions_union:
            heights = np.array([counts[cond_idx].get(sess, 0) for cond_idx in range(n_cond)])
            if np.all(heights == 0):
                continue

            # draw each stack segment separately so we can annotate it
            for cond_idx in range(n_cond):
                h = heights[cond_idx]
                if h == 0:
                    continue

                x_pos = x[cond_idx]
                y_bottom = bottom[cond_idx]

                ax.bar(x_pos, h, bottom=y_bottom, color=color_map[sess])

                # text in the middle of this segment
                y_center = y_bottom + h / 2.0
                ax.text(
                    x_pos,
                    y_center,
                    f"{sess}\n(n={h})",
                    ha="center",
                    va="center",
                    fontsize=5,
                )

                # update the bottom for this condition
                bottom[cond_idx] += h

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels_short, rotation=20, ha='right')
        ax.set_ylabel("# neurons")
        ax.set_title(f"{title_suffix}\nTotal n cells across sessions: {total_n_all}")

        # add total n on top of each full bar (sum over sessions)
        for cond_idx in range(n_cond):
            total_n_cond = sum(counts[cond_idx].values())
            if total_n_cond > 0:
                y_pos = bottom[cond_idx] + 0.02 * bottom.max()
                ax.text(
                    x[cond_idx],
                    y_pos,
                    f"n={total_n_cond}",
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )

    plot_one_panel(axs[0], res_new, "NEW data")
    plot_one_panel(axs[1], res_old, "OLD data")

    plt.tight_layout()
    plt.show()

    session_idx_dict = {
        "NEW": {
            cond_labels_short[i]: cond_session_idx_lists_new[i]
            for i in range(len(cond_labels_short))
        },
        "OLD": {
            cond_labels_short[i]: cond_session_idx_lists_old[i]
            for i in range(len(cond_labels_short))
        },
    }

    return session_idx_dict

    
def debug_print_new_sessions():
    base_path = '/Users/michaelfinch/CA1-interneuron-GLM'
    mat_path_new = f"{base_path}/datasets/NDNF_E0A1B1_251107.mat"

    sessions_new = load_sessions_from_mat(mat_path_new)
    meta_new, name_list_new = build_new_meta_from_sessions(sessions_new)
    # meta_new: list of (idx, animal, date_int, cue)
    # name_list_new[idx]: string label, e.g. "CG189_250213_1B"

    idx_session_name_dict = {}

    print("INDEX | SESSION_NAME        | ANIMAL  | DATE   | CUE")
    print("------------------------------------------------------")
    for (idx, animal, date, cue) in meta_new:
        session_name = name_list_new[idx]
        print(f"{idx:5d} | {session_name:18s} | {animal:6s} | {date} | {cue}")

        idx_session_name_dict[idx] = session_name

    return idx_session_name_dict






def session_to_animal_ID_newest():
    """
    Returns a dict:
        session_index (int) -> animal_id (str)
    for the NEW data sessions.
    """

    # Animals that saw both A1 and B1 (your original mapping)
    animals_seen_both_A1_B1_dict = {
        "CG189": [1, 15, 30],
        "CG190": [2, 16, 31],
        "MV177": [9, 21, 35],
        "MV180": [10, 22, 36],
        "MV196": [12, 24, 38],
        "MV219": [13, 26, 40],
        "MV228": [14, 28, 41],
        "MV171": [20, 34],
        "MV191": [23, 37],
        "MV200": [25, 39],
    }

    # Animals that (in your note) saw B1 but not A1
    animals_seen_B1_no_A1 = {
        "CG186": [0, 29],
        "CG191": [3, 32],
        "MV166": [33],
    }

    # Animals that saw A1 but not B1
    animals_seen_A1_no_B1 = {
        "LM084": [4, 17],
        "MV161": [6, 18],
        "MV170": [8, 19],
        "MV222": [27],
    }

    idx_to_animal = {}

    for animal, idx_list in animals_seen_both_A1_B1_dict.items():
        for idx in idx_list:
            idx_to_animal[idx] = animal

    for animal, idx_list in animals_seen_B1_no_A1.items():
        for idx in idx_list:
            idx_to_animal[idx] = animal

    for animal, idx_list in animals_seen_A1_no_B1.items():
        for idx in idx_list:
            idx_to_animal[idx] = animal

    return idx_to_animal


def session_paths_old():
    return [
        '/Volumes/Imaging15/LM083/LM083_240604_E/anal.mat',   # 0
        '/Volumes/Imaging15/LM098/LM098_240807_E/anal.mat',   # 1
        '/Volumes/Imaging15/MV153/MV153_241029_E/anal.mat',   # 2
        '/Volumes/Imaging15/MV161/MV161_241218_E/anal.mat',   # 3
        '/Volumes/Imaging15/CG186/CG186_250123_E/anal.mat',   # 4
        '/Volumes/Imaging15/MV166/MV166_250207_E/anal.mat',   # 5
        '/Volumes/Imaging15/CG189/CG189_250211_E/anal.mat',   # 6
        '/Volumes/Imaging15/MV168/MV168_250214_E/anal.mat',   # 7
        '/Volumes/Imaging15/CG190/CG190_250214_E/anal.mat',   # 8
        '/Volumes/Imaging15/CG191/CG191_250216_E/anal.mat',   # 9
        '/Volumes/Imaging15/MV169/MV169_250222_E/anal.mat',   # 10
        '/Volumes/Imaging15/MV170/MV170_250222_E/anal.mat',   # 11
        '/Volumes/Imaging15/MV171/MV171_250303_E/anal.mat',   # 12
        '/Volumes/Imaging15/MV177/MV177_250326_E/anal.mat',   # 13
        '/Volumes/Imaging15/MV180/MV180_250328_E/anal.mat',   # 14
        '/Volumes/Imaging15/MV191/MV191_250511_E/anal.mat',   # 15
        '/Volumes/Imaging15/MV196/MV196_250523_E/anal.mat',   # 16
        '/Volumes/Imaging15/MV197/MV197_250523_E/anal.mat',   # 17
        '/Volumes/Imaging15/LM084/LM084_240610_1A/anal.mat',  # 18
        '/Volumes/Imaging15/LM098/LM098_240809_1A/anal.mat',  # 19
        '/Volumes/Imaging15/MV161/MV161_241219_1A/anal.mat',  # 20
        '/Volumes/Imaging15/CG186/CG186_250125_1A/anal.mat',  # 21
        '/Volumes/Imaging15/MV166/MV166_250210_1A/anal.mat',  # 22
        '/Volumes/Imaging15/CG189/CG189_250215_1A/anal.mat',  # 23
        '/Volumes/Imaging15/CG190/CG190_250220_1A/anal.mat',  # 24
        '/Volumes/Imaging15/CG191/CG191_250220_1A/anal.mat',  # 25
        '/Volumes/Imaging15/MV170/MV170_250228_1A/anal.mat',  # 26
        '/Volumes/Imaging15/MV171/MV171_250306_1A/anal.mat',  # 27
        '/Volumes/Imaging15/MV191/MV191_250516_1A/anal.mat',  # 28
        '/Volumes/Imaging15/MV196/MV196_250528_1A/anal.mat',  # 29
        '/Volumes/Imaging15/MV197/MV197_250530_1A/anal.mat',  # 30
        '/Volumes/Imaging15/CG189/CG189_250213_1B/anal.mat',  # 31
        '/Volumes/Imaging15/CG190/CG190_250217_1B/anal.mat',  # 32
        '/Volumes/Imaging15/CG191/CG191_250218_1B/anal.mat',  # 33
        '/Volumes/Imaging15/MV171/MV171_250315_1B/anal.mat',  # 34
        '/Volumes/Imaging15/MV177/MV177_250406_1B/anal.mat',  # 35
        '/Volumes/Imaging15/MV180/MV180_250408_1B/anal.mat',  # 36
        '/Volumes/Imaging15/MV191/MV191_250514_1B/anal.mat',  # 37
        '/Volumes/Imaging15/MV196/MV196_250526_1B/anal.mat',  # 38
    ]



def preprocess_data2(filepath, normalize=True, new_NDNF=False, use_final=False):
    factors_dict = {}
    activity_dict = {}

    if new_NDNF:
        with h5py.File(filepath, 'r') as f:
            if use_final:
                animal_group = f['animals']
            else:
                animal_group = f['animal']
            shiftR_refs = animal_group['ShiftR'][:]
            shiftRunning_refs = animal_group['ShiftRunning'][:]
            shiftL_refs = animal_group['ShiftL'][:]
            shiftV_refs = animal_group['ShiftV'][:]

            for animal_idx in range(len(shiftR_refs)):
                delta_f = np.array(f[shiftR_refs[animal_idx][0]])
                delta_f = delta_f.swapaxes(0, 2)
                velocity = np.array(f[shiftRunning_refs[animal_idx][0]]).T
                lick_rate = np.array(f[shiftL_refs[animal_idx][0]]).T
                reward_loc = np.array(f[shiftV_refs[animal_idx][0]]).T

                if delta_f.shape[1] > 1:
                    delta_f = delta_f[:, 1:, :]  # remove duplicate neuron

                num_trials = min(delta_f.shape[1], velocity.shape[1], lick_rate.shape[1], reward_loc.shape[1])

                delta_f = delta_f[:, :num_trials, :]
                velocity = velocity[:, :num_trials]
                lick_rate = lick_rate[:, :num_trials]
                reward_loc = reward_loc[:, :num_trials]

                nan_trials = (
                        np.any(np.isnan(lick_rate), axis=0) |
                        np.any(np.isnan(reward_loc), axis=0) |
                        np.any(np.isnan(velocity), axis=0) |
                        np.any(np.isnan(delta_f), axis=(0, 2))
                )

                animal_key = f'animal_{animal_idx + 1}'
                factors_dict[animal_key] = {
                    "Licks": lick_rate[:, ~nan_trials],
                    "Reward_loc": reward_loc[:, ~nan_trials],
                    "Velocity": velocity[:, ~nan_trials]
                }

                if normalize:
                    for var in factors_dict[animal_key]:
                        factors_dict[animal_key][var] = ((factors_dict[animal_key][var] - np.min(factors_dict[animal_key][var])) /
                                                         (np.max(factors_dict[animal_key][var]) - np.min(factors_dict[animal_key][var])))

                activity_dict[animal_key] = {}
                for neuron_idx in range(delta_f.shape[2]):  # loop over neurons
                    neuron_activity = delta_f[:, :, neuron_idx]  # (trial, bin)
                    if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0):
                        continue

                    cleaned_activity = neuron_activity[:, ~nan_trials]
                    if normalize:
                        cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)
                    neuron_key = f'cell_{neuron_idx + 1}'
                    activity_dict[animal_key][neuron_key] = cleaned_activity


    else:
        data_dict = mat73.loadmat(filepath)

        # Setup position variables
        num_spatial_bins = 10
        position_matrix = np.zeros((50, num_spatial_bins))
        bin_size = 50 // num_spatial_bins
        for i in range(num_spatial_bins):
            position_matrix[i * bin_size:(i + 1) * bin_size, i] = 1

        for animal_idx, (delta_f, velocity, lick_rate, reward_loc) in enumerate(
                zip(data_dict['animal']['ShiftR'], data_dict['animal']['ShiftRunning'], data_dict['animal']['ShiftLrate'], data_dict['animal']['ShiftV'])):

            num_trials = min(delta_f.shape[1], lick_rate.shape[1], reward_loc.shape[1], velocity.shape[1])
            delta_f = delta_f[:, :num_trials, :]
            velocity = velocity[:, :num_trials]
            lick_rate = lick_rate[:, :num_trials]
            reward_loc = reward_loc[:, :num_trials]

            nan_trials = (
                    np.any(np.isnan(lick_rate), axis=0) |
                    np.any(np.isnan(reward_loc), axis=0) |
                    np.any(np.isnan(velocity), axis=0) |
                    np.any(np.isnan(delta_f), axis=(0, 2)))

            animal_key = f'animal_{animal_idx + 1}'
            factors_dict[animal_key] = {
                "Licks": lick_rate[:, ~nan_trials],
                "Reward_loc": reward_loc[:, ~nan_trials],
                "Velocity": velocity[:, ~nan_trials]}

            # Add position info
            num_trials = factors_dict[animal_key]["Velocity"].shape[1]
            for bin_idx in range(num_spatial_bins):
                bin_key = f"Position_{bin_idx + 1}"
                factors_dict[animal_key][bin_key] = np.tile(position_matrix[:, bin_idx][:, np.newaxis], num_trials)

            if normalize:
                for var in factors_dict[animal_key]:
                    factors_dict[animal_key][var] = (
                            (factors_dict[animal_key][var] - np.min(factors_dict[animal_key][var])) /
                            (np.max(factors_dict[animal_key][var]) - np.min(factors_dict[animal_key][var])))

            activity_dict[animal_key] = {}
            for neuron_idx in range(delta_f.shape[2]):
                neuron_activity = delta_f[:, :, neuron_idx]
                if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0):
                    continue
                cleaned_activity = neuron_activity[:, ~nan_trials]
                if normalize:
                    cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)
                neuron_key = f'cell_{neuron_idx + 1}'
                activity_dict[animal_key][neuron_key] = cleaned_activity

    return activity_dict, factors_dict


def load_data_regular(file_path=r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM", name="NDNFanalC", new_NDNF=True, use_final=False):
    file_path = file_path
    filename = name
    filepath = os.path.join(file_path, "datasets", filename + ".mat")

    activity_dict, factors_dict = preprocess_data2(filepath, normalize=True, new_NDNF=new_NDNF, use_final=use_final)

    filtered_factors_dict = subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

    GLM_params, double_predicted_activity_dict_NDNF_new = fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
    double_residual_activity_dict_NDNF_new = get_residual_activity_dict(activity_dict, double_predicted_activity_dict_NDNF_new)

    return GLM_params, activity_dict, double_predicted_activity_dict_NDNF_new, factors_dict, filtered_factors_dict, double_residual_activity_dict_NDNF_new


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

def subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"]):
    filtered_factors_dict = {}
    for animal in factors_dict:
        filtered_factors_dict[animal] = {}
        for variable in variables_to_keep:
            filtered_factors_dict[animal][variable] = factors_dict[animal][variable]
    return filtered_factors_dict



def get_labels_all_different_Ks_single(model_20_NDNF_resid, which_vectors: int):
    
    w1 = model_20_NDNF_resid.vectors[which_vectors][0]
    f1 = model_20_NDNF_resid.vectors[which_vectors][1]
    F = f1.detach().cpu().numpy()   # (latents, cells, pos) = (20, 115, 50)
    W = w1.detach().cpu().numpy()   # (latents, trials) = (20, 100)

    # Build X so rows = cells (115)
    if which_vectors == 0:
        # Use latent×pos per cell, flattened: (115, 20*50)
        X = np.moveaxis(F, 1, 0)              # (cells=115, latents=20, pos=50)
        X = X.reshape(X.shape[0], -1)         # (115, 1000)

    elif which_vectors == 1:
        X = W.T  # (115, 20) mean over pos

    else:
        X = np.moveaxis(F, 2, 0)     # -> (cells=115, latents=20, trials=100)
        X = X.reshape(X.shape[0], -1)  # -> (115, 20*100) = (115, 2000)

    # Standardize then KMeans over K=1..10
    Xz = StandardScaler().fit_transform(X)
    labels_cells_dict_all_K = {K: KMeans(n_clusters=K, n_init=100, random_state=42).fit_predict(Xz) for K in range(1, 11)}
    return labels_cells_dict_all_K


def get_activity_for_type(use_new_data=True, use_fixed_track=True):
    if use_new_data:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E0A1B1_251107", new_NDNF=True, use_final=True)
        first_an_idx = 14
        last_an_idx = 29

    else:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)
        first_an_idx = 17
        last_an_idx = 31
        
    cue_residual_activity_dict_NDNF_newest = {}
    for idx, session in enumerate(residual_activity_dict_NDNF_newest):
        print(f"idx {idx}")
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                cue_residual_activity_dict_NDNF_newest[idx] = residual_activity_dict_NDNF_newest[session]
        else:
            if idx > last_an_idx-1:
                cue_residual_activity_dict_NDNF_newest[idx] = residual_activity_dict_NDNF_newest[session]

    return cue_residual_activity_dict_NDNF_newest




def get_celltype_0_1_lists(labels_fixed_new,  new_fixed_data, sessions_list):

    cellType_0_indices = np.where(labels_fixed_new==0)[0]
    cellType_1_indices = np.where(labels_fixed_new==1)[0]

    cellType_0_list = []
    cellType_1_list = []

    count = 0

    cell_list = []
    for session in new_fixed_data:
        if session in sessions_list:
            for cell in new_fixed_data[session]:
                if count in cellType_0_indices:
                    print(new_fixed_data[session][cell].shape)
                    cellType_0_list.append(np.mean(new_fixed_data[session][cell], axis=1))
                else:
                    cellType_1_list.append(np.mean(new_fixed_data[session][cell], axis=1))
                count+=1

    return cellType_0_list, cellType_1_list




def prepare_celltype_lists(use_new_data: bool):
    """
    Returns a dict with:
      cellType_0_lists: [list_for_cond0, list_for_cond1, list_for_cond2, list_for_cond3]
      cellType_1_lists: same as above for type 1
      labels_conditions: list of 4 condition labels (strings)
    Each list_for_condX is a list of (n_cells_in_condX) traces, each trace length = n_pos_bins.
    """
    base_path = '/Users/michaelfinch/CA1-interneuron-GLM'

    # --- condition labels (same for new/old) ---
    labels_conditions = [
        "Cue B1: B1 seen first",
        "Cue B1: A1 seen first",
        "Fixed A1: A1 seen first",
        "Fixed A1: B1 seen first",
    ]

    if use_new_data:
        # --- load new data ---
        (GLM_params_NDNF_newest,
         activity_dict_NDNF_newest,
         double_predicted_activity_dict_NDNF_newest,
         factors_dict_NDNF_newest,
         filtered_factors_dict_NDNF_newest,
         residual_activity_dict_NDNF_newest) = load_data_regular(
            file_path=base_path,
            name="NDNF_E0A1B1_251107",
            new_NDNF=True,
            use_final=True
        )

        # session groupings (NEW)
        b1_first_b1_indices = [30, 31, 38, 40, 41, 37, 39]
        a1_when_b1_first    = [15, 16, 24, 26, 28, 23, 25]
        a1_first_a1_indices = [21, 22, 20]
        b1_when_a1_first    = [35, 36, 34]

        # --- sliceTCA labels for FIXED track ---
        with open(f'{base_path}/datasets/model_20_NDNF_resid_0x0_fixed_new_data.pkl', 'rb') as f:
            sliceTCA_model_fixed = pickle.load(f)

        labels_cells_dict_all_K_fixed = get_labels_all_different_Ks_single(
            sliceTCA_model_fixed, which_vectors=1
        )
        labels_fixed = np.asarray(labels_cells_dict_all_K_fixed[2])

        fixed_data = get_activity_for_type(use_new_data=True, use_fixed_track=True)

        # type lists for FIXED conditions
        cellType_0_list_a1_when_b1_first, \
        cellType_1_list_a1_when_b1_first = get_celltype_0_1_lists(labels_fixed, fixed_data, a1_when_b1_first)

        cellType_0_list_a1_first_a1_indices, \
        cellType_1_list_a1_first_a1_indices = get_celltype_0_1_lists(labels_fixed, fixed_data, a1_first_a1_indices)

        # --- sliceTCA labels for CUE track ---
        with open(f'{base_path}/datasets/model_20_NDNF_resid_0x0_cue_new_data.pkl', 'rb') as f:
            sliceTCA_model_cue = pickle.load(f)

        labels_cells_dict_all_K_cue = get_labels_all_different_Ks_single(
            sliceTCA_model_cue, which_vectors=1
        )
        labels_cue = np.asarray(labels_cells_dict_all_K_cue[2])

        cue_data = get_activity_for_type(use_new_data=True, use_fixed_track=False)

        # type lists for CUE conditions
        cellType_0_list_a1_b1_first_b1_indices, \
        cellType_1_list_b1_first_b1_indices = get_celltype_0_1_lists(labels_cue, cue_data, b1_first_b1_indices)

        cellType_0_list_a1_first_b1_when_a1_first, \
        cellType_1_list_b1_when_a1_first = get_celltype_0_1_lists(labels_cue, cue_data, b1_when_a1_first)

    else:
        # --- load old data ---
        (GLM_params_NDNF_newest,
         activity_dict_NDNF_newest,
         double_predicted_activity_dict_NDNF_newest,
         factors_dict_NDNF_newest,
         filtered_factors_dict_NDNF_newest,
         residual_activity_dict_NDNF_newest) = load_data_regular(
            file_path=base_path,
            name="NDNF_E1A1B",
            new_NDNF=True,
            use_final=False
        )

        # session groupings (OLD)
        b1_first_b1_indices = [31, 32, 33, 35, 36, 37, 38]
        a1_when_b1_first    = [23, 24, 25, 28, 29]
        a1_first_a1_indices = [18, 19, 20, 21, 22, 26, 27, 30]
        b1_when_a1_first    = [34]

        # --- sliceTCA labels for FIXED track ---
        save_path_fixed = f"{base_path}/Clean_notebooks_to_date/pickle_every_slice_type_NDNF.pkl"
        with open(save_path_fixed, 'rb') as f:
            models_dict = pickle.load(f)
            sliceTCA_model_fixed = models_dict[20]['model']

        labels_cells_dict_all_K_fixed = get_labels_all_different_Ks_single(
            sliceTCA_model_fixed, which_vectors=1
        )
        labels_fixed = np.asarray(labels_cells_dict_all_K_fixed[2])

        fixed_data = get_activity_for_type(use_new_data=False, use_fixed_track=True)

        cellType_0_list_a1_when_b1_first, \
        cellType_1_list_a1_when_b1_first = get_celltype_0_1_lists(labels_fixed, fixed_data, a1_when_b1_first)

        cellType_0_list_a1_first_a1_indices, \
        cellType_1_list_a1_first_a1_indices = get_celltype_0_1_lists(labels_fixed, fixed_data, a1_first_a1_indices)

        # --- sliceTCA labels for CUE track ---
        save_path_cue = f"{base_path}/Clean_notebooks_to_date/model_20_NDNF_resid_0x0_cue.pkl"
        with open(save_path_cue, 'rb') as f:
            sliceTCA_model_cue = pickle.load(f)

        labels_cells_dict_all_K_cue = get_labels_all_different_Ks_single(
            sliceTCA_model_cue, which_vectors=1
        )
        labels_cue = np.asarray(labels_cells_dict_all_K_cue[2])

        cue_data = get_activity_for_type(use_new_data=False, use_fixed_track=False)

        cellType_0_list_a1_b1_first_b1_indices, \
        cellType_1_list_b1_first_b1_indices = get_celltype_0_1_lists(labels_cue, cue_data, b1_first_b1_indices)

        cellType_0_list_a1_first_b1_when_a1_first, \
        cellType_1_list_b1_when_a1_first = get_celltype_0_1_lists(labels_cue, cue_data, b1_when_a1_first)

    # --- pack into consistent order across new/old ---
    # 0: Cue B1, B1 first
    # 1: Cue B1, A1 first
    # 2: Fixed A1, A1 first
    # 3: Fixed A1, B1 first
    cellType_0_lists = [
        cellType_0_list_a1_b1_first_b1_indices,
        cellType_0_list_a1_first_b1_when_a1_first,
        cellType_0_list_a1_first_a1_indices,
        cellType_0_list_a1_when_b1_first,
    ]
    cellType_1_lists = [
        cellType_1_list_b1_first_b1_indices,
        cellType_1_list_b1_when_a1_first,
        cellType_1_list_a1_first_a1_indices,
        cellType_1_list_a1_when_b1_first,
    ]

    return {
        "cellType_0_lists": cellType_0_lists,
        "cellType_1_lists": cellType_1_lists,
        "labels_conditions": labels_conditions,
    }


def plot_celltype_summary_new_vs_old():
    # get data for new and old
    res_new = prepare_celltype_lists(use_new_data=True)
    res_old = prepare_celltype_lists(use_new_data=False)

    labels_conditions = res_new["labels_conditions"]  # same for old

    cellType_0_lists_new = res_new["cellType_0_lists"]
    cellType_1_lists_new = res_new["cellType_1_lists"]

    cellType_0_lists_old = res_old["cellType_0_lists"]
    cellType_1_lists_old = res_old["cellType_1_lists"]

    # set colors for the 4 conditions (consistent across all subplots)
    colors = ['C0', 'C1', 'C2', 'C3']

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    # row 0: cell type 0
    # col 0: new data
    # col 1: old data

    # --- Helper: plot means for a given cell-type list set into given axis ---
    def plot_means_for_dataset(ax, cellType_lists, title_prefix):
        for cond_idx in range(4):
            traces = cellType_lists[cond_idx]
            if len(traces) == 0:
                continue
            arr = np.array(traces)
            mean_trace = arr.mean(axis=0)
            sem_trace = sem(arr, axis=0)

            x = np.arange(len(mean_trace))
            ax.plot(x, mean_trace, color=colors[cond_idx], label=labels_conditions[cond_idx])
            ax.fill_between(x, mean_trace - sem_trace, mean_trace + sem_trace,
                            color=colors[cond_idx], alpha=0.2)

        ax.set_xlabel("Position bins")
        ax.set_ylabel("Z-scored dF/F")
        ax.set_title(title_prefix)

    # --- Cell type 0 ---
    ax_00 = axs[0, 0]
    ax_01 = axs[0, 1]

    plot_means_for_dataset(ax_00, cellType_0_lists_new,  "Cell type 0 – NEW")
    plot_means_for_dataset(ax_01, cellType_0_lists_old,  "Cell type 0 – OLD")

    # --- Cell type 1 ---
    ax_10 = axs[1, 0]
    ax_11 = axs[1, 1]

    plot_means_for_dataset(ax_10, cellType_1_lists_new, "Cell type 1 – NEW")
    plot_means_for_dataset(ax_11, cellType_1_lists_old, "Cell type 1 – OLD")

    # Put a single legend in the top-right subplot
    handles, labels = ax_00.get_legend_handles_labels()
    ax_01.legend(handles, labels, fontsize=8, frameon=False, loc='upper right')

    plt.tight_layout()
    plt.show()




def get_session_contribution_counts(use_new_data: bool):
    """
    Compute # cells per session per condition for NEW or OLD data.

    Returns dict with:
      - counts: list of 4 dicts, counts[cond_idx][session_label] = n_cells
      - cond_labels_short: list of 4 short condition labels
      - cond_labels_long: list of 4 long condition labels
      - all_sessions: sorted list of session_label strings
      - session_total_n: dict session_label -> total n across all 4 conditions
      - total_n_all: sum of session_total_n.values()
      - dataset_name: "NEW" or "OLD"
    """

    base_path = '/Users/michaelfinch/CA1-interneuron-GLM'

    cond_labels_long = [
        "Cue B1 when B1 seen first",
        "Cue B1 when A1 seen first",
        "Fixed A1 when A1 seen first",
        "Fixed A1 when B1 seen first",
    ]
    cond_labels_short = [
        "Cue-B1/B1first",
        "Cue-B1/A1first",
        "Fix-A1/A1first",
        "Fix-A1/B1first",
    ]

    # -------------------------
    # 1) Load data and define session groupings + labels
    # -------------------------
    # if use_new_data:
    dataset_name = "NEW"
    (GLM_params_NDNF_newest,
    activity_dict_NDNF_newest,
    double_predicted_activity_dict_NDNF_newest,
    factors_dict_NDNF_newest,
    filtered_factors_dict_NDNF_newest,
    residual_activity_dict_NDNF_newest) = load_data_regular(
        file_path=base_path,
        name="NDNF_E0A1B1_251107",
        new_NDNF=True,
        use_final=True
    )

    # --- NEW: read sessions from .mat, build meta + groups ---
    mat_path_new = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat"
    with h5py.File(mat_path_new, "r") as f:
        raw = f["sessions"][:]
        sessions_new = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in raw]

    meta_new, name_list_new = build_new_meta_from_sessions(sessions_new)
    groups_new = classify_sessions_from_meta(meta_new)

    b1_first_b1_indices = groups_new["B1_first_or_only"]
    a1_first_a1_indices = groups_new["A1_first_or_only"]
    a1_when_b1_first    = groups_new["A1_after_B1"]
    b1_when_a1_first    = groups_new["B1_after_A1"]

    # label sessions by their animal_date string
    index_to_label = {idx: name_list_new[idx] for idx in range(len(name_list_new))}

    return groups_new, index_to_label


    # else:
    #     dataset_name = "OLD"
    #     (GLM_params_NDNF_newest,
    #      activity_dict_NDNF_newest,
    #      double_predicted_activity_dict_NDNF_newest,
    #      factors_dict_NDNF_newest,
    #      filtered_factors_dict_NDNF_newest,
    #      residual_activity_dict_NDNF_newest) = load_data_regular(
    #         file_path=base_path,
    #         name="NDNF_E1A1B",
    #         new_NDNF=True,
    #         use_final=False
    #     )

    #     # session groupings (OLD)
    #     b1_first_b1_indices = [31, 32, 33, 35, 36, 37, 38]
    #     a1_when_b1_first    = [23, 24, 25, 28, 29]
    #     a1_first_a1_indices = [18, 19, 20, 21, 22, 26, 27, 30]
    #     b1_when_a1_first    = [34]

    #     # OLD: animal IDs from folder names
    #     paths = session_paths_old()
    #     index_to_label = {}
    #     for idx, p in enumerate(paths):
    #         folder = p.split('/')[-2]         # e.g. 'CG189_250213_1B'
    #         animal_id = folder.split('_')[0]  # 'CG189'
    #         index_to_label[idx] = animal_id

    # # Bundle condition session lists in a fixed order:
    # # 0: Cue B1, B1 first
    # # 1: Cue B1, A1 first
    # # 2: Fixed A1, A1 first
    # # 3: Fixed A1, B1 first
    # cond_session_idx_lists = [
    #     b1_first_b1_indices,
    #     b1_when_a1_first,
    #     a1_first_a1_indices,
    #     a1_when_b1_first,
    # ]

    # # -------------------------
    # # 2) Get cue vs fixed activity dicts (filtered by first/last_an_idx)
    # # -------------------------
    # fixed_data = get_activity_for_type(use_new_data=use_new_data, use_fixed_track=True)
    # cue_data   = get_activity_for_type(use_new_data=use_new_data, use_fixed_track=False)

    # # -------------------------
    # # 3) Count cells per session per condition
    # # -------------------------
    # counts = [defaultdict(int) for _ in range(4)]  # counts[cond_idx][session_label] = n_cells

    # for cond_idx in range(4):
    #     if cond_idx in [0, 1]:
    #         data_dict = cue_data
    #     else:
    #         data_dict = fixed_data

    #     session_indices = cond_session_idx_lists[cond_idx]
    #     for idx in session_indices:
    #         if idx not in data_dict:
    #             continue
    #         session_label = index_to_label.get(idx, f"idx_{idx}")
    #         n_cells = len(data_dict[idx])
    #         counts[cond_idx][session_label] += n_cells

    # # -------------------------
    # # 4) Aggregate totals per session and overall
    # # -------------------------
    # all_sessions = sorted(set(sess for c in counts for sess in c.keys()))

    # # each session belongs to exactly one condition list in your setup,
    # # so summing across conditions should give total n in that dataset
    # session_total_n = {}
    # for sess in all_sessions:
    #     session_total_n[sess] = sum(counts[cond_idx].get(sess, 0) for cond_idx in range(4))

    # total_n_all = sum(session_total_n.values())

    # return {
    #     "counts": counts,
    #     "cond_labels_short": cond_labels_short,
    #     "cond_labels_long": cond_labels_long,
    #     "all_sessions": all_sessions,
    #     "session_total_n": session_total_n,
    #     "total_n_all": total_n_all,
    #     "dataset_name": dataset_name,
    # }

def build_new_meta_from_sessions(sessions):
    """
    sessions: list of strings from the NEW .mat file.
      - entries with '_' are session IDs like 'CG189_250215'
      - entries without '_' are track types: 'E0', 'A1', 'B1'

    Returns:
      meta: list of (idx, animal, date_int, cue) for each session
            where cue in {'E', 'A1', 'B1'}.
      names: list of session name strings in the same idx order.
    """
    # Separate by presence of underscore
    names  = [s for s in sessions if '_' in s]
    tracks = [s for s in sessions if '_' not in s]

    if len(names) != len(tracks):
        raise ValueError(
            f"Mismatch: {len(names)} session names with '_' "
            f"but {len(tracks)} track codes without '_'."
        )

    meta = []
    for idx, (name, track) in enumerate(zip(names, tracks)):
        try:
            animal, date_str = name.split('_')
        except ValueError:
            raise ValueError(f"Could not split session name '{name}' into animal/date")

        date = int(date_str)

        if track == 'E0':
            cue = 'E'
        elif track == 'A1':
            cue = 'A1'
        elif track == 'B1':
            cue = 'B1'
        else:
            raise ValueError(f"Unknown track type '{track}' for session '{name}'")

        meta.append((idx, animal, date, cue))

    return meta, names

import h5py
import numpy as np

def load_sessions_from_mat(mat_path):
    """
    Load 'sessions' from a MATLAB v7.3 HDF5 .mat file and return a
    flat Python list of strings.
    """
    sessions_list = []

    with h5py.File(mat_path, "r") as f:
        sess_ds = f["sessions"]  # the MATLAB cell array or char array

        if sess_ds.dtype == h5py.ref_dtype:
            # sessions is a cell array of object references
            refs = sess_ds[()].ravel()
            for ref in refs:
                dset = f[ref]
                arr = dset[()]

                if arr.dtype.kind in ("S", "U"):
                    # string-like array
                    if arr.dtype.kind == "S":
                        s = arr.tobytes().decode("utf-8")
                    else:
                        s = "".join(arr.flatten().tolist())
                else:
                    # numeric char codes
                    s = "".join(chr(int(c)) for c in arr.flatten())
                sessions_list.append(s)

        else:
            # fallback: sessions stored as 2D char/string array
            arr = sess_ds[()]
            arr = np.array(arr)

            if arr.dtype.kind in ("S", "U"):
                for row in arr.reshape(-1, arr.shape[-1]):
                    if arr.dtype.kind == "S":
                        s = row.tobytes().decode("utf-8")
                    else:
                        s = "".join(row.tolist())
                    sessions_list.append(s)
            else:
                raise ValueError(f"Unhandled sessions dtype: {sess_ds.dtype}")

    return sessions_list


def classify_sessions_from_meta(meta):
    """
    Given (idx, animal, date, cue), classify A1/B1 sessions into 4 groups:
      - B1_first_or_only
      - A1_first_or_only
      - A1_after_B1
      - B1_after_A1
    using TRUE chronological order within each animal (sorted by date).
    """
    # animal -> list of (idx, date, cue)
    animals = {}
    for idx, animal, date, cue in meta:
        animals.setdefault(animal, []).append((idx, date, cue))

    for a in animals:
        animals[a] = sorted(animals[a], key=lambda x: x[1])

    B1_first = []
    A1_first = []
    A1_after_B1 = []
    B1_after_A1 = []

    for animal, sessions in animals.items():
        saw_A1 = False
        saw_B1 = False
        for idx, date, cue in sessions:
            if cue == "B1":
                if not saw_A1:
                    B1_first.append(idx)
                else:
                    B1_after_A1.append(idx)
                saw_B1 = True
            elif cue == "A1":
                if not saw_B1:
                    A1_first.append(idx)
                else:
                    A1_after_B1.append(idx)
                saw_A1 = True
            else:
                # cue == 'E' -> exploration, skip
                pass

    # sanity check: each A1/B1 in exactly one group
    all_ab = sorted([idx for idx, animal, date, cue in meta if cue in ("A1", "B1")])
    combined = sorted(B1_first + A1_first + A1_after_B1 + B1_after_A1)
    missing = sorted(set(all_ab) - set(combined))
    duplicated = [i for i in combined if combined.count(i) > 1]

    print("NEW data — missing A1/B1 sessions from groups:", missing)
    print("NEW data — duplicated indices across groups:", duplicated)
    if not missing and not duplicated:
        print("✅ NEW: All A1/B1 sessions covered exactly once.")

    return {
        "B1_first_or_only": sorted(B1_first),
        "A1_first_or_only": sorted(A1_first),
        "A1_after_B1":      sorted(A1_after_B1),
        "B1_after_A1":      sorted(B1_after_A1),
        "ALL_A1_B1":        all_ab,
    }

def build_old_meta_from_paths(paths):
    """
    paths: list from session_paths_old()
           e.g. .../CG189/CG189_250213_1B/anal.mat

    Returns meta: list of (idx, animal, date_int, cue)
    """
    meta = []
    for idx, p in enumerate(paths):
        folder = p.split('/')[-2]         # 'CG189_250213_1B'
        parts = folder.split('_')
        animal = parts[0]                 # 'CG189'
        date = int(parts[1])              # 250213
        suffix = parts[2]                 # 'E', '1A', '1B'

        if suffix == "E":
            cue = "E"
        elif suffix == "1A":
            cue = "A1"
        elif suffix == "1B":
            cue = "B1"
        else:
            raise ValueError(f"Unknown suffix {suffix} in folder {folder}")

        meta.append((idx, animal, date, cue))

    return meta


# def get_session_contribution_counts(use_new_data: bool):
#     base_path = '/Users/michaelfinch/CA1-interneuron-GLM'

#     cond_labels_long = [
#         "Cue B1 when B1 seen first",
#         "Cue B1 when A1 seen first",
#         "Fixed A1 when A1 seen first",
#         "Fixed A1 when B1 seen first",
#     ]
#     cond_labels_short = [
#         "Cue-B1/B1first",
#         "Cue-B1/A1first",
#         "Fix-A1/A1first",
#         "Fix-A1/B1first",
#     ]

#     if use_new_data:
#         dataset_name = "NEW"
#         (GLM_params_NDNF_newest,
#          activity_dict_NDNF_newest,
#          double_predicted_activity_dict_NDNF_newest,
#          factors_dict_NDNF_newest,
#          filtered_factors_dict_NDNF_newest,
#          residual_activity_dict_NDNF_newest) = load_data_regular(
#             file_path=base_path,
#             name="NDNF_E0A1B1_251107",
#             new_NDNF=True,
#             use_final=True
#         )

#         # --- NEW: robustly load sessions from .mat and classify ---
#         mat_path_new = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat"
#         sessions_new = load_sessions_from_mat(mat_path_new)

#         meta_new, name_list_new = build_new_meta_from_sessions(sessions_new)
#         groups_new = classify_sessions_from_meta(meta_new)

#         b1_first_b1_indices = groups_new["B1_first_or_only"]
#         a1_first_a1_indices = groups_new["A1_first_or_only"]
#         a1_when_b1_first    = groups_new["A1_after_B1"]
#         b1_when_a1_first    = groups_new["B1_after_A1"]

#         # label sessions by their animal_date string
#         index_to_label = {idx: name_list_new[idx] for idx in range(len(name_list_new))}


#     else:
#         dataset_name = "OLD"
#         (GLM_params_NDNF_newest,
#          activity_dict_NDNF_newest,
#          double_predicted_activity_dict_NDNF_newest,
#          factors_dict_NDNF_newest,
#          filtered_factors_dict_NDNF_newest,
#          residual_activity_dict_NDNF_newest) = load_data_regular(
#             file_path=base_path,
#             name="NDNF_E1A1B",
#             new_NDNF=True,
#             use_final=False
#         )

#         paths_old = session_paths_old()
#         meta_old  = build_old_meta_from_paths(paths_old)
#         groups_old = classify_sessions_from_meta(meta_old)

#         b1_first_b1_indices = groups_old["B1_first_or_only"]
#         a1_first_a1_indices = groups_old["A1_first_or_only"]
#         a1_when_b1_first    = groups_old["A1_after_B1"]
#         b1_when_a1_first    = groups_old["B1_after_A1"]

#         # # label old sessions by animal ID
#         # index_to_label = {}
#         # for idx, p in enumerate(paths_old):
#         #     folder = p.split('/')[-2]
#         #     animal_id = folder.split('_')[0]
#         #     index_to_label[idx] = animal_id

#         paths = session_paths_old()
#         index_to_label = {}
#         for idx, p in enumerate(paths):
#             folder = p.split('/')[-2]           # 'CG189_250213_1B'
#             parts = folder.split('_')
#             animal_id = parts[0]                # 'CG189'
#             date_str  = parts[1]                # '250213'
#             # label as animal + date (no track, since track is encoded by condition)
#             index_to_label[idx] = f"{animal_id}_{date_str}"

#     # ---- condition → which sessions ----
#     cond_session_idx_lists = [
#         b1_first_b1_indices,  # 0: Cue B1 when B1 first
#         b1_when_a1_first,     # 1: Cue B1 when A1 first
#         a1_first_a1_indices,  # 2: Fixed A1 when A1 first
#         a1_when_b1_first,     # 3: Fixed A1 when B1 first
#     ]

#     # ---- get cue vs fixed activity dicts (your existing function) ----
#     fixed_data = get_activity_for_type(use_new_data=use_new_data, use_fixed_track=True)
#     cue_data   = get_activity_for_type(use_new_data=use_new_data, use_fixed_track=False)

#     from collections import defaultdict
#     counts = [defaultdict(int) for _ in range(4)]

#     for cond_idx in range(4):
#         data_dict = cue_data if cond_idx in [0, 1] else fixed_data
#         session_indices = cond_session_idx_lists[cond_idx]

#         for idx in session_indices:
#             if idx not in data_dict:
#                 continue
#             session_label = index_to_label.get(idx, f"idx_{idx}")
#             n_cells = len(data_dict[idx])
#             counts[cond_idx][session_label] += n_cells

#     all_sessions = sorted(set(sess for c in counts for sess in c.keys()))
#     session_total_n = {
#         sess: sum(counts[ci].get(sess, 0) for ci in range(4))
#         for sess in all_sessions
#     }
#     total_n_all = sum(session_total_n.values())

#     return {
#         "counts": counts,
#         "cond_labels_short": cond_labels_short,
#         "cond_labels_long": cond_labels_long,
#         "all_sessions": all_sessions,
#         "session_total_n": session_total_n,
#         "total_n_all": total_n_all,
#         "dataset_name": dataset_name,
#         "cond_session_idx_lists":cond_session_idx_lists
#     }



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




def get_animal_clean_dict_activity(filepath, use_final=True):

    with h5py.File(filepath, 'r') as f:
        if use_final:
            animal_group = f['animals']
        else:
            animal_group = f['animal']

        print(f"animal_group.keys() {animal_group.keys()}")

        shiftR_refs = animal_group['ShiftR'][:]

        shiftRunning_refs = animal_group['ShiftRunning'][:]

        if use_final:
            shiftL_refs = animal_group['ShiftL'][:]
        else:
            shiftL_refs = animal_group['ShiftLrate'][:]

        animal_clean_dict_activity = {}


        animal_trials_original = []
        animal_trials_clean = []

        count = 0

        animal_vel_dict = {}
        animal_lick_dict = {}

        trials_to_remove_local = []
        for animal_idx in range(len(shiftR_refs)):
            delta_f = np.array(f[shiftR_refs[animal_idx][0]])

            animal_trials_original.append(delta_f.shape[1])

            vel = f[shiftRunning_refs[animal_idx][0]]
            vel = np.array(vel).T

            lick = f[shiftL_refs[animal_idx][0]]
            lick = np.array(lick).T


            vel_clean = np.empty(vel.shape)
            lick_clean = np.empty(lick.shape)

        
            trials_to_remove_list = []

            delta_f_clean = np.empty(delta_f.shape)


            # print(f"delta_f.shape {delta_f.shape}")

            
            if animal_idx == 22:
                valid_cells = range(1, delta_f.shape[0])
            else:
                valid_cells = range(delta_f.shape[0])

            for cell in valid_cells: #range(delta_f.shape[0]):
                
            
                cell_data = delta_f[cell,:,:].T


                for trial in range(cell_data.shape[1]):

                    trial_data = cell_data[:, trial]

                    vel_data_trial = vel[:,trial]
                    lick_data_trial = lick[:,trial]

                    if np.any(np.isnan(trial_data)) or np.any(np.isnan(vel_data_trial)):

                        has_5_nans = has_run_of_n_nans(trial_data, n=5)

                        has_5_nans_vel = has_run_of_n_nans(vel_data_trial, n=5)
                        
                        if has_5_nans or has_5_nans_vel:                            
                            if count == 105:
                                trials_to_remove_local.append(trial)

                            
                            if trial not in trials_to_remove_list:
                                trials_to_remove_list.append(trial)
                        else:

                            trial_data = interp_nans_1d(trial_data)
                            delta_f_clean[cell, trial, :] = trial_data

                            trial_data_vel = interp_nans_1d(vel_data_trial)
                            vel_clean[:,trial] = trial_data_vel

                            trial_data_lick = interp_nans_1d(lick_data_trial)
                            lick_clean[:,trial] = trial_data_lick


                    else:
                        delta_f_clean[cell, trial, :] = trial_data

                        vel_clean[:,trial] = vel_data_trial 
                        lick_clean[:,trial] = lick_data_trial 


                count+=1




            trials_to_remove_array = np.array(trials_to_remove_list) 

            if len(trials_to_remove_array) !=0:

                # print(f"trials_to_remove_array {trials_to_remove_array}")
                mask = np.ones(cell_data.shape[1], dtype=bool)
                mask[trials_to_remove_array] = False
                delta_f_clean = delta_f_clean[:, mask,:]

                vel_clean = vel_clean[:, mask]
                lick_clean = lick_clean[:, mask]

            animal_trials_clean.append(delta_f_clean.shape[1])

            cell_dict = {}


            for cell in valid_cells:#range(delta_f.shape[0]):

                cell_data = delta_f_clean[cell,:,:]
                

                mean = np.mean(cell_data)
                std = np.std(cell_data)

                if std == 0 or not np.isfinite(std):
                    print(f" -> zero or bad std for this cell, skipping")
                    continue

                cell_data = (cell_data - mean) / std
                cell_dict[f"cell_{cell+1}"] = cell_data.T
                

            animal_clean_dict_activity[f"animal_{animal_idx+1}"] = cell_dict
            animal_vel_dict[f"animal_{animal_idx+1}"] = {"Velocity":vel_clean}
            animal_lick_dict[f"animal_{animal_idx+1}"] = {"Licks":lick_clean}

        return animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict


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


def plot_cps(cp_dict_NDNF, idx_session_name_dict, first_an_idx, last_an_idx, early_cp=True, use_fixed=True):

    animals = list(cp_dict_NDNF.keys())

    cp_early_list = []
    cp_late_list = []


    cp_early_by_animal = []
    cp_late_by_animal = []

    for animal in animals:
        early_vals = []
        late_vals = []
        for cell in cp_dict_NDNF[animal]:
            cp_early = cp_dict_NDNF[animal][cell][0]
            cp_late = cp_dict_NDNF[animal][cell][1]

            cp_early_list.append(cp_early)
            cp_late_list.append(cp_late)

            early_vals.append(cp_early)
            late_vals.append(cp_late)

        cp_early_by_animal.append(np.array(early_vals))
        cp_late_by_animal.append(np.array(late_vals))

    print(f"np.min(cp_early_list) {np.min(cp_early_list)}")


    import math

    n_animals = len(cp_early_by_animal)
    ncols = math.ceil(n_animals / 2)
    nrows = 2

    fig, axs = plt.subplots(
        nrows, ncols,
        figsize=(3 * ncols, 4 * nrows),
        sharey=True
    )
        

    if use_fixed:
        if early_cp:
            fig.suptitle("Fixed Track Early CP", fontsize=10)
            color="blue"
        else:
            fig.suptitle("Fixed Track Late CP", fontsize=10)
            color="orange"
    else:
        if early_cp:
            fig.suptitle("Cued Track Early CP", fontsize=10)
            color="blue"
        else:
            fig.suptitle("Cued Track Late CP", fontsize=10)
            color="orange"

    bins = np.linspace(0, 300, 31)  # 30 bins

    # Flatten axes array for easy indexing
    axs = axs.ravel()




    for idx in range(n_animals):
        if early_cp:
            cps = cp_early_by_animal[idx]
        else:
            cps = cp_late_by_animal[idx]

        if use_fixed:
            real_idx = idx+first_an_idx+1
        else:
            real_idx = idx+last_an_idx 

        session_name = idx_session_name_dict[idx]

        axs[idx].hist(cps, bins=bins, color=color, alpha=0.7, edgecolor="black")
        axs[idx].set_title(session_name, fontsize=10)
        axs[idx].set_xlabel("Trial # for Changepoint")
        if idx % ncols == 0:  # first column in each row
            axs[idx].set_ylabel("Number of Cells")

    # Hide any unused subplots if n_animals is odd
    for j in range(n_animals, len(axs)):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()


        


def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict, num_clusters=8, reassign_clusters=False, x00=True, umap=True, contiguous=True, ranks=20):

    internals_per_animal_dict_EC_animal_x00_regkmean = {}
    
    for idx, animal in enumerate(residual_activity_dict):
        internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}
        for idt, cell in enumerate(residual_activity_dict[animal]):

            cell_data = residual_activity_dict[animal][cell].T
            cell_data = ((cell_data-np.min(cell_data)) / np.max(cell_data) - np.min(cell_data))
            cell_data_3d = np.expand_dims(cell_data, axis=1)
            cell_data_3d = torch.from_numpy(cell_data_3d)
            cell_model = NDNF_fixed_model_dict[idx][idt]

            internals_dict = get_animal_model_reconstruction_dict_mod(cell_model, cell_data_3d, max_clusters=num_clusters, display=False, reassign_small_clusters=reassign_clusters, x00=x00, use_umap=umap, use_breakpoints=contiguous)

            internals_per_animal_dict_EC_animal_x00_regkmean_cell[idt] = internals_dict
        
        internals_per_animal_dict_EC_animal_x00_regkmean[idx] = internals_per_animal_dict_EC_animal_x00_regkmean_cell

    return internals_per_animal_dict_EC_animal_x00_regkmean



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





def run(use_new_data, use_fixed_track):

    # groups, data_indices = get_session_contribution_counts(use_new_data=use_new_data)

    # groups_new = {
    # "B1_first_or_only": [29, 30, 31, 32, 33, 37, 38, 39, 40, 41],
    # "A1_first_or_only": [17, 18, 19, 20, 21, 22, 27],
    # "A1_after_B1":      [15, 16, 23, 24, 25, 26, 28],
    # "B1_after_A1":      [34, 35, 36],
    # }

    # groups_new["ALL_A1_B1"] = sorted(
    #     groups_new["B1_first_or_only"]
    #     + groups_new["A1_first_or_only"]
    #     + groups_new["A1_after_B1"]
    #     + groups_new["B1_after_A1"]
    # )
    # # -> [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]

    # meta_new = [
    # (0,  "CG186", 250123, "E"),
    # (1,  "CG189", 250211, "E"),
    # (2,  "CG190", 250214, "E"),
    # (3,  "CG191", 250216, "E"),
    # (4,  "LM084", 240608, "E"),
    # (5,  "LM098", 240807, "E"),
    # (6,  "MV161", 241218, "E"),
    # (7,  "MV169", 250222, "E"),
    # (8,  "MV170", 250222, "E"),
    # (9,  "MV177", 250325, "E"),
    # (10, "MV180", 250328, "E"),
    # (11, "MV195", 250522, "E"),
    # (12, "MV196", 250523, "E"),
    # (13, "MV219", 250818, "E"),
    # (14, "MV228", 250920, "E"),
    # (15, "CG189", 250215, "A1"),
    # (16, "CG190", 250220, "A1"),
    # (17, "LM084", 240610, "A1"),
    # (18, "MV161", 241219, "A1"),
    # (19, "MV170", 250228, "A1"),
    # (20, "MV171", 250306, "A1"),
    # (21, "MV177", 250328, "A1"),
    # (22, "MV180", 250331, "A1"),
    # (23, "MV191", 250516, "A1"),
    # (24, "MV196", 250528, "A1"),
    # (25, "MV200", 250617, "A1"),
    # (26, "MV219", 250822, "A1"),
    # (27, "MV222", 250830, "A1"),
    # (28, "MV228", 250924, "A1"),
    # (29, "CG186", 250131, "B1"),
    # (30, "CG189", 250213, "B1"),
    # (31, "CG190", 250217, "B1"),
    # (32, "CG191", 250218, "B1"),
    # (33, "MV166", 250222, "B1"),
    # (34, "MV171", 250315, "B1"),
    # (35, "MV177", 250506, "B1"),
    # (36, "MV180", 250408, "B1"),
    # (37, "MV191", 250514, "B1"),
    # (38, "MV196", 250526, "B1"),
    # (39, "MV200", 250615, "B1"),
    # (40, "MV219", 250820, "B1"),
    # (41, "MV228", 250922, "B1"),]

    # groups_new = classify_sessions_from_meta(meta_new)



    # print(f"groups {groups}")
    # print(f"data_indices {data_indices}")

    # idx_session_name_dict = debug_print_new_sessions()

    # session_idx_dict = plot_session_contributions_new_vs_old()

    # session_idx_new = session_idx_dict["NEW"]
    
    # cue_list = ['Cue-B1/B1first', 'Cue-B1/A1first']

    # fixed_list = ['Fix-A1/A1first', 'Fix-A1/B1first']

    # session_idx_old = session_idx_dict["OLD"]


    filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat'

    animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict = get_animal_clean_dict_activity(filepath, use_final=True)


    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

    residual_activity_dict_NDNF_new = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)


    first_an_idx = 14
    last_an_idx = 29



    if use_fixed_track:

        clean_resid_activity_dict_NDNF_newest = {}

        clean_velocity_dict_NDNF_newest = {}

        clean_lick_dict_NDNF_newest = {}

        for idx, animal in enumerate(residual_activity_dict_NDNF_new):
            if first_an_idx < idx < last_an_idx:
                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]


        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_fixed_model_dict_clean.pkl', 'rb') as f:
            NDNF_model_dict_clean  = pickle.load(f)


        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_fixed_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)

        
        
    else:

        clean_resid_activity_dict_NDNF_newest = {}

        clean_velocity_dict_NDNF_newest = {}

        clean_lick_dict_NDNF_newest = {}

        for idx, animal in enumerate(residual_activity_dict_NDNF_new):
            if idx > last_an_idx-1:
                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]

            

        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_cued_model_dict_clean.pkl', 'rb') as f:
            NDNF_model_dict_clean = pickle.load(f)

        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_cued_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)

            
    
    residual_activity_dict_NDNF_newest = clean_resid_activity_dict_NDNF_newest

            
    


    
    # reassigned_dict_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=8, reassign_clusters=True, x00=True, umap=False, contiguous=False, ranks=20)

    contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)

    

    

    contig_dict = reshape_contig_dict(contig_dict_all_cell_tca, NDNF_model_dict_clean)


    cp_dict_NDNF = get_cp_dict(contig_dict)

    if use_fixed_track:
        idx_session_name_dict = ["CG189_250215",
"CG190_250220",
"LM084_240610",
"MV161_241219",
"MV170_250228",
"MV171_250306",
"MV177_250328",
"MV180_250331",
"MV191_250516",
"MV196_250528",
"MV200_250617",
"MV219_250822",
"MV222_250830",
"MV228_250924",]
    
    else:
        idx_session_name_dict = ["CG186_250131",
"CG189_250213",
"CG190_250217",
"CG191_250218",
"MV166_250222",
"MV171_250315",
"MV177_250506",
"MV180_250408",
"MV191_250514",
"MV196_250526",
"MV200_250615",
"MV219_250820",
"MV228_250922"]

    plot_cps(cp_dict_NDNF, idx_session_name_dict, first_an_idx, last_an_idx, early_cp=True, use_fixed=use_fixed_track)
    plot_cps(cp_dict_NDNF, idx_session_name_dict, first_an_idx, last_an_idx, early_cp=False, use_fixed=use_fixed_track)

    



    


    # base_path = '/Users/michaelfinch/CA1-interneuron-GLM'
    # mat_name = "NDNF_E0A1B1_251107"
    # (GLM_params_NDNF_newest,
    #     activity_dict_NDNF_newest,
    #     double_predicted_activity_dict_NDNF_newest,
    #     factors_dict_NDNF_newest,
    #     filtered_factors_dict_NDNF_newest,
    #     residual_activity_dict_NDNF_newest) = load_data_regular(file_path=base_path, name=mat_name, new_NDNF=True, use_final=True)


    # GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_old = load_data_regular(
    #         file_path=base_path,
    #         name="NDNF_E1A1B",
    #         new_NDNF=True,
    #         use_final=False)

    fig, axs = plt.subplots(2, 2, figsize=(10,4))  # a bit wider for legend

    for i, name in enumerate(session_idx_new):
        if name in fixed_list:
            cell_list = []
            # cell_labels = []   # animal IDs per trace

            # Collect per-cell traces and their animal labels
            for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
                if idx in session_idx_new[name]:
                    # this_label = index_to_label.get(idx, f"idx_{idx}")
                    for cell in residual_activity_dict_NDNF_newest[animal]:
                        trace = np.mean(residual_activity_dict_NDNF_newest[animal][cell], axis=1)
                        cell_list.append(trace)
                        # cell_labels.append(this_label)

            cell_array = np.array(cell_list)
            mean_cell_array = np.mean(cell_array, axis=0)
            sem_cell_array = sem(cell_array, axis=0)

            
            axs[0,0].plot(mean_cell_array, label=name)
            axs[0,0].fill_between(
                range(len(mean_cell_array)),
                mean_cell_array - sem_cell_array,
                mean_cell_array + sem_cell_array,
                alpha=0.2,
            )
            axs[0,0].set_ylabel("Z-Scored DF/F")
            axs[0,0].set_xlabel("Position Bins")
            axs[0,0].set_title("Fixed New")
            axs[0,0].set_ylim(-0.5, 1.5)
            axs[0,0].legend()


    for i, name in enumerate(session_idx_new):
        if name in cue_list:
            cell_list = []
            # cell_labels = []   # animal IDs per trace

            # Collect per-cell traces and their animal labels
            for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
                if idx in session_idx_new[name]:
                    # this_label = index_to_label.get(idx, f"idx_{idx}")
                    for cell in residual_activity_dict_NDNF_newest[animal]:
                        trace = np.mean(residual_activity_dict_NDNF_newest[animal][cell], axis=1)
                        cell_list.append(trace)
                        # cell_labels.append(this_label)

            cell_array = np.array(cell_list)
            mean_cell_array = np.mean(cell_array, axis=0)
            sem_cell_array = sem(cell_array, axis=0)

            
            axs[0,1].plot(mean_cell_array, label=name)
            axs[0,1].fill_between(
                range(len(mean_cell_array)),
                mean_cell_array - sem_cell_array,
                mean_cell_array + sem_cell_array,
                alpha=0.2,
            )
            axs[0,1].set_ylabel("Z-Scored DF/F")
            axs[0,1].set_xlabel("Position Bins")
            axs[0,1].set_title("Cue New")
            axs[0,1].set_ylim(-0.5, 1.5)
            axs[0,1].legend()


    
    for i, name in enumerate(session_idx_old):
        if name in fixed_list:
            cell_list = []
            # cell_labels = []   # animal IDs per trace

            # Collect per-cell traces and their animal labels
            for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
                if idx in session_idx_old[name]:
                    # this_label = index_to_label.get(idx, f"idx_{idx}")
                    for cell in residual_activity_dict_NDNF_newest[animal]:
                        trace = np.mean(residual_activity_dict_NDNF_newest[animal][cell], axis=1)
                        cell_list.append(trace)
                        # cell_labels.append(this_label)

            cell_array = np.array(cell_list)
            mean_cell_array = np.mean(cell_array, axis=0)
            sem_cell_array = sem(cell_array, axis=0)

            
            axs[1,0].plot(mean_cell_array, label=name)
            axs[1,0].fill_between(
                range(len(mean_cell_array)),
                mean_cell_array - sem_cell_array,
                mean_cell_array + sem_cell_array,
                alpha=0.2,
            )
            axs[1,0].set_ylabel("Z-Scored DF/F")
            axs[1,0].set_xlabel("Position Bins")
            axs[1,0].set_title("Fixed Old")
            axs[1,0].set_ylim(-0.5, 1.5)
            axs[1,0].legend()


    for i, name in enumerate(session_idx_old):
        if name in cue_list:
            cell_list = []
            # cell_labels = []   # animal IDs per trace

            # Collect per-cell traces and their animal labels
            for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
                if idx in session_idx_old[name]:
                    # this_label = index_to_label.get(idx, f"idx_{idx}")
                    for cell in residual_activity_dict_NDNF_newest[animal]:
                        trace = np.mean(residual_activity_dict_NDNF_newest[animal][cell], axis=1)
                        cell_list.append(trace)
                        # cell_labels.append(this_label)

            cell_array = np.array(cell_list)
            mean_cell_array = np.mean(cell_array, axis=0)
            sem_cell_array = sem(cell_array, axis=0)

            
            axs[1,1].plot(mean_cell_array, label=name)
            axs[1,1].fill_between(
                range(len(mean_cell_array)),
                mean_cell_array - sem_cell_array,
                mean_cell_array + sem_cell_array,
                alpha=0.2,
            )
            axs[1,1].set_ylabel("Z-Scored DF/F")
            axs[1,1].set_xlabel("Position Bins")
            axs[1,1].set_title("Cue Old")
            axs[1,1].set_ylim(-0.5, 1.5)
            axs[1,1].legend()

    plt.tight_layout()
    plt.show()




    # with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_new.pkl', 'rb') as f:
    #     fixed_new_labels = pickle.load(f)
    #     print(len(fixed_new_labels))

    # with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_new.pkl', 'rb') as f:
    #     cue_new_labels = pickle.load(f)
    #     print(len(cue_new_labels))

    # with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_old.pkl', 'rb') as f:
    #     fixed_old_labels = pickle.load(f)
    #     print(len(fixed_old_labels))

    # with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_old.pkl', 'rb') as f:
    #     cue_old_labels = pickle.load(f)
    #     print(len(cue_old_labels))


    # GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E0A1B1_251107", new_NDNF=True, use_final=True)
    first_an_idx_new = 14
    last_an_idx_new = 29

    # GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_old = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)
    first_an_idx_old = 17
    last_an_idx_old = 31


    cue_residual_activity_dict_NDNF_newest = {}
    fixed_residual_activity_dict_NDNF_newest = {}

    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if first_an_idx_new < idx < last_an_idx_new:
            fixed_residual_activity_dict_NDNF_newest[idx] = residual_activity_dict_NDNF_newest[animal]
        if idx > last_an_idx_new-1:
            cue_residual_activity_dict_NDNF_newest[idx] = residual_activity_dict_NDNF_newest[animal]


    cue_residual_activity_dict_NDNF_old = {}
    fixed_residual_activity_dict_NDNF_old = {}

    for idx, animal in enumerate(residual_activity_dict_NDNF_old):
        if first_an_idx_old < idx < last_an_idx_old:
            fixed_residual_activity_dict_NDNF_old[idx] = residual_activity_dict_NDNF_old[animal]
        if idx > last_an_idx_old-1:
            cue_residual_activity_dict_NDNF_old[idx] = residual_activity_dict_NDNF_old[animal]



    def get_session_type_count_per_cell_type(residual_activity_dict, labels, session_type):

        type0_idx = np.where(labels == 0)[0]
        type1_idx = np.where(labels == 1)[0]

        cell_type_0_dict = {"same_type_first":0, "other_type_first":0, "overall_len":len(type0_idx)}
        cell_type_1_dict = {"same_type_first":0, "other_type_first":0, "overall_len":len(type1_idx)}

        global_cell_idx = 0  

        cell_type0_same_data_list = []
        cell_type0_other_data_list = []

        cell_type1_same_data_list = []
        cell_type1_other_data_list = []

        for session in residual_activity_dict:
            session_is_in_type = session in session_type

            for cell in residual_activity_dict[session]:

                if global_cell_idx in type0_idx:
                    if session_is_in_type:
                        cell_type_0_dict["same_type_first"] += 1
                        cell_type0_same_data_list.append(np.mean(residual_activity_dict[session][cell], axis=1))
                    else:
                        cell_type_0_dict["other_type_first"] += 1
                        cell_type0_other_data_list.append(np.mean(residual_activity_dict[session][cell], axis=1))

                else:  # cell type 1
                    if session_is_in_type:
                        cell_type_1_dict["same_type_first"] += 1
                        cell_type1_same_data_list.append(np.mean(residual_activity_dict[session][cell], axis=1))

                    else:
                        cell_type_1_dict["other_type_first"] += 1
                        cell_type1_other_data_list.append(np.mean(residual_activity_dict[session][cell], axis=1))

                global_cell_idx += 1  # advance only here


        data_dict = {"cell_type0_same_data_list":cell_type0_same_data_list,
                     "cell_type0_other_data_list":cell_type0_other_data_list,
                     "cell_type1_same_data_list":cell_type1_same_data_list,
                     "cell_type1_other_data_list":cell_type1_other_data_list}

        return cell_type_0_dict, cell_type_1_dict, data_dict

    
    cell_type_0_dict_fixed_new, cell_type_1_dict_fixed_new, data_dict_fixed_new = \
        get_session_type_count_per_cell_type(
            fixed_residual_activity_dict_NDNF_newest,
            fixed_new_labels,
            session_idx_new['Fix-A1/A1first']
        )

    n0_same_new  = cell_type_0_dict_fixed_new["same_type_first"]
    n0_other_new = cell_type_0_dict_fixed_new["other_type_first"]
    n1_same_new  = cell_type_1_dict_fixed_new["same_type_first"]
    n1_other_new = cell_type_1_dict_fixed_new["other_type_first"]

    type0_same_fraction_new  = n0_same_new  / cell_type_0_dict_fixed_new["overall_len"]
    type0_other_fraction_new = n0_other_new / cell_type_0_dict_fixed_new["overall_len"]
    type1_same_fraction_new  = n1_same_new  / cell_type_1_dict_fixed_new["overall_len"]
    type1_other_fraction_new = n1_other_new / cell_type_1_dict_fixed_new["overall_len"]

    data_list_new0 = [type0_same_fraction_new, type0_other_fraction_new]
    data_list_new1 = [type1_same_fraction_new, type1_other_fraction_new]
    n_list_new0    = [n0_same_new, n0_other_new]
    n_list_new1    = [n1_same_new, n1_other_new]


    # --- OLD: fixed (A1-first) ---
    cell_type_0_dict_fixed_old, cell_type_1_dict_fixed_old, data_dict_fixed_old = \
        get_session_type_count_per_cell_type(
            fixed_residual_activity_dict_NDNF_old,
            fixed_old_labels,
            session_idx_old['Fix-A1/A1first']
        )

    n0_same_old  = cell_type_0_dict_fixed_old["same_type_first"]
    n0_other_old = cell_type_0_dict_fixed_old["other_type_first"]
    n1_same_old  = cell_type_1_dict_fixed_old["same_type_first"]
    n1_other_old = cell_type_1_dict_fixed_old["other_type_first"]

    type0_same_fraction_old  = n0_same_old  / cell_type_0_dict_fixed_old["overall_len"]
    type0_other_fraction_old = n0_other_old / cell_type_0_dict_fixed_old["overall_len"]
    type1_same_fraction_old  = n1_same_old  / cell_type_1_dict_fixed_old["overall_len"]
    type1_other_fraction_old = n1_other_old / cell_type_1_dict_fixed_old["overall_len"]

    data_list_old0 = [type0_same_fraction_old, type0_other_fraction_old]
    data_list_old1 = [type1_same_fraction_old, type1_other_fraction_old]
    n_list_old0    = [n0_same_old, n0_other_old]
    n_list_old1    = [n1_same_old, n1_other_old]


    # --- BAR PLOTTING --- #
    labels_list = ["same type first", "other type first"]
    x = np.arange(len(labels_list))  # [0, 1]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Helper to draw bars + n counts
    def plot_with_n(ax, data, n_list, title):
        bars = ax.bar(x, data, color=["#4C72B0", "#55A868"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction")
        ax.set_title(title)

        # print n on top of each bar
        for xi, bar, n in zip(x, bars, n_list):
            height = bar.get_height()
            ax.text(
                xi,
                height + 0.03,     # little offset above bar
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=9
            )


    # --- OLD fixed — cell type 0 ---
    plot_with_n(axs[0,0], data_list_old0, n_list_old0, "OLD – Fixed – Cell Type 0")

    # --- OLD fixed — cell type 1 ---
    plot_with_n(axs[0,1], data_list_old1, n_list_old1, "OLD – Fixed – Cell Type 1")

    # --- NEW fixed — cell type 0 ---
    plot_with_n(axs[1,0], data_list_new0, n_list_new0, "NEW – Fixed – Cell Type 0")

    # --- NEW fixed — cell type 1 ---
    plot_with_n(axs[1,1], data_list_new1, n_list_new1, "NEW – Fixed – Cell Type 1")

    plt.tight_layout()
    plt.show()

    mat_path_new = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat"

    with h5py.File(mat_path_new, "r") as f:
        sess_ds = f["sessions"][:]    # e.g. shape (84, nChars)

    # Convert each row into a proper UTF-8 string
    sessions_new = []
    for row in sess_ds:
        try:
            # row is something like array([67,71,49,56,54, ...])
            s = row.astype("uint8").tobytes().decode("utf-8").strip()
        except Exception:
            try:
                s = row.tobytes().decode("utf-8").strip()
            except Exception:
                s = str(row).strip()
        sessions_new.append(s)

    # Now extract animal IDs (e.g. "CG186" from "CG186_250131")
    idx_to_animal_new = {}
    for idx, name in enumerate(sessions_new):
        if "_" in name:
            idx_to_animal_new[idx] = name.split("_")[0]



    # OLD dataset: from paths
    paths_old = session_paths_old()
    idx_to_animal_old = {}
    for idx, p in enumerate(paths_old):
        folder = p.split("/")[-2]        # e.g. "CG189_250215_1A"
        animal = folder.split("_")[0]    # "CG189"
        idx_to_animal_old[idx] = animal

    # --------------------------------------------------
    # 2) Build table rows for FIXED track data only
    #     (keys are session indices like 15..30)
    # --------------------------------------------------

    rows = []

    all_session_idxs = sorted(
        set(list(fixed_residual_activity_dict_NDNF_newest.keys()) +
            list(fixed_residual_activity_dict_NDNF_old.keys()))
    )

    for sess_idx in all_session_idxs:

        # figure out a nice animal label
        name_new = idx_to_animal_new.get(sess_idx)
        name_old = idx_to_animal_old.get(sess_idx)

        if name_new is not None:
            animal_label = name_new
        elif name_old is not None:
            animal_label = name_old
        else:
            animal_label = f"idx_{sess_idx}"

        # --- NEW data presence ---
        if sess_idx in fixed_residual_activity_dict_NDNF_newest:
            cells_new = fixed_residual_activity_dict_NDNF_newest[sess_idx]
            n_cells_new = len(cells_new)

            first_cell_key_new = next(iter(cells_new))
            arr_new = cells_new[first_cell_key_new]       # shape: (50, n_trials)
            n_trials_new = arr_new.shape[1]

            present_new = "✔"
        else:
            n_cells_new = ""
            n_trials_new = ""
            present_new = "✗"

        # --- OLD data presence ---
        if sess_idx in fixed_residual_activity_dict_NDNF_old:
            cells_old = fixed_residual_activity_dict_NDNF_old[sess_idx]
            n_cells_old = len(cells_old)

            first_cell_key_old = next(iter(cells_old))
            arr_old = cells_old[first_cell_key_old]       # shape: (50, n_trials)
            n_trials_old = arr_old.shape[1]

            present_old = "✔"
        else:
            n_cells_old = ""
            n_trials_old = ""
            present_old = "✗"

        rows.append({
            "Animal": animal_label,
            "# Cells NEW": n_cells_new,
            "# Cells OLD": n_cells_old,
            "# Trials NEW": n_trials_new,
            "# Trials OLD": n_trials_old,
            "Present in NEW?": present_new,
            "Present in OLD?": present_old,
        })

    table_df = pd.DataFrame(rows)
    # optional: sort by animal name
    table_df = table_df.sort_values("Animal").reset_index(drop=True)

    print(table_df)

    fig_table, ax_table = plt.subplots(figsize=(10, len(table_df)*0.4))
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    plt.show()

    print(f"idx_to_animal_new {idx_to_animal_new}")

    print(f"idx_to_animal_old {idx_to_animal_old}")


    def get_session_id_to_name_dicts():
        """
        Returns:
        idx_to_name_new: {session_idx (NEW) -> 'Animal_Date_Track'}
        idx_to_name_old: {session_idx (OLD) -> 'Animal_Date_TrackCode'}
        NEW uses the .mat "sessions" + meta.
        OLD uses session_paths_old() folder names.
        """
        base_path = '/Users/michaelfinch/CA1-interneuron-GLM'

        # ---------- NEW ----------
        # (This load is just to keep things consistent with how you're using the dataset elsewhere)
        _ = load_data_regular(
            file_path=base_path,
            name="NDNF_E0A1B1_251107",
            new_NDNF=True,
            use_final=True,
        )

        mat_path_new = "/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat"
        sessions_new = load_sessions_from_mat(mat_path_new)
        meta_new, name_list_new = build_new_meta_from_sessions(sessions_new)
        # meta_new: list of (idx, animal, date, cue) where cue ∈ {'E','A1','B1'}
        # name_list_new[idx] is e.g. 'CG189_250215'

        idx_to_name_new = {}
        for idx, animal, date, cue in meta_new:
            if cue == "E":
                continue  # skip exploration
            base_name = name_list_new[idx]
            full_name = f"{base_name}_{cue}"
            idx_to_name_new[idx] = full_name

        # ---------- OLD ----------
        _ = load_data_regular(
            file_path=base_path,
            name="NDNF_E1A1B",
            new_NDNF=True,
            use_final=False,
        )

        paths_old = session_paths_old()
        # folders like 'CG189_250213_1B'
        idx_to_name_old = {}
        for idx, p in enumerate(paths_old):
            folder = p.split('/')[-2]               # 'CG189_250213_1B'
            idx_to_name_old[idx] = folder

        return idx_to_name_new, idx_to_name_old
    
    idx_to_name_new, idx_to_name_old = get_session_id_to_name_dicts()

    save_list = [idx_to_name_new, idx_to_name_old]

    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/ndnf_cell_types/saved_stuff.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(save_list, f)

    print("NEW mapping:", idx_to_name_new)
    print("OLD mapping:", idx_to_name_old)

    # Example lookup:
    # session id 7 in NEW:
    print("NEW 7 ->", idx_to_name_new.get(7))
    # session id 12 in OLD:
    print("OLD 12 ->", idx_to_name_old.get(12))





@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
@click.option('--use_new_data/--use_old_data', default=True, help="Use the Final NDNF data")

def cli(use_new_data, use_fixed_track):
    run(use_new_data, use_fixed_track)

if __name__ == "__main__":
    cli()

