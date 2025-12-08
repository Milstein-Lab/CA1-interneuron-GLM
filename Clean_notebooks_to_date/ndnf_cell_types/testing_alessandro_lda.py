
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca

from scipy.spatial.distance import mahalanobis

from sklearn.linear_model import LinearRegression

# import utils as ut
# import plot as pt
plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

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

    ax.set_xlabel("LDA1 (pre vs post reward)", fontsize=title_fs-1)
    ax.set_ylabel("LDA2 (early vs late trials)", fontsize=title_fs-1)
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

    ax.set_title(f"{title} \n Mahal. Dist. Early vs Late Pre-Reward={d_pre_0:.2f} \n Mahal. Dist. Early vs Late Post-Reward={d_post_0:.2f}", fontsize=title_fs)


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

    #     MSE_per_animal_list = []

    #     if use_animal:

    #         for animal_idx, animal_model in enumerate(model_list_per_animal):
    #             animal_tensor = tensor_per_animal_list[animal_idx]  # (trials, cells, pos)
    #             reconstruction_full_animal = animal_model.construct().numpy(force=True)

    #             if reconstruction_full_animal.shape != animal_tensor.shape:
    #                 raise ValueError(
    #                     f"Shape mismatch for k={k}, animal {animal_idx}: "
    #                     f"tensor {animal_tensor.shape}, recon {reconstruction_full_animal.shape}"
    #                 )
                
    #             print(f"reconstruction_full_animal.shape {reconstruction_full_animal.shape}")

    #             MSE = np.mean((animal_tensor - reconstruction_full_animal) ** 2)
    #             MSE_per_animal_list.append(MSE)

    #     else:
    #         # print("made it hereeeeeeeeee")
    #         for animal_idx, animal_model_list in enumerate(model_list_per_animal):
    #             MSE_per_cells = []
    #             for cell in range(len(animal_model_list)):
    #                 cell_model = animal_model_list[cell]
    #                 reconstruction_full_cell = cell_model.construct().numpy(force=True)
                    
    #                 cell_tensor = tensor_per_animal_list[animal_idx][cell]

    #                 print(f"animal_idx {animal_idx} cell {cell}")

    #                 if reconstruction_full_cell.shape != cell_tensor.shape:
    #                     raise ValueError(
    #                         f"Shape mismatch for k={k}, animal {animal_idx}: "
    #                         f"tensor {cell_tensor.shape}, recon {reconstruction_full_cell.shape}"
    #                     )
                    
    #                 # print(f"reconstruction_full_cell.shape {reconstruction_full_cell.shape}")
                    
    #                 MSE = np.mean((cell_tensor - reconstruction_full_cell) ** 2)
    #                 MSE_per_cells.append(MSE)

    #             MSE_per_animal_list.append(np.mean(MSE_per_cells))


    #     MSE_per_animal_array = np.array(MSE_per_animal_list)
    #     MSE_an_av_per_latent.append(MSE_per_animal_array.mean())
    #     MSE_an_sem_per_latent.append(sem(MSE_per_animal_array))

    # return MSE_an_av_per_latent, MSE_an_sem_per_latent, k_values

def run():

    models_dir = "./per_k_pickles_ndnf_per_cell"

    latent_model20_per_cell_list = get_mse_from_model_filepath(models_dir)

    just_fixed_animals_list = []

    for i in range(len(latent_model20_per_cell_list)):
         if 14 < i < 29:
             just_fixed_animals_list.append(latent_model20_per_cell_list[i])
             

    example_model = just_fixed_animals_list[0][0]

    X_all = example_model.vectors[0][0].detach().numpy().T


    reg_motion = LinearRegression()

    # motion is shape (n_trials,)
    reg_motion.fit(X_all, motion)
    X_lda2 = X_all @ reg_motion.coef_

    
    plt.plot(X_all[0,:])
    plt.show()

    

if __name__ == "__main__":
    run()