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

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *


from modelling_to_date_utils import *
from SliceTCA_example import *

from scipy.stats import ttest_rel

import click
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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

def get_max_proportion(early_labels, use_max=True):
    unique_early_labels = np.unique(early_labels)
    if use_max:
        og_proportion=0
    else:
        og_proportion=1000
    good_dict=None
    for unique_label in unique_early_labels:
        amount = len(np.where(early_labels==unique_label)[0])
        len_early_labels = len(early_labels)
        proportion_early = amount / len_early_labels

        if use_max:
            if proportion_early > og_proportion:
                good_dict= {"unique_label":unique_label,
                        "fraction":proportion_early}
                og_proportion=proportion_early
        else:
            if proportion_early < og_proportion:
                good_dict= {"unique_label":unique_label,
                        "fraction":proportion_early}
                og_proportion=proportion_early
    return good_dict

# def get_lists_out_of_dicts(fixed_TT_data, fixed_activity_dict_NDNF_newest, cp_dict_NDNF):
#     TT_list = []
#     for animal in fixed_TT_data:
#         for cell in fixed_TT_data[animal]:
#             TT_list.append(fixed_TT_data[animal][cell][1][f"cell_{cell}"])

#     print(len(TT_list)) 

#     NDNF_activity_list = []
#     for animal in fixed_activity_dict_NDNF_newest:
#         for cell in fixed_activity_dict_NDNF_newest[animal]:
#             NDNF_activity_list.append(fixed_activity_dict_NDNF_newest[animal][cell])

#     print(len(NDNF_activity_list)) 

#     cp_list_NDNF = []

#     for animal in cp_dict_NDNF:
#         for cell in cp_dict_NDNF[animal]:
#             cp_list_NDNF.append(cp_dict_NDNF[animal][cell])

#     print(len(cp_list_NDNF)) 

    # return TT_list, NDNF_activity_list, cp_list_NDNF

def find_elbow_point(y_vals, min_index=2):

    x = np.arange(len(y_vals))
    y = y_vals

    # First and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Compute distances to the line
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_p1 = np.vstack((x - p1[0], y - p1[1])).T
    scalar_proj = np.dot(vec_from_p1, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_from_p1 - proj, axis=1)

    # Force elbow to be at least min_index (default = 2)
    elbow_idx = np.argmax(dist_to_line[min_index:]) + min_index
    return int(elbow_idx)

# def get_most_expressed_cluster(TT_list, activity_list, cp_list_NDNF, early_late_none="early", to_include=None, most_expressed=True):
    
#     if np.any(to_include) == None:
#         indices_to_include = np.arange(115)
#     else:
#         indices_to_include = to_include

#     most_expressed_label_dict = {}

#     elbow_kmeans_array = np.empty(len(indices_to_include))

#     for j, cell in enumerate(indices_to_include):
              
#         activity_array = activity_list[cell]

#         labels_example = TT_list[cell]["labels_dict"]["clusters_chosen_3"]

#         MSE_dict = TT_list[cell]["MSE_dict"]

#         MSE_array = np.empty(len(MSE_dict))

#         for id, clusters_chosen in enumerate(MSE_dict):
#             MSE = MSE_dict[clusters_chosen]
#             MSE_array[id] = MSE
        
#         elbow_kmeans = find_elbow_point(MSE_array)

#         elbow_kmeans_array[j] = elbow_kmeans
        
#         if early_late_none=='early':
#             cp_early = cp_list_NDNF[cell][0]
#             labels = labels_example[:cp_early]

#         elif early_late_none=='late':
#             cp_late = cp_list_NDNF[cell][1]
#             labels = labels_example[cp_late:]

#         elif early_late_none=='none':
#             labels = labels_example
#         else:
#             raise ValueError("Invalid learning chunk")


#         good_dict = get_max_proportion(labels, use_max=most_expressed)

#         correct_indices = np.where(labels==good_dict["unique_label"])[0]
#         activity_array_sliced = activity_array[:,correct_indices]


#         most_expressed_label_dict[cell] = {"label":good_dict["unique_label"],
#                                                         "cluster_activity": activity_array_sliced, "fraction":good_dict["fraction"]}

#     return most_expressed_label_dict, elbow_kmeans_array



def eval_proportion(most_expressed_label_dict_animal_early, most_expressed_label_dict_animal_late, group=None, ax=None):
    early_vals = []
    late_vals  = []

    for cell in most_expressed_label_dict_animal_early:
        early_vals.append(most_expressed_label_dict_animal_early[cell]["fraction"])
        late_vals.append(most_expressed_label_dict_animal_late[cell]["fraction"])

    early_vals = np.array(early_vals)
    late_vals  = np.array(late_vals)

    # paired t test
    t,p = ttest_rel(early_vals, late_vals)

    # plot all individual paired points
    for e,l in zip(early_vals, late_vals):
        ax.plot([0,1], [e,l], color='gray', alpha=0.4)
        ax.plot(0, e, 'o', color='black')
        ax.plot(1, l, 'o', color='black')

    # plot means
    ax.plot([0,1], [early_vals.mean(), late_vals.mean()],
            color='red', marker='o', linewidth=2)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['Early','Late'])
    ax.set_ylabel("Fraction of Trials in Learning Block")
    ax.set_title(f"{group} Paired t-test (p = {p:.3f})")


def get_activity_cut_learn(fixed_residual_activity_dict_NDNF_newest, cp_dict_NDNF):
    activity_list_early = []
    activity_list_late = []

    for idx, animal in enumerate(fixed_residual_activity_dict_NDNF_newest):
        for idt, cell in enumerate(fixed_residual_activity_dict_NDNF_newest[animal]):
            data = fixed_residual_activity_dict_NDNF_newest[animal][cell]
            cp_early = cp_dict_NDNF[idx][idt][0]
            cp_late = cp_dict_NDNF[idx][idt][0]

            early_data = data[:,:cp_early]
            late_data = data[:,-cp_late:]

            mean_early_data = np.mean(early_data, axis=1)
            activity_list_early.append(mean_early_data)
            mean_late_data = np.mean(late_data, axis=1)
            activity_list_late.append(mean_late_data)




    activity_early_array = np.array(activity_list_early)
    activity_array_late = np.array(activity_list_late)

    return activity_early_array, activity_array_late



# def plot_no_learn_cell_types(most_expressed_label_dict_animal_cluster_0_all, most_expressed_label_dict_animal_cluster_1_all, elbow_kmeans_array, group=None, most_expressed=None):

#     fig, axs = plt.subplots(4,2, figsize=(8,12))

#     if most_expressed:
#         fig.suptitle("Most Expressed Cluster Across Trials")
#     else:
#         fig.suptitle("Least Expressed Cluster Across Trials")

#     mean_0_list = []
#     mean_1_list = []

#     mean_0_fractions_list = []
#     mean_1_fractions_list = []

#     for cell in most_expressed_label_dict_animal_cluster_0_all:
#         array0 = most_expressed_label_dict_animal_cluster_0_all[cell]["cluster_activity"]
#         mean0 = np.mean(array0, axis=1)
#         mean_0_list.append(mean0)
#         axs[0,0].plot(mean0)
#         mean_0_fractions_list.append(most_expressed_label_dict_animal_cluster_0_all[cell]["fraction"])
#     for cell in most_expressed_label_dict_animal_cluster_1_all:
#         array1 = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
#         mean1 = np.mean(array1, axis=1)
#         mean_1_list.append(mean1)
#         axs[0,1].plot(mean1)
#         mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])



#     axs[0,0].set_title("Group0")
#     axs[0,1].set_title("Group1")

#     mean_0_array = np.array(mean_0_list)
#     mean_1_array = np.array(mean_1_list)

#     mean_mean_0_array = np.mean(mean_0_array, axis=0)
#     mean_mean_1_array = np.mean(mean_1_array, axis=0)

#     sem_0_array = sem(mean_0_array, axis=0)
#     sem_1_array = sem(mean_1_array, axis=0)

#     axs[1,0].plot(mean_mean_0_array, label='Group0')
#     axs[1,1].plot(mean_mean_1_array, label='Group1')
#     axs[1,0].fill_between(range(len(mean_mean_0_array)), mean_mean_0_array-sem_0_array, mean_mean_0_array+sem_0_array, alpha=0.2)
#     axs[1,1].fill_between(range(len(mean_mean_1_array)), mean_mean_1_array-sem_1_array, mean_mean_1_array+sem_1_array, alpha=0.2)

#     axs[2,0].hist(mean_0_fractions_list)
#     axs[2,0].set_xlabel("Fraction of Total Trials")
#     axs[2,0].set_ylabel("Cells")

#     axs[2,1].hist(mean_1_fractions_list)
#     axs[2,1].set_xlabel("Fraction of Total Trials")
#     axs[2,1].set_ylabel("Cells")

#     axs[3,0].plot(mean_mean_0_array, label='Early')
#     axs[3,0].plot(mean_mean_1_array, label='Late')
#     axs[3,0].fill_between(range(len(mean_mean_0_array)), mean_mean_0_array-sem_0_array, mean_mean_0_array+sem_0_array, alpha=0.2)
#     axs[3,0].fill_between(range(len(mean_mean_1_array)), mean_mean_1_array-sem_1_array, mean_mean_1_array+sem_1_array, alpha=0.2)
#     axs[3,0].set_title(group)
#     axs[3,0].legend()

#     axs[3,1].hist(elbow_kmeans_array, bins=[1.5,2.5,3.5,4.5,5.5])
#     axs[3,1].set_xticks([2,3,4,5])
#     axs[3,1].set_xlim(1.5,5.5)

#     axs[3,1].set_title("Elbow Num Clusters Distribution")
#     axs[3,1].set_xlabel("Num Clusters")
#     axs[3,1].set_ylabel("Num Cells")
        
#     plt.tight_layout()
#     plt.show()

# def plot_clustered_data_learn(means_dict_cluster_0x0_raw, activity_early_array, activity_array_late, K=2, title=""):
#     data_good = means_dict_cluster_0x0_raw[K]["labels_loc_dict"]

#     fig, axs = plt.subplots(1,len(data_good), figsize=(4*len(data_good), 4))
#     fig.suptitle(title)

#     for i in data_good:
#         labels = data_good[i]
#         n=len(labels)
#         sliced_early = activity_early_array[labels,:]
#         mean_sliced_early = np.mean(sliced_early, axis=0)
#         sem_sliced_early = sem(sliced_early, axis=0)

#         # sliced_data_early_dict[i] = sliced_early
#         sliced_late = activity_array_late[labels,:]
#         mean_sliced_late = np.mean(sliced_late, axis=0)
#         sem_sliced_late = sem(sliced_late, axis=0)

#         # sliced_data_late_dict[i] = sliced_late
#         axs[i].plot(mean_sliced_early, label='Early')
#         axs[i].fill_between(range(len(mean_sliced_early)), mean_sliced_early-sem_sliced_early, mean_sliced_early+sem_sliced_early, alpha=0.2)

#         axs[i].plot(mean_sliced_late, label="Late")
#         axs[i].fill_between(range(len(mean_sliced_late)), mean_sliced_late-sem_sliced_late, mean_sliced_late+sem_sliced_late, alpha=0.2)
#         axs[i].set_title(f"Cluster {i} n={n}")
#         axs[i].legend()

#     plt.tight_layout()
#     plt.show()


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

    # Standardize then KMeans over K=1..10
    Xz = StandardScaler().fit_transform(X)
    labels_cells_dict_all_K = {K: KMeans(n_clusters=K, n_init=100, random_state=42).fit_predict(Xz) for K in range(1, 11)}
    return labels_cells_dict_all_K



# def plot_early_late_activity(
#     most_expressed_label_dict_animal_early,
#     most_expressed_label_dict_animal_late,
#     elbow_kmeans_array,
#     group=None,
#     most_expressed=True
# ):
#     # ---------- compute means/sems once ----------
#     mean_early_list = []
#     mean_late_list  = []

#     for cell in most_expressed_label_dict_animal_early:
#         early_array = most_expressed_label_dict_animal_early[cell]["cluster_activity"]  # (clusters, time) or similar
#         late_array  = most_expressed_label_dict_animal_late[cell]["cluster_activity"]

#         mean_early = np.mean(early_array, axis=1)  # per-cluster mean trace (adjust if your axis differs)
#         mean_late  = np.mean(late_array,  axis=1)

#         mean_early_list.append(mean_early)
#         mean_late_list.append(mean_late)

#     mean_early_array = np.asarray(mean_early_list)
#     mean_late_array  = np.asarray(mean_late_list)

#     mean_mean_early_array = np.mean(mean_early_array, axis=0)
#     mean_mean_late_array  = np.mean(mean_late_array,  axis=0)

#     sem_early_array = sem(mean_early_array, axis=0, nan_policy='omit')
#     sem_late_array  = sem(mean_late_array,  axis=0, nan_policy='omit')

#     # ---------- figure with nested grids ----------
#     fig = plt.figure(figsize=(18, 5), constrained_layout=True)
#     title_txt = "Most Highly Expressed Cluster Per Cell" if most_expressed else "Most Highly Expressed Cluster Per Cell"
#     if group:
#         fig.suptitle(f"{title_txt} — {group}", y=1.05)
#     else:
#         fig.suptitle(title_txt, y=1.05)

#     # Outer grid: 2 rows × 4 cols; first column is wide to host a 2×2 sub-grid
#     outer = fig.add_gridspec(2, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.1, hspace=0.01)

#     # ----- Left block: 2×2 subgrid (your original first figure) -----
#     left = outer[:, 0].subgridspec(2, 2, wspace=0.01, hspace=0.1, height_ratios=(1.0,1.0), width_ratios=(0.1,0.1))
#     ax00 = fig.add_subplot(left[0, 0])
#     ax01 = fig.add_subplot(left[0, 1])
#     ax10 = fig.add_subplot(left[1, 0])
#     ax11 = fig.add_subplot(left[1, 1])

#     # Top row: per-cell traces (Early / Late)
#     for cell in most_expressed_label_dict_animal_early:
#         early_array = most_expressed_label_dict_animal_early[cell]["cluster_activity"]
#         late_array  = most_expressed_label_dict_animal_late[cell]["cluster_activity"]
#         ax00.plot(np.mean(early_array, axis=1))
#         ax01.plot(np.mean(late_array,  axis=1))

#     ax00.set_title("Early")
#     ax01.set_title("Late")

#     # Bottom row: mean ± SEM
#     tE = np.arange(len(mean_mean_early_array))
#     tL = np.arange(len(mean_mean_late_array))
#     ax10.plot(mean_mean_early_array, label='Early')
#     ax11.plot(mean_mean_late_array,  label='Late')

#     ax10.fill_between(tE, mean_mean_early_array - sem_early_array,
#                            mean_mean_early_array + sem_early_array, alpha=0.2)
#     ax11.fill_between(tL, mean_mean_late_array  - sem_late_array,
#                            mean_mean_late_array  + sem_late_array,  alpha=0.2)

#     ax10.legend(loc='upper right', frameon=False)
#     ax11.legend(loc='upper right', frameon=False)

#     # ----- Right three single panels (formerly 1×3) -----
#     axA = fig.add_subplot(outer[:, 1])  # summary line plot
#     axB = fig.add_subplot(outer[:, 2])  # proportions (your eval_proportion)
#     axC = fig.add_subplot(outer[:, 3])  # histogram

#     # Summary mean ± SEM overlay
#     axA.plot(mean_mean_early_array, label='Early')
#     axA.plot(mean_mean_late_array,  label='Late')
#     axA.fill_between(tE, mean_mean_early_array - sem_early_array,
#                           mean_mean_early_array + sem_early_array, alpha=0.2)
#     axA.fill_between(tL, mean_mean_late_array  - sem_late_array,
#                           mean_mean_late_array  + sem_late_array,  alpha=0.2)
#     if group:
#         axA.set_title(group)
#     axA.legend(frameon=False)

#     # Proportions panel (reuse your existing helper)
#     # Make sure eval_proportion accepts an axes handle.
#     eval_proportion(most_expressed_label_dict_animal_early,
#                     most_expressed_label_dict_animal_late,
#                     group="All Cells", ax=axB)

#     # Histogram
#     axC.hist(elbow_kmeans_array, bins='auto')
#     axC.set_title("Early Elbow Num Clusters Distribution")
#     axC.set_xlabel("Number of Clusters")
#     axC.set_ylabel("Number of Cells")

    plt.show()

def plot_clustered_data_learn(means_dict_cluster_0x0_raw, activity_early_array, activity_array_late, K=2, title="", cue=False, ax_list=None):
    data_good = means_dict_cluster_0x0_raw[K]["labels_loc_dict"]

    # fig, axs = plt.subplots(1,2, figsize=(4*len(data_good), 4))
    # fig.suptitle(title)

    for idx, i in enumerate(data_good):
        labels = data_good[i]
        n=len(labels)
        sliced_early = activity_early_array[labels,:]
        mean_sliced_early = np.mean(sliced_early, axis=0)
        sem_sliced_early = sem(sliced_early, axis=0)

        # sliced_data_early_dict[i] = sliced_early
        sliced_late = activity_array_late[labels,:]
        mean_sliced_late = np.mean(sliced_late, axis=0)
        sem_sliced_late = sem(sliced_late, axis=0)

        # sliced_data_late_dict[i] = sliced_late
        ax_list[idx].plot(mean_sliced_early, label='Early')
        ax_list[idx].fill_between(range(len(mean_sliced_early)), mean_sliced_early-sem_sliced_early, mean_sliced_early+sem_sliced_early, alpha=0.2)

        ax_list[idx].plot(mean_sliced_late, label="Late")
        ax_list[idx].fill_between(range(len(mean_sliced_late)), mean_sliced_late-sem_sliced_late, mean_sliced_late+sem_sliced_late, alpha=0.2)
        ax_list[idx].set_title(f"Cluster {i} n={n}")
        if cue:
            ax_list[idx].axvline(10,linestyle='--', color='r', label="Cue")
        ax_list[idx].legend()

    # plt.tight_layout()
    # plt.show()

def plot_reconstructions(labels_cells_dict_all_K_NDNF, fixed_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="", plot=False):


    synthetic_mean_array = {}
    means_dict_cluster= {}
    for num_clusters in labels_cells_dict_all_K_NDNF:
        data_truncated_array_NDNF = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
        # print(data_truncated_array_NDNF.shape)
        labels = labels_cells_dict_all_K_NDNF[num_clusters]
        # print(labels)
        uniq = np.unique(labels)
        real_data_mean_array = np.empty(data_truncated_array_NDNF.shape)
        mean_data_dict = {}

        r_vel_dict_per_clust = {}
        r_lick_dict_per_clust = {}

        cell_ids_per_cluster_dict = {}

        vel_data_sliced_dict = {}
        lick_data_sliced_dict = {}

        vel_array = r_dict_vel["array_data"]
        lick_array = r_dict_licks["array_data"]

        labels_loc_dict = {}

        mean_data_list = []
        fraction_cells_list = []
        for i in uniq:
            labels_loc = np.where(labels==i)[0]
            cell_ids_per_cluster_dict[i] = labels_loc
            fraction_cells = (len(labels_loc) / data_truncated_array_NDNF.shape[0])*100
            fraction_cells_list.append(fraction_cells)
            real_data_array_sliced = data_truncated_array_NDNF[labels_loc,:,:]
            mean_real_data_array_sliced = np.mean(real_data_array_sliced, axis=0)
            mean_data_list.append(mean_real_data_array_sliced)
            real_data_mean_array[labels_loc,:,:] = mean_real_data_array_sliced

            labels_loc_dict[i] = labels_loc
            
            r_list_vel = np.array(r_dict_vel["r_list"])

            r_list_licks = np.array(r_dict_licks["r_list"])

            

            vel_data_sliced_dict[i] = vel_array[labels_loc,:,:]

            lick_data_sliced_dict[i] = lick_array[labels_loc,:,:]

            r_vel_dict_per_clust[i] = r_list_vel[labels_loc]
            r_lick_dict_per_clust[i] = r_list_licks[labels_loc]



        MSE_reco_vs_real = np.mean(np.square(data_truncated_array_NDNF-real_data_mean_array))

        mean_data_dict = {"mean_data_list":mean_data_list,
                        "fraction_cells":fraction_cells_list,
                        "MSE_reco_vs_real":MSE_reco_vs_real,
                        "r_vel_dict_per_clust":r_vel_dict_per_clust,
                        "r_lick_dict_per_clust":r_lick_dict_per_clust,
                        "cell_ids_per_cluster_dict":cell_ids_per_cluster_dict,
                        "vel_data_sliced_dict":vel_data_sliced_dict,
                        "lick_data_sliced_dict":lick_data_sliced_dict,
                        "labels_loc_dict":labels_loc_dict}
        
        means_dict_cluster[num_clusters] = mean_data_dict
        synthetic_mean_array[num_clusters] = real_data_mean_array



    data_truncated_array_NDNF = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
    data_truncated_array_NDNF_ta = np.mean(data_truncated_array_NDNF, axis=2),
    if plot:
        fig, axs = plt.subplots(2,len(synthetic_mean_array), figsize=(30,8))
        plt.suptitle(f"Reconstructed vs Real Trial Averaged Data {prefix}")
        for num_clusters in synthetic_mean_array:
            reconstructed_data = synthetic_mean_array[num_clusters]
            reconstructed_data_ta = np.mean(reconstructed_data, axis=2)
            axs[0,num_clusters-1].imshow(reconstructed_data_ta, aspect='auto')
            axs[0,num_clusters-1].set_ylabel("Cell ID")
            axs[0,num_clusters-1].set_title(f"Reconstructed K={num_clusters}")
            axs[0,num_clusters-1].set_xlabel("Position Bin")
            axs[1,num_clusters-1].imshow(data_truncated_array_NDNF_ta, aspect='auto')
            axs[1,num_clusters-1].set_title("Real T.A. Data")
            axs[1,num_clusters-1].set_xlabel("Position Bin")
        plt.tight_layout()
        plt.show()

    return means_dict_cluster


def plot_mean_resid(residual_activity_dict_NDNF_newest, title, ax=None, plot=False):
    cue_residual_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if idx > 30:
            cue_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

    listy = []
    for animal in cue_residual_activity_dict_NDNF_newest:
        print(animal)
        for cell in cue_residual_activity_dict_NDNF_newest[animal]:
            listy.append(np.mean(cue_residual_activity_dict_NDNF_newest[animal][cell], axis=1))

    good_array = np.array(listy)
    means = np.mean(good_array, axis=0)
    sems = sem(good_array, axis=0)

    if plot:
        ax.plot(means)
        ax.axvline(10, linestyle="--", color='red', label='Cue')
        ax.fill_between(range(len(means)), means+sems, means-sems, alpha=0.2)
        ax.legend()
        ax.set_title(title)

    return cue_residual_activity_dict_NDNF_newest

def get_cells_per_animal_dict(fixed_activity_dict_NDNF_newest):
    cells_per_animal_dict = {}

    count=0
    for animal in fixed_activity_dict_NDNF_newest:
        per_animal_list=[]
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            per_animal_list.append(count)
            count+=1
        cells_per_animal_dict[animal] = per_animal_list

    return cells_per_animal_dict

# def plot_cluster_traces_by_animal(
#     means_dict_cluster,                 # output from plot_reconstructions (has cell_ids_per_cluster_dict)
#     fixed_activity_dict_NDNF_newest,    # to recompute TA traces
#     cells_per_animal_dict,              # {animal: [global_cell_ids...]}
#     K, ncol=None,                                  # which K to plot
#     ylim=(-1.1, 2.6), spacing=None,  
#     title_prefix=""):
#     # --- build cell_id -> animal map ---
#     cell_to_animal = {}
#     for animal, cell_ids in cells_per_animal_dict.items():
#         for cid in cell_ids:
#             cell_to_animal[cid] = animal

#     # --- get TA data (cells × pos) ---
#     data = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
#     ta = data.mean(axis=2)                                                   # (cells, pos)
#     n_cells, n_pos = ta.shape

#     # --- clusters & subplot layout ---
#     clust_idx_dict = means_dict_cluster[K]["cell_ids_per_cluster_dict"]      # {cluster_label: array(cell_ids)}
#     uniq = sorted(clust_idx_dict.keys())
#     label_to_col = {lab: j for j, lab in enumerate(uniq)}

#     # --- color map per animal ---
#     animals = sorted(set(cell_to_animal.values()))
#     cmap = plt.get_cmap("tab20", len(animals))
#     animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

#     fig, axs = plt.subplots(1, len(uniq), figsize=(4*len(uniq), 6), sharey=True)
#     if len(uniq) == 1:
#         axs = np.array([axs])

#     for lab in uniq:
#         ax = axs[label_to_col[lab]]
#         idx = np.asarray(clust_idx_dict[lab])
#         traces = ta[idx]                                   # (n_k, n_pos)

#         # plot each cell trace colored by its animal
#         for cid in idx:
#             a = cell_to_animal.get(int(cid), "unknown")
#             color = animal_to_color.get(a, (0.5,0.5,0.5,0.6))
#             ax.plot(traces[np.where(idx==cid)[0][0], :], lw=1.0, alpha=0.7, color=color)

#         # overlay mean ± SEM (neutral color)
#         m = traces.mean(axis=0)
#         s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
#         ax.plot(m, lw=2.0, color="k")
#         ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color="k")

#         ax.set_title(f"Cluster {lab} (n={len(idx)})")
#         ax.set_xlabel("Position bins")
#         ax.set_ylim(*ylim)
#     axs[0].set_ylabel("Z-scored dF/F")
#     fig.suptitle(f"{title_prefix} Traces colored by animal — K={K}", y=1.02, fontsize=12)

#     # legend outside: one entry per animal
#     handles = [plt.Line2D([0],[0], color=animal_to_color[a], lw=2) for a in animals]
#     labels = [f"{a}" for a in animals]
#     fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
#     fig.subplots_adjust(top=0.85, right=0.98, left=0.07, bottom=spacing)
#     plt.show()


def plot_cluster_traces_by_animal(
    means_dict_cluster,                 # output from plot_reconstructions (has cell_ids_per_cluster_dict)
    fixed_activity_dict_NDNF_newest,    # to recompute TA traces
    cells_per_animal_dict,              # {animal: [global_cell_ids...]}
    K, 
    ncol=1,                             # legend columns
    ylim=(-1.1, 2.6), 
    spacing=0.25,                       # horizontal spacing between subplots (wspace)
    title_prefix="",
    axs=None,                           # list/array of Axes (len >= #clusters) or None
    legend="bottom",                    # "bottom", "right", "inside", or None
color_list=None):
    """
    Plots trial-averaged (TA) traces per cluster, coloring each trace by its animal.
    If `axs` is None, creates a new 1×(#clusters) figure (shared y). Otherwise draws into provided axes.
    Legend shows animals with their colors and can be placed at 'bottom', 'right', 'inside', or disabled with None.
    """

    # --- build cell_id -> animal map ---
    cell_to_animal = {}
    for animal, cell_ids in cells_per_animal_dict.items():
        for cid in cell_ids:
            cell_to_animal[int(cid)] = animal

    # --- get TA data (cells × pos) ---
    data = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
    ta = data.mean(axis=2)                                                   # (cells, pos)
    _, n_pos = ta.shape

    # --- clusters & subplot layout ---
    clust_idx_dict = means_dict_cluster[K]["cell_ids_per_cluster_dict"]      # {cluster_label: array(cell_ids)}
    uniq = sorted(clust_idx_dict.keys())
    n_panels = len(uniq)
    label_to_col = {lab: j for j, lab in enumerate(uniq)}

    # --- color map per animal ---
    animals = sorted(set(cell_to_animal.values()))
    cmap = plt.get_cmap("tab20", max(1, len(animals)))
    animal_to_color = {a: cmap(i % cmap.N) for i, a in enumerate(animals)}

    # --- axes handling ---
    created_fig = False
    if axs is None:
        fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6), sharey=True)
        if n_panels == 1:
            axs = np.array([axs])
        created_fig = True
    else:
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = np.array([axs])
        else:
            axs = np.array(axs).ravel()
        if len(axs) < n_panels:
            raise ValueError(f"Provided axs has length {len(axs)} but need at least {n_panels} axes.")
        fig = axs[0].figure

    # adjust spacing if requested
    if spacing is not None:
        fig.subplots_adjust(wspace=spacing)

    # --- draw each cluster ---
    for i, lab in enumerate(uniq):
        ax = axs[label_to_col[lab]]
        idx = np.asarray(clust_idx_dict[lab], dtype=int)
        if idx.size == 0:
            ax.set_title(f"Cluster {lab} (n=0)")
            ax.set_xlim(0, n_pos - 1)
            ax.set_ylim(*ylim)
            continue

        traces = ta[idx]  # (n_k, n_pos)

        # plot each cell trace colored by its animal
        for j, cid in enumerate(idx):
            a = cell_to_animal.get(int(cid), "unknown")
            color = animal_to_color.get(a, (0.5, 0.5, 0.5, 0.6))
            ax.plot(traces[j, :], lw=1.0, alpha=0.7, color=color)

        # overlay mean ± SEM (neutral color)
        m = traces.mean(axis=0)
        s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
        ax.plot(m, lw=2.0, color=color_list[i], zorder=5)
        # ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color=color_list[i], zorder=4)

        ax.set_title(f"{title_prefix} Cluster {lab} (n={len(idx)})".strip())
        ax.set_xlabel("Position bins")
        ax.set_ylim(*ylim)

    axs[0].set_ylabel("Z-scored dF/F")

    # # --- legend handling ---
    # if legend is not None and len(animals) > 0:
    #     # make legend handles
    #     handles = []
    #     labels = []
    #     for a in animals:
    #         handles.append(plt.Line2D([0], [0], color=animal_to_color[a], lw=3))
    #         labels.append(str(a))

    #     if legend == "inside":
    #         axs[-1].legend(handles, labels, frameon=True, loc="upper right", ncol=ncol)
    #     elif legend == "right":
    #         # put one consolidated legend on the right of the rightmost axis
    #         leg = axs[-1].legend(handles, labels, frameon=False, loc="center left",
    #                              bbox_to_anchor=(1.02, 0.5), ncol=ncol)
    #         fig.subplots_adjust(right=0.78 if ncol == 1 else 0.88)
    #     elif legend == "bottom":
    #         # global legend at bottom of figure
    #         fig.legend(handles, labels, loc="lower center", ncol=max(ncol, 1), frameon=False)
    #         fig.subplots_adjust(bottom=0.15)
    #     # else: unknown string → skip legend silently

    # if created_fig:
    #     plt.show()

    return fig, axs


    # --- title ---
    # fig.suptitle(f"{title_prefix} Traces colored by animal — K={K}", y=0.98, fontsize=12)

    # # --- legend ---
    # if legend is not None and len(animals) > 0:
    #     handles = [plt.Line2D([0],[0], color=animal_to_color[a], lw=2) for a in animals]
    #     labels = [f"{a}" for a in animals]

    #     if legend == "bottom":
    #         fig.legend(handles, labels, loc="lower center", ncol=ncol or legend_ncol, frameon=False)
    #         if created_fig:
    #             fig.subplots_adjust(bottom=spacing if spacing is not None else 0.12, top=0.90, left=0.07, right=0.98)
    #     elif legend == "right":
    #         # put legend to the right of the last axis
    #         axs[-1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
    #                        frameon=False, ncol=ncol or legend_ncol)
    #         if created_fig:
    #             fig.subplots_adjust(right=0.80, top=0.90, left=0.07, bottom=0.12)
    #     elif legend == "inside":
    #         axs[-1].legend(handles, labels, loc="upper right", frameon=False, ncol=ncol or legend_ncol)

    # only show if we created the figure




# def plot_cluster_animal_composition_stacked(means_dict_cluster, cells_per_animal_dict, K,
#                                             title_prefix="", show_percent_labels=False):
#     # --- build cell_id -> animal map ---
#     cell_to_animal = {}
#     for animal, cell_ids in cells_per_animal_dict.items():
#         for cid in cell_ids:
#             cell_to_animal[int(cid)] = animal

#     # --- cluster membership dict for this K ---
#     clust_idx_dict = means_dict_cluster[K]["cell_ids_per_cluster_dict"]  # {cluster_label: array(cell_ids)}
#     clusters = sorted(clust_idx_dict.keys())  # keep order stable

#     # --- all animals & consistent colors ---
#     animals = sorted(set(cell_to_animal.values()))
#     cmap = plt.get_cmap("tab20", len(animals))
#     animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

#     # --- counts matrix: rows=clusters, cols=animals ---
#     counts = np.zeros((len(clusters), len(animals)), dtype=int)
#     for r, clab in enumerate(clusters):
#         for cid in clust_idx_dict[clab]:
#             a = cell_to_animal.get(int(cid), None)
#             if a is not None:
#                 c = animals.index(a)
#                 counts[r, c] += 1

#     totals = counts.sum(axis=1)  # cells per cluster

#     # --- plot stacked bars (height = counts, stacked by animal) ---
#     x = np.arange(len(clusters))
#     fig, ax = plt.subplots(figsize=(1.2*len(clusters)+3, 5))
#     bottom = np.zeros_like(totals, dtype=float)

#     bars = []
#     for j, a in enumerate(animals):
#         b = ax.bar(x, counts[:, j], bottom=bottom, width=0.8,
#                    color=animal_to_color[a], label=f"{a}")
#         bars.append(b)
#         bottom += counts[:, j]

#     # x-ticks as cluster labels with n
#     ax.set_xticks(x, [f"C{clab}\n(n={totals[i]})" for i, clab in enumerate(clusters)])
#     ax.set_ylabel("# cells")
#     ax.set_title(f"{title_prefix} K={K}")

#     # optional % labels inside segments
#     if show_percent_labels:
#         with np.errstate(divide='ignore', invalid='ignore'):
#             props = counts / totals[:, None]
#             props[np.isnan(props)] = 0.0
#         for i in range(len(clusters)):
#             cum = 0.0
#             for j in range(len(animals)):
#                 h = counts[i, j]
#                 if h > 0 and props[i, j] >= 0.07:  # only label if ≥7% to avoid clutter
#                     ax.text(x[i], cum + h/2.0, f"{props[i, j]*100:.0f}%",
#                             ha="center", va="center", fontsize=8, color="white")
#                 cum += h

#     # legend outside
#     ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
#     fig.subplots_adjust(right=0.78)


def plot_cluster_animal_composition_stacked(
    means_dict_cluster,
    cells_per_animal_dict,
    K,
    title_prefix="",
    show_percent_labels=False,
    ax=None,
    legend="outside",   # "outside", "inside", or None
    legend_ncol=1
):
    # --- build cell_id -> animal map ---
    cell_to_animal = {}
    for animal, cell_ids in cells_per_animal_dict.items():
        for cid in cell_ids:
            cell_to_animal[int(cid)] = animal

    # --- cluster membership dict for this K ---
    clust_idx_dict = means_dict_cluster[K]["cell_ids_per_cluster_dict"]  # {cluster_label: array(cell_ids)}
    clusters = sorted(clust_idx_dict.keys())  # keep order stable

    # --- all animals & consistent colors ---
    animals = sorted(set(cell_to_animal.values()))
    cmap = plt.get_cmap("tab20", len(animals))
    animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

    # --- counts matrix: rows=clusters, cols=animals ---
    import numpy as np
    counts = np.zeros((len(clusters), len(animals)), dtype=int)
    for r, clab in enumerate(clusters):
        for cid in clust_idx_dict[clab]:
            a = cell_to_animal.get(int(cid), None)
            if a is not None:
                c = animals.index(a)
                counts[r, c] += 1

    totals = counts.sum(axis=1)  # cells per cluster
    x = np.arange(len(clusters))

    # --- axes handling ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2*len(clusters)+3, 5))
        created_fig = True

    # --- plot stacked bars (height = counts, stacked by animal) ---
    bottom = np.zeros_like(totals, dtype=float)
    for j, a in enumerate(animals):
        ax.bar(x, counts[:, j], bottom=bottom, width=0.8,
               color=animal_to_color[a], label=f"{a}")
        bottom += counts[:, j]

    # x-ticks as cluster labels with n
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{clab}\n(n={totals[i]})" for i, clab in enumerate(clusters)])
    ax.set_ylabel("# cells")
    ax.set_title(f"{title_prefix} K={K}")

    # optional % labels inside segments
    if show_percent_labels:
        with np.errstate(divide='ignore', invalid='ignore'):
            props = counts / totals[:, None]
            props[np.isnan(props)] = 0.0
        for i in range(len(clusters)):
            cum = 0.0
            for j in range(len(animals)):
                h = counts[i, j]
                if h > 0 and props[i, j] >= 0.07:  # label only if ≥7% to avoid clutter
                    ax.text(x[i], cum + h/2.0, f"{props[i, j]*100:.0f}%",
                            ha="center", va="center", fontsize=8, color="white")
                cum += h

    # legend placement
    if legend == "outside":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=legend_ncol)
        # don't adjust figure if we're drawing into a provided ax
        if created_fig:
            fig.subplots_adjust(right=0.78)
    elif legend == "inside":
        ax.legend(loc="upper right", frameon=False)
    else:
        pass  # no legend

    return ax



def get_r_list(fixed_activity_dict_NDNF_newest, factors_dict_NDNF_newest, data_truncated_array_EC, data_to_corr=None):

    r_list_vel = []
    vel_per_animal = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            vel = factors_dict_NDNF_newest[animal][data_to_corr][:,:data_truncated_array_EC.shape[2]]
            vel_per_animal.append(vel)
            data = fixed_activity_dict_NDNF_newest[animal][cell][:,:data_truncated_array_EC.shape[2]]
            r, _ = pearsonr(vel.flatten(), data.flatten())
            r_list_vel.append(r)

    array_data = np.array(vel_per_animal)
    r_dict_vel = {"r_list":r_list_vel,
                  "array_data":array_data}

    return r_dict_vel


def get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest):
    min_val = 10000

    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            data = fixed_activity_dict_NDNF_newest[animal][cell]
            if data.shape[1] < min_val:
                min_val = data.shape[1]

    data_truncated_list = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            data_truncated = fixed_activity_dict_NDNF_newest[animal][cell][:,:min_val]
            data_truncated_list.append(data_truncated)


    data_truncated_array = np.array(data_truncated_list)

    return data_truncated_array


def plot_lick_vel_data_clust(means_dict_cluster_0x0_raw, num_clusters=3, use_vel=False, title=None, ax=None, color_list=None):

    if use_vel:
        vel_data = means_dict_cluster_0x0_raw[num_clusters]["vel_data_sliced_dict"]
    else:
        vel_data = means_dict_cluster_0x0_raw[num_clusters]["lick_data_sliced_dict"]


    # fig, axs = plt.subplots(1,len(vel_data), figsize=(len(vel_data)*4, 4))
    for i, clust in enumerate(vel_data):
        vel_array = vel_data[clust]
        trial_av_vel_array = np.mean(vel_array, axis=2)

        mean_over_cells = np.mean(trial_av_vel_array, axis=0)
        sem_over_cells = sem(trial_av_vel_array, axis=0)

        ax.plot(mean_over_cells, label=f"Cluster {clust} n={vel_array.shape[0]}", color=color_list[i])
        ax.fill_between(range(len(mean_over_cells)), mean_over_cells-sem_over_cells, mean_over_cells+sem_over_cells, alpha=0.2, color=color_list[i])
        # plt.title(f"Cluster {clust}")
        ax.set_xlabel("Position Bins")
        if use_vel:
            ax.set_ylabel(f"Velocity (meters/sec)")
        else:
            ax.set_ylabel(f"Normalized Lick Rate")
    ax.set_title(title)
    ax.legend()




def lda_with_orthogonal_axis_2d(X, labels, title_prefix=""):
    """
    Returns a 2D embedding:
    axis 1: Fisher LDA direction (max class separation)
    axis 2: top-variance direction orthogonal to LDA (via PCA in orth subspace)
    """
    X = np.asarray(X)
    y = np.asarray(labels)
    n_samples, n_features = X.shape
    if n_features < 2:
        raise ValueError("Need at least 2 features to build an orthogonal second axis.")

    # --- 1) Fit LDA (1 component) and get the discriminant vector w ---
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)

    # For binary classes, coef_ has shape (1, n_features)
    w = lda.coef_[0].astype(float)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        # fallback: if degenerate, use mean-difference direction
        classes = np.unique(y)
        mu0 = X[y == classes[0]].mean(axis=0)
        mu1 = X[y == classes[1]].mean(axis=0)
        w = (mu1 - mu0)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            raise ValueError("Could not determine a discriminant direction (degenerate data).")
    w /= w_norm  # unit vector

    # --- 2) Build the orthogonal subspace and find its top-variance direction ---
    # Project data onto the orthogonal complement of w
    # X_perp = X - (X w) w
    Xw = X @ w
    X_perp = X - np.outer(Xw, w)

    # Center X_perp (important for PCA)
    X_perp_centered = X_perp - X_perp.mean(axis=0, keepdims=True)

    # If all variance orthogonal to w vanishes, fall back to a random orth direction
    if np.allclose(np.var(X_perp_centered, axis=0).sum(), 0.0, atol=1e-12):
        # Random orth direction (Gram-Schmidt)
        rand_vec = np.random.randn(n_features)
        orth = rand_vec - (rand_vec @ w) * w
        if np.linalg.norm(orth) < 1e-12:
            # try again
            rand_vec = np.random.randn(n_features)
            orth = rand_vec - (rand_vec @ w) * w
        orth /= np.linalg.norm(orth)
    else:
        # PCA in the orthogonal subspace to get the most informative 2nd axis
        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(X_perp_centered)
        # PCA component is in the *feature* space of X_perp_centered, already orthogonal to w
        orth = pca.components_[0]
        # Keep it strictly orthogonal (numerical safety)
        orth = orth - (orth @ w) * w
        orth /= np.linalg.norm(orth)

    # --- 3) Project original X onto (w, orth) for 2D embedding ---
    X_2d = np.column_stack((X @ w, X @ orth))
    return X_2d





def plot_lda_projection(X, labels, title_prefix="LDA Projection", cluster_subset=None):
    """
    If cluster_subset is given (e.g., (0,1)), restrict to those two classes.
    For 2 classes -> 1D LDA projection (strip+hist).
    For >=3 classes -> 2D LDA scatter (first two discriminants).
    """
    labels = np.asarray(labels)

    # Optional: restrict to chosen clusters (e.g., (0,1))
    if cluster_subset is not None:
        mask = np.isin(labels, cluster_subset)
        X = X[mask]
        labels = labels[mask]

    uniq = np.unique(labels)
    uniq = np.unique(labels)

    if len(uniq) == 2:
        # 1) LDA axis
        lda = LinearDiscriminantAnalysis(n_components=1).fit(X, labels)
        w = lda.coef_[0].astype(float)
        w /= np.linalg.norm(w) + 1e-12

        # 2) Orthogonal top-variance axis (PCA in subspace ⟂ w)
        Xw = X @ w
        X_perp = X - np.outer(Xw, w)
        X_perp -= X_perp.mean(axis=0, keepdims=True)

        pca_orth = PCA(n_components=1, svd_solver="full").fit(X_perp)
        orth = pca_orth.components_[0]
        # enforce exact orthogonality (numerical safety)
        orth = orth - (orth @ w) * w
        orth /= np.linalg.norm(orth) + 1e-12

        X_2d = np.column_stack((X @ w, X @ orth))

        # fig, ax = plt.subplots(figsize=(7.5, 5))
        for k in uniq:
            m = labels == k
            ax.scatter(X_2d[m, 0], X_2d[m, 1], s=40, alpha=0.6, label=f"C{k} (n={m.sum()})")
        ax.set_xlabel("LDA axis (max class separation)")
        ax.set_ylabel("Orthogonal axis (top variance ⟂ LDA)")
        ax.set_title(f"LDA + Orthogonal PCA {title_prefix}")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
        fig.subplots_adjust(right=0.78)
        plt.show()

    else:
        # standard LDA with up to 2 components for K>=3
        lda = LinearDiscriminantAnalysis(n_components=min(2, len(uniq)-1))
        X_lda = lda.fit_transform(X, labels)

        fig, ax = plt.subplots(figsize=(7.5, 5))
        for k in uniq:
            m = labels == k
            ax.scatter(X_lda[m, 0], X_lda[m, 1], s=40, alpha=0.6, label=f"C{k} (n={m.sum()})")
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("LDA Component 2")
        ax.set_title(f"LDA Projection {title_prefix}")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
        fig.subplots_adjust(right=0.78)
        # plt.show()

def get_cell_features(w1, f1, *,
                      feature_mode="latent_pos_flat",  # "latent_pos_flat" | "latent_trials_flat" | "loadings" | "pos_profile"
                      n_cells_expected=None,
                      n_latents_expected=None):
    """
    Build per-cell feature matrix X (rows=cells).
    Handles arbitrary axis orders from SliceTCA outputs across (x00, 0x0, 00x).
    w1, f1 are torch tensors from model.vectors[which_vectors].

    feature_mode:
      - "latent_pos_flat":   (cells, latents*pos)        e.g., (115, 20*50)
      - "latent_trials_flat":(cells, latents*trials)     e.g., (115, 20*100)
      - "loadings":          (cells, latents)            e.g., (115, 20)
      - "pos_profile":       (cells, pos) via weighted latent templates

    Notes:
      • If a chosen mode can’t be formed from the available axes, it raises a helpful error.
      • Pass n_cells_expected=115 (and n_latents_expected=20) to make detection strict.
    """
    # to numpy
    W = w1.detach().cpu().numpy()
    F = f1.detach().cpu().numpy()

    # small helpers
    def find_axis_by_size(shape, target):
        return shape.index(target) if (target in shape) else None

    def move_front(a, ax):
        return np.moveaxis(a, ax, 0)

    # Try to detect latents axis (often 20)
    Lw = find_axis_by_size(W.shape, n_latents_expected) if n_latents_expected else None
    Lf = find_axis_by_size(F.shape, n_latents_expected) if n_latents_expected else None
    # Detect cells axis (often 115)
    Cw = find_axis_by_size(W.shape, n_cells_expected) if n_cells_expected else None
    Cf = find_axis_by_size(F.shape, n_cells_expected) if n_cells_expected else None

    # ---- Option 1: LOADINGS (cells × latents) ----
    if feature_mode == "0x0":
        # Prefer a (latents, cells) slice if present (W or F that directly has cells)
        if (Lw is not None) and (Cw is not None) and W.ndim == 2:
            # W (latents, cells) -> (cells, latents)
            X = np.moveaxis(W, (Lw, Cw), (1, 0))
            X = X.T  # (cells, latents)
            return X
        # Else try from F by reducing non-latent dims
        if Cf is not None and Lf is not None and F.ndim >= 2:
            # bring cells front, latents second
            order = [Cf, Lf] + [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            G = np.transpose(F, order)     # (cells, latents, ...)
            if G.ndim == 2:
                return G  # already (cells, latents)
            # average remaining dims (e.g., trials/pos) → (cells, latents)
            X = G.mean(axis=tuple(range(2, G.ndim)))
            return X
        raise ValueError("Cannot form 'loadings': no (latents,cells) pairing found in W/F.")

    # ---- Option 2: latent × pos flattened (cells × (L*P)) ----
    if feature_mode == "x00":
        # We need a tensor that contains (latents, cells, pos) in some order
        # Check F first (most common)
        if Cf is not None and Lf is not None and 3 <= F.ndim <= 4:
            # find a pos-like axis: pick the remaining non-cells, non-latents axis
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            if not rem:
                raise ValueError("No pos axis found in F for latent_pos_flat.")
            pos_ax = rem[0]
            # reorder to (cells, latents, pos, [maybe ...])
            order = [Cf, Lf, pos_ax] + [ax for ax in rem[1:]]
            G = np.transpose(F, order)
            # if more dims exist, mean over them
            if G.ndim > 3:
                G = G.mean(axis=tuple(range(3, G.ndim)))
            C, L, P = G.shape
            X = G.reshape(C, L * P)
            return X
        # Else: if W has (latents, pos) and *no* cells, we can’t build per-cell features from W alone.
        raise ValueError("Cannot form 'latent_pos_flat' from current W/F: need latents+cells+pos together (usually in F).")

    # ---- Option 3: latent × trials flattened (cells × (L*T)) ----
    if feature_mode == "00x":
        # Need (latents, trials, cells) in some order
        if Cf is not None and Lf is not None and 3 <= F.ndim <= 4:
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            if not rem:
                raise ValueError("No trials axis found in F for latent_trials_flat.")
            tr_ax = rem[0]
            order = [Cf, Lf, tr_ax] + [ax for ax in rem[1:]]
            G = np.transpose(F, order)  # (cells, latents, trials, [maybe ...])
            if G.ndim > 3:
                G = G.mean(axis=tuple(range(3, G.ndim)))
            C, L, T = G.shape
            X = G.reshape(C, L * T)
            return X
        raise ValueError("Cannot form 'latent_trials_flat': need latents+cells+trials together (usually in F).")

    # ---- Option 4: pos_profile (cells × pos), via weighted templates if possible ----
    if feature_mode == "pos_profile":
        # If we have (latents,cells, pos) in F: average latents → (cells, pos)
        if Cf is not None and Lf is not None and 3 <= F.ndim:
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            pos_ax = rem[0] if rem else None
            if pos_ax is not None:
                order = [Cf, Lf, pos_ax] + [ax for ax in rem[1:]]
                G = np.transpose(F, order)  # (cells, latents, pos, ...)
                if G.ndim > 3:
                    G = G.mean(axis=tuple(range(3, G.ndim)))
                X = G.mean(axis=1)  # avg over latents -> (cells, pos)
                return X
        raise ValueError("Cannot form 'pos_profile' from current W/F.")

    raise ValueError(f"Unknown feature_mode: {feature_mode}")
    



# def get_selectivity_each_trial_cell_type(activity_dict_EC, cells_list, neg_sel=True, trial_av=False):
#     """
#     - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
#     returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually
#     """

#     count=0


#     animal_average_selectivity_dict = {}
#     for animal in activity_dict_EC:
#         cell_dict = {}
#         for cell in activity_dict_EC[animal]:
#             print(f"count {count} cells_list {cells_list} ")
#             if count in cells_list:
#                 cell_data = activity_dict_EC[animal][cell]
#                 if trial_av:
#                     trial_av_activity = np.mean(cell_data, axis=1) 
#                     selectivity_trial_av = Vinje2000(trial_av_activity, norm='none', negative_selectivity=neg_sel)
#                     cell_dict[cell] = selectivity_trial_av
#                 else:
#                     trial_selectivity_list = []
#                     for trial in range(cell_data.shape[1]):
#                         trial_activity = cell_data[:,trial] 
#                         selectivity_trial = Vinje2000(trial_activity, norm='none', negative_selectivity=neg_sel)
#                         trial_selectivity_list.append(selectivity_trial)
                    
#                     percentile_average_selectivity = np.mean(trial_selectivity_list)
#                     cell_dict[cell] = percentile_average_selectivity
#             count+=1
#         animal_average_selectivity_dict[animal] = cell_dict
#     return animal_average_selectivity_dict

def Vinje2000(tuning_curve, norm='None', negative_selectivity=False):
    if norm == 'min_max':
        tuning_curve = (tuning_curve - np.min(tuning_curve)) / (np.max(tuning_curve) - np.min(tuning_curve))
        if negative_selectivity:
            tuning_curve = np.absolute(1 - tuning_curve)
    elif norm == 'z_score':
        tuning_curve = (tuning_curve - np.mean(tuning_curve)) / np.std(tuning_curve)
        if negative_selectivity:
            tuning_curve = np.absolute(1 - tuning_curve)
    A = np.mean(tuning_curve) ** 2 / np.mean(tuning_curve ** 2)
    return (1 - A) / (1 - 1 / len(tuning_curve))


def get_selectivity_each_trial_cell_type(activity_dict_EC, cells_list, neg_sel=True, trial_av=True, norm=None):
    count = 0
    cells_set = set(int(x) for x in cells_list)

    out = {}
    animals_dict_data = {}
    for animal in activity_dict_EC:
        cell_dict = {}
        cell_dict_data = {}
        for cell in activity_dict_EC[animal]:
            # print(f"count {count} cells_list {cells_list}")
            if count in cells_set:
                cell_data = activity_dict_EC[animal][cell]
                if trial_av:
                    trial_av_activity = np.mean(cell_data, axis=1)
                    cell_dict_data[cell] = cell_data
                    val = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)

                else:
                    vals = [Vinje2000(cell_data[:, tr], norm='none', negative_selectivity=neg_sel)
                            for tr in range(cell_data.shape[1])]
                    val = float(np.mean(vals)) if len(vals) else np.nan
                cell_dict[cell] = val
            count += 1  # increment on EVERY cell
        out[animal] = cell_dict
        animals_dict_data[animal] = cell_dict_data
    return out, animals_dict_data



# def get_binned_data_for_CDF(animal_average_selectivity_dict_SST_r, n_bins=20):

#     """
#     params animal_average_selectivity_dict_SST_r: selectivity value for every cell in a dict
#     returns binned_data: the cells with selectiivty values that fit within each bin (percent of data)
#     """

#     selectivity_list = []
#     for animal in animal_average_selectivity_dict_SST_r:
#         for cell in animal_average_selectivity_dict_SST_r[animal]:
#             val = animal_average_selectivity_dict_SST_r[animal][cell]
#             selectivity_list.append(val)

#     selectivity_array = np.array(selectivity_list)

#     edges = np.quantile(selectivity_array, np.linspace(0, 1, n_bins+1))  

#     binned_data = []

#     for idx in range(n_bins):
#         low = edges[idx]
#         high = edges[idx + 1]

#         # Include low, exclude high except last bin
#         if idx < 19:
#             in_bin = selectivity_array[(selectivity_array >= low) & (selectivity_array < high)]
#         else:
#             in_bin = selectivity_array[(selectivity_array >= low) & (selectivity_array <= high)]

#         binned_data.append(in_bin)

#     return binned_data


# def get_mean_sem_lists(binned_data):
#     mean_data_list = []
#     sem_data_list = []
#     for i in range(len(binned_data)):
#         bin_data = binned_data[i]
#         mean_data_list.append(np.mean(bin_data))
#         sem_data_list.append(sem(bin_data))
#     return mean_data_list, sem_data_list

def get_binned_data_for_CDF(animal_average_selectivity_dict, n_bins=20):
    # collect values
    vals = []
    for animal in animal_average_selectivity_dict:
        for cell in animal_average_selectivity_dict[animal]:
            v = animal_average_selectivity_dict[animal][cell]
            vals.append(v)

    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]  # drop NaN/inf
    if arr.size == 0:
        # return n_bins empty arrays to keep the downstream shape
        return [np.array([]) for _ in range(n_bins)]

    # if fewer points than bins, reduce bins to available size
    n_eff = min(n_bins, arr.size)

    # quantile edges can repeat when many identical values
    edges = np.quantile(arr, np.linspace(0, 1, n_eff + 1))

    binned = []
    for i in range(n_eff):
        low, high = edges[i], edges[i+1]
        if i < n_eff - 1:
            mask = (arr >= low) & (arr < high)
        else:
            mask = (arr >= low) & (arr <= high)
        binned.append(arr[mask])

    # pad if n_eff < n_bins (keeps plotting code unchanged)
    if n_eff < n_bins:
        binned += [np.array([])] * (n_bins - n_eff)

    return binned

def get_mean_sem_lists(binned_data):
    mean_data_list, sem_data_list = [], []
    for bin_data in binned_data:
        n = bin_data.size
        if n == 0:
            mean_data_list.append(np.nan)  # or choose a sentinel
            sem_data_list.append(0.0)
        elif n == 1:
            mean_data_list.append(float(bin_data[0]))
            sem_data_list.append(0.0)
        else:
            mean_data_list.append(float(np.mean(bin_data)))
            sem_data_list.append(float(sem(bin_data)))  # ddof handled by scipy
    return mean_data_list, sem_data_list



def plot_the_CDF_early_late(binned_data_0_early, binned_data_1_early, binned_data_0_late, binned_data_1_late, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
    mean_0_early, sem_0_early = get_mean_sem_lists(binned_data_0_early)
    mean_1_early, sem_1_early = get_mean_sem_lists(binned_data_1_early)

    mean_0_late, sem_0_late = get_mean_sem_lists(binned_data_0_late)
    mean_1_late, sem_1_late = get_mean_sem_lists(binned_data_1_late)


    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5

    # Plot: horizontal bars (x=selectivity, y=percentile)
    # plt.figure(figsize=(7, 6))

    ax.errorbar(mean_0_early, percentiles, xerr=sem_0_early, fmt='o-', label='Cell Type 0 Early', color='purple', capsize=3)
    ax.errorbar(mean_0_late, percentiles, xerr=sem_0_late, fmt='o-', label='Cell Type 0 Late', color='magenta', capsize=3)
    ax.errorbar(mean_1_early, percentiles, xerr=sem_1_early, fmt='o-', label='Cell Type 1 Early', color='red', capsize=3)
    ax.errorbar(mean_1_late, percentiles, xerr=sem_1_late, fmt='o-', label='Cell Type 1 Late', color='orange', capsize=3)

    ax.set_ylabel("Percentile of Cells")
    ax.set_xlabel("Selectivity")
    ax.set_title(title)
    ax.legend(fontsize=6)


def plot_the_CDF_celltypes(binned_data_0, binned_data_1, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
    mean_0, sem_0 = get_mean_sem_lists(binned_data_0)
    mean_1, sem_1 = get_mean_sem_lists(binned_data_1)

    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5

    # Plot: horizontal bars (x=selectivity, y=percentile)
    # plt.figure(figsize=(7, 6))

    ax.errorbar(mean_0, percentiles, xerr=sem_0, fmt='o-', label='Cell Type 0', color='purple', capsize=3)
    ax.errorbar(mean_1, percentiles, xerr=sem_1, fmt='o-', label='Cell Type 1', color='red', capsize=3)

    ax.set_ylabel("Percentile of Cells")
    ax.set_xlabel("Selectivity")
    ax.set_title(title)
    ax.legend()




# def _flatten_selectivities(sel_dict):
#     vals = [v for d in sel_dict.values() for v in d.values()]
#     arr = np.asarray(vals, float)
#     return arr[np.isfinite(arr)]

# def _ecdf(arr):
#     if arr.size == 0:
#         return np.array([]), np.array([])
#     x = np.sort(arr)
#     y = (np.arange(1, x.size + 1) / x.size) * 100.0  # percent
#     return x, y

# def plot_ecdf_celltypes(sel_dict_0, sel_dict_1, title="Selectivity CDF"):
#     a0 = _flatten_selectivities(sel_dict_0)
#     a1 = _flatten_selectivities(sel_dict_1)

#     x0, y0 = _ecdf(a0)
#     x1, y1 = _ecdf(a1)

#     plt.figure(figsize=(7,6))
#     if x0.size:
#         plt.step(x0, y0, where='post', label='Cell Type 0')
#     if x1.size:
#         plt.step(x1, y1, where='post', label='Cell Type 1')
#     plt.xlabel("Selectivity")
#     plt.ylabel("Cumulative % of cells")
#     plt.title(title)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def get_selectivity_each_trial_early_late_cluster(activity_dict_EC, cp_dict_EC, cell_cluster_list, neg_sel=True, trial_av=False, eml="early", norm=None):
    """
    - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
    returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually
    """

    count=0

    animal_dict_data = {}

    animal_average_selectivity_dict = {}
    for idx, animal in enumerate(activity_dict_EC):
        cell_dict = {}
        cell_dict_data = {}
        for idt, cell in enumerate(activity_dict_EC[animal]):
            
            if count in cell_cluster_list:

                cp = cp_dict_EC[idx][idt]
                early_cut = cp[0]
                late_cut = cp[1]
                cell_data = activity_dict_EC[animal][cell]
                if trial_av:
                    if eml=="early":
                        cell_data_early = cell_data[:,:early_cut]
                        trial_av_activity = np.mean(cell_data_early, axis=1) 
                        cell_dict_data[cell] = trial_av_activity
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict[cell] = selectivity_trial_av
                    elif eml=="middle":
                        cell_data_late = cell_data[:,early_cut:late_cut]
                        trial_av_activity = np.mean(cell_data_late, axis=1) 
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict_data[cell] = trial_av_activity
                        cell_dict[cell] = selectivity_trial_av
                    elif eml=="late":
                        cell_data_late = cell_data[:,-late_cut:]
                        trial_av_activity = np.mean(cell_data_late, axis=1) 
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict_data[cell] = trial_av_activity
                        cell_dict[cell] = selectivity_trial_av
                    else:
                        raise ValueError("improper eml")
                else:

                    ####### have to fix this 
                    trial_selectivity_list = []
                    if eml=="early":
                        for trial in range(cell_data.shape[1]):
                            if trial <= early_cut:
                                trial_activity = cell_data[:,trial] 
                                selectivity_trial = Vinje2000(trial_activity, norm=norm, negative_selectivity=neg_sel)
                                trial_selectivity_list.append(selectivity_trial)
                                cell_dict_data[cell] = trial_activity
                    else:
                        for trial in range(cell_data.shape[1]):
                            if trial >= late_cut:
                                trial_activity = cell_data[:,trial] 
                                selectivity_trial = Vinje2000(trial_activity, norm=norm, negative_selectivity=neg_sel)
                                trial_selectivity_list.append(selectivity_trial)
                                cell_dict_data[cell] = trial_activity

                    percentile_average_selectivity = np.mean(trial_selectivity_list)
                    cell_dict[cell] = percentile_average_selectivity
            count+=1
            animal_average_selectivity_dict[animal] = cell_dict
            animal_dict_data[animal] = cell_dict_data
    return animal_average_selectivity_dict, animal_dict_data



def get_selectivity_array(animal_average_selectivity_dict):

    all_cell_selectivity = []

    for animal in animal_average_selectivity_dict:
        for cell in animal_average_selectivity_dict[animal]:
            selectivity_per_bin = animal_average_selectivity_dict[animal][cell]
            if len(selectivity_per_bin) == 10:  # sanity check
                all_cell_selectivity.append(selectivity_per_bin)

    all_cell_selectivity = np.array(all_cell_selectivity)  # shape: [n_cells, 10]

    return all_cell_selectivity


def plot_selectivity_over_trials(group_0_selectivity, group_1_selectivity, color_list=None, ax=None):

    mean_selectivity_0 = np.mean(group_0_selectivity, axis=0)
    sem_selectivity_0 = sem(group_0_selectivity, axis=0)

    mean_selectivity_1 = np.mean(group_1_selectivity, axis=0)
    sem_selectivity_1 = sem(group_1_selectivity, axis=0)

    x = np.arange(1, 11) 

    ax.plot(x, mean_selectivity_0, color=color_list[0], label='Cell Type 0')
    ax.fill_between(x, mean_selectivity_0 - sem_selectivity_0, mean_selectivity_0 + sem_selectivity_0, alpha=0.2, color=color_list[0])
    ax.plot(x, mean_selectivity_1, color=color_list[1], label='Cell Type 1')
    ax.fill_between(x, mean_selectivity_1 - sem_selectivity_1, mean_selectivity_1 + sem_selectivity_1, alpha=0.2, color=color_list[1])
    ax.set_xticks(ticks=x, labels=[f"{int(p)}%" for p in np.linspace(0, 100, 10)])
    ax.set_xlabel("Percentile of Trials")
    ax.set_ylabel("Average Selectivity Across Cells")
    ax.set_title("Selectivity Across Trials")
    ax.legend()


def get_percentlie_slices(activity_dict_SST):
    percentile_slices = {}

    for animal in activity_dict_SST:
        percentile_slices_cell = {}
        for cell in activity_dict_SST[animal]:
            data = activity_dict_SST[animal][cell]
            num_trials = data.shape[1]

            cut_indices = [int(p * num_trials / 10) for p in range(1, 10)]  
            cut_indices = [0] + cut_indices + [num_trials] 

            cell_slices = []
            for idx in range(10):
                start = cut_indices[idx]
                end = cut_indices[idx + 1]
                data_slice = data[:, start:end]
                cell_slices.append(data_slice)

            percentile_slices_cell[cell] = cell_slices
        percentile_slices[animal] = percentile_slices_cell

    return percentile_slices


def selectivity_from_percentile_slices(percentile_slices, norm='min_max', neg_sel=False):
    """
    percentile_slices[animal][cell] = list of 10 arrays, each (pos, trials_in_bin)
    Returns array of shape (n_cells, 10) with one scalar selectivity per bin.
    """
    rows = []
    for animal in percentile_slices:
        for cell, list10 in percentile_slices[animal].items():
            if len(list10) != 10:
                continue
            vec = []
            for sl in list10:
                # sl: (pos, trials) – average across trials, then compute selectivity across pos
                if sl.size == 0:
                    vec.append(np.nan)
                    continue
                ta = np.mean(sl, axis=1)  # (pos,)
                vec.append(Vinje2000(ta, norm=norm, negative_selectivity=neg_sel))
            rows.append(vec)
    return np.asarray(rows, dtype=float)


def plot_selectivity_seperated_by_learn_stage(type0, type1, colors_list=None, ax=None):

    NDNF_means0, NDNF_sems0 = get_mean_selelectivity_by_cutpoint(type0)

    NDNF_means1, NDNF_sems1 = get_mean_selelectivity_by_cutpoint(type1)

    x = np.arange(3)
    labels = ["Early", "Middle", "Late"]

    ax.errorbar(x, NDNF_means0, yerr=NDNF_sems0, color=colors_list[0], label="Cell Type 0", capsize=4, fmt='-o')
    ax.errorbar(x, NDNF_means1, yerr=NDNF_sems1, color=colors_list[1], label="Cell Type 1", capsize=4, fmt='-o')

    ax.set_xticks(x, labels)
    ax.set_ylabel("Average Selectivity Across Cells")
    ax.set_xlabel("Contiguous K-Means Learning Stage")
    ax.set_title("Selectivity by Learning Stage")
    ax.legend()
    

def get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_SST):
    early_list_SST = []
    middle_list_SST = []
    late_list_SST = []

    for animal in animal_average_selectivity_dict_SST:
        for cell in animal_average_selectivity_dict_SST[animal]:
            early_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["early_selectivity"])
            middle_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["middle_selectivity"])
            late_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["late_selectivity"])

    early_mean = np.mean(early_list_SST)
    middle_mean = np.mean(middle_list_SST)
    late_mean = np.mean(late_list_SST)

    early_sem = sem(early_list_SST)
    middle_sem = sem(middle_list_SST)
    late_sem = sem(late_list_SST)

    SST_means = [early_mean, middle_mean, late_mean]
    SST_sems = [early_sem, middle_sem, late_sem]
    return SST_means, SST_sems


def get_animal_average_selectivity_dict_eml(residual_activity_dict_SST, cp_dict_SST, neg_sel=True, trial_av=False):
   
    animal_average_selectivity_dict = {}
    for idx, animal in enumerate(residual_activity_dict_SST):
        average_selectivity_dict_cell = {}
        for idt, cell in enumerate(residual_activity_dict_SST[animal]):
            cell_data = residual_activity_dict_SST[animal][cell]

            cp_e = cp_dict_SST[idx][idt][0]
            cp_l = cp_dict_SST[idx][idt][1]
            
            for i in range(len(cell_data)):

                if trial_av:
                    trial_av_activity_e = np.mean(cell_data[:,:cp_e], axis=1)
                    trial_av_activity_m = np.mean(cell_data[:,cp_e:cp_l], axis=1)
                    trial_av_activity_l = np.mean(cell_data[:,-cp_l:], axis=1)

                    selectivity_trial_av_e = Vinje2000(trial_av_activity_e, norm='none', negative_selectivity=neg_sel)
                    selectivity_trial_av_m = Vinje2000(trial_av_activity_m, norm='none', negative_selectivity=neg_sel)
                    selectivity_trial_av_l = Vinje2000(trial_av_activity_l, norm='none', negative_selectivity=neg_sel)

                    average_selectivity_dict_cell[cell] = {"early_selectivity": selectivity_trial_av_e,
                                                           "middle_selectivity": selectivity_trial_av_m,
                                                           "late_selectivity": selectivity_trial_av_l,
                                                           }
                else:
                    
                    early_list = []
                    middle_list = []
                    late_list = []

                    for trial in range(cell_data.shape[1]):
                        if trial <= cp_e:
                            data_trial = cell_data[:,trial]
                            selectivity = Vinje2000(data_trial, norm='none', negative_selectivity=neg_sel)
                            early_list.append(selectivity)
                        elif cp_e < trial < cp_l: 
                            data_trial = cell_data[:,trial]
                            selectivity = Vinje2000(data_trial, norm='none', negative_selectivity=neg_sel)
                            middle_list.append(selectivity)
                        elif trial >= cp_l: 
                            data_trial = cell_data[:,trial]
                            selectivity = Vinje2000(data_trial, norm='none', negative_selectivity=neg_sel)
                            late_list.append(selectivity)

                    trial_av_selectivity_early = np.mean(early_list)
                    trial_av_selectivity_middle = np.mean(middle_list)
                    trial_av_selectivity_late = np.mean(late_list)
                    
                    average_selectivity_dict_cell[cell] = {"early_selectivity": trial_av_selectivity_early,
                                                           "middle_selectivity": trial_av_selectivity_middle,
                                                           "late_selectivity": trial_av_selectivity_late,
                                                           }
                    

        animal_average_selectivity_dict[animal] = average_selectivity_dict_cell
    
    return animal_average_selectivity_dict




def collect_eml_data(animal_average_selectivity_dict_NDNF_0_early):

    cell_list = []

    for animal in animal_average_selectivity_dict_NDNF_0_early:
        for cell in animal_average_selectivity_dict_NDNF_0_early[animal]:
            cell_list.append(animal_average_selectivity_dict_NDNF_0_early[animal][cell])

    cell_array = np.array(cell_list)

    return cell_array
            

def plot_eml_data(animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=None, color_list=None):

    cell_array_0_early = collect_eml_data(animal_average_selectivity_dict_NDNF_0_early)
    cell_array_1_early = collect_eml_data(animal_average_selectivity_dict_NDNF_1_early)

    cell_array_0_middle = collect_eml_data(animal_average_selectivity_dict_NDNF_0_middle)
    cell_array_1_middle = collect_eml_data(animal_average_selectivity_dict_NDNF_1_middle)

    cell_array_0_late = collect_eml_data(animal_average_selectivity_dict_NDNF_0_late)
    cell_array_1_late = collect_eml_data(animal_average_selectivity_dict_NDNF_1_late)


    cell_array_0_early_mean = np.mean(cell_array_0_early)
    cell_array_0_early_sem = sem(cell_array_0_early)

    cell_array_1_early_mean = np.mean(cell_array_1_early)
    cell_array_1_early_sem = sem(cell_array_1_early)

    cell_array_0_middle_mean = np.mean(cell_array_0_middle)
    cell_array_0_middle_sem = sem(cell_array_0_middle)

    cell_array_1_middle_mean = np.mean(cell_array_1_middle)
    cell_array_1_middle_sem = sem(cell_array_1_middle)

    cell_array_0_late_mean = np.mean(cell_array_0_late)
    cell_array_0_late_sem = sem(cell_array_0_late)

    cell_array_1_late_mean = np.mean(cell_array_1_late)
    cell_array_1_late_sem = sem(cell_array_1_late)

    means0 = [cell_array_0_early_mean,cell_array_0_middle_mean, cell_array_0_late_mean]
    sems0 = [cell_array_0_early_sem,cell_array_0_middle_sem, cell_array_0_late_sem]

    means1 = [cell_array_1_early_mean,cell_array_1_middle_mean, cell_array_1_late_mean]
    sems1 = [cell_array_1_early_sem,cell_array_1_middle_sem, cell_array_1_late_sem]

    ax.errorbar(range(len(means0)), means0, yerr=sems0, label="Cell Type 0", color=color_list[0])
    ax.errorbar(range(len(means1)), means1, yerr=sems1, label="Cell Type 1", color=color_list[1])
    ax.set_xticks(np.arange(3), ["Early", "Middle", "Late"])
    ax.legend()





def run():
    GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="EC_GLM", new_NDNF=False)


    GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)

    # fixed_residual_activity_dict_NDNF_newest = {}
    # for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
    #     if 17 < idx < 31:
    #         fixed_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

    # fixed_activity_dict_NDNF_newest = {}
    # for idx, animal in enumerate(activity_dict_NDNF_newest):
    #     if 17 < idx < 31:
    #         fixed_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]

    mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
    cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)
    
    cued_cell_NDNF_model_ranks20_contig_x00_cue = {20: {}}
    for idx, animal in enumerate(cell_NDNF_model_ranks20_contig_x00_cue[20]):
        if idx>30:
            cued_cell_NDNF_model_ranks20_contig_x00_cue[20][animal-31] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]


    cued_factors_dict_NDNF_newest = {}
    for idx, animal in enumerate(factors_dict_NDNF_newest):
        if idx > 30:
            cued_factors_dict_NDNF_newest[f"animal_{idx+1}"] = factors_dict_NDNF_newest[animal]


    cued_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(activity_dict_NDNF_newest):
        if idx> 30:
            cued_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]

    cued_residual_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if idx> 30:
            cued_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]


############ make this for the cued list and then make plot the run and licks

# 
    data_truncated_array_NDNF = get_truncated_to_min_data_array(cued_activity_dict_NDNF_newest)


    r_dict_vel = get_r_list(cued_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, data_truncated_array_NDNF, data_to_corr="Velocity")
    r_dict_licks = get_r_list(cued_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, data_truncated_array_NDNF, data_to_corr="Licks")

    
    mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
    cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)

        
    

    cued_cp_dict_NDNF = get_cp_dict(cued_cell_NDNF_model_ranks20_contig_x00_cue)


    save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/model_20_NDNF_resid_0x0_cue.pkl"


    with open(save_path, 'rb') as f:
        model_20_NDNF_resid_0x0_cue = pickle.load(f)
        print(save_path)


    fig, axs = plt.subplots(3,4, figsize=(15,15))


    labels_cells_dict_all_K_NDNF_0x0_resid_cue = get_labels_all_different_Ks_single(model_20_NDNF_resid_0x0_cue, which_vectors=1)


    cue_residual_activity_dict_NDNF_newest = plot_mean_resid(residual_activity_dict_NDNF_newest, title="Residuals", ax=axs[0,0], plot=False)


    means_dict_cluster_0x0_cue_resid = plot_reconstructions(labels_cells_dict_all_K_NDNF_0x0_resid_cue, cue_residual_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="NDNF 0x0 Cue Resid")

    activity_early_array, activity_array_late = get_activity_cut_learn(cue_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF)
    plot_clustered_data_learn(means_dict_cluster_0x0_cue_resid, activity_early_array, activity_array_late, K=2, title="Cued Track Residuals NDNF Clustered Changepoint", ax_list=[axs[2,0],axs[2,1]])

    cells_per_animal_dict = get_cells_per_animal_dict(cue_residual_activity_dict_NDNF_newest)


    color_list = ["purple", "red"]


    plot_cluster_traces_by_animal(means_dict_cluster_0x0_cue_resid,
                                cue_residual_activity_dict_NDNF_newest,
                                cells_per_animal_dict,
                                K=2,
                                ncol=5, spacing=0.2,
                                title_prefix="Residuals", axs=[axs[0,0],axs[0,1]], color_list = color_list)
    

    plot_cluster_animal_composition_stacked(means_dict_cluster_0x0_cue_resid, cells_per_animal_dict, K=2,
                                            title_prefix="NDNF 0x0 Residual Cue", show_percent_labels=True, ax=axs[1,0])



    plot_lick_vel_data_clust(means_dict_cluster_0x0_cue_resid, num_clusters=2, use_vel=False, title="Licks NDNF Cued Track", ax=axs[1,2], color_list=color_list)
    plot_lick_vel_data_clust(means_dict_cluster_0x0_cue_resid, num_clusters=2, use_vel=True, title="Velocity NDNF Cued Track", ax=axs[1,1], color_list=color_list)

    which_vectors=1

    labels_cells_dict_all_K_NDNF = get_labels_all_different_Ks_single(model_20_NDNF_resid_0x0_cue, which_vectors=which_vectors)

    labels = np.asarray(labels_cells_dict_all_K_NDNF[2])

    w1 = model_20_NDNF_resid_0x0_cue.vectors[which_vectors][0]
    f1 = model_20_NDNF_resid_0x0_cue.vectors[which_vectors][1]

    labels = np.asarray(labels_cells_dict_all_K_NDNF[2])
    n_latents_expected = 20
    n_cells_expected = len(labels)  

    X = get_cell_features(
        w1, f1,
        feature_mode="0x0",
        n_cells_expected=n_cells_expected,
        n_latents_expected=n_latents_expected,)


    
    if X.shape[0] != len(labels) and X.shape[1] == len(labels):
        X = X.T
    elif X.shape[0] != len(labels):
        raise ValueError(f"Shape mismatch: X.shape={X.shape}, labels={len(labels)}")


    # plot_lda_projection(X, labels, title_prefix="LDA Projection", cluster_subset=None, ax=axs[2,2])

    uniq = np.unique(labels)
    X_2d = lda_with_orthogonal_axis_2d(X, labels, title_prefix="")

    for i, k in enumerate(uniq):
        m = labels == k
        axs[0,2].scatter(X_2d[m, 0], X_2d[m, 1], s=40, alpha=0.6, color=color_list[i], label=f"C{k} (n={m.sum()})")

    axs[0,2].set_xlabel("LDA Component 1")
    axs[0,2].set_ylabel("LDA Component 2")

    leg = axs[0,2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.subplots_adjust(right=0.78)

    cells_list_0 = np.where(labels==0)[0]
    cells_list_1 = np.where(labels==1)[0]

    # sel0 = get_selectivity_each_trial_cell_type(cued_residual_activity_dict_NDNF_newest, cells_list_0, neg_sel=False, trial_av=True, norm="min_max")
    # sel1 = get_selectivity_each_trial_cell_type(cued_residual_activity_dict_NDNF_newest, cells_list_1, neg_sel=False, trial_av=True, norm="min_max")

    # plot_ecdf_celltypes(sel0, sel1, title="Selectivity CDF")


    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(cued_residual_activity_dict_NDNF_newest, cells_list_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(cued_residual_activity_dict_NDNF_newest, cells_list_1, neg_sel=False, trial_av=True, norm="min_max")


    binned_data_NDNF_0 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0, n_bins=20)
    binned_data_NDNF_1 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1, n_bins=20)

    plot_the_CDF_celltypes(binned_data_NDNF_0, binned_data_NDNF_1, title="Selectivity Distribution", n_bins = 20, ax=axs[2,2])



    
    animal_average_selectivity_dict_NDNF_0_early,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="early", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_early,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="early", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_late,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="late", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_late,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="late", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_middle,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="middle", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_middle,_ = get_selectivity_each_trial_early_late_cluster(cued_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="middle", norm="min_max")

    print(f"animal_average_selectivity_dict_NDNF_1_middle {animal_average_selectivity_dict_NDNF_1_middle}")


    binned_data_NDNF_0_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_early, n_bins=20)
    binned_data_NDNF_1_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_early, n_bins=20)

    binned_data_NDNF_0_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_late, n_bins=20)
    binned_data_NDNF_1_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_late, n_bins=20)


    plot_the_CDF_early_late(binned_data_NDNF_0_early, binned_data_NDNF_1_early, binned_data_NDNF_0_late, binned_data_NDNF_1_late, title="Selectivity Distribution Early Late Learn", n_bins = 20, ax=axs[2,3])



    percentile_slices_NDNF0 = get_percentlie_slices(animals_dict_data_NDNF_0)
    percentile_slices_NDNF1 = get_percentlie_slices(animals_dict_data_NDNF_1)
    

    all_cells_NDNF_0 = selectivity_from_percentile_slices(percentile_slices_NDNF0, norm='min_max', neg_sel=False)
    all_cells_NDNF_1 = selectivity_from_percentile_slices(percentile_slices_NDNF1, norm='min_max', neg_sel=False)

    plot_selectivity_over_trials(all_cells_NDNF_0, all_cells_NDNF_1, color_list=color_list, ax=axs[1,3])


    plot_eml_data(animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=axs[0,3], color_list=color_list)



    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    run()

