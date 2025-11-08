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


def plot_mean_resid(residual_activity_dict_NDNF_newest, title, ax=None):
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
    ncol=None,                          # legend columns
    ylim=(-1.1, 2.6), 
    spacing=None,  
    title_prefix="",
    axs=None,                           # <-- pass a list/array of Axes (len >= #clusters) or None
    legend="bottom",                    # "bottom", "right", "inside", or None
    legend_ncol=1
):
    """
    If `axs` is None, creates a new figure with one row of subplots (one per cluster).
    If `axs` is provided, it must be an array-like of Axes with length >= number of clusters.
    Only calls plt.show() if it created the figure.
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
    cmap = plt.get_cmap("tab20", len(animals))
    animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

    # --- axes handling ---
    created_fig = False
    if axs is None:
        fig, axs = plt.subplots(1, n_panels, figsize=(4*n_panels, 6), sharey=True)
        created_fig = True
    else:
        # normalize axs to a 1D numpy array for indexing
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = np.array([axs])
        else:
            axs = np.array(axs).ravel()
        if len(axs) < n_panels:
            raise ValueError(f"Provided axs has length {len(axs)} but need at least {n_panels} axes.")
        fig = axs[0].figure  # figure hosting the provided axes

    if n_panels == 1 and axs.ndim == 0:
        axs = np.array([axs])

    # --- draw each cluster ---
    for lab in uniq:
        ax = axs[label_to_col[lab]]
        idx = np.asarray(clust_idx_dict[lab], dtype=int)
        traces = ta[idx]                                   # (n_k, n_pos)

        # plot each cell trace colored by its animal
        # (map cid -> row index once for speed)
        row_index = {cid: i for i, cid in enumerate(idx)}
        for cid in idx:
            a = cell_to_animal.get(int(cid), "unknown")
            color = animal_to_color.get(a, (0.5, 0.5, 0.5, 0.6))
            ax.plot(traces[row_index[int(cid)], :], lw=1.0, alpha=0.7, color=color)

        # overlay mean ± SEM (neutral color)
        if traces.size > 0:
            m = traces.mean(axis=0)
            s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
            ax.plot(m, lw=2.0, color="k")
            ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color="k")

        ax.set_title(f"Cluster {lab} (n={len(idx)})")
        ax.set_xlabel("Position bins")
        ax.set_ylim(*ylim)

    axs[0].set_ylabel("Z-scored dF/F")

    # --- title ---
    fig.suptitle(f"{title_prefix} Traces colored by animal — K={K}", y=0.98, fontsize=12)

    # --- legend ---
    if legend is not None and len(animals) > 0:
        handles = [plt.Line2D([0],[0], color=animal_to_color[a], lw=2) for a in animals]
        labels = [f"{a}" for a in animals]

        if legend == "bottom":
            fig.legend(handles, labels, loc="lower center", ncol=ncol or legend_ncol, frameon=False)
            if created_fig:
                fig.subplots_adjust(bottom=spacing if spacing is not None else 0.12, top=0.90, left=0.07, right=0.98)
        elif legend == "right":
            # put legend to the right of the last axis
            axs[-1].legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                           frameon=False, ncol=ncol or legend_ncol)
            if created_fig:
                fig.subplots_adjust(right=0.80, top=0.90, left=0.07, bottom=0.12)
        elif legend == "inside":
            axs[-1].legend(handles, labels, loc="upper right", frameon=False, ncol=ncol or legend_ncol)

    # only show if we created the figure
    if created_fig:
        plt.show()

    return axs



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


def plot_lick_vel_data_clust(means_dict_cluster_0x0_raw, num_clusters=3, use_vel=False, title=None, ax=None):

    if use_vel:
        vel_data = means_dict_cluster_0x0_raw[num_clusters]["vel_data_sliced_dict"]
    else:
        vel_data = means_dict_cluster_0x0_raw[num_clusters]["lick_data_sliced_dict"]


    # fig, axs = plt.subplots(1,len(vel_data), figsize=(len(vel_data)*4, 4))
    for clust in vel_data:
        vel_array = vel_data[clust]
        trial_av_vel_array = np.mean(vel_array, axis=2)

        mean_over_cells = np.mean(trial_av_vel_array, axis=0)
        sem_over_cells = sem(trial_av_vel_array, axis=0)

        ax.plot(mean_over_cells, label=f"Cluster {clust} n={vel_array.shape[0]}")
        ax.fill_between(range(len(mean_over_cells)), mean_over_cells-sem_over_cells, mean_over_cells+sem_over_cells, alpha=0.2)
        # plt.title(f"Cluster {clust}")
        ax.set_xlabel("Position Bins")
        if use_vel:
            ax.set_ylabel(f"Velocity (meters/sec)")
        else:
            ax.set_ylabel(f"Normalized Lick Rate")
    ax.set_title(title)
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


    fig, axs = plt.subplots(3,3, figsize=(15,15))


    labels_cells_dict_all_K_NDNF_0x0_resid_cue = get_labels_all_different_Ks_single(model_20_NDNF_resid_0x0_cue, which_vectors=1)


    cue_residual_activity_dict_NDNF_newest = plot_mean_resid(residual_activity_dict_NDNF_newest, title="Residuals NDNF Cue Track", ax=axs[0,0])


    means_dict_cluster_0x0_cue_resid = plot_reconstructions(labels_cells_dict_all_K_NDNF_0x0_resid_cue, cue_residual_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="NDNF 0x0 Cue Resid")

    activity_early_array, activity_array_late = get_activity_cut_learn(cue_residual_activity_dict_NDNF_newest, cued_cp_dict_NDNF)
    plot_clustered_data_learn(means_dict_cluster_0x0_cue_resid, activity_early_array, activity_array_late, K=2, title="Cued Track Residuals NDNF Clustered Changepoint", ax_list=[axs[2,0],axs[2,1]])

    cells_per_animal_dict = get_cells_per_animal_dict(cue_residual_activity_dict_NDNF_newest)

    plot_cluster_traces_by_animal(means_dict_cluster_0x0_cue_resid,
                                cue_residual_activity_dict_NDNF_newest,
                                cells_per_animal_dict,
                                K=2,
                                ncol=5, spacing=0.2,
                                title_prefix="NDNF Cue Track 0x0 Residuals", axs=[axs[0,1],axs[0,2]])
    

    plot_cluster_animal_composition_stacked(means_dict_cluster_0x0_cue_resid, cells_per_animal_dict, K=2,
                                            title_prefix="NDNF 0x0 Residual Cue", show_percent_labels=True, ax=axs[1,0])



    plot_lick_vel_data_clust(means_dict_cluster_0x0_cue_resid, num_clusters=2, use_vel=False, title="Licks NDNF Cued Track", ax=axs[1,2])
    plot_lick_vel_data_clust(means_dict_cluster_0x0_cue_resid, num_clusters=2, use_vel=True, title="Velocity NDNF Cued Track", ax=axs[1,1])

    plt.tight_layout()
    plt.show()




# @click.command()
# @click.option(
#     '--most-expressed/--no-most-expressed',
#     default=False,
#     help="Use the 'most expressed' scanning logic."
# )
# def cli(most_expressed):
#     run(most_expressed)

if __name__ == "__main__":
    run()#cli()

