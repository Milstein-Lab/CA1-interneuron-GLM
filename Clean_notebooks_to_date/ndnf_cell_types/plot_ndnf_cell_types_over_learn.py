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

def get_lists_out_of_dicts(fixed_TT_data, fixed_activity_dict_NDNF_newest, cp_dict_NDNF):
    TT_list = []
    for animal in fixed_TT_data:
        for cell in fixed_TT_data[animal]:
            TT_list.append(fixed_TT_data[animal][cell][1][f"cell_{cell}"])

    print(len(TT_list)) 

    NDNF_activity_list = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            NDNF_activity_list.append(fixed_activity_dict_NDNF_newest[animal][cell])

    print(len(NDNF_activity_list)) 

    cp_list_NDNF = []

    for animal in cp_dict_NDNF:
        for cell in cp_dict_NDNF[animal]:
            cp_list_NDNF.append(cp_dict_NDNF[animal][cell])

    print(len(cp_list_NDNF)) 

    return TT_list, NDNF_activity_list, cp_list_NDNF

def find_elbow_point(y_vals, min_index=2):
    from scipy.spatial.distance import cdist
    import numpy as np

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

def get_most_expressed_cluster(TT_list, activity_list, cp_list_NDNF, early_late_none="early", to_include=None, most_expressed=True):
    
    if np.any(to_include) == None:
        indices_to_include = np.arange(115)
    else:
        indices_to_include = to_include

    most_expressed_label_dict = {}

    elbow_kmeans_array = np.empty(len(indices_to_include))

    for j, cell in enumerate(indices_to_include):
              
        activity_array = activity_list[cell]

        labels_example = TT_list[cell]["labels_dict"]["clusters_chosen_3"]

        MSE_dict = TT_list[cell]["MSE_dict"]

        MSE_array = np.empty(len(MSE_dict))

        for id, clusters_chosen in enumerate(MSE_dict):
            MSE = MSE_dict[clusters_chosen]
            MSE_array[id] = MSE
        
        elbow_kmeans = find_elbow_point(MSE_array)

        elbow_kmeans_array[j] = elbow_kmeans
        
        if early_late_none=='early':
            cp_early = cp_list_NDNF[cell][0]
            labels = labels_example[:cp_early]

        elif early_late_none=='late':
            cp_late = cp_list_NDNF[cell][1]
            labels = labels_example[cp_late:]

        elif early_late_none=='none':
            labels = labels_example
        else:
            raise ValueError("Invalid learning chunk")


        good_dict = get_max_proportion(labels, use_max=most_expressed)

        correct_indices = np.where(labels==good_dict["unique_label"])[0]
        activity_array_sliced = activity_array[:,correct_indices]


        most_expressed_label_dict[cell] = {"label":good_dict["unique_label"],
                                                        "cluster_activity": activity_array_sliced, "fraction":good_dict["fraction"]}

    return most_expressed_label_dict, elbow_kmeans_array



def eval_proportion(most_expressed_label_dict_animal_early, most_expressed_label_dict_animal_late, group=None, ax=None, color=None):
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
            color=color, marker='o', linewidth=2)

    ax.set_xticks([0,1])
    ax.set_xticklabels(['Early','Late'])
    ax.set_ylabel("Fraction of Trials")
    ax.set_title(f"{group} \n Paired t-test (p = {p:.3f})")


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



# def plot_no_learn_cell_types(most_expressed_label_dict_animal_cluster_0_all, most_expressed_label_dict_animal_cluster_1_all, elbow_kmeans_array, group=None, most_expressed=None, axs_list=None):

    
#     counter=0
#     count = axs_list[counter]
    
#     # fig, axs = plt.subplots(4,2, figsize=(8,12))

#     mean_0_list = []
#     mean_1_list = []

#     mean_0_fractions_list = []
#     mean_1_fractions_list = []

#     count.set_title("Group0")
#     for cell in most_expressed_label_dict_animal_cluster_0_all:
#         array0 = most_expressed_label_dict_animal_cluster_0_all[cell]["cluster_activity"]
#         mean0 = np.mean(array0, axis=1)
#         mean_0_list.append(mean0)
#         count.plot(mean0)
#         mean_0_fractions_list.append(most_expressed_label_dict_animal_cluster_0_all[cell]["fraction"])

#     counter+=1

#     count.set_title("Group1")
#     for cell in most_expressed_label_dict_animal_cluster_1_all:
#         array1 = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
#         mean1 = np.mean(array1, axis=1)
#         mean_1_list.append(mean1)
#         count.plot(mean1)
#         mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])

#     counter+=1

#     mean_0_array = np.array(mean_0_list)
#     mean_1_array = np.array(mean_1_list)

#     mean_mean_0_array = np.mean(mean_0_array, axis=0)
#     mean_mean_1_array = np.mean(mean_1_array, axis=0)

#     sem_0_array = sem(mean_0_array, axis=0)
#     sem_1_array = sem(mean_1_array, axis=0)

#     count.plot(mean_mean_0_array, label='Group0')
#     count.fill_between(range(len(mean_mean_0_array)), mean_mean_0_array-sem_0_array, mean_mean_0_array+sem_0_array, alpha=0.2)
#     counter+=1

#     count.plot(mean_mean_1_array, label='Group1')
#     count.fill_between(range(len(mean_mean_1_array)), mean_mean_1_array-sem_1_array, mean_mean_1_array+sem_1_array, alpha=0.2)
#     counter+=1
    
#     count.hist(mean_0_fractions_list)
#     count.set_xlabel("Fraction of Total Trials")
#     count.set_ylabel("Cells")
#     counter+=1


#     count.hist(mean_1_fractions_list)
#     count.set_xlabel("Fraction of Total Trials")
#     count.set_ylabel("Cells")
#     counter+=1

#     count.plot(mean_mean_0_array, label='Early')
#     count.plot(mean_mean_1_array, label='Late')
#     count.fill_between(range(len(mean_mean_0_array)), mean_mean_0_array-sem_0_array, mean_mean_0_array+sem_0_array, alpha=0.2)
#     count.fill_between(range(len(mean_mean_1_array)), mean_mean_1_array-sem_1_array, mean_mean_1_array+sem_1_array, alpha=0.2)
#     count.set_title(group)
#     count.legend()
#     counter+=1

#     count.hist(elbow_kmeans_array, bins=[1.5,2.5,3.5,4.5,5.5])
#     count.set_xticks([2,3,4,5])
#     count.set_xlim(1.5,5.5)
#     count.set_title("Elbow Num Clusters Distribution")
#     count.set_xlabel("Num Clusters")
#     count.set_ylabel("Num Cells")
#     counter+=1

#     # plt.tight_layout()
#     # plt.show()


def plot_no_learn_cell_types(
    most_expressed_label_dict_animal_cluster_0_all,
    most_expressed_label_dict_animal_cluster_1_all,
    least_expressed_label_dict_animal_all_group0,
    least_expressed_label_dict_animal_all_group1,
    group=None,
    most_expressed=True,
    axs_list=None, color_dict=None
):
    """
    Expects axs_list length == 8 laid out however you want.
    Panel order:
      0: Group0 per-cell traces
      1: Group1 per-cell traces
      2: Group0 mean ± SEM
      3: Group1 mean ± SEM
      4: Group0 fraction histogram
      5: Group1 fraction histogram
      6: Overlay (Group0 vs Group1) mean ± SEM
      7: Elbow histogram (distribution of chosen K)
    """
    if axs_list is None or len(axs_list) < 8:
        raise ValueError("axs_list must be provided with at least 8 axes.")

    title_txt = "Most Expressed Cluster Across Trials" if most_expressed else "Least Expressed Cluster Across Trials"
    if group:
        axs_list[0].figure.suptitle(f"{title_txt} — {group}")
    else:
        axs_list[0].figure.suptitle(title_txt)

    # --- collect means per cell for each group ---
    mean_0_list_most = []
    mean_0_list_least = []
    mean_1_list_most = []
    mean_1_list_least = []
    most_mean_0_fractions_list = []
    least_mean_0_fractions_list = []
    most_mean_1_fractions_list = []
    least_mean_1_fractions_list = []

    # Panel 0: Group0 per-cell traces
    # ax = axs_list[0]
    axs_list[0].set_title("Cell Type 0 \n Most Expressed")
    axs_list[2].set_title("Cell Type 0 \n Least Expressed")
    for cell in most_expressed_label_dict_animal_cluster_0_all:
        arr_most = most_expressed_label_dict_animal_cluster_0_all[cell]["cluster_activity"]
        arr_least = least_expressed_label_dict_animal_all_group0[cell]["cluster_activity"]
        arr_most_mean = np.mean(arr_most, axis=1)  # avg over trials/time dim as you intended
        arr_least_mean = np.mean(arr_least, axis=1)
        mean_0_list_most.append(arr_most_mean)
        mean_0_list_least.append(arr_least_mean)
        axs_list[0].plot(arr_most_mean, alpha=0.4, color='gray')
        axs_list[2].plot(arr_least_mean, alpha=0.4, color='gray')
        most_mean_0_fractions_list.append(most_expressed_label_dict_animal_cluster_0_all[cell]["fraction"])
        least_mean_0_fractions_list.append(least_expressed_label_dict_animal_all_group0[cell]["fraction"])


    # ax = axs_list[1]
    axs_list[1].set_title("Cell Type 1 \n Most Expressed")
    axs_list[3].set_title("Cell Type 1 \n Least Expressed")
    for cell in most_expressed_label_dict_animal_cluster_1_all:
        arr_most = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
        arr_least = least_expressed_label_dict_animal_all_group1[cell]["cluster_activity"]
        arr_most_mean = np.mean(arr_most, axis=1)  # avg over trials/time dim as you intended
        arr_least_mean = np.mean(arr_least, axis=1)
        mean_1_list_most.append(arr_most_mean)
        mean_1_list_least.append(arr_least_mean)
        axs_list[1].plot(arr_most_mean, alpha=0.4, color='gray')
        axs_list[3].plot(arr_least_mean, alpha=0.4, color='gray')
        most_mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])
        least_mean_1_fractions_list.append(least_expressed_label_dict_animal_all_group1[cell]["fraction"])

    # # Panel 1: Group1 per-cell traces
    # ax = axs_list[1]
    # ax.set_title("Group1: per-cell")
    # for cell in most_expressed_label_dict_animal_cluster_1_all:
    #     arr = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
    #     m = np.mean(arr, axis=1)
    #     mean_1_list.append(m)
    #     ax.plot(m, alpha=0.4)
    #     mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])

    # Convert to arrays
    mean_0_array_most = np.array(mean_0_list_most)
    mean_1_array_most = np.array(mean_1_list_most)

    mean_0_array_least = np.array(mean_0_list_least)
    mean_1_array_least = np.array(mean_1_list_least)


    mean_mean_0_array_most = np.mean(mean_0_array_most, axis=0)
    mean_mean_1_array_most = np.mean(mean_1_array_most, axis=0)

    mean_mean_0_array_least = np.mean(mean_0_array_least, axis=0)
    mean_mean_1_array_least = np.mean(mean_1_array_least, axis=0)


    sem_0_most = sem(mean_0_array_most, axis=0, nan_policy='omit') 
    sem_1_most = sem(mean_1_array_most, axis=0, nan_policy='omit')

    
    sem_0_least = sem(mean_0_array_least, axis=0, nan_policy='omit') 
    sem_1_least = sem(mean_1_array_least, axis=0, nan_policy='omit')

    axs_list[0].plot(mean_mean_0_array_most, linewidth=4, color=color_dict["Most_0"])
    axs_list[0].set_xlabel("Position Bins")
    axs_list[2].set_xlabel("Position Bins")
    axs_list[0].set_ylabel("Z-Scored DF/F")
    axs_list[2].set_ylabel("Z-Scored DF/F")
    axs_list[0].set_ylim(-1.5, 4)
    axs_list[2].plot(mean_mean_0_array_least, linewidth=4, color=color_dict["Least_0"])
    axs_list[2].set_ylim(-1.5, 4)

    axs_list[1].set_xlabel("Position Bins")
    axs_list[3].set_xlabel("Position Bins")
    axs_list[1].set_ylabel("Z-Scored DF/F")
    axs_list[3].set_ylabel("Z-Scored DF/F")
    axs_list[1].plot(mean_mean_1_array_most, linewidth=4, color=color_dict["Most_1"])
    axs_list[1].set_ylim(-1.5, 4)
    axs_list[3].plot(mean_mean_1_array_least, linewidth=4, color=color_dict["Least_1"])
    axs_list[3].set_ylim(-1.5, 4)

    ax = axs_list[4]
    ax.set_title("Cell Type 0")
    ax.set_xlabel("Position Bins")
    ax.set_ylabel("Z-Scored DF/F")
    ax.plot(mean_mean_0_array_most, label="Most Expressed Trial Type", color=color_dict["Most_0"])
    ax.fill_between(range(len(mean_mean_0_array_most)), mean_mean_0_array_most - sem_0_most, mean_mean_0_array_most + sem_0_most, alpha=0.2, color=color_dict["Most_0"])
    ax.plot(mean_mean_0_array_least, label="Least Expressed Trial Type", color=color_dict["Least_0"])
    ax.fill_between(range(len(mean_mean_0_array_least)), mean_mean_0_array_least - sem_0_least, mean_mean_0_array_least + sem_0_least, alpha=0.2, color=color_dict["Least_0"])
    ax.legend(fontsize=6)

    ax = axs_list[5]
    ax.set_xlabel("Position Bins")
    ax.set_title("Cell Type 1")
    ax.plot(mean_mean_1_array_most, label="Most Expressed Trial Type", color=color_dict["Most_1"])
    ax.fill_between(range(len(mean_mean_1_array_most)), mean_mean_1_array_most - sem_1_most, mean_mean_1_array_most + sem_1_most, alpha=0.2, color=color_dict["Most_1"])
    ax.plot(mean_mean_1_array_least, label="Least Expressed Trial Type", color=color_dict["Least_1"])
    ax.fill_between(range(len(mean_mean_1_array_least)), mean_mean_1_array_least - sem_1_least, mean_mean_1_array_least + sem_1_least, alpha=0.2, color=color_dict["Least_1"])
    ax.set_ylabel("Z-Scored DF/F")
    ax.legend(fontsize=6)
    
    # # Panel 2: Group0 mean±SEM
    # ax = axs_list[2]
    # ax.set_title("Group0: mean ± SEM")
    # if mean_mean_0.size:
    #     t = np.arange(len(mean_mean_0))
    #     ax.plot(mean_mean_0, label='Group0')
    #     ax.fill_between(t, mean_mean_0 - sem_0, mean_mean_0 + sem_0, alpha=0.2)
    #     ax.legend(frameon=False)

    # # Panel 3: Group1 mean±SEM
    # ax = axs_list[3]
    # ax.set_title("Group1: mean ± SEM")
    # if mean_mean_1.size:
    #     t = np.arange(len(mean_mean_1))
    #     ax.plot(mean_mean_1, label='Group1')
    #     ax.fill_between(t, mean_mean_1 - sem_1, mean_mean_1 + sem_1, alpha=0.2)
    #     ax.legend(frameon=False)

    # Panel 4: Group0 fraction histogram
    # ax = axs_list[4]
    # ax.hist(most_mean_0_fractions_list, bins='auto', alpha=0.2)
    # ax.hist(least_mean_0_fractions_list, bins='auto', alpha=0.2)
    # ax.set_title("Group0: fraction of trials")
    # ax.set_xlabel("Fraction")
    # ax.set_ylabel("Cells")

    all_vals = np.concatenate([most_mean_0_fractions_list, least_mean_0_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    print(f"len(most_mean_0_fractions_list) {len(most_mean_0_fractions_list)}")

    ax = axs_list[6]
    ax.hist(most_mean_0_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_0"])
    ax.hist(least_mean_0_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_0"])
    ax.set_title("Cell Type 0")
    ax.set_xlabel("Fraction of Trials"); ax.set_ylabel("Number of Cells")
    ax.legend(frameon=False)

    # Panel 5: Group1 fraction histogram
    # ax = axs_list[5]
    # ax.hist(most_mean_1_fractions_list, bins='auto', alpha=0.2)
    # ax.hist(least_mean_1_fractions_list, bins='auto', alpha=0.2)
    # ax.set_title("Group1: fraction of trials")
    # ax.set_xlabel("Fraction")
    # ax.set_ylabel("Cells")

    all_vals = np.concatenate([most_mean_1_fractions_list, least_mean_1_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    ax = axs_list[7]
    ax.hist(most_mean_1_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_1"])
    ax.hist(least_mean_1_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_1"])
    ax.set_title("Cell Type 1")
    ax.set_xlabel("Fraction of Trials"); ax.set_ylabel("Number of Cells")
    ax.legend(frameon=False)

    # # Panel 6: Overlay Group0 vs Group1 mean±SEM
    # ax = axs_list[6]
    # ax.set_title("Overlay: Group0 vs Group1")
    # if mean_mean_0.size:
    #     t0 = np.arange(len(mean_mean_0))
    #     ax.plot(mean_mean_0, label='Group0')
    #     ax.fill_between(t0, mean_mean_0 - sem_0, mean_mean_0 + sem_0, alpha=0.2)
    # if mean_mean_1.size:
    #     t1 = np.arange(len(mean_mean_1))
    #     ax.plot(mean_mean_1, label='Group1')
    #     ax.fill_between(t1, mean_mean_1 - sem_1, mean_mean_1 + sem_1, alpha=0.2)
    # ax.legend(frameon=False)

    # Panel 7: Elbow histogram
    


def plot_clustered_data_learn(means_dict_cluster_0x0_raw, activity_early_array, activity_array_late, K=2, title=""):
    data_good = means_dict_cluster_0x0_raw[K]["labels_loc_dict"]

    fig, axs = plt.subplots(1,len(data_good), figsize=(4*len(data_good), 4))
    fig.suptitle(title)

    for i in data_good:
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
        axs[i].plot(mean_sliced_early, label='Early')
        axs[i].fill_between(range(len(mean_sliced_early)), mean_sliced_early-sem_sliced_early, mean_sliced_early+sem_sliced_early, alpha=0.2)

        axs[i].plot(mean_sliced_late, label="Late")
        axs[i].fill_between(range(len(mean_sliced_late)), mean_sliced_late-sem_sliced_late, mean_sliced_late+sem_sliced_late, alpha=0.2)
        axs[i].set_title(f"Cluster {i} n={n}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def get_labels_all_different_Ks_single(model_20_NDNF_resid, which_vectors: int):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

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


def plot_lick_vel_data_clust(means_dict_cluster_0x0_raw, num_clusters=3, use_vel=False, title=None):

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

        plt.plot(mean_over_cells, label=f"Cluster {clust} n={vel_array.shape[0]}")
        plt.fill_between(range(len(mean_over_cells)), mean_over_cells-sem_over_cells, mean_over_cells+sem_over_cells, alpha=0.2)
        # plt.title(f"Cluster {clust}")
        plt.xlabel("Position Bins")
        if use_vel:
            plt.ylabel(f"Velocity (meters/sec)")
        else:
            plt.ylabel(f"Normalized Lick Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_early_late_activity(
    most_expressed_label_dict_animal_early,
    most_expressed_label_dict_animal_late,
    elbow_kmeans_array,
    group=None,
    most_expressed=True):
    # ---------- compute means/sems once ----------
    mean_early_list = []
    mean_late_list  = []

    for cell in most_expressed_label_dict_animal_early:
        early_array = most_expressed_label_dict_animal_early[cell]["cluster_activity"]  # (clusters, time) or similar
        late_array  = most_expressed_label_dict_animal_late[cell]["cluster_activity"]

        mean_early = np.mean(early_array, axis=1)  # per-cluster mean trace (adjust if your axis differs)
        mean_late  = np.mean(late_array,  axis=1)

        mean_early_list.append(mean_early)
        mean_late_list.append(mean_late)

    mean_early_array = np.asarray(mean_early_list)
    mean_late_array  = np.asarray(mean_late_list)

    mean_mean_early_array = np.mean(mean_early_array, axis=0)
    mean_mean_late_array  = np.mean(mean_late_array,  axis=0)

    sem_early_array = sem(mean_early_array, axis=0, nan_policy='omit')
    sem_late_array  = sem(mean_late_array,  axis=0, nan_policy='omit')

    # ---------- figure with nested grids ----------
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    title_txt = "Most Highly Expressed Cluster Per Cell" if most_expressed else "Most Highly Expressed Cluster Per Cell"
    if group:
        fig.suptitle(f"{title_txt} — {group}", y=1.05)
    else:
        fig.suptitle(title_txt, y=1.05)

    # Outer grid: 2 rows × 4 cols; first column is wide to host a 2×2 sub-grid
    outer = fig.add_gridspec(2, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.1, hspace=0.01)

    # ----- Left block: 2×2 subgrid (your original first figure) -----
    left = outer[:, 0].subgridspec(2, 2, wspace=0.01, hspace=0.1, height_ratios=(1.0,1.0), width_ratios=(0.1,0.1))
    ax00 = fig.add_subplot(left[0, 0])
    ax01 = fig.add_subplot(left[0, 1])
    ax10 = fig.add_subplot(left[1, 0])
    ax11 = fig.add_subplot(left[1, 1])

    # Top row: per-cell traces (Early / Late)
    for cell in most_expressed_label_dict_animal_early:
        early_array = most_expressed_label_dict_animal_early[cell]["cluster_activity"]
        late_array  = most_expressed_label_dict_animal_late[cell]["cluster_activity"]
        ax00.plot(np.mean(early_array, axis=1))
        ax01.plot(np.mean(late_array,  axis=1))

    ax00.set_title("Early")
    ax01.set_title("Late")

    # Bottom row: mean ± SEM
    tE = np.arange(len(mean_mean_early_array))
    tL = np.arange(len(mean_mean_late_array))
    ax10.plot(mean_mean_early_array, label='Early')
    ax11.plot(mean_mean_late_array,  label='Late')

    ax10.fill_between(tE, mean_mean_early_array - sem_early_array,
                           mean_mean_early_array + sem_early_array, alpha=0.2)
    ax11.fill_between(tL, mean_mean_late_array  - sem_late_array,
                           mean_mean_late_array  + sem_late_array,  alpha=0.2)

    ax10.legend(loc='upper right', frameon=False)
    ax11.legend(loc='upper right', frameon=False)

    # ----- Right three single panels (formerly 1×3) -----
    axA = fig.add_subplot(outer[:, 1])  # summary line plot
    axB = fig.add_subplot(outer[:, 2])  # proportions (your eval_proportion)
    axC = fig.add_subplot(outer[:, 3])  # histogram

    # Summary mean ± SEM overlay
    axA.plot(mean_mean_early_array, label='Early')
    axA.plot(mean_mean_late_array,  label='Late')
    axA.fill_between(tE, mean_mean_early_array - sem_early_array,
                          mean_mean_early_array + sem_early_array, alpha=0.2)
    axA.fill_between(tL, mean_mean_late_array  - sem_late_array,
                          mean_mean_late_array  + sem_late_array,  alpha=0.2)
    if group:
        axA.set_title(group)
    axA.legend(frameon=False)

    # Proportions panel (reuse your existing helper)
    # Make sure eval_proportion accepts an axes handle.
    eval_proportion(most_expressed_label_dict_animal_early,
                    most_expressed_label_dict_animal_late,
                    group="All Cells", ax=axB)

    # Histogram
    axC.hist(elbow_kmeans_array, bins='auto')
    axC.set_title("Early Elbow Num Clusters Distribution")
    axC.set_xlabel("Number of Clusters")
    axC.set_ylabel("Number of Cells")

    plt.show()

def plot_early_late_activity_light(
    most_expressed_label_dict_animal_early,
    most_expressed_label_dict_animal_late,
    ax_list = None,
    most_expressed=True, group=None):
    # ---------- compute means/sems once ----------
    mean_early_list = []
    mean_late_list  = []

    for cell in most_expressed_label_dict_animal_early:
        early_array = most_expressed_label_dict_animal_early[cell]["cluster_activity"]  # (clusters, time) or similar
        late_array  = most_expressed_label_dict_animal_late[cell]["cluster_activity"]

        mean_early = np.mean(early_array, axis=1)  # per-cluster mean trace (adjust if your axis differs)
        mean_late  = np.mean(late_array,  axis=1)

        mean_early_list.append(mean_early)
        mean_late_list.append(mean_late)

    mean_early_array = np.asarray(mean_early_list)
    mean_late_array  = np.asarray(mean_late_list)

    mean_mean_early_array = np.mean(mean_early_array, axis=0)
    mean_mean_late_array  = np.mean(mean_late_array,  axis=0)

    sem_early_array = sem(mean_early_array, axis=0, nan_policy='omit')
    sem_late_array  = sem(mean_late_array,  axis=0, nan_policy='omit')

    count = 0
    counter = ax_list[count]

    counter.plot(mean_mean_early_array, label='Early')
    counter.fill_between(range(len(mean_mean_early_array)), mean_mean_early_array - sem_early_array,
                          mean_mean_early_array + sem_early_array, alpha=0.2)
    counter.set_ylim(-0.5,1.5)
    
    counter.plot(mean_mean_late_array,  label='Late')
    counter.fill_between(range(len(mean_mean_late_array)), mean_mean_late_array  - sem_late_array,
                          mean_mean_late_array  + sem_late_array,  alpha=0.2)
    
    counter.set_ylim(-0.5,1.5)

    counter.legend(frameon=False)
    
    if most_expressed:
        counter.set_title(f"Most Expressed Trial Type \n {group}")
    else:
        counter.set_title(f"Least Expressed Trial Type \n {group}")

    count+=1
    counter = ax_list[count]

    # Proportions panel (reuse your existing helper)
    # Make sure eval_proportion accepts an axes handle.
    eval_proportion(most_expressed_label_dict_animal_early,
                    most_expressed_label_dict_animal_late,
                    group="All Cells", ax=counter)

    # # Histogram
    # axC.hist(elbow_kmeans_array, bins='auto')
    # axC.set_title("Early Elbow Num Clusters Distribution")
    # axC.set_xlabel("Number of Clusters")
    # axC.set_ylabel("Number of Cells")

    # plt.show()


def run(most_expressed):
    GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="EC_GLM", new_NDNF=False)


    GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)

    fixed_residual_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if 17 < idx < 31:
            fixed_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

    fixed_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(activity_dict_NDNF_newest):
        if 17 < idx < 31:
            fixed_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]


    
    mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
    cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)

        
    fixed_cell_NDNF_model_ranks20_contig_x00_cue = {20: {}}
    for idx, animal in enumerate(cell_NDNF_model_ranks20_contig_x00_cue[20]):
        if 17<idx<31:
            fixed_cell_NDNF_model_ranks20_contig_x00_cue[20][animal-18] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]


    cp_dict_NDNF = get_cp_dict(fixed_cell_NDNF_model_ranks20_contig_x00_cue)

    save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/pickle_every_slice_type_NDNF.pkl"

    with open (save_path, 'rb') as f:
        model_and_mse_dict_NDNF = pickle.load(f)
        print(save_path)

    model_20_NDNF_raw_0x0 = model_and_mse_dict_NDNF[20]['model']
    labels_cells_dict_all_K_NDNF_0x0_raw = get_labels_all_different_Ks_single(model_20_NDNF_raw_0x0, which_vectors=1)


    # activity_early_array, activity_array_late = get_activity_cut_learn(fixed_activity_dict_NDNF_newest, cp_dict_NDNF)
    # plot_clustered_data_learn(means_dict_cluster_0x0_raw, activity_early_array, activity_array_late, K=2, title="Raw NDNF Clustered Changepoint")


    cell_type_labels = labels_cells_dict_all_K_NDNF_0x0_raw[2]

    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/ndnf_cell_types/cell_types_labels.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(cell_type_labels, f)


    cells_group_0 = np.where(cell_type_labels==0)[0]
    cells_group_1 = np.where(cell_type_labels==1)[0]

    mse_dir ='/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_regkmean_reassign_x00_cell'
    real_final_NDNF_model_ranks20_regkmean_reassign_x00_cell = get_model_data_per_cell(mse_dir)

    

    
    trial_type_data = real_final_NDNF_model_ranks20_regkmean_reassign_x00_cell[20]
    fixed_TT_data = {}
    for idx, animal in enumerate(trial_type_data):
        if 17 < idx < 31:
            fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]


    TT_list, NDNF_activity_list, cp_list_NDNF = get_lists_out_of_dicts(fixed_TT_data, fixed_residual_activity_dict_NDNF_newest, cp_dict_NDNF)



    most_expressed_label_dict_animal_early, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", most_expressed=most_expressed)
    most_expressed_label_dict_animal_late, elbow_kmeans_array = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", most_expressed=most_expressed)


    most_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=most_expressed)
    most_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=most_expressed)


    most_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=True)
    most_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=True)



    most_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=True)
    most_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=True)


    least_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=False)
    least_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=False)

    
    
    least_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=False)
    least_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=False)


    least_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=False)
    least_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=False)





    # fig, axs = plt.subplots(4,4)

    # if most_expressed:
    #     fig.suptitle("Most Expressed Cluster Across Trials")
    # else:
    #     fig.suptitle("Least Expressed Cluster Across Trials")

    fig, axs = plt.subplots(4, 4, figsize=(10, 12))  # or your preferred size

    colors_dict = {"Most_0":"orange",
                   "Least_0":"magenta",
                   "Most_1":"purple",
                   "Least_1":"red"}

    axs_list = [
        axs[0,0], axs[0,1],
        axs[1,0], axs[1,1],
        axs[2,0], axs[2,1],
        axs[3,0], axs[3,1],
    ]
    plot_no_learn_cell_types(
        most_expressed_label_dict_animal_all_group0,
        most_expressed_label_dict_animal_all_group1,
        least_expressed_label_dict_animal_all_group0,
        least_expressed_label_dict_animal_all_group1,
        group=None,
        most_expressed=most_expressed,
        axs_list=axs_list, color_dict=colors_dict
    )

    # axs_list2 = [axs[0,2], axs[0,3]]
    # plot_early_late_activity_light(
    # most_expressed_label_dict_animal_early_group0,
    # most_expressed_label_dict_animal_late_group0,
    # ax_list = axs_list2,
    # most_expressed=True,
    # group="Cell Type 0")

    eval_proportion(most_expressed_label_dict_animal_early_group0,
                    most_expressed_label_dict_animal_late_group0,
                    group="Cell Type 0 Most Expressed", ax=axs[0,2], color=colors_dict["Most_0"])


    # axs_list3 = [axs[1,2], axs[1,3]]
    # plot_early_late_activity_light(
    # least_expressed_label_dict_animal_early_group0,
    # least_expressed_label_dict_animal_late_group0,
    # ax_list = axs_list3,
    # most_expressed=False,
    # group="Cell Type 0")


    eval_proportion(least_expressed_label_dict_animal_early_group0,
                    least_expressed_label_dict_animal_late_group0,
                    group="Cell Type 0 Least Expressed", ax=axs[1,2], color=colors_dict["Least_0"])


    # axs_list4 = [axs[2,2], axs[2,3]]
    # plot_early_late_activity_light(
    # most_expressed_label_dict_animal_early_group1,
    # most_expressed_label_dict_animal_late_group1,
    # ax_list = axs_list4,
    # most_expressed=True,
    # group="Cell Type 1")


    eval_proportion(most_expressed_label_dict_animal_early_group1,
                    most_expressed_label_dict_animal_late_group1,
                    group="Cell Type 1 Most Expressed", ax=axs[2,2], color=colors_dict["Most_1"])


    # axs_list5 = [axs[3,2], axs[3,3]]
    # plot_early_late_activity_light(
    # least_expressed_label_dict_animal_early_group1,
    # least_expressed_label_dict_animal_late_group1,
    # ax_list = axs_list5,
    # most_expressed=False,
    # group="Cell Type 1")


    eval_proportion(least_expressed_label_dict_animal_early_group1,
                least_expressed_label_dict_animal_late_group1,
                group="Cell Type 1 Least Expressed", ax=axs[3,2], color=colors_dict["Least_1"])
    

    ax=axs[0,3]
    ax.hist(elbow_kmeans_array_group0_most, bins=[1.5,2.5,3.5,4.5,5.5], color=colors_dict["Most_0"])
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Most Expressed Cell Type 0")
    ax.set_xlabel("Num Clusters")
    ax.set_ylabel("Num Cells")

    ax=axs[1,3]
    ax.hist(elbow_kmeans_array_group0_least, bins=[1.5,2.5,3.5,4.5,5.5], color=colors_dict["Least_0"])
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Least Expressed Cell Type 0")
    ax.set_xlabel("Num Clusters")
    ax.set_ylabel("Num Cells")

    ax=axs[2,3]
    ax.hist(elbow_kmeans_array_group1_most, bins=[1.5,2.5,3.5,4.5,5.5], color=colors_dict["Most_1"])
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Most Expressed Cell Type 1")
    ax.set_xlabel("Num Clusters")
    ax.set_ylabel("Num Cells")

    ax=axs[3,3]
    ax.hist(elbow_kmeans_array_group1_least, bins=[1.5,2.5,3.5,4.5,5.5], color=colors_dict["Least_1"])
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Least Expressed Cell Type 1")
    ax.set_xlabel("Num Clusters")
    ax.set_ylabel("Num Cells")


    plt.tight_layout()
    plt.show()




    # plot_no_learn_cell_types(most_expressed_label_dict_animal_all_group0, most_expressed_label_dict_animal_all_group1, elbow_kmeans_array, group=None, most_expressed=most_expressed, axs_list=[axs[0,0], axs[0,1], axs[1,0], axs[1,1], axs[2,0], axs[2,1], axs[3,0], axs[3,1]])

    # plot_early_late_activity(most_expressed_label_dict_animal_early_group0, most_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0, group="Cell Type 0")

    # plot_early_late_activity(most_expressed_label_dict_animal_early_group1, most_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1, group="Cell Type 1")

    # plt.tight_layout()
    # plt.show()

    
@click.command()
@click.option(
    '--most-expressed/--no-most-expressed',
    default=True,
    help="Use the 'most expressed' scanning logic."
)
def cli(most_expressed):
    run(most_expressed)

if __name__ == "__main__":
    cli()

