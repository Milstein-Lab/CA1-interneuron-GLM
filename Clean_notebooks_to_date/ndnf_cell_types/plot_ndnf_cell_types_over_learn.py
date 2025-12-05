import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import slicetca

from scipy.spatial.distance import mahalanobis

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


from matplotlib.lines import Line2D



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

    print(f"len(TT_list){len(TT_list)} len(activity_list){len(activity_list)} len(cp_list_NDNF){len(cp_list_NDNF)}")
    
    if np.any(to_include) == None:
        indices_to_include = np.arange(len(TT_list))
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



def eval_proportion(title_fs, most_expressed_label_dict_animal_early, most_expressed_label_dict_animal_late, group=None, ax=None, color=None):
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
    ax.set_xticklabels(['Early','Late'], fontsize=title_fs-1)
    ax.set_ylabel("Fraction of Trials", fontsize=title_fs-1)
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


def plot_no_learn_cell_types(title_fs,
    most_expressed_label_dict_animal_cluster_0_all,
    most_expressed_label_dict_animal_cluster_1_all,
    least_expressed_label_dict_animal_all_group0,
    least_expressed_label_dict_animal_all_group1,
    group=None,
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
    axs_list[0].set_title("Cell Type 0 \n Most Expressed", fontsize=title_fs)
    axs_list[2].set_title("Cell Type 0 \n Least Expressed", fontsize=title_fs)
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
    axs_list[1].set_title("Cell Type 1 \n Most Expressed", fontsize=title_fs)
    axs_list[3].set_title("Cell Type 1 \n Least Expressed", fontsize=title_fs)
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
    axs_list[0].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[2].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[0].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[2].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[0].set_ylim(-1.5, 4)
    axs_list[2].plot(mean_mean_0_array_least, linewidth=4, color=color_dict["Least_0"])
    axs_list[2].set_ylim(-1.5, 4)

    axs_list[1].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[3].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[1].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[3].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[1].plot(mean_mean_1_array_most, linewidth=4, color=color_dict["Most_1"])
    axs_list[1].set_ylim(-1.5, 4)
    axs_list[3].plot(mean_mean_1_array_least, linewidth=4, color=color_dict["Least_1"])
    axs_list[3].set_ylim(-1.5, 4)

    ax = axs_list[4]
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Position Bins", fontsize=title_fs-1)
    ax.set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    ax.plot(mean_mean_0_array_most, label="Most Expressed Trial Type", color=color_dict["Most_0"])
    ax.fill_between(range(len(mean_mean_0_array_most)), mean_mean_0_array_most - sem_0_most, mean_mean_0_array_most + sem_0_most, alpha=0.2, color=color_dict["Most_0"])
    ax.plot(mean_mean_0_array_least, label="Least Expressed Trial Type", color=color_dict["Least_0"])
    ax.fill_between(range(len(mean_mean_0_array_least)), mean_mean_0_array_least - sem_0_least, mean_mean_0_array_least + sem_0_least, alpha=0.2, color=color_dict["Least_0"])
    ax.legend(fontsize=title_fs-2)

    ax = axs_list[5]
    ax.set_xlabel("Position Bins", fontsize=title_fs-1)
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.plot(mean_mean_1_array_most, label="Most Expressed Trial Type", color=color_dict["Most_1"])
    ax.fill_between(range(len(mean_mean_1_array_most)), mean_mean_1_array_most - sem_1_most, mean_mean_1_array_most + sem_1_most, alpha=0.2, color=color_dict["Most_1"])
    ax.plot(mean_mean_1_array_least, label="Least Expressed Trial Type", color=color_dict["Least_1"])
    ax.fill_between(range(len(mean_mean_1_array_least)), mean_mean_1_array_least - sem_1_least, mean_mean_1_array_least + sem_1_least, alpha=0.2, color=color_dict["Least_1"])
    ax.set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    ax.legend(fontsize=title_fs-2)
    
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
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Fraction of Trials",fontsize=title_fs-1); ax.set_ylabel("Number of Cells",fontsize=title_fs-1)
    ax.legend(frameon=False,fontsize=title_fs-1)

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
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.set_xlabel("Fraction of Trials",fontsize=title_fs-1); ax.set_ylabel("Number of Cells",fontsize=title_fs-1)
    ax.legend(frameon=False,fontsize=title_fs-1)

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
    eval_proportion(title_fs, most_expressed_label_dict_animal_early,
                    most_expressed_label_dict_animal_late,
                    group="All Cells", ax=axB)

    # Histogram
    axC.hist(elbow_kmeans_array, bins='auto')
    axC.set_title("Early Elbow Num Clusters Distribution")
    axC.set_xlabel("Number of Clusters")
    axC.set_ylabel("Number of Cells")

    plt.show()

# def plot_early_late_activity_light(
#     most_expressed_label_dict_animal_early,
#     most_expressed_label_dict_animal_late,
#     ax_list = None,
#     most_expressed=True, group=None):
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

#     count = 0
#     counter = ax_list[count]

#     counter.plot(mean_mean_early_array, label='Early')
#     counter.fill_between(range(len(mean_mean_early_array)), mean_mean_early_array - sem_early_array,
#                           mean_mean_early_array + sem_early_array, alpha=0.2)
#     counter.set_ylim(-0.5,1.5)
    
#     counter.plot(mean_mean_late_array,  label='Late')
#     counter.fill_between(range(len(mean_mean_late_array)), mean_mean_late_array  - sem_late_array,
#                           mean_mean_late_array  + sem_late_array,  alpha=0.2)
    
#     counter.set_ylim(-0.5,1.5)

#     counter.legend(frameon=False)
    
#     if most_expressed:
#         counter.set_title(f"Most Expressed Trial Type \n {group}")
#     else:
#         counter.set_title(f"Least Expressed Trial Type \n {group}")

#     count+=1
#     counter = ax_list[count]

#     # Proportions panel (reuse your existing helper)
#     # Make sure eval_proportion accepts an axes handle.
#     eval_proportion(title_fs, most_expressed_label_dict_animal_early,
#                     most_expressed_label_dict_animal_late,
#                     group="All Cells", ax=counter)


def eval_proportion_two_groups(title_fs,
    early_dict_g0,
    late_dict_g0,
    early_dict_g1,
    late_dict_g1,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=("C0", "C1"),
    ax=None,
    title="Fraction of trials", Most=None, 
):
    """
    Plot early vs late fraction for two cell types on the same axis.

    For each group:
      - compute mean ± SEM for Early and Late
      - plot a line connecting Early→Late means
      - add vertical SEM errorbars at Early and Late

    No individual cell points are shown.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    # ---- helper to extract early/late arrays from dict ----
    def dict_to_arrays(d_early, d_late):
        early_vals = []
        late_vals  = []
        for cell in d_early:
            early_vals.append(d_early[cell]["fraction"])
            late_vals.append(d_late[cell]["fraction"])
        early_vals = np.array(early_vals)
        late_vals  = np.array(late_vals)
        return early_vals, late_vals

    # ---- Group 0 ----
    early0, late0 = dict_to_arrays(early_dict_g0, late_dict_g0)
    mean_e0 = early0.mean()
    mean_l0 = late0.mean()
    sem_e0  = early0.std(ddof=1) / np.sqrt(len(early0))
    sem_l0  = late0.std(ddof=1) / np.sqrt(len(late0))

    # optional: paired t-test for group 0
    t0, p0 = ttest_rel(early0, late0)

    # ---- Group 1 ----
    early1, late1 = dict_to_arrays(early_dict_g1, late_dict_g1)
    mean_e1 = early1.mean()
    mean_l1 = late1.mean()
    sem_e1  = early1.std(ddof=1) / np.sqrt(len(early1))
    sem_l1  = late1.std(ddof=1) / np.sqrt(len(late1))

    # optional: paired t-test for group 1
    t1, p1 = ttest_rel(early1, late1)

    # x positions: 0 = Early, 1 = Late
    x_early = 0.1
    x_late  = 0.9

    # ---- plot Group 0 ----
    ax.plot(
        [x_early, x_late],
        [mean_e0, mean_l0],
        color=colors[0],
        marker='o',
        linewidth=2,
        label=f"{group_labels[0]} (p={p0:.3f})"
    )
    ax.errorbar(
        x_early, mean_e0, yerr=sem_e0,
        fmt='none', ecolor=colors[0], elinewidth=1.5, capsize=4, linestyle='--'
    )
    ax.errorbar(
        x_late, mean_l0, yerr=sem_l0,
        fmt='none', ecolor=colors[0], elinewidth=1.5, capsize=4, linestyle='--'
    )

    # ---- plot Group 1 ----
    ax.plot(
        [x_early, x_late],
        [mean_e1, mean_l1],
        color=colors[1],
        marker='o',
        linewidth=2,
        label=f"{group_labels[1]} (p={p1:.3f})"
    )
    ax.errorbar(
        x_early, mean_e1, yerr=sem_e1,
        fmt='none', ecolor=colors[1], elinewidth=1.5, capsize=4, linestyle='--'
    )
    ax.errorbar(
        x_late, mean_l1, yerr=sem_l1,
        fmt='none', ecolor=colors[1], elinewidth=1.5, capsize=4, linestyle='--'
    )

    # ---- cosmetics ----
    ax.set_xticks([0.1, 0.9])
    ax.set_xlim([0., 1.])
    ax.set_xticklabels(['Early', 'Late'],fontsize=title_fs-1)
    ax.set_ylabel("Fraction of Trials",fontsize=title_fs-1)
    if Most:
        ax.set_ylim(0.5,0.75)
    else:
        ax.set_ylim(0.05,0.3)
    ax.set_title(title, fontsize=title_fs)
    ax.legend(frameon=False, fontsize=5)


    return {
        "group0": {"early": early0, "late": late0, "p": p0},
        "group1": {"early": early1, "late": late1, "p": p1}
    }



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


def plot_LDA_hist(title_fs, group_1_array, title=None, color_list=None, ax=None):
    X = group_1_array            # (n_samples, n_cells) = (5000, 59)
    n_samples, n_cells = X.shape
    n_pos_bins = 50
    n_trials   = n_samples // n_pos_bins
    assert n_samples == n_pos_bins * n_trials

    print(f"n_cells = {n_cells}, n_trials = {n_trials}, n_pos_bins = {n_pos_bins}")

    # per-sample pos index within trial: 0..49 repeated over trials
    pos_idx   = np.tile(np.arange(n_pos_bins), n_trials)

    # -------------------------
    # 1) BINARY LABELS: start vs end of track
    #    0: bins 0–24 (first half)
    #    1: bins 25–49 (second half)
    # -------------------------
    start_end_labels = (pos_idx >= 25).astype(int)   # 0 = start, 1 = end

    # -------------------------
    # TRIAL QUINTILES for learning (early vs late 1/5 of trials)
    # -------------------------
    trial_idx = np.repeat(np.arange(n_trials), n_pos_bins)  # 0..n_trials-1 per sample
    trials_per_quint = n_trials // 5  # assume divisible by 5

    trial_quint = trial_idx // trials_per_quint  # 0..4
    early_trials_mask = (trial_quint == 0)
    late_trials_mask  = (trial_quint == 4)
    mid_trials_mask   = ~(early_trials_mask | late_trials_mask)

    # -------------------------
    # 2) z-score across samples for each cell
    # -------------------------
    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # -------------------------
    # 3) LDA with 2 classes (start vs end)
    #    With 2 classes, there is only 1 discriminant axis.
    # -------------------------
    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd") #eigen
    X_lda1 = lda.fit_transform(X_z, start_end_labels).ravel()   # shape (n_samples,)

    print("LDA explained variance ratio (start vs end):", lda.explained_variance_ratio_)


    bins = 40

    ax.hist(X_lda1[early_trials_mask], bins=bins, alpha=0.5,
             density=True, label="Early Trials", color=color_list[0])
    ax.hist(X_lda1[late_trials_mask], bins=bins, alpha=0.5,
             density=True, label="Late Trials", color=color_list[1])

    ax.set_xlabel("LDA 1 (start vs end axis)", fontsize=title_fs-1)
    ax.set_ylabel("density", fontsize=title_fs-1)
    ax.set_title(f"{title} LDA1 (Pre- vs Post-Reward Bins) \n Early and Late Learn Trials", fontsize=title_fs)
    ax.legend(fontsize=title_fs-3)



def plot_LDA_hist_compare_types(
        title_fs,
        group0_array,   # shape (n_samples, n_cells0)
        group1_array,   # shape (n_samples, n_cells1)
        title="Cell-type LDA comparison",
        color_dict=None,
        ax=None):

    """
    Compare Early-vs-Late LDA for Cell Type 0 and Cell Type 1.
    Uses pos-bin samples (same granularity as your original method).

    Colors needed:
        color_dict = {
            "g0_early": "blue",
            "g0_late":  "lightblue",
            "g1_early": "red",
            "g1_late":  "salmon"
        }
    """

    # default colors if none passed
    if color_dict is None:
        color_dict = {
            "g0_early": "blue",
            "g0_late":  "cyan",
            "g1_early": "red",
            "g1_late":  "orange",
        }

    # prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = ax.figure

    # -----------------------------------
    # Helper to compute LDA1 for one group
    # -----------------------------------
    def get_LDA1_for_group(X):
        n_samples, n_cells = X.shape
        n_pos_bins = 50
        n_trials   = n_samples // n_pos_bins
        assert n_samples == n_pos_bins * n_trials

        # reshape to (trials, pos_bins, cells)
        X_3d = X.reshape(n_trials, n_pos_bins, n_cells)

        # ------------------------------------
        # 1) Collapse to TRIAL-LEVEL vectors: mean over pos bins
        # ------------------------------------
        X_trials = X_3d.mean(axis=1)           # shape: (n_trials, n_cells)

        # ------------------------------------
        # 2) Trial quintiles: now on trials, not samples
        # ------------------------------------
        trials_per_quint = n_trials // 5
        trial_quint = np.arange(n_trials) // trials_per_quint  # shape (n_trials,)

        early_mask = (trial_quint == 0)       # first quintile of trials
        late_mask  = (trial_quint == 4)       # last quintile of trials

        # labels PER TRIAL: 0 = early, 1 = late
        y_trials = np.zeros(n_trials, dtype=int)
        y_trials[late_mask] = 1

        # ------------------------------------
        # 3) z-score across trials (per cell)
        # ------------------------------------
        X_z = (X_trials - X_trials.mean(axis=0)) / (X_trials.std(axis=0) + 1e-8)

        # ------------------------------------
        # 4) LDA on trial-level data
        # ------------------------------------
        lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
        X_lda1_trials = lda.fit_transform(X_z, y_trials).ravel()   # shape (n_trials,)

        return X_lda1_trials, early_mask, late_mask, lda.explained_variance_ratio_


    lda0, early0, late0, var0 = get_LDA1_for_group(group0_array)
    lda1, early1, late1, var1 = get_LDA1_for_group(group1_array)

    print("Group 0 (cell type 0) LDA variance:", var0)
    print("Group 1 (cell type 1) LDA variance:", var1)

    bins = 5

    ax.hist(lda0[early0], bins=bins, alpha=0.45, density=True,
            color=color_dict["g0_early"], label="Type 0 — Early")

    ax.hist(lda0[late0], bins=bins, alpha=0.45, density=True,
            color=color_dict["g0_late"], label="Type 0 — Late")

    ax.hist(lda1[early1], bins=bins, alpha=0.45, density=True,
            color=color_dict["g1_early"], label="Type 1 — Early")

    ax.hist(lda1[late1], bins=bins, alpha=0.45, density=True,
            color=color_dict["g1_late"], label="Type 1 — Late")

    ax.set_xlabel("LDA1 (Early ↔ Late learning axis)", fontsize=title_fs-1)
    ax.set_ylabel("Density", fontsize=title_fs-1)
    ax.set_title(f"{title}\nComparison of Cell Types on Learning Axis", fontsize=title_fs)
    # ax.set_xlim(-2.5, 2.5)
    ax.legend(fontsize=title_fs-3)

    return lda0, lda1, early0, late0, early1, late1


    


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


def plot_LDA_hist_compare_types_shared(
        title_fs,
        group0_array,   # shape (n_samples, n_cells0) = (n_trials * n_pos_bins, n_cells0)
        group1_array,   # shape (n_samples, n_cells1) = (n_trials * n_pos_bins, n_cells1)
        title="NDNF Cell Type Comparison",
        color_dict=None,
        ax=None):
    """
    Early-vs-Late LDA using a *shared* discriminant axis for both cell types.

    - One LDA is fit on trial-averaged activity of BOTH cell types concatenated.
    - Then we decompose the LDA weight vector into contributions from:
        - Type 0 features
        - Type 1 features
    - For each trial we compute:
        - LDA score from Type 0 alone
        - LDA score from Type 1 alone
      using the *same* discriminant axis.
    - We then plot 4 histograms (Type0/Type1 × Early/Late) on the same x-axis.

    Returns
    -------
    fig, ax, lda0_scores, lda1_scores, early_mask, late_mask
        Scores can be used for ANOVA.
    """

    # default colors if none passed
    if color_dict is None:
        color_dict = {
            "g0_early": "royalblue",
            "g0_late":  "lightblue",
            "g1_early": "crimson",
            "g1_late":  "pink",
        }

    # -----------------------------
    n_samples0, n_cells0 = group0_array.shape
    n_samples1, n_cells1 = group1_array.shape
    assert n_samples0 == n_samples1, "Both groups must have same #samples (trials*pos_bins)"

    n_pos_bins = 50
    n_trials   = n_samples0 // n_pos_bins
    assert n_samples0 == n_pos_bins * n_trials, "n_samples must equal n_trials * n_pos_bins"

    # reshape to (trials, pos_bins, cells)
    X0_3d = group0_array.reshape(n_trials, n_pos_bins, n_cells0)
    X1_3d = group1_array.reshape(n_trials, n_pos_bins, n_cells1)

    # trial-averaged activity: (n_trials, n_cells)
    X0_trials = X0_3d.mean(axis=1)
    X1_trials = X1_3d.mean(axis=1)

    # -----------------------------
    # Trial-level early/late labels
    # -----------------------------
    trials_per_quint = n_trials // 5
    trial_idx = np.arange(n_trials)
    trial_quint = trial_idx // trials_per_quint  # 0..4

    early_mask = (trial_quint == 0)
    late_mask  = (trial_quint == 4)

    y_trials = np.zeros(n_trials, dtype=int)
    y_trials[late_mask] = 1           # 0 = early, 1 = late

    # -----------------------------
    # Build combined feature matrix
    # -----------------------------
    X_comb_trials = np.concatenate([X0_trials, X1_trials], axis=1)  # (n_trials, n_cells0 + n_cells1)

    # z-score using shared mean/std
    mean_comb = X_comb_trials.mean(axis=0)
    std_comb  = X_comb_trials.std(axis=0) + 1e-8
    X_comb_z  = (X_comb_trials - mean_comb) / std_comb

    # -----------------------------
    # Fit ONE LDA on combined data
    # -----------------------------
    lda = LinearDiscriminantAnalysis(n_components=1, solver="svd")
    lda.fit(X_comb_z, y_trials)
    print("Shared LDA explained variance ratio:", lda.explained_variance_ratio_)

    # LDA weight vector in combined feature space
    # (n_features, 1) -> (n_features,)
    w = lda.scalings_[:, 0]

    # split weights into type 0 and type 1 parts
    w0 = w[:n_cells0]          # weights for Type 0 cells
    w1 = w[n_cells0:]          # weights for Type 1 cells

    # -----------------------------
    # Compute per-type LDA scores
    # using same z-scales and weights
    # -----------------------------
    # z-score each type using SAME combined mean/std slices
    X0_z = (X0_trials - mean_comb[:n_cells0]) / std_comb[:n_cells0]
    X1_z = (X1_trials - mean_comb[n_cells0:]) / std_comb[n_cells0:]

    # trial-level scores per type
    lda0_scores = X0_z.dot(w0)    # shape (n_trials,)
    lda1_scores = X1_z.dot(w1)    # shape (n_trials,)

    # -----------------------------
    # Plot 4 histograms on same axis
    # -----------------------------
    bins = 10

    # ax.hist(lda0_scores[early_mask], bins=bins, alpha=0.45, density=True,
    #         color=color_dict["g0_early"], label="Type 0 — Early")

    # ax.hist(lda0_scores[late_mask], bins=bins, alpha=0.45, density=True,
    #         color=color_dict["g0_late"], label="Type 0 — Late")

    # ax.hist(lda1_scores[early_mask], bins=bins, alpha=0.45, density=True,
    #         color=color_dict["g1_early"], label="Type 1 — Early")

    # ax.hist(lda1_scores[late_mask], bins=bins, alpha=0.45, density=True,
    #         color=color_dict["g1_late"], label="Type 1 — Late")

    # ax.set_xlabel("LDA1 (Early ↔ Late learning axis)", fontsize=title_fs - 1)
    # ax.set_ylabel("Density", fontsize=title_fs - 1)
    # ax.set_title(f"{title}\nComparison of Cell Types on Shared Learning Axis",
    #              fontsize=title_fs)
    # ax.legend(fontsize=title_fs - 3)

    return lda0_scores, lda1_scores, early_mask, late_mask



def plot_peak_trough_histograms_wrap(clusters_dict_NDNF_early, use_argmax=True, title=None, ylim=None, ax=None, color=None, alpha=None):
    all_means_list = []

    for animal in clusters_dict_NDNF_early:
        for cell in clusters_dict_NDNF_early[animal]:
            for i in range(len(clusters_dict_NDNF_early[animal][cell])):
                data = clusters_dict_NDNF_early[animal][cell][i]
                # if np.mean(data != 0):
                mean_data = np.mean(data, axis=0)
                if use_argmax:
                    all_means_list.append(np.argmax(mean_data))
                    ax.set_title(f"Position of Peak Mean Activity in Cluster {title}", fontsize=10)
                else:
                    all_means_list.append(np.argmin(mean_data))
                    ax.set_title(f"Position of Trough Mean Activity in Cluster {title}", fontsize=10)

    wrap_count = 0
    else_count = 0
    
    for i in range(len(all_means_list)-1):
        if all_means_list[i] > 45 and all_means_list[i+1] < 45:
            wrap_count+=1
            early_peak = all_means_list[i] 
            late_peak = all_means_list[i+1] 
            if np.random.rand() > 0.5:
                all_means_list[i] = late_peak
            else:
                all_means_list[i] = early_peak
            all_means_list[i+1] = np.nan
        else:
            else_count+=1

    print(f"wrap_count {wrap_count}, else_count, {else_count}")
    
    if use_argmax:
        color='forestgreen'
    else:
        color='darkred'

    ax.hist(all_means_list, bins=50, alpha=alpha, color=color)
    ax.set_ylabel("Number of Clusters")

    ax.set_xlabel("Position Bins")
    ax.set_ylim(0,ylim)

    return all_means_list


def plot_butterfly_hist(
    title_fs,
    argmax_list_early_0, argmax_list_late_0,
    argmin_list_early_0, argmin_list_late_0,
    ax=None, colors_list=None, title=None):

    bins = np.arange(0, 51)  # 50 position bins

    # ---- ARGMAX histograms (top) ----
    ax.hist(np.array(argmax_list_early_0),
            bins=bins, alpha=0.4,
            color=colors_list[0])

    ax.hist(np.array(argmax_list_late_0),
            bins=bins, alpha=0.4,
            color=colors_list[1])

    # ---- ARGMIN histograms (bottom, mirrored) ----
    weights_early_min = -np.ones_like(argmin_list_early_0, dtype=float)
    weights_late_min  = -np.ones_like(argmin_list_late_0, dtype=float)

    ax.hist(np.array(argmin_list_early_0),
            bins=bins, alpha=0.4, weights=weights_early_min,
            color=colors_list[2])

    ax.hist(np.array(argmin_list_late_0),
            bins=bins, alpha=0.4, weights=weights_late_min,
            color=colors_list[3])

    # Zero line
    ax.axhline(0, color="k", linewidth=1)

    # Axis labels & title
    ax.set_xlabel("Position bin", fontsize=title_fs-1)
    ax.set_ylabel("Cluster Count", fontsize=title_fs-1)
    ax.set_title(f"{title} Cluster Count per Pos Bin", fontsize=title_fs)

    # ---- Symmetric y-limits and custom ticks (25 → 0 → 25 style) ----
    y_min, y_max = ax.get_ylim()
    max_val = max(abs(y_min), abs(y_max))
    ax.set_ylim(-max_val, max_val)

    # Hard-code symmetric ticks (e.g., -25..0..25, labeled as 25..0..25)
    n_ticks = 5  # per side
    pos_ticks = np.linspace(0, max_val, n_ticks+1)     # 0..26
    neg_ticks = -pos_ticks[1:][::-1]              # -26..-small
    ticks = np.concatenate([neg_ticks, [0.0], pos_ticks[1:]])
    tick_labels = [f"{int(abs(t))}" for t in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # ---- Separate legends: upper right for Max, lower right for Min ----
    # Proxy handles for legend (so we don't depend on hist's internal patches)
    handles_max = [
        Line2D([0], [0], color=colors_list[0], lw=3, label="Max Early"),
        Line2D([0], [0], color=colors_list[1], lw=3, label="Max Late"),
    ]
    legend_max = ax.legend(handles=handles_max,
                        loc="upper right", fontsize=title_fs-3,
                        title="Max Loc")
    ax.add_artist(legend_max)  # keep this one when adding second legend

    handles_min = [
        Line2D([0], [0], color=colors_list[2], lw=3, linestyle="--", label="Min Early"),
        Line2D([0], [0], color=colors_list[3], lw=3, linestyle="--", label="Min Late"),
    ]
    legend_min = ax.legend(handles=handles_min,
                        loc="lower right", fontsize=title_fs-3,
                        title="Min Loc")

    return ax



    

def get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True):

    trial_type_activity_list_all_cells = []

    argmax_list = []
    argmin_list = []

    argmax_amp_list = [[] for _ in range(50)]
    argmin_amp_list = [[] for _ in range(50)]


    for idx, cell in enumerate(cells_group_0):
        num_clusters_chosen = elbow_kmeans_array_group0_most[idx]
        labels = TT_list[cell]['labels_dict'][f'clusters_chosen_{int(num_clusters_chosen)}']
        # tt_cluster_indices = TT_list[cell]['indices_for_cluster_number'][f'clusters_chosen_{int(num_clusters_chosen)}']
        trial_type_activity_list = []

        cp_list = cp_list_NDNF[cell]
        
        if use_early:
            cp = cp_list[0]
        else:
            cp = cp_list[1]

        unique_labels = np.unique(labels)
        # for tt_cluster in range(len(tt_cluster_indices)):
        for tt_cluster in range(len(unique_labels)):
            data_for_cell = NDNF_activity_list[cell]
            # trial_indices = tt_cluster_indices[tt_cluster]
            trial_indices = np.where(labels==tt_cluster)[0]

            if use_early:
                valid_trial_indices = np.where(trial_indices<cp)[0]
            else:
                valid_trial_indices = np.where(trial_indices>cp)[0]

            data_slice_trail_type = data_for_cell[:,valid_trial_indices]
            trial_type_activity_list.append(data_slice_trail_type)
            mean_data_slice_trail_type= np.mean(data_slice_trail_type, axis=1)
            max_loc = np.argmax(mean_data_slice_trail_type)
            min_loc = np.argmin(mean_data_slice_trail_type)
            argmax_amp_list[max_loc].append(mean_data_slice_trail_type[max_loc])
            argmin_amp_list[min_loc].append(mean_data_slice_trail_type[min_loc])
            argmax_list.append(max_loc)
            argmin_list.append(min_loc)
        trial_type_activity_list_all_cells.append(trial_type_activity_list)

    return argmax_list, argmin_list, argmax_amp_list, argmin_amp_list
    
def rebin_means_sems(means, sems, bins_per_group=10):
    """
    Take per-pos means & sems (length 50) and collapse into coarser bins.
    Returns x positions (bin centers), rebinned means, rebinned sems.
    """
    means = np.asarray(means, float)
    sems  = np.asarray(sems, float)

    n_old = len(means)
    assert n_old % bins_per_group == 0, "50 must be divisible by bins_per_group"
    n_groups = n_old // bins_per_group

    x_coarse = []
    m_coarse = []
    s_coarse = []

    for g in range(n_groups):
        sl = slice(g * bins_per_group, (g + 1) * bins_per_group)
        m_block = np.nanmean(means[sl])
        s_block = np.nanmean(sems[sl])   # approx; for viz only

        x_center = (g * bins_per_group + (g + 1) * bins_per_group - 1) / 2.0

        x_coarse.append(x_center)
        m_coarse.append(m_block)
        s_coarse.append(s_block)

    return np.array(x_coarse), np.array(m_coarse), np.array(s_coarse)




def run(use_fixed_track, use_new_data):
    # GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="SSTindivsomata_GLM", new_NDNF=False)
    # GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="EC_GLM", new_NDNF=False)

    if use_new_data:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E0A1B1_251107", new_NDNF=True, use_final=True)
        first_an_idx = 14
        last_an_idx = 29

    else:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)
        first_an_idx = 17
        last_an_idx = 31

    title_fs = 8

    fixed_residual_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                fixed_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]
        else:
            if idx > last_an_idx-1:
                fixed_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

    fixed_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(activity_dict_NDNF_newest):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                fixed_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]
        else:
            if idx > last_an_idx-1:
                fixed_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]


    if use_new_data:

        if use_fixed_track:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_fixed_contig_models_NDNF.pkl', 'rb') as f:
                contig_dict = pickle.load(f)

        else:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_cued_contig_models_NDNF.pkl', 'rb') as f:
                contig_dict = pickle.load(f)


    else:
        mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/'
        cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)


        contig_dict = {20: {}}
        for idx, animal in enumerate(cell_NDNF_model_ranks20_contig_x00_cue[20]):
            if use_fixed_track:
                if first_an_idx<idx<last_an_idx:
                    contig_dict[20][animal-(first_an_idx+1)] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]
            else:
                if idx>last_an_idx-1:
                    contig_dict[20][animal-(last_an_idx+1)] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]

    
    
    cp_dict_NDNF = get_cp_dict(contig_dict)




    ################## if we are building the cell type labels from scratch ########################

            # save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/pickle_every_slice_type_NDNF.pkl"

            # with open (save_path, 'rb') as f:
            #     model_and_mse_dict_NDNF = pickle.load(f)
            #     print(save_path)

            # model_20_NDNF_raw_0x0 = model_and_mse_dict_NDNF[20]['model']
            # labels_cells_dict_all_K_NDNF_0x0_raw = get_labels_all_different_Ks_single(model_20_NDNF_raw_0x0, which_vectors=1)


            # cell_type_labels = labels_cells_dict_all_K_NDNF_0x0_raw[2]

            # cells_group_0 = np.where(cell_type_labels==0)[0]
            # cells_group_1 = np.where(cell_type_labels==1)[0]

    ##############################################################################################



    if use_new_data:
        if use_fixed_track:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/model_20_NDNF_resid_0x0_fixed_new_data.pkl', 'rb') as f:
                sliceTCA_model = pickle.load(f)
        else:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/model_20_NDNF_resid_0x0_cue_new_data.pkl', 'rb') as f:
                sliceTCA_model = pickle.load(f)
    else:
        if use_fixed_track:

            save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/pickle_every_slice_type_NDNF.pkl"

            with open (save_path, 'rb') as f:
                model = pickle.load(f)
                sliceTCA_model = model[20]['model']
        else:
            save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/model_20_NDNF_resid_0x0_cue.pkl"
            with open (save_path, 'rb') as f:
                sliceTCA_model = pickle.load(f)



    labels_cells_dict_all_K_NDNF = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

    cell_type_labels = labels_cells_dict_all_K_NDNF[2]

    cells_group_0 = np.where(cell_type_labels==0)[0]
    cells_group_1 = np.where(cell_type_labels==1)[0]

    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(fixed_residual_activity_dict_NDNF_newest, cells_group_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(fixed_residual_activity_dict_NDNF_newest, cells_group_1, neg_sel=False, trial_av=True, norm="min_max")

    animal_average_selectivity_dict_NDNF_0_list = []
    for animal in animal_average_selectivity_dict_NDNF_0:
        for cell in animal_average_selectivity_dict_NDNF_0[animal]:
            animal_average_selectivity_dict_NDNF_0_list.append(animal_average_selectivity_dict_NDNF_0[animal][cell])

    animal_average_selectivity_dict_NDNF_1_list = []
    for animal in animal_average_selectivity_dict_NDNF_1:
        for cell in animal_average_selectivity_dict_NDNF_1[animal]:
            animal_average_selectivity_dict_NDNF_1_list.append(animal_average_selectivity_dict_NDNF_1[animal][cell])

    if np.mean(animal_average_selectivity_dict_NDNF_0_list) > np.mean(animal_average_selectivity_dict_NDNF_1_list):
        inverted_labels = 1 - cell_type_labels
        cells_group_0 = np.where(inverted_labels==0)[0]
        cells_group_1 = np.where(inverted_labels==1)[0]
        cell_type_labels = inverted_labels

    
    # if use_new_data:
    #     if use_fixed_track:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_new.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_new.pkl')
    #     else:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_new.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_new.pkl')
    # else:
    #     if use_fixed_track:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_old.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_old.pkl')
    #     else:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_old.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_old.pkl')





    if use_new_data:

        fixed_TT_data = {}

        if use_fixed_track:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_fixed_reassign_models_NDNF.pkl', 'rb') as f:
                reassigned_dict = pickle.load(f)

            trial_type_data = reassigned_dict[20]
            for idx, animal in enumerate(trial_type_data):
                fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

        else:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_cued_reassign_models_NDNF.pkl', 'rb') as f:
                reassigned_dict = pickle.load(f)

            trial_type_data = reassigned_dict[20]
            for idx, animal in enumerate(trial_type_data):
                fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    else:
        mse_dir ='/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_regkmean_reassign_x00_cell'
        reassigned_dict = get_model_data_per_cell(mse_dir)

    

        print(f"reassigned_dict.keys() {reassigned_dict.keys()}")
        trial_type_data = reassigned_dict[20]
        print(f"trial_type_data.keys() {trial_type_data.keys()}")
        fixed_TT_data = {}
        for idx, animal in enumerate(trial_type_data):
            if use_fixed_track:
                if first_an_idx < idx < last_an_idx:
                    fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]
            else:
                if idx > last_an_idx-1:
                    fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    print(f"cp_dict_NDNF {cp_dict_NDNF}")
            

    TT_list, NDNF_activity_list, cp_list_NDNF = get_lists_out_of_dicts(fixed_TT_data, fixed_residual_activity_dict_NDNF_newest, cp_dict_NDNF)





    # most_expressed_label_dict_animal_early, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", most_expressed=most_expressed)
    # most_expressed_label_dict_animal_late, elbow_kmeans_array = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", most_expressed=most_expressed)


    most_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=True)
    most_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=True)


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




    # print(f"len(elbow_kmeans_array_group0_most) {len(elbow_kmeans_array_group0_most)}")
    # print(f"len(TT_list) {TT_list[0]['labels_dict'].keys()}")
    # print(f"len(TT_list) {TT_list[0]['indices_for_cluster_number']['clusters_chosen_2']}")

    
    argmax_list_early_0, argmin_list_early_0, argmax_amp_list_early_0, argmin_amp_list_early_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True)
    argmax_list_late_0, argmin_list_late_0, argmax_amp_list_late_0, argmin_amp_list_late_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=False)

    argmax_list_early_1, argmin_list_early_1, argmax_amp_list_early_1, argmin_amp_list_early_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=True)
    argmax_list_late_1, argmin_list_late_1, argmax_amp_list_late_1, argmin_amp_list_late_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=False)    

    def mean_and_sem(argmax_amp_list):
        means_list = []
        sems_list = []

        for i in range(len(argmax_amp_list)):
            pos_bin_vals = argmax_amp_list[i]
            if len(pos_bin_vals)>1:
                means_list.append(np.mean(pos_bin_vals))
                sems_list.append(sem(pos_bin_vals))
            elif len(pos_bin_vals)==1:
                means_list.append(pos_bin_vals[0])
                sems_list.append(0.0)
            else:
                means_list.append(np.nan)
                sems_list.append(np.nan)

        return means_list, sems_list

    means_list_early_0, sems_list_early_0 = mean_and_sem(argmax_amp_list_early_0)
    means_list_late_0, sems_list_late_0 = mean_and_sem(argmax_amp_list_late_0)
    means_list_early_1, sems_list_early_1 = mean_and_sem(argmax_amp_list_early_1)
    means_list_late_1, sems_list_late_1 = mean_and_sem(argmax_amp_list_late_1)

    means_list_early_0min, sems_list_early_0min = mean_and_sem(argmin_amp_list_early_0)
    means_list_late_0min, sems_list_late_0min = mean_and_sem(argmin_amp_list_late_0)
    means_list_early_1min, sems_list_early_1min = mean_and_sem(argmin_amp_list_early_1)
    means_list_late_1min, sems_list_late_1min = mean_and_sem(argmin_amp_list_late_1)


    
        # --- rebin to 10 coarse pos bins (5 original bins per coarse bin) ---
    x10_0_e_max, m10_0_e_max, s10_0_e_max = rebin_means_sems(means_list_early_0,  sems_list_early_0)
    _,         m10_0_l_max, s10_0_l_max   = rebin_means_sems(means_list_late_0,   sems_list_late_0)

    _,         m10_0_e_min, s10_0_e_min   = rebin_means_sems(means_list_early_0min, sems_list_early_0min)
    _,         m10_0_l_min, s10_0_l_min   = rebin_means_sems(means_list_late_0min,  sems_list_late_0min)

    x10_1_e_max, m10_1_e_max, s10_1_e_max = rebin_means_sems(means_list_early_1,  sems_list_early_1)
    _,         m10_1_l_max, s10_1_l_max   = rebin_means_sems(means_list_late_1,   sems_list_late_1)

    _,         m10_1_e_min, s10_1_e_min   = rebin_means_sems(means_list_early_1min, sems_list_early_1min)
    _,         m10_1_l_min, s10_1_l_min   = rebin_means_sems(means_list_late_1min,  sems_list_late_1min)





    fig, axs = plt.subplots(4, 4, figsize=(10, 12))  



    axs[1,3].errorbar(x10_0_e_max, m10_0_e_max, yerr=s10_0_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_max, yerr=s10_0_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_e_min, yerr=s10_0_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_min, yerr=s10_0_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[1,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    axs[1,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    axs[1,3].set_title("Cell Type 0 Max/Min Amplitude", fontsize=title_fs)
    axs[1,3].legend(fontsize=title_fs-2)

   
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_max, yerr=s10_1_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_max, yerr=s10_1_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_min, yerr=s10_1_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_min, yerr=s10_1_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[3,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    axs[3,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    axs[3,3].set_title("Cell Type 1 Max/Min Amplitude", fontsize=title_fs)
    axs[3,3].legend(fontsize=title_fs-2)

 


    plot_butterfly_hist(title_fs, argmax_list_early_0, argmax_list_late_0, argmin_list_early_0, argmin_list_late_0, ax=axs[0,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 0")
    plot_butterfly_hist(title_fs, argmax_list_early_1, argmax_list_late_1, argmin_list_early_1, argmin_list_late_1, ax=axs[2,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 1")



    if use_new_data:
        if use_fixed_track:
            fig.suptitle("New Data Fixed Track")
        else:
            fig.suptitle("New Data Cued Track")
    else:
        if use_fixed_track:
            fig.suptitle("Old Data Fixed Track")
        else:
            fig.suptitle("Old Data Cued Track")


    colors_dict = {"Most_0":"orange",
                   "Least_0":"magenta",
                   "Most_1":"purple",
                   "Least_1":"red"}

    axs_list = [
        axs[0,0], axs[0,1],
        axs[1,0], axs[1,1],
        axs[2,0], axs[2,1],
        axs[3,0], axs[3,1],]
    
    plot_no_learn_cell_types(title_fs,
        most_expressed_label_dict_animal_all_group0,
        most_expressed_label_dict_animal_all_group1,
        least_expressed_label_dict_animal_all_group0,
        least_expressed_label_dict_animal_all_group1,
        group=None,axs_list=axs_list, color_dict=colors_dict)

    eval_proportion_two_groups(title_fs,
    most_expressed_label_dict_animal_early_group0,
    most_expressed_label_dict_animal_late_group0,
    most_expressed_label_dict_animal_early_group1,
    most_expressed_label_dict_animal_late_group1,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=(colors_dict["Most_0"], colors_dict["Most_1"]),
    ax=axs[0,2],
    title="Most expressed: Early vs Late", Most=True)

    eval_proportion_two_groups(title_fs,
    least_expressed_label_dict_animal_early_group0,
    least_expressed_label_dict_animal_late_group0,
    least_expressed_label_dict_animal_early_group1,
    least_expressed_label_dict_animal_late_group1,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=(colors_dict["Least_0"], colors_dict["Least_1"]),
    ax=axs[1,2],
    title="Least expressed: Early vs Late", Most=False)

    ax=axs[2,2]
    ax.hist(elbow_kmeans_array_group0_most, bins=[1.5,2.5,3.5,4.5,5.5], color='red')
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Number of Trial Type Clusters", fontsize=title_fs-1)
    ax.set_ylabel("Number of Cells", fontsize=title_fs-1)

    ax=axs[3,2]
    ax.hist(elbow_kmeans_array_group1_most, bins=[1.5,2.5,3.5,4.5,5.5], color='purple')
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.set_xlabel("Number of Trial Type Clusters", fontsize=title_fs-1)
    ax.set_ylabel("Number of Cells", fontsize=title_fs-1)


    fixed_residual_list = []

    to_plot_list = []

    min_t = 10000

    for animal in fixed_residual_activity_dict_NDNF_newest:
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            data = fixed_residual_activity_dict_NDNF_newest[animal][cell]
            data_flat = data.flatten()
            if data.shape[1] < min_t:
                min_t = data.shape[1]

    for animal in fixed_residual_activity_dict_NDNF_newest:
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            data_trunc = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,:min_t]
            to_plot_list.append(np.mean(data_trunc, axis=1))
            fixed_residual_list.append(data_trunc.flatten())

    fixed_residual_array = np.array(fixed_residual_list)
    fixed_residual_array_correct_shape = fixed_residual_array.T

    group_1_array = fixed_residual_array_correct_shape[:,cells_group_1]

    group_0_array = fixed_residual_array_correct_shape[:,cells_group_0]


    # plot_LDA1_LDA2_state_space_prepost(title_fs, group_0_array, title="Cell Type 0", ax=axs[2,3])

    # plot_LDA1_LDA2_state_space_prepost(title_fs, group_1_array, title="Cell Type 1", ax=axs[3,3])


    # plot_LDA_hist(title_fs, group_0_array, title="Cell Type 0", color_list=["blue", "orange"], ax=axs[0,3])
    # plot_LDA_hist(title_fs, group_1_array, title="Cell Type 1", color_list=["red", "green"], ax=axs[1,3])

    # lda0, lda1, early0, late0, early1, late1 = plot_LDA_hist_compare_types(title_fs,group0_array=group_0_array,group1_array=group_1_array, title="NDNF Cell Type Comparison",
    # color_dict={
    #     "g0_early": "blue",
    #     "g0_late": "green",
    #     "g1_early": "red",
    #     "g1_late": "magenta",},ax=axs[0,3])

    # print(f"len(lda0[early0]) {len(lda0[early0])} len(lda0[late0]) {len(lda0[late0])}") #, late0, early1, late1


    # def build_clusters_dict():






    plt.tight_layout()
    plt.show()

@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
@click.option('--use_new_data/--use_old_data', default=True, help="Use the Final NDNF data")

def cli(use_fixed_track, use_new_data):
    run(use_fixed_track, use_new_data)

if __name__ == "__main__":
    cli()

