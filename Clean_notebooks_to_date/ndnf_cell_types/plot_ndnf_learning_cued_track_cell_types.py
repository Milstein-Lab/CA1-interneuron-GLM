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

from scipy.spatial.distance import mahalanobis

plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9



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

def plot_clustered_data_learn(title_fs, labels_list, activity_early_array, activity_array_late, K=2, title="", cue=False, ax_list=None): #means_dict_cluster_0x0_raw, 
    # data_good = means_dict_cluster_0x0_raw[K]["labels_loc_dict"]

    color_list_dict = [{"Early":"purple", "Late":"magenta"},{"Early":"red", "Late":"orange"}] 

    for idx in range(len(labels_list)):
        # labels = data_good[i]
        labels = labels_list[idx]
        n=len(labels)
        sliced_early = activity_early_array[labels,:]
        mean_sliced_early = np.mean(sliced_early, axis=0)
        sem_sliced_early = sem(sliced_early, axis=0)

        # sliced_data_early_dict[i] = sliced_early
        sliced_late = activity_array_late[labels,:]
        mean_sliced_late = np.mean(sliced_late, axis=0)
        sem_sliced_late = sem(sliced_late, axis=0)

        # sliced_data_late_dict[i] = sliced_late
        ax_list[idx].plot(mean_sliced_early, label='Early', color=color_list_dict[idx]["Early"])
        ax_list[idx].fill_between(range(len(mean_sliced_early)), mean_sliced_early-sem_sliced_early, mean_sliced_early+sem_sliced_early, alpha=0.2, color=color_list_dict[idx]["Early"])

        ax_list[idx].plot(mean_sliced_late, label="Late", color=color_list_dict[idx]["Late"])
        ax_list[idx].fill_between(range(len(mean_sliced_late)), mean_sliced_late-sem_sliced_late, mean_sliced_late+sem_sliced_late, alpha=0.2, color=color_list_dict[idx]["Late"])
        ax_list[idx].set_title(f"Cluster {idx} n={n}", fontsize=title_fs)
        if cue:
            ax_list[idx].axvline(10,linestyle='--', color='r', label="Cue")
        ax_list[idx].legend(fontsize=title_fs-2)
        ax_list[idx].set_ylabel("Z-Scored dF/F", fontsize=title_fs-1)
        ax_list[idx].set_xlabel("Position bins", fontsize=title_fs-1)
        ax_list[idx].set_ylim(-0.5, 1.0)


    # plt.tight_layout()
    # plt.show()

def plot_reconstructions(labels_list, fixed_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="", plot=False):


    synthetic_mean_array = {}
    means_dict_cluster= {}
    for num_clusters in range(len(labels_list)):
        data_truncated_array_NDNF, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
        labels = labels_list[num_clusters]
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



    data_truncated_array_NDNF, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
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


# def plot_mean_resid(residual_activity_dict_NDNF_newest, title, ax=None, plot=False):
#     cue_residual_activity_dict_NDNF_newest = {}
#     for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
#         if idx > 30:
#             cue_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

#     listy = []
#     for animal in cue_residual_activity_dict_NDNF_newest:
#         for cell in cue_residual_activity_dict_NDNF_newest[animal]:
#             listy.append(np.mean(cue_residual_activity_dict_NDNF_newest[animal][cell], axis=1))

#     good_array = np.array(listy)
#     means = np.mean(good_array, axis=0)
#     sems = sem(good_array, axis=0)

#     if plot:
#         ax.plot(means)
#         ax.axvline(10, linestyle="--", color='red', label='Cue')
#         ax.fill_between(range(len(means)), means+sems, means-sems, alpha=0.2)
#         ax.legend()
#         ax.set_title(title)

#     return cue_residual_activity_dict_NDNF_newest

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


# def plot_cluster_traces_by_animal(title_fs,
#     labels,
#     #means_dict_cluster,                 # output from plot_reconstructions (has cell_ids_per_cluster_dict)
#     fixed_activity_dict_NDNF_newest,    # to recompute TA traces
#     cells_per_animal_dict,              # {animal: [global_cell_ids...]}
#     K, 
#     ncol=1,                             # legend columns
#     ylim=(-1.1, 2.6), 
#     spacing=0.25,                       # horizontal spacing between subplots (wspace)
#     title_prefix="",
#     axs=None,                           # list/array of Axes (len >= #clusters) or None
#     legend="bottom",                    # "bottom", "right", "inside", or None
# color_list=None):
#     """
#     Plots trial-averaged (TA) traces per cluster, coloring each trace by its animal.
#     If `axs` is None, creates a new 1×(#clusters) figure (shared y). Otherwise draws into provided axes.
#     Legend shows animals with their colors and can be placed at 'bottom', 'right', 'inside', or disabled with None.
#     """

#     cell_to_animal = {}
#     for animal, cell_ids in cells_per_animal_dict.items():
#         for cid in cell_ids:
#             cell_to_animal[int(cid)] = animal

#     data = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
#     ta = data.mean(axis=2)                                                   # (cells, pos)
#     _, n_pos = ta.shape

#     uniq = np.unique(labels)
#     n_panels = len(uniq)
#     labels_list = []
#     for i in uniq:
#         cells_data = np.where()
#         labels_list.append(cells_data)


#     # --- color map per animal ---
#     animals = sorted(set(cell_to_animal.values()))
#     cmap = plt.get_cmap("tab20", max(1, len(animals)))
#     animal_to_color = {a: cmap(i % cmap.N) for i, a in enumerate(animals)}

#     # --- axes handling ---
#     created_fig = False
#     if axs is None:
#         fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 6), sharey=True)
#         if n_panels == 1:
#             axs = np.array([axs])
#         created_fig = True
#     else:
#         if not isinstance(axs, (list, tuple, np.ndarray)):
#             axs = np.array([axs])
#         else:
#             axs = np.array(axs).ravel()
#         if len(axs) < n_panels:
#             raise ValueError(f"Provided axs has length {len(axs)} but need at least {n_panels} axes.")
#         fig = axs[0].figure

#     # adjust spacing if requested
#     if spacing is not None:
#         fig.subplots_adjust(wspace=spacing)

#     # --- draw each cluster ---
#     for i in range(len(uniq)):
#         idx = np.asarray(clust_idx_dict[lab], dtype=int)
#         if idx.size == 0:
#             axs[i].set_title(f"Cluster {i} (n=0)")
#             axs[i].set_xlim(0, n_pos - 1)
#             axs[i].set_ylim(*ylim)
#             continue

#         traces = ta[idx]  # (n_k, n_pos)

#         # plot each cell trace colored by its animal
#         for j, cid in enumerate(idx):
#             a = cell_to_animal.get(int(cid), "unknown")
#             color = animal_to_color.get(a, (0.5, 0.5, 0.5, 0.6))
#             ax.plot(traces[j, :], lw=1.0, alpha=0.7, color=color)

#         # overlay mean ± SEM (neutral color)
#         m = traces.mean(axis=0)
#         s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
#         ax.plot(m, lw=2.0, color=color_list[i], zorder=5)
#         # ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color=color_list[i], zorder=4)

#         ax.set_title(f"{title_prefix} Cluster {lab} (n={len(idx)})".strip(), fontsize=title_fs)
#         ax.set_xlabel("Position bins", fontsize=title_fs-1)
#         ax.set_ylim(*ylim)

#     axs[0].set_ylabel("Z-scored dF/F", fontsize=title_fs-1)
#     axs[1].set_ylabel("Z-scored dF/F", fontsize=title_fs-1)

#     return fig, axs

def plot_cluster_traces_by_animal(
    title_fs,
    labels,                          # 1D array, length = n_cells
    fixed_activity_dict_NDNF_newest, # to recompute TA traces
    cells_per_animal_dict,           # {animal: [global_cell_ids...]}
    K=None,                          # not really needed anymore
    ncol=1,
    ylim=(-1.1, 2.6),
    spacing=0.25,
    title_prefix="",
    axs=None,
    legend="bottom",
    color_list=None,
):
    """
    Plots trial-averaged (TA) traces per label (0/1), coloring each trace by its animal.
    `labels` is a 1D array (n_cells,) with cluster IDs (e.g., 0 and 1).
    """

    cell_to_animal = {}
    for animal, cell_ids in cells_per_animal_dict.items():
        for cid in cell_ids:
            cell_to_animal[int(cid)] = animal

    data, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
    ta = data.mean(axis=2)                                                   # (cells, pos)
    n_cells, n_pos = ta.shape

    labels = np.asarray(labels)
    if labels.shape[0] != n_cells:
        raise ValueError(f"labels has length {labels.shape[0]} but ta has {n_cells} cells")

    uniq = np.unique(labels)
    n_panels = len(uniq)
    label_to_col = {lab: j for j, lab in enumerate(uniq)}

    animals = sorted(set(cell_to_animal.values()))
    cmap = plt.get_cmap("tab20", max(1, len(animals)))
    animal_to_color = {a: cmap(i % cmap.N) for i, a in enumerate(animals)}

    for i, lab in enumerate(uniq):
        ax = axs[label_to_col[lab]]

        idx = np.where(labels == lab)[0]
        if idx.size == 0:
            ax.set_title(f"{title_prefix} Cluster {lab} (n=0)", fontsize=title_fs)
            ax.set_xlim(0, n_pos - 1)
            ax.set_ylim(*ylim)
            continue

        traces = ta[idx]  # (n_k, n_pos)

        for j, cid in enumerate(idx):
            a = cell_to_animal.get(int(cid), "unknown")
            color = animal_to_color.get(a, (0.5, 0.5, 0.5, 0.6))
            ax.plot(traces[j, :], lw=1.0, alpha=0.7, color=color)

        m = traces.mean(axis=0)
        s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
        if color_list is not None:
            main_color = color_list[i]
        else:
            main_color = "k"
        ax.plot(m, lw=2.0, color=main_color, zorder=5)
    
        ax.set_title(f"{title_prefix} Cluster {lab} (n={len(idx)})".strip(), fontsize=title_fs)
        ax.set_xlabel("Position bins", fontsize=title_fs - 1)
        ax.set_ylim(*ylim)

    axs[0].set_ylabel("Z-scored dF/F", fontsize=title_fs - 1)
    axs[1].set_ylabel("Z-scored dF/F", fontsize=title_fs - 1)




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
    title_fs,
    labels_list,             # list of arrays: one array of global cell IDs per cluster
    cells_per_animal_dict,   # {animal: [global_cell_ids...]}
    K,
    title_prefix="",
    show_percent_labels=False,
    ax=None,
    legend="outside",        # "outside", "inside", or None
    legend_ncol=1,
):
    """
    Stacked bar plot of animal composition per cluster.

    labels_list: list of 1D arrays
        labels_list[k] = array of global cell IDs belonging to cluster k.
    """

    # --- map cell -> animal ---
    cell_to_animal = {}
    for animal, cell_ids in cells_per_animal_dict.items():
        for cid in cell_ids:
            cell_to_animal[int(cid)] = animal

    n_clusters = len(labels_list)

    # --- animals & colors ---
    animals = sorted(set(cell_to_animal.values()))
    cmap = plt.get_cmap("tab20", len(animals))
    animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

    # --- counts[cluster, animal] ---
    counts = np.zeros((n_clusters, len(animals)), dtype=int)

    for r, cell_ids in enumerate(labels_list):
        cell_ids = np.asarray(cell_ids, dtype=int)
        for cid in cell_ids:
            a = cell_to_animal.get(int(cid), None)
            if a is not None:
                j = animals.index(a)
                counts[r, j] += 1

    totals = counts.sum(axis=1)              # total cells per cluster
    x = np.arange(n_clusters)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))
    else:
        fig = ax.figure

    # --- stacked bars ---
    bottom = np.zeros_like(totals, dtype=float)
    for j, a in enumerate(animals):
        ax.bar(
            x,
            counts[:, j],
            bottom=bottom,
            width=0.8,
            color=animal_to_color[a],
            label=str(a),
        )
        bottom += counts[:, j]

    # --- x labels & title ---
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{i}\n(n={totals[i]})" for i in range(n_clusters)],
        fontsize=title_fs - 1,
    )
    ax.set_ylabel("# cells", fontsize=title_fs - 1)
    ax.set_title(f"{title_prefix}", fontsize=title_fs)

    # --- optional percent labels ---
    if show_percent_labels:
        with np.errstate(divide="ignore", invalid="ignore"):
            props = counts / totals[:, None]
            props[np.isnan(props)] = 0.0
        for i in range(n_clusters):
            cum = 0.0
            for j in range(len(animals)):
                h = counts[i, j]
                if h > 0 and props[i, j] >= 0.07:
                    ax.text(
                        x[i],
                        cum + h / 2.0,
                        f"{props[i, j]*100:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
                cum += h

    # --- legend ---
    if legend == "outside":
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            ncol=legend_ncol,
            fontsize=title_fs - 2,
        )
    elif legend == "inside":
        ax.legend(loc="upper right", frameon=False)

    return fig, ax



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

    return data_truncated_array, min_val

def get_mean_behav_factor_per_cell(residual_activity_dict, factors_dict, min_num_trials, factor:str):

    data_list = []
    for animal in residual_activity_dict:
        for cell in residual_activity_dict[animal]:
            data_list.append(factors_dict[animal][factor][:, :min_num_trials])

    return data_list


def plot_lick_vel_data_clust(title_fs, labels_list, mean_factor_list, use_vel=False, title=None, ax=None, color_list=None):


    for i in range(len(labels_list)):

        vels_sliced_list = []

        labels = labels_list[i]

        for cell in labels:
            vels_sliced = mean_factor_list[cell]
            vels_sliced_list.append(vels_sliced)

        vel_array = np.array(vels_sliced_list)

        ta_vel_array = np.mean(vel_array, axis=2)

        mean_over_cells = np.mean(ta_vel_array, axis=0)
        sem_over_cells = sem(ta_vel_array, axis=0)

        ax.plot(mean_over_cells, label=f"Cluster {i} n={vel_array.shape[0]}", color=color_list[i])
        ax.fill_between(range(len(mean_over_cells)), mean_over_cells-sem_over_cells, mean_over_cells+sem_over_cells, alpha=0.2, color=color_list[i])
        # plt.title(f"Cluster {clust}")
        ax.set_xlabel("Position Bins", fontsize=title_fs-1)
        if use_vel:
            ax.set_ylabel(f"Velocity (meters/sec)", fontsize=title_fs-1)
        else:
            ax.set_ylabel(f"Normalized Lick Rate", fontsize=title_fs-1)
    ax.set_title(title, fontsize=title_fs)
    ax.legend(fontsize=title_fs-2)




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



def plot_the_CDF_early_late(title_fs, binned_data_0_early, binned_data_1_early, binned_data_0_late, binned_data_1_late, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
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

    ax.set_ylabel("Percentile of Cells", fontsize=title_fs-1)
    ax.set_xlabel("Selectivity", fontsize=title_fs-1)
    ax.set_title(title, fontsize=title_fs)
    ax.legend(fontsize=6)


def plot_the_CDF_celltypes(title_fs, binned_data_0, binned_data_1, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
    mean_0, sem_0 = get_mean_sem_lists(binned_data_0)
    mean_1, sem_1 = get_mean_sem_lists(binned_data_1)

    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5

    # Plot: horizontal bars (x=selectivity, y=percentile)
    # plt.figure(figsize=(7, 6))

    ax.errorbar(mean_0, percentiles, xerr=sem_0, fmt='o-', label='Cell Type 0', color='purple', capsize=3)
    ax.errorbar(mean_1, percentiles, xerr=sem_1, fmt='o-', label='Cell Type 1', color='red', capsize=3)

    ax.set_ylabel("Percentile of Cells", fontsize=title_fs-1)
    ax.set_xlabel("Selectivity", fontsize=title_fs-1)
    ax.set_title(title, fontsize=title_fs)
    ax.legend(fontsize=title_fs-2)



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


def plot_selectivity_over_trials(title_fs, group_0_selectivity, group_1_selectivity, color_list=None, ax=None):

    mean_selectivity_0 = np.mean(group_0_selectivity, axis=0)
    sem_selectivity_0 = sem(group_0_selectivity, axis=0)

    mean_selectivity_1 = np.mean(group_1_selectivity, axis=0)
    sem_selectivity_1 = sem(group_1_selectivity, axis=0)

    x = np.arange(1, 11) 

    ax.plot(x, mean_selectivity_0, color=color_list[0], label='Cell Type 0')
    ax.fill_between(x, mean_selectivity_0 - sem_selectivity_0, mean_selectivity_0 + sem_selectivity_0, alpha=0.2, color=color_list[0])
    ax.plot(x, mean_selectivity_1, color=color_list[1], label='Cell Type 1')
    ax.fill_between(x, mean_selectivity_1 - sem_selectivity_1, mean_selectivity_1 + sem_selectivity_1, alpha=0.2, color=color_list[1])
    ax.set_xticks(ticks=x, labels=[f"{int(p)}" for p in np.linspace(0, 100, 10)])
    ax.set_xlabel("Percentile of Trials (%)", fontsize=title_fs-1)
    ax.set_ylabel("Average Selectivity Across Cells", fontsize=title_fs-1)
    ax.set_title("Selectivity Across Trials", fontsize=title_fs)
    ax.legend(fontsize=title_fs-2)


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


def plot_cell_clusters_LDA_space(title_fs, which_vectors, model_20_NDNF_resid_0x0_cue, labels, ax=None, color_list=None):

    w1 = model_20_NDNF_resid_0x0_cue.vectors[which_vectors][0]
    f1 = model_20_NDNF_resid_0x0_cue.vectors[which_vectors][1]

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
        ax.scatter(X_2d[m, 0], X_2d[m, 1], s=40, alpha=0.6, color=color_list[i], label=f"C{k} (n={m.sum()})")

    ax.set_xlabel("LDA Component 1", fontsize=title_fs-1)
    ax.set_ylabel("LDA Component 2", fontsize=title_fs-1)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1, fontsize=title_fs-2)
    # fig.subplots_adjust(right=0.78)

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
            

def plot_eml_data(title_fs, animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=None, color_list=None):

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

    ax.errorbar(range(len(means0)), means0, yerr=sems0, label="Cell Type 0", color=color_list[0], marker="o", capsize=3)
    ax.errorbar(range(len(means1)), means1, yerr=sems1, label="Cell Type 1", color=color_list[1], marker="o", capsize=3)
    ax.set_xticks(np.arange(3), ["Early", "Middle", "Late"])
    ax.set_ylabel("Average Selectivity Across Cells", fontsize=title_fs-1)
    ax.set_xlabel("Contiguous K-Means Learning Stage", fontsize=title_fs-1)
    ax.set_title("Selectivity by Unsupervised Learning Stage", fontsize=title_fs)
    ax.legend(fontsize=title_fs-2)



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
    
    
    # d_pre_1 = mahal_group_distance(X1, mask_early_pre1, mask_late_pre1)
    # d_post_1 = mahal_group_distance(X1, mask_early_post1, mask_late_post1)


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

    ax.set_title(f"{title} \n Mahal. Dist. Early vs Late Pre={d_pre_0:.2f}, Post={d_post_0:.2f}", fontsize=title_fs-1)


    # return mask_early_pre, mask_early_post, mask_late_pre, mask_late_post, LDA1, LDA2



def run(use_fixed_track, use_new_data):
    
    

    if use_new_data:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E0A1B1_251107", new_NDNF=True, use_final=True)
        first_an_idx = 14
        last_an_idx = 29

    else:
        GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)
        first_an_idx = 17
        last_an_idx = 31


    mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
    cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)
    
    cued_cell_NDNF_model_ranks20_contig_x00_cue = {20: {}}
    for idx, animal in enumerate(cell_NDNF_model_ranks20_contig_x00_cue[20]):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                cued_cell_NDNF_model_ranks20_contig_x00_cue[20][animal-(first_an_idx+1)] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]
        else:
            if idx>last_an_idx-1:
                cued_cell_NDNF_model_ranks20_contig_x00_cue[20][animal-last_an_idx] = cell_NDNF_model_ranks20_contig_x00_cue[20][animal]

    cued_factors_dict_NDNF_newest = {}
    for idx, animal in enumerate(factors_dict_NDNF_newest):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                cued_factors_dict_NDNF_newest[f"animal_{idx+1}"] = factors_dict_NDNF_newest[animal]
        else:
            if idx > last_an_idx-1:
                cued_factors_dict_NDNF_newest[f"animal_{idx+1}"] = factors_dict_NDNF_newest[animal]
        

    cued_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(activity_dict_NDNF_newest):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                cued_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]
        else:
            if idx > last_an_idx-1:
                cued_activity_dict_NDNF_newest[f"animal_{idx+1}"] = activity_dict_NDNF_newest[animal]
        
        
    cue_residual_activity_dict_NDNF_newest = {}
    for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
        if use_fixed_track:
            if first_an_idx < idx < last_an_idx:
                cue_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]
        else:
            if idx > last_an_idx-1:
                cue_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]



    # cue_residual_activity_dict_NDNF_newest = cued_activity_dict_NDNF_newest
            

    data_truncated_array_NDNF, min_num_trials = get_truncated_to_min_data_array(cued_activity_dict_NDNF_newest)


    r_dict_vel = get_r_list(cued_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, data_truncated_array_NDNF, data_to_corr="Velocity")
    r_dict_licks = get_r_list(cued_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, data_truncated_array_NDNF, data_to_corr="Licks")

################ old method of getting the cp_dict #########
    # mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
    # cell_NDNF_model_ranks20_contig_x00_cue = get_model_data_per_cell(mse_dir)

    # cued_cp_dict_NDNF = get_cp_dict(cued_cell_NDNF_model_ranks20_contig_x00_cue)
##########################################################


    if use_new_data:

        if use_fixed_track:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_fixed_contig_models_NDNF.pkl', 'rb') as f:
                contig_data_dict = pickle.load(f)

        else:
            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_cued_contig_models_NDNF.pkl', 'rb') as f:
                contig_data_dict = pickle.load(f)
            

        

    else:
        mse_dir = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_contig_x00_cell'
        data_dict = get_model_data_per_cell(mse_dir)


        contig_data_dict = {20: {}}
        for idx, animal in enumerate(data_dict[20]):
            if use_fixed_track:
                if first_an_idx<idx<last_an_idx:
                    
                    contig_data_dict[20][animal-(first_an_idx+1)] = data_dict[20][animal]
            else:
                if idx>last_an_idx-1:
                    contig_data_dict[20][animal-(last_an_idx)] = data_dict[20][animal]


    print(f"contig_data_dict[20].keys() {contig_data_dict[20].keys()}")

    cp_dict_NDNF = get_cp_dict(contig_data_dict)

    print(f"cp_dict_NDNF.keys() {cp_dict_NDNF.keys()}")




##################### load in all the sliceTCA models ############################
    
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
                models_dict = pickle.load(f)
                sliceTCA_model = models_dict[20]['model']

        else:
            save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/model_20_NDNF_resid_0x0_cue.pkl"
            with open (save_path, 'rb') as f:
                sliceTCA_model = pickle.load(f)

    labels_cells_dict_all_K_NDNF = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

###################### old ##########################################

    # save_path="/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/model_20_NDNF_resid_0x0_cue.pkl"
    # with open(save_path, 'rb') as f:
    #     model_20_NDNF_resid_0x0_cue = pickle.load(f)
    #     print(save_path)

    # labels_cells_dict_all_K_NDNF_0x0_resid_cue = get_labels_all_different_Ks_single(model_20_NDNF_resid_0x0_cue, which_vectors=1)

###################### old ##########################################


    

    fig, axs = plt.subplots(4,4, figsize=(20,20))


    if use_fixed_track and use_new_data:
        fig.suptitle("Fixed Track Newest Data")
    elif use_fixed_track and not use_new_data:
        fig.suptitle("Fixed Track Old Data")
    elif not use_fixed_track and use_new_data:
        fig.suptitle("Cued Track Newest Data")
    else:
        fig.suptitle("Cued Track Old Data")

    title_fs=9


    # cue_residual_activity_dict_NDNF_newest = plot_mean_resid(residual_activity_dict_NDNF_newest, title="Residuals", ax=axs[0,0], plot=False)

    

    activity_early_array, activity_array_late = get_activity_cut_learn(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF)
    

    cells_per_animal_dict = get_cells_per_animal_dict(cue_residual_activity_dict_NDNF_newest)

    color_list=["purple", "red"]


    which_vectors=1
    labels = np.asarray(labels_cells_dict_all_K_NDNF[2])

    cells_list_0 = np.where(labels==0)[0]
    cells_list_1 = np.where(labels==1)[0]

    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(cue_residual_activity_dict_NDNF_newest, cells_list_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(cue_residual_activity_dict_NDNF_newest, cells_list_1, neg_sel=False, trial_av=True, norm="min_max")

    animal_average_selectivity_dict_NDNF_0_list = []
    for animal in animal_average_selectivity_dict_NDNF_0:
        for cell in animal_average_selectivity_dict_NDNF_0[animal]:
            animal_average_selectivity_dict_NDNF_0_list.append(animal_average_selectivity_dict_NDNF_0[animal][cell])

    animal_average_selectivity_dict_NDNF_1_list = []
    for animal in animal_average_selectivity_dict_NDNF_1:
        for cell in animal_average_selectivity_dict_NDNF_1[animal]:
            animal_average_selectivity_dict_NDNF_1_list.append(animal_average_selectivity_dict_NDNF_1[animal][cell])

    if np.mean(animal_average_selectivity_dict_NDNF_0_list) > np.mean(animal_average_selectivity_dict_NDNF_1_list):
        inverted_labels = 1 - labels
        cells_list_0 = np.where(inverted_labels==0)[0]
        cells_list_1 = np.where(inverted_labels==1)[0]
        labels = inverted_labels

    labels_list = [cells_list_0, cells_list_1]

    means_dict_cluster_0x0_cue_resid = plot_reconstructions(labels_list, cue_residual_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="NDNF 0x0 Cue Resid")

    plot_cluster_traces_by_animal(title_fs, labels,
                                cue_residual_activity_dict_NDNF_newest,
                                cells_per_animal_dict,
                                K=2,
                                ncol=5, spacing=0.2,
                                title_prefix="", axs=[axs[0,0],axs[0,1]], color_list = color_list)
    

    plot_cell_clusters_LDA_space(title_fs, which_vectors, sliceTCA_model, labels, ax=axs[0,2], color_list=color_list)        

    plot_cluster_animal_composition_stacked(title_fs,labels_list, cells_per_animal_dict, K=2,
                                            title_prefix="Fraction of Clusters Cells by Animal", show_percent_labels=True, ax=axs[1,0])

    mean_vel_list = get_mean_behav_factor_per_cell(cue_residual_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, min_num_trials, factor="Velocity")
    plot_lick_vel_data_clust(title_fs, labels_list, mean_vel_list, use_vel=True, title="Weighted Av. Animal Velocity", ax=axs[1,1], color_list=color_list)
    mean_lick_list = get_mean_behav_factor_per_cell(cue_residual_activity_dict_NDNF_newest, cued_factors_dict_NDNF_newest, min_num_trials, factor="Licks")
    plot_lick_vel_data_clust(title_fs, labels_list, mean_lick_list, use_vel=False, title="Weighted Av. Animal Lick", ax=axs[1,2], color_list=color_list)



    plot_clustered_data_learn(title_fs, labels_list, activity_early_array, activity_array_late, K=2, title="Cued Track Residuals NDNF Clustered Changepoint", ax_list=[axs[2,0],axs[2,1]])
    
    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(cue_residual_activity_dict_NDNF_newest, cells_list_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(cue_residual_activity_dict_NDNF_newest, cells_list_1, neg_sel=False, trial_av=True, norm="min_max")

    binned_data_NDNF_0 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0, n_bins=20)
    binned_data_NDNF_1 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1, n_bins=20)

    plot_the_CDF_celltypes(title_fs, binned_data_NDNF_0, binned_data_NDNF_1, title="Selectivity Distribution", n_bins = 20, ax=axs[2,2])

    animal_average_selectivity_dict_NDNF_0_early,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="early", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_early,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="early", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_late,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="late", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_late,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="late", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_middle,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="middle", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_middle,_ = get_selectivity_each_trial_early_late_cluster(cue_residual_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="middle", norm="min_max")

    plot_eml_data(title_fs, animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=axs[0,3], color_list=color_list)
    
    binned_data_NDNF_0_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_early, n_bins=20)
    binned_data_NDNF_1_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_early, n_bins=20)

    binned_data_NDNF_0_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_late, n_bins=20)
    binned_data_NDNF_1_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_late, n_bins=20)


    plot_the_CDF_early_late(title_fs, binned_data_NDNF_0_early, binned_data_NDNF_1_early, binned_data_NDNF_0_late, binned_data_NDNF_1_late, title="Selectivity Distribution Early Late Learn", n_bins = 20, ax=axs[2,3])



    percentile_slices_NDNF0 = get_percentlie_slices(animals_dict_data_NDNF_0)
    percentile_slices_NDNF1 = get_percentlie_slices(animals_dict_data_NDNF_1)
    

    all_cells_NDNF_0 = selectivity_from_percentile_slices(percentile_slices_NDNF0, norm='min_max', neg_sel=False)
    all_cells_NDNF_1 = selectivity_from_percentile_slices(percentile_slices_NDNF1, norm='min_max', neg_sel=False)

    plot_selectivity_over_trials(title_fs, all_cells_NDNF_0, all_cells_NDNF_1, color_list=color_list, ax=axs[1,3])




    fixed_residual_list = []

    to_plot_list = []

    min_t = 10000

    for animal in cue_residual_activity_dict_NDNF_newest:
        for cell in cue_residual_activity_dict_NDNF_newest[animal]:
            data = cue_residual_activity_dict_NDNF_newest[animal][cell]
            data_flat = data.flatten()
            if data.shape[1] < min_t:
                min_t = data.shape[1]

    for animal in cue_residual_activity_dict_NDNF_newest:
        for cell in cue_residual_activity_dict_NDNF_newest[animal]:
            data_trunc = cue_residual_activity_dict_NDNF_newest[animal][cell][:,:min_t]
            to_plot_list.append(np.mean(data_trunc, axis=1))
            fixed_residual_list.append(data_trunc.flatten())

    fixed_residual_array = np.array(fixed_residual_list)
    fixed_residual_array_correct_shape = fixed_residual_array.T

    group_1_array = fixed_residual_array_correct_shape[:,cells_list_1]

    group_0_array = fixed_residual_array_correct_shape[:,cells_list_0]


    plot_LDA1_LDA2_state_space_prepost(title_fs, group_0_array, title="Cell Type 0", ax=axs[3,2])

    plot_LDA1_LDA2_state_space_prepost(title_fs, group_1_array, title="Cell Type 1", ax=axs[3,3])


    plot_LDA_hist(title_fs, group_0_array, title="Cell Type 0", color_list=["blue", "orange"], ax=axs[3,0])
    plot_LDA_hist(title_fs, group_1_array, title="Cell Type 1", color_list=["red", "green"], ax=axs[3,1])







    fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=4.0)
    plt.show()



@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
@click.option('--use_new_data/--use_old_data', default=True, help="Use the Final NDNF data")

def cli(use_fixed_track, use_new_data):
    run(use_fixed_track, use_new_data)

if __name__ == "__main__":
    cli()
