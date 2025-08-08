import operator
import random
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt

import sys
from scipy.stats import sem
sys.path.append(r'C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM')

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *


from modelling_to_date_utils import *


import time



def get_means_list(residual_activity_dict_EC):
    means_list = []
    trials_list_wraparound = []

    clusters_dict_labelled_field_type_wraparound = {}

    count = 0

    for animal in residual_activity_dict_EC:
        cell_dict = {}
        for cell in residual_activity_dict_EC[animal]:

            wraparound_field = {}

            data = residual_activity_dict_EC[animal][cell]
            trial_list = []

            for trial in range(data.shape[0] - 1):  # avoid out-of-bounds on next trial
                current_data = data[:, trial]
                next_data = data[:, trial + 1]

                peak_current = np.argmax(current_data)
                peak_next = np.argmax(next_data)

                mean_current = np.mean(current_data)
                threshold_current = 1.5 * mean_current
                mean_next = np.mean(current_data)
                threshold_next = 1.5 * mean_next

                if peak_current > 45 and peak_next < 5 and peak_current > threshold_current and peak_next > threshold_next:
                    trial_list.append(trial)
                    means_list.append(np.mean(residual_activity_dict_EC[animal][cell][:, trial]))
                    trials_list_wraparound.append(residual_activity_dict_EC[animal][cell][:, trial])
                    means_list.append(np.mean(residual_activity_dict_EC[animal][cell][:, trial + 1]))
                    trials_list_wraparound.append(residual_activity_dict_EC[animal][cell][:, trial + 1])

            cell_dict[cell] = trial_list

        clusters_dict_labelled_field_type_wraparound[animal] = cell_dict

    return means_list

def get_clusters_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln="nothing"):

    """
    - takes in the contiguous cutpoints slice TCA model and the K-Means slice TCA model
    - use elbow kmeans to loop through every number of cluster (up to 8) to get the optimal number of clusters via reconstruction MSE of cluster average reconstruction vs real data for the cell
    - returns a dict where every cell's activity is seperated by its cluster via trial indices for each cluster
    - since we are seperating by trial indices we can ask whether the indices are within the early sliceTCA changepoint or in late and seperate the data by learning
    """

    clusters_dict_EC = {}

    for animal_num, animal in enumerate(residual_activity_dict_EC):
        clusters_cell_EC = {}
        for cell_num, cell in enumerate(residual_activity_dict_EC[animal]):
            #             cell_num = int(cell.split("_")[-1])  # e.g., from 'cell_3' -> 3
            #             elbow_kmeans = get_elbow_score_data(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=animal_num, cell=cell_num)
            # ✅ skip if cell is missing in model dict
            try:
                elbow_kmeans = get_elbow_score_data(
                    testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell,
                    animal=animal_num,
                    cell=cell_num
                )
            except KeyError:
                print(f"⚠️ Skipping animal {animal_num}, cell {cell_num} — not in model dict")
                continue

            #             elbow_kmeans = get_elbow_score_data(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=animal_num, cell=cell_num)
            clusters = elbow_kmeans + 1
            clusters_list = plot_per_cell_clustering_internals_single_cluster(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal_id=animal_num, cell_id=cell_num, num_clusters=clusters, plot=False, early_late_nothing=eln)
            clusters_cell_EC[cell] = clusters_list

        clusters_dict_EC[animal] = clusters_cell_EC

    return clusters_dict_EC

def get_clusters_dict_field_type(means_list, clusters_dict_EC, residual_activity_dict_EC, use_peak=True):

    '''
    -finds the argmax or argmin of each clusters' average activity from the cell dict of activity seperated into clusters and then labells that cluster as a before, near(reward), after or wraparound if the peak was near the end of the trial and there was a peak near the start (within 5 position bins) of the next trial
    '''

    rounded_means_set = set(round(m, 5) for m in means_list)

    # for animal in clusters_dict_EC:
    #     for cell in clusters_dict_EC[animal]:

    overall_animal_cluster_trial_averages = {}

    clusters_dict_labelled_field_type = {}

    for animal in clusters_dict_EC:
        cell_dict = {}
        cell_dict_labelled = {}
        for cell in clusters_dict_EC[animal]:
            cell_data = clusters_dict_EC[animal][cell]
            num_clusters = len(cell_data)
            before_field = {}
            near_field = {}
            after_field = {}
            noisy_field = {}
            wraparound_field = {}

            total_num_trials = residual_activity_dict_EC[animal][cell].shape[1]

            count = 0

            clusters_dict = {}
            for i in range(num_clusters):

                cluster_length = cell_data[i].shape[0]
                count += cluster_length

                trial_av = np.mean(cell_data[i], axis=0)
                if len(trial_av) > 0:
                    if use_peak:
                        peak = np.argmax(trial_av)
                        peak_amp = max(trial_av)
                        compare_op = operator.gt

                    else:
                        peak = np.argmin(trial_av)
                        peak_amp = min(trial_av)
                        compare_op = operator.lt

                    mean_trial_av = np.mean(trial_av)
                    #                     cutoff = mean_trial_av * 1.5

                    if use_peak:
                        cutoff = mean_trial_av * 0.8
                    else:
                        cutoff = mean_trial_av * 2.0  # Or whatever gets you a stricter min threshold

                    if compare_op(peak_amp, cutoff):
                        if peak <= 15:
                            filtered_trials = []
                            wraparound_trials = []

                            for trial in cell_data[i]:
                                trial_data = trial
                                mean_trial_data = np.mean(trial_data)

                                if round(mean_trial_data, 5) in rounded_means_set:
                                    wraparound_trials.append(trial_data)
                                else:
                                    filtered_trials.append(trial_data)

                            filtered_array = np.array(filtered_trials)
                            wraparound_array = np.array(wraparound_trials)

                            clusters_dict[i] = {f"before_field": filtered_array,
                                                "wraparound_field": wraparound_array}
                            before_field[i] = filtered_array
                            wraparound_field[i] = wraparound_array

                        elif 15 < peak < 35:
                            filtered_trials = []
                            wraparound_trials = []

                            for trial in cell_data[i]:
                                trial_data = trial
                                mean_trial_data = np.mean(trial_data)

                                if round(mean_trial_data, 5) in rounded_means_set:
                                    wraparound_trials.append(trial_data)
                                else:
                                    filtered_trials.append(trial_data)

                            filtered_array = np.array(filtered_trials)
                            wraparound_array = np.array(wraparound_trials)

                            clusters_dict[i] = {f"near_field": filtered_array,
                                                "wraparound_field": wraparound_array}

                            near_field[i] = filtered_array
                            wraparound_field[i] = wraparound_array

                        elif peak >= 35:
                            filtered_trials = []
                            wraparound_trials = []

                            for trial in cell_data[i]:
                                trial_data = trial
                                mean_trial_data = np.mean(trial_data)

                                mean_trial_data = np.mean(trial_data)
                                if round(mean_trial_data, 5) in rounded_means_set:
                                    wraparound_trials.append(trial_data)
                                else:
                                    filtered_trials.append(trial_data)

                            filtered_array = np.array(filtered_trials)
                            wraparound_array = np.array(wraparound_trials)

                            clusters_dict[i] = {f"after_field": filtered_array,
                                                "wraparound_field": wraparound_array}

                            after_field[i] = filtered_array
                            wraparound_field[i] = wraparound_array

                    else:
                        filtered_trials = []
                        wraparound_trials = []

                        for trial in cell_data[i]:
                            trial_data = trial
                            mean_trial_data = np.mean(trial_data)

                            if round(mean_trial_data, 5) in rounded_means_set:
                                wraparound_trials.append(trial_data)
                            else:
                                filtered_trials.append(trial_data)

                        if len(filtered_trials) > 0:
                            filtered_array = np.array(filtered_trials)
                            wraparound_array = np.array(wraparound_trials)

                            noisy_field[i] = filtered_array
                            wraparound_field[i] = wraparound_array
                            clusters_dict[i] = {"noisy_field": filtered_array,
                                                "wraparound_field": wraparound_array}

            #                         print(f"Potential noisy: {animal} cell {cell} cluster {i} with amp={peak_amp:.2f}, mean={mean_trial_av:.2f}, cutoff={cutoff:.2f}")

            #                         if len(filtered_trials) > 0:
            #                             filtered_array = np.array(filtered_trials)
            #                             noisy_field[i] = filtered_array
            #                             clusters_dict[i] = {"noisy_field": filtered_array}

            # #                         filtered_trials = []
            # #                         wraparound_trials = []

            # #                         for trial in cell_data[i]:
            # #                             trial_data = trial
            # #                             mean_trial_data = np.mean(trial_data)

            # #                             mean_trial_data = np.mean(trial_data)
            # #                             if round(mean_trial_data, 5) in rounded_means_set:
            # #                                 wraparound_trials.append(trial_data)
            # #                             else:
            # #                                 filtered_trials.append(trial_data)

            # #                         filtered_array = np.array(filtered_trials)
            # #                         wraparound_array = np.array(wraparound_trials)

            # #                         clusters_dict[i] = {f"noisy_field":filtered_array,
            # #                                            "wraparound_field": wraparound_array}

            #                             noisy_field[i] = filtered_array
            #                             wraparound_field[i] = wraparound_array

            cells_dict = {"before_field": before_field,
                          "near_field": near_field,
                          "after_field": after_field,
                          "noisy_field": noisy_field,
                          "wraparound_field": wraparound_field}

            cell_dict[cell] = cells_dict

            cell_dict_labelled[cell] = clusters_dict
            print(f" count {count} total {total_num_trials}")
        clusters_dict_labelled_field_type[animal] = cell_dict_labelled
        overall_animal_cluster_trial_averages[animal] = cell_dict

    return clusters_dict_labelled_field_type, overall_animal_cluster_trial_averages

def plot_per_cell_clustering_internals_single_cluster(cell_EC_model_ranks20_contig_x00, cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, residual_activity_dict_NDNF, animal_id=1, cell_id=1, num_clusters=4, plot=True, early_late_nothing="nothing"):
    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_NDNF, animal_TCA=False)

    tensor_list_by_animal_all_NDNF = []
    for animal in residual_activity_dict_NDNF:
        neural_data = ut.get_animal_neural_tensor(residual_activity_dict_NDNF, animal=animal)
        neural_data_tensor = torch.tensor(neural_data)
        # Normalize per cell
        for i in range(neural_data_tensor.shape[1]):
            cell = neural_data_tensor[:, i, :]
            min_val = cell.min()
            max_val = cell.max()
            neural_data_tensor[:, i, :] = (cell - min_val) / (max_val - min_val + 1e-8)
        tensor_list_by_animal_all_NDNF.append(neural_data_tensor)

    real_activity = tensor_list_by_animal_all_NDNF[animal_id][:, cell_id, :].detach().numpy()

    #     for culsters_chosen in cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["TCA_reconstructions_dict"]:

    TCA_reco = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["TCA_reconstructions_dict"][f"clusters_chosen_{num_clusters}"]

    animal_reco = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["Recon_by_cluster_av_dict"][f"clusters_chosen_{num_clusters}"]

    indices_dict = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["indices_for_cluster_number"][f"clusters_chosen_{num_clusters}"]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        im1 = (axs[0].imshow(real_activity, aspect='auto', vmin=0, vmax=1))
        axs[0].set_title("Raw Data")
        fig.colorbar(im1, ax=axs[0])
        im2 = axs[1].imshow(TCA_reco, aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im2, ax=axs[1])
        im3 = axs[2].imshow(animal_reco, aspect='auto', vmin=0, vmax=1)
        fig.colorbar(im3, ax=axs[2])
        axs[1].set_title("SliceTCA Reconstruction")
        axs[2].set_title("Cluster Average Reconstruction")
        plt.tight_layout()
        plt.show()

    num_clusters = len(indices_dict)
    clusters_list = []

    if plot:
        fig, axs = plt.subplots(2, num_clusters, figsize=(num_clusters * 4, 6), squeeze=False)

    for n in range(num_clusters):
        indices = indices_dict[n]

        if early_late_nothing == "nothing":
            indices_list = indices
        elif early_late_nothing == "early":
            indices_list = [i for i in indices if i < animal_first_changepoints_list[animal_id][cell_id]]
        elif early_late_nothing == "late":
            indices_list = [i for i in indices if i > animal_second_changepoints_list[animal_id][cell_id]]
        else:
            raise ValueError(f"Invalid value for early_late_nothing: {early_late_nothing}")

        cluster = real_activity[indices_list, :]
        clusters_list.append(cluster)

        #         cluster = real_activity[indices_list, :]  # trials x bins
        #         clusters_list.append(cluster)

        if plot:
            axs[0, n].imshow(cluster, aspect='auto', vmin=0, vmax=1)
            axs[0, n].set_title(f"Cluster {n}")
            axs[0, n].set_xlabel("Position")
            axs[0, n].set_ylabel("Trial")

            # Plot the cluster mean
            axs[1, n].plot(np.mean(cluster, axis=0))
            axs[1, n].set_xlabel("Position")
            axs[1, n].set_ylabel("Mean activity")
            axs[1, n].set_ylim(0, 1)
    if plot:
        plt.tight_layout()
        plt.show()

    return clusters_list

def get_count(animal_percent_dict_EC_early_peak, field_type="before_percent"):
    count_early = 0
    total_count = 0
    for animal in animal_percent_dict_EC_early_peak:  # ['animal_1']['cell_2']["before_percent"]:
        for cell in animal_percent_dict_EC_early_peak[animal]:
            if animal_percent_dict_EC_early_peak[animal][cell][field_type] > 0.0:
                count_early += 1

            if animal_percent_dict_EC_early_peak[animal][cell]:
                total_count += 1

    probability = count_early / total_count

    return probability

def plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="near_field", plot=False):
    near_field_list = []

    for animal in clusters_dict_labelled_field_type_early_SST_trough:
        for cell in clusters_dict_labelled_field_type_early_SST_trough[animal]:
            for i in clusters_dict_labelled_field_type_early_SST_trough[animal][cell]:
                entry = clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i]
                if field_type in entry:
                    arr = entry[field_type]
                    if len(arr) == 0:
                        continue  # skip empty arrays

                if field_type in clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i]:
                    means = np.mean(clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i][field_type], axis=0)
                    if means.shape != (50,):
                        continue

                    near_field_list.append(means)

    near_field_array = np.array(near_field_list)

    if plot:

        for i in near_field_list:
            plt.plot(i, color='gray')
        plt.plot(np.mean(near_field_array, axis=0), color='red')
        plt.show()

        mean_near_field_array = np.mean(near_field_array, axis=0)
        sem_near_field_array = sem(near_field_array, axis=0)

        plt.plot(mean_near_field_array)
        plt.fill_between(range(len(mean_near_field_array)), mean_near_field_array - sem_near_field_array, mean_near_field_array + sem_near_field_array, alpha=0.2)

    return near_field_array

def plot_clustered_averages(clusters_dict_labelled_field_type_early_SST_trough, clusters_dict_labelled_field_type_late_SST_trough, clusters_dict_labelled_field_type_early_SST_peak, clusters_dict_labelled_field_type_late_SST_peak, use_trough=True, plot=False, cell_type="SST"):

    """
    - uses plot cluster peaks to give you a average of the trials in the cluster for every cluster grouping - roughtly 1 of each per cell - does so for our 4 types of clusters/trial types - before, near, after and wraparound
    - then gets the mean and sem across cells
    - plots the array which is the trial average of every cluster across the whole dataset and the mean of them
    """
    if use_trough:
        near_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="near_field", plot=False)
        before_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="before_field", plot=False)
        after_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="after_field", plot=False)
        noisy_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="noisy_field", plot=False)
        wraparound_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="wraparound_field", plot=False)

        near_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="near_field", plot=False)
        before_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="before_field", plot=False)
        after_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="after_field", plot=False)
        noisy_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="noisy_field", plot=False)
        wraparound_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="wraparound_field", plot=False)

        mean_near_field_array_early = np.mean(near_field_array_early, axis=0)
        sem_near_field_array_early = sem(near_field_array_early, axis=0)

        mean_before_field_array_early = np.mean(before_field_array_early, axis=0)
        sem_before_field_array_early = sem(before_field_array_early, axis=0)

        mean_after_field_array_early = np.mean(after_field_array_early, axis=0)
        sem_after_field_array_early = sem(after_field_array_early, axis=0)

        mean_noisy_field_array_early = np.mean(noisy_field_array_early, axis=0)
        sem_noisy_field_array_early = sem(noisy_field_array_early, axis=0)

        mean_wraparound_field_array_early = np.mean(wraparound_field_array_early, axis=0)
        sem_wraparound_field_array_early = sem(wraparound_field_array_early, axis=0)

        mean_near_field_array_late = np.mean(near_field_array_late, axis=0)
        sem_near_field_array_late = sem(near_field_array_late, axis=0)

        mean_before_field_array_late = np.mean(before_field_array_late, axis=0)
        sem_before_field_array_late = sem(before_field_array_late, axis=0)

        mean_after_field_array_late = np.mean(after_field_array_late, axis=0)
        sem_after_field_array_late = sem(after_field_array_late, axis=0)

        mean_noisy_field_array_late = np.mean(noisy_field_array_late, axis=0)
        sem_noisy_field_array_late = sem(noisy_field_array_late, axis=0)

        mean_wraparound_field_array_late = np.mean(wraparound_field_array_late, axis=0)
        sem_wraparound_field_array_late = sem(wraparound_field_array_late, axis=0)

    else:
        near_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="near_field", plot=False)
        before_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="before_field", plot=False)
        after_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="noisy_field", plot=False)
        wraparound_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="wraparound_field", plot=False)

        near_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="near_field", plot=False)
        before_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="before_field", plot=False)
        after_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="noisy_field", plot=False)
        wraparound_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="wraparound_field", plot=False)

        mean_near_field_array_early = np.mean(near_field_array_early, axis=0)
        sem_near_field_array_early = sem(near_field_array_early, axis=0)

        mean_before_field_array_early = np.mean(before_field_array_early, axis=0)
        sem_before_field_array_early = sem(before_field_array_early, axis=0)

        mean_after_field_array_early = np.mean(after_field_array_early, axis=0)
        sem_after_field_array_early = sem(after_field_array_early, axis=0)

        mean_noisy_field_array_early = np.mean(noisy_field_array_early, axis=0)
        sem_noisy_field_array_early = sem(noisy_field_array_early, axis=0)

        mean_wraparound_field_array_early = np.mean(wraparound_field_array_early, axis=0)
        sem_wraparound_field_array_early = sem(wraparound_field_array_early, axis=0)

        mean_near_field_array_late = np.mean(near_field_array_late, axis=0)
        sem_near_field_array_late = sem(near_field_array_late, axis=0)

        mean_before_field_array_late = np.mean(before_field_array_late, axis=0)
        sem_before_field_array_late = sem(before_field_array_late, axis=0)

        mean_after_field_array_late = np.mean(after_field_array_late, axis=0)
        sem_after_field_array_late = sem(after_field_array_late, axis=0)

        mean_noisy_field_array_late = np.mean(noisy_field_array_late, axis=0)
        sem_noisy_field_array_late = sem(noisy_field_array_late, axis=0)

        mean_wraparound_field_array_late = np.mean(wraparound_field_array_late, axis=0)
        sem_wraparound_field_array_late = sem(wraparound_field_array_late, axis=0)

    if plot:
        fig, axs = plt.subplots(4, 4, figsize=(15, 12))
        plt.suptitle(cell_type)

        if use_trough:
            axs[0, 0].set_title("Before Reward Early Trough")
            axs[0, 1].set_title("Near Reward Early Trough")
            axs[0, 2].set_title("After Reward Early Trough")
            axs[0, 3].set_title("Wraparound Early Trough")
        else:
            axs[0, 0].set_title("Before Reward Early Peak")
            axs[0, 1].set_title("Near Reward Early Peak")
            axs[0, 2].set_title("After Reward Early Peak")
            axs[0, 3].set_title("Wraparound Early Peak")

        axs[1, 0].imshow(before_field_array_early, aspect='auto')

        axs[1, 1].imshow(near_field_array_early, aspect='auto')

        axs[1, 2].imshow(after_field_array_early, aspect='auto')

        axs[1, 3].imshow(wraparound_field_array_early, aspect='auto')

        axs[0, 1].plot(mean_near_field_array_early, color='r')
        axs[0, 1].fill_between(range(len(mean_near_field_array_early)), mean_near_field_array_early - sem_near_field_array_early, mean_near_field_array_early + sem_near_field_array_early, color='r', alpha=0.2)

        axs[0, 0].plot(mean_before_field_array_early, color='b')
        axs[0, 0].fill_between(range(len(mean_before_field_array_early)), mean_before_field_array_early - sem_before_field_array_early, mean_before_field_array_early + sem_before_field_array_early, color='b', alpha=0.2)

        axs[0, 2].plot(mean_after_field_array_early, color='purple')
        axs[0, 2].fill_between(range(len(mean_after_field_array_early)), mean_after_field_array_early - sem_after_field_array_early, mean_after_field_array_early + sem_after_field_array_early, color='purple', alpha=0.2)

        axs[0, 3].plot(mean_wraparound_field_array_early, color='g')
        axs[0, 3].fill_between(range(len(mean_wraparound_field_array_early)), mean_wraparound_field_array_early - sem_wraparound_field_array_early, mean_wraparound_field_array_early + sem_wraparound_field_array_early, color='g', alpha=0.2)

        axs[2, 0].plot(mean_before_field_array_late, color='b')
        axs[2, 0].fill_between(range(len(mean_before_field_array_late)), mean_before_field_array_late - sem_before_field_array_late, mean_before_field_array_late + sem_before_field_array_late, color='b', alpha=0.2)

        axs[2, 1].plot(mean_near_field_array_late, color='r')
        axs[2, 1].fill_between(range(len(mean_near_field_array_late)), mean_near_field_array_late - sem_near_field_array_late, mean_near_field_array_late + sem_near_field_array_late, color='r', alpha=0.2)

        axs[2, 2].plot(mean_after_field_array_late, color='purple')
        axs[2, 2].fill_between(range(len(mean_after_field_array_late)), mean_after_field_array_late - sem_after_field_array_late, mean_after_field_array_late + sem_after_field_array_late, color='purple', alpha=0.2)

        axs[2, 3].plot(mean_wraparound_field_array_late, color='g')
        axs[2, 3].fill_between(range(len(mean_wraparound_field_array_late)), mean_wraparound_field_array_late - sem_wraparound_field_array_late, mean_wraparound_field_array_late + sem_wraparound_field_array_late, color='g', alpha=0.2)

        if use_trough:
            axs[2, 0].set_title("Before Reward Late Trough")
            axs[2, 1].set_title("Near Reward Late Trough")
            axs[2, 2].set_title("After Reward Late Trough")
            axs[2, 3].set_title("Wraparound Late Trough")
        else:
            axs[2, 0].set_title("Before Reward Late Peak")
            axs[2, 1].set_title("Near Reward Late Peak")
            axs[2, 2].set_title("After Reward Late Peak")
            axs[2, 3].set_title("Wraparound Late Peak")

        axs[3, 0].imshow(before_field_array_late, aspect='auto')

        axs[3, 1].imshow(near_field_array_late, aspect='auto')

        axs[3, 2].imshow(after_field_array_late, aspect='auto')

        axs[3, 3].imshow(wraparound_field_array_late, aspect='auto')

        plt.tight_layout()
        plt.plot()

    else:
        return before_field_array_early, near_field_array_early, after_field_array_early, wraparound_field_array_early, before_field_array_late, near_field_array_late, after_field_array_late, wraparound_field_array_late


def plot_clustered_averages_deterministic(clusters_dict_labelled_field_type_early_SST_trough, clusters_dict_labelled_field_type_late_SST_trough, plot=False, cell_type="SST"):

    near_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="near_field", plot=False)
    before_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="before_field", plot=False)
    after_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="after_field", plot=False)
    noisy_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="noisy_field", plot=False)
    wraparound_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type="wraparound_field", plot=False)

    near_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="near_field", plot=False)
    before_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="before_field", plot=False)
    after_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="after_field", plot=False)
    noisy_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="noisy_field", plot=False)
    wraparound_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_trough, field_type="wraparound_field", plot=False)

    mean_near_field_array_early = np.mean(near_field_array_early, axis=0)
    sem_near_field_array_early = sem(near_field_array_early, axis=0)

    mean_before_field_array_early = np.mean(before_field_array_early, axis=0)
    sem_before_field_array_early = sem(before_field_array_early, axis=0)

    mean_after_field_array_early = np.mean(after_field_array_early, axis=0)
    sem_after_field_array_early = sem(after_field_array_early, axis=0)

    mean_noisy_field_array_early = np.mean(noisy_field_array_early, axis=0)
    sem_noisy_field_array_early = sem(noisy_field_array_early, axis=0)

    mean_wraparound_field_array_early = np.mean(wraparound_field_array_early, axis=0)
    sem_wraparound_field_array_early = sem(wraparound_field_array_early, axis=0)

    mean_near_field_array_late = np.mean(near_field_array_late, axis=0)
    sem_near_field_array_late = sem(near_field_array_late, axis=0)

    mean_before_field_array_late = np.mean(before_field_array_late, axis=0)
    sem_before_field_array_late = sem(before_field_array_late, axis=0)

    mean_after_field_array_late = np.mean(after_field_array_late, axis=0)
    sem_after_field_array_late = sem(after_field_array_late, axis=0)

    mean_noisy_field_array_late = np.mean(noisy_field_array_late, axis=0)
    sem_noisy_field_array_late = sem(noisy_field_array_late, axis=0)

    mean_wraparound_field_array_late = np.mean(wraparound_field_array_late, axis=0)
    sem_wraparound_field_array_late = sem(wraparound_field_array_late, axis=0)


    if plot:
        fig, axs = plt.subplots(4, 4, figsize=(15, 12))
        plt.suptitle(cell_type)

        axs[0, 0].set_title("Before Reward Early Deterministic")
        axs[0, 1].set_title("Near Reward Early Peak Deterministic")
        axs[0, 2].set_title("After Reward Early Peak Deterministic")
        axs[0, 3].set_title("Wraparound Early Peak Deterministic")

        axs[1, 0].imshow(before_field_array_early, aspect='auto')

        axs[1, 1].imshow(near_field_array_early, aspect='auto')

        axs[1, 2].imshow(after_field_array_early, aspect='auto')

        axs[1, 3].imshow(wraparound_field_array_early, aspect='auto')

        axs[0, 1].plot(mean_near_field_array_early, color='r')
        axs[0, 1].fill_between(range(len(mean_near_field_array_early)), mean_near_field_array_early - sem_near_field_array_early, mean_near_field_array_early + sem_near_field_array_early, color='r', alpha=0.2)

        axs[0, 0].plot(mean_before_field_array_early, color='b')
        axs[0, 0].fill_between(range(len(mean_before_field_array_early)), mean_before_field_array_early - sem_before_field_array_early, mean_before_field_array_early + sem_before_field_array_early, color='b', alpha=0.2)

        axs[0, 2].plot(mean_after_field_array_early, color='purple')
        axs[0, 2].fill_between(range(len(mean_after_field_array_early)), mean_after_field_array_early - sem_after_field_array_early, mean_after_field_array_early + sem_after_field_array_early, color='purple', alpha=0.2)

        axs[0, 3].plot(mean_wraparound_field_array_early, color='g')
        axs[0, 3].fill_between(range(len(mean_wraparound_field_array_early)), mean_wraparound_field_array_early - sem_wraparound_field_array_early, mean_wraparound_field_array_early + sem_wraparound_field_array_early, color='g', alpha=0.2)

        axs[2, 0].plot(mean_before_field_array_late, color='b')
        axs[2, 0].fill_between(range(len(mean_before_field_array_late)), mean_before_field_array_late - sem_before_field_array_late, mean_before_field_array_late + sem_before_field_array_late, color='b', alpha=0.2)

        axs[2, 1].plot(mean_near_field_array_late, color='r')
        axs[2, 1].fill_between(range(len(mean_near_field_array_late)), mean_near_field_array_late - sem_near_field_array_late, mean_near_field_array_late + sem_near_field_array_late, color='r', alpha=0.2)

        axs[2, 2].plot(mean_after_field_array_late, color='purple')
        axs[2, 2].fill_between(range(len(mean_after_field_array_late)), mean_after_field_array_late - sem_after_field_array_late, mean_after_field_array_late + sem_after_field_array_late, color='purple', alpha=0.2)

        axs[2, 3].plot(mean_wraparound_field_array_late, color='g')
        axs[2, 3].fill_between(range(len(mean_wraparound_field_array_late)), mean_wraparound_field_array_late - sem_wraparound_field_array_late, mean_wraparound_field_array_late + sem_wraparound_field_array_late, color='g', alpha=0.2)


        axs[2, 0].set_title("Before Reward Late Deterministic")
        axs[2, 1].set_title("Near Reward Late Deterministic")
        axs[2, 2].set_title("After Reward Late Deterministic")
        axs[2, 3].set_title("Wraparound Late Deterministic")

        axs[3, 0].imshow(before_field_array_late, aspect='auto')

        axs[3, 1].imshow(near_field_array_late, aspect='auto')

        axs[3, 2].imshow(after_field_array_late, aspect='auto')

        axs[3, 3].imshow(wraparound_field_array_late, aspect='auto')

        plt.tight_layout()
        plt.plot()

    else:
        return before_field_array_early, near_field_array_early, after_field_array_early, wraparound_field_array_early, before_field_array_late, near_field_array_late, after_field_array_late, wraparound_field_array_late


def generate_gaussian(length=50, peak_position=25, std=5, amplitude=1.0):
    """
    Generate a Gaussian array of given length, peaking at `peak_position`.

    Parameters:
    - length: total number of bins (default 50)
    - peak_position: where the Gaussian peaks (can be float)
    - std: standard deviation of the Gaussian
    - amplitude: height of the peak

    Returns:
    - 1D numpy array of shape (length,)
    """
    x = np.arange(length)
    gaussian = amplitude * np.exp(-0.5 * ((x - peak_position) / std) ** 2)
    return gaussian

def random_timeseries(initial_value: float, volatility: float, count: int, rng):
    time_series = [initial_value, ]
    for _ in range(count):
        time_series.append(time_series[-1] + initial_value * random.gauss(0, 1) * volatility)
    return time_series

def remove_duplicate_trials(cells_with_A, cells_with_B, trials_A_dict, trials_B_dict, rng):
    for cell in np.intersect1d(cells_with_A, cells_with_B):
        trials_A = trials_A_dict[cell]
        trials_B = trials_B_dict[cell]

        overlapping_trials = np.intersect1d(trials_A, trials_B)

        for trial in overlapping_trials:
            winner = rng.choice(["A", "B"])
            if winner == "A":
                trials_B = trials_B[trials_B != trial]
            else:
                trials_A = trials_A[trials_A != trial]

        trials_A_dict[cell] = trials_A
        trials_B_dict[cell] = trials_B

        still_overlapping = np.intersect1d(trials_A, trials_B)

    return trials_A_dict, trials_B_dict


def remove_all_duplicates(trial_dicts, cells_with_types, cell_array_early_list, rng):
    """
    trial_dicts: dict of field_type -> {cell_id -> np.array of trial indices}
    cells_with_types: dict of field_type -> np.array of cell ids
    """
    all_types = list(trial_dicts.keys())

    for cell in range(len(cell_array_early_list)):
        if all(cell in cells_with_types[field] for field in all_types):
            # Get all trial sets
            trial_sets = {ftype: set(trial_dicts[ftype][cell]) for ftype in all_types}
            combined_trials = set.union(*trial_sets.values())

            new_assignments = {ftype: set() for ftype in all_types}

            used_trials = set()

            for t in combined_trials:
                # Get all field types that had this trial
                present_in = [ftype for ftype in all_types if t in trial_sets[ftype]]
                if len(present_in) > 1:
                    # Randomly pick one to keep it
                    winner = rng.choice(present_in)
                else:
                    winner = present_in[0]
                new_assignments[winner].add(t)
                used_trials.add(t)

                total_trials = cell_array_early_list[cell].shape[1]

                # Try to reassign the losers
                losers = [ftype for ftype in present_in if ftype != winner]
                for loser in losers:
                    # Find unused trial index
                    available = [new_t for new_t in range(total_trials) if new_t not in used_trials]
                    if available:
                        new_t = rng.choice(available)
                        new_assignments[loser].add(new_t)
                        used_trials.add(new_t)
                    else:
                        print(f"⚠️ No available trials left to reassign for cell {cell}, {loser}")

            # Save new assignments
            for ftype in all_types:
                trial_dicts[ftype][cell] = np.array(sorted(new_assignments[ftype]))

    return trial_dicts

def reconstruct_activity_from_clusters5(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, example_before_field_early, example_near_field_early, example_after_field_early, example_wraparound_field_early, example_before_field_late, example_near_field_late, example_after_field_late, example_wraparound_field_late, percent_lists_EC_early_peak, percent_lists_EC_late_peak,
                                        percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, inits, vol, cell_type="EC Trough", plot=False):

    """
    - consult the percent of cells that have the given trial type and then random choice select cells to express that field type
    - within each cell that gets a field get a number of trials that will express based on the proportion and then randomly select which trials out of all possible trials for that cell will be assigned that field types activity

    """
    rng = np.random.default_rng(seed=42)

    num_cells = len(percent_lists_EC_early_peak[0])

    before_percent_array = np.array(percent_lists_EC_early_peak[0])
    near_percent_array = np.array(percent_lists_EC_early_peak[1])
    after_percent_array = np.array(percent_lists_EC_early_peak[2])
    wraparound_percent_array = np.array(percent_lists_EC_early_peak[3])

    before_percent_array_late = np.array(percent_lists_EC_late_peak[0])
    near_percent_array_late = np.array(percent_lists_EC_late_peak[1])
    after_percent_array_late = np.array(percent_lists_EC_late_peak[2])
    wraparound_percent_array_late = np.array(percent_lists_EC_late_peak[3])

    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, animal_TCA=False)

    flat_list_first = [item for sublist in animal_first_changepoints_list for item in sublist]
    flat_list_second = [item for sublist in animal_second_changepoints_list for item in sublist]

    stack_num_trials_list = np.vstack([flat_list_first, flat_list_second])

    num_trials_list = np.sum(stack_num_trials_list, axis=0)

    available_cells_before_early = percent_of_cells_list_EC_early_trough[0] * len(num_trials_list)
    available_cells_near_early = percent_of_cells_list_EC_early_trough[1] * len(num_trials_list)
    available_cells_after_early = percent_of_cells_list_EC_early_trough[2] * len(num_trials_list)
    available_cells_wraparound_early = percent_of_cells_list_EC_early_trough[3] * len(num_trials_list)

    cells_with_before_field_early = rng.choice(num_cells, int(available_cells_before_early), replace=False)
    cells_with_near_field_early = rng.choice(num_cells, int(available_cells_near_early), replace=False)
    cells_with_after_field_early = rng.choice(num_cells, int(available_cells_after_early), replace=False)
    cells_with_wraparound_field_early = rng.choice(num_cells, int(available_cells_wraparound_early), replace=False)

    available_cells_before_late = percent_of_cells_list_EC_late_trough[0] * len(num_trials_list)
    available_cells_near_late = percent_of_cells_list_EC_late_trough[1] * len(num_trials_list)
    available_cells_after_late = percent_of_cells_list_EC_late_trough[2] * len(num_trials_list)
    available_cells_wraparound_late = percent_of_cells_list_EC_late_trough[3] * len(num_trials_list)

    cells_with_before_field_late = rng.choice(num_cells, int(available_cells_before_late), replace=False)
    cells_with_near_field_late = rng.choice(num_cells, int(available_cells_near_late), replace=False)
    cells_with_after_field_late = rng.choice(num_cells, int(available_cells_after_late), replace=False)
    cells_with_wraparound_field_late = rng.choice(num_cells, int(available_cells_wraparound_late), replace=False)

    cell_array_early_list = []
    cell_array_late_list = []

    for cell in range(len(num_trials_list)):
        rt_early = np.array(random_timeseries2(initial_value=inits, volatility=vol, count=49, rng=rng))
        rt_late = np.array(random_timeseries2(initial_value=inits, volatility=vol, count=49, rng=rng))
        #         cell_array = np.tile(example_[:, np.newaxis], (1,100))
        #         cell_array_late = np.tile(example_[:, np.newaxis], (1,100))

        #     cell_array = np.tile(rt_early[:, np.newaxis], (1,num_trials_early))
        #     cell_array_late = np.tile(rt_late[:, np.newaxis], (1,num_trials_late))
        # cell_array = np.tile(example_[:, np.newaxis], (1,100))

        num_trials_early = flat_list_first[cell]
        num_trials_late = num_trials_list[cell] - flat_list_second[cell]

        cell_array = np.tile(rt_early[:, np.newaxis], (1, num_trials_early))
        cell_array_early_list.append(cell_array)
        cell_array_late = np.tile(rt_late[:, np.newaxis], (1, num_trials_late))
        cell_array_late_list.append(cell_array_late)

    trial_indices_before_dict_early = {}
    trial_indices_near_dict_early = {}
    trial_indices_after_dict_early = {}
    trial_indices_wraparound_dict_early = {}

    trial_indices_before_dict_late = {}
    trial_indices_near_dict_late = {}
    trial_indices_after_dict_late = {}
    trial_indices_wraparound_dict_late = {}

    for cell in range(len(num_trials_list)):
        num_trials_early = flat_list_first[cell]
        num_trials_late = num_trials_list[cell] - flat_list_second[cell]

        available_trials_early = np.arange(num_trials_early)
        remaining_trials_early = available_trials_early.copy()

        available_trials_late = np.arange(num_trials_late)
        remaining_trials_late = available_trials_late.copy()

        if cell in cells_with_before_field_early:
            proportion_early_before = (before_percent_array[cell] / 100) * num_trials_early
            trial_indices_before = rng.choice(available_trials_early, int(proportion_early_before), replace=False)
            trial_indices_before_dict_early[cell] = trial_indices_before

        if cell in cells_with_near_field_early:
            proportion_early_near = (near_percent_array[cell] / 100) * num_trials_early
            trial_indices_near = rng.choice(available_trials_early, int(proportion_early_near), replace=False)
            trial_indices_near_dict_early[cell] = trial_indices_near

        if cell in cells_with_after_field_early:
            proportion_early_after = (after_percent_array[cell] / 100) * num_trials_early
            trial_indices_after = rng.choice(available_trials_early, int(proportion_early_after), replace=False)
            trial_indices_after_dict_early[cell] = trial_indices_after

        if cell in cells_with_wraparound_field_early:
            proportion_early_wraparound = (wraparound_percent_array[cell] / 100) * num_trials_early
            trial_indices_wraparound = rng.choice(available_trials_early, int(proportion_early_wraparound), replace=False)
            trial_indices_wraparound_dict_early[cell] = trial_indices_wraparound

        if cell in cells_with_before_field_late:
            proportion_late_before = (before_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_before = rng.choice(available_trials_late, int(proportion_late_before), replace=False)
            trial_indices_before_dict_late[cell] = trial_indices_before

        if cell in cells_with_near_field_late:
            proportion_late_near = (near_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_near = rng.choice(available_trials_late, int(proportion_late_near), replace=False)
            trial_indices_near_dict_late[cell] = trial_indices_near

        if cell in cells_with_after_field_late:
            proportion_late_after = (after_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_after = rng.choice(available_trials_late, int(proportion_late_after), replace=False)
            trial_indices_after_dict_late[cell] = trial_indices_after

        if cell in cells_with_wraparound_field_late:
            proportion_late_wraparound = (wraparound_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_wraparound = rng.choice(available_trials_late, int(proportion_late_wraparound), replace=False)
            trial_indices_wraparound_dict_late[cell] = trial_indices_wraparound

    #     trial_dicts_early = {
    #     "before": trial_indices_before_dict_early,
    #     "near": trial_indices_near_dict_early,
    #     "after": trial_indices_after_dict_early,
    #     "wraparound": trial_indices_wraparound_dict_early
    #     }
    #     cells_with_early = {
    #         "before": cells_with_before_field_early,
    #         "near": cells_with_near_field_early,
    #         "after": cells_with_after_field_early,
    #         "wraparound": cells_with_wraparound_field_early
    #     }

    #     trial_dicts_early = remove_all_duplicates(trial_dicts_early, cells_with_early, cell_array_early_list)

    trial_indices_before_dict_early, trial_indices_near_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_near_field_early, trial_indices_before_dict_early, trial_indices_near_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_after_field_early, trial_indices_near_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_after_field_early, trial_indices_before_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_wraparound_field_early, trial_indices_before_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_wraparound_field_early, trial_indices_near_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_after_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_after_field_early, cells_with_wraparound_field_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early, rng)

    for cell, trial_indices in trial_indices_before_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["before"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                cell_array_early_list[cell][:, t] = np.mean(example_before_field_early, axis=0)
            else:
                print("improper alignment")
    for cell, trial_indices in trial_indices_near_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["near"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                cell_array_early_list[cell][:, t] = np.mean(example_near_field_early, axis=0)
            else:
                print("improper alignment")

    for cell, trial_indices in trial_indices_after_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["after"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                cell_array_early_list[cell][:, t] = np.mean(example_after_field_early, axis=0)
            else:
                print("improper alignment")

    for cell, trial_indices in trial_indices_wraparound_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["wraparound"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                cell_array_early_list[cell][:, t] = np.mean(example_wraparound_field_early, axis=0)
            else:
                print("improper alignment")

    #     "before": trial_indices_before_dict_late,
    #     "near": trial_indices_near_dict_late,
    #     "after": trial_indices_after_dict_late,
    #     "wraparound": trial_indices_wraparound_dict_late}
    #     cells_with_late = {
    #         "before": cells_with_before_field_late,
    #         "near": cells_with_near_field_late,
    #         "after": cells_with_after_field_late,
    #         "wraparound": cells_with_wraparound_field_late}

    #     trial_dicts_late = remove_all_duplicates(trial_dicts_late, cells_with_late, cell_array_late_list)

    trial_indices_before_dict_late, trial_indices_near_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_near_field_late, trial_indices_before_dict_late, trial_indices_near_dict_late, rng)
    trial_indices_near_dict_late, trial_indices_after_dict_late = remove_duplicate_trials(cells_with_near_field_late, cells_with_after_field_late, trial_indices_near_dict_late, trial_indices_after_dict_late, rng)
    trial_indices_before_dict_late, trial_indices_after_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_after_field_late, trial_indices_before_dict_late, trial_indices_after_dict_late, rng)
    trial_indices_before_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_wraparound_field_late, trial_indices_before_dict_late, trial_indices_wraparound_dict_late, rng)
    trial_indices_near_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_near_field_late, cells_with_wraparound_field_late, trial_indices_near_dict_late, trial_indices_wraparound_dict_late, rng)
    trial_indices_after_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_after_field_late, cells_with_wraparound_field_late, trial_indices_after_dict_late, trial_indices_wraparound_dict_late, rng)

    for cell, trial_indices in trial_indices_before_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["before"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                cell_array_late_list[cell][:, t] = np.mean(example_before_field_late, axis=0)
            else:
                print("improper alignment")
    for cell, trial_indices in trial_indices_near_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["near"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                cell_array_late_list[cell][:, t] = np.mean(example_near_field_late, axis=0)
            else:
                print("improper alignment")

    for cell, trial_indices in trial_indices_after_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["after"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                cell_array_late_list[cell][:, t] = np.mean(example_after_field_late, axis=0)
            else:
                print("improper alignment")

    for cell, trial_indices in trial_indices_wraparound_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["wraparound"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                cell_array_late_list[cell][:, t] = np.mean(example_wraparound_field_late, axis=0)
            else:
                print("improper alignment")

    full_activity_list = []
    full_activity_trial_av_list = []

    trial_averaged_cell_list_early = []
    trial_averaged_cell_list_late = []

    for i in range(len(cell_array_early_list)):
        early_component = cell_array_early_list[i]
        late_component = cell_array_late_list[i]

        full_actiivty = np.concatenate([early_component, late_component], axis=1)

        full_activity_list.append(full_actiivty)

        full_activity_trial_av_list.append(np.mean(full_actiivty, axis=1))

        full_activity_trial_av_array = np.array(full_activity_trial_av_list)

        trial_averaged_cell_list_early.append(np.mean(early_component, axis=1))
        trial_averaged_cell_list_late.append(np.mean(late_component, axis=1))

    trial_averaged_cell_array_early = np.array(trial_averaged_cell_list_early)
    trial_averaged_cell_array_late = np.array(trial_averaged_cell_list_late)

    if plot:
        fig, axs = plt.subplots(5, 4, figsize=(15, 20))
        plt.suptitle(cell_type)

        axs[0, 0].plot(np.mean(example_before_field_early, axis=0), label="Early")
        axs[0, 0].plot(np.mean(example_before_field_late, axis=0), label="Late")
        axs[0, 0].set_title("Inputs Before Field")
        axs[0, 0].set_ylabel("Activity")
        axs[0, 0].set_xlabel("Position Bins")
        axs[0, 0].legend()

        axs[0, 1].plot(np.mean(example_near_field_early, axis=0), label="Early")
        axs[0, 1].plot(np.mean(example_near_field_late, axis=0), label="Late")
        axs[0, 1].set_title("Inputs Near Field")
        axs[0, 1].set_ylabel("Activity")
        axs[0, 1].set_xlabel("Position Bins")
        axs[0, 1].legend()

        axs[0, 2].plot(np.mean(example_after_field_early, axis=0), label="Early")
        axs[0, 2].plot(np.mean(example_after_field_late, axis=0), label="Late")
        axs[0, 2].set_title("Inputs After Field")
        axs[0, 2].set_ylabel("Activity")
        axs[0, 2].set_xlabel("Position Bins")
        axs[0, 2].legend()

        axs[0, 3].plot(np.mean(example_wraparound_field_early, axis=0), label="Early")
        axs[0, 3].plot(np.mean(example_wraparound_field_late, axis=0), label="Late")
        axs[0, 3].set_title("Inputs Wraparound Field")
        axs[0, 3].set_ylabel("Activity")
        axs[0, 3].set_xlabel("Position Bins")
        axs[0, 3].legend()

        axs[1, 0].hist(before_percent_array, bins=20)
        axs[1, 0].set_title(f"Early Before Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[0] * 100:.1f}%")
        axs[1, 0].set_ylabel("Number of Cells")
        axs[1, 0].set_xlabel("Percent of Trials")

        axs[1, 1].hist(near_percent_array, bins=20)
        axs[1, 1].set_title(f"Early Near Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[1] * 100:.1f}%")
        axs[1, 1].set_ylabel("Number of Cells")
        axs[1, 1].set_xlabel("Percent of Trials")

        axs[1, 2].hist(after_percent_array, bins=20)
        axs[1, 2].set_title(f"Early After Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[2] * 100:.1f}%")
        axs[1, 2].set_ylabel("Number of Cells")
        axs[1, 2].set_xlabel("Percent of Trials")

        axs[1, 3].hist(wraparound_percent_array, bins=20)
        axs[1, 3].set_title(f"Early Wraparound Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[3] * 100:.1f}%")
        axs[1, 3].set_ylabel("Number of Cells")
        axs[1, 3].set_xlabel("Percent of Trials")

        axs[2, 0].hist(before_percent_array_late, bins=20)
        #         axs[2,0].set_title(f"Late Before Field Probability \n Percent of Cells={available_trials_before_late:.1f}%")
        axs[2, 0].set_title("Late Before Field Probability")
        axs[2, 0].set_ylabel("Number of Cells")
        axs[2, 0].set_xlabel("Percent of Trials")

        axs[2, 1].hist(near_percent_array_late, bins=20)
        #         axs[2,1].set_title(f"Late Near Field Probability \n Percent of Cells={available_trials_near_late:.1f}%")
        axs[2, 1].set_title("Late Near Field Probability")
        axs[2, 1].set_ylabel("Number of Cells")
        axs[2, 1].set_xlabel("Percent of Trials")

        axs[2, 2].hist(after_percent_array_late, bins=20)
        #         axs[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
        axs[2, 2].set_title("Late After Field Probability")
        axs[2, 2].set_ylabel("Number of Cells")
        axs[2, 2].set_xlabel("Percent of Trials")

        axs[2, 3].hist(wraparound_percent_array_late, bins=20)
        #         axs[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
        axs[2, 3].set_title("Late Wraparound Field Probability")
        axs[2, 3].set_ylabel("Number of Cells")
        axs[2, 3].set_xlabel("Percent of Trials")

        axs[4, 0].imshow(trial_averaged_cell_array_early, aspect='auto')
        axs[4, 0].set_ylabel("Cell ID")
        axs[4, 0].set_xlabel("Position Bin")

        means_early = np.mean(trial_averaged_cell_array_early, axis=0)
        sems_early = sem(trial_averaged_cell_array_early, axis=0)
        axs[3, 0].plot(means_early, color='orange')
        axs[3, 0].fill_between(range(len(means_early)), means_early + sems_early, means_early - sems_early, alpha=0.2, color='orange')
        axs[3, 0].set_xlabel("Position Bins")
        axs[3, 0].set_ylabel("Activity")
        axs[3, 0].set_title("Generative Early Learn")

        axs[4, 1].imshow(trial_averaged_cell_array_late, aspect='auto')
        axs[4, 1].set_ylabel("Cell ID")
        axs[4, 1].set_xlabel("Position Bin")

        means_late = np.mean(trial_averaged_cell_array_late, axis=0)
        sems_late = sem(trial_averaged_cell_array_late, axis=0)
        axs[3, 1].plot(means_late, color='r')
        axs[3, 1].fill_between(range(len(means_late)), means_late + sems_late, means_late - sems_late, alpha=0.2, color='r')
        axs[3, 1].set_xlabel("Position Bins")
        axs[3, 1].set_ylabel("Activity")
        axs[3, 1].set_title("Generative Late Learn")

        axs[4, 2].imshow(full_activity_trial_av_array, aspect='auto')
        axs[4, 2].set_ylabel("Cell ID")
        axs[4, 2].set_xlabel("Position Bin")

        means_overall = np.mean(full_activity_trial_av_list, axis=0)
        sems_overall = sem(full_activity_trial_av_list, axis=0)
        axs[3, 2].plot(means_overall, color='purple')
        axs[3, 2].fill_between(range(len(means_overall)), means_overall + sems_overall, means_overall - sems_overall, alpha=0.2, color='purple')
        axs[3, 2].set_xlabel("Position Bins")
        axs[3, 2].set_ylabel("Activity")
        axs[3, 2].set_title("Generative Overall Average")

        plt.tight_layout()
        plt.show()
    else:
        return cell_array_early_list, cell_array_late_list, full_activity_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, full_activity_trial_av_array

def get_early_vs_late_activity(cell_SST_model_ranks20_contig_x00):
    data_early_list = []
    data_late_list = []

    for animal in cell_SST_model_ranks20_contig_x00[20]:
        for cell in cell_SST_model_ranks20_contig_x00[20][animal]:
            data_early = cell_SST_model_ranks20_contig_x00[20][animal][cell][1][f'cell_{cell}']["cluster_trial_mean_dict"]["clusters_chosen_3"][0]
            data_early_list.append(data_early)
            data_late = cell_SST_model_ranks20_contig_x00[20][animal][cell][1][f'cell_{cell}']["cluster_trial_mean_dict"]["clusters_chosen_3"][2]
            data_late_list.append(data_late)

    data_early_array = np.array(data_early_list)
    data_late_array = np.array(data_late_list)

    return data_early_array, data_late_array

def min_max_my_data(array_data):
    norm_data = (array_data - np.min(array_data)) / (np.max(array_data) - np.min(array_data))
    return norm_data

def compare_within_subtype(cell_SST_model_ranks20_contig_x00, trial_averaged_cell_array_early_SST_trough, trial_averaged_cell_array_late_SST_trough, full_activity_trial_av_array_SST_trough, residual_activity_dict_SST, cell_type="SST Peak"):
    early_array_SST, late_array_SST = get_early_vs_late_activity(cell_SST_model_ranks20_contig_x00)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    early_array_SST_min_max = min_max_my_data(early_array_SST)
    late_array_SST_min_max = min_max_my_data(late_array_SST)

    trial_averaged_cell_array_early_SST_trough_min_max = min_max_my_data(trial_averaged_cell_array_early_SST_trough)
    trial_averaged_cell_array_late_SST_trough_min_max = min_max_my_data(trial_averaged_cell_array_late_SST_trough)

    mean_early_array_SST = np.mean(early_array_SST, axis=0)
    sem_early_array_SST = sem(early_array_SST, axis=0)
    mean_late_array_SST = np.mean(late_array_SST, axis=0)
    sem_early_array_SST = sem(late_array_SST, axis=0)

    #     mean_early_array_SST_min_max = np.mean(early_array_SST_min_max, axis=0)
    #     sem_early_array_SST_min_max = sem(early_array_SST_min_max, axis=0)
    #     mean_late_array_SST_min_max = np.mean(late_array_SST_min_max, axis=0)
    #     sem_late_array_SST_min_max = sem(late_array_SST_min_max, axis=0)

    mean_early_array_SST_min_max = min_max_my_data(mean_early_array_SST)
    sem_early_array_SST_min_max = min_max_my_data(sem_early_array_SST)
    mean_late_array_SST_min_max = min_max_my_data(mean_late_array_SST)
    sem_late_array_SST_min_max = min_max_my_data(sem_early_array_SST)

    mean_SST_early_model = np.mean(trial_averaged_cell_array_early_SST_trough, axis=0)
    sem_SST_early_model = sem(trial_averaged_cell_array_early_SST_trough, axis=0)
    mean_SST_late_model = np.mean(trial_averaged_cell_array_late_SST_trough, axis=0)
    sem_SST_late_model = sem(trial_averaged_cell_array_late_SST_trough, axis=0)

    #     mean_SST_early_model_min_max = np.mean(trial_averaged_cell_array_early_SST_trough_min_max, axis=0)
    #     sem_SST_early_model_min_max = sem(trial_averaged_cell_array_early_SST_trough_min_max, axis=0)
    #     mean_SST_late_model_min_max = np.mean(trial_averaged_cell_array_late_SST_trough_min_max, axis=0)
    #     sem_SST_late_model_min_max = sem(trial_averaged_cell_array_late_SST_trough_min_max, axis=0)

    mean_SST_early_model_min_max = min_max_my_data(mean_SST_early_model)
    sem_SST_early_model_min_max = min_max_my_data(sem_SST_early_model)
    mean_SST_late_model_min_max = min_max_my_data(mean_SST_late_model)
    sem_SST_late_model_min_max = min_max_my_data(sem_SST_late_model)

    axs[0, 0].plot(mean_early_array_SST, color='cyan', label="Real")
    axs[0, 0].fill_between(range(len(mean_early_array_SST)), mean_early_array_SST + sem_early_array_SST, mean_early_array_SST - sem_early_array_SST, alpha=0.2, color='cyan')
    axs[0, 0].plot(mean_SST_early_model, label="Model", color='cyan', linestyle="--", linewidth=2)
    axs[0, 0].fill_between(range(len(mean_SST_early_model)), mean_SST_early_model + sem_SST_early_model, mean_SST_early_model - sem_SST_early_model, alpha=0.2, color='cyan')
    axs[0, 0].set_title(f"{cell_type} Early")
    axs[0, 0].legend()

    axs[1, 0].plot(mean_late_array_SST, color='b', label="Real")
    axs[1, 0].fill_between(range(len(mean_late_array_SST)), mean_late_array_SST + sem_early_array_SST, mean_late_array_SST - sem_early_array_SST, alpha=0.2, color='b')
    axs[1, 0].plot(mean_SST_late_model, label="Model", color='b', linestyle="--", linewidth=2)
    axs[1, 0].fill_between(range(len(mean_SST_late_model)), mean_SST_late_model + sem_SST_late_model, mean_SST_late_model - sem_SST_late_model, alpha=0.2, color='b')
    axs[1, 0].set_title(f"{cell_type} Late")
    axs[1, 0].legend()

    axs[0, 1].plot(mean_early_array_SST, color='cyan', label="Early")
    axs[0, 1].fill_between(range(len(mean_early_array_SST)), mean_early_array_SST + sem_early_array_SST, mean_early_array_SST - sem_early_array_SST, alpha=0.2, color='cyan')
    axs[0, 1].plot(mean_late_array_SST, color='b', label="Late")
    axs[0, 1].fill_between(range(len(mean_late_array_SST)), mean_late_array_SST + sem_early_array_SST, mean_late_array_SST - sem_early_array_SST, alpha=0.2, color='b')
    axs[0, 1].set_title(f"{cell_type} Real")
    axs[0, 1].legend()

    axs[1, 1].plot(mean_SST_early_model, label="Early", color='cyan', linestyle="--", linewidth=2)
    axs[1, 1].fill_between(range(len(mean_SST_early_model)), mean_SST_early_model + sem_SST_early_model, mean_SST_early_model - sem_SST_early_model, alpha=0.2, color='cyan')
    axs[1, 1].plot(mean_SST_late_model, label="Late", color='b', linestyle="--", linewidth=2)
    axs[1, 1].fill_between(range(len(mean_SST_late_model)), mean_SST_late_model + sem_SST_late_model, mean_SST_late_model - sem_SST_late_model, alpha=0.2, color='b')
    axs[1, 1].set_title(f"{cell_type} Model")
    axs[1, 1].legend()

    axs[0, 2].plot(mean_early_array_SST_min_max, color='cyan', label="Real")
    #     axs[0,2].fill_between(range(len(mean_early_array_SST_min_max)), mean_early_array_SST_min_max+sem_early_array_SST_min_max, mean_early_array_SST_min_max-sem_early_array_SST_min_max, alpha=0.2, color='cyan')
    axs[0, 2].plot(mean_SST_early_model_min_max, label="Model", color='cyan', linestyle='--', linewidth=2)
    #     axs[0,2].fill_between(range(len(mean_SST_early_model_min_max)), mean_SST_early_model_min_max+sem_SST_early_model_min_max, mean_SST_early_model_min_max-sem_SST_early_model_min_max, alpha=0.2, color='cyan')
    axs[0, 2].set_title(f"{cell_type} Early Min Max")
    axs[0, 2].legend()

    axs[1, 2].plot(mean_late_array_SST_min_max, color='b', label="Real")
    #     axs[1,2].fill_between(range(len(mean_late_array_SST_min_max)), mean_late_array_SST_min_max+sem_late_array_SST_min_max, mean_late_array_SST_min_max-sem_late_array_SST_min_max, alpha=0.2, color='b')
    axs[1, 2].plot(mean_SST_late_model_min_max, label="Model", color='b', linestyle='--', linewidth=2)
    #     axs[1,2].fill_between(range(len(mean_SST_late_model_min_max)), mean_SST_late_model_min_max+sem_SST_late_model_min_max, mean_SST_late_model_min_max-sem_SST_late_model_min_max, alpha=0.2, color='b')
    axs[1, 2].set_title(f"{cell_type} Late Min Max")
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()

def split_data(testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell):
    fixed_model_NDNF_dict = {20:{}}
    testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell[20]
    for idx, animal in enumerate(testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell[20]):
        if animal>8:
            fixed_model_NDNF_dict[20][idx-9] = testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell[20][animal]

    return fixed_model_NDNF_dict



def plot_clustered_averages2(clusters_dict_labelled_field_type_early_SST_peak, clusters_dict_labelled_field_type_late_SST_peak, clusters_dict_labelled_field_type_middle_SST_peak, use_trough=True, plot=False, cell_type="SST", return_array=True):
    """
    - params: early, middle and late cluster dicts cut by contiguous kmeans
    - returns: either dicts seperated by cell and animal for the cluster mean activity or arrays where all the clusters trial averages are clumped togther in a 2d array - dependent on the return_array flag
    """

    if return_array:
        near_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="near_field", plot=False)
        before_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="before_field", plot=False)
        after_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_early = plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_peak, field_type="wraparound_field", plot=False)

        near_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="near_field", plot=False)
        before_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="before_field", plot=False)
        after_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_late = plot_cluster_peaks(clusters_dict_labelled_field_type_late_SST_peak, field_type="wraparound_field", plot=False)

        near_field_array_middle = plot_cluster_peaks(clusters_dict_labelled_field_type_middle_SST_peak, field_type="near_field", plot=False)
        before_field_array_middle = plot_cluster_peaks(clusters_dict_labelled_field_type_middle_SST_peak, field_type="before_field", plot=False)
        after_field_array_middle = plot_cluster_peaks(clusters_dict_labelled_field_type_middle_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_middle = plot_cluster_peaks(clusters_dict_labelled_field_type_middle_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_middle = plot_cluster_peaks(clusters_dict_labelled_field_type_middle_SST_peak, field_type="wraparound_field", plot=False)

        mean_near_field_array_early = np.mean(near_field_array_early, axis=0)
        sem_near_field_array_early = sem(near_field_array_early, axis=0)

        mean_before_field_array_early = np.mean(before_field_array_early, axis=0)
        sem_before_field_array_early = sem(before_field_array_early, axis=0)

        mean_after_field_array_early = np.mean(after_field_array_early, axis=0)
        sem_after_field_array_early = sem(after_field_array_early, axis=0)

        mean_noisy_field_array_early = np.mean(noisy_field_array_early, axis=0)
        sem_noisy_field_array_early = sem(noisy_field_array_early, axis=0)

        mean_wraparound_field_array_early = np.mean(wraparound_field_array_early, axis=0)
        sem_wraparound_field_array_early = sem(wraparound_field_array_early, axis=0)

        mean_near_field_array_late = np.mean(near_field_array_late, axis=0)
        sem_near_field_array_late = sem(near_field_array_late, axis=0)

        mean_before_field_array_late = np.mean(before_field_array_late, axis=0)
        sem_before_field_array_late = sem(before_field_array_late, axis=0)

        mean_after_field_array_late = np.mean(after_field_array_late, axis=0)
        sem_after_field_array_late = sem(after_field_array_late, axis=0)

        mean_noisy_field_array_late = np.mean(noisy_field_array_late, axis=0)
        sem_noisy_field_array_late = sem(noisy_field_array_late, axis=0)

        mean_wraparound_field_array_late = np.mean(wraparound_field_array_late, axis=0)
        sem_wraparound_field_array_late = sem(wraparound_field_array_late, axis=0)

        mean_near_field_array_middle = np.mean(near_field_array_middle, axis=0)
        sem_near_field_array_middle = sem(near_field_array_middle, axis=0)

        mean_before_field_array_middle = np.mean(before_field_array_middle, axis=0)
        sem_before_field_array_middle = sem(before_field_array_middle, axis=0)

        mean_after_field_array_middle = np.mean(after_field_array_middle, axis=0)
        sem_after_field_array_middle = sem(after_field_array_middle, axis=0)

        mean_noisy_field_array_middle = np.mean(noisy_field_array_middle, axis=0)
        sem_noisy_field_array_middle = sem(noisy_field_array_middle, axis=0)

        mean_wraparound_field_array_middle = np.mean(wraparound_field_array_middle, axis=0)
        sem_wraparound_field_array_middle = sem(wraparound_field_array_middle, axis=0)

        if plot:
            fig, axs = plt.subplots(6, 4, figsize=(15, 12))
            plt.suptitle(cell_type)

            if use_trough:
                axs[0, 0].set_title("Before Reward Early Trough")
                axs[0, 1].set_title("Near Reward Early Trough")
                axs[0, 2].set_title("After Reward Early Trough")
                axs[0, 3].set_title("Wraparound Early Trough")
            else:
                axs[0, 0].set_title("Before Reward Early Peak")
                axs[0, 1].set_title("Near Reward Early Peak")
                axs[0, 2].set_title("After Reward Early Peak")
                axs[0, 3].set_title("Wraparound Early Peak")

            axs[1, 0].imshow(before_field_array_early, aspect='auto')

            axs[1, 1].imshow(near_field_array_early, aspect='auto')

            axs[1, 2].imshow(after_field_array_early, aspect='auto')

            axs[1, 3].imshow(wraparound_field_array_early, aspect='auto')

            axs[0, 1].plot(mean_near_field_array_early, color='r')
            axs[0, 1].fill_between(range(len(mean_near_field_array_early)), mean_near_field_array_early - sem_near_field_array_early, mean_near_field_array_early + sem_near_field_array_early, color='r', alpha=0.2)

            axs[0, 0].plot(mean_before_field_array_early, color='b')
            axs[0, 0].fill_between(range(len(mean_before_field_array_early)), mean_before_field_array_early - sem_before_field_array_early, mean_before_field_array_early + sem_before_field_array_early, color='b', alpha=0.2)

            axs[0, 2].plot(mean_after_field_array_early, color='purple')
            axs[0, 2].fill_between(range(len(mean_after_field_array_early)), mean_after_field_array_early - sem_after_field_array_early, mean_after_field_array_early + sem_after_field_array_early, color='purple', alpha=0.2)

            axs[0, 3].plot(mean_wraparound_field_array_early, color='g')
            axs[0, 3].fill_between(range(len(mean_wraparound_field_array_early)), mean_wraparound_field_array_early - sem_wraparound_field_array_early, mean_wraparound_field_array_early + sem_wraparound_field_array_early, color='g', alpha=0.2)

            axs[2, 0].plot(mean_before_field_array_late, color='b')
            axs[2, 0].fill_between(range(len(mean_before_field_array_late)), mean_before_field_array_late - sem_before_field_array_late, mean_before_field_array_late + sem_before_field_array_late, color='b', alpha=0.2)

            axs[2, 1].plot(mean_near_field_array_late, color='r')
            axs[2, 1].fill_between(range(len(mean_near_field_array_late)), mean_near_field_array_late - sem_near_field_array_late, mean_near_field_array_late + sem_near_field_array_late, color='r', alpha=0.2)

            axs[2, 2].plot(mean_after_field_array_late, color='purple')
            axs[2, 2].fill_between(range(len(mean_after_field_array_late)), mean_after_field_array_late - sem_after_field_array_late, mean_after_field_array_late + sem_after_field_array_late, color='purple', alpha=0.2)

            axs[2, 3].plot(mean_wraparound_field_array_late, color='g')
            axs[2, 3].fill_between(range(len(mean_wraparound_field_array_late)), mean_wraparound_field_array_late - sem_wraparound_field_array_late, mean_wraparound_field_array_late + sem_wraparound_field_array_late, color='g', alpha=0.2)

            axs[4, 0].plot(mean_before_field_array_middle, color='b')
            axs[4, 0].fill_between(range(len(mean_before_field_array_middle)), mean_before_field_array_middle - sem_before_field_array_middle, mean_before_field_array_middle + sem_before_field_array_middle, color='b', alpha=0.2)

            axs[4, 1].plot(mean_near_field_array_middle, color='r')
            axs[4, 1].fill_between(range(len(mean_near_field_array_middle)), mean_near_field_array_middle - sem_near_field_array_middle, mean_near_field_array_middle + sem_near_field_array_middle, color='r', alpha=0.2)

            axs[4, 2].plot(mean_after_field_array_middle, color='purple')
            axs[4, 2].fill_between(range(len(mean_after_field_array_middle)), mean_after_field_array_middle - sem_after_field_array_middle, mean_after_field_array_middle + sem_after_field_array_middle, color='purple', alpha=0.2)

            axs[4, 3].plot(mean_wraparound_field_array_middle, color='g')
            axs[4, 3].fill_between(range(len(mean_wraparound_field_array_middle)), mean_wraparound_field_array_middle - sem_wraparound_field_array_middle, mean_wraparound_field_array_middle + sem_wraparound_field_array_middle, color='g', alpha=0.2)

            if use_trough:
                axs[2, 0].set_title("Before Reward Late Trough")
                axs[2, 1].set_title("Near Reward Late Trough")
                axs[2, 2].set_title("After Reward Late Trough")
                axs[2, 3].set_title("Wraparound Late Trough")
            else:
                axs[2, 0].set_title("Before Reward Late Peak")
                axs[2, 1].set_title("Near Reward Late Peak")
                axs[2, 2].set_title("After Reward Late Peak")
                axs[2, 3].set_title("Wraparound Late Peak")

            if use_trough:
                axs[4, 0].set_title("Before Reward Middle Trough")
                axs[4, 1].set_title("Near Reward Middle Trough")
                axs[4, 2].set_title("After Reward Middle Trough")
                axs[4, 3].set_title("Wraparound Middle Trough")
            else:
                axs[4, 0].set_title("Before Reward Middle Peak")
                axs[4, 1].set_title("Near Reward Middle Peak")
                axs[4, 2].set_title("After Reward Middle Peak")
                axs[4, 3].set_title("Wraparound Middle Peak")

            axs[3, 0].imshow(before_field_array_late, aspect='auto')

            axs[3, 1].imshow(near_field_array_late, aspect='auto')

            axs[3, 2].imshow(after_field_array_late, aspect='auto')

            axs[3, 3].imshow(wraparound_field_array_late, aspect='auto')

            axs[5, 0].imshow(before_field_array_middle, aspect='auto')

            axs[5, 1].imshow(near_field_array_middle, aspect='auto')

            axs[5, 2].imshow(after_field_array_middle, aspect='auto')

            axs[5, 3].imshow(wraparound_field_array_middle, aspect='auto')

            plt.tight_layout()
            plt.plot()

        else:
            return before_field_array_early, near_field_array_early, after_field_array_early, wraparound_field_array_early, before_field_array_late, near_field_array_late, after_field_array_late, wraparound_field_array_late, before_field_array_middle, near_field_array_middle, after_field_array_middle, wraparound_field_array_middle
    else:
        near_field_array_early = plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_peak, field_type="near_field", plot=False)
        before_field_array_early = plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_peak, field_type="before_field", plot=False)
        after_field_array_early = plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_early = plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_early = plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_peak, field_type="wraparound_field", plot=False)

        near_field_array_late = plot_cluster_peaks2(clusters_dict_labelled_field_type_late_SST_peak, field_type="near_field", plot=False)
        before_field_array_late = plot_cluster_peaks2(clusters_dict_labelled_field_type_late_SST_peak, field_type="before_field", plot=False)
        after_field_array_late = plot_cluster_peaks2(clusters_dict_labelled_field_type_late_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_late = plot_cluster_peaks2(clusters_dict_labelled_field_type_late_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_late = plot_cluster_peaks2(clusters_dict_labelled_field_type_late_SST_peak, field_type="wraparound_field", plot=False)

        near_field_array_middle = plot_cluster_peaks2(clusters_dict_labelled_field_type_middle_SST_peak, field_type="near_field", plot=False)
        before_field_array_middle = plot_cluster_peaks2(clusters_dict_labelled_field_type_middle_SST_peak, field_type="before_field", plot=False)
        after_field_array_middle = plot_cluster_peaks2(clusters_dict_labelled_field_type_middle_SST_peak, field_type="after_field", plot=False)
        noisy_field_array_middle = plot_cluster_peaks2(clusters_dict_labelled_field_type_middle_SST_peak, field_type="noisy_field", plot=False)
        wraparound_field_array_middle = plot_cluster_peaks2(clusters_dict_labelled_field_type_middle_SST_peak, field_type="wraparound_field", plot=False)

        return before_field_array_early, near_field_array_early, after_field_array_early, wraparound_field_array_early, before_field_array_late, near_field_array_late, after_field_array_late, wraparound_field_array_late, before_field_array_middle, near_field_array_middle, after_field_array_middle, wraparound_field_array_middle


def get_field_type_percents2(clusters_dict_labelled_field_type, cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, e_or_l="e"):
    """
    - get a count of the total number of each of our 4 categories appearing across trials for each cell and then divide by total num trials for that cell to get a probability of trial appearance
    """

    before_percent_list = []
    near_percent_list = []
    after_percent_list = []
    wraparound_percent_list = []
    noisy_percent_list = []

    animal_percent_dict = {}

    counter = 0
    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, animal_TCA=False)

    resid_dict_2 = {}
    for idx, animal in enumerate(residual_activity_dict_EC):
        cell_dict_2 = {}
        for idt, cell in enumerate(residual_activity_dict_EC[animal]):
            cell_dict_2[idt] = residual_activity_dict_EC[animal][cell]
            counter += 1

        resid_dict_2[idx] = cell_dict_2

    for idx, animal in enumerate(clusters_dict_labelled_field_type):
        cell_dict = {}

        for idt, cell in enumerate(clusters_dict_labelled_field_type[animal]):

            cp_e = animal_first_changepoints_list[idx][idt]
            cp_l = animal_second_changepoints_list[idx][idt]
            tots = resid_dict_2[idx][idt].shape[1]

            total_num_trials_late = (tots - cp_l) - 1

            total_num_trials_middle = tots - (cp_e + total_num_trials_late) - 2

            print(f"cp_e {cp_e} cp_l {cp_l} tots {tots} tots_late {total_num_trials_late} tots_mid {total_num_trials_middle}")

            if e_or_l == "e":
                total_num_trials = cp_e
            elif e_or_l == "l":
                total_num_trials = total_num_trials_late
            else:
                total_num_trials = total_num_trials_middle

                # Initialize field type counts
            before_count = 0
            near_count = 0
            after_count = 0
            wraparound_count = 0
            noisy_count = 0

            for i in clusters_dict_labelled_field_type[animal][cell]:
                for field_type, trial_array in clusters_dict_labelled_field_type[animal][cell][i].items():
                    count = len(trial_array)
                    if field_type == "before_field":
                        before_count += count
                    elif field_type == "near_field":
                        near_count += count
                    elif field_type == "after_field":
                        after_count += count
                    elif field_type == "wraparound_field":
                        wraparound_count += count
                    elif field_type == "noisy_field":
                        noisy_count += count

            #             print(f"counter {counter} np.sum {np.sum([before_count, near_count, after_count, wraparound_count, noisy_count])} totals trials {total_num_trials}")

            counted_total_num_trials = before_count + near_count + after_count + wraparound_count + noisy_count
            print(f"counted_total_num_trials {counted_total_num_trials}")

            if total_num_trials > 0:
                before_percent = before_count / total_num_trials * 100
                near_percent = near_count / total_num_trials * 100
                after_percent = after_count / total_num_trials * 100
                wraparound_percent = wraparound_count / total_num_trials * 100
                noisy_percent = noisy_count / total_num_trials * 100

                before_percent_list.append(before_percent)
                near_percent_list.append(near_percent)
                after_percent_list.append(after_percent)
                wraparound_percent_list.append(wraparound_percent)
                noisy_percent_list.append(noisy_percent)

                cell_dict[cell] = {
                    "before_percent": before_percent,
                    "near_percent": near_percent,
                    "after_percent": after_percent,
                    "wraparound_percent": wraparound_percent,
                    "noisy_percent": noisy_percent
                }

        #                 print(f"{animal} {cell} → total trials: {total_num_trials} total labeled: {before_count + near_count + after_count + wraparound_count + noisy_count}")

        animal_percent_dict[animal] = cell_dict

    percent_lists = [
        before_percent_list,
        near_percent_list,
        after_percent_list,
        wraparound_percent_list,
        noisy_percent_list,
    ]

    return animal_percent_dict, percent_lists


def get_cells_percents(animal_percent_dict_EC_early_peak, animal_percent_dict_EC_late_peak, animal_percent_dict_EC_middle_peak, animal_percent_dict_EC_early_trough, animal_percent_dict_EC_late_trough, animal_percent_dict_EC_middle_trough):
    count_early_peak_EC_before = get_count(animal_percent_dict_EC_early_peak, field_type="before_percent")
    count_late_peak_EC_before = get_count(animal_percent_dict_EC_late_peak, field_type="before_percent")
    count_middle_peak_EC_before = get_count(animal_percent_dict_EC_middle_peak, field_type="before_percent")

    count_early_trough_EC_before = get_count(animal_percent_dict_EC_early_trough, field_type="before_percent")
    count_late_trough_EC_before = get_count(animal_percent_dict_EC_late_trough, field_type="before_percent")
    count_middle_trough_EC_before = get_count(animal_percent_dict_EC_middle_trough, field_type="before_percent")

    count_early_peak_EC_near = get_count(animal_percent_dict_EC_early_peak, field_type="near_percent")
    count_late_peak_EC_near = get_count(animal_percent_dict_EC_late_peak, field_type="near_percent")
    count_middle_peak_EC_near = get_count(animal_percent_dict_EC_middle_peak, field_type="near_percent")

    count_early_trough_EC_near = get_count(animal_percent_dict_EC_early_trough, field_type="near_percent")
    count_late_trough_EC_near = get_count(animal_percent_dict_EC_late_trough, field_type="near_percent")
    count_middle_trough_EC_near = get_count(animal_percent_dict_EC_middle_trough, field_type="near_percent")

    count_early_peak_EC_after = get_count(animal_percent_dict_EC_early_peak, field_type="after_percent")
    count_late_peak_EC_after = get_count(animal_percent_dict_EC_late_peak, field_type="after_percent")
    count_middle_peak_EC_after = get_count(animal_percent_dict_EC_middle_peak, field_type="after_percent")

    count_early_trough_EC_after = get_count(animal_percent_dict_EC_early_trough, field_type="after_percent")
    count_late_trough_EC_after = get_count(animal_percent_dict_EC_late_trough, field_type="after_percent")
    count_middle_trough_EC_after = get_count(animal_percent_dict_EC_middle_trough, field_type="after_percent")

    count_early_peak_EC_wraparound = get_count(animal_percent_dict_EC_early_peak, field_type="wraparound_percent")
    count_late_peak_EC_wraparound = get_count(animal_percent_dict_EC_late_peak, field_type="wraparound_percent")
    count_middle_peak_EC_wraparound = get_count(animal_percent_dict_EC_middle_peak, field_type="wraparound_percent")

    count_early_trough_EC_wraparound = get_count(animal_percent_dict_EC_early_trough, field_type="wraparound_percent")
    count_late_trough_EC_wraparound = get_count(animal_percent_dict_EC_late_trough, field_type="wraparound_percent")
    count_middle_trough_EC_wraparound = get_count(animal_percent_dict_EC_middle_trough, field_type="wraparound_percent")

    count_early_peak_EC_noisy = get_count(animal_percent_dict_EC_early_peak, field_type="noisy_percent")
    count_late_peak_EC_noisy = get_count(animal_percent_dict_EC_late_peak, field_type="noisy_percent")
    count_middle_peak_EC_noisy = get_count(animal_percent_dict_EC_middle_peak, field_type="noisy_percent")

    count_early_trough_EC_noisy = get_count(animal_percent_dict_EC_early_trough, field_type="noisy_percent")
    count_late_trough_EC_noisy = get_count(animal_percent_dict_EC_late_trough, field_type="noisy_percent")
    count_middle_trough_EC_noisy = get_count(animal_percent_dict_EC_middle_trough, field_type="noisy_percent")

    percent_of_cells_list_EC_early_peak = [count_early_peak_EC_before, count_early_peak_EC_near, count_early_peak_EC_after, count_early_peak_EC_wraparound, count_early_peak_EC_noisy]
    percent_of_cells_list_EC_late_peak = [count_late_peak_EC_before, count_late_peak_EC_near, count_late_peak_EC_after, count_late_peak_EC_wraparound, count_late_peak_EC_noisy]
    percent_of_cells_list_EC_middle_peak = [count_middle_peak_EC_before, count_middle_peak_EC_near, count_middle_peak_EC_after, count_middle_peak_EC_wraparound, count_middle_peak_EC_noisy]

    percent_of_cells_list_EC_early_trough = [count_early_trough_EC_before, count_early_trough_EC_near, count_early_trough_EC_after, count_early_trough_EC_wraparound, count_early_trough_EC_noisy]
    percent_of_cells_list_EC_late_trough = [count_late_trough_EC_before, count_late_trough_EC_near, count_late_trough_EC_after, count_late_trough_EC_wraparound, count_late_trough_EC_noisy]
    percent_of_cells_list_EC_middle_trough = [count_middle_trough_EC_before, count_middle_trough_EC_near, count_middle_trough_EC_after, count_middle_trough_EC_wraparound, count_middle_trough_EC_noisy]

    return percent_of_cells_list_EC_early_peak, percent_of_cells_list_EC_late_peak, percent_of_cells_list_EC_middle_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, percent_of_cells_list_EC_middle_trough

def random_timeseries2(initial_value: float, volatility: float, count: int, rng):
    time_series = [initial_value]
    for _ in range(count):
        noise = rng.normal(0, 1)  # Use NumPy's RNG
        next_val = time_series[-1] + initial_value * noise * volatility
        time_series.append(next_val)
    return time_series

def get_means_array(final_weighted_means_per_cell_before_early):
    final_weighted_means_per_cell_before_early_array = np.array(final_weighted_means_per_cell_before_early)
    mean_final_weighted_means_per_cell_before_early_array = np.mean(final_weighted_means_per_cell_before_early_array, axis=0)
    return mean_final_weighted_means_per_cell_before_early_array



def reconstruct_activity_from_clusters_every_cell_diff_latent(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, before_field_array_early_trough_EC_dict, near_field_array_early_trough_EC_dict, after_field_array_early_trough_EC_dict, wraparound_field_array_early_trough_EC_dict, before_field_array_late_trough_EC_dict, near_field_array_late_trough_EC_dict,
                                                              after_field_array_late_trough_EC_dict, wraparound_field_array_late_trough_EC_dict, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, inits, vol, cell_type="EC Trough", plot=False, seed=42):
    """
    - same function as the other reconstruction except we are no longer reusing the same across-cell average trace slotted into every trial across all cells, now we are using a different average trace for each cell's trials but it is just a single trace per trial
    - uses the function get_weighted_activityies_per_cell in case there were multiple of the same type of field per cell (ex / 2 clusters that were both before fields with different kinetics) and gets a weighted average of them to hand into the before trials for that given cell
    """

    rng = np.random.default_rng(seed)

    start = time.time()

    num_cells = len(percent_lists_EC_early_peak[0])

    before_percent_array = np.array(percent_lists_EC_early_peak[0])
    near_percent_array = np.array(percent_lists_EC_early_peak[1])
    after_percent_array = np.array(percent_lists_EC_early_peak[2])
    wraparound_percent_array = np.array(percent_lists_EC_early_peak[3])

    before_percent_array_late = np.array(percent_lists_EC_late_peak[0])
    near_percent_array_late = np.array(percent_lists_EC_late_peak[1])
    after_percent_array_late = np.array(percent_lists_EC_late_peak[2])
    wraparound_percent_array_late = np.array(percent_lists_EC_late_peak[3])

    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, animal_TCA=False)

    flat_list_first = [item for sublist in animal_first_changepoints_list for item in sublist]
    flat_list_second = [item for sublist in animal_second_changepoints_list for item in sublist]

    stack_num_trials_list = np.vstack([flat_list_first, flat_list_second])

    num_trials_list = np.sum(stack_num_trials_list, axis=0)

    available_cells_before_early = percent_of_cells_list_EC_early_trough[0] * len(num_trials_list)
    available_cells_near_early = percent_of_cells_list_EC_early_trough[1] * len(num_trials_list)
    available_cells_after_early = percent_of_cells_list_EC_early_trough[2] * len(num_trials_list)
    available_cells_wraparound_early = percent_of_cells_list_EC_early_trough[3] * len(num_trials_list)

    cells_with_before_field_early = rng.choice(num_cells, int(available_cells_before_early), replace=False)
    cells_with_near_field_early = rng.choice(num_cells, int(available_cells_near_early), replace=False)
    cells_with_after_field_early = rng.choice(num_cells, int(available_cells_after_early), replace=False)
    cells_with_wraparound_field_early = rng.choice(num_cells, int(available_cells_wraparound_early), replace=False)

    available_cells_before_late = percent_of_cells_list_EC_late_trough[0] * len(num_trials_list)
    available_cells_near_late = percent_of_cells_list_EC_late_trough[1] * len(num_trials_list)
    available_cells_after_late = percent_of_cells_list_EC_late_trough[2] * len(num_trials_list)
    available_cells_wraparound_late = percent_of_cells_list_EC_late_trough[3] * len(num_trials_list)

    cells_with_before_field_late = rng.choice(num_cells, int(available_cells_before_late), replace=False)
    cells_with_near_field_late = rng.choice(num_cells, int(available_cells_near_late), replace=False)
    cells_with_after_field_late = rng.choice(num_cells, int(available_cells_after_late), replace=False)
    cells_with_wraparound_field_late = rng.choice(num_cells, int(available_cells_wraparound_late), replace=False)

    cell_array_early_list = []
    cell_array_late_list = []

    for cell in range(len(num_trials_list)):
        rt_early = np.array(random_timeseries2(initial_value=inits, volatility=vol, count=49, rng=rng))
        rt_late = np.array(random_timeseries2(initial_value=inits, volatility=vol, count=49, rng=rng))

        num_trials_early = flat_list_first[cell]
        num_trials_late = num_trials_list[cell] - flat_list_second[cell]

        cell_array = np.tile(rt_early[:, np.newaxis], (1, num_trials_early))
        cell_array_early_list.append(cell_array)
        cell_array_late = np.tile(rt_late[:, np.newaxis], (1, num_trials_late))
        cell_array_late_list.append(cell_array_late)

    trial_indices_before_dict_early = {}
    trial_indices_near_dict_early = {}
    trial_indices_after_dict_early = {}
    trial_indices_wraparound_dict_early = {}

    trial_indices_before_dict_late = {}
    trial_indices_near_dict_late = {}
    trial_indices_after_dict_late = {}
    trial_indices_wraparound_dict_late = {}

    for cell in range(len(num_trials_list)):
        num_trials_early = flat_list_first[cell]
        num_trials_late = num_trials_list[cell] - flat_list_second[cell]

        available_trials_early = np.arange(num_trials_early)
        remaining_trials_early = available_trials_early.copy()

        available_trials_late = np.arange(num_trials_late)
        remaining_trials_late = available_trials_late.copy()

        if cell in cells_with_before_field_early:
            proportion_early_before = (before_percent_array[cell] / 100) * num_trials_early
            trial_indices_before = rng.choice(available_trials_early, int(proportion_early_before), replace=False)
            trial_indices_before_dict_early[cell] = trial_indices_before

        if cell in cells_with_near_field_early:
            proportion_early_near = (near_percent_array[cell] / 100) * num_trials_early
            trial_indices_near = rng.choice(available_trials_early, int(proportion_early_near), replace=False)
            trial_indices_near_dict_early[cell] = trial_indices_near

        if cell in cells_with_after_field_early:
            proportion_early_after = (after_percent_array[cell] / 100) * num_trials_early
            trial_indices_after = rng.choice(available_trials_early, int(proportion_early_after), replace=False)
            trial_indices_after_dict_early[cell] = trial_indices_after

        if cell in cells_with_wraparound_field_early:
            proportion_early_wraparound = (wraparound_percent_array[cell] / 100) * num_trials_early
            trial_indices_wraparound = rng.choice(available_trials_early, int(proportion_early_wraparound), replace=False)
            trial_indices_wraparound_dict_early[cell] = trial_indices_wraparound

        if cell in cells_with_before_field_late:
            proportion_late_before = (before_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_before = rng.choice(available_trials_late, int(proportion_late_before), replace=False)
            trial_indices_before_dict_late[cell] = trial_indices_before

        if cell in cells_with_near_field_late:
            proportion_late_near = (near_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_near = rng.choice(available_trials_late, int(proportion_late_near), replace=False)
            trial_indices_near_dict_late[cell] = trial_indices_near

        if cell in cells_with_after_field_late:
            proportion_late_after = (after_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_after = rng.choice(available_trials_late, int(proportion_late_after), replace=False)
            trial_indices_after_dict_late[cell] = trial_indices_after

        if cell in cells_with_wraparound_field_late:
            proportion_late_wraparound = (wraparound_percent_array_late[cell] / 100) * num_trials_late
            trial_indices_wraparound = rng.choice(available_trials_late, int(proportion_late_wraparound), replace=False)
            trial_indices_wraparound_dict_late[cell] = trial_indices_wraparound

    trial_indices_before_dict_early, trial_indices_near_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_near_field_early, trial_indices_before_dict_early, trial_indices_near_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_after_field_early, trial_indices_near_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_after_field_early, trial_indices_before_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_wraparound_field_early, trial_indices_before_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_wraparound_field_early, trial_indices_near_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_after_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_after_field_early, cells_with_wraparound_field_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early, rng)

    final_weighted_means_per_cell_before_early = get_weighted_activityies_per_cell(before_field_array_early_trough_EC_dict, field_type="before_field")
    final_weighted_means_per_cell_near_early = get_weighted_activityies_per_cell(near_field_array_early_trough_EC_dict, field_type="near_field")
    final_weighted_means_per_cell_after_early = get_weighted_activityies_per_cell(after_field_array_early_trough_EC_dict, field_type="after_field")
    final_weighted_means_per_cell_wraparound_early = get_weighted_activityies_per_cell(wraparound_field_array_early_trough_EC_dict, field_type="wraparound_field")

    count_before_early = 0
    for cell, trial_indices in trial_indices_before_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["before"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_before_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_before_early[count_before_early]
            else:
                print("improper alignment")
        count_before_early += 1

    count_near_early = 0
    for cell, trial_indices in trial_indices_near_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["near"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_near_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_near_early[count_near_early]
            else:
                print("improper alignment")

        count_near_early += 1

    count_after_early = 0
    for cell, trial_indices in trial_indices_after_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["after"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_after_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_after_early[count_after_early]
            else:
                print("improper alignment")

        count_after_early += 1

    count_wraparound_early = 0
    for cell, trial_indices in trial_indices_wraparound_dict_early.items():
        #     for cell, trial_indices in trial_dicts_early["wraparound"].items():
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_wraparound_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_wraparound_early[count_wraparound_early]
            else:
                print("improper alignment")

        count_wraparound_early += 1

    trial_indices_before_dict_late, trial_indices_near_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_near_field_late, trial_indices_before_dict_late, trial_indices_near_dict_late, rng)
    trial_indices_near_dict_late, trial_indices_after_dict_late = remove_duplicate_trials(cells_with_near_field_late, cells_with_after_field_late, trial_indices_near_dict_late, trial_indices_after_dict_late, rng)
    trial_indices_before_dict_late, trial_indices_after_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_after_field_late, trial_indices_before_dict_late, trial_indices_after_dict_late, rng)
    trial_indices_before_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_before_field_late, cells_with_wraparound_field_late, trial_indices_before_dict_late, trial_indices_wraparound_dict_late, rng)
    trial_indices_near_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_near_field_late, cells_with_wraparound_field_late, trial_indices_near_dict_late, trial_indices_wraparound_dict_late, rng)
    trial_indices_after_dict_late, trial_indices_wraparound_dict_late = remove_duplicate_trials(cells_with_after_field_late, cells_with_wraparound_field_late, trial_indices_after_dict_late, trial_indices_wraparound_dict_late, rng)

    final_weighted_means_per_cell_before_late = get_weighted_activityies_per_cell(before_field_array_late_trough_EC_dict, field_type="before_field")
    final_weighted_means_per_cell_near_late = get_weighted_activityies_per_cell(near_field_array_late_trough_EC_dict, field_type="near_field")
    final_weighted_means_per_cell_after_late = get_weighted_activityies_per_cell(after_field_array_late_trough_EC_dict, field_type="after_field")
    final_weighted_means_per_cell_wraparound_late = get_weighted_activityies_per_cell(wraparound_field_array_late_trough_EC_dict, field_type="wraparound_field")

    count_before_late = 0
    for cell, trial_indices in trial_indices_before_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["before"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                #                 cell_array_late_list[cell][:, t] = example_before_field_late
                cell_array_late_list[cell][:, t] = final_weighted_means_per_cell_before_late[count_before_late]
            else:
                print("improper alignment")
        count_before_late += 1

    count_near_late = 0
    for cell, trial_indices in trial_indices_near_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["near"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                #                 cell_array_late_list[cell][:, t] = example_near_field_late
                cell_array_late_list[cell][:, t] = final_weighted_means_per_cell_near_late[count_near_late]
            else:
                print("improper alignment")

        count_near_late += 1

    count_after_late = 0
    for cell, trial_indices in trial_indices_after_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["after"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                #                 cell_array_late_list[cell][:, t] = example_after_field_late
                cell_array_late_list[cell][:, t] = final_weighted_means_per_cell_after_late[count_after_late]
            else:
                print("improper alignment")

        count_after_late += 1

    count_wraparound_late = 0
    for cell, trial_indices in trial_indices_wraparound_dict_late.items():
        #     for cell, trial_indices in trial_dicts_late["wraparound"].items():
        for t in trial_indices:
            if t < cell_array_late_list[cell].shape[1]:
                #                 cell_array_late_list[cell][:, t] = example_wraparound_field_late
                cell_array_late_list[cell][:, t] = final_weighted_means_per_cell_wraparound_late[count_wraparound_late]
            else:
                print("improper alignment")

        count_wraparound_late += 1

    full_activity_list = []
    full_activity_trial_av_list = []

    trial_averaged_cell_list_early = []
    trial_averaged_cell_list_late = []

    for i in range(len(cell_array_early_list)):
        early_component = cell_array_early_list[i]
        late_component = cell_array_late_list[i]

        full_actiivty = np.concatenate([early_component, late_component], axis=1)

        full_activity_list.append(full_actiivty)

        full_activity_trial_av_list.append(np.mean(full_actiivty, axis=1))

        full_activity_trial_av_array = np.array(full_activity_trial_av_list)

        trial_averaged_cell_list_early.append(np.mean(early_component, axis=1))
        trial_averaged_cell_list_late.append(np.mean(late_component, axis=1))

    trial_averaged_cell_array_early = np.array(trial_averaged_cell_list_early)
    trial_averaged_cell_array_late = np.array(trial_averaged_cell_list_late)

    if plot:
        fig, axs = plt.subplots(6, 4, figsize=(15, 20))
        plt.suptitle(cell_type)

        #         def get_means_array(final_weighted_means_per_cell_before_early):
        #             final_weighted_means_per_cell_before_early_array = np.array(final_weighted_means_per_cell_before_early)
        #             mean_final_weighted_means_per_cell_before_early_array = np.mean(final_weighted_means_per_cell_before_early_array, axis=1)
        #             return mean_final_weighted_means_per_cell_before_early_array

        mean_final_weighted_means_per_cell_before_early_array = get_means_array(final_weighted_means_per_cell_before_early)

        for i in range(len(final_weighted_means_per_cell_before_early)):
            axs[0, 0].plot(final_weighted_means_per_cell_before_early[i], alpha=0.2)
        axs[0, 0].plot(mean_final_weighted_means_per_cell_before_early_array, color='r', linestyle='--', label='mean')
        axs[0, 0].set_title("Inputs Before Field Early")
        axs[0, 0].set_ylabel("Activity")
        axs[0, 0].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_before_late = get_means_array(final_weighted_means_per_cell_before_late)

        for i in range(len(final_weighted_means_per_cell_before_late)):
            axs[1, 0].plot(final_weighted_means_per_cell_before_late[i], alpha=0.2)
        axs[1, 0].plot(mean_final_weighted_means_per_cell_before_late, color='r', linestyle='--', label='mean')
        axs[1, 0].set_title("Inputs Before Field Late")
        axs[1, 0].set_ylabel("Activity")
        axs[1, 0].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_near_early = get_means_array(final_weighted_means_per_cell_near_early)

        for i in range(len(final_weighted_means_per_cell_near_early)):
            axs[0, 1].plot(final_weighted_means_per_cell_near_early[i], alpha=0.2)
        axs[0, 1].plot(mean_final_weighted_means_per_cell_near_early, color='r', linestyle='--', label='mean')
        axs[0, 1].set_title("Inputs Near Field Early")
        axs[0, 1].set_ylabel("Activity")
        axs[0, 1].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_near_late = get_means_array(final_weighted_means_per_cell_near_late)

        for i in range(len(final_weighted_means_per_cell_near_late)):
            axs[1, 1].plot(final_weighted_means_per_cell_near_late[i], alpha=0.2)
        axs[1, 1].plot(mean_final_weighted_means_per_cell_near_late, color='r', linestyle='--', label='mean')
        axs[1, 1].set_title("Inputs Near Field Late")
        axs[1, 1].set_ylabel("Activity")
        axs[1, 1].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_after_early = get_means_array(final_weighted_means_per_cell_after_early)

        for i in range(len(final_weighted_means_per_cell_after_early)):
            axs[0, 2].plot(final_weighted_means_per_cell_after_early[i], alpha=0.2)
        axs[0, 2].plot(mean_final_weighted_means_per_cell_after_early, color='r', linestyle='--', label='mean')
        axs[0, 2].set_title("Inputs After Field Early")
        axs[0, 2].set_ylabel("Activity")
        axs[0, 2].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_after_late = get_means_array(final_weighted_means_per_cell_after_late)

        for i in range(len(final_weighted_means_per_cell_after_late)):
            axs[1, 2].plot(final_weighted_means_per_cell_after_late[i], alpha=0.2)
        axs[1, 2].plot(mean_final_weighted_means_per_cell_after_late, color='r', linestyle='--', label='mean')
        axs[1, 2].set_title("Inputs After Field Late")
        axs[1, 2].set_ylabel("Activity")
        axs[1, 2].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_wraparound_early = get_means_array(final_weighted_means_per_cell_wraparound_early)

        for i in range(len(final_weighted_means_per_cell_wraparound_early)):
            axs[0, 3].plot(final_weighted_means_per_cell_wraparound_early[i], alpha=0.2)
        axs[0, 3].plot(mean_final_weighted_means_per_cell_wraparound_early, color='r', linestyle='--', label='mean')
        axs[0, 3].set_title("Inputs Wraparound Field Early")
        axs[0, 3].set_ylabel("Activity")
        axs[0, 3].set_xlabel("Position Bins")

        mean_final_weighted_means_per_cell_wraparound_late = get_means_array(final_weighted_means_per_cell_wraparound_late)

        for i in range(len(final_weighted_means_per_cell_wraparound_late)):
            axs[1, 3].plot(final_weighted_means_per_cell_wraparound_late[i], alpha=0.2)
        axs[1, 3].plot(mean_final_weighted_means_per_cell_wraparound_late, color='r', linestyle='--', label='mean')
        axs[1, 3].set_title("Inputs Wraparound Field Late")
        axs[1, 3].set_ylabel("Activity")
        axs[1, 3].set_xlabel("Position Bins")

        axs[2, 0].hist(before_percent_array, bins=20)
        axs[2, 0].set_title(f"Early Before Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[0] * 100:.1f}%")
        axs[2, 0].set_ylabel("Number of Cells")
        axs[2, 0].set_xlabel("Percent of Trials")

        axs[2, 1].hist(near_percent_array, bins=20)
        axs[2, 1].set_title(f"Early Near Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[1] * 100:.1f}%")
        axs[2, 1].set_ylabel("Number of Cells")
        axs[2, 1].set_xlabel("Percent of Trials")

        axs[2, 2].hist(after_percent_array, bins=20)
        axs[2, 2].set_title(f"Early After Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[2] * 100:.1f}%")
        axs[2, 2].set_ylabel("Number of Cells")
        axs[2, 2].set_xlabel("Percent of Trials")

        axs[2, 3].hist(wraparound_percent_array, bins=20)
        axs[2, 3].set_title(f"Early Wraparound Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[3] * 100:.1f}%")
        axs[2, 3].set_ylabel("Number of Cells")
        axs[2, 3].set_xlabel("Percent of Trials")

        axs[3, 0].hist(before_percent_array_late, bins=20)
        #         ax[2,0].set_title(f"Late Before Field Probability \n Percent of Cells={available_trials_before_late:.1f}%")
        axs[3, 0].set_title("Late Before Field Probability")
        axs[3, 0].set_ylabel("Number of Cells")
        axs[3, 0].set_xlabel("Percent of Trials")

        axs[3, 1].hist(near_percent_array_late, bins=20)
        #         axs[2,1].set_title(f"Late Near Field Probability \n Percent of Cells={available_trials_near_late:.1f}%")
        axs[3, 1].set_title("Late Near Field Probability")
        axs[3, 1].set_ylabel("Number of Cells")
        axs[3, 1].set_xlabel("Percent of Trials")

        axs[3, 2].hist(after_percent_array_late, bins=20)
        #         ax[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
        axs[3, 2].set_title("Late After Field Probability")
        axs[3, 2].set_ylabel("Number of Cells")
        axs[3, 2].set_xlabel("Percent of Trials")

        axs[3, 3].hist(wraparound_percent_array_late, bins=20)
        #         axs[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
        axs[3, 3].set_title("Late Wraparound Field Probability")
        axs[3, 3].set_ylabel("Number of Cells")
        axs[3, 3].set_xlabel("Percent of Trials")

        axs[5, 0].imshow(trial_averaged_cell_array_early, aspect='auto')
        axs[5, 0].set_ylabel("Cell ID")
        axs[5, 0].set_xlabel("Position Bin")

        means_early = np.mean(trial_averaged_cell_array_early, axis=0)
        sems_early = sem(trial_averaged_cell_array_early, axis=0)
        axs[4, 0].plot(means_early, color='orange')
        axs[4, 0].fill_between(range(len(means_early)), means_early + sems_early, means_early - sems_early, alpha=0.2, color='orange')
        axs[4, 0].set_xlabel("Position Bins")
        axs[4, 0].set_ylabel("Activity")
        axs[4, 0].set_title("Generative Early Learn")

        axs[5, 1].imshow(trial_averaged_cell_array_late, aspect='auto')
        axs[5, 1].set_ylabel("Cell ID")
        axs[5, 1].set_xlabel("Position Bin")

        means_late = np.mean(trial_averaged_cell_array_late, axis=0)
        sems_late = sem(trial_averaged_cell_array_late, axis=0)
        axs[4, 1].plot(means_late, color='r')
        axs[4, 1].fill_between(range(len(means_late)), means_late + sems_late, means_late - sems_late, alpha=0.2, color='r')
        axs[4, 1].set_xlabel("Position Bins")
        axs[4, 1].set_ylabel("Activity")
        axs[4, 1].set_title("Generative Late Learn")

        axs[5, 2].imshow(full_activity_trial_av_array, aspect='auto')
        axs[5, 2].set_ylabel("Cell ID")
        axs[5, 2].set_xlabel("Position Bin")

        means_overall = np.mean(full_activity_trial_av_list, axis=0)
        sems_overall = sem(full_activity_trial_av_list, axis=0)
        axs[4, 2].plot(means_overall, color='purple')
        axs[4, 2].fill_between(range(len(means_overall)), means_overall + sems_overall, means_overall - sems_overall, alpha=0.2, color='purple')
        axs[4, 2].set_xlabel("Position Bins")
        axs[4, 2].set_ylabel("Activity")
        axs[4, 2].set_title("Generative Overall Average")

        plt.tight_layout()
        plt.show()
    else:
        return cell_array_early_list, cell_array_late_list, full_activity_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, full_activity_trial_av_array

def get_weighted_activityies_per_cell(field_dict, field_type="before_field"):
    final_weighted_means_per_cell = []

    for animal in field_dict:
        for cell in field_dict[animal]:
            field_list = field_dict[animal][cell].get(field_type, [])
            if len(field_list) == 0:
                continue

            # Extract weights and field means in a single pass
            weights = []
            field_means = []
            for arr in field_list:
                if arr.shape[0] == 0:
                    continue  # skip empty arrays
                weights.append(arr.shape[0])
                field_means.append(np.mean(arr, axis=0))

            if len(weights) == 0:
                continue  # skip if all were empty

            weights_np = np.array(weights)
            weights_np = weights_np / weights_np.sum()  # normalize

            stacked = np.vstack(field_means)  # (n_trials, 50)
            weighted_mean = np.average(stacked, axis=0, weights=weights_np)
            final_weighted_means_per_cell.append(weighted_mean)

    return final_weighted_means_per_cell

def plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_trough, field_type="near_field", plot=False):

    near_field_list = []
    near_field_dict = {}

    for animal in clusters_dict_labelled_field_type_early_SST_trough:
        cell_dict = {}
        for cell in clusters_dict_labelled_field_type_early_SST_trough[animal]:
            field_dict = {}
            just_this_cell_list = []
            for i in clusters_dict_labelled_field_type_early_SST_trough[animal][cell]:
                entry = clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i]
                if field_type in entry:
                    arr = entry[field_type]
                    if len(arr) == 0:
                        continue  # skip empty arrays
                if field_type in clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i]:
                    means = clusters_dict_labelled_field_type_early_SST_trough[animal][cell][i][field_type]

                    near_field_list.append(means)
                    just_this_cell_list.append(means)

                field_dict[field_type] = just_this_cell_list
            cell_dict[cell] = field_dict
        near_field_dict[animal] = cell_dict


    if plot:

        for i in near_field_list:
            plt.plot(i, color='gray')
        plt.plot(np.mean(near_field_array, axis=0), color='red')
        plt.show()

        mean_near_field_array = np.mean(near_field_array, axis=0)
        sem_near_field_array = sem(near_field_array, axis=0)

        plt.plot(mean_near_field_array)
        plt.fill_between(range(len(mean_near_field_array)), mean_near_field_array-sem_near_field_array, mean_near_field_array+sem_near_field_array, alpha=0.2)

    return near_field_dict


def reconstruct_activity_from_clusters_every_cell_diff_latent_split(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, early_data_list_dict, late_data_list_dict, middle_data_list_dict, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_lists_EC_middle_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, percent_of_cells_list_EC_middle_trough,
                                                                    inits, vol, cell_type="EC Trough", plot=False, seed=42, num_t_e=20, num_t_m=40, num_t_l=20):
    """
    - same function as the other reconstruction except we are no longer reusing the same across-cell average trace slotted into every trial across all cells, now we are using a different average trace for each cell's trials but it is just a single trace per trial
    - uses the function get_weighted_activityies_per_cell in case there were multiple of the same type of field per cell (ex / 2 clusters that were both before fields with different kinetics) and gets a weighted average of them to hand into the before trials for that given cell
    """

    rng = np.random.default_rng(seed)

    before_field_array_early_trough_EC_dict = early_data_list_dict[0]
    near_field_array_early_trough_EC_dict = early_data_list_dict[1]
    after_field_array_early_trough_EC_dict = early_data_list_dict[2]
    wraparound_field_array_early_trough_EC_dict = early_data_list_dict[3]

    before_field_array_late_trough_EC_dict = late_data_list_dict[0]
    near_field_array_late_trough_EC_dict = late_data_list_dict[1]
    after_field_array_late_trough_EC_dict = late_data_list_dict[2]
    wraparound_field_array_late_trough_EC_dict = late_data_list_dict[3]

    before_field_array_middle_trough_EC_dict = middle_data_list_dict[0]
    near_field_array_middle_trough_EC_dict = middle_data_list_dict[1]
    after_field_array_middle_trough_EC_dict = middle_data_list_dict[2]
    wraparound_field_array_middle_trough_EC_dict = middle_data_list_dict[3]

    num_cells = len(percent_lists_EC_early_peak[0])

    before_percent_array = np.array(percent_lists_EC_early_peak[0])
    near_percent_array = np.array(percent_lists_EC_early_peak[1])
    after_percent_array = np.array(percent_lists_EC_early_peak[2])
    wraparound_percent_array = np.array(percent_lists_EC_early_peak[3])

    before_percent_array_late = np.array(percent_lists_EC_late_peak[0])
    near_percent_array_late = np.array(percent_lists_EC_late_peak[1])
    after_percent_array_late = np.array(percent_lists_EC_late_peak[2])
    wraparound_percent_array_late = np.array(percent_lists_EC_late_peak[3])

    before_percent_array_middle = np.array(percent_lists_EC_middle_peak[0])
    near_percent_array_middle = np.array(percent_lists_EC_middle_peak[1])
    after_percent_array_middle = np.array(percent_lists_EC_middle_peak[2])
    wraparound_percent_array_middle = np.array(percent_lists_EC_middle_peak[3])

    animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, animal_TCA=False)

    flat_list_first = [item for sublist in animal_first_changepoints_list for item in sublist]
    flat_list_second = [item for sublist in animal_second_changepoints_list for item in sublist]

    stack_num_trials_list = np.vstack([flat_list_first, flat_list_second])

    num_trials_list = np.sum(stack_num_trials_list, axis=0)

    cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early = randomly_pick_cells(percent_of_cells_list_EC_early_trough, num_trials_list, num_cells, rng)
    cells_with_before_field_late, cells_with_near_field_late, cells_with_after_field_late, cells_with_wraparound_field_late = randomly_pick_cells(percent_of_cells_list_EC_late_trough, num_trials_list, num_cells, rng)
    cells_with_before_field_middle, cells_with_near_field_middle, cells_with_after_field_middle, cells_with_wraparound_field_middle = randomly_pick_cells(percent_of_cells_list_EC_middle_trough, num_trials_list, num_cells, rng)

    trial_indices_before_dict_early, trial_indices_near_dict_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early = get_trial_indices(num_trials_list, before_percent_array, near_percent_array, after_percent_array, wraparound_percent_array, cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early, rng=rng, num_trials=num_t_e)
    trial_indices_before_dict_late, trial_indices_near_dict_late, trial_indices_after_dict_late, trial_indices_wraparound_dict_late = get_trial_indices(num_trials_list, before_percent_array, near_percent_array, after_percent_array, wraparound_percent_array, cells_with_before_field_late, cells_with_near_field_late, cells_with_after_field_late, cells_with_wraparound_field_late, rng=rng, num_trials=num_t_l)
    trial_indices_before_dict_middle, trial_indices_near_dict_middle, trial_indices_after_dict_middle, trial_indices_wraparound_dict_middle = get_trial_indices(num_trials_list, before_percent_array, near_percent_array, after_percent_array, wraparound_percent_array, cells_with_before_field_middle, cells_with_near_field_middle, cells_with_after_field_middle, cells_with_wraparound_field_middle, rng=rng, num_trials=num_t_m)

    cells_lists = [cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early]
    trials_lists = [trial_indices_before_dict_early, trial_indices_near_dict_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early]
    field_array_lists = [before_field_array_early_trough_EC_dict, near_field_array_early_trough_EC_dict, after_field_array_early_trough_EC_dict, wraparound_field_array_early_trough_EC_dict]
    cell_array_early_list, final_weights_early = slot_trials_into_spots(num_trials_list, cells_lists, trials_lists, field_array_lists, inits, vol, rng=rng, num_t=num_t_e)

    cells_lists = [cells_with_before_field_late, cells_with_near_field_late, cells_with_after_field_late, cells_with_wraparound_field_late]
    trials_lists = [trial_indices_before_dict_late, trial_indices_near_dict_late, trial_indices_after_dict_late, trial_indices_wraparound_dict_late]
    field_array_lists = [before_field_array_late_trough_EC_dict, near_field_array_late_trough_EC_dict, after_field_array_late_trough_EC_dict, wraparound_field_array_late_trough_EC_dict]
    cell_array_late_list, final_weights_late = slot_trials_into_spots(num_trials_list, cells_lists, trials_lists, field_array_lists, inits, vol, rng=rng, num_t=num_t_l)

    cells_lists = [cells_with_before_field_middle, cells_with_near_field_middle, cells_with_after_field_middle, cells_with_wraparound_field_middle]
    trials_lists = [trial_indices_before_dict_middle, trial_indices_near_dict_middle, trial_indices_after_dict_middle, trial_indices_wraparound_dict_middle]
    field_array_lists = [before_field_array_middle_trough_EC_dict, near_field_array_middle_trough_EC_dict, after_field_array_middle_trough_EC_dict, wraparound_field_array_middle_trough_EC_dict]
    cell_array_middle_list, final_weights_middle = slot_trials_into_spots(num_trials_list, cells_lists, trials_lists, field_array_lists, inits, vol, rng=rng, num_t=num_t_m)

    full_activity_list = []
    full_activity_trial_av_list = []

    trial_averaged_cell_list_early = []
    trial_averaged_cell_list_late = []
    trial_averaged_cell_list_middle = []

    for i in range(len(cell_array_early_list)):
        early_component = cell_array_early_list[i]
        late_component = cell_array_late_list[i]
        middle_component = cell_array_middle_list[i]

        full_actiivty = np.concatenate([early_component, middle_component, late_component], axis=1)

        full_activity_list.append(full_actiivty)

        full_activity_trial_av_list.append(np.mean(full_actiivty, axis=1))

        full_activity_trial_av_array = np.array(full_activity_trial_av_list)

        trial_averaged_cell_list_early.append(np.mean(early_component, axis=1))
        trial_averaged_cell_list_late.append(np.mean(late_component, axis=1))
        trial_averaged_cell_list_middle.append(np.mean(middle_component, axis=1))

    trial_averaged_cell_array_early = np.array(trial_averaged_cell_list_early)
    trial_averaged_cell_array_late = np.array(trial_averaged_cell_list_late)
    trial_averaged_cell_array_middle = np.array(trial_averaged_cell_list_middle)

    if plot:
        plot_generative_model(final_weights_early, final_weights_late, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, full_activity_trial_av_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, trial_averaged_cell_array_middle, cell_type=cell_type)


    else:
        return cell_array_early_list, cell_array_late_list, cell_array_middle_list, full_activity_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, full_activity_trial_av_array


def plot_generative_model(final_weights_early, final_weights_late, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, full_activity_trial_av_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, trial_averaged_cell_array_middle, cell_type="SST"):
    final_weighted_means_per_cell_before_early = final_weights_early[0]
    final_weighted_means_per_cell_near_early = final_weights_early[1]
    final_weighted_means_per_cell_after_early = final_weights_early[2]
    final_weighted_means_per_cell_wraparound_early = final_weights_early[3]

    final_weighted_means_per_cell_before_late = final_weights_late[0]
    final_weighted_means_per_cell_near_late = final_weights_late[1]
    final_weighted_means_per_cell_after_late = final_weights_late[2]
    final_weighted_means_per_cell_wraparound_late = final_weights_late[3]

    before_percent_array = percent_lists_EC_early_peak[0]
    near_percent_array = percent_lists_EC_early_peak[1]
    after_percent_array = percent_lists_EC_early_peak[2]
    wraparound_percent_array = percent_lists_EC_early_peak[3]

    before_percent_array_late = percent_lists_EC_late_peak[0]
    near_percent_array_late = percent_lists_EC_late_peak[1]
    after_percent_array_late = percent_lists_EC_late_peak[2]
    wraparound_percent_array_late = percent_lists_EC_late_peak[3]

    fig, axs = plt.subplots(6, 4, figsize=(15, 20))
    plt.suptitle(cell_type)

    #         def get_means_array(final_weighted_means_per_cell_before_early):
    #             final_weighted_means_per_cell_before_early_array = np.array(final_weighted_means_per_cell_before_early)
    #             mean_final_weighted_means_per_cell_before_early_array = np.mean(final_weighted_means_per_cell_before_early_array, axis=1)
    #             return mean_final_weighted_means_per_cell_before_early_array

    mean_final_weighted_means_per_cell_before_early_array = get_means_array(final_weighted_means_per_cell_before_early)

    for i in range(len(final_weighted_means_per_cell_before_early)):
        axs[0, 0].plot(final_weighted_means_per_cell_before_early[i], alpha=0.2)
    axs[0, 0].plot(mean_final_weighted_means_per_cell_before_early_array, color='r', linestyle='--', label='mean')
    axs[0, 0].set_title("Inputs Before Field Early")
    axs[0, 0].set_ylabel("Activity")
    axs[0, 0].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_before_late = get_means_array(final_weighted_means_per_cell_before_late)

    for i in range(len(final_weighted_means_per_cell_before_late)):
        axs[1, 0].plot(final_weighted_means_per_cell_before_late[i], alpha=0.2)
    axs[1, 0].plot(mean_final_weighted_means_per_cell_before_late, color='r', linestyle='--', label='mean')
    axs[1, 0].set_title("Inputs Before Field Late")
    axs[1, 0].set_ylabel("Activity")
    axs[1, 0].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_near_early = get_means_array(final_weighted_means_per_cell_near_early)

    for i in range(len(final_weighted_means_per_cell_near_early)):
        axs[0, 1].plot(final_weighted_means_per_cell_near_early[i], alpha=0.2)
    axs[0, 1].plot(mean_final_weighted_means_per_cell_near_early, color='r', linestyle='--', label='mean')
    axs[0, 1].set_title("Inputs Near Field Early")
    axs[0, 1].set_ylabel("Activity")
    axs[0, 1].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_near_late = get_means_array(final_weighted_means_per_cell_near_late)

    for i in range(len(final_weighted_means_per_cell_near_late)):
        axs[1, 1].plot(final_weighted_means_per_cell_near_late[i], alpha=0.2)
    axs[1, 1].plot(mean_final_weighted_means_per_cell_near_late, color='r', linestyle='--', label='mean')
    axs[1, 1].set_title("Inputs Near Field Late")
    axs[1, 1].set_ylabel("Activity")
    axs[1, 1].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_after_early = get_means_array(final_weighted_means_per_cell_after_early)

    for i in range(len(final_weighted_means_per_cell_after_early)):
        axs[0, 2].plot(final_weighted_means_per_cell_after_early[i], alpha=0.2)
    axs[0, 2].plot(mean_final_weighted_means_per_cell_after_early, color='r', linestyle='--', label='mean')
    axs[0, 2].set_title("Inputs After Field Early")
    axs[0, 2].set_ylabel("Activity")
    axs[0, 2].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_after_late = get_means_array(final_weighted_means_per_cell_after_late)

    for i in range(len(final_weighted_means_per_cell_after_late)):
        axs[1, 2].plot(final_weighted_means_per_cell_after_late[i], alpha=0.2)
    axs[1, 2].plot(mean_final_weighted_means_per_cell_after_late, color='r', linestyle='--', label='mean')
    axs[1, 2].set_title("Inputs After Field Late")
    axs[1, 2].set_ylabel("Activity")
    axs[1, 2].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_wraparound_early = get_means_array(final_weighted_means_per_cell_wraparound_early)

    for i in range(len(final_weighted_means_per_cell_wraparound_early)):
        axs[0, 3].plot(final_weighted_means_per_cell_wraparound_early[i], alpha=0.2)
    axs[0, 3].plot(mean_final_weighted_means_per_cell_wraparound_early, color='r', linestyle='--', label='mean')
    axs[0, 3].set_title("Inputs Wraparound Field Early")
    axs[0, 3].set_ylabel("Activity")
    axs[0, 3].set_xlabel("Position Bins")

    mean_final_weighted_means_per_cell_wraparound_late = get_means_array(final_weighted_means_per_cell_wraparound_late)

    for i in range(len(final_weighted_means_per_cell_wraparound_late)):
        axs[1, 3].plot(final_weighted_means_per_cell_wraparound_late[i], alpha=0.2)
    axs[1, 3].plot(mean_final_weighted_means_per_cell_wraparound_late, color='r', linestyle='--', label='mean')
    axs[1, 3].set_title("Inputs Wraparound Field Late")
    axs[1, 3].set_ylabel("Activity")
    axs[1, 3].set_xlabel("Position Bins")

    axs[2, 0].hist(before_percent_array, bins=20)
    axs[2, 0].set_title(f"Early Before Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[0] * 100:.1f}%")
    axs[2, 0].set_ylabel("Number of Cells")
    axs[2, 0].set_xlabel("Percent of Trials")

    axs[2, 1].hist(near_percent_array, bins=20)
    axs[2, 1].set_title(f"Early Near Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[1] * 100:.1f}%")
    axs[2, 1].set_ylabel("Number of Cells")
    axs[2, 1].set_xlabel("Percent of Trials")

    axs[2, 2].hist(after_percent_array, bins=20)
    axs[2, 2].set_title(f"Early After Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[2] * 100:.1f}%")
    axs[2, 2].set_ylabel("Number of Cells")
    axs[2, 2].set_xlabel("Percent of Trials")

    axs[2, 3].hist(wraparound_percent_array, bins=20)
    axs[2, 3].set_title(f"Early Wraparound Field Probability \n Percent of Cells={percent_of_cells_list_EC_early_trough[3] * 100:.1f}%")
    axs[2, 3].set_ylabel("Number of Cells")
    axs[2, 3].set_xlabel("Percent of Trials")

    axs[3, 0].hist(before_percent_array_late, bins=20)
    #         ax[2,0].set_title(f"Late Before Field Probability \n Percent of Cells={available_trials_before_late:.1f}%")
    axs[3, 0].set_title("Late Before Field Probability")
    axs[3, 0].set_ylabel("Number of Cells")
    axs[3, 0].set_xlabel("Percent of Trials")

    axs[3, 1].hist(near_percent_array_late, bins=20)
    #         axs[2,1].set_title(f"Late Near Field Probability \n Percent of Cells={available_trials_near_late:.1f}%")
    axs[3, 1].set_title("Late Near Field Probability")
    axs[3, 1].set_ylabel("Number of Cells")
    axs[3, 1].set_xlabel("Percent of Trials")

    axs[3, 2].hist(after_percent_array_late, bins=20)
    #         ax[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
    axs[3, 2].set_title("Late After Field Probability")
    axs[3, 2].set_ylabel("Number of Cells")
    axs[3, 2].set_xlabel("Percent of Trials")

    axs[3, 3].hist(wraparound_percent_array_late, bins=20)
    #         axs[2,2].set_title(f"Late After Field Probability \n Percent of Cells={available_trials_after_late:.1f}%")
    axs[3, 3].set_title("Late Wraparound Field Probability")
    axs[3, 3].set_ylabel("Number of Cells")
    axs[3, 3].set_xlabel("Percent of Trials")

    axs[5, 0].imshow(trial_averaged_cell_array_early, aspect='auto')
    axs[5, 0].set_ylabel("Cell ID")
    axs[5, 0].set_xlabel("Position Bin")

    means_early = np.mean(trial_averaged_cell_array_early, axis=0)
    sems_early = sem(trial_averaged_cell_array_early, axis=0)
    axs[4, 0].plot(means_early, color='orange')
    axs[4, 0].fill_between(range(len(means_early)), means_early + sems_early, means_early - sems_early, alpha=0.2, color='orange')
    axs[4, 0].set_xlabel("Position Bins")
    axs[4, 0].set_ylabel("Activity")
    axs[4, 0].set_title("Generative Early Learn")

    axs[5, 1].imshow(trial_averaged_cell_array_late, aspect='auto')
    axs[5, 1].set_ylabel("Cell ID")
    axs[5, 1].set_xlabel("Position Bin")

    means_late = np.mean(trial_averaged_cell_array_late, axis=0)
    sems_late = sem(trial_averaged_cell_array_late, axis=0)
    axs[4, 1].plot(means_late, color='r')
    axs[4, 1].fill_between(range(len(means_late)), means_late + sems_late, means_late - sems_late, alpha=0.2, color='r')
    axs[4, 1].set_xlabel("Position Bins")
    axs[4, 1].set_ylabel("Activity")
    axs[4, 1].set_title("Generative Late Learn")

    full_activity_trial_av_array = np.array(full_activity_trial_av_list)

    axs[5, 2].imshow(full_activity_trial_av_array, aspect='auto')
    axs[5, 2].set_ylabel("Cell ID")
    axs[5, 2].set_xlabel("Position Bin")

    means_overall = np.mean(full_activity_trial_av_list, axis=0)
    sems_overall = sem(full_activity_trial_av_list, axis=0)
    axs[4, 2].plot(means_overall, color='purple')
    axs[4, 2].fill_between(range(len(means_overall)), means_overall + sems_overall, means_overall - sems_overall, alpha=0.2, color='purple')
    axs[4, 2].set_xlabel("Position Bins")
    axs[4, 2].set_ylabel("Activity")
    axs[4, 2].set_title("Generative Overall Average")

    plt.tight_layout()
    plt.show()


def slot_trials_into_spots(animal_first_changepoints_list, cells_lists, trials_lists, field_array_lists, inits, vol, rng=None, num_t=20):
    cells_with_before_field_early = cells_lists[0]
    cells_with_near_field_early = cells_lists[1]
    cells_with_after_field_early = cells_lists[2]
    cells_with_wraparound_field_early = cells_lists[3]

    trial_indices_before_dict_early = trials_lists[0]
    trial_indices_near_dict_early = trials_lists[1]
    trial_indices_after_dict_early = trials_lists[2]
    trial_indices_wraparound_dict_early = trials_lists[3]

    before_field_array_early_trough_EC_dict = field_array_lists[0]
    near_field_array_early_trough_EC_dict = field_array_lists[1]
    after_field_array_early_trough_EC_dict = field_array_lists[2]
    wraparound_field_array_early_trough_EC_dict = field_array_lists[3]

    cell_array_early_list = get_cell_array(animal_first_changepoints_list, inits, vol, rng=rng, num_trials_early=num_t)

    trial_indices_before_dict_early, trial_indices_near_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_near_field_early, trial_indices_before_dict_early, trial_indices_near_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_after_field_early, trial_indices_near_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_after_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_after_field_early, trial_indices_before_dict_early, trial_indices_after_dict_early, rng)
    trial_indices_before_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_before_field_early, cells_with_wraparound_field_early, trial_indices_before_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_near_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_near_field_early, cells_with_wraparound_field_early, trial_indices_near_dict_early, trial_indices_wraparound_dict_early, rng)
    trial_indices_after_dict_early, trial_indices_wraparound_dict_early = remove_duplicate_trials(cells_with_after_field_early, cells_with_wraparound_field_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early, rng)

    final_weighted_means_per_cell_before_early = get_weighted_activityies_per_cell(before_field_array_early_trough_EC_dict, field_type="before_field")
    final_weighted_means_per_cell_near_early = get_weighted_activityies_per_cell(near_field_array_early_trough_EC_dict, field_type="near_field")
    final_weighted_means_per_cell_after_early = get_weighted_activityies_per_cell(after_field_array_early_trough_EC_dict, field_type="after_field")
    final_weighted_means_per_cell_wraparound_early = get_weighted_activityies_per_cell(wraparound_field_array_early_trough_EC_dict, field_type="wraparound_field")

    count_before_early = 0
    for cell, trial_indices in trial_indices_before_dict_early.items():
        if count_before_early >= len(final_weighted_means_per_cell_before_early):
            continue
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_before_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_before_early[count_before_early]
            else:
                print("improper alignment")
        count_before_early += 1

    count_near_early = 0
    for cell, trial_indices in trial_indices_near_dict_early.items():
        if count_near_early >= len(final_weighted_means_per_cell_near_early):
            continue
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_near_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_near_early[count_near_early]
            else:
                print("improper alignment")

        count_near_early += 1

    count_after_early = 0
    for cell, trial_indices in trial_indices_after_dict_early.items():
        if count_after_early >= len(final_weighted_means_per_cell_after_early):
            continue
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_after_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_after_early[count_after_early]
            else:
                print("improper alignment")

        count_after_early += 1

    count_wraparound_early = 0
    for cell, trial_indices in trial_indices_wraparound_dict_early.items():
        if count_wraparound_early >= len(final_weighted_means_per_cell_wraparound_early):
            #             print(f"⚠️ Ran out of wraparound weighted means! Skipping cell {cell}")
            continue
        for t in trial_indices:
            if t < cell_array_early_list[cell].shape[1]:
                #                 cell_array_early_list[cell][:, t] = example_wraparound_field_early
                cell_array_early_list[cell][:, t] = final_weighted_means_per_cell_wraparound_early[count_wraparound_early]
            else:
                print("improper alignment")

        count_wraparound_early += 1

    final_weights_early = [final_weighted_means_per_cell_before_early, final_weighted_means_per_cell_near_early, final_weighted_means_per_cell_after_early, final_weighted_means_per_cell_wraparound_early]

    return cell_array_early_list, final_weights_early


def get_trial_indices(animal_first_changepoints_list, before_percent_array, near_percent_array, after_percent_array, wraparound_percent_array, cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early, rng=None, num_trials=20):
    trial_indices_before_dict_early = {}
    trial_indices_near_dict_early = {}
    trial_indices_after_dict_early = {}
    trial_indices_wraparound_dict_early = {}

    for cell in range(len(animal_first_changepoints_list)):

        available_trials_early = np.arange(num_trials)
        remaining_trials_early = available_trials_early.copy()

        if cell in cells_with_before_field_early:
            proportion_early_before = (before_percent_array[cell] / 100) * num_trials
            trial_indices_before = rng.choice(available_trials_early, int(proportion_early_before), replace=False)
            trial_indices_before_dict_early[cell] = trial_indices_before

        if cell in cells_with_near_field_early:
            proportion_early_near = (near_percent_array[cell] / 100) * num_trials
            trial_indices_near = rng.choice(available_trials_early, int(proportion_early_near), replace=False)
            trial_indices_near_dict_early[cell] = trial_indices_near

        if cell in cells_with_after_field_early:
            proportion_early_after = (after_percent_array[cell] / 100) * num_trials
            trial_indices_after = rng.choice(available_trials_early, int(proportion_early_after), replace=False)
            trial_indices_after_dict_early[cell] = trial_indices_after

        if cell in cells_with_wraparound_field_early:
            proportion_early_wraparound = (wraparound_percent_array[cell] / 100) * num_trials
            trial_indices_wraparound = rng.choice(available_trials_early, int(proportion_early_wraparound), replace=False)
            trial_indices_wraparound_dict_early[cell] = trial_indices_wraparound

    return trial_indices_before_dict_early, trial_indices_near_dict_early, trial_indices_after_dict_early, trial_indices_wraparound_dict_early


def get_cell_array(animal_first_changepoints_list, inits, vol, rng=None, num_trials_early=20):
    cell_array_early_list = []

    for cell in range(len(animal_first_changepoints_list)):
        rt_early = np.array(random_timeseries2(initial_value=inits, volatility=vol, count=49, rng=rng))

        cell_array = np.tile(rt_early[:, np.newaxis], (1, num_trials_early))
        cell_array_early_list.append(cell_array)

    return cell_array_early_list

def randomly_pick_cells(percent_of_cells_list_EC_early_trough, animal_first_changepoints_list, num_cells, rng):
    available_cells_before_early = percent_of_cells_list_EC_early_trough[0] * len(animal_first_changepoints_list)
    available_cells_near_early = percent_of_cells_list_EC_early_trough[1] * len(animal_first_changepoints_list)
    available_cells_after_early = percent_of_cells_list_EC_early_trough[2] * len(animal_first_changepoints_list)
    available_cells_wraparound_early = percent_of_cells_list_EC_early_trough[3] * len(animal_first_changepoints_list)

    cells_with_before_field_early = rng.choice(num_cells, int(available_cells_before_early), replace=False)
    cells_with_near_field_early = rng.choice(num_cells, int(available_cells_near_early), replace=False)
    cells_with_after_field_early = rng.choice(num_cells, int(available_cells_after_early), replace=False)
    cells_with_wraparound_field_early = rng.choice(num_cells, int(available_cells_wraparound_early), replace=False)

    return cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early


