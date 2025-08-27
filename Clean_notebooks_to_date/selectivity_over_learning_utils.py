

file_path = '/Users/michaelfinch/CA1-interneuron-GLM' #Mac
#file_path=r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM"

import sys
sys.path.append(file_path)

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *


from modelling_to_date_utils import *

def get_fixed_model_dict_NDNF_newest(cell_NDNF_model_ranks20_contig_x00):
    fixed_model_dict_NDNF_newest = {20:{}}
    for animal in cell_NDNF_model_ranks20_contig_x00[20]:
        if 17 < animal < 31:
            fixed_model_dict_NDNF_newest[20][animal-18] = cell_NDNF_model_ranks20_contig_x00[20][animal]
    return fixed_model_dict_NDNF_newest

def subset_factors(factors_dict_NDNF_newest):
    fixed_factors_dict_NDNF_newest = {}
    for idx, animal in enumerate(factors_dict_NDNF_newest):
        if 17 < idx < 31:
            fixed_factors_dict_NDNF_newest[f"animal_{idx+1}"] = factors_dict_NDNF_newest[animal]

    return fixed_factors_dict_NDNF_newest

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

def get_selectivity_each_trial(activity_dict_EC, neg_sel=True, trial_av=False):
    """
    - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
    returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually
    """

    animal_average_selectivity_dict = {}
    for animal in activity_dict_EC:
        cell_dict = {}
        for cell in activity_dict_EC[animal]:
            cell_data = activity_dict_EC[animal][cell]
            if trial_av:
                trial_av_activity = np.mean(cell_data, axis=1) 
                selectivity_trial_av = Vinje2000(trial_av_activity, norm='none', negative_selectivity=neg_sel)
                cell_dict[cell] = selectivity_trial_av
            else:
                trial_selectivity_list = []
                for trial in range(cell_data.shape[1]):
                    trial_activity = cell_data[:,trial] 
                    selectivity_trial = Vinje2000(trial_activity, norm='none', negative_selectivity=neg_sel)
                    trial_selectivity_list.append(selectivity_trial)
                
                percentile_average_selectivity = np.mean(trial_selectivity_list)
                cell_dict[cell] = percentile_average_selectivity
        animal_average_selectivity_dict[animal] = cell_dict
    return animal_average_selectivity_dict

def get_selectivity_each_trial_early_late(activity_dict_EC, cp_dict_EC, neg_sel=True, trial_av=False, use_early=True):
    """
    - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
    returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually
    """

    animal_average_selectivity_dict = {}
    for idx, animal in enumerate(activity_dict_EC):
        cell_dict = {}
        for idt, cell in enumerate(activity_dict_EC[animal]):
            cp = cp_dict_EC[idx][idt]
            early_cut = cp[0]
            late_cut = cp[1]
            cell_data = activity_dict_EC[animal][cell]
            if trial_av:
                if use_early:
                    cell_data_early = cell_data[:,:early_cut]
                    trial_av_activity = np.mean(cell_data_early, axis=1) 
                    selectivity_trial_av = Vinje2000(trial_av_activity, norm='none', negative_selectivity=neg_sel)
                    cell_dict[cell] = selectivity_trial_av
                else:
                    cell_data_late = cell_data[:,:-late_cut:]
                    trial_av_activity = np.mean(cell_data_late, axis=1) 
                    selectivity_trial_av = Vinje2000(trial_av_activity, norm='none', negative_selectivity=neg_sel)
                    cell_dict[cell] = selectivity_trial_av
            else:
                trial_selectivity_list = []
                if use_early:
                    for trial in range(cell_data.shape[1]):
                        if trial <= early_cut:
                            trial_activity = cell_data[:,trial] 
                            selectivity_trial = Vinje2000(trial_activity, norm='none', negative_selectivity=neg_sel)
                            trial_selectivity_list.append(selectivity_trial)
                else:
                    for trial in range(cell_data.shape[1]):
                        if trial >= late_cut:
                            trial_activity = cell_data[:,trial] 
                            selectivity_trial = Vinje2000(trial_activity, norm='none', negative_selectivity=neg_sel)
                            trial_selectivity_list.append(selectivity_trial)

                percentile_average_selectivity = np.mean(trial_selectivity_list)
                cell_dict[cell] = percentile_average_selectivity
        animal_average_selectivity_dict[animal] = cell_dict
    return animal_average_selectivity_dict

def get_binned_data_for_CDF(animal_average_selectivity_dict_SST_r, n_bins=20):

    """
    params animal_average_selectivity_dict_SST_r: selectivity value for every cell in a dict
    returns binned_data: the cells with selectiivty values that fit within each bin (percent of data)
    """

    selectivity_list = []
    for animal in animal_average_selectivity_dict_SST_r:
        for cell in animal_average_selectivity_dict_SST_r[animal]:
            val = animal_average_selectivity_dict_SST_r[animal][cell]
            selectivity_list.append(val)

    selectivity_array = np.array(selectivity_list)

    edges = np.quantile(selectivity_array, np.linspace(0, 1, n_bins+1))  

    binned_data = []

    for idx in range(n_bins):
        low = edges[idx]
        high = edges[idx + 1]

        # Include low, exclude high except last bin
        if idx < 19:
            in_bin = selectivity_array[(selectivity_array >= low) & (selectivity_array < high)]
        else:
            in_bin = selectivity_array[(selectivity_array >= low) & (selectivity_array <= high)]

        binned_data.append(in_bin)

    return binned_data

def get_mean_sem_lists(binned_data):
    mean_data_list = []
    sem_data_list = []
    for i in range(len(binned_data)):
        bin_data = binned_data[i]
        mean_data_list.append(np.mean(bin_data))
        sem_data_list.append(sem(bin_data))
    return mean_data_list, sem_data_list

def plot_the_CDF(binned_data_SST, binned_data_EC, binned_data_NDNF, title="Selectivity Distribution Across Cells +-SEM"):
    mean_SST, sem_SST = get_mean_sem_lists(binned_data_SST)
    mean_EC, sem_EC = get_mean_sem_lists(binned_data_EC)
    mean_NDNF, sem_NDNF = get_mean_sem_lists(binned_data_NDNF)

    # Y-axis = percentiles (center of each bin: 2.5%, 7.5%, ..., 97.5%)
    n_bins = 20
    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5

    # Plot: horizontal bars (x=selectivity, y=percentile)
    plt.figure(figsize=(7, 6))

    plt.errorbar(mean_SST, percentiles, xerr=sem_SST, fmt='o-', label='SST', color='blue', capsize=3)
    plt.errorbar(mean_EC, percentiles, xerr=sem_EC, fmt='o-', label='EC', color='green', capsize=3)
    plt.errorbar(mean_NDNF, percentiles, xerr=sem_NDNF, fmt='o-', label='NDNF', color='orange', capsize=3)

    plt.ylabel("Percentile of Cells")
    plt.xlabel("Selectivity")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_the_CDF_early_late(binned_data_SST_e, binned_data_EC_e, binned_data_NDNF_e, binned_data_SST_l, binned_data_EC_l, binned_data_NDNF_l, title="Selectivity Distribution Across Cells +-SEM"):
    
    mean_SST_e, sem_SST_e = get_mean_sem_lists(binned_data_SST_e)
    mean_EC_e, sem_EC_e = get_mean_sem_lists(binned_data_EC_e)
    mean_NDNF_e, sem_NDNF_e = get_mean_sem_lists(binned_data_NDNF_e)

    mean_SST_l, sem_SST_l = get_mean_sem_lists(binned_data_SST_l)
    mean_EC_l, sem_EC_l = get_mean_sem_lists(binned_data_EC_l)
    mean_NDNF_l, sem_NDNF_l = get_mean_sem_lists(binned_data_NDNF_l)

    n_bins = 20
    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins) 

    plt.figure(figsize=(7, 6))

    plt.errorbar(mean_SST_e, percentiles, xerr=sem_SST_e, fmt='o-', label='SST Early', color='cyan', capsize=3)
    plt.errorbar(mean_EC_e, percentiles, xerr=sem_EC_e, fmt='o-', label='EC Early', color='k', capsize=3)
    plt.errorbar(mean_NDNF_e, percentiles, xerr=sem_NDNF_e, fmt='o-', label='NDNF Early', color='orange', capsize=3)
    plt.errorbar(mean_SST_l, percentiles, xerr=sem_SST_l, fmt='o-', label='SST Late', color='blue', capsize=3)
    plt.errorbar(mean_EC_l, percentiles, xerr=sem_EC_l, fmt='o-', label='EC Late', color='green', capsize=3)
    plt.errorbar(mean_NDNF_l, percentiles, xerr=sem_NDNF_l, fmt='o-', label='NDNF Late', color='red', capsize=3)

    plt.ylabel("Percentile of Cells")
    plt.xlabel("Selectivity")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

def plot_selectivity_seperated_by_learn_stage(animal_average_selectivity_dict_SST, animal_average_selectivity_dict_NDNF, animal_average_selectivity_dict_EC):

    SST_means, SST_sems = get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_SST)

    NDNF_means, NDNF_sems = get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_NDNF)

    EC_means, EC_sems = get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_EC)

    x = np.arange(3)
    labels = ["Early", "Middle", "Late"]

    plt.figure(figsize=(6, 4))

    plt.errorbar(x, SST_means, yerr=SST_sems, color='b', label="SST", capsize=4, fmt='-o')
    plt.errorbar(x, NDNF_means, yerr=NDNF_sems, color='orange', label="NDNF", capsize=4, fmt='-o')
    plt.errorbar(x, EC_means, yerr=EC_sems, color='green', label="EC", capsize=4, fmt='-o')

    plt.xticks(x, labels)
    plt.ylabel("Average Selectivity Across Cells")
    plt.xlabel("Contiguous K-Means Learning Stage")
    plt.title("Selectivity by Learning Stage")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_selectivity_over_trials(all_cell_selectivity_SST, all_cell_selectivity_EC, all_cell_selectivity_NDNF):

    mean_selectivity_SST = np.mean(all_cell_selectivity_SST, axis=0)
    sem_selectivity_SST = sem(all_cell_selectivity_SST, axis=0)

    mean_selectivity_EC = np.mean(all_cell_selectivity_EC, axis=0)
    sem_selectivity_EC = sem(all_cell_selectivity_EC, axis=0)

    mean_selectivity_NDNF = np.mean(all_cell_selectivity_NDNF, axis=0)
    sem_selectivity_NDNF = sem(all_cell_selectivity_NDNF, axis=0)

    # Step 4: plot
    x = np.arange(1, 11)  # Percentile bins: 1 to 10

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_selectivity_SST, color='b', label='SST')
    plt.fill_between(x, mean_selectivity_SST - sem_selectivity_SST, mean_selectivity_SST + sem_selectivity_SST, alpha=0.2, color='b')
    plt.plot(x, mean_selectivity_EC, color='green', label='EC')
    plt.fill_between(x, mean_selectivity_EC - sem_selectivity_EC, mean_selectivity_EC + sem_selectivity_EC, alpha=0.2, color='green')
    plt.plot(x, mean_selectivity_NDNF, color='orange', label='NDNF')
    plt.fill_between(x, mean_selectivity_NDNF - sem_selectivity_NDNF, mean_selectivity_NDNF + sem_selectivity_NDNF, alpha=0.2, color='orange')
    plt.xticks(ticks=x, labels=[f"{int(p)}%" for p in np.linspace(0, 100, 10)])
    plt.xlabel("Percentile of Trials")
    plt.ylabel("Average Selectivity Across Cells")
    plt.title("Selectivity Across Trials")
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_selectivity_array(animal_average_selectivity_dict):

    # Step 1: gather data into a list of [n_cells x 10] rows
    all_cell_selectivity = []

    for animal in animal_average_selectivity_dict:
        for cell in animal_average_selectivity_dict[animal]:
            selectivity_per_bin = animal_average_selectivity_dict[animal][cell]
            if len(selectivity_per_bin) == 10:  # sanity check
                all_cell_selectivity.append(selectivity_per_bin)

    # Step 2: convert to numpy array
    all_cell_selectivity = np.array(all_cell_selectivity)  # shape: [n_cells, 10]

    return all_cell_selectivity

def get_animal_average_selectivity_dict(percentile_slices, neg_sel=True, trial_av=False):
    animal_average_selectivity_dict = {}
    for animal in percentile_slices:
        cell_dict = {}
        for cell in percentile_slices[animal]:
            cell_data = percentile_slices[animal][cell]
            average_selectivity_list = []
            for i in range(len(cell_data)):
                activity_cut = cell_data[i]

                if trial_av:
                    trial_av_activity = np.mean(activity_cut, axis=1) 
                    selectivity_trial_av = Vinje2000(trial_av_activity, norm='none', negative_selectivity=neg_sel)
                    average_selectivity_list.append(selectivity_trial_av)
                else:
                    trial_selectivity_list = []
                    for trial in range(activity_cut.shape[1]):
                        trial_activity = activity_cut[:,trial] 
                        selectivity_trial = Vinje2000(trial_activity, norm='none', negative_selectivity=neg_sel)
                        trial_selectivity_list.append(selectivity_trial)
                    
                    percentile_average_selectivity = np.mean(trial_selectivity_list)
                    average_selectivity_list.append(percentile_average_selectivity)
            cell_dict[cell] = average_selectivity_list
        animal_average_selectivity_dict[animal] = cell_dict
    return animal_average_selectivity_dict

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










