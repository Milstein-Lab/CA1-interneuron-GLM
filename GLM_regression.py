import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
import scipy.stats as stats
from sklearn.cluster import KMeans
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import sem
from collections import defaultdict
import os
import sys

plt.rcParams.update({'font.size': 10,
                    'axes.spines.right': False,
                    'axes.spines.top':   False,
                    'legend.frameon':       False,})


def preprocess_data(filepath, normalize=True):
    data_dict = mat73.loadmat(filepath)

    # Define new position variables to use as input for the GLM
    num_spatial_bins = 10
    position_matrix = np.zeros((50, num_spatial_bins))
    bin_size = 50 // num_spatial_bins
    for i in range(num_spatial_bins):
        position_matrix[i * bin_size:(i + 1) * bin_size, i] = 1

    factors_dict = {}
    activity_dict = {}
    for animal_idx, (delta_f, velocity, lick_rate, reward_loc) in enumerate(zip(data_dict['animal']['ShiftR'], data_dict['animal']['ShiftRunning'], data_dict['animal']['ShiftLrate'],data_dict['animal']['ShiftV'])):
        num_trials = min(delta_f.shape[1], lick_rate.shape[1], reward_loc.shape[1], velocity.shape[1])
        lick_rate = lick_rate[:, :num_trials]
        reward_loc = reward_loc[:, :num_trials]
        velocity = velocity[:, :num_trials]
        delta_f = delta_f[:, :num_trials, :]

        # Exclude trials with NaNs
        nan_trials_licks = np.any(np.isnan(lick_rate), axis=0)
        nan_trials_reward = np.any(np.isnan(reward_loc), axis=0)
        nan_trials_velocity = np.any(np.isnan(velocity), axis=0)
        nan_trials_activity = np.any(np.isnan(delta_f), axis=(0, 2))
        nan_trials = nan_trials_licks | nan_trials_reward | nan_trials_velocity | nan_trials_activity

        animal_key = f'animal_{animal_idx + 1}'
        factors_dict[animal_key] = {"Licks": lick_rate[:, ~nan_trials],
                                    "Reward_loc": reward_loc[:, ~nan_trials],
                                    "Velocity": velocity[:, ~nan_trials]}

        # Add a factor for each position variable (from 1 to num_spatial_bins)
        num_trials = factors_dict[animal_key]["Velocity"].shape[1]
        for bin_idx in range(num_spatial_bins):
            bin_key = f"Position_{bin_idx + 1}"
            factors_dict[animal_key][bin_key] = np.tile(position_matrix[:, bin_idx][:, np.newaxis], num_trials) # Copy the position variable for each trial

        if normalize: # Normalize behavioral factors to [0,1]
            for var_name in factors_dict[animal_key]:
                factors_dict[animal_key][var_name] = (factors_dict[animal_key][var_name] - np.min(factors_dict[animal_key][var_name])) / (np.max(factors_dict[animal_key][var_name]) - np.min(factors_dict[animal_key][var_name]))

        activity_dict[animal_key] = {}
        for neuron_idx in range(delta_f.shape[2]):
            neuron_activity = delta_f[:, :, neuron_idx]
            if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0): # Don't save empty/silent neurons
                continue
            cleaned_activity = neuron_activity[:, ~nan_trials]

            if normalize:  # Z-score the neuron activity (df/f)
                cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)

            neuron_key = f'cell_{neuron_idx + 1}'
            activity_dict[animal_key][neuron_key] = cleaned_activity

    return activity_dict, factors_dict


def trial_average(activity_list):
    trial_average_list = []

    for i in activity_list:
        trial_average = np.mean(i, axis=1)
        trial_average_list.append(trial_average)

    return trial_average_list


def normalize_data(neuron_dict):
    for var_name in neuron_dict:
        if var_name == "Activity": # Z-score the neuron activity (df/f)
            neuron_dict[var_name] = (neuron_dict[var_name] - np.mean(neuron_dict[var_name])) / np.std(neuron_dict[var_name])
        else: # Normalize the other variables to [0,1]
            neuron_dict[var_name] = (neuron_dict[var_name] - np.min(neuron_dict[var_name])) / (np.max(neuron_dict[var_name]) - np.min(neuron_dict[var_name]))
            

def subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"]):
    filtered_factors_dict = {}
    for animal in factors_dict:
        filtered_factors_dict[animal] = {}
        for variable in variables_to_keep:
            if variable in factors_dict[animal]:
                filtered_factors_dict[animal][variable] = factors_dict[animal][variable]
            else:
                raise ValueError(f"Variable '{variable}' not found in neuron data for {neuron} in {animal}.")
    return filtered_factors_dict


def fit_GLM_population(factors_dict, activity_dict, quintile=None, regression='ridge', renormalize=True, alphas=None):
    GLM_params = {}
    predicted_activity_dict = {}

    for animal in factors_dict:
        GLM_params[animal] = {}
        predicted_activity_dict[animal] = {}
        _factors_dict = factors_dict[animal].copy()

        if quintile is not None:
            num_trials = _factors_dict['Activity'].shape[1]
            start_idx, end_idx = get_quintile_indices(num_trials, quintile)
            for var in _factors_dict:
                _factors_dict[var] = _factors_dict[var][:, start_idx:end_idx]

        if renormalize:
            normalize_data(_factors_dict)

        for neuron_idx in activity_dict[animal]:
            neuron_activity = activity_dict[animal][neuron_idx]

            neuron_GLM_params, neuron_predicted_activity = fit_GLM(_factors_dict, neuron_activity, regression, alphas)

            GLM_params[animal][neuron_idx] = neuron_GLM_params
            predicted_activity_dict[animal][neuron_idx] = neuron_predicted_activity.reshape(
                activity_dict[animal][neuron_idx].shape)

    return GLM_params, predicted_activity_dict


def fit_GLM(factors_dict, neuron_activity, regression='ridge', alphas=None):
    neuron_activity_flat = neuron_activity.flatten()
    flattened_data = flatten_data(factors_dict)

    variable_names = [var for var in flattened_data]
    design_matrix_X = np.stack([flattened_data[var] for var in variable_names], axis=1)

    if regression == 'lasso':
        model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
    elif regression == 'ridge':
        model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
    elif regression == 'elastic':
        l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000],
                             l1_ratio=l1_ratio, cv=None)
    else:
        raise ValueError("Regression type must be 'lasso' or 'ridge'")

    model.fit(design_matrix_X, neuron_activity_flat)

    neuron_predicted_activity = model.predict(design_matrix_X)

    trialavg_neuron_activity = np.mean(neuron_activity, axis=1)
    trialavg_predicted_activity = np.mean(neuron_predicted_activity.reshape(neuron_activity.shape), axis=1)
    pearson_R = np.corrcoef(trialavg_predicted_activity, trialavg_neuron_activity)[0, 1]

    neuron_GLM_params = {}
    neuron_GLM_params['weights'] = {var: model.coef_[idx] for idx, var in enumerate(variable_names)}
    neuron_GLM_params['intercept'] = model.intercept_
    neuron_GLM_params['alpha'] = model.alpha_ if regression == 'ridge' else model.alpha_
    neuron_GLM_params['l1_ratio'] = model.l1_ratio_ if regression == 'elastic' else None
    neuron_GLM_params['R2'] = model.score(design_matrix_X, neuron_activity_flat)
    neuron_GLM_params['pearson_R'] = pearson_R
    neuron_GLM_params['model'] = model

    return neuron_GLM_params, neuron_predicted_activity


def fit_behavior_GLM(animal_activity_dict, behavior_data, regression='ridge', alphas=None):
    neural_data = []
    for cell, data in animal_activity_dict.items():
        neural_data.append(data.flatten())

    design_matrix_X = np.stack(neural_data, axis=1)
    behavior_data_flattened = behavior_data.flatten()

    if regression == 'lasso':
        model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
    elif regression == 'ridge':
        model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
    elif regression == 'elastic':
        l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000],
                             l1_ratio=l1_ratio, cv=None)
    else:
        raise ValueError("Regression type must be 'lasso' or 'ridge'")

    model.fit(design_matrix_X, behavior_data_flattened)

    predicted_behavior = model.predict(design_matrix_X)
    pearson_R = np.corrcoef(predicted_behavior, behavior_data_flattened)[0, 1]

    animal_GLM_params = {}
    animal_GLM_params['alpha'] = model.alpha_ if regression == 'ridge' else model.alpha_
    animal_GLM_params['l1_ratio'] = model.l1_ratio_ if regression == 'elastic' else None
    animal_GLM_params['R2'] = model.score(design_matrix_X, behavior_data_flattened)
    animal_GLM_params['pearson_R'] = pearson_R
    animal_GLM_params['model'] = model

    return animal_GLM_params, predicted_behavior.reshape(behavior_data.shape)


def get_animal_mean_sem(activity_dict, cell_residual_list):
    animal_mean_list = []
    animal_sem_list = []

    animal_mean_residuals = []
    animal_sem_residuals = []

    count = 0
    for animal in activity_dict:
        per_animal_data = []
        per_animal_residuals = []

        for neuron in activity_dict[animal]:
            neuron_mean_activity = np.mean(activity_dict[animal][neuron], axis=1)
            neuron_mean_residual = np.mean(cell_residual_list[count], axis=1)

            per_animal_data.append(neuron_mean_activity)
            per_animal_residuals.append(neuron_mean_residual)

            count += 1

        per_animal_data = np.array(per_animal_data)
        per_animal_residuals = np.array(per_animal_residuals)

        mean_animal = np.mean(per_animal_data, axis=0)
        sem_animal = sem(per_animal_data, axis=0)
        mean_residual = np.mean(per_animal_residuals, axis=0)
        sem_residuals = sem(per_animal_residuals, axis=0)

        animal_mean_list.append(mean_animal)
        animal_sem_list.append(sem_animal)
        animal_mean_residuals.append(mean_residual)
        animal_sem_residuals.append(sem_residuals)

    return animal_mean_list, animal_sem_list, animal_mean_residuals, animal_sem_residuals


def get_per_animal_mean_r2(r2_variable_activity_dict_SST):
    mean_r2_per_animal = []
    sem_r2_per_animal = []

    for animal in r2_variable_activity_dict_SST:
        neurons = []
        for neuron in r2_variable_activity_dict_SST[animal]:
            neurons.append(r2_variable_activity_dict_SST[animal][neuron])
        mean_r2_per_animal.append(np.mean(neurons))
        sem_r2_per_animal.append(sem(neurons))

    return mean_r2_per_animal, sem_r2_per_animal


def get_neuron_activity_prediction_residual(activity_dict, predicted_activity_dict):
    neuron_activity_list = []
    predictions_list = []
    cell_residual_list = []

    for animal in activity_dict:
        if animal not in predicted_activity_dict:
            print(f"Warning: {animal} not found in predicted_activity_dict. Skipping.")
            continue

        for neuron in activity_dict[animal]:
            if neuron not in predicted_activity_dict[animal]:
                print(f"Warning: {neuron} not found in predicted_activity_dict[{animal}]. Skipping.")
                continue

            neuron_activity = activity_dict[animal][neuron]
            prediction = predicted_activity_dict[animal][neuron]

            residual = np.array(neuron_activity) - np.array(prediction)

            neuron_activity_list.append(neuron_activity)
            predictions_list.append(prediction)
            cell_residual_list.append(residual)

    return neuron_activity_list, predictions_list, cell_residual_list


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


def get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=False):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)

    if residual:
        trial_av_activity_SST = trial_average(cell_residual_list_SST)

    else:
        trial_av_activity_SST = trial_average(neuron_activity_list_SST)

    SST_negative_selectivity = []
    SST_factor_list = []
    for i in trial_av_activity_SST:
        selectivity = Vinje2000(i, norm='min_max', negative_selectivity=False)
        negative_selectivity = Vinje2000(i, norm='min_max', negative_selectivity=True)
        SST_factor_list.append(selectivity)
        SST_negative_selectivity.append(negative_selectivity)

    return SST_factor_list, SST_negative_selectivity


def setup_CDF_plotting_for_selectivity(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False):

    SST_factor_list, SST_negative_selectivity = get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=residual)
    NDNF_factor_list, NDNF_negative_selectivity = get_selectivity_for_plotting(activity_dict_NDNF, predicted_activity_dict_NDNF, residual=residual)
    EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(activity_dict_EC, predicted_activity_dict_EC, residual=residual)

    mean_quantiles_SST, sem_quantiles_SST = get_quantiles_for_cdf(activity_dict_SST, SST_factor_list, n_bins=20)

    mean_quantiles_NDNF, sem_quantiles_NDNF = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_factor_list, n_bins=20)

    mean_quantiles_EC, sem_quantiles_EC = get_quantiles_for_cdf(activity_dict_EC, EC_factor_list, n_bins=20)

    mean_quantiles_list = [mean_quantiles_SST, mean_quantiles_NDNF, mean_quantiles_EC]

    sem_quantiles_list = [sem_quantiles_SST, sem_quantiles_NDNF, sem_quantiles_EC]

    mean_quantiles_SST_negative, sem_quantiles_SST_negative = get_quantiles_for_cdf(activity_dict_SST,
                                                                                    SST_negative_selectivity, n_bins=20)

    mean_quantiles_NDNF_negative, sem_quantiles_NDNF_negative = get_quantiles_for_cdf(activity_dict_NDNF,
                                                                                      NDNF_negative_selectivity,
                                                                                      n_bins=20)

    mean_quantiles_EC_negative, sem_quantiles_EC_negative = get_quantiles_for_cdf(activity_dict_EC,
                                                                                  EC_negative_selectivity, n_bins=20)

    mean_quantiles_list_negative = [mean_quantiles_SST_negative, mean_quantiles_NDNF_negative,
                                    mean_quantiles_EC_negative]

    sem_quantiles_list_negative = [sem_quantiles_SST_negative, sem_quantiles_NDNF_negative, sem_quantiles_EC_negative]

    return mean_quantiles_list, sem_quantiles_list, mean_quantiles_list_negative, sem_quantiles_list_negative


def get_argmin_argmax_for_plotting(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(
        activity_dict_EC, predicted_activity_dict_EC)

    if residual:
        trial_av_activity_SST = trial_average(cell_residual_list_SST)
        trial_av_activity_NDNF = trial_average(cell_residual_list_NDNF)
        trial_av_activity_EC = trial_average(cell_residual_list_EC)

    else:
        trial_av_activity_SST = trial_average(neuron_activity_list_SST)
        trial_av_activity_NDNF = trial_average(neuron_activity_list_NDNF)
        trial_av_activity_EC = trial_average(neuron_activity_list_EC)

    if which_to_plot == "argmin":
        SST_factor_list = []
        for i in trial_av_activity_SST:
            argmin_SST = np.argmin(i)
            SST_factor_list.append(argmin_SST)

        NDNF_factor_list = []
        for i in trial_av_activity_NDNF:
            argmin_NDNF = np.argmin(i)
            NDNF_factor_list.append(argmin_NDNF)

        EC_factor_list = []
        for i in trial_av_activity_EC:
            argmin_EC = np.argmin(i)
            EC_factor_list.append(argmin_EC)

    elif which_to_plot == "argmax":
        SST_factor_list = []
        for i in trial_av_activity_SST:
            argmax_SST = np.argmax(i)
            SST_factor_list.append(argmax_SST)

        NDNF_factor_list = []
        for i in trial_av_activity_NDNF:
            argmax_NDNF = np.argmax(i)
            NDNF_factor_list.append(argmax_NDNF)

        EC_factor_list = []
        for i in trial_av_activity_EC:
            argmax_EC = np.argmax(i)
            EC_factor_list.append(argmax_EC)

    else:
        raise ValueError("options are argmin or argmax")

    return SST_factor_list, NDNF_factor_list, EC_factor_list


def setup_CDF_plotting_and_plot_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):

    SST_factor_list, NDNF_factor_list, EC_factor_list = get_argmin_argmax_for_plotting(activity_dict_SST,
                                                                                       predicted_activity_dict_SST,
                                                                                       activity_dict_NDNF,
                                                                                       predicted_activity_dict_NDNF,
                                                                                       activity_dict_EC,
                                                                                       predicted_activity_dict_EC,
                                                                                       residual=residual,
                                                                                       which_to_plot=which_to_plot)
    mean_quantiles_SST, sem_quantiles_SST = get_quantiles_for_cdf(activity_dict_SST, SST_factor_list, n_bins=20)

    mean_quantiles_NDNF, sem_quantiles_NDNF = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_factor_list, n_bins=20)

    mean_quantiles_EC, sem_quantiles_EC = get_quantiles_for_cdf(activity_dict_EC, EC_factor_list, n_bins=20)

    mean_quantiles_list = [mean_quantiles_SST, mean_quantiles_NDNF, mean_quantiles_EC]

    sem_quantiles_list = [sem_quantiles_SST, sem_quantiles_NDNF, sem_quantiles_EC]

    if which_to_plot == "argmin":
        if residual:
            plot_cdf(mean_quantiles_list, sem_quantiles_list, "Argmin for Residuals", "Position Bin of Minimum Firing",
                     n_bins=20)

        else:
            plot_cdf(mean_quantiles_list, sem_quantiles_list, "Argmin for Raw Data", "Position Bin of Minimum Firing",
                     n_bins=20)

    elif which_to_plot == "argmax":
        if residual:
            plot_cdf(mean_quantiles_list, sem_quantiles_list, "Argmax for Residuals", "Position Bin of Maximum Firing",
                     n_bins=20)

        else:
            plot_cdf(mean_quantiles_list, sem_quantiles_list, "Argmax for Raw Data", "Position Bin of Maximum Firing",
                     n_bins=20)

    else:
        raise ValueError("options are argmin or argmax")


def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"):

    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")

    r2_SST_above_zero = {}
    r2_SST_below_zero = {}

    for animal in r2_variable_activity_dict_SST:
        r2_SST_above_zero[animal] = {}
        r2_SST_below_zero[animal] = {}

        for neuron in r2_variable_activity_dict_SST[animal]:

            if r2_variable_activity_dict_SST[animal][neuron] >= 0:
                r2_SST_above_zero[animal][neuron] = r2_variable_activity_dict_SST[animal][neuron]
            else:
                r2_SST_below_zero[animal][neuron] = r2_variable_activity_dict_SST[animal][neuron]

    return r2_SST_above_zero, r2_SST_below_zero


def filter_activity_by_r2(activity_dict, predicted_activity_dict, r2_dict, filtered_factors_dict_SST, residual=True):

    velocity_dict = {}
    for animal in activity_dict:
        velocity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            velocity_dict[animal][neuron] = filtered_factors_dict_SST[animal]["Velocity"]

    velocity_list_out = []

    filtered_activity = {}

    for animal in r2_dict:
        if animal in activity_dict and animal in predicted_activity_dict:
            filtered_activity[animal] = {}
            for neuron in r2_dict[animal]:
                if neuron in activity_dict[animal] and neuron in predicted_activity_dict[animal]:
                    if residual:
                        filtered_activity[animal][neuron] = (activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron])
                        velocity_list_out.append(velocity_dict[animal][neuron])
                    else:
                        filtered_activity[animal][neuron] = activity_dict[animal][neuron]
                        velocity_list_out.append(velocity_dict[animal][neuron])

    return filtered_activity, velocity_list_out


def compute_mean_and_sem(activity_list):
    activity_array = np.array(activity_list)
    mean_activity = np.mean(activity_array, axis=0)
    sem_activity = np.std(activity_array, axis=0) / np.sqrt(activity_array.shape[0])
    return mean_activity, sem_activity


def get_factor_as_list(activity_SST_above):
    activity_SST_above_list = []
    for animal in activity_SST_above:
        for neuron in activity_SST_above[animal]:
            activity_SST_above_list.append(activity_SST_above[animal][neuron])

    return activity_SST_above_list


def get_activity_by_animal(activity_SST_above):
    per_animal_activity_SST_above = []

    for animal in activity_SST_above:
        animal_mean = []

        for neuron in activity_SST_above[animal]:
            trial_av = np.mean(activity_SST_above[animal][neuron], axis=1)
            animal_mean.append(trial_av)

        if not animal_mean: continue
        animal_mean_array = np.stack(animal_mean)
        per_animal_activity_SST_above.append(np.mean(animal_mean_array, axis=0))

    return per_animal_activity_SST_above

def get_activity_by_animal_quintile_split(activity_SST_above):
    per_animal_activity_SST_above = []

    for animal in activity_SST_above:
        animal_mean = []

        for neuron in activity_SST_above[animal]:
            trial_av = np.mean(activity_SST_above[animal][neuron], axis=1)
            animal_mean.append(trial_av)

        if not animal_mean: continue
        animal_mean_array = np.stack(animal_mean)
        per_animal_activity_SST_above.append(np.mean(animal_mean_array, axis=0))

    return per_animal_activity_SST_above


def get_animal_vel_correlations_activity(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):

    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")
    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, variable_to_correlate="Velocity")
    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, variable_to_correlate="Velocity")

    activity_SST_above = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_above_zero, residual=residual)
    activity_SST_below = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_below_zero, residual=residual)

    per_animal_activity_SST_above = get_activity_by_animal(activity_SST_above)
    per_animal_activity_SST_below = get_activity_by_animal(activity_SST_below)

    activity_NDNF_above = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_above_zero, residual=residual)
    activity_NDNF_below = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_below_zero, residual=residual)

    per_animal_activity_NDNF_above = get_activity_by_animal(activity_NDNF_above)
    per_animal_activity_NDNF_below = get_activity_by_animal(activity_NDNF_below)

    activity_EC_above = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_above_zero, residual=residual)
    activity_EC_below = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_below_zero, residual=residual)

    per_animal_activity_EC_above = get_activity_by_animal(activity_EC_above)
    per_animal_activity_EC_below = get_activity_by_animal(activity_EC_below)

    return per_animal_activity_SST_above, per_animal_activity_SST_below, per_animal_activity_NDNF_above, per_animal_activity_NDNF_below, per_animal_activity_EC_above, per_animal_activity_EC_below


def get_activity_by_r2_groups(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):

    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")
    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, variable_to_correlate="Velocity")
    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, variable_to_correlate="Velocity")

    activity_SST_above = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_above_zero, residual=residual)
    activity_SST_below = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_below_zero, residual=residual)

    per_animal_activity_SST_above = get_activity_by_animal(activity_SST_above)
    per_animal_activity_SST_below = get_activity_by_animal(activity_SST_below)

    activity_NDNF_above = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_above_zero, residual=residual)
    activity_NDNF_below = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_below_zero, residual=residual)

    per_animal_activity_NDNF_above = get_activity_by_animal(activity_NDNF_above)
    per_animal_activity_NDNF_below = get_activity_by_animal(activity_NDNF_below)

    activity_EC_above = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_above_zero, residual=residual)
    activity_EC_below = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_below_zero, residual=residual)

    per_animal_activity_EC_above = get_activity_by_animal(activity_EC_above)
    per_animal_activity_EC_below = get_activity_by_animal(activity_EC_below)

    activity_SST_above = get_factor_as_list(activity_SST_above)
    activity_SST_below = get_factor_as_list(activity_SST_below)
    activity_NDNF_above = get_factor_as_list(activity_NDNF_above)
    activity_NDNF_below = get_factor_as_list(activity_NDNF_below)
    activity_EC_above = get_factor_as_list(activity_EC_above)
    activity_EC_below = get_factor_as_list(activity_EC_below)

    trial_av_activity_SST_above = trial_average(activity_SST_above)
    trial_av_activity_SST_below = trial_average(activity_SST_below)

    trial_av_activity_NDNF_above = trial_average(activity_NDNF_above)
    trial_av_activity_NDNF_below = trial_average(activity_NDNF_below)

    trial_av_activity_EC_above = trial_average(activity_EC_above)
    trial_av_activity_EC_below = trial_average(activity_EC_below)

    mean_SST_above, sem_SST_above = compute_mean_and_sem(trial_av_activity_SST_above)
    mean_SST_below, sem_SST_below = compute_mean_and_sem(trial_av_activity_SST_below)

    mean_NDNF_above, sem_NDNF_above = compute_mean_and_sem(trial_av_activity_NDNF_above)
    mean_NDNF_below, sem_NDNF_below = compute_mean_and_sem(trial_av_activity_NDNF_below)

    mean_EC_above, sem_EC_above = compute_mean_and_sem(trial_av_activity_EC_above)
    mean_EC_below, sem_EC_below = compute_mean_and_sem(trial_av_activity_EC_below)

    return mean_SST_above, sem_SST_above, mean_SST_below, sem_SST_below, mean_NDNF_above, sem_NDNF_above, mean_NDNF_below, sem_NDNF_below, mean_EC_above, sem_EC_above, mean_EC_below, sem_EC_below


def split_selectivity_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=False, compute_negative=False):

    SST_factor_list, SST_negative_selectivity = get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=residual)
    NDNF_factor_list, NDNF_negative_selectivity = get_selectivity_for_plotting(activity_dict_NDNF, predicted_activity_dict_NDNF, residual=residual)
    EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(activity_dict_EC, predicted_activity_dict_EC, residual=residual)

    SST_selectivity_list = SST_negative_selectivity if compute_negative else SST_factor_list
    NDNF_selectivity_list = NDNF_negative_selectivity if compute_negative else NDNF_factor_list
    EC_selectivity_list = EC_negative_selectivity if compute_negative else EC_factor_list

    neuron_mapping_SST = [(animal, neuron) for animal in activity_dict_SST for neuron in activity_dict_SST[animal]]
    neuron_mapping_NDNF = [(animal, neuron) for animal in activity_dict_NDNF for neuron in activity_dict_NDNF[animal]]
    neuron_mapping_EC = [(animal, neuron) for animal in activity_dict_EC for neuron in activity_dict_EC[animal]]


    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST,
                                                                       filtered_factors_dict_SST,
                                                                       variable_to_correlate="Velocity")
    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF,
                                                                         predicted_activity_dict_NDNF,
                                                                         filtered_factors_dict_NDNF,
                                                                         variable_to_correlate="Velocity")
    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC,
                                                                     filtered_factors_dict_EC,
                                                                     variable_to_correlate="Velocity")

    SST_above_zero, SST_below_zero, animal_list_above_zero_SST, animal_list_below_zero_SST = split_by_r2(neuron_mapping_SST, SST_selectivity_list, r2_SST_above_zero,
                                                 r2_SST_below_zero)
    NDNF_above_zero, NDNF_below_zero, animal_list_above_zero_NDNF, animal_list_below_zero_NDNF = split_by_r2(neuron_mapping_NDNF, NDNF_selectivity_list, r2_NDNF_above_zero,
                                                   r2_NDNF_below_zero)
    EC_above_zero, EC_below_zero, animal_list_above_zero_EC, animal_list_below_zero_EC = split_by_r2(neuron_mapping_EC, EC_selectivity_list, r2_EC_above_zero,
                                               r2_EC_below_zero)

    return SST_above_zero, SST_below_zero, animal_list_above_zero_SST, animal_list_below_zero_SST, NDNF_above_zero, NDNF_below_zero, animal_list_above_zero_NDNF, animal_list_below_zero_NDNF, EC_above_zero, EC_below_zero, animal_list_above_zero_EC, animal_list_below_zero_EC


def setup_CDF_plotting_split_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=False, compute_negative=False):

    SST_factor_list, SST_negative_selectivity = get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=residual)
    NDNF_factor_list, NDNF_negative_selectivity = get_selectivity_for_plotting(activity_dict_NDNF, predicted_activity_dict_NDNF, residual=residual)
    EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(activity_dict_EC, predicted_activity_dict_EC, residual=residual)

    SST_factor_above_zero, SST_factor_below_zero, animal_list_above_zero_SST, animal_list_below_zero_SST, NDNF_factor_above_zero, NDNF_factor_below_zero, animal_list_above_zero_NDNF, animal_list_below_zero_NDNF, EC_factor_above_zero, EC_factor_below_zero, animal_list_above_zero_EC, animal_list_below_zero_EC = split_selectivity_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=residual, compute_negative=compute_negative)
    #
    # print(f"animal_list_below_zero_SST {animal_list_below_zero_SST}")
    # print(f"SST_factor_below_zero {SST_factor_below_zero}")

    mean_quantiles_SST_high, sem_quantiles_SST_high = get_quantiles_for_cdf_r2_split(animal_list_above_zero_SST, SST_factor_above_zero,
                                                                            n_bins=20)
    mean_quantiles_SST_low, sem_quantiles_SST_low = get_quantiles_for_cdf_r2_split(animal_list_below_zero_SST, SST_factor_below_zero,
                                                                          n_bins=20)

    mean_quantiles_NDNF_high, sem_quantiles_NDNF_high = get_quantiles_for_cdf_r2_split(animal_list_above_zero_NDNF,
                                                                              NDNF_factor_above_zero, n_bins=20)
    mean_quantiles_NDNF_low, sem_quantiles_NDNF_low = get_quantiles_for_cdf_r2_split(animal_list_below_zero_NDNF, NDNF_factor_below_zero,
                                                                            n_bins=20)

    mean_quantiles_EC_high, sem_quantiles_EC_high = get_quantiles_for_cdf_r2_split(animal_list_above_zero_EC, EC_factor_above_zero,
                                                                          n_bins=20)
    mean_quantiles_EC_low, sem_quantiles_EC_low = get_quantiles_for_cdf_r2_split(animal_list_below_zero_EC, EC_factor_below_zero,
                                                                        n_bins=20)


    mean_quantiles_list = [mean_quantiles_SST_high, mean_quantiles_SST_low, mean_quantiles_NDNF_high,
                           mean_quantiles_NDNF_low, mean_quantiles_EC_high, mean_quantiles_EC_low]

    sem_quantiles_list = [sem_quantiles_SST_high, sem_quantiles_SST_low, sem_quantiles_NDNF_high,
                          sem_quantiles_NDNF_low, sem_quantiles_EC_high, sem_quantiles_EC_low]


    return  mean_quantiles_list,  sem_quantiles_list


def split_by_r2(neuron_mapping, factor_list, r2_above_zero, r2_below_zero):
    above_zero = []
    below_zero = []
    animal_list_above_zero = []
    animal_list_below_zero = []

    for idx, (animal, neuron) in enumerate(neuron_mapping):
        if animal in r2_above_zero and neuron in r2_above_zero[animal]:
            above_zero.append(factor_list[idx])
            animal_list_above_zero.append(animal)
        elif animal in r2_below_zero and neuron in r2_below_zero[animal]:
            below_zero.append(factor_list[idx])
            animal_list_below_zero.append(animal)

    return above_zero, below_zero, animal_list_above_zero, animal_list_below_zero


def compute_mean_and_sem_for_quintiles(activity_dict, predicted_activity_dict):

    first_quintile_trial_av_raw = []
    last_quintile_trial_av_raw = []
    first_quintile_residuals = []
    last_quintile_residuals = []
    animal_mean_first_quintile_residuals = []
    animal_mean_last_quintile_residuals = []

    for animal in activity_dict:
        neuron_mean_first_quintile_residuals = []
        neuron_mean_last_quintile_residuals = []
        for neuron in activity_dict[animal]:

            raw_quintiles = split_into_quintiles(activity_dict[animal][neuron])
            predicted_quintiles = split_into_quintiles(predicted_activity_dict[animal][neuron])

            trial_av_first_quintile = np.mean(raw_quintiles[0], axis=1)
            trial_av_first_quintile_predicted = np.mean(predicted_quintiles[0], axis=1)

            first_quintile_trial_av_raw.append(trial_av_first_quintile)

            trial_av_last_quintile = np.mean(raw_quintiles[-1], axis=1)
            trial_av_last_quintile_predicted = np.mean(predicted_quintiles[-1], axis=1)

            last_quintile_trial_av_raw.append(trial_av_last_quintile)

            first_quintile_residual = trial_av_first_quintile - trial_av_first_quintile_predicted
            last_quintile_residual = trial_av_last_quintile - trial_av_last_quintile_predicted

            neuron_mean_first_quintile_residuals.append(first_quintile_residual)
            neuron_mean_last_quintile_residuals.append(last_quintile_residual)

            first_quintile_residuals.append(first_quintile_residual)
            last_quintile_residuals.append(last_quintile_residual)


        animal_mean_first_quintile_residuals.append(np.mean(np.stack(neuron_mean_first_quintile_residuals), axis=0))
        animal_mean_last_quintile_residuals.append(np.mean(np.stack(neuron_mean_last_quintile_residuals), axis=0))

    first_quintile_residuals = np.array(first_quintile_residuals)
    last_quintile_residuals = np.array(last_quintile_residuals)

    first_mean_residual = np.mean(first_quintile_residuals, axis=0)  # Mean across neurons
    first_sem_residual = np.std(first_quintile_residuals, axis=0) / np.sqrt(first_quintile_residuals.shape[0])

    last_mean_residual = np.mean(last_quintile_residuals, axis=0)
    last_sem_residual = np.std(last_quintile_residuals, axis=0) / np.sqrt(last_quintile_residuals.shape[0])


    return first_quintile_trial_av_raw, last_quintile_trial_av_raw, first_mean_residual, first_sem_residual, last_mean_residual, last_sem_residual, animal_mean_first_quintile_residuals, animal_mean_last_quintile_residuals, first_quintile_residuals, last_quintile_residuals


def get_max_or_min(trial_av_list, argmax_or_argmin="argmax"):
    value_list = []
    for i in trial_av_list:
        if argmax_or_argmin == "argmax":
            value_list.append(np.argmax(i))
        elif argmax_or_argmin == "argmin":
            value_list.append(np.argmin(i))
        else:
            raise ValueError("argmax_or_argmin arguements are eihter argmax or argmin")

    return value_list


def get_argmin_argmax_split_learning_histogram(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, argmax_or_argmin=None):


    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(
        activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q5_NDNF = split_activity_and_prediction_into_quintiles(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, prediction_list_EC_q1, prediction_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(
        activity_dict_EC, predicted_activity_dict_EC)

    argmax_SST_q1 = get_max_or_min(residual_q1_SST, argmax_or_argmin=argmax_or_argmin)
    argmax_SST_q5 = get_max_or_min(residual_q5_SST, argmax_or_argmin=argmax_or_argmin)

    argmax_NDNF_q1 = get_max_or_min(residual_q1_NDNF, argmax_or_argmin=argmax_or_argmin)
    argmax_NDNF_q5 = get_max_or_min(residual_q5_NDNF, argmax_or_argmin=argmax_or_argmin)

    argmax_EC_q1 = get_max_or_min(residual_q1_EC, argmax_or_argmin=argmax_or_argmin)
    argmax_EC_q5 = get_max_or_min(residual_q5_EC, argmax_or_argmin=argmax_or_argmin)

    return argmax_SST_q1, argmax_SST_q5, argmax_NDNF_q1, argmax_NDNF_q5, argmax_EC_q1, argmax_EC_q5


def get_pop_correlation_to_variable(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)

    filtered_input_variables = {var: [] for var in filtered_factors_dict_SST[next(iter(filtered_factors_dict_SST))].keys()}

    for animal in activity_dict_SST:
        for neuron in activity_dict_SST[animal]:
            for var in filtered_factors_dict_SST[animal]:
                filtered_input_variables[var].append(filtered_factors_dict_SST[animal][var])

    r2_variable_activity_dict = {}
    r2_variable_residual_dict = {}

    idx = 0
    for animal in activity_dict_SST:
        r2_variable_activity_dict[animal] = {}
        r2_variable_residual_dict[animal] = {}
        for neuron in activity_dict_SST[animal]:
            flat_neuron_activity = neuron_activity_list_SST[idx].flatten()
            flat_residual = cell_residual_list_SST[idx].flatten()
            flat_variable_of_interest = filtered_input_variables[variable_to_correlate][idx].flatten()
            r2_variable_activity, _ = pearsonr(flat_neuron_activity, flat_variable_of_interest)
            r2_variable_residual, _ = pearsonr(flat_residual, flat_variable_of_interest)

            r2_variable_activity_dict[animal][neuron] = r2_variable_activity
            r2_variable_residual_dict[animal][neuron] = r2_variable_residual

            idx += 1

    return r2_variable_activity_dict, r2_variable_residual_dict


def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"):
    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(activity_dict_SST,
                                                                                                   predicted_activity_dict_SST,
                                                                                                   filtered_factors_dict_SST,
                                                                                                   variable_to_correlate="Velocity")

    r2_SST_above_zero = {}
    r2_SST_below_zero = {}

    for animal in r2_variable_activity_dict_SST:
        r2_SST_above_zero[animal] = {}
        r2_SST_below_zero[animal] = {}

        for neuron in r2_variable_activity_dict_SST[animal]:

            if r2_variable_activity_dict_SST[animal][neuron] >= 0:
                r2_SST_above_zero[animal][neuron] = r2_variable_activity_dict_SST[animal][neuron]
            else:
                r2_SST_below_zero[animal][neuron] = r2_variable_activity_dict_SST[animal][neuron]

    return r2_SST_above_zero, r2_SST_below_zero


def split_argmin_argmax_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, r2_SST_above_zero, r2_SST_below_zero, r2_NDNF_above_zero, r2_NDNF_below_zero, r2_EC_above_zero, r2_EC_below_zero, residual=False, which_to_plot="argmin"):


    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(activity_dict_EC, predicted_activity_dict_EC)

    trial_av_activity_SST = trial_average(cell_residual_list_SST) if residual else trial_average(neuron_activity_list_SST)
    trial_av_activity_NDNF = trial_average(cell_residual_list_NDNF) if residual else trial_average(neuron_activity_list_NDNF)
    trial_av_activity_EC = trial_average(cell_residual_list_EC) if residual else trial_average(neuron_activity_list_EC)

    SST_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_SST]
    NDNF_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_NDNF]
    EC_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_EC]

    neuron_mapping_SST = [(animal, neuron) for animal in activity_dict_SST for neuron in activity_dict_SST[animal]]
    neuron_mapping_NDNF = [(animal, neuron) for animal in activity_dict_NDNF for neuron in activity_dict_NDNF[animal]]
    neuron_mapping_EC = [(animal, neuron) for animal in activity_dict_EC for neuron in activity_dict_EC[animal]]

    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")

    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, variable_to_correlate="Velocity")

    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, variable_to_correlate="Velocity")

    SST_above_zero, SST_below_zero, animal_list_above_zero_SST, animal_list_below_zero_SST = split_by_r2(neuron_mapping_SST, SST_factor_list, r2_SST_above_zero, r2_SST_below_zero)

    NDNF_above_zero, NDNF_below_zero, animal_list_above_zero_NDNF, animal_list_below_zero_NDNF = split_by_r2(neuron_mapping_NDNF, NDNF_factor_list, r2_NDNF_above_zero, r2_NDNF_below_zero)

    EC_above_zero, EC_below_zero, animal_list_above_zero_EC, animal_list_below_zero_EC = split_by_r2(neuron_mapping_EC, EC_factor_list, r2_EC_above_zero, r2_EC_below_zero)

    return SST_above_zero, SST_below_zero, NDNF_above_zero, NDNF_below_zero, EC_above_zero, EC_below_zero


def setup_CDF_plotting_argmin_argmax_split_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=False, which_to_plot="argmin"):

    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")
    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, variable_to_correlate="Velocity")
    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, variable_to_correlate="Velocity")

    SST_above_zero, SST_below_zero, NDNF_above_zero, NDNF_below_zero, EC_above_zero, EC_below_zero = split_argmin_argmax_by_r2(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, r2_SST_above_zero, r2_SST_below_zero, r2_NDNF_above_zero, r2_NDNF_below_zero, r2_EC_above_zero, r2_EC_below_zero, residual=residual, which_to_plot=which_to_plot)

    mean_quantiles_SST_high, sem_quantiles_SST_high = get_quantiles_for_cdf(activity_dict_SST, SST_above_zero, n_bins=20)
    mean_quantiles_SST_low, sem_quantiles_SST_low = get_quantiles_for_cdf(activity_dict_SST, SST_below_zero, n_bins=20)

    mean_quantiles_NDNF_high, sem_quantiles_NDNF_high = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_above_zero, n_bins=20)
    mean_quantiles_NDNF_low, sem_quantiles_NDNF_low = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_below_zero, n_bins=20)

    mean_quantiles_EC_high, sem_quantiles_EC_high = get_quantiles_for_cdf(activity_dict_EC, EC_above_zero, n_bins=20)
    mean_quantiles_EC_low, sem_quantiles_EC_low = get_quantiles_for_cdf(activity_dict_EC, EC_below_zero, n_bins=20)

    mean_quantiles_list = [mean_quantiles_SST_high, mean_quantiles_SST_low, mean_quantiles_NDNF_high, mean_quantiles_NDNF_low, mean_quantiles_EC_high, mean_quantiles_EC_low]

    sem_quantiles_list = [sem_quantiles_SST_high, sem_quantiles_SST_low, sem_quantiles_NDNF_high, sem_quantiles_NDNF_low, sem_quantiles_EC_high, sem_quantiles_EC_low]

    return mean_quantiles_list, sem_quantiles_list


def split_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(
        activity_dict_EC, predicted_activity_dict_EC)

    trial_av_activity_SST = trial_average(cell_residual_list_SST) if residual else trial_average(
        neuron_activity_list_SST)
    trial_av_activity_NDNF = trial_average(cell_residual_list_NDNF) if residual else trial_average(
        neuron_activity_list_NDNF)
    trial_av_activity_EC = trial_average(cell_residual_list_EC) if residual else trial_average(neuron_activity_list_EC)

    SST_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_SST]
    NDNF_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_NDNF]
    EC_factor_list = [np.argmin(i) if which_to_plot == "argmin" else np.argmax(i) for i in trial_av_activity_EC]

    return SST_factor_list, NDNF_factor_list, EC_factor_list


def split_into_quintiles(array):
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    total_columns = array.shape[1]
    quintile_size = total_columns // 5

    usable_columns = quintile_size * 5
    truncated_array = array[:, :usable_columns]

    split_array = np.split(truncated_array, 5, axis=1)

    return split_array


def split_activity_and_prediction_into_quintiles(activity_dict_SST, predicted_activity_dict_SST):
    activity_list_SST_q1 = []
    activity_list_SST_q5 = []

    for animal in activity_dict_SST:
        for neuron in activity_dict_SST[animal]:
            sst_quintiles = split_into_quintiles(activity_dict_SST[animal][neuron])
            first_quintile = sst_quintiles[0]
            last_quintile = sst_quintiles[-1]
            activity_list_SST_q1.append(first_quintile)
            activity_list_SST_q5.append(last_quintile)

    prediction_list_SST_q1 = []
    prediction_list_SST_q5 = []

    for animal in predicted_activity_dict_SST:
        for neuron in predicted_activity_dict_SST[animal]:
            sst_quintiles_prediction = split_into_quintiles(predicted_activity_dict_SST[animal][neuron])
            first_quintile_prediction = sst_quintiles_prediction[0]
            last_quintile_prediction = sst_quintiles_prediction[-1]
            prediction_list_SST_q1.append(first_quintile_prediction)
            prediction_list_SST_q5.append(last_quintile_prediction)


    residual_q1 = np.array(trial_average(activity_list_SST_q1)) - np.array(trial_average(prediction_list_SST_q1))
    residual_q5 = np.array(trial_average(activity_list_SST_q5)) - np.array(trial_average(prediction_list_SST_q5))

    return activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1, residual_q5


def get_quantiles_for_cdf_list(animal_ID_list, values_list, n_bins=None):
    animal_to_values = defaultdict(list)

    for animal, value in zip(animal_ID_list, values_list):
        animal_to_values[animal].append(value)

    animal_to_values = dict(animal_to_values)

    edges_list = []
    sep_dict = {}

    for key, value in animal_to_values.items():
        animal_edges = np.quantile(value, np.linspace(0, 1, n_bins))
        edges_list.append(animal_edges)

    stacked_edges = np.vstack(edges_list)

    mean_list = []
    sem_list = []

    for i in range(stacked_edges.shape[1]):
        column = stacked_edges[:, i]
        mean_list.append(np.mean(column))
        sem_list.append((np.std(column) / np.sqrt(len(column))))

    return mean_list, sem_list


def quint_list(activity_SST_above):
    first_quintile_list = []
    last_quintile_list = []

    for i in activity_SST_above:
        sst_quintiles = split_into_quintiles(i)
        first_quintile = sst_quintiles[0]
        first_quintile_list.append(first_quintile)
        last_quintile = sst_quintiles[-1]
        last_quintile_list.append(last_quintile)
    return first_quintile_list, last_quintile_list


def get_activity_by_r2_groups_plus_learning(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):
    r2_SST_above_zero, r2_SST_below_zero = get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity")
    r2_NDNF_above_zero, r2_NDNF_below_zero = get_r2_above_and_below_zero(activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, variable_to_correlate="Velocity")
    r2_EC_above_zero, r2_EC_below_zero = get_r2_above_and_below_zero(activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, variable_to_correlate="Velocity")

    activity_SST_above, velocity_list_out_SST_above = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_above_zero, filtered_factors_dict_SST, residual=residual)
    activity_SST_below, velocity_list_out_SST_below = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_below_zero, filtered_factors_dict_SST, residual=residual)

    activity_NDNF_above, velocity_list_out_NDNF_above = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_above_zero, filtered_factors_dict_NDNF, residual=residual)
    activity_NDNF_below, velocity_list_out_NDNF_above = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_below_zero, filtered_factors_dict_NDNF, residual=residual)

    activity_EC_above, velocity_list_out_EC_above = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_above_zero, filtered_factors_dict_EC, residual=residual)
    activity_EC_below, velocity_list_out_EC_above = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_below_zero, filtered_factors_dict_EC, residual=residual)

    per_animal_activity_SST_above = get_activity_by_animal(activity_SST_above)
    per_animal_activity_SST_below = get_activity_by_animal(activity_SST_below)

    per_animal_activity_NDNF_above = get_activity_by_animal(activity_NDNF_above)
    per_animal_activity_NDNF_below = get_activity_by_animal(activity_NDNF_below)

    per_animal_activity_EC_above = get_activity_by_animal(activity_EC_above)
    per_animal_activity_EC_below = get_activity_by_animal(activity_EC_below)

    activity_SST_above = get_factor_as_list(activity_SST_above)
    activity_SST_below = get_factor_as_list(activity_SST_below)
    activity_NDNF_above = get_factor_as_list(activity_NDNF_above)
    activity_NDNF_below = get_factor_as_list(activity_NDNF_below)
    activity_EC_above = get_factor_as_list(activity_EC_above)
    activity_EC_below = get_factor_as_list(activity_EC_below)

    first_quintile_list_SST_above, last_quintile_list_SST_above = quint_list(activity_SST_above)
    first_quintile_list_SST_below, last_quintile_list_SST_below = quint_list(activity_SST_below)

    first_quintile_list_NDNF_above, last_quintile_list_NDNF_above = quint_list(activity_NDNF_above)
    first_quintile_list_NDNF_below, last_quintile_list_NDNF_below = quint_list(activity_NDNF_below)

    first_quintile_list_EC_above, last_quintile_list_EC_above = quint_list(activity_EC_above)
    first_quintile_list_EC_below, last_quintile_list_EC_below = quint_list(activity_EC_below)

    ######################

    correlation_above_list_all_trials = []
    correlation_below_list_all_trials = []

    correlation_above_list_q1 = []
    correlation_below_list_q1 = []

    correlation_above_list_q5 = []
    correlation_below_list_q5 = []

    for i in range(len(activity_SST_above)):
        flat_activity_SST_above = activity_SST_above[i].flatten()
        flat_velocity_SST_above = velocity_list_out_SST_above[i].flatten()
        correlation_above, _ = pearsonr(flat_activity_SST_above, flat_velocity_SST_above)
        correlation_above_list_all_trials.append(correlation_above)

        quintiles = split_into_quintiles(activity_SST_above[i])
        quintiles_first = quintiles[0].flatten()
        quintiles_last = quintiles[-1].flatten()
        quintiles_velocity = split_into_quintiles(velocity_list_out_SST_above[i])
        quintiles_velocity_first = quintiles_velocity[0].flatten()
        quintiles_velocity_last = quintiles_velocity[-1].flatten()

        correlation_first_above, _ = pearsonr(quintiles_first, quintiles_velocity_first)
        correlation_above_list_q1.append(correlation_first_above)
        correlation_last_above, _ = pearsonr(quintiles_last, quintiles_velocity_last)
        correlation_above_list_q5.append(correlation_last_above)

    for i in range(len(activity_SST_below)):
        flat_activity_SST_below = activity_SST_below[i].flatten()
        flat_velocity_SST_below = velocity_list_out_SST_below[i].flatten()
        correlation_below, _ = pearsonr(flat_activity_SST_below, flat_velocity_SST_below)
        correlation_below_list_all_trials.append(correlation_below)

        quintiles = split_into_quintiles(activity_SST_below[i])
        quintiles_first = quintiles[0].flatten()
        quintiles_last = quintiles[-1].flatten()
        quintiles_velocity = split_into_quintiles(velocity_list_out_SST_below[i])
        quintiles_velocity_first = quintiles_velocity[0].flatten()
        quintiles_velocity_last = quintiles_velocity[-1].flatten()

        correlation_first_below, _ = pearsonr(quintiles_first, quintiles_velocity_first)
        correlation_last_below, _ = pearsonr(quintiles_last, quintiles_velocity_last)
        correlation_below_list_q1.append(correlation_first_below)
        correlation_below_list_q5.append(correlation_last_below)

    data = [correlation_above_list_all_trials, correlation_below_list_all_trials, correlation_above_list_q1, correlation_below_list_q1, correlation_above_list_q5, correlation_below_list_q5]

    labels = ["Above All", "Below All", "Above Q1", "Below Q1", "Above Q5", "Below Q5"]

    plt.figure(figsize=(10, 6))

    correlation_lists = [
        correlation_above_list_all_trials,
        correlation_below_list_all_trials,
        correlation_above_list_q1,
        correlation_above_list_q5,
        correlation_below_list_q1,
        correlation_below_list_q5
    ]

    labels = ["Above All", "Below All", "Above Q1", "Above Q5", "Below Q1", "Below Q5"]

    plt.figure(figsize=(10, 6))

    x_positions = list(range(len(labels)))

    for i, (corr_list, label) in enumerate(zip(correlation_lists, labels)):
        x = np.full(len(corr_list), i)  # Assign x position
        plt.scatter(x, corr_list, color='gray', alpha=0.7)  # Plot dots
        mean_value = np.mean(corr_list)
        plt.plot([i - 0.2, i + 0.2], [mean_value, mean_value], color='black', lw=2)  # Plot mean line

    if len(correlation_above_list_q1) == len(correlation_above_list_q5):
        for q1, q5 in zip(correlation_above_list_q1, correlation_above_list_q5):
            plt.plot([x_positions[2], x_positions[3]], [q1, q5], color='red', linestyle='--', alpha=0.5)

    if len(correlation_below_list_q1) == len(correlation_below_list_q5):
        for q1, q5 in zip(correlation_below_list_q1, correlation_below_list_q5):
            plt.plot([x_positions[4], x_positions[5]], [q1, q5], color='blue', linestyle='--', alpha=0.5)

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.ylabel("Correlation Values")
    plt.title("R² Values Across Conditions with Q1-Q5 Connections")
    plt.tight_layout()
    plt.show()

    #####################################

    trial_av_activity_SST_above_q1 = trial_average(first_quintile_list_SST_above)
    trial_av_activity_SST_above_q5 = trial_average(last_quintile_list_SST_above)
    trial_av_activity_SST_below_q1 = trial_average(first_quintile_list_SST_below)
    trial_av_activity_SST_below_q5 = trial_average(last_quintile_list_SST_below)

    trial_average_activity_list_SST = [trial_av_activity_SST_above_q1, trial_av_activity_SST_above_q5, trial_av_activity_SST_below_q1, trial_av_activity_SST_below_q5]

    trial_av_activity_NDNF_above_q1 = trial_average(first_quintile_list_NDNF_above)
    trial_av_activity_NDNF_above_q5 = trial_average(last_quintile_list_NDNF_above)
    trial_av_activity_NDNF_below_q1 = trial_average(first_quintile_list_NDNF_below)
    trial_av_activity_NDNF_below_q5 = trial_average(last_quintile_list_NDNF_below)

    trial_av_activity_EC_above_q1 = trial_average(first_quintile_list_EC_above)
    trial_av_activity_EC_above_q5 = trial_average(last_quintile_list_EC_above)
    trial_av_activity_EC_below_q1 = trial_average(first_quintile_list_EC_below)
    trial_av_activity_EC_below_q5 = trial_average(last_quintile_list_EC_below)

    mean_SST_above_q1, sem_SST_above_q1 = compute_mean_and_sem(trial_av_activity_SST_above_q1)
    mean_SST_above_q5, sem_SST_above_q5 = compute_mean_and_sem(trial_av_activity_SST_above_q5)
    mean_SST_below_q1, sem_SST_below_q1 = compute_mean_and_sem(trial_av_activity_SST_below_q1)
    mean_SST_below_q5, sem_SST_below_q5 = compute_mean_and_sem(trial_av_activity_SST_below_q5)

    mean_NDNF_above_q1, sem_NDNF_above_q1 = compute_mean_and_sem(trial_av_activity_NDNF_above_q1)
    mean_NDNF_above_q5, sem_NDNF_above_q5 = compute_mean_and_sem(trial_av_activity_NDNF_above_q5)
    mean_NDNF_below_q1, sem_NDNF_below_q1 = compute_mean_and_sem(trial_av_activity_NDNF_below_q1)
    mean_NDNF_below_q5, sem_NDNF_below_q5 = compute_mean_and_sem(trial_av_activity_NDNF_below_q5)

    mean_EC_above_q1, sem_EC_above_q1 = compute_mean_and_sem(trial_av_activity_EC_above_q1)
    mean_EC_above_q5, sem_EC_above_q5 = compute_mean_and_sem(trial_av_activity_EC_above_q5)
    mean_EC_below_q1, sem_EC_below_q1 = compute_mean_and_sem(trial_av_activity_EC_below_q1)
    mean_EC_below_q5, sem_EC_below_q5 = compute_mean_and_sem(trial_av_activity_EC_below_q5)

    mean_above_list = [mean_SST_above_q1, mean_SST_above_q5, mean_NDNF_above_q1, mean_NDNF_above_q5, mean_EC_above_q1, mean_EC_above_q5]
    mean_below_list = [mean_SST_below_q1, mean_SST_below_q5, mean_NDNF_below_q1, mean_NDNF_below_q5, mean_EC_below_q1, mean_EC_below_q5]

    sem_above_list = [sem_SST_above_q1, sem_SST_above_q5, sem_NDNF_above_q1, sem_NDNF_above_q5, sem_EC_above_q1, sem_EC_above_q5]
    sem_below_list = [sem_SST_below_q1, sem_SST_below_q5, sem_NDNF_below_q1, sem_NDNF_below_q5, sem_EC_below_q1, sem_EC_below_q5]

    return mean_above_list, mean_below_list, sem_above_list, sem_below_list, trial_average_activity_list_SST


def confirm_model(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict, activity_dict, cell_number, variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"], variable_into_GLM="Velocity", cell_type="SST", quintile="Q1"):
    input_variables = {var: [] for var in factors_dict[next(iter(factors_dict))].keys()}
    filtered_input_variables = {var: [] for var in filtered_factors_dict[next(iter(filtered_factors_dict))].keys()}

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            for var in factors_dict[animal]:
                input_variables[var].append(factors_dict[animal][var])
            for var in filtered_factors_dict[animal]:
                filtered_input_variables[var].append(filtered_factors_dict[animal][var])

    neuron_activity_list, predictions_list, cell_residual_list = get_neuron_activity_prediction_residual(activity_dict, predicted_activity_dict)

    velocity_vs_activity_correlation_list = []
    velocity_vs_residuals_correlation_list = []

    cell_activity = neuron_activity_list[cell_number]
    cell_residual = cell_residual_list[cell_number]
    variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]

    sst_quintiles_activity = split_into_quintiles(cell_activity)
    sst_quintiles_residual = split_into_quintiles(cell_residual)
    sst_quintiles_velocity = split_into_quintiles(variable_of_interest)

    q1_activity = sst_quintiles_activity[0].flatten()
    q2_activity = sst_quintiles_activity[1].flatten()
    q3_activity = sst_quintiles_activity[2].flatten()
    q4_activity = sst_quintiles_activity[3].flatten()
    q5_activity = sst_quintiles_activity[4].flatten()

    q1_residual = sst_quintiles_residual[0].flatten()
    q2_residual = sst_quintiles_residual[1].flatten()
    q3_residual = sst_quintiles_residual[2].flatten()
    q4_residual = sst_quintiles_residual[3].flatten()
    q5_residual = sst_quintiles_residual[4].flatten()
    #
    # for animal in activity_dict:
    #     sst_quintiles_animal
    q1_velocity = sst_quintiles_velocity[0]
    q2_velocity = sst_quintiles_velocity[1]
    q3_velocity = sst_quintiles_velocity[2]
    q4_velocity = sst_quintiles_velocity[3]
    q5_velocity = sst_quintiles_velocity[4]

    oveall_mean_q1_velocity = np.mean(q1_velocity)
    oveall_mean_q2_velocity = np.mean(q2_velocity)
    oveall_mean_q3_velocity = np.mean(q3_velocity)
    oveall_mean_q4_velocity = np.mean(q4_velocity)
    oveall_mean_q5_velocity = np.mean(q5_velocity)

    sem_velocity_q1 = sem(sst_quintiles_velocity[0], axis=1)
    sem_velocity_q2 = sem(sst_quintiles_velocity[1], axis=1)
    sem_velocity_q3 = sem(sst_quintiles_velocity[2], axis=1)
    sem_velocity_q4 = sem(sst_quintiles_velocity[3], axis=1)
    sem_velocity_q5 = sem(sst_quintiles_velocity[4], axis=1)

    mean_velocity_q1 = np.mean(sst_quintiles_velocity[0], axis=1)
    mean_velocity_q2 = np.mean(sst_quintiles_velocity[1], axis=1)
    mean_velocity_q3 = np.mean(sst_quintiles_velocity[2], axis=1)
    mean_velocity_q4 = np.mean(sst_quintiles_velocity[3], axis=1)
    mean_velocity_q5 = np.mean(sst_quintiles_velocity[4], axis=1)

    print(mean_velocity_q1.shape)

    plt.figure()
    plt.plot(mean_velocity_q1, color='b', label=f"Q1 {oveall_mean_q1_velocity:.3f}")
    plt.fill_between(range(len(mean_velocity_q1)), mean_velocity_q1 + sem_velocity_q1, mean_velocity_q1 - sem_velocity_q1, color='b', alpha=0.2)
    plt.plot(mean_velocity_q2, color='r', label=f"Q2 {oveall_mean_q2_velocity:.3f}")
    plt.fill_between(range(len(mean_velocity_q2)), mean_velocity_q2 + sem_velocity_q2, mean_velocity_q2 - sem_velocity_q2, color='r', alpha=0.2)
    plt.plot(mean_velocity_q3, color='orange', label=f"Q3 {oveall_mean_q3_velocity:.3f}")
    plt.fill_between(range(len(mean_velocity_q3)), mean_velocity_q3 + sem_velocity_q3, mean_velocity_q3 - sem_velocity_q3, color='orange', alpha=0.2)
    plt.plot(mean_velocity_q4, color='cyan', label=f"Q4 {oveall_mean_q4_velocity:.3f}")
    plt.fill_between(range(len(mean_velocity_q4)), mean_velocity_q4 + sem_velocity_q4, mean_velocity_q4 - sem_velocity_q4, color='cyan', alpha=0.2)
    plt.plot(mean_velocity_q5, color='green', label=f"Q5 {oveall_mean_q5_velocity:.3f}")
    plt.fill_between(range(len(mean_velocity_q5)), mean_velocity_q5 + sem_velocity_q5, mean_velocity_q5 - sem_velocity_q5, color='green', alpha=0.2)
    plt.ylabel("Velocity")
    plt.xlabel("Position Bin")
    plt.legend()
    plt.show()

    q1_velocity = sst_quintiles_velocity[0].flatten()
    q2_velocity = sst_quintiles_velocity[1].flatten()
    q3_velocity = sst_quintiles_velocity[2].flatten()
    q4_velocity = sst_quintiles_velocity[3].flatten()
    q5_velocity = sst_quintiles_velocity[4].flatten()

    activity_vs_velocity_q1, _ = pearsonr(q1_activity, q1_velocity)
    activity_vs_velocity_q2, _ = pearsonr(q1_activity, q2_velocity)
    activity_vs_velocity_q3, _ = pearsonr(q1_activity, q3_velocity)
    activity_vs_velocity_q4, _ = pearsonr(q1_activity, q4_velocity)
    activity_vs_velocity_q5, _ = pearsonr(q1_activity, q5_velocity)

    residual_vs_velocity_q1, _ = pearsonr(q1_residual, q1_velocity)
    residual_vs_velocity_q2, _ = pearsonr(q1_residual, q2_velocity)
    residual_vs_velocity_q3, _ = pearsonr(q1_residual, q3_velocity)
    residual_vs_velocity_q4, _ = pearsonr(q1_residual, q4_velocity)
    residual_vs_velocity_q5, _ = pearsonr(q1_residual, q5_velocity)

    activity_vs_velocity_list = [activity_vs_velocity_q1, activity_vs_velocity_q2, activity_vs_velocity_q3, activity_vs_velocity_q4, activity_vs_velocity_q5]
    residual_vs_velocity_list = [residual_vs_velocity_q1, residual_vs_velocity_q2, residual_vs_velocity_q3, residual_vs_velocity_q4, residual_vs_velocity_q5]

    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    x_activity = np.ones(len(activity_vs_velocity_list)) * 1  # Group 1
    x_residual = np.ones(len(residual_vs_velocity_list)) * 2  # Group 2

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(x_activity, activity_vs_velocity_list, color='blue', label="Activity vs Velocity")
    ax.scatter(x_residual, residual_vs_velocity_list, color='red', label="Residual vs Velocity")

    for i in range(len(activity_vs_velocity_list)):
        ax.plot([x_activity[i], x_residual[i]], [activity_vs_velocity_list[i], residual_vs_velocity_list[i]],
                color='gray', linestyle='-', alpha=0.7)

    ax.hlines(np.mean(activity_vs_velocity_list), xmin=0.9, xmax=1.1, colors='blue', linestyles='dashed', linewidth=2)
    ax.hlines(np.mean(residual_vs_velocity_list), xmin=1.9, xmax=2.1, colors='red', linestyles='dashed', linewidth=2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Activity vs Velocity", "Residual vs Velocity"])
    ax.set_ylabel("Pearson Correlation")

    for i, label in enumerate(quintile_labels):
        ax.text(1.05, activity_vs_velocity_list[i], label, fontsize=10, verticalalignment='center', color='blue')
        ax.text(2.05, residual_vs_velocity_list[i], label, fontsize=10, verticalalignment='center', color='red')

    # Show the plot
    plt.tight_layout()
    plt.show()

    sst_quintiles_residual = split_into_quintiles(cell_residual)
    sst_quintiles_velocity = split_into_quintiles(variable_of_interest)
    q1_residual = sst_quintiles_residual[0]
    q2_residual = sst_quintiles_residual[1]
    q3_residual = sst_quintiles_residual[2]
    q4_residual = sst_quintiles_residual[3]
    q5_residual = sst_quintiles_residual[4]

    q1_velocity = sst_quintiles_velocity[0]
    q2_velocity = sst_quintiles_velocity[1]
    q3_velocity = sst_quintiles_velocity[2]
    q4_velocity = sst_quintiles_velocity[3]
    q5_velocity = sst_quintiles_velocity[4]

    correlation_list_q1 = []
    correlation_list_q2 = []
    correlation_list_q3 = []
    correlation_list_q4 = []
    correlation_list_q5 = []

    for trial in range(q1_residual.shape[1]):
        residual_vel_correlation, _ = pearsonr(q1_residual[:, trial], q1_velocity[:, trial])
        correlation_list_q1.append(residual_vel_correlation)

    for trial in range(q2_residual.shape[1]):
        residual_vel_correlation, _ = pearsonr(q2_residual[:, trial], q2_velocity[:, trial])
        correlation_list_q2.append(residual_vel_correlation)

    for trial in range(q3_residual.shape[1]):
        residual_vel_correlation, _ = pearsonr(q3_residual[:, trial], q3_velocity[:, trial])
        correlation_list_q3.append(residual_vel_correlation)

    for trial in range(q4_residual.shape[1]):
        residual_vel_correlation, _ = pearsonr(q4_residual[:, trial], q4_velocity[:, trial])
        correlation_list_q4.append(residual_vel_correlation)

    for trial in range(q5_residual.shape[1]):
        residual_vel_correlation, _ = pearsonr(q5_residual[:, trial], q5_velocity[:, trial])
        correlation_list_q5.append(residual_vel_correlation)

    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    correlation_lists = [correlation_list_q1, correlation_list_q2, correlation_list_q3, correlation_list_q4, correlation_list_q5]

    correlation_arrays = [np.array(corrs) for corrs in correlation_lists]

    x_positions = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, correlations in enumerate(correlation_arrays):
        x_jitter = np.random.uniform(-0.05, 0.05, size=len(correlations))
        ax.scatter(np.full_like(correlations, x_positions[i]) + x_jitter, correlations, alpha=0.6, label=f"{quintile_labels[i]}")

        mean_corr = np.mean(correlations)
        ax.plot([x_positions[i] - 0.2, x_positions[i] + 0.2], [mean_corr, mean_corr], color='black', linewidth=2)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(quintile_labels, fontsize=12)
    ax.set_ylabel("Pearson Correlation (r)", fontsize=12)
    ax.set_title(f"Velocity vs Residuals Correlation Across Quintiles Cell#{cell_number}", fontsize=14)

    plt.show()


def get_selectivity_for_plotting_lists(neuron_activity_list_SST, neuron_activity_list_NDNF, neuron_activity_list_EC):

    trial_av_activity_SST = neuron_activity_list_SST
    trial_av_activity_NDNF = neuron_activity_list_NDNF
    trial_av_activity_EC = neuron_activity_list_EC

    SST_negative_selectivity = []
    SST_factor_list = []
    for i in trial_av_activity_SST:
        selectivity = Vinje2000(i, norm='min_max', negative_selectivity=False)
        negative_selectivity = Vinje2000(i, norm='min_max', negative_selectivity=True)
        SST_factor_list.append(selectivity)
        SST_negative_selectivity.append(negative_selectivity)

    NDNF_negative_selectivity = []
    NDNF_factor_list = []
    for i in trial_av_activity_NDNF:
        selectivity = Vinje2000(i, norm='min_max', negative_selectivity=False)
        negative_selectivity = Vinje2000(i, norm='min_max', negative_selectivity=True)
        NDNF_factor_list.append(selectivity)
        NDNF_negative_selectivity.append(negative_selectivity)

    EC_negative_selectivity = []
    EC_factor_list = []
    for i in trial_av_activity_EC:
        selectivity = Vinje2000(i, norm='min_max', negative_selectivity=False)
        negative_selectivity = Vinje2000(i, norm='min_max', negative_selectivity=True)
        EC_factor_list.append(selectivity)
        EC_negative_selectivity.append(negative_selectivity)

    return SST_factor_list, SST_negative_selectivity, NDNF_factor_list, NDNF_negative_selectivity, EC_factor_list, EC_negative_selectivity


def get_animal_ID_list(activity_dict_SST):
    animal_ID_list = []
    for animal in activity_dict_SST:
        for neuron in activity_dict_SST[animal]:
            animal_ID_list.append(animal)
    return animal_ID_list


def get_argmin_argmax_split_learning(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):

    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(
        activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q5_NDNF = split_activity_and_prediction_into_quintiles(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, prediction_list_EC_q1, prediction_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(
        activity_dict_EC, predicted_activity_dict_EC)


    argmax_SST_q1 = get_max_or_min(residual_q1_SST, argmax_or_argmin="argmax")
    argmin_SST_q1 = get_max_or_min(residual_q1_SST, argmax_or_argmin="argmin")
    argmax_SST_q5 = get_max_or_min(residual_q5_SST, argmax_or_argmin="argmax")
    argmin_SST_q5 = get_max_or_min(residual_q5_SST, argmax_or_argmin="argmin")

    argmax_NDNF_q1 = get_max_or_min(residual_q1_NDNF, argmax_or_argmin="argmax")
    argmin_NDNF_q1 = get_max_or_min(residual_q1_NDNF, argmax_or_argmin="argmin")
    argmax_NDNF_q5 = get_max_or_min(residual_q5_NDNF, argmax_or_argmin="argmax")
    argmin_NDNF_q5 = get_max_or_min(residual_q5_NDNF, argmax_or_argmin="argmin")

    argmax_EC_q1 = get_max_or_min(residual_q1_EC, argmax_or_argmin="argmax")
    argmin_EC_q1 = get_max_or_min(residual_q1_EC, argmax_or_argmin="argmin")
    argmax_EC_q5 = get_max_or_min(residual_q5_EC, argmax_or_argmin="argmax")
    argmin_EC_q5 = get_max_or_min(residual_q5_EC, argmax_or_argmin="argmin")

    animal_ID_list_SST = get_animal_ID_list(activity_dict_SST)
    animal_ID_list_NDNF = get_animal_ID_list(activity_dict_NDNF)
    animal_ID_list_EC = get_animal_ID_list(activity_dict_EC)

    mean_quantiles_SST_q1, sem_quantiles_SST_q1 = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                             argmax_SST_q1, n_bins=20)
    mean_quantiles_SST_q5, sem_quantiles_SST_q5 = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                             argmax_SST_q5, n_bins=20)

    mean_quantiles_NDNF_q1, sem_quantiles_NDNF_q1 = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                               argmax_NDNF_q1, n_bins=20)
    mean_quantiles_NDNF_q5, sem_quantiles_NDNF_q5 = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                               argmax_NDNF_q5, n_bins=20)

    mean_quantiles_EC_q1, sem_quantiles_EC_q1 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           argmax_EC_q1, n_bins=20)
    mean_quantiles_EC_q5, sem_quantiles_EC_q5 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           argmax_EC_q5, n_bins=20)

    mean_quantiles_SST_q1_argmin, sem_quantiles_SST_q1_argmin = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                           argmin_SST_q1,
                                                                                           n_bins=20)
    mean_quantiles_SST_q5_argmin, sem_quantiles_SST_q5_argmin = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                           argmin_SST_q5,
                                                                                           n_bins=20)

    mean_quantiles_NDNF_q1_argmin, sem_quantiles_NDNF_q1_argmin = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                             argmin_NDNF_q1,
                                                                                             n_bins=20)
    mean_quantiles_NDNF_q5_argmin, sem_quantiles_NDNF_q5_argmin = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                             argmin_NDNF_q5,
                                                                                             n_bins=20)

    mean_quantiles_EC_q1_argmin, sem_quantiles_EC_q1_argmin = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                         argmin_EC_q1,
                                                                                         n_bins=20)
    mean_quantiles_EC_q5_argmin, sem_quantiles_EC_q5_argmin = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                         argmin_EC_q5,
                                                                                         n_bins=20)

    argmax_mean_list = [mean_quantiles_SST_q1, mean_quantiles_SST_q5, mean_quantiles_NDNF_q1, mean_quantiles_NDNF_q5,
                        mean_quantiles_EC_q1, mean_quantiles_EC_q5]
    argmax_sem_list = [sem_quantiles_SST_q1, sem_quantiles_SST_q5, sem_quantiles_NDNF_q1, sem_quantiles_NDNF_q5,
                       sem_quantiles_EC_q1, sem_quantiles_EC_q5]

    argmin_mean_list = [mean_quantiles_SST_q1_argmin, mean_quantiles_SST_q5_argmin, mean_quantiles_NDNF_q1_argmin,
                        mean_quantiles_NDNF_q5_argmin, mean_quantiles_EC_q1_argmin, mean_quantiles_EC_q5_argmin]
    argmin_sem_list = [sem_quantiles_SST_q1_argmin, sem_quantiles_SST_q5_argmin, sem_quantiles_NDNF_q1_argmin,
                       sem_quantiles_NDNF_q5_argmin, sem_quantiles_EC_q1_argmin, sem_quantiles_EC_q5_argmin]

    return argmax_mean_list, argmax_sem_list, argmin_mean_list, argmin_sem_list


def get_quantiles_for_cdf(activity_dict, values_list, n_bins=None):
    animal_ID_list = []
    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            animal_ID_list.append(animal)

    animal_to_values = defaultdict(list)

    for animal, value in zip(animal_ID_list, values_list):
        animal_to_values[animal].append(value)

    animal_to_values = dict(animal_to_values)

    edges_list = []
    sep_dict = {}

    for key, value in animal_to_values.items():
        animal_edges = np.quantile(value, np.linspace(0, 1, n_bins))
        edges_list.append(animal_edges)

    stacked_edges = np.vstack(edges_list)

    mean_list = []
    sem_list = []

    for i in range(stacked_edges.shape[1]):
        column = stacked_edges[:, i]
        mean_list.append(np.mean(column))
        sem_list.append((np.std(column) / np.sqrt(len(column))))

    return mean_list, sem_list


def get_quantiles_for_cdf_r2_split(animal_ID_list, values_list, n_bins=None):

    animal_to_values = defaultdict(list)

    for animal, value in zip(animal_ID_list, values_list):
        animal_to_values[animal].append(value)

    animal_to_values = dict(animal_to_values)

    edges_list = []
    sep_dict = {}

    for key, value in animal_to_values.items():
        animal_edges = np.quantile(value, np.linspace(0, 1, n_bins))
        edges_list.append(animal_edges)

    stacked_edges = np.vstack(edges_list)

    mean_list = []
    sem_list = []

    for i in range(stacked_edges.shape[1]):
        column = stacked_edges[:, i]
        mean_list.append(np.mean(column))
        sem_list.append((np.std(column) / np.sqrt(len(column))))

    return mean_list, sem_list


def setup_argmin_argmax_cdf_plotting(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False):

    SST_argmax_list, NDNF_argmax_list, EC_argmax_list = split_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=residual, which_to_plot="argmax")

    SST_argmin_list, NDNF_argmin_list, EC_argmin_list = split_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC,predicted_activity_dict_EC,residual=residual, which_to_plot="argmin")

    animal_ID_list_SST = get_animal_ID_list(activity_dict_SST)
    animal_ID_list_NDNF = get_animal_ID_list(activity_dict_NDNF)
    animal_ID_list_EC = get_animal_ID_list(activity_dict_EC)

    mean_quantiles_SST_argmax, sem_quantiles_SST_argmax = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                     SST_argmax_list, n_bins=20)
    mean_quantiles_SST_argmin, sem_quantiles_SST_argmin = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                     SST_argmin_list, n_bins=20)
    mean_quantiles_NDNF_argmax, sem_quantiles_NDNF_argmax = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                       NDNF_argmax_list, n_bins=20)
    mean_quantiles_NDNF_argmin, sem_quantiles_NDNF_argmin = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                       NDNF_argmin_list, n_bins=20)
    mean_quantiles_EC_argmax, sem_quantiles_EC_argmax = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_argmax_list,
                                                                                   n_bins=20)
    mean_quantiles_EC_argmin, sem_quantiles_EC_argmin = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_argmin_list,
                                                                                   n_bins=20)

    mean_quantiles_list_argmax = [mean_quantiles_SST_argmax, mean_quantiles_NDNF_argmax, mean_quantiles_EC_argmax]
    sem_quantiles_list_argmax = [sem_quantiles_SST_argmax, sem_quantiles_NDNF_argmax, sem_quantiles_EC_argmax]

    mean_quantiles_list_argmin = [mean_quantiles_SST_argmin, mean_quantiles_NDNF_argmin, mean_quantiles_EC_argmin]
    sem_quantiles_list_argmin = [sem_quantiles_SST_argmin, sem_quantiles_NDNF_argmin, sem_quantiles_EC_argmin]

    return mean_quantiles_list_argmax, sem_quantiles_list_argmax, mean_quantiles_list_argmin, sem_quantiles_list_argmin


def compute_r_and_model(x, y):

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r_value, _ = pearsonr(x, y)
    return r_value, y_pred


def compute_residual_activity(activity_dict, predicted_activity_dict):
    predicted_activity_list = []
    neuron_activity_list = []
    residuals_list = []

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            predicted_activity = predicted_activity_dict[animal][neuron]
            neuron_activity = activity_dict[animal][neuron]
            residual = neuron_activity - predicted_activity

            neuron_activity_list.append(neuron_activity)
            predicted_activity_list.append(predicted_activity)
            residuals_list.append(residual)

    return predicted_activity_list, neuron_activity_list, residuals_list


def get_residual_activity_dict(activity_dict, predicted_activity_dict):
    residual_activity_dict = {}
    for animal in activity_dict:
        residual_activity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            residual_activity_dict[animal][neuron] = activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
    return residual_activity_dict


def flatten_data(neuron_dict):
    flattened_data = {}
    for var in neuron_dict:
        flattened_data[var] = neuron_dict[var].flatten()
    return flattened_data


def get_quintile_indices(num_trials, quintile=None):
    quintile_indices = [(i * num_trials) // 5 for i in range(6)]
    start_idx = quintile_indices[quintile - 1]
    end_idx = quintile_indices[quintile]
    return start_idx,end_idx


def filter_neurons_by_metric(reorganized_data, GLM_params, variable_list, metric, threshold, scale_type='value', keep='top'):
    GLM_r2 = get_GLM_R2(GLM_params)
    weights = get_GLM_weights(GLM_params, variable_list)
    consequtive_trials_correlations, _,_ = compute_temporal_correlation(reorganized_data)

    avg_residuals, _ = compute_velocity_subtracted_residuals(reorganized_data, variable_list, quintile=None)
    spatial_selectivity_index = compute_spatial_selectivity_index(avg_residuals)

    metrics_dict = {**weights, 'R2': GLM_r2, 
                    'trial correlations': consequtive_trials_correlations,
                    'spatial selectivity': spatial_selectivity_index}
    
    if metric not in metrics_dict:
        raise ValueError(f"Invalid metric: {metric}. Choose from {list(metrics_dict.keys())}")
    metric_values = metrics_dict[metric]
    
    filtered_GLM = copy.deepcopy(GLM_params)
    filtered_data = copy.deepcopy(reorganized_data)
    neuron_idx = 0
    for animal in filtered_GLM:
        neurons_to_remove = []
        for neuron in filtered_GLM[animal]:
            metric_value = metric_values[neuron_idx]

            # Define thresholds based on scale type
            if scale_type == 'std':
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                upper_bound = mean_val + threshold * std_val
                lower_bound = mean_val - threshold * std_val

            elif scale_type == 'percentile':
                upper_bound = np.percentile(metric_values, 100 - threshold)
                lower_bound = np.percentile(metric_values, threshold)

            elif scale_type == 'value':
                upper_bound = threshold
                lower_bound = threshold

            # Apply filtering based on filter type
            if (keep == 'top' and metric_value < upper_bound) or \
               (keep == 'bottom' and metric_value > lower_bound):
                neurons_to_remove.append(neuron)

            neuron_idx += 1

        for neuron in neurons_to_remove:
            filtered_GLM[animal].pop(neuron)
            filtered_data[animal].pop(neuron)
        
    filtered_GLM = {animal: filtered_GLM[animal] for animal in filtered_GLM if filtered_GLM[animal] != {}}
    filtered_data = {animal: filtered_data[animal] for animal in filtered_data if filtered_data[animal] != {}}

    return filtered_data, filtered_GLM


def compute_temporal_correlation(reorganized_data):
    r_all_neurons = []
    r_per_animal = []
    r_dict = {}

    def compute_r_consecutive_pairs(neuron_activity):
        r_values = []
        for col in range(neuron_activity.shape[1] - 1):
            X = neuron_activity[:, col]
            Y = neuron_activity[:, col + 1]
            if np.std(X) > 0 and np.std(Y) > 0:
                r = np.corrcoef(X, Y)[0, 1]
            else:
                r = np.nan  # Handle cases with no variability
            r_values.append(r)
        return r_values

    for animal_key in reorganized_data:
        r_animal_neurons = []
        r_dict[animal_key] = {}

        for neuron_key in reorganized_data[animal_key]:
            neuron_data = reorganized_data[animal_key][neuron_key]
            neuron_activity = neuron_data[:, 0, :]

            # Remove trials with NaNs
            valid_trials_mask = ~np.isnan(neuron_activity).any(axis=0)
            neuron_activity = neuron_activity[:, valid_trials_mask]

            # Remove bins with NaNs
            valid_bins_mask = ~np.isnan(neuron_activity).any(axis=1)
            neuron_activity = neuron_activity[valid_bins_mask, :]

            if neuron_activity.shape[1] > 1:
                r_values = compute_r_consecutive_pairs(neuron_activity)
                mean_r_value = np.nanmean(r_values)
                r_animal_neurons.append(mean_r_value)
                r_all_neurons.append(mean_r_value)
                r_dict[animal_key][neuron_key] = mean_r_value

        r_per_animal.append(np.nanmean(r_animal_neurons))

    return r_all_neurons, r_per_animal, r_dict


def create_variable_lists(predicted_activity_dict, neuron_activity_list, variable_to_correlate_list):
    predicted_variable_lists = {}

    for key in variable_to_correlate_list:
        predicted_list = []

        for i, animal in enumerate(predicted_activity_dict):
            for neuron in predicted_activity_dict[animal]:
                predicted_activity = predicted_activity_dict[animal][neuron][key]
                predicted_list.append(predicted_activity)

        variable_lists[key] = predicted_list

    return predicted_variable_lists


def compute_residual_activity_min_max(GLM_params, reorganized_data, quintile=None):
    residual_activity = {}
    avg_residuals = []
    for animal in GLM_params:
        residual_activity[animal] = {}
        for neuron in GLM_params[animal]:
            # Pre-process the data
            neuron_data = reorganized_data[animal][neuron]
            if quintile is not None:
                num_trials = neuron_data.shape[2]
                start_idx, end_idx = get_quintile_indices(num_trials, quintile)
                neuron_data = neuron_data[:, :, start_idx:end_idx]

            flattened_data = flatten_data(neuron_data)
            neuron_activity = neuron_data[:, 0, :]

            # Predict activity and compute residuals
            glm_model = GLM_params[animal][neuron]['model']
            flattened_input_variables = flattened_data[:, 1:]
            predicted_activity = glm_model.predict(flattened_input_variables)
            predicted_activity = predicted_activity.reshape(neuron_activity.shape)
            normalized_trials = []
            for trials in predicted_activity:
                trial = []
                for bins in trials:
                    norm_bin = (bins - np.min(trials)) / (np.max(trials) - np.min(trials))
                    trial.append(norm_bin)
                trial_array = np.array(trial)
                normalized_trials.append(trial_array)
            predicted_activity = np.vstack(normalized_trials)

            residual_activity[animal][neuron] = neuron_activity - predicted_activity

            avg_residuals.append(np.mean(residual_activity[animal][neuron], axis=1))

    avg_residuals = np.array(avg_residuals)

    normalized_average_residuals = []
    for cell in avg_residuals:
        cells = []
        for bins in cell:
            norm_bin = (bins - np.min(cell)) / (np.max(cell) - np.min(cell))
            cells.append(norm_bin)
        cell_array = np.array(cells)
        normalized_average_residuals.append(cell_array)
    avg_residuals = np.vstack(normalized_average_residuals)

    return residual_activity, avg_residuals


def remove_variables_from_glm(GLM_params, vars_to_remove, variable_list):
    GLM_params_ = copy.deepcopy(GLM_params)

    idx_to_remove = [variable_list[1:].index(var) for var in vars_to_remove if var in variable_list[1:]]

    modified_GLM_params = {}
    for animal_key, neurons in GLM_params_.items():
        modified_GLM_params[animal_key] = {}
        for neuron_idx, params in neurons.items():
            params['model'].coef_[idx_to_remove] = 0
            if "intercept" in vars_to_remove:
                params['model'].intercept_ = 0
                params['intercept'] = 0
            modified_GLM_params[animal_key][neuron_idx] = params

    return modified_GLM_params


def compute_variable_subtracted_residuals(reorganized_data, variable_list, variables_to_remove, quintile):
    GLM_params = fit_GLM(reorganized_data, quintile=quintile, regression='ridge', renormalize=False)
    vars_to_remove = variable_list.copy()[1:] + ['intercept']
    removing_list = []

    for variable in variables_to_remove:
        if variable in variable_list:
            index = variable_list.index(variable)
            removing_list.append(index)

    for index in removing_list:
        vars_to_remove.remove(variable_list[index])

    filtered_GLM_params = remove_variables_from_glm(GLM_params, vars_to_remove, variable_list)
    residual_activity, avg_residuals, predicted_activity = compute_residual_activity(filtered_GLM_params, reorganized_data,
                                                                 quintile=quintile)
    return avg_residuals, GLM_params, residual_activity, predicted_activity


def normalize(x, norm, per_cell=True):
    if per_cell:
        if norm == 'min_max':
            return (x - np.min(x, axis=1, keepdims=True)) / (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))
        elif norm == 'Z_score':
            return (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
    else:
        if norm == 'min_max':
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        elif norm == 'Z_score':
            return (x - np.mean(x)) / np.std(x)
    return x


def get_min_maxed_residuals_argmin_argmax_selectivity(avg_residuals):
    argmax_list = []
    argmin_list = []
    selectivity = []

    normalized_average_residuals = []
    for cell in avg_residuals:
        cell_selectivity = Vinje2000(cell, norm='min_max')
        selectivity.append(cell_selectivity)
        norm_bin = (cell - np.min(cell)) / (np.max(cell) - np.min(cell))
        normalized_average_residuals.append(norm_bin)
        argmax_list.append(np.argmax(norm_bin))
        argmin_list.append(np.argmin(norm_bin))

    avg_residuals_min_max = np.vstack(normalized_average_residuals)

    return avg_residuals_min_max, argmax_list, argmin_list, selectivity


def split_into_quintiles(array):
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    total_columns = array.shape[1]
    quintile_size = total_columns // 5

    usable_columns = quintile_size * 5
    truncated_array = array[:, :usable_columns]

    split_array = np.split(truncated_array, 5, axis=1)

    return split_array


def select_neuron(GLM_params, variable_list, sort_by='R2', animal=None, cell=None):

    variable_list = variable_list[1:]

    flattened_data = []

    if animal is not None:
        if animal not in GLM_params:
            raise KeyError(f"{animal} not found in GLM_params")
        if cell is not None and cell not in GLM_params[animal]:
            raise KeyError(f"{cell} not found in GLM_params for {animal}")
    
    for animal_id in GLM_params:
        for neuron_id in GLM_params[animal_id]:
            metrics = GLM_params[animal_id][neuron_id]
            row = {'animal': animal_id, 'neuron': neuron_id}
            row.update(metrics)
            flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    if sort_by in variable_list: # If sort_by is a variable, sort by the magnitude of the specified weight
        var_idx = variable_list.index(sort_by)
        df[f'{sort_by}_weight_magnitude'] = df['weights'].apply(lambda w: np.abs(w[var_idx])) # Add a column for the magnitude of the specified weight
        df_sorted = df.sort_values(by=[f'{sort_by}_weight_magnitude', 'R2'], ascending=[False, False])
        top_neuron = df_sorted.iloc[0]
        top_neuron_weight = top_neuron['weights'][var_idx]
        print(f"Top neuron for {sort_by}, with weight: {top_neuron_weight}")
    else:
        if sort_by != 'R2':
            raise ValueError(f"Invalid sort_by value: {sort_by}. Must be one of {variable_list} or 'R2'")
        df_sorted = df.sort_values(by=sort_by, ascending=False)

    if animal is not None:
        df_sorted = df_sorted[df_sorted['animal'] == animal]

    if cell is not None:
        df_sorted = df_sorted[df_sorted['neuron'] == cell]

    top_neuron = df_sorted.iloc[0]
    animal, cell = top_neuron['animal'], top_neuron['neuron']

    return animal, cell


def compute_spatial_selectivity_index(avg_residuals):
    spatial_selectivity_index = []
    for cell_residual in avg_residuals:
        # Renormalize residuals between 0 and 1
        cell_residual = (cell_residual - np.min(cell_residual)) / (np.max(cell_residual) - np.min(cell_residual))

        # Compute bimodality coefficient
        skewness = stats.skew(cell_residual)
        kurt = stats.kurtosis(cell_residual, fisher=False)  # Use fisher=False to get Pearson kurtosis
        n = len(cell_residual)
        bimodality_coefficient = (skewness**2 + 1) / (kurt + (3*(n-1)**2)/((n-2)*(n-3)))

        # Compute distance between k-means centers
        kmeans = KMeans(n_clusters=2).fit(cell_residual.reshape(-1, 1))
        centers = kmeans.cluster_centers_
        distance = np.abs(centers[0] - centers[1])

        # Combine the metrics
        spatial_selectivity_index.append(bimodality_coefficient * distance[0])
    return spatial_selectivity_index


def compute_velocity_subtracted_residuals(reorganized_data, variable_list, quintile):
    GLM_params = fit_GLM(reorganized_data, quintile=quintile, regression='ridge', renormalize=False)
    vars_to_remove = variable_list.copy()[1:] + ['intercept']
    vars_to_remove.remove('Velocity')
    filtered_GLM_params = remove_variables_from_glm(GLM_params, vars_to_remove, variable_list)
    residual_activity, avg_residuals, predicted_activity_list = compute_residual_activity(filtered_GLM_params, reorganized_data, quintile=quintile)
    return avg_residuals, GLM_params


def get_GLM_R2(GLM_params):
    all_R2_values = np.array([GLM_params[animal][neuron]['R2_trialavg'] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_R2_values


def get_GLM_weights(GLM_params, variable_list):
    all_weights = {}
    for i,var_name in enumerate(variable_list[1:]):
        all_weights[var_name] = np.array([GLM_params[animal][neuron]['weights'][i] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_weights


def calculate_delta_weights(GLM_params_first, GLM_params_last):
    assert GLM_params_first.keys() == GLM_params_last.keys(), "Animal keys do not match between the two GLM parameters dictionaries."

    delta_params = {}
    for animal in GLM_params_first:
        delta_params[animal] = {}
        for cell in GLM_params_first[animal]:
            weights_first = GLM_params_first[animal][cell]['weights']
            weights_last = GLM_params_last[animal][cell]['weights']
            intercept_first = GLM_params_first[animal][cell]['intercept']
            intercept_last = GLM_params_last[animal][cell]['intercept']

            delta_weights = weights_last - weights_first
            delta_intercept = intercept_last - intercept_first
            delta_params[animal][cell] = {'weights':delta_weights, 'intercept':delta_intercept}
    return delta_params


if __name__ == "__main__":

    datasets = ["SSTindivsomata_GLM.mat"]#, "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    filepath_list = ["SSTindivsomata_GLM.mat", "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    filepaths = {
        'NDNF': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\NDNFindivsomata_GLM.mat",
        'EC': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_GLM.mat",
        'SST': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\SSTindivsomata_GLM.mat"
    }

    get_delta_weights_and_plot(filepath_list)

    # variable_list: ['Activity', 'Licks', 'R_loc', 'Speed', '#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10']
    vars_to_remove = [2]  # speed

    filtered_GLM_params = remove_variables_from_glm(GLM_params, vars_to_remove)

    for dataset_path in datasets:
        reorganized_data, variable_list = load_data(dataset_path)
        GLM_params = fit_GLM(reorganized_data, quintile=1, regression=lasso)

        fig = plt.figure(figsize=(10,5))
        axes = gs.GridSpec(nrows=1, ncols=3)
        fig.suptitle(dataset_path)

        ax = fig.add_subplot(axes[0,0])
        plot_example_neuron(reorganized_data, variable_list, model_name=dataset_path, fig=fig, ax=ax)

        ax = fig.add_subplot(axes[0,2])
        plot_GLM_summary_data(GLM_params, variable_list, dataset_path, ax=ax)

        fig.savefig(f"figures/{dataset_path[:-4]}.png", dpi=300)