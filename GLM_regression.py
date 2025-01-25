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


def plot_cell_trial_average_variable_subtraction(activity_dict_SST, activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_SST, predicted_activity_dict_NDNF, predicted_activity_dict_EC):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(activity_dict_EC, predicted_activity_dict_EC)

    trial_av_neuron_activity_list_SST = trial_average(neuron_activity_list_SST)
    trial_av_mean_cell_residual_list_SST = trial_average(cell_residual_list_SST)

    trial_av_neuron_activity_list_NDNF = trial_average(neuron_activity_list_NDNF)
    trial_av_mean_cell_residual_list_NDNF = trial_average(cell_residual_list_NDNF)

    trial_av_neuron_activity_list_EC = trial_average(neuron_activity_list_EC)
    trial_av_mean_cell_residual_list_EC = trial_average(cell_residual_list_EC)

    neuron_activity_list_SST_array = np.stack(trial_av_neuron_activity_list_SST)
    neuron_activity_list_NDNF_array = np.stack(trial_av_neuron_activity_list_NDNF)
    neuron_activity_list_EC_array = np.stack(trial_av_neuron_activity_list_EC)

    neuron_residual_list_SST_array = np.stack(trial_av_mean_cell_residual_list_SST)
    neuron_residual_list_NDNF_array = np.stack(trial_av_mean_cell_residual_list_NDNF)
    neuron_residual_list_EC_array = np.stack(trial_av_mean_cell_residual_list_EC)

    cell_av_neuron_activity_list_SST = np.mean(neuron_activity_list_SST_array, axis=0)
    cell_av_neuron_activity_list_NDNF = np.mean(neuron_activity_list_NDNF_array, axis=0)
    cell_av_neuron_activity_list_EC = np.mean(neuron_activity_list_EC_array, axis=0)

    cell_av_residual_list_SST = np.mean(neuron_residual_list_SST_array, axis=0)
    cell_av_residual_list_NDNF = np.mean(neuron_residual_list_NDNF_array, axis=0)
    cell_av_residual_list_EC = np.mean(neuron_residual_list_EC_array, axis=0)

    sem_neuron_activity_list_SST = np.std(neuron_activity_list_SST_array, axis=0) / np.sqrt(
        neuron_activity_list_SST_array.shape[0])
    sem_residual_list_SST = np.std(neuron_residual_list_SST_array, axis=0) / np.sqrt(
        neuron_residual_list_SST_array.shape[0])

    sem_neuron_activity_list_NDNF = np.std(neuron_activity_list_NDNF_array, axis=0) / np.sqrt(
        neuron_activity_list_NDNF_array.shape[0])
    sem_residual_list_NDNF = np.std(neuron_residual_list_NDNF_array, axis=0) / np.sqrt(
        neuron_residual_list_NDNF_array.shape[0])

    sem_neuron_activity_list_EC = np.std(neuron_activity_list_EC_array, axis=0) / np.sqrt(
        neuron_activity_list_EC_array.shape[0])
    sem_residual_list_EC = np.std(neuron_residual_list_EC_array, axis=0) / np.sqrt(
        neuron_residual_list_EC_array.shape[0])

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_SST, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_SST)),
                     cell_av_neuron_activity_list_SST - sem_neuron_activity_list_SST,
                     cell_av_neuron_activity_list_SST + sem_neuron_activity_list_SST,
                     color='k', alpha=0.2)
    plt.plot(cell_av_residual_list_SST, color='b', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_SST)),
                     cell_av_residual_list_SST - sem_residual_list_SST,
                     cell_av_residual_list_SST + sem_residual_list_SST,
                     color='b', alpha=0.2)
    plt.title("SST Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_NDNF, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_NDNF)),
                     cell_av_neuron_activity_list_NDNF - sem_neuron_activity_list_NDNF,
                     cell_av_neuron_activity_list_NDNF + sem_neuron_activity_list_NDNF,
                     color='k', alpha=0.2)

    plt.plot(cell_av_residual_list_NDNF, color='orange', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_NDNF)),
                     cell_av_residual_list_NDNF - sem_residual_list_NDNF,
                     cell_av_residual_list_NDNF + sem_residual_list_NDNF,
                     color='orange', alpha=0.2)

    plt.title("NDNF Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_EC, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_EC)),
                     cell_av_neuron_activity_list_EC - sem_neuron_activity_list_EC,
                     cell_av_neuron_activity_list_EC + sem_neuron_activity_list_EC,
                     color='k', alpha=0.2)

    plt.plot(cell_av_residual_list_EC, color='g', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_EC)),
                     cell_av_residual_list_EC - sem_residual_list_EC,
                     cell_av_residual_list_EC + sem_residual_list_EC,
                     color='g', alpha=0.2)

    plt.title("EC Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()


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


def plot_single_animal_average_trace(
    animal_mean_list_SST, animal_mean_residuals_SST,
    activity_dict_SST, predicted_activity_dict_SST,
    filtered_factors_dict_SST, cell_type="SST"
):
    """
    Plot velocity, raw activity, and residual activity across animals, along with their means.

    Args:
        animal_mean_list_SST: List of mean raw activities for each animal.
        animal_mean_residuals_SST: List of mean residual activities for each animal.
        activity_dict_SST: Dictionary of actual activity.
        predicted_activity_dict_SST: Dictionary of predicted activity.
        filtered_factors_dict_SST: Dictionary of factors, e.g., velocity.
        cell_type: The type of cell to plot (e.g., "SST", "NDNF", "EC").
    """
    # Calculate R² correlations
    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"
    )
    mean_r2_per_animal_raw, sem_r2_per_animal_raw = get_per_animal_mean_r2(r2_variable_activity_dict_SST)
    mean_r2_per_animal_residual, sem_r2_per_animal_residual = get_per_animal_mean_r2(r2_variable_residual_dict_SST)

    # Define the color to plot based on cell type
    color_map = {"SST": "blue", "NDNF": "orange", "EC": "green"}
    color_to_plot = color_map.get(cell_type, "gray")

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Velocity Plot
    trial_av_animal_list = []
    for animal in filtered_factors_dict_SST:
        value = filtered_factors_dict_SST[animal]["Velocity"]
        trial_av_animal_list.append(np.mean(value, axis=1))

    velocity_array = np.stack(trial_av_animal_list) * 100
    mean_velocity = np.mean(velocity_array, axis=0)

    for trial in velocity_array:
        axs[0].plot(trial, color="gray", alpha=0.5, linewidth=0.8)
    axs[0].plot(mean_velocity, linewidth=3, color=color_to_plot, alpha=0.9, label="Mean Velocity")
    axs[0].set_title("Velocity Per Animal Trial Averaged", fontsize=14)
    axs[0].set_xlabel("Position Bins", fontsize=12)
    axs[0].set_ylabel("cm Per Second", fontsize=12)
    axs[0].legend(fontsize=10)

    # Raw Activity Plot
    animal_mean_mean_list_SST = np.stack(animal_mean_list_SST)
    animal_mean_mean_list_SST = np.mean(animal_mean_mean_list_SST, axis=0)

    for animal_to_plot in range(len(animal_mean_list_SST)):
        axs[1].plot(animal_mean_list_SST[animal_to_plot], color="gray", alpha=0.5, linewidth=0.8)
    axs[1].plot(animal_mean_mean_list_SST, color=color_to_plot, alpha=0.9, linewidth=3, label="Mean Raw Activity")
    axs[1].set_title(f"{cell_type} Raw Activity Across Animals", fontsize=14)
    axs[1].set_xlabel("Position Bin", fontsize=12)
    axs[1].set_ylabel("DF/F", fontsize=12)
    axs[1].set_ylim(-1, 1)
    axs[1].legend(fontsize=10)

    # Residual Activity Plot
    animal_mean_mean_residual_list_SST = np.stack(animal_mean_residuals_SST)
    animal_mean_mean_residual_list_SST = np.mean(animal_mean_mean_residual_list_SST, axis=0)

    for animal_to_plot in range(len(animal_mean_residuals_SST)):
        axs[2].plot(animal_mean_residuals_SST[animal_to_plot], color="gray", alpha=0.5)
    axs[2].plot(animal_mean_mean_residual_list_SST, color=color_to_plot, alpha=0.9, linewidth=3, label="Mean Residual Activity")
    axs[2].set_title(f"{cell_type} Residual Activity Across Animals", fontsize=14)
    axs[2].set_xlabel("Position Bin", fontsize=12)
    axs[2].set_ylabel("DF/F", fontsize=12)
    axs[2].set_ylim(-1, 1)
    axs[2].legend(fontsize=10)

    # Finalize layout
    fig.tight_layout()
    plt.show()

    # Plot population correlation
    plot_pop_correlation(
        mean_r2_per_animal_raw, mean_r2_per_animal_residual,
        variable_to_correlate="Velocity", cell_type=cell_type,
        color=color_to_plot, dictionary=False
    )

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




# def get_neuron_activity_prediction_residual(activity_dict, predicted_activity_dict):
#     neuron_activity_list = []
#     predictions_list = []
#     cell_residual_list = []
#
#     for animal in activity_dict:
#         for neuron in activity_dict[animal]:
#             neuron_activity = activity_dict[animal][neuron]
#             prediction = predicted_activity_dict[animal][neuron]
#
#             residual = np.array(neuron_activity) - np.array(prediction)
#
#             neuron_activity_list.append(neuron_activity)
#             predictions_list.append(prediction)
#             cell_residual_list.append(residual)
#
#     return neuron_activity_list, predictions_list, cell_residual_list


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



def plot_correlations_single_variable_GLM(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict,
                                          activity_dict, cell_number,
                                          variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"],
                                          variable_into_GLM="Velocity"):
    input_variables = {var: [] for var in factors_dict[next(iter(factors_dict))].keys()}
    filtered_input_variables = {var: [] for var in filtered_factors_dict[next(iter(filtered_factors_dict))].keys()}

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            for var in factors_dict[animal]:
                input_variables[var].append(factors_dict[animal][var])
            for var in filtered_factors_dict[animal]:
                filtered_input_variables[var].append(filtered_factors_dict[animal][var])

    neuron_activity_list, predictions_list, cell_residual_list = get_neuron_activity_prediction_residual(
        activity_dict, predicted_activity_dict)

    cell_activity = neuron_activity_list[cell_number]
    cell_prediciton = predictions_list[cell_number]
    cell_residual = cell_residual_list[cell_number]

    flat_neuron_activity = cell_activity.flatten()
    flat_prediction_total = cell_prediciton.flatten()
    flat_residual_total = cell_residual.flatten()

    num_keys = len(variable_to_correlate_list)
    fig, axs = plt.subplots(1, num_keys, figsize=(6 * (num_keys), 5))

    for idx, key in enumerate(variable_to_correlate_list):
        flat_neuron_activity = neuron_activity_list[cell_number].flatten()
        flat_variable = input_variables[key][cell_number].flatten()

        model_activity = LinearRegression()
        model_activity.fit(flat_variable.reshape(-1, 1), flat_neuron_activity)
        y_pred_activity = model_activity.predict(flat_variable.reshape(-1, 1))
        r2_activity, _ = pearsonr(flat_neuron_activity, y_pred_activity)

        axs[idx].scatter(flat_variable, flat_neuron_activity, label="Data", alpha=0.6)
        axs[idx].plot(flat_variable, y_pred_activity, color='r', label="Best Fit", linewidth=2)
        axs[idx].set_title(f"{key} vs Activity\nR value: {r2_activity:.3f}")
        axs[idx].set_xlabel(key)
        axs[idx].set_ylabel("Activity")
        axs[idx].legend()

    plt.tight_layout()
    plt.show()

    variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]
    flat_variable_of_interest = variable_of_interest.flatten()

    r2_total, y_pred_total = compute_r_and_model(flat_prediction_total, flat_neuron_activity)
    r2_residual, y_pred_total_residual = compute_r_and_model(flat_residual_total, flat_neuron_activity)
    r2_variable_residual, y_pred_variable_residual = compute_r_and_model(flat_residual_total, flat_variable_of_interest)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(flat_prediction_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[0].plot(flat_prediction_total, y_pred_total, color='r', label="Best Fit", linewidth=2)
    axs[0].set_title(f"Prediction based on just {variable_into_GLM} vs Activity\nR value: {r2_total:.3f}")
    axs[0].set_xlabel(f"{variable_into_GLM} Prediction")
    axs[0].set_ylabel("Activity")
    axs[0].legend()

    axs[1].scatter(flat_residual_total, flat_variable_of_interest, label="Data", alpha=0.6)
    axs[1].plot(flat_residual_total, y_pred_variable_residual, color='r', label="Best Fit", linewidth=2)
    axs[1].set_title(f"Residuals vs Variable: {variable_into_GLM}\nR value: {r2_variable_residual:.3f}")
    axs[1].set_xlabel("Residuals")
    axs[1].set_ylabel(f"Variable: {variable_into_GLM}")
    axs[1].legend()

    axs[2].scatter(flat_residual_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[2].plot(flat_residual_total, y_pred_total_residual, color='r', label="Best Fit", linewidth=2)
    axs[2].set_title(f"Residuals vs Activity\nR value: {r2_residual:.3f}")
    axs[2].set_xlabel("Residuals")
    axs[2].set_ylabel("Activity")
    axs[2].legend()

    plt.tight_layout()
    plt.show()

    flat_residual_total = flat_residual_total.reshape(neuron_activity_list[cell_number].shape)
    flat_neuron_activity = flat_neuron_activity.reshape(neuron_activity_list[cell_number].shape)

    plt.figure()
    plt.plot(np.mean(predictions_list[cell_number], axis=1), color='b',
             label=f"Prediction Based on {variable_into_GLM}")
    plt.plot(np.mean(flat_neuron_activity, axis=1), color='k', label=f"Raw Activity for Cell#{cell_number}")
    plt.plot(np.mean(flat_residual_total, axis=1), color='r', label=f"{variable_into_GLM}-Subtracted Residual")
    plt.title(f"Cell#{cell_number} Trial Averaged")
    plt.xlabel('Position Bin')
    plt.ylabel('DF/F')
    plt.legend()
    plt.show()


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


def get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                 predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC,
                                 residual=False):
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


def plot_selectivity_frequency_split_by_r2(SST_list_above, SST_list_below, NDNF_list_above, NDNF_list_below,
                                           EC_list_above, EC_list_below, name=None):
    bin_edges = np.arange(0, 1.1, 0.1)
    bin_centers = bin_edges[:-1] + 0.05
    bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    SST_hist_above, _ = np.histogram(SST_list_above, bins=bin_edges)
    NDNF_hist_above, _ = np.histogram(NDNF_list_above, bins=bin_edges)
    EC_hist_above, _ = np.histogram(EC_list_above, bins=bin_edges)

    SST_fraction_above = SST_hist_above / np.sum(SST_hist_above)
    NDNF_fraction_above = NDNF_hist_above / np.sum(NDNF_hist_above)
    EC_fraction_above = EC_hist_above / np.sum(EC_hist_above)

    SST_hist_below, _ = np.histogram(SST_list_below, bins=bin_edges)
    NDNF_hist_below, _ = np.histogram(NDNF_list_below, bins=bin_edges)
    EC_hist_below, _ = np.histogram(EC_list_below, bins=bin_edges)

    SST_fraction_below = SST_hist_below / np.sum(SST_hist_below)
    NDNF_fraction_below = NDNF_hist_below / np.sum(NDNF_hist_below)
    EC_fraction_below = EC_hist_below / np.sum(EC_hist_below)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, SST_fraction_above, marker='o', label=f'SST High R Vel', linestyle='-', color='b')
    plt.plot(bin_centers, NDNF_fraction_above, marker='o', label=f'NDNF High R Vel', linestyle='-', color='orange')
    plt.plot(bin_centers, EC_fraction_above, marker='o', label=f'EC High R Vel', linestyle='-', color='green')
    plt.plot(bin_centers, SST_fraction_below, marker='o', label=f'SST Low R Vel', linestyle='-', color='c')
    plt.plot(bin_centers, NDNF_fraction_below, marker='o', label=f'NDNF Low R Vel', linestyle='-', color='red')
    plt.plot(bin_centers, EC_fraction_below, marker='o', label=f'EC Low R Vel', linestyle='-', color='gray')

    plt.xlabel('Selectivity')
    plt.ylabel('Fraction of Cells')
    plt.title(f'{name} Split by Velocity Correlation')
    plt.xticks(bin_centers[::2], bin_labels[::2])
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_selectivity_frequency(SST_list, NDNF_list, EC_list, name=None):
    bin_edges = np.arange(0, 1.1, 0.1)
    bin_centers = bin_edges[:-1] + 0.05
    bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    SST_hist, _ = np.histogram(SST_list, bins=bin_edges)
    NDNF_hist, _ = np.histogram(NDNF_list, bins=bin_edges)
    EC_hist, _ = np.histogram(EC_list, bins=bin_edges)

    SST_fraction = SST_hist / np.sum(SST_hist)
    NDNF_fraction = NDNF_hist / np.sum(NDNF_hist)
    EC_fraction = EC_hist / np.sum(EC_hist)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, SST_fraction, marker='o', label=f'SST {name}', linestyle='-')
    plt.plot(bin_centers, NDNF_fraction, marker='o', label=f'NDNF {name}', linestyle='-')
    plt.plot(bin_centers, EC_fraction, marker='o', label=f'EC {name}', linestyle='-')

    plt.xlabel('Selectivity')
    plt.ylabel('Fraction of Cells')
    plt.title(f'{name}')
    plt.xticks(bin_centers[::2], bin_labels[::2])
    plt.legend()

    plt.tight_layout()
    plt.show()

def setup_CDF_plotting_and_plot_selectivity(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                            predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC,
                                            residual=False):
    SST_factor_list, SST_negative_selectivity, NDNF_factor_list, NDNF_negative_selectivity, EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(
        activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC, residual=residual)

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

    if residual:
        plot_cdf(mean_quantiles_list, sem_quantiles_list, "Selectivity for Residuals", "Vinje Selectivity Index",
                 n_bins=20)
        plot_cdf(mean_quantiles_list_negative, sem_quantiles_list_negative, "Negative Selectivity for Residuals",
                 "Negative Vinje Selectivity Index", n_bins=20)


    else:
        plot_cdf(mean_quantiles_list, sem_quantiles_list, "Selectivity for Raw Data", "Vinje Selectivity Index",
                 n_bins=20)
        plot_cdf(mean_quantiles_list_negative, sem_quantiles_list_negative, "Negative Selectivity for Raw Data",
                 "Negative Vinje Selectivity Index", n_bins=20)


def get_argmin_argmax_for_plotting(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                   predicted_activity_dict_NDNF, activity_dict_EC,
                                   predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):
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


def setup_CDF_plotting_and_plot_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                              predicted_activity_dict_NDNF, activity_dict_EC,
                                              predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):
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



def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                variable_to_correlate="Velocity"):
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


def get_pop_correlation_to_variable(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                    variable_to_correlate="Velocity"):
    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)

    filtered_input_variables = {var: [] for var in
                                filtered_factors_dict_SST[next(iter(filtered_factors_dict_SST))].keys()}
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


def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                variable_to_correlate="Velocity"):
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


# def filter_activity_by_r2(activity_dict, predicted_activity_dict, r2_dict, residual=True):
#
#     filtered_activity = {}
#     for animal in r2_dict:
#         if animal in activity_dict:
#             filtered_activity[animal] = {neuron: activity_dict[animal][neuron] for neuron in r2_dict[animal]}
#     return filtered_activity
def filter_activity_by_r2(activity_dict, predicted_activity_dict, r2_dict, residual=True):

    filtered_activity = {}

    for animal in r2_dict:
        if animal in activity_dict and animal in predicted_activity_dict:
            filtered_activity[animal] = {}
            for neuron in r2_dict[animal]:
                if neuron in activity_dict[animal] and neuron in predicted_activity_dict[animal]:
                    if residual:
                        # Subtract predicted activity from actual activity to compute residuals
                        filtered_activity[animal][neuron] = (
                            activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
                        )
                    else:
                        # Use raw activity
                        filtered_activity[animal][neuron] = activity_dict[animal][neuron]

    return filtered_activity


def plot_mean_and_sem_by_r2(activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero, cell_type):
    activity_above_zero = filter_activity_by_r2(activity_dict_SST, r2_SST_above_zero)
    activity_below_zero = filter_activity_by_r2(activity_dict_SST, r2_SST_below_zero)

    first_mean_above, first_sem_above, last_mean_above, last_sem_above = compute_mean_and_sem_for_quintiles(
        activity_above_zero)
    first_mean_below, first_sem_below, last_mean_below, last_sem_below = compute_mean_and_sem_for_quintiles(
        activity_below_zero)

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(first_mean_above)), first_mean_above, yerr=first_sem_above, fmt='o-', color='blue',
                 label=f'{cell_type} Above Zero - First Quintile')
    plt.errorbar(range(len(last_mean_above)), last_mean_above, yerr=last_sem_above, fmt='o--', color='cyan',
                 label=f'{cell_type} Above Zero - Last Quintile')

    plt.errorbar(range(len(first_mean_below)), first_mean_below, yerr=first_sem_below, fmt='o-', color='orange',
                 label=f'{cell_type} Below Zero - First Quintile')
    plt.errorbar(range(len(last_mean_below)), last_mean_below, yerr=last_sem_below, fmt='o--', color='red',
                 label=f'{cell_type} Below Zero - Last Quintile')

    plt.xlabel("Position Bin")
    plt.ylabel("Z-Score Mean Activity")
    plt.title(f"Activity Split by Velocity Correlation ({cell_type})")
    plt.legend()
    plt.show()


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


def plot_activity_by_r2_groups(activity_dict_SST, predicted_activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero,
                               activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_above_zero, r2_NDNF_below_zero,
                               activity_dict_EC, predicted_activity_dict_EC, r2_EC_above_zero, r2_EC_below_zero):

    activity_SST_above = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_above_zero, residual=True)
    activity_SST_below = filter_activity_by_r2(activity_dict_SST, predicted_activity_dict_SST, r2_SST_below_zero, residual=True)

    activity_NDNF_above = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_above_zero, residual=True)
    activity_NDNF_below = filter_activity_by_r2(activity_dict_NDNF, predicted_activity_dict_NDNF, r2_NDNF_below_zero, residual=True)

    activity_EC_above = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_above_zero, residual=True)
    activity_EC_below = filter_activity_by_r2(activity_dict_EC, predicted_activity_dict_EC, r2_EC_below_zero, residual=True)

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

    plt.figure(figsize=(12, 8))

    plt.errorbar(range(len(mean_SST_above)), mean_SST_above, yerr=sem_SST_above, fmt='o-', color='blue',
                 label='SST Above Zero')
    plt.errorbar(range(len(mean_SST_below)), mean_SST_below, yerr=sem_SST_below, fmt='o--', color='cyan',
                 label='SST Below Zero')

    plt.errorbar(range(len(mean_NDNF_above)), mean_NDNF_above, yerr=sem_NDNF_above, fmt='o-', color='orange',
                 label='NDNF Above Zero')
    plt.errorbar(range(len(mean_NDNF_below)), mean_NDNF_below, yerr=sem_NDNF_below, fmt='o--', color='red',
                 label='NDNF Below Zero')

    plt.errorbar(range(len(mean_EC_above)), mean_EC_above, yerr=sem_EC_above, fmt='o-', color='green',
                 label='EC Above Zero')
    plt.errorbar(range(len(mean_EC_below)), mean_EC_below, yerr=sem_EC_below, fmt='o--', color='gray',
                 label='EC Below Zero')

    plt.xlabel("Position Bins")
    plt.ylabel("Mean Activity")
    plt.title("Activity Split By Correlation To Velocity")
    plt.legend()
    plt.show()


def plot_positive_negative_selectivity_by_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                                   predicted_activity_dict_NDNF, activity_dict_EC,
                                                   predicted_activity_dict_EC):
    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5 = split_activity_and_prediction_into_quintiles(
        activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5 = split_activity_and_prediction_into_quintiles(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, predicted_activity_list_EC_q1, predicted_activity_list_EC_q5 = split_activity_and_prediction_into_quintiles(
        activity_dict_EC, predicted_activity_dict_EC)

    trial_av_activity_SST_q1 = trial_average(activity_list_SST_q1)
    trial_av_activity_NDNF_q1 = trial_average(activity_list_NDNF_q1)
    trial_av_activity_EC_q1 = trial_average(activity_list_EC_q1)

    trial_av_activity_SST_q5 = trial_average(activity_list_SST_q5)
    trial_av_activity_NDNF_q5 = trial_average(activity_list_NDNF_q5)
    trial_av_activity_EC_q5 = trial_average(activity_list_EC_q5)

    argmax_SST_q1 = get_max_or_min(trial_av_activity_SST_q1, argmax_or_argmin="argmax")
    argmin_SST_q1 = get_max_or_min(trial_av_activity_SST_q1, argmax_or_argmin="argmin")
    argmax_SST_q5 = get_max_or_min(trial_av_activity_SST_q5, argmax_or_argmin="argmax")
    argmin_SST_q5 = get_max_or_min(trial_av_activity_SST_q5, argmax_or_argmin="argmin")

    argmax_NDNF_q1 = get_max_or_min(trial_av_activity_NDNF_q1, argmax_or_argmin="argmax")
    argmin_NDNF_q1 = get_max_or_min(trial_av_activity_NDNF_q1, argmax_or_argmin="argmin")
    argmax_NDNF_q5 = get_max_or_min(trial_av_activity_NDNF_q5, argmax_or_argmin="argmax")
    argmin_NDNF_q5 = get_max_or_min(trial_av_activity_NDNF_q5, argmax_or_argmin="argmin")

    argmax_EC_q1 = get_max_or_min(trial_av_activity_EC_q1, argmax_or_argmin="argmax")
    argmin_EC_q1 = get_max_or_min(trial_av_activity_EC_q1, argmax_or_argmin="argmin")
    argmax_EC_q5 = get_max_or_min(trial_av_activity_EC_q5, argmax_or_argmin="argmax")
    argmin_EC_q5 = get_max_or_min(trial_av_activity_EC_q5, argmax_or_argmin="argmin")

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
                                                                               argmin_NDNF_q5, n_bins=20)

    mean_quantiles_EC_q1, sem_quantiles_EC_q1 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           argmax_EC_q1, n_bins=20)
    mean_quantiles_EC_q5, sem_quantiles_EC_q5 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           argmax_EC_q5, n_bins=20)

    mean_quantiles_SST_q1_negative, sem_quantiles_SST_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                               argmin_SST_q1,
                                                                                               n_bins=20)
    mean_quantiles_SST_q5_negative, sem_quantiles_SST_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                               argmin_SST_q5,
                                                                                               n_bins=20)

    mean_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                                 argmin_NDNF_q1,
                                                                                                 n_bins=20)
    mean_quantiles_NDNF_q5_negative, sem_quantiles_NDNF_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                                 argmin_NDNF_q5,
                                                                                                 n_bins=20)

    mean_quantiles_EC_q1_negative, sem_quantiles_EC_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                             argmin_EC_q1,
                                                                                             n_bins=20)
    mean_quantiles_EC_q5_negative, sem_quantiles_EC_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                             argmin_EC_q5,
                                                                                             n_bins=20)

    positive_mean_list = [mean_quantiles_SST_q1, mean_quantiles_SST_q5, mean_quantiles_NDNF_q1, mean_quantiles_NDNF_q5,
                          mean_quantiles_EC_q1, mean_quantiles_EC_q5]
    positive_sem_list = [sem_quantiles_SST_q1, sem_quantiles_SST_q5, sem_quantiles_NDNF_q1, sem_quantiles_NDNF_q5,
                         sem_quantiles_EC_q1, sem_quantiles_EC_q5]

    negative_mean_list = [mean_quantiles_SST_q1_negative, mean_quantiles_SST_q5_negative,
                          mean_quantiles_NDNF_q1_negative, mean_quantiles_NDNF_q5_negative,
                          mean_quantiles_EC_q1_negative, mean_quantiles_EC_q5_negative]
    negative_sem_list = [sem_quantiles_SST_q1_negative, sem_quantiles_SST_q5_negative, sem_quantiles_NDNF_q1_negative,
                         sem_quantiles_NDNF_q5_negative, sem_quantiles_EC_q1_negative, sem_quantiles_EC_q5_negative]

    plot_cdf_split_learning(positive_mean_list, positive_sem_list, title="Positive Selectivity", x_title="Selectivity",
                            n_bins=20)
    plot_cdf_split_learning(negative_mean_list, negative_sem_list, title="Negative Selectivity", x_title="Selectivity",
                            n_bins=20)





def compute_mean_and_sem_for_r2_groups(activity_dict, r2_above_zero, r2_below_zero):
    activity_above_zero = filter_activity_by_r2(activity_dict, r2_above_zero)
    activity_below_zero = filter_activity_by_r2(activity_dict, r2_below_zero)

    first_mean_above, first_sem_above, last_mean_above, last_sem_above = compute_mean_and_sem_for_quintiles(
        activity_above_zero)

    first_mean_below, first_sem_below, last_mean_below, last_sem_below = compute_mean_and_sem_for_quintiles(
        activity_below_zero)

    return (first_mean_above, first_sem_above, last_mean_above, last_sem_above), \
        (first_mean_below, first_sem_below, last_mean_below, last_sem_below)


def plot_mean_and_sem_by_r2(activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero, cell_type):
    above_zero_results, below_zero_results = compute_mean_and_sem_for_r2_groups(
        activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero)

    plt.figure(figsize=(10, 6))

    first_mean, first_sem, last_mean, last_sem = above_zero_results
    plt.errorbar(range(len(first_mean)), first_mean, yerr=first_sem, fmt='o-', color='blue',
                 label='Above Zero - First Quintile')
    plt.errorbar(range(len(last_mean)), last_mean, yerr=last_sem, fmt='o--', color='cyan',
                 label='Above Zero - Last Quintile')

    first_mean, first_sem, last_mean, last_sem = below_zero_results
    plt.errorbar(range(len(first_mean)), first_mean, yerr=first_sem, fmt='o-', color='orange',
                 label='Below Zero - First Quintile')
    plt.errorbar(range(len(last_mean)), last_mean, yerr=last_sem, fmt='o--', color='red',
                 label='Below Zero - Last Quintile')

    plt.xlabel("Position Bin")
    plt.ylabel("Z-Score Mean Activity")
    plt.title("Activity Split by Velocity Correlation")
    plt.legend()
    plt.show()


def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                variable_to_correlate="Velocity"):
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


def split_selectivity_by_r2(activity_dict_SST, predicted_activity_dict_SST,
                            activity_dict_NDNF, predicted_activity_dict_NDNF,
                            activity_dict_EC, predicted_activity_dict_EC,
                            residual=False, compute_negative=False):

    SST_factor_list, SST_negative_selectivity, NDNF_factor_list, NDNF_negative_selectivity, EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(
        activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC, residual=residual)

    SST_selectivity_list = SST_negative_selectivity if compute_negative else SST_factor_list
    NDNF_selectivity_list = NDNF_negative_selectivity if compute_negative else NDNF_factor_list
    EC_selectivity_list = EC_negative_selectivity if compute_negative else EC_factor_list

    neuron_mapping_SST = [(animal, neuron) for animal in activity_dict_SST for neuron in activity_dict_SST[animal]]
    neuron_mapping_NDNF = [(animal, neuron) for animal in activity_dict_NDNF for neuron in activity_dict_NDNF[animal]]
    neuron_mapping_EC = [(animal, neuron) for animal in activity_dict_EC for neuron in activity_dict_EC[animal]]

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_SST = "SSTindivsomata_GLM"
    filepath_SST = os.path.join(datasets_dir, filename_SST + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_NDNF = "NDNFindivsomata_GLM"
    filepath_NDNF = os.path.join(datasets_dir, filename_NDNF + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_EC = "EC_GLM"
    filepath_EC = os.path.join(datasets_dir, filename_EC + ".mat")

    activity_dict_SST, factors_dict_SST = preprocess_data(filepath_SST, normalize=True)
    activity_dict_NDNF, factors_dict_NDNF = preprocess_data(filepath_NDNF, normalize=True)
    activity_dict_EC, factors_dict_EC = preprocess_data(filepath_EC, normalize=True)

    filtered_factors_dict_SST = subset_variables_from_data(factors_dict_SST, variables_to_keep=["Velocity"])
    filtered_factors_dict_NDNF = subset_variables_from_data(factors_dict_NDNF, variables_to_keep=["Velocity"])
    filtered_factors_dict_EC = subset_variables_from_data(factors_dict_EC, variables_to_keep=["Velocity"])

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

    SST_above_zero, SST_below_zero = split_by_r2(neuron_mapping_SST, SST_selectivity_list, r2_SST_above_zero,
                                                 r2_SST_below_zero)
    NDNF_above_zero, NDNF_below_zero = split_by_r2(neuron_mapping_NDNF, NDNF_selectivity_list, r2_NDNF_above_zero,
                                                   r2_NDNF_below_zero)
    EC_above_zero, EC_below_zero = split_by_r2(neuron_mapping_EC, EC_selectivity_list, r2_EC_above_zero,
                                               r2_EC_below_zero)

    return SST_above_zero, SST_below_zero, NDNF_above_zero, NDNF_below_zero, EC_above_zero, EC_below_zero


def plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, title=None, x_title=None, n_bins=None):
    bin_centers = np.arange(1, n_bins + 1)

    plt.figure(figsize=(10, 6))

    plt.errorbar(mean_quantiles_list[0], bin_centers, xerr=sem_quantiles_list[0], fmt='o-', color='blue', ecolor='blue',
                 capsize=8, label="SST above 0 R vs Vel")
    plt.errorbar(mean_quantiles_list[1], bin_centers, xerr=sem_quantiles_list[1], fmt='o-', color='c', ecolor='c',
                 capsize=8, label="SST below 0 R vs Vel")

    plt.errorbar(mean_quantiles_list[2], bin_centers, xerr=sem_quantiles_list[2], fmt='o-', color='orange',
                 ecolor='orange',
                 capsize=8, label="NDNF above 0 R vs Vel")
    plt.errorbar(mean_quantiles_list[3], bin_centers, xerr=sem_quantiles_list[3], fmt='o-', color='r', ecolor='r',
                 capsize=8, label="NDNF below 0 R vs Vel")

    plt.errorbar(mean_quantiles_list[4], bin_centers, xerr=sem_quantiles_list[4], fmt='o-', color='green',
                 ecolor='green',
                 capsize=8, label="EC above 0 R vs Vel")
    plt.errorbar(mean_quantiles_list[5], bin_centers, xerr=sem_quantiles_list[5], fmt='o-', color='gray', ecolor='gray',
                 capsize=8, label="EC below 0 R vs Vel")

    plt.ylabel("Percentile of Data")
    plt.yticks(ticks=bin_centers, labels=[f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)])
    plt.xlabel(f" Mean {x_title}")
    plt.title(title)
    plt.legend()

    plt.show()


def setup_CDF_plotting_and_plot_selectivity_split_by_r2(activity_dict_SST, predicted_activity_dict_SST,
                                                        activity_dict_NDNF,
                                                        predicted_activity_dict_NDNF, activity_dict_EC,
                                                        predicted_activity_dict_EC,
                                                        residual=False):

    SST_factor_list, SST_negative_selectivity, NDNF_factor_list, NDNF_negative_selectivity, EC_factor_list, EC_negative_selectivity = get_selectivity_for_plotting(
        activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC, residual=residual)

    SST_factor_above_zero, SST_factor_below_zero, NDNF_factor_above_zero, NDNF_factor_below_zero, EC_factor_above_zero, EC_factor_below_zero = split_selectivity_by_r2(
        activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
        predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC,
        residual=residual, compute_negative=False)

    SST_factor_above_zero_negative, SST_factor_below_zero_negative, NDNF_factor_above_zero_negative, NDNF_factor_below_zero_negative, EC_factor_above_zero_negative, EC_factor_below_zero_negative = split_selectivity_by_r2(
        activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
        predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC,
        residual=residual, compute_negative=True)

    mean_quantiles_SST_high, sem_quantiles_SST_high = get_quantiles_for_cdf(activity_dict_SST, SST_factor_above_zero,
                                                                            n_bins=20)
    mean_quantiles_SST_low, sem_quantiles_SST_low = get_quantiles_for_cdf(activity_dict_SST, SST_factor_below_zero,
                                                                          n_bins=20)

    mean_quantiles_NDNF_high, sem_quantiles_NDNF_high = get_quantiles_for_cdf(activity_dict_NDNF,
                                                                              NDNF_factor_above_zero, n_bins=20)
    mean_quantiles_NDNF_low, sem_quantiles_NDNF_low = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_factor_below_zero,
                                                                            n_bins=20)

    mean_quantiles_EC_high, sem_quantiles_EC_high = get_quantiles_for_cdf(activity_dict_EC, EC_factor_above_zero,
                                                                          n_bins=20)
    mean_quantiles_EC_low, sem_quantiles_EC_low = get_quantiles_for_cdf(activity_dict_EC, SST_factor_below_zero,
                                                                        n_bins=20)

    mean_quantiles_SST_high_negative, sem_quantiles_SST_high_negative = get_quantiles_for_cdf(activity_dict_SST,
                                                                                              SST_factor_above_zero_negative,
                                                                                              n_bins=20)
    mean_quantiles_SST_low_negative, sem_quantiles_SST_low_negative = get_quantiles_for_cdf(activity_dict_SST,
                                                                                            SST_factor_below_zero_negative,
                                                                                            n_bins=20)

    mean_quantiles_NDNF_high_negative, sem_quantiles_NDNF_high_negative = get_quantiles_for_cdf(activity_dict_NDNF,
                                                                                                NDNF_factor_above_zero_negative,
                                                                                                n_bins=20)
    mean_quantiles_NDNF_low_negative, sem_quantiles_NDNF_low_negative = get_quantiles_for_cdf(activity_dict_NDNF,
                                                                                              NDNF_factor_below_zero_negative,
                                                                                              n_bins=20)

    mean_quantiles_EC_high_negative, sem_quantiles_EC_high_negative = get_quantiles_for_cdf(activity_dict_EC,
                                                                                            EC_factor_above_zero_negative,
                                                                                            n_bins=20)
    mean_quantiles_EC_low_negative, sem_quantiles_EC_low_negative = get_quantiles_for_cdf(activity_dict_EC,
                                                                                          SST_factor_below_zero_negative,
                                                                                          n_bins=20)

    mean_quantiles_list = [mean_quantiles_SST_high, mean_quantiles_SST_low, mean_quantiles_NDNF_high,
                           mean_quantiles_NDNF_low, mean_quantiles_EC_high, mean_quantiles_EC_low]

    sem_quantiles_list = [sem_quantiles_SST_high, sem_quantiles_SST_low, sem_quantiles_NDNF_high,
                          sem_quantiles_NDNF_low, sem_quantiles_EC_high, sem_quantiles_EC_low]

    mean_quantiles_list_negative = [mean_quantiles_SST_high_negative, mean_quantiles_SST_low_negative,
                                    mean_quantiles_NDNF_high_negative, mean_quantiles_NDNF_low_negative,
                                    mean_quantiles_EC_high_negative, mean_quantiles_EC_low_negative]

    sem_quantiles_list_negative = [sem_quantiles_SST_high_negative, sem_quantiles_SST_low_negative,
                                   sem_quantiles_NDNF_high_negative, sem_quantiles_NDNF_low_negative,
                                   sem_quantiles_EC_high_negative, sem_quantiles_EC_low_negative]


    plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, "Selectivity", "Vinje Selectivity Index", n_bins=20)
    plot_cdf_split_r2(mean_quantiles_list_negative, sem_quantiles_list_negative, "Negative Selectivity", "Negative Vinje Selectivity Index", n_bins=20)






def split_by_r2(neuron_mapping, factor_list, r2_above_zero, r2_below_zero):
    above_zero = []
    below_zero = []

    for idx, (animal, neuron) in enumerate(neuron_mapping):
        if animal in r2_above_zero and neuron in r2_above_zero[animal]:
            above_zero.append(factor_list[idx])
        elif animal in r2_below_zero and neuron in r2_below_zero[animal]:
            below_zero.append(factor_list[idx])

    return above_zero, below_zero




def compute_mean_and_sem_for_quintiles(activity_dict, predicted_activity_dict):

    first_quintile_residuals = []
    last_quintile_residuals = []
    animal_mean_first_quintile_residuals = []
    animal_mean_last_quintile_residuals = []

    for animal in activity_dict:
        neuron_mean_first_quintile_residuals = []
        neuron_mean_last_quintile_residuals = []
        for neuron in activity_dict[animal]:
            # Split each neuron's raw and predicted data into quintiles
            raw_quintiles = split_into_quintiles(activity_dict[animal][neuron])
            predicted_quintiles = split_into_quintiles(predicted_activity_dict[animal][neuron])

            # Compute trial-averaged first and last quintile residuals
            trial_av_first_quintile = np.mean(raw_quintiles[0], axis=1)
            trial_av_first_quintile_predicted = np.mean(predicted_quintiles[0], axis=1)

            trial_av_last_quintile = np.mean(raw_quintiles[-1], axis=1)
            trial_av_last_quintile_predicted = np.mean(predicted_quintiles[-1], axis=1)

            first_quintile_residual = trial_av_first_quintile - trial_av_first_quintile_predicted
            last_quintile_residual = trial_av_last_quintile - trial_av_last_quintile_predicted

            # Append residuals for this neuron
            neuron_mean_first_quintile_residuals.append(first_quintile_residual)
            neuron_mean_last_quintile_residuals.append(last_quintile_residual)

            # Store the individual residuals for overall computation
            first_quintile_residuals.append(first_quintile_residual)
            last_quintile_residuals.append(last_quintile_residual)


        animal_mean_first_quintile_residuals.append(np.mean(np.stack(neuron_mean_first_quintile_residuals), axis=0))
        animal_mean_last_quintile_residuals.append(np.mean(np.stack(neuron_mean_last_quintile_residuals), axis=0))

    # Convert residuals to arrays
    first_quintile_residuals = np.array(first_quintile_residuals)
    last_quintile_residuals = np.array(last_quintile_residuals)

    # Compute mean and SEM across all neurons
    first_mean_residual = np.mean(first_quintile_residuals, axis=0)  # Mean across neurons
    first_sem_residual = np.std(first_quintile_residuals, axis=0) / np.sqrt(first_quintile_residuals.shape[0])

    last_mean_residual = np.mean(last_quintile_residuals, axis=0)
    last_sem_residual = np.std(last_quintile_residuals, axis=0) / np.sqrt(last_quintile_residuals.shape[0])


    return (
        first_mean_residual,
        first_sem_residual,
        last_mean_residual,
        last_sem_residual,
        animal_mean_first_quintile_residuals,
        animal_mean_last_quintile_residuals, )





# def compute_mean_and_sem_for_quintiles(activity_dict, predicted_activity_dict):
#     """
#     Compute the mean and SEM for the first and last quintiles across neurons and animals.
#     """
#     first_quintile_means = []
#     last_quintile_means = []
#
#     for animal in activity_dict:
#         for neuron in activity_dict[animal]:
#             # Split each neuron's data into quintiles
#             quintiles = split_into_quintiles(activity_dict[animal][neuron])
#             first_quintile = quintiles[0]
#             last_quintile = quintiles[-1]
#
#             # Compute the mean of the first and last quintiles
#             first_quintile_mean = np.mean(first_quintile, axis=1)  # Average over trials
#             last_quintile_mean = np.mean(last_quintile, axis=1)
#
#             first_quintile_means.append(first_quintile_mean)
#             last_quintile_means.append(last_quintile_mean)
#
#     # Convert to arrays
#     first_quintile_means = np.array(first_quintile_means)
#     last_quintile_means = np.array(last_quintile_means)
#
#     # Compute mean and SEM across neurons
#     first_mean = np.mean(first_quintile_means, axis=0)  # Average over neurons
#     first_sem = np.std(first_quintile_means, axis=0) / np.sqrt(first_quintile_means.shape[0])
#
#     last_mean = np.mean(last_quintile_means, axis=0)
#     last_sem = np.std(last_quintile_means, axis=0) / np.sqrt(last_quintile_means.shape[0])
#
#     return first_mean, first_sem, last_mean, last_sem


def plot_frequency_hist_split_by_r2(SST_list_above, SST_list_below, NDNF_list_above, NDNF_list_below, EC_list_above,
                                    EC_list_below, selectivity_or_arg="selectivity", name=None, residual=True):
    if selectivity_or_arg == "selectivity":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif selectivity_or_arg == "arg":
        bin_edges = np.arange(0, 51, 5)
        bin_centers = bin_edges[:-1] + 2.5
        bin_labels = [f"{start}-{start + 4}" for start in bin_edges[:-1]]

    else:
        raise ValueError(" selectivity_or_arg takes either selectivity or arg")

    SST_hist_above, _ = np.histogram(SST_list_above, bins=bin_edges)
    NDNF_hist_above, _ = np.histogram(NDNF_list_above, bins=bin_edges)
    EC_hist_above, _ = np.histogram(EC_list_above, bins=bin_edges)

    SST_fraction_above = SST_hist_above / np.sum(SST_hist_above)
    NDNF_fraction_above = NDNF_hist_above / np.sum(NDNF_hist_above)
    EC_fraction_above = EC_hist_above / np.sum(EC_hist_above)

    SST_hist_below, _ = np.histogram(SST_list_below, bins=bin_edges)
    NDNF_hist_below, _ = np.histogram(NDNF_list_below, bins=bin_edges)
    EC_hist_below, _ = np.histogram(EC_list_below, bins=bin_edges)

    SST_fraction_below = SST_hist_below / np.sum(SST_hist_below)
    NDNF_fraction_below = NDNF_hist_below / np.sum(NDNF_hist_below)
    EC_fraction_below = EC_hist_below / np.sum(EC_hist_below)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, SST_fraction_above, marker='o', label=f'SST above 0 R vs Vel', linestyle='-', color='b')
    plt.plot(bin_centers, NDNF_fraction_above, marker='o', label=f'NDNF above 0 R vs Vel', linestyle='-', color='orange')
    plt.plot(bin_centers, EC_fraction_above, marker='o', label=f'EC above 0 R vs Vel', linestyle='-', color='green')
    plt.plot(bin_centers, SST_fraction_below, marker='o', label=f'SST below 0 R vs Vel', linestyle='-', color='c')
    plt.plot(bin_centers, NDNF_fraction_below, marker='o', label=f'NDNF below 0 R vs Vel', linestyle='-', color='red')
    plt.plot(bin_centers, EC_fraction_below, marker='o', label=f'EC Low below 0 R vs Vel', linestyle='-', color='gray')

    plt.xlabel(name)
    plt.ylabel('Fraction of Cells')
    plt.title(f'{name} Split by Velocity Correlation residual={residual}')
    plt.xticks(bin_centers, bin_labels)
    #     plt.xticks(bin_centers[::2], bin_labels[::2])
    plt.legend()

    plt.tight_layout()
    plt.show()


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


def plot_frequency_hist_learning(SST_list_above, SST_list_below, NDNF_list_above, NDNF_list_below, EC_list_above,
                                 EC_list_below, selectivity_or_arg="selectivity", name=None):
    if selectivity_or_arg == "selectivity":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif selectivity_or_arg == "arg":
        bin_edges = np.arange(0, 51, 5)
        bin_centers = bin_edges[:-1] + 2.5
        bin_labels = [f"{start}-{start + 4}" for start in bin_edges[:-1]]

    else:
        raise ValueError(" selectivity_or_arg takes either selectivity or arg")

    SST_hist_above, _ = np.histogram(SST_list_above, bins=bin_edges)
    NDNF_hist_above, _ = np.histogram(NDNF_list_above, bins=bin_edges)
    EC_hist_above, _ = np.histogram(EC_list_above, bins=bin_edges)

    SST_fraction_above = SST_hist_above / np.sum(SST_hist_above)
    NDNF_fraction_above = NDNF_hist_above / np.sum(NDNF_hist_above)
    EC_fraction_above = EC_hist_above / np.sum(EC_hist_above)

    SST_hist_below, _ = np.histogram(SST_list_below, bins=bin_edges)
    NDNF_hist_below, _ = np.histogram(NDNF_list_below, bins=bin_edges)
    EC_hist_below, _ = np.histogram(EC_list_below, bins=bin_edges)

    SST_fraction_below = SST_hist_below / np.sum(SST_hist_below)
    NDNF_fraction_below = NDNF_hist_below / np.sum(NDNF_hist_below)
    EC_fraction_below = EC_hist_below / np.sum(EC_hist_below)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, SST_fraction_above, marker='o', label=f'SST Early Learn (Q1)', linestyle='-', color='b')
    plt.plot(bin_centers, NDNF_fraction_above, marker='o', label=f'SST Late Learn (Q5)', linestyle='-', color='cyan')
    plt.plot(bin_centers, EC_fraction_above, marker='o', label=f'NDNF Early Learn (Q1)', linestyle='-', color='orange')
    plt.plot(bin_centers, SST_fraction_below, marker='o', label=f'NDNF Late Learn (Q5)', linestyle='-', color='red')
    plt.plot(bin_centers, NDNF_fraction_below, marker='o', label=f'EC Early Learn (Q1)', linestyle='-', color='green')
    plt.plot(bin_centers, EC_fraction_below, marker='o', label=f'EC Late Learn (Q5)', linestyle='-', color='gray')

    plt.xlabel(name)
    plt.ylabel('Fraction of Cells')
    plt.title(f'{name} Split by Quintile')
    plt.xticks(bin_centers, bin_labels)
    #     plt.xticks(bin_centers[::2], bin_labels[::2])
    plt.legend()

    plt.tight_layout()
    plt.show()


def get_argmin_argmax_split_learning_histogram(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                     predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):
    # activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5 = split_activity_and_prediction_into_quintiles(
    #     activity_dict_SST, predicted_activity_dict_SST)
    # activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5 = split_activity_and_prediction_into_quintiles(
    #     activity_dict_NDNF, predicted_activity_dict_NDNF)
    # activity_list_EC_q1, activity_list_EC_q5, prediction_list_EC_q1, prediction_list_EC_q5 = split_activity_and_prediction_into_quintiles(
    #     activity_dict_EC, predicted_activity_dict_EC)
    #
    # trial_av_activity_SST_q1 = trial_average(activity_list_SST_q1)
    # trial_av_activity_NDNF_q1 = trial_average(activity_list_NDNF_q1)
    # trial_av_activity_EC_q1 = trial_average(activity_list_EC_q1)
    #
    # trial_av_activity_SST_q5 = trial_average(activity_list_SST_q5)
    # trial_av_activity_NDNF_q5 = trial_average(activity_list_NDNF_q5)
    # trial_av_activity_EC_q5 = trial_average(activity_list_EC_q5)
    #
    # argmax_SST_q1 = get_max_or_min(trial_av_activity_SST_q1, argmax_or_argmin="argmax")
    # argmin_SST_q1 = get_max_or_min(trial_av_activity_SST_q1, argmax_or_argmin="argmin")
    # argmax_SST_q5 = get_max_or_min(trial_av_activity_SST_q5, argmax_or_argmin="argmax")
    # argmin_SST_q5 = get_max_or_min(trial_av_activity_SST_q5, argmax_or_argmin="argmin")
    #
    # argmax_NDNF_q1 = get_max_or_min(trial_av_activity_NDNF_q1, argmax_or_argmin="argmax")
    # argmin_NDNF_q1 = get_max_or_min(trial_av_activity_NDNF_q1, argmax_or_argmin="argmin")
    # argmax_NDNF_q5 = get_max_or_min(trial_av_activity_NDNF_q5, argmax_or_argmin="argmax")
    # argmin_NDNF_q5 = get_max_or_min(trial_av_activity_NDNF_q5, argmax_or_argmin="argmin")
    #
    # argmax_EC_q1 = get_max_or_min(trial_av_activity_EC_q1, argmax_or_argmin="argmax")
    # argmin_EC_q1 = get_max_or_min(trial_av_activity_EC_q1, argmax_or_argmin="argmin")
    # argmax_EC_q5 = get_max_or_min(trial_av_activity_EC_q5, argmax_or_argmin="argmax")
    # argmin_EC_q5 = get_max_or_min(trial_av_activity_EC_q5, argmax_or_argmin="argmin")

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



    plot_frequency_hist_learning(argmax_SST_q1, argmax_SST_q5, argmax_NDNF_q1, argmax_NDNF_q5, argmax_EC_q1,
                                 argmax_EC_q5, selectivity_or_arg="arg", name="Argmax")
    plot_frequency_hist_learning(argmin_SST_q1, argmin_SST_q5, argmin_NDNF_q1, argmin_NDNF_q5, argmin_EC_q1,
                                 argmin_EC_q5, selectivity_or_arg="arg", name="Argmin")


def plot_first_and_last_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):

    first_mean_SST, first_sem_SST, last_mean_SST, last_sem_SST, animal_mean_residual_first_SST, animal_mean_residual_last_SST = compute_mean_and_sem_for_quintiles(activity_dict_SST, predicted_activity_dict_SST)

    first_mean_NDNF, first_sem_NDNF, last_mean_NDNF, last_sem_NDNF, animal_mean_residual_first_NDNF, animal_mean_residual_last_NDNF = compute_mean_and_sem_for_quintiles(activity_dict_NDNF, predicted_activity_dict_NDNF)

    first_mean_EC, first_sem_EC, last_mean_EC, last_sem_EC, animal_mean_residual_first_EC, animal_mean_residual_last_EC = compute_mean_and_sem_for_quintiles(activity_dict_EC, predicted_activity_dict_EC)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    axs[0, 0].errorbar(range(len(first_mean_SST)), first_mean_SST, yerr=first_sem_SST, fmt='o-', color='blue', label='SST First Quintile')
    axs[0, 0].errorbar(range(len(last_mean_SST)), last_mean_SST, yerr=last_sem_SST, fmt='o--', color='cyan', label='SST Last Quintile')
    axs[0, 0].set_xlabel("Position Bin")
    axs[0, 0].set_ylabel("z-score DF/F")
    axs[0, 0].set_title("SST Quintile Activity")
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 0].legend()  # Add legend to this subplot only


    for animal in range(len(animal_mean_residual_first_SST)):
        axs[0, 1].plot(animal_mean_residual_first_SST[animal], color='blue', alpha=0.5)
    axs[0, 1].plot(np.mean(np.stack(animal_mean_residual_first_SST), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean Q1")
    axs[0, 1].set_xlabel("Position Bin")
    axs[0, 1].set_ylabel("z-score DF/F")
    axs[0, 1].set_title("SST Quintile Activity Per Animal")
    axs[0, 1].set_ylim(-1, 1)
    axs[0, 1].legend()

    for animal in range(len(animal_mean_residual_last_SST)):
        axs[0, 2].plot(animal_mean_residual_last_SST[animal], color='cyan', alpha=0.5)
    axs[0, 2].plot(np.mean(np.stack(animal_mean_residual_last_SST), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean Q5")
    axs[0, 2].set_xlabel("Position Bin")
    axs[0, 2].set_ylabel("z-score DF/F")
    axs[0, 2].set_title("SST Quintile Activity Per Animal")
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 2].legend()

    # NDNF Plots
    axs[1, 0].errorbar(range(len(first_mean_NDNF)), first_mean_NDNF, yerr=first_sem_NDNF, fmt='o-', color='orange', label='NDNF First Quintile')
    axs[1, 0].errorbar(range(len(last_mean_NDNF)), last_mean_NDNF, yerr=last_sem_NDNF, fmt='o--', color='red', label='NDNF Last Quintile')
    axs[1, 0].set_xlabel("Position Bin")
    axs[1, 0].set_ylabel("z-score DF/F")
    axs[1, 0].set_title("NDNF Quintile Activity")
    axs[1, 0].set_ylim(-1, 1)
    axs[1, 0].legend()

    for animal in range(len(animal_mean_residual_first_NDNF)):
        axs[1, 1].plot(animal_mean_residual_first_NDNF[animal], color='orange', alpha=0.5)
    axs[1, 1].plot(np.mean(np.stack(animal_mean_residual_first_NDNF), axis=0), color='red', alpha=0.9, linewidth=4,label="NDNF Mean Q1")
    axs[1, 1].set_xlabel("Position Bin")
    axs[1, 1].set_ylabel("z-score DF/F")
    axs[1, 1].set_title("NDNF Quintile Activity Per Animal")
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].legend()

    for animal in range(len(animal_mean_residual_last_NDNF)):
        axs[1, 2].plot(animal_mean_residual_last_NDNF[animal], color='red', alpha=0.5)
    axs[1, 2].plot(np.mean(np.stack(animal_mean_residual_last_NDNF), axis=0), color='red', alpha=0.9, linewidth=4, label="NDNF Mean Q5")
    axs[1, 2].set_xlabel("Position Bin")
    axs[1, 2].set_ylabel("z-score DF/F")
    axs[1, 2].set_title("NDNF Quintile Activity Per Animal")
    axs[1, 2].set_ylim(-1, 1)
    axs[1, 2].legend()

    # EC Plots
    axs[2, 0].errorbar(range(len(first_mean_EC)), first_mean_EC, yerr=first_sem_EC, fmt='o-', color='green', label='EC First Quintile')
    axs[2, 0].errorbar(range(len(last_mean_EC)), last_mean_EC, yerr=last_sem_EC, fmt='o--', color='gray', label='EC Last Quintile')
    axs[2, 0].set_xlabel("Position Bin")
    axs[2, 0].set_ylabel("z-score DF/F")
    axs[2, 0].set_title("EC Quintile Activity")
    axs[2, 0].set_ylim(-1, 1)
    axs[2, 0].legend()

    for animal in range(len(animal_mean_residual_first_EC)):
        axs[2, 1].plot(animal_mean_residual_first_EC[animal], color='green', alpha=0.5)
    axs[2, 1].plot(np.mean(np.stack(animal_mean_residual_first_EC), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean Q1")
    axs[2, 1].set_xlabel("Position Bin")
    axs[2, 1].set_ylabel("z-score DF/F")
    axs[2, 1].set_title("EC Quintile Activity Per Animal")
    axs[2, 1].set_ylim(-1, 1)
    axs[2, 1].legend()

    for animal in range(len(animal_mean_residual_last_EC)):
        axs[2, 2].plot(animal_mean_residual_last_EC[animal], color='gray', alpha=0.5)
    axs[2, 2].plot(np.mean(np.stack(animal_mean_residual_last_EC), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean Q5")
    axs[2, 2].set_xlabel("Position Bin")
    axs[2, 2].set_ylabel("z-score DF/F")
    axs[2, 2].set_title("EC Quintile Activity Per Animal")
    axs[2, 2].set_ylim(-1, 1)
    axs[2, 2].legend()

    # Adjust layout
    fig.tight_layout()
    plt.show()



def get_pop_correlation_to_variable(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                    variable_to_correlate="Velocity"):
    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)

    filtered_input_variables = {var: [] for var in
                                filtered_factors_dict_SST[next(iter(filtered_factors_dict_SST))].keys()}
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


def get_r2_above_and_below_zero(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
                                variable_to_correlate="Velocity"):
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



def compute_mean_and_sem_for_r2_groups(activity_dict, r2_above_zero, r2_below_zero):
    activity_above_zero = filter_activity_by_r2(activity_dict, r2_above_zero)
    activity_below_zero = filter_activity_by_r2(activity_dict, r2_below_zero)

    first_mean_above, first_sem_above, last_mean_above, last_sem_above = compute_mean_and_sem_for_quintiles(
        activity_above_zero)

    first_mean_below, first_sem_below, last_mean_below, last_sem_below = compute_mean_and_sem_for_quintiles(
        activity_below_zero)

    return (first_mean_above, first_sem_above, last_mean_above, last_sem_above), \
        (first_mean_below, first_sem_below, last_mean_below, last_sem_below)


def plot_mean_and_sem_by_r2(activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero, cell_type):
    above_zero_results, below_zero_results = compute_mean_and_sem_for_r2_groups(
        activity_dict_SST, r2_SST_above_zero, r2_SST_below_zero)

    plt.figure(figsize=(10, 6))

    first_mean, first_sem, last_mean, last_sem = above_zero_results
    plt.errorbar(range(len(first_mean)), first_mean, yerr=first_sem, fmt='o-', color='blue',
                 label='Above Zero - First Quintile')
    plt.errorbar(range(len(last_mean)), last_mean, yerr=last_sem, fmt='o--', color='cyan',
                 label='Above Zero - Last Quintile')

    first_mean, first_sem, last_mean, last_sem = below_zero_results
    plt.errorbar(range(len(first_mean)), first_mean, yerr=first_sem, fmt='o-', color='orange',
                 label='Below Zero - First Quintile')
    plt.errorbar(range(len(last_mean)), last_mean, yerr=last_sem, fmt='o--', color='red',
                 label='Below Zero - Last Quintile')

    plt.xlabel("Position Bin")
    plt.ylabel("Z-Score Mean Activity")
    plt.title(f"Activity Split by Velocity Correlation for {cell_type}")
    plt.legend()
    plt.show()




def split_argmin_argmax_by_r2(activity_dict_SST, predicted_activity_dict_SST,
                              activity_dict_NDNF, predicted_activity_dict_NDNF,
                              activity_dict_EC, predicted_activity_dict_EC,
                              r2_SST_above_zero, r2_SST_below_zero,
                              r2_NDNF_above_zero, r2_NDNF_below_zero,
                              r2_EC_above_zero, r2_EC_below_zero,
                              residual=False, which_to_plot="argmin"):
    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_SST = "SSTindivsomata_GLM"
    filepath_SST = os.path.join(datasets_dir, filename_SST + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_NDNF = "NDNFindivsomata_GLM"
    filepath_NDNF = os.path.join(datasets_dir, filename_NDNF + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_EC = "EC_GLM"
    filepath_EC = os.path.join(datasets_dir, filename_EC + ".mat")

    activity_dict_SST, factors_dict_SST = preprocess_data(filepath_SST, normalize=True)
    activity_dict_NDNF, factors_dict_NDNF = preprocess_data(filepath_NDNF, normalize=True)
    activity_dict_EC, factors_dict_EC = preprocess_data(filepath_EC, normalize=True)

    filtered_factors_dict_SST = subset_variables_from_data(factors_dict_SST, variables_to_keep=["Velocity"])
    filtered_factors_dict_NDNF = subset_variables_from_data(factors_dict_NDNF, variables_to_keep=["Velocity"])
    filtered_factors_dict_EC = subset_variables_from_data(factors_dict_EC, variables_to_keep=["Velocity"])

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

    SST_above_zero, SST_below_zero = split_by_r2(neuron_mapping_SST, SST_factor_list, r2_SST_above_zero,
                                                 r2_SST_below_zero)
    NDNF_above_zero, NDNF_below_zero = split_by_r2(neuron_mapping_NDNF, NDNF_factor_list, r2_NDNF_above_zero,
                                                   r2_NDNF_below_zero)
    EC_above_zero, EC_below_zero = split_by_r2(neuron_mapping_EC, EC_factor_list, r2_EC_above_zero, r2_EC_below_zero)

    print(f"residual {residual}")

    return SST_above_zero, SST_below_zero, NDNF_above_zero, NDNF_below_zero, EC_above_zero, EC_below_zero


def setup_CDF_plotting_and_plot_argmin_argmax_split_by_r2(activity_dict_SST, predicted_activity_dict_SST,
                                                          activity_dict_NDNF,
                                                          predicted_activity_dict_NDNF, activity_dict_EC,
                                                          predicted_activity_dict_EC, residual=False,
                                                          which_to_plot="argmin"):
    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_SST = "SSTindivsomata_GLM"
    filepath_SST = os.path.join(datasets_dir, filename_SST + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_NDNF = "NDNFindivsomata_GLM"
    filepath_NDNF = os.path.join(datasets_dir, filename_NDNF + ".mat")

    datasets_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "datasets"))
    filename_EC = "EC_GLM"
    filepath_EC = os.path.join(datasets_dir, filename_EC + ".mat")

    activity_dict_SST, factors_dict_SST = preprocess_data(filepath_SST, normalize=True)
    activity_dict_NDNF, factors_dict_NDNF = preprocess_data(filepath_NDNF, normalize=True)
    activity_dict_EC, factors_dict_EC = preprocess_data(filepath_EC, normalize=True)

    filtered_factors_dict_SST = subset_variables_from_data(factors_dict_SST, variables_to_keep=["Velocity"])
    filtered_factors_dict_NDNF = subset_variables_from_data(factors_dict_NDNF, variables_to_keep=["Velocity"])
    filtered_factors_dict_EC = subset_variables_from_data(factors_dict_EC, variables_to_keep=["Velocity"])

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

    SST_above_zero, SST_below_zero, NDNF_above_zero, NDNF_below_zero, EC_above_zero, EC_below_zero = split_argmin_argmax_by_r2(
        activity_dict_SST, predicted_activity_dict_SST,
        activity_dict_NDNF, predicted_activity_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC,
        r2_SST_above_zero, r2_SST_below_zero,
        r2_NDNF_above_zero, r2_NDNF_below_zero,
        r2_EC_above_zero, r2_EC_below_zero,
        residual=residual, which_to_plot=which_to_plot)

    mean_quantiles_SST_high, sem_quantiles_SST_high = get_quantiles_for_cdf(activity_dict_SST, SST_above_zero,
                                                                            n_bins=20)
    mean_quantiles_SST_low, sem_quantiles_SST_low = get_quantiles_for_cdf(activity_dict_SST, SST_below_zero, n_bins=20)

    mean_quantiles_NDNF_high, sem_quantiles_NDNF_high = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_above_zero,
                                                                              n_bins=20)
    mean_quantiles_NDNF_low, sem_quantiles_NDNF_low = get_quantiles_for_cdf(activity_dict_NDNF, NDNF_below_zero,
                                                                            n_bins=20)

    mean_quantiles_EC_high, sem_quantiles_EC_high = get_quantiles_for_cdf(activity_dict_EC, EC_above_zero, n_bins=20)
    mean_quantiles_EC_low, sem_quantiles_EC_low = get_quantiles_for_cdf(activity_dict_EC, SST_below_zero, n_bins=20)

    mean_quantiles_list = [mean_quantiles_SST_high, mean_quantiles_SST_low, mean_quantiles_NDNF_high,
                           mean_quantiles_NDNF_low, mean_quantiles_EC_high, mean_quantiles_EC_low]

    sem_quantiles_list = [sem_quantiles_SST_high, sem_quantiles_SST_low, sem_quantiles_NDNF_high,
                          sem_quantiles_NDNF_low, sem_quantiles_EC_high, sem_quantiles_EC_low]

    print(f"residual {residual}")
    if which_to_plot == "argmin":
        if residual:

            plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, "Argmin Split by Velocity Correlation Residual",
                              "Position Bin of Minimum Firing",
                              n_bins=20)

        else:

            plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, "Argmin Split by Velocity Correlation Raw",
                              "Position Bin of Minimum Firing",
                              n_bins=20)

    elif which_to_plot == "argmax":
        if residual:

            plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, "Argmax Split by Velocity Correlation Residual",
                              "Position Bin of Minimum Firing",
                              n_bins=20)

        else:

            plot_cdf_split_r2(mean_quantiles_list, sem_quantiles_list, "Argmax Split by Velocity Correlation Raw",
                              "Position Bin of Minimum Firing",
                              n_bins=20)

    else:
        raise ValueError("options are argmin or argmax")


def split_argmin_argmax(activity_dict_SST, predicted_activity_dict_SST,
                        activity_dict_NDNF, predicted_activity_dict_NDNF,
                        activity_dict_EC, predicted_activity_dict_EC,
                        residual=False, which_to_plot="argmin"):
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

def plot_position_frequency(SST_list, NDNF_list, EC_list, selectivity_or_arg="selectivity", name=None):

    if selectivity_or_arg == "selectivity":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif selectivity_or_arg == "arg":
        bin_edges = np.arange(0, 51, 5)
        bin_centers = bin_edges[:-1] + 2.5
        bin_labels = [f"{start}-{start + 4}" for start in bin_edges[:-1]]

    SST_hist, _ = np.histogram(SST_list, bins=bin_edges)
    NDNF_hist, _ = np.histogram(NDNF_list, bins=bin_edges)
    EC_hist, _ = np.histogram(EC_list, bins=bin_edges)

    SST_fraction = SST_hist / np.sum(SST_hist)
    NDNF_fraction = NDNF_hist / np.sum(NDNF_hist)
    EC_fraction = EC_hist / np.sum(EC_hist)

    bin_centers = bin_edges[:-1] + 2.5

    plt.figure(figsize=(8, 6))

    plt.plot(bin_centers, SST_fraction, marker='o', label=f'SST {name}', linestyle='-')
    plt.plot(bin_centers, NDNF_fraction, marker='o', label=f'NDNF {name}', linestyle='-')
    plt.plot(bin_centers, EC_fraction, marker='o', label=f'EC {name}', linestyle='-')

    plt.xlabel('Position Bin')
    plt.ylabel('Fraction of Cells')
    plt.title(f'{name}')
    plt.xticks(bin_centers, bin_labels, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()


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

    # print("Trial Average (activity_list_SST_q1):", trial_average(activity_list_SST_q1))
    # print("Trial Average (prediction_list_SST_q1):", trial_average(prediction_list_SST_q1))
    #
    # residual_q1 = trial_average(activity_list_SST_q1) - trial_average(prediction_list_SST_q1)
    #
    # residual_q5 = trial_average(activity_list_SST_q5) - trial_average(prediction_list_SST_q5)

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


def plot_cdf_split_learning(mean_quantiles_list, sem_quantiles_list, title=None, x_title=None, n_bins=None):
    bin_centers = np.arange(1, n_bins + 1)

    plt.figure(figsize=(10, 6))

    plt.errorbar(mean_quantiles_list[0], bin_centers, xerr=sem_quantiles_list[0], fmt='o-', color='blue', ecolor='blue',
                 capsize=8, label="SST Early Learn (Q1)")
    plt.errorbar(mean_quantiles_list[1], bin_centers, xerr=sem_quantiles_list[1], fmt='o-', color='c', ecolor='c',
                 capsize=8, label="SST Late Learn (Q5)")

    plt.errorbar(mean_quantiles_list[2], bin_centers, xerr=sem_quantiles_list[2], fmt='o-', color='orange',
                 ecolor='orange',
                 capsize=8, label="NDNF Early Learn (Q1)")
    plt.errorbar(mean_quantiles_list[3], bin_centers, xerr=sem_quantiles_list[3], fmt='o-', color='r', ecolor='r',
                 capsize=8, label="NDNF Early Late Learn (Q5)")

    plt.errorbar(mean_quantiles_list[4], bin_centers, xerr=sem_quantiles_list[4], fmt='o-', color='green',
                 ecolor='green',
                 capsize=8, label="EC Early Learn (Q1)")
    plt.errorbar(mean_quantiles_list[5], bin_centers, xerr=sem_quantiles_list[5], fmt='o-', color='gray', ecolor='gray',
                 capsize=8, label="EC Late Learn (Q5)")

    plt.ylabel("Percentile of Data")
    plt.yticks(ticks=bin_centers, labels=[f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)])
    plt.xlabel(f" Mean {x_title}")
    plt.title(title)
    plt.legend()

    plt.show()


def get_selectivity_for_plotting_lists(neuron_activity_list_SST, neuron_activity_list_NDNF, neuron_activity_list_EC):
    # trial_av_activity_SST = trial_average(neuron_activity_list_SST)
    # trial_av_activity_NDNF = trial_average(neuron_activity_list_NDNF)
    # trial_av_activity_EC = trial_average(neuron_activity_list_EC)

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


def plot_positive_negative_selectivity_by_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                                   predicted_activity_dict_NDNF, activity_dict_EC,
                                                   predicted_activity_dict_EC):
    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(
        activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q1_NDNF = split_activity_and_prediction_into_quintiles(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, predicted_activity_list_EC_q1, predicted_activity_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(
        activity_dict_EC, predicted_activity_dict_EC)

    SST_positive_selectivity_q1, SST_negative_selectivity_q1, NDNF_positive_selectivity_q1, NDNF_negative_selectivity_q1, EC_positive_selectivity_q1, EC_negative_selectivity_q1 = get_selectivity_for_plotting_lists(
        residual_q1_SST, residual_q1_NDNF, residual_q1_EC)

    SST_positive_selectivity_q5, SST_negative_selectivity_q5, NDNF_positive_selectivity_q5, NDNF_negative_selectivity_q5, EC_positive_selectivity_q5, EC_negative_selectivity_q5 = get_selectivity_for_plotting_lists(
        residual_q5_SST, residual_q1_NDNF, residual_q5_EC)

    animal_ID_list_SST = get_animal_ID_list(activity_dict_SST)
    animal_ID_list_NDNF = get_animal_ID_list(activity_dict_NDNF)
    animal_ID_list_EC = get_animal_ID_list(activity_dict_EC)

    mean_quantiles_SST_q1, sem_quantiles_SST_q1 = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                             SST_positive_selectivity_q1, n_bins=20)
    mean_quantiles_SST_q5, sem_quantiles_SST_q5 = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                             SST_positive_selectivity_q5, n_bins=20)

    mean_quantiles_NDNF_q1, sem_quantiles_NDNF_q1 = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                               NDNF_positive_selectivity_q1, n_bins=20)
    mean_quantiles_NDNF_q5, sem_quantiles_NDNF_q5 = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                               NDNF_positive_selectivity_q5, n_bins=20)

    mean_quantiles_EC_q1, sem_quantiles_EC_q1 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           EC_positive_selectivity_q1, n_bins=20)
    mean_quantiles_EC_q5, sem_quantiles_EC_q5 = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                           EC_positive_selectivity_q5, n_bins=20)

    mean_quantiles_SST_q1_negative, sem_quantiles_SST_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                               SST_negative_selectivity_q1,
                                                                                               n_bins=20)
    mean_quantiles_SST_q5_negative, sem_quantiles_SST_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_SST,
                                                                                               SST_negative_selectivity_q5,
                                                                                               n_bins=20)

    mean_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                                 NDNF_negative_selectivity_q1,
                                                                                                 n_bins=20)
    mean_quantiles_NDNF_q5_negative, sem_quantiles_NDNF_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF,
                                                                                                 NDNF_negative_selectivity_q5,
                                                                                                 n_bins=20)

    mean_quantiles_EC_q1_negative, sem_quantiles_EC_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                             EC_negative_selectivity_q1,
                                                                                             n_bins=20)
    mean_quantiles_EC_q5_negative, sem_quantiles_EC_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_EC,
                                                                                             EC_negative_selectivity_q5,
                                                                                             n_bins=20)

    positive_mean_list = [mean_quantiles_SST_q1, mean_quantiles_SST_q5, mean_quantiles_NDNF_q1, mean_quantiles_NDNF_q5,
                          mean_quantiles_EC_q1, mean_quantiles_EC_q5]
    positive_sem_list = [sem_quantiles_SST_q1, sem_quantiles_SST_q5, sem_quantiles_NDNF_q1, sem_quantiles_NDNF_q5,
                         sem_quantiles_EC_q1, sem_quantiles_EC_q5]

    negative_mean_list = [mean_quantiles_SST_q1_negative, mean_quantiles_SST_q5_negative,
                          mean_quantiles_NDNF_q1_negative, mean_quantiles_NDNF_q5_negative,
                          mean_quantiles_EC_q1_negative, mean_quantiles_EC_q5_negative]
    negative_sem_list = [sem_quantiles_SST_q1_negative, sem_quantiles_SST_q5_negative, sem_quantiles_NDNF_q1_negative,
                         sem_quantiles_NDNF_q5_negative, sem_quantiles_EC_q1_negative, sem_quantiles_EC_q5_negative]

    plot_cdf_split_learning(positive_mean_list, positive_sem_list, title="Positive Selectivity", x_title="Selectivity",
                            n_bins=20)
    plot_cdf_split_learning(negative_mean_list, negative_sem_list, title="Negative Selectivity", x_title="Selectivity",
                            n_bins=20)


def get_argmin_argmax_split_learning(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                                     predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):
    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(
        activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q5_NDNF = split_activity_and_prediction_into_quintiles(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, prediction_list_EC_q1, prediction_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(
        activity_dict_EC, predicted_activity_dict_EC)

    # trial_av_activity_SST_q1 = trial_average(activity_list_SST_q1)
    # trial_av_activity_NDNF_q1 = trial_average(activity_list_NDNF_q1)
    # trial_av_activity_EC_q1 = trial_average(activity_list_EC_q1)
    #
    # trial_av_activity_SST_q5 = trial_average(activity_list_SST_q5)
    # trial_av_activity_NDNF_q5 = trial_average(activity_list_NDNF_q5)
    # trial_av_activity_EC_q5 = trial_average(activity_list_EC_q5)

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

    argmin_mean_list = [mean_quantiles_SST_q1_argmin, mean_quantiles_SST_q5_argmin,
                        mean_quantiles_NDNF_q1_argmin, mean_quantiles_NDNF_q5_argmin,
                        mean_quantiles_EC_q1_argmin, mean_quantiles_EC_q5_argmin]
    argmin_sem_list = [sem_quantiles_SST_q1_argmin, sem_quantiles_SST_q5_argmin, sem_quantiles_NDNF_q1_argmin,
                       sem_quantiles_NDNF_q5_argmin, sem_quantiles_EC_q1_argmin, sem_quantiles_EC_q5_argmin]

    plot_cdf_split_learning(argmax_mean_list, argmax_sem_list, title="Argmax", x_title="Argmax",
                            n_bins=20)
    plot_cdf_split_learning(argmin_mean_list, argmin_sem_list, title="Argmin", x_title="Argmin",
                            n_bins=20)


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


def setup_argmin_argmax_cdf_plotting_and_plot(activity_dict_SST, predicted_activity_dict_SST,
                                              activity_dict_NDNF, predicted_activity_dict_NDNF,
                                              activity_dict_EC, predicted_activity_dict_EC,
                                              residual=False):
    SST_argmax_list, NDNF_argmax_list, EC_argmax_list = split_argmin_argmax(activity_dict_SST,
                                                                            predicted_activity_dict_SST,
                                                                            activity_dict_NDNF,
                                                                            predicted_activity_dict_NDNF,
                                                                            activity_dict_EC,
                                                                            predicted_activity_dict_EC,
                                                                            residual=residual, which_to_plot="argmax")
    SST_argmin_list, NDNF_argmin_list, EC_argmin_list = split_argmin_argmax(activity_dict_SST,
                                                                            predicted_activity_dict_SST,
                                                                            activity_dict_NDNF,
                                                                            predicted_activity_dict_NDNF,
                                                                            activity_dict_EC,
                                                                            predicted_activity_dict_EC,
                                                                            residual=residual, which_to_plot="argmin")

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

    if residual:
        plot_cdf(mean_quantiles_list_argmax, sem_quantiles_list_argmax, title="Argmax Residual",
                 x_title="Bin of Peak Firing", n_bins=20)
        plot_cdf(mean_quantiles_list_argmin, sem_quantiles_list_argmin, title="Argmin Residual",
                 x_title="Bin of Minimum Firing", n_bins=20)

    else:
        plot_cdf(mean_quantiles_list_argmax, sem_quantiles_list_argmax, title="Argmax Raw Data",
                 x_title="Bin of Peak Firing", n_bins=20)
        plot_cdf(mean_quantiles_list_argmin, sem_quantiles_list_argmin, title="Argmin Raw Data",
                 x_title="Bin of Minimum Firing", n_bins=20)

#
def plot_cdf(mean_quantiles_list, sem_quantiles_list, title=None, x_title=None, n_bins=None):
    bin_centers = np.arange(1, n_bins + 1)

    plt.figure(figsize=(10, 6))

    plt.errorbar(mean_quantiles_list[0], bin_centers, xerr=sem_quantiles_list[0], fmt='o-', color='blue', ecolor='blue',
                 capsize=8, label="SST")
    plt.errorbar(mean_quantiles_list[1], bin_centers, xerr=sem_quantiles_list[1], fmt='o-', color='orange',
                 ecolor='orange', capsize=8, label="NDNF")
    plt.errorbar(mean_quantiles_list[2], bin_centers, xerr=sem_quantiles_list[2], fmt='o-', color='green',
                 ecolor='green', capsize=8, label="EC")

    plt.ylabel("Percentile of Data")
    plt.yticks(ticks=bin_centers, labels=[f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)])
    plt.xlabel(f" Mean {x_title}")
    plt.title(title)
    plt.legend()

    plt.show()


def compute_r_and_model(x, y):

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    r_value, _ = pearsonr(x, y)
    return r_value, y_pred


def plot_pop_correlation(r2_variable_activity_dict, r2_variable_residual_dict, variable_to_correlate="Velocity", cell_type="SST", color="blue",dictionary=True):

    if dictionary:
        r2_variable_activity_list = []
        r2_variable_residual_list = []

        for animal in r2_variable_activity_dict:
            for neuron in r2_variable_activity_dict[animal]:
                r2_variable_activity_list.append(r2_variable_activity_dict[animal][neuron])
                r2_variable_residual_list.append(r2_variable_residual_dict[animal][neuron])

    else:
        r2_variable_activity_list = r2_variable_activity_dict
        r2_variable_residual_list = r2_variable_residual_dict


    r2_variable_activity_array = np.array(r2_variable_activity_list)
    r2_variable_residual_array = np.array(r2_variable_residual_list)

    mean_activity = np.mean(r2_variable_activity_array)
    sem_activity = np.std(r2_variable_activity_array) / np.sqrt(len(r2_variable_activity_array))

    mean_residuals = np.mean(r2_variable_residual_array)
    sem_residuals = np.std(r2_variable_residual_array) / np.sqrt(len(r2_variable_residual_array))

    x_positions = [0, 1]
    x_jitter_activity = np.random.normal(x_positions[0], 0.05, size=len(r2_variable_activity_array))
    x_jitter_residuals = np.random.normal(x_positions[1], 0.05, size=len(r2_variable_residual_array))

    plt.figure(figsize=(8, 6))

    # Plot individual data points
    plt.scatter(x_jitter_activity, r2_variable_activity_array, color='black', alpha=0.8, label=f'Activity vs {variable_to_correlate}', zorder=3)
    plt.scatter(x_jitter_residuals, r2_variable_residual_array, color=color, alpha=0.8, label=f'Residuals vs {variable_to_correlate}', zorder=3)

    # Connect corresponding points with grey lines
    for i in range(len(r2_variable_activity_array)):
        plt.plot([x_jitter_activity[i], x_jitter_residuals[i]],
                 [r2_variable_activity_array[i], r2_variable_residual_array[i]],
                 color='gray', alpha=0.6, linewidth=0.8)

    # Plot mean and SEM for activity as horizontal lines
    plt.hlines(mean_activity, x_positions[0] - 0.3, x_positions[0] - 0.1, color='black', linewidth=2, label='Mean (Activity)')
    plt.fill_betweenx([mean_activity - sem_activity, mean_activity + sem_activity],
                      x_positions[0] - 0.3, x_positions[0] - 0.1, color='black', alpha=0.2)

    # Plot mean and SEM for residuals as horizontal lines
    plt.hlines(mean_residuals, x_positions[1] + 0.2, x_positions[1] + 0.3, color=color, linewidth=2, label='Mean (Residuals)')
    plt.fill_betweenx([mean_residuals - sem_residuals, mean_residuals + sem_residuals],
                      x_positions[1] + 0.2, x_positions[1] + 0.3, color=color, alpha=0.2)

    # Set plot aesthetics
    plt.xticks(x_positions, [f'Activity vs {variable_to_correlate}', f'Residuals vs {variable_to_correlate}'])
    plt.ylabel("R Value")
    plt.title(f"R Values for Activity vs {variable_to_correlate} and Residuals vs {variable_to_correlate} for {cell_type}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Show the plot
    plt.show()



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


def plot_temporal_correlation(filepaths, output_to_plot='r', data_to_plot='all'):
    overall_r_per_dataset = {}
    per_animal_r_per_dataset = {}

    for label, filepath in filepaths.items():
        if data_to_plot == 'all' or data_to_plot.lower() == label.lower():
            reorganized_data, _ = preprocess_data(filepath)
            r_all_neurons, r_per_animal, _ = compute_temporal_correlation(reorganized_data)

            if output_to_plot.lower() == 'r2':
                r_all_neurons = np.array(r_all_neurons) ** 2
                r_per_animal = np.array(r_per_animal) ** 2

            overall_r_per_dataset[label] = r_all_neurons
            per_animal_r_per_dataset[label] = r_per_animal

    plot_results(overall_r_per_dataset, per_animal_r_per_dataset, output_to_plot)


def plot_results(overall_r_per_dataset, per_animal_r_per_dataset, output_to_plot):
    dataset_labels = list(overall_r_per_dataset.keys())
    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(dataset_labels))
    jitter_strength = 0.05

    for i, label in enumerate(dataset_labels):
        all_neurons_r = overall_r_per_dataset[label]
        per_animal_r = per_animal_r_per_dataset[label]

        jitter_neuron = np.random.normal(0, jitter_strength, size=len(all_neurons_r))
        jitter_animal = np.random.normal(0, jitter_strength, size=len(per_animal_r))

        plt.scatter(np.full(len(all_neurons_r), i) + jitter_neuron, all_neurons_r, color='grey', alpha=0.5,
                    label='Neurons' if i == 0 else "")

        plt.scatter(np.full(len(per_animal_r), i) + jitter_animal, per_animal_r, color='k',
                    label='Per Animal' if i == 0 else "")

        mean_r = np.mean(all_neurons_r)
        error = sem(all_neurons_r)
        plt.errorbar(i, mean_r, yerr=error, fmt='o', color='r', label='Overall' if i == 0 else "")

    plt.xticks(x_positions, dataset_labels)
    plt.ylabel(f'{output_to_plot.upper()} value')
    plt.title(f'Average {output_to_plot.upper()} for NDNF, EC, and SST Neurons with SEM')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_activity_residuals_correlation(reorganized_data, predicted_activity_list, neuron_activity_list, residuals_list,
                                        cell_number, variable_to_corelate=["Velocity"]):
    velocity_list = []
    for key, value in reorganized_data.items():
        for key2, value2 in value.items():
            velocity = value2["Velocity"]
            velocity_list.append(velocity)

def plot_correlations_overview(
        GLM_params, factors_dict, predicted_activity_dict, activity_dict, cell_number,
        variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"], include_intercept=True
):
    neuron_activity_list = []

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            neuron_activity_list.append(activity_dict[animal][neuron])

    flat_neuron_activity = neuron_activity_list[cell_number].flatten()

    combined_predictions, input_variables, variable_predictions = get_predictions_and_input_variables(
        GLM_params, factors_dict, include_intercept
    )

    flat_prediction_total = combined_predictions[cell_number].flatten()
    flat_residual_total = flat_neuron_activity - flat_prediction_total

    model_total = LinearRegression()
    model_total.fit(flat_prediction_total.reshape(-1, 1), flat_neuron_activity)
    y_pred_total = model_total.predict(flat_prediction_total.reshape(-1, 1))
    r2_total, _ = pearsonr(flat_neuron_activity, y_pred_total)

    model_total_residual = LinearRegression()
    model_total_residual.fit(flat_residual_total.reshape(-1, 1), flat_neuron_activity)
    y_pred_total_residual = model_total_residual.predict(flat_residual_total.reshape(-1, 1))
    r2_residual, _ = pearsonr(flat_neuron_activity, y_pred_total_residual)

    variable_vs_activity_correlation_dict = {key: [] for key in variable_to_correlate_list}
    prediction_vs_activity_correlation_dict = {key: [] for key in variable_to_correlate_list}
    residuals_vs_activity_correlation_dict = {key: [] for key in variable_to_correlate_list}

    residuals_vs_variable_correlation_dict = {key: [] for key in variable_to_correlate_list}

    num_keys = len(variable_to_correlate_list)
    fig, axs = plt.subplots(4, num_keys, figsize=(6 * (num_keys), 15))

    for idx, key in enumerate(variable_to_correlate_list):
        flat_neuron_activity = neuron_activity_list[cell_number].flatten()
        flat_variable = input_variables[key][cell_number].flatten()
        flat_predicted_activity = variable_predictions[key][cell_number].flatten()
        flat_residual = flat_neuron_activity - flat_predicted_activity

        model_activity = LinearRegression()
        model_activity.fit(flat_variable.reshape(-1, 1), flat_neuron_activity)
        y_pred_activity = model_activity.predict(flat_variable.reshape(-1, 1))
        r2_activity, _ = pearsonr(flat_neuron_activity, y_pred_activity)

        axs[0, idx].scatter(flat_variable, flat_neuron_activity, label="Data", alpha=0.6)
        axs[0, idx].plot(flat_variable, y_pred_activity, color='r', label="Best Fit", linewidth=2)
        axs[0, idx].set_title(f"{key} vs Activity\nR value: {r2_activity:.3f}")
        axs[0, idx].set_xlabel(key)
        axs[0, idx].set_ylabel("Activity")
        axs[0, idx].legend()

        model_pred = LinearRegression()
        model_pred.fit(flat_predicted_activity.reshape(-1, 1), flat_neuron_activity)
        y_pred_prediction = model_pred.predict(flat_predicted_activity.reshape(-1, 1))
        r2_prediction, _ = pearsonr(flat_neuron_activity, flat_predicted_activity)

        axs[1, idx].scatter(flat_predicted_activity, flat_neuron_activity, label="Data", alpha=0.6)
        axs[1, idx].plot(flat_predicted_activity, y_pred_prediction, color='r', label="Best Fit", linewidth=2)
        axs[1, idx].set_title(f"Prediction ({key}) vs Activity\nR value: {r2_prediction:.3f}")
        axs[1, idx].set_xlabel(f"Prediction ({key})")
        axs[1, idx].set_ylabel("Activity")
        axs[1, idx].legend()

        ##### residuals vs the input variable

        model_residual_variable = LinearRegression()
        model_residual_variable.fit(flat_residual.reshape(-1, 1), flat_variable)
        y_pred_residual_variable = model_residual_variable.predict(flat_residual.reshape(-1, 1))
        r2_residual_variable, _ = pearsonr(flat_residual, flat_variable)

        axs[2, idx].scatter(flat_residual, flat_variable, label="Data", alpha=0.6)
        axs[2, idx].plot(flat_residual, y_pred_residual_variable, color='r', label="Best Fit", linewidth=2)
        axs[2, idx].set_title(f"Residuals ({key}) vs {key}\nR value: {r2_residual_variable:.3f}")
        axs[2, idx].set_xlabel("Residuals")
        axs[2, idx].set_ylabel(f"{key} variable")
        axs[2, idx].legend()

        #### residuals vs the actual activity

        model_resid = LinearRegression()
        model_resid.fit(flat_residual.reshape(-1, 1), flat_neuron_activity)
        y_pred_residual = model_resid.predict(flat_residual.reshape(-1, 1))
        r2_residual, _ = pearsonr(flat_neuron_activity, flat_residual)

        axs[3, idx].scatter(flat_residual, flat_neuron_activity, label="Data", alpha=0.6)
        axs[3, idx].plot(flat_residual, y_pred_residual, color='r', label="Best Fit", linewidth=2)
        axs[3, idx].set_title(f"Residuals ({key}) vs Activity\nR value: {r2_residual:.3f}")
        axs[3, idx].set_xlabel("Residuals")
        axs[3, idx].set_ylabel("Activity")
        axs[3, idx].legend()

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(flat_prediction_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[0].plot(flat_prediction_total, y_pred_total, color='r', label="Best Fit", linewidth=2)
    axs[0].set_title(f"Combined Prediction vs Activity\nR value: {r2_total:.3f}")
    axs[0].set_xlabel("Combined Prediction")
    axs[0].set_ylabel("Activity")
    axs[0].legend()

    axs[1].scatter(flat_residual_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[1].plot(flat_residual_total, y_pred_total_residual, color='r', label="Best Fit", linewidth=2)
    axs[1].set_title(f"Residuals vs Activity\nR value: {r2_residual:.3f}")
    axs[1].set_xlabel("Residuals")
    axs[1].set_ylabel("Activity")
    axs[1].legend()

    fig.show()

    return variable_vs_activity_correlation_dict, prediction_vs_activity_correlation_dict, residuals_vs_activity_correlation_dict


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
def plot_activity_residuals_correlation(factors_dict, predicted_activity_list, neuron_activity_list, residuals_list,
                                        cell_number, variable_to_corelate="Velocity"):
    variable_data_list = []
    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            variable_data_list.append(factors_dict[animal][variable_to_corelate])
            print(f"factors_dict[key][variable_to_corelate].shape {factors_dict[animal][variable_to_corelate].shape}")

    print(f"len(variable_data_list) {len(variable_data_list)}")

    trial_average_variable_data_list = []
    for i in variable_data_list:
        trial_average_variable_data_list.append(np.mean(i, axis=1))

    print(f"len(trial_average_variable_data_list) {len(trial_average_variable_data_list)}")

    trial_av_prediction_list = []
    for i in predicted_activity_list:
        trial_av_prediction_list.append(np.mean(i, axis=1))

    trial_av_neuron_activity_list = []
    for i in neuron_activity_list:
        trial_av_neuron_activity_list.append(np.mean(i, axis=1))

    trial_av_residuals_list = []
    for i in residuals_list:
        trial_av_residuals_list.append(np.mean(i, axis=1))

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].plot(trial_av_neuron_activity_list[cell_number], color='k', label='neuron activity')
    axs[0, 0].set_title(f"Firing Rate for Cell#{cell_number}", fontsize=6)
    axs[0, 0].set_ylim(-1.5, 1)

    axs[0, 1].plot(trial_average_variable_data_list[cell_number], color='g')
    axs[0, 1].set_title(f"Run Velocity for Animal, Cell#{cell_number}", fontsize=6)
    axs[0, 1].set_ylim(-1.5, 1)

    axs[1, 0].plot(trial_av_residuals_list[cell_number], color='r', label='residuals')
    axs[1, 0].set_title(f"Residuals: Firing Rate - Velocity-Only Prediction of FR for Cell#{cell_number}", fontsize=6)
    axs[1, 0].set_ylim(-1.5, 1)

    axs[1, 1].plot(trial_av_prediction_list[cell_number], color='b', label='velocity prediction')
    axs[1, 1].set_title(f"Prediction of Firing Based on Velocity for Cell#{cell_number}", fontsize=6)
    axs[1, 1].set_ylim(-1.5, 1)

    r2_list_residuals = []
    r2_list_activity = []
    y_pred_activity_list = []
    y_pred_residuals_list = []

    for i in range(len(variable_data_list)):
        velocity_flat = variable_data_list[i].flatten()
        activity_flat = neuron_activity_list[i].flatten()
        residuals_flat = residuals_list[i].flatten()

        model_activity = LinearRegression()
        model_activity.fit(velocity_flat.reshape(-1, 1), activity_flat)
        y_pred_activity = model_activity.predict(velocity_flat.reshape(-1, 1))
        r2_activity = r2_score(activity_flat, y_pred_activity)
        r2_list_activity.append(r2_activity)
        y_pred_activity_list.append(y_pred_activity)

        model_residuals = LinearRegression()
        model_residuals.fit(velocity_flat.reshape(-1, 1), residuals_flat)
        y_pred_residuals = model_residuals.predict(velocity_flat.reshape(-1, 1))
        r2_residuals = r2_score(residuals_flat, y_pred_residuals)
        r2_list_residuals.append(r2_residuals)
        y_pred_residuals_list.append(y_pred_residuals)

    velocity_flat = variable_data_list[cell_number].flatten()
    activity_flat = neuron_activity_list[cell_number].flatten()
    residuals_flat = residuals_list[cell_number].flatten()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(velocity_flat, activity_flat, color="g", s=10, alpha=0.3, label="Data")
    axs[0].plot(
        velocity_flat, y_pred_activity_list[cell_number], color="r", label=f"R² = {r2_list_activity[cell_number]:.3f}"
    )
    axs[0].set_title(f"Activity vs Velocity (Cell #{cell_number})")
    axs[0].set_xlabel("Velocity")
    axs[0].set_ylabel("Activity")
    axs[0].legend()

    axs[1].scatter(velocity_flat, residuals_flat, color="b", s=10, alpha=0.3, label="Data")
    axs[1].plot(
        velocity_flat, y_pred_residuals_list[cell_number], color="r", label=f"R² = {r2_list_residuals[cell_number]:.3f}"
    )
    axs[1].set_title(f"Residuals vs Velocity (Cell #{cell_number})")
    axs[1].set_xlabel("Velocity")
    axs[1].set_ylabel("Residuals")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return r2_list_residuals, r2_list_activity


def plot_r_difference_vs_variable(r2_list_activity, r2_list_residuals, variable_to_compare="Velocity"):
    r2_list_activity = np.array(r2_list_activity)
    r2_list_residuals = np.array(r2_list_residuals)

    mean_activity = np.mean(r2_list_activity)
    mean_residuals = np.mean(r2_list_residuals)
    sem_activity = sem(r2_list_activity)
    sem_residuals = sem(r2_list_residuals)

    positions = [0, 1]

    plt.figure(figsize=(8, 6))

    # Scatter points
    x_jitter_activity = np.random.normal(positions[0], 0.05, size=len(r2_list_activity))
    x_jitter_residuals = np.random.normal(positions[1], 0.05, size=len(r2_list_residuals))
    plt.scatter(x_jitter_activity, r2_list_activity, color='black', alpha=0.8, label='R² Activity', zorder=3)
    plt.scatter(x_jitter_residuals, r2_list_residuals, color='red', alpha=0.8, label='R² Residuals', zorder=3)

    # Connecting lines
    for i in range(len(r2_list_activity)):
        plt.plot([x_jitter_activity[i], x_jitter_residuals[i]],
                 [r2_list_activity[i], r2_list_residuals[i]],
                 color='gray', alpha=0.6, linewidth=0.8)

    # Mean and SEM lines for Activity
    plt.hlines(mean_activity, positions[0] - 0.2, positions[0] + 0.2, color='black', linewidth=2, zorder=4)
    plt.vlines(positions[0], mean_activity - sem_activity, mean_activity + sem_activity, color='black', linewidth=2)

    # Mean and SEM lines for Residuals
    plt.hlines(mean_residuals, positions[1] - 0.2, positions[1] + 0.2, color='red', linewidth=2, zorder=4)
    plt.vlines(positions[1], mean_residuals - sem_residuals, mean_residuals + sem_residuals, color='red', linewidth=2)

    # Formatting
    plt.xticks(positions, [f'Activity vs {variable_to_compare}', f'Residuals vs {variable_to_compare}'])
    plt.ylabel("R Value")
    plt.title(f"Activity vs {variable_to_compare} and Residuals vs {variable_to_compare}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()





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


def plot_sorted_activity(data, sorted_indices, title, ylabel, xlabel):
    plt.figure()
    plt.imshow(data[sorted_indices, :], aspect='auto')
    plt.title(title)
    plt.colorbar(label='Activity')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def plot_trial_averages(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF,
                        predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False,
                        which_to_plot="argmin"):
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

    trial_av_activity_SST_stack = np.stack(trial_av_activity_SST)
    trial_av_activity_NDNF_stack = np.stack(trial_av_activity_NDNF)
    trial_av_activity_EC_stack = np.stack(trial_av_activity_EC)

    print(f"trial_av_activity_SST_stack.shape {trial_av_activity_SST_stack.shape}")

    trial_av_activity_SST = normalize(trial_av_activity_SST_stack, norm="z_score", per_cell=True)
    trial_av_activity_NDNF = normalize(trial_av_activity_NDNF_stack, norm="z_score", per_cell=True)
    trial_av_activity_EC = normalize(trial_av_activity_EC_stack, norm="z_score", per_cell=True)

    if which_to_plot == "argmin":
        sorted_indices_SST = np.argsort(np.argmin(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmin(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmin(trial_av_activity_EC, axis=1))
    elif which_to_plot == "argmax":
        sorted_indices_SST = np.argsort(np.argmax(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmax(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmax(trial_av_activity_EC, axis=1))
    else:
        raise ValueError("Options for which_to_plot are 'argmin' or 'argmax'")

    plot_sorted_activity(trial_av_activity_SST, sorted_indices_SST,
                         f"{'Residuals' if residual else 'Raw Data'} SST ({which_to_plot})", "Cell ID", "Position Bins")

    plot_sorted_activity(
        trial_av_activity_NDNF, sorted_indices_NDNF,
        f"{'Residuals' if residual else 'Raw Data'} NDNF ({which_to_plot})", "Cell ID", "Position Bins")

    plot_sorted_activity(
        trial_av_activity_EC, sorted_indices_EC, f"{'Residuals' if residual else 'Raw Data'} EC ({which_to_plot})",
        "Cell ID", "Position Bins")

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


def plot_example_neuron(reorganized_data, GLM_params, variable_list, sort_by='R2', animal=None, cell=None, ax=None):
    if animal is None or cell is None:
        animal, cell = select_neuron(GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell)
        print(f"Best neuron: {cell}, {animal}")

    neuron_data = reorganized_data[animal][cell]
    flattened_data = flatten_data(neuron_data)

    input_variables = neuron_data[:,1:,:]
    neuron_activity = neuron_data[:,0,:]

    if ax is None:
        fig = plt.figure(figsize=(14, 4))
        ax_ = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        ax_ = ax
    ax_.axis('off')

    axes = gs.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=ax_, wspace=0., width_ratios=[0.3, 0.22, 0.48])
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    avg_variables = np.mean(input_variables, axis=2)
    def plot_example_neuron_variables(variables, variable_list, ax, fig, weights=None):
        variable_list = variable_list[1:]
        height_ratios = np.ones(variables.shape[1])
        height_ratios[-10:] = 0.5
        axes = gs.GridSpecFromSubplotSpec(nrows=variables.shape[1], ncols=1, subplot_spec=ax, hspace=0.5, height_ratios=height_ratios)
        for i in range(variables.shape[1]):
            ax = fig.add_subplot(axes[i])
            ax.plot(variables[:, i], color='k', linewidth=1)
            ax.set_ylabel(variable_list[i], rotation=0, ha='right', va='center', labelpad=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if weights is not None:
                ax.scatter([50],[0.3], c='k', s=abs(weights[i])*20)
        ax.set_xlabel('Position', labelpad=-10)
        ax.set_xticks([0,50])

        # Draw vertical line across all plots (remove axes and make the background transparent)
        ax = fig.add_subplot(axes[:])
        ax.vlines(25, 0, 1, linestyles='--', color='r', alpha=0.7)
        ax.set_xlim([0, 49])
        ax.set_ylim([0, 1])
        ax.axis('off')
        ax.patch.set_alpha(0)
    plot_example_neuron_variables(avg_variables, variable_list, ax, fig)

    ax = fig.add_subplot(axes[1])
    ax.axis('off')
    def plot_weight_lines(GLM_params, animal, cell, ax):
        weights = GLM_params[animal][cell]['weights']
        if len(weights) > 13:
            line_spacing = -np.ones(len(weights)) * 1.4
        else:
            line_spacing = -np.ones(len(weights)) * 1.45
        line_spacing[-10:] *= 0.615
        y = np.cumsum(line_spacing) - 0.4/line_spacing[0]
        w_max = np.max(np.abs(weights))
        for i,w in enumerate(weights):
            if abs(w)<0.05:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='lightgray', linestyle='--', linewidth=1)
            elif w < 0:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='deepskyblue', linewidth=abs(w/w_max)*4)
            else:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='black', linewidth=abs(w/w_max)*4)
            line.set_solid_capstyle('round')
        ax.set_ylim([-len(weights),0])

        ax.plot([0,0], [0,0], color='lightgray', linewidth=1.5, linestyle='--', label='Small weights')
        ax.plot([0,0], [0,0], color='deepskyblue', linewidth=1.5, label='Negative weights')
        ax.plot([0,0], [0,0], color='black', linewidth=1.5, label='Positive weights')
        ax.legend(fontsize=10, loc='upper right', frameon=False, handlelength=1.5, handletextpad=0.5, labelspacing=0.2, borderpad=0)
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        ax.text(1, -13, f'Max weight: {max_weight:.2f}', ha='right', va='bottom', fontsize=10)
        ax.text(1, -12, f'Min weight: {min_weight:.2f}', ha='right', va='bottom', fontsize=10)
    plot_weight_lines(GLM_params, animal, cell, ax)

    # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][cell]['model']
    flattened_input_variables = flattened_data[:,1:]
    predicted_activity = glm_model.predict(flattened_input_variables)

    pearson_R = np.corrcoef(predicted_activity, flattened_data[:,0])[0,1]
    print("pearson R2 across all trials:", pearson_R**2)

    predicted_activity = predicted_activity.reshape(neuron_activity.shape)
    avg_predicted_activity = np.mean(predicted_activity, axis=1)
    std_predicted_activity = np.std(predicted_activity, axis=1)
    sem_predicted_activity = std_predicted_activity / np.sqrt(predicted_activity.shape[1])
    avg_neuron_activity = np.mean(neuron_activity, axis=1)
    std_neuron_activity = np.std(neuron_activity, axis=1)
    sem_neuron_activity = std_neuron_activity / np.sqrt(neuron_activity.shape[1])

    pearson_R = np.corrcoef(avg_predicted_activity, avg_neuron_activity)[0,1]
    print("pearson R2 (average prediction vs average activity):", pearson_R**2)

    axes = gs.GridSpecFromSubplotSpec(nrows=3, ncols=3, subplot_spec=ax_, wspace=0., width_ratios=[0.3, 0.3, 0.4], height_ratios=[0.2,1,0.2])
    ax = fig.add_subplot(axes[1,2])
    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-sem_neuron_activity, avg_neuron_activity+sem_neuron_activity, alpha=0.1, color='k')
    ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-sem_predicted_activity, avg_predicted_activity+sem_predicted_activity, alpha=0.1, color='gray')
    ax.set_xlabel("Position")
    ax.set_ylabel("dF/F activity (Z-scored)")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))


def plot_GLM_summary_data(GLM_params, variable_list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    jitter = 0.5
    animal_xoffset = 0.2

    params_all_animals = []
    for animal_id in GLM_params:
        params_all_neurons = []
        for neuron_id in GLM_params[animal_id]:
            weights = GLM_params[animal_id][neuron_id]['weights']
            intercept = GLM_params[animal_id][neuron_id]['intercept']
            all_params = np.concatenate([weights, [intercept]])
            params_all_neurons.append(all_params)
            jittered_x = np.arange(len(all_params)) + np.random.uniform(0.3, jitter, len(all_params))
            ax.scatter(jittered_x, all_params, color='grey', alpha=0.2, s=10)

        params_all_neurons = np.array(params_all_neurons)
        mean_params = np.mean(params_all_neurons, axis=0)
        params_all_animals.append(mean_params)
        ax.scatter(np.arange(len(mean_params))+animal_xoffset, mean_params, color='black', label=f'Animal {animal_id}', s=20)

    params_all_animals = np.array(params_all_animals)
    global_mean = np.mean(params_all_animals, axis=0)
    global_std = np.std(params_all_animals, axis=0)
    ax.errorbar(np.arange(len(global_mean)), global_mean, yerr=global_std, fmt='o', color='red', ecolor='red', capsize=5, label='Average of all animals', markersize=7)

    ax.set_xticks(np.arange(len(global_mean)))
    xtick_labels = variable_list[1:] + ['Intercept']
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel('Weights')

    ax.hlines(0, -0.5, len(variable_list) - 0.5, linestyles='--', color='black', alpha=0.5)
    ax.set_xlim([-0.5, len(global_mean) - 0.4])


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


def plot_quintile_comparison(reorganized_data, variable_list, filename, quintiles=(1,5), save=False):
    avg_residuals_ls = []
    GLM_params_ls = []
    for quintile in quintiles:
        avg_residuals, GLM_params = compute_velocity_subtracted_residuals(reorganized_data, variable_list, quintile)
        avg_residuals_ls.append(avg_residuals)
        GLM_params_ls.append(GLM_params)

    sorting_idx = np.argsort(np.argmax(avg_residuals_ls[1], axis=1))
    avg_residuals_ls = [avg_residuals[sorting_idx] for avg_residuals in avg_residuals_ls]

    fig = plt.figure(figsize=(14, 14))
    axes = gs.GridSpec(nrows=3, ncols=3, hspace=0.2, wspace=0.4, height_ratios=[3,1,2])
    fig.suptitle(filename, fontsize=15, y=0.93)

    vmax = np.max([np.abs(avg_residuals_ls[0]), np.abs(avg_residuals_ls[1])])
    plot_scale = np.max([np.abs(np.mean(avg_residuals_ls[0], axis=0))+np.std(avg_residuals_ls[0], axis=0), np.abs(np.mean(avg_residuals_ls[1], axis=0))+np.std(avg_residuals_ls[1], axis=0)])

    for col, quintile_data in enumerate(avg_residuals_ls):
        # Plot heatmap
        ax = fig.add_subplot(axes[0,col])
        im = ax.imshow(quintile_data, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
        x,y,w,h = ax.get_position().bounds
        cax = fig.add_axes([x+w*1.02, y, w*0.05, h])
        cbar = plt.colorbar(im, cax=cax)
        ax.set_xlim([0, 50])
        if quintiles[col] == 1:
            ax.set_title(f'First 20% of trials')
        elif quintiles[col] == 5:
            ax.set_title(f'Last 20% of trials')
        else:
            ax.set_title(f'Quintile {quintiles[col]}')
        if col == 0:
            ax.set_ylabel('Neuron #')

        # Plot average activity
        ax = fig.add_subplot(axes[1,col])
        ax.plot(np.mean(quintile_data, axis=0), c='k', lw=2)
        std = np.std(quintile_data, axis=0)
        sem = std / np.sqrt(quintile_data.shape[0])
        ax.fill_between(np.arange(50), np.mean(quintile_data, axis=0)-std, np.mean(quintile_data, axis=0)+std, color='gray', alpha=0.2)
        ax.hlines(0, 0, 50, color='gray', linestyle='--')
        ax.set_xlim([0, 50])
        if col == 0:
            ax.set_ylabel('dF/F residual activity \n(Z-scored)')
        ax.set_xlabel('Position')
        ax.set_ylim([-plot_scale, plot_scale])

    # Plot delta
    delta_residuals = avg_residuals_ls[1] - avg_residuals_ls[0]
    ax = fig.add_subplot(axes[0,2])
    vmax = np.max(np.abs(delta_residuals))
    im = ax.imshow(delta_residuals, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    x,y,w,h = ax.get_position().bounds
    cax = fig.add_axes([x+w*1.02, y, w*0.05, h])
    cbar = fig.colorbar(im, cax=cax)
    ax.set_xlim([0, 50])
    ax.set_title('Difference')

    ax = fig.add_subplot(axes[1,2])
    ax.plot(np.mean(delta_residuals, axis=0), c='k', lw=2)
    std = np.std(delta_residuals, axis=0)
    sem = std / np.sqrt(delta_residuals.shape[0])
    ax.fill_between(np.arange(50), np.mean(delta_residuals, axis=0)-std, np.mean(delta_residuals, axis=0)+std, color='gray', alpha=0.2)
    ax.set_xlim([0, 50])
    ax.set_xlabel('Position')
    ax.hlines(0, 0, 50, color='gray', linestyle='--')

    ax = fig.add_subplot(axes[2,:])
    delta_weights = calculate_delta_weights(GLM_params_ls[0], GLM_params_ls[1])
    plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=ax)

    if save:
        fig.savefig(f'figures/{filename.split(".")[0]}_residuals.png', dpi=300)
        fig.savefig(f'figures/{filename.split(".")[0]}_residuals.svg', dpi=300)


def get_GLM_R2(GLM_params):
    all_R2_values = np.array([GLM_params[animal][neuron]['R2_trialavg'] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_R2_values


def get_GLM_weights(GLM_params, variable_list):
    all_weights = {}
    for i,var_name in enumerate(variable_list[1:]):
        all_weights[var_name] = np.array([GLM_params[animal][neuron]['weights'][i] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_weights


def plot_R2_distribution(GLM_params, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,5))
    else:
        fig = ax.get_figure()

    if isinstance(GLM_params, list):
        model_list = GLM_params
    else:
        model_list = [GLM_params]

    all_R2_values = {}
    for i, model_params in enumerate(model_list, start=1):
        all_R2_values[i] = []
        animal_avg_R2_values = []
        for animal in model_params:
            animal_R2_values = []
            for neuron in model_params[animal]:
                all_R2_values[i].append(model_params[animal][neuron]['R2_trialavg'])
                animal_R2_values.append(model_params[animal][neuron]['R2_trialavg'])
            animal_avg_R2_values.append(np.mean(animal_R2_values))
        all_R2_values[i] = np.array(all_R2_values[i])

        jitter = 0.2
        jittered_x = i*np.ones(all_R2_values[i].shape) + np.random.uniform(0.1, jitter, all_R2_values[i].shape)
        ax.scatter(jittered_x, all_R2_values[i], color='grey', alpha=0.2, s=10)
        ax.scatter(i*np.ones(len(animal_avg_R2_values)), animal_avg_R2_values, color='black', label='Average R2 value', s=20)
        ax.errorbar(i*0.9, np.mean(animal_avg_R2_values), yerr=np.std(animal_avg_R2_values), fmt='o', color='red', ecolor='red',
                    capsize=5, label='Average of all animals', markersize=7)

    ax.set_ylabel("R² value")
    ax.set_xlim([0.8,2*i])
    ax.set_ylim([0,1])
    if len(model_list) > 1:
        # ax.set_xticks([0.8, i+0.2])
        # ax.set_xticklabels(['First quintile', 'Last quintile'])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['All data', 'No licks', 'No Reward loc'])
        ax.set_xlim([0.8,3.2])
        if title is not None:
            fig.suptitle(title)
            fig.savefig(f"figures/R2_distribution_{title}.png", bbox_inches='tight')
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    # Statistical test
    if len(model_list) == 2:
        t, p = stats.ttest_ind(all_R2_values[1], all_R2_values[2])
        if p < 0.001:
            ax.text(0.1, 0.2, f"p = {p:.2e}", transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.1, 0.2, f"p = {p:.3f}", transform=ax.transAxes, fontsize=12)


def plot_combined_figure(reorganized_data, GLM_params, variable_list, sort_by='R2', animal=None, cell=None, model_name=None, save=False):
    animal, cell = select_neuron(GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell)

    fig = plt.figure(figsize=(10,6))
    axes = gs.GridSpec(nrows=1, ncols=1, top=1, bottom=0.5, left=0, right=1)
    ax = fig.add_subplot(axes[0])
    plot_example_neuron(reorganized_data, GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell, ax=ax)

    # Plot R2 distribution
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.4, bottom=0, left=0., right=0.2)
    ax = fig.add_subplot(axes[0])
    plot_R2_distribution(GLM_params, ax=ax)

    # Plot summary data
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.4, bottom=0, left=0.3, right=1)
    ax = fig.add_subplot(axes[0])
    plot_GLM_summary_data(GLM_params, variable_list, ax=ax)

    if save and model_name is not None:
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{cell}.png", bbox_inches='tight', dpi=300)
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{cell}.svg", bbox_inches='tight', dpi=300)


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


def plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    plot_GLM_summary_data(delta_weights, variable_list, ax=ax)
    ax.set_ylabel('Δ Weights\n(Last - First Quintile)')

    if model_name is not None:
        ax.set_title(model_name)

    if model_name is not None and save:
        fig.savefig(f"figures/{model_name}_delta_weights.png", dpi=300)




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