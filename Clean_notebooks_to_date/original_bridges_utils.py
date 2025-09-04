################ og bridges utils 

import matplotlib.pyplot as plt
import os
import torch
import slicetca
import mat73
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
import h5py

# import utils as ut
# import plot as pt
plt.rcParams.update({'font.size': 12,
                     'axes.spines.right': False,
                     'axes.spines.top':   False,
                     'legend.frameon':    False,})

import sys
from scipy.stats import sem
sys.path.append('/Users/michaelfinch/CA1-interneuron-GLM')

# from utils_TCA_clustering_scratchpad import *
# from GLM_regression_plotting import *


# from modelling_to_date_utils import *
# from SliceTCA_example import *


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

def preprocess_data2(filepath, normalize=True, new_NDNF=False):
    factors_dict = {}
    activity_dict = {}

    if new_NDNF:
        with h5py.File(filepath, 'r') as f:
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

def load_data_regular(file_path=r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM", name="NDNFanalC", new_NDNF=True):
    file_path = file_path
    filename = name
    filepath = os.path.join(file_path, "datasets", filename + ".mat")

    activity_dict, factors_dict = preprocess_data2(filepath, normalize=True, new_NDNF=new_NDNF)

    filtered_factors_dict = subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

    GLM_params, double_predicted_activity_dict_NDNF_new = fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
    double_residual_activity_dict_NDNF_new = get_residual_activity_dict(activity_dict, double_predicted_activity_dict_NDNF_new)

    return GLM_params, activity_dict, double_predicted_activity_dict_NDNF_new, factors_dict, filtered_factors_dict, double_residual_activity_dict_NDNF_new

def add_vel_contribution_to_residuals(scaled_data_Hz_dict, GLM_params, animal_velocity_dict):
    animal_dict={}
    for animal in scaled_data_Hz_dict:
        cell_dict = {}
        for cell in scaled_data_Hz_dict[animal]:
            animal_velocity = animal_velocity_dict[animal]
            data = scaled_data_Hz_dict[animal][cell]
            weights = GLM_params[animal][cell]['weights']["Velocity"]
            intercept = GLM_params[animal][cell]['intercept']
            data = data + (weights * animal_velocity) + intercept
            cell_dict[cell] = data
        animal_dict[animal] = cell_dict

    return animal_dict

def get_scaled_data_Hz_dict(activity_dict_EC, Hz_SF=50):
    scaled_data_Hz_dict={}
    for animal in activity_dict_EC:
        scaled_data_Hz_dict_cell = {}
        for cell in activity_dict_EC[animal]:
            activity = activity_dict_EC[animal][cell][:,:58]
            min_max_actiivty_list = []
            for i in range(activity.shape[1]):
                trial_activity = activity[:, i]
                min_max_actiivty = (trial_activity - (np.min(trial_activity))) / (np.max(trial_activity) - (np.min(trial_activity)))
                scaled_data_Hz = min_max_actiivty * Hz_SF
                min_max_actiivty_list.append(scaled_data_Hz)
            min_max_actiivty_array = np.array(min_max_actiivty_list)
            scaled_data_Hz_dict_cell[cell] = min_max_actiivty_array.T
        scaled_data_Hz_dict[animal] = scaled_data_Hz_dict_cell
    return scaled_data_Hz_dict

def do_the_interpolation(scaled_data_Hz_dict, an_velocity=None):
    
    padded_warped_activity_dict = {}

    dt_constant = 0.001

    for animal in scaled_data_Hz_dict:
        padded_cell = {}
        for cell in scaled_data_Hz_dict[animal]:

            summed_dendrite = scaled_data_Hz_dict[animal][cell]

            # if vel_applied == 'constant':
            #     an_velocity = np.full((summed_dendrite.shape), 0.43) #0.43 meters per second animal velocity 
            # else:
            an_velocity = an_velocity

            total_time_sec = 4.71657036 

            dt=total_time_sec/50
            dx=180/50

            proper_velocity=an_velocity*100

            animal_velocity_constant= np.full((summed_dendrite.shape), dx/dt)

            # if vel_applied=="constant":
            #     dt = dx / animal_velocity_constant
            # else:
            dt = dx / proper_velocity

            time_bins = np.cumsum(dt, axis=0)
            time_bins_ms = time_bins * 1

            num_trials = summed_dendrite.shape[1]
            trial_warped_activity = []
            max_len = 0

            for t in range(num_trials):
                if np.any(np.isnan(time_bins[:, t])):
                    continue
                total_time = time_bins[-1, t]

                time_axis_constant = np.arange(0, total_time, dt_constant)
                
               	firing = summed_dendrite[:, t]

                warped_firing = np.interp(time_axis_constant, time_bins_ms[:,t], firing)

                trial_warped_activity.append(warped_firing)
                if len(warped_firing) > max_len:
                    max_len = len(warped_firing)


            # if vel_applied=="constant":
            #     padded_warped_activity = np.full((num_trials, max_len), np.nan)
            #     for i, trace in enumerate(trial_warped_activity):
            #             padded_warped_activity[i, :len(trace)] = trace 

            #     padded_cell[cell] = padded_warped_activity
            # else:
            padded_cell[cell] = trial_warped_activity
        padded_warped_activity_dict[animal] = padded_cell

    return padded_warped_activity_dict, an_velocity

def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None, rng=None):
    """
    Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
    Poisson process with a refractory period after each spike.
    :param rate: instantaneous rates in time (Hz)
    :param t: corresponding time values (ms)
    :param dt: temporal resolution for spike times (ms)
    :param refractory: absolute deadtime following a spike (ms)
    :param generator: :class:'random.Random()'
    :return: list of m spike times (ms)
    """
    if generator is None:
        generator = rng
    interp_t = np.arange(t[0], t[-1] + dt, dt)
    interp_rate = np.interp(interp_t, t, rate)
    interp_rate /= 1000.
    non_zero = np.where(interp_rate > 0.)[0]
    interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
    spike_times = []
    max_rate = np.max(interp_rate)
    i = 0
    ISI_memory = 0.
    while i < len(interp_t):
        x = generator.random()
        if x > 0.:
            ISI = -np.log(x) / max_rate
            i += int(ISI / dt)
            ISI_memory += ISI
            if (i < len(interp_t)) and (generator.random() <= interp_rate[i] / max_rate) and ISI_memory >= 0.:
                spike_times.append(interp_t[i])

