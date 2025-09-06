import numpy as np
import os
import torch
import slicetca
import mat73
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
import h5py
import matplotlib.pyplot as plt

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

def get_scaled_data_Hz_dict(activity_dict_EC, Hz_SF=50.0, eps=1e-12):
    out = {}
    num_cells_per_animal = {}
    for animal in activity_dict_EC:
        counter=0
        per_cell = {}
        for cell, A in activity_dict_EC[animal].items():
            counter+=1
            A = np.asarray(A[:, :58], dtype=float)  # (n_pos, 58)
            B = np.empty_like(A, dtype=float)
            for i in range(A.shape[1]):
                x = A[:, i]
                denom = max(np.nanmax(x) - np.nanmin(x), eps)
                B[:, i] = ((x - np.nanmin(x)) / denom) * Hz_SF
            per_cell[cell] = B
        num_cells_per_animal[animal] = counter
        out[animal] = per_cell
    return out, num_cells_per_animal


# def get_scaled_data_Hz_dict(activity_dict_EC, Hz_SF=50):
#     scaled_data_Hz_dict={}
#     for animal in activity_dict_EC:
#         scaled_data_Hz_dict_cell = {}
#         for cell in activity_dict_EC[animal]:
#             activity = activity_dict_EC[animal][cell][:,:58]
#             min_max_actiivty_list = []
#             for i in range(activity.shape[1]):
#                 trial_activity = activity[:, i]
#                 min_max_actiivty = (trial_activity - (np.min(trial_activity))) / (np.max(trial_activity) - (np.min(trial_activity)))
#                 scaled_data_Hz = min_max_actiivty * Hz_SF
#                 min_max_actiivty_list.append(scaled_data_Hz)
#             min_max_actiivty_array = np.array(min_max_actiivty_list)
#             scaled_data_Hz_dict_cell[cell] = min_max_actiivty_array.T
#         scaled_data_Hz_dict[animal] = scaled_data_Hz_dict_cell
#     return scaled_data_Hz_dict

# def do_the_interpolation(scaled_data_Hz_dict, an_velocity=None):
#     padded_warped_activity_dict = {}
#     dt_constant = 0.001  # 1 ms
#     total_time_sec = 4.71657036
#     npos = 50
#     dt_nominal = total_time_sec / npos
#     dx = 180 / npos

#     for animal in scaled_data_Hz_dict:
#         padded_cell = {}
#         for cell in scaled_data_Hz_dict[animal]:
#             firing_mat = scaled_data_Hz_dict[animal][cell]  # shape (npos, n_trials)
#             vel = an_velocity  # (npos, n_trials)

#             # velocity in cm/s
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 proper_velocity = vel * 100.0
#                 # replace invalid or tiny velocities with NaN so we can skip those trials
#                 proper_velocity = np.where(
#                     ~(np.isfinite(proper_velocity)) | (proper_velocity <= 1e-6),
#                     np.nan, proper_velocity
#                 )
#                 dt = dx / proper_velocity  # seconds/position bin

#             # cumulative time axis per trial
#             time_bins = np.cumsum(dt, axis=0)  # shape (npos, n_trials)
#             # require strictly finite & increasing time axis
#             ok_trial = []
#             trial_warped_activity = []
#             max_len = 0

#             num_trials = firing_mat.shape[1]
#             for t in range(num_trials):
#                 tb = time_bins[:, t]
#                 if (not np.all(np.isfinite(tb))) or (np.any(np.diff(tb) <= 0)):
#                     # bad velocity for this trial → skip
#                     continue

#                 time_axis = np.arange(0.0, tb[-1], dt_constant)
#                 firing = firing_mat[:, t]
#                 # guard: if firing has NaNs, drop this trial
#                 if not np.all(np.isfinite(firing)):
#                     continue

#                 warped = np.interp(time_axis, tb, firing)
#                 # final guard
#                 if not np.all(np.isfinite(warped)) or warped.size == 0:
#                     continue

#                 trial_warped_activity.append(warped)
#                 max_len = max(max_len, warped.size)
#                 ok_trial.append(t)

#             # if you need padded arrays, you can pad here; otherwise keep list
#             print(f"len(trial_warped_activity {len(trial_warped_activity)}")
#             padded_cell[cell] = trial_warped_activity

#         padded_warped_activity_dict[animal] = padded_cell

#     return padded_warped_activity_dict, an_velocity



# def do_the_interpolation(scaled_data_Hz_dict, an_velocity=None, dt_constant=0.001):
#     """
#     Keep EXACTLY n_trials outputs per cell by repairing velocity rather than skipping trials.
#     Returns:
#         padded_warped_activity_dict[animal][cell] -> list of length n_trials (each 1D float32)
#         an_velocity (unchanged)
#     """
#     padded_warped_activity_dict = {}
#     npos = 50
#     dx = 180.0 / npos  # cm per bin (match your units)

#     for animal in scaled_data_Hz_dict:
#         padded_cell = {}
#         for cell in scaled_data_Hz_dict[animal]:
#             firing_mat = scaled_data_Hz_dict[animal][cell]   # (npos, n_trials)
#             vel_mat    = an_velocity                         # (npos, n_trials)
#             n_pos, n_trials = firing_mat.shape
#             assert n_pos == npos, f"expected {npos} pos bins, got {n_pos}"

#             trial_warped_activity = []
#             for t in range(n_trials):
#                 firing = firing_mat[:, t].astype(np.float64, copy=False)
#                 vel_cm = (vel_mat[:, t] * 100.0).astype(np.float64, copy=False)

#                 # 1) repair velocity (no zeros/NaNs)
#                 vel_cm = _sanitize_velocity_cm_s(vel_cm)

#                 if np.any(np.isnan(vel_cm)):
#                     print('oops nan')

#                 if np.any(np.any(vel_cm==0)):
#                     print('oops vel')


#                 # 2) build strictly increasing time edges
#                 dt_s   = dx / vel_cm
#                 edges  = np.concatenate(([0.0], np.cumsum(dt_s)))
#                 total_time = float(edges[-1])

#                 # 3) constant time axis and interpolate
#                 if np.isfinite(total_time) and total_time > 0 and np.isfinite(firing).sum() >= 2:
#                     t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
#                     # use bin centers for interpolation (edges[1:] are the bin ends)
#                     time_pts = edges[1:]  # length n_pos
#                     warped = np.interp(t_axis, time_pts, firing)
#                     warped = warped.astype(np.float32, copy=False)
#                     if warped.size == 0 or not np.isfinite(warped).any():
#                         warped = np.full(1, np.nan, dtype=np.float32)
#                 else:
#                     print("problem")
#                     # placeholder if something is still wrong
#                     warped = np.full(1, np.nan, dtype=np.float32)

#                 trial_warped_activity.append(warped)

#                 if np.any(np.isnan(warped)):

#             # IMPORTANT: we return a list with length == n_trials
#             padded_cell[cell] = trial_warped_activity

#         padded_warped_activity_dict[animal] = padded_cell

#     return padded_warped_activity_dict, an_velocity



def do_the_interpolation(
    scaled_data_Hz_dict,
    an_velocity=None,
    dt_constant=0.001,      # seconds
    min_vel_cm_s=1e-3,      # clamp ultra small velocities
    use_bin_centers=True,
    verbose=True,
    log_limit=30,
    fallback_mode="zero"    # "zero" | "first" | "nan"
    ):
    """
    Returns:
        padded_warped_activity_dict[animal][cell] -> list of length n_trials (each 1D float32)
        an_velocity (unchanged)
    """
    assert isinstance(an_velocity, np.ndarray) and an_velocity.ndim == 2, "an_velocity must be (npos, n_trials)"
    npos = 50
    assert an_velocity.shape[0] == npos, f"an_velocity must have npos={npos} rows"
    dx = 180.0 / npos  # cm per bin

    def _log(msg):
        if verbose and (len(bad_trials) < log_limit):
            print(msg)

    padded_warped_activity_dict = {}
    bad_trials = []  # collect a few bad ones to summarize once

    for animal in scaled_data_Hz_dict:
        padded_cell = {}
        for cell in scaled_data_Hz_dict[animal]:
            firing_mat = scaled_data_Hz_dict[animal][cell]   # (npos, n_trials)
            assert firing_mat.shape[0] == npos, f"[{animal}][{cell}] firing_mat shape {firing_mat.shape}, expected ({npos}, n_trials)"
            n_pos, n_trials = firing_mat.shape
            assert an_velocity.shape[1] == n_trials, f"[{animal}][{cell}] vel trials {an_velocity.shape[1]} != firing trials {n_trials}"

            trial_warped_activity = []

            for t in range(n_trials):
                firing = firing_mat[:, t].astype(np.float64, copy=False)
                vel_cm = (an_velocity[:, t] * 100.0).astype(np.float64, copy=False)  # cm/s

                # (A) Velocity checks (you said no NaNs/zeros; still guard & clamp tiny)
                vel_bad = (~np.isfinite(vel_cm)) | (vel_cm <= 0)
                if vel_bad.any():
                    _log(f"[interp][{animal}][{cell}][t={t}] velocity non-finite or <=0 at {vel_bad.sum()} bins")
                # clamp tiny to avoid huge dt_s
                vel_cm = np.nan_to_num(vel_cm, nan=min_vel_cm_s, posinf=min_vel_cm_s, neginf=min_vel_cm_s)
                vel_cm = np.clip(vel_cm, min_vel_cm_s, None)

                # (B) Firing checks
                finite_firing = np.isfinite(firing)
                n_finite = int(finite_firing.sum())
                if n_finite < 2:
                    _log(f"[interp][{animal}][{cell}][t={t}] firing has only {n_finite} finite samples (need >=2)")
                firing_for_interp = np.where(finite_firing, firing, 0.0)  # prevent interp NaNs

                # (C) Build time grid
                dt_s = dx / vel_cm  # seconds per bin
                edges = np.concatenate(([0.0], np.cumsum(dt_s)))
                total_time = float(edges[-1])

                # centers are safer; ends also ok if strictly increasing
                time_pts = 0.5*(edges[:-1] + edges[1:]) if use_bin_centers else edges[1:]
                # quick domain sanity
                monotonic = np.all(np.diff(time_pts) > 0)
                if not monotonic:
                    _log(f"[interp][{animal}][{cell}][t={t}] time_pts not strictly increasing (velocity grid issue)")

                # (D) Build uniform t_axis
                if np.isfinite(total_time) and total_time > 0:
                    t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
                else:
                    t_axis = np.array([], dtype=np.float64)

                # (E) Diagnose common NaN causes
                if t_axis.size == 0:
                    # total_time < dt_constant or invalid → the classic "empty grid" cause
                    _log(f"[interp][{animal}][{cell}][t={t}] empty t_axis: total_time={total_time:.6g}, dt={dt_constant}, "
                         f"vel[min,max]=({vel_cm.min():.3g},{vel_cm.max():.3g}), dt_s[min,max]=({dt_s.min():.3g},{dt_s.max():.3g})")

                # (F) Interpolate or fallback
                if t_axis.size >= 1 and monotonic and n_finite >= 2:
                    # np.interp never returns NaNs if inputs are finite; we ensured that
                    warped = np.interp(t_axis, time_pts, firing_for_interp).astype(np.float32, copy=False)
                else:
                    # fallback — choose policy
                    if fallback_mode == "zero":
                        warped = np.zeros(1, dtype=np.float32)
                    elif fallback_mode == "first":
                        # pick first finite firing (or 0)
                        fv = firing[finite_firing][0] if n_finite >= 1 else 0.0
                        warped = np.array([fv], dtype=np.float32)
                    else:  # "nan"
                        warped = np.full(1, np.nan, dtype=np.float32)

                    bad_trials.append((animal, cell, t))

                # Final assert: keep no-NaN if you want to protect downstream
                if not np.isfinite(warped).all():
                    _log(f"[interp][{animal}][{cell}][t={t}] warped still has NaNs (likely due to fallback='nan').")
                trial_warped_activity.append(warped)

            padded_cell[cell] = trial_warped_activity
        padded_warped_activity_dict[animal] = padded_cell

    if verbose and bad_trials:
        print(f"[interp] {len(bad_trials)} trials used fallback (most common cause: tiny total_time < dt). "
              f"Examples (up to {log_limit}):")
        for a, c, t in bad_trials[:log_limit]:
            print(f"    animal={a}, cell={c}, trial={t}")

    return padded_warped_activity_dict, an_velocity



def get_plateau_and_cumulative_ragged(
    padded_warped_activity_dict,
    dend_threshold,
    plateau_len=300,   # samples (300 @ 1 ms = 300 ms)
    refractory=800,    # samples
    scan_step=100      # samples
):
    plateau_dict_animal = {}
    counts_dict_animal = {}

    for animal, cells in padded_warped_activity_dict.items():
        plateau_dict_cell = {}
        counts_dict_cell = {}

        for cell, trials in cells.items():  # trials: List[np.ndarray] (ragged)
            plateau_arrays = []
            starts_per_trial = []

            for x in trials:
                # guard for bad/empty trials
                if not isinstance(x, np.ndarray) or x.size == 0 or ~np.isfinite(x).any():
                    plateau_arrays.append(np.zeros(1, dtype=np.uint8))
                    starts_per_trial.append(0)
                    continue

                x = np.asarray(x, dtype=float)
                marks = np.zeros_like(x, dtype=np.uint8)

                i = 0
                N = x.size
                n_starts = 0
                while i < N:
                    if x[i] > dend_threshold:
                        end = min(i + plateau_len, N)
                        marks[i:end] = 1
                        n_starts += 1
                        i += refractory
                    else:
                        i += scan_step

                plateau_arrays.append(marks)
                starts_per_trial.append(n_starts)

            # ----- PAD & STACK to get a true 2-D array (trials × timebins) -----
            if len(plateau_arrays) == 0:
                plateau_2d = np.zeros((0, 0), dtype=np.uint8)
                cumulative = np.zeros((0,), dtype=int)
            else:
                max_len = max(arr.size for arr in plateau_arrays)
                if max_len == 0:
                    plateau_2d = np.zeros((len(plateau_arrays), 0), dtype=np.uint8)
                else:
                    plateau_2d = np.zeros((len(plateau_arrays), max_len), dtype=np.uint8)
                    for i, arr in enumerate(plateau_arrays):
                        plateau_2d[i, :arr.size] = arr  # pad with 0s to the right

                cumulative = np.cumsum(np.asarray(starts_per_trial, dtype=int))

            plateau_dict_cell[cell] = plateau_2d          # shape: (n_trials, max_len)
            counts_dict_cell[cell] = cumulative           # shape: (n_trials,)

        plateau_dict_animal[animal] = plateau_dict_cell
        counts_dict_animal[animal] = counts_dict_cell

    return plateau_dict_animal, counts_dict_animal




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
                ISI_memory = -refractory
    return np.array(spike_times)

def exp_kernel(tau_ms, dt_ms, n_taus=5, norm="area", target=1.0):
    """
    Create a causal exponential kernel e^{-t/tau}.
    norm: "area" -> sum(kernel) == 1
          "peak" -> max(kernel) == 1
    target: scales the chosen normalization (e.g., target mV or arbitrary units).
    """
    klen = int(np.ceil(n_taus * tau_ms / dt_ms))
    t = np.arange(klen) * dt_ms
    k = np.exp(-t / tau_ms)

    if norm == "area":
        k /= k.sum() + 1e-12         # unit L1
    elif norm == "peak":
        k /= k.max() + 1e-12         # unit peak
    else:
        raise ValueError("norm must be 'area' or 'peak'")

    return k * target                # final amplitude per spike

def get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest, which_type=None):

    if which_type == "EC_animal_average":
        an_velocity_real_list = []
        for animal in factors_dict_EC:
                an_velocity_real_list.append(factors_dict_EC[animal]["Velocity"][:,:58])

        an_velocity_real_array = np.array(an_velocity_real_list)
        an_velocity_real_array_mean_animal = np.nanmean(an_velocity_real_array, axis=0)
        return an_velocity_real_array_mean_animal
    
    elif which_type == "repeated_waveform":
        an_velocity_real_list_all = []
        for animal in factors_dict_EC:
                an_velocity_real_list_all.append(factors_dict_EC[animal]["Velocity"][:,:58])
        for animal in factors_dict_SST:
                an_velocity_real_list_all.append(factors_dict_SST[animal]["Velocity"][:,:58])
        for animal in fixed_filtered_factors_dict_NDNF_newest:
                an_velocity_real_list_all.append(fixed_filtered_factors_dict_NDNF_newest[animal]["Velocity"][:,:58])

        an_velocity_real_array_all = np.array(an_velocity_real_list_all)
        an_velocity_real_array_mean_animal_all = np.mean(an_velocity_real_array_all, axis=0)
        mean_velocity = np.mean(an_velocity_real_array_mean_animal_all, axis=1)
        mean_vel_2d = np.tile(mean_velocity[:,None], (1,58))
        return mean_vel_2d
    
    elif which_type == "constant":
         constant_vel = np.full((50,58), 0.43)
         return constant_vel

def get_epsp_dict(padded_warped_activity_dict, tau_ms=None, amp=None, seed=None):

    dt_constant = 0.001

    dt_ms = dt_constant * 1000.0      # 1 ms

    tau_ms  = tau_ms
    dt_ms   = dt_constant * 1000.0      # 1 ms
    AMP     = amp                      # mV
    MODE    = "peak"                    # "area" or "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

    rng = np.random.default_rng(seed)


    animal_dict = {}
    for animal in padded_warped_activity_dict:
        cell_dict = {}
        for cell in padded_warped_activity_dict[animal]:
            epsps_dict= {}
            spike_times_dict = {}
            spike_train_dict = {}

            padded_warped_activity = padded_warped_activity_dict[animal][cell]


            for trial in range(len(padded_warped_activity)):
                
                example_padded_warped_activity = padded_warped_activity[trial]
                if trial > 0:
                    example_previous_pad = padded_warped_activity[trial-1]
                else:
                    example_previous_pad = padded_warped_activity[trial+1]

                L_prev = example_previous_pad.shape[0]
                L_curr = example_padded_warped_activity.shape[0]

                two_track_length = np.concatenate([example_previous_pad, example_padded_warped_activity], axis=0)
                t_ms = np.arange(two_track_length.shape[0]) * dt_ms

                spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
                st_curr = spike_times[spike_times >= L_prev] - L_prev
                spike_times_dict[trial] = st_curr

                spike_train = np.zeros(two_track_length.shape, dtype=np.uint8)
                spike_train[spike_times] = 1
                spike_train_curr = spike_train[L_prev : L_prev + L_curr]
                spike_train_dict[trial] = spike_train_curr

                epsps = np.convolve(spike_train, kernel, mode='full')[:len(spike_train)]
                epsps_curr = epsps[L_prev : L_prev + L_curr]
                epsps_dict[trial] = epsps_curr
                # epsps_scaled = epsps-60
                # dendrite.append(epsps_scaled)
            


            cell_dict[cell] = {"epsps":epsps_dict,
                                "spike_times":spike_times_dict,
                                "spike_train":spike_train_dict}
        animal_dict[animal] = cell_dict

    return animal_dict, kernel

def get_dend_vm(epsp_dict, Vrest=-60.0, epsp_sf=0.1):
    cell_epsp_mats = []
    cell_spike_mats = []

    # --- per cell: build (n_trials, T_cell) as float so we can NaN-pad ---
    for animal in epsp_dict:
        for cell in epsp_dict[animal]:
            epsp = epsp_dict[animal][cell]["epsps"]         # dict: trial -> 1D array (float)
            spik = epsp_dict[animal][cell]["spike_train"]   # dict: trial -> 1D array (uint8)

            if not epsp:  # skip truly empty cells
                continue

            # Per-cell max lengths (time)
            max_len_epsp = max(len(epsp[t]) for t in epsp)
            max_len_spik = max(len(spik[t]) for t in spik)

            # EPSPs -> (n_trials, max_len_epsp), float with NaN padding
            epsp_trials = []
            for t in range(len(epsp)):
                v = np.asarray(epsp[t], dtype=np.float32)
                if v.size < max_len_epsp:
                    v = np.pad(v, (0, max_len_epsp - v.size), mode="constant", constant_values=np.nan)
                epsp_trials.append(v)
            epsp_mat = np.vstack(epsp_trials).astype(np.float32, copy=False)
            cell_epsp_mats.append(epsp_mat)

            # Spikes -> (n_trials, max_len_spik), cast to float before NaN padding
            spk_trials = []
            for t in range(len(spik)):
                v = np.asarray(spik[t], dtype=np.float32)  # cast BEFORE padding so NaN is valid
                if v.size < max_len_spik:
                    v = np.pad(v, (0, max_len_spik - v.size), mode="constant", constant_values=np.nan)
                spk_trials.append(v)
            spk_mat = np.vstack(spk_trials).astype(np.float32, copy=False)
            cell_spike_mats.append(spk_mat)

    if not cell_epsp_mats:
        raise ValueError("No EPSP matrices were built (empty epsp_dict?).")

    # --- across cells: pad to GLOBAL (n_trials, T) so we can stack cleanly ---
    global_T = max(m.shape[1] for m in cell_epsp_mats)
    global_N = max(m.shape[0] for m in cell_epsp_mats)

    def pad_to_global(mat):
        n, t = mat.shape
        dn = global_N - n
        dt = global_T - t
        if dn > 0 or dt > 0:
            mat = np.pad(mat, ((0, max(dn,0)), (0, max(dt,0))),
                         mode="constant", constant_values=np.nan)
        return mat.astype(np.float32, copy=False)

    epsp_stack = np.stack([pad_to_global(m) for m in cell_epsp_mats], axis=0)  # (n_cells, N, T)

    # --- masked SUM across cells, keeping NaN where no data exists ---
    valid_counts = np.sum(~np.isnan(epsp_stack), axis=0)   # (N, T)
    summed = np.nansum(epsp_stack, axis=0)                 # (N, T)
    summed[valid_counts == 0] = np.nan

    # center per trial using nanmean
    trial_means = np.nanmean(summed, axis=1, keepdims=True)  # (N, 1)
    summed_centered = summed - trial_means

    dend_Vm = Vrest + epsp_sf * summed_centered  # (N, T)

    return dend_Vm, epsp_stack, cell_spike_mats


def plot_dend_vm_example(dend_Vm, epsp_list, spike_list, residual_activity_dict_EC, an_velocity):
    

    data_list = []
    for animal in residual_activity_dict_EC:
        for cell in residual_activity_dict_EC[animal]:
            data_list.append(residual_activity_dict_EC[animal][cell][:,:58])

    data_array = np.array(data_list)
    summed_dendrite = np.sum(data_array, axis=0)

    padded_warped_activity_EC, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=0.001, dend_threshold=-20, vel_applied="real")

    fig, axs= plt.subplots(2,2, figsize=(12,9))
    im = axs[0,0].imshow(padded_warped_activity_EC, aspect='auto')
    axs[0,0].set_title("Dendrite DF/F")
    axs[0,0].set_ylabel("Trials")
    axs[0,0].set_xlabel("Time (ms)")
    fig.colorbar(im, ax=axs[0,0], label="Z-Scored Summed DF/F")


    mean_padded_warped_activity_EC = np.nanmean(padded_warped_activity_EC, axis=0)
    sem_padded_warped_activity_EC = sem(padded_warped_activity_EC, axis=0, nan_policy='omit')

    axs[0,1].plot(mean_padded_warped_activity_EC)
    axs[0,1].fill_between(range(len(mean_padded_warped_activity_EC)), mean_padded_warped_activity_EC-sem_padded_warped_activity_EC, mean_padded_warped_activity_EC+sem_padded_warped_activity_EC, alpha=0.2)
    axs[0,1].set_title("Dendrite Trial Averaged DF/F")
    axs[0,1].set_xlabel("Time (ms)")
    axs[0,1].set_ylabel('Z-Scored Summed DF/F')



    fig.suptitle("Dend Vm from Summed Already Convolved EPSPs For Each EC Input")

    mean_dend = np.nanmean(dend_Vm, axis=0)
    sem_dend = sem(dend_Vm, axis=0, nan_policy='omit')

    axs[1,1].plot(mean_dend)
    axs[1,1].fill_between(range(len(mean_dend)), mean_dend+sem_dend, mean_dend-sem_dend, alpha=0.2)
    axs[1,1].axhline(Vrest, color='b', linestyle='--', label="Vrest")
    axs[1,1].legend()
    axs[1,1].set_title("Dendrite Vm")
    axs[1,1].set_ylabel("Vm (mV)")
    axs[1,1].set_xlabel("Time (ms)")

    im = axs[1,0].imshow(dend_Vm, aspect='auto')
    axs[1,0].set_title("Dendrite Vm")
    axs[1,0].set_ylabel("Trial")
    axs[1,0].set_xlabel("Time (ms)")
    fig.colorbar(im, ax=axs[1,0], label='Vm (mV)')

    plt.tight_layout()
    plt.plot()

    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].plot(spike_list[0][0,3000:3300])
    axs[0].set_title("First Cell First Trial EC Turned into Spikes Using Poisson")
    axs[0].set_xlabel("Time (ms)")

    axs[1].plot(epsp_list[0][0,3000:3300])
    axs[1].set_title("First Cell First Trial EC Convolved Spikes")
    axs[1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

    def pad_sequences(seqs, pad_value=np.nan, dtype=float):
        """Pad 1D arrays in `seqs` to the same length."""
        seqs = [np.asarray(s, dtype=dtype) for s in seqs]
        maxlen = max(len(s) for s in seqs)
        out = np.full((len(seqs), maxlen), pad_value, dtype=dtype)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return out

def plot_single_cell_trial_comparison(scaled_data_Hz_dict, padded_warped_activity_dict):


    fig, axs = plt.subplots(2,3, figsize=(16,8))

    first_ec_cell_rate_position = scaled_data_Hz_dict['animal_1']['cell_1']
    im = axs[0,0].imshow(first_ec_cell_rate_position.T, aspect='auto')
    axs[0,0].set_xlabel("Position Bins")
    axs[0,0].set_ylabel("Trials")
    axs[0,0].set_title("EC Cell 1 Rate Over Position")
    fig.colorbar(im, ax=axs[0,0], label="Hz")

    mean_dendrite = np.mean(first_ec_cell_rate_position, axis=1)
    sem_dendrite = sem(first_ec_cell_rate_position, axis=1)

    axs[1,0].plot(mean_dendrite)
    axs[1,0].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2)
    axs[1,0].set_title("Trial Averaged Dendrite Vm")
    axs[1,0].set_xlabel("Time (ms)")
    axs[1,0].set_ylabel("Mean Activity")

    first_ec_cell_rate_time = padded_warped_activity_dict['animal_1']['cell_1']
    max_len_padded = max(len(first_ec_cell_rate_time[t]) for t in range(len(first_ec_cell_rate_time)))
    padded_trials = []
    for t in range(len(first_ec_cell_rate_time)):
        v = first_ec_cell_rate_time[t]
        if len(v) < max_len_padded:
            v = np.pad(v, (0, max_len_padded - len(v)), constant_values=np.nan)
        padded_trials.append(v.astype(np.float32, copy=False))

    padded_trials_array = np.array(padded_trials)
    im = axs[0,1].imshow(padded_trials_array, aspect='auto', interpolation='none')
    axs[0,1].set_xlabel("Time (ms)")
    axs[0,1].set_ylabel("Trials")
    axs[0,1].set_title("EC Cell 1 Rate Over Time")
    fig.colorbar(im, ax=axs[0,1], label="Hz")

    mean_dendrite = np.mean(padded_trials_array, axis=0)
    sem_dendrite = sem(padded_trials_array, axis=0)

    axs[1,1].plot(mean_dendrite)
    axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2)
    axs[1,1].set_title("Trial Averaged Dendrite Vm")
    axs[1,1].set_xlabel("Time (ms)")
    axs[1,1].set_ylabel("Mean Activity")



    first_ec_cell_rate_time = padded_warped_activity_dict['animal_1']['cell_1']
    max_len_padded = max(len(first_ec_cell_rate_time[t]) for t in range(len(first_ec_cell_rate_time)))
    padded_trials = []
    for t in range(len(first_ec_cell_rate_time)):
        v = first_ec_cell_rate_time[t]
        if len(v) < max_len_padded:
            v = np.pad(v, (0, max_len_padded - len(v)), constant_values=np.nan)
        padded_trials.append(v.astype(np.float32, copy=False))



    spike_dict=epsp_dict['animal_1']['cell_1']["spike_train"]
    max_len_spike = max(len(spike_dict[t]) for t in range(len(spike_dict)))
    trial_list = []
    for t in range(len(spike_dict)):
        v = spike_dict[t]
        if len(v) < max_len_spike:
            v = np.pad(v, (0, max_len_spike - len(v)), constant_values=np.nan)
        trial_list.append(v.astype(np.float32, copy=False))
    trial_array = np.array(trial_list)
    im = axs[0,2].imshow(trial_array, aspect='auto')
    axs[0,2].set_xlabel("Time (ms)")
    axs[0,2].set_ylabel("Trials")
    axs[0,2].set_title("EC Cell 1 Spikes Over Time")
    axs[0,2].set_xlabel("Time (ms)")
    fig.colorbar(im, ax=axs[0,2])


    mean_dendrite = np.mean(trial_array, axis=0)
    sem_dendrite = sem(trial_array, axis=0)

    sigma = 10  # adjust this value depending on how much smoothing you want

    mean_dendrite_smooth = gaussian_filter1d(mean_dendrite, sigma=sigma)
    sem_dendrite_smooth = gaussian_filter1d(sem_dendrite, sigma=sigma)

    axs[1,2].plot(mean_dendrite_smooth)
    axs[1,2].fill_between(range(len(mean_dendrite_smooth)), mean_dendrite_smooth+sem_dendrite_smooth, mean_dendrite_smooth-sem_dendrite_smooth, alpha=0.2)
    axs[1,2].set_title("Smoothed Trial Averaged Dendrite Vm")
    axs[1,2].set_xlabel("Time (ms)")
    axs[1,2].set_ylabel("Mean Activity")

    plt.tight_layout()
    plt.plot()

    fig, axs = plt.subplots(2,1, figsize=(16,3))
    axs[0].plot(first_ec_cell_spikes[11,:])
    axs[0].set_title("Spikes Trial 11")
    axs[1].plot(first_ec_cell_rate_time[11], label="rate")
    axs[1].set_title("Rate Trial 11")
    axs[1].set_xlabel("Time (ms)")

    plt.tight_layout()
    plt.show()

def get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=0.001, dend_threshold=None, vel_applied=None):


    animal_velocity_constant = np.full((summed_dendrite.shape), 23)
    total_time_sec = 4.71657036 

    dt=total_time_sec/50
    dx=180/50

    proper_velocity=an_velocity*100

    animal_velocity_constant= np.full((summed_dendrite.shape), dx/dt)

    if vel_applied=="constant":
        dt = dx / animal_velocity_constant
    else:
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
                
    padded_warped_activity = np.full((num_trials, max_len), np.nan) 
    for i, trace in enumerate(trial_warped_activity):
        padded_warped_activity[i, :len(trace)] = trace

        
        
    mean_dend_time = np.mean(padded_warped_activity, axis=0)
    sem_dend_time = sem(padded_warped_activity, axis=0)


    x_start = 0
    time_bin_size_ms=1
    num_bins = len(mean_dend_time)
    x_end = num_bins * time_bin_size_ms 


    time_bin_size_ms = 1

    num_bins = len(mean_dend_time)
    x_time_ms = np.arange(num_bins) * time_bin_size_ms  # x-axis in ms


    position_bins, num_trials = animal_velocity_constant.shape


    flat_padded_warped_activity = padded_warped_activity.flatten()
    flat_plateau_array = np.zeros_like(flat_padded_warped_activity)

    just_plateau_starts = np.zeros_like(flat_padded_warped_activity)
    just_plateau_starts_list = []
    i = 0
    while i < len(flat_padded_warped_activity):
        if flat_padded_warped_activity[i] > dend_threshold:
            flat_plateau_array[i:i+300] = 1
            just_plateau_starts[i] = 1
            just_plateau_starts_list.append(i)
            
            i += 800
        else:
            i += 100


    plateau_array = flat_plateau_array.reshape(padded_warped_activity.shape)

    just_plateau_starts_reshape = just_plateau_starts.reshape(padded_warped_activity.shape)

    just_plateau_starts_sums = np.sum(just_plateau_starts_reshape, axis=1)

    # Compute the cumulative sum of plateaus across trials
    cumulative_plateau_counts = np.cumsum(just_plateau_starts_sums)


    return padded_warped_activity, summed_dendrite, cumulative_plateau_counts

def bin_vm_time_to_position(Vm_trials, dt_ms, vel_pos_trial_m_s,
                            track_len_cm=180.0, sample_variance=False):
    """
    Inputs
    -------
    Vm_trials : array, shape (n_trials, T)
        Membrane potential vs time for each trial. Units: mV (or your unit).
        Assumed to be sampled at a fixed step dt_ms.

    dt_ms : float
        Time step per sample in milliseconds (e.g., 1.0 for 1 ms sampling).

    vel_pos_trial_m_s : array, shape (n_pos, n_trials)
        Velocity per *position bin* for each trial, in meters/second.
        Row k is the velocity the animal has while traversing position bin k
        on that trial.

    track_len_cm : float
        Physical track length in centimeters (default 180 cm).

    sample_variance : bool
        If True, return *sample* variance (ddof=1) within each bin; otherwise
        return population variance (ddof=0).

    Returns
    -------
    centers_cm : array, shape (n_pos,)
        Physical centers of the position bins, in centimeters.

    Vm_mean : array, shape (n_trials, n_pos)
        Time-weighted mean Vm within each position bin for each trial.

    Vm_var : array, shape (n_trials, n_pos)
        Variance of Vm across the ~100 ms spent in each bin, per trial.
        (Population variance by default; sample variance if sample_variance=True.)

    occ_ms : array, shape (n_trials, n_pos)
        Time spent in each position bin (occupancy), in milliseconds.
    """

    # Unpack dimensions: n_trials trials, T timepoints
    n_trials, T = Vm_trials.shape                     # (n_trials, T)
    # Velocity table has n_pos bins × n_trials columns (one column per trial)
    n_pos, n_trials_v = vel_pos_trial_m_s.shape       # (n_pos, n_trials)
    assert n_trials == n_trials_v, "Trials mismatch between Vm and velocity arrays."

    # Position bin edges along the track in cm: (n_pos+1,)
    edges_cm   = np.linspace(0.0, track_len_cm, n_pos + 1)   # [0, ..., 180]
    # Bin centers in cm: (n_pos,)
    centers_cm = 0.5 * (edges_cm[:-1] + edges_cm[1:])
    # Bin width in cm (uniform): scalar
    bin_width_cm = track_len_cm / n_pos

    # Absolute sample times for one trial in ms: (T,)
    t_ms = np.arange(T, dtype=float) * float(dt_ms)

    # Allocate outputs:
    # time-weighted mean per trial × bin
    Vm_mean = np.full((n_trials, n_pos), np.nan, dtype=float)   # (n_trials, n_pos)
    # variance per trial × bin
    Vm_var  = np.full((n_trials, n_pos), np.nan, dtype=float)   # (n_trials, n_pos)
    # occupancy time in ms per trial × bin
    occ_ms  = np.zeros((n_trials, n_pos), dtype=float)          # (n_trials, n_pos)

    # ---- Loop over trials (each column of vel_pos_trial_m_s corresponds to a trial)
    for tr in range(n_trials):
        # Velocity profile for this trial in cm/s: (n_pos,)
        v_cm_s = np.asarray(vel_pos_trial_m_s[:, tr], float) * 100.0
        # Avoid zeros so time per bin is finite
        v_cm_s = np.maximum(v_cm_s, 1e-9)

        # Time spent in each position bin on this trial, in ms: (n_pos,)
        # Δt_bin = Δx / v, where Δx in cm, v in cm/s; convert s→ms by ×1000
        dt_per_bin_ms = (bin_width_cm / v_cm_s) * 1000.0

        # Cumulative time edges for the lap in ms: (n_pos+1,)
        # edges_ms[k] ≤ t < edges_ms[k+1] means "we're in bin k"
        edges_ms = np.concatenate(([0.0], np.cumsum(dt_per_bin_ms)))

        # Restrict to samples that occur during the lap; boolean mask: (T,)
        valid = t_ms < edges_ms[-1]

        # Vm samples for this trial within the lap: (T_valid,)
        vm = Vm_trials[tr, valid]
        # Map each valid sample time to its position-bin index: (T_valid,)
        # searchsorted(..., side="right") - 1 gives k with edges[k] ≤ t < edges[k+1]
        idx = np.searchsorted(edges_ms, t_ms[valid], side="right") - 1
        # Clip just in case due to floating arithmetic
        idx = np.clip(idx, 0, n_pos - 1)

        # ---- Compute per-bin mean/variance using np.mean/np.var over the samples
        #      that fell into each bin (all samples have equal weight dt_ms here).
        for k in range(n_pos):
            # All Vm samples assigned to bin k for this trial: (n_k,)
            vals = vm[idx == k]
            n = vals.size  # scalar
            if n:
                # Mean within bin k for trial tr
                Vm_mean[tr, k] = np.mean(vals)

                # Variance within bin k for trial tr
                # ddof=0 → population variance; ddof=1 → sample variance (n>1).
                dd = 1 if (sample_variance and n > 1) else 0
                Vm_var[tr, k]  = np.var(vals)

                # Occupancy time for this bin on this trial, in ms
                occ_ms[tr, k]  = n * dt_ms
            else:
                # Leave as NaN/0.0 if no samples fell in this bin
                Vm_mean[tr, k] = np.nan
                Vm_var[tr, k]  = np.nan
                occ_ms[tr, k]  = 0.0

    return centers_cm, Vm_mean, Vm_var, occ_ms

def get_internal_counts(an_velocity, plateau_array, dx=None, dt_constant=None):
    
    plateau_start_positions_counter = np.zeros(50)

    n_trials, n_time = plateau_array.shape
    n_pos = 50


    vel_cm_s = an_velocity * 100.0  # (50, n_trials)

    plateau_time_per_pos_s   = np.zeros(n_pos, dtype=float)  # duration in seconds
    plateau_starts_per_pos   = np.zeros(n_pos, dtype=int)    # onset counts

    # time axis for the (max) warped length
    time_axis = np.arange(n_time) * dt_constant  # seconds

    for t in range(n_trials):
        v = vel_cm_s[:, t].astype(float)  # (50,)
        
        dt_trial = dx / v   #time spent in each position bin
        edges = np.concatenate([[0.0], np.cumsum(dt_trial)])  #cumulative edges in time 
        total_T  = edges[-1]

        # --- get mask for this trial
        valid_mask = time_axis < total_T
        if not np.any(valid_mask):
            continue

        pos_idx_for_time = np.searchsorted(edges, time_axis[valid_mask], side='right') - 1 #map the time sample onto position bins
        pos_idx_for_time = np.clip(pos_idx_for_time, 0, n_pos-1)

        # weight by dt to convert samples → seconds
        plateau_mask = plateau_array[t, valid_mask].astype(bool)
        w = plateau_mask.astype(float) * dt_constant  # seconds
        add_s = np.bincount(pos_idx_for_time, weights=w, minlength=n_pos)
        plateau_time_per_pos_s += add_s

        # --- COUNT OF PLATEAU STARTS per pos-bin (onsets)
        row = plateau_array[t, :np.count_nonzero(valid_mask)]  # truncate to trial time
        starts_idx = np.flatnonzero((np.pad(row.astype(int), (1,0))[:-1] == 0) &
                                    (np.pad(row.astype(int), (1,0))[1:]  == 1))
        if starts_idx.size:
            start_times = starts_idx * dt_constant
            pos_of_starts = np.searchsorted(edges, start_times, side='right') - 1
            pos_of_starts = np.clip(pos_of_starts, 0, n_pos-1)
            np.add.at(plateau_starts_per_pos, pos_of_starts, 1)

    for trial in range(plateau_array.shape[0]):
        velocity_trial = an_velocity[:, trial] *100.0
#         velocity_trial = proper_velocity
        dt_trial = dx / velocity_trial  # in seconds
        time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

        plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
        plateau_start_times = plateau_start_indices * dt_constant  # in seconds
        # plateau_start_times_list.append(plateau_start_times)

        for pt_start_time in plateau_start_times:
            if pt_start_time != 0.0:
                for pos_idx in range(50):
                    if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
                        plateau_start_positions_counter[pos_idx] += 1
                        break

    return plateau_starts_per_pos, plateau_time_per_pos_s, plateau_start_positions_counter

def get_plateau_array_dict(dend_Vm_dict, dend_threshold=None):
    just_plateau_starts_sums_dict = {}
    plateau_array_dict = {}

    for seed in dend_Vm_dict:
        dend_Vm = dend_Vm_dict[seed]
        x_start = 0
        time_bin_size_ms=1
        num_bins = dend_Vm.shape[1]
        x_end = num_bins * time_bin_size_ms 
        time_bin_size_ms = 1

        x_time_ms = np.arange(num_bins) * time_bin_size_ms  # x-axis in ms

        flat_padded_warped_activity = dend_Vm.flatten()
        flat_plateau_array = np.zeros_like(flat_padded_warped_activity)

        just_plateau_starts = np.zeros_like(flat_padded_warped_activity)
        just_plateau_starts_list = []
        i = 0
        while i < len(flat_padded_warped_activity):
            if flat_padded_warped_activity[i] > dend_threshold:
                flat_plateau_array[i:i+300] = 1
                just_plateau_starts[i] = 1
                just_plateau_starts_list.append(i)
                
                i += 800
            else:
                i += 100


        plateau_array = flat_plateau_array.reshape(dend_Vm.shape)

        just_plateau_starts_reshape = just_plateau_starts.reshape(dend_Vm.shape)

        just_plateau_starts_sums = np.sum(just_plateau_starts_reshape, axis=1)

        just_plateau_starts_sums_dict[seed] = just_plateau_starts_sums
        plateau_array_dict[seed] = plateau_array

    return just_plateau_starts_sums_dict, plateau_array_dict

def get_summed_dendrite_EC_DFF(residual_activity_dict_EC):
        data_list = []
        for animal in residual_activity_dict_EC:
            for cell in residual_activity_dict_EC[animal]:
                data_list.append(residual_activity_dict_EC[animal][cell][:,:58])

        data_array = np.array(data_list)
        summed_dendrite = np.sum(data_array, axis=0)
        return summed_dendrite
   
def get_plateau_array(dend_Vm, dend_threshold=None):
    x_start = 0
    time_bin_size_ms=1
    num_bins = dend_Vm.shape[1]
    x_end = num_bins * time_bin_size_ms 
    time_bin_size_ms = 1

    x_time_ms = np.arange(num_bins) * time_bin_size_ms  # x-axis in ms

    flat_padded_warped_activity = dend_Vm.flatten()
    flat_plateau_array = np.zeros_like(flat_padded_warped_activity)

    just_plateau_starts = np.zeros_like(flat_padded_warped_activity)
    just_plateau_starts_list = []
    i = 0
    while i < len(flat_padded_warped_activity):
        if flat_padded_warped_activity[i] > dend_threshold:
            flat_plateau_array[i:i+300] = 1
            just_plateau_starts[i] = 1
            just_plateau_starts_list.append(i)
            
            i += 800
        else:
            i += 100


    plateau_array = flat_plateau_array.reshape(dend_Vm.shape)

    just_plateau_starts_reshape = just_plateau_starts.reshape(dend_Vm.shape)

    just_plateau_starts_sums = np.sum(just_plateau_starts_reshape, axis=1)

    return just_plateau_starts_sums, plateau_array

def get_velocity_array_every_animal(factors_dict_EC, n_trials=58):
    velocity_dict = {}
    for animal, d in factors_dict_EC.items():
        v = np.asarray(d["Velocity"][:, :n_trials], dtype=np.float32)  
        v[v == 0] = np.nan  
        velocity_dict[animal] = v
    return velocity_dict

def _sanitize_velocity_cm_s(v_m_s, min_vel_cm_s=1e-3*100):
    """
    v_m_s : (n_pos,) velocity in m/s.
    Returns finite, strictly positive cm/s with simple interpolation across NaNs/<=0.
    """
    v = np.asarray(v_m_s, dtype=float) * 100.0  # m/s -> cm/s
    bad = ~np.isfinite(v) | (v <= 0)
    if bad.any():
        good_idx = np.flatnonzero(~bad)
        bad_idx  = np.flatnonzero(bad)
        if good_idx.size >= 2:
            v[bad] = np.interp(bad_idx, good_idx, v[good_idx])
        elif good_idx.size == 1:
            v[bad] = v[good_idx[0]]
        else:
            v[:] = 10.0  # fallback 10 cm/s if everything is bad
    return np.maximum(v, min_vel_cm_s)

def do_the_interpolation_an(scaled_data_Hz_dict, an_velocity_dict, dt_constant=0.001):
    """
    scaled_data_Hz_dict : {cell: array (n_pos, n_trials)}    # e.g., (50, 58)
    an_velocity_dict    : array (n_pos, n_trials) in m/s     # same shape as each cell
    dt_constant         : seconds per output sample (e.g., 0.001 s = 1 ms)

    Returns
    -------
    padded_warped_activity_dict_cells : {cell: list of length n_trials, each 1-D array}
    an_velocity_dict : unchanged
    """
    padded_warped_activity_dict_cells = {}

    n_pos, n_trials = next(iter(scaled_data_Hz_dict.values())).shape
    dx = 180.0 / n_pos  

    vel = np.asarray(an_velocity_dict)
    if vel.shape != (n_pos, n_trials):
        raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

    for cell, summed_dendrite in scaled_data_Hz_dict.items():
        if summed_dendrite.shape != (n_pos, n_trials):
            raise ValueError(f"{cell} has shape {summed_dendrite.shape}, expected {(n_pos, n_trials)}")

        trial_warped_activity = []

        for t in range(n_trials):
            v_cm_s = _sanitize_velocity_cm_s(vel[:, t])      
            dt_s   = dx / v_cm_s                             
            edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
            total_time = float(edges_s[-1])

            # constant-time axis in seconds
            t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)

            firing = summed_dendrite[:, t].astype(np.float64, copy=False)
            valid = np.isfinite(firing)
            if valid.sum() >= 2:
                time_points = np.cumsum(dt_s)  # length n_pos
                w = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
            else:
                w = np.full(1, np.nan, dtype=np.float32)

            trial_warped_activity.append(w)

        padded_warped_activity_dict_cells[cell] = trial_warped_activity

    return padded_warped_activity_dict_cells, an_velocity_dict

def get_epsp_dict_animal(padded_warped_activity_dicts, tau_ms=None, amp=None, seed=None):

    dt_constant = 0.001

    dt_ms = dt_constant * 1000.0      # 1 ms

    tau_ms  = tau_ms
    dt_ms   = dt_constant * 1000.0      # 1 ms
    AMP     = amp                      # mV
    MODE    = "peak"                    # "area" or "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

    rng = np.random.default_rng(seed)


    # animal_dict = {}
    # for animal in padded_warped_activity_dict:

    

    cell_dict = {}
    for cell in padded_warped_activity_dicts:

        trial_count = 0
        epsps_dict= {}
        spike_times_dict = {}
        spike_train_dict = {}

        padded_warped_activity = padded_warped_activity_dicts[cell]

        for trial in range(len(padded_warped_activity)):
            
            example_padded_warped_activity = padded_warped_activity[trial]
            if trial > 0:
                example_previous_pad = padded_warped_activity[trial-1]
            else:
                example_previous_pad = padded_warped_activity[trial+1]

            L_prev = example_previous_pad.shape[0]
            L_curr = example_padded_warped_activity.shape[0]

            two_track_length = np.concatenate([example_previous_pad, example_padded_warped_activity], axis=0)
            t_ms = np.arange(two_track_length.shape[0]) * dt_ms

            spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
            st_curr = spike_times[spike_times >= L_prev] - L_prev
            spike_times_dict[trial] = st_curr

            spike_train = np.zeros(two_track_length.shape, dtype=np.uint8)
            spike_train[spike_times] = 1
            spike_train_curr = spike_train[L_prev : L_prev + L_curr]
            spike_train_dict[trial] = spike_train_curr

            epsps = np.convolve(spike_train, kernel, mode='full')[:len(spike_train)]

            epsps_curr = epsps[L_prev : L_prev + L_curr]
            epsps_dict[trial] = epsps_curr
            # epsps_scaled = epsps-60
            # dendrite.append(epsps_scaled)
            trial_count+=1


        cell_dict[cell] = {"epsps":epsps_dict,
                            "spike_times":spike_times_dict,
                            "spike_train":spike_train_dict}

    return cell_dict, kernel

def get_dend_vm_from_cells(cells_dict, Vrest=-60.0, epsp_sf=0.1):
    """
    Build dendritic Vm for a *single animal* given per-cell EPSPs/spike trains.

    Parameters
    ----------
    cells_dict : dict
        {cell: {"epsps": {trial_id: 1D array}, "spike_train": {trial_id: 1D array}}}
    Vrest : float
        Resting potential (mV).
    epsp_sf : float
        Scale factor for EPSP sum → Vm.

    Returns
    -------
    dend_Vm : (n_trials, T) float32
        Vrest + epsp_sf * (sum across cells, per-trial centered).
    sum_epsp_centered : (n_trials, T) float32
        Summed EPSPs across cells (masked by NaN) and centered per trial.
    spikes_per_cell : {cell: (n_trials, T_spk) float32}
        Spike trains per cell, aligned & padded across trials (NaN for padding).
    trial_ids : list
        The trial order used for rows in the outputs.
    """

    def _trial_sort_key(x):
        # ints first in numeric order, then strings
        try:
            return (0, int(x))
        except (ValueError, TypeError):
            return (1, str(x))

    if not cells_dict:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, 0), dtype=np.float32),
                {},
                [])

    # ---- discover global trial set and global time lengths
    trial_ids = set()
    global_T_epsp = 0
    global_T_spk  = 0
    for cell, payload in cells_dict.items():
        ceps = payload["epsps"]
        cspk = payload["spike_train"]
        trial_ids.update(ceps.keys())
        for v in ceps.values():
            global_T_epsp = max(global_T_epsp, len(v))
        for v in cspk.values():
            global_T_spk = max(global_T_spk, len(v))

    trial_ids = sorted(trial_ids, key=_trial_sort_key)
    n_trials = len(trial_ids)

    # ---- build (n_trials, T) per-cell EPSP and spike matrices (padded with NaN)
    cell_epsp_mats = []
    spikes_per_cell = {}
    for cell, payload in cells_dict.items():
        ceps = payload["epsps"]
        cspk = payload["spike_train"]

        epsp_rows, spk_rows = [], []
        for tid in trial_ids:
            ev = np.asarray(ceps.get(tid, []), dtype=np.float32)
            sv = np.asarray(cspk.get(tid, []), dtype=np.float32)

            if ev.size < global_T_epsp:
                ev = np.pad(ev, (0, global_T_epsp - ev.size), constant_values=np.nan)
            if sv.size < global_T_spk:
                sv = np.pad(sv, (0, global_T_spk - sv.size), constant_values=np.nan)

            epsp_rows.append(ev)
            spk_rows.append(sv)

        epsp_mat = np.vstack(epsp_rows).astype(np.float32, copy=False)  # (n_trials, global_T_epsp)
        spk_mat  = np.vstack(spk_rows).astype(np.float32, copy=False)   # (n_trials, global_T_spk)

        cell_epsp_mats.append(epsp_mat)
        spikes_per_cell[cell] = spk_mat

    # ---- stack cells -> (n_cells, n_trials, T) and masked-sum across cells
    epsp_stack = np.stack(cell_epsp_mats, axis=0)           # (n_cells, n_trials, global_T_epsp)
    valid_counts = np.sum(~np.isnan(epsp_stack), axis=0)    # (n_trials, T)
    sum_epsp = np.nansum(epsp_stack, axis=0)                # (n_trials, T)
    sum_epsp[valid_counts == 0] = np.nan                    # keep NaN where no cell contributed

    # ---- per-trial centering (across time)
    trial_means = np.nanmean(sum_epsp, axis=1, keepdims=True)  # (n_trials, 1)
    sum_epsp_centered = sum_epsp - trial_means

    # ---- dendritic Vm
    dend_Vm = Vrest + epsp_sf * sum_epsp_centered

    return dend_Vm, sum_epsp_centered, spikes_per_cell

def bin_vm_time_to_position_nans(
    Vm_trials,                       # (n_trials, T) Vm vs time
    dt_ms,                           # ms per sample (e.g., 1.0)
    vel_pos_trial_m_s,               # (n_pos, n_trials) velocity by position bin (m/s)
    track_len_cm=180.0,
    sample_variance=False,
    occupancy_mode="velocity",       # "velocity" or "samples"
    min_vel_cm_s=1e-3*100            # minimum velocity (cm/s) to avoid div-by-zero
):
    """
    Returns:
      centers_cm: (n_pos,)
      Vm_mean:    (n_trials, n_pos)
      Vm_var:     (n_trials, n_pos)
      occ_ms:     (n_trials, n_pos)
    """
    n_trials, T = Vm_trials.shape
    n_pos, n_trials_v = vel_pos_trial_m_s.shape
    if n_trials != n_trials_v:
        raise ValueError(f"Trials mismatch: Vm has {n_trials}, vel has {n_trials_v}")

    edges_cm   = np.linspace(0.0, track_len_cm, n_pos + 1)
    centers_cm = 0.5 * (edges_cm[:-1] + edges_cm[1:])
    bin_width_cm = track_len_cm / n_pos

    t_ms = np.arange(T, dtype=float) * float(dt_ms)

    Vm_mean = np.full((n_trials, n_pos), np.nan, dtype=float)
    Vm_var  = np.full((n_trials, n_pos), np.nan, dtype=float)
    occ_ms  = np.zeros((n_trials, n_pos), dtype=float)

    # helper to fill NaNs and nonpositive in velocity along the position axis (per trial)
    def sanitize_velocity(v_cm_s):
        v = np.array(v_cm_s, dtype=float, copy=True)
        bad = ~np.isfinite(v) | (v <= 0)
        if bad.any():
            good_idx = np.flatnonzero(~bad)
            bad_idx  = np.flatnonzero(bad)
            if good_idx.size >= 2:
                v[bad] = np.interp(bad_idx, good_idx, v[good_idx])
            elif good_idx.size == 1:
                v[bad] = v[good_idx[0]]
            else:
                # no good data: fall back to a safe speed (10 cm/s)
                v[:] = 10.0
        # enforce a minimum positive velocity
        v = np.maximum(v, min_vel_cm_s)
        return v

    for tr in range(n_trials):
        # 1) velocity (cm/s), filled across position
        v_cm_s = sanitize_velocity(vel_pos_trial_m_s[:, tr] * 100.0)

        # 2) time per position bin (ms) from velocity
        dt_per_bin_ms = (bin_width_cm / v_cm_s) * 1000.0
        edges_ms = np.concatenate(([0.0], np.cumsum(dt_per_bin_ms)))
        if not np.isfinite(edges_ms[-1]):
            # last guard; shouldn't trigger after sanitize_velocity
            continue

        # 3) valid Vm samples for this trial: inside the lap AND finite
        valid_time = t_ms < edges_ms[-1]
        valid_vm   = np.isfinite(Vm_trials[tr])
        valid = valid_time & valid_vm
        if not np.any(valid):
            # no data this trial
            if occupancy_mode == "velocity":
                occ_ms[tr, :] = dt_per_bin_ms
            continue

        vm = Vm_trials[tr, valid]
        tm = t_ms[valid]

        # 4) assign each sample to a position bin
        idx = np.searchsorted(edges_ms, tm, side="right") - 1
        idx = np.clip(idx, 0, n_pos - 1)

        # 5) per-bin stats
        dd = 1 if sample_variance else 0
        for k in range(n_pos):
            vals = vm[idx == k]
            n = vals.size
            if n:
                Vm_mean[tr, k] = np.nanmean(vals)
                Vm_var[tr, k]  = np.nanvar(vals, ddof=dd)
                if occupancy_mode == "samples":
                    occ_ms[tr, k]  = n * dt_ms

        # 6) occupancy from velocity if requested (recommended)
        if occupancy_mode == "velocity":
            occ_ms[tr, :] = dt_per_bin_ms

    return centers_cm, Vm_mean, Vm_var, occ_ms

def plot_dendrite_spikes_multiple_seeds(dend_Vm_dict, an_velocity, residual_activity_dict_EC, animal, animal_by_animal=False, dend_threshold=None, tau=None, num_seeds=None):

    fig, axs = plt.subplots(4,4, figsize=(20,20))

    plt.suptitle(f'Tau {tau}ms Num Seeds {num_seeds} Animal: {animal}')

    def get_summed_dendrite_EC_DFF(residual_activity_dict_EC, animal_by_animal=animal_by_animal):
        if animal_by_animal:
            data_list = []
            for cell in residual_activity_dict_EC:
                data_list.append(residual_activity_dict_EC[cell][:,:58])

            data_array = np.array(data_list)
            summed_dendrite = np.sum(data_array, axis=0)
        else:
            data_list = []
            for animal in residual_activity_dict_EC:
                for cell in residual_activity_dict_EC[animal]:
                    data_list.append(residual_activity_dict_EC[animal][cell][:,:58])

            data_array = np.array(data_list)
            summed_dendrite = np.sum(data_array, axis=0)

        return summed_dendrite

    # def get_summed_dendrite_EC_DFF(residual_activity_dict_EC):
    #     data_list = []
    #     for animal in residual_activity_dict_EC:
    #         for cell in residual_activity_dict_EC[animal]:
    #             data_list.append(residual_activity_dict_EC[animal][cell][:,:58])

    #     data_array = np.array(data_list)
    #     summed_dendrite = np.sum(data_array, axis=0)
    #     return summed_dendrite
    
    dend_Vm_list = []

    for seed in dend_Vm_dict:
        dend_Vm_list.append(dend_Vm_dict[seed])

    dend_Vm_array = np.array(dend_Vm_list)

    mean_dend_seeds = np.nanmean(dend_Vm_array, axis=0)

    im=axs[1,0].imshow(mean_dend_seeds, aspect='auto', interpolation='none')
    axs[1,0].set_ylabel("Trial")
    axs[1,0].set_xlabel("Time (ms)")
    axs[1,0].set_title("Mean Dendrite Vm over Seeds as Sum of Spiking EC")
    fig.colorbar(im, ax=axs[1,0], label="mV")

    mean_dendrite = np.nanmean(mean_dend_seeds, axis=0)
    sem_dendrite = sem(mean_dend_seeds, axis=0, nan_policy='omit')

    axs[1,1].plot(mean_dendrite)
    axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2)
    axs[1,1].set_title("Trial Averaged Dendrite Vm")
    axs[1,1].set_xlabel("Time (ms)")
    axs[1,1].set_ylabel("mV")

    # centers_cm, Vm_binned, occ_ms = bin_vm_time_to_position(mean_dend_seeds, dt_ms=1.0, vel_pos_trial_m_s=an_velocity)

    # centers_cm, Vm_mean, Vm_var, occ_ms = bin_vm_time_to_position(mean_dend_seeds, dt_ms=1.0, vel_pos_trial_m_s=an_velocity)
    centers_cm, Vm_mean, Vm_var, occ_ms  = bin_vm_time_to_position_nans(mean_dend_seeds, dt_ms=1.0, vel_pos_trial_m_s=an_velocity, track_len_cm=180.0, sample_variance=False, occupancy_mode="velocity", min_vel_cm_s=1e-3*100)
    Vm_mean_across_trials = np.mean(Vm_mean, axis=0) 
    Vm_sem_across_trials = sem(Vm_mean, axis=0) 

    im = axs[1,2].imshow(Vm_mean, aspect='auto')
    axs[1,2].set_ylabel("Trials")
    axs[1,2].set_xlabel("Position Bins")
    axs[1,2].set_title("Spiking Model Mean Across Seeds Binned")
    fig.colorbar(im, ax=axs[1,2])

    axs[1,3].plot(Vm_mean_across_trials)
    axs[1,3].fill_between(range(len(Vm_mean_across_trials)), Vm_mean_across_trials+Vm_sem_across_trials, Vm_mean_across_trials-Vm_sem_across_trials, alpha=0.2)
    axs[1,3].set_ylabel("Vm (mV)")
    axs[1,3].set_xlabel("Position Bins")
    axs[1,3].set_title("Trial Averaged Binned Spiking Model Across Seeds")




    summed_dendrite = get_summed_dendrite_EC_DFF(residual_activity_dict_EC)

    im = axs[0,2].imshow(summed_dendrite.T, aspect='auto')
    axs[0,2].set_title(f"Dendrite DF/F Animal: {animal}")
    axs[0,2].set_ylabel("Trials")
    axs[0,2].set_xlabel("Position Bins")
    fig.colorbar(im, ax=axs[0,2], label="Z-Scored Summed DF/F")

    mean_summed_dend = np.mean(summed_dendrite, axis=1)
    sem_summed_dend = sem(summed_dendrite, axis=1, nan_policy='omit')

    axs[0,3].plot(mean_summed_dend)
    axs[0,3].fill_between(range(len(mean_summed_dend)), mean_summed_dend-sem_summed_dend, mean_summed_dend+sem_summed_dend, alpha=0.2)
    axs[0,3].set_title(f"Dendrite Trial Averaged DF/F Animal: {animal}")
    axs[0,3].set_xlabel("Position Bins")
    axs[0,3].set_ylabel('Z-Scored Summed DF/F')


    padded_warped_activity_EC, summed_dendrite, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=0.001, dend_threshold=dend_threshold, vel_applied="real")

    im = axs[0,0].imshow(padded_warped_activity_EC, aspect='auto')
    axs[0,0].set_title("Dendrite DF/F")
    axs[0,0].set_ylabel("Trials")
    axs[0,0].set_xlabel("Time (ms)")
    fig.colorbar(im, ax=axs[0,0], label="Z-Scored Summed DF/F")


    # mean_curve = np.nanmean(padded_warped_activity_EC[:,:array_time_len], axis=0)
    # sem_curve  = sem(padded_warped_activity_EC[:,:array_time_len], axis=0, ddof=1, nan_policy='omit')

    mean_curve = np.nanmean(padded_warped_activity_EC, axis=0)
    sem_curve  = sem(padded_warped_activity_EC, axis=0, ddof=1, nan_policy='omit')

    axs[0,1].plot(mean_curve)
    axs[0,1].fill_between(range(len(mean_curve)),
                        mean_curve - sem_curve,
                        mean_curve + sem_curve,
                        alpha=0.2)
    axs[0,1].set_title("Dendrite Trial Averaged DF/F")
    axs[0,1].set_xlabel("Time (ms)")
    axs[0,1].set_ylabel("Z-Scored Summed DF/F")



    
    im = axs[2,0].imshow(an_velocity.T, aspect='auto')
    axs[2,0].set_title("Velocity")
    axs[2,0].set_xlabel("Position Bins")
    axs[2,0].set_ylabel("Trials")
    fig.colorbar(im, ax=axs[2,0])


    total_track_length=1.8 #meters
    num_pos_bins=50
    dist_per_pos_bin = total_track_length / num_pos_bins
    occupancy = dist_per_pos_bin/an_velocity

    axs[3,0].plot(occupancy, color='r', linewidth=2)
    axs[3,0].set_title("Occupancy")
    axs[3,0].set_xlabel("Position Bins")
    axs[3,0].set_ylabel("Seconds")
    
    just_plateau_starts_sums_dict, plateau_array_dict = get_plateau_array_dict(dend_Vm_dict, dend_threshold)


    im = axs[2,1].imshow(plateau_array_dict[0], aspect='auto', interpolation='none', cmap='grey')
    axs[2,1].set_title(f"Example Seed Plateaus \n Dendrite Threshold={dend_threshold}")
    axs[2,1].set_ylabel("Trial")
    axs[2,1].set_xlabel("Time (ms)")
    fig.colorbar(im, ax=axs[2,1])


    just_plateau_starts_sums_list = [just_plateau_starts_sums_dict[seed] for seed in just_plateau_starts_sums_dict]
    just_plateau_starts_sums_array = np.array(just_plateau_starts_sums_list)

    mean_vals = np.nanmean(just_plateau_starts_sums_array, axis=0)
    sem_vals  = sem(just_plateau_starts_sums_array, axis=0, nan_policy='omit')

    x = np.arange(mean_vals.shape[0])

    # axs[2,2].plot(x, mean_vals, color='k', linewidth=2, label="Mean across seeds")
    # axs[2,2].fill_between(x, mean_vals - sem_vals, mean_vals + sem_vals, color='gray', alpha=0.2, label="SEM")
    axs[2,2].errorbar(range(len(mean_vals)), mean_vals, yerr=sem_vals, fmt='o-', color='k', capsize=2, markersize=5)
    axs[2,2].set_title("Plateau Count Over Trials")
    axs[2,2].set_ylabel("Plateau Count")
    axs[2,2].set_xlabel("Session Length (%)")
    axs[2,2].set_xticks([0, len(x)//4, len(x)//2, 3*len(x)//4, len(x)-1], labels=["0", "25", "50", "75", "100"])
    axs[2,2].legend()



    cumulative_plateau_counts_array = np.cumsum(just_plateau_starts_sums_array, axis=0)

    mean_cumulative_plateau_counts_array = np.nanmean(cumulative_plateau_counts_array, axis=0)
    sem_cumulative_plateau_counts_array = sem(cumulative_plateau_counts_array, axis=0, nan_policy='omit')

    # cumulative_plateau_counts = np.cumsum(just_plateau_starts_sums)
    axs[2,3].errorbar(range(len(mean_cumulative_plateau_counts_array)), mean_cumulative_plateau_counts_array, yerr=sem_cumulative_plateau_counts_array, fmt='o-', color='k', capsize=2, markersize=5)
    # axs[2,3].plot(mean_cumulative_plateau_counts_array, color='k', linewidth=4)
    # axs[2,3].fill_between(range(len(mean_cumulative_plateau_counts_array)), mean_cumulative_plateau_counts_array+sem_cumulative_plateau_counts_array, mean_cumulative_plateau_counts_array-sem_cumulative_plateau_counts_array, alpha=0.2)
    axs[2,3].set_title("Cumulative Plateau Count Over Trials \n Averaged Over Seeds")
    axs[2,3].set_ylabel("Cumulative Plateau Count")
    axs[2,3].set_xlabel("Session Length (%)")
    axs[2,3].set_xticks([0, len(cumulative_plateau_counts)//4, len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts)//4 + len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts) - 1], 
                        labels=["0", '25', "50", '75', "100"])



    track_len_cm = 180.0
    n_pos = an_velocity.shape[0]    

    dx = track_len_cm / n_pos       

    dt_constant = 1.0 / 1000.0  

    summed_plateaus_list = []

    plateau_counts_per_time_list = []


    for seed in plateau_array_dict:
        indiv_plateau_array = plateau_array_dict[seed]
        plateau_counts_per_time = np.sum(indiv_plateau_array, axis=0)
        plateau_counts_per_time_list.append(plateau_counts_per_time)

        starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, indiv_plateau_array, dx=dx, dt_constant=dt_constant)

        n_bins = 5
        bin_size = int(50 / n_bins)

        summed_plateaus = np.zeros(n_bins)

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(plateau_start_positions_counter[start:end])
            summed_plateaus[i] = summed_data

        summed_plateaus_list.append(summed_plateaus)

    summed_plateaus_array = np.array(summed_plateaus_list)
    mean_summed_plateaus_array = np.nanmean(summed_plateaus_array, axis=0)
    sem_summed_plateaus_array = sem(summed_plateaus_array, axis=0, nan_policy='omit')        

    plateau_counts_per_time_array = np.array(plateau_counts_per_time_list)

    mean_plateau_counts_per_time_array = np.nanmean(plateau_counts_per_time_array, axis=0)
    sem_plateau_counts_per_time_array = sem(plateau_counts_per_time_array, axis=0, nan_policy='omit')

    # dt = 1 ms from your dt_constant, so 20 samples per 20 ms bin
    samples_per_bin = 1000
    T = plateau_counts_per_time_array.shape[1]
    trim = (T // samples_per_bin) * samples_per_bin

    # drop tail so reshape works
    A = plateau_counts_per_time_array[:, :trim]
    A = A.reshape(A.shape[0], -1, samples_per_bin).sum(axis=2)  # (n_seeds, n_time_bins)

    mean_time_binned = np.nanmean(A, axis=0)
    sem_time_binned  = sem(A, axis=0, nan_policy='omit')
    x_time = np.arange(mean_time_binned.size)

    mean_Vm_var = np.nanmean(Vm_var, axis=0)
    sem_Vm_var = sem(Vm_var, axis=0, nan_policy='omit')
    axs[3,1].bar(range(len(mean_Vm_var)), mean_Vm_var, yerr=sem_Vm_var, capsize=0.5, error_kw={'lw':2, 'capthick':0.2})
    axs[3,1].set_xlabel("Position Bin")
    axs[3,1].set_ylabel("Mean +-SEM Across Trials")
    axs[3,1].set_title("Mean Seeds Dend Vm Variance Across Trials")
    # axs[3,1].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])


    axs[3,2].bar(range(len(mean_summed_plateaus_array)), mean_summed_plateaus_array, yerr=sem_summed_plateaus_array, edgecolor='k', capsize=6, error_kw={'lw':2, 'capthick':2})
    axs[3,2].set_xlabel("Position Bin")
    axs[3,2].set_ylabel("Plateau Count")
    axs[3,2].set_title("Plateau Count per Track Section")
    axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])

    fraction_plateaus_list = []

    for i in range(len(summed_plateaus_list)):
        summed_plateaus = summed_plateaus_list[i]
        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        fraction_plateaus_list.append(fraction_plateaus)

    fraction_plateaus_array = np.array(fraction_plateaus_list)
    mean_fraction_plateaus_array = np.nanmean(fraction_plateaus_array, axis=0)
    sem_fraction_plateaus_array = sem(fraction_plateaus_array, axis=0, nan_policy='omit')

    # axs[3,3].plot(mean_fraction_plateaus_array*100, marker='o', yerr=sem_fraction_plateaus_array, color='k', markersize=7)
    axs[3,3].errorbar(range(len(mean_fraction_plateaus_array)), mean_fraction_plateaus_array*100, yerr=sem_fraction_plateaus_array*100, fmt='o-', color='k', capsize=4, markersize=7)
    axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
    axs[3,3].set_xlabel("Grouped Position Bins")
    axs[3,3].set_ylabel("% of Total Plateaus")
    axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])


    plt.tight_layout()
    plt.show()


 
# def plot_dendrite_spikes_multiple_seeds(dend_Vm_dict, an_velocity, residual_activity_dict_EC, padded_warped_activity_EC, summed_dendrite, just_plateau_starts_sums_dict, plateau_array_dict, dend_threshold=None, tau=None, num_seeds=None):

#     fig, axs = plt.subplots(4,4, figsize=(20,20))

#     plt.suptitle(f'Tau {tau}ms Num Seeds {num_seeds}')

#     def get_summed_dendrite_EC_DFF(residual_activity_dict_EC):
#         data_list = []
#         for animal in residual_activity_dict_EC:
#             for cell in residual_activity_dict_EC[animal]:
#                 data_list.append(residual_activity_dict_EC[animal][cell][:,:58])

#         data_array = np.array(data_list)
#         summed_dendrite = np.sum(data_array, axis=0)
#         return summed_dendrite
    
#     summed_dendrite = get_summed_dendrite_EC_DFF(residual_activity_dict_EC)

#     im = axs[0,2].imshow(summed_dendrite.T, aspect='auto')
#     axs[0,2].set_title("Dendrite DF/F")
#     axs[0,2].set_ylabel("Trials")
#     axs[0,2].set_xlabel("Position Bins")
#     fig.colorbar(im, ax=axs[0,2], label="Z-Scored Summed DF/F")

#     mean_summed_dend = np.mean(summed_dendrite, axis=1)
#     sem_summed_dend = sem(summed_dendrite, axis=1, nan_policy='omit')

#     axs[0,3].plot(mean_summed_dend)
#     axs[0,3].fill_between(range(len(mean_summed_dend)), mean_summed_dend-sem_summed_dend, mean_summed_dend+sem_summed_dend, alpha=0.2)
#     axs[0,3].set_title("Dendrite Trial Averaged DF/F")
#     axs[0,3].set_xlabel("Position Bins")
#     axs[0,3].set_ylabel('Z-Scored Summed DF/F')


#     im = axs[0,0].imshow(padded_warped_activity_EC, aspect='auto')
#     axs[0,0].set_title("Dendrite DF/F")
#     axs[0,0].set_ylabel("Trials")
#     axs[0,0].set_xlabel("Time (ms)")
#     fig.colorbar(im, ax=axs[0,0], label="Z-Scored Summed DF/F")


#     mean_padded_warped_activity_EC = np.nanmean(padded_warped_activity_EC, axis=0)
#     sem_padded_warped_activity_EC = sem(padded_warped_activity_EC, axis=0, nan_policy='omit')

#     axs[0,1].plot(mean_padded_warped_activity_EC)
#     axs[0,1].fill_between(range(len(mean_padded_warped_activity_EC)), mean_padded_warped_activity_EC-sem_padded_warped_activity_EC, mean_padded_warped_activity_EC+sem_padded_warped_activity_EC, alpha=0.2)
#     axs[0,1].set_title("Dendrite Trial Averaged DF/F")
#     axs[0,1].set_xlabel("Time (ms)")
#     axs[0,1].set_ylabel('Z-Scored Summed DF/F')

#     dend_Vm_list = []

#     for seed in dend_Vm_dict:
#         dend_Vm_list.append(dend_Vm_dict[seed])

#     dend_Vm_array = np.array(dend_Vm_list)

#     mean_dend_seeds = np.mean(dend_Vm_array, axis=0)

#     im=axs[1,0].imshow(mean_dend_seeds, aspect='auto', interpolation='none')
#     axs[1,0].set_ylabel("Trial")
#     axs[1,0].set_xlabel("Time (ms)")
#     axs[1,0].set_title("Mean Dendrite Vm over Seeds as Sum of Spiking EC")
#     fig.colorbar(im, ax=axs[1,0], label="mV")

#     mean_dendrite = np.nanmean(mean_dend_seeds, axis=0)
#     sem_dendrite = sem(mean_dend_seeds, axis=0, nan_policy='omit')

#     axs[1,1].plot(mean_dendrite)
#     axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2)
#     axs[1,1].set_title("Trial Averaged Dendrite Vm")
#     axs[1,1].set_xlabel("Time (ms)")
#     axs[1,1].set_ylabel("mV")

#     # centers_cm, Vm_binned, occ_ms = bin_vm_time_to_position(mean_dend_seeds, dt_ms=1.0, vel_pos_trial_m_s=an_velocity)
#     centers_cm, Vm_mean, Vm_var, occ_ms = bin_vm_time_to_position(mean_dend_seeds, dt_ms=1.0, vel_pos_trial_m_s=an_velocity)
#     Vm_mean_across_trials = np.nanmean(Vm_mean, axis=0) 
#     Vm_sem_across_trials = sem(Vm_mean, axis=0, nan_policy='omit') 

#     im = axs[1,2].imshow(Vm_mean, aspect='auto')
#     axs[1,2].set_ylabel("Trials")
#     axs[1,2].set_xlabel("Position Bins")
#     axs[1,2].set_title("Spiking Model Mean Across Seeds Binned")
#     fig.colorbar(im, ax=axs[1,2])

#     axs[1,3].plot(Vm_mean_across_trials)
#     axs[1,3].fill_between(range(len(Vm_mean_across_trials)), Vm_mean_across_trials+Vm_sem_across_trials, Vm_mean_across_trials-Vm_sem_across_trials, alpha=0.2)
#     axs[1,3].set_ylabel("Vm (mV)")
#     axs[1,3].set_xlabel("Position Bins")
#     axs[1,3].set_title("Trial Averaged Binned Spiking Model Across Seeds")

#     im = axs[2,0].imshow(an_velocity.T, aspect='auto')
#     axs[2,0].set_title("Velocity")
#     axs[2,0].set_xlabel("Position Bins")
#     axs[2,0].set_ylabel("Trials")
#     fig.colorbar(im, ax=axs[2,0])


#     total_track_length=1.8 #meters
#     num_pos_bins=50
#     dist_per_pos_bin = total_track_length / num_pos_bins
#     occupancy = dist_per_pos_bin/an_velocity

#     axs[3,0].plot(occupancy, color='r', linewidth=2)
#     axs[3,0].set_title("Occupancy")
#     axs[3,0].set_xlabel("Position Bins")
#     axs[3,0].set_ylabel("Seconds")
    


#     im = axs[2,1].imshow(plateau_array_dict[0], aspect='auto', interpolation='none', cmap='grey')
#     axs[2,1].set_title(f"Example Seed Plateaus \n Dendrite Threshold={dend_threshold}")
#     axs[2,1].set_ylabel("Trial")
#     axs[2,1].set_xlabel("Time (ms)")
#     fig.colorbar(im, ax=axs[2,1])



    


#     # just_plateau_starts_sums_list = [just_plateau_starts_sums_dict[seed] for seed in just_plateau_starts_sums_dict]
#     # just_plateau_starts_sums_array = np.array(just_plateau_starts_sums_list)

#     # mean_vals = np.nanmean(just_plateau_starts_sums_array, axis=0)
#     # sem_vals  = sem(just_plateau_starts_sums_array, axis=0, nan_policy='omit')

#     # n_trials = mean_vals.shape[0]

#     # x = np.arange(mean_vals.shape[0])

#     # # axs[2,2].plot(x, mean_vals, color='k', linewidth=2, label="Mean across seeds")
#     # # axs[2,2].fill_between(x, mean_vals - sem_vals, mean_vals + sem_vals, color='gray', alpha=0.2, label="SEM")
#     # axs[2,2].errorbar(range(len(mean_vals)), mean_vals, yerr=sem_vals, fmt='o-', color='k', capsize=2, markersize=5)
#     # axs[2,2].set_title("Plateau Count Over Trials")
#     # axs[2,2].set_ylabel("Plateau Count")
#     # axs[2,2].set_xlabel("Session Length (%)")
#     # axs[2,2].set_xticks([0, len(x)//4, len(x)//2, 3*len(x)//4, len(x)-1], labels=["0", "25", "50", "75", "100"])
#     # axs[2,2].legend()



#     # cumulative_plateau_counts_array = np.cumsum(just_plateau_starts_sums_array, axis=0)

#     # mean_cumulative_plateau_counts_array = np.nanmean(cumulative_plateau_counts_array, axis=0)
#     # sem_cumulative_plateau_counts_array = sem(cumulative_plateau_counts_array, axis=0, nan_policy='omit')

#     # # cumulative_plateau_counts = np.cumsum(just_plateau_starts_sums)
#     # axs[2,3].errorbar(range(len(mean_cumulative_plateau_counts_array)), mean_cumulative_plateau_counts_array, yerr=sem_cumulative_plateau_counts_array, fmt='o-', color='k', capsize=2, markersize=5)
#     # # axs[2,3].plot(mean_cumulative_plateau_counts_array, color='k', linewidth=4)
#     # # axs[2,3].fill_between(range(len(mean_cumulative_plateau_counts_array)), mean_cumulative_plateau_counts_array+sem_cumulative_plateau_counts_array, mean_cumulative_plateau_counts_array-sem_cumulative_plateau_counts_array, alpha=0.2)
#     # axs[2,3].set_title("Cumulative Plateau Count Over Trials \n Averaged Over Seeds")
#     # axs[2,3].set_ylabel("Cumulative Plateau Count")
#     # axs[2,3].set_xlabel("Session Length (%)")
#     # axs[2,3].set_xticks([0, len(n_trials)//4, len(n_trials)//2, len(n_trials)//4 + len(n_trials)//2, len(n_trials) - 1], 
#     #                     labels=["0", '25', "50", '75', "100"])


#     just_plateau_starts_sums_list = [just_plateau_starts_sums_dict[s] for s in sorted(just_plateau_starts_sums_dict)]
#     just_plateau_starts_sums_array = np.asarray(just_plateau_starts_sums_list, dtype=float)  # (n_seeds, n_trials)

#     # ----- per-trial mean ± sem (plateau starts)
#     mean_vals = np.nanmean(just_plateau_starts_sums_array, axis=0)
#     sem_vals  = sem(just_plateau_starts_sums_array, axis=0, nan_policy='omit')
#     n_trials = mean_vals.shape[0]
#     axs[2,2].errorbar(range(n_trials), mean_vals, yerr=sem_vals, fmt='o-', color='k', capsize=2, markersize=5)
#     axs[2,2].set_title("Plateau Count Over Trials")
#     axs[2,2].set_ylabel("Plateau Count")
#     axs[2,2].set_xlabel("Session Length (%)")
#     axs[2,2].set_xticks([0, n_trials//4, n_trials//2, 3*n_trials//4, n_trials-1],
#                         labels=["0", "25", "50", "75", "100"])
#     axs[2,2].legend()

#     # ----- cumulative per seed -> mean ± sem across seeds
#     cumulative_plateau_counts_array = np.cumsum(just_plateau_starts_sums_array, axis=0)  # (n_seeds, n_trials)
#     mean_cum = np.nanmean(cumulative_plateau_counts_array, axis=0)
#     sem_cum  = sem(cumulative_plateau_counts_array, axis=0, nan_policy='omit')
#     axs[2,3].errorbar(range(n_trials), mean_cum, yerr=sem_cum, fmt='o-', color='k', capsize=2, markersize=5)
#     axs[2,3].set_title("Cumulative Plateau Count Over Trials\nAveraged Over Seeds")
#     axs[2,3].set_ylabel("Cumulative Plateau Count")
#     axs[2,3].set_xlabel("Session Length (%)")
#     axs[2,3].set_xticks([0, n_trials//4, n_trials//2, 3*n_trials//4, n_trials-1],
#                         labels=["0", "25", "50", "75", "100"])


#     track_len_cm = 180.0
#     n_pos = an_velocity.shape[0]    

#     dx = track_len_cm / n_pos       

#     dt_constant = 1.0 / 1000.0  

#     summed_plateaus_list = []

#     plateau_counts_per_time_list = []


#     for seed in plateau_array_dict:
#         indiv_plateau_array = plateau_array_dict[seed]
#         plateau_counts_per_time = np.sum(indiv_plateau_array, axis=0)
#         plateau_counts_per_time_list.append(plateau_counts_per_time)

#         starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, indiv_plateau_array, dx=dx, dt_constant=dt_constant)

#         n_bins = 5
#         bin_size = int(50 / n_bins)

#         summed_plateaus = np.zeros(n_bins)

#         for i in range(n_bins):
#             start = i * bin_size
#             end = (i + 1) * bin_size
#             summed_data = np.sum(plateau_start_positions_counter[start:end])
#             summed_plateaus[i] = summed_data

#         summed_plateaus_list.append(summed_plateaus)

#     summed_plateaus_array = np.array(summed_plateaus_list)
#     mean_summed_plateaus_array = np.nanmean(summed_plateaus_array, axis=0)
#     sem_summed_plateaus_array = sem(summed_plateaus_array, axis=0, nan_policy='omit')        

#     plateau_counts_per_time_array = np.array(plateau_counts_per_time_list)

#     mean_plateau_counts_per_time_array = np.nanmean(plateau_counts_per_time_array, axis=0)
#     sem_plateau_counts_per_time_array = sem(plateau_counts_per_time_array, axis=0, nan_policy='omit')

#     # dt = 1 ms from your dt_constant, so 20 samples per 20 ms bin
#     samples_per_bin = 1000
#     T = plateau_counts_per_time_array.shape[1]
#     trim = (T // samples_per_bin) * samples_per_bin

#     # drop tail so reshape works
#     A = plateau_counts_per_time_array[:, :trim]
#     A = A.reshape(A.shape[0], -1, samples_per_bin).sum(axis=2)  # (n_seeds, n_time_bins)

#     mean_time_binned = np.nanmean(A, axis=0)
#     sem_time_binned  = sem(A, axis=0, nan_policy='omit')
#     x_time = np.arange(mean_time_binned.size)

#     mean_Vm_var = np.nanmean(Vm_var, axis=0)
#     sem_Vm_var = sem(Vm_var, axis=0, nan_policy='omit')
#     axs[3,1].bar(range(len(mean_Vm_var)), mean_Vm_var, yerr=sem_Vm_var, capsize=0.5, error_kw={'lw':2, 'capthick':0.2})
#     axs[3,1].set_xlabel("Position Bin")
#     axs[3,1].set_ylabel("Mean +-SEM Across Trials")
#     axs[3,1].set_title("Mean Seeds Dend Vm Variance Across Trials")
#     # axs[3,1].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])


#     axs[3,2].bar(range(len(mean_summed_plateaus_array)), mean_summed_plateaus_array, yerr=sem_summed_plateaus_array, edgecolor='k', capsize=6, error_kw={'lw':2, 'capthick':2})
#     axs[3,2].set_xlabel("Position Bin")
#     axs[3,2].set_ylabel("Plateau Count")
#     axs[3,2].set_title("Plateau Count per Track Section")
#     axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])

#     fraction_plateaus_list = []

#     for i in range(len(summed_plateaus_list)):
#         summed_plateaus = summed_plateaus_list[i]
#         total_plateaus = np.sum(summed_plateaus)
#         fraction_plateaus = summed_plateaus / total_plateaus
#         fraction_plateaus_list.append(fraction_plateaus)

#     fraction_plateaus_array = np.array(fraction_plateaus_list)
#     mean_fraction_plateaus_array = np.nanmean(fraction_plateaus_array, axis=0)
#     sem_fraction_plateaus_array = sem(fraction_plateaus_array, axis=0, nan_policy='omit')

#     # axs[3,3].plot(mean_fraction_plateaus_array*100, marker='o', yerr=sem_fraction_plateaus_array, color='k', markersize=7)
#     axs[3,3].errorbar(range(len(mean_fraction_plateaus_array)), mean_fraction_plateaus_array*100, yerr=sem_fraction_plateaus_array*100, fmt='o-', color='k', capsize=4, markersize=7)
#     axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
#     axs[3,3].set_xlabel("Grouped Position Bins")
#     axs[3,3].set_ylabel("% of Total Plateaus")
#     axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])


#     plt.tight_layout()
#     plt.show()



# def do_the_interpolation(scaled_data_Hz_dict, an_velocity=None):
    
#     padded_warped_activity_dict = {}

#     dt_constant = 0.001

#     for animal in scaled_data_Hz_dict:
#         padded_cell = {}
#         for cell in scaled_data_Hz_dict[animal]:

#             summed_dendrite = scaled_data_Hz_dict[animal][cell]

#             # if vel_applied == 'constant':
#             #     an_velocity = np.full((summed_dendrite.shape), 0.43) #0.43 meters per second animal velocity 
#             # else:
#             an_velocity = an_velocity

#             total_time_sec = 4.71657036 

#             dt=total_time_sec/50
#             dx=180/50

#             proper_velocity=an_velocity*100

#             animal_velocity_constant= np.full((summed_dendrite.shape), dx/dt)

#             # if vel_applied=="constant":
#             #     dt = dx / animal_velocity_constant
#             # else:
#             dt = dx / proper_velocity

#             time_bins = np.cumsum(dt, axis=0)
#             time_bins_ms = time_bins * 1

#             num_trials = summed_dendrite.shape[1]
#             trial_warped_activity = []
#             max_len = 0

#             for t in range(num_trials):
#                 if np.any(np.isnan(time_bins[:, t])):
#                     continue
#                 total_time = time_bins[-1, t]

#                 time_axis_constant = np.arange(0, total_time, dt_constant)
                
#                 firing = summed_dendrite[:, t]

#                 warped_firing = np.interp(time_axis_constant, time_bins_ms[:,t], firing)

#                 trial_warped_activity.append(warped_firing)
#                 if len(warped_firing) > max_len:
#                     max_len = len(warped_firing)


#             # if vel_applied=="constant":
#             #     padded_warped_activity = np.full((num_trials, max_len), np.nan)
#             #     for i, trace in enumerate(trial_warped_activity):
#             #             padded_warped_activity[i, :len(trace)] = trace 

#             #     padded_cell[cell] = padded_warped_activity
#             # else:
#             padded_cell[cell] = trial_warped_activity
#         padded_warped_activity_dict[animal] = padded_cell

#     return padded_warped_activity_dict, an_velocity


