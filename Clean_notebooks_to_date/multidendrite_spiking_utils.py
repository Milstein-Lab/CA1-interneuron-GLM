
import random

from spiking_model_utils import *


import pathlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pickle
import click
from spiking_model_utils import *
from pathlib import Path
import sys
import platform

def sample_equal_weights(mask, value=1.0):
    weights = np.zeros_like(mask, dtype=float)
    weights[mask] = value
    return weights

def sample_weights(distribution, mask, rng, mean=0.1, std=0.5):
    weights = np.zeros_like(mask, dtype=float)
    n_samples = np.sum(mask)

    if distribution == "Uniform":
        samples = rng.uniform(low=mean - std, high=mean + std, size=n_samples)
    elif distribution == "Normal":
        samples = rng.normal(loc=mean, scale=std, size=n_samples)
        samples = np.clip(samples, 0, None)
    elif distribution == "Lognormal":
        samples = rng.lognormal(mean=np.log(mean), sigma=std, size=n_samples)
    elif distribution == "Equal":
        samples = np.full(n_samples, mean, dtype=float)
    else:
        raise ValueError("Invalid distribution")

    weights[mask] = samples
    return weights

def get_dendrite_activity(weights, EC_input_matrix, n_dendrites, n_EC):
    EC_flat = EC_input_matrix.reshape(n_EC, -1)
    dendrite_flat = weights @ EC_flat
    return dendrite_flat.reshape(n_dendrites, 50, 58)
    
def loss_fn_multi(params, activity_EC, activity_SST, activity_NDNF):
    ndnf_sf, sst_sf = params
    summed = activity_EC - (sst_sf * activity_SST + ndnf_sf * activity_NDNF)
    
    summed_mean = np.mean(summed, axis=0)
    summed_means = np.mean(summed_mean, axis=1)
    summed_overall_means = np.mean(summed_means)
    
    main_loss = (summed_overall_means ** 2)  # This penalizes all deviations from 0

    return main_loss #+ 0.1 * equal_scaling_penalty

def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
    time_series = [initial_value, ]
    for _ in range(count):
        time_series.append(time_series[-1] + initial_value * random.gauss(0, 1) * volatility)
    return time_series
    
def multi_wrap_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, vel_applied='real', add_inh=None, SST_bias_factor=None, dist=None):

    if vel_applied=="real":
        constant_vel=False
        real_vel=True
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist)
    elif vel_applied=="constant":
        constant_vel=True
        real_vel=False
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist)
    else:
        constant_vel=False
        real_vel=False
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=False,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist)

    return an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF
        

    if which_type == "EC_animal_average":
        an_velocity_real_list = []
        for animal in factors_dict_EC:
                an_velocity_real_list.append(factors_dict_EC[animal]["Velocity"][:,:58])

        an_velocity_real_array = np.array(an_velocity_real_list)
        an_velocity_real_array_mean_animal = np.nanmean(an_velocity_real_array, axis=0)
        return an_velocity_real_array_mean_animal

def get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest, which_type=None):

    if which_type == "EC_animal_average":
        an_velocity_real_list = []
        for animal in factors_dict_EC:
            an_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
            an_velocity = sanitize_velocity_cm_s(v_m_s, min_vel_cm_s=1e-3*100)
            an_velocity_real_list.append(an_velocity)

        an_velocity_real_array = np.array(an_velocity_real_list)
        an_velocity_real_array_mean_animal = np.nanmean(an_velocity_real_array, axis=0)
        return an_velocity_real_array_mean_animal
    
def sanitize_velocity_cm_s(v_m_s, min_vel_cm_s=1e-3*100):
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

def get_activity_multidendrite2(animal_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, dt_constant, dx, dend_threshold=20, vel_applied="real", example_cell=15, include_inhibition=None, use_model_EC=False):

    if use_model_EC:
        
        dend_list = []
        for j in range(792):
            ts_list = []
            for i in range(58):
                ts = random_timeseries(1.0, 0.005, 49)
                ts_list.append(ts)

            dend_contribution_EC = np.array(ts_list).T
            dend_list.append(dend_contribution_EC)
            
        EC_input_matrix = np.array(dend_list)


    if include_inhibition == 'both':
        dend_activity = activity_EC - (activity_NDNF*NDNF_sf_opt + activity_SST*SST_sf_opt)

    elif include_inhibition == 'sst':
        dend_activity = activity_EC - (activity_SST*SST_sf_opt)
        
    else:
        dend_activity = activity_EC 

    # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)
        
    plateau_positions_counter = np.zeros(50)
    plateau_start_positions_counter = np.zeros(50)


    plateau_array_per_dendrite_list = []

    num_plateaus_per_dend_list = []

    dendrite_plateau_mask = np.zeros((dend_activity.shape[0], 50), dtype=bool)

    padded_warped_activity_list = []
    
    plateau_start_times_list_mega_list = []

    # n_timebins, n_trials = plateau_array.shape



    for d_idx in range(dend_activity.shape[0]):
 
        position_bins, num_trials = animal_velocity.shape

        padded_warped_activity = activity_EC[d_idx, :, :]

        flat_padded_warped_activity = padded_warped_activity.flatten()
        flat_plateau_array = np.zeros_like(flat_padded_warped_activity)

        i = 0
        while i < len(flat_padded_warped_activity):
            if flat_padded_warped_activity[i] > dend_threshold:
                flat_plateau_array[i:i+300] = 1

                i += 800
            else:
                i += 100


        plateau_array = flat_plateau_array.reshape(padded_warped_activity.shape)
        plateau_array_per_dendrite_list.append(plateau_array)


        plateau_start_times_list = []

        for trial in range(num_trials):
            velocity_trial = animal_velocity[:, trial]
    #         velocity_trial = proper_velocity
            dt_trial = dx / velocity_trial  # in seconds
            time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

            plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
            plateau_start_times = plateau_start_indices * dt_constant  # in seconds
            plateau_start_times_list.append(plateau_start_times)

            for pt_start_time in plateau_start_times:
                if pt_start_time != 0.0:
                    for pos_idx in range(50):
                        if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
                            plateau_start_positions_counter[pos_idx] += 1
                            break

        plateau_start_times_list_mega_list.append(plateau_start_times_list)

        num_plateaus_list = []


        for trial in range(num_trials):
            velocity_trial = animal_velocity[:, trial]
    #         velocity_trial = proper_velocity
            dt_trial = dx / velocity_trial  # in seconds
            time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

            plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]

            plateau_start_times = plateau_start_indices * dt_constant  # in seconds
            num_plateaus_list.append(len(plateau_start_times))

            for pt_start_time in plateau_start_times:
                if pt_start_time != 0.0:
                    for pos_idx in range(50):
                        if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
                            dendrite_plateau_mask[d_idx, pos_idx] = True
                            break

        num_plateaus_per_dend_list.append(np.sum(num_plateaus_list))





        num_time_bins = padded_warped_activity.shape[1]



        position_bins = 50

        for trial in range(num_trials):
            velocity_trial = animal_velocity[:, trial]  # shape (50,)
            dt_trial = dx / velocity_trial              # shape (50,)
            bin_edges = np.concatenate([[0], np.cumsum(dt_trial)])  # shape (51,)

            time_bins = np.arange(num_time_bins) * dt_constant  # shape (num_time_bins,)

            plateau_indices = np.where(plateau_array[trial] == 1)[0]  # shape (n_plateaus,)

            if len(plateau_indices) == 0:
                continue

            pt_times = time_bins[plateau_indices] 

            pos_bin_idxs = np.searchsorted(bin_edges, pt_times, side='right') - 1

            valid_mask = (pos_bin_idxs >= 0) & (pos_bin_idxs < position_bins)
            pos_bin_idxs = pos_bin_idxs[valid_mask]

            bin_counts = np.bincount(pos_bin_idxs, minlength=position_bins)

            # --- Accumulate into global counter
            plateau_positions_counter += bin_counts
            
    plateau_start_times_list = plateau_start_times_list_mega_list[example_cell]
    
    EC_used = use_model_EC

    


    return plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, plateau_start_times_list_mega_list, num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list
    

def _broadcast_to_shape(x, target_shape, name):
    if np.isscalar(x):
        return np.full(target_shape, float(x), dtype=float)
    x = np.asarray(x, dtype=float)
    # allow (n_pos,) or (n_trials,) to expand
    if x.ndim == 1:
        if x.shape[0] == target_shape[0]:
            return np.broadcast_to(x[:, None], target_shape)
        if x.shape[0] == target_shape[1]:
            return np.broadcast_to(x[None, :], target_shape)
    try:
        return np.broadcast_to(x, target_shape)
    except Exception as e:
        raise ValueError(f"Cannot broadcast {name} {x.shape} -> {target_shape}: {e}")

def _first_nonfinite_idx(arr):
    bad = ~np.isfinite(arr)
    if not bad.any():
        return None
    idx = np.argwhere(bad)[0]
    return tuple(idx)

def add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict, GLM_params, animal_velocity_dict):
    out = {}
    for animal in scaled_data_Hz_dict:
        V_raw = np.asarray(animal_velocity_dict[animal], dtype=float)  # (n_pos, 58)
        V = clean_velocity(V_raw, min_v=1e-6)                          # <- sanitize here

        per_cell = {}
        for cell, D in scaled_data_Hz_dict[animal].items():
            D = np.asarray(D, dtype=float)
            if D.shape != V.shape:
                raise ValueError(f"{animal}/{cell}: data {D.shape} vs vel {V.shape} mismatch")

            W = GLM_params[animal][cell]['weights'].get("Velocity", 0.0)
            b = GLM_params[animal][cell]['intercept']

            # Broadcast helpers
            def _to_shape(x, shape, name):
                if np.isscalar(x):
                    return np.full(shape, float(x), dtype=float)
                a = np.asarray(x, dtype=float)
                if a.shape == shape: return a
                if a.ndim == 1:
                    if a.shape[0] == shape[0]: return np.broadcast_to(a[:, None], shape)
                    if a.shape[0] == shape[1]: return np.broadcast_to(a[None, :], shape)
                return np.broadcast_to(a, shape)

            Wb = _to_shape(W, D.shape, "weights['Velocity']")
            bb = _to_shape(b, D.shape, "intercept")

            # sanity
            for name, X in [("data", D), ("W", Wb), ("b", bb), ("V", V)]:
                if not np.isfinite(X).all():
                    idx = tuple(np.argwhere(~np.isfinite(X))[0])
                    raise ValueError(f"{animal}/{cell}: {name} non-finite at {idx}")

            Y = D + (Wb * V) + bb
            if not np.isfinite(Y).all():
                idx = tuple(np.argwhere(~np.isfinite(Y))[0])
                raise ValueError(
                    f"{animal}/{cell}: output non-finite at {idx} "
                    f"(D={D[idx]}, W={Wb[idx]}, V={V[idx]}, b={bb[idx]})"
                )

            per_cell[cell] = Y.astype(np.float32)
        out[animal] = per_cell
    return out

def clean_velocity(V, min_v=1e-6, max_v=None):
    """
    V: (n_pos, n_trials) velocity matrix (float)
    - Interpolates non-finite entries along position within each trial.
    - If an entire trial is non-finite, replaces it with per-position median across other trials.
    - Clamps negative/too-small velocities.
    """
    V = np.asarray(V, dtype=float).copy()
    n_pos, n_trials = V.shape
    x = np.arange(n_pos)

    # Pass 1: interpolate along position within each trial
    for t in range(n_trials):
        col = V[:, t]
        good = np.isfinite(col)
        if good.any():
            # edge-fill using first/last finite
            left_val  = col[good][0]
            right_val = col[good][-1]
            col[~good] = np.interp(x[~good], x[good], col[good], left=left_val, right=right_val)
            V[:, t] = col
        else:
            # whole trial bad: fill with per-position median across trials (ignoring NaNs)
            med = np.nanmedian(V, axis=1)
            # if med still has NaNs (everything is bad), fall back to zeros
            med = np.nan_to_num(med, nan=0.0)
            V[:, t] = med

    # Pass 2: clamp negatives and very small velocities (avoid future dx/v blowups)
    V = np.where(~np.isfinite(V), 0.0, V)     # just in case
    V = np.where(V < min_v, min_v, V)         # enforce strictly positive
    if max_v is not None:
        V = np.clip(V, None, max_v)
    return V


def _pad_rows_to_2d(rows, fill=np.nan):
    """
    rows: iterable of 1D arrays/lists (ragged by time)
    returns: 2D float array (n_trials, max_T_cell) padded with NaN
    """
    # convert each row to float array; allow empty rows
    rows_np = [np.asarray(r, dtype=float).ravel() for r in rows]
    max_T = max((len(r) for r in rows_np), default=0)
    out = np.full((len(rows_np), max_T), fill, dtype=float)
    for i, r in enumerate(rows_np):
        if r.size:
            out[i, :r.size] = r
    return out  # shape (n_trials, max_T_cell)

def pad_stack_same_trials_ragged(padded_warped_activity_dict, n_trials_expected=58):
    """
    padded_warped_activity_dict[animal][cell] = either:
      - 2D array (n_trials, T_cell), or
      - list/iterable of length n_trials with 1D arrays of variable length
    Returns:
      stacked: (n_cells_total, n_trials_expected, max_T_global)
      keys:    [(animal, cell), ...]
      max_T_global: int
    """
    per_cell = []
    keys = []
    max_T_global = 0

    for animal, cells in padded_warped_activity_dict.items():
        for cell, A in cells.items():
            # Normalize to 2D float array (n_trials, T_cell)
            if isinstance(A, np.ndarray) and A.ndim == 2:
                A2 = A.astype(float, copy=False)
            else:
                # Assume ragged-by-trial
                A2 = _pad_rows_to_2d(A)

            if A2.shape[0] != n_trials_expected:
                raise ValueError(f"{animal}/{cell}: expected {n_trials_expected} trials, got {A2.shape[0]}")

            per_cell.append(A2)
            keys.append((animal, cell))
            max_T_global = max(max_T_global, A2.shape[1])

    # Second pass: pad each cell array to global max_T and stack
    n_cells = len(per_cell)
    stacked = np.full((n_cells, n_trials_expected, max_T_global), np.nan, dtype=float)
    for i, A2 in enumerate(per_cell):
        T = A2.shape[1]
        stacked[i, :, :T] = A2

    return stacked, keys, max_T_global

def activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1,
                        center_across="time_trials", dtype=np.float32):
    A = np.asarray(activity_EC, dtype=float)  # (D, T, N)
    if A.ndim != 3:
        raise ValueError(f"activity_EC must be 3D; got {A.shape}")

    if center_across == "time":
        mu = np.nanmean(A, axis=1, keepdims=True)          # (D,1,N)  <-- removes trial offsets
    elif center_across == "time_trials":
        print("worked")
        mu = np.nanmean(A, axis=(1,2), keepdims=True)       # (D,1,1)  <-- keeps trial variability
    elif center_across == "none":
        mu = 0.0                                            # no centering
    else:
        raise ValueError("center_across must be 'time', 'time_trials', or 'none'")

    A_centered = A - mu
    dend_Vm = Vrest + vm_scale * A_centered
    return dend_Vm.astype(dtype), A_centered.astype(dtype), (mu if np.isscalar(mu) else mu.astype(dtype))




def get_epsp_dict_multi(padded_warped_activity_dict, tau_ms=None, amp=None, seed=None):

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

            print(f"len(padded_warped_activity) {len(padded_warped_activity)}")


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


    
def plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None):
    
    if include_inhibition == 'both':
        # dend_activity = activity_EC - (activity_NDNF*NDNF_sf_opt + activity_SST*SST_sf_opt)

        # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)

        fig, axs = plt.subplots(5,4, figsize=(30,25))
        fig.suptitle(f"Ratio SST Contribution : NDNF Contribution = {SST_contribution_sum / NDNF_contribution_sum:.3f}", y=1.0)

        activity_EC_trial_av = np.mean(activity_EC, axis=2)
        mean_activity_EC_trial_av = np.mean(activity_EC_trial_av, axis=0)
        for i in range(activity_EC_trial_av.shape[0]):
            axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
        axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,0].set_title("EC Input To Each Dendrite SF=1")
        axs[0,0].set_ylabel("Activity")
        axs[0,0].set_xlabel("Position Bins")


        activity_SST_trial_av = np.mean(activity_SST, axis=2)
        mean_activity_SST_trial_av = np.mean(activity_SST_trial_av, axis=0)
        for i in range(activity_SST_trial_av.shape[0]):
            axs[0,1].plot(activity_SST_trial_av[i,:], alpha=0.2)
        axs[0,1].plot(mean_activity_SST_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,1].set_title(f"SST Input To Each Dendrite SF={SST_sf_opt:.3f}")
        axs[0,1].set_ylabel("Activity")
        axs[0,1].set_xlabel("Position Bins")

        activity_NDNF_trial_av = np.mean(activity_NDNF, axis=2)
        mean_activity_NDNF_trial_av = np.mean(activity_NDNF_trial_av, axis=0)
        for i in range(activity_NDNF_trial_av.shape[0]):
            axs[0,2].plot(activity_NDNF_trial_av[i,:], alpha=0.2)
        axs[0,2].plot(mean_activity_NDNF_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,2].set_title(f"NDNF Input To Each Dendrite SF={NDNF_sf_opt:.3f}")
        axs[0,2].set_ylabel("Activity")
        axs[0,2].set_xlabel("Position Bins")

        im1 = axs[1,0].imshow(weights_EC, aspect='auto')
        axs[1,0].set_title(f"EC Weights: {dist}")
        axs[1,0].set_ylabel("Dendrites")
        axs[1,0].set_xlabel("Input Cells")
        fig.colorbar(im1, ax=axs[1,0])

        im1 = axs[1,1].imshow(weights_SST, aspect='auto')
        axs[1,1].set_title(f"SST Weights: Equal")
        axs[1,1].set_ylabel("Dendrites")
        axs[1,1].set_xlabel("Input Cells")
        fig.colorbar(im1, ax=axs[1,1])

        im1 = axs[1,2].imshow(weights_NDNF, aspect='auto')
        axs[1,2].set_title(f"NDNF Weights: Equal")
        axs[1,2].set_ylabel("Dendrites")
        axs[1,2].set_xlabel("Input Cells")
        fig.colorbar(im1, ax=axs[1,2])

        mean_pad = np.mean(activity_EC, axis=0)

        im4 = axs[2,2].imshow(mean_pad, aspect='auto', interpolation='None')
        axs[2,2].set_title("Mean Over Dendrites")
        axs[2,2].set_ylabel("Trials")
        axs[2,2].set_xlabel("Time (ms)")
        fig.colorbar(im4, ax=axs[2,2])

        means = np.mean(mean_pad, axis=0)
        # means = means / np.max(means)
        axs[2,3].plot(means)
        axs[2,3].set_title("Mean of Dendrites and Trials")
        axs[2,3].set_xlabel("Time (ms)")

        axs[0,3].plot(an_velocity, color='r')
        axs[0,3].set_xlabel("Position Bins")
        axs[0,3].set_ylabel("Meters / Second")
        axs[0,3].set_title("Velocity")

        distance = 3.6
        velocity_cm = an_velocity*100

        occupancy = distance / velocity_cm
        
        axs[1,3].plot(occupancy, color='r')
        axs[1,3].set_xlabel("Position Bins")
        axs[1,3].set_ylabel("Seconds")
        axs[1,3].set_title("Occupancy")

        ims = axs[3,0].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray')
        axs[3,0].set_title(f"Ex/ Dendrite Plateaus Over Time \n Dendrite Threshold={dend_threshold}")
        axs[3,0].set_xlabel("Time (ms)")
        axs[3,0].set_ylabel("Trials")
        fig.colorbar(ims, ax=axs[3,0])


        # axs[1,2].hist(weights_EC.flatten(), bins=50)
        # axs[1,2].set_title(f"Weights - {dist} Dist")
        # axs[1,2].set_ylabel("Count")
        # axs[1,2].set_xlabel("Weight")

        mean_dend_activity = np.mean(dend_activity, axis=0)

        im4 = axs[2,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
        axs[2,0].set_title("Mean Over Dendrites")
        axs[2,0].set_ylabel("Trials")
        axs[2,0].set_xlabel("Position Bins")
        fig.colorbar(im4, ax=axs[2,0])

        num_plateaus_per_trial_list_across_dends = []
        for dend in range(len(plateau_start_times_list_mega_list)):
            num_plateaus_per_trial = []
            dend_plateaus = plateau_start_times_list_mega_list[dend]
            for trial in range(len(dend_plateaus)):
                num_plateaus_per_trial.append(len(dend_plateaus[trial]))

            num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

        num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
        mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
        sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

        axs[4,2].set_title("Mean # Plateaus Per Trial Across Dendrites")
        axs[4,2].set_ylabel("Mean # Plateaus Across Dendrites")
        axs[4,2].set_xlabel("Session Length (%)")
        axs[4,2].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], 
                                labels=["0", '25', "50", '75', "100"])
        axs[4,2].plot(mean_plat_per_trial, color='k')
        axs[4,2].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        
        means = np.nanmean(mean_dend_activity, axis=1)
        
        axs[2,1].plot(means)
        axs[2,1].set_title("Mean of Dendrites and Trials")
        axs[2,1].set_xlabel("Position Bins")

        axs[3,1].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
        axs[3,1].set_title(f"Plateau Time Across All Dendrites")
        axs[3,1].set_ylabel("Time (ms)")
        axs[3,1].set_xlabel("Position Bins")


        n_bins = 10
        bin_size = int(50 / n_bins)

        summed_plateaus = np.zeros(n_bins)

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(plateau_start_positions_counter[start:end])
            summed_plateaus[i] = summed_data

        axs[3,2].bar(range(len(summed_plateaus)), summed_plateaus)
        axs[3,2].set_xlabel("Position Bin")
        axs[3,2].set_ylabel("Plateau Count")
        axs[3,2].set_title("Plateau Onset Count per Track Section")
        # axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])
        axs[3,2].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
        axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
        axs[3,3].set_xlabel("Grouped Position Bins")
        axs[3,3].set_ylabel("% of Total Plateaus")
        # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])
        axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

        axs[4,0].hist(num_plateaus_per_dend_list, bins=10)
        axs[4,0].set_xlabel("# of Plateaus")
        axs[4,0].set_ylabel("# of Cells")

        number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
        axs[4,1].plot(number_dendrites_counter)
        axs[4,1].set_title("Percent of Dendrites with Plateau at Location")
        axs[4,1].set_ylabel("Percent")
        axs[4,1].set_xlabel("Position Bin")

 

        cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
        mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
        sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)


        axs[4,3].set_title("Cumulative # Plateaus Across Trials")
        axs[4,3].set_ylabel("# Plateaus Mean +- SEM Across Dends")
        axs[4,3].set_xlabel("Session Length (%)")
        axs[4,3].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
                                labels=["0", '25', "50", '75', "100"])
        axs[4,3].plot(mean_plateaus_all_dends, color='k')
        axs[4,3].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
        

        plt.tight_layout()
        plt.show()



    elif include_inhibition == 'sst':

        fig, axs = plt.subplots(5,4, figsize=(30,25))


        activity_EC_trial_av = np.mean(activity_EC, axis=2)
        mean_activity_EC_trial_av = np.mean(activity_EC_trial_av, axis=0)
        for i in range(activity_EC_trial_av.shape[0]):
            axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
        axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,0].set_title("EC Input To Each Dendrite")
        axs[0,0].set_ylabel("Activity")
        axs[0,0].set_xlabel("Position Bins")


        activity_SST_trial_av = np.mean(activity_SST, axis=2)
        mean_activity_SST_trial_av = np.mean(activity_SST_trial_av, axis=0)
        for i in range(activity_SST_trial_av.shape[0]):
            axs[0,1].plot(activity_SST_trial_av[i,:], alpha=0.2)
        axs[0,1].plot(mean_activity_SST_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,1].set_title("SST Input To Each Dendrite")
        axs[0,1].set_ylabel("Activity")
        axs[0,1].set_xlabel("Position Bins")

        axs[0,2].plot(an_velocity, color='r')
        axs[0,2].set_xlabel("Position Bins")
        axs[0,2].set_ylabel("Meters / Second")
        axs[0,2].set_title("Velocity")

        an_velocity_cm = an_velocity*100
        distance = 3.6

        occupancy = distance / an_velocity_cm

        axs[0,3].plot(occupancy, color='r')
        axs[0,3].set_xlabel("Position Bins")
        axs[0,3].set_ylabel("Seconds")
        axs[0,3].set_title("Occupancy")

        im1 = axs[1,0].imshow(weights_EC, aspect='auto')
        axs[1,0].set_title(f"EC Weights {dist}")
        axs[1,0].set_ylabel("Dendrites")
        axs[1,0].set_xlabel("Input Cells")
        fig.colorbar(im1, ax=axs[1,0])


        padded_warped_activity_array = np.array(padded_warped_activity_list)

        mean_pad = np.mean(padded_warped_activity_array, axis=0)

        im4 = axs[2,2].imshow(mean_pad, aspect='auto', interpolation='None')
        axs[2,2].set_title("Mean Over Dendrites")
        axs[2,2].set_ylabel("Trials")
        axs[2,2].set_xlabel("Time (ms)")
        fig.colorbar(im4, ax=axs[2,2])

        means = np.mean(mean_pad, axis=0)
        # means = means / np.max(means)
        axs[2,3].plot(means)
        axs[2,3].set_title("Mean of Dendrites and Trials")
        axs[2,3].set_xlabel("Time (ms)")

        ims = axs[3,0].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray')
        axs[3,0].set_title(f"Ex/ Dendrite Plateaus Over Time \n Dendrite Threshold={dend_threshold}")
        axs[3,0].set_xlabel("Time (ms)")
        axs[3,0].set_ylabel("Trials")
        fig.colorbar(ims, ax=axs[3,0])

        im1 = axs[1,1].imshow(weights_SST, aspect='auto')
        axs[1,1].set_title(f"SST Weights: Equal")
        axs[1,1].set_ylabel("Dendrites")
        axs[1,1].set_xlabel("Input Cells")
        fig.colorbar(im1, ax=axs[1,1])


        axs[1,2].hist(weights_EC.flatten(), bins=50)
        axs[1,2].set_title(f"EC Weights: {dist} Distriution")
        axs[1,2].set_ylabel("Count")
        axs[1,2].set_xlabel("Weight")

        axs[1,3].hist(weights_SST.flatten(), bins=50)
        axs[1,3].set_title(f"SST Weights: Equal Distriution")
        axs[1,3].set_ylabel("Count")
        axs[1,3].set_xlabel("Weight")

        mean_dend_activity = np.mean(dend_activity, axis=0)

        im4 = axs[2,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
        axs[2,0].set_title("Mean Over Dendrites")
        axs[2,0].set_ylabel("Trials")
        axs[2,0].set_xlabel("Position Bins")
        fig.colorbar(im4, ax=axs[2,0])

        num_plateaus_per_trial_list_across_dends = []
        for dend in range(len(plateau_start_times_list_mega_list)):
            num_plateaus_per_trial = []
            dend_plateaus = plateau_start_times_list_mega_list[dend]
            for trial in range(len(dend_plateaus)):
                num_plateaus_per_trial.append(len(dend_plateaus[trial]))

            num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

        num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
        mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
        sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

        axs[4,2].set_title("Mean # Plateaus Per Trial Across Dendrites")
        axs[4,2].set_ylabel("Mean # Plateaus Across Dendrites")
        axs[4,2].set_xlabel("Session Length (%)")
        axs[4,2].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], 
                                labels=["0%", '25', "50%", '75', "100%"])
        axs[4,2].plot(mean_plat_per_trial, color='k')
        axs[4,2].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        
        means = np.nanmean(mean_dend_activity, axis=1)
        
        axs[2,1].plot(means)
        axs[2,1].set_title("Mean of Dendrites and Trials")
        axs[2,1].set_xlabel("Position Bins")

        axs[3,1].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
        axs[3,1].set_title(f"Plateau Time Across All Dendrites")
        axs[3,1].set_ylabel("Time (ms)")
        axs[3,1].set_xlabel("Position Bins")

        # axs[3,1].bar(range(len(plateau_start_positions_counter)), plateau_start_positions_counter)
        # axs[3,1].set_title(f"Plateau Count Across All Dendrites")
        # axs[3,1].set_ylabel("Plateau Count")
        # axs[3,1].set_xlabel("Position Bins")


        # n_bins = 10
        # bin_size = int(50 / n_bins)

        # summed_plateaus = np.zeros(n_bins)

        # for i in range(n_bins):
        #     start = i * bin_size
        #     end = (i + 1) * bin_size
        #     summed_data = np.sum(plateau_positions_counter[start:end])
        #     summed_plateaus[i] = summed_data

        # axs[4,3].bar(range(len(summed_plateaus)), summed_plateaus)
        # axs[4,3].set_xlabel("Position Bin Quintile")
        # axs[4,3].set_ylabel("Time (ms)")
        # axs[4,3].set_title("Plateau Time per Track Section")
        # axs[4,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

        # y_time_ms = plateau_counts_per_time * (dt_constant * 1000.0)  # ms
        # x_time_ms = np.arange(len(plateau_counts_per_time)) * (dt_constant * 1000.0)
        # n_trials = plateau_array.shape[0]
        # y_mean_time_ms = (plateau_counts_per_time / n_trials) * (dt_constant * 1000.0)
        # axs[3,0].bar(x_time_ms, y_mean_time_ms)
        # axs[3,0].set_ylabel("Mean plateau time per trial (ms)")
        # axs[3,0].set_xlabel("Time (ms)")


        n_bins = 10
        bin_size = int(50 / n_bins)

        summed_plateaus = np.zeros(n_bins)

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(plateau_start_positions_counter[start:end])
            summed_plateaus[i] = summed_data

        axs[3,2].bar(range(len(summed_plateaus)), summed_plateaus)
        axs[3,2].set_xlabel("Position Bin")
        axs[3,2].set_ylabel("Plateau Count")
        axs[3,2].set_title("Plateau Onset Count per Track Section")
        # axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])
        axs[3,2].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
        axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
        axs[3,3].set_xlabel("Grouped Position Bins")
        axs[3,3].set_ylabel("% of Total Plateaus")
        axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
        # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])

        cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
        mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
        sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)

        axs[4,3].set_title("Cumulative # Plateaus Across Trials")
        axs[4,3].set_ylabel("# Plateaus Mean +- SEM Across Dends")
        axs[4,3].set_xlabel("Session Length (%)")
        axs[4,3].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
                             labels=["0", '25', "50", '75', "100%"])
        axs[4,3].plot(mean_plateaus_all_dends, color='k')
        axs[4,3].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
            

        # variance_array = np.var(dend_activity, axis=(1, 2))  # Per dendrite variance

        # axs[5,2].hist(variance_array, bins=50)
        # axs[5,2].set_title(f"Variance in Activity {dist}")
        # axs[5,2].set_xlabel("Variance")
        # axs[5,2].set_ylabel("Number of Dendrites")


      

        number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
        axs[4,1].plot(number_dendrites_counter)
        axs[4,1].set_title("Percent of Dendrites with Plateau at Location")
        axs[4,1].set_ylabel("Percent")
        axs[4,1].set_xlabel("Position Bin")

        axs[4,0].hist(num_plateaus_per_dend_list, bins=10, edgecolor='k')
        axs[4,0].set_xlabel("# of Plateaus")
        axs[4,0].set_ylabel("# of Cells")



        plt.tight_layout()
        plt.show()



    else:
        # dend_activity = activity_EC 

        # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)

        fig, axs = plt.subplots(4,4, figsize=(25,20))

        print(f"activity_EC.shape {activity_EC.shape}")

        D, N, T = activity_EC.shape            # (dendrites, trials, timebins)
        dt = 0.001                             # seconds per time bin
        t_ms = np.arange(T) * dt * 1000.0      # time (ms), length T

        # trial-averaged time series per dendrite (ignore NaN padding)
        activity_EC_trial_av = np.nanmean(activity_EC, axis=1)   # (D, T)

        for i in range(D):
            y = np.ma.masked_invalid(activity_EC_trial_av[i])    # mask any NaNs in ragged tail
            axs[0, 0].plot(t_ms, y, alpha=0.2)

        axs[0, 0].set_title("EC Input To Each Dendrite")
        axs[0, 0].set_ylabel("Summed Z-Scored Activity")
        axs[0, 0].set_xlabel("time (ms)")
        axs[0, 0].set_xlim(0, 6000)            # show full 0..6000 ms even if data ends earlier

        # activity_EC_trial_av = np.mean(activity_EC, axis=1)
        # # mean_activity_EC_trial_av = np.nanmean(activity_EC_trial_av, axis=0)
        # for i in range(activity_EC_trial_av.shape[0]):
        #     axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
        # axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
        # axs[0,0].set_title("EC Input To Each Dendrite")
        # axs[0,0].set_ylabel("Summed Z-Scored Activity")
        # axs[0,0].set_xlabel("Position Bins")

        axs[0,3].plot(an_velocity, color='r')
        axs[0,3].set_xlabel("Position Bins")
        axs[0,3].set_ylabel("Meters / Second")
        axs[0,3].set_title("Velocity")

        an_velocity_cm = an_velocity*100
        distance = 3.6

        axs[1,3].plot(distance/an_velocity_cm, color='r')
        axs[1,3].set_xlabel("Position Bins")
        axs[1,3].set_ylabel("Seconds")
        axs[1,3].set_title("Occupancy")

        distance=3.6
        velocity_cm_s = an_velocity*100

        # axs[0,2].plot(distance / velocity_cm_s, color='purple')
        # axs[0,2].set_xlabel("Position Bins")
        # axs[0,2].set_ylabel("Seconds")
        # axs[0,2].set_title("Occupancy")

        im1 = axs[0,1].imshow(weights_EC, aspect='auto')
        axs[0,1].set_title(f"EC Weights: {dist} Distribution")
        axs[0,1].set_ylabel("Dendrites")
        axs[0,1].set_xlabel("Input Cells") 
        fig.colorbar(im1, ax=axs[0,1])



        mean_pad = np.mean(activity_EC, axis=0)
        mean_pad = mean_pad.T

        im4 = axs[2,0].imshow(mean_pad, aspect='auto', interpolation='None')
        axs[2,0].set_title("Mean Over Dendrites")
        axs[2,0].set_ylabel("Trials")
        axs[2,0].set_xlabel("Time (ms)")
        fig.colorbar(im4, ax=axs[2,0])

        means = np.mean(mean_pad, axis=0)
        # means = means / np.max(means)
        axs[2,1].set_ylabel("Summed Z-Scored Activity")
        axs[2,1].plot(means)
        axs[2,1].set_title("Mean of Dendrites and Trials")
        axs[2,1].set_xlabel("Time (ms)")

        ims = axs[1,2].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray', interpolation='none')
        axs[1,2].set_title(f"Example Dendrite Plateaus \n Dendrite Threshold={dend_threshold}")
        axs[1,2].set_xlabel("Time (ms)")
        axs[1,2].set_ylabel("Trials")
        fig.colorbar(ims, ax=axs[1,2])

        axs[0,2].hist(weights_EC.flatten(), bins=50)
        axs[0,2].set_title(f"EC Weights: {dist} Distribution")
        axs[0,2].set_ylabel("Count")
        axs[0,2].set_xlabel("Weight")

        mean_dend_activity = np.nanmean(dend_activity, axis=0).T

        im4 = axs[1,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
        axs[1,0].set_title("Mean Over Dendrites")
        axs[1,0].set_ylabel("Trials")
        axs[1,0].set_xlabel("Position Bins")
        axs[1,0].set_xlim(0, 6000)
        fig.colorbar(im4, ax=axs[1,0])

        num_plateaus_per_trial_list_across_dends = []
        for dend in range(len(plateau_start_times_list_mega_list)):
            num_plateaus_per_trial = []
            dend_plateaus = plateau_start_times_list_mega_list[dend]
            for trial in range(len(dend_plateaus)):
                num_plateaus_per_trial.append(len(dend_plateaus[trial]))

            num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

        num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
        mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
        sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

        axs[3,1].set_title("Mean # Plateaus Per Trial Across Dendrites")
        axs[3,1].set_ylabel("Mean # Plateaus Across Dendrites")
        axs[3,1].set_xlabel("Session Length (%)")
        axs[3,1].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], labels=["0", '25', "50", '75', "100"])
        axs[3,1].plot(mean_plat_per_trial, color='k')
        axs[3,1].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        

        # padded_warped_activity_array, key_list, max_len = pad_stack_same_trials_ragged(padded_warped_activity_dict, n_trials_expected=58)

        for i in range(mean_pad.shape[1]):
            axs[1,1].plot(t_ms, mean_pad[:,i])
        axs[1,1].set_ylabel("Summed Z-Scored Activity")
        axs[1,1].set_title("Mean of Dendrites and Trials")
        axs[1,1].set_xlabel("Time (ms)")
        axs[1,1].set_xlim(0, 6000) 

        axs[2,2].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
        axs[2,2].set_title(f"Plateau Time Across All Dendrites")
        axs[2,2].set_ylabel("Time (ms)")
        axs[2,2].set_xlabel("Position Bins")


        n_bins = 10
        bin_size = int(50 / n_bins)

        summed_plateaus = np.zeros(n_bins)

        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(plateau_start_positions_counter[start:end])
            summed_plateaus[i] = summed_data

        axs[2,3].bar(range(len(summed_plateaus)), summed_plateaus)
        axs[2,3].set_xlabel("Position Bin")
        axs[2,3].set_ylabel("Plateau Count")
        axs[2,3].set_title("Plateau Onset Count per Track Section")
        axs[2,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
        # axs[2,3].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])


        cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
        mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
        sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)


        axs[3,0].set_title("Cumulative # Plateaus Across Trials")
        axs[3,0].set_ylabel("# Plateaus Mean +- SEM Across Dends")
        axs[3,0].set_xlabel("Session Length (%)")
        axs[3,0].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
                             labels=["0", '25', "50", '75', "100"])
        axs[3,0].plot(mean_plateaus_all_dends, color='k')
        axs[3,0].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
            

        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
        axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
        axs[3,3].set_xlabel("Grouped Position Bins")
        axs[3,3].set_ylabel("% of Total Plateaus")
        axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
        # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])


        # variance_array = np.var(dend_activity, axis=(1, 2))  # Per dendrite variance

        # axs[5,2].hist(variance_array, bins=50)
        # axs[5,2].set_title(f"Variance in Activity {dist}")
        # axs[5,2].set_xlabel("Variance")
        # axs[5,2].set_ylabel("Number of Dendrites")


        # axs[3,0].hist(num_plateaus_per_dend_list, bins=10, edgecolor='k')
        # axs[3,0].set_xlabel("# of Plateaus")
        # axs[3,0].set_ylabel("# of Cells")

        number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
        axs[3,2].plot(number_dendrites_counter)
        axs[3,2].set_title("Percent of Dendrites with Plateau at Location")
        axs[3,2].set_ylabel("Percent")
        axs[3,2].set_xlabel("Position Bin")

 

        plt.tight_layout()
        plt.show()

        return mean_pad



# def zscore_2d(array, axis=None, eps=1e-12):
    
#     arr = np.asarray(array, dtype=float)
#     mean = np.mean(arr, axis=axis, keepdims=True)
#     std = np.std(arr, axis=axis, keepdims=True)
#     return (arr - mean) / (std + eps)


def get_plateau_internals(
    plateau_dict_animal,         # dict[animal][cell] -> plateau_array (n_trials, n_time)
    activity_EC,                 # array for sizing only; if you don't need it, you can drop it
    animal_velocity,             # (50, n_trials) in cm/s (or m/s scaled consistently with dx)
    dt_constant,                 # seconds per time bin in plateau_array
    dx,                          # cm per position bin
    n_pos=50
):
    # Treat each (animal, cell) as one "dendrite" row in outputs
    # Gather a stable list of (animal, cell, plateau_array)
    entries = []
    for animal in plateau_dict_animal:
        for cell in plateau_dict_animal[animal]:
            pa = plateau_dict_animal[animal][cell]  # (n_trials, n_time)
            if pa is None:
                continue
            entries.append((animal, cell, pa))

    n_dend = len(entries)
    dendrite_plateau_mask = np.zeros((n_dend, n_pos), dtype=bool)

    # Global counters across ALL animals/cells (match original intent)
    plateau_positions_counter = np.zeros(n_pos, dtype=float)
    plateau_start_positions_counter = np.zeros(n_pos, dtype=float)

    # Per-dendrite (per-entry) containers
    plateau_array_per_dendrite_list = []
    plateau_start_times_list_mega_list = []  # list over dendrites -> list over trials -> array of start times
    num_plateaus_per_dend_list = []

    # Optional: zscore activity_EC if you still want it around; otherwise drop it from returns
    # (This mirrors your original top function but isn't used below.)
    # If activity_EC is shaped (n_dend, n_pos, n_trials), we keep it; else ignore.
    # dend_activity = zscore_2d(activity_EC, axis=None, eps=1e-12)  # if you really need it

    # Iterate over each "dendrite" (animal, cell)
    for d_idx, (animal, cell, plateau_array) in enumerate(entries):
        n_trials, n_time = plateau_array.shape
        plateau_array_per_dendrite_list.append(plateau_array)

        # Per-dend start-time lists
        dend_plateau_start_times_list = []
        num_plateaus_this_dend = 0

        # For mapping times->position we use the SAME animal_velocity for all entries
        # If each animal/cell has its own velocity, pass that in instead of a single 'animal_velocity'.
        position_bins = n_pos

        # Precompute time axis for this plateau array
        time_bins = np.arange(n_time) * dt_constant  # seconds

        # --- START COUNTS (onsets only) + dendrite mask
        for trial in range(n_trials):
            velocity_trial = animal_velocity[:, trial]  # (50,)
            dt_trial = dx / velocity_trial              # seconds per position bin
            bin_edges = np.concatenate([[0.0], np.cumsum(dt_trial)])  # shape (51,)

            # onset indices (0->1 transitions)
            start_idx = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
            start_times = start_idx * dt_constant
            dend_plateau_start_times_list.append(start_times)
            num_plateaus_this_dend += len(start_times)

            # map starts to position bins
            if start_times.size:
                pos_bin_idxs = np.searchsorted(bin_edges, start_times, side='right') - 1
                valid = (pos_bin_idxs >= 0) & (pos_bin_idxs < position_bins)
                pos_bin_idxs = pos_bin_idxs[valid]
                if pos_bin_idxs.size:
                    counts = np.bincount(pos_bin_idxs, minlength=position_bins)
                    plateau_start_positions_counter += counts
                    # mark mask where this dendrite ever had a start
                    dendrite_plateau_mask[d_idx, counts > 0] = True

        plateau_start_times_list_mega_list.append(dend_plateau_start_times_list)
        num_plateaus_per_dend_list.append(num_plateaus_this_dend)

        # --- ALL-PLATEAU COUNTS (timepoints where plateau==1)
        for trial in range(n_trials):
            velocity_trial = animal_velocity[:, trial]     # (50,)
            dt_trial = dx / velocity_trial
            bin_edges = np.concatenate([[0.0], np.cumsum(dt_trial)])
            pt_idx = np.where(plateau_array[trial] == 1)[0]
            if pt_idx.size == 0:
                continue
            pt_times = time_bins[pt_idx]
            pos_bin_idxs = np.searchsorted(bin_edges, pt_times, side='right') - 1
            valid = (pos_bin_idxs >= 0) & (pos_bin_idxs < position_bins)
            pos_bin_idxs = pos_bin_idxs[valid]
            if pos_bin_idxs.size:
                plateau_positions_counter += np.bincount(pos_bin_idxs, minlength=position_bins)

    # For parity with the original returns:
    # 'time_each_pos_bin_starts' has no single well-defined value if velocity differs per trial;
    # returning the LAST computed per-trial edges would be misleading. Instead, return None
    # or a dict if you truly need it for inspection.
    time_each_pos_bin_starts = None
    EC_used = False  # since this path consumes precomputed plateau arrays; no synthetic EC was used

    return (
        plateau_positions_counter,
        plateau_start_positions_counter,
        plateau_array_per_dendrite_list,
        dendrite_plateau_mask,
        time_each_pos_bin_starts,
        plateau_start_times_list_mega_list,
        EC_used,
        num_plateaus_per_dend_list,
    )




def get_dend_vm_multi(epsp_dict, Vrest=-60.0, epsp_sf=0.1):
    cell_epsp_mats = []
    cell_spike_mats = []

    # --- per cell: pad trials to that cell's max trial length ---
    for animal in epsp_dict:
        for cell in epsp_dict[animal]:
            epsp = epsp_dict[animal][cell]["epsps"]         # dict: trial -> 1D array
            spik = epsp_dict[animal][cell]["spike_train"]   # dict: trial -> 1D array

            # Per-cell max lengths
            max_len_epsp = max(len(epsp[t]) for t in epsp)
            max_len_spik = max(len(spik[t]) for t in spik)

            # EPSPs -> (n_trials, max_len_epsp)
            epsp_trials = []
            for t in range(len(epsp)):
                v = epsp[t]
                if len(v) < max_len_epsp:
                    v = np.pad(v, (0, max_len_epsp - len(v)), constant_values=np.nan)
                epsp_trials.append(v.astype(np.float32, copy=False))
            epsp_mat = np.vstack(epsp_trials)  # (n_trials, max_len_epsp)
            cell_epsp_mats.append(epsp_mat)

            # Spikes -> (n_trials, max_len_spik)
            spk_trials = []
            for t in range(len(spik)):
                v = spik[t]
                if len(v) < max_len_spik:
                    v = np.pad(v, (0, max_len_spik - len(v)), constant_values=np.nan)
                spk_trials.append(v.astype(np.float32, copy=False))
            spk_mat = np.vstack(spk_trials)
            cell_spike_mats.append(spk_mat)

    # --- across cells: pad to GLOBAL max length so we can stack cleanly ---
    # (trials can differ across cells; we’ll align by time axis length only)

    global_T = max(m.shape[1] for m in cell_epsp_mats)
    epsp_stack = []
    for m in cell_epsp_mats:
        if m.shape[1] < global_T:
            m = np.pad(m, ((0,0),(0, global_T - m.shape[1])), constant_values=np.nan)
            
        epsp_stack.append(m)
    # shape: (n_cells, n_trials, T) after stacking along a new axis
    epsp_stack = np.stack(epsp_stack, axis=0)

    # --- masked SUM across cells, keeping NaN where no data exists ---
    # count how many non-NaN cells contribute at each (trial, time) bin
    valid_counts = np.sum(~np.isnan(epsp_stack), axis=0)       # (n_trials, T)
    summed = np.nansum(epsp_stack, axis=0)                     # (n_trials, T)
    # where count == 0, set to NaN (instead of 0 from nansum)
    summed[valid_counts == 0] = np.nan

    # center per trial using nanmean (does not turn NaNs into zeros)
    trial_means = np.nanmean(summed, axis=1, keepdims=True)    # (n_trials, 1)
    summed_centered = summed - trial_means                     # (n_trials, T)

    # dendritic Vm (same shape)
    dend_Vm = Vrest + epsp_sf * summed_centered

    return dend_Vm, epsp_stack, cell_spike_mats

 


# def get_plateau_internals(plateau_dict_animal, activity_EC, animal_velocity, dt_constant, dx):
    
#     dend_activity = activity_EC
#     dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)
#     dendrite_plateau_mask = np.zeros((dend_activity.shape[0], 50), dtype=bool)

#     for animal in plateau_dict_animal:
#         for cell in plateau_dict_animal[animal]:
#             plateau_array = plateau_dict_animal[animal][cell]

            
                
#             plateau_positions_counter = np.zeros(50)
#             plateau_start_positions_counter = np.zeros(50)

#             threshold_list = []

#             variance_list = []

#             plateau_array_per_dendrite_list = []

#             num_plateaus_per_dend_list = []

            

#             padded_warped_activity_list = []
            
#             plateau_start_times_list_mega_list = []


#             plateau_start_times_list = []

#             for d_idx in range(dend_activity.shape[0]):
#                 for trial in range(plateau_array.shape[0]):
#                     velocity_trial = animal_velocity[:, trial]
#             #         velocity_trial = proper_velocity
#                     dt_trial = dx / velocity_trial  # in seconds
#                     time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

#                     plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
#                     plateau_start_times = plateau_start_indices * dt_constant  # in seconds
#                     plateau_start_times_list.append(plateau_start_times)

#                     for pt_start_time in plateau_start_times:
#                         if pt_start_time != 0.0:
#                             for pos_idx in range(50):
#                                 if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
#                                     plateau_start_positions_counter[pos_idx] += 1
#                                     break

#                 plateau_start_times_list_mega_list.append(plateau_start_times_list)

#                 num_plateaus_list = []
#                 for trial in range(plateau_array.shape[0]):
#                     velocity_trial = animal_velocity[:, trial]
#             #         velocity_trial = proper_velocity
#                     dt_trial = dx / velocity_trial  # in seconds
#                     time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

#                     plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]

#                     plateau_start_times = plateau_start_indices * dt_constant  # in seconds
#                     num_plateaus_list.append(len(plateau_start_times))

#                     for pt_start_time in plateau_start_times:
#                         if pt_start_time != 0.0:
#                             for pos_idx in range(50):
#                                 if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
#                                     dendrite_plateau_mask[d_idx, pos_idx] = True
#                                     break

#                 num_plateaus_per_dend_list.append(np.sum(num_plateaus_list))



#                 num_time_bins = plateau_array.shape[1]


#                 num_trials, num_time_bins = plateau_array.shape
#                 position_bins = 50



#                 for trial in range(num_trials):
#                     velocity_trial = animal_velocity[:, trial]  # shape (50,)
#                     dt_trial = dx / velocity_trial              # shape (50,)
#                     bin_edges = np.concatenate([[0], np.cumsum(dt_trial)])  # shape (51,)

#                     time_bins = np.arange(num_time_bins) * dt_constant  # shape (num_time_bins,)

#                     plateau_indices = np.where(plateau_array[trial] == 1)[0]  # shape (n_plateaus,)

#                     if len(plateau_indices) == 0:
#                         continue

#                     pt_times = time_bins[plateau_indices] 

#                     pos_bin_idxs = np.searchsorted(bin_edges, pt_times, side='right') - 1

#                     valid_mask = (pos_bin_idxs >= 0) & (pos_bin_idxs < position_bins)
#                     pos_bin_idxs = pos_bin_idxs[valid_mask]

#                     bin_counts = np.bincount(pos_bin_idxs, minlength=position_bins)

#                     # --- Accumulate into global counter
#                     plateau_positions_counter += bin_counts
                    
#         # plateau_start_times_list = plateau_start_times_list_mega_list[example_cell]
        
#         # EC_used = use_model_EC

            


#     return  plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, num_plateaus_per_dend_list
    