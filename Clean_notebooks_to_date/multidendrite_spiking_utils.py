
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

import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# def sample_equal_weights(mask, value=1.0):
#     weights = np.zeros_like(mask, dtype=float)
#     weights[mask] = value
#     return weights



# def sample_weights(distribution, mask, rng, mean=0.1, std=0.5):
#     # normalize incoming token
#     tok = str(distribution).strip().lower()
#     if tok in ("uniform", "u", "uni"):
#         mode = "Uniform"
#     elif tok in ("normal", "n", "gaussian", "norm"):
#         mode = "Normal"
#     elif tok in ("lognormal", "log-norm", "lognorm", "ln"):
#         mode = "Lognormal"
#     elif tok in ("equal", "const", "constant"):
#         mode = "Equal"
#     else:
#         raise ValueError(f"Invalid distribution: {distribution!r}")

#     weights = np.zeros_like(mask, dtype=float)
#     n_samples = int(np.sum(mask))

#     if mode == "Uniform":
#         samples = rng.uniform(low=mean - std, high=mean + std, size=n_samples)
#     elif mode == "Normal":
#         samples = rng.normal(loc=mean, scale=std, size=n_samples)
#         samples = np.clip(samples, 0, None)
#     elif mode == "Lognormal":
#         samples = rng.lognormal(mean=np.log(mean), sigma=std, size=n_samples)
#     elif mode == "None" or mode == 'none':
#         print("It was none all along")
#     else:  # "Equal"
#         samples = np.full(n_samples, mean, dtype=float)

#     weights[mask] = samples
#     return weights


# 1) sample_weights: safe "None" handling
def sample_weights(distribution, mask, rng, mean=0.1, std=0.5):
    tok = str(distribution).strip().lower()
    if tok in ("uniform", "u", "uni"):
        mode = "Uniform"
    elif tok in ("normal", "n", "gaussian", "norm"):
        mode = "Normal"
    elif tok in ("lognormal", "log-norm", "lognorm", "ln"):
        mode = "Lognormal"
    elif tok in ("equal", "const", "constant"):
        mode = "Equal"
    elif tok in ("none",):
        mode = "None"
    else:
        raise ValueError(f"Invalid distribution: {distribution!r}")

    weights = np.zeros(mask.shape, dtype=np.float32)
    n_samples = int(np.sum(mask))

    if mode == "Uniform":
        samples = rng.uniform(low=mean - std, high=mean + std, size=n_samples).astype(np.float32, copy=False)
        weights[mask] = samples
    elif mode == "Normal":
        samples = rng.normal(loc=mean, scale=std, size=n_samples).astype(np.float32, copy=False)
        np.maximum(samples, 0, out=samples)
        weights[mask] = samples
    elif mode == "Lognormal":
        samples = rng.lognormal(mean=np.log(mean), sigma=std, size=n_samples).astype(np.float32, copy=False)
        weights[mask] = samples
    elif mode == "Equal":
        weights[mask] = np.float32(mean)
    else:  # "None" -> leave zeros
        pass

    return weights



# def get_dendrite_activity(weights, EC_input_matrix, n_dendrites, n_EC):
#     EC_flat = EC_input_matrix.reshape(n_EC, -1)
#     dendrite_flat = weights @ EC_flat
#     return dendrite_flat.reshape(n_dendrites, 50, 58)

def get_dendrite_activity(weights, EC_input_matrix, n_dendrites, n_EC):
    # ensure float32 & contiguous BEFORE matmul to avoid upcasts/copies
    W = np.asarray(weights, dtype=np.float32, order="C")
    E = np.asarray(EC_input_matrix, dtype=np.float32, order="C")
    EC_flat = E.reshape(n_EC, -1)                   # view, not copy (C order)
    dendrite_flat = W @ EC_flat                     # float32 GEMM
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
        

    # if which_type == "EC_animal_average":
    #     an_velocity_real_list = []
    #     for animal in factors_dict_EC:
    #             an_velocity_real_list.append(factors_dict_EC[animal]["Velocity"][:,:58])

    #     an_velocity_real_array = np.array(an_velocity_real_list)
    #     an_velocity_real_array_mean_animal = np.nanmean(an_velocity_real_array, axis=0)
    #     return an_velocity_real_array_mean_animal

def sanitize_velocity_cm_s(v_in_m_per_s, min_vel_cm_s=1e-3 * 100):
    """
    Convert m/s -> cm/s and make strictly positive.
    Accepts 1D (n_pos,) or 2D (n_pos, n_trials). Returns same shape.
    Fills NaNs/<=0 by interpolation along the position axis, per trial.
    """

    v = np.asarray(v_in_m_per_s, dtype=np.float32) * np.float32(100.0)  # m/s -> cm/s

    def _sanitize_1d(x):
        # operate on the passed slice, not on the outer 'v'
        bad = ~np.isfinite(x) | (x <= 0)
        if bad.any():
            good_idx = np.flatnonzero(~bad)
            bad_idx  = np.flatnonzero(bad)
            if good_idx.size >= 2:
                x[bad] = np.interp(bad_idx, good_idx, x[good_idx])
            elif good_idx.size == 1:
                x[bad] = x[good_idx[0]]
            else:
                x[:] = np.float32(10.0)  # fallback if everything is bad
        return np.maximum(x, np.float32(min_vel_cm_s))

    if v.ndim == 1:
        return _sanitize_1d(v)
    if v.ndim == 2:
        out = v.copy()
        for t in range(out.shape[1]):
            out[:, t] = _sanitize_1d(out[:, t])
        return out
    raise ValueError(f"sanitize_velocity_cm_s: expected 1D or 2D, got shape {v.shape}")


def get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest,
                       which_type=None):
    if which_type == "EC_animal_average":
        arrs = []
        for animal, fd in factors_dict_EC.items():
            v = np.asarray(fd["Velocity"])[:, :58]  # expect (50,58)
            if v.shape[0] != 50:
                raise ValueError(f"{animal}: expected 50 position bins, got {v.shape[0]}")
            v = sanitize_velocity_cm_s(v, min_vel_cm_s=1e-3 * 100)  # cm/s, finite, >0
            arrs.append(v.astype(np.float32, copy=False))
        arrs = np.asarray(arrs, dtype=np.float32)  # (n_animals, 50, 58)
        return np.nanmean(arrs, axis=0)            # (50, 58)
    # ... keep your other modes if needed



# def get_activity_multidendrite2(animal_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, dt_constant, dx, dend_threshold=20, vel_applied="real", example_cell=15, include_inhibition=None, use_model_EC=False):

#     if use_model_EC:
        
#         dend_list = []
#         for j in range(792):
#             ts_list = []
#             for i in range(58):
#                 ts = random_timeseries(1.0, 0.005, 49)
#                 ts_list.append(ts)

#             dend_contribution_EC = np.array(ts_list).T
#             dend_list.append(dend_contribution_EC)
            
#         EC_input_matrix = np.array(dend_list)


#     if include_inhibition == 'both':
#         dend_activity = activity_EC - (activity_NDNF*NDNF_sf_opt + activity_SST*SST_sf_opt)

#     elif include_inhibition == 'sst':
#         dend_activity = activity_EC - (activity_SST*SST_sf_opt)
        
#     else:
#         dend_activity = activity_EC 

#     # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)
        
#     plateau_positions_counter = np.zeros(50)
#     plateau_start_positions_counter = np.zeros(50)


#     plateau_array_per_dendrite_list = []

#     num_plateaus_per_dend_list = []

#     dendrite_plateau_mask = np.zeros((dend_activity.shape[0], 50), dtype=bool)

#     padded_warped_activity_list = []
    
#     plateau_start_times_list_mega_list = []

#     # n_timebins, n_trials = plateau_array.shape



#     for d_idx in range(dend_activity.shape[0]):
 
#         position_bins, num_trials = animal_velocity.shape

#         padded_warped_activity = activity_EC[d_idx, :, :]

#         flat_padded_warped_activity = padded_warped_activity.flatten()
#         flat_plateau_array = np.zeros_like(flat_padded_warped_activity)

#         i = 0
#         while i < len(flat_padded_warped_activity):
#             if flat_padded_warped_activity[i] > dend_threshold:
#                 flat_plateau_array[i:i+300] = 1

#                 i += 800
#             else:
#                 i += 100


#         plateau_array = flat_plateau_array.reshape(padded_warped_activity.shape)
#         plateau_array_per_dendrite_list.append(plateau_array)


#         plateau_start_times_list = []

#         for trial in range(num_trials):
#             velocity_trial = animal_velocity[:, trial]
#     #         velocity_trial = proper_velocity
#             dt_trial = dx / velocity_trial  # in seconds
#             time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

#             plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
#             plateau_start_times = plateau_start_indices * dt_constant  # in seconds
#             plateau_start_times_list.append(plateau_start_times)

#             for pt_start_time in plateau_start_times:
#                 if pt_start_time != 0.0:
#                     for pos_idx in range(50):
#                         if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
#                             plateau_start_positions_counter[pos_idx] += 1
#                             break

#         plateau_start_times_list_mega_list.append(plateau_start_times_list)

#         num_plateaus_list = []


#         for trial in range(num_trials):
#             velocity_trial = animal_velocity[:, trial]
#     #         velocity_trial = proper_velocity
#             dt_trial = dx / velocity_trial  # in seconds
#             time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

#             plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]

#             plateau_start_times = plateau_start_indices * dt_constant  # in seconds
#             num_plateaus_list.append(len(plateau_start_times))

#             for pt_start_time in plateau_start_times:
#                 if pt_start_time != 0.0:
#                     for pos_idx in range(50):
#                         if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
#                             dendrite_plateau_mask[d_idx, pos_idx] = True
#                             break

#         num_plateaus_per_dend_list.append(np.sum(num_plateaus_list))





#         num_time_bins = padded_warped_activity.shape[1]



#         position_bins = 50

#         for trial in range(num_trials):
#             velocity_trial = animal_velocity[:, trial]  # shape (50,)
#             dt_trial = dx / velocity_trial              # shape (50,)
#             bin_edges = np.concatenate([[0], np.cumsum(dt_trial)])  # shape (51,)

#             time_bins = np.arange(num_time_bins) * dt_constant  # shape (num_time_bins,)

#             plateau_indices = np.where(plateau_array[trial] == 1)[0]  # shape (n_plateaus,)

#             if len(plateau_indices) == 0:
#                 continue

#             pt_times = time_bins[plateau_indices] 

#             pos_bin_idxs = np.searchsorted(bin_edges, pt_times, side='right') - 1

#             valid_mask = (pos_bin_idxs >= 0) & (pos_bin_idxs < position_bins)
#             pos_bin_idxs = pos_bin_idxs[valid_mask]

#             bin_counts = np.bincount(pos_bin_idxs, minlength=position_bins)

#             # --- Accumulate into global counter
#             plateau_positions_counter += bin_counts
            
#     plateau_start_times_list = plateau_start_times_list_mega_list[example_cell]
    
#     EC_used = use_model_EC

    


#     return plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, plateau_start_times_list_mega_list, num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list
    

def get_activity_multidendrite2(
    animal_velocity,
    activity_EC,
    activity_NDNF,
    activity_SST,
    NDNF_sf_opt,
    SST_sf_opt,
    dt_constant,
    dx,
    dend_threshold=20,
    vel_applied="real",
    example_cell=15,
    include_inhibition=None,
    use_model_EC=False,
    *,
    store_arrays=False,   # <---- NEW: keep False during optimization
):
    """
    Returns (same positions as before):
      plateau_positions_counter: (50,) int32
      plateau_start_positions_counter: (50,) int32
      plateau_array_per_dendrite_list: list[ (T,N) uint8 ] or None
      dendrite_plateau_mask: (D,50) uint8
      plateau_start_times_list_mega_list: list[list[np.ndarray]] or None
      num_plateaus_per_dend_list: list[float]
      dend_activity: (D,T,N) float32
      padded_warped_activity_list: None (kept for API compatibility)
    """

    # ---- prep EC activity (float32, contiguous) ----
    A_EC = np.asarray(activity_EC, dtype=np.float32, order="C")  # (D, T, N)

    # ---- choose activity with/without inhibition (float32) ----
    if include_inhibition == 'both':
        dend_activity = A_EC - (np.asarray(activity_NDNF, dtype=np.float32) * np.float32(NDNF_sf_opt)
                                + np.asarray(activity_SST, dtype=np.float32) * np.float32(SST_sf_opt))
    elif include_inhibition == 'sst':
        dend_activity = A_EC - (np.asarray(activity_SST, dtype=np.float32) * np.float32(SST_sf_opt))
    else:
        dend_activity = A_EC  # already float32

    D, T, N = dend_activity.shape
    position_bins, num_trials = animal_velocity.shape  # expect (50, 58)
    assert position_bins == 50, f"expected 50 pos bins, got {position_bins}"

    bin_edges_cache = []
    for trial in range(num_trials):
        v = animal_velocity[:, trial].astype(np.float32, copy=False)
        dt_trial = (np.float32(dx) / v).astype(np.float32, copy=False)
        bin_edges_cache.append(np.concatenate(([np.float32(0.0)], np.cumsum(dt_trial, dtype=np.float32))))

    # ---- outputs (small dtypes) ----
    plateau_positions_counter = np.zeros(50, dtype=np.int32)
    plateau_start_positions_counter = np.zeros(50, dtype=np.int32)
    dendrite_plateau_mask = np.zeros((D, 50), dtype=np.uint8)
    num_plateaus_per_dend_list = []

    # big structures only if explicitly requested
    plateau_array_per_dendrite_list = [] if store_arrays else None
    plateau_start_times_list_mega_list = [] if store_arrays else None
    padded_warped_activity_list = None  # never stored in hot path

    # ---- per-dendrite loop (reuse buffers; avoid Python lists of big arrays) ----
    # NOTE: we allocate a single flat buffer per dendrite and reuse it; no growth.
    flat_len = T * N
    flat_plateau = np.zeros(flat_len, dtype=np.uint8)  # reused per dendrite

    # Precompute time grid used for mapping indices -> time
    time_bins = (np.arange(T, dtype=np.int32) * np.float32(dt_constant)).astype(np.float32, copy=False)

    for d_idx in range(D):
        # view only (no copy)
        padded_warped_activity = dend_activity[d_idx, :, :]          # (T, N)
        flat_padded = padded_warped_activity.ravel(order="C")        # view

        # zero the reusable flat mask
        flat_plateau.fill(0)

        # ---- build plateau mask sparsely (uint8) ----
        i = 0
        thr = np.float32(dend_threshold)
        # (keep this identical to your rule)
        while i < flat_len:
            if flat_padded[i] > thr:
                end = i + 300
                if end > flat_len:
                    end = flat_len
                flat_plateau[i:end] = 1
                i += 800
            else:
                i += 100

        # reshape only when needed
        if store_arrays:
            plateau_array = flat_plateau.reshape(T, N)  # uint8 view
            plateau_array_per_dendrite_list.append(plateau_array)

        # ---- count plateaus per trial, map starts and positions to 50 bins ----
        num_plateaus_sum = 0

        if store_arrays:
            plateau_start_times_list = []

        for trial in range(num_trials):
            # per-trial mask (view)
            trial_mask = flat_plateau[trial::N][:T].view()  # (T,) stepping by N
            # start indices where mask rises 0->1
            rising = np.flatnonzero(np.diff(np.pad(trial_mask, (1, 0), mode='constant')) == 1).astype(np.int32)

            num_plateaus_sum += rising.size

            # time spent in each position bin this trial
            velocity_trial = animal_velocity[:, trial].astype(np.float32, copy=False)
            dt_trial = (np.float32(dx) / velocity_trial).astype(np.float32, copy=False)  # (50,)
            bin_edges = bin_edges_cache[trial]

            # bin_edges = np.concatenate(([np.float32(0.0)], np.cumsum(dt_trial, dtype=np.float32)))  # (51,)

            # map plateau starts to position bins
            if rising.size:
                start_times = time_bins[rising]  # (k,)
                # searchsorted expects ascending bin_edges
                pos_idx = np.searchsorted(bin_edges, start_times, side='right').astype(np.int32) - 1
                # keep valid ones
                vmask = (pos_idx >= 0) & (pos_idx < 50)
                pos_idx = pos_idx[vmask]
                if pos_idx.size:
                    # accumulate start positions
                    # faster than bincount for small k, but either is fine
                    counts = np.bincount(pos_idx, minlength=50).astype(np.int32, copy=False)
                    plateau_start_positions_counter += counts

            # mark dendrite bins that had any plateau on this trial
            if rising.size:
                # compute ALL plateau times (not just starts) if needed:
                # here we only have starts; to keep mask semantics close,
                # we mark bins that hosted a start (cheap)
                dendrite_plateau_mask[d_idx, pos_idx] = 1

            if store_arrays:
                plateau_start_times_list.append(start_times if rising.size else np.empty((0,), dtype=np.float32))

        num_plateaus_per_dend_list.append(float(num_plateaus_sum))

        if store_arrays:
            plateau_start_times_list_mega_list.append(plateau_start_times_list)

        # ---- map *all* plateau samples to positions counter (optional, cheap approx) ----
        # If you need the original "plateau_positions_counter" (not just starts),
        # approximate by sampling where mask==1 at each trial:
        for trial in range(num_trials):
            trial_mask = flat_plateau[trial::N][:T]
            if not np.any(trial_mask):
                continue
            velocity_trial = animal_velocity[:, trial].astype(np.float32, copy=False)
            dt_trial = (np.float32(dx) / velocity_trial).astype(np.float32, copy=False)
            # bin_edges = np.concatenate(([np.float32(0.0)], np.cumsum(dt_trial, dtype=np.float32)))
            bin_edges = bin_edges_cache[trial]
            pt_times = time_bins[trial_mask.astype(bool, copy=False)]
            if pt_times.size:
                pos_idx = np.searchsorted(bin_edges, pt_times, side='right').astype(np.int32) - 1
                vmask = (pos_idx >= 0) & (pos_idx < 50)
                pos_idx = pos_idx[vmask]
                if pos_idx.size:
                    counts = np.bincount(pos_idx, minlength=50).astype(np.int32, copy=False)
                    plateau_positions_counter += counts

    return (plateau_positions_counter,
            plateau_start_positions_counter,
            plateau_array_per_dendrite_list,
            dendrite_plateau_mask,
            plateau_start_times_list_mega_list,
            num_plateaus_per_dend_list,
            dend_activity,                    # float32
            None)                             # padded_warped_activity_list (not used in hot path)




def get_activity_multidendrite2_multiple_seeds(animal_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, dt_constant, dx, dend_threshold=20, vel_applied="real", example_cell=15, include_inhibition=None, use_model_EC=False):

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
        dend_activity_dict = activity_EC 

    # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)
        
    plateau_positions_counter = np.zeros(50)
    plateau_start_positions_counter = np.zeros(50)


    plateau_array_per_dendrite_list = []

    num_plateaus_per_dend_list = []


    seed_list = []
    for seed in dend_activity_dict:
        seed_data = dend_activity_dict[seed]
        seed_list.append(seed_data)


    seed_array = np.array(seed_list)

    dend_activity = np.mean(seed_array, axis=0)


    dendrite_plateau_mask = np.zeros((dend_activity.shape[0], 50), dtype=bool)

    padded_warped_activity_list = []
    
    plateau_start_times_list_mega_list = []

    # n_timebins, n_trials = plateau_array.shape


    for d_idx in range(dend_activity.shape[0]):
 
        position_bins, num_trials = animal_velocity.shape

        padded_warped_activity = dend_activity[d_idx, :, :]

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
    

def get_dend_vm_from_cells_multi(cells_dict, Vrest=-60.0, epsp_sf=0.1):
    if not cells_dict:
        z = np.zeros((0, 0), dtype=np.float32)
        return z, z, {}, []

    def _trial_sort_key(x):
        try: return (0, int(x))
        except (ValueError, TypeError): return (1, str(x))

    trial_ids = set()
    global_T_epsp = 0
    global_T_spk  = 0
    for payload in cells_dict.values():
        ceps = payload["epsps"]; cspk = payload["spike_train"]
        trial_ids.update(ceps.keys())
        for v in ceps.values(): global_T_epsp = max(global_T_epsp, len(v))
        for v in cspk.values(): global_T_spk  = max(global_T_spk,  len(v))

    trial_ids = sorted(trial_ids, key=_trial_sort_key)
    n_trials = len(trial_ids)

    cell_epsp_mats = []
    spikes_per_cell = {}
    for cell, payload in cells_dict.items():
        ceps = payload["epsps"]; cspk = payload["spike_train"]
        epsp_rows, spk_rows = [], []
        for tid in trial_ids:
            ev = np.asarray(ceps.get(tid, []), dtype=np.float32)
            sv = np.asarray(cspk.get(tid, []), dtype=np.float32)
            if ev.size < global_T_epsp:
                ev = np.pad(ev, (0, global_T_epsp - ev.size), constant_values=np.float32(np.nan))
            if sv.size < global_T_spk:
                sv = np.pad(sv, (0, global_T_spk - sv.size), constant_values=np.float32(np.nan))
            epsp_rows.append(ev); spk_rows.append(sv)
        epsp_mat = np.vstack(epsp_rows).astype(np.float32, copy=False)
        spk_mat  = np.vstack(spk_rows).astype(np.float32,  copy=False)
        cell_epsp_mats.append(epsp_mat)
        spikes_per_cell[cell] = spk_mat

    epsp_stack = np.stack(cell_epsp_mats, axis=0).astype(np.float32, copy=False)
    valid_counts = np.sum(~np.isnan(epsp_stack), axis=0).astype(np.int32, copy=False)
    sum_epsp = np.nansum(epsp_stack, axis=0).astype(np.float32, copy=False)
    sum_epsp[valid_counts == 0] = np.float32(np.nan)

    # keep float32: nanmean returns float64, so cast back
    trial_means = np.nanmean(sum_epsp, axis=1, keepdims=True).astype(np.float32, copy=False)
    sum_epsp_centered = (sum_epsp - trial_means).astype(np.float32, copy=False)
    dend_Vm = np.float32(Vrest) + np.float32(epsp_sf) * sum_epsp_centered

    return dend_Vm, epsp_stack, spikes_per_cell



# def get_dend_vm_from_cells_multi(cells_dict, Vrest=-60.0, epsp_sf=0.1):
#     """
#     Build dendritic Vm for a *single animal* given per-cell EPSPs/spike trains.

#     Parameters
#     ----------
#     cells_dict : dict
#         {cell: {"epsps": {trial_id: 1D array}, "spike_train": {trial_id: 1D array}}}
#     Vrest : float
#         Resting potential (mV).
#     epsp_sf : float
#         Scale factor for EPSP sum → Vm.

#     Returns
#     -------
#     dend_Vm : (n_trials, T) float32
#         Vrest + epsp_sf * (sum across cells, per-trial centered).
#     sum_epsp_centered : (n_trials, T) float32
#         Summed EPSPs across cells (masked by NaN) and centered per trial.
#     spikes_per_cell : {cell: (n_trials, T_spk) float32}
#         Spike trains per cell, aligned & padded across trials (NaN for padding).
#     trial_ids : list
#         The trial order used for rows in the outputs.
#     """

#     def _trial_sort_key(x):
#         # ints first in numeric order, then strings
#         try:
#             return (0, int(x))
#         except (ValueError, TypeError):
#             return (1, str(x))

#     if not cells_dict:
#         return (np.zeros((0, 0), dtype=np.float32),
#                 np.zeros((0, 0), dtype=np.float32),
#                 {},
#                 [])

#     # ---- discover global trial set and global time lengths
#     trial_ids = set()
#     global_T_epsp = 0
#     global_T_spk  = 0
#     for cell, payload in cells_dict.items():
#         ceps = payload["epsps"]
#         cspk = payload["spike_train"]
#         trial_ids.update(ceps.keys())
#         for v in ceps.values():
#             global_T_epsp = max(global_T_epsp, len(v))
#         for v in cspk.values():
#             global_T_spk = max(global_T_spk, len(v))

#     trial_ids = sorted(trial_ids, key=_trial_sort_key)
#     n_trials = len(trial_ids)

#     # ---- build (n_trials, T) per-cell EPSP and spike matrices (padded with NaN)
#     cell_epsp_mats = []
#     spikes_per_cell = {}
#     for cell, payload in cells_dict.items():
#         ceps = payload["epsps"]
#         cspk = payload["spike_train"]

#         epsp_rows, spk_rows = [], []
#         for tid in trial_ids:
#             ev = np.asarray(ceps.get(tid, []), dtype=np.float32)
#             sv = np.asarray(cspk.get(tid, []), dtype=np.float32)

#             if ev.size < global_T_epsp:
#                 ev = np.pad(ev, (0, global_T_epsp - ev.size), constant_values=np.nan)
#             if sv.size < global_T_spk:
#                 sv = np.pad(sv, (0, global_T_spk - sv.size), constant_values=np.nan)

#             epsp_rows.append(ev)
#             spk_rows.append(sv)

#         epsp_mat = np.vstack(epsp_rows).astype(np.float32, copy=False)  # (n_trials, global_T_epsp)
#         spk_mat  = np.vstack(spk_rows).astype(np.float32, copy=False)   # (n_trials, global_T_spk)

#         cell_epsp_mats.append(epsp_mat)
#         spikes_per_cell[cell] = spk_mat

#     # ---- stack cells -> (n_cells, n_trials, T) and masked-sum across cells
#     epsp_stack = np.stack(cell_epsp_mats, axis=0)           # (n_cells, n_trials, global_T_epsp)
#     valid_counts = np.sum(~np.isnan(epsp_stack), axis=0)    # (n_trials, T)
#     sum_epsp = np.nansum(epsp_stack, axis=0)                # (n_trials, T)
#     sum_epsp[valid_counts == 0] = np.nan                    # keep NaN where no cell contributed

#     # ---- per-trial centering (across time)
#     trial_means = np.nanmean(sum_epsp, axis=1, keepdims=True)  # (n_trials, 1)
#     sum_epsp_centered = sum_epsp - trial_means

#     # ---- dendritic Vm
#     dend_Vm = Vrest + epsp_sf * sum_epsp_centered

#     return dend_Vm, epsp_stack, spikes_per_cell



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
        print(f"animal {animal}")
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

# def clean_velocity(V, min_v=1e-6, max_v=None):
#     """
#     V: (n_pos, n_trials) velocity matrix (float)
#     - Interpolates non-finite entries along position within each trial.
#     - If an entire trial is non-finite, replaces it with per-position median across other trials.
#     - Clamps negative/too-small velocities.
#     """
#     V = np.asarray(V, dtype=float).copy()
#     n_pos, n_trials = V.shape
#     x = np.arange(n_pos)

#     # Pass 1: interpolate along position within each trial
#     for t in range(n_trials):
#         col = V[:, t]
#         good = np.isfinite(col)
#         if good.any():
#             # edge-fill using first/last finite
#             left_val  = col[good][0]
#             right_val = col[good][-1]
#             col[~good] = np.interp(x[~good], x[good], col[good], left=left_val, right=right_val)
#             V[:, t] = col
#         else:
#             # whole trial bad: fill with per-position median across trials (ignoring NaNs)
#             med = np.nanmedian(V, axis=1)
#             # if med still has NaNs (everything is bad), fall back to zeros
#             med = np.nan_to_num(med, nan=0.0)
#             V[:, t] = med

#     # Pass 2: clamp negatives and very small velocities (avoid future dx/v blowups)
#     V = np.where(~np.isfinite(V), 0.0, V)     # just in case
#     V = np.where(V < min_v, min_v, V)         # enforce strictly positive
#     if max_v is not None:
#         V = np.clip(V, None, max_v)
#     return V


def clean_velocity(V, min_v=1e-6, max_v=None):
    V = np.asarray(V, dtype=np.float32).copy()
    n_pos, n_trials = V.shape
    x = np.arange(n_pos, dtype=np.int32)

    for t in range(n_trials):
        col = V[:, t]
        good = np.isfinite(col)
        if good.any():
            left_val  = col[good][0]
            right_val = col[good][-1]
            col[~good] = np.interp(x[~good], x[good], col[good], left=left_val, right=right_val).astype(np.float32)
            V[:, t] = col
        else:
            med = np.nanmedian(V, axis=1).astype(np.float32)
            med = np.nan_to_num(med, nan=0.0).astype(np.float32)
            V[:, t] = med

    V = np.where(~np.isfinite(V), np.float32(0.0), V)
    V = np.where(V < min_v, np.float32(min_v), V)
    if max_v is not None:
        V = np.clip(V, None, np.float32(max_v))
    return V.astype(np.float32, copy=False)




# def _pad_rows_to_2d(rows, fill=np.nan):
#     """
#     rows: iterable of 1D arrays/lists (ragged by time)
#     returns: 2D float array (n_trials, max_T_cell) padded with NaN
#     """
#     # convert each row to float array; allow empty rows
#     rows_np = [np.asarray(r, dtype=float).ravel() for r in rows]
#     max_T = max((len(r) for r in rows_np), default=0)
#     out = np.full((len(rows_np), max_T), fill, dtype=float)
#     for i, r in enumerate(rows_np):
#         if r.size:
#             out[i, :r.size] = r
#     return out  # shape (n_trials, max_T_cell)

# def pad_stack_same_trials_ragged(padded_warped_activity_dict, n_trials_expected=58):
#     """
#     padded_warped_activity_dict[animal][cell] = either:
#       - 2D array (n_trials, T_cell), or
#       - list/iterable of length n_trials with 1D arrays of variable length
#     Returns:
#       stacked: (n_cells_total, n_trials_expected, max_T_global)
#       keys:    [(animal, cell), ...]
#       max_T_global: int
#     """
#     per_cell = []
#     keys = []
#     max_T_global = 0

#     for animal, cells in padded_warped_activity_dict.items():
#         for cell, A in cells.items():
#             # Normalize to 2D float array (n_trials, T_cell)
#             if isinstance(A, np.ndarray) and A.ndim == 2:
#                 A2 = A.astype(float, copy=False)
#             else:
#                 # Assume ragged-by-trial
#                 A2 = _pad_rows_to_2d(A)

#             if A2.shape[0] != n_trials_expected:
#                 raise ValueError(f"{animal}/{cell}: expected {n_trials_expected} trials, got {A2.shape[0]}")

#             per_cell.append(A2)
#             keys.append((animal, cell))
#             max_T_global = max(max_T_global, A2.shape[1])

#     # Second pass: pad each cell array to global max_T and stack
#     n_cells = len(per_cell)
#     stacked = np.full((n_cells, n_trials_expected, max_T_global), np.nan, dtype=float)
#     for i, A2 in enumerate(per_cell):
#         T = A2.shape[1]
#         stacked[i, :, :T] = A2

#     return stacked, keys, max_T_global


def _pad_rows_to_2d(rows, fill=np.float32(np.nan)):
    rows_np = [np.asarray(r, dtype=np.float32).ravel() for r in rows]
    max_T = max((len(r) for r in rows_np), default=0)
    out = np.full((len(rows_np), max_T), np.float32(fill), dtype=np.float32)
    for i, r in enumerate(rows_np):
        if r.size:
            out[i, :r.size] = r
    return out

def pad_stack_same_trials_ragged(padded_warped_activity_dict, n_trials_expected=58):
    per_cell = []
    keys = []
    max_T_global = 0
    for animal, cells in padded_warped_activity_dict.items():
        for cell, A in cells.items():
            if isinstance(A, np.ndarray) and A.ndim == 2:
                A2 = np.asarray(A, dtype=np.float32, order="C")
            else:
                A2 = _pad_rows_to_2d(A)
            if A2.shape[0] != n_trials_expected:
                raise ValueError(f"{animal}/{cell}: expected {n_trials_expected} trials, got {A2.shape[0]}")
            per_cell.append(A2)
            keys.append((animal, cell))
            max_T_global = max(max_T_global, A2.shape[1])

    n_cells = len(per_cell)
    stacked = np.full((n_cells, n_trials_expected, max_T_global),
                      np.float32(np.nan), dtype=np.float32)
    for i, A2 in enumerate(per_cell):
        T = A2.shape[1]
        stacked[i, :, :T] = A2
    return stacked, keys, max_T_global




# def activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1,
#                         center_across="time_trials", dtype=np.float32):
#     A = np.asarray(activity_EC, dtype=float)  # (D, T, N)
#     if A.ndim != 3:
#         raise ValueError(f"activity_EC must be 3D; got {A.shape}")

#     if center_across == "time":
#         mu = np.nanmean(A, axis=1, keepdims=True)          # (D,1,N)  <-- removes trial offsets
#     elif center_across == "time_trials":
#         mu = np.nanmean(A, axis=(1,2), keepdims=True)       # (D,1,1)  <-- keeps trial variability
#     elif center_across == "none":
#         mu = 0.0                                            # no centering
#     else:
#         raise ValueError("center_across must be 'time', 'time_trials', or 'none'")

#     A_centered = A - mu
#     dend_Vm = Vrest + vm_scale * A_centered
#     return dend_Vm.astype(dtype), A_centered.astype(dtype), (mu if np.isscalar(mu) else mu.astype(dtype))

def activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1,
                        center_across="time_trials", dtype=np.float32):
    A = np.asarray(activity_EC, dtype=np.float32, order="C")
    if A.ndim != 3:
        raise ValueError(f"activity_EC must be 3D; got {A.shape}")

    if center_across == "time":
        mu = np.nanmean(A, axis=1, keepdims=True).astype(np.float32, copy=False)
    elif center_across == "time_trials":
        mu = np.nanmean(A, axis=(1,2), keepdims=True).astype(np.float32, copy=False)
    elif center_across == "none":
        mu = np.float32(0.0)
    else:
        raise ValueError("center_across must be 'time', 'time_trials', or 'none'")

    A_centered = (A - mu).astype(np.float32, copy=False)
    dend_Vm = np.float32(Vrest) + np.float32(vm_scale) * A_centered
    return dend_Vm, A_centered, (mu if np.isscalar(mu) else mu.astype(np.float32, copy=False))



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


# def get_dendrite_activity_multi(weights, EC_input_matrix, n_dendrites, n_EC):
            
#             E, T, N = EC_input_matrix.shape
#             EC_flat = EC_input_matrix.reshape(E, T*N)      # row-major: blocks of N per time bin
#             dendrite_flat = weights @ EC_flat              # (D, T*N)
#             return dendrite_flat.reshape(n_dendrites, T, N)


# 2) get_dendrite_activity_multi: enforce float32 & views
def get_dendrite_activity_multi(weights, EC_input_matrix, n_dendrites, n_EC):
    W = np.asarray(weights, dtype=np.float32, order="C")
    E = np.asarray(EC_input_matrix, dtype=np.float32, order="C")  # (E, T, N)
    E_cnt, T, N = E.shape
    assert E_cnt == n_EC, f"EC_input_matrix first dim {E_cnt} != n_EC {n_EC}"
    EC_flat = E.reshape(n_EC, T * N)            # view
    dendrite_flat = W @ EC_flat                 # float32 GEMM
    return dendrite_flat.reshape(n_dendrites, T, N)




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




def _ten_bin_fraction_from_counter(cnt50: np.ndarray) -> np.ndarray:
    cnt50 = np.asarray(cnt50, float)
    agg10 = np.add.reduceat(cnt50, np.arange(0, 50, 5))
    s = agg10.sum()
    return agg10 / s if s > 0 else np.full(10, 0.1, float)

def _build_static_inputs(spike_model: "SpikingModel", animal: str):
    cfg = spike_model.cfg
    factors_dict_EC  = spike_model.data["factors_dict_EC"]
    activity_dict_EC = spike_model.data["activity_dict_EC"]

    vel_by_animal = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
    animal_velocity = vel_by_animal[animal]  # (50, T)

    ret = get_scaled_data_Hz_dict({animal: activity_dict_EC[animal]}, Hz_SF=cfg.hz_sf)
    if isinstance(ret, tuple) and len(ret) == 2:
        scaled_data_Hz_dict, cells_per_animal_dict = ret
        n_EC = int(cells_per_animal_dict[animal])
    else:
        scaled_data_Hz_dict = ret
        n_EC = len(scaled_data_Hz_dict[animal])

    pwa_cell_dict, _ = do_the_interpolation_an(
        scaled_data_Hz_dict[animal], animal_velocity, dt_constant=cfg.dt_constant
    )

    return dict(
        animal=animal,
        animal_velocity=animal_velocity,
        pwa_cell_dict=pwa_cell_dict,
        n_EC=n_EC,
        n_dendrites=100,
        dt_constant=cfg.dt_constant,
        dx=cfg.dx,
        vrest=cfg.vrest,
        epsp_sf=cfg.epsp_sf,
        dist=cfg.dist,  # will overwrite below
    )


def _build_static_inputs_animal_av(spike_model: "SpikingModel", animal_average=False):
    cfg = spike_model.cfg
    factors_dict_EC  = spike_model.data["factors_dict_EC"]
    factors_dict_SST = spike_model.data["factors_dict_SST"]
    fixed_filtered_factors_dict_NDNF_newest = spike_model.data["fixed_filtered_factors_dict_NDNF_newest"]
    residual_activity_dict_SST = spike_model.data["residual_activity_dict_SST"]
    residual_activity_dict_EC = spike_model.data["residual_activity_dict_EC"]
    fixed_residual_activity_dict_NDNF_newest = spike_model.data["fixed_residual_activity_dict_NDNF_newest"]
    GLM_params_EC = spike_model.data["GLM_params_EC"]
    GLM_params_SST = spike_model.data["GLM_params_SST"]
    GLM_params_NDNF = spike_model.data["GLM_params_NDNF"]


    an_velocity = get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest, which_type="EC_animal_average")


    an_velocity_dict = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
    scaled_data_Hz_dict_resid_EC, cells_per_animal_dict = get_scaled_data_Hz_dict(residual_activity_dict_EC, Hz_SF=50)
    scaled_data_Hz_dict_EC = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_EC, GLM_params_EC, an_velocity_dict)

    # scaled_data_Hz_dict_resid_SST = get_scaled_data_Hz_dict(residual_activity_dict_SST, Hz_SF=50)
    # scaled_data_Hz_dict_SST = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_SST, GLM_params_SST, an_velocity_dict)

    # scaled_data_Hz_dict_resid_NDNF = get_scaled_data_Hz_dict(fixed_residual_activity_dict_NDNF_newest, Hz_SF=50)
    # scaled_data_Hz_dict_NDNF = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_NDNF, GLM_params_NDNF, an_velocity_dict)

    pwa_cell_dict, _ = do_the_interpolation(scaled_data_Hz_dict_EC, an_velocity=an_velocity)

    return dict(
        animal_velocity=an_velocity,
        pwa_cell_dict=pwa_cell_dict,
        n_EC=792,
        n_dendrites=100,
        dt_constant=cfg.dt_constant,
        dx=cfg.dx,
        vrest=cfg.vrest,
        epsp_sf=cfg.epsp_sf,
        dist=cfg.dist,  # will overwrite below
    )

def _eval_one_seed(params, seed, static, animal_average=False):
    import os
    import numpy as np
    try:
        # Linux: ru_maxrss in KB; divide by 1024 → MB
        import resource
        def _rss_mb():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        def _rss_mb():
            return float("nan")
    try:
        from mpi4py import MPI
        _rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        _rank = -1  # not under MPI

    tau_ms         = float(params["tau_ms"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])
    dend_threshold = float(params["dend_threshold"])

    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = np.asarray(static["animal_velocity"], dtype=np.float32, order="C")
    pwa_cell_dict   = static["pwa_cell_dict"]

    # --- EPSPs (ensure float32 path) ---
    if animal_average:
        epsp_cells, _ = get_epsp_dict_multi(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=seed)
        _, epsp_input_matrix, _ = get_dend_vm(epsp_cells, Vrest=np.float32(static["vrest"]), epsp_sf=np.float32(static["epsp_sf"]))
        epsp_eTN = np.transpose(epsp_input_matrix, (0, 2, 1)) 
    else:
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(seed))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=np.float32(static["vrest"]), epsp_sf=np.float32(static["epsp_sf"]))

    # want (E, T, N)
    if epsp_eTN.shape[1] != animal_velocity.shape[1]:
        epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))
    epsp_eTN = np.ascontiguousarray(epsp_eTN, dtype=np.float32)

    # --- Weights (float32) ---
    rng = np.random.default_rng(12345 + int(seed))
    connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=np.bool_)
    weights_EC = sample_weights(
        static["dist"], connection_mask_EC, rng=rng,
        mean=weights_mean, std=weights_std
    ).astype(np.float32, copy=False)

    # --- EC activity -> dendritic Vm (float32, contiguous) ---
    activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
    activity_EC = np.asarray(activity_EC, dtype=np.float32, order="C")
    dend_Vm, _, _ = activity_to_dend_vm(
        activity_EC, Vrest=np.float32(-70.0), vm_scale=np.float32(0.1), center_across="time_trials"
    )

    # --- Plateaus: ask NOT to build/store huge arrays ---
    res = get_activity_multidendrite2(
        animal_velocity, dend_Vm,
        activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
        dt_constant=static["dt_constant"], dx=static["dx"], dend_threshold=np.float32(dend_threshold),
        vel_applied="real", example_cell=15, include_inhibition=None, use_model_EC=False,
        store_arrays=False,  # <---- IMPORTANT
    )
    # Pull only what you need
    start_pos_cnt50 = res[1]                      # (50,) int counts
    num_plateaus_per_dend_list = res[5]           # list/array per dendrite

    # --- instrumentation: memory + shapes (before freeing) ---
    def _mb(x):
        try:
            return x.nbytes / 1e6
        except Exception:
            return 0.0
    print(
        f"animal_average={animal_average}",
        f"[SEED] rank={_rank} pid={os.getpid()} seed={int(seed)} "
        f"RSS={_rss_mb():.1f}MB | "
        f"epsp_eTN(shape={tuple(epsp_eTN.shape)}, { _mb(epsp_eTN):.1f}MB) "
        f"W({ _mb(weights_EC):.1f}MB) "
        f"actEC(shape={tuple(activity_EC.shape)}, { _mb(activity_EC):.1f}MB) "
        f"Vm(shape={tuple(dend_Vm.shape)}, { _mb(dend_Vm):.1f}MB) "
        f"n_dendrites={n_dendrites} n_EC={n_EC}",
        flush=True
    )

    # --- runtime hygiene: drop big temps immediately ---
    try: del epsp_cells
    except NameError: pass
    try: del epsp_eTN
    except NameError: pass
    try: del weights_EC
    except NameError: pass
    try: del activity_EC
    except NameError: pass
    try: del dend_Vm
    except NameError: pass
    # Optional: import gc; gc.collect()

    return start_pos_cnt50, np.asarray(num_plateaus_per_dend_list, dtype=np.float32)



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

 


# def plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=False):
    
#     if include_inhibition == 'both':
#         # dend_activity = activity_EC - (activity_NDNF*NDNF_sf_opt + activity_SST*SST_sf_opt)

#         # dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)

#         fig, axs = plt.subplots(5,4, figsize=(30,25))
#         fig.suptitle(f"Ratio SST Contribution : NDNF Contribution = {SST_contribution_sum / NDNF_contribution_sum:.3f}", y=1.0)

#         activity_EC_trial_av = np.mean(activity_EC, axis=2)
#         mean_activity_EC_trial_av = np.mean(activity_EC_trial_av, axis=0)
#         for i in range(activity_EC_trial_av.shape[0]):
#             axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
#         axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
#         axs[0,0].set_title("EC Input To Each Dendrite SF=1")
#         axs[0,0].set_ylabel("Activity")
#         axs[0,0].set_xlabel("Position Bins")


#         activity_SST_trial_av = np.mean(activity_SST, axis=2)
#         mean_activity_SST_trial_av = np.mean(activity_SST_trial_av, axis=0)
#         for i in range(activity_SST_trial_av.shape[0]):
#             axs[0,1].plot(activity_SST_trial_av[i,:], alpha=0.2)
#         axs[0,1].plot(mean_activity_SST_trial_av, linewidth=3, color='r', linestyle='--')
#         axs[0,1].set_title(f"SST Input To Each Dendrite SF={SST_sf_opt:.3f}")
#         axs[0,1].set_ylabel("Activity")
#         axs[0,1].set_xlabel("Position Bins")

#         activity_NDNF_trial_av = np.mean(activity_NDNF, axis=2)
#         mean_activity_NDNF_trial_av = np.mean(activity_NDNF_trial_av, axis=0)
#         for i in range(activity_NDNF_trial_av.shape[0]):
#             axs[0,2].plot(activity_NDNF_trial_av[i,:], alpha=0.2)
#         axs[0,2].plot(mean_activity_NDNF_trial_av, linewidth=3, color='r', linestyle='--')
#         axs[0,2].set_title(f"NDNF Input To Each Dendrite SF={NDNF_sf_opt:.3f}")
#         axs[0,2].set_ylabel("Activity")
#         axs[0,2].set_xlabel("Position Bins")

#         im1 = axs[1,0].imshow(weights_EC, aspect='auto')
#         axs[1,0].set_title(f"EC Weights: {dist}")
#         axs[1,0].set_ylabel("Dendrites")
#         axs[1,0].set_xlabel("Input Cells")
#         fig.colorbar(im1, ax=axs[1,0])

#         im1 = axs[1,1].imshow(weights_SST, aspect='auto')
#         axs[1,1].set_title(f"SST Weights: Equal")
#         axs[1,1].set_ylabel("Dendrites")
#         axs[1,1].set_xlabel("Input Cells")
#         fig.colorbar(im1, ax=axs[1,1])

#         im1 = axs[1,2].imshow(weights_NDNF, aspect='auto')
#         axs[1,2].set_title(f"NDNF Weights: Equal")
#         axs[1,2].set_ylabel("Dendrites")
#         axs[1,2].set_xlabel("Input Cells")
#         fig.colorbar(im1, ax=axs[1,2])

#         mean_pad = np.mean(activity_EC, axis=0)

#         im4 = axs[2,2].imshow(mean_pad, aspect='auto', interpolation='None')
#         axs[2,2].set_title("Mean Over Dendrites")
#         axs[2,2].set_ylabel("Trials")
#         axs[2,2].set_xlabel("Time (ms)")
#         fig.colorbar(im4, ax=axs[2,2])

#         means = np.mean(mean_pad, axis=0)
#         # means = means / np.max(means)
#         axs[2,3].plot(means)
#         axs[2,3].set_title("Mean of Dendrites and Trials")
#         axs[2,3].set_xlabel("Time (ms)")

#         axs[0,3].plot(an_velocity, color='r')
#         axs[0,3].set_xlabel("Position Bins")
#         axs[0,3].set_ylabel("Meters / Second")
#         axs[0,3].set_title("Velocity")

#         distance = 3.6
#         velocity_cm = an_velocity*100

#         occupancy = distance / velocity_cm
        
#         axs[1,3].plot(occupancy, color='r')
#         axs[1,3].set_xlabel("Position Bins")
#         axs[1,3].set_ylabel("Seconds")
#         axs[1,3].set_title("Occupancy")

#         ims = axs[3,0].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray')
#         axs[3,0].set_title(f"Ex/ Dendrite Plateaus Over Time \n Dendrite Threshold={dend_threshold}")
#         axs[3,0].set_xlabel("Time (ms)")
#         axs[3,0].set_ylabel("Trials")
#         fig.colorbar(ims, ax=axs[3,0])


#         # axs[1,2].hist(weights_EC.flatten(), bins=50)
#         # axs[1,2].set_title(f"Weights - {dist} Dist")
#         # axs[1,2].set_ylabel("Count")
#         # axs[1,2].set_xlabel("Weight")

#         mean_dend_activity = np.mean(dend_activity, axis=0)

#         im4 = axs[2,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
#         axs[2,0].set_title("Mean Over Dendrites")
#         axs[2,0].set_ylabel("Trials")
#         axs[2,0].set_xlabel("Position Bins")
#         fig.colorbar(im4, ax=axs[2,0])

#         num_plateaus_per_trial_list_across_dends = []
#         for dend in range(len(plateau_start_times_list_mega_list)):
#             num_plateaus_per_trial = []
#             dend_plateaus = plateau_start_times_list_mega_list[dend]
#             for trial in range(len(dend_plateaus)):
#                 num_plateaus_per_trial.append(len(dend_plateaus[trial]))

#             num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

#         num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
#         mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
#         sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

#         axs[4,2].set_title("Mean # Plateaus Per Trial Across Dendrites")
#         axs[4,2].set_ylabel("Mean # Plateaus Across Dendrites")
#         axs[4,2].set_xlabel("Session Length (%)")
#         axs[4,2].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], 
#                                 labels=["0", '25', "50", '75', "100"])
#         axs[4,2].plot(mean_plat_per_trial, color='k')
#         axs[4,2].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        
#         means = np.nanmean(mean_dend_activity, axis=1)
        
#         axs[2,1].plot(means)
#         axs[2,1].set_title("Mean of Dendrites and Trials")
#         axs[2,1].set_xlabel("Position Bins")

#         axs[3,1].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
#         axs[3,1].set_title(f"Plateau Time Across All Dendrites")
#         axs[3,1].set_ylabel("Time (ms)")
#         axs[3,1].set_xlabel("Position Bins")


#         n_bins = 10
#         bin_size = int(50 / n_bins)

#         summed_plateaus = np.zeros(n_bins)

#         for i in range(n_bins):
#             start = i * bin_size
#             end = (i + 1) * bin_size
#             summed_data = np.sum(plateau_start_positions_counter[start:end])
#             summed_plateaus[i] = summed_data

#         axs[3,2].bar(range(len(summed_plateaus)), summed_plateaus)
#         axs[3,2].set_xlabel("Position Bin")
#         axs[3,2].set_ylabel("Plateau Count")
#         axs[3,2].set_title("Plateau Onset Count per Track Section")
#         # axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])
#         axs[3,2].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

#         total_plateaus = np.sum(summed_plateaus)
#         fraction_plateaus = summed_plateaus / total_plateaus
#         axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
#         axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
#         axs[3,3].set_xlabel("Grouped Position Bins")
#         axs[3,3].set_ylabel("% of Total Plateaus")
#         # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])
#         axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

#         axs[4,0].hist(num_plateaus_per_dend_list, bins=10)
#         axs[4,0].set_xlabel("# of Plateaus")
#         axs[4,0].set_ylabel("# of Cells")

#         number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
#         axs[4,1].plot(number_dendrites_counter)
#         axs[4,1].set_title("Percent of Dendrites with Plateau at Location")
#         axs[4,1].set_ylabel("Percent")
#         axs[4,1].set_xlabel("Position Bin")

 

#         cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
#         mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
#         sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)


#         axs[4,3].set_title("Cumulative # Plateaus Across Trials")
#         axs[4,3].set_ylabel("# Plateaus Mean +- SEM Across Dends")
#         axs[4,3].set_xlabel("Session Length (%)")
#         axs[4,3].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
#                                 labels=["0", '25', "50", '75', "100"])
#         axs[4,3].plot(mean_plateaus_all_dends, color='k')
#         axs[4,3].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
        

#         plt.tight_layout()
#         plt.show()



#     elif include_inhibition == 'sst':

#         fig, axs = plt.subplots(5,4, figsize=(30,25))


#         activity_EC_trial_av = np.mean(activity_EC, axis=2)
#         mean_activity_EC_trial_av = np.mean(activity_EC_trial_av, axis=0)
#         for i in range(activity_EC_trial_av.shape[0]):
#             axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
#         axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
#         axs[0,0].set_title("EC Input To Each Dendrite")
#         axs[0,0].set_ylabel("Activity")
#         axs[0,0].set_xlabel("Position Bins")


#         activity_SST_trial_av = np.mean(activity_SST, axis=2)
#         mean_activity_SST_trial_av = np.mean(activity_SST_trial_av, axis=0)
#         for i in range(activity_SST_trial_av.shape[0]):
#             axs[0,1].plot(activity_SST_trial_av[i,:], alpha=0.2)
#         axs[0,1].plot(mean_activity_SST_trial_av, linewidth=3, color='r', linestyle='--')
#         axs[0,1].set_title("SST Input To Each Dendrite")
#         axs[0,1].set_ylabel("Activity")
#         axs[0,1].set_xlabel("Position Bins")

#         axs[0,2].plot(an_velocity, color='r')
#         axs[0,2].set_xlabel("Position Bins")
#         axs[0,2].set_ylabel("Meters / Second")
#         axs[0,2].set_title("Velocity")

#         an_velocity_cm = an_velocity*100
#         distance = 3.6

#         occupancy = distance / an_velocity_cm

#         axs[0,3].plot(occupancy, color='r')
#         axs[0,3].set_xlabel("Position Bins")
#         axs[0,3].set_ylabel("Seconds")
#         axs[0,3].set_title("Occupancy")

#         im1 = axs[1,0].imshow(weights_EC, aspect='auto')
#         axs[1,0].set_title(f"EC Weights {dist}")
#         axs[1,0].set_ylabel("Dendrites")
#         axs[1,0].set_xlabel("Input Cells")
#         fig.colorbar(im1, ax=axs[1,0])


#         padded_warped_activity_array = np.array(padded_warped_activity_list)

#         mean_pad = np.mean(padded_warped_activity_array, axis=0)

#         im4 = axs[2,2].imshow(mean_pad, aspect='auto', interpolation='None')
#         axs[2,2].set_title("Mean Over Dendrites")
#         axs[2,2].set_ylabel("Trials")
#         axs[2,2].set_xlabel("Time (ms)")
#         fig.colorbar(im4, ax=axs[2,2])

#         means = np.mean(mean_pad, axis=0)
#         # means = means / np.max(means)
#         axs[2,3].plot(means)
#         axs[2,3].set_title("Mean of Dendrites and Trials")
#         axs[2,3].set_xlabel("Time (ms)")

#         ims = axs[3,0].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray')
#         axs[3,0].set_title(f"Ex/ Dendrite Plateaus Over Time \n Dendrite Threshold={dend_threshold}")
#         axs[3,0].set_xlabel("Time (ms)")
#         axs[3,0].set_ylabel("Trials")
#         fig.colorbar(ims, ax=axs[3,0])

#         im1 = axs[1,1].imshow(weights_SST, aspect='auto')
#         axs[1,1].set_title(f"SST Weights: Equal")
#         axs[1,1].set_ylabel("Dendrites")
#         axs[1,1].set_xlabel("Input Cells")
#         fig.colorbar(im1, ax=axs[1,1])


#         axs[1,2].hist(weights_EC.flatten(), bins=50)
#         axs[1,2].set_title(f"EC Weights: {dist} Distriution")
#         axs[1,2].set_ylabel("Count")
#         axs[1,2].set_xlabel("Weight")

#         axs[1,3].hist(weights_SST.flatten(), bins=50)
#         axs[1,3].set_title(f"SST Weights: Equal Distriution")
#         axs[1,3].set_ylabel("Count")
#         axs[1,3].set_xlabel("Weight")

#         mean_dend_activity = np.mean(dend_activity, axis=0)

#         im4 = axs[2,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
#         axs[2,0].set_title("Mean Over Dendrites")
#         axs[2,0].set_ylabel("Trials")
#         axs[2,0].set_xlabel("Position Bins")
#         fig.colorbar(im4, ax=axs[2,0])

#         num_plateaus_per_trial_list_across_dends = []
#         for dend in range(len(plateau_start_times_list_mega_list)):
#             num_plateaus_per_trial = []
#             dend_plateaus = plateau_start_times_list_mega_list[dend]
#             for trial in range(len(dend_plateaus)):
#                 num_plateaus_per_trial.append(len(dend_plateaus[trial]))

#             num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

#         num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
#         mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
#         sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

#         axs[4,2].set_title("Mean # Plateaus Per Trial Across Dendrites")
#         axs[4,2].set_ylabel("Mean # Plateaus Across Dendrites")
#         axs[4,2].set_xlabel("Session Length (%)")
#         axs[4,2].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], 
#                                 labels=["0%", '25', "50%", '75', "100%"])
#         axs[4,2].plot(mean_plat_per_trial, color='k')
#         axs[4,2].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        
#         means = np.nanmean(mean_dend_activity, axis=1)
        
#         axs[2,1].plot(means)
#         axs[2,1].set_title("Mean of Dendrites and Trials")
#         axs[2,1].set_xlabel("Position Bins")

#         axs[3,1].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
#         axs[3,1].set_title(f"Plateau Time Across All Dendrites")
#         axs[3,1].set_ylabel("Time (ms)")
#         axs[3,1].set_xlabel("Position Bins")

#         # axs[3,1].bar(range(len(plateau_start_positions_counter)), plateau_start_positions_counter)
#         # axs[3,1].set_title(f"Plateau Count Across All Dendrites")
#         # axs[3,1].set_ylabel("Plateau Count")
#         # axs[3,1].set_xlabel("Position Bins")


#         # n_bins = 10
#         # bin_size = int(50 / n_bins)

#         # summed_plateaus = np.zeros(n_bins)

#         # for i in range(n_bins):
#         #     start = i * bin_size
#         #     end = (i + 1) * bin_size
#         #     summed_data = np.sum(plateau_positions_counter[start:end])
#         #     summed_plateaus[i] = summed_data

#         # axs[4,3].bar(range(len(summed_plateaus)), summed_plateaus)
#         # axs[4,3].set_xlabel("Position Bin Quintile")
#         # axs[4,3].set_ylabel("Time (ms)")
#         # axs[4,3].set_title("Plateau Time per Track Section")
#         # axs[4,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

#         # y_time_ms = plateau_counts_per_time * (dt_constant * 1000.0)  # ms
#         # x_time_ms = np.arange(len(plateau_counts_per_time)) * (dt_constant * 1000.0)
#         # n_trials = plateau_array.shape[0]
#         # y_mean_time_ms = (plateau_counts_per_time / n_trials) * (dt_constant * 1000.0)
#         # axs[3,0].bar(x_time_ms, y_mean_time_ms)
#         # axs[3,0].set_ylabel("Mean plateau time per trial (ms)")
#         # axs[3,0].set_xlabel("Time (ms)")


#         n_bins = 10
#         bin_size = int(50 / n_bins)

#         summed_plateaus = np.zeros(n_bins)

#         for i in range(n_bins):
#             start = i * bin_size
#             end = (i + 1) * bin_size
#             summed_data = np.sum(plateau_start_positions_counter[start:end])
#             summed_plateaus[i] = summed_data

#         axs[3,2].bar(range(len(summed_plateaus)), summed_plateaus)
#         axs[3,2].set_xlabel("Position Bin")
#         axs[3,2].set_ylabel("Plateau Count")
#         axs[3,2].set_title("Plateau Onset Count per Track Section")
#         # axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])
#         axs[3,2].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

#         total_plateaus = np.sum(summed_plateaus)
#         fraction_plateaus = summed_plateaus / total_plateaus
#         axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
#         axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
#         axs[3,3].set_xlabel("Grouped Position Bins")
#         axs[3,3].set_ylabel("% of Total Plateaus")
#         axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
#         # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])

#         cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
#         mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
#         sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)

#         axs[4,3].set_title("Cumulative # Plateaus Across Trials")
#         axs[4,3].set_ylabel("# Plateaus Mean +- SEM Across Dends")
#         axs[4,3].set_xlabel("Session Length (%)")
#         axs[4,3].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
#                              labels=["0", '25', "50", '75', "100%"])
#         axs[4,3].plot(mean_plateaus_all_dends, color='k')
#         axs[4,3].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
            

#         # variance_array = np.var(dend_activity, axis=(1, 2))  # Per dendrite variance

#         # axs[5,2].hist(variance_array, bins=50)
#         # axs[5,2].set_title(f"Variance in Activity {dist}")
#         # axs[5,2].set_xlabel("Variance")
#         # axs[5,2].set_ylabel("Number of Dendrites")


      

#         number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
#         axs[4,1].plot(number_dendrites_counter)
#         axs[4,1].set_title("Percent of Dendrites with Plateau at Location")
#         axs[4,1].set_ylabel("Percent")
#         axs[4,1].set_xlabel("Position Bin")

#         axs[4,0].hist(num_plateaus_per_dend_list, bins=10, edgecolor='k')
#         axs[4,0].set_xlabel("# of Plateaus")
#         axs[4,0].set_ylabel("# of Cells")



#         plt.tight_layout()
#         plt.show()



#     else:
        

#         fig, axs = plt.subplots(4,4, figsize=(25,20))

#         if animal_by_animal:
#             fig.suptitle(f"Animal: {animal}")

#         print(f"activity_EC.shape {activity_EC.shape}")

#         D, N, T = activity_EC.shape            # (dendrites, trials, timebins)
#         dt = 0.001                             # seconds per time bin
#         t_ms = np.arange(T) * dt * 1000.0      # time (ms), length T

#         # trial-averaged time series per dendrite (ignore NaN padding)
#         activity_EC_trial_av = np.nanmean(activity_EC, axis=1)   # (D, T)

#         for i in range(D):
#             y = np.ma.masked_invalid(activity_EC_trial_av[i])    # mask any NaNs in ragged tail
#             axs[0, 0].plot(t_ms, y, alpha=0.2)

#         axs[0, 0].set_title("EC Input To Each Dendrite")
#         axs[0, 0].set_ylabel("Summed Z-Scored Activity")
#         axs[0, 0].set_xlabel("time (ms)")
#         if not animal_by_animal:
#             axs[0, 0].set_xlim(0, 6000)            # show full 0..6000 ms even if data ends earlier

#         # activity_EC_trial_av = np.mean(activity_EC, axis=1)
#         # # mean_activity_EC_trial_av = np.nanmean(activity_EC_trial_av, axis=0)
#         # for i in range(activity_EC_trial_av.shape[0]):
#         #     axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
#         # axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
#         # axs[0,0].set_title("EC Input To Each Dendrite")
#         # axs[0,0].set_ylabel("Summed Z-Scored Activity")
#         # axs[0,0].set_xlabel("Position Bins")

#         axs[0,3].plot(an_velocity, color='r')
#         axs[0,3].set_xlabel("Position Bins")
#         axs[0,3].set_ylabel("Meters / Second")
#         axs[0,3].set_title("Velocity")

#         an_velocity_cm = an_velocity*100
#         distance = 3.6

#         axs[1,3].plot(distance/an_velocity_cm, color='r')
#         axs[1,3].set_xlabel("Position Bins")
#         axs[1,3].set_ylabel("Seconds")
#         axs[1,3].set_title("Occupancy")

#         distance=3.6
#         velocity_cm_s = an_velocity*100

#         # axs[0,2].plot(distance / velocity_cm_s, color='purple')
#         # axs[0,2].set_xlabel("Position Bins")
#         # axs[0,2].set_ylabel("Seconds")
#         # axs[0,2].set_title("Occupancy")

#         im1 = axs[0,1].imshow(weights_EC, aspect='auto')
#         axs[0,1].set_title(f"EC Weights: {dist} Distribution")
#         axs[0,1].set_ylabel("Dendrites")
#         axs[0,1].set_xlabel("Input Cells") 
#         fig.colorbar(im1, ax=axs[0,1])



#         mean_pad = np.mean(activity_EC, axis=0)
#         mean_pad = mean_pad.T

#         im4 = axs[2,0].imshow(mean_pad, aspect='auto', interpolation='None')
#         axs[2,0].set_title("Mean Over Dendrites")
#         axs[2,0].set_ylabel("Trials")
#         axs[2,0].set_xlabel("Time (ms)")
#         fig.colorbar(im4, ax=axs[2,0])

#         means = np.mean(mean_pad, axis=0)
#         # means = means / np.max(means)
#         axs[2,1].set_ylabel("Summed Z-Scored Activity")
#         axs[2,1].plot(means)
#         axs[2,1].set_title("Mean of Dendrites and Trials")
#         axs[2,1].set_xlabel("Time (ms)")

#         ims = axs[1,2].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray', interpolation='none')
#         axs[1,2].set_title(f"Example Dendrite Plateaus \n Dendrite Threshold={dend_threshold}")
#         axs[1,2].set_xlabel("Time (ms)")
#         axs[1,2].set_ylabel("Trials")
#         fig.colorbar(ims, ax=axs[1,2])

#         axs[0,2].hist(weights_EC.flatten(), bins=50)
#         axs[0,2].set_title(f"EC Weights: {dist} Distribution")
#         axs[0,2].set_ylabel("Count")
#         axs[0,2].set_xlabel("Weight")

#         mean_dend_activity = np.nanmean(dend_activity, axis=0).T

#         im4 = axs[1,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
#         axs[1,0].set_title("Mean Over Dendrites")
#         axs[1,0].set_ylabel("Trials")
#         axs[1,0].set_xlabel("Position Bins")
#         if not animal_by_animal:
#             axs[1,0].set_xlim(0, 6000)
#         fig.colorbar(im4, ax=axs[1,0])

#         num_plateaus_per_trial_list_across_dends = []
#         for dend in range(len(plateau_start_times_list_mega_list)):
#             num_plateaus_per_trial = []
#             dend_plateaus = plateau_start_times_list_mega_list[dend]
#             for trial in range(len(dend_plateaus)):
#                 num_plateaus_per_trial.append(len(dend_plateaus[trial]))

#             num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)

#         num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
#         mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
#         sem_plat_per_trial = sem(num_plateaus_per_trial_array, axis=0)

#         axs[3,1].set_title("Mean # Plateaus Per Trial Across Dendrites")
#         axs[3,1].set_ylabel("Mean # Plateaus Across Dendrites")
#         axs[3,1].set_xlabel("Session Length (%)")
#         axs[3,1].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], labels=["0", '25', "50", '75', "100"])
#         axs[3,1].plot(mean_plat_per_trial, color='k')
#         axs[3,1].fill_between(range(len(mean_plat_per_trial)), mean_plat_per_trial - sem_plat_per_trial, mean_plat_per_trial + sem_plat_per_trial, alpha=0.2, color='k')
        

#         # padded_warped_activity_array, key_list, max_len = pad_stack_same_trials_ragged(padded_warped_activity_dict, n_trials_expected=58)

#         for i in range(mean_pad.shape[1]):
#             if not animal_by_animal:
#                 axs[1,1].plot(t_ms, mean_pad[:,i])
#             else:
#                 axs[1,1].plot(mean_pad[:,i])
#         axs[1,1].set_ylabel("Summed Z-Scored Activity")
#         axs[1,1].set_title("Mean of Dendrites and Trials")
#         axs[1,1].set_xlabel("Time (ms)")
#         if not animal_by_animal:
#             axs[1,1].set_xlim(0, 6000) 

#         axs[2,2].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
#         axs[2,2].set_title(f"Plateau Time Across All Dendrites")
#         axs[2,2].set_ylabel("Time (ms)")
#         axs[2,2].set_xlabel("Position Bins")


#         n_bins = 10
#         bin_size = int(50 / n_bins)

#         summed_plateaus = np.zeros(n_bins)

#         for i in range(n_bins):
#             start = i * bin_size
#             end = (i + 1) * bin_size
#             summed_data = np.sum(plateau_start_positions_counter[start:end])
#             summed_plateaus[i] = summed_data

#         axs[2,3].bar(range(len(summed_plateaus)), summed_plateaus)
#         axs[2,3].set_xlabel("Position Bin")
#         axs[2,3].set_ylabel("Plateau Count")
#         axs[2,3].set_title("Plateau Onset Count per Track Section")
#         axs[2,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
#         # axs[2,3].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])


#         cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
#         mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
#         sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)


#         axs[3,0].set_title("Cumulative # Plateaus Across Trials")
#         axs[3,0].set_ylabel("# Plateaus Mean +- SEM Across Dends")
#         axs[3,0].set_xlabel("Session Length (%)")
#         axs[3,0].set_xticks([0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2, len(mean_plateaus_all_dends) - 1], 
#                              labels=["0", '25', "50", '75', "100"])
#         axs[3,0].plot(mean_plateaus_all_dends, color='k')
#         axs[3,0].fill_between(range(len(mean_plateaus_all_dends)), mean_plateaus_all_dends - sem_plateaus_all_dends, mean_plateaus_all_dends + sem_plateaus_all_dends, alpha=0.2, color='k')
            

#         total_plateaus = np.sum(summed_plateaus)
#         fraction_plateaus = (summed_plateaus / total_plateaus)*100
#         axs[3,3].plot(fraction_plateaus, marker='o', color='k', markersize=7)
#         axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
#         axs[3,3].set_xlabel("Grouped Position Bins")
#         axs[3,3].set_ylabel("% of Total Plateaus")
#         axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
#         # axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])


#         # variance_array = np.var(dend_activity, axis=(1, 2))  # Per dendrite variance

#         # axs[5,2].hist(variance_array, bins=50)
#         # axs[5,2].set_title(f"Variance in Activity {dist}")
#         # axs[5,2].set_xlabel("Variance")
#         # axs[5,2].set_ylabel("Number of Dendrites")


#         # axs[3,0].hist(num_plateaus_per_dend_list, bins=10, edgecolor='k')
#         # axs[3,0].set_xlabel("# of Plateaus")
#         # axs[3,0].set_ylabel("# of Cells")

#         number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0) # shape (50,)
#         axs[3,2].plot(number_dendrites_counter)
#         axs[3,2].set_title("Percent of Dendrites with Plateau at Location")
#         axs[3,2].set_ylabel("Percent")
#         axs[3,2].set_xlabel("Position Bin")

 

#         plt.tight_layout()
#         plt.show()

#         return mean_plateaus_all_dends, fraction_plateaus





def plot_multidendrite_EC_multiple_seeds(weights_EC, weights_SST, weights_NDNF,  dend_vm_per_seed_dict, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity_dict, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=False):
    
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
        

        fig, axs = plt.subplots(4,4, figsize=(25,20))

        if animal_by_animal:
            fig.suptitle(f"Animal: {animal}")

        # activity_EC = np.mean(seed_array, axis=0)

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
        if not animal_by_animal:
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


        seed_list = []

        for seed in dend_vm_per_seed_dict:
            seed_list.append(dend_vm_per_seed_dict[seed])

        seed_array = np.array(seed_list)

        average_seeds_dend = np.mean(seed_array, axis=0)

        mean_pad = np.mean(average_seeds_dend, axis=0)
        sem_pad = sem(average_seeds_dend, axis=0)

        im4 = axs[2,0].imshow(mean_pad, aspect='auto', interpolation='None')
        axs[2,0].set_title("Mean Over Dendrites over Seeds")
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

        # seed_list = []
        # for seed in dend_activity_dict:
        #     seed_list.append(dend_activity_dict[seed])


        # seed_array = np.array(seed_list)
        # activity_EC = np.mean(activity_EC, axis=0)

        # dend_activity = np.mean(seed_list, axis=0)

        mean_EC_activity = np.nanmean(activity_EC, axis=0).T

        im4 = axs[1,0].imshow(mean_EC_activity.T, aspect='auto', interpolation=None)
        axs[1,0].set_title("Mean Input Activity")
        axs[1,0].set_ylabel("Trials")
        axs[1,0].set_xlabel("Position Bins")
        if not animal_by_animal:
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


        axs[1,1].plot(mean_pad)
        axs[1,1].plot(range(len(mean_pad)), mean_pad, mean_pad-sem_pad, mean_pad+sem_pad, alpha=0.2)
        axs[1,1].set_ylabel("Summed Z-Scored Activity")
        axs[1,1].set_title("Mean of Dendrites and Trials")
        axs[1,1].set_xlabel("Time (ms)")
        if not animal_by_animal:
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
        fraction_plateaus = (summed_plateaus / total_plateaus)*100
        axs[3,3].plot(fraction_plateaus, marker='o', color='k', markersize=7)
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

        return mean_plateaus_all_dends, fraction_plateaus, plateau_start_positions_counter


def plot_multidendrite_EC_err_across_seeds(tau_ms,
    seeds, last_EPSP, weights_EC, weights_SST, weights_NDNF,  dend_vm_per_seed_dict,
    activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt,
    padded_warped_activity_list, an_velocity, dend_activity_dict, dend_threshold,
    _pos_cnt_dict, start_pos_cnt50_dict, _plateau_arr_list_dict, _mask_dict,  _starts_list_dict,
    dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition=None,
    NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=False
):
    fig, axs = plt.subplots(4,4, figsize=(15,10))

    # ↓↓↓ font sizes: title -2pt, labels -2pt

    def _pt(v):  # convert named sizes like 'large' to points
        return FontProperties(size=v).get_size_in_points()
    title_fs = max(1, _pt(mpl.rcParams['axes.titlesize']) - 4)
    label_fs = max(1, _pt(mpl.rcParams['axes.labelsize']) - 5)

    if animal_by_animal:
        fig.suptitle(f"Animal: {animal} Seeds: {seeds} Tau (ms): {tau_ms:.3f}", fontsize=title_fs)

    D, N, T = activity_EC.shape            # (dendrites, trials, timebins)
    dt = 0.001                             # seconds per time bin
    t_ms = np.arange(T) * dt * 1000.0      # time (ms), length T

    activity_EC_trial_av = np.nanmean(activity_EC, axis=1)   # (D, T)
    for i in range(D):
        y = np.ma.masked_invalid(activity_EC_trial_av[i])
        axs[0, 0].plot(t_ms, y, alpha=0.2)

    axs[0, 0].set_title("EC Input To Each Dendrite", fontsize=title_fs)
    axs[0, 0].set_ylabel("Summed Z-Scored Activity", fontsize=label_fs)
    axs[0, 0].set_xlabel("Time (ms)", fontsize=label_fs)
    if not animal_by_animal:
        axs[0, 0].set_xlim(0, 6000)

    axs[0,3].plot(an_velocity, color='r')
    axs[0,3].set_xlabel("Position Bins", fontsize=label_fs)
    axs[0,3].set_ylabel("Meters / Second", fontsize=label_fs)
    axs[0,3].set_title("Velocity", fontsize=title_fs)

    an_velocity_cm = an_velocity*100
    distance = 3.6

    axs[1,3].plot(distance/an_velocity_cm, color='r')
    axs[1,3].set_xlabel("Position Bins", fontsize=label_fs)
    axs[1,3].set_ylabel("Seconds", fontsize=label_fs)
    axs[1,3].set_title("Occupancy", fontsize=title_fs)

    distance=3.6
    velocity_cm_s = an_velocity*100

    axs[0,1].plot(last_EPSP[0,0,:1000])
    axs[0,1].set_xlabel("Time (ms)", fontsize=label_fs)
    axs[0,1].set_ylabel("EPSP Amplitude (mV)", fontsize=label_fs)
    axs[0,1].set_title("EPSP Example Train", fontsize=title_fs)

    # im1 = axs[0,1].imshow(weights_EC, aspect='auto')
    # axs[0,1].set_title(f"EC Weights: {dist} Distribution", fontsize=title_fs)
    # axs[0,1].set_ylabel("Dendrites", fontsize=label_fs)
    # axs[0,1].set_xlabel("Input Cells", fontsize=label_fs)
    # fig.colorbar(im1, ax=axs[0,1])

    seed_list = []
    for seed in dend_vm_per_seed_dict:
        seed_list.append(np.mean(dend_vm_per_seed_dict[seed], axis=0))
    seed_array = np.array(seed_list)
    average_seeds_dend = np.mean(seed_array, axis=0)
    print(f"average_seeds_dend.shape {average_seeds_dend.shape}")
    mean_pad = np.nanmean(average_seeds_dend, axis=0)
    print(f"mean_pad.shape {mean_pad.shape}")
    sem_pad = sem(average_seeds_dend, axis=0, nan_policy="omit")

    im4 = axs[2,0].imshow(average_seeds_dend, aspect='auto', interpolation='None')
    axs[2,0].set_title("Mean Over Dendrites over Seeds", fontsize=title_fs)
    axs[2,0].set_ylabel("Trials", fontsize=label_fs)
    axs[2,0].set_xlabel("Time (ms)", fontsize=label_fs)
    fig.colorbar(im4, ax=axs[2,0], label="mV")

    axs[2,1].set_ylabel("mV", fontsize=label_fs)
    axs[2,1].plot(mean_pad)
    axs[2,1].fill_between(range(len(mean_pad)), mean_pad-sem_pad, mean_pad+sem_pad, alpha=0.2)
    axs[2,1].set_title("Mean of Dendrites and Trials", fontsize=title_fs)
    axs[2,1].set_xlabel("Time (ms)", fontsize=label_fs)

    seed = seeds[0]
    ims = axs[1,2].imshow(_plateau_arr_list_dict[seed][example_cell], aspect='auto', cmap='gray', interpolation='none')
    axs[1,2].set_title(f"Seed#{seed} Dendrite#{example_cell} Plateaus \n Dendrite Threshold={dend_threshold}", fontsize=title_fs)
    axs[1,2].set_xlabel("Time (ms)", fontsize=label_fs)
    axs[1,2].set_ylabel("Trials", fontsize=label_fs)
    fig.colorbar(ims, ax=axs[1,2])

    axs[0,2].hist(weights_EC.flatten(), bins=50)
    axs[0,2].set_title(f"EC Weights: {dist} Distribution", fontsize=title_fs)
    axs[0,2].set_ylabel("Count", fontsize=label_fs)
    axs[0,2].set_xlabel("Weight", fontsize=label_fs)

    mean_EC_activity = np.nanmean(activity_EC, axis=0).T
    im4 = axs[1,0].imshow(mean_EC_activity.T, aspect='auto', interpolation=None)
    axs[1,0].set_title("Mean Input Activity", fontsize=title_fs)
    axs[1,0].set_ylabel("Trials", fontsize=label_fs)
    axs[1,0].set_xlabel("Time (ms)", fontsize=label_fs)
    if not animal_by_animal:
        axs[1,0].set_xlim(0, 6000)
    cb = fig.colorbar(im4, ax=axs[1,0], label="Summed Z-Scored Activity")
    cb.set_label("Summed Z-Scored Activity", fontsize=label_fs)

    print(f"activity_EC.shape {activity_EC.shape}")

    mean_EC_activity = np.nanmean(activity_EC, axis=0)
    mean_mean = np.nanmean(mean_EC_activity, axis=0)
    sem_mean = sem(mean_EC_activity, axis=0, nan_policy='omit')
    axs[1,1].plot(mean_mean)
    axs[1,1].fill_between(range(len(mean_mean)), mean_mean-sem_mean, mean_mean+sem_mean, alpha=0.2)
    axs[1,1].set_title("Trial Averaged Input Activity", fontsize=title_fs)
    axs[1,1].set_xlabel("Trial Averaged Input Activity", fontsize=title_fs)
    axs[1,1].set_ylabel("Summed Z-Scored Activity", fontsize=label_fs)

    per_seed_list = []
    for seed in _starts_list_dict:
        num_plateaus_per_trial_list_across_dends = []
        for dend in range(len(_starts_list_dict[seed])):
            num_plateaus_per_trial = []
            dend_plateaus = _starts_list_dict[seed][dend]
            for trial in range(len(dend_plateaus)):
                num_plateaus_per_trial.append(len(dend_plateaus[trial]))
            num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)
        num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
        mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
        per_seed_list.append(mean_plat_per_trial)

    per_seed_array = np.array(per_seed_list)
    mean_plat_over_seeds = np.mean(per_seed_array, axis=0)
    sem_plat_over_seeds = sem(per_seed_array, axis=0)
    
    axs[3,1].set_title("Mean # Plateaus Per Trial Across Dendrites", fontsize=title_fs)
    axs[3,1].set_ylabel("Mean Plateaus +- SEM Seeds", fontsize=label_fs)
    axs[3,1].set_xlabel("Session Length (%)", fontsize=label_fs)
    axs[3,1].set_xticks([0, len(mean_plat_over_seeds)//4, len(mean_plat_over_seeds)//2,
         len(mean_plat_over_seeds)//4 + len(mean_plat_over_seeds)//2,
         len(mean_plat_over_seeds) - 1],
        labels=["0", '25', "50", '75', "100"])
    axs[3,1].errorbar(range(len(mean_plat_over_seeds)), mean_plat_over_seeds, yerr=sem_plat_over_seeds, color='k', marker='o', markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    


    _pos_cnt_dict_list = []
    for seed in _pos_cnt_dict:
        print(f"_pos_cnt_dict[seed].shape {_pos_cnt_dict[seed].shape}")
        _pos_cnt_dict_list.append(_pos_cnt_dict[seed])
    _pos_cnt_dict_array = np.array(_pos_cnt_dict_list)

    mean_pos_cnt_dict_array = np.mean(_pos_cnt_dict_array, axis=0)
    sem_pos_cnt_dict_array = sem(_pos_cnt_dict_array, axis=0)

    err_kw = dict(ecolor='k', elinewidth=0.8, capsize=1, capthick=0.8)
    axs[2,2].bar(range(len(mean_pos_cnt_dict_array)), mean_pos_cnt_dict_array, yerr=sem_pos_cnt_dict_array, capsize=4, error_kw=err_kw)
    axs[2,2].set_title(f"Plateau Time Across All Dendrites", fontsize=title_fs)
    axs[2,2].set_ylabel("Time (ms) +- SEM Across Seeds", fontsize=label_fs)
    axs[2,2].set_xlabel("Position Bins", fontsize=label_fs)

    summed_plateaus_over_seeds = []
    for seed in start_pos_cnt50_dict:
        start_pos_cnt50_list = start_pos_cnt50_dict[seed]
        n_bins = 10
        bin_size = int(50 / n_bins)
        summed_plateaus = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(start_pos_cnt50_list[start:end])
            summed_plateaus[i] = summed_data
        summed_plateaus_over_seeds.append(summed_plateaus)

    summed_plateaus_over_seeds_array = np.array(summed_plateaus_over_seeds)
    mean_summed_plateaus_over_seeds_array = np.mean(summed_plateaus_over_seeds_array, axis=0)
    sem_summed_plateaus_over_seeds_array = sem(summed_plateaus_over_seeds_array, axis=0)

    err_kw = dict(ecolor='k', elinewidth=0.8, capsize=2, capthick=0.8)
    axs[2,3].bar(range(len(mean_summed_plateaus_over_seeds_array)), mean_summed_plateaus_over_seeds_array, yerr=sem_summed_plateaus_over_seeds_array, error_kw=err_kw)
    axs[2,3].set_xlabel("Position Bin", fontsize=label_fs)
    axs[2,3].set_ylabel("Plateau Count +- SEM over Seeds", fontsize=label_fs)
    axs[2,3].set_title("Plateau Onset Count per Track Section", fontsize=title_fs)
    axs[2,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

    cumsum_plateaus_all_dends = np.cumsum(num_plateaus_per_trial_array, axis=1)
    mean_plateaus_all_dends = np.mean(cumsum_plateaus_all_dends, axis=0)
    sem_plateaus_all_dends = sem(cumsum_plateaus_all_dends, axis=0)

    axs[3,0].set_title("Cumulative # Plateaus Across Trials", fontsize=title_fs)
    axs[3,0].set_ylabel("# Plateaus Mean +- SEM Across Dends", fontsize=label_fs)
    axs[3,0].set_xlabel("Session Length (%)", fontsize=label_fs)
    axs[3,0].set_xticks(
        [0, len(mean_plateaus_all_dends)//4, len(mean_plateaus_all_dends)//2,
         len(mean_plateaus_all_dends)//4 + len(mean_plateaus_all_dends)//2,
         len(mean_plateaus_all_dends) - 1],
        labels=["0", '25', "50", '75', "100"])
    
    axs[3,0].plot(mean_plateaus_all_dends, color='k')
    axs[3,0].fill_between(
        range(len(mean_plateaus_all_dends)),
        mean_plateaus_all_dends - sem_plateaus_all_dends,
        mean_plateaus_all_dends + sem_plateaus_all_dends,
        alpha=0.2, color='k')
    
    print(f"summed_plateaus_over_seeds_array {summed_plateaus_over_seeds_array.shape}")


    # fraction_plateaus_list = []
    # for i in range(summed_plateaus_over_seeds_array.shape[0]):
    #     summed_plateaus = summed_plateaus_over_seeds_array[i, :]
    #     total_plateaus = np.sum(summed_plateaus)
    #     print(f"total_plateaus {total_plateaus}")
    #     fraction_plateaus = (summed_plateaus / total_plateaus)*100
    #     fraction_plateaus_list.append(fraction_plateaus)

    # fraction_plateaus_array = np.array(fraction_plateaus_list)
    # mean_fraction_plateaus_array = np.mean(fraction_plateaus_array, axis=0)
    # sem_fraction_plateaus_array = sem(fraction_plateaus_array, axis=0, nan_policy='omit')

    arr = np.asarray(summed_plateaus_over_seeds)  # expect (n_seeds, n_bins)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")

    # If bins are on axis 0, transpose to (n_seeds, n_bins)
    if arr.shape[0] in (10, 50) and arr.shape[1] not in (10, 50):
        arr = arr.T

    # Compute per-seed totals and normalize safely
    totals = arr.sum(axis=1)  # (n_seeds,)
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = arr / totals[:, None] * 100.0  # (n_seeds, n_bins)
        frac[~np.isfinite(frac)] = np.nan     # seeds with zero total → NaN

    # Mean and SEM across seeds, bin-wise
    mean_fraction_plateaus_array = np.nanmean(frac, axis=0)
    n_eff = np.sum(~np.isnan(frac), axis=0).clip(min=1)
    sem_fraction_plateaus_array = np.nanstd(frac, axis=0, ddof=1) / np.sqrt(n_eff)

    # Plot with visible errorbar caps
    x = np.arange(mean_fraction_plateaus_array.size)
    axs[3,3].errorbar(
        x, mean_fraction_plateaus_array,
        yerr=sem_fraction_plateaus_array,
        fmt='o-', color='k', ms=2,
        elinewidth=0.8, capsize=2, capthick=0.8
    )
    axs[3,3].set_title("% of Plateaus in Grouped Position Bin", fontsize=title_fs)
    axs[3,3].set_xlabel("Grouped Position Bins", fontsize=label_fs)
    axs[3,3].set_ylabel("% of Total Plateaus", fontsize=label_fs)
    axs[3,3].set_xticks(np.arange(n_bins), ["1-5","6-10","11-15","16-20","21-25","26-30","31-35","36-40","41-45","46-50"], fontsize=7)

    # Optional quick sanity prints:
    # print("arr shape:", arr.shape)
    # print("totals per seed:", totals)
    # print("any zero totals?", np.any(totals == 0))




    # axs[3,3].errorbar(range(len(mean_fraction_plateaus_array)), mean_fraction_plateaus_array, yerr=sem_fraction_plateaus_array, marker='o', color='k', markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    # axs[3,3].set_title("% of Plateaus in Grouped Position Bin", fontsize=title_fs)
    # axs[3,3].set_xlabel("Grouped Position Bins", fontsize=label_fs)
    # axs[3,3].set_ylabel("% of Total Plateaus", fontsize=label_fs)
    # axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

    num_dendrites_across_seeds =[]
    for seed in _mask_dict:
        dendrite_plateau_mask = _mask_dict[seed]
        number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0)
        num_dendrites_across_seeds.append(number_dendrites_counter)

    num_dendrites_across_seeds_array = np.array(num_dendrites_across_seeds)
    mean_num_dendrites_across_seeds = np.mean(num_dendrites_across_seeds_array, axis=0)
    sem_num_dendrites_across_seeds = np.mean(num_dendrites_across_seeds_array, axis=0)

    axs[3,2].plot(mean_num_dendrites_across_seeds)
    axs[3,2].errorbar(range(len(mean_num_dendrites_across_seeds)), mean_num_dendrites_across_seeds, yerr=sem_num_dendrites_across_seeds, color='k', marker='o', markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    axs[3,2].set_title("Percent of Dendrites with Plateau at Location", fontsize=title_fs)
    axs[3,2].set_ylabel("Percent +- SEM Across Seeds", fontsize=label_fs)
    axs[3,2].set_xlabel("Position Bin", fontsize=label_fs)

    plt.tight_layout()
    plt.show()

    return mean_plateaus_all_dends


