# (ca1_env) michaelfinch@nbp-25-228-133 Clean_notebooks_to_date % python build_a_model_object.py simulate \
#   -s 0 -s 1 \
#   -o /Users/michaelfinch/CA1-interneuron-GLM/tmp/spike_sim.pkl \
#   --dend-threshold -70 \
#   --vel-applied real \ -o ./spike_sim.pkl \                                                                                   
#   --include-inhibition


from __future__ import annotations  # defer evaluation of type hints

# stdlib / third-party
import os, time, gc, json, pickle, random, yaml
from dataclasses import dataclass, replace, asdict
from typing import Any, Dict, List, Tuple, Optional, Iterable, Literal

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import click

# your project imports (AFTER third-party)
from Fixing_dend_models_presentation import *
from spiking_model_utils import load_data_regular




def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
    # time_series = []
    # for _ in range(count+1):
    #     initial_value += random.gauss(0, 1) * volatility
    #     time_series.append(initial_value)
    # return time_series
    time_series = []
    for _ in range(count+1):
        time_series.append(initial_value + random.gauss(0, 1) * volatility)
    return time_series

def _sanitize_velocity_cm_s(v_in_m_per_s, min_vel_cm_s=1e-3 * 100):
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

def exp_kernel(tau_ms, dt_ms, n_taus=5, norm="peak", target=1.0):
            # simple single-exponential fallback
            L = int(np.ceil(10.0 * tau_ms / max(dt_ms, 1e-6)))
            t = np.arange(max(L, 1), dtype=np.float32) * np.float32(dt_ms)
            k = np.exp(-t / np.float32(tau_ms)).astype(np.float32, copy=False)
            if norm == "peak":
                m = float(k.max()) if k.size else 1.0
                k = (np.float32(target) * k) / np.float32(max(m, 1e-12))
            elif norm == "area":
                area = float(k.sum()) * (dt_ms / 1000.0)
                k = (np.float32(target) * k) / np.float32(max(area, 1e-12))
            return k

# def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None, rng=None):
#     """
#     Given a time series of instantaneous spike rates in Hz, produce a spike train
#     consistent with an inhomogeneous Poisson process with an absolute refractory period.
#     - rate: Hz at times t (ms)
#     - t:    ms (monotonic)
#     - dt:   ms grid for interpolation & returned spike times
#     - refractory: ms absolute deadtime
#     """
#     # Choose RNG (backward compatible with your original signature)
#     if generator is None and rng is None:
#         rng = np.random.default_rng()
#     if rng is None:
#         # wrap a Python random.Random as a minimal interface
#         class _PyRandWrapper:
#             def random(self, size=None):
#                 if size is None:
#                     return generator.random()
#                 # vectorized draw
#                 return np.array([generator.random() for _ in range(size)], dtype=float)
#             def exponential(self, scale, size=None):
#                 # inverse-transform using .random()
#                 u = self.random(size=size)
#                 return -np.log(u) * scale
#         rng = _PyRandWrapper()

#     # Interpolate onto uniform grid (ms). Same as your original.
#     t0 = float(t[0])
#     t_end = float(t[-1])
#     interp_t = np.arange(t0, t_end + dt, dt, dtype=float)
#     interp_rate = np.interp(interp_t, t, rate).astype(float, copy=False)

#     # Convert Hz -> kHz (per ms), then apply refractory correction (identical math)
#     interp_rate /= 1000.0
#     mask = interp_rate > 0.0
#     interp_rate[mask] = 1.0 / (1.0 / interp_rate[mask] - refractory)

#     max_rate = float(np.max(interp_rate))
#     if not np.isfinite(max_rate) or max_rate <= 0.0:
#         return np.empty(0, dtype=float)

#     # Ogata thinning: jump in time with Exp(max_rate), accept with r(t)/max_rate,
#     # and enforce absolute refractory in ms.
#     # We work in continuous time then snap to the dt-grid indices.
#     spikes = []
#     t_curr = t0
#     last_spike_t = -math.inf  # enforce refractory in ms
#     # Draw in batches to minimize Python overhead
#     batch = 1024

#     while t_curr < t_end:
#         # Draw a batch of proposed ISIs (ms)
#         isis = rng.exponential(scale=1.0/max_rate, size=batch)
#         # Cumulative jump times for the whole batch
#         cum = np.cumsum(isis)
#         # Turn into absolute proposal times (ms)
#         props = t_curr + cum

#         # Stop at first proposal beyond t_end; process those within
#         valid_n = int(np.searchsorted(props, t_end, side='right'))
#         if valid_n == 0:
#             # advance time and continue
#             t_curr = props[-1]
#             continue

#         # Convert proposal times to nearest grid index (floor)
#         idx = np.floor((props[:valid_n] - t0) / dt).astype(int)
#         idx = np.clip(idx, 0, interp_t.size - 1)
#         accept_prob = interp_rate[idx] / max_rate

#         # Draw uniforms for accept/reject
#         u = rng.random(size=valid_n)
#         accepted = u <= accept_prob

#         # Apply refractory: keep only those with (props - last_spike_t) >= refractory
#         if accepted.any():
#             for k in np.nonzero(accepted)[0]:
#                 tp = props[k]
#                 if tp - last_spike_t >= refractory:
#                     spikes.append(tp)
#                     last_spike_t = tp

#         # Advance base time by the last ISI in the batch
#         t_curr = props[valid_n-1] if valid_n > 0 else props[-1]

#     return np.asarray(spikes, dtype=float)

def epsps_event_add(spike_idx, T, kernel):
    """
    Exact causal conv with `kernel` using event-driven accumulation.
    spike_idx: 1D int array of spike times (samples)
    T: length of output trace
    kernel: 1D float array (causal; length K)
    returns: 1D float array of length T
    """
    out = np.zeros(T, dtype=np.float32)
    K = kernel.shape[0]
    for s in spike_idx:
        if 0 <= s < T:
            end = min(T, s + K)
            out[s:end] += kernel[:(end - s)]
    return out

def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
    # time_series = []
    # for _ in range(count+1):
    #     initial_value += random.gauss(0, 1) * volatility
    #     time_series.append(initial_value)
    # return time_series
    time_series = []
    for _ in range(count+1):
        time_series.append(initial_value + random.gauss(0, 1) * volatility)
    return time_series

def plot_multidendrite_EC_err_across_seeds(tau_ms,
    seeds, last_EPSP, weights_EC, weights_SST, weights_NDNF,  dend_vm_per_seed_dict,
    activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt,
    padded_warped_activity_list, an_velocity, dend_threshold,
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

def get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=None, constant_vel=None, use_residuals=True, use_model_EC=False, multiple_dendrites=False, add_inh=None, seed=0, SST_bias_factor=None, dist=None, use_averaged_velocity=None, make_it_spike=False, store_intermediates=False):
    
    SEED = seed
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    n_EC = 792
    n_SST = 75
    n_NDNF = 115
    n_dendrites=100


    dt_constant=0.001

    tau_ms  = 5.0
    dt_ms   = dt_constant * 1000.0      # 1 ms
    AMP     = 1.0                      # mV
    MODE    = "peak"                    # "area" or "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)


    if use_model_EC:

        if make_it_spike:
            
            n_EC = 792
            pos_bins = 50
            n_trials = 58

            animal_velocity_list_cm_sec = get_vel_formatted(activity_dict_EC, factors_dict_EC)
            data_normalized_list, r2_list = get_correlation(activity_dict_EC, factors_dict_EC)
            overall_mean = np.mean(data_normalized_list)
            synthetic_data_plus_vel_list, synthetic_activity = get_synthetic_data(n_EC, pos_bins, n_trials, overall_mean, animal_velocity_list_cm_sec, r2_list)

            animal_velocity_array = np.array(animal_velocity_list_cm_sec)
            an_velocity = np.mean(animal_velocity_array, axis=0) / 100


            print(f"len(synthetic_data_plus_vel_list) {len(synthetic_data_plus_vel_list)}")
            print(f"synthetic_data_plus_vel_list[0].shape {synthetic_data_plus_vel_list[0].shape}")

            dend_vm_list, weights_EC, last_EPSP = turn_rates_into_spikes(synthetic_data_plus_vel_list, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng)

            num_trials   = len(dend_vm_list)
            n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


            # dend_vm_list: list of 2D arrays, each (n_dendrites, T_i)

            num_trials = len(dend_vm_list)
            n_dendrites_ = dend_vm_list[0].shape[0]
            max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

            dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

            for i, arr in enumerate(dend_vm_list):
                if arr.shape[0] != n_dendrites_:
                    raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
                Ti = arr.shape[1]               # current trial's time length
                dend_vm_padded[i, :, :Ti] = arr # pad along time axis


            print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

            Vm_list = []
            for dend in range(dend_vm_padded.shape[1]):
                trials_by_time = dend_vm_padded[:,dend,:]
                Vm, _, _ = activity_to_dend_vm_2d(
                trials_by_time,
                Vrest=-70.0,
                vm_scale=0.1,
                center_across="time")
                Vm_list.append(Vm)

            activity_EC = np.array(Vm_list)

            activity_NDNF=0
            activity_SST=0 
            NDNF_sf_opt=0 
            SST_sf_opt=0 
            NDNF_contribution_sum=0
            SST_contribution_sum=0
            weights_SST=0
            weights_NDNF=0

            return an_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP

        
        else:
            random.seed(seed)
            ts_list = []
            for i in range(58):
                ts = random_timeseries(1.0, 1., 49)
                ts_list.append(ts)

            dend_contribution_EC = np.array(ts_list).T
            print(dend_contribution_EC.shape)

            if constant_vel:
                an_velocity = np.full(dend_contribution_EC.shape, 0.43)
            else:
                an_velocity = np.tile(mean_new_average_vel_array[:,None], (1,58))


            # dend_contribution_EC /= dend_contribution_EC

            return dend_contribution_EC, an_velocity 
            

        # rng = np.random.default_rng(seed=40)

        # pos_bins = 150
        # real_pos=50
        # trials = 58
        # dend_contribution_EC = np.zeros((pos_bins, trials))

        # # Fixed Gaussian centers
        # base_centers = np.arange(0, pos_bins+1, 5)  # bin 2, 7, 12, ..., 47
        # width = 2.5
        # for i in range(trials):
        #     for center in base_centers:
        #         # Add jitter to the center (±1 bin)
        #         jittered_center = center + rng.choice([-1, 0, 1])

        #         # Make sure jittered center stays within bounds
        #         jittered_center = np.clip(jittered_center, 0, pos_bins - 1)

        #         amplitude = rng.uniform(0.01, 0.02)  # still jitter amplitude too
        #         gaussian = norm.pdf(np.arange(pos_bins), loc=jittered_center, scale=width)
        #         gaussian /= np.max(gaussian)
        #         dend_contribution_EC[:, i] += amplitude * gaussian

        # final_dend = dend_contribution_EC[real_pos:real_pos*2, :]
        # # Normalize globally if needed
        # final_dend /= np.max(final_dend)
        # if constant_vel:
        #     an_velocity = np.full(final_dend.shape, 0.43)
        # else:
        #     an_velocity = np.tile(mean_new_average_vel_array[:,None], (1,58))
            
        # return final_dend, an_velocity 
    
    else:
        if use_residuals:
            dend_contribution_EC, an_velocity_EC, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_NDNF, an_velocity_NDNF, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_SST, an_velocity_SST, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            
            EC = dend_contribution_EC.copy()
            NDNF = dend_contribution_NDNF.copy()
            SST = dend_contribution_SST.copy()
            
            if add_inh=='neither':
                an_velocity = an_velocity_EC 
                SST_sf_opt = 0
                NDNF_sf_opt = 0
                NDNF_contribution_sum = 0
                SST_contribution_sum = 0

            elif add_inh=='sst':
                SST_sf_opt, info = fit_sst_scale_to_cancel_ec(dend_contribution_EC, dend_contribution_SST)
                NDNF_sf_opt=0
                NDNF_contribution_sum = 0
                SST_contribution_sum = 0

                an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST], axis=0), axis=0)

            else:
                # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
                # NDNF_sf_opt, SST_sf_opt = result.x

                res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
                NDNF_sf_opt = res["ndnf_sf"]
                SST_sf_opt  = res["sst_sf"]
                NDNF_contribution_sum = res["contrib_L2_ndnf"]
                SST_contribution_sum = res["contrib_L2_sst"]

                an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST, an_velocity_NDNF], axis=0), axis=0)


        else:
            dend_contribution_EC, an_velocity_EC, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_NDNF, an_velocity_SST, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_SST, an_velocity_NDNF, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            
            EC = dend_contribution_EC.copy()
            NDNF = dend_contribution_NDNF.copy()
            SST = dend_contribution_SST.copy()

            if add_inh=='neither':
                an_velocity = an_velocity_EC 
                SST_sf_opt = 0
                NDNF_sf_opt = 0
                NDNF_contribution_sum = 0
                SST_contribution_sum = 0

            elif add_inh=='sst':
                SST_sf_opt, info = fit_sst_scale_to_cancel_ec(dend_contribution_EC, dend_contribution_SST)
                NDNF_sf_opt=0
                NDNF_contribution_sum = 0
                SST_contribution_sum = 0
                an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST], axis=0), axis=0)

            else:
                # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
                # NDNF_sf_opt, SST_sf_opt = result.x
                res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
                NDNF_sf_opt = res["ndnf_sf"]
                SST_sf_opt  = res["sst_sf"]
                NDNF_contribution_sum = res["contrib_L2_ndnf"]
                SST_contribution_sum = res["contrib_L2_sst"]

                an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST, an_velocity_NDNF], axis=0), axis=0)


        if multiple_dendrites:

            dend_contribution_EC, an_velocity, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_NDNF, an_velocity, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            dend_contribution_SST, an_velocity, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            


            if make_it_spike:

                try:
                    import psutil
                    _PROC = psutil.Process(os.getpid())
                except Exception:
                    psutil = None
                    _PROC = None


                # def turn_rates_into_spikes(dend_list_EC):


                #     EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
                #     SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
                #     NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

                    

                #     n_pos = EC_input_matrix.shape[1]
                #     n_trials = EC_input_matrix.shape[2]
                #     dx = 180.0 / n_pos  

                #     vel = an_velocity
                #     if vel.shape != (n_pos, n_trials):
                #         raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

                #     L_prev = 500
    
                #     # precomputed_prepend = np.zeros(L_prev)
                #     weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng)
                #     max_length = 0
                #     dend_vm_list = []  

                #     proc_peak_start = _get_peak_rss_bytes()   # peak at loop start

                #     warped_list = []

                #     last_EPSP = None

                #     for t in range(EC_input_matrix.shape[2]):

                #         trial_wall_start = time.perf_counter()
                #         trial_peak_before = _get_peak_rss_bytes()
                #         trial_curr_before = _get_current_rss_bytes()

                #         start_time = time.time()
                #         v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]) 
                #         dt_s   = dx / v_cm_s                             
                #         edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
                #         total_time = float(edges_s[-1])





                #         # constant-time axis in seconds
                #         t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
                #         print(f"len(t_axis) {len(t_axis)}")
                #         max_time_over_cells = 0
                #         T = t_axis.size

                #         firing_example = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
                #         valid = np.isfinite(firing_example)
                #         time_points = np.cumsum(dt_s)
                        
                #         firing_zero = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
                #         warped_zero = np.interp(t_axis, time_points[valid], firing_zero[valid]).astype(np.float32, copy=False)
                #         holder = warped_zero[:L_prev]

                #         warped_rows = [] if store_intermediates else None

                #         rows = np.empty((n_EC, T), dtype=np.float32)   

                #         two_track_length = np.empty(len(holder) + len(warped_zero))

                #         t_ms = np.arange(two_track_length.size, dtype=int) * dt_ms


                #         for cell in range(n_EC):
                #             firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
                #             valid = np.isfinite(firing)
                #             if valid.sum() >= 2:
                #                 time_points = np.cumsum(dt_s)  # length n_pos

                #                 warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                #                 holder = warped[:L_prev]

                #             else:
                #                 warped = np.full(1, np.nan, dtype=np.float32)
                            
                #             if store_intermediates:
                #                 warped_rows.append(warped)

                #             two_track_length[:len(holder)] = holder
                #             two_track_length[-len(warped):] = warped

                #             spike_times = get_inhom_poisson_spike_times_by_thinning_real(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
                            
                #             epsps = epsps_event_add(spike_times, two_track_length.shape[0], kernel).astype(np.float16, copy=False)
                #             epsps = epsps[L_prev:]

                #             last_EPSP = epsps

                            
                #             if len(epsps) > max_time_over_cells:
                #                 max_time_over_cells = len(epsps)
                #                 print(f"trial {t} len(epsps) {len(epsps)}")
                #             rows[cell,:] = epsps


                #         warped_list.append(warped_rows) if store_intermediates else None
                        
                #         X_trial = np.stack(rows, axis=0).astype(np.float16, copy=False)

                        

                #         dendy_list = []

                #         for dend in range(n_dendrites):
                #             w = weights_EC[:, dend]            # shape (n_EC,)
                #             vm_t = w @ X_trial 
                #             dendy_list.append(vm_t)
                            
                #         dend_vm_over_time = np.array(dendy_list, dtype=np.float16)


                #         # dend_vm_over_time = weights_EC @ X_trial

                #         if len(dend_vm_over_time) > max_length:
                #             max_length = len(dend_vm_over_time)

                #         dend_vm_list.append(dend_vm_over_time)

                        
                #         end_time = time.time()

                #         end_time = time.time()
                #         # --- memory + timing report ---
                #         trial_wall_end = time.time()
                #         trial_peak_after = _get_peak_rss_bytes()
                #         trial_curr_after = _get_current_rss_bytes()

                #         trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
                #             else (trial_peak_after - trial_peak_before)
                #         proc_peak_total  = trial_peak_after
                #         proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
                #             else (trial_peak_after - proc_peak_start)

                #         print(
                #             f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
                #             f"RSS now={_fmt_bytes(trial_curr_after)}  "
                #             f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
                #             f"proc peak={_fmt_bytes(proc_peak_total)}  "
                #             f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
                #         )
                #         print(f"total time {end_time-start_time}")

                #         del rows, X_trial, dend_vm_over_time, epsps, 
                
                #     return dend_vm_list

                print(f"len(dend_list_EC) {len(dend_list_EC)}")
                print(f"dend_list_EC[0].shape {dend_list_EC[0].shape}")

                dend_vm_list, weights_EC, last_EPSP = turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng)


                num_trials   = len(dend_vm_list)
                n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


                print(f"num_trials{num_trials} n_dendrites_ {n_dendrites_}")

                # dend_vm_list: list of 2D arrays, each (n_dendrites, T_i)

                num_trials = len(dend_vm_list)
                n_dendrites_ = dend_vm_list[0].shape[0]
                max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

                print(f"num_trials={num_trials}  n_dendrites_={n_dendrites_}  max_T={max_T}")

                # Preallocate (num_trials, n_dendrites, max_T)
                dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

                for i, arr in enumerate(dend_vm_list):
                    if arr.shape[0] != n_dendrites_:
                        raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
                    Ti = arr.shape[1]               # current trial's time length
                    dend_vm_padded[i, :, :Ti] = arr # pad along time axis


                print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

                Vm_list = []
                for dend in range(dend_vm_padded.shape[1]):
                    trials_by_time = dend_vm_padded[:,dend,:]
                    Vm, _, _ = activity_to_dend_vm_2d(
                    trials_by_time,
                    Vrest=-70.0,
                    vm_scale=0.1,
                    center_across="time")
                    Vm_list.append(Vm)

                activity_EC = np.array(Vm_list)

                activity_NDNF=0
                activity_SST=0 
                NDNF_sf_opt=0 
                SST_sf_opt=0 
                NDNF_contribution_sum=0
                SST_contribution_sum=0
                weights_SST=0
                weights_NDNF=0

            else:
                n_EC = 792
                n_SST = 75
                n_NDNF = 115
                n_dendrites=100


                EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
                SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
                NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

                weights_EC = sample_weights(dist, n_dendrites, n_EC, rng=rng)
                weights_SST = sample_weights('Equal', n_dendrites, n_SST, rng=rng)
                weights_NDNF = sample_weights('Equal', n_dendrites, n_NDNF, rng=rng)

                activity_EC = get_dendrite_activity(weights_EC, EC_input_matrix, n_dendrites, n_EC)
                activity_SST = get_dendrite_activity(weights_SST, SST_input_matrix, n_dendrites, n_SST)
                activity_NDNF = get_dendrite_activity(weights_NDNF, NDNF_input_matrix, n_dendrites, n_NDNF)

                EC = activity_EC.copy()
                NDNF = activity_NDNF.copy()
                SST = activity_SST.copy()

                if add_inh=='sst':
                    SST_sf_opt, info = fit_sst_scale_to_cancel_ec(activity_EC, activity_SST)
                    NDNF_sf_opt=0
                else:
                    # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
                    # NDNF_sf_opt, SST_sf_opt = result.x
                    res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
                    NDNF_sf_opt = res["ndnf_sf"]
                    SST_sf_opt  = res["sst_sf"]
                    NDNF_contribution_sum = res["contrib_L2_ndnf"]
                    SST_contribution_sum = res["contrib_L2_sst"]


            return an_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP
        else:
            return dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, last_EPSP

def turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=None, n_dendrites=100, store_intermediates=False, rng=None, debug=False):

    print(f"dt_constant {dt_constant}")

    dt_ms = dt_constant*1000.

    EC_input_matrix = np.stack(dend_list_EC, axis=0)

    print(f"an_velocity.shape {an_velocity.shape}")
    # SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
    # NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

    n_EC = EC_input_matrix.shape[0]
    n_pos = EC_input_matrix.shape[1]
    n_trials = EC_input_matrix.shape[2]

    dx = 180.0 / n_pos  

    vel = an_velocity
    if vel.shape != (n_pos, n_trials):
        raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

    L_prev = 500

    # precomputed_prepend = np.zeros(L_prev)
    weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng)
    max_length = 0
    dend_vm_list = []  

    if debug:
        proc_peak_start = _get_peak_rss_bytes()   # peak at loop start

    warped_list = []

    last_EPSP = None

    for t in range(EC_input_matrix.shape[2]):

        if debug:
            trial_wall_start = time.perf_counter()
            trial_peak_before = _get_peak_rss_bytes()
            trial_curr_before = _get_current_rss_bytes()

        start_time = time.time()
        v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]) 
        dt_s   = dx / v_cm_s                             
        edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
        total_time = float(edges_s[-1])


        # constant-time axis in seconds
        t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
        print(f"len(t_axis) {len(t_axis)}")
        max_time_over_cells = 0
        T = t_axis.size

        firing_example = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
        valid = np.isfinite(firing_example)
        time_points = np.cumsum(dt_s)
        
        firing_zero = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
        warped_zero = np.interp(t_axis, time_points[valid], firing_zero[valid]).astype(np.float32, copy=False)
        holder = warped_zero[:L_prev]

        warped_rows = [] if store_intermediates else None

        rows = np.empty((n_EC, T), dtype=np.float32)   

        two_track_length = np.empty(len(holder) + len(warped_zero))

        t_ms = np.arange(two_track_length.size, dtype=int) * dt_ms


        for cell in range(n_EC):
            firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
            valid = np.isfinite(firing)
            if valid.sum() >= 2:
                time_points = np.cumsum(dt_s)  # length n_pos

                warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                holder = warped[:L_prev]
            else:
                warped = np.full(1, np.nan, dtype=np.float32)
            
            if store_intermediates:
                warped_rows.append(warped)

            two_track_length[:len(holder)] = holder
            two_track_length[-len(warped):] = warped

            spike_times = get_inhom_poisson_spike_times_by_thinning_real(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
            
            epsps = epsps_event_add(spike_times, two_track_length.shape[0], kernel).astype(np.float16, copy=False)
            epsps = epsps[L_prev:]

            last_EPSP = epsps

            
            if len(epsps) > max_time_over_cells:
                max_time_over_cells = len(epsps)
                print(f"trial {t} len(epsps) {len(epsps)}")
            rows[cell,:] = epsps


        warped_list.append(warped_rows) if store_intermediates else None
        
        X_trial = np.stack(rows, axis=0).astype(np.float16, copy=False)

        

        dendy_list = []

        for dend in range(n_dendrites):
            w = weights_EC[:, dend]            # shape (n_EC,)
            vm_t = w @ X_trial 
            dendy_list.append(vm_t)
            
        dend_vm_over_time = np.array(dendy_list, dtype=np.float16)


        # dend_vm_over_time = weights_EC @ X_trial

        if len(dend_vm_over_time) > max_length:
            max_length = len(dend_vm_over_time)

        dend_vm_list.append(dend_vm_over_time)

        
        if debug:
            end_time = time.time()

            end_time = time.time()
            # --- memory + timing report ---
            trial_wall_end = time.time()
            trial_peak_after = _get_peak_rss_bytes()
            trial_curr_after = _get_current_rss_bytes()

            trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
                else (trial_peak_after - trial_peak_before)
            proc_peak_total  = trial_peak_after
            proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
                else (trial_peak_after - proc_peak_start)

            print(
                f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
                f"RSS now={_fmt_bytes(trial_curr_after)}  "
                f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
                f"proc peak={_fmt_bytes(proc_peak_total)}  "
                f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
            )
            print(f"total time {end_time-start_time}")

        del rows, X_trial, dend_vm_over_time, epsps, 

    return dend_vm_list, weights_EC, last_EPSP

def load_yaml_cfg(path: str) -> Tuple[SimConfig, StoreFlags]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    sim_keys = {"dt_constant","dx","L_prev","refractory_ms","seed","store","debug"}
    flags = raw.get("store_flags", {}) or {}
    cfg_kwargs = {k: raw[k] for k in sim_keys if k in raw}
    cfg = SimConfig(**cfg_kwargs)
    flg = StoreFlags(**{k: v for k, v in flags.items() if k in StoreFlags().__dict__})
    return cfg, flg

def _get_peak_rss_bytes():
    """
    Peak RSS for the process since start.
    macOS: ru_maxrss is bytes; Linux: kilobytes → convert to bytes.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024

def _get_current_rss_bytes():
    """
    Current RSS snapshot (needs psutil; returns None if unavailable).
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None

@dataclass
class StoreFlags:
    spikes: bool = False
    epsps: bool = False
    warp_axes: bool = False

@dataclass
class SimConfig:
    dt_constant: float = 0.001
    dx: float = None  # set from EC shape if not provided
    L_prev: int = 0
    refractory_ms: float = 3.0
    seed: int = 42
    store: str = "mean"     # or "all"
    debug: bool = False
    dist: str = "Lognormal" # label for weights

class SpikeSimModel:
    def __init__(
        self,
        # EC_input_matrix: np.ndarray,      # (n_EC, n_pos, n_trials) Hz
        # an_velocity: np.ndarray,          # (n_pos, n_trials) cm/s
        kernel: np.ndarray,               # EPSP kernel (samples)
        dist_for_weights,                 # callable or string label
        weights_SST=None, weights_NDNF=None,
        config: Optional[SimConfig]=None,
        flags: Optional[StoreFlags]=None,
    ):
        # self.EC = np.asarray(EC_input_matrix, dtype=np.float32, order="C")
        # self.vel = np.asarray(an_velocity, dtype=np.float32, order="C")
        self.kernel = np.asarray(kernel, dtype=np.float32, order="C")
        self.W_dist = dist_for_weights
        self.W_dist_name = dist_for_weights if isinstance(dist_for_weights, str) else getattr(dist_for_weights, "__name__", "weights")
        self.W_SST = weights_SST
        self.W_NDNF = weights_NDNF

        # self.n_EC, self.n_pos, self.n_trials = self.EC.shape
        # assert self.vel.shape == (self.n_pos, self.n_trials), f"velocity shape mismatch: {self.vel.shape} vs {(self.n_pos, self.n_trials)}"

        self.cfg = config or SimConfig()
        # if self.cfg.dx is None:
            # self.cfg.dx = 180.0 / self.n_pos
        self.flags = flags or StoreFlags()

        self._precomputed = None
        self._results: Dict[int, Dict[str, Any]] = {}
        # optional plot context
        self.tau_ms: Optional[float] = None
        self.dend_threshold: Optional[float] = None
        self.dist: str = self.W_dist_name
        self.seeds: List[int] = []
        # evaluate() aggregates / plotting hooks
        self._pos_cnt_dict = {}
        self.start_pos_cnt50_dict = {}
        self._plateau_arr_list_dict = {}
        self._mask_dict = {}
        self._starts_list_dict = {}
        self.num_plateaus_per_dend_list_dict = {}
        self.dend_activity_dict = {}
        self.padded_warped_activity_list_dict = {}


    # --------- public API ----------------------------------------------------
    def store_intermediates(self, **kwargs) -> "SpikeSimModel":
        """Enable/disable what to keep. Example:
           model.store_intermediates(spikes=False, epsps=True, warp_axes=True)
        """
        self.flags = replace(self.flags, **kwargs)
        return self
    
    def __getstate__(self):
        state = self.__dict__.copy()

        results = state.get("_results", {})
        slim_results = {}
        for seed, R in results.items():
            slim_R = {
                "Vm": R.get("Vm"),
                "weights_EC": R.get("weights_EC"),
                "last_epsp_example": R.get("last_epsp_example"),
                "metrics": R.get("metrics"),
            }
            slim_results[seed] = slim_R
        state["_results"] = slim_results

        # ensure plot keys exist
        for k in ("_pos_cnt_dict","start_pos_cnt50_dict","_plateau_arr_list_dict",
                "_mask_dict","_starts_list_dict","num_plateaus_per_dend_list_dict",
                "dend_activity_dict","padded_warped_activity_list_dict","seeds","tau_ms"):
            state.setdefault(k, {} if k.endswith("_dict") else None)

        # keep small helpful context
        state.setdefault("vel", getattr(self, "vel", None))
        state.setdefault("activity_EC", None)
        state.setdefault("activity_SST", None)
        state.setdefault("activity_NDNF", None)
        state.setdefault("SST_sf_opt", None)
        state.setdefault("NDNF_sf_opt", None)
        state.setdefault("dend_threshold", getattr(self, "dend_threshold", None))
        state.setdefault("animal", None)

        # drop heavy/non-picklables
        state["_precomputed"] = None
        state.pop("rows_buf", None)
        state.pop("rate_buf", None)

        # keep a label for the weight distribution
        state["W_dist"] = None
        state["W_dist_name"] = getattr(self, "W_dist_name", None)
        return state

    def evaluate(self,important_dict: Dict[str, Any],seeds: List[int],*,dend_threshold: float,include_inhibition: bool = True,vel_applied: str = "real",example_cell: int = 15,target_total: float = 120.0,target_frac: np.ndarray = None) -> Tuple[float, Dict[str, float]]:
        if target_frac is None:
            target_frac = np.full(10, 0.1, dtype=float)

        dvms = important_dict["dend_vm_per_seed_dict"]
        num_plateaus_per_dend = important_dict["num_plateaus_per_dend_dict"]

        outs = []
        for s in seeds:
            num_per_dend = np.asarray(num_plateaus_per_dend[s], dtype=int)  # (n_dend,)
            total_plateaus = float(num_per_dend.sum())
            if total_plateaus <= 0:
                frac10 = np.zeros(10, float)
            else:
                frac10 = np.full(10, 0.1, dtype=float)
            violations = np.maximum(0.0, num_per_dend - 2.0)  # plateaus per dend > 2 as a violation
            outs.append((total_plateaus, frac10, violations, num_per_dend))

        # Aggregate
        totals     = [o[0] for o in outs]
        frac_sum   = sum(o[1] for o in outs)
        violations = [float(o[2].sum()) for o in outs]
        total_violations = float(np.sum(violations))

        active_fracs, f12_act_list = [], []
        for (_t, _frac10, _viol, num_per_dend) in outs:
            active_mask = (num_per_dend > 0)
            active_fracs.append(float(active_mask.mean()))
            if active_mask.any():
                f12 = (num_per_dend == 1) | (num_per_dend == 2)
                f12_act_list.append(float(f12[active_mask].mean()))
            else:
                f12_act_list.append(0.0)

        frac_active = float(np.mean(active_fracs)) if active_fracs else 0.0
        f12_active  = float(np.mean(f12_act_list)) if f12_act_list else 0.0
        mean_total  = float(np.mean(totals)) if totals else 0.0

        if mean_total < 1.0:
            loss = 1e7 + (mean_total - target_total) ** 2
            metrics = dict(mean_total=mean_total, violations=total_violations,
                        frac_active=frac_active, f12_active=f12_active,
                        mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)
        elif total_violations > 0:
            loss = float(1e6 * total_violations)
            metrics = dict(mean_total=mean_total, violations=total_violations,
                        frac_active=frac_active, f12_active=f12_active,
                        mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)
        else:
            mse_total = (mean_total - target_total) ** 2
            mean_frac = frac_sum / max(len(seeds), 1)
            mse_frac  = float(np.mean((mean_frac - target_frac) ** 2))
            w_sparsity, w_shape12 = 2.0, 1.0
            pen_sparsity = (frac_active - 0.30) ** 2
            pen_12only   = (1.0 - f12_active) ** 2
            loss = float(mse_total + mse_frac + w_sparsity*pen_sparsity + w_shape12*pen_12only)
            metrics = dict(mean_total=mean_total, violations=total_violations,
                        frac_active=frac_active, f12_active=f12_active,
                        mse_total=float(mse_total), mse_frac=float(mse_frac),
                        pen_sparsity=float(pen_sparsity), pen_12only=float(pen_12only))
        # stash a copy for __getstate__
        self._results["metrics"] = metrics
        self.dend_threshold = dend_threshold
        return loss, metrics



        ############ figure out how to use this to get the outputs that I want 



    def plot_from_pickle(path, *, plot_fn, animal="animal-1", **extra):
        P = SpikeSimModel.load_pickle(path)
        seeds = P["seeds"]
        results = P["results"]

        dend_vm_per_seed_dict = {int(s): results[str(s)]["Vm"] for s in results}
        last_seed = seeds[0]
        last = results[str(last_seed)]
        last_EPSP = last.get("last_epsp_example", np.zeros((1,1,1000), np.float32))
        weights_EC = last.get("weights_EC", None)

        return plot_fn(
            tau_ms=P.get("tau_ms", 0.0),
            seeds=seeds,
            last_EPSP=last_EPSP,
            weights_EC=weights_EC,
            weights_SST=extra.get("weights_SST", 0),
            weights_NDNF=extra.get("weights_NDNF", 0),
            dend_vm_per_seed_dict=dend_vm_per_seed_dict,
            activity_EC=extra["activity_EC"],        # you provide (D, trials, T) if needed
            activity_SST=extra.get("activity_SST", 0),
            activity_NDNF=extra.get("activity_NDNF", 0),
            SST_sf_opt=extra.get("SST_sf_opt", 0.0),
            NDNF_sf_opt=extra.get("NDNF_sf_opt", 0.0),
            padded_warped_activity_list=P["padded_warped_activity_list_dict"].get(str(last_seed), []),
            an_velocity=P["velocity"][:, 0],         # 1D for your plot
            dend_activity_dict=P["dend_activity_dict"].get(str(last_seed), {}),
            dend_threshold=extra["dend_threshold"],
            _pos_cnt_dict=P["_pos_cnt_dict"],
            start_pos_cnt50_dict=P["start_pos_cnt50_dict"],
            _plateau_arr_list_dict=P["_plateau_arr_list_dict"],
            _mask_dict=P["_mask_dict"],
            _starts_list_dict=P["_starts_list_dict"],
            dist=extra.get("dist", "weights"),
            num_plateaus_per_dend_list=P["num_plateaus_per_dend_list_dict"].get(str(last_seed), []),
            animal=animal,
            example_cell=extra.get("example_cell", 1),
            include_inhibition=extra.get("include_inhibition", "neither"),
            NDNF_contribution_sum=extra.get("NDNF_contribution_sum", None),
            SST_contribution_sum=extra.get("SST_contribution_sum", None),
            animal_by_animal=True,
        )



    def export_intermediates(self, important_dict: Dict[str, Any], path: str):
        with open(path, "wb") as f:
            pickle.dump(important_dict, f)
        print(f"pickle saved to {path}")

    @staticmethod
    def load_intermediates_for_plotting(path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return pickle.load(f)
        
    
    def _precompute_once(self):
        if self._precomputed is not None:
            return
        # put any conv kernels, dt, time axes here if you need them
        self._precomputed = True

    def _simulate_one_seed(self, seed: int):
        # Hitches: all of these attributes must exist on self BEFORE simulate()
        residual_activity_dict_EC = self.residual_activity_dict_EC
        fixed_residual_activity_dict_NDNF_newest = self.fixed_residual_activity_dict_NDNF_newest
        residual_activity_dict_SST = self.residual_activity_dict_SST
        factors_dict_EC = self.factors_dict_EC
        factors_dict_SST = self.factors_dict_SST
        factors_dict_NDNF_newest = self.factors_dict_NDNF_newest
        GLM_params_EC = self.GLM_params_EC
        GLM_params_NDNF_newest = self.GLM_params_NDNF_newest
        GLM_params_SST = self.GLM_params_SST
        mean_new_average_vel_array = self.mean_new_average_vel_array
        real_vel = self.real_vel
        constant_vel = self.constant_vel
        add_inh = self.add_inh
        make_it_spike = self.make_it_spike
        SST_bias_factor = self.SST_bias_factor
        dist = self.dist
        vel_applied = self.vel_applied
        use_averaged_velocity = self.use_averaged_velocity

        activity_NDNF = 0
        activity_SST  = 0

        (an_velocity, dendrite_contribution_EC, NDNF_pop_list, SST_pop_list,
        NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum,
        weights_EC, weights_SST, weights_NDNF, last_EPSP) = get_dend_contribution(
            residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest,
            residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest,
            GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array,
            real_vel=real_vel, constant_vel=constant_vel, use_residuals=True, use_model_EC=self.use_model_EC,
            multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor,
            dist=dist, use_averaged_velocity=use_averaged_velocity, make_it_spike=make_it_spike,
            seed=seed, store_intermediates=False)

        (padded_warped_activity_list, dend_vm, plateau_positions_counter,
        plateau_start_positions_counter, plateau_array_per_dendrite_list,
        dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list,
        EC_used, dist, num_plateaus_per_dend_list) = get_activity_multidendrite(
            an_velocity, dendrite_contribution_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt,
            dend_threshold=-70, vel_applied=vel_applied, example_cell=17, dist=dist, n_dendrites=100,
            n_SST=75, n_EC=792, n_NDNF=73, include_inhibition=add_inh, use_model_EC=False, make_it_spike=make_it_spike)

        return (dend_vm, plateau_positions_counter, padded_warped_activity_list,
                plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,
                num_plateaus_per_dend_list, plateau_start_times_list_mega_list, last_EPSP,
                weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST, activity_NDNF,
                SST_sf_opt, NDNF_sf_opt)

    def simulate(self, seeds, export=False, plot=False):
        self._precompute_once()
        seeds = list(seeds)
        self.seeds = seeds

        dend_vm_per_seed_dict = {}
        _pos_cnt_dict = {}
        padded_warped_activity_list_dict = {}
        start_pos_cnt50_dict = {}
        _plateau_arr_list_dict = {}
        _mask_dict = {}
        num_plateaus_per_dend_dict = {}
        _starts_list_dict = {}
        last_EPSP_dict = {}
        weights_EC_dict = {}
        weights_SST_dict = {}
        weights_NDNF_dict = {}
        an_velocity_dict = {}
        activity_SST_dict = {}
        activity_NDNF_dict = {}
        SST_sf_opt_dict = {}
        NDNF_sf_opt_dict = {}

        for seed in seeds:
            (dend_vm, plateau_positions_counter, padded_warped_activity_list,
            plateau_start_positions_counter, plateau_array_per_dendrite_list,
            dendrite_plateau_mask, num_plateaus_per_dend_list, plateau_start_times_list_mega_list,
            last_EPSP, weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST,
            activity_NDNF, SST_sf_opt, NDNF_sf_opt) = self._simulate_one_seed(int(seed))

            dend_vm_per_seed_dict[seed] = dend_vm
            _pos_cnt_dict[seed] = plateau_positions_counter
            padded_warped_activity_list_dict[seed] = padded_warped_activity_list
            start_pos_cnt50_dict[seed] = plateau_start_positions_counter
            _plateau_arr_list_dict[seed] = plateau_array_per_dendrite_list
            _mask_dict[seed] = dendrite_plateau_mask
            num_plateaus_per_dend_dict[seed] = num_plateaus_per_dend_list
            _starts_list_dict[seed] = plateau_start_times_list_mega_list
            last_EPSP_dict[seed] = last_EPSP
            weights_EC_dict[seed] = weights_EC
            weights_SST_dict[seed] = weights_SST
            weights_NDNF_dict[seed] = weights_NDNF
            an_velocity_dict[seed] = an_velocity
            activity_SST_dict[seed] = activity_SST
            activity_NDNF_dict[seed] = activity_NDNF
            SST_sf_opt_dict[seed] = SST_sf_opt
            NDNF_sf_opt_dict[seed] = NDNF_sf_opt

        # stash for plotting/evaluate
        important_dict = dict(
            dend_vm_per_seed_dict=dend_vm_per_seed_dict,
            _pos_cnt_dict=_pos_cnt_dict,
            padded_warped_activity_list_dict=padded_warped_activity_list_dict,
            start_pos_cnt50_dict=start_pos_cnt50_dict,
            _plateau_arr_list_dict=_plateau_arr_list_dict,
            _mask_dict=_mask_dict,
            num_plateaus_per_dend_dict=num_plateaus_per_dend_dict,
            _starts_list_dict=_starts_list_dict,
            last_EPSP=last_EPSP_dict,
            weights_EC=weights_EC_dict,
            weights_SST_dict=weights_SST_dict,
            weights_NDNF_dict=weights_NDNF_dict,
            an_velocity_dict=an_velocity_dict,
            activity_SST_dict=activity_SST_dict,
            activity_NDNF_dict=activity_NDNF_dict,
            SST_sf_opt_dict=SST_sf_opt_dict,
            NDNF_sf_opt_dict=NDNF_sf_opt_dict,
        )

        # cache for plotting convenience
        self.padded_warped_activity_list_dict = padded_warped_activity_list_dict
        self._pos_cnt_dict = _pos_cnt_dict
        self.start_pos_cnt50_dict = start_pos_cnt50_dict
        self._plateau_arr_list_dict = _plateau_arr_list_dict
        self._mask_dict = _mask_dict
        self._starts_list_dict = _starts_list_dict
        self.num_plateaus_per_dend_list_dict = num_plateaus_per_dend_dict

        if plot:
            # you referenced plot_multidendrite_EC_err_across_seeds2; call your wrapper here if needed
            pass

        return important_dict


   
@click.group(context_settings=dict(help_option_names=["-h","--help"]))
def cli():
    """SpikeSimModel CLI: simulate→evaluate→export, or load→plot."""
    pass


@cli.command("simulate")
@click.option("-s", "--seed", "seeds", multiple=True, type=int, required=True)
@click.option("-o", "--save-path", type=click.Path(dir_okay=False, writable=True, resolve_path=True), required=True)
@click.option("--dend-threshold", type=float, required=True)
@click.option("--include-inhibition/--no-include-inhibition", default=True)
@click.option("--vel-applied", type=click.Choice(["real", "constant", "average"]), default="real")
def simulate_cmd(seeds, save_path, dend_threshold, include_inhibition, vel_applied):
    # 1) Load data (your helper)
    GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(
        file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(
        file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="EC_GLM", new_NDNF=False)
    GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(
        file_path='/Users/michaelfinch/CA1-interneuron-GLM', name="NDNF_E1A1B", new_NDNF=True)

    # 2) Fix NDNF subsets (your logic)
    fixed_residual_activity_dict_NDNF_newest = {f"animal_{idx+1}": residual_activity_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(residual_activity_dict_NDNF_newest)
                                               if 17 < idx < 31}
    fixed_filtered_factors_dict_NDNF_newest = {f"animal_{idx+1}": filtered_factors_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(filtered_factors_dict_NDNF_newest)
                                               if 17 < idx < 31}

    # 3) Kernel
    dt_constant = 0.0001
    tau_ms  = 5.0
    dt_ms   = dt_constant * 1000.0
    AMP     = 1.0
    MODE    = "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

    # 4) Get EC/SST/NDNF contributions to build EC_input_matrix and velocity
    use_averaged_velocity = "actual_velocity" #"cell_type_av" 
    dist = "Lognormal"
    add_inh = 'neither'
    make_it_spike = True
    SST_bias_multi = 1.4
    real_vel=True
    constant_vel=False
    use_model_EC = True
    SST_bias_factor=2.0


    mean_new_average_vel_array = get_real_velocity_array(filtered_factors_dict_EC, filtered_factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest)


    cfg, flg = load_yaml_cfg("example_config.yaml")
    model = SpikeSimModel(kernel=kernel,dist_for_weights=dist,weights_SST=None, weights_NDNF=None,config=cfg, flags=flg)

    # 6) Attach all the attributes `_simulate_one_seed` expects:
    model.residual_activity_dict_EC = residual_activity_dict_EC
    model.fixed_residual_activity_dict_NDNF_newest = fixed_residual_activity_dict_NDNF_newest
    model.residual_activity_dict_SST = residual_activity_dict_SST
    model.factors_dict_EC = factors_dict_EC
    model.factors_dict_SST = factors_dict_SST
    model.factors_dict_NDNF_newest = factors_dict_NDNF_newest
    model.GLM_params_EC = GLM_params_EC
    model.GLM_params_NDNF_newest = GLM_params_NDNF_newest
    model.GLM_params_SST = GLM_params_SST
    model.mean_new_average_vel_array = None # provide if used
    model.real_vel = (vel_applied == "real")
    model.constant_vel = (vel_applied == "constant")
    model.add_inh = add_inh
    model.make_it_spike = make_it_spike
    model.SST_bias_factor = SST_bias_multi
    model.dist = dist
    model.vel_applied = vel_applied
    model.use_averaged_velocity = use_averaged_velocity
    model.use_model_EC = False
    model.tau_ms = tau_ms
    model.dend_threshold = dend_threshold
    model.mean_new_average_vel_array = mean_new_average_vel_array


    important = model.simulate(seeds=seeds, export=False, plot=False)
    loss, metrics = model.evaluate(important, seeds=list(seeds), dend_threshold=dend_threshold)
    print(f"[EVAL] loss={loss:.6g}  metrics={metrics}")

    state = model.__getstate__()
    state["loss"] = float(loss)
    state["metrics"] = {k: float(v) if v is not None else None for k, v in metrics.items()}

    with open(save_path, "wb") as f:
        pickle.dump(state, f)
    click.echo(f"Saved slim model to {save_path}")


def _load_light_model(path: str):
    with open(path, "rb") as f:
        state = pickle.load(f)
    obj = SpikeSimModel(
        EC_input_matrix=np.zeros((1,1,1), dtype=np.float32),
        an_velocity=np.zeros((1,1), dtype=np.float32),
        kernel=np.zeros((10,), dtype=np.float32),
        dist_for_weights="",
    )
    obj.__dict__.update(state)
    return obj

@cli.command("plot")
@click.option("-i", "--pickle-path", type=click.Path(exists=True, dir_okay=False, resolve_path=True), required=True)
@click.option("--out", type=click.Path(dir_okay=False, resolve_path=True))
def plot_cmd(pickle_path, out):
    model = _load_light_model(pickle_path)
    if model.dend_threshold is None:
        raise click.UsageError("dend_threshold not found in pickle. Provide it at simulate time.")

    # Use your wrapper to call plot_multidendrite_EC_err_across_seeds2
    # Example (adapt to your function’s signature):
    # plot_multidendrite_EC_err_across_seeds2(...)

    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
        click.echo(f"Saved figure to: {out}")


if __name__ == "__main__":
    cli()