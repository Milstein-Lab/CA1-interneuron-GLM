# (ca1_env) michaelfinch@nbp-25-228-133 Clean_notebooks_to_date % python build_a_model_object.py simulate \
#   -s 0 -s 1 \
#   -o /Users/michaelfinch/CA1-interneuron-GLM/tmp/spike_sim.pkl \
#   --dend-threshold -70 \
#   --vel-applied real \ 
#   --o ./spike_sim.pkl \                                                                                   
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

from mpi4py import MPI
import os, sys, time

import psutil, os, gc, time, numpy as np

def report_mem(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    print(f"[{time.time():.3f}] {tag} RSS={rss:.2f} GB", flush=True)

def nbytes(arr, name):
    try:
        print(f"  {name}.nbytes={arr.nbytes/1e6:.1f} MB", flush=True)
    except Exception:
        pass




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

# ###### og ##### loss = 0.00018764
def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):
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
        generator = np.random.default_rng()
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




# loss = 0.0002368
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
#     # last_spike_t = -math.inf  # enforce refractory in ms
#     last_spike_t = -np.inf
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



# def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None):

#     """
#     Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
#     Poisson process with a refractory period after each spike.
#     :param rate: instantaneous rates in time (Hz)
#     :param t: corresponding time values (ms)
#     :param dt: temporal resolution for spike times (ms)
#     :param refractory: absolute deadtime following a spike (ms)
#     :param generator: :class:'np.random.RandomState()'
#     :return: list of m spike times (ms)
#     """
#     if generator is None:
#         generator = np.random.default_rng()
#     interp_t = np.arange(t[0], t[-1] + dt, dt)
#     try:
#         interp_rate = np.interp(interp_t, t, rate)
#     except Exception as e:
#         print('t shape: %s rate shape: %s' % (str(t.shape), str(rate.shape)))
#         sys.stdout.flush()
#         time.sleep(0.1)
#         raise(e)
#     interp_rate /= 1000.
#     spike_times = []
#     non_zero = np.where(interp_rate > 1.e-100)[0]
#     if len(non_zero) == 0:
#         return spike_times
#     interp_rate[non_zero] = 1. / (1. / interp_rate[non_zero] - refractory)
#     max_rate = np.max(interp_rate)
#     if not max_rate > 0.:
#         return spike_times
#     i = 0
#     ISI_memory = 0.
#     while i < len(interp_t):
#         x = generator.uniform(0.0, 1.0)
#         if x > 0.:
#             ISI = -np.log(x) / max_rate
#             i += int(ISI / dt)
#             ISI_memory += ISI
#             if (i < len(interp_t)) and (generator.uniform(0.0, 1.0) <= (interp_rate[i] / max_rate)) and \
#                     ISI_memory >= 0.:
#                 spike_times.append(interp_t[i])
#                 ISI_memory = -refractory
#     return np.asarray(spike_times, dtype=float)


def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean from: {x!r}")



# def get_dend_contribution(kernel, dt_constant, residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=None, constant_vel=None, use_residuals=True, use_model_EC=False, multiple_dendrites=False, add_inh=None, seed=0, SST_bias_factor=None, dist=None, use_averaged_velocity=None, make_it_spike=False, store_intermediates=False, animal_by_animal=False, input_animal=None, include_beta=None, flat_input=None, optimization_time=False, debug=False, mean=None, std=None):
    
#     SEED = seed
#     np.random.seed(SEED)
#     random.seed(SEED)
#     rng = np.random.default_rng(SEED)

#     animal_by_animal = to_bool(animal_by_animal)
#     flat_input = to_bool(flat_input)
#     include_beta = to_bool(include_beta)


#     if use_model_EC:

#         if make_it_spike:
            
#             n_EC = 792
#             pos_bins = 50
#             n_trials = 58

#             data_list_normalized = []

#             for animal in residual_activity_dict_EC:
#                 for cell in residual_activity_dict_EC[animal]:
#                     data_normalized = residual_activity_dict_EC[animal][cell][:,:58]
#                     data_normalized = (data_normalized - np.min(data_normalized)) / (np.max(data_normalized) - np.min(data_normalized)) *50
#                     data_list_normalized.append(data_normalized)


#             data_array_normalized = np.array(data_list_normalized)

#             overall_mu= np.mean(data_list_normalized)
#             overall_std = np.std(data_list_normalized)


#             animal_velocity_list_cm_sec = get_vel_formatted(residual_activity_dict_EC, factors_dict_EC)
#             data_normalized_list, r2_list = get_correlation(residual_activity_dict_EC, factors_dict_EC)
#             overall_mean = np.mean(data_normalized_list)
#             synthetic_data_plus_vel_list, synthetic_activity = get_synthetic_data(n_EC, pos_bins, n_trials, overall_mean, animal_velocity_list_cm_sec, r2_list)

#             animal_velocity_array = np.array(animal_velocity_list_cm_sec)
#             an_velocity = np.mean(animal_velocity_array, axis=0) / 100

#             if optimization_time:
#                 dend_vm_list, weights_EC, last_EPSP = turn_rates_into_spikes(synthetic_data_plus_vel_list, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng, debug=debug, optimization_time=False, mean=mean, std=std)
#             else:
#                 dend_vm_list, weights_EC, last_EPSP, warped_list = turn_rates_into_spikes(synthetic_data_plus_vel_list, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng, debug=debug, optimization_time=True, mean=mean, std=std)

#             num_trials   = len(dend_vm_list)
#             n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


#             # dend_vm_list: list of 2D arrays, each (n_dendrites, T_i)

#             num_trials = len(dend_vm_list)
#             n_dendrites_ = dend_vm_list[0].shape[0]
#             max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

#             dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

#             for i, arr in enumerate(dend_vm_list):
#                 if arr.shape[0] != n_dendrites_:
#                     raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
#                 Ti = arr.shape[1]               # current trial's time length
#                 dend_vm_padded[i, :, :Ti] = arr # pad along time axis


#             print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

#             Vm_list = []
#             for dend in range(dend_vm_padded.shape[1]):
#                 trials_by_time = dend_vm_padded[:,dend,:]
#                 Vm, _, _ = activity_to_dend_vm_2d(
#                 trials_by_time,
#                 Vrest=-70.0,
#                 vm_scale=0.1,
#                 center_across="time")
#                 Vm_list.append(Vm)

#             activity_EC = np.array(Vm_list)

#             activity_NDNF=0
#             activity_SST=0 
#             NDNF_sf_opt=0 
#             SST_sf_opt=0 
#             NDNF_contribution_sum=0
#             SST_contribution_sum=0
#             weights_SST=0
#             weights_NDNF=0

#             return an_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP

        
#         else:
#             random.seed(seed)
#             ts_list = []
#             for i in range(58):
#                 ts = random_timeseries(1.0, 1., 49)
#                 ts_list.append(ts)

#             dend_contribution_EC = np.array(ts_list).T
#             print(dend_contribution_EC.shape)

#             if constant_vel:
#                 an_velocity = np.full(dend_contribution_EC.shape, 0.43)
#             else:
#                 an_velocity = np.tile(mean_new_average_vel_array[:,None], (1,58))


#             # dend_contribution_EC /= dend_contribution_EC

#             return dend_contribution_EC, an_velocity 
            
    
#     else:
#         if multiple_dendrites:

#             animal_by_animal = to_bool(animal_by_animal)
#             include_beta = to_bool(include_beta)
#             flat_input = to_bool(flat_input)
#             constant_vel = to_bool(constant_vel)
            
#             if animal_by_animal:

#                 an_velocity, dend_list_EC = get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=True, animal_used=input_animal)
    
#             else:

#                 an_velocity, dend_list_EC = get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=False, animal_used=input_animal)

#             if make_it_spike:

#                 try:
#                     import psutil
#                     _PROC = psutil.Process(os.getpid())
#                 except Exception:
#                     psutil = None
#                     _PROC = None

#                 if optimization_time:
#                     dend_vm_list, weights_EC, last_EPSP = turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, rng=rng, debug=debug, optimization_time=True, mean=mean, std=std)
#                 else:
#                     dend_vm_list, weights_EC, last_EPSP, warped_list = turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, rng=rng, debug=debug, optimization_time=False, mean=mean, std=std)

#                 activity_NDNF=0
#                 activity_SST=0 
#                 NDNF_sf_opt=0 
#                 SST_sf_opt=0 
#                 NDNF_contribution_sum=0
#                 SST_contribution_sum=0
#                 weights_SST=0
#                 weights_NDNF=0

#             else:
#                 n_EC = 792
#                 n_SST = 75
#                 n_NDNF = 115
#                 n_dendrites=100


#                 EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
#                 SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
#                 NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

#                 weights_EC = sample_weights(dist, n_dendrites, n_EC, rng=rng)
#                 weights_SST = sample_weights('Equal', n_dendrites, n_SST, rng=rng)
#                 weights_NDNF = sample_weights('Equal', n_dendrites, n_NDNF, rng=rng)

#                 activity_EC = get_dendrite_activity(weights_EC, EC_input_matrix, n_dendrites, n_EC)
#                 activity_SST = get_dendrite_activity(weights_SST, SST_input_matrix, n_dendrites, n_SST)
#                 activity_NDNF = get_dendrite_activity(weights_NDNF, NDNF_input_matrix, n_dendrites, n_NDNF)

#                 EC = activity_EC.copy()
#                 NDNF = activity_NDNF.copy()
#                 SST = activity_SST.copy()

#                 if add_inh=='sst':
#                     SST_sf_opt, info = fit_sst_scale_to_cancel_ec(activity_EC, activity_SST)
#                     NDNF_sf_opt=0
#                 else:
#                     res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
#                     NDNF_sf_opt = res["ndnf_sf"]
#                     SST_sf_opt  = res["sst_sf"]
#                     NDNF_contribution_sum = res["contrib_L2_ndnf"]
#                     SST_contribution_sum = res["contrib_L2_sst"]


#             if optimization_time:
#                 return an_velocity, dend_vm_list, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP #, min_trial_length
#             else:
#                 return an_velocity, dend_vm_list, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_list
#         else:
#             return dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, last_EPSP #, min_trial_length









def get_dend_contribution(kernel, dt_constant, residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, NDNF_sf_opt, SST_sf_opt, dist=None, include_inh=False, constant_vel=None, multiple_dendrites=False, seed=0, make_it_spike=False, animal_by_animal=False, input_animal=None, include_beta=None, flat_input=None, optimization_time=False, debug=False, mean=None, std=None):
    
    SEED = seed
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    animal_by_animal = to_bool(animal_by_animal)
    flat_input = to_bool(flat_input)
    include_beta = to_bool(include_beta)

    # else:
    if multiple_dendrites:

        animal_by_animal = to_bool(animal_by_animal)
        include_beta = to_bool(include_beta)
        flat_input = to_bool(flat_input)
        constant_vel = to_bool(constant_vel)
        
        if animal_by_animal:

            an_velocity, dend_list_EC = get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=True, animal_used=input_animal)

        else:

            if include_inh:
                an_velocity_EC, dend_list_EC = get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=False, animal_used=input_animal)
                an_velocity_SST, dend_list_SST = get_dend_VM_cell_type_new(residual_activity_dict_SST, factors_dict_SST, GLM_params_SST, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=False, animal_used=input_animal)
                an_velocity_NDNF, dend_list_NDNF = get_dend_VM_cell_type_new(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, GLM_params_NDNF_newest, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=False, animal_used=input_animal)


                an_velocity_stack = np.stack([an_velocity_EC, an_velocity_SST, an_velocity_NDNF])
                an_velocity = np.mean(an_velocity_stack, axis=0)


            else:
                an_velocity, dend_list_EC = get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=include_beta, const_vel=constant_vel, flat_input=flat_input, animal_by_animal=False, animal_used=input_animal)

                # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/an_vel_pkl.pkl"
                # with open(save_path, 'wb') as f:
                #     pickle.dump(an_velocity, f)
        if make_it_spike:

            try:
                import psutil
                _PROC = psutil.Process(os.getpid())
            except Exception:
                psutil = None
                _PROC = None
            
            if include_inh:
                
                dend_vm_list, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_list = turn_rates_into_spikes_add_inh(
                    dend_list_EC, dend_list_NDNF, dend_list_SST, an_velocity, dist, kernel, SST_sf_opt, NDNF_sf_opt,
                    dt_constant=dt_constant, n_dendrites=100, rng=rng, debug=debug, optimization_time=True, mean=mean,
                    std=std)
                return an_velocity, dend_vm_list, activity_NDNF, activity_SST, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_list
            else:
                
                activity_NDNF=0
                activity_SST=0
                NDNF_contribution_sum=0
                SST_contribution_sum=0
                weights_SST=0
                weights_NDNF=0
                
                if optimization_time:
                    internals_dict = dict(dend_list_EC=dend_list_EC, an_velocity=an_velocity, dist=dist, kernel=kernel,
                                          dt_constant=dt_constant)
                
                dend_vm_list, weights_EC, last_EPSP, warped_list = (
                    turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=dt_constant,
                                           n_dendrites=100, rng=rng, debug=debug, optimization_time=optimization_time,
                                           mean=mean, std=std))
                return an_velocity, dend_vm_list, activity_NDNF, activity_SST, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_list #, min_trial_length
                    

# (an_velocity, dend_activity, NDNF_pop_list, SST_pop_list, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF, last_EPSP) = get_dend_contribution(self.kernel, self.dt_constant,
#                 residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest,
#                 residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest,
#                 GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, self.NDNF_sf_opt, self.SST_sf_opt, include_inh=self.include_inh,
#                 real_vel=real_vel, constant_vel=constant_vel, use_residuals=True, use_model_EC=self.use_model_EC,
#                 multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor,
#                 dist=dist, use_averaged_velocity=use_averaged_velocity, make_it_spike=make_it_spike,
#                 seed=seed, animal_by_animal=self.animal_by_animal, input_animal=self.input_animal, include_beta=self.include_beta, flat_input=self.flat_input, optimization_time=True, debug=debug, mean=mean, std=std)


    # if use_model_EC:

    #     if make_it_spike:
            
    #         n_EC = 792
    #         pos_bins = 50
    #         n_trials = 58

    #         data_list_normalized = []

    #         for animal in residual_activity_dict_EC:
    #             for cell in residual_activity_dict_EC[animal]:
    #                 data_normalized = residual_activity_dict_EC[animal][cell][:,:58]
    #                 data_normalized = (data_normalized - np.min(data_normalized)) / (np.max(data_normalized) - np.min(data_normalized)) *50
    #                 data_list_normalized.append(data_normalized)


    #         data_array_normalized = np.array(data_list_normalized)

    #         overall_mu= np.mean(data_list_normalized)
    #         overall_std = np.std(data_list_normalized)


    #         animal_velocity_list_cm_sec = get_vel_formatted(residual_activity_dict_EC, factors_dict_EC)
    #         data_normalized_list, r2_list = get_correlation(residual_activity_dict_EC, factors_dict_EC)
    #         overall_mean = np.mean(data_normalized_list)
    #         synthetic_data_plus_vel_list, synthetic_activity = get_synthetic_data(n_EC, pos_bins, n_trials, overall_mean, animal_velocity_list_cm_sec, r2_list)

    #         animal_velocity_array = np.array(animal_velocity_list_cm_sec)
    #         an_velocity = np.mean(animal_velocity_array, axis=0) / 100

    #         if optimization_time:
    #             dend_vm_list, weights_EC, last_EPSP = turn_rates_into_spikes(synthetic_data_plus_vel_list, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng, debug=debug, optimization_time=False, mean=mean, std=std)
    #         else:
    #             dend_vm_list, weights_EC, last_EPSP, warped_list = turn_rates_into_spikes(synthetic_data_plus_vel_list, an_velocity, dist, kernel, dt_constant=dt_constant, n_dendrites=100, store_intermediates=False, rng=rng, debug=debug, optimization_time=True, mean=mean, std=std)

    #         num_trials   = len(dend_vm_list)
    #         n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


    #         # dend_vm_list: list of 2D arrays, each (n_dendrites, T_i)

    #         num_trials = len(dend_vm_list)
    #         n_dendrites_ = dend_vm_list[0].shape[0]
    #         max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

    #         dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

    #         for i, arr in enumerate(dend_vm_list):
    #             if arr.shape[0] != n_dendrites_:
    #                 raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
    #             Ti = arr.shape[1]               # current trial's time length
    #             dend_vm_padded[i, :, :Ti] = arr # pad along time axis


    #         print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

    #         Vm_list = []
    #         for dend in range(dend_vm_padded.shape[1]):
    #             trials_by_time = dend_vm_padded[:,dend,:]
    #             Vm, _, _ = activity_to_dend_vm_2d(
    #             trials_by_time,
    #             Vrest=-70.0,
    #             vm_scale=0.1,
    #             center_across="time")
    #             Vm_list.append(Vm)

    #         activity_EC = np.array(Vm_list)

    #         activity_NDNF=0
    #         activity_SST=0 
    #         NDNF_contribution_sum=0
    #         SST_contribution_sum=0
    #         weights_SST=0
    #         weights_NDNF=0

    #         return an_velocity, activity_EC, activity_NDNF, activity_SST, weights_EC, weights_SST, weights_NDNF, last_EPSP

        
    #     else:
    #         random.seed(seed)
    #         ts_list = []
    #         for i in range(58):
    #             ts = random_timeseries(1.0, 1., 49)
    #             ts_list.append(ts)

    #         dend_contribution_EC = np.array(ts_list).T
    #         print(dend_contribution_EC.shape)

    #         if constant_vel:
    #             an_velocity = np.full(dend_contribution_EC.shape, 0.43)
    #         else:
    #             an_velocity = np.tile(mean_new_average_vel_array[:,None], (1,58))


    #         # dend_contribution_EC /= dend_contribution_EC

    #         return dend_contribution_EC, an_velocity 
            
    












# --- memory debug helpers ---
import os, psutil

_PEAK_RSS = 0  # bytes

def _get_current_rss_bytes():
    try:
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return 0

def _get_peak_rss_bytes():
    global _PEAK_RSS
    rss = _get_current_rss_bytes()
    if rss > _PEAK_RSS:
        _PEAK_RSS = rss
    return _PEAK_RSS

def _fmt_bytes(n):
    try:
        for unit in ("B","KB","MB","GB","TB"):
            if n < 1024.0:
                return f"{n:,.1f} {unit}"
            n /= 1024.0
    except Exception:
        pass
    return f"{n} B"


# def turn_rates_into_spikes(
#     dend_list_EC, an_velocity, dist, kernel,
#     dt_constant=None, n_dendrites=100, rng=None, debug=False, optimization_time=None
# ):
#     assert dt_constant is not None
#     dt_ms = float(dt_constant * 1000.0)

#     if optimization_time:
#         # ----- OPT PATH (no warped_list returned) -----
#         EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
#         n_EC, n_pos, n_trials = EC_input_matrix.shape
#         dx = np.float32(180.0 / n_pos)

#         weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng).astype(np.float32, copy=False)
#         kernel = np.asarray(kernel, dtype=np.float32)

#         L_prev = 1200
#         last_EPSP = None
#         dend_vm_list = []

#         if debug:
#             proc_peak_start = _get_peak_rss_bytes()

#         for t in range(n_trials):
#             if debug:
#                 trial_wall_start = time.perf_counter()
#                 trial_peak_before = _get_peak_rss_bytes()
#                 trial_curr_before = _get_current_rss_bytes()

#             v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)
#             dt_s   = dx / v_cm_s
#             time_points = np.cumsum(dt_s, dtype=np.float32)
#             total_time  = float(time_points[-1])
#             t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)
#             T = t_axis.size

#             rows = np.empty((n_EC, T), dtype=np.float32)

#             firing0 = EC_input_matrix[0, :, t]
#             valid0  = np.isfinite(firing0)
#             warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
#                        if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
#             holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

#             two_len = len(holder) + T + len(holder)
#             two_track_length = np.empty(two_len, dtype=np.float32)
#             t_ms = (np.arange(two_len, dtype=np.float32) * np.float32(dt_ms))

#             for cell in range(n_EC):
#                 firing = EC_input_matrix[cell, :, t]
#                 valid  = np.isfinite(firing)
#                 if valid.sum() >= 2:
#                     warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
#                 else:
#                     warped = np.full(1, np.nan, dtype=np.float32)

#                 hL = len(holder)
#                 two_track_length[:hL] = holder
#                 two_track_length[hL:hL + warped.size] = warped
#                 two_track_length[hL + warped.size:] = holder
#                 holder = warped[-L_prev:] if warped.size >= L_prev else warped

#                 spike_times = get_inhom_poisson_spike_times_by_thinning(
#                     two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng
#                 ).astype(np.int32, copy=False)

#                 epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
#                 epsps = epsps[hL:hL + T]
#                 rows[cell, :] = epsps
#                 last_EPSP = epsps

#             dend_vm_over_time = (weights_EC.T @ rows).astype(np.float32, copy=False)  # (n_dendrites, T)
#             dend_vm_list.append(dend_vm_over_time)

#             if debug:
#                 trial_wall_end = time.perf_counter()
#                 trial_peak_after = _get_peak_rss_bytes()
#                 trial_curr_after = _get_current_rss_bytes()
#                 trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
#                     else (trial_peak_after - trial_peak_before)
#                 proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
#                     else (trial_peak_after - proc_peak_start)
#                 print(
#                     f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
#                     f"RSS now={_fmt_bytes(trial_curr_after)}  "
#                     f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
#                     f"proc peak={_fmt_bytes(trial_peak_after)}  "
#                     f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
#                 )

#             del rows, two_track_length, warped0, warped, firing, firing0

#         num_trials = len(dend_vm_list)
#         n_dendrites_ = dend_vm_list[0].shape[0]
#         max_T = max(arr.shape[1] for arr in dend_vm_list)

#         dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
#         for i, arr in enumerate(dend_vm_list):
#             Ti = arr.shape[1]
#             dend_vm_padded[i, :, :Ti] = arr

#         dend_contribution_EC = dend_vm_padded  # (trials, dendrites, time) f32

#         if dend_contribution_EC.shape[1] != 100:
#             print("dend_contribution_EC dendrites in the wrong axis")

#         Vm_dict = {}
#         for d in range(dend_contribution_EC.shape[1]):
#             vm_array = dend_contribution_EC[:, d, :]  # f32
#             Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
#             Vm_dict[d] = Vm.astype(np.float32, copy=False)

#         dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
#         return dend_activity, weights_EC, last_EPSP

#     else:
#         # ----- FULL PATH (also returns warped_list) -----
#         EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
#         n_EC, n_pos, n_trials = EC_input_matrix.shape
#         dx = np.float32(180.0 / n_pos)

#         weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng).astype(np.float32, copy=False)
#         kernel = np.asarray(kernel, dtype=np.float32)

#         L_prev = 1200
#         last_EPSP = None
#         dend_vm_list = []
#         warped_list = []

#         if debug:
#             proc_peak_start = _get_peak_rss_bytes()

#         for t in range(n_trials):
#             if debug:
#                 trial_wall_start = time.perf_counter()
#                 trial_peak_before = _get_peak_rss_bytes()
#                 trial_curr_before = _get_current_rss_bytes()

#             v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)
#             dt_s   = dx / v_cm_s
#             time_points = np.cumsum(dt_s, dtype=np.float32)
#             total_time  = float(time_points[-1])
#             t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)
#             T = t_axis.size

#             rows = np.empty((n_EC, T), dtype=np.float32)
#             firing0 = EC_input_matrix[0, :, t]
#             valid0  = np.isfinite(firing0)
#             warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
#                        if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
#             holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

#             two_len = len(holder) + T + len(holder)
#             two_track_length = np.empty(two_len, dtype=np.float32)
#             t_ms = (np.arange(two_len, dtype=np.float32) * np.float32(dt_ms))

#             warped_array = np.empty((n_EC, T), dtype=np.float32)

#             for cell in range(n_EC):
#                 firing = EC_input_matrix[cell, :, t]
#                 valid  = np.isfinite(firing)
#                 if valid.sum() >= 2:
#                     warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
#                 else:
#                     warped = np.full(1, np.nan, dtype=np.float32)

#                 hL = len(holder)
#                 two_track_length[:hL] = holder
#                 two_track_length[hL:hL + warped.size] = warped
#                 two_track_length[hL + warped.size:] = holder
#                 holder = warped[-L_prev:] if warped.size >= L_prev else warped

#                 spike_times = get_inhom_poisson_spike_times_by_thinning(
#                     two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng
#                 ).astype(np.int32, copy=False)

#                 epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
#                 epsps = epsps[hL:hL + T]
#                 rows[cell, :] = epsps
#                 warped_array[cell, :] = warped
#                 last_EPSP = epsps

#             dend_vm_over_time = (weights_EC.T @ rows).astype(np.float32, copy=False)
#             dend_vm_list.append(dend_vm_over_time)
#             warped_list.append(warped_array)

#             if debug:
#                 trial_wall_end = time.perf_counter()
#                 trial_peak_after = _get_peak_rss_bytes()
#                 trial_curr_after = _get_current_rss_bytes()
#                 trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
#                     else (trial_peak_after - trial_peak_before)
#                 proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
#                     else (trial_peak_after - proc_peak_start)
#                 print(
#                     f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
#                     f"RSS now={_fmt_bytes(trial_curr_after)}  "
#                     f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
#                     f"proc peak={_fmt_bytes(trial_peak_after)}  "
#                     f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
#                 )

#             del rows, two_track_length, warped0, warped, firing, firing0

#         num_trials = len(dend_vm_list)
#         n_dendrites_ = dend_vm_list[0].shape[0]
#         max_T = max(arr.shape[1] for arr in dend_vm_list)

#         dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
#         for i, arr in enumerate(dend_vm_list):
#             Ti = arr.shape[1]
#             dend_vm_padded[i, :, :Ti] = arr

#         dend_contribution_EC = dend_vm_padded  # (trials, dendrites, time) f32

#         if dend_contribution_EC.shape[1] != 100:
#             print("dend_contribution_EC dendrites in the wrong axis")

#         Vm_dict = {}
#         for d in range(dend_contribution_EC.shape[1]):
#             vm_array = dend_contribution_EC[:, d, :]
#             Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
#             Vm_dict[d] = Vm.astype(np.float32, copy=False)

#         dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
#         return dend_activity, weights_EC, last_EPSP, warped_list


# def turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=None, n_dendrites=100, rng=None, debug=False, optimization_time=None, mean=None, std=None):
    
#     assert dt_constant is not None
#     dt_ms = float(dt_constant * 1000.0)

#     # Store EC inputs in f16 to keep memory low; upcast when needed
#     EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float16, copy=False)
#     n_EC, n_pos, n_trials = EC_input_matrix.shape
#     dx = np.float32(180.0 / n_pos)

#     # Weights in f32; sanitize/normalize if you had heavy tails
#     weights_EC = sample_weights(dist, n_EC, n_dendrites,
#                                 rng=rng, mean=mean, std=std).astype(np.float32, copy=False)
#     # Optional: tame tails + per-dendrite L1 norm
#     weights_EC[~np.isfinite(weights_EC)] = 0.0
#     colsum = weights_EC.sum(axis=0, keepdims=True)
#     weights_EC /= (colsum + 1e-8)

#     kernel = np.asarray(kernel, dtype=np.float16)  # storage OK in f16

#     # ---- pre-scan trial lengths to get T_max ----
#     T_list = []
#     for t in range(n_trials):
#         v = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)
#         dt_s = dx / v
#         total_time = float(np.cumsum(dt_s, dtype=np.float32)[-1])
#         T_list.append(int(np.ceil(total_time / float(dt_constant))))
#     T_max = int(max(T_list))

#     # ---- pre-allocate outputs (f16 to save RAM) ----
#     dend_vm_padded = np.full((n_trials, n_dendrites, T_max), np.nan, dtype=np.float16)

#     # Reusable scratch buffers (largest needed)
#     L_prev = 1200
#     two_len_max = L_prev + T_max + L_prev
#     two_track = np.empty(two_len_max, dtype=np.float16)   # reused every cell
#     t_idx_full = np.arange(two_len_max, dtype=np.int32)

#     # Optional: only build warped_list when needed
#     if not optimization_time:
#         warped_list = []

#     # ---- block size over EC cells ----
#     block = 64  # tune (32/64/96) based on memory
#     # Accumulator for each trial’s dendrite×time (f32 compute; downcast later)
#     accum = np.empty((n_dendrites, T_max), dtype=np.float32)

#     for t in range(n_trials):
#         # Build time axis
#         v = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)
#         dt_s = dx / v
#         time_points = np.cumsum(dt_s, dtype=np.float32)
#         total_time = float(time_points[-1])
#         t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)
#         T = t_axis.size

#         # Prime holder using cell 0
#         firing0 = EC_input_matrix[0, :, t].astype(np.float32, copy=False)
#         valid0 = np.isfinite(firing0)
#         warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
#                    if valid0.sum() >= 2 else np.zeros(1, dtype=np.float32))
#         holder = warped0.astype(np.float16, copy=False)
#         holder = holder[-L_prev:] if holder.size >= L_prev else holder

#         if not optimization_time:
#             warped_array = np.empty((n_EC, T), dtype=np.float16)

#         # zero the accumulator for this trial
#         accum[:, :T] = 0.0

#         # ---- process EC cells in blocks ----
#         for start in range(0, n_EC, block):
#             end = min(start + block, n_EC)
#             B = end - start

#             # rows for this block (B × T) as f16 storage, cast to f32 at matmul
#             rows_block = np.empty((B, T), dtype=np.float16)

#             for bi, cell in enumerate(range(start, end)):
#                 firing = EC_input_matrix[cell, :, t].astype(np.float32, copy=False)
#                 valid = np.isfinite(firing)
#                 warped = (np.interp(t_axis, time_points[valid], firing[valid])
#                           if valid.sum() >= 2 else np.zeros(1, dtype=np.float32))

#                 # stitch holder | warped | holder into two_track (all f16)
#                 hL = len(holder)
#                 two_len = hL + warped.size + hL
#                 tt = two_track[:two_len]
#                 tt[:hL] = holder
#                 w16 = warped.astype(np.float16, copy=False)
#                 tt[hL:hL + w16.size] = w16
#                 tt[hL + w16.size:] = holder

#                 # update holder for next cell
#                 holder = w16[-L_prev:] if w16.size >= L_prev else w16

#                 # thinning; upcast to f32 for the function if it expects f32
#                 spike_times = get_inhom_poisson_spike_times_by_thinning(
#                     tt.astype(np.float32, copy=False), t_idx_full[:two_len],
#                     dt=dt_ms, refractory=3., generator=None, rng=rng).astype(np.int32, copy=False)

#                 epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
#                 last_EPSP = epsps
#                 rows_block[bi, :] = epsps[hL:hL + T].astype(np.float16, copy=False)

#                 if not optimization_time:
#                     warped_array[cell, :] = w16  # store warped for debug/viz

#             # matmul for this block: (n_EC_block × n_dendrites)^T @ (n_EC_block × T)
#             # weights slice is f32; rows upcast to f32 for compute
#             W_blk = weights_EC[start:end, :]                           # (B, n_dendrites) f32
#             accum[:, :T] += (W_blk.T @ rows_block.astype(np.float32))  # (n_dendrites, T)

#         # store this trial’s result in f16
#         dend_vm_padded[t, :, :T] = accum[:, :T].astype(np.float16, copy=False)

#         if not optimization_time:
#             warped_list.append(warped_array)

#     # final tensor
#     dend_contribution_EC = dend_vm_padded  # (trials, dendrites, T_max) f16

#     # Vm transform: compute f32, store f16
#     Vm_dict = {}
#     for d in range(dend_contribution_EC.shape[1]):
#         vm_array = dend_contribution_EC[:, d, :].astype(np.float32, copy=False)
#         Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
#         Vm_dict[d] = Vm.astype(np.float16, copy=False)
#     dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float16, copy=False)

#     if not optimization_time:
#         return dend_activity, weights_EC, last_EPSP, warped_list
#     else:
#         return dend_activity, weights_EC, last_EPSP



############################################################################################################# good version 



def turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=None, n_dendrites=100, rng=None, debug=False, optimization_time=None, mean=None, std=None):

    # if optimization_time:
    assert dt_constant is not None
    dt_ms = float(dt_constant * 1000.0)

    EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
    n_EC, n_pos, n_trials = EC_input_matrix.shape

    dx = np.float32(180.0 / n_pos)

    weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std).astype(np.float32, copy=False)

    kernel = np.asarray(kernel, dtype=np.float32)

    L_prev = 1200  
    last_EPSP = None
    dend_vm_list = []  
    if debug:
        proc_peak_start = _get_peak_rss_bytes()
    
    warped_list = []

    for t in range(n_trials):
        if debug:
            trial_wall_start = time.perf_counter()
            trial_peak_before = _get_peak_rss_bytes()
            trial_curr_before = _get_current_rss_bytes()


        v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)  # (n_pos,)
        dt_s   = dx / v_cm_s                                                                 # (n_pos,)
        time_points = np.cumsum(dt_s, dtype=np.float32)                                      # len n_pos
        total_time  = float(time_points[-1])
        t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)           # (T,)
        T = t_axis.size

        rows = np.empty((n_EC, T), dtype=np.float32)
        firing0 = EC_input_matrix[0, :, t].astype(np.float32, copy=False)
        valid0  = np.isfinite(firing0)
        warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
                if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
        warped0 = warped0.astype(np.float32, copy=False)
        holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

        two_len = len(holder) + T + len(holder)
        two_track_length = np.empty(two_len, dtype=np.float32)
        t_idx = np.arange(two_len, dtype=np.int32)

        if not optimization_time:
            warped_array = np.empty((n_EC, T), dtype=np.float32)

        for cell in range(n_EC):
            firing = EC_input_matrix[cell, :, t].astype(np.float32, copy=False)
            valid  = np.isfinite(firing)
            if valid.sum() >= 2:
                warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
            else:
                warped = np.full(1, np.nan, dtype=np.float32)

            hL = len(holder)
            two_track_length[:hL] = holder
            two_track_length[hL:hL + warped.size] = warped
            two_track_length[hL + warped.size:] = holder

            holder = warped[-L_prev:] if warped.size >= L_prev else warped

            spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_idx, dt=dt_ms, refractory=3., generator=rng).astype(np.int32, copy=False)

            epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
            epsps = epsps[hL:hL + T]  

            rows[cell, :] = epsps.astype(np.float32, copy=False)
            if not optimization_time:
                warped_array[cell, :] = warped

            last_EPSP = epsps  

        dend_vm_over_time = (weights_EC.astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T)

        dend_vm_list.append(dend_vm_over_time)

        if (t & 7) == 7:
            import gc; gc.collect()

        if not optimization_time:
            warped_list.append(warped_array.astype(np.float32, copy=False))

        if debug:
            trial_wall_end = time.perf_counter()
            trial_peak_after = _get_peak_rss_bytes()
            trial_curr_after = _get_current_rss_bytes()
            trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
                else (trial_peak_after - trial_peak_before)
            proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
                else (trial_peak_after - proc_peak_start)
            print(
                f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
                f"RSS now={_fmt_bytes(trial_curr_after)}  "
                f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
                f"proc peak={_fmt_bytes(trial_peak_after)}  "
                f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
            )

        del rows, two_track_length, warped0, warped, firing, firing0

    num_trials = len(dend_vm_list)
    n_dendrites_ = dend_vm_list[0].shape[0]
    max_T = max(arr.shape[1] for arr in dend_vm_list)

    dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
    for i, arr in enumerate(dend_vm_list):
        Ti = arr.shape[1]
        dend_vm_padded[i, :, :Ti] = arr

    dend_contribution_EC = dend_vm_padded  # (trials, dendrites, time) f16

    if dend_contribution_EC.shape[1] != 100:
        print("dend_contribution_EC dendrites in the wrong axis")

    Vm_dict = {}
    for d in range(dend_contribution_EC.shape[1]):
        vm_array = dend_contribution_EC[:, d, :].astype(np.float32, copy=False)
        Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
        Vm_dict[d] = Vm.astype(np.float32, copy=False)

    dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
    
    return dend_activity, weights_EC, last_EPSP, warped_list


##############################################################################################################





def turn_rates_into_spikes_add_inh(dend_list_EC, dend_list_NDNF, dend_list_SST, an_velocity, dist, kernel, SST_sf_opt, NDNF_sf_opt, dt_constant=None, n_dendrites=100, rng=None, debug=False, optimization_time=None, mean=None, std=None):

    # if optimization_time:
    assert dt_constant is not None
    dt_ms = float(dt_constant * 1000.0)

    EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
    n_EC, n_pos, n_trials = EC_input_matrix.shape


    SST_input_matrix = np.stack(dend_list_SST, axis=0).astype(np.float32, copy=False)
    n_SST, n_pos, n_trials = SST_input_matrix.shape


    NDNF_input_matrix = np.stack(dend_list_NDNF, axis=0).astype(np.float32, copy=False)
    n_NDNF, n_pos, n_trials = NDNF_input_matrix.shape

    input_pop_list = [EC_input_matrix, SST_input_matrix, NDNF_input_matrix]

    dx = np.float32(180.0 / n_pos)

    weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std).astype(np.float32, copy=False)

    weights_SST = sample_weights("Equal", n_SST, n_dendrites, rng=rng, mean=SST_sf_opt).astype(np.float32, copy=False)

    weights_NDNF = sample_weights("Equal", n_NDNF, n_dendrites, rng=rng, mean=NDNF_sf_opt).astype(np.float32, copy=False)

    weights_list = [weights_EC, weights_SST, weights_NDNF]


    # W64 = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std)  # float64 by default
    # print("pre-cast finite?", np.isfinite(W64).all(), "max:", W64.max())

    # W16 = W64.astype(np.float16, copy=False)
    # print("post-cast finite?", np.isfinite(W16).all(), "num inf:", np.isinf(W16).sum())

    # kernel used by epsp add — keep as f16
    kernel = np.asarray(kernel, dtype=np.float32)

    L_prev = 1200  # history prepend/append
    last_EPSP = None
   

    if debug:
        proc_peak_start = _get_peak_rss_bytes()
    
    dend_vm_dict = {}

    label_list = ["EC", "SST", "NDNF"]


    for idx, input_population in enumerate(input_pop_list):

        label = label_list[idx]

        if debug:
            trial_wall_start = time.perf_counter()
            trial_peak_before = _get_peak_rss_bytes()
            trial_curr_before = _get_current_rss_bytes()

        dend_vm_list = []  # per-trial, shape (n_dendrites, T_t) in f16

        warped_list = []


        for t in range(n_trials):

            v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)  # (n_pos,)
            dt_s   = dx / v_cm_s                                                                 # (n_pos,)
            time_points = np.cumsum(dt_s, dtype=np.float32)                                      # len n_pos
            total_time  = float(time_points[-1])
            t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)           # (T,)
            T = t_axis.size
        
            # rows = np.empty((n_EC, T), dtype=np.float32)
            rows = np.zeros((input_population.shape[0], T), dtype=np.float32)

            firing0 = input_population[0, :, t].astype(np.float32, copy=False)
            valid0  = np.isfinite(firing0)
            warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
                    if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
            warped0 = warped0.astype(np.float32, copy=False)
            holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

            # total two-track buffer in f16
            two_len = len(holder) + T + len(holder)
            two_track_length = np.empty(two_len, dtype=np.float32)
            t_idx = np.arange(two_len, dtype=np.int32)

            if not optimization_time:
                warped_array = np.empty((input_population.shape[0], T), dtype=np.float32)

            for cell in range(input_population.shape[0]):
                firing = input_population[cell, :, t].astype(np.float32, copy=False)
                valid  = np.isfinite(firing)
                if valid.sum() >= 2:
                    warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                else:
                    warped = np.full(1, np.nan, dtype=np.float32)

                hL = len(holder)
                two_track_length[:hL] = holder
                two_track_length[hL:hL + warped.size] = warped
                two_track_length[hL + warped.size:] = holder

                holder = warped[-L_prev:] if warped.size >= L_prev else warped

                spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_idx, dt=dt_ms, refractory=3., generator=rng).astype(np.int32, copy=False)

                epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
                epsps = epsps[hL:hL + T]  # crop back to the middle warped region (length T)

                # rows[cell, :] = epsps.astype(np.float32, copy=False)
                if label in ("SST", "NDNF"):
                    epsps *= -1.0

                rows[cell, :] += epsps.astype(np.float32, copy=False)
                if not optimization_time:
                    warped_array[cell, :] = warped

                last_EPSP = epsps  

            # ----- dendritic VM = W^T @ rows ; do compute in f32 then downcast -----
            # weights_EC: (n_EC, n_dendrites), rows: (n_EC, T)
            # dend_vm_over_time = (weights_EC.astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T) #### dendrite contribution

            dend_vm_contribution = (weights_list[idx].astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T) #### dendrite contribution

            dend_vm_list.append(dend_vm_contribution)
            
            if (t & 7) == 7:
                import gc; gc.collect()

            if not optimization_time:
                warped_list.append(warped_array.astype(np.float32, copy=False))

            if debug:
                trial_wall_end = time.perf_counter()
                trial_peak_after = _get_peak_rss_bytes()
                trial_curr_after = _get_current_rss_bytes()
                trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
                    else (trial_peak_after - trial_peak_before)
                proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
                    else (trial_peak_after - proc_peak_start)
                print(
                    f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
                    f"RSS now={_fmt_bytes(trial_curr_after)}  "
                    f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
                    f"proc peak={_fmt_bytes(trial_peak_after)}  "
                    f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
                )

            # free large temporaries as we go
            del rows, two_track_length, warped0, warped, firing, firing0

    # ----- optional padding (still f16) -----
        num_trials = len(dend_vm_list)
        n_dendrites_ = dend_vm_list[0].shape[0]
        max_T = max(arr.shape[1] for arr in dend_vm_list)

        dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
        for i, arr in enumerate(dend_vm_list):
            Ti = arr.shape[1]
            dend_vm_padded[i, :, :Ti] = arr

        dend_contribution = dend_vm_padded  # (trials, dendrites, time) f16

        dend_vm_dict[label] = dend_contribution
        warped_dict[label] = warped_list

    overall_dend_contribution = dend_vm_dict["EC"] + dend_vm_dict["SST"] + dend_vm_dict["NDNF"]

    if dend_contribution.shape[1] != 100:
        print("dend_contribution_EC dendrites in the wrong axis")

    # ----- Vm transform (compute f32; store f16) -----
    Vm_dict = {}
    for d in range(overall_dend_contribution.shape[1]):
        vm_array = overall_dend_contribution[:, d, :].astype(np.float32, copy=False)
        Vm, _, _ = activity_to_dend_vm_2d(
            vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
        Vm_dict[d] = Vm.astype(np.float32, copy=False)

    dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)

    if not optimization_time:
        return dend_activity, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_dict
    else:
        return dend_activity, weights_EC, last_EPSP

    # else:

    #     assert dt_constant is not None
    #     dt_ms = float(dt_constant * 1000.0)

    #     EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float16, copy=False)
    #     n_EC, n_pos, n_trials = EC_input_matrix.shape
    #     dx = np.float32(180.0 / n_pos)

    #     weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std).astype(np.float16, copy=False)

    #     # kernel used by epsp add — keep as f16
    #     kernel = np.asarray(kernel, dtype=np.float16)

    #     L_prev = 1200  # history prepend/append
    #     last_EPSP = None
    #     dend_vm_list = []  # per-trial, shape (n_dendrites, T_t) in f16

    #     if debug:
    #         proc_peak_start = _get_peak_rss_bytes()

    #     warped_list = []

    #     for t in range(n_trials):
    #         if debug:
    #             trial_wall_start = time.perf_counter()
    #             trial_peak_before = _get_peak_rss_bytes()
    #             trial_curr_before = _get_current_rss_bytes()

    #         # ----- build time axis (f32 compute, then cast) -----
    #         v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)  # (n_pos,)
    #         dt_s   = dx / v_cm_s                                                                 # (n_pos,)
    #         time_points = np.cumsum(dt_s, dtype=np.float32)                                      # len n_pos
    #         total_time  = float(time_points[-1])
    #         t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32)           # (T,)
    #         T = t_axis.size

    #         # Prealloc buffers (f16 storage)
    #         rows = np.empty((n_EC, T), dtype=np.float16)
    #         firing0 = EC_input_matrix[0, :, t].astype(np.float32, copy=False)
    #         valid0  = np.isfinite(firing0)
    #         warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
    #                 if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
    #         warped0 = warped0.astype(np.float16, copy=False)
    #         holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

    #         # total two-track buffer in f16
    #         two_len = len(holder) + T + len(holder)
    #         two_track_length = np.empty(two_len, dtype=np.float16)
    #         t_idx = np.arange(two_len, dtype=np.int32)

    #         warped_array = np.empty((n_EC, T), dtype=np.float16)
            

    #         for cell in range(n_EC):
    #             firing = EC_input_matrix[cell, :, t].astype(np.float32, copy=False)
    #             valid  = np.isfinite(firing)
    #             if valid.sum() >= 2:
    #                 warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float16, copy=False)
    #             else:
    #                 warped = np.full(1, np.nan, dtype=np.float16)

    #             # stitch [holder | warped | holder] into two_track_length
    #             hL = len(holder)
    #             two_track_length[:hL] = holder
    #             two_track_length[hL:hL + warped.size] = warped
    #             two_track_length[hL + warped.size:] = holder

    #             # update holder for next cell
    #             holder = warped[-L_prev:] if warped.size >= L_prev else warped

    #             # spikes + EPSPs
    #             spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_idx, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(np.int32, copy=False)

    #             epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float16, copy=False)
    #             epsps = epsps[hL:hL + T]  # crop back to the middle warped region (length T)

    #             rows[cell, :] = epsps
    #             warped_array[cell, :] = warped

    #             last_EPSP = epsps  

    #         # ----- dendritic VM = W^T @ rows ; do compute in f32 then downcast -----
    #         # weights_EC: (n_EC, n_dendrites), rows: (n_EC, T)
    #         dend_vm_over_time = (weights_EC.astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float16, copy=False)  # (n_dendrites, T)

    #         dend_vm_list.append(dend_vm_over_time)
    #         warped_list.append(warped_array)

    #         if debug:
    #             trial_wall_end = time.perf_counter()
    #             trial_peak_after = _get_peak_rss_bytes()
    #             trial_curr_after = _get_current_rss_bytes()
    #             trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
    #                 else (trial_peak_after - trial_peak_before)
    #             proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
    #                 else (trial_peak_after - proc_peak_start)
    #             print(
    #                 f"trial {t:02d}:  time={trial_wall_end - trial_wall_start:6.3f}s  "
    #                 f"RSS now={_fmt_bytes(trial_curr_after)}  "
    #                 f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
    #                 f"proc peak={_fmt_bytes(trial_peak_after)}  "
    #                 f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
    #             )

    #         # free large temporaries as we go
    #         del rows, two_track_length, warped0, warped, firing, firing0

    #     # ----- optional padding (still f16) -----
    #     num_trials = len(dend_vm_list)
    #     n_dendrites_ = dend_vm_list[0].shape[0]
    #     max_T = max(arr.shape[1] for arr in dend_vm_list)

    #     dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float16)
    #     for i, arr in enumerate(dend_vm_list):
    #         Ti = arr.shape[1]
    #         dend_vm_padded[i, :, :Ti] = arr

    #     dend_contribution_EC = dend_vm_padded  # (trials, dendrites, time) f16

    #     if dend_contribution_EC.shape[1] != 100:
    #         print("dend_contribution_EC dendrites in the wrong axis")

    #     # ----- Vm transform (compute f32; store f16) -----
    #     Vm_dict = {}
    #     for d in range(dend_contribution_EC.shape[1]):
    #         vm_array = dend_contribution_EC[:, d, :].astype(np.float32, copy=False)
    #         Vm, _, _ = activity_to_dend_vm_2d(
    #             vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
    #         Vm_dict[d] = Vm.astype(np.float16, copy=False)

    #     dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float16, copy=False)

    #     return dend_activity, weights_EC, last_EPSP, warped_list
    




# def turn_rates_into_spikes(dend_list_EC, an_velocity, dist, kernel, dt_constant=None, n_dendrites=100,rng=None, debug=False):

#     print(f"dt_constant {dt_constant}")

#     dt_ms = dt_constant*1000.

#     EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float16, copy=False)

#     # for i in range(EC_input_matrix.shape[0]):
#     #     EC_input_matrix[i]

#     # SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
#     # NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

#     n_EC = EC_input_matrix.shape[0]
#     n_pos = EC_input_matrix.shape[1]
#     n_trials = EC_input_matrix.shape[2]

#     dx = np.float32(180.0 / n_pos)

#     # vel = an_velocity
#     # if vel.shape != (n_pos, n_trials):
#     #     raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

#     # if constant_vel:
#     #     an_velocity = np.full((an_velocity.shape), 43.)

#     L_prev = 1200

#     # precomputed_prepend = np.zeros(L_prev)
#     weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng).astype(np.float16, copy=False)
#     max_length = 0
#     dend_vm_list = []  

#     kernel = np.asarray(kernel, dtype=np.float16)

#     if debug:
#         proc_peak_start = _get_peak_rss_bytes()   # peak at loop start

#     warped_list = []

#     last_EPSP = None

    

#     min_trial_length = 1000000
    

#     for t in range(EC_input_matrix.shape[2]):
        

#         if debug:
#             trial_wall_start = time.perf_counter()
#             trial_peak_before = _get_peak_rss_bytes()
#             trial_curr_before = _get_current_rss_bytes()

#         start_time = time.time()
#         v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)  # (n_pos,)
#         dt_s   = dx / v_cm_s                             
#         edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
#         total_time = float(edges_s[-1])


#         # constant-time axis in seconds
#         t_axis = np.arange(0.0, total_time, float(dt_constant), dtype=np.float32) 
        

#         max_time_over_cells = 0
#         T = t_axis.size

#         # Prealloc buffers (f16 storage)
#         rows = np.empty((n_EC, T), dtype=np.float16)
#         # we’ll reuse holder as f16
#         # first cell’s warped determines initial holder
#         firing0 = EC_input_matrix[0, :, t].astype(np.float32, copy=False)
#         valid0  = np.isfinite(firing0)
#         warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
#                    if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
#         warped0 = warped0.astype(np.float16, copy=False)
#         holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

#         # total two-track buffer in f16
#         two_len = len(holder) + T + len(holder)
#         two_track_length = np.empty(two_len, dtype=np.float16)
#         # time index for spike gen: plain indices; dt passed separately
#         t_idx = np.arange(two_len, dtype=np.int32)

#         firing_example = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
#         valid = np.isfinite(firing_example)
#         time_points = np.cumsum(dt_s)
        
#         firing_zero = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
#         warped_zero = np.interp(t_axis, time_points[valid], firing_zero[valid]).astype(np.float32, copy=False)
#         #min_trial_length = warped_zero.shape[0]

#         holder = warped_zero[-L_prev:]
#         # ender = warped_zero[-L_prev:]

#         # warped_rows = [] 

#         rows = np.empty((n_EC, T), dtype=np.float32)   

#         two_track_length = np.empty(len(holder) + len(warped_zero) + len(holder))
#         # two_track_length = np.empty(len(holder) + len(warped_zero))

#         # two_track_length_list = []
#         # EPSP_LIST = []

#         t_ms = np.arange(two_track_length.size, dtype=int) * dt_ms

        


#         for cell in range(n_EC):
#             firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
#             valid = np.isfinite(firing)
#             if valid.sum() >= 2:
#                 time_points = np.cumsum(dt_s)  # length n_pos

#                 warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                
#             else:
#                 warped = np.full(1, np.nan, dtype=np.float32)

#             # warped_rows.append(warped)

#             two_track_length[:len(holder)] = holder
#             two_track_length[-len(holder):] = holder
#             two_track_length[len(holder):len(holder)+len(warped)] = warped

#             holder = warped[-L_prev:]


#             spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 

#             epsps = epsps_event_add(spike_times, two_track_length.shape[0], kernel).astype(np.float16, copy=False)
#             epsps = epsps[L_prev:L_prev+warped.size]


#             last_EPSP = epsps

            
#             if len(epsps) > max_time_over_cells:
#                 max_time_over_cells = len(epsps)
#             rows[cell,:] = epsps

#         # epsp_array = np.array(EPSP_LIST)
#         # plt.plot(np.mean(epsp_array, axis=0))
#         # plt.title(f"trial {t}")
#         # plt.show()

#         # two_track_length_array = np.array(two_track_length_list)
#         # plt.plot(np.mean(two_track_length_array, axis=0))
#         # plt.title("mean two track length")
#         # plt.show()

        
        
#         #warped_rows_array = np.array(warped_rows)
#         #warped_list.append(warped_rows_array) 
        


#         # X_trial = np.stack(rows, axis=0).astype(np.float16, copy=False)

  

#         dendy_list = []

#         for dend in range(n_dendrites):
#             w = weights_EC[:, dend]            # shape (n_EC,)
#             vm_t = w @ rows 
#             dendy_list.append(vm_t)
            
#         dend_vm_over_time = np.array(dendy_list, dtype=np.float16)


#         # dend_vm_over_time = weights_EC @ X_trial

#         if len(dend_vm_over_time) > max_length:
#             max_length = len(dend_vm_over_time)

#         dend_vm_list.append(dend_vm_over_time)



#     num_trials   = len(dend_vm_list)
#     n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


#     print(f"num_trials{num_trials} n_dendrites_ {n_dendrites_}")

#     num_trials = len(dend_vm_list)
#     n_dendrites_ = dend_vm_list[0].shape[0]
#     max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

#     print(f"num_trials={num_trials}  n_dendrites_={n_dendrites_}  max_T={max_T}")

#     dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

#     for i, arr in enumerate(dend_vm_list):
#         if arr.shape[0] != n_dendrites_:
#             raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
#         Ti = arr.shape[1]               # current trial's time length
#         dend_vm_padded[i, :, :Ti] = arr # pad along time axis

#     dend_contribution_EC = dend_vm_padded

#     if dend_contribution_EC.shape[1] != 100:
#         print("dend_contribution_EC dendrites in the wrong axis")



#     Vm_dict = {}
#     for dend in range(dend_contribution_EC.shape[1]):
#         vm_array_shaped = dend_contribution_EC[:, dend, :]
#         Vm, _, _ = activity_to_dend_vm_2d(
#             vm_array_shaped,
#             Vrest=-70.0,
#             vm_scale=0.1,
#             center_across="time")
#         Vm_dict[dend] = Vm

#     vm_list = []
#     for trial in Vm_dict:
#         vm_list.append(Vm_dict[trial])

#     dend_activity = np.array(vm_list)



    
#     if debug:
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

#     del rows, dend_vm_over_time, epsps, 

#     return dend_activity, weights_EC, last_EPSP #, min_trial_length #, warped_list

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

import pickle
from types import SimpleNamespace

class DummySimConfig(SimpleNamespace): pass
class DummyStoreFlags(SimpleNamespace): pass

class DummyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "SimConfig":
            return DummySimConfig
        if module == "__main__" and name == "StoreFlags":
            return DummyStoreFlags
        return super().find_class(module, name)


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
        kernel: np.ndarray,               # EPSP kernel (samples)
        dist_for_weights,                 # callable or string label
        weights_SST=None, weights_NDNF=None,
        config: Optional[SimConfig]=None,
        flags: Optional[StoreFlags]=None,):
        self.kernel = np.asarray(kernel, dtype=np.float32, order="C")
        self.W_dist = dist_for_weights
        self.W_dist_name = dist_for_weights if isinstance(dist_for_weights, str) else getattr(dist_for_weights, "__name__", "weights")
        self.W_SST = weights_SST
        self.W_NDNF = weights_NDNF


        self.cfg = config or SimConfig()
        # if self.cfg.dx is None:
        #     self.cfg.dx = 180.0 / self.n_pos
        self.flags = flags or StoreFlags()

        self._precomputed = None
        self._results: Dict[int, Dict[str, Any]] = {}
        self.tau_ms: Optional[float] = None
        self.dend_threshold: Optional[float] = None
        self.dist: str = self.W_dist_name
        self.seeds: List[int] = []
        self._pos_cnt_dict = {}
        self.start_pos_cnt50_dict = {}
        self._plateau_arr_list_dict = {}
        self._mask_dict = {}
        self._starts_list_dict = {}
        self.num_plateaus_per_dend_list_dict = {}
        self.dend_activity_dict = {}
        self.padded_warped_activity_list_dict = {}


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

    def evaluate(self, christine_overrepresentation_array=None, seeds=None,*,dend_threshold: float,include_inhibition: bool = True,vel_applied: str = "real",example_cell: int = 15,target_total: float = 120.0,target_frac: np.ndarray = None) -> Tuple[float, Dict[str, float]]:

        start_pos_cnt50_dict = self.start_pos_cnt50_dict

        plateau_list = self._plateau_arr_list_dict

        first_seed = seeds[0]

        n_dendrites = len(plateau_list[first_seed])

        dendrites_with_plateau_count = 0
        total_dends = 0

        

        for seed in range(len(plateau_list)):
            for dendrite in range(len(plateau_list[seed])):
                dendrite_plateau_array = plateau_list[seed][dendrite]
                
                if np.any(dendrite_plateau_array==1):
                    dendrites_with_plateau_count+=1

                total_dends +=1

        frac_dends_with_plateau = dendrites_with_plateau_count / total_dends


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


        arr = np.asarray(summed_plateaus_over_seeds)  # expect (n_seeds, n_bins)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape}")

        # If bins are on axis 0, transpose to (n_seeds, n_bins)
        if arr.shape[0] in (10, 50) and arr.shape[1] not in (10, 50):
            arr = arr.T

        mean_over_seeds = np.mean(arr, axis=0)


        mean_over_seeds = np.mean(arr, axis=0)

        totals = arr.sum(axis=1)  # shape (n_seeds,)
        valid = totals > 0        # seeds that actually have any plateau events

        if not np.any(valid):
            p_model_allcells = np.zeros_like(christine_overrepresentation_array, dtype=float)
        else:
            frac = np.empty_like(arr, dtype=float)
            frac[:] = np.nan
            frac[valid] = arr[valid] / totals[valid, None]   # rows sum to 1 for valid seeds

            p_model = np.nanmean(frac, axis=0)               # (10,)
            p_model = np.nan_to_num(p_model, nan=0.0)        # guard rare all-NaN columns
            p_model_allcells = p_model * frac_dends_with_plateau

        p_chr = christine_overrepresentation_array / 100.0
        p_chr_allcells = p_chr * 0.25


        loss = np.mean((p_model_allcells - p_chr_allcells)**2)
        return loss



    def plot_from_pickle(path, *, plot_fn, animal="animal-1", **extra):
        P = SpikeSimModel.load_pickle(path)
        seeds = P["seeds"]
        results = P["results"]

        dend_vm_per_seed_dict = {int(s): results[str(s)]["Vm"] for s in results}
        last_seed = seeds[0]
        last = results[str(last_seed)]
        last_EPSP = last.get("last_epsp_example", np.zeros((1,1,1000), np.float32))
        weights_EC = last.get("weights_EC", None)

        return plot_fn(tau_ms=P.get("tau_ms", 0.0),
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
            animal_by_animal=True,)



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
        self._precomputed = True

    def _simulate_one_seed(self, seed: int, debug=False):
        
        if debug:
            import psutil, os
            print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss/1e9, flush=True) 

        activity_NDNF = 0
        activity_SST  = 0

        if debug:

            rank = MPI.COMM_WORLD.Get_rank()
            pid = os.getpid()

            t0 = time.time()
            print(f"[{t0:.3f}] rank={rank} pid={pid} START seed={seed}", flush=True)
            sys.stdout.flush()
            report_mem("pre get_dend_contribution")

        (an_velocity, dend_activity, NDNF_pop_list, SST_pop_list, NDNF_contribution_sum, SST_contribution_sum,
         weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_list) = (
            get_dend_contribution(self.kernel, self.dt_constant, self.residual_activity_dict_EC,
                                  self.fixed_residual_activity_dict_NDNF_newest,self.residual_activity_dict_SST,
                                  self.factors_dict_EC, self.factors_dict_SST, self.factors_dict_NDNF_newest,
                                  self.GLM_params_EC, self.GLM_params_NDNF_newest, self.GLM_params_SST,
                                  self.NDNF_sf_opt, self.SST_sf_opt, include_inh=self.include_inh,
                                  multiple_dendrites=True, dist=self.dist, make_it_spike=self.make_it_spike, seed=seed,
                                  animal_by_animal=self.animal_by_animal, input_animal=self.input_animal,
                                  constant_vel=self.constant_vel, include_beta=self.include_beta,
                                  flat_input=self.flat_input, optimization_time=self.optimization_time, debug=debug,
                                  mean=self.mean, std=self.std))
        
        if debug:
            report_mem("post get_dend_contribution")

            nbytes(weights_EC, "weights_EC")
            nbytes(weights_SST, "weights_SST")
            nbytes(weights_NDNF, "weights_NDNF")

        if debug:
            report_mem("pre get_activity_multidendrite")

        (padded_warped_activity_list, plateau_positions_counter,
        plateau_start_positions_counter, plateau_array_per_dendrite_list,
        dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list,
        EC_used, num_plateaus_per_dend_list) = (
            get_activity_multidendrite(an_velocity, dend_activity,dend_threshold=self.dend_threshold, example_cell=17,
                                       n_dendrites=100, make_it_spike=self.make_it_spike))
        
        if debug:
            report_mem("post get_activity_multidendrite")

        return (dend_activity, plateau_positions_counter, padded_warped_activity_list,
                    plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,
                    num_plateaus_per_dend_list, plateau_start_times_list_mega_list, last_EPSP,
                    weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST, activity_NDNF, warped_list)
        
            
    def simulate(self, seeds, export=False, plot=False, debug=False):
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
        dend_contribution_EC_dict = {}
        dend_activity_dict = {}
        warped_list_dict = {}


        for seed in seeds:

            if self.optimization_time:
                (dend_activity, plateau_positions_counter, padded_warped_activity_list,
                plateau_start_positions_counter, plateau_array_per_dendrite_list,
                dendrite_plateau_mask, num_plateaus_per_dend_list, plateau_start_times_list_mega_list,
                last_EPSP, weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST,
                activity_NDNF) = self._simulate_one_seed(int(seed), debug=debug)
            else:
                (dend_activity, plateau_positions_counter, padded_warped_activity_list,
                plateau_start_positions_counter, plateau_array_per_dendrite_list,
                dendrite_plateau_mask, num_plateaus_per_dend_list, plateau_start_times_list_mega_list,
                last_EPSP, weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST,
                activity_NDNF, warped_list) = self._simulate_one_seed(int(seed), debug=debug)


            #min_trial_length

            if self.optimization_time:

                # dend_vm_per_seed_dict[seed] = dend_vm
                # _pos_cnt_dict[seed] = plateau_positions_counter
                # padded_warped_activity_list_dict[seed] = padded_warped_activity_list
                start_pos_cnt50_dict[seed] = plateau_start_positions_counter
                _plateau_arr_list_dict[seed] = plateau_array_per_dendrite_list
                # _mask_dict[seed] = dendrite_plateau_mask
                # num_plateaus_per_dend_dict[seed] = num_plateaus_per_dend_list
                # _starts_list_dict[seed] = plateau_start_times_list_mega_list
                # last_EPSP_dict[seed] = last_EPSP
                # weights_EC_dict[seed] = weights_EC
                # weights_SST_dict[seed] = weights_SST
                # weights_NDNF_dict[seed] = weights_NDNF
                # an_velocity_dict[seed] = an_velocity
                # activity_SST_dict[seed] = activity_SST
                # activity_NDNF_dict[seed] = activity_NDNF
                # SST_sf_opt_dict[seed] = SST_sf_opt
                # NDNF_sf_opt_dict[seed] = NDNF_sf_opt
                # dend_contribution_EC_dict[seed] = dend_contribution_EC

            else:
                # dend_vm_per_seed_dict[seed] = dend_vm
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
                # dend_contribution_EC_dict[seed] = dend_contribution_EC
                dend_activity_dict[seed] = dend_activity
                warped_list_dict[seed] = warped_list


        if self.optimization_time:

            self.start_pos_cnt50_dict = start_pos_cnt50_dict
            self._plateau_arr_list_dict = _plateau_arr_list_dict


        else:

            
            
            
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
                warped_list_dict = warped_list_dict
            )

            # cache for plotting convenience
            self.dend_vm_per_seed_dict=dend_vm_per_seed_dict
            self.padded_warped_activity_list_dict = padded_warped_activity_list_dict
            self._pos_cnt_dict = _pos_cnt_dict
            self.start_pos_cnt50_dict = start_pos_cnt50_dict
            self._plateau_arr_list_dict = _plateau_arr_list_dict
            self._mask_dict = _mask_dict
            self._starts_list_dict = _starts_list_dict
            self.num_plateaus_per_dend_list_dict = num_plateaus_per_dend_dict
            self.last_EPSP_dict = last_EPSP_dict
            self.weights_EC_dict = weights_EC_dict
            self.weights_SST_dict = weights_SST_dict
            self.weights_NDNF_dict = weights_NDNF_dict
            self.dend_contribution_EC_dict = dend_contribution_EC_dict
            self.an_velocity_dict = an_velocity_dict
            self.activity_SST_dict = activity_SST_dict
            self.activity_NDNF_dict = activity_NDNF_dict
            self.SST_sf_opt_dict = SST_sf_opt_dict
            self.NDNF_sf_opt_dict = NDNF_sf_opt_dict
            # self.min_trial_length = min_trial_length
            # self.dend_list_EC_interp = dend_list_EC_interp
            self.dend_activity_dict = dend_activity_dict
            self.warped_list_dict = warped_list_dict



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

    dt_constant = 0.0001
    tau_ms  = 5.0
    dt_ms   = dt_constant * 1000.0
    AMP     = 1.0
    MODE    = "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

    use_averaged_velocity = "actual_velocity" #"cell_type_av" 
    dist = "Lognormal"
    add_inh = 'neither'
    make_it_spike = True
    SST_bias_multi = 1.4

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
    model.mean_new_average_vel_array = None 
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
    model.animal_by_animal = True
    model.input_animal = "animal_1"



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

    with open(pickle_path, "rb") as f:
        state = DummyUnpickler(f).load()

    # Optional: normalize for downstream code
    if isinstance(state.get("cfg"), (DummySimConfig, SimpleNamespace)):
        state["cfg"] = vars(state["cfg"])
    if isinstance(state.get("flags"), (DummyStoreFlags, SimpleNamespace)):
        state["flags"] = vars(state["flags"])


    



    # model = _load_light_model(pickle_path)
    # if model.dend_threshold is None:
    #     raise click.UsageError("dend_threshold not found in pickle. Provide it at simulate time.")
    print(state.keys())
    print(f'state["flags"] {state["flags"]}')
    print(f'state["cfg"] {state["cfg"]}')

    plot_multidendrite_EC_err_across_seeds(tau_ms = state['tau_ms'],
    seeds = state["seeds"], last_EPSP = state["last_EPSP_dict"][0], weights_EC = state["weights_EC_dict"][0], weights_SST = state["weights_SST_dict"][0], weights_NDNF = state["weights_NDNF_dict"][0], dend_vm_per_seed_dict = state["dend_vm_per_seed_dict"],
    activity_EC = state["dend_contribution_EC_dict"][0], activity_SST = state["activity_SST"], activity_NDNF = state["activity_NDNF"], SST_sf_opt = state["SST_sf_opt"], NDNF_sf_opt = state["NDNF_sf_opt"],
    padded_warped_activity_list = state["padded_warped_activity_list_dict"], an_velocity = state["an_velocity_dict"][0], dend_threshold = state["dend_threshold"],
    _pos_cnt_dict = state["_pos_cnt_dict"], start_pos_cnt50_dict = state["start_pos_cnt50_dict"], _plateau_arr_list_dict = state["_plateau_arr_list_dict"], _mask_dict = state["_mask_dict"], _starts_list_dict = state["_starts_list_dict"],
    dist = state["dist"], num_plateaus_per_dend_list = state["num_plateaus_per_dend_list_dict"], animal=state["input_animal"], example_cell=17, include_inhibition=False, #include inhibiiton,
    NDNF_contribution_sum = None, #state["NDNF_contribution_sum"], 
    SST_contribution_sum = None, #state["SST_contribution_sum"], 
    animal_by_animal = state["animal_by_animal"]) #state["animal_by_animal"])

    warped_list_dict


    # tau_ms
    # seeds
    # *** last_epsp
    # *** weights ec 
    # 'W_SST',
    # 'W_NDNF',
    # dend_vm_per_seed_dict
    # dend_activity_dict
    # 'activity_EC', 'activity_SST', 'activity_NDNF', 
    # 'SST_sf_opt', 'NDNF_sf_opt'



    # dict_keys(['kernel', 'W_dist', 'W_dist_name',  'cfg', 'flags', '_precomputed', '_results', 'tau_ms', 'dend_threshold', 'dist', '_pos_cnt_dict', 'start_pos_cnt50_dict', '_plateau_arr_list_dict', '_mask_dict', '_starts_list_dict', 'num_plateaus_per_dend_list_dict', '', 'padded_warped_activity_list_dict', 'residual_activity_dict_EC', 'fixed_residual_activity_dict_NDNF_newest', 'residual_activity_dict_SST', 'factors_dict_EC', 'factors_dict_SST', 'factors_dict_NDNF_newest', 'GLM_params_EC', 'GLM_params_NDNF_newest', 'GLM_params_SST', 'mean_new_average_vel_array', 'real_vel', 'constant_vel', 'add_inh', 'make_it_spike', 'SST_bias_factor', 'vel_applied', 'use_averaged_velocity', 'use_model_EC', 'vel', , 'animal'])

    if out:
        plt.savefig(out, dpi=200, bbox_inches="tight")
        click.echo(f"Saved figure to: {out}")


if __name__ == "__main__":
    cli()