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
from scipy.stats import sem
import numpy.typing as npt
import matplotlib.pyplot as plt
import click

# your project imports (AFTER third-party)
# from Fixing_dend_models_presentation import *
from spiking_model_utils import load_data_regular

from mpi4py import MPI
import os, sys, time, psutil, resource
from time import perf_counter
_PROC = psutil.Process(os.getpid())



def exp_kernel(tau_ms, dt_ms, n_taus=5, norm="peak", target=1.0):
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


def str_true_false_to_bool(s):
    """
    Accepts 'true' or 'false' (any case, with spaces). 
    Returns True/False. Raises ValueError otherwise.
    """
    if isinstance(s, bool):
        return s
    if not isinstance(s, str):
        raise ValueError("Expected a string 'true' or 'false'")
    t = s.strip().lower()
    if t == "true":
        return True
    if t == "false":
        return False
    raise ValueError(f"Unrecognized boolean string: {s!r}")


def sample_weights(distribution, n_EC, n_dendrites, rng, mean=0.1, std=0.5):

    if distribution == "Uniform":
        samples = rng.uniform(low=mean - std, high=mean + std, size=(n_EC, n_dendrites))
    elif distribution == "Normal":
        samples = rng.normal(loc=mean, scale=std, size=(n_EC, n_dendrites))
        samples = np.clip(samples, 0, None)
    elif distribution == "Lognormal":
        # samples = rng.lognormal(mean=np.log(mean), sigma=std, size=n_samples)
        samples = rng.lognormal(mean=mean, sigma=std, size=(n_EC, n_dendrites))
    elif distribution == "Equal":
        samples = np.full((n_EC, n_dendrites), mean, dtype=float)
    else:
        raise ValueError("Invalid distribution")

    return samples


def plot_seed_average(christine_field_building_array, christine_overrepresentation_array_scaled, mean_histogram_array, sem_histogram_array, mean_pc_array, sem_pc_array, loss):

    fig, axs = plt.subplots(1,2, figsize=(10,5))

    def _interp_to(x_src, y_src, x_dst):
        m = np.isfinite(x_src) & np.isfinite(y_src)
        xs = np.asarray(x_src)[m]
        ys = np.asarray(y_src)[m]
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        return np.interp(x_dst, xs, ys, left=ys[0], right=ys[-1])

    Tn = len(mean_pc_array)
    x_model0 = np.linspace(0, 100, Tn)  # common x in [0,100]

    x_exp0_raw = christine_field_building_array[:, 0]
    y_exp0_raw = christine_field_building_array[:, 1]
    x_exp0 = x_exp0_raw * 100.0 if np.nanmax(x_exp0_raw) <= 1.5 else x_exp0_raw
    y_exp0 = y_exp0_raw / 100.0  # if exp is in percent, match model's 0–1

    y_exp0_on_model = _interp_to(x_exp0, y_exp0, x_model0)


    axs[0].set_xticks([0, 25, 50, 75, 100], labels=["0","25","50","75","100"])
    axs[0].fill_between(x_model0, mean_pc_array+sem_pc_array, mean_pc_array-sem_pc_array, alpha=0.2, color='b')
    axs[0].plot(x_model0, mean_pc_array, label='model', color='b')
    axs[0].plot(x_model0, y_exp0_on_model, label='experiment', color='k', linewidth=3)
    axs[0].set_xlabel("Session Length (%)")
    axs[0].set_ylabel("Fraction of CA1 Dendrites (%)")
    axs[0].set_title(f"MSE={np.mean(np.square(y_exp0_on_model-mean_pc_array))}")
    axs[0].set_xlim(0, 100)
    axs[0].legend()

    axs[1].fill_between(range(len(mean_histogram_array)), mean_histogram_array+sem_histogram_array, mean_histogram_array-sem_histogram_array, alpha=0.2, color='b')
    axs[1].set_ylabel("Fraction of CA1 PCs (%)")
    axs[1].plot(mean_histogram_array, label='model', color='b')
    axs[1].plot(christine_overrepresentation_array_scaled, label='experiment', color='k', linewidth=3)
    lin_aray = np.linspace(0, 180, 10)
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels(lin_aray.astype(int))
    axs[1].set_xlabel("PF Peak Location (cm)")
    axs[1].set_title(f"MSE={loss}")
    axs[1].legend()

    plt.show()


def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
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

def _get_peak_rss_bytes():
    # macOS: bytes; Linux: kilobytes → convert to bytes
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru if sys.platform == "darwin" else ru * 1024

def _get_current_rss_bytes():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss

def _mb(nbytes):
    return (nbytes or 0) / (1024.0 * 1024.0)

def log_mem(tag=""):
    cur = _get_current_rss_bytes()
    peak = _get_peak_rss_bytes()
    print(f"[mem]{' '+tag if tag else ''}  now={_mb(cur):.1f} MB   peak={_mb(peak):.1f} MB")


def turn_rates_into_spikes(dend_list_EC, an_velocity, weights_pop_dict, kernel, dt=None, n_dendrites=100, rng=None, debug=False, store_intermediates=None, mean=None, std=None):

    assert dt is not None
    dt_ms = float(dt * 1000.0)

    EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
    n_EC, n_pos, n_trials = EC_input_matrix.shape

    dx = np.float32(180.0 / n_pos)

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
        t_axis = np.arange(0.0, total_time, float(dt), dtype=np.float32)           # (T,)
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

        if not store_intermediates:
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
            if not store_intermediates:
                warped_array[cell, :] = warped

            last_EPSP = epsps  

        dend_vm_over_time = (weights_pop_dict.astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T)

        dend_vm_list.append(dend_vm_over_time)

    if (t & 7) == 7:
        import gc; gc.collect()

    if not store_intermediates:
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
            f"(+{_fmt_bytes(proc_peak_since_start)} since start)")

    del rows, two_track_length, warped0, warped, firing, firing0

    num_trials = len(dend_vm_list)
    n_dendrites_ = dend_vm_list[0].shape[0]
    max_T = max(arr.shape[1] for arr in dend_vm_list)

    dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
    for i, arr in enumerate(dend_vm_list):
        Ti = arr.shape[1]
        dend_vm_padded[i, :, :Ti] = arr

    dend_contribution = dend_vm_padded  # (trials, dendrites, time) f16

    if dend_contribution.shape[1] != 100:
        print("dend_contribution_EC dendrites in the wrong axis")
    
    return dend_contribution, last_EPSP, warped_list


def resample_trace_to_position_bins(trace_time, dt, vel_pos_mps, track_len_m=1.8, n_bins_target=1000):

    vel = np.asarray(vel_pos_mps, float)
    n_pos = vel.size
    dx = float(track_len_m) / n_pos                   
    vel = np.clip(vel, 1e-6, None)                    
    dt_pos = dx / vel                                 
    t_edges = np.concatenate([[0.0], np.cumsum(dt_pos)])       
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])             

    T = trace_time.size
    t_trace = np.arange(T, dtype=float) * dt                    

    pos_src = np.arange(n_pos, dtype=float) + 0.5               
    pos_tgt = (np.arange(n_bins_target, dtype=float) + 0.5) * (n_pos / n_bins_target)

    t_at_pos_tgt = np.interp(pos_tgt, pos_src, t_centers, left=t_centers[0], right=t_centers[-1])

    trace_pos = np.interp(t_at_pos_tgt, t_trace, trace_time,left=trace_time[0], right=trace_time[-1])

    return trace_pos

def activity_to_dend_vm_2d(
    A_trials_time,
    Vrest=-70.0,
    vm_scale=0.1,
    center_across="time",   # "time" | "time_trials" | "none"
    dtype=np.float32):
    """
    A_trials_time: (n_trials, T) activity (may contain NaNs)
    Returns:
      Vm: (n_trials, T) in mV
      A_centered: centered activity (same shape)
      mu: the mean(s) removed (scalar or array)
    """
    A = np.asarray(A_trials_time, dtype=dtype, order="C")
    if A.ndim != 2:
        raise ValueError(f"A must be 2D (trials, time); got {A.shape}")

    if center_across == "time":
        # per-trial time mean -> each trial mean becomes Vrest after scaling/offset
        mu = np.nanmean(A, axis=1, keepdims=True).astype(dtype, copy=False)
    elif center_across == "time_trials":
        # single global mean over all trials and time
        mu = np.nanmean(A, keepdims=True).astype(dtype, copy=False)
    elif center_across == "none":
        mu = dtype(0.0)
    else:
        raise ValueError("center_across must be 'time', 'time_trials', or 'none'")

    A_centered = (A - mu).astype(dtype, copy=False)
    Vm = dtype(Vrest) + dtype(vm_scale) * A_centered
    return Vm, A_centered, (mu if np.isscalar(mu) else mu.astype(dtype, copy=False))

    

def scale_dendrite(dend_contribution_summed):
    Vm_dict = {}
    for d in range(dend_contribution_summed.shape[1]):
        vm_array = dend_contribution_summed[:, d, :].astype(np.float32, copy=False)
        Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
        Vm_dict[d] = Vm.astype(np.float32, copy=False)

    dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
    return dend_activity


def get_padded_dend_from_psps(psps_list):
    num_trials = len(psps_list)
    n_dendrites_ = psps_list[0].shape[0]
    max_T = max(arr.shape[1] for arr in psps_list)

    dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
    for i, arr in enumerate(psps_list):
        Ti = arr.shape[1]
        dend_vm_padded[i, :, :Ti] = arr

    dend_contribution = dend_vm_padded  # (trials, dendrites, time) f16

    if dend_contribution.shape[1] != 100:
        print("dend_contribution_EC dendrites in the wrong axis")
    return(dend_contribution)


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
    multiple_dendrites: bool = True
    make_it_spike: bool = True
    num_dendrites: int = 100


class SpikeSimModel:
    def __init__(self,
                kernel: np.ndarray,          # EPSP kernel (samples)
                weight_config_dict,                 # dict
                dt=0.001,  # sec
                track_len=180.,
                store_intermediates=None,
                plot=False,
                multiple_dendrites=True,
                residuals_activity_dict = None,
                make_it_spike = None,
                GLM_params_dict=None,
                behav_factors_dict=None,
                max_num_trials=58,
                num_pos_bins=50,
                av_animals_velocity=0.43,
                hz_target_for_scaling=50,
                constant_vel=None, 
                include_beta=None,
                flat_input=None,
                dend_threshold=None,
                tau_ms=None,
                sim_config = None,
                debug = False,
                n_cells_dict = None,
                christine_overrep_array=None,
                christine_field_building_array=None,
                GLM_param_dict=None):
        
        self.kernel = kernel
        self.weight_config_dict = weight_config_dict
        self.residuals_activity_dict = residuals_activity_dict
        self.GLM_params_dict = GLM_params_dict
        self.behav_factors_dict = behav_factors_dict
        self.multiple_dendrites = bool(multiple_dendrites)
        self.make_it_spike = make_it_spike
        self._precomputed = None
        self._results: Dict[int, Dict[str, Any]] = {}
        self.tau_ms: Optional[float] = None
        self.dend_threshold: Optional[float] = None
        self._pos_cnt_dict = {}
        self.start_pos_cnt50_dict = {}
        self._plateau_arr_list_dict = {}
        self._mask_dict = {}
        self._starts_list_dict = {}
        self.num_plateaus_per_dend_list_dict = {}
        self.dend_activity_dict = {}
        self.padded_warped_activity_list_dict = {}
        self.tau_ms = tau_ms 
        self.dend_threshold = dend_threshold 
        self.dt = dt
        self.track_len = track_len
        self.max_num_trials = max_num_trials
        self.num_pos_bins = num_pos_bins
        self.dx = track_len/num_pos_bins
        self.av_animals_velocity = av_animals_velocity
        self.hz_target_for_scaling = hz_target_for_scaling
        self.constant_vel=constant_vel
        self.include_beta=include_beta
        self.flat_input=flat_input
        self.num_dendrites = sim_config.num_dendrites
        self.debug = debug
        self.n_cells_dict = n_cells_dict
        self.christines_overrepresentation_array = christine_overrep_array
        self.store_intermediates = store_intermediates
        self.plot = plot
        self.christine_field_building_array = christine_field_building_array
        self.GLM_param_dict = GLM_param_dict


    def modify_activity(self, residuals_activity_dict_pop, behav_factors_dict_pop, GLM_params_dict_pop):

        const_vel = self.constant_vel
        flat_input = self.flat_input
        include_beta = self.include_beta

        data_list_normalized = []

        for animal in residuals_activity_dict_pop:
            for cell in residuals_activity_dict_pop[animal]:
                data_normalized = residuals_activity_dict_pop[animal][cell][:,:self.max_num_trials]
                data_normalized = ((data_normalized - np.min(data_normalized)) / (np.max(data_normalized) - np.min(data_normalized))) *self.hz_target_for_scaling
                data_list_normalized.append(data_normalized)


        mu= np.mean(data_list_normalized)
        sigma = np.std(data_list_normalized)
        
        dendrite_list = []

        animal_velocity_list = []

        for animal in residuals_activity_dict_pop:
            for cell in residuals_activity_dict_pop[animal]:
                data_full = residuals_activity_dict_pop[animal][cell]

                if flat_input:
                    data = np.zeros((self.num_pos_bins,self.max_num_trials))

                    weights = GLM_params_dict_pop[animal][cell]['weights']["Velocity"]

                    if const_vel:
                        animal_velocity = np.full((self.num_pos_bins, self.max_num_trials), self.av_animals_velocity)

                        animal_velocity_list.append(animal_velocity)

                        if include_beta:
                        
                            data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                        else:
                            data = data + mu

                        dendrite_list.append(data)

                    else:
                        animal_velocity = behav_factors_dict_pop[animal]["Velocity"][:,:self.max_num_trials]
                        animal_velocity_list.append(animal_velocity)
                        
                        if include_beta:
                        
                            data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                        else:
                            data = data + mu

                        dendrite_list.append(data)

                else:
                    
                    ########### hand in velocity then min max 50 

                    data = data_full[:,:self.max_num_trials]

                    weights = GLM_params_dict_pop[animal][cell]['weights']["Velocity"]

                    if const_vel:
                        animal_velocity = np.full((self.num_pos_bins, self.max_num_trials), self.av_animals_velocity)

                        animal_velocity_list.append(animal_velocity)

                        if include_beta:
                        
                            data = data + (weights * animal_velocity) #*sigma + intercept

                            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *self.hz_target_for_scaling)

                        else:
                            # data = data + mu
                            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *self.hz_target_for_scaling)

                        dendrite_list.append(data)

                    else:
                        animal_velocity = behav_factors_dict_pop[animal]["Velocity"][:,:self.max_num_trials]
                        animal_velocity_list.append(animal_velocity)
                        
                        if include_beta:
                        
                            data = data + (weights * animal_velocity) #*sigma #+ intercept
                            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *self.hz_target_for_scaling)

                        else:
                            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *self.hz_target_for_scaling)


                        dendrite_list.append(data)

                                    
        an_velocity = np.array(animal_velocity_list)
        an_velocity = np.nanmean(an_velocity, axis=0)

        dendrite_array = np.array(dendrite_list).astype(np.float32)

        return an_velocity, dendrite_array  #, dendrite_ta_list



    def process_inputs_and_velocity(self):

        animal_velocity_list = []

        processed_binned_activity_dict = {}

        for pop in self.weight_config_dict:
            this_animal_velocity, this_processed_binned_activity = self.modify_activity(self.residuals_activity_dict[pop], self.behav_factors_dict[pop], self.GLM_params_dict[pop])
            animal_velocity_list.append(this_animal_velocity)
            processed_binned_activity_dict[pop] = this_processed_binned_activity

        animal_velocity_array = np.array(animal_velocity_list)
        animal_velocity = np.mean(animal_velocity_array, axis=0)
    
        return animal_velocity, processed_binned_activity_dict
    
    def init_weights(self):
        
        weights_pop_dict = {}

        for pop in self.weight_config_dict:
            weights_pop = sample_weights(self.weight_config_dict[pop]["dist_type"], self.n_cells_dict[pop], self.num_dendrites, rng=self.rng, mean=self.weight_config_dict[pop]["mean"], std=self.weight_config_dict[pop]["std"]).astype(np.float32, copy=False)
            weights_pop *= self.weight_config_dict[pop]["sign"]
            weights_pop_dict[pop] = weights_pop

        return weights_pop_dict
    
    # def get_dend_contribution_old(self, trial, prev_trial_end_values_dict, total_time_ms_prev_trial):

    #     self.dt_ms = float(self.dt * 1000.0)

    #     self.L_prev = 1200
    #     self.L_prev_slice = int(round(self.L_prev / self.dt_ms))
    #     self.last_psp_dict = {}  
    #     if self.debug:
    #         proc_peak_start = _get_peak_rss_bytes()
        
    #     rate_over_time_dict = {}

    #     dend_contribution_dict = {}

    #     new_end_times = {}

    #     v_cm_s = _sanitize_velocity_cm_s(self.an_velocity[:, trial]).astype(np.float32, copy=False)  
    #     dt_s   = self.dx / v_cm_s                                                                
    #     self.time_points = np.cumsum(dt_s, dtype=np.float32)                                     
    #     total_time  = float(self.time_points[-1])
    #     self.t_axis = np.arange(0.0, total_time, float(self.dt), dtype=np.float32)         
    #     self.T = self.t_axis.size
    #     total_time_ms = int(total_time*1000)


    #     for pop in self.weight_config_dict:

    #         input_rate_matrix = np.stack(self.processed_binned_activity_dict[pop], axis=0).astype(np.float32, copy=False)

    #         n_cells = self.n_cells_dict[pop]

    #         rate_over_time_array = np.empty((n_cells, self.T))

    #         rows = np.empty((n_cells, self.T))

    #         end_times_this_trial = [None] * n_cells

    #         prev_tails_this_pop = end_times_this_trial if (prev_trial_end_values_dict is None or prev_trial_end_values_dict.get(pop) is None) else prev_trial_end_values_dict[pop]

    #         last_psp = None

    #         for cell in range(n_cells):

    #             previous_trial_prepend_chunk = prev_tails_this_pop[cell]

    #             rate_over_time, spike_train = self.convert_rates_to_spikes(trial, cell, input_rate_matrix, previous_trial_prepend_chunk)
    #             T = self.T + self.L_prev_slice
    #             psps = self.epsps_event_add(spike_train, T).astype(np.float32)

    #             tail_this_trial = rate_over_time[-self.L_prev_slice:]

    #             end_times_this_trial[cell] = tail_this_trial

    #             psps = psps[self.L_prev_slice:]

    #             if self.store_intermediates:
    #                 rate_over_time_array[cell, :] = rate_over_time
    #                 last_psp = psps    
                        
    #             rows[cell, :] = psps.astype(np.float32, copy=False)


    #         new_end_times[pop] = end_times_this_trial

    #         if self.store_intermediates:
    #             rate_over_time_dict[pop] = rate_over_time_array

    #             self.last_psp_dict[pop] = last_psp

    #         dend_vm_over_time = (self.weights_pop_dict[pop].astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T)

    #         dend_contribution_dict[pop] = dend_vm_over_time
            
    #     return dend_contribution_dict, new_end_times, total_time_ms, rate_over_time_dict
    

    def get_dend_contribution(self, trial, prev_trial_end_values_dict, total_time_ms_prev_trial):

        self.dt_ms = float(self.dt * 1000.0)

        self.L_prev = 1200
        self.L_prev_slice = int(round(self.L_prev / self.dt_ms))
        self.last_psp_dict = {}  
        if self.debug:
            proc_peak_start = _get_peak_rss_bytes()
        
        rate_over_time_dict = {}

        dend_contribution_dict = {}

        new_end_times = {}

        v_cm_s = _sanitize_velocity_cm_s(self.an_velocity[:, trial]).astype(np.float32, copy=False)  
        dt_s   = self.dx / v_cm_s                                                                
        self.time_points = np.cumsum(dt_s, dtype=np.float32)                                     
        total_time  = float(self.time_points[-1])
        self.t_axis = np.arange(0.0, total_time, float(self.dt), dtype=np.float32)         
        self.T = self.t_axis.size
        total_time_ms = int(total_time*1000)

        print(f"self.debug {self.debug}")

        if self.debug:
            log_mem("B")

        for pop in self.weight_config_dict:

            n_cells = self.n_cells_dict[pop]

            if self.store_intermediates:

                rate_over_time_array = np.empty((n_cells, self.T))

            rows = np.empty((n_cells, self.T))

            end_times_this_trial = [None] * n_cells

            prev_tails_this_pop = end_times_this_trial if (prev_trial_end_values_dict is None or prev_trial_end_values_dict.get(pop) is None) else prev_trial_end_values_dict[pop]

            last_psp = None

            # Wpop = self.weights_pop_dict[pop].astype(np.float32, copy=False)  # (n_cells, n_dendrites) or (n_dendrites, n_cells)? assume later usage
            # if Wpop.shape[0] == n_cells:      # ensure shape is (n_dendrites, n_cells)
            #     Wpop = Wpop.T
            # dend_accum = np.zeros((Wpop.shape[0], self.T), dtype=np.float32)


            for cell in range(n_cells):

                previous_trial_prepend_chunk = prev_tails_this_pop[cell]

                rate_over_time, spike_times, end_of_spike_train= self.convert_rates_to_spikes(trial, cell, self.processed_binned_activity_dict[pop], previous_trial_prepend_chunk)
                

                end_times_this_trial[cell] = end_of_spike_train
                
                
                T = self.T + self.L_prev_slice
                psps = self.epsps_event_add(spike_times, T).astype(np.float32)

                psps = psps[self.L_prev_slice:]


                if self.store_intermediates:
                    rate_over_time_array[cell, :] = rate_over_time
                    last_psp = psps    
                        
                rows[cell, :] = psps.astype(np.float32, copy=False)
                
            #     w = Wpop[:, cell]            # (n_dendrites,)
            #     dend_accum += (w[:, None] * psps)

            # dend_contribution_dict[pop] = dend_accum


            self.prev_T = self.T


            new_end_times[pop] = end_times_this_trial

            if self.store_intermediates:
                rate_over_time_dict[pop] = rate_over_time_array

                self.last_psp_dict[pop] = last_psp

            dend_vm_over_time = (self.weights_pop_dict[pop].astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T)
            dend_contribution_dict[pop] = dend_vm_over_time

        gc.collect()
        
        if self.debug:
            log_mem("C")
            
        return dend_contribution_dict, new_end_times, total_time_ms, rate_over_time_dict
    


    def scale_dendrite(dend_contribution_summed):
        Vm_dict = {}
        for d in range(dend_contribution_summed.shape[1]):
            vm_array = dend_contribution_summed[:, d, :].astype(np.float32, copy=False)
            Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
            Vm_dict[d] = Vm.astype(np.float32, copy=False)

        dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
        return dend_activity


    def get_rate_vs_time(self, firing):
        valid  = np.isfinite(firing)
        if valid.sum() >= 2:
            rate_over_time = np.interp(self.t_axis, self.time_points[valid], firing[valid]).astype(np.float32, copy=False)
        else:
            rate_over_time = np.full(1, np.nan, dtype=np.float32)
        
        return rate_over_time
    
    def epsps_event_add(self, spike_idx, T):
        """
        Exact causal conv with `kernel` using event-driven accumulation.
        spike_idx: 1D int array of spike times (samples)
        T: length of output trace
        kernel: 1D float array (causal; length K)
        returns: 1D float array of length T
        """
        out = np.zeros(T, dtype=np.float32)
        K = self.kernel.shape[0]
        for s in spike_idx:
            if 0 <= s < T:
                end = min(T, s + K)
                out[s:end] += self.kernel[:(end - s)]
        return out


    def convert_spikes_to_psps(self, spike_idx):

        T = int(self.T)
        L_prev = int(self.L_prev)

        kernel = np.asarray(self.kernel, dtype=np.float32)
        K = kernel.shape[0]

        buffer_len = T + L_prev
        out = np.zeros(buffer_len, dtype=np.float32)

        spike_idx_shifted = np.asarray(spike_idx, dtype=np.int64) + L_prev

        for s in spike_idx_shifted:
            if s < 0 or s >= buffer_len:
                continue
            L = min(K, buffer_len - s)   # truncate kernel at buffer end
            if L <= 0:
                continue
            out[s:s+L] += kernel[:L]

        return out[L_prev:L_prev + T]

    def convert_rates_to_spikes_old(self, trial, cell, input_rate_matrix, end_of_previous_to_prepend):

        firing_rate_binned_trial = input_rate_matrix[cell, :, trial].astype(np.float32, copy=False)

        rate_over_time = self.get_rate_vs_time(firing_rate_binned_trial)

        end_current_trial = rate_over_time[-self.L_prev_slice:]

        if trial ==0:
            rate_over_time_prepend = np.concatenate([end_current_trial, rate_over_time])
        else:
            rate_over_time_prepend = np.concatenate([end_of_previous_to_prepend, rate_over_time])

        spike_train = get_inhom_poisson_spike_times_by_thinning(rate_over_time_prepend, np.arange((self.T+self.L_prev_slice), dtype=np.int32), dt=self.dt_ms, refractory=3., generator=self.rng).astype(np.int32, copy=False)

        return rate_over_time, spike_train
    

    def convert_rates_to_spikes(self, trial, cell, input_rate_matrix, end_of_previous_to_prepend):

        firing_rate_binned_trial = input_rate_matrix[cell, :, trial].astype(np.float32, copy=False)

        rate_over_time = self.get_rate_vs_time(firing_rate_binned_trial)

        spike_times = get_inhom_poisson_spike_times_by_thinning(rate_over_time, np.arange((self.T), dtype=np.int32), dt=self.dt_ms, refractory=3., generator=self.rng).astype(np.int32, copy=False)

        # spike_train = np.zeros(self.T).astype(np.float16)
        # spike_train[spike_times] = 1
        # segregation_point = self.T - self.L_prev_slice
        # end_of_spike_train = spike_train[segregation_point:].astype(np.float16)

        segregation_point = self.T - self.L_prev_slice

        end_spike_times = spike_times[spike_times >= segregation_point] - segregation_point

        if trial == 0:
            prev_part = end_spike_times
        else:
            if end_of_previous_to_prepend is None:
                prev_part = np.asarray(end_spike_times, dtype=np.int32)
            else:
                prev_part = np.asarray(end_of_previous_to_prepend, dtype=np.int32)

        spike_times_concatenated = np.concatenate([prev_part, spike_times + self.L_prev_slice]).astype(np.int32, copy=False)

        if self.store_intermediates:
            return rate_over_time, spike_times_concatenated, end_spike_times
        else:
            return 0, spike_times_concatenated, end_spike_times

    




    # def convert_rates_to_spikes(self, trial, cell, input_rate_matrix, prev_tail_unshifted):
    #     """
    #     prev_tail_unshifted: indices from previous trial in [T_prev-L .. T_prev-1], or None
    #     Returns:
    #     rate_over_time_or_None, spikes_concat (int32 indices in [0..T+L-1]),
    #     tail_unshifted (int32 indices for next trial in [T-L..T-1])
    #     """
    #     firing_rate_binned = input_rate_matrix[cell, :, trial].astype(np.float32, copy=False)
    #     rate_over_time = self.get_rate_vs_time(firing_rate_binned)

    #     T = rate_over_time.shape[0]
    #     L = self.L_prev_slice

    #     # 1) Draw spike indices for THIS trial in [0..T-1]
    #     spike_idx = get_inhom_poisson_spike_times_by_thinning(
    #         rate_over_time,
    #         np.arange(T, dtype=np.int32),
    #         dt=self.dt_ms, refractory=3., generator=self.rng
    #     ).astype(np.int32, copy=False)

    #     # 2) Tail for NEXT trial (UNSHIFTED, in this trial's frame): [T-L .. T-1]
    #     if L > 0:
    #         tail_mask = (spike_idx >= (T - L))
    #         tail_unshifted = spike_idx[tail_mask]
    #     else:
    #         tail_unshifted = np.empty((0,), dtype=np.int32)

    #     # 3) Map PREVIOUS trial's tail to [0..L-1] for prepend
    #     if trial == 0 or prev_tail_unshifted is None or prev_tail_unshifted.size == 0:
    #         prev_pos = np.empty((0,), dtype=np.int32)
    #     else:
    #         T_prev = self.prev_T            # set in caller after each trial
    #         prev_pos = prev_tail_unshifted - T_prev + L
    #         prev_pos = prev_pos[(prev_pos >= 0) & (prev_pos < L)]

    #     # 4) Current spikes go to [L .. L+T-1]
    #     curr_pos = spike_idx + L

    #     # 5) Final indices for convolution over length T+L (all int32)
    #     spikes_concat = np.concatenate([prev_pos, curr_pos]).astype(np.int32, copy=False)

    #     return (rate_over_time if self.store_intermediates else None), spikes_concat, tail_unshifted

    

    def analyze(self):

        n_pos = 50
        n_trials = 58
        dx=180/50
        dt = 0.001

        self.plateau_array_per_dend_list = []

        plateau_start_positions_counter = np.zeros((self.num_dendrites, n_pos))

        plateau_start_positions_counter_overall = np.zeros(n_pos)

        plateau_start_times_list_mega_list = []

        n_dendrites, n_trials, T_max = self.dend_activity.shape

        ragged_dend_list = []
        valid_lengths_per_dend = []  

        for d_idx in range(self.dend_activity.shape[0]):
            ragged_trial_list = []
            trial_lengths = [] 

            sum_data = 0
            for trial in range(self.dend_activity.shape[1]):
                trial_activity = self.dend_activity[d_idx,trial,:]
                valid = ~np.isnan(trial_activity)
                valid_trial_activity = trial_activity[valid]
                sum_data+=len(valid_trial_activity)
                trial_lengths.append(len(valid_trial_activity))
                ragged_trial_list.append(valid_trial_activity)

            

            ragged_trial_list_flat = np.hstack(ragged_trial_list)

            ragged_dend_list.append(ragged_trial_list_flat)
            valid_lengths_per_dend.append(trial_lengths)

        for d_idx in range(n_dendrites):

            flat_signal = ragged_dend_list[d_idx]                  
            flat_plateau_array = np.zeros_like(flat_signal, dtype=np.uint8)

            i = 0
            while i < flat_signal.size:
                if flat_signal[i] > float(self.dend_threshold):
                    flat_plateau_array[i:i+300] = 1
                    i += 800
                else:
                    i += 100

            lengths = valid_lengths_per_dend[d_idx]
            cursor = 0
            per_trial_padded = []
            for L in lengths:
                if L > 0:
                    seg = flat_plateau_array[cursor:cursor+L]
                    cursor += L
                    padded = np.zeros(T_max, dtype=np.uint8)
                    padded[:L] = seg
                else:
                    padded = np.zeros(T_max, dtype=np.uint8)
                per_trial_padded.append(padded)

            plateau_array = np.stack(per_trial_padded, axis=0)   
            
            self.plateau_array_per_dend_list.append(plateau_array)
                        
            proper_velocity = self.an_velocity*100

            animal_velocity = proper_velocity

            plateau_start_times_list = []
            
            for trial in range(plateau_array.shape[0]):
                velocity_trial = animal_velocity[:, trial]
                dt_trial = dx / velocity_trial  # in seconds
                time_each_pos_bin_starts = np.concatenate([[0], np.cumsum(dt_trial)])

                plateau_start_indices = np.where(np.diff(np.pad(plateau_array[trial], (1, 0))) == 1)[0]
                plateau_start_times = plateau_start_indices * dt  # in seconds
                plateau_start_times_list.append(plateau_start_times)

                for pt_start_time in plateau_start_times:
                    if pt_start_time != 0.0:
                        for pos_idx in range(50):
                            if time_each_pos_bin_starts[pos_idx] <= pt_start_time < time_each_pos_bin_starts[pos_idx + 1]:
                                plateau_start_positions_counter[d_idx, pos_idx] = 1
                                plateau_start_positions_counter_overall[pos_idx] += 1
                                break

            plateau_start_times_list_mega_list.append(plateau_start_times_list)

        return plateau_start_positions_counter, plateau_start_times_list_mega_list, plateau_start_positions_counter_overall
    
    def simulate(self, seed=0, debug=False):

        # SEED = seed
        # np.random.seed(SEED)
        # random.seed(SEED)
        self.rng = np.random.default_rng(seed)

        self.an_velocity, self.processed_binned_activity_dict = self.process_inputs_and_velocity()
        self.weights_pop_dict = self.init_weights()

        pops = list(self.weight_config_dict.keys())

        dend_vm_lists_dict = {pop: [] for pop in pops}

        prepend_storage_dict = {pop: None for pop in pops}

        total_time_ms_prev_trial = None


        self.prev_T = None


        rate_over_time_dict_list_per_trial = []


        padded_dend_contribution_dict = {}
        total_padded = None
        
        start_time = perf_counter() 

        for trial in range(self.max_num_trials):
            if self.debug:
                log_mem(f"A")
            dend_contrib_dict, prepend_storage_dict_new, total_time_ms_prev_trial, rate_over_time_dict = self.get_dend_contribution(trial, prepend_storage_dict, total_time_ms_prev_trial)
            if self.debug:
                log_mem(f"x")
            if self.store_intermediates:
                rate_over_time_dict_list_per_trial.append(rate_over_time_dict)
            for pop in pops:
                dend_vm_lists_dict[pop].append(dend_contrib_dict[pop])
            prepend_storage_dict = prepend_storage_dict_new

        end_time = perf_counter()      # ← call it
        print(f"total_time {end_time - start_time:.4f}s")


        for pop in pops:
            pop_padded = get_padded_dend_from_psps(dend_vm_lists_dict[pop])  
            pop_padded *= self.weight_config_dict[pop]["sf"]
            padded_dend_contribution_dict[pop] = pop_padded
            total_padded = pop_padded if total_padded is None else (total_padded + pop_padded)



        self.padded_dend_contribution_dict = padded_dend_contribution_dict
        self.total_padded_dend_contribution = total_padded

        self.dend_activity = scale_dendrite(total_padded)  


        self.plateau_start_positions_counter, self.dend_plateaus_list, self.plateau_start_positions_counter_overall = self.analyze()

        self.rate_over_time_dict_list_per_trial = rate_over_time_dict_list_per_trial


        num_dendrites_with_plateau = np.sum(self.plateau_start_positions_counter, axis=1)

        num_dendrites_with_plateau_mask = num_dendrites_with_plateau > 0

        num_dendrites_with_plateau_count = np.sum(num_dendrites_with_plateau_mask)


        summed_plateaus_per_position_bin = np.sum(self.plateau_start_positions_counter, axis=0)

        self.percent_dendrites_per_plateau_location = summed_plateaus_per_position_bin / self.num_dendrites

        binned_plateaus_as_fraction_dendrites = summed_plateaus_per_position_bin / np.sum(summed_plateaus_per_position_bin)

        binned_plateaus_as_fraction_dendrites_scaled_by_frac_active_plateaus = binned_plateaus_as_fraction_dendrites * (num_dendrites_with_plateau_count/self.num_dendrites)
        

        n_bins = 10
        bin_size = int(50 / n_bins)
        self.model_histogram_scaled = np.zeros(n_bins)
        model_histogram = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(binned_plateaus_as_fraction_dendrites_scaled_by_frac_active_plateaus[start:end])
            summed_data_og = np.sum(binned_plateaus_as_fraction_dendrites[start:end])
            self.model_histogram_scaled[i] = summed_data
            model_histogram[i] = summed_data_og

        self.model_histogram = model_histogram*100

        self.christine_overrepresentation_array_scaled = (self.christines_overrepresentation_array / 100) * 0.25

        # if self.store_intermediates:
        frac_dends_cum = self.get_frac_dends_cum()

        return self.model_histogram_scaled, frac_dends_cum, self.christine_overrepresentation_array_scaled


    def get_frac_dends_cum(self):
            num_plateaus_per_trial_list_across_dends = []
            dend_plateaus = self.dend_plateaus_list     # list of length n_dends; each is list over trials
            for dend in range(len(dend_plateaus)):
                per_trial_counts = [len(dend_plateaus[dend][trial])
                                    for trial in range(len(dend_plateaus[dend]))]
                num_plateaus_per_trial_list_across_dends.append(per_trial_counts)

            num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)  # (n_dends, n_trials)

            had_any_this_trial = (num_plateaus_per_trial_array > 0).astype(int)

            ever_had_by_trial = (np.cumsum(had_any_this_trial, axis=1) > 0).astype(float)  # (n_dends, n_trials)

            frac_dends_cum = ever_had_by_trial.mean(axis=0)  # (n_trials,)

            return frac_dends_cum

    def plot_summary(self):


        fig, axs = plt.subplots(4,4, figsize=(15,10))

        if self.constant_vel:
            vel_str = "Flat Velocity"
        else:
            vel_str = "Real Velocity"
        
        if self.flat_input:
            input_str = "Synthetic Data"
        else:
            input_str = "Real Data"
        
        if self.include_beta:
            beta_str = "Real Beta"
        else:
            beta_str = "No Beta"


        def _pt(v):  
            return FontProperties(size=v).get_size_in_points()
        title_fs = max(1, _pt(mpl.rcParams['axes.titlesize']) - 4)
        label_fs = max(1, _pt(mpl.rcParams['axes.labelsize']) - 5)


        if self.animal_by_animal:
            fig.suptitle(f"{input_str} {vel_str} {beta_str} Data From {animal} Only Tau (ms): {self.tau_ms:.3f}")
        else:
            fig.suptitle(f"{input_str} {vel_str} {beta_str} Data From All EC Cells, Tau (ms): {self.tau_ms:.3f}")

        pops = list(self.weight_config_dict.keys())

        trial_pop_dict = {}
        n_bins_target = 1000

        for trial in range(len(self.rate_over_time_dict_list_per_trial)):
            vel_pos_mps = self.an_velocity[:, trial]
            dict_for_trial = self.rate_over_time_dict_list_per_trial[trial]
            for pop in pops:
                arr = dict_for_trial[pop]                  # (n_cells, T_trial)
                n_cells = arr.shape[0]
                trial_mat = np.empty((n_cells, n_bins_target), dtype=np.float32)

                for cell in range(n_cells):
                    dend_vm_trace = arr[cell, :]
                    y = resample_trace_to_position_bins(
                        dend_vm_trace,
                        dt=self.dt,
                        vel_pos_mps=vel_pos_mps,
                        track_len_m=self.track_len/100.0,  # if self.track_len is in cm
                        n_bins_target=n_bins_target
                    )
                    trial_mat[cell, :] = y

                trial_pop_dict.setdefault(pop, []).append(trial_mat)


        for pop in trial_pop_dict:
            fine_binned_array = np.array(trial_pop_dict[pop])
            trial_av = np.mean(fine_binned_array, axis=0)
            cell_av = np.mean(trial_av, axis=0)
            cell_sem = sem(trial_av, axis=0)
            axs[0,0].plot(cell_av, label=pop)
            axs[0,0].fill_between(range(len(cell_av)), cell_av+cell_sem, cell_av-cell_sem, alpha=0.2)

        axs[0,0].set_title("Input Activity", fontsize=title_fs)
        axs[0,0].set_xlabel("Position Bins", fontsize=label_fs)
        axs[0,0].set_ylabel("DF/F Z-scored Scaled to Hz", fontsize=label_fs)
        axs[0,0].legend()

        for pop in self.last_psp_dict:
            PSP_example = self.last_psp_dict[pop] * self.weight_config_dict[pop]["sign"]
            axs[0,1].plot(PSP_example[:1000], label=pop)
        axs[0,1].set_xlabel("Time (ms)", fontsize=label_fs)
        axs[0,1].set_ylabel("EPSP Amplitude (mV)", fontsize=label_fs)
        axs[0,1].set_title("EPSP / IPSP Example Trains", fontsize=title_fs)
        axs[0,1].legend()

        ax = axs[0, 2]

        max_w = 25
        bins = np.linspace(0, max_w, 51)  # shared edges for all pops

        for pop, W in self.weights_pop_dict.items():
            w = np.asarray(W, dtype=np.float32).ravel()
            w = np.clip(w, 0, max_w)
            w = w[np.isfinite(w)]           # drop NaNs/inf to avoid empty plots
            if w.size == 0:
                continue
            ax.hist(
                w, bins=bins,
                histtype="step",            # outlines so they overlay cleanly
                linewidth=2,
                label=pop
            )

        ax.set_title(f"Weights: overlay by population", fontsize=title_fs)
        ax.set_ylabel("Count", fontsize=label_fs)
        ax.set_xlabel("Weight", fontsize=label_fs)
        ax.legend(frameon=False)

        axs[0,3].plot(self.an_velocity, color='r')
        axs[0,3].set_xlabel("Position Bins", fontsize=label_fs)
        axs[0,3].set_ylabel("Meters / Second", fontsize=label_fs)
        axs[0,3].set_title("Velocity", fontsize=title_fs)

        an_velocity_cm = self.an_velocity*100

        axs[1,3].plot(self.dx/an_velocity_cm, color='r')
        axs[1,3].set_xlabel("Position Bins", fontsize=label_fs)
        axs[1,3].set_ylabel("Seconds", fontsize=label_fs)
        axs[1,3].set_title("Occupancy", fontsize=title_fs)

        var_per_cell = []

        for cell in range(self.dend_activity.shape[0]):
            dendrite = self.dend_activity[cell,:,:]
            var_trials = np.var(dendrite, axis=0)
            var_per_cell.append(var_trials)

        var_per_array = np.array(var_per_cell)

        mean_var_per_array = np.nanmean(var_per_array, axis=0)

        bins = np.array_split(np.arange(50), 10)   # handles any remainder too

        binned_mean = np.array([np.nanmean(mean_var_per_array[idx]) for idx in bins])

        cell_bin_means = np.array([np.nanmean(var_per_array[:, idx], axis=1) for idx in bins]).T
        n_non_nan = np.sum(np.isfinite(cell_bin_means), axis=0)
        binned_sem = np.nanstd(cell_bin_means, axis=0, ddof=1) / np.sqrt(np.maximum(n_non_nan, 1))

        x = np.arange(len(binned_mean))
        axs[1,0].bar(x, binned_mean, yerr=binned_sem, capsize=3)
        axs[1,0].set_xticks(x)
        axs[1,0].set_xticklabels([f"{idx[0]+1}-{idx[-1]+1}" for idx in bins], fontsize=5)
        axs[1,0].set_xlabel("Binned Position", fontsize=label_fs)
        axs[1,0].set_ylabel("Mean variance across trials", fontsize=label_fs)

        X = np.asarray(self.dend_activity)

        n_bins_target = 1000

        overall_array = np.empty((self.num_dendrites, self.max_num_trials, n_bins_target))

        for cell in range(X.shape[0]):
            for trial in range(X.shape[1]):
                vel_pos_mps = self.an_velocity[:,trial]
                time_array = X[cell,trial,:]
                resampled = resample_trace_to_position_bins(time_array, self.dt, vel_pos_mps, track_len_m=self.track_len/100, n_bins_target=n_bins_target)
                overall_array[cell,trial,:] = resampled

        print(f"self.dend_activity.shape {self.dend_activity.shape}")


        dend_av_fine_x = np.mean(overall_array, axis=0)
        im4 = axs[2,0].imshow(dend_av_fine_x, aspect='auto', interpolation='none')
        axs[2,0].set_title("Mean over dendrites over seeds", fontsize=title_fs)
        axs[2,0].set_ylabel("Trials", fontsize=label_fs)
        axs[2,0].set_xlabel("Position Bins", fontsize=label_fs)
        fig.colorbar(im4, ax=axs[2,0], label="mV")

        trial_av = np.mean(dend_av_fine_x, axis=0)  
        sem_trial = sem(dend_av_fine_x, nan_policy='omit')
        axs[2,1].plot(trial_av)
        axs[2,1].fill_between(range(len(trial_av)), trial_av+sem_trial, trial_av-sem_trial, alpha=0.2)
        axs[2,1].set_title("Mean of trials", fontsize=title_fs)
        axs[2,1].set_xlabel("Position Bins", fontsize=label_fs)
        axs[2,1].set_ylabel("mV", fontsize=label_fs)


        plat_array_per_dendrite = []

        for dendrite in range(len(self.plateau_array_per_dend_list)):

            plateau_array = self.plateau_array_per_dend_list[dendrite]

            plateau_pos_list = []

            for t in range(plateau_array.shape[0]):
                sig_time = plateau_array[t]                 # shape (T,), 0/1 per ms
                v        = self.an_velocity[:, t]                        # shape (50,), m/s per position bin
                dx_m     = 1.8 / 50.0                       # meters per bin
                dt_ms    = 1.0
                T        = sig_time.size

                dt_per_bin_ms = (dx_m / np.maximum(v, 1e-6)) * 1000.0              # (50,)
                dt_per_bin_ms *= (T*dt_ms) / np.sum(dt_per_bin_ms)

                edges_ms   = np.concatenate([[0.0], np.cumsum(dt_per_bin_ms)])     # (51,)
                centers_ms = 0.5 * (edges_ms[:-1] + edges_ms[1:])                  # (50,)

                t_ms = np.arange(T, dtype=np.float32) * dt_ms                      # (T,)

                plat_pos = np.interp(centers_ms, t_ms, sig_time)                   # (50,)

                plateau_pos_list.append(plat_pos)

            plateau_pos_array = np.stack(plateau_pos_list)

            plat_array_per_dendrite.append(plateau_pos_array)


        n_bins_target=1000

        plateau_array_threed = np.empty((self.num_dendrites, self.max_num_trials, n_bins_target))

        for dendrite in range(len(self.plateau_array_per_dend_list)):

            plateau_array = self.plateau_array_per_dend_list[dendrite]

            for trial in range(plateau_array.shape[0]):
                vel_pos_mps = self.an_velocity[:,trial]
                time_array = plateau_array[trial]
                resampled = resample_trace_to_position_bins(time_array, self.dt, vel_pos_mps, track_len_m=self.track_len/100, n_bins_target=n_bins_target)
                plateau_array_threed[dendrite, trial, :] = resampled

        sum_over_trials = np.sum(plateau_array_threed, axis=1)
        sorted_sum_over_trials = np.argsort(np.argmax(sum_over_trials, axis=1))

        ims = axs[1,2].imshow(sum_over_trials[sorted_sum_over_trials,:], aspect='auto', cmap='gray', interpolation='none')
        axs[1,2].set_title(f"Mean plateaus per position (sum over trials)\nDendrite Threshold={self.dend_threshold:.2f}",
                        fontsize=title_fs)
        axs[1,2].set_xlabel("Position Bins", fontsize=label_fs)
        axs[1,2].set_ylabel("Dendrite index", fontsize=label_fs)
        cbar = fig.colorbar(ims, ax=axs[1,2], label="plateau count")
        cbar.set_label("plateau count", fontsize=label_fs)


        dend_contr_over_x_fine = np.empty((self.max_num_trials, self.num_dendrites, n_bins_target))
    
        for pop in self.padded_dend_contribution_dict:
            X = self.padded_dend_contribution_dict[pop]
            for trial in range(X.shape[0]):
                for dend in range(X.shape[1]):
                    vel_pos_mps = self.an_velocity[:,trial]
                    time_array = X[trial,dend,:]
                    resampled = resample_trace_to_position_bins(time_array, self.dt, vel_pos_mps, track_len_m=self.track_len/100, n_bins_target=n_bins_target)
                    dend_contr_over_x_fine[trial,dend,:] = resampled

            mean_dend_contr = np.mean(dend_contr_over_x_fine, axis=(0,1))
            axs[2,2].plot(mean_dend_contr)
        axs[2,2].set_xlabel("Position Bins", fontsize=label_fs)
        axs[2,2].set_ylabel("A.U.", fontsize=label_fs)


        start_pos_cnt50_list = self.plateau_start_positions_counter_overall
        n_bins = 10
        bin_size = int(50 / n_bins)
        summed_plateaus = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(start_pos_cnt50_list[start:end])
            summed_plateaus[i] = summed_data
        
        
        err_kw = dict(ecolor='k', elinewidth=0.8, capsize=2, capthick=0.8)
        axs[2,3].bar(range(len(summed_plateaus)), summed_plateaus)
        axs[2,3].set_xlabel("Position Bin", fontsize=label_fs)
        axs[2,3].set_ylabel("Plateau Count", fontsize=label_fs)
        axs[2,3].set_title("Plateau Onset Count per Track Section", fontsize=title_fs)
        axs[2,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=5)


        frac_dends_cum = self.get_frac_dends_cum()

        axs[3,0].clear()
        axs[3,0].set_title("Cumulative fraction of dendrites with plateau", fontsize=title_fs)
        axs[3,0].set_ylabel("Fraction of dendrites", fontsize=label_fs)
        axs[3,0].set_xlabel("Session Length (%)", fontsize=label_fs)
        # axs[3,0].set_ylim(0, 0.5)

        Tn = len(frac_dends_cum)
        axs[3,0].set_xticks([0, Tn//4, Tn//2, (3*Tn)//4, Tn-1], labels=["0","25","50","75","100"])

        axs[3,0].plot(frac_dends_cum, color='k', label="Model")

        axs[3,0].plot(Tn-1, 0.25, marker='*', markersize=10, color='r', label='Experimental Target')
        axs[3,0].legend(fontsize=8)

        num_plateaus_per_trial_list_across_dends = []

        for dend in range(len(self.dend_plateaus_list)):
            num_plateaus_per_trial = []
            dend_plateaus = self.dend_plateaus_list[dend]
            for trial in range(len(dend_plateaus)):
                num_plateaus_per_trial.append(len(dend_plateaus[trial]))
            num_plateaus_per_trial_list_across_dends.append(num_plateaus_per_trial)
        num_plateaus_per_trial_array = np.array(num_plateaus_per_trial_list_across_dends)
        mean_plat_per_trial = np.mean(num_plateaus_per_trial_array, axis=0)
        
        axs[3,1].set_title("Mean # Plateaus Per Trial Across Dendrites", fontsize=title_fs)
        axs[3,1].set_ylabel("Mean Plateaus", fontsize=label_fs)
        axs[3,1].set_xlabel("Session Length (%)", fontsize=label_fs)
        axs[3,1].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2,
            len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2,
            len(mean_plat_per_trial) - 1],
            labels=["0", '25', "50", '75', "100"])
        axs[3,1].plot(mean_plat_per_trial)

        axs[3,2].plot(self.percent_dendrites_per_plateau_location, marker='o')
        axs[3,2].set_ylabel("Percent of Dendrites", fontsize=label_fs)
        axs[3,2].set_xlabel("Position Bins", fontsize=label_fs)

        axs[3,3].plot(self.model_histogram, label='Model')
        axs[3,3].plot(self.christines_overrepresentation_array, label='Experimental', color='k')
        axs[3,3].legend(fontsize=5)
        lin_aray = np.linspace(0, 180, 10)
        axs[3,3].set_xticks(range(10))
        axs[3,3].set_xticklabels(lin_aray.astype(int))
        axs[3,3].set_xlabel("PF Peak Location (cm)", fontsize=label_fs)
        axs[3,3].set_ylabel("Fraction of CA1 PCs (%)", fontsize=label_fs)

        plt.tight_layout()
        plt.show()                        


            
    def simulate_old(self, seeds, export=False, plot=False, debug=False):
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
