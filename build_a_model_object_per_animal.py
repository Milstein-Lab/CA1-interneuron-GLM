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
import os, sys, time, psutil
_PROC = psutil.Process(os.getpid())

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

def random_timeseries(initial_value: float, volatility: float, count: int) -> list:
    time_series = []
    for _ in range(count+1):
        time_series.append(initial_value + random.gauss(0, 1) * volatility)
    return time_series


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


def turn_rates_into_spikes(dend_list_EC, an_velocity, weights_pop_dict, kernel, dt=None, n_dendrites=100, rng=None, debug=False, store_intermediates=None, mean=None, std=None):

    # if optimization_time:
    assert dt is not None
    dt_ms = float(dt * 1000.0)


    # pop_dend_vm_dict = {}

    # for pop in processed_binned_activity_dict:

        # dend_list_EC = processed_binned_activity_dict[pop]
        # an_velocity = animal_velocity_dict[pop]

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

    dend_contribution = dend_vm_padded  # (trials, dendrites, time) f16

    if dend_contribution.shape[1] != 100:
        print("dend_contribution_EC dendrites in the wrong axis")
    
    return dend_contribution, last_EPSP, warped_list


##############################################################################################################





# def turn_rates_into_spikes_add_inh(dend_list_EC, dend_list_NDNF, dend_list_SST, an_velocity, dist, kernel, SST_sf_opt, NDNF_sf_opt, dt=None, n_dendrites=100, rng=None, debug=False, optimization_time=None, mean=None, std=None):

#     # if optimization_time:
#     assert dt is not None
#     dt_ms = float(dt * 1000.0)

#     EC_input_matrix = np.stack(dend_list_EC, axis=0).astype(np.float32, copy=False)
#     n_EC, n_pos, n_trials = EC_input_matrix.shape


#     SST_input_matrix = np.stack(dend_list_SST, axis=0).astype(np.float32, copy=False)
#     n_SST, n_pos, n_trials = SST_input_matrix.shape


#     NDNF_input_matrix = np.stack(dend_list_NDNF, axis=0).astype(np.float32, copy=False)
#     n_NDNF, n_pos, n_trials = NDNF_input_matrix.shape

#     input_pop_list = [EC_input_matrix, SST_input_matrix, NDNF_input_matrix]

#     dx = np.float32(180.0 / n_pos)

#     weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std).astype(np.float32, copy=False)

#     weights_SST = sample_weights("Equal", n_SST, n_dendrites, rng=rng, mean=SST_sf_opt).astype(np.float32, copy=False)

#     weights_NDNF = sample_weights("Equal", n_NDNF, n_dendrites, rng=rng, mean=NDNF_sf_opt).astype(np.float32, copy=False)

#     weights_list = [weights_EC, weights_SST, weights_NDNF]


#     # W64 = sample_weights(dist, n_EC, n_dendrites, rng=rng, mean=mean, std=std)  # float64 by default
#     # print("pre-cast finite?", np.isfinite(W64).all(), "max:", W64.max())

#     # W16 = W64.astype(np.float16, copy=False)
#     # print("post-cast finite?", np.isfinite(W16).all(), "num inf:", np.isinf(W16).sum())

#     # kernel used by epsp add — keep as f16
#     kernel = np.asarray(kernel, dtype=np.float32)

#     L_prev = 1200  # history prepend/append
#     last_EPSP = None
   

#     if debug:
#         proc_peak_start = _get_peak_rss_bytes()
    
#     dend_vm_dict = {}

#     label_list = ["EC", "SST", "NDNF"]


#     for idx, input_population in enumerate(input_pop_list):

#         label = label_list[idx]

#         if debug:
#             trial_wall_start = time.perf_counter()
#             trial_peak_before = _get_peak_rss_bytes()
#             trial_curr_before = _get_current_rss_bytes()

#         dend_vm_list = []  # per-trial, shape (n_dendrites, T_t) in f16

#         warped_list = []


#         for t in range(n_trials):

#             v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t]).astype(np.float32, copy=False)  # (n_pos,)
#             dt_s   = dx / v_cm_s                                                                 # (n_pos,)
#             time_points = np.cumsum(dt_s, dtype=np.float32)                                      # len n_pos
#             total_time  = float(time_points[-1])
#             t_axis = np.arange(0.0, total_time, float(dt), dtype=np.float32)           # (T,)
#             T = t_axis.size
        
#             # rows = np.empty((n_EC, T), dtype=np.float32)
#             rows = np.zeros((input_population.shape[0], T), dtype=np.float32)

#             firing0 = input_population[0, :, t].astype(np.float32, copy=False)
#             valid0  = np.isfinite(firing0)
#             warped0 = (np.interp(t_axis, time_points[valid0], firing0[valid0])
#                     if valid0.sum() >= 2 else np.full(1, np.nan, dtype=np.float32))
#             warped0 = warped0.astype(np.float32, copy=False)
#             holder = warped0[-L_prev:] if warped0.size >= L_prev else warped0

#             # total two-track buffer in f16
#             two_len = len(holder) + T + len(holder)
#             two_track_length = np.empty(two_len, dtype=np.float32)
#             t_idx = np.arange(two_len, dtype=np.int32)

#             if not optimization_time:
#                 warped_array = np.empty((input_population.shape[0], T), dtype=np.float32)

#             for cell in range(input_population.shape[0]):
#                 firing = input_population[cell, :, t].astype(np.float32, copy=False)
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

#                 spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_idx, dt=dt_ms, refractory=3., generator=rng).astype(np.int32, copy=False)

#                 epsps = epsps_event_add(spike_times, two_len, kernel).astype(np.float32, copy=False)
#                 epsps = epsps[hL:hL + T]  # crop back to the middle warped region (length T)

#                 # rows[cell, :] = epsps.astype(np.float32, copy=False)
#                 if label in ("SST", "NDNF"):
#                     epsps *= -1.0

#                 rows[cell, :] += epsps.astype(np.float32, copy=False)
#                 if not optimization_time:
#                     warped_array[cell, :] = warped

#                 last_EPSP = epsps  

#             # ----- dendritic VM = W^T @ rows ; do compute in f32 then downcast -----
#             # weights_EC: (n_EC, n_dendrites), rows: (n_EC, T)
#             # dend_vm_over_time = (weights_EC.astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T) #### dendrite contribution

#             dend_vm_contribution = (weights_list[idx].astype(np.float32, copy=False).T @ rows.astype(np.float32, copy=False)).astype(np.float32, copy=False)  # (n_dendrites, T) #### dendrite contribution

#             dend_vm_list.append(dend_vm_contribution)
            
#             if (t & 7) == 7:
#                 import gc; gc.collect()

#             if not optimization_time:
#                 warped_list.append(warped_array.astype(np.float32, copy=False))

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

#             # free large temporaries as we go
#             del rows, two_track_length, warped0, warped, firing, firing0

#     # ----- optional padding (still f16) -----
#         num_trials = len(dend_vm_list)
#         n_dendrites_ = dend_vm_list[0].shape[0]
#         max_T = max(arr.shape[1] for arr in dend_vm_list)

#         dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
#         for i, arr in enumerate(dend_vm_list):
#             Ti = arr.shape[1]
#             dend_vm_padded[i, :, :Ti] = arr

#         dend_contribution = dend_vm_padded  # (trials, dendrites, time) f16

#         dend_vm_dict[label] = dend_contribution
#         warped_dict[label] = warped_list

#     overall_dend_contribution = dend_vm_dict["EC"] + dend_vm_dict["SST"] + dend_vm_dict["NDNF"]

#     if dend_contribution.shape[1] != 100:
#         print("dend_contribution_EC dendrites in the wrong axis")

#     # ----- Vm transform (compute f32; store f16) -----
#     Vm_dict = {}
#     for d in range(overall_dend_contribution.shape[1]):
#         vm_array = overall_dend_contribution[:, d, :].astype(np.float32, copy=False)
#         Vm, _, _ = activity_to_dend_vm_2d(
#             vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
#         Vm_dict[d] = Vm.astype(np.float32, copy=False)

#     dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)

#     if not optimization_time:
#         return dend_activity, weights_EC, weights_SST, weights_NDNF, last_EPSP, warped_dict
#     else:
#         return dend_activity, weights_EC, last_EPSP

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
    

def scale_dendrite(dend_contribution_summed):
                Vm_dict = {}
                for d in range(dend_contribution_summed.shape[1]):
                    vm_array = dend_contribution_summed[:, d, :].astype(np.float32, copy=False)
                    Vm, _, _ = activity_to_dend_vm_2d(vm_array, Vrest=-70.0, vm_scale=0.1, center_across="time")
                    Vm_dict[d] = Vm.astype(np.float32, copy=False)

                dend_activity = np.stack([Vm_dict[k] for k in sorted(Vm_dict.keys())], axis=0).astype(np.float32, copy=False)
                return dend_activity


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
                dx=3.6,  # cm
                store_intermediates=None,
                multiple_dendrites=True,
                residuals_activity_dict = None,
                make_it_spike = None,
                GLM_params_dict=None,
                behav_factors_dict=None,
                animal_by_animal=None,
                input_animal = None,
                max_num_trials=58,
                num_pos_bins=50,
                av_animals_velocity=0.43,
                hz_target_for_scaling=50,
                constant_vel=None, 
                include_beta=None,
                flat_input=None,
                dend_threshold=None,
                tau_ms=None,
                EC_weights_mean=None,
                EC_weights_std=None,
                sim_config = None,
                debug = False):
        
        self.kernel = kernel
        self.weight_config_dict = weight_config_dict
        self.residuals_activity_dict = residuals_activity_dict
        self.GLM_params_dict = GLM_params_dict
        self.behav_factors_dict = behav_factors_dict
        self.multiple_dendrites = bool(multiple_dendrites)
        self.make_it_spike = make_it_spike
        self.animal_by_animal = animal_by_animal
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
        self.EC_weights_mean = EC_weights_mean 
        self.EC_weights_std = EC_weights_std 
        self.input_animal = input_animal
        self.dt = dt
        self.max_num_trials = max_num_trials
        self.num_pos_bins = num_pos_bins
        self.av_animals_velocity = av_animals_velocity
        self.hz_target_for_scaling = hz_target_for_scaling
        self.const_vel=constant_vel
        self.include_beta=include_beta
        self.flat_input=flat_input
        self.num_dendrites = sim_config.num_dendrites
        self.debug = debug



        self.store_intermediates = store_intermediates

        ####



    # def store_intermediates(self, **kwargs) -> "SpikeSimModel":
    #     """Enable/disable what to keep. Example:
    #        model.store_intermediates(spikes=False, epsps=True, warp_axes=True)
    #     """
    #     self.flags = replace(self.flags, **kwargs)
    #     return self
    
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

    def modify_activity(self, residuals_activity_dict_pop, behav_factors_dict_pop, GLM_params_dict_pop):
        #residual_activity_dict_EC include_beta=True, const_vel=True, flat_input=False 
        # animal_by_animal = to_bool(animal_by_animal)

        const_vel = self.const_vel
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
                            # data = data + mu
                            data = ((data - np.min(data)) / (np.max(data) - np.min(data)) *self.hz_target_for_scaling)


                        dendrite_list.append(data)

                                    
        an_velocity = np.array(animal_velocity_list)
        an_velocity = np.nanmean(an_velocity, axis=0)

        return an_velocity, dendrite_list  #, dendrite_ta_list


    #   if self.make_it_spike:



    def make_the_dendrite(self, seed=0, debug=False):
        
        SEED = seed
        np.random.seed(SEED)
        random.seed(SEED)
        rng = np.random.default_rng(SEED)
        
            
        animal_velocity_list = []
        processed_binned_activity_dict = {}

        weights_pop_dict = {}
        
        for pop in self.weight_config_dict:
            this_animal_velocity, this_processed_binned_activity = self.modify_activity(self.residuals_activity_dict[pop], self.behav_factors_dict[pop], self.GLM_params_dict[pop])

            n_cells = len(this_processed_binned_activity)
            
            weights_pop = sample_weights(self.weight_config_dict[pop]["dist_type"], n_cells, self.num_dendrites, rng=rng, mean=self.weight_config_dict[pop]["mean"], std=self.weight_config_dict[pop]["std"]).astype(np.float32, copy=False)
            weights_pop_dict[pop] = weights_pop
            
            processed_binned_activity_dict[pop] = this_processed_binned_activity
            animal_velocity_list.append(this_animal_velocity)

        an_velocity_stack = np.array(animal_velocity_list)

        if an_velocity_stack.ndim == 3:
            an_velocity = np.mean(an_velocity_stack, axis=0)
        else:
            pass


        summed_dendrite_input = []

        last_EPSP_dict = {}
        warped_list_dict = {}
        for pop in self.weight_config_dict:
            dend_contribution, last_EPSP, warped_list = (
            turn_rates_into_spikes(processed_binned_activity_dict[pop], an_velocity, weights_pop_dict[pop],
                                        self.kernel, dt=self.dt,
                                        n_dendrites=self.num_dendrites, rng=rng, debug=self.debug,
                                        store_intermediates=self.store_intermediates))
            
            last_EPSP_dict[pop] = last_EPSP
            warped_list_dict[pop] = warped_list


            dend_contribution_scaled = dend_contribution * self.weight_config_dict[pop]["sign"]

            summed_dendrite_input.append(dend_contribution_scaled)

        summed_dendrite_input_array = np.array(summed_dendrite_input)

        
        summed_dendrite_input_array = np.mean(summed_dendrite_input_array, axis=0)

        final_scaled_dendrite = scale_dendrite(summed_dendrite_input_array)

            
        return an_velocity, final_scaled_dendrite, weights_pop_dict, last_EPSP_dict, warped_list_dict  # , min_trial_length #psp_list_dict
    
    def simulate(self, seed=0, debug=False):
        
        if debug:
            import psutil, os
            print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss/1e9, flush=True) 

        if debug:

            rank = MPI.COMM_WORLD.Get_rank()
            pid = os.getpid()

            t0 = time.time()
            print(f"[{t0:.3f}] rank={rank} pid={pid} START seed={seed}", flush=True)
            sys.stdout.flush()
            report_mem("pre get_dend_contribution")



        an_velocity, dend_activity, weights_pop_dict, last_EPSP_dict, warped_list_dict = self.make_the_dendrite(seed=seed, debug=False)  #psp_list_dict

        print(f"self.dend_threshold {self.dend_threshold} n_dendrites {self.num_dendrites}")

        if debug:
            report_mem("pre get_activity_multidendrite")

        (plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list,
        dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list,num_plateaus_per_dend_list) = get_activity_multidendrite(an_velocity, dend_activity,dend_threshold=self.dend_threshold, example_cell=17,n_dendrites=self.num_dendrites) #padded_warped_activity_list


        return (dend_activity, plateau_positions_counter,
                    plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,
                    num_plateaus_per_dend_list, plateau_start_times_list_mega_list, last_EPSP_dict,
                    weights_pop_dict, an_velocity, warped_list_dict) #padded_warped_activity_list






    # def simulate(self, seed=0, debug=False):
    #     if debug:
    #         import psutil, os
    #         print("RSS GB:", psutil.Process(os.getpid()).memory_info().rss/1e9, flush=True)

    #     activity_NDNF = 0
    #     activity_SST  = 0

    #     rank = MPI.COMM_WORLD.Get_rank()
    #     pid  = os.getpid()
    #     tag  = f"[SIM seed={seed} rank={rank} pid={pid}]"

    #     if debug:
    #         t0 = time.time()
    #         print(f"{tag} START {t0:.3f}", flush=True)
    #         sys.stdout.flush()
    #         report_mem(f"{tag} pre make_the_dendrite")

    #     # ---- before get_activity_multidendrite
    #     print(f"{tag} ENTER make_the_dendrite", flush=True)
    #     an_velocity, dend_activity, weights_pop_dict, last_EPSP_dict, warped_list_dict = \
    #         self.make_the_dendrite(seed=seed, debug=False)
    #     print(f"{tag} OK make_the_dendrite", flush=True)
    #     print(f"{tag} dend_activity type={type(dend_activity)} shape={getattr(dend_activity, 'shape', None)}", flush=True)

    #     if debug:
    #         report_mem(f"{tag} pre get_activity_multidendrite")

    #     print(f"{tag} ENTER get_activity_multidendrite", flush=True)
    #     try:
    #         result = get_activity_multidendrite(
    #             an_velocity, dend_activity,
    #             dend_threshold=self.dend_threshold,
    #             example_cell=17,
    #             n_dendrites=self.num_dendrites
    #         )
    #     except Exception as e:
    #         import traceback
    #         print(f"{tag} EXCEPTION in get_activity_multidendrite: {repr(e)}", flush=True)
    #         traceback.print_exc()
    #         raise

    #     print(f"{tag} RETURNED from get_activity_multidendrite type={type(result)}", flush=True)
    #     if result is None:
    #         print(f"{tag} ERROR: get_activity_multidendrite returned None", flush=True)
    #         raise RuntimeError("get_activity_multidendrite() returned None")

    #     if not isinstance(result, tuple):
    #         print(f"{tag} ERROR: expected tuple, got {type(result)}", flush=True)
    #         raise TypeError("get_activity_multidendrite() did not return a tuple")

    #     if len(result) != 8:
    #         print(f"{tag} ERROR: expected len 8, got {len(result)}", flush=True)
    #         raise ValueError("get_activity_multidendrite() returned wrong arity")

    #     (padded_warped_activity_list,
    #     plateau_positions_counter,
    #     plateau_start_positions_counter,
    #     plateau_array_per_dendrite_list,
    #     dendrite_plateau_mask,
    #     time_each_pos_bin_starts,
    #     plateau_start_times_list_mega_list,
    #     num_plateaus_per_dend_list) = result

    #     if debug:
    #         report_mem(f"{tag} post get_activity_multidendrite")

    #     print(f"{tag} RETURN simulate()", flush=True)
    #     return (dend_activity, plateau_positions_counter, padded_warped_activity_list,
    #             plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,
    #             num_plateaus_per_dend_list, plateau_start_times_list_mega_list, last_EPSP_dict,
    #             weights_pop_dict, an_velocity, activity_SST, activity_NDNF, warped_list_dict)


            
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