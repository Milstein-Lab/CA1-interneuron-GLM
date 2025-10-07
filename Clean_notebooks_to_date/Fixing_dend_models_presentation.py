import sys
from scipy.stats import sem
sys.path.append('/Users/michaelfinch/CA1-interneuron-GLM')

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *

from modelling_to_date_utils import *

import os, sys, time, resource

import time

import random 

import matplotlib as mpl
from matplotlib.font_manager import FontProperties


from scipy.signal import fftconvolve

def fit_equal_contrib_L2(EC, NDNF, SST, eps=1e-12):
    EC   = np.asarray(EC, dtype=float).ravel()
    NDNF = np.asarray(NDNF, dtype=float).ravel()
    SST  = np.asarray(SST, dtype=float).ravel()

    ndnf_norm = np.linalg.norm(NDNF)
    sst_norm  = np.linalg.norm(SST)
    if sst_norm < eps:
        raise ValueError("SST vector near zero; cannot enforce L2 equality.")
    r = ndnf_norm / (sst_norm + eps)

    Z = NDNF + r * SST
    denom = np.dot(Z, Z)
    if denom < eps:
        raise ValueError("Design vector Z is near zero; cannot solve.")

    ndnf_sf = np.dot(Z, EC) / denom
    sst_sf  = r * ndnf_sf

    # diagnostics
    contrib_ndnf = ndnf_norm * ndnf_sf
    contrib_sst  = sst_norm  * sst_sf
    fit = ndnf_sf * Z
    residual = EC - fit
    mse = np.mean(residual**2)

    return {
        "ndnf_sf": ndnf_sf,
        "sst_sf": sst_sf,
        "ratio_r": r,
        "mse": mse,
        "contrib_L2_ndnf": contrib_ndnf,
        "contrib_L2_sst": contrib_sst,
    }

def fit_sst_scale_to_cancel_ec(EC, SST, mask=None, nonneg=True, eps=1e-12):
    """
    Least-squares scalar s minimizing ||EC - s*SST||_2 over mask.
    Returns (s, info) where info has diagnostics.
    """
    EC = np.asarray(EC, dtype=float)
    SST = np.asarray(SST, dtype=float)
    assert EC.shape == SST.shape, f"Shape mismatch: EC{EC.shape} vs SST{SST.shape}"

    if mask is None:
        mask = np.isfinite(EC) & np.isfinite(SST)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(EC) & np.isfinite(SST)

    if not np.any(mask):
        return 0.0, {"reason": "empty mask (no finite entries)"}

    ec = EC[mask].ravel()
    sst = SST[mask].ravel()

    den = float(np.dot(sst, sst))
    if den < eps:
        return 0.0, {"reason": "near-zero SST energy", "den": den, "n": sst.size}

    num = float(np.dot(ec, sst))
    s = num / den
    if nonneg:
        s = max(0.0, s)

    return s, {"num": num, "den": den, "n": sst.size}

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

def get_dend_VM_cell_type_new(residual_activity_dict_EC, factors_dict_EC, GLM_params_EC, include_beta=True, const_vel=True, flat_input=False, animal_by_animal=False, animal_used=None):

    animal_by_animal = to_bool(animal_by_animal)
    flat_input = to_bool(flat_input)
    include_beta = to_bool(include_beta)

    data_list_normalized = []

    for animal in residual_activity_dict_EC:
        for cell in residual_activity_dict_EC[animal]:
            data_normalized = residual_activity_dict_EC[animal][cell][:,:58]
            data_normalized = ((data_normalized - np.min(data_normalized)) / (np.max(data_normalized) - np.min(data_normalized))) *50
            data_list_normalized.append(data_normalized)


    data_array_normalized = np.array(data_list_normalized)

    mu= np.mean(data_list_normalized)
    sigma = np.std(data_list_normalized)

    print(f"mu={mu} sigma={sigma}")

    dendrite = np.zeros((50, 58))
    
    dendrite_list = []


    #dendrite_ta_list = []


    # if use_averaged_velocity=="cell_type_av":
    #     animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))

    animal_velocity_list = []


    if animal_by_animal:
        animal = animal_used
        for cell in residual_activity_dict_EC[animal]:
            data = residual_activity_dict_EC[animal][cell]
            data = data[:,:58]

            if flat_input:
                data = np.zeros((data.shape))

            weights = GLM_params_EC[animal][cell]['weights']["Velocity"]
            # intercept = GLM_params[animal][cell]['intercept']

            if const_vel:
                animal_velocity = np.full((50, 58), 0.43)
                animal_velocity_list.append(animal_velocity)

                if include_beta:
                
                    data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                else:
                    data = data + mu

                #dendrite_ta_list.append(np.mean(data, axis=1))


            else:
                # if use_averaged_velocity=="cell_type_av":
                #     animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))
                # elif use_averaged_velocity=="actual_velocity":
                animal_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
                animal_velocity_list.append(animal_velocity)
                
                if include_beta:
                
                    data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                else:
                    data = data + mu

                # dendrite_ta_list.append(np.mean(data, axis=1))



            dendrite += data
            dendrite_list.append(data)


    else:
        for animal in residual_activity_dict_EC:
            for cell in residual_activity_dict_EC[animal]:
                data = residual_activity_dict_EC[animal][cell]
                data = data[:,:58]

                if flat_input:
                    data = np.zeros((data.shape))

                weights = GLM_params_EC[animal][cell]['weights']["Velocity"]
                # intercept = GLM_params[animal][cell]['intercept']

                if const_vel:
                    animal_velocity = np.full((50, 58), 0.43)
                    animal_velocity_list.append(animal_velocity)

                    if include_beta:
                    
                        data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                    else:
                        data = data + mu

                    #dendrite_ta_list.append(np.mean(data, axis=1))


                else:
                    # if use_averaged_velocity=="cell_type_av":
                    #     animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))
                    # elif use_averaged_velocity=="actual_velocity":
                    animal_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
                    animal_velocity_list.append(animal_velocity)
                    
                    if include_beta:
                    
                        data = data + mu + (weights * animal_velocity * sigma) #+ intercept

                    else:
                        data = data + mu

                    dendrite#_ta_list.append(np.mean(data, axis=1))



                dendrite += data
                dendrite_list.append(data)

                                            
    # if add_vel_contribution:
    #     if use_averaged_velocity=="actual_velocity":
    #         # if add_inh == 'both' or add_inh == 'sst':
    #         animal_velocity_array = np.array(animal_velocity_list)
    #         animal_velocity = np.mean(animal_velocity_array, axis=0)
    #         # else:
    #         #     animal_velocity_array = np.array(animal_velocity_list)
    #         #     animal_velocity = np.mean(animal_velocity_array, axis=2)

    an_velocity = np.array(animal_velocity_list)
    an_velocity = np.nanmean(an_velocity, axis=0)

    return an_velocity, dendrite_list  #, dendrite_ta_list

# def get_dend_VM_cell_type(residual_activity_dict, factors_dict_EC, mean_new_average_vel_array, GLM_params, dend_sf, amplitude=-5, vel=23, norm=False, add_vel_contribution=False, const_vel=True, use_averaged_velocity=None, add_inh=None, normalize_hz=False, make_it_spike=False, animal_by_animal=False, animal=None):

#     dendrite = np.zeros((50, 58))
    
#     dendrite_list = []


#     dendrite_ta_list = []


#     if use_averaged_velocity=="cell_type_av":
#         animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))

#     animal_velocity_list = []



#     for animal in residual_activity_dict:
#         for cell in residual_activity_dict[animal]:
#             data = residual_activity_dict[animal][cell]
#             data = data[:,:58]

#             weights = GLM_params[animal][cell]['weights']["Velocity"]
#             intercept = GLM_params[animal][cell]['intercept']

#             if const_vel:
#                 animal_velocity = np.full((50, 58), vel)
                
#                 data = data + (weights * animal_velocity) + intercept

#                 dendrite_ta_list.append(np.mean(data, axis=1))


#                 # if make_it_spike:
#                 #     data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50
                
                

#             elif add_vel_contribution:
#                 if use_averaged_velocity=="cell_type_av":
#                     animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))
#                 elif use_averaged_velocity=="actual_velocity":
#                     animal_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
#                     animal_velocity_list.append(animal_velocity)
                
#                 data = data + (weights * animal_velocity) + intercept

#                 # if make_it_spike:
#                 #     data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50


#             # if norm == "min_max":
#             #     data = normalize(data, norm='min_max', per_cell=False) * 10

#             #     if make_it_spike:
#             #         data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50

#             dendrite += data
#             dendrite_list.append(data)

#     dendrite = dendrite * dend_sf

#     if add_vel_contribution:
#         if use_averaged_velocity=="actual_velocity":
#             # if add_inh == 'both' or add_inh == 'sst':
#             animal_velocity_array = np.array(animal_velocity_list)
#             animal_velocity = np.mean(animal_velocity_array, axis=0)
#             # else:
#             #     animal_velocity_array = np.array(animal_velocity_list)
#             #     animal_velocity = np.mean(animal_velocity_array, axis=2)


#     for i in range(len(dendrite_ta_list)):
#         plt.plot(dendrite_ta_list[i])
#     plt.show()


#     return animal_velocity, dendrite_list


# def get_dend_VM_cell_type(residual_activity_dict, factors_dict_EC, mean_new_average_vel_array, GLM_params, dend_sf, amplitude=-5, vel=23, norm=False, add_vel_contribution=False, const_vel=True, use_averaged_velocity=None, add_inh=None, normalize_hz=False, make_it_spike=False, animal_by_animal=False, animal=None):

#     # dendrite = np.zeros((50, 58))

#     if use_averaged_velocity=="cell_type_av":
#         animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))

#     animal_velocity_list = []
    
#     dendrite_list = []

#     data_list_normalized = []

#     means_list = []

#     for animal in residual_activity_dict:
#         for cell in residual_activity_dict[animal]:
#             data_normalized = residual_activity_dict[animal][cell][:,:58]
#             data_normalized = (data_normalized - np.min(data_normalized)) / (np.max(data_normalized) - np.min(data_normalized)) *50
#             data_list_normalized.append(data_normalized)

            
#     print(f"np.mean(means_list) {np.mean(means_list)}")

#     # data_array_normalized = np.array(data_list_normalized)

#     overall_mu= np.mean(data_list_normalized)
#     overall_std = np.std(data_list_normalized)

#     print(f"overall_mu {overall_mu} overall_std {overall_std}")

#     ta_data_list = []


#     if animal_by_animal:

#         print(f"residual_activity_dict.keys() {residual_activity_dict.keys()}")

#         for cell in residual_activity_dict[animal]:
#             data = residual_activity_dict[animal][cell]
#             data = data[:,:58]
#             weights = GLM_params[animal][cell]['weights']["Velocity"]
#             intercept = GLM_params[animal][cell]['intercept']

#             if const_vel:
#                 animal_velocity = np.full((50, 58), vel)


#                 data_offset_by_mu = data + overall_mu
#                 data = data_offset_by_mu + (weights * animal_velocity * overall_std) 
#                 print(f"data.shape {data.shape}")
#                 ta_data_list.append(np.mean(data, axis=1))



                
#                 # data = data + (weights * animal_velocity) + intercept
#                 # data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50


#             elif add_vel_contribution:
#                 if use_averaged_velocity=="cell_type_av":
#                     animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))
#                 elif use_averaged_velocity=="actual_velocity":
#                     animal_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
#                     animal_velocity_list.append(animal_velocity)


#                 data_offset_by_mu = data + overall_mu
#                 data = data_offset_by_mu + (weights * animal_velocity * overall_std) 
#                 print(f"data.shape {data.shape}")
#                 ta_data_list.append(np.mean(data, axis=1))
                

#                 # data = data + (weights * animal_velocity) + intercept
#                 # # if make_it_spike:
#                 # data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50


#             # if norm == "min_max":
#             #     data = normalize(data, norm='min_max', per_cell=False) * 10

#             #     if make_it_spike:
#             #         print("WE MADE IT ALL THE WAY HEREEEEEEEEEEEE")

#             #         data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50

#             # dendrite += data
#             dendrite_list.append(data)


#     else:
#         for animal in residual_activity_dict:
#             for cell in residual_activity_dict[animal]:
#                 data = residual_activity_dict[animal][cell]
#                 data = data[:,:58]

#                 weights = GLM_params[animal][cell]['weights']["Velocity"]
#                 # intercept = GLM_params[animal][cell]['intercept']

#                 if const_vel:
#                     animal_velocity = np.full((50, 58), vel)
                    
#                     # data = data + (weights * animal_velocity) + intercept


#                     # if make_it_spike:
#                     #     print("WE MADE IT SPIKE")
#                     #     data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50

#                     data_offset_by_mu = data + overall_mu
#                     data = data_offset_by_mu + (weights * animal_velocity * overall_std)
                    
                    

#                 elif add_vel_contribution:
#                     if use_averaged_velocity=="cell_type_av":
#                         animal_velocity = np.tile(mean_new_average_vel_array[:, np.newaxis], (1, 58))
#                     elif use_averaged_velocity=="actual_velocity":
#                         animal_velocity = factors_dict_EC[animal]["Velocity"][:,:58]
#                         animal_velocity_list.append(animal_velocity)

#                     data_offset_by_mu = data_array + overall_mu
#                     data = data_offset_by_mu + (weights * animal_velocity * overall_std)
                    
#                 #     data = data + (weights * animal_velocity) + intercept

#                 #     if make_it_spike:
#                 #         data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50


#                 # if norm == "min_max":
#                 #     data = normalize(data, norm='min_max', per_cell=False) * 10

#                 #     if make_it_spike:
#                 #         data = (data - np.max(data)) / (np.min(data)-np.max(data)) * 50

#                 # dendrite += data
#                 dendrite_list.append(data)

#     # for i in range(len(ta_data_list)):
#     #     plt.plot(ta_data_list[i])
#     # plt.show()

#     # dendrite = dendrite * dend_sf

#     if add_vel_contribution:
#         if use_averaged_velocity=="actual_velocity":
#             # if add_inh == 'both' or add_inh == 'sst':
#             animal_velocity_array = np.array(animal_velocity_list)
#             animal_velocity = np.mean(animal_velocity_array, axis=0)
#             # else:
#             #     animal_velocity_array = np.array(animal_velocity_list)
#             #     animal_velocity = np.mean(animal_velocity_array, axis=2)

    
#     return animal_velocity, dendrite_list

def get_real_velocity_array(filtered_factors_dict_EC, filtered_factors_dict_SST, filtered_factors_dict):
    new_average_vel = []

    for animal in filtered_factors_dict_EC:
        for cell in filtered_factors_dict_EC[animal]:
            new_average_vel.append(np.mean(filtered_factors_dict_EC[animal][cell], axis=1))

    for animal in filtered_factors_dict_SST:
        for cell in filtered_factors_dict_SST[animal]:
            new_average_vel.append(np.mean(filtered_factors_dict_SST[animal][cell], axis=1))

    for idx, animal in enumerate(filtered_factors_dict):
        if idx > 8:
            for cell in filtered_factors_dict[animal]:
                new_average_vel.append(np.mean(filtered_factors_dict[animal][cell], axis=1))


    new_average_vel_array = np.array(new_average_vel)
    mean_new_average_vel_array = np.mean(new_average_vel_array, axis=0)
    
    return mean_new_average_vel_array

def fit_equal_contrib_L2(EC, NDNF, SST, eps=1e-12, SST_bias_factor=1.0):
    """
    Enforce L2 contribution: ||SST||*sst_sf = (SST_bias_factor) * ||NDNF||*ndnf_sf
    and fit EC ≈ ndnf_sf*NDNF + sst_sf*SST in least squares.

    Returns dict with ndnf_sf, sst_sf, ratio_r_b, mse, contribs.
    """
    EC   = np.asarray(EC, dtype=float).ravel()
    NDNF = np.asarray(NDNF, dtype=float).ravel()
    SST  = np.asarray(SST, dtype=float).ravel()

    ndnf_norm = np.linalg.norm(NDNF)
    sst_norm  = np.linalg.norm(SST)
    if sst_norm < eps:
        raise ValueError("SST vector near zero; cannot enforce L2 equality.")
    if ndnf_norm < eps:
        raise ValueError("NDNF vector near zero; cannot enforce L2 equality.")

    # enforce biased equality of contributions
    r_b = SST_bias_factor * (ndnf_norm / (sst_norm + eps))  # sst_sf = r_b * ndnf_sf

    Z = NDNF + r_b * SST
    denom = np.dot(Z, Z)
    if denom < eps:
        raise ValueError("Design vector Z is near zero; cannot solve.")

    ndnf_sf = np.dot(Z, EC) / denom
    sst_sf  = r_b * ndnf_sf

    fit = ndnf_sf * Z
    residual = EC - fit
    mse = np.mean(residual**2)

    contrib_ndnf = ndnf_norm * ndnf_sf
    contrib_sst  = sst_norm  * sst_sf

    return {
        "ndnf_sf": ndnf_sf,
        "sst_sf": sst_sf,
        "ratio_r_b": r_b,
        "mse": mse,
        "contrib_L2_ndnf": contrib_ndnf,
        "contrib_L2_sst": contrib_sst,
    }

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

# def get_inhom_poisson_spike_times_by_thinning(rate, t, dt=0.02, refractory=3., generator=None, rng=None):
#     """
#     Given a time series of instantaneous spike rates in Hz, produce a spike train consistent with an inhomogeneous
#     Poisson process with a refractory period after each spike.

#     :param rate: instantaneous rates in time (Hz)
#     :param t: corresponding time values (ms)
#     :param dt: temporal resolution for spike times (ms)
#     :param refractory: absolute deadtime following a spike (ms)
#     :param generator: random.Random()-like or NumPy Generator (falls back to rng if None)
#     :return: 1D np.array of spike times (ms)
#     """
#     # --- tiny micro-optimizations, same math/IO ---
#     rate = np.asarray(rate, dtype=np.float64)
#     t    = np.asarray(t,    dtype=np.float64)

#     # prefer 'generator', fall back to 'rng'
#     if generator is None:
#         generator = rng

#     # get a scalar uniform sampler without conditionals in the loop
#     # supports both random.Random and np.random.Generator
#     rand = generator.random if hasattr(generator, "random") else generator

#     # build interpolation grid (identical spacing)
#     t0 = float(t[0]); t1 = float(t[-1])
#     n_steps = int(np.floor((t1 - t0) / dt)) + 1
#     interp_t = t0 + np.arange(n_steps, dtype=np.float64) * dt  # == np.arange(t0, t1+dt, dt)

#     # interpolate rate and convert Hz -> per ms
#     interp_rate = np.interp(interp_t, t, rate).astype(np.float64, copy=False) / 1000.0

#     # refractory adjustment only where rate > 0
#     pos = interp_rate > 0.0
#     # r' = 1 / (1/r - refractory)  (units: ms^-1, refractory in ms)
#     interp_rate[pos] = 1.0 / (1.0 / interp_rate[pos] - refractory)

#     # if all rates are <= 0, return empty (same semantics, faster early-exit)
#     max_rate = float(np.max(interp_rate))
#     if not np.isfinite(max_rate) or max_rate <= 0.0:
#         return np.empty(0, dtype=np.float64)

#     inv_max_rate = 1.0 / max_rate   # hoist division out of loop
#     inv_dt       = 1.0 / dt

#     spike_times = []
#     append = spike_times.append      # local binding for speed

#     i = 0
#     ISI_memory = 0.0

#     # loop structure identical; keep exact accept/reject logic
#     while i < n_steps:
#         # guard against rare 0.0 exactly to avoid log(0)
#         x = rand()
#         if x == 0.0:
#             x = np.finfo(np.float64).tiny

#         ISI = -np.log(x) * inv_max_rate       # == -log(x)/max_rate
#         i += int(ISI * inv_dt)                # == int(ISI/dt)
#         ISI_memory += ISI

#         if i < n_steps:
#             # second uniform for thinning
#             y = rand()
#             if (y <= interp_rate[i] * inv_max_rate) and (ISI_memory >= 0.0):
#                 append(interp_t[i])
#                 ISI_memory = -refractory

#     return np.asarray(spike_times, dtype=np.float64)

# import numpy as np
# import math

# def get_inhom_poisson_spike_times_by_thinning_new(rate, t, dt=0.02, refractory=3., generator=None, rng=None):
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

def get_inhom_poisson_spike_times_by_thinning_ms(
    rate_hz,               # instantaneous rate in Hz
    t_ms,                  # time axis in ms (same length as rate_hz)
    dt_ms=1.0,             # step (ms)
    refractory_ms=3.0,     # absolute refractory (ms)
    generator=None,        # optional: python random.Random()
    rng=None               # optional: numpy Generator
):
    """
    ms-native thinning sampler (very close to your original).
    Returns spike times in ms as float64.
    """
    import numpy as np

    # Choose RNG
    if generator is None:
        if rng is None:
            import random
            generator = random.Random()
        else:
            # wrap numpy Generator to a .random() float API
            class _Wrap:
                def __init__(self, g): self.g = g
                def random(self): return float(self.g.random())
            generator = _Wrap(rng)

    # Build a uniform ms grid and interpolate rate onto it
    t_ms = np.asarray(t_ms, dtype=np.float64)
    interp_t_ms = np.arange(t_ms[0], t_ms[-1] + dt_ms, dt_ms, dtype=np.float64)

    interp_rate_hz = np.interp(interp_t_ms, t_ms, np.asarray(rate_hz, dtype=np.float64))
    # sanitize: NaNs/inf -> 0; negative -> 0
    if not np.all(np.isfinite(interp_rate_hz)):
        np.nan_to_num(interp_rate_hz, copy=False)
    np.maximum(interp_rate_hz, 0.0, out=interp_rate_hz)

    # Convert to spikes/ms
    rate_per_ms = interp_rate_hz / 1000.0

    # Effective rate with absolute refractory (in ms)
    # same algebra as your original but in ms-units:
    non_zero = rate_per_ms > 0.0
    if np.any(non_zero):
        rate_per_ms[non_zero] = 1.0 / (1.0 / rate_per_ms[non_zero] - refractory_ms)
        # clip negatives that can arise if refractory is too large for the rate
        np.maximum(rate_per_ms, 0.0, out=rate_per_ms)

    max_rate_per_ms = float(np.max(rate_per_ms)) if rate_per_ms.size else 0.0
    if not np.isfinite(max_rate_per_ms) or max_rate_per_ms <= 0.0:
        return np.empty(0, dtype=np.float64)

    # Thinning loop
    spike_times = []
    i = 0
    ISI_memory_ms = 0.0
    n = len(interp_t_ms)

    while i < n:
        x = generator.random()
        if x <= 0.0:
            # extremely rare; skip to avoid log(0)
            i += 1
            continue
        # draw ISI from exp with max rate (units: ms)
        ISI_ms = -np.log(x) / max_rate_per_ms
        i += int(ISI_ms / dt_ms)
        ISI_memory_ms += ISI_ms
        if i < n:
            # accept with probability rate/max_rate
            if generator.random() <= (rate_per_ms[i] / max_rate_per_ms) and ISI_memory_ms >= 0.0:
                spike_times.append(interp_t_ms[i])
                ISI_memory_ms = -refractory_ms

    return np.array(spike_times, dtype=np.float64)

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

def _fmt_bytes(n):
    if n is None: return "n/a"
    for u in ("B","KB","MB","GB","TB"):
        if n < 1024 or u == "TB":
            return f"{n:,.1f} {u}"
        n /= 1024.0

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

def get_contr_wrapper(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, vel_applied='real', add_inh=None, SST_bias_factor=None, use_averaged_velocity=None):

    if vel_applied=="real":
        constant_vel=False
        real_vel=True
        dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, use_averaged_velocity=use_averaged_velocity)
    elif vel_applied=="constant":
        constant_vel=True
        real_vel=False
        dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, use_averaged_velocity=use_averaged_velocity)
    elif vel_applied=="model":
        constant_vel=True
        real_vel=False
        dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True, add_inh=add_inh, use_model_EC=True, SST_bias_factor=SST_bias_factor, use_averaged_velocity=use_averaged_velocity)   
    else:
        constant_vel=False
        real_vel=False
        dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=False, add_inh=add_inh, SST_bias_factor=SST_bias_factor, use_averaged_velocity=use_averaged_velocity)

    return dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum

def count_plateaus_by_position_from_abs(plateau_abs_times,       # 1D array, seconds
                                    bin_starts_by_trial,      # shape (n_trials, 51), seconds, relative per trial
                                    trial_starts_abs=None):   # shape (n_trials,), seconds
    plateau_abs_times = np.asarray(plateau_abs_times, float)
    n_trials, n_edges = bin_starts_by_trial.shape
    n_pos_bins = n_edges - 1

    # derive trial starts if not provided (assumes no gaps)
    if trial_starts_abs is None:
        trial_durations = bin_starts_by_trial[:, -1]
        trial_starts_abs = np.concatenate([[0.0], np.cumsum(trial_durations[:-1])])

    counts = np.zeros(n_pos_bins, dtype=int)

    for t in range(n_trials):
        start = trial_starts_abs[t]
        end   = start + bin_starts_by_trial[t, -1]
        mask = (plateau_abs_times >= start) & (plateau_abs_times < end)
        if not np.any(mask):
            continue
        rel = plateau_abs_times[mask] - start  # seconds within this trial

        edges = bin_starts_by_trial[t]  # len 51
        pos_idx = np.searchsorted(edges, rel, side='right') - 1
        pos_idx = np.clip(pos_idx, 0, n_pos_bins-1)
        np.add.at(counts, pos_idx, 1)

    return counts

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


    return padded_warped_activity, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts

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

def zscore_2d(array, axis=None, eps=1e-12):
    
    arr = np.asarray(array, dtype=float)
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    return (arr - mean) / (std + eps)

def plot_the_dendrite(NDNF_sf_opt, SST_sf_opt, dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, mean_new_average_vel_array, NDNF_contribution_sum, SST_contribution_sum, dend_threshold=20, vel_applied=None, include_inhibition="both", ylim=None, pad_with_means=True):

    dx = 180/50 #track len (cm) / num pos bins
        
    if include_inhibition == 'both':
        summed_dendrite = dend_contribution_EC - (dend_contribution_NDNF*NDNF_sf_opt + dend_contribution_SST*SST_sf_opt)
    #     summed_dendrite = dend_contribution_EC - (SST_sf_opt * dend_contribution_SST*2)

        fig, axs = plt.subplots(4, 4, figsize=(25, 25))  

        fig.suptitle(f"Ratio SST Contribution : NDNF Contribution = {SST_contribution_sum / NDNF_contribution_sum:.3f}", y=1.0)

        EC_normalized = dend_contribution_EC/ np.max(dend_contribution_EC)

        mean_EC_dend = np.mean(EC_normalized, axis=1)
        sem_EC_dend = sem(EC_normalized, axis=1)
        axs[0,0].plot(mean_EC_dend, color='g', label="EC Vel")
        axs[0,0].fill_between(range(len(mean_EC_dend)), mean_EC_dend+sem_EC_dend, mean_EC_dend-sem_EC_dend, alpha=0.2, color='g')
        axs[0,0].set_title(f"Mean EC Dend Contribution SF={1.0}")
        axs[0,0].set_xlabel("Position Bin")
        axs[0,0].set_ylabel("Activity")
        axs[0,0].set_ylim(0,1)
            
        SST_normalized = dend_contribution_SST/ np.max(dend_contribution_SST)

        mean_SST_dend = np.mean(SST_normalized, axis=1)
        sem_SST_dend = sem(SST_normalized, axis=1)
        axs[0,1].plot(mean_SST_dend, color='b', label="SST Vel")
        axs[0,1].fill_between(range(len(mean_SST_dend)), mean_SST_dend+sem_SST_dend, mean_SST_dend-sem_SST_dend, alpha=0.2, color='b')
        axs[0,1].set_title(f"Mean SST Dend Contribution SF={SST_sf_opt:.3f}")
        axs[0,1].set_xlabel("Position Bin")
        axs[0,1].set_ylabel("Activity")
        axs[0,1].set_ylim(0,1)
        
        NDNF_normalized = dend_contribution_NDNF/ np.max(dend_contribution_NDNF)

        mean_NDNF_dend = np.mean(NDNF_normalized, axis=1)
        sem_NDNF_dend = sem(NDNF_normalized, axis=1)
        axs[0,2].plot(mean_NDNF_dend, color='orange', label="NDNF Vel")
    #     axs[0,1].plot(np.zeros(50), color='orange', label="NDNF Vel")
        axs[0,2].fill_between(range(len(mean_NDNF_dend)), mean_NDNF_dend+sem_NDNF_dend, mean_NDNF_dend-sem_NDNF_dend, alpha=0.2, color='orange')
        axs[0,2].set_title(f"Mean NDNF Dend Contribution SF={NDNF_sf_opt:.3f}")
        axs[0,2].set_xlabel("Position Bin")
        axs[0,2].set_ylabel("Activity")
        axs[0,2].set_ylim(0,1)

        summed_dendrite = zscore_2d(summed_dendrite, axis=None, eps=1e-12)
                    
        im11 = axs[1,0].imshow(summed_dendrite.T, aspect='auto')
        axs[1,0].set_title("Dendrite Over Position")
        axs[1,0].set_xlabel("Position Bin")
        axs[1,0].set_ylabel("Trials")
        fig.colorbar(im11, ax=axs[1,0])

        mean_dendrite = np.mean(summed_dendrite, axis=1)
        sem_dendrite = sem(summed_dendrite, axis=1)

        axs[1,1].plot(mean_dendrite, color='k')
        axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2, color='k')
        axs[1,1].set_title("Trial Averaged Dendrite Vm")
        axs[1,1].set_xlabel("Position Bin")
        axs[1,1].set_ylabel("Normalized Dendrite Vm")

        print(f"an_velocity.shape {an_velocity.shape}")

        im = axs[1,2].imshow(an_velocity.T, aspect='auto')
        axs[1,2].set_title("Average Animal Velocity")
        axs[1,2].set_ylabel("Trials")
        axs[1,2].set_xlabel("Position Bin")
        fig.colorbar(im, ax=axs[1,2], label="Meters/Second")

        axs[0,3].axis("off")

        occupancy = (3.6 / (an_velocity[:,0]*100))

        axs[1,3].plot(occupancy, color='r', linewidth=2)
        axs[1,3].set_title("Occupancy")
        axs[1,3].set_ylabel("Seconds")
        axs[1,3].set_xlabel("Position Bin")


        dt_constant = 0.001
        padded_warped_activity, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=dt_constant, dend_threshold=dend_threshold, vel_applied=vel_applied)

        if pad_with_means:
            A = padded_warped_activity.copy()
            row_means = np.nanmean(A, axis=1)                 # mean per column, skip NaNs
            row_means = np.where(np.isnan(row_means), 0, row_means)  # fallback if a column is all-NaN
            inds = np.where(np.isnan(A))
            A[inds] = row_means[inds[0]]
            padded_warped_activity = A

        im = axs[2,0].imshow(padded_warped_activity, aspect='auto', interpolation=None)
        axs[2,0].set_title("Dendrite Over Time")
        axs[2,0].set_xlabel("Time (ms)")
        axs[2,0].set_ylabel("Trials")
        fig.colorbar(im, ax=axs[2,0])

        mean_pad = np.nanmean(padded_warped_activity, axis=0)
        sem_pad = sem(padded_warped_activity, axis=0, nan_policy='omit')

        axs[2,1].plot(mean_pad, color='k')
        axs[2,1].fill_between(range(len(mean_pad)), mean_pad+sem_pad, mean_pad-sem_pad, color='k', alpha=0.2)
        axs[2,1].set_title("Trial Averaged Dendrite Vm")
        axs[2,1].set_xlabel("Time (ms)")
        axs[2,1].set_ylabel("Normalized Dendrite Vm")

        im2 = axs[2,2].imshow(plateau_array, aspect='auto', interpolation=None, cmap="gray")
        axs[2,2].set_title(f"Plateaus Over Time \n Dend Threshold={dend_threshold}")
        axs[2,2].set_xlabel("Time (ms)")
        axs[2,2].set_ylabel("Trials")
        fig.colorbar(im2, ax=axs[2,2])


        plateau_counts_per_time = np.sum(plateau_array, axis=0)
        axs[2,3].bar(range(len(plateau_counts_per_time)), plateau_counts_per_time)
        axs[2,3].set_xlabel("Time (ms)")
        axs[2,3].set_ylabel("Count")
        axs[2,3].set_title("Plateau Counts per Time")


        starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, plateau_array, dx=dx, dt_constant=dt_constant)


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
        axs[3,2].set_title("Plateau Count per Track Section")
        axs[3,2].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"])


  
         
        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
        axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
        axs[3,3].set_xlabel("Grouped Position Bins")
        axs[3,3].set_ylabel("% of Total Plateaus")
        axs[3,3].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"])

        axs[3,0].plot(cumulative_plateau_counts, color='k', linewidth=4)
        axs[3,0].set_title("Cumulative Plateau Count Over Trials")
        axs[3,0].set_ylabel("Cumulative Plateau Count")
        axs[3,0].set_xlabel("Session Length (%)")
        axs[3,0].set_xticks([0, len(cumulative_plateau_counts)//4, len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts)//4 + len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts) - 1], 
                            labels=["0", '25', "50", '75', "100"])
        

        print(f"just_plateau_starts_sums {len(just_plateau_starts_sums)}")

        mean_plat_per_trial = just_plateau_starts_sums
        axs[3,1].set_title("# Plateaus Per Trial Across Dendrites")
        axs[3,1].set_ylabel("# Plateaus")
        axs[3,1].set_xlabel("Session Length (%)")
        axs[3,1].set_xticks([0, len(mean_plat_per_trial)//4, len(mean_plat_per_trial)//2, len(mean_plat_per_trial)//4 + len(mean_plat_per_trial)//2, len(mean_plat_per_trial) - 1], 
                                labels=["0", '25', "50", '75', "100"])
        axs[3,1].plot(mean_plat_per_trial, color='k')
        

        plt.tight_layout()
        plt.show()

        return padded_warped_activity

    elif include_inhibition == 'sst':

        summed_dendrite = dend_contribution_EC - (dend_contribution_SST*SST_sf_opt)
    #     summed_dendrite = dend_contribution_EC - (SST_sf_opt * dend_contribution_SST*2)

        fig, axs = plt.subplots(4, 4, figsize=(25, 25))  
            

        EC_normalized = dend_contribution_EC/ np.max(dend_contribution_EC)

        mean_EC_dend = np.mean(EC_normalized, axis=1)
        sem_EC_dend = sem(EC_normalized, axis=1)
        axs[0,0].plot(mean_EC_dend, color='g', label="EC Vel")
        axs[0,0].fill_between(range(len(mean_EC_dend)), mean_EC_dend+sem_EC_dend, mean_EC_dend-sem_EC_dend, alpha=0.2, color='g')
        axs[0,0].set_title(f"Mean EC Dend Contribution SF={1.0}")
        axs[0,0].set_xlabel("Position Bin")
        axs[0,0].set_ylabel("Activity")
        axs[0,0].set_ylim(0,1)

                    
        SST_normalized = dend_contribution_SST/ np.max(dend_contribution_SST)

        mean_SST_dend = np.mean(SST_normalized, axis=1)
        sem_SST_dend = sem(SST_normalized, axis=1)
        axs[0,1].plot(mean_SST_dend, color='b', label="SST Vel")
        axs[0,1].fill_between(range(len(mean_SST_dend)), mean_SST_dend+sem_SST_dend, mean_SST_dend-sem_SST_dend, alpha=0.2, color='b')
        axs[0,1].set_title(f"Mean SST Dend Contribution SF={SST_sf_opt:.3f}")
        axs[0,1].set_xlabel("Position Bin")
        axs[0,1].set_ylabel("Activity")
        axs[0,1].set_ylim(0,1)


        axs[0,2].plot(an_velocity[:,0], color='r', linewidth=2)
        axs[0,2].set_title("Average Animal Velocity")
        axs[0,2].set_ylabel("meters/sec")
        axs[0,2].set_xlabel("Position Bin")

        occupancy = 3.6/(an_velocity[:,0]*100)

        axs[1,2].plot(occupancy, color='r', linewidth=2)
        axs[1,2].set_title("Occupancy")
        axs[1,2].set_ylabel("seconds")
        axs[1,2].set_xlabel("Position Bin")

        
        summed_dendrite = zscore_2d(summed_dendrite, axis=None, eps=1e-12)

        dt_constant = 0.001
        padded_warped_activity, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=dt_constant, dend_threshold=dend_threshold, vel_applied=vel_applied)

        print("NaNs left:", np.isnan(padded_warped_activity).sum())  # should be 0
        
        if pad_with_means:
            A = padded_warped_activity.copy()
            row_means = np.nanmean(A, axis=1)                 # mean per column, skip NaNs
            row_means = np.where(np.isnan(row_means), 0, row_means)  # fallback if a column is all-NaN
            inds = np.where(np.isnan(A))
            A[inds] = row_means[inds[0]]
            padded_warped_activity = A

      
        mean_dendrite = np.mean(summed_dendrite, axis=1)
        sem_dendrite = sem(summed_dendrite, axis=1)

        im11 = axs[1,0].imshow(summed_dendrite.T, aspect='auto')
        axs[1,0].set_title("Dendrite Over Position")
        axs[1,0].set_xlabel("Position Bin")
        axs[1,0].set_ylabel("Trials")
        fig.colorbar(im11, ax=axs[1,0])

        axs[1,1].plot(mean_dendrite, color='k')
        axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2, color='k')
        axs[1,1].set_title("Trial Averaged Dendrite Vm")
        axs[1,1].set_xlabel("Position Bin")
        axs[1,1].set_ylabel("Normalized Dendrite Vm")

        axs[3,1].plot(just_plateau_starts_sums, color='k', linewidth=2)
        axs[3,1].set_title("Plateau Count Over Trials")
        axs[3,1].set_ylabel("Plateau Count")
        axs[3,1].set_xlabel("Session Length (%)")
        axs[3,1].set_xticks([0, len(just_plateau_starts_sums)//4, len(just_plateau_starts_sums)//2, len(just_plateau_starts_sums)//4 + len(just_plateau_starts_sums)//2, len(just_plateau_starts_sums) - 1], 
                            labels=["0", '25', "50", '75', "100"])


        im = axs[2,0].imshow(padded_warped_activity, aspect='auto', interpolation=None)
        axs[2,0].set_title("Dendrite Over Time")
        axs[2,0].set_xlabel("Time (ms)")
        axs[2,0].set_ylabel("Trials")
        fig.colorbar(im, ax=axs[2,0])

        mean_dend_time = np.nanmean(padded_warped_activity, axis=0)
        sem_dend_time = sem(padded_warped_activity, axis=0, nan_policy='omit')

        axs[2,1].plot(mean_dend_time, color='k')
        axs[2,1].fill_between(range(len(mean_dend_time)), mean_dend_time+sem_dend_time, mean_dend_time-sem_dend_time, color='k', alpha=0.2)
        axs[2,1].set_title("Trial Averaged Dendrite Vm")
        axs[2,1].set_xlabel("Time (ms)")
        axs[2,1].set_ylabel("Normalized Dendrite Vm")

        im2 = axs[2,2].imshow(plateau_array, aspect='auto', interpolation=None, cmap="gray")
        axs[2,2].set_title(f"Plateaus Over Time \n Dend Threshold={dend_threshold}")
        axs[2,2].set_xlabel("Time (ms)")
        axs[2,2].set_ylabel("Trials")
        fig.colorbar(im2, ax=axs[2,2])

        plateau_counts_per_time = np.sum(plateau_array, axis=0)
        axs[2,3].bar(range(len(plateau_counts_per_time)), plateau_counts_per_time)
        axs[2,3].set_xlabel("Time (ms)")
        axs[2,3].set_ylabel("Count")
        axs[2,3].set_title("Plateau Counts per Time")


        starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, plateau_array, dx=dx, dt_constant=dt_constant)


        n_bins = 5
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
        axs[3,2].set_title("Plateau Count per Track Section")
        axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"], fontsize=7)
        axs[3,2].set_ylim(0,ylim)


          # Plot the cumulative plateau counts
        axs[3,0].plot(cumulative_plateau_counts, color='k', linewidth=4)
        axs[3,0].set_title("Cumulative Plateau Count Over Trials")
        axs[3,0].set_ylabel("Cumulative Plateau Count")
        axs[3,0].set_xlabel("Session Length (%)")
        axs[3,0].set_xticks([0, len(cumulative_plateau_counts)//4, len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts)//4 + len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts) - 1], 
                            labels=["0", '25', "50", '75', "100"])
        

        total_plateaus = np.sum(summed_plateaus)
        fraction_plateaus = summed_plateaus / total_plateaus
        axs[3,3].plot(fraction_plateaus*100, marker='o', color='k', markersize=7)
        axs[3,3].set_title("% of Plateaus in Grouped Position Bin")
        axs[3,3].set_xlabel("Grouped Position Bins")
        axs[3,3].set_ylabel("% of Total Plateaus")
        axs[3,3].set_xticks(np.arange(5), ["1-10", "11-20", "21-30", "31-40", "41-50"])

        axs[0, 3].axis("off")
        axs[1, 3].axis("off")

        
        plt.tight_layout()
        plt.show()

        return padded_warped_activity


    else:
        summed_dendrite = dend_contribution_EC

        fig, axs = plt.subplots(4,3, figsize=(20, 25))  
        # plt.suptitle(f"include_inhibition={include_inhibition}, use_model_EC={use_model_EC}, use_residuals=False, velocity modulation={vel_applied}")
        
        EC_normalized = dend_contribution_EC/ np.max(dend_contribution_EC)

        mean_EC_dend = np.mean(EC_normalized, axis=1)
        sem_EC_dend = sem(EC_normalized, axis=1)
        axs[0,0].plot(mean_EC_dend, color='g', label="EC Vel")
        axs[0,0].fill_between(range(len(mean_EC_dend)), mean_EC_dend+sem_EC_dend, mean_EC_dend-sem_EC_dend, alpha=0.2, color='g')
        axs[0,0].set_title(f"Mean EC Dend Contribution SF={1.0}")
        axs[0,0].set_xlabel("Position Bin")
        axs[0,0].set_ylabel("Activity")
        axs[0,0].set_ylim(0,1)

        im = axs[0,1].imshow(an_velocity.T, aspect='auto')
        axs[0,1].set_title("Average Animal Velocity")
        axs[0,1].set_ylabel("Trials")
        axs[0,1].set_xlabel("Position Bin")
        fig.colorbar(im, ax=axs[0,1], label="Meters/Second")
        

        vel_cm_sec = np.mean(an_velocity, axis=1)*100
        distance = dx
        axs[0,2].plot(distance/vel_cm_sec, color='r', linewidth=2)
        axs[0,2].set_ylabel("Seconds")
        axs[0,2].set_xlabel("Position Bins")
        axs[0,2].set_title("Average Animal Occupancy")
        
        
        summed_dendrite = zscore_2d(summed_dendrite, axis=None, eps=1e-12)

        im11 = axs[1,0].imshow(summed_dendrite.T, aspect='auto')
        axs[1,0].set_title("Dendrite Over Position")
        axs[1,0].set_xlabel("Position Bin")
        axs[1,0].set_ylabel("Trials")
        fig.colorbar(im11, ax=axs[1,0])

        mean_dendrite = np.mean(summed_dendrite, axis=1)
        sem_dendrite = sem(summed_dendrite, axis=1)

        axs[1,1].plot(mean_dendrite, color='k')
        axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2, color='k')
        axs[1,1].set_title("Trial Averaged Dendrite Vm")
        axs[1,1].set_xlabel("Position Bin")
        axs[1,1].set_ylabel("Normalized Dendrite Vm")

        print(f"an_velocity.shape {an_velocity.shape}")

        dt_constant = 0.001
        padded_warped_activity, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=dt_constant, dend_threshold=dend_threshold, vel_applied=vel_applied)
        
        if pad_with_means:
            A = padded_warped_activity.copy()
            row_means = np.nanmean(A, axis=1)                 # mean per column, skip NaNs
            row_means = np.where(np.isnan(row_means), 0, row_means)  # fallback if a column is all-NaN
            inds = np.where(np.isnan(A))
            A[inds] = row_means[inds[0]]
            padded_warped_activity = A

        print("NaNs left:", np.isnan(padded_warped_activity).sum())  # should be 0


        print(f"padded_warped_activity.shape early {np.any(np.isnan(padded_warped_activity))}")

        axs[1,2].plot(just_plateau_starts_sums, color='k', linewidth=2)
        axs[1,2].set_title("Plateau Count Over Trials")
        axs[1,2].set_ylabel("Plateau Count")
        axs[1,2].set_xlabel("Session Length (%)")
        axs[1,2].set_xticks([0, len(just_plateau_starts_sums)//4, len(just_plateau_starts_sums)//2, len(just_plateau_starts_sums)//4+len(just_plateau_starts_sums)//2,len(just_plateau_starts_sums) - 1], labels=["0", "25", "50", "75", "100"])

        # padded_warped_activity = zscore_2d(padded_warped_activity, axis=None, eps=1e-12)
        print(f"padded_warped_activity.shape late {np.any(np.isnan(padded_warped_activity))}")

        im = axs[2,0].imshow(padded_warped_activity, aspect='auto', resample=False)
        axs[2,0].set_title("Dendrite Over Time")
        axs[2,0].set_xlabel("Time (ms)")
        axs[2,0].set_ylabel("Trials")
        fig.colorbar(im, ax=axs[2,0])

        meany = np.nanmean(padded_warped_activity, axis=0)
        sems = sem(padded_warped_activity, axis=0, nan_policy='omit')

        axs[2,1].plot(meany, color='k')
        axs[2,1].fill_between(range(len(meany)), meany+sems, meany-sems, color='k', alpha=0.2)
        axs[2,1].set_title("Trial Averaged Dendrite Vm")
        axs[2,1].set_xlabel("Time (ms)")
        axs[2,1].set_ylabel("Normalized Dendrite Vm")


        im2 = axs[2,2].imshow(plateau_array, aspect='auto', interpolation=None, cmap="gray")
        axs[2,2].set_title(f"Plateaus Over Time \n Dend Threshold={dend_threshold}")
        axs[2,2].set_xlabel("Time (ms)")
        axs[2,2].set_ylabel("Trials")
        fig.colorbar(im2, ax=axs[2,2])



        plateau_counts_per_time = np.sum(plateau_array, axis=0)
        axs[3,1].bar(range(len(plateau_counts_per_time)), plateau_counts_per_time)
        axs[3,1].set_xlabel("Time (ms)")
        axs[3,1].set_ylabel("Count")
        axs[3,1].set_title("Plateau Counts per Time")




        starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, plateau_array, dx=dx, dt_constant=dt_constant)


        n_bins = 5
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
        axs[3,2].set_title("Plateau Count per Track Section")
        axs[3,2].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"])

        axs[3,0].plot(cumulative_plateau_counts, color='k', linewidth=4)
        axs[3,0].set_title("Cumulative Plateau Count Over Trials")
        axs[3,0].set_ylabel("Cumulative Plateau Count")
        axs[3,0].set_xlabel("Session Length (%)")
        axs[3,0].set_xticks([0, len(cumulative_plateau_counts)//4, len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts)//4 + len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts) - 1], 
                            labels=["0", '25', "50", '75', "100"])
      
        plt.tight_layout()
        plt.show()

        return padded_warped_activity

def plot_multidendrite_EC_err_across_seeds(dend_list_EC_interp, tau_ms,
    seeds, last_EPSP, weights_EC, weights_SST, weights_NDNF,  dend_vm_per_seed_dict,
    activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt,
    padded_warped_activity_list, an_velocity, dend_threshold,
    _pos_cnt_dict, start_pos_cnt50_dict, _plateau_arr_list_dict, _mask_dict,  _starts_list_dict,
    dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition=None,
    NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=False, make_it_spike=None, constant_vel=None, include_beta=None, flat_input=None):
    fig, axs = plt.subplots(4,4, figsize=(15,10))

    

    animal_by_animal = to_bool(animal_by_animal)
    constant_vel = to_bool(constant_vel)
    flat_input = to_bool(flat_input)
    include_beta = to_bool(include_beta)

    if constant_vel:
        vel_str = "Flat Velocity"
    else:
        vel_str = "Real Velocity"
    
    if flat_input:
        input_str = "Synthetic Data"
    else:
        input_str = "Real Data"
    
    if include_beta:
        beta_str = "Real Beta"
    else:
        beta_str = "No Beta"


    def _pt(v):  # convert named sizes like 'large' to points
        return FontProperties(size=v).get_size_in_points()
    title_fs = max(1, _pt(mpl.rcParams['axes.titlesize']) - 4)
    label_fs = max(1, _pt(mpl.rcParams['axes.labelsize']) - 5)

    animal_by_animal = to_bool(animal_by_animal)

    if animal_by_animal:
        fig.suptitle(f"{input_str} {vel_str} {beta_str} Data From {animal} Only Tau (ms): {tau_ms:.3f}")
    else:
        fig.suptitle(f"{input_str} {vel_str} {beta_str} Data From All EC Cells, Tau (ms): {tau_ms:.3f}")

    # num_trials   = len(dend_list_EC_interp)
    # n_dendrites_ = dend_list_EC_interp[0].shape[0]   # assume constant across trials


    # print(f"num_trials{num_trials} n_dendrites_ {n_dendrites_}")

    # num_trials = len(dend_list_EC_interp)
    # n_dendrites_ = dend_list_EC_interp[0].shape[0]
    # max_T = max(arr.shape[1] for arr in dend_list_EC_interp)  # <-- time axis length

    # print(f"num_trials={num_trials}  n_dendrites_={n_dendrites_}  max_T={max_T}")

    # dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

    # for i, arr in enumerate(dend_list_EC_interp):
    #     if arr.shape[0] != n_dendrites_:
    #         raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
    #     Ti = arr.shape[1]               # current trial's time length
    #     dend_vm_padded[i, :, :Ti] = arr # pad along time axis


    # dend_array_EC = dend_vm_padded

    # print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

    clean_trials = []
    for i, arr in enumerate(dend_list_EC_interp):
        arr = np.asarray(arr)
        if arr is None or arr.size == 0:
            raise ValueError(f"Trial {i}: empty array with shape {arr.shape}")
        if arr.ndim == 1:
            # Treat as (T,) -> (n_dendrites=1, T)
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"Trial {i}: expected 2D (n_dendrites, T), got {arr.ndim}D with shape {arr.shape}")
        clean_trials.append(arr)

    dend_list_EC_interp = clean_trials

    num_trials = len(dend_list_EC_interp)
    n_dendrites_ = dend_list_EC_interp[0].shape[0]
    for i, arr in enumerate(dend_list_EC_interp):
        if arr.shape[0] != n_dendrites_:
            raise ValueError(f"Trial {i}: n_dendrites mismatch {arr.shape[0]} != {n_dendrites_}")

    max_T = max(arr.shape[1] for arr in dend_list_EC_interp)
    print(f"num_trials={num_trials}  n_dendrites_={n_dendrites_}  max_T={max_T}")

    dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)
    for i, arr in enumerate(dend_list_EC_interp):
        Ti = arr.shape[1]
        dend_vm_padded[i, :, :Ti] = arr

    print(f"dend_vm_padded.shape {dend_vm_padded.shape}")

    mean_over_dend_per_trial = np.nanmean(dend_vm_padded, axis=1)     # (trial, time)
    grand_mean = np.nanmean(mean_over_dend_per_trial, axis=0)[:7000]        # (time,)
    grand_sem  = sem(mean_over_dend_per_trial, axis=0, nan_policy="omit")[:7000]

    axs[0,0].plot(grand_mean)
    axs[0,0].fill_between(range(len(grand_mean)), grand_mean+grand_sem, grand_mean-grand_sem, alpha=0.2)
    axs[0,0].set_ylabel("Average EC Activity (Hz)",  fontsize=label_fs)
    axs[0,0].set_xlabel("Time (ms)", fontsize=label_fs)
    axs[0,0].set_title("EC Input Rate", fontsize=title_fs)

    # if constant_vel==True and include_beta==False and flat_input

    # if make_it_spike:
    #     activity_list = []
    #     for trial in activity_EC:
    #         activity_list.append(activity_EC[trial])
    #     activity_EC = np.array(activity_list)
    #     print(f"activity_EC.shape {activity_EC.shape}")


    # D, N, T = activity_EC.shape            # (dendrites, trials, timebins)
    # dt = 0.001                             # seconds per time bin
    # t_ms = np.arange(T) * dt * 1000.0      # time (ms), length T

    # activity_EC_trial_av = np.nanmean(activity_EC, axis=1)   # (D, T)
    # for i in range(D):
    #     y = np.ma.masked_invalid(activity_EC_trial_av[i])
    #     axs[0, 0].plot(t_ms, y, alpha=0.2)

    # axs[0, 0].set_title("EC Input To Each Dendrite", fontsize=title_fs)
    # axs[0, 0].set_ylabel("Summed Z-Scored Activity", fontsize=label_fs)
    # axs[0, 0].set_xlabel("Time (ms)", fontsize=label_fs)
    # if not animal_by_animal:
    #     axs[0, 0].set_xlim(0, 6000)

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

    # axs[0,1].plot(last_EPSP[0,0,:1000])
    axs[0,1].plot(last_EPSP)
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

    mean_EC_activity = np.nanmean(activity_EC, axis=1)
    im4 = axs[1,0].imshow(mean_EC_activity, aspect='auto', interpolation=None)
    axs[1,0].set_title("Mean Input Activity", fontsize=title_fs)
    axs[1,0].set_ylabel("Trials", fontsize=label_fs)
    axs[1,0].set_xlabel("Time (ms)", fontsize=label_fs)
    # if not animal_by_animal:
    #     axs[1,0].set_xlim(0, 6000)
    cb = fig.colorbar(im4, ax=axs[1,0], label="Summed Z-Scored Activity")
    cb.set_label("Summed Z-Scored Activity", fontsize=label_fs)

    print(f"activity_EC.shape {activity_EC.shape}")

    mean_EC_activity = np.nanmean(activity_EC, axis=1)
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


    seeds = sorted(_mask_dict.keys())
    n_seeds = len(seeds)

    sample = _mask_dict[seeds[0]]
    # If your mask is (n_dendrites, n_pos), dend_axis=0; if (n_pos, n_dendrites), dend_axis=1
    # Pick the axis that's size 100 (or your known n_dendrites); fallback: the smaller axis
    n_dendrites = 100  # set this to your actual count
    if n_dendrites in sample.shape:
        dend_axis = list(sample.shape).index(n_dendrites)
    else:
        dend_axis = 0 if sample.shape[0] <= sample.shape[1] else 1  # heuristic

    # sum over dendrites, leaving position bins
    per_seed_counts = []
    for seed in seeds:
        mask = _mask_dict[seed].astype(bool)
        counts_per_pos = mask.sum(axis=dend_axis)  # length = n_pos
        per_seed_counts.append(counts_per_pos)

    per_seed_counts = np.asarray(per_seed_counts)                  # (n_seeds, n_pos)
    per_seed_pct = (per_seed_counts / float(n_dendrites)) * 100.0  # percent per seed

    mean_pct = per_seed_pct.mean(axis=0)                           # (n_pos,)
    sem_pct  = per_seed_pct.std(axis=0, ddof=1) / np.sqrt(n_seeds) # (n_pos,)

    x = np.arange(mean_pct.size)
    axs[3, 2].plot(x, mean_pct, lw=1.2)
    axs[3, 2].errorbar(x, mean_pct, yerr=sem_pct, color='k', marker='o',
                    markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)

    axs[3, 2].set_title("Percent of Dendrites with Plateau at Location", fontsize=title_fs)
    axs[3, 2].set_ylabel("Percent ± SEM Across Seeds", fontsize=label_fs)
    axs[3, 2].set_xlabel("Position Bin", fontsize=label_fs)
    axs[3, 2].set_ylim(0, 100)                      

    plt.tight_layout()
    plt.show()                        

    # num_dendrites_across_seeds =[]
    # for seed in _mask_dict:
    #     dendrite_plateau_mask = _mask_dict[seed]
    #     number_dendrites_counter = np.sum(dendrite_plateau_mask, axis=0)
    #     num_dendrites_across_seeds.append(number_dendrites_counter)

    # num_dendrites_across_seeds_array = np.array(num_dendrites_across_seeds)
    # mean_num_dendrites_across_seeds = np.mean(num_dendrites_across_seeds_array, axis=0)
    # sem_num_dendrites_across_seeds = np.mean(num_dendrites_across_seeds_array, axis=0)

    # axs[3,2].plot(mean_num_dendrites_across_seeds)
    # axs[3,2].errorbar(range(len(mean_num_dendrites_across_seeds)), mean_num_dendrites_across_seeds, yerr=sem_num_dendrites_across_seeds, color='k', marker='o', markersize=2, elinewidth=0.8, capsize=2, capthick=0.8)
    # axs[3,2].set_title("Percent of Dendrites with Plateau at Location", fontsize=title_fs)
    # axs[3,2].set_ylabel("Percent +- SEM Across Seeds", fontsize=label_fs)
    # axs[3,2].set_xlabel("Position Bin", fontsize=label_fs)

    # plt.tight_layout()
    # plt.show()

    return mean_plateaus_all_dends

def plot_the_model_EC(dend_contribution_EC, an_velocity, dend_threshold=250, vel_applied=None, include_inhibition=None, ylim=40):
    
    print(f"an_velocity.shape {an_velocity.shape}")

    fig, axs = plt.subplots(4, 3, figsize=(20, 20))  

    mean_EC_dend = np.mean(dend_contribution_EC, axis=1)
    sem_EC_dend = sem(dend_contribution_EC, axis=1)
    axs[0,0].plot(mean_EC_dend, color='g', label="EC Vel")
    axs[0,0].fill_between(range(len(mean_EC_dend)), mean_EC_dend+sem_EC_dend, mean_EC_dend-sem_EC_dend, alpha=0.2, color='g')
    axs[0,0].set_title(f"Mean EC Dend Contribution SF={1.0}")
    axs[0,0].set_xlabel("Position Bin")
    axs[0,0].set_ylabel("Activity")
    axs[0,0].set_ylim(0,axs[0,0].get_ylim()[1])
        
    axs[0,1].plot(an_velocity[:,0], linewidth=2, color='r')
    axs[0,1].set_ylabel("Meters / Second")
    axs[0,1].set_xlabel("Position Bins")
    axs[0,1].set_title("Average Animal Velocity")

    distance = 3.6
    occupancy = distance / (an_velocity[:,0]*100)

    axs[0,2].plot(occupancy, linewidth=2, color='purple')
    axs[0,2].set_ylabel("Seconds")
    axs[0,2].set_xlabel("Position Bins")
    axs[0,2].set_title("Average Animal Occupancy")
                
    im11 = axs[1,0].imshow(dend_contribution_EC.T, aspect='auto', interpolation='none')
    axs[1,0].set_title("Dendrite Over Position")
    axs[1,0].set_xlabel("Position Bin")
    axs[1,0].set_ylabel("Trials")
    fig.colorbar(im11, ax=axs[1,0])

    mean_dendrite = np.mean(dend_contribution_EC, axis=1)
    sem_dendrite = sem(dend_contribution_EC, axis=1)

    axs[1,1].plot(mean_dendrite, color='k')
    axs[1,1].fill_between(range(len(mean_dendrite)), mean_dendrite+sem_dendrite, mean_dendrite-sem_dendrite, alpha=0.2, color='k')
    axs[1,1].set_title("Trial Averaged Dend Vm")
    axs[1,1].set_xlabel("Position Bin")
    axs[1,1].set_ylabel("Dendrite Vm")

    dt_constant = 0.001
    padded_warped_activity, summed_dendrite, an_velocity, just_plateau_starts_list, plateau_array, mean_dend_time, sem_dend_time, x_time_ms, just_plateau_starts_sums, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, dend_contribution_EC, dt_constant=dt_constant, dend_threshold=dend_threshold, vel_applied=vel_applied)

    axs[1,2].plot(just_plateau_starts_sums, color='k', linewidth=2)
    axs[1,2].set_title("Plateau Count Over Trials")
    axs[1,2].set_ylabel("Plateau Count")
    axs[1,2].set_xlabel("Session Length (%)")
    axs[1,2].set_xticks([0, len(just_plateau_starts_sums)//4, len(just_plateau_starts_sums)//2, len(just_plateau_starts_sums)//4 + len(just_plateau_starts_sums)//2, len(just_plateau_starts_sums) - 1], 
                        labels=["0", '25', "50", '75', "100"])


    im = axs[2,0].imshow(padded_warped_activity, aspect='auto', interpolation='none')
    axs[2,0].set_title("Dendrite Over Time")
    axs[2,0].set_xlabel("Time (ms)")
    axs[2,0].set_ylabel("Trials")
    fig.colorbar(im, ax=axs[2,0])

    axs[2,1].plot(x_time_ms, mean_dend_time, color='k')
    axs[2,1].fill_between(x_time_ms, mean_dend_time+sem_dend_time, mean_dend_time-sem_dend_time, color='k', alpha=0.2)
    axs[2,1].set_title("Trial Averaged Dend Vm")
    axs[2,1].set_xlabel("Time (ms)")
    axs[2,1].set_ylabel("Vm")

    im2 = axs[2,2].imshow(plateau_array, aspect='auto', interpolation='none', cmap="gray")
    axs[2,2].set_title(f"Plateaus Over Time \n Dend Threshold={dend_threshold}")
    axs[2,2].set_xlabel("Time (ms)")
    axs[2,2].set_ylabel("Trials")
    fig.colorbar(im2, ax=axs[2,2])


    plateau_counts_per_time = np.sum(plateau_array, axis=0)
    axs[3,0].bar(range(len(plateau_counts_per_time)), plateau_counts_per_time)
    axs[3,0].set_xlabel("Time (ms)")
    axs[3,0].set_ylabel("Count")
    axs[3,0].set_title("Plateau Counts per Time")

    dx=3.6
    starts_per_pos, time_per_pos_s, plateau_start_positions_counter = get_internal_counts(an_velocity, plateau_array, dx=dx, dt_constant=dt_constant)

    n_bins = 5
    bin_size = int(50 / n_bins)

    summed_plateaus = np.zeros(n_bins)

    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        summed_data = np.sum(plateau_start_positions_counter[start:end])
        summed_plateaus[i] = summed_data

    axs[3,1].bar(range(len(summed_plateaus)), summed_plateaus)
    axs[3,1].set_xlabel("Position Bin")
    axs[3,1].set_ylabel("Plateau Count")
    axs[3,1].set_ylim(0, ylim)
    axs[3,1].set_title("Plateau Onset Count per Track Section")
    # axs[3,1].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)
    axs[3,1].set_xticks(np.arange(n_bins), ["1-10", "11-20", "21-30", "31-40", "41-50"], fontsize=7)


    axs[3,2].plot(cumulative_plateau_counts, color='k', linewidth=4)
    axs[3,2].set_title("Cumulative Plateau Count Over Trials")
    axs[3,2].set_ylabel("Cumulative Plateau Count")
    axs[3,2].set_xlabel("Session Length (%)")
    axs[3,2].set_xticks([0, len(cumulative_plateau_counts)//4, len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts)//4 + len(cumulative_plateau_counts)//2, len(cumulative_plateau_counts) - 1], 
                        labels=["0", '25', "50", '75', "100"])

    plt.tight_layout()
    plt.show()

def sample_equal_weights(mask, value=1.0):
    weights = np.zeros_like(mask, dtype=float)
    weights[mask] = value
    return weights

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

            print(f"animal_velocity.shape {animal_velocity.shape}")
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

    print(f"samples.shape {samples.shape}")

    return samples

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

def multi_wrap_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, vel_applied='real', add_inh=None, SST_bias_factor=None, dist=None, use_averaged_velocity=None, make_it_spike=False, seed=None):

    if vel_applied=="real":
        constant_vel=False
        real_vel=True
        
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist, use_averaged_velocity=use_averaged_velocity, make_it_spike=make_it_spike, seed=seed)
    elif vel_applied=="constant":
        constant_vel=True
        real_vel=False
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=True,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist, use_averaged_velocity=use_averaged_velocity, make_it_spike=make_it_spike, seed=seed)
    else:
        constant_vel=False
        real_vel=False
        an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=real_vel, constant_vel=constant_vel, use_residuals=False,  multiple_dendrites=True, add_inh=add_inh, SST_bias_factor=SST_bias_factor, dist=dist, use_averaged_velocity=use_averaged_velocity, make_it_spike=make_it_spike, seed=seed)

    return an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF
        
def get_activity_multidendrite(an_velocity, dend_vm_list, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, dend_threshold=20, vel_applied="real", example_cell=15, dist="Uniform", n_dendrites=100, n_SST=75, n_EC=792, n_NDNF=73, include_inhibition=True, use_model_EC=False, make_it_spike=None):


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


    if make_it_spike:

        n_pos = 50
        n_trials = 58
        dx=180/50
        dt_constant = 0.001

        # vm_list = []

        # for trial in range(dend_contribution_EC.shape[0]):
        #     vm_list.append(dend_contribution_EC[trial])

        # vm_array = np.array(vm_list)
        # vm_array = np.swapaxes(vm_array, 0, 1)
        # if vm_array.shape[1] != 100:
        #     raise ValueError(f"dendrites in wrong axis !=100")
        # if vm_array.shape[0] != 58:
        #     raise ValueError(f"trials in wrong axis !=58")

        num_trials   = len(dend_vm_list)
        n_dendrites_ = dend_vm_list[0].shape[0]   # assume constant across trials


        print(f"num_trials{num_trials} n_dendrites_ {n_dendrites_}")

        num_trials = len(dend_vm_list)
        n_dendrites_ = dend_vm_list[0].shape[0]
        max_T = max(arr.shape[1] for arr in dend_vm_list)  # <-- time axis length

        print(f"num_trials={num_trials}  n_dendrites_={n_dendrites_}  max_T={max_T}")

        dend_vm_padded = np.full((num_trials, n_dendrites_, max_T), np.nan, dtype=np.float32)

        for i, arr in enumerate(dend_vm_list):
            if arr.shape[0] != n_dendrites_:
                raise ValueError(f"Trial {i}: n_dendrites mismatch: {arr.shape[0]} != {n_dendrites_}")
            Ti = arr.shape[1]               # current trial's time length
            dend_vm_padded[i, :, :Ti] = arr # pad along time axis

        dend_contribution_EC = dend_vm_padded

        if dend_contribution_EC.shape[1] != 100:
            print("dend_contribution_EC dendrites in the wrong axis")

    

        Vm_dict = {}
        for dend in range(dend_contribution_EC.shape[1]):
            vm_array_shaped = dend_contribution_EC[:, dend, :]
            Vm, _, _ = activity_to_dend_vm_2d(
                vm_array_shaped,
                Vrest=-70.0,
                vm_scale=0.1,
                center_across="time")
            Vm_dict[dend] = Vm

        vm_list = []
        for trial in Vm_dict:
            vm_list.append(Vm_dict[trial])

        dend_activity = np.array(vm_list)

        plateau_positions_counter = np.zeros(n_pos)
        plateau_start_positions_counter = np.zeros(n_pos)
        plateau_array_per_dendrite_list = []
        num_plateaus_per_dend_list = []
        dendrite_plateau_mask = np.zeros((dend_activity.shape[0], n_pos), dtype=bool)
        padded_warped_activity_list = []
        plateau_start_times_list_mega_list = []

        for d_idx in range(dend_activity.shape[0]):

            padded_warped_activity=dend_activity[d_idx,:,:]
            padded_warped_activity_list.append(padded_warped_activity)

            flat_padded_warped_activity = padded_warped_activity.flatten()
            flat_plateau_array = np.zeros_like(flat_padded_warped_activity)


            i = 0
            while i < len(flat_padded_warped_activity):
                if flat_padded_warped_activity[i] > float(dend_threshold):
                    flat_plateau_array[i:i+300] = 1

                    i += 800
                else:
                    i += 100

            plateau_array = flat_plateau_array.reshape(padded_warped_activity.shape)
            plateau_array_per_dendrite_list.append(plateau_array)
                        
            proper_velocity = an_velocity*100

            animal_velocity = proper_velocity

            plateau_start_times_list = []
            
            for trial in range(plateau_array.shape[0]):
                velocity_trial = animal_velocity[:, trial]
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
            for trial in range(plateau_array.shape[0]):
                velocity_trial = animal_velocity[:, trial]
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


            num_trials, num_time_bins = plateau_array.shape
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


        return padded_warped_activity_list, dend_contribution_EC, dend_activity, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list    
            
                

    else:
        if include_inhibition == 'both':
            dend_activity = activity_EC - (activity_NDNF*NDNF_sf_opt + activity_SST*SST_sf_opt)

        elif include_inhibition == 'sst':
            dend_activity = activity_EC - (activity_SST*SST_sf_opt)
            
        else:
            dend_activity = activity_EC 

        dend_activity = zscore_2d(dend_activity, axis=None, eps=1e-12)

    
        
        plateau_positions_counter = np.zeros(50)
        plateau_start_positions_counter = np.zeros(50)

        threshold_list = []

        variance_list = []

        plateau_array_per_dendrite_list = []

        num_plateaus_per_dend_list = []

        dendrite_plateau_mask = np.zeros((dend_activity.shape[0], 50), dtype=bool)

        padded_warped_activity_list = []
        
        plateau_start_times_list_mega_list = []


        for d_idx in range(dend_activity.shape[0]):
            doi=dend_activity[d_idx,:,:]

            dt= 4.71657036/50
            dx=180/50

            animal_velocity_constant= np.full((doi.shape), dx/dt)

            proper_velocity = an_velocity*100


            dt_constant = 0.001

            if vel_applied=="constant":
                dt = dx / animal_velocity_constant
            else:
                dt = dx / proper_velocity
            time_bins = np.cumsum(dt, axis=0)
            time_bins_ms = time_bins * 1

            num_trials = dend_activity.shape[2]
            trial_warped_activity = []
            max_len = 0

            for t in range(num_trials):
                if np.any(np.isnan(time_bins[:, t])):
                    continue
                total_time = time_bins[-1, t]

                time_axis_constant = np.arange(0, total_time, dt_constant)

                firing = doi[:, t]

                warped_firing = np.interp(time_axis_constant, time_bins[:,t], firing)

                trial_warped_activity.append(warped_firing)
                if len(warped_firing) > max_len:
                    max_len = len(warped_firing)

            padded_warped_activity = np.full((num_trials, max_len), np.nan) 
            for i, trace in enumerate(trial_warped_activity):
                padded_warped_activity[i, :len(trace)] = trace


            padded_warped_activity_list.append(padded_warped_activity)

            position_bins, num_trials = animal_velocity_constant.shape

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

            animal_velocity = proper_velocity

            plateau_start_times_list = []

            
            for trial in range(plateau_array.shape[0]):
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
            for trial in range(plateau_array.shape[0]):
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

            animal_velocity = proper_velocity


            num_trials, num_time_bins = plateau_array.shape
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

        


        return padded_warped_activity_list, dend_activity, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list    
        
def plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None):
    
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


        activity_EC_trial_av = np.mean(activity_EC, axis=2)
        mean_activity_EC_trial_av = np.mean(activity_EC_trial_av, axis=0)
        for i in range(activity_EC_trial_av.shape[0]):
            axs[0,0].plot(activity_EC_trial_av[i,:], alpha=0.2)
        axs[0,0].plot(mean_activity_EC_trial_av, linewidth=3, color='r', linestyle='--')
        axs[0,0].set_title("EC Input To Each Dendrite")
        axs[0,0].set_ylabel("Summed Z-Scored Activity")
        axs[0,0].set_xlabel("Position Bins")

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

        padded_warped_activity_array = np.array(padded_warped_activity_list)

        mean_pad = np.mean(padded_warped_activity_array, axis=0)

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

        ims = axs[1,2].imshow(plateau_array_per_dendrite_list[example_cell], aspect='auto', cmap='gray')
        axs[1,2].set_title(f"Ex/ Dendrite Plateaus Over Time \n Dendrite Threshold={dend_threshold}")
        axs[1,2].set_xlabel("Time (ms)")
        axs[1,2].set_ylabel("Trials")
        fig.colorbar(ims, ax=axs[1,2])

        axs[0,2].hist(weights_EC.flatten(), bins=50)
        axs[0,2].set_title(f"EC Weights: {dist} Distribution")
        axs[0,2].set_ylabel("Count")
        axs[0,2].set_xlabel("Weight")

        mean_dend_activity = np.mean(dend_activity, axis=0)

        im4 = axs[1,0].imshow(mean_dend_activity.T, aspect='auto', interpolation=None)
        axs[1,0].set_title("Mean Over Dendrites")
        axs[1,0].set_ylabel("Trials")
        axs[1,0].set_xlabel("Position Bins")
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
        

        means = np.nanmean(mean_dend_activity, axis=1)
        
        axs[1,1].plot(means)
        axs[1,1].set_ylabel("Summed Z-Scored Activity")
        axs[1,1].set_title("Mean of Dendrites and Trials")
        axs[1,1].set_xlabel("Position Bins")

        axs[2,2].bar(range(len(plateau_positions_counter)), plateau_positions_counter)
        axs[2,2].set_title(f"Plateau Time Across All Dendrites")
        axs[2,2].set_ylabel("Time (ms)")
        axs[2,2].set_xlabel("Position Bins")

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

        # axs[5,0].bar(range(len(summed_plateaus)), summed_plateaus)
        # axs[5,0].set_xlabel("Position Bin Quintile")
        # axs[5,0].set_ylabel("Time (ms)")
        # axs[5,0].set_title("Plateau Time per Track Section")
        # axs[5,0].set_xticks(np.arange(n_bins), ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50"], fontsize=7)

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


# SEED = 42
                # np.random.seed(SEED)
                # random.seed(SEED)
                # rng = np.random.default_rng(SEED)
                # n_EC = 792
                # weights_EC = sample_weights(dist, n_EC, rng=rng)
                # EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
                # L_prev=500
                # precomputed_prepend = np.array(random_timeseries(np.mean(EC_input_matrix), np.std(EC_input_matrix), L_prev))
                # v_safe   = np.apply_along_axis(_sanitize_velocity_cm_s, 0, an_velocity)
                # n_pos = EC_input_matrix.shape[1]
                # n_trials = EC_input_matrix.shape[2]
                # dx = 180.0 / n_pos  
                # dt_s_all = dx / v_safe
                # time_points_all = np.cumsum(dt_s_all, axis=0)
                # total_time_per_trial = time_points_all[-1, :]                       # (n_trials,)
                # T_warp_per_trial = np.floor(total_time_per_trial / dt_constant + 1e-12).astype(int)
                # dend_vm_holder = np.zeros((n_trials, 50000))

                # t_axis_list = [np.arange(T_warp_per_trial[t], dtype=np.float64) * dt_constant for t in range(n_trials)]
                # T2_per_trial = L_prev + T_warp_per_trial
                # dt_ms = dt_constant * 1000.0

                # for t in range(n_trials):
                #     start_time = time.time()
                #     T_warp = int(T_warp_per_trial[t])
                #     T2 = int(T2_per_trial[t])
                #     t_axis = t_axis_list[t]
                #     time_points = time_points_all[:, t]         # precomputed
                #     t_ms = np.arange(T2, dtype=np.float32) * dt_ms

                #     rows = np.zeros((n_EC, T_warp), dtype=np.float32)
                #     rate_buf = np.empty(T2, dtype=np.float32)
                #     rate_buf[:L_prev] = precomputed_prepend[:L_prev]

                #     for cell in range(n_EC):
                #         firing = EC_input_matrix[cell, :, t]
                #         valid  = np.isfinite(firing)
                #         if valid.sum() < 2:
                #             continue
                #         warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                #         # guaranteed len(warped) == T_warp because t_axis came from T_warp
                #         rate_buf[L_prev:L_prev+T_warp] = warped

                #         spike_times = get_inhom_poisson_spike_times_by_thinning(rate_buf[:T2], t_ms, dt=dt_ms, refractory=3., rng=rng).astype(int)
                #         st_curr = spike_times[spike_times >= L_prev] - L_prev
                #         rows[cell, :T_warp] = epsps_event_add(st_curr, T_warp, kernel).astype(np.float32, copy=False)
                #     end_time = time.time()
                #     print(f"total trial time trial {t} time {end_time-start_time}")
                #     dend_vm_over_time = weights_EC @ rows #X_trial
                #     dend_vm_holder[t,:len(dend_vm_over_time)] = dend_vm_over_time
     
                # ---- one-time setup (above the loop) ----
                
                
                # SEED = 42
                # np.random.seed(SEED); random.seed(SEED); rng = np.random.default_rng(SEED)

                # n_EC = 792
                # weights_EC = sample_weights(dist, n_EC, rng=rng).astype(np.float32, copy=False)

                # EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0).astype(np.float32, copy=False)
                # L_prev = 500
                # precomputed_prepend = np.array(
                #     random_timeseries(float(np.nanmean(EC_input_matrix)), float(np.nanstd(EC_input_matrix)), L_prev),
                #     dtype=np.float32
                # )

                # n_pos = EC_input_matrix.shape[1]
                # n_trials = EC_input_matrix.shape[2]
                # dx = 180.0 / n_pos
                # dt_ms = float(dt_constant) * 1000.0

                # # PRECOMPUTE (no interpolation just for shapes)
                # v_safe   = np.apply_along_axis(_sanitize_velocity_cm_s, 0, an_velocity)       # (n_pos, n_trials)
                # dt_s_all = dx / v_safe                                                        # (n_pos, n_trials)
                # time_points_all = np.cumsum(dt_s_all, axis=0)                                 # (n_pos, n_trials)
                # total_time_per_trial = time_points_all[-1, :]                                 # (n_trials,)
                # # match np.arange(0, total_time, dt_constant) length -> floor
                # T_warp_per_trial = np.floor(total_time_per_trial / dt_constant + 1e-12).astype(np.int32)
                # T2_per_trial    = (L_prev + T_warp_per_trial).astype(np.int32)

                # # prebuild per-trial axes (seconds, ms)
                # t_axis_list = [np.arange(T_warp_per_trial[t], dtype=np.float64) * dt_constant for t in range(n_trials)]
                # t_ms_list   = [np.arange(int(T2_per_trial[t]), dtype=np.float32) * dt_ms       for t in range(n_trials)]

                # # reusable buffers (avoid per-trial allocations)
                # T_warp_max = int(T_warp_per_trial.max())
                # T2_max     = int(T2_per_trial.max())
                # rows_buf   = np.zeros((n_EC, T_warp_max), dtype=np.float32)                    # reused per trial
                # rate_buf   = np.empty(T2_max, dtype=np.float32)
                # rate_buf[:L_prev] = precomputed_prepend

                # # final holder (2D: trials × time) — size to your real max, not a magic 50k
                # dend_vm_holder = np.zeros((n_trials, T2_max), dtype=np.float32)

                # # minor micro-opts
                # interp = np.interp; isfin = np.isfinite

                # # (optional) quiet GC while we loop
                # import gc, time
                # gc_on = gc.isenabled()
                # if gc_on: gc.disable()

                # # ---- trial loop ----
                # for t in range(n_trials):
                #     t0 = time.perf_counter()

                #     T_warp = int(T_warp_per_trial[t]); T2 = int(T2_per_trial[t])
                #     t_axis = t_axis_list[t]
                #     t_ms   = t_ms_list[t]
                #     time_points = time_points_all[:, t]  # precomputed cumulative time

                #     # get a view into the reusable rows buffer and zero only the active slice
                #     rows = rows_buf[:, :T_warp]
                #     rows.fill(0.0)

                #     for cell in range(n_EC):
                #         firing = EC_input_matrix[cell, :, t]        # float32, no copy
                #         valid  = isfin(firing)
                #         if valid.sum() < 2: 
                #             continue

                #         warped = interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                #         # fill the reusable rate buffer: prepend is already set
                #         rate_buf[L_prev:L_prev+T_warp] = warped

                #         spike_times = get_inhom_poisson_spike_times_by_thinning(
                #             rate_buf[:T2], t_ms, dt=dt_ms, refractory=3., rng=rng
                #         ).astype(int)
                #         st_curr = spike_times[spike_times >= L_prev] - L_prev

                #         rows[cell, :T_warp] = epsps_event_add(st_curr, T_warp, kernel).astype(np.float32, copy=False)

                #     # (matmul usually fast; not included in timing if you want apples-to-apples)
                #     vm = weights_EC @ rows  # (T_warp,) if weights_EC is (n_EC,); else (n_dendrites, T_warp)
                #     trace = np.nanmean(vm, axis=0) if vm.ndim == 2 else vm
                #     dend_vm_holder[t, :T_warp] = trace

                #     print(f"[trial {t}] {time.perf_counter() - t0:.3f}s")
                # # ---- end loop ----

                # if gc_on: gc.enable()
                                

                # activity_EC = get_dendrite_activity(weights_EC, EC_input_matrix, n_dendrites, n_EC)
                # activity_SST = get_dendrite_activity(weights_SST, SST_input_matrix, n_dendrites, n_SST)
                # activity_NDNF = get_dendrite_activity(weights_NDNF, NDNF_input_matrix, n_dendrites, n_NDNF)

                # EC = activity_EC.copy()
                # NDNF = activity_NDNF.copy()
                # SST = activity_SST.copy()

                # if add_inh=='sst':
                #     SST_sf_opt, info = fit_sst_scale_to_cancel_ec(activity_EC, activity_SST)
                #     NDNF_sf_opt=0
                # else:
                #     # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
                #     # NDNF_sf_opt, SST_sf_opt = result.x
                #     res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
                #     NDNF_sf_opt = res["ndnf_sf"]
                #     SST_sf_opt  = res["sst_sf"]
                #     NDNF_contribution_sum = res["contrib_L2_ndnf"]
                #     SST_contribution_sum = res["contrib_L2_sst"]




##### my versiion 

                # import os, sys, time, resource

                # import sys, os, resource
                # def _fmt_bytes(n):
                #     for u in ("B","KB","MB","GB","TB"):
                #         if n < 1024 or u == "TB": return f"{n:,.1f} {u}"
                #         n /= 1024.0
                # def _peak_rss_bytes():
                #     ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                #     return ru if sys.platform == "darwin" else ru * 1024  # mac: bytes, linux: KB
                # def _curr_rss_bytes():
                #     try:
                #         import psutil
                #         return psutil.Process(os.getpid()).memory_info().rss
                #     except Exception:
                #         return None

                # SEED = 42
                # np.random.seed(SEED)
                # random.seed(SEED)
                # rng = np.random.default_rng(SEED)

                # n_EC = 792
                # n_SST = 75
                # n_NDNF = 115
                # n_dendrites=100
                # n_trials = 58


                # EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
                # SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
                # NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

            
                # n_pos = EC_input_matrix.shape[1]
                # n_trials = EC_input_matrix.shape[2]
                # dx = 180.0 / n_pos  

                # vel = an_velocity
                # if vel.shape != (n_pos, n_trials):
                #     raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

                # L_prev = 500
  
                # precomputed_prepend = random_timeseries(np.mean(EC_input_matrix), np.std(EC_input_matrix) ,L_prev)
                

                # weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng)
                    

                # proc_peak_start = _peak_rss_bytes()
                # run_curr_start  = _curr_rss_bytes()
                # print(f"[run start] RSS={_fmt_bytes(run_curr_start or 0)} peak={_fmt_bytes(proc_peak_start)}")
                
                # max_length = 0
                # dend_vm_per_trial_dict = {}

                # for t in range(EC_input_matrix.shape[2]):
                #     trial_wall_start  = time.time()
                #     trial_peak_before = _peak_rss_bytes()
                #     trial_curr_before = _curr_rss_bytes()


                #     start_time = time.time()
                #     v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t])      
                #     dt_s   = dx / v_cm_s                             
                #     edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
                #     total_time = float(edges_s[-1])

                #     # constant-time axis in seconds
                #     t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
                #     max_time_over_cells = 0

                #     firing_example = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
                #     valid = np.isfinite(firing_example)
                #     time_points = np.cumsum(dt_s)
                #     warped_example = np.interp(t_axis, time_points[valid], firing_example[valid]).astype(np.float32, copy=False)

                #     two_track_length_example = np.concatenate([precomputed_prepend, warped_example], axis=0)
                #     t_ms = np.arange(two_track_length_example.shape[0]) * dt_ms


                #     rows= []
                #     for cell in range(n_EC):
                #         firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
                #         valid = np.isfinite(firing)
                #         if valid.sum() >= 2:
                #             time_points = np.cumsum(dt_s)  # length n_pos

                #             # start_time = time.time()
                #             warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                #             # end_time = time.time()
                #             # print(f"interpolation_time = {end_time-start_time} ")
                #         else:
                #             warped = np.full(1, np.nan, dtype=np.float32)

                        
                #         two_track_length = np.concatenate([precomputed_prepend, warped], axis=0)
                #         spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
                #         st_curr = spike_times[spike_times >= L_prev] - L_prev

                #         spike_train = np.zeros(warped.shape, dtype=np.uint8)
                #         spike_train[st_curr] = 1

                #         # start_time = time.time()
                #         epsps = epsps_event_add(st_curr, warped.shape[0], kernel)

                #         if len(epsps) > max_time_over_cells:
                #             max_time_over_cells = len(epsps)
                #             print(f"trial {t} len(epsps) {len(epsps)}")
                #         rows.append(epsps.astype(np.float32, copy=False))

                #     X_trial = np.stack(rows, axis=0)  # (n_EC, padded time)


                #     dend_vm_per_dend_list = []
                #     for i in range(weights_EC.shape[1]):
                #         dend_vm_over_time = weights_EC[:,i] @ X_trial
                #         dend_vm_per_dend_list.append(dend_vm_over_time)
                #         if len(dend_vm_over_time) > max_length:
                #             max_length = len(dend_vm_over_time)

                #     dend_vm_per_trial_dict[t] = dend_vm_per_dend_list

                #     trial_wall_end   = time.time()
                #     trial_peak_after = _peak_rss_bytes()
                #     trial_curr_after = _curr_rss_bytes()

                #     trial_peak_delta      = (trial_peak_after - trial_peak_before) if (trial_peak_after and trial_peak_before) else None
                #     proc_peak_since_start = (trial_peak_after - proc_peak_start)   if (trial_peak_after and proc_peak_start) else None

                #     print(
                #         f"trial {t:02d}: T_warp={T_warp:6d}  time={trial_wall_end - trial_wall_start:6.3f}s  "
                #         f"RSS now={_fmt_bytes(trial_curr_after or 0)}  "
                #         f"trial peak Δ={_fmt_bytes(trial_peak_delta or 0)}  "
                #         f"proc peak={_fmt_bytes(trial_peak_after)}  "
                #         f"(+{_fmt_bytes(proc_peak_since_start or 0)} since start)"
                #     )




                # SEED = 42
                # np.random.seed(SEED)
                # random.seed(SEED)
                # rng = np.random.default_rng(SEED)

                # n_EC = 792
                # n_SST = 75
                # n_NDNF = 115
                # n_dendrites=100


                # EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
                # SST_input_matrix = np.stack(dend_list_SST[:n_SST], axis=0)
                # NDNF_input_matrix = np.stack(dend_list_NDNF[:n_NDNF], axis=0)

            
                # n_pos = EC_input_matrix.shape[1]
                # n_trials = EC_input_matrix.shape[2]
                # dx = 180.0 / n_pos  

                # vel = an_velocity
                # if vel.shape != (n_pos, n_trials):
                #     raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

                # L_prev = 500
  
                # precomputed_prepend = random_timeseries(np.mean(EC_input_matrix), np.std(EC_input_matrix) ,L_prev)

                # weights_EC = sample_weights(dist, n_EC, rng=rng)
                # max_length = 0
                # dend_vm_list = []  # list of (n_EC, T_fixed)

                # for t in range(EC_input_matrix.shape[2]):
                #     start_time = time.time()
                #     v_cm_s = _sanitize_velocity_cm_s(an_velocity[:, t])      
                #     dt_s   = dx / v_cm_s                             
                #     edges_s = np.concatenate(([0.0], np.cumsum(dt_s)))
                #     total_time = float(edges_s[-1])

                #     # constant-time axis in seconds
                #     t_axis = np.arange(0.0, total_time, dt_constant, dtype=np.float64)
                #     max_time_over_cells = 0

                #     firing_example = EC_input_matrix[0, :, t].astype(np.float64, copy=False)
                #     valid = np.isfinite(firing_example)
                #     time_points = np.cumsum(dt_s)
                #     warped_example = np.interp(t_axis, time_points[valid], firing_example[valid]).astype(np.float32, copy=False)

                #     two_track_length_example = np.concatenate([precomputed_prepend, warped_example], axis=0)
                #     t_ms = np.arange(two_track_length_example.shape[0]) * dt_ms


                #     rows= []
                #     for cell in range(n_EC):
                #         firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
                #         valid = np.isfinite(firing)
                #         if valid.sum() >= 2:
                #             time_points = np.cumsum(dt_s)  # length n_pos

                #             # start_time = time.time()
                #             warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
                #             # end_time = time.time()
                #             # print(f"interpolation_time = {end_time-start_time} ")
                #         else:
                #             warped = np.full(1, np.nan, dtype=np.float32)

                        
                #         two_track_length = np.concatenate([precomputed_prepend, warped], axis=0)
                #         spike_times = get_inhom_poisson_spike_times_by_thinning(two_track_length, t_ms, dt=dt_ms, refractory=3., generator=None, rng=rng).astype(int) 
                #         st_curr = spike_times[spike_times >= L_prev] - L_prev

                #         spike_train = np.zeros(warped.shape, dtype=np.uint8)
                #         spike_train[st_curr] = 1

                #         # start_time = time.time()
                #         epsps = epsps_event_add(st_curr, warped.shape[0], kernel)

                #         if len(epsps) > max_time_over_cells:
                #             max_time_over_cells = len(epsps)
                #             print(f"trial {t} len(epsps) {len(epsps)}")
                #         rows.append(epsps.astype(np.float32, copy=False))

                #     X_trial = np.stack(rows, axis=0)  # (n_EC, padded time)

                #     dend_vm_over_time = weights_EC @ X_trial

                #     if len(dend_vm_over_time) > max_length:
                #         max_length = len(dend_vm_over_time)

                #     dend_vm_list.append(dend_vm_over_time)
                #     end_time = time.time()
                #     print(f"total time {end_time-start_time}")
                
                # dend_vm_padded_list = []
                # for trial in range(len(dend_vm_list)):
                #     dend_vm = dend_vm_list[trial]
                #     if len(dend_vm) < max_length:
                #         padded_dend = np.pad(dend_vm, np.nan)
                #         dend_vm_padded_list.append(padded_dend)

                # dend_vm_padded_array = np.array(dend_vm_padded_list)

   
                # # 1. Pre-allocate the final array with NaNs.
                # # The shape should be (number of trials, max_length).
                # num_trials = len(dend_vm_list)
                # dend_vm_padded_array = np.full((num_trials, max_length), np.nan, dtype=np.float32)

                # # 2. Iterate and copy the data into the pre-allocated array.
                # for trial_index, dend_vm in enumerate(dend_vm_list):
                #     current_length = len(dend_vm)
                #     if current_length <= max_length:
                #         dend_vm_padded_array[trial_index, :current_length] = dend_vm


                # Vm, _, _ = activity_to_dend_vm_2d(
                #     dend_vm_padded_array,
                #     Vrest=-70.0,
                #     vm_scale=0.1,
                #     center_across="time")
                
                # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/vm_test/vm_test.pkl"


                # important_dict = {"dend_vm_padded_array":dend_vm_padded_array,
                #                   "Vm":Vm}

                # with open(save_path, 'wb') as f:
                #     pickle.dump(important_dict, f)
                #     print(f"pickle saved to {save_path}")
                    

                # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/vm_test/vm_test.pkl"
                # with open(save_path, 'rb') as f:
                #     important_dict = pickle.load(f)

                
                # Vm = important_dict["Vm"]
                # print(f"Vm.shape {Vm.shape}")

                # activity_NDNF=0
                # activity_SST=0 
                # NDNF_sf_opt=0 
                # SST_sf_opt=0 
                # NDNF_contribution_sum=0
                # SST_contribution_sum=0
                # weights_SST=0
                # weights_NDNF=0

                # --- setup (same as yours) ---

                

                    # end_time = time.time()
                    # # --- memory + timing report ---
                    # trial_wall_end = time.time()
                    # trial_peak_after = _get_peak_rss_bytes()
                    # trial_curr_after = _get_current_rss_bytes()

                    # trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
                    #     else (trial_peak_after - trial_peak_before)
                    # proc_peak_total  = trial_peak_after
                    # proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
                    #     else (trial_peak_after - proc_peak_start)

                    # print(
                    #     f"trial {t:02d}: T_warp={T_warp:6d}  time={trial_wall_end - trial_wall_start:6.3f}s  "
                    #     f"RSS now={_fmt_bytes(trial_curr_after)}  "
                    #     f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
                    #     f"proc peak={_fmt_bytes(proc_peak_total)}  "
                    #     f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
                    # )
                    # print(f"total time {end_time-start_time}")
                    

                    # dend_vm_array.append(dend_vm_over_time)
                    
                
                # dend_vm_padded_list = []
                # for trial in range(len(dend_vm_list)):
                #     dend_vm = dend_vm_list[trial]
                #     if len(dend_vm) < max_length:
                #         padded_dend = np.pad(dend_vm, np.nan)
                #         dend_vm_padded_list.append(padded_dend)

                # dend_vm_padded_array = np.array(dend_vm_padded_list)





# def get_dend_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, factors_dict_EC, factors_dict_SST, factors_dict_NDNF_newest, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, real_vel=None, constant_vel=None, use_residuals=True, use_model_EC=False, multiple_dendrites=False, add_inh=None, seed=0, SST_bias_factor=None, dist=None, use_averaged_velocity=None, make_it_spike=False):

#     if use_model_EC:

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
        

#         # rng = np.random.default_rng(seed=40)

#         # pos_bins = 150
#         # real_pos=50
#         # trials = 58
#         # dend_contribution_EC = np.zeros((pos_bins, trials))

#         # # Fixed Gaussian centers
#         # base_centers = np.arange(0, pos_bins+1, 5)  # bin 2, 7, 12, ..., 47
#         # width = 2.5
#         # for i in range(trials):
#         #     for center in base_centers:
#         #         # Add jitter to the center (±1 bin)
#         #         jittered_center = center + rng.choice([-1, 0, 1])

#         #         # Make sure jittered center stays within bounds
#         #         jittered_center = np.clip(jittered_center, 0, pos_bins - 1)

#         #         amplitude = rng.uniform(0.01, 0.02)  # still jitter amplitude too
#         #         gaussian = norm.pdf(np.arange(pos_bins), loc=jittered_center, scale=width)
#         #         gaussian /= np.max(gaussian)
#         #         dend_contribution_EC[:, i] += amplitude * gaussian

#         # final_dend = dend_contribution_EC[real_pos:real_pos*2, :]
#         # # Normalize globally if needed
#         # final_dend /= np.max(final_dend)
#         # if constant_vel:
#         #     an_velocity = np.full(final_dend.shape, 0.43)
#         # else:
#         #     an_velocity = np.tile(mean_new_average_vel_array[:,None], (1,58))
            
#         # return final_dend, an_velocity 
    
#     else:
#         if use_residuals:
#             dend_contribution_EC, an_velocity_EC, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_NDNF, an_velocity_NDNF, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_SST, an_velocity_SST, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            
#             EC = dend_contribution_EC.copy()
#             NDNF = dend_contribution_NDNF.copy()
#             SST = dend_contribution_SST.copy()
            
#             if add_inh=='neither':
#                 an_velocity = an_velocity_EC 
#                 SST_sf_opt = 0
#                 NDNF_sf_opt = 0
#                 NDNF_contribution_sum = 0
#                 SST_contribution_sum = 0

#             elif add_inh=='sst':
#                 SST_sf_opt, info = fit_sst_scale_to_cancel_ec(dend_contribution_EC, dend_contribution_SST)
#                 NDNF_sf_opt=0
#                 NDNF_contribution_sum = 0
#                 SST_contribution_sum = 0

#                 an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST], axis=0), axis=0)

#             else:
#                 # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
#                 # NDNF_sf_opt, SST_sf_opt = result.x

#                 res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
#                 NDNF_sf_opt = res["ndnf_sf"]
#                 SST_sf_opt  = res["sst_sf"]
#                 NDNF_contribution_sum = res["contrib_L2_ndnf"]
#                 SST_contribution_sum = res["contrib_L2_sst"]

#                 an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST, an_velocity_NDNF], axis=0), axis=0)


#         else:
#             dend_contribution_EC, an_velocity_EC, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_NDNF, an_velocity_SST, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_SST, an_velocity_NDNF, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            
#             EC = dend_contribution_EC.copy()
#             NDNF = dend_contribution_NDNF.copy()
#             SST = dend_contribution_SST.copy()

#             if add_inh=='neither':
#                 an_velocity = an_velocity_EC 
#                 SST_sf_opt = 0
#                 NDNF_sf_opt = 0
#                 NDNF_contribution_sum = 0
#                 SST_contribution_sum = 0

#             elif add_inh=='sst':
#                 SST_sf_opt, info = fit_sst_scale_to_cancel_ec(dend_contribution_EC, dend_contribution_SST)
#                 NDNF_sf_opt=0
#                 NDNF_contribution_sum = 0
#                 SST_contribution_sum = 0
#                 an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST], axis=0), axis=0)

#             else:
#                 # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
#                 # NDNF_sf_opt, SST_sf_opt = result.x
#                 res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
#                 NDNF_sf_opt = res["ndnf_sf"]
#                 SST_sf_opt  = res["sst_sf"]
#                 NDNF_contribution_sum = res["contrib_L2_ndnf"]
#                 SST_contribution_sum = res["contrib_L2_sst"]

#                 an_velocity = np.nanmean(np.stack([an_velocity_EC, an_velocity_SST, an_velocity_NDNF], axis=0), axis=0)


#         if multiple_dendrites:

#             dend_contribution_EC, an_velocity, dend_list_EC = get_dend_VM_cell_type(residual_activity_dict_EC, factors_dict_EC, mean_new_average_vel_array, GLM_params_EC, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_NDNF, an_velocity, dend_list_NDNF = get_dend_VM_cell_type(fixed_residual_activity_dict_NDNF_newest, factors_dict_NDNF_newest, mean_new_average_vel_array, GLM_params_NDNF_newest, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
#             dend_contribution_SST, an_velocity, dend_list_SST = get_dend_VM_cell_type(residual_activity_dict_SST, factors_dict_SST, mean_new_average_vel_array, GLM_params_SST, 1.0, amplitude=-0.5, vel=0.43, norm="min_max", add_vel_contribution=real_vel, const_vel=constant_vel, use_averaged_velocity=use_averaged_velocity, add_inh=add_inh, make_it_spike=make_it_spike)
            
#             #### fix this 
#             dt_constant=0.0001

#             tau_ms  = 5.0
#             dt_ms   = dt_constant * 1000.0      # 1 ms
#             AMP     = 1.0                      # mV
#             MODE    = "peak"                    # "area" or "peak"
#             kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

            


#             if make_it_spike:
                
                
#                 SEED = seed
#                 # SEED = 42
#                 np.random.seed(SEED); random.seed(SEED)
#                 rng = np.random.default_rng(SEED)

#                 n_EC = 792; n_SST = 75; n_NDNF = 115; n_dendrites = 100
#                 EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0).astype(np.float32, copy=False)  # (n_EC, n_pos, n_trials)
#                 n_pos, n_trials = EC_input_matrix.shape[1], EC_input_matrix.shape[2]
#                 dx = 180.0 / n_pos
#                 vel = an_velocity
#                 if vel.shape != (n_pos, n_trials):
#                     raise ValueError(f"velocity shape {vel.shape} != {(n_pos, n_trials)}")

#                 L_prev = 500
#                 dt_ms = np.float32(dt_constant * 1000.0)
#                 precomputed_prepend = np.asarray(
#                     random_timeseries(np.mean(EC_input_matrix), np.std(EC_input_matrix), L_prev),
#                     dtype=np.float32
#                 )

#                 # weights: (n_EC, n_dendrites)
#                 weights_EC = sample_weights(dist, n_EC, n_dendrites, rng=rng).astype(np.float32, copy=False)
#                 WT = weights_EC.T  # (n_dendrites, n_EC)

#                 # --- precompute warped lengths & axes per trial ---
#                 v_safe= np.apply_along_axis(_sanitize_velocity_cm_s, 0, vel)  # (n_pos, n_trials)
#                 dt_s_all = dx / v_safe
#                 time_points_all = np.cumsum(dt_s_all, axis=0)                    # (n_pos, n_trials)
#                 total_time = time_points_all[-1, :]                              # (n_trials,)
#                 T_warp_per_trial = np.floor(total_time / dt_constant + 1e-12).astype(np.int32)
#                 t_axis_list = [np.arange(int(T_warp_per_trial[t]), dtype=np.float64) * dt_constant for t in range(n_trials)]

#                 # --- reusable buffers ---
#                 T_warp_max = int(T_warp_per_trial.max())
#                 rows = np.zeros((n_EC, T_warp_max), dtype=np.float32)                # EPSPs per cell
#                 rate_buf = np.empty(L_prev + T_warp_max, dtype=np.float32)
#                 rate_buf[:L_prev] = precomputed_prepend[:L_prev]

#                 # --- results: dict of per-trial arrays (n_dendrites, T_warp[t]) ---
#                 vm_trials = {}               # vm_trials[t] -> array (n_dendrites, T_warp[t])

#                 # Optional: time blocking size to keep temporaries tiny
#                 B = 2048

#                 proc_peak_start = _get_peak_rss_bytes()

#                 for t in range(n_trials):
#                     trial_wall_start = time.time()
#                     trial_peak_before = _get_peak_rss_bytes()
#                     trial_curr_before = _get_current_rss_bytes()

#                     T_warp = int(T_warp_per_trial[t])
#                     T2     = L_prev + T_warp
#                     t_axis = t_axis_list[t]
#                     time_points = time_points_all[:, t]

#                     rows[:, :T_warp] = 0.0
#                     t_ms_vec = (np.arange(T2, dtype=np.float32) * dt_ms)

#                     for cell in range(n_EC):
#                         firing = EC_input_matrix[cell, :, t].astype(np.float64, copy=False)
#                         valid  = np.isfinite(firing)
#                         if valid.sum() < 2:
#                             continue

#                         warped = np.interp(t_axis, time_points[valid], firing[valid]).astype(np.float32, copy=False)
#                         r = rate_buf[:T2]
#                         r[L_prev:T2] = warped

#                         spike_times = get_inhom_poisson_spike_times_by_thinning(
#                             r, t_ms_vec, dt=dt_ms, refractory=3.0, generator=None, rng=rng
#                         ).astype(np.int32, copy=False)
#                         st_curr = spike_times[spike_times >= L_prev] - L_prev

#                         rows[cell, :T_warp] = epsps_event_add(st_curr, T_warp, kernel).astype(np.float32, copy=False)

#                     vm_t = np.empty((n_dendrites, T_warp), dtype=np.float16)
#                     for b in range(0, T_warp, B):
#                         e = min(b + B, T_warp)
#                         vm_t[:, b:e] = (WT @ rows[:, b:e]).astype(np.float16, copy=False)

#                     vm_trials[t] = vm_t.astype(np.float16, copy=False)

#                     del vm_t, warped, r, spike_times, st_curr

#                     # --- memory + timing report ---
#                     trial_wall_end = time.time()
#                     trial_peak_after = _get_peak_rss_bytes()
#                     trial_curr_after = _get_current_rss_bytes()

#                     trial_peak_delta = None if (trial_peak_before is None or trial_peak_after is None) \
#                         else (trial_peak_after - trial_peak_before)
#                     proc_peak_total  = trial_peak_after
#                     proc_peak_since_start = None if (proc_peak_start is None or trial_peak_after is None) \
#                         else (trial_peak_after - proc_peak_start)

#                     print(
#                         f"trial {t:02d}: T_warp={T_warp:6d}  time={trial_wall_end - trial_wall_start:6.3f}s  "
#                         f"RSS now={_fmt_bytes(trial_curr_after)}  "
#                         f"trial peak Δ={_fmt_bytes(trial_peak_delta)}  "
#                         f"proc peak={_fmt_bytes(proc_peak_total)}  "
#                         f"(+{_fmt_bytes(proc_peak_since_start)} since start)"
#                     )


                
#                 save_path = "/Users/michaelfinch/CA1-interneuron-GLM/vm_test/vm_test.pkl"


#                 with open(save_path, 'wb') as f:
#                     pickle.dump(vm_trials, f)
#                     print(f"pickle saved to {save_path}")

#                 Vm_dict = {}
#                 for i in range(dend_vm_array.shape[0]):
#                     Vm, _, _ = activity_to_dend_vm_2d(
#                         dend_vm_array[i,:,:],
#                         Vrest=-70.0,
#                         vm_scale=0.1,
#                         center_across="time")
#                     Vm_dict[i] = Vm
                

                    

#                 save_path = "/Users/michaelfinch/CA1-interneuron-GLM/vm_test/vm_test.pkl"
#                 with open(save_path, 'rb') as f:
#                     Vm = pickle.load(f)

                


#                 activity_EC = Vm
#                 weights_EC = WT
#                 activity_NDNF=0
#                 activity_SST=0 
#                 NDNF_sf_opt=0 
#                 SST_sf_opt=0 
#                 NDNF_contribution_sum=0
#                 SST_contribution_sum=0
#                 weights_SST=0
#                 weights_NDNF=0

#             else:
#                 SEED = 42
#                 np.random.seed(SEED)
#                 random.seed(SEED)
#                 rng = np.random.default_rng(SEED)

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
#                     # result = minimize(loss_fn, x0=[1.0, 1.0], bounds=[(0, 5), (0, 5)], args=(EC, NDNF, SST))
#                     # NDNF_sf_opt, SST_sf_opt = result.x
#                     res = fit_equal_contrib_L2(EC, NDNF, SST, SST_bias_factor=SST_bias_factor)
#                     NDNF_sf_opt = res["ndnf_sf"]
#                     SST_sf_opt  = res["sst_sf"]
#                     NDNF_contribution_sum = res["contrib_L2_ndnf"]
#                     SST_contribution_sum = res["contrib_L2_sst"]


#             return an_velocity, activity_EC, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF
#         else:
#             return dend_contribution_EC, dend_contribution_NDNF, dend_contribution_SST, an_velocity, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum
                