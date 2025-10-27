#/ocean/projects/bio240068p/mfinch/six_configs

# (ca1_env) michaelfinch@diampillion Clean_notebooks_to_date % python -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --pop_size=1 --path_length=1 --max_iter=1 \
#   --network_start_seed=0 --network_end_seed=5 \
#   --vel_applied=real --animal_by_animal=False \
#   --constant_vel=False --include_beta=False --flat_input=True \
#   --disp


# mpiexec -n 4 python -m mpi4py.futures -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=mpi \
#   --pop_size=1 --path_length=1 --max_iter=1 \
#   --num-workers=3 \
#   --num_network_seeds=3 \
#   --disp --network_start_seed=0 --network_end_seed=2 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True

# Clean_notebooks_to_date % mpiexec -n 6 python -m mpi4py.futures -m nested.analyze \ 
#   --config-file-path=config/ec_nested.yaml \
#   --framework=mpi \   
#   --num-workers=5 \
#   --num_network_seeds=5 \
#   --disp --network_start_seed=0 --vel_applied='real' --param-file-path=model_key_yaml_opt.yaml --model-key=oct21_justec_5seeds_serial_number1_velTbetaFinputT --animal_by_animal=False --constant_vel=True --include_beta=False --flat_input=True --optimization_time=True

# mpiexec -n 4 python -m mpi4py.futures -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=mpi \
#   --num-workers=3 \
#   --pop_size=1 --path_length=1 --max_iter=1 \
#   --num_network_seeds=1 \
#   --disp --network_start_seed=0 --network_end_seed=2 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True

# python -m nested.analyze \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --num_network_seeds=1 \
#   --disp --network_start_seed=0 --network_end_seed=2 --plot --model-key=animal_6_precompute_spikes --param-file-path=model_key_yaml_opt.yaml --vel_applied='real' --dend_threshold=-69.0 --tau_ms=10.0 --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True 





#  Clean_notebooks_to_date % mpiexec -n 6 python -m mpi4py.futures -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=mpi \
#   --num-workers=5 \
#   --pop_size=6 --path_length=1 --max_iter=1 \
#   --num_network_seeds=2 \
#   --disp --network_start_seed=0 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True --debug=True

# python -m nested.analyze \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --num_network_seeds=2 \
#   --disp --network_start_seed=0 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True --debug=True --optimization_time=False --plot --param-file-path=model_key_yaml_opt.yaml --model-key=oct21_justec_5seeds_serial_number1_velTbetaFinputT

import nested
import optuna
from optuna.trial import TrialState
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict

from mpi4py import MPI
import os, resource, sys, time

from spiking_model_utils import load_data_regular
from build_a_model_object_per_animal import *


import os, time, psutil, gc

def _rank_tag():
    comm = MPI.COMM_WORLD
    return f"[rank={comm.Get_rank()} pid={os.getpid()}]"

def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def log_mem(tag):
    print(f"[{time.time():.3f}] {tag} RSS={rss_gb():.2f} GB", flush=True)

def deep_nbytes(obj, seen=None):
    """Crude deep size for numpy/list/tuple/dict (bytes)."""
    if seen is None: seen = set()
    oid = id(obj)
    if oid in seen: return 0
    seen.add(oid)
    if isinstance(obj, np.ndarray): return obj.nbytes
    if isinstance(obj, (list, tuple)): return sum(deep_nbytes(x, seen) for x in obj)
    if isinstance(obj, dict): return sum(deep_nbytes(k, seen)+deep_nbytes(v, seen) for k,v in obj.items())
    return 0


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

import pickle


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

context = Context()


def config_worker():

    start = int(context.network_start_seed)
    end   = start + int(context.num_network_seeds)
    seeds_array = list(range(start, end))

    GLM_params_SST, _, _, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(
        file_path=context.data_root, name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, _, _, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(
        file_path=context.data_root, name="EC_GLM", new_NDNF=False)
    GLM_params_NDNF_newest, _, _, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(
        file_path=context.data_root, name="NDNF_E1A1B", new_NDNF=True)

    fixed_residual_activity_dict_NDNF_newest = {f"animal_{idx+1}": residual_activity_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(residual_activity_dict_NDNF_newest)
                                               if 17 < idx < 31}
    fixed_filtered_factors_dict_NDNF_newest = {f"animal_{idx+1}": filtered_factors_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(filtered_factors_dict_NDNF_newest)
                                               if 17 < idx < 31}

    christine_save_path = (f"{context.data_root}/datasets/christines_overrepresentation_pkl.pkl")
    with open(christine_save_path, 'rb') as f:
        christine_overrepresentation_array = pickle.load(f)

    dx=180./50.

    mean_new_average_vel_array = get_real_velocity_array(filtered_factors_dict_EC, filtered_factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest)

    debug = str_true_false_to_bool(context.debug)
    if debug:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[rank={rank} pid={os.getpid()}] seeds_array={list(seeds_array)}", flush=True)

    context.update(locals())

import pickle
from types import SimpleNamespace



def get_args():
    seed_list = [context.seeds_array]
    return seed_list


def compute_features(params, network_seed, previous_features=None, model_id=None, export=False, plot=False):

  

    tau_ms = params[0]
    dend_threshold = params[1]
    weights_mean = params[2]
    weights_std  = params[3]

    dt_ms   = context.dt_constant * 1000.0
    AMP     = 1.0
    MODE    = "peak"

    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print("[params from optimizer]", params, flush=True)
        print("[context overrides]",
              {"tau_ms": getattr(context, "tau_ms", None),
               "dend_threshold": getattr(context, "dend_threshold", None)}, flush=True)

    cfg = {"dt_constant":context.dt_constant,
           "dx":context.dx,
           "store": context.store}

    flg = {"spikes": context.spikes,
    "epsps": context.epsps,
    "warp_axes": context.warp_axes}

    debug = str_true_false_to_bool(context.debug)

    if debug:
        log_mem("A: before building model")


    model = SpikeSimModel(kernel=kernel,dist_for_weights=context.dist,weights_SST=None, weights_NDNF=None,config=cfg, flags=flg)

    if debug:
        log_mem("B: after building empty model")

    model.residual_activity_dict_EC = context.residual_activity_dict_EC
    model.fixed_residual_activity_dict_NDNF_newest = context.fixed_residual_activity_dict_NDNF_newest
    model.residual_activity_dict_SST = context.residual_activity_dict_SST
    model.factors_dict_EC = context.factors_dict_EC
    model.factors_dict_SST = context.factors_dict_SST
    model.factors_dict_NDNF_newest = context.factors_dict_NDNF_newest
    model.GLM_params_EC = context.GLM_params_EC
    model.GLM_params_NDNF_newest = context.GLM_params_NDNF_newest
    model.GLM_params_SST = context.GLM_params_SST
    model.mean_new_average_vel_array = context.mean_new_average_vel_array

    model.real_vel = (context.vel_applied == "real")
    model.constant_vel = context.constant_vel
    # model.add_inh = context.add_inh             # ← was bare add_inh
    model.make_it_spike = context.make_it_spike
    # model.SST_bias_factor = context.SST_bias_multi
    model.dist = context.dist
    # model.vel_applied = context.vel_applied
    model.use_averaged_velocity = context.use_averaged_velocity
    # model.use_model_EC = context.use_model_EC
    model.tau_ms = tau_ms #context.tau_ms
    model.dend_threshold = dend_threshold #context.dend_threshold
    model.animal_by_animal = context.animal_by_animal
    model.input_animal = context.input_animal
    model.kernel = kernel
    model.dt_constant = context.dt_constant
    model.include_beta = context.include_beta
    model.flat_input = context.flat_input
    model.mean = weights_mean
    model.std = weights_std
    model.include_inh = False

    model.NDNF_sf_opt = 0
    model.SST_sf_opt = 0


    optimization_time = str_true_false_to_bool(context.optimization_time)

    t0 = time.time()
    rank = MPI.COMM_WORLD.Get_rank()
    host = MPI.Get_processor_name()
    pid  = os.getpid()
    


    model.optimization_time = optimization_time

    print(f"model.optimization_time {model.optimization_time}")

    if debug:
        log_mem("C: after attaching attrs, pre simulate")

    if model.optimization_time:
        (dend_activity, plateau_positions_counter, padded_warped_activity_list,
        start_pos_cnt50_dict, _plateau_arr_list_dict,
        dendrite_plateau_mask, num_plateaus_per_dend_list, plateau_start_times_list_mega_list,
        last_EPSP, weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST,
        activity_NDNF) = model._simulate_one_seed(int(network_seed), debug=debug)

        


        t1 = time.time()
        features_dict = {
        "seed": network_seed,
        "rank": rank,
        "pid": pid,
        "host": host,
        "t_start": t0,
        "t_end": t1,
        "duration_s": t1 - t0,
        "start_pos_cnt50_dict":start_pos_cnt50_dict,
        "_plateau_arr_list_dict":_plateau_arr_list_dict,
        }
        return features_dict

    else:
        (dend_activity, plateau_positions_counter, padded_warped_activity_list,
                start_pos_cnt50_dict, _plateau_arr_list_dict,
                dendrite_plateau_mask, num_plateaus_per_dend_list, plateau_start_times_list_mega_list,
                last_EPSP, weights_EC, weights_SST, weights_NDNF, an_velocity, activity_SST,
                activity_NDNF, warped_list) = model._simulate_one_seed(int(network_seed), debug=debug)

        
                

        if debug:
            log_mem("D: after simulate returned")

        t1 = time.time()
        features_dict = {
            "seed": network_seed,
            "rank": rank,
            "pid": pid,
            "host": host,
            "t_start": t0,
            "t_end": t1,
            "duration_s": t1 - t0,
            "start_pos_cnt50_dict":start_pos_cnt50_dict,
            "_plateau_arr_list_dict":_plateau_arr_list_dict,
            "plateau_positions_counter":plateau_positions_counter,
            "dendrite_plateau_mask":dendrite_plateau_mask,
            "num_plateaus_per_dend_list":num_plateaus_per_dend_list,
            "plateau_start_times_list_mega_list":plateau_start_times_list_mega_list,
            "last_EPSP":last_EPSP,
            "weights_EC":weights_EC,
            "weights_SST":weights_SST,
            "weights_NDNF":weights_NDNF,
            "an_velocity":an_velocity,
            "activity_SST":activity_SST,
            "activity_NDNF":activity_NDNF,
            "SST_sf_opt" : model.SST_sf_opt,
            "NDNF_sf_opt": model.NDNF_sf_opt,
            # dend_contribution_EC_dict[seed] = dend_contribution_EC
            "dend_activity":dend_activity,
            "warped_list":warped_list,
            "activity_SST":activity_SST,
            "activity_NDNF":activity_NDNF,
            # dend_contribution_EC_dict[seed] = dend_contribution_EC
            "tau_ms":tau_ms,
            "dend_threshold":dend_threshold,
            "dist":context.dist,
            "residual_activity_dict_EC":context.residual_activity_dict_EC,
            "constant_vel":context.constant_vel,
            "include_beta":context.include_beta, 
            "flat_input":context.flat_input,
            "dt_constant":context.dt_constant}
        
        
        return features_dict
        

def get_objectives(features, model_id=None, export=False, plot=False):

    objectives = {}
    objectives['total_error'] = features["total_error"]

    return features, objectives



def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):
    """Average across seeds (nested calls this once per trial with the list from get_args())."""

    try:
        rows = []
        for fd in features_dict_list:
            rows.append((fd["seed"], fd["rank"], fd["pid"], fd["host"],
                         fd["t_start"], fd["t_end"], fd["duration_s"]))
        # sort by start time
        rows.sort(key=lambda r: r[4])
        print("\n[filter_features] seed run summary (sorted by start time):", flush=True)
        for s, r, p, h, ts, te, dur in rows:
            print(f"  seed={s:3d} rank={r:3d} pid={p:6d} host={h:>20s}  "
                  f"start={ts:.3f}  end={te:.3f}  dur={dur:.2f}s", flush=True)

        # quick concurrency check: count overlaps with previous job end
        overlaps = 0
        for i in range(1, len(rows)):
            if rows[i][4] < rows[i-1][5]:  # start_i < end_(i-1)
                overlaps += 1
        uniq_hosts = {h for *_, h, _, _, _ in rows}
        uniq_ranks = {r for *_, r, _, _, _ in rows}

        print(f"[filter_features] unique seeds={len(rows)}  unique ranks={len(uniq_ranks)}  "
              f"unique hosts={len(uniq_hosts)}  overlaps_detected={overlaps}", flush=True)
    except Exception as e:
        print(f"[filter_features] telemetry print failed: {e}", flush=True)

    
    
    christine_overrepresentation_array = context.christine_overrepresentation_array

    # seed_list = [seed for seed in list(context.seeds_array)]

    dendrites_with_plateau_count = 0
    total_dends = 0

    summed_plateaus_over_seeds = []


    for seed in range(len(features_dict_list)):
        plateau_list = features_dict_list[seed]["_plateau_arr_list_dict"]
        for dendrite in range(len(plateau_list)):
            dendrite_plateau_array = plateau_list[dendrite]
            if np.any(dendrite_plateau_array==1):
                dendrites_with_plateau_count+=1

            total_dends +=1

        start_pos_cnt50_list = features_dict_list[seed]["start_pos_cnt50_dict"]

        n_bins = 10
        bin_size = int(50 / n_bins)
        summed_plateaus = np.zeros(n_bins)
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size
            summed_data = np.sum(start_pos_cnt50_list[start:end])
            summed_plateaus[i] = summed_data
        summed_plateaus_over_seeds.append(summed_plateaus)

    # print(f"plateau_start_times_list_mega_list len {len(plateau_start_times_list_mega_list)} plateau_start_times_list_mega_list[0]  {len(plateau_start_times_list_mega_list[0])} plateau_start_times_list_mega_list[0][0]  {plateau_start_times_list_mega_list[0][0].shape}")

    # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/plat_array.pkl"

    # with open(save_path, 'wb') as f:
    #     pickle.dump(features_dict_list[seed]["_plateau_arr_list_dict"], f)
    # print(f" pickle dumped to {save_path}")


        

    frac_dends_with_plateau = dendrites_with_plateau_count / total_dends

    arr = np.asarray(summed_plateaus_over_seeds)  # expect (n_seeds, n_bins)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")

    if arr.shape[0] in (10, 50) and arr.shape[1] not in (10, 50):
        arr = arr.T

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

    features = {}

    loss = np.mean((p_model_allcells - p_chr_allcells)**2)


    # plt.figure()
    # plt.title(f"loss {loss}")
    # plt.plot(p_model_allcells, color='b', label='Model')
    # plt.plot(p_chr_allcells, color='k', label='Experimental Target')
    # plt.show()


    features['total_error'] = loss



    if plot:

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
        dend_activity_dict = {}
        warped_list_dict = {}


        for seed in range(len(features_dict_list)):
            
            _pos_cnt_dict[seed] = features_dict_list[seed]["plateau_positions_counter"]
            # padded_warped_activity_list_dict[seed] = features_dict_list[seed]["padded_warped_activity_list"]
            start_pos_cnt50_dict[seed] = features_dict_list[seed]["start_pos_cnt50_dict"]
            _plateau_arr_list_dict[seed] = features_dict_list[seed]["_plateau_arr_list_dict"]
            _mask_dict[seed] = features_dict_list[seed]["dendrite_plateau_mask"]
            num_plateaus_per_dend_dict[seed] = features_dict_list[seed]["num_plateaus_per_dend_list"]
            _starts_list_dict[seed] = features_dict_list[seed]["plateau_start_times_list_mega_list"]
            last_EPSP = features_dict_list[seed]["last_EPSP"]
            weights_EC_dict[seed] = features_dict_list[seed]["weights_EC"]
            weights_SST_dict[seed] = features_dict_list[seed]["weights_SST"]
            weights_NDNF_dict[seed] = features_dict_list[seed]["weights_NDNF"]
            an_velocity_dict[seed] = features_dict_list[seed]["an_velocity"]
            activity_SST_dict[seed] = features_dict_list[seed]["activity_SST"]
            activity_NDNF_dict[seed] = features_dict_list[seed]["activity_NDNF"]
            SST_sf_opt_dict[seed] = features_dict_list[seed]["SST_sf_opt"]
            NDNF_sf_opt_dict[seed] = features_dict_list[seed]["NDNF_sf_opt"]
            # dend_contribution_EC_dict[seed] = dend_contribution_EC
            dend_activity_dict[seed] = features_dict_list[seed]["dend_activity"]
            warped_list_dict[seed] = features_dict_list[seed]["warped_list"]
            tau_ms = features_dict_list[seed]["tau_ms"]
            dend_threshold = features_dict_list[seed]["dend_threshold"]
            dist = features_dict_list[seed]["dist"]
            residual_activity_dict_EC = features_dict_list[seed]["residual_activity_dict_EC"]
            constant_vel = features_dict_list[seed]["constant_vel"]
            flat_input = features_dict_list[seed]["flat_input"]
            include_beta = features_dict_list[seed]["include_beta"]
            dt_constant = features_dict_list[seed]["dt_constant"]

        # model.last_EPSP = important_dict["last_EPSP"]
        # model.weights_EC=important_dict["weights_EC"]
        # model.weights_SST = important_dict["weights_SST_dict"]
        # model.weights_NDNF = important_dict["weights_NDNF_dict"]
        # model.dend_vm_per_seed_dict = important_dict["dend_vm_per_seed_dict"]
        # model.NDNF_sf_opt = important_dict["NDNF_sf_opt_dict"]
        # model.activity_SST = important_dict["activity_SST_dict"]
        # model.activity_NDNF = important_dict["activity_NDNF_dict"]
        # model.SST_sf_opt = important_dict["SST_sf_opt_dict"]
        # model.padded_warped_activity_list = important_dict["padded_warped_activity_list_dict"]
        # model._pos_cnt_dict = important_dict["_pos_cnt_dict"]
        # model.start_pos_cnt50_dict = important_dict["start_pos_cnt50_dict"]
        # model._plateau_arr_list_dict = important_dict["_plateau_arr_list_dict"]
        # model._mask_dict = important_dict["_mask_dict"]
        # model._starts_list_dict = important_dict["_starts_list_dict"]
        # model.num_plateaus_per_dend_list = important_dict["num_plateaus_per_dend_dict"]
        # model.warped_list_dict = important_dict["warped_list_dict"]

        input_animal=0
        animal_by_animal = False

        save_path2 = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/warped_pkl.pkl"
        with open(save_path2, 'wb') as f:
            pickle.dump(warped_list_dict, f)

        important_dict = dict(
        warped_list_dict=warped_list_dict,
        residual_activity_dict_EC=residual_activity_dict_EC,
        tau_ms=tau_ms,
        seeds=context.seeds_array,
        last_EPSP=last_EPSP,
        weights_EC_dict=weights_EC_dict,
        weights_SST_dict=weights_SST_dict,
        weights_NDNF_dict=weights_NDNF_dict,
        dend_vm_per_seed_dict=dend_vm_per_seed_dict,
        activity_SST_dict=activity_SST_dict,
        activity_NDNF_dict=activity_NDNF_dict,
        SST_sf_opt_dict=SST_sf_opt_dict,
        NDNF_sf_opt_dict=NDNF_sf_opt_dict,
        padded_warped_activity_list_dict=padded_warped_activity_list_dict,
        an_velocity_dict=an_velocity_dict,
        dend_threshold=dend_threshold,
        _pos_cnt_dict=_pos_cnt_dict,
        start_pos_cnt50_dict=start_pos_cnt50_dict,
        _plateau_arr_list_dict=_plateau_arr_list_dict,
        _mask_dict=_mask_dict,
        _starts_list_dict=_starts_list_dict,
        dist=dist,
        num_plateaus_per_dend_dict=num_plateaus_per_dend_dict,
        animal=0,
        example_cell=17,
        include_inhibition=False,
        NDNF_contribution_sum=None,
        SST_contribution_sum=None,
        animal_by_animal=animal_by_animal,
        constant_vel=constant_vel,
        include_beta=include_beta,
        flat_input=flat_input,
        dt_constant=dt_constant)
        

        # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/dend_activity_dict.pkl"
        # with open(save_path, 'wb') as f:
        #     pickle.dump(dend_activity_dict, f)

        plot_multidendrite_EC_err_across_seeds(loss, context.christine_overrepresentation_array, warped_list_dict, residual_activity_dict_EC, tau_ms = tau_ms,
        seeds = context.seeds_array, last_EPSP = last_EPSP, weights_EC = weights_EC_dict[0], weights_SST = weights_SST_dict[0], weights_NDNF =weights_NDNF_dict[0], dend_vm_per_seed_dict = dend_activity_dict, activity_SST = activity_SST_dict[0], activity_NDNF = activity_NDNF_dict[0], SST_sf_opt = SST_sf_opt_dict[0], NDNF_sf_opt = NDNF_sf_opt_dict[0],
        padded_warped_activity_list = padded_warped_activity_list_dict, an_velocity = an_velocity_dict[0], dend_threshold = dend_threshold,
        _pos_cnt_dict = _pos_cnt_dict, start_pos_cnt50_dict = start_pos_cnt50_dict, _plateau_arr_list_dict = _plateau_arr_list_dict, _mask_dict = _mask_dict, _starts_list_dict = _starts_list_dict,
        dist = dist, num_plateaus_per_dend_list = num_plateaus_per_dend_dict, animal=input_animal, example_cell=17, include_inhibition=False, #include inhibiiton,
        NDNF_contribution_sum = None, #state["NDNF_contribution_sum"], 
        SST_contribution_sum = None, #state["SST_contribution_sum"], 
        animal_by_animal = animal_by_animal, constant_vel=constant_vel, include_beta=include_beta, flat_input=flat_input, dt_constant=dt_constant) #state["animal_by_animal"])

    


    return features 

    