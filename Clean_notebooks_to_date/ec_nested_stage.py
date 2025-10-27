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

import nested
import optuna
from optuna.trial import TrialState
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict

# from build_a_model_object import (exp_kernel, load_yaml_cfg)

from mpi4py import MPI
import os, resource, sys

#from Fixing_dend_models_presentation import *
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
    end   = int(context.network_end_seed)      
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

    seeds_array = np.arange(int(context.network_start_seed), int(context.network_end_seed))

    debug = str_true_false_to_bool(context.debug)
    if debug:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"[rank={rank} pid={os.getpid()}] seeds_array={list(seeds_array)}", flush=True)

    context.update(locals())


def get_args():
    context.seed_range = context.seeds_array   
    return [[0]]  

def compute_features(params, network_seed, model_id=None, export=False, plot=False):

    load = str_true_false_to_bool(context.load)

    if load:
        with open(context.save_path, 'rb') as f:
            model = pickle.load(f)
            

        plot_multidendrite_EC_err_across_seeds(model.warped_list_dict[0], model.residual_activity_dict_EC, tau_ms = model.tau_ms,
            seeds = model.seeds_array, last_EPSP = model.last_EPSP[0], weights_EC = model.weights_EC[0], weights_SST = model.weights_SST[0], weights_NDNF =model.weights_NDNF[0], dend_vm_per_seed_dict = model.dend_activity_dict, activity_SST = model.activity_SST[0], activity_NDNF = model.activity_NDNF[0], SST_sf_opt = model.SST_sf_opt[0], NDNF_sf_opt = model.NDNF_sf_opt[0],
            padded_warped_activity_list = model.padded_warped_activity_list, an_velocity = model.an_velocity_dict[0], dend_threshold = model.dend_threshold,
            _pos_cnt_dict = model._pos_cnt_dict, start_pos_cnt50_dict = model.start_pos_cnt50_dict, _plateau_arr_list_dict = model._plateau_arr_list_dict, _mask_dict = model._mask_dict, _starts_list_dict = model._starts_list_dict,
            dist = model.dist, num_plateaus_per_dend_list = model.num_plateaus_per_dend_list, animal=model.input_animal, example_cell=17, include_inhibition=False, #include inhibiiton,
            NDNF_contribution_sum = None, #state["NDNF_contribution_sum"], 
            SST_contribution_sum = None, #state["SST_contribution_sum"], 
            animal_by_animal = model.animal_by_animal, constant_vel=model.constant_vel, include_beta=model.include_beta, flat_input=model.flat_input, dt_constant=model.dt_constant) #state["animal_by_animal"])
    
    else:


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
        model.add_inh = context.add_inh             # ← was bare add_inh
        model.make_it_spike = context.make_it_spike
        model.SST_bias_factor = context.SST_bias_multi
        model.dist = context.dist
        model.vel_applied = context.vel_applied
        model.use_averaged_velocity = context.use_averaged_velocity
        model.use_model_EC = context.use_model_EC
        model.tau_ms = tau_ms #context.tau_ms
        model.dend_threshold = dend_threshold #context.dend_threshold
        model.animal_by_animal = context.animal_by_animal
        model.input_animal = context.input_animal
        model.kernel = kernel
        model.dt_constant = context.dt_constant
        model.include_beta = context.include_beta
        model.flat_input = context.flat_input
        model.weights_mean = weights_mean
        model.weights_std = weights_std
        model.seeds_array = context.seeds_array

        optimization_time = str_true_false_to_bool(context.optimization_time)
        


        model.optimization_time = optimization_time

        if debug:
            log_mem("C: after attaching attrs, pre simulate")

        if model.optimization_time:
            model.simulate(seeds=context.seeds_array, export=False, plot=False, debug=debug) 
            
        else:
            important_dict = model.simulate(seeds=context.seeds_array, export=False, plot=False, debug=debug)  

        if debug:
            log_mem("D: after simulate returned")
        
        loss = model.evaluate(christine_overrepresentation_array=context.christine_overrepresentation_array,
        seeds=context.seeds_array,
        dend_threshold=model.dend_threshold)

        print(f"loss {loss}")



        if plot:

            model.last_EPSP = important_dict["last_EPSP"]
            model.weights_EC=important_dict["weights_EC"]
            model.weights_SST = important_dict["weights_SST_dict"]
            model.weights_NDNF = important_dict["weights_NDNF_dict"]
            model.dend_vm_per_seed_dict = important_dict["dend_vm_per_seed_dict"]
            model.NDNF_sf_opt = important_dict["NDNF_sf_opt_dict"]
            model.activity_SST = important_dict["activity_SST_dict"]
            model.activity_NDNF = important_dict["activity_NDNF_dict"]
            model.SST_sf_opt = important_dict["SST_sf_opt_dict"]
            model.padded_warped_activity_list = important_dict["padded_warped_activity_list_dict"]
            model._pos_cnt_dict = important_dict["_pos_cnt_dict"]
            model.start_pos_cnt50_dict = important_dict["start_pos_cnt50_dict"]
            model._plateau_arr_list_dict = important_dict["_plateau_arr_list_dict"]
            model._mask_dict = important_dict["_mask_dict"]
            model._starts_list_dict = important_dict["_starts_list_dict"]
            model.num_plateaus_per_dend_list = important_dict["num_plateaus_per_dend_dict"]
            model.warped_list_dict = important_dict["warped_list_dict"]

            save_path = context.save_path
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"saved to save_path: {save_path}")

            plot_multidendrite_EC_err_across_seeds(model.warped_list_dict[0], model.residual_activity_dict_EC, tau_ms = model.tau_ms,
            seeds = model.seeds_array, last_EPSP = model.last_EPSP[0], weights_EC = model.weights_EC[0], weights_SST = model.weights_SST[0], weights_NDNF =model.weights_NDNF[0], dend_vm_per_seed_dict = model.dend_activity_dict, activity_SST = model.activity_SST[0], activity_NDNF = model.activity_NDNF[0], SST_sf_opt = model.SST_sf_opt[0], NDNF_sf_opt = model.NDNF_sf_opt[0],
            padded_warped_activity_list = model.padded_warped_activity_list, an_velocity = model.an_velocity_dict[0], dend_threshold = model.dend_threshold,
            _pos_cnt_dict = model._pos_cnt_dict, start_pos_cnt50_dict = model.start_pos_cnt50_dict, _plateau_arr_list_dict = model._plateau_arr_list_dict, _mask_dict = model._mask_dict, _starts_list_dict = model._starts_list_dict,
            dist = model.dist, num_plateaus_per_dend_list = model.num_plateaus_per_dend_list, animal=model.input_animal, example_cell=17, include_inhibition=False, #include inhibiiton,
            NDNF_contribution_sum = None, #state["NDNF_contribution_sum"], 
            SST_contribution_sum = None, #state["SST_contribution_sum"], 
            animal_by_animal = model.animal_by_animal, constant_vel=model.constant_vel, include_beta=model.include_beta, flat_input=model.flat_input, dt_constant=model.dt_constant) #state["animal_by_animal"])

#activity_EC = model.dend_contribution_EC_dict[0],
    feats = {"total_error": float(loss)}

    return feats


def get_objectives(features, model_id=None, export=False, plot=False):

    objectives = {}
    objectives['total_error'] = features["total_error"]

    return features, objectives



# def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):
#     """Average across seeds (nested calls this once per trial with the list from get_args())."""
#     out = {}
#     keys = features_dict_list[0].keys()
#     for k in keys:
#         vals = [fd[k] for fd in features_dict_list if fd[k] is not None]
#         out[k] = float(np.mean(vals)) if vals else 0.0
#     return out


# def get_objectives_multi(features, model_id=None, export=False, plot=False):

#     new_features = dict()
#     objectives = dict()

#     for key, value in features.items():
#         if 'residual' in key:
#             objectives[key] = value
#         else:
#             new_features[key] = value


#     return new_features, objectives


# def get_objectives(features, model_id=None, export=False, plot=False):

#     features, multi_objectives = get_objectives_multi(features, model_id, export, plot)
#     objectives = {}
#     objectives['total_error'] = np.sum(list(multi_objectives.values()))

#     return features, objectives