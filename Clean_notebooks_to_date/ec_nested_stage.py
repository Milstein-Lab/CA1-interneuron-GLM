# python -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --pop_size=1 --path_length=1 --max_iter=1 \
#   --num_network_seeds=1 \
#   --disp --vel_applied='real' --animal_by_animal=True --input_animal='animal_1' --dend_threshold=-69. --network_start_seed=0 --network_end_seed=2

# python -m nested.analyze \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --num_network_seeds=1 \
#   --disp --network_start_seed=0 --network_end_seed=2  --animal_by_animal=True --input_animal='animal_2' --plot --model-key=animal_6_precompute_spikes --param-file-path=model_key_yaml_opt.yaml --vel_applied='real'

import nested
import optuna
from optuna.trial import TrialState
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict

from build_a_model_object import (exp_kernel, load_yaml_cfg)

from mpi4py import MPI
import os, resource, sys

from Fixing_dend_models_presentation import *
from spiking_model_utils import load_data_regular
from build_a_model_object_per_animal import *


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

context = Context()


def config_worker():

    GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(
        file_path=context.data_root, name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(
        file_path=context.data_root, name="EC_GLM", new_NDNF=False)
    GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(
        file_path=context.data_root, name="NDNF_E1A1B", new_NDNF=True)

    fixed_residual_activity_dict_NDNF_newest = {f"animal_{idx+1}": residual_activity_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(residual_activity_dict_NDNF_newest)
                                               if 17 < idx < 31}
    fixed_filtered_factors_dict_NDNF_newest = {f"animal_{idx+1}": filtered_factors_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(filtered_factors_dict_NDNF_newest)
                                               if 17 < idx < 31}

    tau_ms = float(context.tau_ms)

    dt_ms   = context.dt_constant * 1000.0
    AMP     = 1.0
    MODE    = "peak"
    kernel  = exp_kernel(tau_ms, dt_ms, n_taus=5, norm=MODE, target=AMP)



    dx=180./50.

    mean_new_average_vel_array = get_real_velocity_array(filtered_factors_dict_EC, filtered_factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest)

    seeds_array = np.arange(int(context.network_start_seed), int(context.network_end_seed))

    # cfg, flg = load_yaml_cfg("example_config.yaml")

    context.update(locals())

def get_args():
    """Controller-safe: derive seeds from kwargs, not from config_worker."""
    start = int(getattr(context, "network_start_seed", 0))
    num   = int(getattr(context, "num_network_seeds", 5))
    seed_range = list(range(start, start + num))
    # optional: stash for workers too (harmless on controller)
    context.seed_range = seed_range
    return [seed_range]

def compute_features(params, network_seed, model_id=None, export=False, plot=False):


    cfg = {"dt_constant":context.dt_constant,
           "dx":context.dx,
           "store": context.store}

    flg = {"spikes": context.spikes,
    "epsps": context.epsps,
    "warp_axes": context.warp_axes}


    model = SpikeSimModel(kernel=context.kernel,dist_for_weights=context.dist,weights_SST=None, weights_NDNF=None,config=cfg, flags=flg)

    # attach all attributes _simulate_one_seed expects
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
    model.tau_ms = context.tau_ms
    model.dend_threshold = context.dend_threshold
    model.animal_by_animal = context.animal_by_animal
    model.input_animal = context.input_animal
    model.kernel = context.kernel
    model.dt_constant = context.dt_constant
    model.include_beta = context.include_beta
    model.flat_input = context.flat_input


    important = model.simulate(seeds=context.seeds_array, export=False, plot=False)
    loss, metrics = model.evaluate(important, seeds=context.seeds_array, dend_threshold=context.dend_threshold)

    if context.export:
        state = model.__getstate__()
        state["loss"] = float(loss)
        state["metrics"] = {k: float(v) if v is not None else None for k, v in metrics.items()}

        with open(context.save_path, "wb") as f:
            pickle.dump(state, f)
        click.echo(f"Saved slim model to {context.save_path}")

    if context.plot:
        state = model.__getstate__()
        state["loss"] = float(loss)
        state["metrics"] = {k: float(v) if v is not None else None for k, v in metrics.items()}

        plot_multidendrite_EC_err_across_seeds(dend_list_EC_interp = state["dend_list_EC_interp"], tau_ms = state['tau_ms'],
        seeds = state["seeds"], last_EPSP = state["last_EPSP_dict"][0], weights_EC = state["weights_EC_dict"][0], weights_SST = state["weights_SST_dict"][0], weights_NDNF = state["weights_NDNF_dict"][0], dend_vm_per_seed_dict = state["dend_vm_per_seed_dict"],
        activity_EC = state["dend_contribution_EC_dict"][0], activity_SST = state["activity_SST"], activity_NDNF = state["activity_NDNF"], SST_sf_opt = state["SST_sf_opt"], NDNF_sf_opt = state["NDNF_sf_opt"],
        padded_warped_activity_list = state["padded_warped_activity_list_dict"], an_velocity = state["an_velocity_dict"][0], dend_threshold = state["dend_threshold"],
        _pos_cnt_dict = state["_pos_cnt_dict"], start_pos_cnt50_dict = state["start_pos_cnt50_dict"], _plateau_arr_list_dict = state["_plateau_arr_list_dict"], _mask_dict = state["_mask_dict"], _starts_list_dict = state["_starts_list_dict"],
        dist = state["dist"], num_plateaus_per_dend_list = state["num_plateaus_per_dend_list_dict"], animal=state["input_animal"], example_cell=17, include_inhibition=False, #include inhibiiton,
        NDNF_contribution_sum = None, #state["NDNF_contribution_sum"], 
        SST_contribution_sum = None, #state["SST_contribution_sum"], 
        animal_by_animal = state["animal_by_animal"], include_beta = state["include_beta"], flat_input=state["include_beta"], constant_vel=state["constant_vel"]) #state["animal_by_animal"])



    feats = {"total_error": float(loss)}
    # for k, v in metrics.items():
    #     # ensure JSON-serializable
    #     feats[f"m_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else v
    return feats



def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):
    """Average across seeds (nested calls this once per trial with the list from get_args())."""
    out = {}
    keys = features_dict_list[0].keys()
    for k in keys:
        vals = [fd[k] for fd in features_dict_list if fd[k] is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def get_objectives_multi(features, model_id=None, export=False, plot=False):

    new_features = dict()
    objectives = dict()

    for key, value in features.items():
        if 'residual' in key:
            objectives[key] = value
        else:
            new_features[key] = value


    return new_features, objectives


def get_objectives(features, model_id=None, export=False, plot=False):

    features, multi_objectives = get_objectives_multi(features, model_id, export, plot)
    objectives = {}
    objectives['total_error'] = np.sum(list(multi_objectives.values()))

    return features, objectives