

# OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
# mpiexec -n 4 python -m mpi4py.futures -m nested.optimize \
#   --config-file-path=/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/ching_lung.yaml \
#   --framework=mpi \
#   --num-workers=3 \
#   --procs-per-worker=1 \
#   --pop_size=200 --path_length=3 --max_iter=50 \
#   --num_network_seeds=1 \
#   --disp

# python -m nested.analyze \
#   --config-file-path=ching_lung.yaml \
#   --framework=serial \
#   --num_network_seeds=1 \
#   --disp --plot --export --model-key=model_test_2 --param-file-path=model_params_ching_lung.yaml --plot_full_intermediates=True

# python -m nested.analyze \
#   --config-file-path=ching_lung.yaml \
#   --framework=serial \
#   --num_network_seeds=1 \
#   --disp --plot --model-key=model_test_2 --param-file-path=model_params_ching_lung.yaml --plot_full_intermediates=False --use_ching_lung=True --special_case=True


import nested
import optuna
from optuna.trial import TrialState
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict

from mpi4py import MPI
import os, resource, sys

from Exploring_BTSP import (objective, plot_full_kernel_bidirectional, plot_tau_buckets, fit_and_plot_btsp_paper_style_from_cleaned, fit_and_plot_full_kernel_paper_style, fit_exp_zerobase, exp_model, plot_fixed_data, get_W, plot_kernels_grid, fit_exp_fixed_c, exp_with_fixed_c, plot_jeff, plot_ching_lung) #plot_btsp_overlay_products_cleaned_mixed

import pickle
import matplotlib.pyplot as plt

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



context = Context()

def config_worker():

    # df_10  = load_two_numeric_cols(context.data_file_path + "ching_lung_csv/backwards_fig_2d_10hz.csv")
    # df_20  = load_two_numeric_cols(context.data_file_path + "ching_lung_csv/backwards_fig_2d_20hz.csv")
    # df_40  = load_two_numeric_cols(context.data_file_path + "ching_lung_csv/backwards_fig_2d_40hz.csv")
    # df_100  = load_two_numeric_cols(context.data_file_path + "ching_lung_csv/backwards_fig_2d_100hz.csv")
    # df_full  = load_two_numeric_cols(context.data_file_path + "ching_lung_csv/full_btsp_kernel_fig2c.csv")

    # df_list = [df_10, df_20, df_40, df_100] # df_full
    # hz_list = [10, 20, 40, 100, 20]
    # string_list = ["10Hz", "20Hz", "40Hz", "100Hz"] #, "20Hz"
    # taus_list  = [1.44, 1.75, 1.80, 1.03] 


    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/pickle_of_all_experimental_data.pkl"
    with open(save_path, 'rb') as f:
        cleaned_data_dict = pickle.load(f)

    save_path = "/Users/michaelfinch/CA1-interneuron-GLM/ching_lung/jeffs_data_csv.pkl"
    with open(save_path, 'rb') as f:
        jeffs_data_dict = pickle.load(f)


    context.update(locals())


def compute_features(params, model_id=None, export=False, plot=False):

    if not isinstance(params, dict):
        try:
            params = param_array_to_dict(params, context.param_names)
        except Exception as e:
            names = ["tau_et", "tau_is", "lam_et", "lam_is", "eta_ms"]
            params = {k: float(v) for k, v in zip(names, list(params))}

    use_ching_lung = str_true_false_to_bool(context.use_ching_lung)
    
    if plot:

        if use_ching_lung:

            plot_ching_lung(context, params)
            

        else:

            plot_jeff(context, params)


    else:
        if use_ching_lung:
            loss = objective(context.cleaned_data_dict, params, export=False, plot=False)
        else:
            # print(f"use_ching_lung {use_ching_lung}")
            loss = objective(context.jeffs_data_dict, params, export=False, plot=False)


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