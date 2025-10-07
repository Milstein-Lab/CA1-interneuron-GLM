#works for reproducing the data from the optimization
#(ca1_env) michaelfinch@nbp-25-225-169 Clean_notebooks_to_date % python -u optuna_multidendrite_spiking.py eval-pickle --pickle-path "/Users/michaelfinch/CA1-interneuron-GLM/optuna_param_pickles/animal_3_best_2.pkl" --animal animal_3 --data-root "/Users/michaelfinch/CA1-interneuron-GLM" --weight-dist Uniform --seed-list "200,201,202,203,204"

# run_optuna.py
import numpy as np
import optuna
from optuna.storages import RDBStorage
import click
import pickle
from joblib import Parallel, delayed
from pathlib import Path
import json 
import sys
import matplotlib.pyplot as plt


def normalize_dist_name(name) -> str:
    m = str(name).strip().lower()
    aliases = {
        "u": "uniform", "uni": "uniform",
        "n": "normal", "gaussian": "normal", "norm": "normal",
        "ln": "lognormal", "lognorm": "lognormal", "log-norm": "lognormal",
    }
    return aliases.get(m, m)


from multidendrite_spiking_main import SpikingModel, SpikingModelConfig
from multidendrite_spiking_utils import (
    get_velocity_array_every_animal, get_scaled_data_Hz_dict, do_the_interpolation_an,
    get_epsp_dict_animal, get_dend_vm_from_cells_multi, sample_weights,
    get_dendrite_activity_multi, activity_to_dend_vm, get_activity_multidendrite2_multiple_seeds, get_activity_multidendrite2, plot_multidendrite_EC_err_across_seeds, 
)

def _compare_metrics_from_pickle(saved, runtime_metrics, seeds_used=None, dist_used=None, tol=1e-9):
    import math
    # pull metrics stored in the pickle (if present)
    pickle_metrics = {}
    if isinstance(saved, dict):
        pickle_metrics = dict(saved.get("best_metrics", {}))

    # fixed display order; fall back to union if something new appears
    order = [
        "mean_total","violations","frac_active","f12_active",
        "mse_total","mse_frac","pen_sparsity","pen_12only"
    ]
    keys = order + sorted(k for k in set(pickle_metrics)|set(runtime_metrics) if k not in order)

    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.6g}"
        return "—" if x is None else str(x)

    print("\n[metrics compare]")
    print(f"{'metric':<16} {'pickle':>18} {'runtime':>18} {'match':>7} {'Δ(abs)':>12}")
    for k in keys:
        pv = pickle_metrics.get(k, None)
        rv = runtime_metrics.get(k, None)
        same = (pv == rv)
        delta = ""
        if isinstance(pv, (int,float)) and isinstance(rv, (int,float)):
            same = math.isclose(float(pv), float(rv), rel_tol=tol, abs_tol=tol)
            delta = f"{abs(float(pv)-float(rv)):.3g}"
        print(f"{k:<16} {_fmt(pv):>18} {_fmt(rv):>18} {'✓' if same else '✗':>7} {delta:>12}")

    # extras (seed list, weight dist) if present
    extra_pickle = {
        "seeds": (list(map(int, saved.get("seed_list_used"))) if isinstance(saved, dict) and saved.get("seed_list_used") else None),
        "weight_dist": (saved.get("weight_dist") if isinstance(saved, dict) else None),
    }
    extra_runtime = {
        "seeds": list(seeds_used) if seeds_used is not None else None,
        "weight_dist": dist_used,
    }
    print("\n[extras]")
    for k in ["seeds","weight_dist"]:
        pv, rv = extra_pickle.get(k), extra_runtime.get(k)
        same = pv == rv
        print(f"{k:<16} {_fmt(pv):>18} {_fmt(rv):>18} {'✓' if same else '✗':>7}")

def _side_by_side(pickle_params, runtime_params, extra_pickle=None, extra_runtime=None):
    import math, json
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.6g}"
        return str(x)
    keys = sorted(set(pickle_params) | set(runtime_params))
    print("\n[param compare]")
    print(f"{'key':<18} {'pickle':>18} {'runtime':>18} {'match':>7}")
    for k in keys:
        pv = pickle_params.get(k, None)
        rv = runtime_params.get(k, None)
        same = (pv == rv)
        # tolerate tiny float roundoff
        if isinstance(pv, (int,float)) and isinstance(rv, (int,float)):
            same = math.isclose(float(pv), float(rv), rel_tol=1e-9, abs_tol=1e-9)
        print(f"{k:<18} {_fmt(pv):>18} {_fmt(rv):>18} {'✓' if same else '✗'}")
    if extra_pickle or extra_runtime:
        print("\n[extra fields]")
        extras = sorted(set((extra_pickle or {}).keys()) | set((extra_runtime or {}).keys()))
        for k in extras:
            print(f"{k:<18} {_fmt((extra_pickle or {}).get(k,'—')):>18} {_fmt((extra_runtime or {}).get(k,'—')):>18} "
                  f"{'✓' if (extra_pickle or {}).get(k)==(extra_runtime or {}).get(k) else '✗'}")
    print()



def evaluate_params_for_seeds(params, static, seeds):
    """
    Recompute the objective on an explicit list of seeds, returning (loss, metrics)
    matching what the optimizer computed.
    """
    tau_ms         = float(params["tau_ms"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])
    dend_threshold = float(params["dend_threshold"])

    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]

    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    # def _eval_one_seed(s):
    #     epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(s))
    #     _, epsp_eTN, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"])
    #     if epsp_eTN.shape[1] != animal_velocity.shape[1]:
    #         epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

    #     rng = np.random.default_rng(12345 + int(s))
    #     connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
    #     weights_EC = sample_weights(static["dist"], connection_mask_EC, rng=rng,
    #                                 mean=weights_mean, std=weights_std)

    #     activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
    #     dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

    #     (_pos_cnt,
    #      start_pos_cnt50,
    #      _plateau_arr_list,
    #      _mask,
    #      _starts_list,
    #      num_plateaus_per_dend_list,
    #      _,
    #      _) = get_activity_multidendrite2(
    #         animal_velocity, dend_Vm,
    #         activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
    #         dt_constant=static["dt_constant"], dx=static["dx"], dend_threshold=dend_threshold,
    #         vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False
    #     )

    #     num_per_dend = np.asarray(num_plateaus_per_dend_list, float)
    #     total_starts = float(np.sum(start_pos_cnt50))
    #     frac10 = ten_bin_fraction_from_counter(np.asarray(start_pos_cnt50, float))
    #     return total_starts, frac10, num_per_dend

    def _eval_one_seed(params, seed, static):
        tau_ms         = float(params["tau_ms"])
        weights_mean   = float(params["weights_mean"])
        weights_std    = float(params["weights_std"])
        dend_threshold = float(params["dend_threshold"])

        n_dendrites     = int(static["n_dendrites"])
        n_EC            = int(static["n_EC"])
        animal_velocity = np.asarray(static["animal_velocity"], dtype=np.float32, order="C")
        pwa_cell_dict   = static["pwa_cell_dict"]

        # --- EPSPs (ensure float32 path) ---
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(seed))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(
            epsp_cells, Vrest=np.float32(static["vrest"]), epsp_sf=np.float32(static["epsp_sf"])
        )
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


    # run serially here for a deterministic replay; or use Parallel(..., prefer="threads")
    outs = [_eval_one_seed(s) for s in seeds]

    totals     = [o[0] for o in outs]
    frac_sum   = sum(o[1] for o in outs)
    violations = [np.maximum(0.0, o[2] - 2.0).sum() for o in outs]
    total_violations = float(np.sum(violations))

    active_fracs, f12_act_list = [], []
    for _t, _frac10, num_per_dend in outs:
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
        metrics = dict(mean_total=mean_total, violations=float(total_violations),
                       frac_active=frac_active, f12_active=f12_active,
                       mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)
    elif total_violations > 0:
        loss = float(1e6 * total_violations)
        metrics = dict(mean_total=mean_total, violations=float(total_violations),
                       frac_active=frac_active, f12_active=f12_active,
                       mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)
    else:
        mse_total = (mean_total - target_total) ** 2
        mean_frac = frac_sum / len(seeds) if len(seeds) > 0 else np.full(10, 0.1)
        mse_frac  = float(np.mean((mean_frac - target_frac) ** 2))
        w_sparsity, w_shape12 = 2.0, 1.0
        pen_sparsity = (frac_active - 0.30) ** 2
        pen_12only   = (1.0 - f12_active) ** 2
        loss = float(mse_total + mse_frac + w_sparsity*pen_sparsity + w_shape12*pen_12only)
        metrics = dict(mean_total=mean_total, violations=float(total_violations),
                       frac_active=frac_active, f12_active=f12_active,
                       mse_total=float(mse_total), mse_frac=float(mse_frac),
                       pen_sparsity=float(pen_sparsity), pen_12only=float(pen_12only))
    return loss, metrics


# def load_plot_pickled_params_by_seeds(save_path, static, seeds, animal, plot=True):
#     """
#     Optional plotting replay wrapper. Loads params from pickle, ensures dist, then calls the
#     multi-seed path you already wired up. Adjust internals as needed if your plotting util
#     expects dicts keyed by seed, etc.
#     """
#     with open(save_path, 'rb') as f:
#         saved = pickle.load(f)
#     params = saved["best_params"] if isinstance(saved, dict) and "best_params" in saved else saved

#     # honor dist saved in pickle if present
#     if isinstance(saved, dict) and "weight_dist" in saved:
#         static["dist"] = normalize_dist_name(saved["weight_dist"])

#     # If your plotting needs per-seed Vm dicts, re-create them here similarly to the commented code.
#     # For strict metric replay, prefer evaluate_params_for_seeds() and skip plotting.
#     loss, metrics = evaluate_params_for_seeds(params, static, seeds)
#     print(f"[eval] recomputed_loss={loss:.6g}")
#     print("[eval] recomputed_metrics:", json.dumps(metrics, indent=2))
#     return loss, metrics



def _compute_loss_metrics_from_outs(outs, n_dendrites):
    import numpy as np
    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    totals = []
    frac_sum = np.zeros(10, float)
    violations = 0.0
    active_fracs, f12_act_list = [], []

    def ten_bin_fraction_from_counter(cnt50):
        cnt50 = np.asarray(cnt50, float)
        agg10 = np.add.reduceat(cnt50, np.arange(0, 50, 5))
        s = agg10.sum()
        return agg10 / s if s > 0 else np.full(10, 0.1, float)

    for start_pos_cnt50, num_per_dend in outs:
        totals.append(float(np.sum(start_pos_cnt50)))
        frac_sum += ten_bin_fraction_from_counter(start_pos_cnt50)
        num_per_dend = np.asarray(num_per_dend, float)
        violations += float(np.maximum(0.0, num_per_dend - 2.0).sum())

        active_mask = (num_per_dend > 0)
        active_fracs.append(float(active_mask.mean()))
        f12_act_list.append(float(((num_per_dend==1) | (num_per_dend==2))[active_mask].mean()) if active_mask.any() else 0.0)

    mean_total = float(np.mean(totals)) if totals else 0.0
    frac_active = float(np.mean(active_fracs)) if active_fracs else 0.0
    f12_active  = float(np.mean(f12_act_list)) if f12_act_list else 0.0

    if mean_total < 1.0:
        loss = 1e7 + (mean_total - target_total) ** 2
        return loss, dict(mean_total=mean_total, violations=violations,
                          frac_active=frac_active, f12_active=f12_active,
                          mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)
    if violations > 0:
        loss = float(1e6 * violations)
        return loss, dict(mean_total=mean_total, violations=violations,
                          frac_active=frac_active, f12_active=f12_active,
                          mse_total=None, mse_frac=None, pen_sparsity=None, pen_12only=None)

    mse_total = (mean_total - target_total) ** 2
    mean_frac = frac_sum / len(outs) if len(outs) > 0 else np.full(10, 0.1)
    mse_frac  = float(np.mean((mean_frac - target_frac) ** 2))
    w_sparsity, w_shape12 = 2.0, 1.0
    pen_sparsity = (frac_active - 0.30) ** 2
    pen_12only   = (1.0 - f12_active) ** 2
    loss = float(mse_total + mse_frac + w_sparsity*pen_sparsity + w_shape12*pen_12only)

    return loss, dict(mean_total=mean_total, violations=violations,
                      frac_active=frac_active, f12_active=f12_active,
                      mse_total=float(mse_total), mse_frac=float(mse_frac),
                      pen_sparsity=float(pen_sparsity), pen_12only=float(pen_12only))


def load_plot_pickled_params_by_seeds(save_path, static, seeds, animal, plot=True, plot_file=None):
    # --- load once ---
    with open(save_path, 'rb') as f:
        saved = pickle.load(f)
    pickle_params = saved["best_params"] if isinstance(saved, dict) and "best_params" in saved else saved
    pickle_dist   = (saved.get("weight_dist") if isinstance(saved, dict) else None)
    pickle_seeds  = (saved.get("seed_list_used") if isinstance(saved, dict) else None)

    # prefer pickle's dist if present
    if pickle_dist:
        static["dist"] = normalize_dist_name(pickle_dist)

    # runtime params (exactly what we’ll use)
    params = {
        "tau_ms": float(pickle_params["tau_ms"]),
        "dend_threshold": float(pickle_params["dend_threshold"]),
        "weights_mean": float(pickle_params["weights_mean"]),
        "weights_std": float(pickle_params["weights_std"]),
    }
    _side_by_side(pickle_params, params,
                  extra_pickle={"weight_dist": normalize_dist_name(pickle_dist) if pickle_dist else None,
                                "seeds": list(map(int, pickle_seeds)) if pickle_seeds else None},
                  extra_runtime={"weight_dist": static["dist"], "seeds": list(map(int, seeds))})

    # --- statics ---
    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]

    # --- simulate each seed ONCE, keep artifacts for plotting and metrics ---
    dend_vm_per_seed_dict = {}
    last_weights_EC = None
    last_activity_EC = None
    outs_for_metrics = []  # (start_pos_cnt50, num_per_dend) per seed
    _pos_cnt_dict = {}
    start_pos_cnt50_dict = {}
    _plateau_arr_list_dict = {}
    _mask_dict = {}
    _starts_list_dict = {}
    num_plateaus_per_dend_list_dict = {}
    dend_activity_dict = {}
    padded_warped_activity_list_dict = {}
    last_EPSP = None

    for s in seeds:
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=params["tau_ms"], amp=1., seed=int(s))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"])
        last_EPSP = epsp_eTN
        if epsp_eTN.shape[1] != animal_velocity.shape[1]:
            epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

        rng = np.random.default_rng(12345 + int(s))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(static["dist"], connection_mask_EC, rng=rng,
                                    mean=params["weights_mean"], std=params["weights_std"])
        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
        dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

        # store artifacts
        dend_vm_per_seed_dict[int(s)] = dend_Vm
        last_weights_EC = weights_EC
        last_activity_EC = activity_EC

        # per-seed analysis (cheap; uses Vm only)
        (_pos_cnt,
         start_pos_cnt50,
         _plateau_arr_list,
         _mask,
         _starts_list,
         num_plateaus_per_dend_list,
        dend_activity,
        padded_warped_activity_list) = get_activity_multidendrite2(
            animal_velocity, dend_Vm,
            activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
            dt_constant=static["dt_constant"], dx=static["dx"], dend_threshold=params["dend_threshold"],
            vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False
        )
        outs_for_metrics.append((start_pos_cnt50, np.asarray(num_plateaus_per_dend_list, float)))

        _pos_cnt_dict[s] = _pos_cnt
        start_pos_cnt50_dict[s] = start_pos_cnt50
        _plateau_arr_list_dict[s] = _plateau_arr_list
        _mask_dict[s] = _mask
        _starts_list_dict[s] = _starts_list
        num_plateaus_per_dend_list_dict[s] = num_plateaus_per_dend_list
        dend_activity_dict[s] = dend_activity
        padded_warped_activity_list_dict[s] = padded_warped_activity_list

    # --- compute identical metrics from the artifacts (no second seed loop) ---
    loss, metrics = _compute_loss_metrics_from_outs(outs_for_metrics, n_dendrites)
    print(f"[eval] recomputed_loss={loss:.6g}")
    print("[eval] recomputed_metrics:", json.dumps(metrics, indent=2))

    print("[helper] load_plot_pickled_params_by_seeds: single-pass version")


    # ---- pack what you want to save ----
    important_dict = {
        "dend_vm_per_seed_dict": dend_vm_per_seed_dict,
        "last_weights_EC": last_weights_EC,
        "last_activity_EC": last_activity_EC,
        "outs_for_metrics": outs_for_metrics,
        "_pos_cnt_dict": _pos_cnt_dict,
        "start_pos_cnt50_dict": start_pos_cnt50_dict,
        "_plateau_arr_list_dict": _plateau_arr_list_dict,
        "_mask_dict": _mask_dict,
        "_starts_list_dict": _starts_list_dict,
        "num_plateaus_per_dend_list_dict": num_plateaus_per_dend_list_dict,
        "dend_activity_dict": dend_activity_dict,
        "padded_warped_activity_list_dict": padded_warped_activity_list_dict,
        "last_EPSP":last_EPSP
    }

    # ---- save ----
    save_path = Path("/Users/michaelfinch/CA1-interneuron-GLM/miscellaneous/seed_artifacts.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(important_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved → {save_path}")

    save_path = Path("/Users/michaelfinch/CA1-interneuron-GLM/miscellaneous/seed_artifacts.pkl")
    with open(save_path, "rb") as f:
        important_dict = pickle.load(f)

    # ---- unpack ----
    dend_vm_per_seed_dict = important_dict["dend_vm_per_seed_dict"]
    last_weights_EC = important_dict["last_weights_EC"]
    last_activity_EC = important_dict["last_activity_EC"]
    outs_for_metrics = important_dict["outs_for_metrics"]
    _pos_cnt_dict = important_dict["_pos_cnt_dict"]
    start_pos_cnt50_dict = important_dict["start_pos_cnt50_dict"]
    _plateau_arr_list_dict = important_dict["_plateau_arr_list_dict"]
    _mask_dict = important_dict["_mask_dict"]
    _starts_list_dict = important_dict["_starts_list_dict"]
    num_plateaus_per_dend_list_dict = important_dict["num_plateaus_per_dend_list_dict"]
    dend_activity_dict = important_dict["dend_activity_dict"]
    padded_warped_activity_list_dict = important_dict["padded_warped_activity_list_dict"]
    last_EPSP = important_dict["last_EPSP"]

    if not plot:
        return loss, metrics

    _ = plot_multidendrite_EC_err_across_seeds(pickle_params["tau_ms"], seeds,last_EPSP,
        last_weights_EC, 0, 0,
        dend_vm_per_seed_dict,
        last_activity_EC, 0, 0,
        0, 0,
        padded_warped_activity_list_dict,
        animal_velocity, dend_activity_dict,
        params["dend_threshold"], _pos_cnt_dict,
        start_pos_cnt50_dict,
        _plateau_arr_list_dict, _mask_dict, _starts_list_dict,
        static['dist'], num_plateaus_per_dend_list_dict,
        animal, example_cell=1,
        include_inhibition="neither",
        NDNF_contribution_sum=None, SST_contribution_sum=None,
        animal_by_animal=True
    )

    if plot_file:
        plt.savefig(plot_file, dpi=200, bbox_inches='tight'); print(f"[eval] plot saved → {plot_file}")
    else:
        plt.show()
    return loss, metrics


def ten_bin_fraction_from_counter(cnt50: np.ndarray) -> np.ndarray:
    cnt50 = np.asarray(cnt50, float)
    agg10 = np.add.reduceat(cnt50, np.arange(0, 50, 5))
    s = agg10.sum()
    return agg10 / s if s > 0 else np.full(10, 1/10, float)

def build_static_inputs_for_animal(spike_model: "SpikingModel", animal: str):
    cfg = spike_model.cfg
    factors_dict_EC  = spike_model.data["factors_dict_EC"]
    activity_dict_EC = spike_model.data["activity_dict_EC"]

    an_velocity_by_animal = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
    animal_velocity = an_velocity_by_animal[animal]  # (50, T)

    ret = get_scaled_data_Hz_dict({animal: activity_dict_EC[animal]}, Hz_SF=cfg.hz_sf)

    # handle both APIs: either a single dict or (dict, cells_per_animal_dict)
    if isinstance(ret, tuple) and len(ret) == 2:
        scaled_data_Hz_dict, cells_per_animal_dict = ret
        n_EC = int(cells_per_animal_dict[animal])
    else:
        scaled_data_Hz_dict = ret
        n_EC = len(scaled_data_Hz_dict[animal])

    # warp/interpolate
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
        dist=cfg.dist,
    )

def priority_loss_single_animal(params, static, seed_list, return_metrics: bool = False):
    """
    Compute loss for one animal by averaging over seeds.
    Optionally returns a metrics dict alongside the loss when return_metrics=True.
    """
    # --- trial params ---
    tau_ms         = float(params["tau_ms"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])
    dend_threshold = float(params["dend_threshold"])

    # --- statics ---
    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]

    # targets: ~30% active dendrites with ~1.5 plateaus each on average
    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    # run all seeds in threads (keeps a single Optuna DB writer)
    def _eval_one_seed(s):
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(s))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"])
        if epsp_eTN.shape[1] != animal_velocity.shape[1]:  # ensure (E, T, N)
            epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

        rng = np.random.default_rng(12345 + int(s))
        # rng = np.random.default_rng((int(s), 0))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(static["dist"], connection_mask_EC, rng=rng, mean=weights_mean, std=weights_std)

        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
        dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

        (_pos_cnt,
         start_pos_cnt50,
         _plateau_arr_list,
         _mask,
         _starts_list,
         num_plateaus_per_dend_list,
         _,
         _) = get_activity_multidendrite2(
            animal_velocity, dend_Vm,
            activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
            dt_constant=static["dt_constant"], dx=static["dx"], dend_threshold=dend_threshold,
            vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False
        )

        num_per_dend = np.asarray(num_plateaus_per_dend_list, float)
        total_starts = float(np.sum(start_pos_cnt50))
        frac10 = ten_bin_fraction_from_counter(np.asarray(start_pos_cnt50, float))
        return total_starts, frac10, num_per_dend

    outs = Parallel(n_jobs=static.get("inner_jobs", 1), prefer="threads")(
        delayed(_eval_one_seed)(s) for s in seed_list
    )

    totals       = [o[0] for o in outs]
    frac_sum     = sum(o[1] for o in outs)
    violations   = [np.maximum(0.0, o[2] - 2.0).sum() for o in outs]
    total_violations = float(np.sum(violations))

    # Active/sparsity metrics
    active_fracs = []
    f12_act_list = []
    for _t, _frac10, num_per_dend in outs:
        active_mask = (num_per_dend > 0)
        active_fracs.append(float(active_mask.mean()))
        if active_mask.any():
            f12 = (num_per_dend == 1) | (num_per_dend == 2)
            f12_act_list.append(float(f12[active_mask].mean()))
        else:
            f12_act_list.append(0.0)

    frac_active = float(np.mean(active_fracs)) if active_fracs else 0.0
    f12_active  = float(np.mean(f12_act_list)) if f12_act_list else 0.0

    # Optional debug print (first seed)
    if outs:
        print(f"[debug] mean_plateaus/dend={outs[0][2].mean():.3f}, total_starts={totals[0]:.0f}")

    mean_total = float(np.mean(totals)) if totals else 0.0

    # guards
    if mean_total < 1.0:
        loss = 1e7 + (mean_total - target_total) ** 2
        metrics = {
            "mean_total": mean_total, "violations": float(total_violations),
            "frac_active": frac_active, "f12_active": f12_active,
            "mse_total": None, "mse_frac": None,
            "pen_sparsity": None, "pen_12only": None,
        }
    elif total_violations > 0:
        loss = float(1e6 * total_violations)
        metrics = {
            "mean_total": mean_total, "violations": float(total_violations),
            "frac_active": frac_active, "f12_active": f12_active,
            "mse_total": None, "mse_frac": None,
            "pen_sparsity": None, "pen_12only": None,
        }
    else:
        mse_total = (mean_total - target_total) ** 2
        mean_frac = frac_sum / len(seed_list) if len(seed_list) > 0 else np.full(10, 0.1)
        mse_frac  = float(np.mean((mean_frac - target_frac) ** 2))

        w_sparsity = 2.0
        w_shape12  = 1.0
        pen_sparsity = (frac_active - 0.30) ** 2
        pen_12only  = (1.0 - f12_active) ** 2

        loss = float(mse_total + mse_frac + w_sparsity*pen_sparsity + w_shape12*pen_12only)
        metrics = {
            "mean_total": mean_total, "violations": float(total_violations),
            "frac_active": frac_active, "f12_active": f12_active,
            "mse_total": float(mse_total), "mse_frac": float(mse_frac),
            "pen_sparsity": float(pen_sparsity), "pen_12only": float(pen_12only),
        }

    return (loss, metrics) if return_metrics else loss

def priority_loss_multi_animal(params, statics_list, seed_list):
    losses = [priority_loss_single_animal(params, st, seed_list) for st in statics_list]
    return float(np.mean(losses))

def parse_seed_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]



@click.group(help="EC Optuna utilities.")
def cli():
    pass

@cli.command("run")
@click.option("--animal", default="animal_1",
              help="Animal key (e.g., 'animal_3') or 'all' to average across all animals.")
@click.option("--storage", default="sqlite:///ec_optuna.db", show_default=True,
              help="Optuna storage URL. Use Postgres for multi-node.")
@click.option("--study", default="ec_param_search", show_default=True,
              help="Optuna study name.")
@click.option("--trials", type=int, default=100, show_default=True)
@click.option("--n-jobs", type=int, default=1, show_default=True,
              help="Parallel workers on *this* node/process.")
@click.option("--seed-list", default="0,1,2,3,4", show_default=True,
              help="Comma-separated seeds averaged inside the objective.")
@click.option("--save-path", type=click.Path(), default="best_params.pkl",
              help="Path to pickle file for saving best params + value.")
@click.option("--data-root", default="/jet/home/mfinch/CA1-interneuron-GLM", show_default=True,
              help="Root directory that contains the 'datasets/' folder.")
@click.option("--inner-jobs", type=int, default=1, show_default=True,
              help="Threads for seed-level parallelism (use with --n-jobs 1).")
@click.option(
    "--weight-dist",
    default="Normal",
    type=click.Choice(["Uniform", "Normal", "Lognormal"], case_sensitive=False),
    show_default=True,
    help="Distribution used by sample_weights for EC→dendrite connections."
)

def run_cmd(animal, storage, study, trials, n_jobs, seed_list, save_path, data_root, inner_jobs, weight_dist):
    data_root = str(Path(data_root).expanduser().resolve())
    datasets_dir = Path(data_root) / "datasets"
    if not datasets_dir.exists():
        raise SystemExit(f"datasets/ not found at: {datasets_dir}. Put your .mat files there or pass --data-root <path>")

    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    all_animals = list(sm.data["activity_dict_EC"].keys())
    chosen = all_animals if animal.lower() == "all" else [animal] if animal in all_animals else None
    if chosen is None:
        raise SystemExit(f"--animal {animal} not found. Available: {all_animals}")

    # build statics then set dist/inner_jobs
    dist = normalize_dist_name(weight_dist)
    statics_list = [build_static_inputs_for_animal(sm, a) for a in chosen]
    for stc in statics_list:
        stc["inner_jobs"] = inner_jobs
        stc["dist"] = dist

    seeds = parse_seed_list(seed_list)

    if storage.startswith("sqlite") and n_jobs != 1:
        click.echo("⚠️ SQLite + --n-jobs>1 can lock. Forcing --n-jobs=1; use --inner-jobs for speed.", err=True)
        n_jobs = 1

    storage_backend = RDBStorage(url=storage, engine_kwargs={"connect_args": {"timeout": 120}})
    st = optuna.create_study(study_name=study, direction="minimize", storage=storage_backend, load_if_exists=True)

    def make_objective(statics_list, seeds):
        def objective(trial):
            params = dict(
                tau_ms=trial.suggest_float("tau_ms", 5.0, 200.0),
                dend_threshold=trial.suggest_float("dend_threshold", -75.0, -65.0),
                weights_mean=trial.suggest_float("weights_mean", 0.1, 1.0),
                weights_std=trial.suggest_float("weights_std", 0.1, 1.0),
            )
            if len(statics_list) == 1:
                loss, metrics = priority_loss_single_animal(params, statics_list[0], seeds, return_metrics=True)
                print(f"[trial {trial.number}] loss={loss:.3f} "
                      f"mean_total={metrics['mean_total']:.2f} "
                      f"active={metrics['frac_active']:.2%} "
                      f"f12={metrics['f12_active']:.2%}")
                for k, v in metrics.items():
                    trial.set_user_attr(k, v)
                return loss
            else:
                return priority_loss_multi_animal(params, statics_list, seeds)
        return objective

    st.optimize(make_objective(statics_list, seeds), n_trials=trials, n_jobs=n_jobs, show_progress_bar=False)

    best_params = st.best_params
    best_value  = st.best_value
    best_trial  = st.best_trial
    best_metrics = getattr(best_trial, "user_attrs", {})

    out = {
    "best_params": best_params,
    "best_value": best_value,
    "animal": animal,
    "best_metrics": best_metrics,
    "best_trial_number": best_trial.number,
    "weight_dist": dist,
    "seed_list_used": seeds,        # <--- add this
    "study_name": study,            # <--- optional but handy
    }

    with open(save_path, "wb") as f:
        pickle.dump(out, f)

    print(f"✅ Saved best trial → {save_path}")
    print("Best params:", best_params)
    print("Best value:", best_value)
    print("Best metrics:", best_metrics)



@cli.command("eval-pickle")
@click.option("--pickle-path", required=True, type=click.Path(exists=True))
@click.option("--animal", required=True)
@click.option("--data-root", required=True, type=click.Path(exists=True))
@click.option("--num-seeds", default=None, type=int, show_default=False)
@click.option("--seed-list", default=None)
@click.option("--no-plot", is_flag=True, default=False)
@click.option("--plot-file", type=click.Path(), default=None, help="Save plot to this file (e.g., results/replay.png)")
@click.option("--weight-dist",
              default="Normal",
              type=click.Choice(["Uniform", "Normal", "Lognormal"], case_sensitive=False),
              show_default=True)


def eval_pickle_cmd(pickle_path, animal, data_root, num_seeds, seed_list, no_plot, weight_dist, plot_file):
    data_root = str(Path(data_root).expanduser().resolve())
    datasets_dir = Path(data_root) / "datasets"
    if not datasets_dir.exists():
        raise SystemExit(f"datasets/ not found at: {datasets_dir}")

    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    static = build_static_inputs_for_animal(sm, animal)
    static["inner_jobs"] = 1
    static["dist"] = normalize_dist_name(weight_dist)

    # ---- pick seeds (CLI > pickle > num_seeds > default 5)
    if seed_list:
        seeds = [int(x) for x in seed_list.split(",") if x.strip()]
    else:
        with open(pickle_path, "rb") as f:
            saved = pickle.load(f)
        seeds = list(map(int, saved.get("seed_list_used", []))) or \
                (list(range(num_seeds)) if num_seeds is not None else [0,1,2,3,4])

    print(f"[eval] pickle={pickle_path} animal={animal} seeds={seeds} dist={static['dist']}")

    # >>> Update your evaluator to accept an explicit seeds list <<<
    # e.g., load_plot_pickled_params_by_seeds(pickle_path, static, seeds, animal, plot=not no_plot)
    load_plot_pickled_params_by_seeds(pickle_path, static, seeds, animal,
                                      plot=(not no_plot), plot_file=plot_file)


if __name__ == "__main__":
    cli()