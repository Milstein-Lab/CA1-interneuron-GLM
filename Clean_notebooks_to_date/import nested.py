import nested
# ec_nested_stage.py
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict
from multidendrite_spiking_main import SpikingModel, SpikingModelConfig
from multidendrite_spiking_utils import (get_velocity_array_every_animal, get_scaled_data_Hz_dict, do_the_interpolation_an, get_epsp_dict_animal, get_dend_vm_from_cells_multi, sample_weights,
    get_dendrite_activity_multi, activity_to_dend_vm, get_activity_multidendrite2,_eval_one_seed, _ten_bin_fraction_from_counter, _build_static_inputs, _build_static_inputs_animal_av, sizeof_pwa_bytes, _precompute_spike_trains_multi, _precompute_spike_trains_animal, rss_now_mb)
import optuna
from optuna.trial import TrialState
from mpi4py import MPI
import os, resource
import time 
from gc import collect
from collections import defaultdict

def _ensure_debug_state():
    if not hasattr(context, "_debug_per_model"):
        context._debug_per_model = defaultdict(list)
    if not hasattr(context, "_model_params"):
        context._model_params = {}
    if not hasattr(context, "_last_params"):
        context._last_params = {}
    if not hasattr(context, "_best_in_gen"):
        context._best_in_gen = {"value": float("inf"), "model_id": None, "params": {}, "metrics": {}}
    if not hasattr(context, "_gen_idx"): context._gen_idx = 0
    if not hasattr(context, "_trial_in_gen"): context._trial_in_gen = 0
    if not hasattr(context, "pop_size_hint"):
        context.pop_size_hint = int(getattr(context, "pop_size_hint", getattr(context, "pop_size", 1)))



context = Context()

import os, sys, resource, atexit

MEM_PEAK_MB = 0.0  # reset at worker init

def report_mem(tag=""):
    bump_mem_peak()
    print(f"[MEM] {tag} rss={current_rss_mb():.1f}MB "
          f"peak_seen={MEM_PEAK_MB:.1f}MB ru_maxrss={ru_maxrss_mb():.1f}MB",
          flush=True)

def current_rss_mb():
    # current resident memory
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except FileNotFoundError:
            return float("nan")
        return float("nan")

def ru_maxrss_mb():
    # OS-reported peak for this process
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024**2) if sys.platform == "darwin" else r / 1024.0

def bump_mem_peak():
    global MEM_PEAK_MB
    now = current_rss_mb()
    peak = ru_maxrss_mb()
    # Track our own max of “current” (per-gen) and OS peak (lifetime)
    MEM_PEAK_MB = max(MEM_PEAK_MB, now, peak)

@atexit.register
def _print_mem_peak_on_exit():
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        rank = -1
    print(f"[MEM-PEAK] rank={rank} pid={os.getpid()} max_seen={MEM_PEAK_MB:.1f} MB ru_maxrss={ru_maxrss_mb():.1f} MB", flush=True)


def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    # how many trials per generation
    pop = int(getattr(context, "pop_size_hint", getattr(context, "pop_size", 1)))

    # print a summary whenever a generation finishes
    if (trial.number + 1) % pop == 0:
        gen = (trial.number + 1)//pop - 1  # 0-based
        bt = study.best_trial
        print(f"\n[GEN-SUMMARY] Generation {gen} finished", flush=True)
        print(f"best model_id: {bt.number}", flush=True)
        print("params:", flush=True)
        for k, v in bt.params.items():
            print(f"  {k}: {v}", flush=True)
        print("objectives:", flush=True)
        print(f"  total_error: {bt.value}", flush=True)



def config_worker():
    
    """Runs once per process. Load data & set up seeds/static."""

    ec_avg = bool(getattr(context, "ec_animal_average", False))

    if ec_avg:
        print("[CTX]", {k: context().get(k) for k in (
            "ec_animal_average","weight_dist","inner_jobs",
            "network_start_seed","num_network_seeds","data_root"
        )}, flush=True)
    else:
        print("[CTX]", {k: context().get(k) for k in (
            "animal","animal_key","weight_dist","inner_jobs",
            "network_start_seed","num_network_seeds","data_root"
        )}, flush=True)

    data_root = str(Path(context.data_root).expanduser().resolve())
    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    if ec_avg:
        static = _build_static_inputs_animal_av(sm)
    else:
        static = _build_static_inputs(sm, getattr(context, "animal", None))

    
    static["inner_jobs"] = int(getattr(context, "inner_jobs", 1))
    static["ec_animal_average"] = ec_avg
    # weight_dist is a string like "Uniform"/"Normal"/"Lognormal"
    static["dist"] = str(getattr(context, "weight_dist", "Normal")).strip().lower()
    context.static = static

    start = int(getattr(context, "network_start_seed", 0))
    num   = int(getattr(context, "num_network_seeds", 5))
    context.seed_range = list(range(start, start + num))

    rank = MPI.COMM_WORLD.Get_rank()
    print(f"[INIT] rank={rank} pid={os.getpid()} RSS={rss_now_mb():.1f} MB "
          f"dist={context.static['dist']} seeds={getattr(context,'seed_range',None)}",
          flush=True)

    spike_seed = int(getattr(context, "spike_seed", 12345))
    dt_ms = float(context.static["dt_constant"] * 1000.0)

    # if ec_avg:
    #     context.static["spike_trains"] = _precompute_spike_trains_multi(static["pwa_cell_dict"], dt_ms, spike_seed)
    # else:
    #     context.static["spike_trains"] = _precompute_spike_trains_animal(static["pwa_cell_dict"], dt_ms, spike_seed)


    # NOW FREE the big float data:
    del static["pwa_cell_dict"]
    collect()

    # reset and bump after we’ve freed the big blob
    global MEM_PEAK_MB
    MEM_PEAK_MB = 0.0
    bump_mem_peak()

    if 'debug' not in context():

        context.debug = False
    context.update(locals())
    _ensure_debug_state()

def get_args():
    """Controller-safe: derive seeds from kwargs, not from config_worker."""
    start = int(getattr(context, "network_start_seed", 0))
    num   = int(getattr(context, "num_network_seeds", 5))
    seed_range = list(range(start, start + num))
    # optional: stash for workers too (harmless on controller)
    context.seed_range = seed_range
    return [seed_range]

# --- optional helper: lazy init if a process missed config_worker ---
def _ensure_initialized():
    if not hasattr(context, "static"):
        data_root = str(Path(context.data_root).expanduser().resolve())
        cfg = SpikingModelConfig(file_path=data_root)
        sm = SpikingModel(cfg); sm.load()
        ec_avg = bool(getattr(context, "ec_animal_average", False))
        if ec_avg:
            static = _build_static_inputs_animal_av(sm)
        else:
            static = _build_static_inputs(sm, getattr(context, "animal", None))
        static["inner_jobs"] = int(getattr(context, "inner_jobs", 1))
        static["dist"] = str(getattr(context, "weight_dist", "Normal")).strip().lower()
        static["ec_animal_average"] = ec_avg
        context.static = static
        _ensure_debug_state()   # <-- add this


    if not hasattr(context, "pop_size_hint"):
        context.pop_size_hint = int(getattr(context, "pop_size_hint", getattr(context, "pop_size", 1)))
    if not hasattr(context, "_gen_idx"): context._gen_idx = 0
    if not hasattr(context, "_trial_in_gen"): context._trial_in_gen = 0
    if not hasattr(context, "_best_in_gen"): context._best_in_gen = {"value": float("inf"), "model_id": None, "params": {}}
    if not hasattr(context, "_model_params"): context._model_params = {}


# at file top (once)
# context._debug_per_model will hold per-model per-seed debug info
if not hasattr(context, "_debug_per_model"):
    context._debug_per_model = defaultdict(list)

def compute_features(params, network_seed, model_id=None, export=False, plot=False):
    _ensure_initialized()
    _ensure_debug_state()
    t0 = time.perf_counter()
    paramsdict = param_array_to_dict(params, context.param_names)

    # stash params for summaries
    if not hasattr(context, "_model_params"):
        context._model_params = {}
    if model_id is not None:
        context._model_params[model_id] = paramsdict
        context._last_model_id = model_id
    context._last_params = paramsdict

    ec_avg = bool(context.static.get("ec_animal_average", False))
    start_pos_cnt50, num_per_dend = _eval_one_seed(paramsdict, int(network_seed), context.static, animal_average=ec_avg)

    # ---- metrics (same as before) ----
    n_dendrites  = int(context.static["n_dendrites"])
    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    total_starts = float(np.sum(start_pos_cnt50))
    frac10       = _ten_bin_fraction_from_counter(np.asarray(start_pos_cnt50, float))
    violations   = float(np.maximum(0.0, num_per_dend - 2.0).sum())

    active_mask  = (num_per_dend > 0)
    frac_active  = float(active_mask.mean())
    f12_active   = float(((num_per_dend == 1) | (num_per_dend == 2))[active_mask].mean()) if active_mask.any() else 0.0

    if total_starts < 1.0:
        loss = 1e7 + (total_starts - target_total) ** 2
        mse_total = mse_frac = pen_sparsity = pen_12only = None
    elif violations > 0:
        loss = float(1e6 * violations)
        mse_total = mse_frac = pen_sparsity = pen_12only = None
    else:
        mse_total   = (total_starts - target_total) ** 2
        mse_frac    = float(np.mean((frac10 - target_frac) ** 2))
        pen_sparsity= (frac_active - 0.30) ** 2
        pen_12only  = (1.0 - f12_active) ** 2
        loss        = float(mse_total + mse_frac + 2.0*pen_sparsity + 1.0*pen_12only)

    # ---- stash debug arrays for summaries (NOT returned) ----
    if model_id is not None:
        context._debug_per_model[model_id].append({
            "total_starts": total_starts,
            "violations": violations,
            "frac_active": frac_active,
            "f12_active": f12_active,
            "frac10_bins": frac10.copy(),     # array
            "target_bins": target_frac.copy(),# array (constant)
            "target_total": target_total,     # scalar (constant for this run)
        })

    # timing/mem prints
    elapsed = time.perf_counter() - t0
    print(f"[SEED-TIME] seed={int(network_seed)} elapsed={elapsed:.2f}s", flush=True)
    report_mem(tag=f"after compute_features seed={int(network_seed)}")

        # ---- RETURN ONLY SCALARS ----
    features = {
        "mean_total": float(total_starts),
        "violations": float(violations),
        "frac_active": float(frac_active),
        "f12_active": float(f12_active),
        "mse_total": None if mse_total is None else float(mse_total),
        "mse_frac": None if mse_frac is None else float(mse_frac),
        "pen_sparsity": None if 'pen_sparsity' not in locals() or pen_sparsity is None else float(pen_sparsity),
        "pen_12only": None if 'pen_12only' not in locals() or pen_12only is None else float(pen_12only),
        "total_error": float(loss),

        # include params for logging on controller
        "param_tau_ms": float(paramsdict["tau_ms"]),
        "param_dend_threshold": float(paramsdict["dend_threshold"]),
        "param_weights_mean": float(paramsdict["weights_mean"]),
        "param_weights_std": float(paramsdict["weights_std"]),

        # include targets as scalars
        "target_total": float(target_total),
    }

    # split the 10-bin vectors into scalars
    for i, v in enumerate(frac10):
        features[f"frac_bin_{i}"] = float(v)
    for i, v in enumerate(target_frac):
        features[f"target_bin_{i}"] = float(v)

    return features


def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):
    _ensure_debug_state()   # <-- add this

    # average scalars coming back from workers
    out = {}
    keys = features_dict_list[0].keys()
    for k in keys:
        vals = [fd[k] for fd in features_dict_list if fd[k] is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0

    # --- per-gen summary (best model’s metrics) ---
    if not hasattr(context, "pop_size_hint"):
        context.pop_size_hint = int(getattr(context, "pop_size_hint", getattr(context, "pop_size", 1)))
    if not hasattr(context, "_gen_idx"): context._gen_idx = 0
    if not hasattr(context, "_trial_in_gen"): context._trial_in_gen = 0
    if not hasattr(context, "_best_in_gen"):
        context._best_in_gen = {"value": float("inf"), "model_id": None, "params": {}, "metrics": {}}
    if not hasattr(context, "_model_params"): context._model_params = {}
    if not hasattr(context, "_last_params"): context._last_params = {}

    # update best-of-generation
    val = out.get("total_error", float("inf"))
    params_for_model = context._model_params.get(model_id, context._last_params)
    if val < context._best_in_gen["value"]:
        context._best_in_gen = {
            "value": val,
            "model_id": model_id,
            "params": dict(params_for_model),
            "metrics": dict(out),  # include averaged achieved + targets & params*
        }

    # count this model toward the generation
    context._trial_in_gen += 1
    if context._trial_in_gen >= int(context.pop_size_hint):
        best = context._best_in_gen
        bm = best.get("metrics", {})

        print(f"\n[GEN-SUMMARY] Generation {context._gen_idx} finished", flush=True)
        print(f"  best model_id: {best.get('model_id')}", flush=True)
        print("  params:", flush=True)
        print(f"    tau_ms: {bm.get('param_tau_ms', float('nan'))}", flush=True)
        print(f"    dend_threshold: {bm.get('param_dend_threshold', float('nan'))}", flush=True)
        print(f"    weights_mean: {bm.get('param_weights_mean', float('nan'))}", flush=True)
        print(f"    weights_std: {bm.get('param_weights_std', float('nan'))}", flush=True)

        target_total = bm.get("target_total", float('nan'))
        total_starts = bm.get("mean_total", float('nan'))
        violations   = bm.get("violations", float('nan'))
        frac_active  = bm.get("frac_active", float('nan'))
        f12_active   = bm.get("f12_active", float('nan'))

        frac_bins   = [bm.get(f"frac_bin_{i}", 0.0)   for i in range(10)]
        target_bins = [bm.get(f"target_bin_{i}", 0.0) for i in range(10)]
        fmt = lambda xs: "[" + ", ".join(f"{x:.3f}" for x in xs) + "]"

        print("  targets vs achieved:", flush=True)
        print(f"    total_starts: {total_starts:.1f}  | target {target_total:.1f}", flush=True)
        print(f"    violations  : {violations:.1f}   | target 0.0", flush=True)
        print(f"    frac_active : {frac_active:.3f}  | target 0.3", flush=True)
        print(f"    f12_active  : {f12_active:.3f}   | target 1.0", flush=True)
        print(f"    frac10 bins : {fmt(frac_bins)}", flush=True)
        print(f"    target bins : {fmt(target_bins)}", flush=True)
        print(f"  total_error: {best.get('value', float('nan'))}", flush=True)

        report_mem(tag=f"end of gen {context._gen_idx}")

        # reset for next gen
        context._gen_idx += 1
        context._trial_in_gen = 0
        context._best_in_gen = {"value": float("inf"), "model_id": None, "params": {}, "metrics": {}}
        context._model_params.clear()

    return out





def get_objectives(features, model_id=None, export=False, plot=False):
    """Move the scalar to objectives so Optuna minimizes it."""
    return features, {"total_error": features["total_error"]}
