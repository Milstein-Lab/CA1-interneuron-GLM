import nested
# ec_nested_stage.py
import numpy as np
from pathlib import Path
from nested.utils import Context, param_array_to_dict
from multidendrite_spiking_main import SpikingModel, SpikingModelConfig
from multidendrite_spiking_utils import (get_velocity_array_every_animal, get_scaled_data_Hz_dict, do_the_interpolation_an, get_epsp_dict_animal, get_dend_vm_from_cells_multi, sample_weights,
    get_dendrite_activity_multi, activity_to_dend_vm, get_activity_multidendrite2,_eval_one_seed, _ten_bin_fraction_from_counter, _build_static_inputs)
import optuna
from optuna.trial import TrialState

context = Context()


def config_worker():
    """Runs once per process. Load data & set up seeds/static."""


    print("[CTX]", {k: context().get(k) for k in (
    "animal","animal_key","weight_dist","inner_jobs",
    "network_start_seed","num_network_seeds","data_root"
)}, flush=True)

    data_root = str(Path(context.data_root).expanduser().resolve())
    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    # animal = getattr(context, "animal", None)
    # if not animal:
    #     animal = getattr(context, "animal", None)
    # # guard against accidental booleans like True/False from CLI flags
    # if not isinstance(animal, str):
    #     raise ValueError(f"Expected animal string like 'animal_3', got {animal!r} ({type(animal).__name__})")


    static = _build_static_inputs(sm, context.animal)
    static["inner_jobs"] = int(getattr(context, "inner_jobs", 1))
    # weight_dist is a string like "Uniform"/"Normal"/"Lognormal"
    static["dist"] = str(getattr(context, "weight_dist", "Normal")).strip().lower()
    context.static = static

    start = int(getattr(context, "network_start_seed", 0))
    num   = int(getattr(context, "num_network_seeds", 5))
    context.seed_range = list(range(start, start + num))

    # >>> Add this for sanity:
    print(
        f"[CONFIG] animal={getattr(context,'animal', None)} "
        f"weight_dist(raw)={getattr(context,'weight_dist', None)} "
        f"dist(normalized)={static['dist']} "
        f"seeds={context.seed_range}",
        flush=True
    )

    if 'debug' not in context():

        context.debug = False
    context.update(locals())

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
        # Minimal, safe init in this process
        data_root = str(Path(context.data_root).expanduser().resolve())
        cfg = SpikingModelConfig(file_path=data_root)
        sm = SpikingModel(cfg); sm.load()
        static = _build_static_inputs(sm, getattr(context, "animal"))
        static["inner_jobs"] = int(getattr(context, "inner_jobs", 1))
        static["dist"] = str(getattr(context, "weight_dist", "Normal")).strip().lower()
        context.static = static

# --- at the top of compute_features, add: ---
def compute_features(params, network_seed, model_id=None, export=False, plot=False):
    _ensure_initialized()
    """Compute per-seed metrics + a scalar loss called total_error."""
    paramsdict = param_array_to_dict(params, context.param_names)
    start_pos_cnt50, num_per_dend = _eval_one_seed(paramsdict, int(network_seed), context.static)

    n_dendrites = int(context.static["n_dendrites"])
    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    total_starts = float(np.sum(start_pos_cnt50))
    frac10 = _ten_bin_fraction_from_counter(np.asarray(start_pos_cnt50, float))
    violations = float(np.maximum(0.0, num_per_dend - 2.0).sum())

    active_mask = (num_per_dend > 0)
    frac_active = float(active_mask.mean())
    f12_active  = float(((num_per_dend == 1) | (num_per_dend == 2))[active_mask].mean()) if active_mask.any() else 0.0

    if total_starts < 1.0:
        loss = 1e7 + (total_starts - target_total) ** 2
        mse_total = mse_frac = pen_sparsity = pen_12only = None
    elif violations > 0:
        loss = float(1e6 * violations)
        mse_total = mse_frac = pen_sparsity = pen_12only = None
    else:
        mse_total = (total_starts - target_total) ** 2
        mse_frac  = float(np.mean((frac10 - target_frac) ** 2))
        pen_sparsity = (frac_active - 0.30) ** 2
        pen_12only   = (1.0 - f12_active) ** 2
        loss = float(mse_total + mse_frac + 2.0*pen_sparsity + 1.0*pen_12only)

    # Return all useful numbers; nested will average them in filter_features
    return {
        "mean_total": float(total_starts),
        "violations": float(violations),
        "frac_active": float(frac_active),
        "f12_active": float(f12_active),
        "mse_total": None if mse_total is None else float(mse_total),
        "mse_frac": None if mse_frac is None else float(mse_frac),
        "pen_sparsity": None if 'pen_sparsity' not in locals() or pen_sparsity is None else float(pen_sparsity),
        "pen_12only": None if 'pen_12only' not in locals() or pen_12only is None else float(pen_12only),
        "total_error": float(loss),   # <- scalar objective for this seed
    }

def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):
    """Average across seeds (nested calls this once per trial with the list from get_args())."""
    out = {}
    keys = features_dict_list[0].keys()
    for k in keys:
        vals = [fd[k] for fd in features_dict_list if fd[k] is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out

def on_trial_end(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    # Only print when a new best is found
    if trial.state == TrialState.COMPLETE and study.best_trial.number == trial.number:
        print(
            f"[BEST @ trial {trial.number}] "
            f"value={study.best_value:.6g} "
            f"params={study.best_trial.params}",
            flush=True
        )

def get_objectives(features, model_id=None, export=False, plot=False):
    """Move the scalar to objectives so Optuna minimizes it."""
    return features, {"total_error": features["total_error"]}
