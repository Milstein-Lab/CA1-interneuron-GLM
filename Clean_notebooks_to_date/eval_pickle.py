

#!/usr/bin/env python
import json
import pickle
from pathlib import Path

import numpy as np
import click

from multidendrite_spiking_main import SpikingModel, SpikingModelConfig
from multidendrite_spiking_utils import (
    get_velocity_array_every_animal, get_scaled_data_Hz_dict, do_the_interpolation_an,
    get_epsp_dict_animal, get_dend_vm_from_cells_multi, sample_weights,
    get_dendrite_activity_multi, activity_to_dend_vm, get_activity_multidendrite2,
plot_multidendrite_EC_multiple_seeds_quick)

from optuna_multidendrite_spiking import *


def load_plot_pickled_params(save_path, static, num_seeds, animal, plot=True):
    """
    Replays 'num_seeds' seeds using params saved in 'save_path' and prints a summary.
    Handles either {'best_params': {...}} or a flat dict of params.
    """
    # --- load params robustly ---
    with open(save_path, "rb") as f:
        obj = pickle.load(f)
    params = obj["best_params"] if isinstance(obj, dict) and "best_params" in obj else obj

    tau            = float(params["tau_ms"])
    dend_threshold = float(params["dend_threshold"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])

    # --- statics ---
    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]
    dist            = static["dist"]

    # --- accumulators for metrics ---
    totals = []
    frac_sum = np.zeros(10, float)
    violations = []
    active_fracs = []
    f12_act_list = []

    # --- stash for optional multi-seed plotting ---
    dend_vm_per_seed_dict = {}
    last_weights_EC = None
    last_activity_EC = None
    # (optional per-seed artifacts for plotting; will fill when available)
    _pos_cnt_all = []
    start_pos_cnt50_all = []
    _plateau_arr_list_all = []
    _mask_all = []
    _starts_list_all = []
    num_plateaus_per_dend_list_all = []
    dend_activity_all = []
    padded_warped_activity_list_all = []

    # --- per-seed replay (ensures we can print per-seed stats) ---
    for s in range(num_seeds):
        # EPSPs
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau, amp=1., seed=int(s))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(
            epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"]
        )
        # ensure (E, T, N)
        if epsp_eTN.shape[1] != animal_velocity.shape[1]:
            epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

        # EC weights
        rng = np.random.default_rng(12345 + int(s))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(dist, connection_mask_EC, rng=rng,
                                    mean=weights_mean, std=weights_std)

        # EC → dend Vm
        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
        dend_Vm, _, _ = activity_to_dend_vm(
            activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials"
        )
        dend_vm_per_seed_dict[s] = dend_Vm
        last_weights_EC = weights_EC
        last_activity_EC = activity_EC

        # Plateau detection (single-seed call; signature differences tolerated)
        _ret = get_activity_multidendrite2(
            animal_velocity, dend_Vm,
            activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
            dt_constant=static["dt_constant"], dx=static["dx"],
            dend_threshold=dend_threshold,
            vel_applied="real", example_cell=15,
            include_inhibition=True, use_model_EC=False
        )
        # Minimum 6 outputs are used; extras (dend_activity, padded_warped_activity_list) captured if present
        # Known earliest 6:
        # 0:_pos_cnt, 1:start_pos_cnt50, 2:_plateau_arr_list, 3:_mask, 4:_starts_list, 5:num_plateaus_per_dend_list
        _pos_cnt                         = _ret[0]
        start_pos_cnt50                  = _ret[1]
        _plateau_arr_list                = _ret[2]
        _mask                            = _ret[3]
        _starts_list                     = _ret[4]
        num_plateaus_per_dend_list       = _ret[5]
        dend_activity                    = _ret[6] if len(_ret) > 6 else None
        padded_warped_activity_list      = _ret[7] if len(_ret) > 7 else None

        # Save for optional plotting
        _pos_cnt_all.append(_pos_cnt)
        start_pos_cnt50_all.append(start_pos_cnt50)
        _plateau_arr_list_all.append(_plateau_arr_list)
        _mask_all.append(_mask)
        _starts_list_all.append(_starts_list)
        num_plateaus_per_dend_list_all.append(num_plateaus_per_dend_list)
        if dend_activity is not None:
            dend_activity_all.append(dend_activity)
        if padded_warped_activity_list is not None:
            padded_warped_activity_list_all.append(padded_warped_activity_list)

        # Metrics for prints/summary
        num_per_dend = np.asarray(num_plateaus_per_dend_list, float)
        total_starts = float(np.sum(start_pos_cnt50))
        frac10 = ten_bin_fraction_from_counter(np.asarray(start_pos_cnt50, float))

        totals.append(total_starts)
        frac_sum += frac10
        violations.append(np.maximum(0.0, num_per_dend - 2.0).sum())

        active_mask = (num_per_dend > 0)
        active_fracs.append(float(active_mask.mean()))
        if active_mask.any():
            f12 = (num_per_dend == 1) | (num_per_dend == 2)
            f12_act_list.append(float(f12[active_mask].mean()))
        else:
            f12_act_list.append(0.0)

        # ---- per-seed print (same style) ----
        print(f"[seed {s}] mean_plateaus/dend={num_per_dend.mean():.3f}, total_starts={total_starts:.0f}")

    # --- aggregate summary (same JSON print) ---
    mean_total  = float(np.mean(totals)) if totals else 0.0
    mean_frac   = frac_sum / max(1, num_seeds)
    total_viol  = float(np.sum(violations))
    frac_active = float(np.mean(active_fracs)) if active_fracs else 0.0
    f12_active  = float(np.mean(f12_act_list)) if f12_act_list else 0.0

    summary = dict(
        params=dict(tau_ms=tau, dend_threshold=dend_threshold,
                    weights_mean=weights_mean, weights_std=weights_std),
        seeds=list(range(num_seeds)),
        mean_total=mean_total,
        frac_active=frac_active,
        f12_active=f12_active,
        total_violations=total_viol,
        frac10=list(np.round(mean_frac, 4)),
        animal=animal,
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # --- optional: try a multi-seed aggregator if your code has one ---
    if plot:
        # If you have a true multi-seed function, prefer it:
        from multidendrite_spiking_utils import get_activity_multidendrite2_multiple_seeds  # type: ignore
        _ = get_activity_multidendrite2_multiple_seeds(
            animal_velocity,
            dend_vm_per_seed_dict,
            activity_NDNF=0, activity_SST=0,
            NDNF_sf_opt=0, SST_sf_opt=0,
            dt_constant=static["dt_constant"], dx=static["dx"],
            dend_threshold=dend_threshold,
            vel_applied="real", example_cell=15,
            include_inhibition=True, use_model_EC=False
        )
        # You can unpack and pass those outputs to your plotting function below if desired.
        

        # Plot hook: call only if your plotting fuction exists and you want a figure.
            # plot_multidendrite_EC_multiple_seeds_quick(last_weights_EC, 0, 0,dend_vm_per_seed_dict,last_activity_EC, 0, 0,0, 0,padded_warped_activity_list_all or None,animal_velocity,dend_activity_all or None,dend_threshold,_pos_cnt_all,start_pos_cnt50_all,_plateau_arr_list_all,_mask_all,_starts_list_all,dist,num_plateaus_per_dend_list_all,animal,example_cell=1, include_inhibition="neither",NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=True)
        plot_multidendrite_EC_multiple_seeds_quick(last_weights_EC, 0, 0,dend_vm_per_seed_dict,last_activity_EC, 0, 0,0, 0, animal_velocity, dend_threshold, _pos_cnt_all, start_pos_cnt50_all, _plateau_arr_list_all, _mask_all,  _starts_list_all, dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=False)
 
    return summary



   
    



# -----------------------
# Turn the script into a 2-command CLI: `optimize` and `eval-pickle`
# -----------------------
import click

@click.group()
def cli():
    """EC Optuna utilities."""
    pass


# === your existing 'main' becomes 'optimize' subcommand ===
@cli.command("optimize")
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
def optimize(animal, storage, study, trials, n_jobs, seed_list, save_path, data_root, inner_jobs):
    # === this is your existing main(...) body unchanged, just wrapped ===
    data_root = str(Path(data_root).expanduser().resolve())
    datasets_dir = Path(data_root) / "datasets"
    if not datasets_dir.exists():
        raise SystemExit(f"datasets/ not found at: {datasets_dir}. Put your .mat files there or pass --data-root <path>")

    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    all_animals = list(sm.data["activity_dict_EC"].keys())
    if animal.lower() == "all":
        chosen = all_animals
    else:
        if animal not in all_animals:
            raise SystemExit(f"--animal {animal} not found. Available: {all_animals}")
        chosen = [animal]

    statics_list = [build_static_inputs_for_animal(sm, a) for a in chosen]
    seeds = parse_seed_list(seed_list)
    for stc in statics_list:
        stc["inner_jobs"] = inner_jobs

    def make_objective(statics_list, seeds):
        def objective(trial):
            params = dict(
                tau_ms         = trial.suggest_float("tau_ms", 5.0, 200.0),
                dend_threshold = trial.suggest_float("dend_threshold", -75.0, -65.0),
                weights_mean   = trial.suggest_float("weights_mean", 0.1, 1.0),
                weights_std    = trial.suggest_float("weights_std", 0.1, 1.0),
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

    if storage.startswith("sqlite") and n_jobs != 1:
        click.echo("⚠️ SQLite + --n-jobs>1 can lock. Forcing --n-jobs=1; use --inner-jobs for speed.", err=True)
        n_jobs = 1

    storage_backend = RDBStorage(url=storage, engine_kwargs={"connect_args": {"timeout": 120}})
    st = optuna.create_study(study_name=study, direction="minimize",
                             storage=storage_backend, load_if_exists=True)

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
    }
    with open(save_path, "wb") as f:
        pickle.dump(out, f)

    print(f"✅ Saved best trial → {save_path}")
    print("Best params:", best_params)
    print("Best value:", best_value)
    print("Best metrics:", best_metrics)


# === new: eval-pickle subcommand that ONLY calls load_plot_pickled_params ===
@cli.command("eval-pickle")
@click.option("--pickle-path", required=True, type=click.Path(exists=True),
              help="Path to params pickle (either flat dict or {'best_params': {...}}).")
@click.option("--animal", required=True, help="Animal key, e.g., 'animal_3'.")
@click.option("--data-root", required=True, type=click.Path(exists=True),
              help="Root directory that contains the 'datasets/' folder.")
@click.option("--num-seeds", default=5, show_default=True, type=int,
              help="How many seeds to replay (0..num_seeds-1).")
@click.option("--no-plot", is_flag=True, default=False, help="Disable plotting.")
def eval_pickle_cmd(pickle_path, animal, data_root, num_seeds, no_plot):
    data_root = str(Path(data_root).expanduser().resolve())
    datasets_dir = Path(data_root) / "datasets"
    if not datasets_dir.exists():
        raise SystemExit(f"datasets/ not found at: {datasets_dir}. Put your .mat files there or pass --data-root <path>")

    cfg = SpikingModelConfig(file_path=data_root)
    sm = SpikingModel(cfg); sm.load()

    static = build_static_inputs_for_animal(sm, animal)
    # you can drop inner_jobs into static if your helpers use it
    static["inner_jobs"] = 1

    print(f"[eval] pickle={pickle_path} animal={animal} seeds=0..{num_seeds-1}")
    load_plot_pickled_params(pickle_path, static, num_seeds, animal, plot=(not no_plot))


if __name__ == "__main__":
    cli()
