# run_optuna.py
import numpy as np
import optuna
from optuna.storages import RDBStorage
import pickle
from joblib import Parallel, delayed
from pathlib import Path

from multidendrite_spiking_main import SpikingModel, SpikingModelConfig
from multidendrite_spiking_utils import (
    get_velocity_array_every_animal, get_scaled_data_Hz_dict, do_the_interpolation_an,
    get_epsp_dict_animal, get_dend_vm_from_cells_multi, sample_weights,
    get_dendrite_activity_multi, activity_to_dend_vm, get_activity_multidendrite2
)
from spiking_model_utils import (
    add_vel_contribution_to_residuals, get_velocity_array, do_the_interpolation,
    get_summed_dendrite_EC_DFF, get_plateau_and_cumulative_ragged,
    get_epsp_dict, get_dend_vm
)

# If you plot inside load_plot_pickled_params, you'll also need:
# from multidendrite_spiking_utils import plot_multidendrite_EC_multiple_seeds

import click


def to_sample_weights_token(name: str) -> str:
    m = str(name).strip().lower()
    if m in ("uniform", "u", "uni"):
        return "Uniform"
    if m in ("normal", "n", "gaussian", "norm"):
        return "Normal"
    if m in ("lognormal", "log-norm", "lognorm", "ln"):
        return "Lognormal"
    if m in ("equal", "const", "constant"):
        return "Equal"
    raise ValueError(f"Unknown weight distribution: {name!r}")



def ten_bin_fraction_from_counter(cnt50: np.ndarray) -> np.ndarray:
    cnt50 = np.asarray(cnt50, float)
    agg10 = np.add.reduceat(cnt50, np.arange(0, 50, 5))
    s = agg10.sum()
    return agg10 / s if s > 0 else np.full(10, 1/10, float)


def build_static_inputs_for_animal(spike_model: "SpikingModel", animal: str):
    cfg = spike_model.cfg
    factors_dict_EC  = spike_model.data["factors_dict_EC"]
    residual_activity_dict_EC = spike_model.data["residual_activity_dict_EC"]
    GLM_params_EC = spike_model.data["GLM_params_EC"]

    an_velocity_dict = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
    scaled_data_Hz_dict_resid, cells_per_animal_dict = get_scaled_data_Hz_dict(residual_activity_dict_EC, Hz_SF=50)
    scaled_data_Hz_dict = add_vel_contribution_to_residuals(scaled_data_Hz_dict_resid, GLM_params_EC, an_velocity_dict)

    an_velocity = get_velocity_array(factors_dict_EC, 0, 0, which_type="EC_animal_average")
    padded_warped_activity_dict, an_velocity = do_the_interpolation(
        scaled_data_Hz_dict, an_velocity=an_velocity
    )
    _ = get_summed_dendrite_EC_DFF(residual_activity_dict_EC)

    n_EC = 0
    for animal in cells_per_animal_dict:
        n_EC+=cells_per_animal_dict[animal]



    return dict(
        animal=animal,
        animal_velocity=an_velocity,
        padded_warped_activity_dict=padded_warped_activity_dict,
        n_dendrites=100,
        n_EC=n_EC,
        dt_constant=cfg.dt_constant,
        dx=cfg.dx,
        vrest=cfg.vrest,
        epsp_sf=cfg.epsp_sf,
        dist=cfg.dist,
        inner_jobs=1,  # can be overridden later
    )


def priority_loss_single_animal(params, static, seed_list, return_metrics: bool = False):
    tau_ms         = float(params["tau_ms"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])
    dend_threshold = float(params["dend_threshold"])

    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    padded_warped_activity_dict = static["padded_warped_activity_dict"]

    # targets
    target_total = 0.30 * n_dendrites * 1.5
    target_frac  = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

    plateau_dict_animal, counts_dict_animal = get_plateau_and_cumulative_ragged(
        padded_warped_activity_dict, dend_threshold, plateau_len=300, refractory=800, scan_step=100
    )

    def _eval_one_seed(s):
        epsp_dict, kernel = get_epsp_dict(padded_warped_activity_dict, tau_ms=tau_ms, amp=1., seed=int(s))
        dend_Vm, epsp_list, spike_list = get_dend_vm(epsp_dict, Vrest=static["vrest"], epsp_sf=static["epsp_sf"])

        rng = np.random.default_rng(12345 + int(s))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(static["dist"], connection_mask_EC, rng=rng,
                                    mean=weights_mean, std=weights_std)

        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_list, n_dendrites, n_EC)
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

    if outs:
        print(f"[debug] mean_plateaus/dend={outs[0][2].mean():.3f}, total_starts={totals[0]:.0f}")

    mean_total = float(np.mean(totals)) if totals else 0.0

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


def load_plot_pickled_params(save_path, static, num_seeds, animal, plot=True):
    with open(save_path, 'rb') as f:
        saved = pickle.load(f)

    # support both schemas
    if isinstance(saved, dict) and "best_params" in saved:
        params = saved["best_params"]
        if "weight_dist" in saved:
            static["dist"] = saved["weight_dist"]
    else:
        params = saved

    tau            = float(params["tau_ms"])
    dend_threshold = float(params["dend_threshold"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])

    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]

    # >>> FIX: normalize *once* to the exact token sample_weights expects
    dist_token = to_sample_weights_token(static.get("dist", "Normal"))
    print(f"[eval] using distribution token for sample_weights → {dist_token}")

    dend_vm_per_seed_dict = {}
    last_weights_EC = None
    last_activity_EC = None

    for s in range(num_seeds):
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau, amp=1., seed=int(s))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"])
        if epsp_eTN.shape[1] != animal_velocity.shape[1]:
            epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

        rng = np.random.default_rng(12345 + int(s))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)

        # >>> FIX: pass the TitleCase token here
        weights_EC = sample_weights(dist_token, connection_mask_EC, rng=rng,
                                    mean=weights_mean, std=weights_std)

        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
        dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

        dend_vm_per_seed_dict[s] = dend_Vm
        last_weights_EC = weights_EC
        last_activity_EC = activity_EC

    (_pos_cnt, start_pos_cnt50, _plateau_arr_list, _mask, _starts_list,
     num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list) = get_activity_multidendrite2(
        animal_velocity, dend_vm_per_seed_dict,
        activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
        dt_constant=static["dt_constant"], dx=static["dx"], dend_threshold=dend_threshold,
        vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False
    )

    if plot:
        cumulative_plateau_counts, plateau_fraction_by_pos_bin, plateau_start_positions_counter  = plot_multidendrite_EC_multiple_seeds(weights_EC, 0, 0, dend_vm_per_seed_dict, activity_EC, 0, 0, 0, 0, padded_warped_activity_list, animal_velocity, dend_activity, dend_threshold, _pos_cnt, plateau_start_positions_counter, _plateau_arr_list, _mask, _starts_list, dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition="neither", NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=True)

@click.group(help="EC Optuna utilities.")
def cli():
    pass


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
    static["inner_jobs"] = 1

    print(f"[eval] pickle={pickle_path} animal={animal} seeds=0..{num_seeds-1}")
    load_plot_pickled_params(pickle_path, static, num_seeds, animal, plot=(not no_plot))


@cli.command("run")
@click.option("--animal", default="animal_1",
              help="Animal key (e.g., 'animal_3') or 'all' to average across all animals.")
@click.option("--storage", default="sqlite:///ec_optuna.db", show_default=True,
              help="Optuna storage URL. Use Postgres for multi-node.")
@click.option("--study", default="ec_param_search", show_default=True,
              help="Optuna study name.")
@click.option("--trials", type=int, default=100, show_default=True)
@click.option("--n-jobs", type=int, default=1, show_default=True,
              help="Parallel workers on *this* process (Optuna-level).")
@click.option("--seed-list", default="0,1,2,3,4", show_default=True,
              help="Comma-separated seeds averaged inside the objective.")
@click.option("--save-path", type=click.Path(), default="best_params.pkl",
              help="Path to pickle file for saving best params + value.")
@click.option("--data-root", default="/jet/home/mfinch/CA1-interneuron-GLM", show_default=True,
              help="Root directory that contains the 'datasets/' folder.")
@click.option("--inner-jobs", type=int, default=1, show_default=True,
              help="Threads for seed-level parallelism (use with --n-jobs 1 when on SQLite).")
def run_cmd(animal, storage, study, trials, n_jobs, seed_list, save_path, data_root, inner_jobs):
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
    for stc in statics_list:
        stc["inner_jobs"] = inner_jobs
    seeds = parse_seed_list(seed_list)

    # SQLite safety
    if storage.startswith("sqlite") and n_jobs != 1:
        click.echo("⚠️ SQLite + --n-jobs>1 can lock. Forcing --n-jobs=1; use --inner-jobs for seed threads.", err=True)
        n_jobs = 1

    storage_backend = RDBStorage(
        url=storage,
        engine_kwargs={"connect_args": {"timeout": 120}}
    )
    st = optuna.create_study(
        study_name=study, direction="minimize",
        storage=storage_backend, load_if_exists=True
    )

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


if __name__ == "__main__":
    cli()
