from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parent   # adjust if modules live elsewhere
sys.path.insert(0, str(REPO_ROOT))

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
)

# ---------- helpers ----------
def ten_bin_fraction_from_counter(cnt50: np.ndarray) -> np.ndarray:
    cnt50 = np.asarray(cnt50, float)
    agg10 = np.add.reduceat(cnt50, np.arange(0, 50, 5))
    s = agg10.sum()
    return agg10 / s if s > 0 else np.full(10, 1/10, float)

def parse_seed_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def build_static_inputs_for_animal(spike_model: "SpikingModel", animal: str):
    cfg = spike_model.cfg
    factors_dict_EC  = spike_model.data["factors_dict_EC"]
    activity_dict_EC = spike_model.data["activity_dict_EC"]

    # velocity (50 position bins × T time)
    an_velocity_by_animal = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
    animal_velocity = an_velocity_by_animal[animal]  # (50, T)

    # scaled EC Hz per animal
    ret = get_scaled_data_Hz_dict({animal: activity_dict_EC[animal]}, Hz_SF=cfg.hz_sf)

    if isinstance(ret, tuple) and len(ret) == 2:
        scaled_data_Hz_dict, cells_per_animal_dict = ret
        n_EC = int(cells_per_animal_dict[animal])
    else:
        scaled_data_Hz_dict = ret
        n_EC = len(scaled_data_Hz_dict[animal])

    # warp/interpolate to PWA
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

def load_params_from_pickle(pickle_path: str):
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    # handle either {tau_ms: ..., ...} OR {"best_params": {...}, ...}
    if isinstance(obj, dict) and "best_params" in obj:
        p = obj["best_params"]
    else:
        p = obj
    out = dict(
        tau_ms=float(p["tau_ms"]),
        dend_threshold=float(p["dend_threshold"]),
        weights_mean=float(p["weights_mean"]),
        weights_std=float(p["weights_std"]),
    )
    return out, obj

def evaluate_params(sm, animal: str, params: dict, seeds, *, verbose=True, plot=False):
    static = build_static_inputs_for_animal(sm, animal)

    tau_ms         = float(params["tau_ms"])
    dend_threshold = float(params["dend_threshold"])
    weights_mean   = float(params["weights_mean"])
    weights_std    = float(params["weights_std"])

    n_dendrites     = int(static["n_dendrites"])
    n_EC            = int(static["n_EC"])
    animal_velocity = static["animal_velocity"]
    pwa_cell_dict   = static["pwa_cell_dict"]

    totals = []
    frac_sum = np.zeros(10, float)
    violations = []
    active_fracs = []
    f12_act_list = []

    # Per-seed replay
    for i, s in enumerate(seeds):
        epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(s))
        _, epsp_eTN, _ = get_dend_vm_from_cells_multi(
            epsp_cells, Vrest=static["vrest"], epsp_sf=static["epsp_sf"]
        )
        # ensure (E, T, N)
        if epsp_eTN.shape[1] != animal_velocity.shape[1]:
            epsp_eTN = np.transpose(epsp_eTN, (0, 2, 1))

        rng = np.random.default_rng(12345 + int(s))
        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(
            static["dist"], connection_mask_EC, rng=rng,
            mean=weights_mean, std=weights_std
        )

        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
        dend_Vm, _, _ = activity_to_dend_vm(
            activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials"
        )

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
            dt_constant=static["dt_constant"], dx=static["dx"],
            dend_threshold=dend_threshold,
            vel_applied="real", example_cell=15,
            include_inhibition=True, use_model_EC=False
        )

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

        if verbose:
            print(f"[seed {s}] mean_plateaus/dend={num_per_dend.mean():.3f}, total_starts={total_starts:.0f}")

    # Aggregate
    mean_total  = float(np.mean(totals)) if totals else 0.0
    mean_frac   = frac_sum / max(1, len(seeds))
    total_viol  = float(np.sum(violations))
    frac_active = float(np.mean(active_fracs)) if active_fracs else 0.0
    f12_active  = float(np.mean(f12_act_list)) if f12_act_list else 0.0

    summary = dict(
        params=dict(tau_ms=tau_ms, dend_threshold=dend_threshold,
                    weights_mean=weights_mean, weights_std=weights_std),
        seeds=list(seeds),
        mean_total=mean_total,
        frac_active=frac_active,
        f12_active=f12_active,
        total_violations=total_viol,
        frac10=list(np.round(mean_frac, 4)),
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

    # Optional plotting hook (only if your function exists & expects these signatures)
    if plot:
        try:
            from multidendrite_spiking_utils import plot_multidendrite_EC_multiple_seeds  # type: ignore
            print("[plot] (placeholder) call your plotting function here with the objects you want to visualize.")
            # You likely want to refactor plotting to accept a small bundle of arrays per seed.
            # Left as a stub to avoid guessing the exact signature in your local code.
        except Exception as e:
            print(f"[plot] skipped (plot function unavailable or import failed): {e}")

    return summary

# ---------- CLI ----------
@click.command()
@click.option("--pickle-path", "pickle_path", required=True, type=click.Path(exists=True),
              help="Path to pickle with params (either {tau_ms,...} or {'best_params':{...}}).")
@click.option("--animal", required=True, help="Animal key (e.g., 'animal_3').")
@click.option("--data-root", required=True, type=click.Path(exists=True),
              help="Root directory that contains the 'datasets/' folder.")
@click.option("--seeds", default="0,1,2,3,4", show_default=True,
              help="Comma-separated seeds to evaluate.")
@click.option("--no-plot", is_flag=True, default=False, help="Disable plotting.")
def main(pickle_path, animal, data_root, seeds, no_plot):
    # config + data
    cfg = SpikingModelConfig(file_path=str(Path(data_root).expanduser().resolve()))
    sm = SpikingModel(cfg)
    sm.load()

    params, raw_obj = load_params_from_pickle(pickle_path)
    print("[loaded params]")
    print(json.dumps(params, indent=2))

    seed_list = parse_seed_list(seeds)
    evaluate_params(sm, animal, params, seed_list, verbose=True, plot=(not no_plot))

if __name__ == "__main__":
    main()
