from dataclasses import dataclass, field
from typing import Dict, Any
import pickle
import click
from spiking_model_utils import *
from pathlib import Path
import sys
import platform


@dataclass
class SpikingModelConfig:
    file_path: str = field(default_factory=lambda: str(Path.home() / "CA1-interneuron-GLM"))
    tau_ms: float = 200.0
    num_seeds: int = 2
    dend_threshold: float = -30.0
    which_velocity: str = "EC_animal_average"   # "EC_animal_average" | "repeated_waveform" | "constant"
    hz_sf: float = 50.0                         # scaling for get_scaled_data_Hz_dict
    vrest: float = -70.0
    epsp_sf: float = 0.1
    dt_constant: float = 0.001                  # seconds (1 ms)


class SpikingModel:
    def __init__(self, cfg: SpikingModelConfig):
        self.cfg = cfg
        # unprocessed data in self.data and the outputs in self.results
        self.data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def load(self) -> None:
        """Load EC/SST/NDNF datasets and keep only the pieces we need downstream."""
        fp = os.environ.get("SPK_MODEL_ROOT", self.cfg.file_path)
        print(f"[load] Using root: {fp}")

        (
            GLM_params_EC,
            activity_dict_EC,
            double_predicted_activity_dict_EC,
            factors_dict_EC,
            filtered_factors_dict_EC,
            residual_activity_dict_EC,
        ) = load_data_regular(file_path=fp, name="EC_GLM", new_NDNF=False)

        (
            GLM_params_SST,
            activity_dict_SST,
            double_predicted_activity_dict_SST,
            factors_dict_SST,
            filtered_factors_dict_SST,
            residual_activity_dict_SST,
        ) = load_data_regular(file_path=fp, name="SSTindivsomata_GLM", new_NDNF=False)

        (
            GLM_params_NDNF_newest,
            activity_dict_NDNF_newest,
            double_predicted_activity_dict_NDNF_newest,
            factors_dict_NDNF_newest,
            filtered_factors_dict_NDNF_newest,
            residual_activity_dict_NDNF_newest,
        ) = load_data_regular(file_path=fp, name="NDNF_E1A1B", new_NDNF=True)

        fixed_filtered_factors_dict_NDNF_newest = {}
        for idx, animal in enumerate(filtered_factors_dict_NDNF_newest):
            if 17 < idx < 31:
                fixed_filtered_factors_dict_NDNF_newest[f"animal_{idx+1}"] = filtered_factors_dict_NDNF_newest[animal]

        self.data.update(
            dict(
                activity_dict_EC=activity_dict_EC,
                residual_activity_dict_EC=residual_activity_dict_EC,
                factors_dict_EC=factors_dict_EC,
                factors_dict_SST=factors_dict_SST,
                fixed_filtered_factors_dict_NDNF_newest=fixed_filtered_factors_dict_NDNF_newest,
                GLM_params_EC=GLM_params_EC,
            )
        )

    def prepare_inputs(self, animal_by_animal: bool = False) -> None:
        activity_dict_EC = self.data["activity_dict_EC"]
        factors_dict_EC = self.data["factors_dict_EC"]
        factors_dict_SST = self.data["factors_dict_SST"]
        fixed_NDNF = self.data["fixed_filtered_factors_dict_NDNF_newest"]
        residual_activity_dict_EC = self.data["residual_activity_dict_EC"]
        GLM_params_EC = self.data["GLM_params_EC"]

        if animal_by_animal:
            scaled_data_Hz_dict = get_scaled_data_Hz_dict(activity_dict_EC, Hz_SF=self.cfg.hz_sf)
        else:
            an_velocity_dict = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
            scaled_data_Hz_dict_resid = get_scaled_data_Hz_dict(residual_activity_dict_EC, Hz_SF=50)
            scaled_data_Hz_dict = add_vel_contribution_to_residuals(
                scaled_data_Hz_dict_resid, GLM_params_EC, an_velocity_dict
            )

        # global (regular) velocity
        if self.cfg.which_velocity == "EC_animal_average":
            an_velocity = get_velocity_array(
                factors_dict_EC, factors_dict_SST, fixed_NDNF, which_type="EC_animal_average"
            )
        elif self.cfg.which_velocity == "repeated_waveform":
            an_velocity = get_velocity_array(
                factors_dict_EC, factors_dict_SST, fixed_NDNF, which_type="repeated_waveform"
            )
        elif self.cfg.which_velocity == "constant":
            an_velocity = get_velocity_array(factors_dict_EC, which_type="constant")
        else:
            raise ValueError("which_velocity must be 'EC_animal_average', 'repeated_waveform', or 'constant'")

        if animal_by_animal:
            # Build per-animal velocity from factors (not activity!)
            an_velocity_by_animal = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)

            # Interpolate/warp per animal
            padded_warped_by_animal = {}
            for animal in scaled_data_Hz_dict:
                pwa_cell_dict, _ = do_the_interpolation_an(
                    scaled_data_Hz_dict[animal],  # {cell: (n_pos, n_trials)}
                    an_velocity_by_animal[animal],  # (n_pos, n_trials)
                    dt_constant=self.cfg.dt_constant,
                )
                padded_warped_by_animal[animal] = pwa_cell_dict

            self.results.update(
                dict(
                    scaled_data_Hz_dict=scaled_data_Hz_dict,
                    velocity_by_animal=an_velocity_by_animal,
                    padded_warped_activity_by_animal=padded_warped_by_animal,
                )
            )
            return

        # Regular (global) path
        padded_warped_activity_dict, an_velocity = do_the_interpolation(scaled_data_Hz_dict, an_velocity=an_velocity)
        
        summed_dendrite = get_summed_dendrite_EC_DFF(self.data["residual_activity_dict_EC"])
        padded_warped_activity_EC, _, cumulative_plateau_counts = get_internals_summed_dendrite(an_velocity, summed_dendrite, dt_constant=self.cfg.dt_constant, dend_threshold=self.cfg.dend_threshold, vel_applied="real")

        self.results.update(
            dict(
                scaled_data_Hz_dict=scaled_data_Hz_dict,
                an_velocity=an_velocity,
                padded_warped_activity_dict=padded_warped_activity_dict,
                summed_dendrite=summed_dendrite,
                padded_warped_activity_EC=padded_warped_activity_dict,
                cumulative_plateau_counts=cumulative_plateau_counts,
            )
        )

    def simulate_seeds(self, animal_by_animal: bool = False, seed_override: int | None = None) -> None:
        tau = self.cfg.tau_ms

        if animal_by_animal:
            pwa_by_animal = self.results["padded_warped_activity_by_animal"]
            vm_by_animal = {}
            spikes_by_animal = {}

            for animal, pwa_cell_dict in pwa_by_animal.items():
                seed_vm = {}
                seed_spikes = {}

                # choose which seeds to run
                seeds = [seed_override] if seed_override is not None else list(range(self.cfg.num_seeds))

                for i in seeds:
                    epsp_cells, kernel = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau, amp=1.0, seed=i)
                    dend_Vm, sum_epsp_centered, spike_mats = get_dend_vm_from_cells(
                        epsp_cells, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf
                    )
                    seed_vm[i] = dend_Vm
                    seed_spikes[i] = spike_mats

                vm_by_animal[animal] = seed_vm
                spikes_by_animal[animal] = seed_spikes

            self.results.update(dict(vm_by_animal=vm_by_animal, spikes_by_animal=spikes_by_animal, last_kernel=kernel))
            return

        # Regular path
        dend_Vm_dict: Dict[int, np.ndarray] = {}
        for i in range(self.cfg.num_seeds):
            epsp_dict, kernel = get_epsp_dict(
                self.results["padded_warped_activity_dict"], tau_ms=tau, amp=1.0, seed=i
            )
            dend_Vm, epsp_list, spike_list = get_dend_vm(
                epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf
            )
            dend_Vm_dict[i] = dend_Vm
            print(f"[seed {i}] dend_vm OK   shape={dend_Vm.shape}")

        self.results.update(dict(dend_Vm_dict=dend_Vm_dict, last_kernel=kernel))

    def compute_plateaus(self) -> None:
        """Detect plateaus from Vm and make arrays for plotting."""
        dend_threshold = self.cfg.dend_threshold
        dend_Vm_dict = self.results["dend_Vm_dict"]

        just_plateau_starts_sums_dict, plateau_array_dict = get_plateau_array_dict(dend_Vm_dict, dend_threshold)
        self.results.update(
            dict(just_plateau_starts_sums_dict=just_plateau_starts_sums_dict, plateau_array_dict=plateau_array_dict)
        )

    def plot(self) -> None:
        """Reproduce your multi-panel figure."""
        r = self.results
        plot_dendrite_spikes_multiple_seeds(
            dend_Vm_dict=r["dend_Vm_dict"],
            an_velocity=r["an_velocity"],
            residual_activity_dict_EC=self.data["residual_activity_dict_EC"],
            padded_warped_activity_EC=r["padded_warped_activity_EC"],
            summed_dendrite=r["summed_dendrite"],
            just_plateau_starts_sums_dict=r["just_plateau_starts_sums_dict"],
            plateau_array_dict=r["plateau_array_dict"],
            dend_threshold=self.cfg.dend_threshold,
            tau=self.cfg.tau_ms,
            num_seeds=self.cfg.num_seeds,
        )

    def run(self, do_plot: bool = True, animal_by_animal: bool = False) -> Dict[str, Any]:
        self.load()
        self.prepare_inputs(animal_by_animal=animal_by_animal)
        self.simulate_seeds(animal_by_animal=animal_by_animal)

        if not animal_by_animal:
            self.compute_plateaus()
            if do_plot:
                self.plot()
        else:
            # Optional: if you want plots per animal, implement self.plot_animal_by_animal()
            if do_plot and "vm_by_animal" in self.results:
                self.plot_animal_by_animal()  # see stub below (optional)

        return self.results

    def plot_quick(self, mode: str = "auto") -> None:
        """
        Plot without recomputing anything. Works for both:
        - regular runs: results['dend_Vm_dict'] + results['an_velocity']
        - animal-by-animal runs: results['vm_by_animal'] + results['velocity_by_animal']
        Calls the util using positional args to avoid keyword mismatches.
        """
        fp = self.cfg.file_path
        (
            GLM_params_EC,
            activity_dict_EC,
            double_predicted_activity_dict_EC,
            factors_dict_EC,
            filtered_factors_dict_EC,
            residual_activity_dict_EC,
        ) = load_data_regular(file_path=fp, name="EC_GLM", new_NDNF=False)
        residual = residual_activity_dict_EC

        # Prefer explicit mode if you add others later; for now 'auto' handles both.
        if "vm_by_animal" in self.results and "velocity_by_animal" in self.results:
            # Animal-by-animal plotting
            vm_by_animal = self.results["vm_by_animal"]
            vel_by_animal = self.results["velocity_by_animal"]

            for animal, seed_vm_dict in vm_by_animal.items():
                vel = vel_by_animal.get(animal, None)
                if vel is None:
                    print(f"[plot_quick] Skipping {animal}: no velocity.")
                    continue
                print(f"[plot_quick] plotting animal={animal}")

                # Call util with positional args only.
                # Try with residual; if the util doesn't accept it, fall back without.
                try:
                    plot_dendrite_spikes_multiple_seeds(
                        seed_vm_dict,
                        vel,
                        residual[animal],
                        animal,
                        self.cfg.dend_threshold,
                        self.cfg.tau_ms,
                        self.cfg.num_seeds,
                    )
                except TypeError:
                    plot_dendrite_spikes_multiple_seeds(
                        seed_vm_dict,
                        vel,
                        self.cfg.dend_threshold,
                        self.cfg.tau_ms,
                        self.cfg.num_seeds,
                    )
            return

        # Regular (global) plotting
        if "dend_Vm_dict" in self.results and "an_velocity" in self.results:
            try:
                plot_dendrite_spikes_multiple_seeds(
                    self.results["dend_Vm_dict"],
                    self.results["an_velocity"],
                    residual,
                    self.cfg.dend_threshold,
                    self.cfg.tau_ms,
                    self.cfg.num_seeds,
                )
            except TypeError:
                plot_dendrite_spikes_multiple_seeds(
                    self.results["dend_Vm_dict"],
                    self.results["an_velocity"],
                    self.cfg.dend_threshold,
                    self.cfg.tau_ms,
                    self.cfg.num_seeds,
                )
            return

        # Nothing recognizable to plot
        have = list(self.results.keys())
        raise KeyError(f"[plot_quick] No VM/velocity found to plot. results has keys: {have}")

    def plot_animal_by_animal(self) -> None:
        """
        For each animal, call the same multi-seed plot using that animal's velocity
        and per-seed Vm dict. We pass None for pooled-only panels.
        """
        r = self.results
        vel_by_animal = r.get("velocity_by_animal", {})
        vm_by_animal = r.get("vm_by_animal", {})

        if not vm_by_animal:
            print("[plot_animal_by_animal] Nothing to plot (vm_by_animal missing).")
            return

        for animal, seed_vm_dict in vm_by_animal.items():
            animal_vel = vel_by_animal.get(animal)
            if animal_vel is None:
                print(f"[plot_animal_by_animal] Skipping {animal}: no velocity.")
                continue

            print(f"[plot_animal_by_animal] plotting animal={animal}")

            # residual_activity_dict_EC = self.data["residual_activity_dict_EC"]
            residual_activity_dict_EC = self.data.get("residual_activity_dict_EC", None)

            animal_by_animal = True

            try:
                plot_dendrite_spikes_multiple_seeds(
                    seed_vm_dict,
                    animal_vel,
                    residual_activity_dict_EC,
                    animal,
                    animal_by_animal=animal_by_animal,
                    dend_threshold=self.cfg.dend_threshold,
                    tau=self.cfg.tau_ms,
                    num_seeds=self.cfg.num_seeds,
                )

            except ValueError as e:
                # Most common cause: a binning helper inside the util unpacks 4 values
                # but your function returns 3. See fix #2 below.
                print(f"[plot_animal_by_animal] Plot helper raised ValueError: {e}")
                raise

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __getstate__(self):
        state = {"cfg": self.cfg}
        r = self.results
        keep = {}

        if "an_velocity" in r and isinstance(r["an_velocity"], np.ndarray):
            keep["an_velocity"] = r["an_velocity"].astype(np.float32, copy=False)

        if "velocity_by_animal" in r:
            keep["velocity_by_animal"] = {a: v.astype(np.float32, copy=False) for a, v in r["velocity_by_animal"].items()}

        if "dend_Vm_dict" in r:
            keep["dend_Vm_dict"] = {k: v.astype(np.float32, copy=False) for k, v in r["dend_Vm_dict"].items()}

        if "vm_by_animal" in r:
            keep["vm_by_animal"] = {
                animal: {seed: vm.astype(np.float32, copy=False) for seed, vm in seed_map.items()}
                for animal, seed_map in r["vm_by_animal"].items()
            }

        for key in ("just_plateau_starts_sums_dict", "plateau_array_dict"):
            if key in r:
                keep[key] = r[key]

        state["results"] = keep
        state["meta"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
        state["data"] = {}
        return state

    def __setstate__(self, state):
        self.cfg = state.get("cfg")
        self.data = state.get("data", {})
        self.results = state.get("results", {})
        self.meta = state.get("meta", {})



        # ---- keep your class above unchanged, but tweak this one small method:
    def save_state(self, path):
        path = Path(path)  # use Path, since it's imported already
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__getstate__(), f)

    @classmethod
    def load_state(cls, path):
        """Recreate a model from a saved state dict."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        m = cls(state["cfg"])     # uses your saved config
        m.__setstate__(state)     # restores results/meta
        return m

    @classmethod
    def load_pickle(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict) and "cfg" in obj:
            m = cls(obj["cfg"])
            m.__setstate__(obj)
            return m
        raise TypeError("Pickle does not contain a SpikingModel or compatible state dict.")


# ---------------- CLI ----------------
@click.command()
@click.option("--do-plot/--no-plot", default=True, help="Show the multi-panel plot at the end.")
@click.option("--save-path", default=None, type=str,
            help="Path to save the pickled model. If omitted, defaults to <file_path>/datasets/spiking_model_run.pkl")
@click.option("--num-seeds", default=2, type=int, show_default=True, help="Number of random seeds to simulate.")
@click.option("--which-velocity", type=click.Choice(["EC_animal_average", "repeated_waveform", "constant"],
            case_sensitive=False), default="EC_animal_average", show_default=True, help="Velocity generator to use.")
@click.option("--animal-by-animal/--regular", default=False, show_default=True,
            help="Run the pipeline per animal (separate velocity/warping/VM per animal).")
@click.option("--load-and-plot", type=str, default=None,
            help="If set, load a previously saved model from this pickle and plot only (no simulation).")
@click.option("--plot-mode", type=click.Choice(["full", "quick"]), default="full",
            help="When loading from pickle, 'quick' needs only Vm+velocity.")
@click.option("--grid-index", type=int, default=None,
            help="Index in [0..(A*S-1)] mapping to (animal, seed). Use with --animal-by-animal.")
@click.option("--max-seed", type=int, default=None,
            help="Seeds per animal for mapping grid-index (default: --num-seeds).")
@click.option("--only-animal", type=str, default=None,
            help="Run only this animal id (e.g. 'animal_1'); implies --animal-by-animal.")
@click.option("--single-seed", type=int, default=None,
            help="Run only this seed value (e.g. 7).")
@click.option("--print-animals", is_flag=True,
            help="Print detected animal IDs and exit.")
@click.option("--file-path", type=str, default=None,
            help="Root folder containing datasets/*.mat. Defaults to $SPK_MODEL_ROOT or ~/CA1-interneuron-GLM.")
@click.option("--tau-ms", type=float, default=200.0, show_default=True,
              help="Membrane time constant (ms).")
@click.option("--dend-threshold", type=float, default=-30.0, show_default=True,
              help="Dendritic plateau threshold (mV).")

def main(do_plot: bool,
        save_path: str | None,
        num_seeds: int,
        which_velocity: str,
        animal_by_animal: bool,
        load_and_plot: str | None,
        plot_mode: str,
        grid_index: int | None,
        max_seed: int | None,
        only_animal: str | None,
        single_seed: int | None,
        print_animals: bool,
        file_path: str | None,
        tau_ms:float, 
        dend_threshold:float):

    # ---- resolve file root once, no hardcoding
    root = file_path or os.environ.get("SPK_MODEL_ROOT") or str(Path.home() / "CA1-interneuron-GLM")

    cfg = SpikingModelConfig(
        file_path=root,
        tau_ms=tau_ms,
        num_seeds=num_seeds,
        dend_threshold=dend_threshold,
        which_velocity=which_velocity,
        hz_sf=50,
        vrest=-70,
        epsp_sf=0.1,
        dt_constant=0.001,
    )

    if print_animals:
        tmp = SpikingModel(cfg)
        tmp.load()
        animals = sorted(tmp.data["activity_dict_EC"].keys())
        for a in animals:
            print(a)
        return

    # If user requested a single (animal, seed) via grid-index OR explicit flags
    seed_override = None
    only_animal_eff = only_animal
    animal_by_animal = animal_by_animal or (only_animal is not None) or (grid_index is not None)

    if grid_index is not None:
        tmp = SpikingModel(cfg)
        tmp.load()
        animals = sorted(tmp.data["activity_dict_EC"].keys())
        S = max_seed if max_seed is not None else num_seeds
        A = len(animals)
        total = A * S
        if grid_index < 0 or grid_index >= total:
            raise ValueError(f"--grid-index out of range 0..{total-1} (A={A}, S={S})")
        only_animal_eff = animals[grid_index // S]
        seed_override   = grid_index % S
    elif single_seed is not None or only_animal is not None:
        seed_override = single_seed  # may be None → run all seeds for that animal

    # Resolve save path early
    if save_path is None:
        save_path = Path(cfg.file_path) / "datasets" / "spiking_model_run.pkl"
    else:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = Path(cfg.file_path) / save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- Mode 1: Load + plot only
    if load_and_plot:
        model = SpikingModel.load_pickle(load_and_plot)
        if file_path:
            model.cfg.file_path = root   # override Bridges path with local path

        if do_plot:
            try:
                if plot_mode == "full":
                    try:
                        if "vm_by_animal" in model.results:
                            model.plot_animal_by_animal()
                        else:
                            model.plot()
                    except Exception:
                        print("[info] Falling back to quick plot (minimal inputs).")
                        model.plot_quick()
                else:
                    model.plot_quick()
            except Exception as e:
                print(f"[plot] Failed while plotting loaded model: {e}")
                raise
        return

    # -------- Mode 2: Full run with checkpoint BEFORE plotting
    model = SpikingModel(cfg)
    model.load()
    model.prepare_inputs(animal_by_animal=animal_by_animal)

    # Restrict to one animal if requested
    if animal_by_animal and only_animal_eff is not None:
        pwa = model.results["padded_warped_activity_by_animal"]
        vel = model.results["velocity_by_animal"]
        if only_animal_eff not in pwa:
            raise KeyError(f"Animal '{only_animal_eff}' not found. Available: {list(pwa.keys())}")
        model.results["padded_warped_activity_by_animal"] = {only_animal_eff: pwa[only_animal_eff]}
        model.results["velocity_by_animal"] = {only_animal_eff: vel[only_animal_eff]}

    # Simulate (optionally one seed)
    model.simulate_seeds(animal_by_animal=animal_by_animal, seed_override=seed_override)

    # SAVE CHECKPOINT BEFORE PLOTTING
    model.save(save_path)
    print(f"[checkpoint] Saved pre-plot model to: {save_path.resolve()}")

    # Plot (full if possible; otherwise quick)
    if do_plot:
        try:
            if not animal_by_animal:
                model.compute_plateaus()
                model.plot()
            else:
                fn = getattr(model, "plot_animal_by_animal", None)
                fn() if callable(fn) else model.plot_quick()
        except Exception as e:
            print(f"[plot] Plot failed: {e}. Results are safe at {save_path.resolve()}")
            raise

    # Save after plotting (same path)
    model.save(save_path)
    print(f"Saved model to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
