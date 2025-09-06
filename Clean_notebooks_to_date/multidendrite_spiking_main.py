
from spiking_model_utils import *
from multidendrite_spiking_utils import *


import pathlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pickle
import click
from spiking_model_utils import *
from pathlib import Path
import sys
import platform



@dataclass
class SpikingModelConfig:
    file_path: str = "/Users/michaelfinch/CA1-interneuron-GLM"
    tau_ms: float = 200.0
    num_seeds: int = 2
    dend_threshold: float = -30.0
    which_velocity: str = "EC_animal_average"   # "EC_animal_average" | "repeated_waveform" | "constant"
    hz_sf: float = 50.0                         # scaling for get_scaled_data_Hz_dict
    vrest: float = -70.0
    epsp_sf: float = 0.1
    dt_constant: float = 0.001,
    dist: str='Uniform',
    dx: float = 180.0/50.0,


class SpikingModel:
    def __init__(self, cfg: SpikingModelConfig):
        self.cfg = cfg
        # unprocessed data in self.data and the outputs in self.results 
        self.data: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def load(self) -> None:
        """Load EC/SST/NDNF datasets and keep only the pieces we need downstream."""
        fp = self.cfg.file_path

        (GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC) = load_data_regular(file_path=fp, name="EC_GLM", new_NDNF=False)

        (GLM_params_SST, activity_dict_SST, double_predicted_activity_dict_SST, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST) = load_data_regular(file_path=fp, name="SSTindivsomata_GLM", new_NDNF=False)

        (GLM_params_NDNF_newest, activity_dict_NDNF_newest, double_predicted_activity_dict_NDNF_newest, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest) = load_data_regular(file_path=fp, name="NDNF_E1A1B", new_NDNF=True)

        fixed_filtered_factors_dict_NDNF_newest = {}
        for idx, animal in enumerate(filtered_factors_dict_NDNF_newest):
            if 17 < idx < 31:
                fixed_filtered_factors_dict_NDNF_newest[f"animal_{idx+1}"] = filtered_factors_dict_NDNF_newest[animal]

        self.data.update(dict(activity_dict_EC=activity_dict_EC, residual_activity_dict_EC=residual_activity_dict_EC, factors_dict_EC=factors_dict_EC, factors_dict_SST=factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest=fixed_filtered_factors_dict_NDNF_newest, GLM_params_EC=GLM_params_EC))


    def prepare_inputs(self, animal_by_animal: bool = False) -> None:

        activity_dict_EC = self.data["activity_dict_EC"]
        factors_dict_EC  = self.data["factors_dict_EC"]
        factors_dict_SST = self.data["factors_dict_SST"]
        fixed_NDNF       = self.data["fixed_filtered_factors_dict_NDNF_newest"]
        residual_activity_dict_EC = self.data["residual_activity_dict_EC"]
        GLM_params_EC = self.data["GLM_params_EC"]

        dist = self.cfg.dist

        print(f"dist {dist}")

        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        rng = np.random.default_rng(SEED)

        n_EC = 792
        n_SST = 75
        n_NDNF = 115
        n_dendrites=100

        def get_dendrite_list(scaled_data_Hz_dict):
            data_list = []
            for animal in scaled_data_Hz_dict:
                for cell in scaled_data_Hz_dict[animal]:
                    data_list.append(scaled_data_Hz_dict[animal][cell])

            return data_list
        
        def sample_weights(distribution, mask, rng, mean=0.1, std=0.5):
            weights = np.zeros_like(mask, dtype=float)
            n_samples = np.sum(mask)

            if distribution == "Uniform":
                samples = rng.uniform(low=mean - std, high=mean + std, size=n_samples)
            elif distribution == "Normal":
                samples = rng.normal(loc=mean, scale=std, size=n_samples)
                samples = np.clip(samples, 0, None)
            elif distribution == "Lognormal":
                samples = rng.lognormal(mean=np.log(mean), sigma=std, size=n_samples)
            elif distribution == "Equal":
                samples = np.full(n_samples, mean, dtype=float)
            else:
                raise ValueError("Invalid distribution")

            weights[mask] = samples
            return weights


        # def get_dendrite_activity(weights, EC_input_matrix, n_dendrites, n_EC):
        #     EC_flat = EC_input_matrix.reshape(n_EC, -1)
        #     dendrite_flat = weights @ EC_flat
        #     return dendrite_flat.reshape(n_dendrites, EC_input_matrix.shape[2], EC_input_matrix.shape[1])

        def get_dendrite_activity_multi(weights, EC_input_matrix, n_dendrites, n_EC):
            E, T, N = EC_input_matrix.shape
            EC_flat = EC_input_matrix.reshape(E, T*N)      # row-major: blocks of N per time bin
            dendrite_flat = weights @ EC_flat              # (D, T*N)
            return dendrite_flat.reshape(n_dendrites, T, N)

                



        an_velocity = get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_NDNF, which_type="EC_animal_average")

        
        


        an_velocity_dict = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
        scaled_data_Hz_dict_resid = get_scaled_data_Hz_dict(residual_activity_dict_EC, Hz_SF=50)
        # scaled_data_Hz_dict = add_vel_contribution_to_residuals(scaled_data_Hz_dict_resid, GLM_params_EC, an_velocity_dict)
        scaled_data_Hz_dict = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid, GLM_params_EC, an_velocity_dict)

        padded_warped_activity_dict, an_velocity = do_the_interpolation(scaled_data_Hz_dict, an_velocity=an_velocity)

        
        # dend_Vm_dict: Dict[int, np.ndarray] = {}

        # # fix this ##-- seeds 
        # # for i in range(2):
        i=0
        # epsp_dict, kernel = get_epsp_dict_multi(padded_warped_activity_dict, tau_ms=self.cfg.tau_ms, amp=1., seed=i)

        # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/epsp_dict.pkl"

        # important_dict = {"activity_EC":activity_EC, 
        #                   "weights_EC":weights_EC}

        # with open(save_path, 'wb') as f:
        #     pickle.dump(epsp_dict, f)
        #     print(f"pickle saved to {save_path}")

        # for animal in epsp_dict:
        #     for cell in epsp_dict[animal]:
        #         padded_warped_length_0 = len(epsp_dict[animal][cell]["epsps"][0])
        #         padded_warped_length_1 = len(epsp_dict[animal][cell]["epsps"][1])
        #         padded_warped_length_2 = len(epsp_dict[animal][cell]["epsps"][2])
        #         print(f"epsp_length {padded_warped_length_0} {padded_warped_length_1} {padded_warped_length_2}")

        save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/epsp_dict.pkl"

        with open(save_path, 'rb') as f:
            epsp_dict = pickle.load(f)
            print(f"pickle loaded from {save_path}")


        _, epsp_input_matrix, spike_list = get_dend_vm(epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
        epsp_input_matrix = np.transpose(epsp_input_matrix, (0, 2, 1)) 
        print(f"epsp_input_matrix shape {epsp_input_matrix.shape}")

        # T, N = epsp_input_matrix.shape[1], epsp_input_matrix.shape[2]
        # dt = 0.001  # s per bin, set to your dt_constant
        # plt.imshow(
        #     epsp_input_matrix[0].T,           # (N, T): trials x time
        #     aspect='auto',
        #     origin='lower',
        #     interpolation='nearest',
        #     extent=[0, T*dt*1000, 0, N]       # x in ms
        # )
        # plt.xlabel('time (ms)')
        # plt.ylabel('trial')
        # plt.title("First EC Cell's EPSPs (trial × time)")
        # plt.show()

        

        # ########## old version for comparison#########
        # dend_list_EC = get_dendrite_list(scaled_data_Hz_dict)
        # EC_input_matrix = np.stack(dend_list_EC[:n_EC], axis=0)
        # print(f"EC_input_matrix.shape {EC_input_matrix.shape}")

        connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
        weights_EC = sample_weights(dist, connection_mask_EC, rng=rng, mean=10.0, std=0.5)

        activity_EC = get_dendrite_activity_multi(weights_EC, epsp_input_matrix, n_dendrites, n_EC)

        # D, T, N = activity_EC.shape  # should be (n_dendrites, T, N)
        # dt = 0.001  # s
        # plt.imshow(activity_EC[0].T, aspect='auto', origin='lower',
        #         interpolation='nearest', extent=[0, T*dt*1000, 0, N])
        # plt.xlabel('time (ms)')
        # plt.ylabel('trial')
        # plt.title('First Dendrite Activity (trial × time)')
        # plt.show()


 
        # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/activity_EC.pkl"

        # important_dict = {"activity_EC":activity_EC, 
        #                   "weights_EC":weights_EC}

        # with open(save_path, 'wb') as f:
        #     pickle.dump(important_dict, f)
        #     print(f"pickle saved to {save_path}")


        # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/activity_EC.pkl"

        # with open(save_path, 'rb') as f:
        #     important_dict = pickle.load(f)
        #     print(f"pickle loaded from {save_path}")


        # activity_EC = important_dict["activity_EC"]
        # weights_EC = important_dict["weights_EC"]

        # print(f"activity_EC.shape {activity_EC.shape}")

        # activity_EC: (n_dendrites, T, 58)
        dend_Vm, activity_centered, trial_means = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

        dend_Vm = np.transpose(dend_Vm, (0, 2, 1)) 

        plt.imshow(dend_Vm[0,:,:].T, aspect='auto')
        plt.title("Dend Vm")
        plt.show()


        summed_dendrite = get_summed_dendrite_EC_DFF(self.data["residual_activity_dict_EC"])

        dend_threshold = self.cfg.dend_threshold


        activity_NDNF=0
        activity_SST=0
        NDNF_sf_opt=0
        SST_sf_opt=0

        weights_SST=0
        weights_NDNF=0

        print(f"self.cfg.dend_threshold {self.cfg.dend_threshold}")

        plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, plateau_start_times_list_mega_list, num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list = get_activity_multidendrite2(an_velocity, dend_Vm, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, self.cfg.dt_constant, self.cfg.dx, dend_threshold=self.cfg.dend_threshold, vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False)


        # dend_Vm_dict: Dict[int, np.ndarray] = {}
        # seeds = [1,2]
        # # seeds = [seed_override] if seed_override is not None else list(range(self.cfg.num_seeds))
        # for i in seeds:
        #     epsp_dict, kernel = get_epsp_dict(self.results["padded_warped_activity_dict"], tau_ms=tau, amp=1., seed=i)
        #     dend_Vm, epsp_list, spike_list = get_dend_vm(epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
        #     dend_Vm_dict[i] = dend_Vm
        # # print(f"[seed {i}] dend_vm OK   shape={dend_Vm.shape}")
    

        mean_pad = plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, dend_Vm, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition="neither", NDNF_contribution_sum=None, SST_contribution_sum=None)
        
        plt.figure(figsize=(16,4))
        for i in range(mean_pad.shape[0]):
            plt.plot(mean_pad[i,:])
        plt.show()

        #### what to do with the plateau dict 

        self.results["animal_by_animal"] = bool(animal_by_animal)

        self.results.update(dict(
            scaled_data_Hz_dict=scaled_data_Hz_dict,
            an_velocity=an_velocity,
            padded_warped_activity_dict=padded_warped_activity_dict,
            summed_dendrite=summed_dendrite,
            cumulative_plateau_counts=counts_dict_animal,
            animal_by_animal = animal_by_animal))
        
        self.results["animal_by_animal"] = bool(animal_by_animal)

        self.results.update(dict(
            scaled_data_Hz_dict=scaled_data_Hz_dict,
            an_velocity=an_velocity,
            padded_warped_activity_dict=padded_warped_activity_dict,
            summed_dendrite=summed_dendrite,
            cumulative_plateau_counts=counts_dict_animal,
            animal_by_animal = animal_by_animal))
        
    def simulate_seeds(self, animal_by_animal: bool = False, seed_override: int | None = None) -> None:
        tau = self.cfg.tau_ms

             # -------- Regular path (pooled) --------
        dend_Vm_dict: Dict[int, np.ndarray] = {}
        seeds = [seed_override] if seed_override is not None else list(range(self.cfg.num_seeds))
        for i in seeds:
            epsp_dict, kernel = get_epsp_dict(self.results["padded_warped_activity_dict"], tau_ms=tau, amp=1., seed=i)
            dend_Vm, epsp_list, spike_list = get_dend_vm(epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
            dend_Vm_dict[i] = dend_Vm
            print(f"[seed {i}] dend_vm OK   shape={dend_Vm.shape}")

        self.results.update(dict(dend_Vm_dict=dend_Vm_dict, last_kernel=kernel, seeds_run=seeds))

    def compute_plateaus(self) -> None:
        """Detect plateaus from Vm and make arrays for plotting."""
        dend_threshold = self.cfg.dend_threshold
        dend_Vm_dict = self.results["dend_Vm_dict"]

        just_plateau_starts_sums_dict, plateau_array_dict = get_plateau_array_dict(dend_Vm_dict, dend_threshold)
        self.results.update(dict(just_plateau_starts_sums_dict=just_plateau_starts_sums_dict, plateau_array_dict=plateau_array_dict))

    def plot(self) -> None:
        """Pooled (global) multi-panel figure."""
        r = self.results
        plot_dendrite_spikes_multiple_seeds(
            r["dend_Vm_dict"],
            r["an_velocity"],
            residual_activity_dict_EC=self.data["residual_activity_dict_EC"],
            animal=None,
            animal_by_animal=False,
            dend_threshold=self.cfg.dend_threshold,
            tau=self.cfg.tau_ms,
            num_seeds=self.cfg.num_seeds)
        
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
        # fp = self.cfg.file_path
        fp="/Users/michaelfinch/CA1-interneuron-GLM"
        GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(file_path=fp, name="EC_GLM", new_NDNF=False)
        residual = residual_activity_dict_EC

        # Prefer explicit mode if you add others later; for now 'auto' handles both.
        if "vm_by_animal" in self.results and "velocity_by_animal" in self.results:
            # Animal-by-animal plotting
            vm_by_animal  = self.results["vm_by_animal"]
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
                    an_velocity = se
                except TypeError:
                    # if your util doesn't accept residuals per animal:
                    plot_dendrite_spikes_multiple_seeds(
                        seed_vm_dict,
                        vel,
                        residual_activity_dict_EC=None,
                        animal=animal,
                        animal_by_animal=True,
                        dend_threshold=self.cfg.dend_threshold,
                        tau=self.cfg.tau_ms,
                        num_seeds=self.cfg.num_seeds,
                    )
                    return
        # Regular (global) plotting
        if "dend_Vm_dict" in self.results and "an_velocity" in self.results:

            an_velocity = self.results["an_velocity"]
            weights_EC = self.results("weights_EC")
            activity_EC = self.results("activity_EC")
            dend_threshold = self.cfg.dend_threshold


            plot_multidendrite_EC(weights_EC, activity_EC, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None)
                
            # plot_dendrite_spikes_multiple_seeds(
            #     self.results["dend_Vm_dict"],
            #     ,
            #     residual_activity_dict_EC=residual,
            #     animal=None,
            #     animal_by_animal=False,
            #     dend_threshold=self.cfg.dend_threshold,
            #     tau=self.cfg.tau_ms,
            #     num_seeds=self.cfg.num_seeds,
            # )
            # return
            # except TypeError:
            #     plot_dendrite_spikes_multiple_seeds(
            #         self.results["dend_Vm_dict"],
            #         self.results["an_velocity"],
            #         self.cfg.dend_threshold,
            #         self.cfg.tau_ms,
            #         self.cfg.num_seeds,
            #     )
            # return

        # Nothing recognizable to plot
        have = list(self.results.keys())
        raise KeyError(f"[plot_quick] No VM/velocity found to plot. results has keys: {have}")


            
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

        keep["animal_by_animal"] = r.get("animal_by_animal", False)

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

    def save_state(self, path):
        path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
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
        # If it's already a SpikingModel, you're done
        if isinstance(obj, cls):
            return obj
        # If it's a compact state dict, rebuild
        if isinstance(obj, dict) and "cfg" in obj:
            m = cls(obj["cfg"])
            m.__setstate__(obj)
            return m
        raise TypeError("Pickle does not contain a SpikingModel or compatible state dict.")

            

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
@click.option("--tau-ms", type=float, default=200.0, show_default=True,
              help="Membrane time constant (ms) for EPSP kernel.")
@click.option("--dend-threshold", type=float, default=-30.0, show_default=True,
              help="Threshold (mV) for plateau detection.")
@click.option("--dist", type=str, default="Uniform", show_default=True,
              help="Weight Dist From EC to CA1 Dendrite")





def main(do_plot: bool, save_path: str | None, num_seeds: int, which_velocity: str, animal_by_animal: bool, load_and_plot: str | None, plot_mode: str, grid_index:int, max_seed:int, only_animal:str, single_seed:int, print_animals:bool, tau_ms: float, dend_threshold: float, dist: str):

    cfg = SpikingModelConfig(
        file_path="/Users/michaelfinch/CA1-interneuron-GLM",
        tau_ms=tau_ms,
        num_seeds=num_seeds,
        dend_threshold=dend_threshold,
        which_velocity=which_velocity,
        hz_sf=50,
        vrest=-70,
        epsp_sf=0.1,
        dt_constant=0.001,
        dist=dist, 
        dx=180./50.,
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



    # Resolve save path early (used for checkpoint as well)
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
        if do_plot:
            try:
                if plot_mode == "full":
                    # Try the full plots first; if missing heavy data, fall back
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
                # Pooled mode has full panels
                model.compute_plateaus()
                model.plot()
            else:
                # Per-animal mode uses minimal plotting
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


        
            
    

# vel_applied = "real"   #real or constant 
    
# wt_dist = "Lognormal"   #Uniform, Constant, Lognormal 


# add_inh = 'neither' #options: both, sst, neither

# dend_threshold = 1.0


# SST_bias_multi = 1.4

# # SST_bias_factor_list = [1.4, 1.6, 1.8, 2.0]

# # for SST_bias_multi in SST_bias_factor_list:


# an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = multi_wrap_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, vel_applied=vel_applied, add_inh=add_inh, SST_bias_factor=SST_bias_multi, dist=wt_dist)
    
# activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list = get_activity_multidendrite(dend_contribution_EC, an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, dend_threshold=dend_threshold, vel_applied=vel_applied, example_cell=17, dist=wt_dist, n_dendrites=100, n_SST=75, n_EC=792, n_NDNF=73, include_inhibition=add_inh, use_model_EC=False)

# plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list, include_inhibition=add_inh, NDNF_contribution_sum=NDNF_contribution_sum, SST_contribution_sum=SST_contribution_sum)
    

        




