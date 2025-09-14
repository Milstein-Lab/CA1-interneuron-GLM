
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

import optuna
from optuna.storages import RDBStorage

def make_objective(static_inputs):
    # close over your preloaded data + functions and return a callable
    def objective(trial):
        params = dict(
            tau_ms=trial.suggest_float("tau_ms", 3.0, 50.0),
            dend_threshold=trial.suggest_float("dend_threshold", -80.0, -50.0),
            weights_mean=trial.suggest_float("weights_mean", 0.5, 12.0),
            weights_std=trial.suggest_float("weights_std", 0.1, 1.5),
        )
        return static_inputs["priority_loss"](params)
    return objective

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
    dt_constant: float = 0.001
    dist: str='Uniform'
    dx: float = 180.0/50.0
    weights_mean: float = 1.0
    weights_std:float = 0.5


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


        fixed_residual_activity_dict_NDNF_newest = {}
        for idx, animal in enumerate(residual_activity_dict_NDNF_newest):
            if 17 < idx < 31:
                fixed_residual_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_newest[animal]

        self.data.update(dict(activity_dict_EC=activity_dict_EC, residual_activity_dict_EC=residual_activity_dict_EC, factors_dict_EC=factors_dict_EC, factors_dict_SST=factors_dict_SST, fixed_filtered_factors_dict_NDNF_newest=fixed_filtered_factors_dict_NDNF_newest, GLM_params_EC=GLM_params_EC, GLM_params_SST=GLM_params_SST, GLM_params_NDNF=GLM_params_NDNF_newest, residual_activity_dict_SST=residual_activity_dict_SST, fixed_residual_activity_dict_NDNF_newest=fixed_residual_activity_dict_NDNF_newest))

    def optimize_params_for_animal(
        self,
        animal: str,
        pwa_cell_dict,              # padded_warped_by_animal[animal]  -> {cell: [warped trials]}
        animal_velocity,            # (50, n_trials)
        n_dendrites: int,
        n_EC: int,
        dist: str,
        seed_list=None,             # seeds for objective averaging
        penalty: float = 1e6
    ):
        """
        Returns: best_params (dict) with keys: tau_ms, dend_threshold, weights_mean, weights_std
        """

        import numpy as np

        if seed_list is None:
            seed_list = list(range(5))  # use 5 seeds for search, validate later with more

        # ---- helpers ----
        def ten_bin_fraction_from_counter(cnt50):
            cnt50 = np.asarray(cnt50, float)
            agg = np.add.reduceat(cnt50, np.arange(0, 50, 5))
            s = agg.sum()
            return agg / s if s > 0 else np.full(10, 1/10, float)

        def eval_one_seed(seed, tau_ms, weights_mean, weights_std, dend_threshold):
            epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(seed))
            dend_Vm_cells, epsp_input_matrix, _ = get_dend_vm_from_cells_multi(
                epsp_cells, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf
            )
            # ensure (E,T,N)
            if epsp_input_matrix.shape[1] == animal_velocity.shape[1]:  # (E,N,T)
                epsp_input_matrix_eTN = np.transpose(epsp_input_matrix, (0, 2, 1))
            else:
                epsp_input_matrix_eTN = epsp_input_matrix

            # weights
            rng = np.random.default_rng(12345 + int(seed))
            connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
            weights_EC = sample_weights(dist, connection_mask_EC, rng=rng,
                                        mean=float(weights_mean), std=float(weights_std))

            # EC -> dend
            activity_EC = get_dendrite_activity_multi(weights_EC, epsp_input_matrix_eTN, n_dendrites, n_EC)
            dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

            (plateau_positions_counter,
            plateau_start_positions_counter,
            plateau_array_per_dendrite_list,
            dendrite_plateau_mask,
            plateau_start_times_list_mega_list,
            num_plateaus_per_dend_list,
            _,
            _) = get_activity_multidendrite2(
                animal_velocity, dend_Vm,
                activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
                dt_constant=self.cfg.dt_constant, dx=self.cfg.dx,
                dend_threshold=float(dend_threshold),
                vel_applied="real", example_cell=15,
                include_inhibition=True, use_model_EC=False
            )

            num_per_dend = np.asarray(num_plateaus_per_dend_list, float)
            start_pos_counter_50 = np.asarray(plateau_start_positions_counter, float)
            return num_per_dend, start_pos_counter_50

        target_total = n_dendrites / 2.0
        target_fraction_10 = np.array([5,5,5,5,20,20,10,10,7,5], float)
        target_fraction_10 /= target_fraction_10.sum()

        SEEDS_OPT = [0,1,2,3,4]

        def priority_loss_for_animal(self, pwa_cell_dict, animal_velocity, n_dendrites, n_EC, dist, params, seed_list):
            def ten_bin_fraction_from_counter(cnt50):
                cnt50 = np.asarray(cnt50, float)
                agg = np.add.reduceat(cnt50, np.arange(0, 50, 5))
                s = agg.sum()
                return agg / s if s > 0 else np.full(10, 1/10, float)

            def eval_one_seed(seed):
                tau_ms         = float(params["tau_ms"])
                weights_mean   = float(params["weights_mean"])
                weights_std    = float(params["weights_std"])
                dend_threshold = float(params["dend_threshold"])
                epsp_cells, _ = get_epsp_dict_animal(pwa_cell_dict, tau_ms=tau_ms, amp=1., seed=int(seed))
                _, epsp_input_matrix, _ = get_dend_vm_from_cells_multi(epsp_cells, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
                epsp_eTN = np.transpose(epsp_input_matrix, (0, 2, 1)) if epsp_input_matrix.shape[1] == animal_velocity.shape[1] else epsp_input_matrix
                rng = np.random.default_rng(12345 + int(seed))
                connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
                weights_EC = sample_weights(dist, connection_mask_EC, rng=rng, mean=weights_mean, std=weights_std)
                activity_EC = get_dendrite_activity_multi(weights_EC, epsp_eTN, n_dendrites, n_EC)
                dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")
                _, start_cnt, _, _, _, num_per_dend, _, _ = get_activity_multidendrite2(
                    animal_velocity, dend_Vm, activity_NDNF=0, activity_SST=0, NDNF_sf_opt=0, SST_sf_opt=0,
                    dt_constant=self.cfg.dt_constant, dx=self.cfg.dx, dend_threshold=dend_threshold,
                    vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False
                )
                return np.asarray(num_per_dend, float), np.asarray(start_cnt, float)

            target_total = n_dendrites / 2.0
            target_frac = np.array([5,5,5,5,20,20,10,10,7,5], float); target_frac /= target_frac.sum()

            violations = 0.0
            totals = []
            frac_sum = np.zeros(10, float)
            for s in seed_list:
                num_per_dend, start_pos_counter_50 = eval_one_seed(s)
                violations += np.maximum(0.0, num_per_dend - 2.0).sum()
                totals.append(start_pos_counter_50.sum())
                frac_sum += ten_bin_fraction_from_counter(start_pos_counter_50)
            if violations > 0:
                return float(1e6 * violations)
            mean_total = float(np.mean(totals))
            mse_total = (mean_total - target_total) ** 2
            mean_frac = frac_sum / len(seed_list)
            mse_frac = float(np.mean((mean_frac - target_frac) ** 2))
            return float(mse_total + mse_frac)

    


    def prepare_inputs(self, animal_by_animal: bool = False, seed_override = float) -> None:

        activity_dict_EC = self.data["activity_dict_EC"]
        factors_dict_EC  = self.data["factors_dict_EC"]
        factors_dict_SST = self.data["factors_dict_SST"]
        fixed_NDNF       = self.data["fixed_filtered_factors_dict_NDNF_newest"]
        residual_activity_dict_EC = self.data["residual_activity_dict_EC"]
        residual_activity_dict_SST = self.data["residual_activity_dict_SST"]
        fixed_residual_activity_dict_NDNF_newest = self.data["fixed_residual_activity_dict_NDNF_newest"]
        GLM_params_EC = self.data["GLM_params_EC"]
        tau = self.cfg.tau_ms

        weights_mean = self.cfg.weights_mean
        weights_std = self.cfg.weights_std

        activity_NDNF=0
        activity_SST=0
        NDNF_sf_opt = 0
        SST_sf_opt = 0
        weights_SST = 0
        weights_NDNF = 0

        if animal_by_animal:
            print("MADE IT TO ANIMAL / ANIMAL")

            dist = self.cfg.dist
            print(f"dist {dist}")

            SEED = 42
            np.random.seed(SEED)
            random.seed(SEED)
            rng = np.random.default_rng(SEED)

            # n_EC = 792
            n_SST = 75
            n_NDNF = 115
            n_dendrites=100


            scaled_data_Hz_dict, cells_per_animal_dict = get_scaled_data_Hz_dict(activity_dict_EC, Hz_SF=self.cfg.hz_sf)

            an_velocity_by_animal = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)

            # Interpolate/warp per animal
            padded_warped_by_animal = {}
            for animal in scaled_data_Hz_dict:
                pwa_cell_dict, _ = do_the_interpolation_an(
                    scaled_data_Hz_dict[animal],      # {cell: (n_pos, n_trials)}
                    an_velocity_by_animal[animal],    # (n_pos, n_trials)
                    dt_constant=self.cfg.dt_constant
                )
                padded_warped_by_animal[animal] = pwa_cell_dict



            # === NEW: optimize before building dend_vm_per_animal_dict ===
            animal_list = ['animal_1']  # or loop all animals
            for animal in animal_list:
                n_EC = cells_per_animal_dict[animal]
                best = self.optimize_params_for_animal(
                    animal=animal,
                    pwa_cell_dict=padded_warped_by_animal[animal],
                    animal_velocity=an_velocity_by_animal[animal],
                    n_dendrites=100,               # your n_dendrites
                    n_EC=n_EC,
                    dist=self.cfg.dist,
                    seed_list=list(range(5))       # fixed seeds during search
                )
                print(f"[{animal}] best params: {best}")

            # Now proceed to (re)build with optimized self.cfg.tau_ms / dend_threshold / weights_mean/std
            dend_vm_per_animal_dict = {}
            animal_weights_dict = {}
            epsp_input_matrix_dict = {}
            for animal in animal_list:
                dend_vm_per_seed_dict = {}
                weights_seed_dict = {}
                epsp_input_matrix_seed_dict = {}
                for i in range(2):  # or your desired seeds to actually run/plot
                    epsp_cells, kernel = get_epsp_dict_animal(
                        padded_warped_by_animal[animal], tau_ms=self.cfg.tau_ms, amp=1., seed=i
                    )
                    dend_Vm, epsp_input_matrix, spike_mats = get_dend_vm_from_cells_multi(
                        epsp_cells, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf
                    )
                    epsp_input_matrix_seed_dict[i] = epsp_input_matrix
                    n_EC = cells_per_animal_dict[animal]
                    connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
                    weights_EC = sample_weights(self.cfg.dist, connection_mask_EC, rng=rng,
                                                mean=self.cfg.weights_mean, std=self.cfg.weights_std)
                    weights_seed_dict[i] = weights_EC
                    # ensure (E,T,N)
                    if epsp_input_matrix.shape[1] == an_velocity_by_animal[animal].shape[1]:
                        epsp_input_matrix = np.transpose(epsp_input_matrix, (0, 2, 1))

                    activity_EC = get_dendrite_activity_multi(weights_EC, epsp_input_matrix, n_dendrites, n_EC)
                    dend_Vm, _, _ = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")
                    dend_vm_per_seed_dict[i] = dend_Vm
                dend_vm_per_animal_dict[animal] = dend_vm_per_seed_dict
                animal_weights_dict[animal] = weights_seed_dict
                epsp_input_matrix_dict[animal] = epsp_input_matrix_seed_dict


            save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/dend_vm_per_animal_dict.pkl"

            important_dict = {"dend_vm_per_animal_dict":dend_vm_per_animal_dict, 
                "weights_EC":animal_weights_dict,
                "epsp_input_matrix":epsp_input_matrix}

            with open(save_path, 'wb') as f:
                pickle.dump(important_dict, f)
                print(f"pickle saved to {save_path}")



            activity_dict_EC = epsp_input_matrix


            for animal in dend_vm_per_animal_dict:

                print(f"dend_vm_per_animal_dict.keys() {dend_vm_per_animal_dict.keys()}")

                dend_vm_per_seed_dict = dend_vm_per_animal_dict[animal]


                print(f"an_velocity_by_animal[animal].shape{an_velocity_by_animal[animal].shape}")


                plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, plateau_start_times_list_mega_list, num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list = get_activity_multidendrite2_multiple_seeds(an_velocity_by_animal[animal], dend_vm_per_seed_dict, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, self.cfg.dt_constant, self.cfg.dx, self.cfg.dend_threshold, vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False)
            

                cumulative_plateau_counts, plateau_fraction_by_pos_bin, plateau_start_positions_counter  = plot_multidendrite_EC_multiple_seeds(weights_EC, weights_SST, weights_NDNF, dend_vm_per_seed_dict, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, an_velocity_by_animal[animal], dend_activity, self.cfg.dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, animal, example_cell=1, include_inhibition="neither", NDNF_contribution_sum=None, SST_contribution_sum=None, animal_by_animal=True)
                
                optimal_total_cumulative_plateau_counts = n_dendrites/2

                MSE_total_plateau_counts = np.mean(np.square(optimal_total_cumulative_plateau_counts - cumulative_plateau_counts))

                optimal_fraction_plateaus_array = np.array([5,5,5,5,20,20,10,10,7,5])

                MSE_plateau_fraction = np.mean(np.square(plateau_fraction_by_pos_bin - optimal_fraction_plateaus_array))

                print(f"plateau_start_positions_counter {plateau_start_positions_counter}")


                    
        else:

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


            an_velocity = get_velocity_array(factors_dict_EC, factors_dict_SST, fixed_NDNF, which_type="EC_animal_average")

            
            an_velocity_dict = get_velocity_array_every_animal(factors_dict_EC, n_trials=58)
            scaled_data_Hz_dict_resid_EC = get_scaled_data_Hz_dict(residual_activity_dict_EC, Hz_SF=50)
            scaled_data_Hz_dict_EC = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_EC, GLM_params_EC, an_velocity_dict)

            scaled_data_Hz_dict_resid_SST = get_scaled_data_Hz_dict(residual_activity_dict_SST, Hz_SF=50)
            scaled_data_Hz_dict_SST = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_SST, GLM_params_EC, an_velocity_dict)

            scaled_data_Hz_dict_resid_NDNF = get_scaled_data_Hz_dict(residual_activity_dict_SST, Hz_SF=50)
            scaled_data_Hz_dict_NDNF = add_vel_contribution_to_residuals_strict(scaled_data_Hz_dict_resid_NDNF, GLM_params_EC, an_velocity_dict)



            padded_warped_activity_dict, an_velocity = do_the_interpolation(scaled_data_Hz_dict_EC, an_velocity=an_velocity)

            
            i=0
            epsp_dict, kernel = get_epsp_dict_multi(padded_warped_activity_dict, tau_ms=tau, amp=1., seed=i)

            save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/epsp_dict.pkl"

            important_dict = {"activity_EC":activity_EC, 
                            "weights_EC":weights_EC}

            with open(save_path, 'wb') as f:
                pickle.dump(epsp_dict, f)
                print(f"pickle saved to {save_path}")

            for animal in epsp_dict:
                for cell in epsp_dict[animal]:
                    padded_warped_length_0 = len(epsp_dict[animal][cell]["epsps"][0])
                    padded_warped_length_1 = len(epsp_dict[animal][cell]["epsps"][1])
                    padded_warped_length_2 = len(epsp_dict[animal][cell]["epsps"][2])
                    print(f"epsp_length {padded_warped_length_0} {padded_warped_length_1} {padded_warped_length_2}")

            save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/epsp_dict.pkl"

            with open(save_path, 'rb') as f:
                epsp_dict = pickle.load(f)
                print(f"pickle loaded from {save_path}")


            _, epsp_input_matrix, spike_list = get_dend_vm(epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
            epsp_input_matrix = np.transpose(epsp_input_matrix, (0, 2, 1)) 
            print(f"epsp_input_matrix shape {epsp_input_matrix.shape}")

    
            connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
            weights_EC = sample_weights(dist, connection_mask_EC, rng=rng, mean=10.0, std=0.5)

            activity_EC = get_dendrite_activity_multi(weights_EC, epsp_input_matrix, n_dendrites, n_EC)

          
            dend_Vm, activity_centered, trial_means = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

            dend_Vm = np.transpose(dend_Vm, (0, 2, 1)) 

            plt.imshow(dend_Vm[0,:,:].T, aspect='auto')
            plt.title("Dend Vm")
            plt.show()


            summed_dendrite = get_summed_dendrite_EC_DFF(self.data["residual_activity_dict_EC"])


            activity_NDNF=0
            activity_SST=0
            NDNF_sf_opt=0
            SST_sf_opt=0

            weights_SST=0
            weights_NDNF=0

            print(f"self.cfg.dend_threshold {self.cfg.dend_threshold}")

            plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, plateau_start_times_list_mega_list, num_plateaus_per_dend_list, dend_activity, padded_warped_activity_list = get_activity_multidendrite2(an_velocity, dend_Vm, activity_NDNF, activity_SST, NDNF_sf_opt, SST_sf_opt, self.cfg.dt_constant, self.cfg.dx, dend_threshold=self.cfg.dend_threshold, vel_applied="real", example_cell=15, include_inhibition=True, use_model_EC=False)
        

            cumulative_plateau_counts, plateau_fraction_by_pos_bin = plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, dend_Vm, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask,  plateau_start_times_list_mega_list, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition="neither", NDNF_contribution_sum=None, SST_contribution_sum=None)
            
            print(f"np.max(cumulative_plateau_counts) {np.max(cumulative_plateau_counts)}")

        self.results["animal_by_animal"] = bool(animal_by_animal)

        self.results.update(dict(
            scaled_data_Hz_dict=scaled_data_Hz_dict,
            an_velocity=an_velocity,
            padded_warped_activity_dict=padded_warped_activity_dict,
            summed_dendrite=summed_dendrite,
            # cumulative_plateau_counts=counts_dict_animal,
            animal_by_animal = animal_by_animal))
        
        self.results["animal_by_animal"] = bool(animal_by_animal)

        

if __name__ == "__main__":
    # 1) Preload data once (velocity, warped EC, etc.) into static_inputs
    #    Build closures you already wrote: eval_one_seed, ten_bin_fraction_from_counter, priority_loss
    static_inputs = build_static_inputs()  # <- you already have this logic in your class; expose it

    # 2) Connect all workers to the same study
    storage = RDBStorage(url="sqlite:///ec_optuna.db")  # OK on one node; use Postgres if across many nodes
    study = optuna.create_study(
        study_name="ec_param_search",
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(make_objective(static_inputs), n_trials=100, show_progress_bar=False)
        
#     def simulate_seeds(self, animal_by_animal: bool = False, seed_override: int | None = None) -> None:
#         tau = self.cfg.tau_ms

#              # -------- Regular path (pooled) --------
#         dend_Vm_dict: Dict[int, np.ndarray] = {}
#         seeds = [seed_override] if seed_override is not None else list(range(self.cfg.num_seeds))
#         for i in seeds:
#             epsp_dict, kernel = get_epsp_dict(self.results["padded_warped_activity_dict"], tau_ms=tau, amp=1., seed=i)
#             dend_Vm, epsp_list, spike_list = get_dend_vm(epsp_dict, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)
#             dend_Vm_dict[i] = dend_Vm
#             print(f"[seed {i}] dend_vm OK   shape={dend_Vm.shape}")

#         self.results.update(dict(dend_Vm_dict=dend_Vm_dict, last_kernel=kernel, seeds_run=seeds))

#     def compute_plateaus(self) -> None:
#         """Detect plateaus from Vm and make arrays for plotting."""
#         dend_threshold = self.cfg.dend_threshold
#         dend_Vm_dict = self.results["dend_Vm_dict"]

#         just_plateau_starts_sums_dict, plateau_array_dict = get_plateau_array_dict(dend_Vm_dict, dend_threshold)
#         self.results.update(dict(just_plateau_starts_sums_dict=just_plateau_starts_sums_dict, plateau_array_dict=plateau_array_dict))

#     def plot(self) -> None:
#         """Pooled (global) multi-panel figure."""
#         r = self.results
#         plot_dendrite_spikes_multiple_seeds(
#             r["dend_Vm_dict"],
#             r["an_velocity"],
#             residual_activity_dict_EC=self.data["residual_activity_dict_EC"],
#             animal=None,
#             animal_by_animal=False,
#             dend_threshold=self.cfg.dend_threshold,
#             tau=self.cfg.tau_ms,
#             num_seeds=self.cfg.num_seeds)
        
#     def run(self, do_plot: bool = True, animal_by_animal: bool = False) -> Dict[str, Any]:
#             self.load()
#             self.prepare_inputs(animal_by_animal=animal_by_animal)
#             self.simulate_seeds(animal_by_animal=animal_by_animal)

#             if not animal_by_animal:
#                 self.compute_plateaus()
#                 if do_plot:
#                     self.plot()
#             else:
#                 # Optional: if you want plots per animal, implement self.plot_animal_by_animal()
#                 if do_plot and "vm_by_animal" in self.results:
#                     self.plot_animal_by_animal()  # see stub below (optional)

#             return self.results
        
#     def plot_quick(self, mode: str = "auto") -> None:
#         """
#         Plot without recomputing anything. Works for both:
#         - regular runs: results['dend_Vm_dict'] + results['an_velocity']
#         - animal-by-animal runs: results['vm_by_animal'] + results['velocity_by_animal']
#         Calls the util using positional args to avoid keyword mismatches.
#         """
#         # fp = self.cfg.file_path
#         fp="/Users/michaelfinch/CA1-interneuron-GLM"
#         GLM_params_EC, activity_dict_EC, double_predicted_activity_dict_EC, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(file_path=fp, name="EC_GLM", new_NDNF=False)
#         residual = residual_activity_dict_EC

#         # Prefer explicit mode if you add others later; for now 'auto' handles both.
#         if "vm_by_animal" in self.results and "velocity_by_animal" in self.results:
#             # Animal-by-animal plotting
#             vm_by_animal  = self.results["vm_by_animal"]
#             vel_by_animal = self.results["velocity_by_animal"]

#             for animal, seed_vm_dict in vm_by_animal.items():
#                 vel = vel_by_animal.get(animal, None)
#                 if vel is None:
#                     print(f"[plot_quick] Skipping {animal}: no velocity.")
#                     continue
#                 print(f"[plot_quick] plotting animal={animal}")

#                 # Call util with positional args only.
#                 # Try with residual; if the util doesn't accept it, fall back without.
#                 try:
#                     an_velocity = se
#                 except TypeError:
#                     # if your util doesn't accept residuals per animal:
#                     plot_dendrite_spikes_multiple_seeds(
#                         seed_vm_dict,
#                         vel,
#                         residual_activity_dict_EC=None,
#                         animal=animal,
#                         animal_by_animal=True,
#                         dend_threshold=self.cfg.dend_threshold,
#                         tau=self.cfg.tau_ms,
#                         num_seeds=self.cfg.num_seeds,
#                     )
#                     return
#         # Regular (global) plotting
#         if "dend_Vm_dict" in self.results and "an_velocity" in self.results:

#             an_velocity = self.results["an_velocity"]
#             weights_EC = self.results("weights_EC")
#             activity_EC = self.results("activity_EC")
#             dend_threshold = self.cfg.dend_threshold


#             plot_multidendrite_EC(weights_EC, activity_EC, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list, example_cell=1, include_inhibition=None, NDNF_contribution_sum=None, SST_contribution_sum=None)
                
#             # plot_dendrite_spikes_multiple_seeds(
#             #     self.results["dend_Vm_dict"],
#             #     ,
#             #     residual_activity_dict_EC=residual,
#             #     animal=None,
#             #     animal_by_animal=False,
#             #     dend_threshold=self.cfg.dend_threshold,
#             #     tau=self.cfg.tau_ms,
#             #     num_seeds=self.cfg.num_seeds,
#             # )
#             # return
#             # except TypeError:
#             #     plot_dendrite_spikes_multiple_seeds(
#             #         self.results["dend_Vm_dict"],
#             #         self.results["an_velocity"],
#             #         self.cfg.dend_threshold,
#             #         self.cfg.tau_ms,
#             #         self.cfg.num_seeds,
#             #     )
#             # return

#         # Nothing recognizable to plot
#         have = list(self.results.keys())
#         raise KeyError(f"[plot_quick] No VM/velocity found to plot. results has keys: {have}")


            
#     def save(self, path: Path) -> None:
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "wb") as f:
#             pickle.dump(self, f)
    
#     def __getstate__(self):
#         state = {"cfg": self.cfg}
#         r = self.results
#         keep = {}

#         if "an_velocity" in r and isinstance(r["an_velocity"], np.ndarray):
#             keep["an_velocity"] = r["an_velocity"].astype(np.float32, copy=False)

#         if "velocity_by_animal" in r:
#             keep["velocity_by_animal"] = {a: v.astype(np.float32, copy=False) for a, v in r["velocity_by_animal"].items()}

#         if "dend_Vm_dict" in r:
#             keep["dend_Vm_dict"] = {k: v.astype(np.float32, copy=False) for k, v in r["dend_Vm_dict"].items()}

#         if "vm_by_animal" in r:
#             keep["vm_by_animal"] = {
#                 animal: {seed: vm.astype(np.float32, copy=False) for seed, vm in seed_map.items()}
#                 for animal, seed_map in r["vm_by_animal"].items()
#             }

#         for key in ("just_plateau_starts_sums_dict", "plateau_array_dict"):
#             if key in r:
#                 keep[key] = r[key]

#         keep["animal_by_animal"] = r.get("animal_by_animal", False)

#         state["results"] = keep
#         state["meta"] = {
#             "python": sys.version,
#             "platform": platform.platform(),
#             "numpy": np.__version__,
#         }
#         state["data"] = {}
#         return state


#     def __setstate__(self, state):
#         self.cfg = state.get("cfg")
#         self.data = state.get("data", {})
#         self.results = state.get("results", {})
#         self.meta = state.get("meta", {})

#     def save_state(self, path):
#         path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "wb") as f:
#             pickle.dump(self.__getstate__(), f)

#     @classmethod
#     def load_state(cls, path):
#         """Recreate a model from a saved state dict."""
#         import pickle
#         with open(path, "rb") as f:
#             state = pickle.load(f)
#         m = cls(state["cfg"])     # uses your saved config
#         m.__setstate__(state)     # restores results/meta
#         return m
    
#     @classmethod
#     def load_pickle(cls, path):
#         with open(path, "rb") as f:
#             obj = pickle.load(f)
#         # If it's already a SpikingModel, you're done
#         if isinstance(obj, cls):
#             return obj
#         # If it's a compact state dict, rebuild
#         if isinstance(obj, dict) and "cfg" in obj:
#             m = cls(obj["cfg"])
#             m.__setstate__(obj)
#             return m
#         raise TypeError("Pickle does not contain a SpikingModel or compatible state dict.")

            

# @click.command()
# @click.option("--do-plot/--no-plot", default=True, help="Show the multi-panel plot at the end.")
# @click.option("--save-path", default=None, type=str,
#               help="Path to save the pickled model. If omitted, defaults to <file_path>/datasets/spiking_model_run.pkl")
# @click.option("--num-seeds", default=2, type=int, show_default=True, help="Number of random seeds to simulate.")
# @click.option("--which-velocity", type=click.Choice(["EC_animal_average", "repeated_waveform", "constant"],
#               case_sensitive=False), default="EC_animal_average", show_default=True, help="Velocity generator to use.")
# @click.option("--animal-by-animal/--regular", default=False, show_default=True,
#               help="Run the pipeline per animal (separate velocity/warping/VM per animal).")
# @click.option("--load-and-plot", type=str, default=None,
#               help="If set, load a previously saved model from this pickle and plot only (no simulation).")
# @click.option("--plot-mode", type=click.Choice(["full", "quick"]), default="full",
#               help="When loading from pickle, 'quick' needs only Vm+velocity.")

# @click.option("--grid-index", type=int, default=None,
#               help="Index in [0..(A*S-1)] mapping to (animal, seed). Use with --animal-by-animal.")
# @click.option("--max-seed", type=int, default=None,
#               help="Seeds per animal for mapping grid-index (default: --num-seeds).")
# @click.option("--only-animal", type=str, default=None,
#               help="Run only this animal id (e.g. 'animal_1'); implies --animal-by-animal.")
# @click.option("--single-seed", type=int, default=None,
#               help="Run only this seed value (e.g. 7).")
# @click.option("--print-animals", is_flag=True,
#               help="Print detected animal IDs and exit.")
# @click.option("--tau-ms", type=float, default=200.0, show_default=True,
#               help="Membrane time constant (ms) for EPSP kernel.")
# @click.option("--dend-threshold", type=float, default=-30.0, show_default=True,
#               help="Threshold (mV) for plateau detection.")
# @click.option("--dist", type=str, default="Uniform", show_default=True,
#               help="Weight Dist From EC to CA1 Dendrite")





# def main(do_plot: bool, save_path: str | None, num_seeds: int, which_velocity: str, animal_by_animal: bool, load_and_plot: str | None, plot_mode: str, grid_index:int, max_seed:int, only_animal:str, single_seed:int, print_animals:bool, tau_ms: float, dend_threshold: float, dist: str):

#     cfg = SpikingModelConfig(
#         file_path="/Users/michaelfinch/CA1-interneuron-GLM",
#         tau_ms=tau_ms,
#         num_seeds=num_seeds,
#         dend_threshold=dend_threshold,
#         which_velocity=which_velocity,
#         hz_sf=50,
#         vrest=-70,
#         epsp_sf=0.1,
#         dt_constant=0.001,
#         dist=dist, 
#         dx=180./50.,
#     )

#     if print_animals:
#         tmp = SpikingModel(cfg)
#         tmp.load()
#         animals = sorted(tmp.data["activity_dict_EC"].keys())
#         for a in animals:
#             print(a)
#         return
    
#     # If user requested a single (animal, seed) via grid-index OR explicit flags
#     seed_override = None
#     only_animal_eff = only_animal
#     animal_by_animal = animal_by_animal or (only_animal is not None) or (grid_index is not None)

#     if grid_index is not None:
#         tmp = SpikingModel(cfg)
#         tmp.load()
#         animals = sorted(tmp.data["activity_dict_EC"].keys())
#         S = max_seed if max_seed is not None else num_seeds
#         A = len(animals)
#         total = A * S
#         if grid_index < 0 or grid_index >= total:
#             raise ValueError(f"--grid-index out of range 0..{total-1} (A={A}, S={S})")
#         only_animal_eff = animals[grid_index // S]
#         seed_override   = grid_index % S
#     elif single_seed is not None or only_animal is not None:
#         seed_override = single_seed  # may be None → run all seeds for that animal



#     # Resolve save path early (used for checkpoint as well)
#     if save_path is None:
#         save_path = Path(cfg.file_path) / "datasets" / "spiking_model_run.pkl"
#     else:
#         save_path = Path(save_path)
#         if not save_path.is_absolute():
#             save_path = Path(cfg.file_path) / save_path
#     save_path.parent.mkdir(parents=True, exist_ok=True)

#     # -------- Mode 1: Load + plot only
#     if load_and_plot:
#         model = SpikingModel.load_pickle(load_and_plot)
#         if do_plot:
#             try:
#                 if plot_mode == "full":
#                     # Try the full plots first; if missing heavy data, fall back
#                     try:
#                         if "vm_by_animal" in model.results:
#                             model.plot_animal_by_animal()
#                         else:
#                             model.plot()
#                     except Exception:
#                         print("[info] Falling back to quick plot (minimal inputs).")
#                         model.plot_quick()
#                 else:
#                     model.plot_quick()
#             except Exception as e:
#                 print(f"[plot] Failed while plotting loaded model: {e}")
#                 raise
#         return

#     # -------- Mode 2: Full run with checkpoint BEFORE plotting
#     model = SpikingModel(cfg)
#     model.load()
#     model.prepare_inputs(animal_by_animal=animal_by_animal)

#     # Restrict to one animal if requested
#     if animal_by_animal and only_animal_eff is not None:
#         pwa = model.results["padded_warped_activity_by_animal"]
#         vel = model.results["velocity_by_animal"]
#         if only_animal_eff not in pwa:
#             raise KeyError(f"Animal '{only_animal_eff}' not found. Available: {list(pwa.keys())}")
#         model.results["padded_warped_activity_by_animal"] = {only_animal_eff: pwa[only_animal_eff]}
#         model.results["velocity_by_animal"] = {only_animal_eff: vel[only_animal_eff]}

#     # Simulate (optionally one seed)
#     model.simulate_seeds(animal_by_animal=animal_by_animal, seed_override=seed_override)

#     # SAVE CHECKPOINT BEFORE PLOTTING
#     model.save(save_path)
#     print(f"[checkpoint] Saved pre-plot model to: {save_path.resolve()}")

#     # Plot (full if possible; otherwise quick)
#     if do_plot:
#         try:
#             if not animal_by_animal:
#                 # Pooled mode has full panels
#                 model.compute_plateaus()
#                 model.plot()
#             else:
#                 # Per-animal mode uses minimal plotting
#                 fn = getattr(model, "plot_animal_by_animal", None)
#                 fn() if callable(fn) else model.plot_quick()
#         except Exception as e:
#             print(f"[plot] Plot failed: {e}. Results are safe at {save_path.resolve()}")
#             raise

#     # Save after plotting (same path)
#     model.save(save_path)
#     print(f"Saved model to: {save_path.resolve()}")


# if __name__ == "__main__":
#     main()


        
            
    

# # vel_applied = "real"   #real or constant 
    
# # wt_dist = "Lognormal"   #Uniform, Constant, Lognormal 


# # add_inh = 'neither' #options: both, sst, neither

# # dend_threshold = 1.0


# # SST_bias_multi = 1.4

# # # SST_bias_factor_list = [1.4, 1.6, 1.8, 2.0]

# # # for SST_bias_multi in SST_bias_factor_list:


# # an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, NDNF_contribution_sum, SST_contribution_sum, weights_EC, weights_SST, weights_NDNF = multi_wrap_contribution(residual_activity_dict_EC, fixed_residual_activity_dict_NDNF_newest, residual_activity_dict_SST, GLM_params_EC, GLM_params_NDNF_newest, GLM_params_SST, mean_new_average_vel_array, vel_applied=vel_applied, add_inh=add_inh, SST_bias_factor=SST_bias_multi, dist=wt_dist)
    
# # activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list = get_activity_multidendrite(dend_contribution_EC, an_velocity, EC_pop_list, NDNF_pop_list, SST_pop_list, NDNF_sf_opt, SST_sf_opt, dend_threshold=dend_threshold, vel_applied=vel_applied, example_cell=17, dist=wt_dist, n_dendrites=100, n_SST=75, n_EC=792, n_NDNF=73, include_inhibition=add_inh, use_model_EC=False)

# # plot_multidendrite_EC(weights_EC, weights_SST, weights_NDNF, activity_EC, activity_SST, activity_NDNF, SST_sf_opt, NDNF_sf_opt, padded_warped_activity_list, an_velocity, dend_activity, dend_threshold, plateau_positions_counter, plateau_start_positions_counter, plateau_array_per_dendrite_list, dendrite_plateau_mask, time_each_pos_bin_starts, plateau_start_times_list_mega_list, EC_used, dist, num_plateaus_per_dend_list, include_inhibition=add_inh, NDNF_contribution_sum=NDNF_contribution_sum, SST_contribution_sum=SST_contribution_sum)
    

        









#             # dend_vm_per_animal_dict = {}

#             # animal_list =['animal_1']

#             # for animal in animal_list:

#             # # seeds = [seed_override] if seed_override is not None else list(range(self.cfg.num_seeds))

#             #     dend_vm_per_seed_dict = {}

#             #     # for i in seeds:
#             #     for i in range(2):

#             #         epsp_cells, kernel = get_epsp_dict_animal(padded_warped_by_animal[animal], tau_ms=tau, amp=1., seed=i)
#             #         dend_Vm, epsp_input_matrix, spike_mats = get_dend_vm_from_cells_multi(epsp_cells, Vrest=self.cfg.vrest, epsp_sf=self.cfg.epsp_sf)

#             #         print(f"epsp_input_matrix.shape {epsp_input_matrix.shape}")

#             #         n_EC = cells_per_animal_dict[animal]
#             #         connection_mask_EC = np.ones((n_dendrites, n_EC), dtype=bool)
#             #         weights_EC = sample_weights(dist, connection_mask_EC, rng=rng, mean=weights_mean, std=weights_std)


#             #         activity_EC = get_dendrite_activity_multi(weights_EC, epsp_input_matrix, n_dendrites, n_EC)

#             #         dend_Vm, activity_centered, trial_means = activity_to_dend_vm(activity_EC, Vrest=-70.0, vm_scale=0.1, center_across="time_trials")

#             #         dend_vm_per_seed_dict[i] = dend_Vm
#             #         # dend_Vm = np.transpose(dend_Vm, (0, 2, 1)) 
#             #     dend_vm_per_animal_dict[animal] = dend_vm_per_seed_dict

#             # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/dend_vm_per_animal_dict.pkl"

#             # important_dict = {"dend_vm_per_animal_dict":dend_vm_per_animal_dict, 
#             #     "weights_EC":weights_EC,
#             #     "epsp_input_matrix":epsp_input_matrix}

#             # with open(save_path, 'wb') as f:
#             #     pickle.dump(important_dict, f)
#             #     print(f"pickle saved to {save_path}")


            
#             # save_path = "/Users/michaelfinch/CA1-interneuron-GLM/Clean_notebooks_to_date/dend_vm_per_animal_dict.pkl"
#             # with open(save_path, 'rb') as f:
#             #     important_dict = pickle.load(f)
#             #     print(f"pickle loaded from {save_path}")

#             # weights_EC = important_dict["weights_EC"]
#             # dend_vm_per_animal_dict = important_dict["dend_vm_per_animal_dict"]
#             # epsp_input_matrix = important_dict["epsp_input_matrix"]
