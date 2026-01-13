import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
import slicetca
from sklearn.cluster import KMeans
from scipy.stats import sem
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import pickle 
from sklearn.decomposition import PCA
import ruptures as rpt
from scipy.spatial.distance import cdist
import click
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import ttest_rel


plt.rcParams['axes.titlesize'] = 8       # all titles
plt.rcParams['axes.labelsize'] = 7       # x and y labels
plt.rcParams['xtick.labelsize'] = 6      # tick labels
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams["legend.fontsize"] = 7
plt.rcParams['savefig.dpi'] = 600




def has_run_of_n_nans(trial_data, n=4):
    
    nan_mask = np.isnan(trial_data)
    if not np.any(nan_mask):
        return False
    
    conv = np.convolve(nan_mask.astype(int), np.ones(n, dtype=int), mode='valid')
    return np.any(conv >= n)


def interp_nans_1d(trial_data):
    x = np.arange(trial_data.size)
    nan_mask = np.isnan(trial_data)

    if not np.any(nan_mask):
        return trial_data

    if not np.any(~nan_mask):  # all NaNs
        return trial_data

    trial_data[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], trial_data[~nan_mask])
    return trial_data



def get_cp_dict(cell_SST_model_ranks20_contig_x00):
    changepoints_dict = {}
    for animal in cell_SST_model_ranks20_contig_x00[20]:
        changepoints_cell_dict = {}
        for cell in cell_SST_model_ranks20_contig_x00[20][animal]:
            labels = cell_SST_model_ranks20_contig_x00[20][animal][cell][1][f'cell_{cell}']["labels_dict"]["clusters_chosen_3"]
            changepoints = np.where(np.diff(labels) != 0)[0]
            changepoints_cell_dict[cell] = changepoints
        changepoints_dict[animal] = changepoints_cell_dict

    return changepoints_dict



def flatten_data(neuron_dict):
    flattened_data = {}
    for var in neuron_dict:
        flattened_data[var] = neuron_dict[var].flatten()
    return flattened_data

def fit_GLM(animal_factors_dict, neuron_activity, regression='linear', alphas=None):
    neuron_activity_flat = neuron_activity.flatten()
    flattened_data = flatten_data(animal_factors_dict)
    variable_names = [var for var in flattened_data]
    design_matrix_X = np.stack([flattened_data[var] for var in variable_names], axis=1)

    if regression == 'linear':
        model = LinearRegression()
    elif regression == 'lasso':
        model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
    elif regression == 'ridge':
        model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
    elif regression == 'elastic':
        l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
        model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000],
                             l1_ratio=l1_ratio, cv=None)

    model.fit(design_matrix_X, neuron_activity_flat)
    neuron_predicted_activity = model.predict(design_matrix_X)

    trialavg_neuron_activity = np.mean(neuron_activity, axis=1)
    trialavg_predicted_activity = np.mean(neuron_predicted_activity.reshape(neuron_activity.shape), axis=1)
    pearson_R = np.corrcoef(trialavg_predicted_activity, trialavg_neuron_activity)[0, 1]
    neuron_GLM_params = {}
    neuron_GLM_params['weights'] = {var: model.coef_[idx] for idx, var in enumerate(variable_names)}
    neuron_GLM_params['intercept'] = model.intercept_
    neuron_GLM_params['alpha'] = model.alpha_ if regression == 'ridge' else None
    neuron_GLM_params['l1_ratio'] = model.l1_ratio_ if regression == 'elastic' else None
    neuron_GLM_params['R2'] = model.score(design_matrix_X, neuron_activity_flat)
    neuron_GLM_params['pearson_R'] = pearson_R
    neuron_GLM_params['model'] = model
    return neuron_GLM_params, neuron_predicted_activity

def get_residual_activity_dict(activity_dict, predicted_activity_dict):
    residual_activity_dict = {}
    for animal in activity_dict:
        residual_activity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            residual_activity_dict[animal][neuron] = activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
    return residual_activity_dict

def fit_GLM_population(factors_dict, activity_dict, quintile=None, regression='ridge', alphas=None):
    GLM_params = {}
    predicted_activity_dict = {}

    for animal in factors_dict:
        GLM_params[animal] = {}
        predicted_activity_dict[animal] = {}
        animal_factors_dict = factors_dict[animal].copy()

        if quintile is not None:
            num_trials = animal_factors_dict['Activity'].shape[1]
            start_idx, end_idx = get_quintile_indices(num_trials, quintile)
            for var in animal_factors_dict:
                animal_factors_dict[var] = animal_factors_dict[var][:, start_idx:end_idx]

        for neuron_idx in activity_dict[animal]:
            neuron_activity = activity_dict[animal][neuron_idx]
            neuron_GLM_params, neuron_predicted_activity = fit_GLM(animal_factors_dict, neuron_activity, regression, alphas)
            GLM_params[animal][neuron_idx] = neuron_GLM_params
            predicted_activity_dict[animal][neuron_idx] = neuron_predicted_activity.reshape(activity_dict[animal][neuron_idx].shape)
                
    return GLM_params, predicted_activity_dict

def subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"]):
    filtered_factors_dict = {}
    for animal in factors_dict:
        filtered_factors_dict[animal] = {}
        for variable in variables_to_keep:
            filtered_factors_dict[animal][variable] = factors_dict[animal][variable]
    return filtered_factors_dict



def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict, num_clusters=8, reassign_clusters=False, x00=True, umap=True, contiguous=True, ranks=20):

    internals_per_animal_dict_EC_animal_x00_regkmean = {}
    
    for idx, animal in enumerate(residual_activity_dict):
        internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}
        for idt, cell in enumerate(residual_activity_dict[animal]):

            cell_data = residual_activity_dict[animal][cell].T
            cell_data = ((cell_data-np.min(cell_data)) / np.max(cell_data) - np.min(cell_data))
            cell_data_3d = np.expand_dims(cell_data, axis=1)
            cell_data_3d = torch.from_numpy(cell_data_3d)
            cell_model = NDNF_fixed_model_dict[animal][cell]


            internals_dict = get_animal_model_reconstruction_dict_mod(cell_model, cell_data_3d, max_clusters=num_clusters, display=False, reassign_small_clusters=reassign_clusters, x00=x00, use_umap=umap, use_breakpoints=contiguous)

            internals_per_animal_dict_EC_animal_x00_regkmean_cell[cell] = internals_dict
        
        internals_per_animal_dict_EC_animal_x00_regkmean[animal] = internals_per_animal_dict_EC_animal_x00_regkmean_cell

    return internals_per_animal_dict_EC_animal_x00_regkmean


def get_animal_model_reconstruction_dict_mod(animal_model, tensor_for_animal, max_clusters=12, display=False, reassign_small_clusters=True, x00=True, use_umap=False, use_breakpoints=False):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    per_cell_internals_dict = {}
    reconstruction_full_animal = animal_model.construct().numpy(force=True)

    if x00:
        w1 = animal_model.vectors[0][0].detach().numpy()
        X = np.abs(w1.T)
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(X)
    else:
        f = animal_model.vectors[2][1].detach()
        f1 = f.permute(1, 0, 2).reshape(f.shape[1], -1)
        f1 = torch.abs(f1).cpu().numpy()
        if use_umap:
            import umap
            umap_model = umap.UMAP(n_components=3, random_state=0)
            X_umap = umap_model.fit_transform(f1)

    cluster_labels_dict = {}
    cluster_pca_dict = {}
    cluster_centroids_dict = {}

    for clusters_chosen in range(1, max_clusters):
        if use_breakpoints:
            print(f"Using breakpoint clustering (clusters = {clusters_chosen})")
            model_input = X_umap if use_umap else (X if x00 else f1)
            n_bkps = clusters_chosen - 1
            algo = rpt.Binseg(model="l2", min_size=3).fit(model_input)
            try:
                bkps = algo.predict(n_bkps=n_bkps)
            except rpt.exceptions.BadSegmentationParameters:
                print("  Skipping due to bad breakpoint config")
                continue

            labels = np.zeros(model_input.shape[0], dtype=int)
            start = 0
            for cluster_id, end in enumerate(bkps):
                labels[start:end] = cluster_id
                start = end

            centroids = np.array([
                model_input[labels == i].mean(axis=0)
                for i in range(clusters_chosen)
            ])
            X_pca = PCA(n_components=3).fit_transform(model_input)

        else:
            kmeans = KMeans(n_clusters=clusters_chosen, random_state=0)
            model_input = X_umap if use_umap else (X if x00 else f1)
            labels = kmeans.fit_predict(model_input)

            centroids = kmeans.cluster_centers_
            if use_umap:
                X_pca = PCA(n_components=3).fit_transform(X_umap)
            else:
                X_pca = PCA(n_components=3).fit_transform(model_input)

        cluster_labels_dict[clusters_chosen] = labels
        cluster_centroids_dict[clusters_chosen] = centroids
        cluster_pca_dict[clusters_chosen] = X_pca

    for cell in range(reconstruction_full_animal.shape[1]):
        #     cell = 0
        print(f"Processing cell {cell}...")

        MSE_dict = {}
        x_pca_dict = {}
        labels_dict = {}
        indices_for_cluster_number = {}
        TCA_reconstructions_dict = {}
        Recon_by_cluster_av_dict = {}
        cluster_trial_mean_dict = {}

        reconstructed_cell = reconstruction_full_animal[:, cell, :]
        real_cell_activity = tensor_for_animal[:, cell, :].detach().numpy()

        for clusters_chosen in range(1, max_clusters):
            labels = cluster_labels_dict[clusters_chosen].copy()
            centroids = cluster_centroids_dict[clusters_chosen]
            X_pca = cluster_pca_dict[clusters_chosen]
            model_input = X_umap if use_umap else (X if x00 else f1)

            print(f"\nclusters_chosen = {clusters_chosen}")
            print("Before reassignment:")
            for cluster_id in range(clusters_chosen):
                count = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} trials")

            if reassign_small_clusters:
                if use_umap:
                    model_input = X_umap
                    centroid_space = np.array([
                        model_input[labels == i].mean(axis=0)
                        for i in range(clusters_chosen)])
                else:
                    model_input = X if x00 else f1
                    centroid_space = centroids
                for cluster_id in range(clusters_chosen):
                    trial_indices = np.where(labels == cluster_id)[0]
                    if len(trial_indices) < 2:
                        print(f"  Reassigning cluster {cluster_id} (size={len(trial_indices)})...")
                        for idx in trial_indices:
                            trial = model_input[idx]
                            dists = cdist([trial], centroid_space)[0]
                            dists[cluster_id] = np.inf
                            new_cluster = np.argmin(dists)
                            labels[idx] = new_cluster

            print("After reassignment:")
            for cluster_id in range(clusters_chosen):
                count = np.sum(labels == cluster_id)
                print(f"  Cluster {cluster_id}: {count} trials")

            x_pca_dict[f"clusters_chosen_{clusters_chosen}"] = X_pca
            labels_dict[f"clusters_chosen_{clusters_chosen}"] = labels

            valid_cluster_mean_trials_list = []
            valid_cluster_indices = []
            cluster_trial_indices = {}

            for n in range(clusters_chosen):
                trial_indices = np.where(labels == n)[0]
                cluster_trial_indices[n] = trial_indices
                if len(trial_indices) == 0:
                    continue
                cluster_trials = real_cell_activity[trial_indices, :]
                mean_cluster = cluster_trials.mean(axis=0)
                valid_cluster_mean_trials_list.append(mean_cluster)
                valid_cluster_indices.append((n, trial_indices))

            empty_cell = np.zeros_like(reconstructed_cell)
            for i, (n, trials) in enumerate(valid_cluster_indices):
                empty_cell[trials, :] = valid_cluster_mean_trials_list[i]

            key = f"clusters_chosen_{clusters_chosen}"
            MSE_dict[key] = np.mean((real_cell_activity - empty_cell) ** 2)
            Recon_by_cluster_av_dict[key] = empty_cell
            TCA_reconstructions_dict[key] = reconstructed_cell
            cluster_trial_mean_dict[key] = valid_cluster_mean_trials_list
            indices_for_cluster_number[key] = cluster_trial_indices

        per_cell_internals_dict[f"cell_{cell}"] = {
            "MSE_dict": MSE_dict,
            "x_pca_dict": x_pca_dict,
            "labels_dict": labels_dict,
            "indices_for_cluster_number": indices_for_cluster_number,
            "TCA_reconstructions_dict": TCA_reconstructions_dict,
            "Recon_by_cluster_av_dict": Recon_by_cluster_av_dict,
            "cluster_trial_mean_dict": cluster_trial_mean_dict,
        }

    return per_cell_internals_dict



def get_max_proportion(early_labels, use_max=True):
    unique_early_labels = np.unique(early_labels)
    if use_max:
        og_proportion=0
    else:
        og_proportion=1000
    good_dict=None
    for unique_label in unique_early_labels:
        amount = len(np.where(early_labels==unique_label)[0])
        len_early_labels = len(early_labels)
        proportion_early = amount / len_early_labels

        if use_max:
            if proportion_early > og_proportion:
                good_dict= {"unique_label":unique_label,
                        "fraction":proportion_early}
                og_proportion=proportion_early
        else:
            if proportion_early < og_proportion:
                good_dict= {"unique_label":unique_label,
                        "fraction":proportion_early}
                og_proportion=proportion_early
    return good_dict



def get_animal_clean_dict_activity(filepath, use_final=True):
    with h5py.File(filepath, "r") as f:
        if use_final:
            animal_group = f["animals"]
        else:
            animal_group = f["animal"]

        shiftR_refs = animal_group["ShiftR"][:]
        shiftRunning_refs = animal_group["ShiftRunning"][:]

        if use_final:
            shiftL_refs = animal_group["ShiftL"][:]
        else:
            shiftL_refs = animal_group["ShiftLrate"][:]

        animal_clean_dict_activity = {}
        animal_trials_original = []
        animal_trials_clean = []

        animal_vel_dict = {}
        animal_lick_dict = {}

        trials_to_remove_local = []  # debug tracking for count == 105
        count = 0  # global cell counter (across animals)

        for animal_idx in range(len(shiftR_refs)):
            # ΔF: (cells, trials, time)
            delta_f = np.array(f[shiftR_refs[animal_idx][0]])
            animal_trials_original.append(delta_f.shape[1])

            # velocity & lick: raw (trials, time?) → transpose: (time, trials)
            vel = np.array(f[shiftRunning_refs[animal_idx][0]]).T
            lick = np.array(f[shiftL_refs[animal_idx][0]]).T

            # --- align trial counts across df / vel / lick ---
            n_df_trials = delta_f.shape[1]
            n_vel_trials = vel.shape[1]
            n_lick_trials = lick.shape[1]

            n_trials = min(n_df_trials, n_vel_trials, n_lick_trials)

            if (n_df_trials, n_vel_trials, n_lick_trials) != (n_trials,) * 3:
                delta_f = delta_f[:, :n_trials, :]
                vel = vel[:, :n_trials]
                lick = lick[:, :n_trials]

            # preallocate clean arrays
            vel_clean = np.empty_like(vel)
            lick_clean = np.empty_like(lick)
            delta_f_clean = np.empty_like(delta_f)

            # list of trials to drop for this animal (union across cells)
            trials_to_remove_list = []

            # special case: skip cell 0 for this animal
            if animal_idx == 22:
                valid_cells = range(1, delta_f.shape[0])
            else:
                valid_cells = range(delta_f.shape[0])

            # --- clean per cell / per trial ---
            for cell in valid_cells:
                # cell_data: (time, trials)
                cell_data = delta_f[cell, :, :].T

                for trial in range(cell_data.shape[1]):
                    trial_data = cell_data[:, trial]
                    vel_data_trial = vel[:, trial]
                    lick_data_trial = lick[:, trial]

                    nan_trial = np.any(np.isnan(trial_data))
                    nan_vel = np.any(np.isnan(vel_data_trial))

                    if nan_trial or nan_vel:
                        # decide: drop or interpolate based on runs of >=5 NaNs
                        too_many_nans = (
                            has_run_of_n_nans(trial_data, n=5)
                            or has_run_of_n_nans(vel_data_trial, n=5)
                        )

                        if too_many_nans:
                            # mark for removal (for all cells later)
                            if count == 105:
                                trials_to_remove_local.append(trial)
                            if trial not in trials_to_remove_list:
                                trials_to_remove_list.append(trial)
                        else:
                            # interpolate and keep this trial
                            clean_trial = interp_nans_1d(trial_data.copy())
                            delta_f_clean[cell, trial, :] = clean_trial

                            clean_vel = interp_nans_1d(vel_data_trial.copy())
                            vel_clean[:, trial] = clean_vel

                            clean_lick = interp_nans_1d(lick_data_trial.copy())
                            lick_clean[:, trial] = clean_lick
                    else:
                        # no NaNs anywhere: just copy
                        delta_f_clean[cell, trial, :] = trial_data
                        vel_clean[:, trial] = vel_data_trial
                        lick_clean[:, trial] = lick_data_trial

                count += 1  # increment per cell

            # --- drop bad trials across all cells for this animal ---
            trials_to_remove_array = np.array(trials_to_remove_list, dtype=int)

            if trials_to_remove_array.size > 0:
                mask = np.ones(delta_f_clean.shape[1], dtype=bool)
                mask[trials_to_remove_array] = False

                delta_f_clean = delta_f_clean[:, mask, :]
                vel_clean = vel_clean[:, mask]
                lick_clean = lick_clean[:, mask]

            animal_trials_clean.append(delta_f_clean.shape[1])

            # --- build per-cell dict with z-scoring ---
            cell_dict = {}
            for cell in valid_cells:
                cell_data = delta_f_clean[cell, :, :]  # (trials, time)

                mean = np.mean(cell_data)
                std = np.std(cell_data)

                if std == 0 or not np.isfinite(std):
                    # print(" -> zero or bad std for this cell, skipping")
                    continue

                cell_z = (cell_data - mean) / std  # (trials, time)
                cell_dict[f"cell_{cell+1}"] = cell_z.T  # (time, trials)

            animal_clean_dict_activity[f"animal_{animal_idx+1}"] = cell_dict
            animal_vel_dict[f"animal_{animal_idx+1}"] = {"Velocity": vel_clean}
            animal_lick_dict[f"animal_{animal_idx+1}"] = {"Licks": lick_clean}

        return (
            animal_clean_dict_activity,
            animal_vel_dict,
            animal_trials_original,
            animal_trials_clean,
            trials_to_remove_local,
            animal_lick_dict,
        )



def reshape_contig_dict(cued_contig_dict, NDNF_cued_model_dict_clean):
    # Match the old structure: outer key is rank (20)
    cued_contig_final = {20: {}}

    for idx, animal in enumerate(NDNF_cued_model_dict_clean):
    # for animal in cued_contig_dict:  # animal index: 0,1,2,...
        cued_contig_final[20][animal] = {}

        print(f"animal {animal} NDNF_cued_model_dict_clean.keys() {NDNF_cued_model_dict_clean.keys()}")

        for cell in cued_contig_dict[animal]:  # cell index: 0,1,2,...
            # 1) SliceTCA model object
            model_obj = NDNF_cued_model_dict_clean[animal][cell]

            # 2) Internals for this cell – currently under "cell_0"
            internals_cell0 = cued_contig_dict[animal][cell]["cell_0"]

            # 3) Rename "cell_0" → f"cell_{cell}" to match old API
            per_cell_internals = {f"cell_{cell}": internals_cell0}

            # 4) Store as [model_obj, per_cell_internals]
            cued_contig_final[20][animal][cell] = [model_obj, per_cell_internals]

    return cued_contig_final


def rebin_means_sems(means, sems, bins_per_group=10):
    """
    Take per-pos means & sems (length 50) and collapse into coarser bins.
    Returns x positions (bin centers), rebinned means, rebinned sems.
    """
    means = np.asarray(means, float)
    sems  = np.asarray(sems, float)

    n_old = len(means)
    assert n_old % bins_per_group == 0, "50 must be divisible by bins_per_group"
    n_groups = n_old // bins_per_group

    x_coarse = []
    m_coarse = []
    s_coarse = []

    for g in range(n_groups):
        sl = slice(g * bins_per_group, (g + 1) * bins_per_group)
        m_block = np.nanmean(means[sl])
        s_block = np.nanmean(sems[sl])   # approx; for viz only

        x_center = (g * bins_per_group + (g + 1) * bins_per_group - 1) / 2.0

        x_coarse.append(x_center)
        m_coarse.append(m_block)
        s_coarse.append(s_block)

    return np.array(x_coarse), np.array(m_coarse), np.array(s_coarse)


def plot_butterfly_hist(
    argmax_list_early_0, argmax_list_late_0,
    argmin_list_early_0, argmin_list_late_0,
    ax=None, colors_list=None, title=None):

    bins = np.arange(0, 51)  # 50 position bins

    # ---- ARGMAX histograms (top) ----
    ax.hist(np.array(argmax_list_early_0),
            bins=bins, alpha=0.4,
            color=colors_list[0])

    ax.hist(np.array(argmax_list_late_0),
            bins=bins, alpha=0.4,
            color=colors_list[1])

    # ---- ARGMIN histograms (bottom, mirrored) ----
    weights_early_min = -np.ones_like(argmin_list_early_0, dtype=float)
    weights_late_min  = -np.ones_like(argmin_list_late_0, dtype=float)

    ax.hist(np.array(argmin_list_early_0),
            bins=bins, alpha=0.4, weights=weights_early_min,
            color=colors_list[2])

    ax.hist(np.array(argmin_list_late_0),
            bins=bins, alpha=0.4, weights=weights_late_min,
            color=colors_list[3])

    # Zero line
    ax.axhline(0, color="k", linewidth=1)

    # Axis labels & title
    ax.set_xlabel("Position bin")
    ax.set_ylabel("Cluster Count")
    ax.set_title(f"{title} Cluster Count per Pos Bin")

    # ---- Symmetric y-limits and custom ticks (25 → 0 → 25 style) ----
    y_min, y_max = ax.get_ylim()
    max_val = max(abs(y_min), abs(y_max))
    ax.set_ylim(-max_val, max_val)

    # Hard-code symmetric ticks (e.g., -25..0..25, labeled as 25..0..25)
    n_ticks = 5  # per side
    pos_ticks = np.linspace(0, max_val, n_ticks+1)     # 0..26
    neg_ticks = -pos_ticks[1:][::-1]              # -26..-small
    ticks = np.concatenate([neg_ticks, [0.0], pos_ticks[1:]])
    tick_labels = [f"{int(abs(t))}" for t in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # ---- Separate legends: upper right for Max, lower right for Min ----
    # Proxy handles for legend (so we don't depend on hist's internal patches)
    handles_max = [
        Line2D([0], [0], color=colors_list[0], lw=3, label="Max Early"),
        Line2D([0], [0], color=colors_list[1], lw=3, label="Max Late"),
    ]
    legend_max = ax.legend(handles=handles_max,
                        loc="upper right",
                        title="Max Loc",)
    ax.add_artist(legend_max)  # keep this one when adding second legend

    handles_min = [
        Line2D([0], [0], color=colors_list[2], lw=3, linestyle="--", label="Min Early"),
        Line2D([0], [0], color=colors_list[3], lw=3, linestyle="--", label="Min Late"),
    ]
    legend_min = ax.legend(handles=handles_min,
                        loc="lower right",
                        title="Min Loc",)

    return ax




def get_lists_out_of_dicts(fixed_TT_data, fixed_activity_dict_NDNF_newest, cp_dict_NDNF):
    TT_list = []
    for animal in fixed_TT_data:
        for cell in fixed_TT_data[animal]:

            TT_list.append(fixed_TT_data[animal][cell][1][f'cell_{cell}'])


    NDNF_activity_list = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            NDNF_activity_list.append(fixed_activity_dict_NDNF_newest[animal][cell])

    cp_list_NDNF = []

    for animal in cp_dict_NDNF:
        for cell in cp_dict_NDNF[animal]:
            cp_list_NDNF.append(cp_dict_NDNF[animal][cell])

    return TT_list, NDNF_activity_list, cp_list_NDNF

def find_elbow_point(y_vals, min_index=2):

    x = np.arange(len(y_vals))
    y = y_vals

    # First and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Compute distances to the line
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_p1 = np.vstack((x - p1[0], y - p1[1])).T
    scalar_proj = np.dot(vec_from_p1, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_from_p1 - proj, axis=1)

    # Force elbow to be at least min_index (default = 2)
    elbow_idx = np.argmax(dist_to_line[min_index:]) + min_index
    return int(elbow_idx)


def get_most_expressed_cluster(TT_list, activity_list, cp_list_NDNF, early_late_none="early", to_include=None, most_expressed=True):

    
    if np.any(to_include) == None:
        indices_to_include = np.arange(len(TT_list))
    else:
        indices_to_include = to_include

    most_expressed_label_dict = {}

    elbow_kmeans_array = np.empty(len(indices_to_include))

    for j, cell in enumerate(indices_to_include):
              
        activity_array = activity_list[cell]

        labels_example = TT_list[cell]["labels_dict"]["clusters_chosen_3"]

        MSE_dict = TT_list[cell]["MSE_dict"]

        MSE_array = np.empty(len(MSE_dict))

        for id, clusters_chosen in enumerate(MSE_dict):
            MSE = MSE_dict[clusters_chosen]
            MSE_array[id] = MSE
        
        elbow_kmeans = find_elbow_point(MSE_array)

        elbow_kmeans_array[j] = elbow_kmeans #+ 1
        
        if early_late_none=='early':
            cp_early = cp_list_NDNF[cell][0]
            labels = labels_example[:cp_early]

        elif early_late_none=='late':
            cp_late = cp_list_NDNF[cell][1]
            labels = labels_example[cp_late:]

        elif early_late_none=='none':
            labels = labels_example
        else:
            raise ValueError("Invalid learning chunk")


        good_dict = get_max_proportion(labels, use_max=most_expressed)

        correct_indices = np.where(labels==good_dict["unique_label"])[0]

        activity_array_sliced = activity_array[:,correct_indices]


        most_expressed_label_dict[cell] = {"label":good_dict["unique_label"],
                                                        "cluster_activity": activity_array_sliced, "fraction":good_dict["fraction"]}

    return most_expressed_label_dict, elbow_kmeans_array



def produce_ta_activity_dict_for_cluster(important_dict):
    sums_list = []

    number_trial_types = []

    ta_activity_dict_for_cluster = {}

    elbow_kmeans_per_cell_list = []

    max_locs_list = []
    min_locs_list = []
    
    max_amplitudes_list = []
    min_amplitudes_list = []

    early_expression_list = []
    middle_expression_list = []
    late_expression_list = []


    for cell in range(len(important_dict["TT_list"])):

        MSE_dict = important_dict["TT_list"][cell]["MSE_dict"]

        MSE_array = np.empty(len(MSE_dict))

        for id, clusters_chosen in enumerate(MSE_dict):
            MSE = MSE_dict[clusters_chosen]
            MSE_array[id] = MSE
        
        elbow_kmeans = find_elbow_point(MSE_array)
        elbow_cluster=f"clusters_chosen_{elbow_kmeans+1}"
        elbow_kmeans_per_cell_list.append(elbow_cluster)

        labels = important_dict["TT_list"][cell]["labels_dict"][elbow_cluster]

        early_cp=important_dict["cp_list_NDNF"][cell][0]
        late_cp=important_dict["cp_list_NDNF"][cell][1]

        label_values = np.unique(labels)

        number_trial_types.append(len(label_values))

        activity_pattern_cell = important_dict["NDNF_activity_list"][cell]

        ta_activity_dict_for_cluster_cell = {}

        for i in label_values:

            trial_type_0_loc = np.where(labels==i)[0]

            overall_expression_prob = len(trial_type_0_loc) / activity_pattern_cell.shape[1]

            den = len(trial_type_0_loc)

            if den == 0:
                print("no zero cells")
                print(labels)


            overall_activity = np.mean(activity_pattern_cell[:,trial_type_0_loc], axis=1)
            baseline_start_overall = overall_activity[0]
            max_loc_overall = np.argmax(overall_activity)
            min_loc_overall = np.argmin(overall_activity)
            max_locs_list.append(max_loc_overall)
            min_locs_list.append(min_loc_overall)

            max_amp_overall = overall_activity[max_loc_overall] 
            min_amp_overall = overall_activity[min_loc_overall] 
            max_amplitudes_list.append(max_amp_overall)
            min_amplitudes_list.append(min_amp_overall)
            
            tt0_early = trial_type_0_loc<early_cp
            tt0_indices_early = trial_type_0_loc[tt0_early]
            total_trials_early = len(tt0_early)
            successes_early = np.sum(tt0_early) ## this is a count of early block successes

            activity_pattern_early = activity_pattern_cell[:,tt0_indices_early]

            tt0_middle = (trial_type_0_loc >= early_cp) & (trial_type_0_loc <  late_cp)
            total_trials_middle = len(tt0_middle)
            successes_middle = np.sum(tt0_middle)
            tt0_indices_middle = trial_type_0_loc[tt0_middle]
            activity_pattern_middle = activity_pattern_cell[:,tt0_indices_middle]

            tt0_late   = trial_type_0_loc >= late_cp
            total_trials_late = len(tt0_late)
            successes_late = np.sum(tt0_late)
            tt0_indices_late = trial_type_0_loc[tt0_late]
            activity_pattern_late = activity_pattern_cell[:,tt0_indices_late]

            ta_activity_early = np.mean(activity_pattern_early, axis=1)
            ta_activity_middle = np.mean(activity_pattern_middle, axis=1)
            ta_activity_late = np.mean(activity_pattern_late, axis=1)

            middle_num_trials = late_cp-early_cp
            late_num_trials = activity_pattern_cell.shape[1] - late_cp

            sum = early_cp+middle_num_trials+late_num_trials


            if activity_pattern_cell.shape[1] != sum:
                raise ValueError("wrong")

            expression_prob_early = successes_early / early_cp
            expression_prob_middle = successes_middle / middle_num_trials
            expression_prob_late = successes_late / late_num_trials

            early_expression_list.append(expression_prob_early)
            middle_expression_list.append(expression_prob_middle)
            late_expression_list.append(expression_prob_late)

            # expression_prob_early = successes_early / total_trials_early
            # expression_prob_middle = successes_middle / total_trials_middle
            # expression_prob_late = successes_late / total_trials_late



            early_activity = np.mean(activity_pattern_cell[:,tt0_indices_early], axis=1)
            baseline_start_early = early_activity[0]
            max_loc_early = np.argmax(early_activity)
            min_loc_early = np.argmin(early_activity)

            max_amp_early = early_activity[max_loc_early]
            min_amp_early = early_activity[min_loc_early]
            
            middle_activity = np.mean(activity_pattern_cell[:,tt0_indices_middle], axis=1)
            baseline_start_middle = middle_activity[0]
            max_loc_middle = np.argmax(middle_activity)
            min_loc_middle = np.argmin(middle_activity)

            max_amp_middle = middle_activity[max_loc_middle]
            min_amp_middle = middle_activity[min_loc_middle]
            

            late_activity = np.mean(activity_pattern_cell[:,tt0_indices_late], axis=1)
            baseline_start_late = late_activity[0]
            max_loc_late = np.argmax(late_activity)
            min_loc_late = np.argmin(late_activity)

            max_amp_late = late_activity[max_loc_late]
            min_amp_late = late_activity[min_loc_late]
            


            act_dict = {"cluster_expression": overall_expression_prob,
                        "early":ta_activity_early,
                        "middle":ta_activity_middle,
                        "late":ta_activity_late,
                        "early_expression":expression_prob_early,
                        "middle_expression":expression_prob_middle,
                        "late_expression": expression_prob_late,
                        "n_early": int(successes_early),
                        "n_middle": int(successes_middle),
                        "n_late": int(successes_late),
                        "overall_activity":overall_activity,
                        "max_loc_overall":max_loc_overall,
                        "min_loc_overall":min_loc_overall,
                        "max_amp_overall":max_amp_overall,
                        "min_amp_overall":min_amp_overall,
                        "baseline_start_overall":baseline_start_overall,
                        # "baseline_end_overall":min_amp_overall,

                        "max_loc_early":max_loc_early,
                        "min_loc_early":min_loc_early,
                        "max_amp_early":max_amp_early,
                        "min_amp_early":min_amp_early,
                        "baseline_start_early":baseline_start_early,

                        "max_loc_middle":max_loc_middle,
                        "min_loc_middle":min_loc_middle,
                        "max_amp_middle":max_amp_middle,
                        "min_amp_middle":min_amp_middle,
                        "baseline_start_middle":baseline_start_middle,

                        "max_loc_late":max_loc_late,
                        "min_loc_late":min_loc_late,
                        "max_amp_late":max_amp_late,
                        "min_amp_late":min_amp_late,
                        "baseline_start_late":baseline_start_late,}

            ta_activity_dict_for_cluster_cell[i] = act_dict

            if np.isnan(expression_prob_early):
                expression_prob_early = 0.

            if np.isnan(expression_prob_middle):
                expression_prob_middle = 0.

            if np.isnan(expression_prob_late):
                expression_prob_late = 0.

            all_list = [expression_prob_early, expression_prob_middle, expression_prob_late]
            sums = np.sum(all_list)

            sums_list.append(sums)


        ta_activity_dict_for_cluster[cell] = ta_activity_dict_for_cluster_cell

    max_locs_array = np.array(max_locs_list)
    min_locs_array = np.array(min_locs_list)
    max_amplitudes_array = np.array(max_amplitudes_list)
    min_amplitudes_array = np.array(min_amplitudes_list)
    early_expression_array = np.array(early_expression_list)
    middle_expression_array = np.array(middle_expression_list)
    late_expression_array = np.array(late_expression_list)

    param_pool = {"max_loc_overall": max_locs_array,
                  "min_loc_overall": min_locs_array,
                  "max_amp_overall": max_amplitudes_array,
                  "min_amp_overall": min_amplitudes_array,
                  "early_expression_array":early_expression_array,
                  "middle_expression_array":middle_expression_array,
                  "late_expression_array":late_expression_array}

    return ta_activity_dict_for_cluster, elbow_kmeans_per_cell_list, param_pool





def get_labels_all_different_Ks_single(model_20_NDNF_resid, which_vectors: int):

    w1 = model_20_NDNF_resid.vectors[which_vectors][0]
    f1 = model_20_NDNF_resid.vectors[which_vectors][1]
    F = f1.detach().cpu().numpy()   # (latents, cells, pos) = (20, 115, 50)
    W = w1.detach().cpu().numpy()   # (latents, trials) = (20, 100)

    print(f"F.shape {F.shape}  W.shape {W.shape}")

    # Build X so rows = cells (115)
    if which_vectors == 0:
        # Use latent×pos per cell, flattened: (115, 20*50)
        X = np.moveaxis(F, 1, 0)              # (cells=115, latents=20, pos=50)
        X = X.reshape(X.shape[0], -1)         # (115, 1000)
        print("X shape (latent×pos flat):", X.shape)

    elif which_vectors == 1:
        X = W.T  # (115, 20) mean over pos
        print("X shape (mean over pos):", X.shape)

    else:
        X = np.moveaxis(F, 2, 0)     # -> (cells=115, latents=20, trials=100)
        X = X.reshape(X.shape[0], -1)  # -> (115, 20*100) = (115, 2000)
        print(X.shape)  # (115, 2000)

    Xz = StandardScaler().fit_transform(X)
    labels_cells_dict_all_K = {K: KMeans(n_clusters=K, n_init=100, random_state=42).fit_predict(Xz) for K in range(1, 11)}
    return labels_cells_dict_all_K



def get_selectivity_each_trial_cell_type(activity_dict_EC, cells_list, neg_sel=True, trial_av=True, norm=None):
    count = 0
    cells_set = set(int(x) for x in cells_list)

    out = {}
    animals_dict_data = {}
    for animal in activity_dict_EC:
        cell_dict = {}
        cell_dict_data = {}
        for cell in activity_dict_EC[animal]:
            # print(f"count {count} cells_list {cells_list}")
            if count in cells_set:
                cell_data = activity_dict_EC[animal][cell]
                if trial_av:
                    trial_av_activity = np.mean(cell_data, axis=1)
                    cell_dict_data[cell] = cell_data
                    val = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)

                else:
                    vals = [Vinje2000(cell_data[:, tr], norm='none', negative_selectivity=neg_sel)
                            for tr in range(cell_data.shape[1])]
                    val = float(np.mean(vals)) if len(vals) else np.nan
                cell_dict[cell] = val
            count += 1  # increment on EVERY cell
        out[animal] = cell_dict
        animals_dict_data[animal] = cell_dict_data
    return out, animals_dict_data


def Vinje2000(tuning_curve, norm='None', negative_selectivity=False):
    if norm == 'min_max':
        tuning_curve = (tuning_curve - np.min(tuning_curve)) / (np.max(tuning_curve) - np.min(tuning_curve))
        if negative_selectivity:
            tuning_curve = np.absolute(1 - tuning_curve)
    elif norm == 'z_score':
        tuning_curve = (tuning_curve - np.mean(tuning_curve)) / np.std(tuning_curve)
        if negative_selectivity:
            tuning_curve = np.absolute(1 - tuning_curve)
    A = np.mean(tuning_curve) ** 2 / np.mean(tuning_curve ** 2)
    return (1 - A) / (1 - 1 / len(tuning_curve))



# everything_dict = {"overall_activity_list_most_array":overall_activity_list_most_array, 
#                     "overall_activity_list_least_array":overall_activity_list_least_array, 
#                     "most_expressed_expression_amount_array":most_expressed_expression_amount_array, 
#                     "least_expressed_expression_amount_array":least_expressed_expression_amount_array,
#                     "argmax_array_early" : argmax_array_early,
#                     "argmin_array_early" : argmin_array_early,
#                     "max_amp_array_early" : max_amp_array_early,
#                     "min_amp_array_early" : min_amp_array_early,
#                     "argmax_array_late" : argmax_array_late,
#                     "argmin_array_late" : argmin_array_late,
#                     "max_amp_array_late" : max_amp_array_late,
#                     "min_amp_array_late" : min_amp_array_late}



def plot_no_learn_data(everything_dict0, everything_dict1, color_dict=None, axs_list=None):

    mean_0_array_most = everything_dict0["overall_activity_list_most_array"]
    mean_1_array_most = everything_dict1["overall_activity_list_most_array"]

    mean_0_array_least = everything_dict0["overall_activity_list_least_array"]
    mean_1_array_least = everything_dict1["overall_activity_list_least_array"]


    axs_list[0].set_title("Cell Type 0 \n Most Expressed")
    axs_list[2].set_title("Cell Type 0 \n Least Expressed")


    axs_list[1].set_title("Cell Type 1 \n Most Expressed")
    axs_list[3].set_title("Cell Type 1 \n Least Expressed")

    mean_mean_0_array_most = np.mean(mean_0_array_most, axis=0)
    mean_mean_1_array_most = np.mean(mean_1_array_most, axis=0)

    mean_mean_0_array_least = np.mean(mean_0_array_least, axis=0)
    mean_mean_1_array_least = np.mean(mean_1_array_least, axis=0)


    sem_0_most = sem(mean_0_array_most, axis=0, nan_policy='omit') 
    sem_1_most = sem(mean_1_array_most, axis=0, nan_policy='omit')

    
    sem_0_least = sem(mean_0_array_least, axis=0, nan_policy='omit') 
    sem_1_least = sem(mean_1_array_least, axis=0, nan_policy='omit')

    axs_list[0].plot(mean_mean_0_array_most, linewidth=4, color=color_dict["Most_0"])
    axs_list[0].set_xlabel("Position Bins")
    axs_list[2].set_xlabel("Position Bins")
    axs_list[0].set_ylabel("Z-Scored DF/F")
    axs_list[2].set_ylabel("Z-Scored DF/F")
    axs_list[0].set_ylim(-1.5, 4)
    axs_list[2].plot(mean_mean_0_array_least, linewidth=4, color=color_dict["Least_0"])
    axs_list[2].set_ylim(-1.5, 4)

    axs_list[1].set_xlabel("Position Bins")
    axs_list[3].set_xlabel("Position Bins")
    axs_list[1].set_ylabel("Z-Scored DF/F")
    axs_list[3].set_ylabel("Z-Scored DF/F")
    axs_list[1].plot(mean_mean_1_array_most, linewidth=4, color=color_dict["Most_1"])
    axs_list[1].set_ylim(-1.5, 4)
    axs_list[3].plot(mean_mean_1_array_least, linewidth=4, color=color_dict["Least_1"])
    axs_list[3].set_ylim(-1.5, 4)

    ax = axs_list[4]
    ax.set_title("Cell Type 0")
    ax.set_xlabel("Position Bins")
    ax.set_ylabel("Z-Scored DF/F")
    ax.plot(mean_mean_0_array_most, label="Most Expressed Trial Type", color=color_dict["Most_0"])
    ax.fill_between(range(len(mean_mean_0_array_most)), mean_mean_0_array_most - sem_0_most, mean_mean_0_array_most + sem_0_most, alpha=0.2, color=color_dict["Most_0"])
    ax.plot(mean_mean_0_array_least, label="Least Expressed Trial Type", color=color_dict["Least_0"])
    ax.fill_between(range(len(mean_mean_0_array_least)), mean_mean_0_array_least - sem_0_least, mean_mean_0_array_least + sem_0_least, alpha=0.2, color=color_dict["Least_0"])
    ax.legend(loc="upper right")

    ax = axs_list[5]
    ax.set_xlabel("Position Bins")
    ax.set_title("Cell Type 1")
    ax.plot(mean_mean_1_array_most, label="Most Expressed Trial Type", color=color_dict["Most_1"])
    ax.fill_between(range(len(mean_mean_1_array_most)), mean_mean_1_array_most - sem_1_most, mean_mean_1_array_most + sem_1_most, alpha=0.2, color=color_dict["Most_1"])
    ax.plot(mean_mean_1_array_least, label="Least Expressed Trial Type", color=color_dict["Least_1"])
    ax.fill_between(range(len(mean_mean_1_array_least)), mean_mean_1_array_least - sem_1_least, mean_mean_1_array_least + sem_1_least, alpha=0.2, color=color_dict["Least_1"])
    ax.set_ylabel("Z-Scored DF/F")
    ax.legend(loc="upper left")
    

    most_mean_0_fractions_list = everything_dict0["most_expressed_expression_amount_array"]
    least_mean_0_fractions_list = everything_dict0["least_expressed_expression_amount_array"]

    most_mean_1_fractions_list = everything_dict1["most_expressed_expression_amount_array"]
    least_mean_1_fractions_list = everything_dict1["least_expressed_expression_amount_array"]

    all_vals = np.concatenate([most_mean_0_fractions_list, least_mean_0_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    print(f"len(most_mean_0_fractions_list) {len(most_mean_0_fractions_list)}")

    ax = axs_list[6]
    ax.hist(most_mean_0_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_0"])
    ax.hist(least_mean_0_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_0"])
    ax.set_title("Cell Type 0")
    ax.set_xlabel("Fraction of Trials"); ax.set_ylabel("Number of Cells")
    ax.legend(frameon=False)

    all_vals = np.concatenate([most_mean_1_fractions_list, least_mean_1_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    ax = axs_list[7]
    ax.hist(most_mean_1_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_1"])
    ax.hist(least_mean_1_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_1"])
    ax.set_title("Cell Type 1")
    ax.set_xlabel("Fraction of Trials"); ax.set_ylabel("Number of Cells")
    ax.legend(frameon=False)

    


def plot_no_learn_cell_types(title_fs,
    most_expressed_label_dict_animal_cluster_0_all,
    most_expressed_label_dict_animal_cluster_1_all,
    least_expressed_label_dict_animal_all_group0,
    least_expressed_label_dict_animal_all_group1,
    group=None,
    axs_list=None, color_dict=None):
    """
    Expects axs_list length == 8 laid out however you want.
    Panel order:
      0: Group0 per-cell traces
      1: Group1 per-cell traces
      2: Group0 mean ± SEM
      3: Group1 mean ± SEM
      4: Group0 fraction histogram
      5: Group1 fraction histogram
      6: Overlay (Group0 vs Group1) mean ± SEM
      7: Elbow histogram (distribution of chosen K)
    """
    if axs_list is None or len(axs_list) < 8:
        raise ValueError("axs_list must be provided with at least 8 axes.")
    
    # --- collect means per cell for each group ---
    mean_0_list_most = []
    mean_0_list_least = []
    mean_1_list_most = []
    mean_1_list_least = []
    most_mean_0_fractions_list = []
    least_mean_0_fractions_list = []
    most_mean_1_fractions_list = []
    least_mean_1_fractions_list = []

    # Panel 0: Group0 per-cell traces
    # ax = axs_list[0]
    axs_list[0].set_title("Cell Type 0 \n Most Expressed", fontsize=title_fs)
    axs_list[2].set_title("Cell Type 0 \n Least Expressed", fontsize=title_fs)
    for cell in most_expressed_label_dict_animal_cluster_0_all:
        arr_most = most_expressed_label_dict_animal_cluster_0_all[cell]["cluster_activity"]
        arr_least = least_expressed_label_dict_animal_all_group0[cell]["cluster_activity"]
        arr_most_mean = np.mean(arr_most, axis=1)  # avg over trials/time dim as you intended
        arr_least_mean = np.mean(arr_least, axis=1)
        mean_0_list_most.append(arr_most_mean)
        mean_0_list_least.append(arr_least_mean)
        axs_list[0].plot(arr_most_mean, alpha=0.4, color='gray')
        axs_list[2].plot(arr_least_mean, alpha=0.4, color='gray')
        most_mean_0_fractions_list.append(most_expressed_label_dict_animal_cluster_0_all[cell]["fraction"])
        least_mean_0_fractions_list.append(least_expressed_label_dict_animal_all_group0[cell]["fraction"])


    # ax = axs_list[1]
    axs_list[1].set_title("Cell Type 1 \n Most Expressed", fontsize=title_fs)
    axs_list[3].set_title("Cell Type 1 \n Least Expressed", fontsize=title_fs)
    for cell in most_expressed_label_dict_animal_cluster_1_all:
        arr_most = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
        arr_least = least_expressed_label_dict_animal_all_group1[cell]["cluster_activity"]
        arr_most_mean = np.mean(arr_most, axis=1)  # avg over trials/time dim as you intended
        arr_least_mean = np.mean(arr_least, axis=1)
        mean_1_list_most.append(arr_most_mean)
        mean_1_list_least.append(arr_least_mean)
        axs_list[1].plot(arr_most_mean, alpha=0.4, color='gray')
        axs_list[3].plot(arr_least_mean, alpha=0.4, color='gray')
        most_mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])
        least_mean_1_fractions_list.append(least_expressed_label_dict_animal_all_group1[cell]["fraction"])

    # # Panel 1: Group1 per-cell traces
    # ax = axs_list[1]
    # ax.set_title("Group1: per-cell")
    # for cell in most_expressed_label_dict_animal_cluster_1_all:
    #     arr = most_expressed_label_dict_animal_cluster_1_all[cell]["cluster_activity"]
    #     m = np.mean(arr, axis=1)
    #     mean_1_list.append(m)
    #     ax.plot(m, alpha=0.4)
    #     mean_1_fractions_list.append(most_expressed_label_dict_animal_cluster_1_all[cell]["fraction"])

    # Convert to arrays
    mean_0_array_most = np.array(mean_0_list_most)
    mean_1_array_most = np.array(mean_1_list_most)

    mean_0_array_least = np.array(mean_0_list_least)
    mean_1_array_least = np.array(mean_1_list_least)


    mean_mean_0_array_most = np.mean(mean_0_array_most, axis=0)
    mean_mean_1_array_most = np.mean(mean_1_array_most, axis=0)

    mean_mean_0_array_least = np.mean(mean_0_array_least, axis=0)
    mean_mean_1_array_least = np.mean(mean_1_array_least, axis=0)


    sem_0_most = sem(mean_0_array_most, axis=0, nan_policy='omit') 
    sem_1_most = sem(mean_1_array_most, axis=0, nan_policy='omit')

    
    sem_0_least = sem(mean_0_array_least, axis=0, nan_policy='omit') 
    sem_1_least = sem(mean_1_array_least, axis=0, nan_policy='omit')

    axs_list[0].plot(mean_mean_0_array_most, linewidth=4, color=color_dict["Most_0"])
    axs_list[0].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[2].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[0].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[2].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[0].set_ylim(-1.5, 4)
    axs_list[2].plot(mean_mean_0_array_least, linewidth=4, color=color_dict["Least_0"])
    axs_list[2].set_ylim(-1.5, 4)

    axs_list[1].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[3].set_xlabel("Position Bins", fontsize=title_fs-1)
    axs_list[1].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[3].set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    axs_list[1].plot(mean_mean_1_array_most, linewidth=4, color=color_dict["Most_1"])
    axs_list[1].set_ylim(-1.5, 4)
    axs_list[3].plot(mean_mean_1_array_least, linewidth=4, color=color_dict["Least_1"])
    axs_list[3].set_ylim(-1.5, 4)

    ax = axs_list[4]
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Position Bins", fontsize=title_fs-1)
    ax.set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    ax.plot(mean_mean_0_array_most, label="Most Expressed Trial Type", color=color_dict["Most_0"])
    ax.fill_between(range(len(mean_mean_0_array_most)), mean_mean_0_array_most - sem_0_most, mean_mean_0_array_most + sem_0_most, alpha=0.2, color=color_dict["Most_0"])
    ax.plot(mean_mean_0_array_least, label="Least Expressed Trial Type", color=color_dict["Least_0"])
    ax.fill_between(range(len(mean_mean_0_array_least)), mean_mean_0_array_least - sem_0_least, mean_mean_0_array_least + sem_0_least, alpha=0.2, color=color_dict["Least_0"])
    ax.legend(fontsize=title_fs-3, loc="upper right")

    ax = axs_list[5]
    ax.set_xlabel("Position Bins", fontsize=title_fs-1)
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.plot(mean_mean_1_array_most, label="Most Expressed Trial Type", color=color_dict["Most_1"])
    ax.fill_between(range(len(mean_mean_1_array_most)), mean_mean_1_array_most - sem_1_most, mean_mean_1_array_most + sem_1_most, alpha=0.2, color=color_dict["Most_1"])
    ax.plot(mean_mean_1_array_least, label="Least Expressed Trial Type", color=color_dict["Least_1"])
    ax.fill_between(range(len(mean_mean_1_array_least)), mean_mean_1_array_least - sem_1_least, mean_mean_1_array_least + sem_1_least, alpha=0.2, color=color_dict["Least_1"])
    ax.set_ylabel("Z-Scored DF/F", fontsize=title_fs-1)
    ax.legend(fontsize=title_fs-3, loc="upper left")
    
    # # Panel 2: Group0 mean±SEM
    # ax = axs_list[2]
    # ax.set_title("Group0: mean ± SEM")
    # if mean_mean_0.size:
    #     t = np.arange(len(mean_mean_0))
    #     ax.plot(mean_mean_0, label='Group0')
    #     ax.fill_between(t, mean_mean_0 - sem_0, mean_mean_0 + sem_0, alpha=0.2)
    #     ax.legend(frameon=False)

    # # Panel 3: Group1 mean±SEM
    # ax = axs_list[3]
    # ax.set_title("Group1: mean ± SEM")
    # if mean_mean_1.size:
    #     t = np.arange(len(mean_mean_1))
    #     ax.plot(mean_mean_1, label='Group1')
    #     ax.fill_between(t, mean_mean_1 - sem_1, mean_mean_1 + sem_1, alpha=0.2)
    #     ax.legend(frameon=False)

    # Panel 4: Group0 fraction histogram
    # ax = axs_list[4]
    # ax.hist(most_mean_0_fractions_list, bins='auto', alpha=0.2)
    # ax.hist(least_mean_0_fractions_list, bins='auto', alpha=0.2)
    # ax.set_title("Group0: fraction of trials")
    # ax.set_xlabel("Fraction")
    # ax.set_ylabel("Cells")

    all_vals = np.concatenate([most_mean_0_fractions_list, least_mean_0_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    print(f"len(most_mean_0_fractions_list) {len(most_mean_0_fractions_list)}")

    ax = axs_list[6]
    ax.hist(most_mean_0_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_0"])
    ax.hist(least_mean_0_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_0"])
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Fraction of Trials",fontsize=title_fs-1); ax.set_ylabel("Number of Cells",fontsize=title_fs-1)
    ax.legend(frameon=False,fontsize=title_fs-1)

    # Panel 5: Group1 fraction histogram
    # ax = axs_list[5]
    # ax.hist(most_mean_1_fractions_list, bins='auto', alpha=0.2)
    # ax.hist(least_mean_1_fractions_list, bins='auto', alpha=0.2)
    # ax.set_title("Group1: fraction of trials")
    # ax.set_xlabel("Fraction")
    # ax.set_ylabel("Cells")

    all_vals = np.concatenate([most_mean_1_fractions_list, least_mean_1_fractions_list])
    bins = np.histogram_bin_edges(all_vals, bins='auto')

    ax = axs_list[7]
    ax.hist(most_mean_1_fractions_list, bins=bins, alpha=0.35, label='Most', edgecolor='none', color=color_dict["Most_1"])
    ax.hist(least_mean_1_fractions_list, bins=bins, alpha=0.35, label='Least', edgecolor='none', color=color_dict["Least_1"])
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.set_xlabel("Fraction of Trials",fontsize=title_fs-1); ax.set_ylabel("Number of Cells",fontsize=title_fs-1)
    ax.legend(frameon=False,fontsize=title_fs-1)

    # # Panel 6: Overlay Group0 vs Group1 mean±SEM
    # ax = axs_list[6]
    # ax.set_title("Overlay: Group0 vs Group1")
    # if mean_mean_0.size:
    #     t0 = np.arange(len(mean_mean_0))
    #     ax.plot(mean_mean_0, label='Group0')
    #     ax.fill_between(t0, mean_mean_0 - sem_0, mean_mean_0 + sem_0, alpha=0.2)
    # if mean_mean_1.size:
    #     t1 = np.arange(len(mean_mean_1))
    #     ax.plot(mean_mean_1, label='Group1')
    #     ax.fill_between(t1, mean_mean_1 - sem_1, mean_mean_1 + sem_1, alpha=0.2)
    # ax.legend(frameon=False)

    # Panel 7: Elbow histogram


def get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True):

    trial_type_activity_list_all_cells = []

    argmax_list = []
    argmin_list = []

    argmax_amp_list = [[] for _ in range(50)]
    argmin_amp_list = [[] for _ in range(50)]


    for idx, cell in enumerate(cells_group_0):
        num_clusters_chosen = elbow_kmeans_array_group0_most[idx]
        labels = TT_list[cell]['labels_dict'][f'clusters_chosen_{int(num_clusters_chosen)}']
        # tt_cluster_indices = TT_list[cell]['indices_for_cluster_number'][f'clusters_chosen_{int(num_clusters_chosen)}']
        trial_type_activity_list = []

        cp_list = cp_list_NDNF[cell]
        
        if use_early:
            cp = cp_list[0]
        else:
            cp = cp_list[1]

        unique_labels = np.unique(labels)
        # for tt_cluster in range(len(tt_cluster_indices)):
        for tt_cluster in range(len(unique_labels)):
            data_for_cell = NDNF_activity_list[cell]
            # trial_indices = tt_cluster_indices[tt_cluster]
            trial_indices = np.where(labels==tt_cluster)[0]

            if use_early:
                valid_trial_indices = np.where(trial_indices<cp)[0]
            else:
                valid_trial_indices = np.where(trial_indices>cp)[0]

            data_slice_trail_type = data_for_cell[:,valid_trial_indices]
            trial_type_activity_list.append(data_slice_trail_type)
            mean_data_slice_trail_type= np.mean(data_slice_trail_type, axis=1)
            max_loc = np.argmax(mean_data_slice_trail_type)
            min_loc = np.argmin(mean_data_slice_trail_type)
            argmax_amp_list[max_loc].append(mean_data_slice_trail_type[max_loc])
            argmin_amp_list[min_loc].append(mean_data_slice_trail_type[min_loc])
            argmax_list.append(max_loc)
            argmin_list.append(min_loc)
        trial_type_activity_list_all_cells.append(trial_type_activity_list)

    return argmax_list, argmin_list, argmax_amp_list, argmin_amp_list



def eval_proportion_two_groups(title_fs,
    early_dict_g0,
    late_dict_g0,
    early_dict_g1,
    late_dict_g1, ymin=None,ymax=None,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=("C0", "C1"),
    ax=None,
    title="Fraction of trials", Most=None, 
):
    """
    Plot early vs late fraction for two cell types on the same axis.

    For each group:
      - compute mean ± SEM for Early and Late
      - plot a line connecting Early→Late means
      - add vertical SEM errorbars at Early and Late

    No individual cell points are shown.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    # ---- helper to extract early/late arrays from dict ----
    def dict_to_arrays(d_early, d_late):
        early_vals = []
        late_vals  = []
        for cell in d_early:
            early_vals.append(d_early[cell]["fraction"])
            late_vals.append(d_late[cell]["fraction"])
        early_vals = np.array(early_vals)
        late_vals  = np.array(late_vals)
        return early_vals, late_vals

    # ---- Group 0 ----
    early0, late0 = dict_to_arrays(early_dict_g0, late_dict_g0)
    mean_e0 = early0.mean()
    mean_l0 = late0.mean()
    sem_e0  = early0.std(ddof=1) / np.sqrt(len(early0))
    sem_l0  = late0.std(ddof=1) / np.sqrt(len(late0))

    # optional: paired t-test for group 0
    t0, p0 = ttest_rel(early0, late0)

    # ---- Group 1 ----
    early1, late1 = dict_to_arrays(early_dict_g1, late_dict_g1)
    mean_e1 = early1.mean()
    mean_l1 = late1.mean()
    sem_e1  = early1.std(ddof=1) / np.sqrt(len(early1))
    sem_l1  = late1.std(ddof=1) / np.sqrt(len(late1))

    # optional: paired t-test for group 1
    t1, p1 = ttest_rel(early1, late1)

    # x positions: 0 = Early, 1 = Late
    x_early = 0.1
    x_late  = 0.9

    # ---- plot Group 0 ----
    ax.plot(
        [x_early, x_late],
        [mean_e0, mean_l0],
        color=colors[0],
        marker='o',
        linewidth=2,
        label=f"{group_labels[0]} (p={p0:.3f})"
    )
    ax.errorbar(
        x_early, mean_e0, yerr=sem_e0,
        fmt='none', ecolor=colors[0], elinewidth=1.5, capsize=4, linestyle='--'
    )
    ax.errorbar(
        x_late, mean_l0, yerr=sem_l0,
        fmt='none', ecolor=colors[0], elinewidth=1.5, capsize=4, linestyle='--'
    )

    # ---- plot Group 1 ----
    ax.plot(
        [x_early, x_late],
        [mean_e1, mean_l1],
        color=colors[1],
        marker='o',
        linewidth=2,
        label=f"{group_labels[1]} (p={p1:.3f})"
    )
    ax.errorbar(
        x_early, mean_e1, yerr=sem_e1,
        fmt='none', ecolor=colors[1], elinewidth=1.5, capsize=4, linestyle='--'
    )
    ax.errorbar(
        x_late, mean_l1, yerr=sem_l1,
        fmt='none', ecolor=colors[1], elinewidth=1.5, capsize=4, linestyle='--'
    )

    # ---- cosmetics ----
    ax.set_xticks([0.1, 0.9])
    ax.set_xlim([0., 1.])
    ax.set_xticklabels(['Early', 'Late'],fontsize=title_fs-1)
    ax.set_ylabel("Fraction of Trials",fontsize=title_fs-1)
    ax.set_ylim(ymin,ymax)
    ax.set_title(title, fontsize=title_fs)
    ax.legend(frameon=False, fontsize=5)


    return {
        "group0": {"early": early0, "late": late0, "p": p0},
        "group1": {"early": early1, "late": late1, "p": p1}
    }




def find_elbow(K, mse):
    # normalize
    x = (K - K.min()) / (K.max() - K.min())
    y = (mse - mse.min()) / (mse.max() - mse.min())

    # line between endpoints
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # distance from each point to the line
    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(d)

    distances = np.array(distances)
    elbow_idx = np.argmax(distances)
    return K[elbow_idx], distances


def trial_types_for_given_k(some_things_dict, k=None, title=None, ymin_act=None, ymax_act=None, ymin_percent=None, ymax_percent=None):



    overall_activity_list_all = some_things_dict["overall_activity_list_all"]
    overall_activity_list_all_array = np.array(overall_activity_list_all)

    cells_for_cluster_list = some_things_dict["cells_for_cluster_list"]
    cells_for_cluster_array = np.array(cells_for_cluster_list)

    expression_list_early_overall = some_things_dict["expression_list_early_overall"]
    expression_array_early_overall = np.array(expression_list_early_overall)

    expression_list_late_overall = some_things_dict["expression_list_late_overall"]
    expression_array_late_overall = np.array(expression_list_late_overall)

    activity_list_early_overall = some_things_dict["activity_list_early_overall"]
    activity_array_early_overall = np.array(activity_list_early_overall)

    activity_list_late_overall = some_things_dict["activity_list_late_overall"]
    activity_array_late_overall = np.array(activity_list_late_overall)

    fig, axs = plt.subplots(3, k+1, figsize=(4*k, 8))

    fig.suptitle(title)

    mse_list = []

    for h in range(1,11):

        kmeans = KMeans(n_clusters=h, random_state=42, n_init='auto')

        labels = kmeans.fit_predict(overall_activity_list_all_array)

        empty_array = np.empty(overall_activity_list_all_array.shape)

        u = np.unique(labels)

        for i in u:

            labels_for_cluster = np.where(labels==i)[0]

            mean_activity_in_cluster = np.mean(overall_activity_list_all_array[labels_for_cluster,:], axis=0)

            empty_array[labels_for_cluster,:] = mean_activity_in_cluster

        mse_list.append(np.mean(np.square(empty_array-overall_activity_list_all_array)))


    K = np.arange(1, len(mse_list)+1)

    mse_array = np.array(mse_list)

    elbow_k, distances = find_elbow(K, mse_array)

    axs[2, 0].plot(K, mse_array, '-o')
    axs[2, 0].set_title(f"Elbow Number of Clusters")
    axs[2, 0].axvline(elbow_k+1, color='r', linestyle='--', label=f'Elbow K={elbow_k+1}')
    axs[2, 0].legend()




    mse_list = []

    labels_list = []

    overall_unique_cells = np.unique(cells_for_cluster_array)
    n_cells = len(overall_unique_cells)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')

    labels = kmeans.fit_predict(overall_activity_list_all_array)

    labels_list.append(labels)

    empty_array = np.empty(overall_activity_list_all_array.shape)

    u = np.unique(labels)



    # Force axs to always be 2D: (3, n_clusters)
    axs = np.atleast_2d(axs)
    if axs.shape[0] != 3:
        axs = axs.reshape(3, -1)

    axs[0, 0].imshow(overall_activity_list_all_array, aspect='auto')
    axs[0, 0].set_ylabel("Cluster ID")
    axs[0, 0].set_xlabel("Position Bins")
    




    for i in u:
        labels_for_cluster = np.where(labels==i)[0]
        
        mean_activity_in_cluster = np.mean(overall_activity_list_all_array[labels_for_cluster,:], axis=0)
        sem_activity_in_cluster = sem(overall_activity_list_all_array[labels_for_cluster,:], axis=0)

        mean_activity_in_cluster_early = np.nanmean(activity_array_early_overall[labels_for_cluster,:], axis=0)
        mean_activity_in_cluster_late = np.nanmean(activity_array_late_overall[labels_for_cluster,:], axis=0)

        sem_activity_in_cluster_early = sem(activity_array_early_overall[labels_for_cluster,:], axis=0, nan_policy='omit')
        sem_activity_in_cluster_late = sem(activity_array_late_overall[labels_for_cluster,:], axis=0, nan_policy='omit')

        num_cells_expressed = np.array(cells_for_cluster_list)[labels_for_cluster]

        unique_cells = np.unique(num_cells_expressed)


        print(f"len(num_cells_expressed) {len(num_cells_expressed)} len(unique_cells) {len(unique_cells)}")

        expression_array_early_overall_cluster = expression_array_early_overall[labels_for_cluster]
        expression_array_late_overall_cluster = expression_array_late_overall[labels_for_cluster]

        mean_expression_in_cluster_early = np.mean(expression_array_early_overall_cluster)
        mean_expression_in_cluster_late = np.mean(expression_array_late_overall_cluster)
        sem_expression_in_cluster_early = sem(expression_array_early_overall_cluster)
        sem_expression_in_cluster_late = sem(expression_array_late_overall_cluster)

        empty_array[labels_for_cluster,:] = mean_activity_in_cluster

        axs[0,i+1].plot(mean_activity_in_cluster)
        axs[0,i+1].fill_between(range(len(mean_activity_in_cluster)), mean_activity_in_cluster-sem_activity_in_cluster, mean_activity_in_cluster+sem_activity_in_cluster, alpha=0.2)
        axs[0,i+1].set_title(f"Cluster Number {i} \n {len(unique_cells)}/{n_cells} Cells = {(len(unique_cells)/n_cells)*100:.2f}%")
        axs[0,i+1].set_ylabel(f"Z-Scored DF/F")
        axs[0,i+1].set_xlabel(f"Position Bins")
        axs[0,i+1].set_ylim(ymin_act, ymax_act)
        

        axs[1,i+1].plot(mean_activity_in_cluster_early, label="Early")
        axs[1,i+1].plot(mean_activity_in_cluster_late, label="Late")
        axs[1,i+1].fill_between(range(len(mean_activity_in_cluster_early)), mean_activity_in_cluster_early+sem_activity_in_cluster_early, mean_activity_in_cluster_early-sem_activity_in_cluster_early, alpha=0.2)
        axs[1,i+1].fill_between(range(len(mean_activity_in_cluster_late)), mean_activity_in_cluster_late+sem_activity_in_cluster_late, mean_activity_in_cluster_late-sem_activity_in_cluster_late, alpha=0.2)
        axs[1,i+1].set_ylabel(f"Z-Scored DF/F")
        axs[1,i+1].set_xlabel(f"Position Bins")
        axs[1,i+1].legend()
        axs[1,i+1].set_ylim(ymin_act, ymax_act)

        x = [0.1, 0.9]

        t, p = ttest_rel(expression_array_early_overall_cluster, expression_array_late_overall_cluster)

        means_list = [mean_expression_in_cluster_early, mean_expression_in_cluster_late]
        
        sems_list = [sem_expression_in_cluster_early, sem_expression_in_cluster_late]

        axs[2,i+1].plot(x, means_list, marker='o')
        axs[2, i+1].errorbar(x, means_list, yerr=sems_list, fmt='o')
        axs[2,i+1].set_title(f"Early vs Late p={p:.3f}")
        axs[2,i+1].set_xticks(x, ["Early", "Late"])
        axs[2,i+1].set_ylim(ymin_percent, ymax_percent)


    axs[1, 0].imshow(empty_array, aspect='auto')
    axs[1, 0].set_ylabel("Cluster ID")
    axs[1, 0].set_xlabel("Position Bins")
    axs[1, 0].set_title(f"Reconstruction K={k}")



    plt.tight_layout()
    plt.show()

    



    return mse_list






def run(use_fixed_track, use_first_or_only, use_all, which_celltype):
    
    title_fs = 8


    filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat'

    animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict = get_animal_clean_dict_activity(filepath)

    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

    residual_activity_dict_NDNF_new = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)


    first_an_idx = 14
    last_an_idx = 29

    

    gospel_labels_dict={
        "A1_first_or_only":[17,18,20,21,22],
        "A1_after_B1":[15,16,19,23,24,25,26,27,28],
        "B1_first_or_only":[30,31,32,37,38,39,40,41],
        "B1_after_A1":[29,33,34,35,36]
    }

    animal_id_per_session_list = [

"CG186_250123",
"CG189_250211",
"CG190_250214",
"CG191_250216",
"LM084_240608",
"LM098_240807",
"MV161_241218",
"MV169_250222",
"MV170_250222",
"MV177_250325",
"MV180_250328",
"MV195_250522",
"MV196_250523",
"MV219_250818",
"MV228_250920",

"CG189_250215",
"CG190_250220",
"LM084_240610",
"MV161_241219",
"MV170_250228",
"MV171_250306",
"MV177_250328",
"MV180_250331",
"MV191_250516",
"MV196_250528",
"MV200_250617",
"MV219_250822",
"MV222_250830",
"MV228_250924",

"CG186_250131",
"CG189_250213",
"CG190_250217",
"CG191_250218",
"MV166_250222",
"MV171_250315",
"MV177_250506",
"MV180_250408",
"MV191_250514",
"MV196_250526",
"MV200_250615",
"MV219_250820",
"MV228_250922"]


    print(f"len(animal_id_per_session_list) {len(animal_id_per_session_list)}")

    animal_ids_for_condition_list = []

    fig, axs = plt.subplots(4, 4, figsize=(10, 9))  
    fig.subplots_adjust(hspace=0.9)


    if which_celltype=="NDNF":

        if use_fixed_track:

            clean_resid_activity_dict_NDNF_newest = {}

            clean_velocity_dict_NDNF_newest = {}

            clean_lick_dict_NDNF_newest = {}

            NDNF_model_dict_clean = {}

            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/better_NDNF_fixed_model_dict_clean.pkl', 'rb') as f:
                NDNF_model_dict  = pickle.load(f)


            save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_fixed_model.pkl'
            with open(save_path, 'rb') as f:
                sliceTCA_model = pickle.load(f)

            cell_count = 0
            for idx, animal in enumerate(residual_activity_dict_NDNF_new):
                if 14 < idx < 29:
                    idx_for_model_clean = idx-15
                    animal_key = f"animal_{idx+1}"
                    for cell in residual_activity_dict_NDNF_new[animal]:
                        cell_count+=1
                    if use_all:
                        if idx in gospel_labels_dict["A1_first_or_only"] or idx in gospel_labels_dict["A1_after_B1"]:
                                print(f"session idx {idx} in gospel_labels_dict[A1_first_or_only] {gospel_labels_dict['A1_first_or_only']}")
                                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                                animal_ids_for_condition_list.append(animal_id_per_session_list[idx])

                                
                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]


                                idx_array_one = np.array(gospel_labels_dict["A1_first_or_only"])
                                idx_array_two = np.array(gospel_labels_dict["A1_after_B1"])

                                total_idx = np.concatenate([idx_array_one, idx_array_two])

                                animal_included_list = np.array(animal_id_per_session_list)[total_idx]

                                fig.suptitle(f"All Fixed Sessions {animal_included_list}")
                    else:
                        if use_first_or_only:
                            if idx in gospel_labels_dict["A1_first_or_only"]:
                                print(f"session idx {idx} in gospel_labels_dict[A1_first_or_only] {gospel_labels_dict['A1_first_or_only']}")
                                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                                animal_ids_for_condition_list.append(animal_id_per_session_list[idx])

                                
                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]


                                idx_array = np.array(gospel_labels_dict["A1_first_or_only"])
                                animal_included_list = np.array(animal_id_per_session_list)[idx_array]

                                fig.suptitle(f"A1_first_or_only: {animal_included_list}")



                        else:
                            if idx in gospel_labels_dict["A1_after_B1"]:
                                print(f"idx {idx} made it here ")
                                print(f"session idx {idx} in gospel_labels_dict[A1_after_B1] {gospel_labels_dict['A1_after_B1']}")
                                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                                animal_ids_for_condition_list.append(animal_id_per_session_list[idx])

                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal]  #[idx_for_model_clean]


                                idx_array = np.array(gospel_labels_dict["A1_after_B1"])
                                animal_included_list = np.array(animal_id_per_session_list)[idx_array]

                                fig.suptitle(f"A1_after_B1: {animal_included_list}")


            count = 0
            binary_array = np.zeros(cell_count)
            for idx, animal in enumerate(residual_activity_dict_NDNF_new):
                if 14 < idx < 29:
                    for cell in residual_activity_dict_NDNF_new[animal]:
                        if use_all:
                            if idx in gospel_labels_dict["A1_first_or_only"] or idx in gospel_labels_dict["A1_after_B1"]:
                                    binary_array[count] = 1
                        else:
                            if use_first_or_only:
                                if idx in gospel_labels_dict["A1_first_or_only"]:
                                    binary_array[count] = 1
                            else:
                                if idx in gospel_labels_dict["A1_after_B1"]:
                                    binary_array[count] = 1
                        count+=1

            print(f"binary_array.shape {binary_array.shape} {binary_array}")

            labels_dict_raw_new = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

            cells_array = labels_dict_raw_new[2] ### will produce an array num cells long

            idx_of_interest = np.where(binary_array==1)[0]

            labels = cells_array[idx_of_interest] 

            print(f"labels {labels}")
            
            
        else:

            NDNF_model_dict_clean = {}

            with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/better_NDNF_cued_model_dict_clean.pkl', 'rb') as f:
                NDNF_model_dict = pickle.load(f)

            save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_cued_model.pkl'
            with open(save_path, 'rb') as f:
                sliceTCA_model = pickle.load(f)

            clean_resid_activity_dict_NDNF_newest = {}

            clean_velocity_dict_NDNF_newest = {}

            clean_lick_dict_NDNF_newest = {}

            
            cell_count = 0
            for idx, animal in enumerate(residual_activity_dict_NDNF_new):
                if idx > 28:
                    idx_for_model_clean = idx-29
                    animal_key = f"animal_{idx+1}"
                    for cell in residual_activity_dict_NDNF_new[animal]:
                        cell_count+=1

                    if use_all:
                        if idx in gospel_labels_dict["B1_first_or_only"] or idx in gospel_labels_dict['B1_after_A1']:
                            print(f"session idx {idx} in gospel_labels_dict[B1_first_or_only] {gospel_labels_dict['B1_first_or_only']}")
                            clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                            clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                            clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                            animal_ids_for_condition_list.append(animal_id_per_session_list[idx])

                            idx_array_one = np.array(gospel_labels_dict["B1_first_or_only"])
                            idx_array_two = np.array(gospel_labels_dict["B1_after_A1"])

                            all_idx = np.concatenate([idx_array_one, idx_array_two])

                            animal_included_list = np.array(animal_id_per_session_list)[all_idx]

                            fig.suptitle(f"All Cued Sessions: {animal_included_list}")

                            NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]

                    else:
                        if use_first_or_only:
                            if idx in gospel_labels_dict["B1_first_or_only"]:
                                print(f"session idx {idx} in gospel_labels_dict[B1_first_or_only] {gospel_labels_dict['B1_first_or_only']}")
                                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                                animal_ids_for_condition_list.append(animal_id_per_session_list[idx])

                                idx_array = np.array(gospel_labels_dict["B1_first_or_only"])
                                animal_included_list = np.array(animal_id_per_session_list)[idx_array]

                                fig.suptitle(f"B1_first_or_only: {animal_included_list}")

                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]

                        else:
                            if idx in gospel_labels_dict["B1_after_A1"]:
                                print(f"session idx {idx} in gospel_labels_dict[B1_after_A1] {gospel_labels_dict['B1_after_A1']}")
                                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
                                clean_lick_dict_NDNF_newest [f"animal_{idx+1}"] = animal_lick_dict[animal]
                                animal_ids_for_condition_list.append(animal_id_per_session_list[idx])
                            
                                idx_array = np.array(gospel_labels_dict["B1_after_A1"])
                                animal_included_list = np.array(animal_id_per_session_list)[idx_array]

                                fig.suptitle(f"B1_after_A1: {animal_included_list}")

                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]



            count = 0
            binary_array = np.zeros(cell_count)
            for idx, animal in enumerate(residual_activity_dict_NDNF_new):
                if idx > 28:
                    for cell in residual_activity_dict_NDNF_new[animal]:
                        if use_all:
                            if idx in gospel_labels_dict["B1_first_or_only"] or idx in gospel_labels_dict["B1_after_A1"]:
                                binary_array[count] = 1
                        else:
                            if use_first_or_only:
                                if idx in gospel_labels_dict["B1_first_or_only"]:
                                    binary_array[count] = 1
                            else:
                                if idx in gospel_labels_dict["B1_after_A1"]:
                                    binary_array[count] = 1
                        count+=1

            print(f"binary_array.shape {binary_array.shape} {binary_array}")

            labels_dict_raw_new = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

            cells_array = labels_dict_raw_new[2] ### will produce an array num cells long

            idx_of_interest = np.where(binary_array==1)[0]

            labels = cells_array[idx_of_interest] 

            print(f"labels {labels}")

                
    elif which_celltype=="EC":
        filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/EC_GLM.mat'

        animal_clean_dict_activity, clean_velocity_dict_NDNF_newest, animal_trials_original, animal_trials_clean, trials_to_remove_local, clean_lick_dict_NDNF_newest = get_animal_clean_dict_activity(filepath, use_final=False)

        GLM_params, predicted_activity_dict = fit_GLM_population(clean_velocity_dict_NDNF_newest, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

        clean_resid_activity_dict_NDNF_newest = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)


        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/EC_model_dict_clean.pkl', 'rb') as f:
                NDNF_model_dict_clean = pickle.load(f)


        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_EC_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)


        labels_dict = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)
        labels = labels_dict[2]


        cell_count = 0
        for animal in clean_resid_activity_dict_NDNF_newest:
            for cell in clean_resid_activity_dict_NDNF_newest[animal]:
                cell_count+=1

        idx_of_interest = np.arange(cell_count)


    elif which_celltype=="SST":

        filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/SSTindivsomata_GLM.mat'

        animal_clean_dict_activity, clean_velocity_dict_NDNF_newest, animal_trials_original, animal_trials_clean, trials_to_remove_local, clean_lick_dict_NDNF_newest = get_animal_clean_dict_activity(filepath, use_final=False)

        GLM_params, predicted_activity_dict = fit_GLM_population(clean_velocity_dict_NDNF_newest, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

        clean_resid_activity_dict_NDNF_newest = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)


        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/SST_model_dict_clean.pkl', 'rb') as f:
            NDNF_model_dict_clean = pickle.load(f)


        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_SST_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)


        labels_dict = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)
        labels = labels_dict[2]


        cell_count = 0
        for animal in clean_resid_activity_dict_NDNF_newest:
            for cell in clean_resid_activity_dict_NDNF_newest[animal]:
                cell_count+=1

        idx_of_interest = np.arange(cell_count)
            

    
    cells_group_0 = np.where(labels==0)[0]
    cells_group_1 = np.where(labels==1)[0]

    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_group_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_group_1, neg_sel=False, trial_av=True, norm="min_max")

    animal_average_selectivity_dict_NDNF_0_list = []
    for animal in animal_average_selectivity_dict_NDNF_0:
        for cell in animal_average_selectivity_dict_NDNF_0[animal]:
            animal_average_selectivity_dict_NDNF_0_list.append(animal_average_selectivity_dict_NDNF_0[animal][cell])

    animal_average_selectivity_dict_NDNF_1_list = []
    for animal in animal_average_selectivity_dict_NDNF_1:
        for cell in animal_average_selectivity_dict_NDNF_1[animal]:
            animal_average_selectivity_dict_NDNF_1_list.append(animal_average_selectivity_dict_NDNF_1[animal][cell])

    if np.mean(animal_average_selectivity_dict_NDNF_0_list) > np.mean(animal_average_selectivity_dict_NDNF_1_list):
        inverted_labels = 1 - labels
        cells_group_0 = np.where(inverted_labels==0)[0]
        cells_group_1 = np.where(inverted_labels==1)[0]
        labels = inverted_labels

    
    
    reassigned_dict_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=8, reassign_clusters=True, x00=True, umap=False, contiguous=False, ranks=20)

    contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)


    print(f"contig_dict_all_cell_tca.keys() {contig_dict_all_cell_tca.keys()}")
    

    contig_dict = reshape_contig_dict(contig_dict_all_cell_tca, NDNF_model_dict_clean)

    reassigned_dict = reshape_contig_dict(reassigned_dict_cell_tca, NDNF_model_dict_clean)

    cp_dict_NDNF = get_cp_dict(contig_dict)


    fixed_TT_data = {}

    print(f"reassigned_dict.keys() {reassigned_dict.keys()}")
    trial_type_data = reassigned_dict[20]
    print(f"trial_type_data.keys() {trial_type_data.keys()}")
    fixed_TT_data = {}

    print(f"use_fixed_track {use_fixed_track}")

    # for idx, animal in enumerate(trial_type_data):
    #     if use_fixed_track:
    #         print(f"first_an_idx {first_an_idx} idx {idx} last_an_idx {last_an_idx}")
    #         if first_an_idx < idx < last_an_idx:
    #             fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]
    #     else:
    #         if idx > last_an_idx-1:
    #             fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    for idx, animal in enumerate(trial_type_data):
        # fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]
        fixed_TT_data[animal] = trial_type_data[animal]

        print(f"animal {animal} ttd {trial_type_data[animal]['cell_2'][0]}")





    TT_list, NDNF_activity_list, cp_list_NDNF = get_lists_out_of_dicts(fixed_TT_data, clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF)

    important_dict = {"TT_list":TT_list,
                      "NDNF_activity_list":NDNF_activity_list,
                      "cp_list_NDNF": cp_list_NDNF}

    # save_path = "/Users/michaelfinch/CA1_interneuron_model/datasets/tt_pkl.pkl"
    # with open(save_path, 'wb') as f:
    #     pickle.dump(important_dict, f)



    # labels_cells_dict_all_K_NDNF = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

    # cell_type_labels = labels_cells_dict_all_K_NDNF[2]

    # cells_group_0 = np.where(cell_type_labels==0)[0]
    # cells_group_1 = np.where(cell_type_labels==1)[0]

    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_group_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_group_1, neg_sel=False, trial_av=True, norm="min_max")

    animal_average_selectivity_dict_NDNF_0_list = []
    for animal in animal_average_selectivity_dict_NDNF_0:
        for cell in animal_average_selectivity_dict_NDNF_0[animal]:
            animal_average_selectivity_dict_NDNF_0_list.append(animal_average_selectivity_dict_NDNF_0[animal][cell])

    animal_average_selectivity_dict_NDNF_1_list = []
    for animal in animal_average_selectivity_dict_NDNF_1:
        for cell in animal_average_selectivity_dict_NDNF_1[animal]:
            animal_average_selectivity_dict_NDNF_1_list.append(animal_average_selectivity_dict_NDNF_1[animal][cell])

    if np.mean(animal_average_selectivity_dict_NDNF_0_list) > np.mean(animal_average_selectivity_dict_NDNF_1_list):
        inverted_labels = 1 - cell_type_labels
        cells_group_0 = np.where(inverted_labels==0)[0]
        cells_group_1 = np.where(inverted_labels==1)[0]
        cell_type_labels = inverted_labels


    print(f"len(TT_list) {len(TT_list)} len(NDNF_activity_list) {len(NDNF_activity_list)} cp_list_NDNF {len(cp_list_NDNF)}")



    # most_expressed_label_dict_animal_early, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", most_expressed=most_expressed)
    # most_expressed_label_dict_animal_late, elbow_kmeans_array = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", most_expressed=most_expressed)






    most_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=True)
    print(f'most_expressed_label_dict_animal_early_group0[0][cluster_activity] {most_expressed_label_dict_animal_early_group0.keys()}')
    most_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=True)


    # most_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=True)
    # most_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=True)



    # most_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=True)
    # most_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=True)


    # least_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=False)
    # least_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=False)

    
    
    # least_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=False)
    # least_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=False)


    # least_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=False)
    # least_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=False)

    ta_activity_dict_for_cluster, elbow_kmeans_per_cell_list, param_pool = produce_ta_activity_dict_for_cluster(important_dict)



    elbow_kmeans_per_cell_list_group_0_list = []
    elbow_kmeans_per_cell_list_group_1_list = []

    import re
    

    for cell in range(len(elbow_kmeans_per_cell_list)):

        text = elbow_kmeans_per_cell_list[cell]
        match = re.search(r'\d+', text)
        result = int(match.group()) if match else None
        
        if cell in cells_group_0:
            elbow_kmeans_per_cell_list_group_0_list.append(result)
        else:
            elbow_kmeans_per_cell_list_group_1_list.append(result)


    def get_activity_most_least_celltype(ta_activity_dict_for_cluster, to_include=None):
        
        most_expressed_expression_amount_list = []
        least_expressed_expression_amount_list = []

        middle_expressed_expression_amount_list = []
        middle_expressed_activity_list = []

        middle_expressed_activity_early_list = []
        middle_expressed_activity_late_list = []

        middle_expressed_expression_list_early = []
        middle_expressed_expression_list_late = []

        overall_activity_list_most = []
        overall_activity_list_least = []

        overall_activity_list_all = []

        argmax_list_early = []
        argmin_list_early = []
        
        max_amp_list_early = []
        min_amp_list_early = []

        argmax_list_late = []
        argmin_list_late = []
        
        max_amp_list_late = []
        min_amp_list_late = []

        expression_list_early_overall = []
        expression_list_late_overall = []

        activity_list_early_overall = []
        activity_list_late_overall = []

        expression_list_late_pop_maxs = []
        expression_list_late_pop_mins = []

        cells_for_cluster_list = []

        expression_list_early_pop_maxs = []
        expression_list_early_pop_mins = []


        for cell in ta_activity_dict_for_cluster:
            if cell in to_include:
                expression_list = []
                activity_list = []

                expression_list_early = []
                expression_list_late = []

                activity_early_list = []
                activity_late_list = []

                for cluster in ta_activity_dict_for_cluster[cell]:
                    cluster_expression = ta_activity_dict_for_cluster[cell][cluster]["cluster_expression"]
                    expression_list.append(cluster_expression)
                    activity_list.append(ta_activity_dict_for_cluster[cell][cluster]["overall_activity"])

                    overall_activity_list_all.append(ta_activity_dict_for_cluster[cell][cluster]["overall_activity"])

                    max_loc_early = ta_activity_dict_for_cluster[cell][cluster]["max_loc_early"]
                    min_loc_early = ta_activity_dict_for_cluster[cell][cluster]["min_loc_early"]
                    max_amp_early = ta_activity_dict_for_cluster[cell][cluster]["max_amp_early"]
                    min_amp_early = ta_activity_dict_for_cluster[cell][cluster]["min_amp_early"]

                    activity_early = ta_activity_dict_for_cluster[cell][cluster]["early"]
                    activity_late = ta_activity_dict_for_cluster[cell][cluster]["late"]

                    cells_for_cluster_list.append(cell)

                    activity_early_list.append(activity_early)
                    activity_late_list.append(activity_late)

                    activity_list_early_overall.append(activity_early)
                    activity_list_late_overall.append(activity_late)

                    expression_list_early.append(ta_activity_dict_for_cluster[cell][cluster]["early_expression"])
                    expression_list_late.append(ta_activity_dict_for_cluster[cell][cluster]["late_expression"])

                    expression_list_early_overall.append(ta_activity_dict_for_cluster[cell][cluster]["early_expression"])
                    expression_list_late_overall.append(ta_activity_dict_for_cluster[cell][cluster]["late_expression"])

                    argmax_list_early.append(max_loc_early)
                    argmin_list_early.append(min_loc_early)
                    max_amp_list_early.append(max_amp_early)
                    min_amp_list_early.append(min_amp_early)
                    
                    max_loc_late = ta_activity_dict_for_cluster[cell][cluster]["max_loc_late"]
                    min_loc_late = ta_activity_dict_for_cluster[cell][cluster]["min_loc_late"]
                    max_amp_late = ta_activity_dict_for_cluster[cell][cluster]["max_amp_late"]
                    min_amp_late = ta_activity_dict_for_cluster[cell][cluster]["min_amp_late"]

                    argmax_list_late.append(max_loc_late)
                    argmin_list_late.append(min_loc_late)
                    max_amp_list_late.append(max_amp_late)
                    min_amp_list_late.append(min_amp_late)


                expression_list_early_pop_maxs.append(np.max(expression_list_early))
                expression_list_early_pop_mins.append(np.min(expression_list_early))

                for i in range(len(expression_list)):
                    
                        middle_expressed_expression_amount_list.append(expression_list[i])
                        middle_expressed_activity_list.append(activity_list[i])
                        middle_expressed_activity_early_list.append(activity_early_list[i])
                        middle_expressed_activity_late_list.append(activity_late_list[i])

                        middle_expressed_expression_list_early.append(expression_list_early[i])
                        middle_expressed_expression_list_late.append(expression_list_late[i])



                
                expression_list_late_pop_maxs.append(np.max(expression_list_late))
                expression_list_late_pop_mins.append(np.min(expression_list_late))
                

                most_expressed_expression = np.argmax(expression_list)
                most_expressed_expression_amount_list.append(expression_list[most_expressed_expression])
                least_expressed_expression = np.argmin(expression_list)
                least_expressed_expression_amount_list.append(expression_list[least_expressed_expression])

                activity_most = activity_list[most_expressed_expression]
                activity_least = activity_list[least_expressed_expression]

                overall_activity_list_most.append(activity_most)
                overall_activity_list_least.append(activity_least)

        most_expressed_expression_amount_array = np.array(most_expressed_expression_amount_list)
        least_expressed_expression_amount_array = np.array(least_expressed_expression_amount_list)

        overall_activity_list_most_array = np.array(overall_activity_list_most)
        overall_activity_list_least_array = np.array(overall_activity_list_least)

        expression_array_early_pop_maxs = np.array(expression_list_early_pop_maxs)
        expression_array_early_pop_mins = np.array(expression_list_early_pop_mins)

        expression_array_late_pop_maxs = np.array(expression_list_late_pop_maxs)
        expression_array_late_pop_mins = np.array(expression_list_late_pop_mins)


        argmax_array_early = np.array(argmax_list_early)
        argmin_array_early = np.array(argmin_list_early)
        
        max_amp_array_early = np.array(max_amp_list_early)
        min_amp_array_early = np.array(min_amp_list_early)

        argmax_array_late = np.array(argmax_list_late)
        argmin_array_late = np.array(argmin_list_late)
        
        max_amp_array_late = np.array(max_amp_list_late)
        min_amp_array_late = np.array(min_amp_list_late)

        everything_dict = {"cells_for_cluster_list":cells_for_cluster_list,
                            "overall_activity_list_most_array":overall_activity_list_most_array, 
                            "expression_list_early_overall" :expression_list_early_overall,
                            "expression_list_late_overall" :expression_list_late_overall,
                            "activity_list_early_overall" :activity_list_early_overall,
                            "activity_list_late_overall" :activity_list_late_overall,
                           "overall_activity_list_least_array":overall_activity_list_least_array, 
                           "most_expressed_expression_amount_array":most_expressed_expression_amount_array, 
                           "least_expressed_expression_amount_array":least_expressed_expression_amount_array,
                           "overall_activity_list_all":overall_activity_list_all,
                           "argmax_array_early" : argmax_array_early,
                            "argmin_array_early" : argmin_array_early,
                            "max_amp_array_early" : max_amp_array_early,
                            "min_amp_array_early" : min_amp_array_early,
                            "argmax_array_late" : argmax_array_late,
                            "argmin_array_late" : argmin_array_late,
                            "max_amp_array_late" : max_amp_array_late,
                            "min_amp_array_late" : min_amp_array_late,
                            "expression_array_early_pop_maxs" : expression_array_early_pop_maxs,
                            "expression_array_early_pop_mins" : expression_array_early_pop_mins,
                            "expression_array_late_pop_maxs" : expression_array_late_pop_maxs,
                            "expression_array_late_pop_mins" : expression_array_late_pop_mins,

}



        return everything_dict

    
    
    everything_dict0 = get_activity_most_least_celltype(ta_activity_dict_for_cluster, to_include=cells_group_0)
    everything_dict1 = get_activity_most_least_celltype(ta_activity_dict_for_cluster, to_include=cells_group_1)


    argmax_list_early_0 = everything_dict0["argmax_array_early"]
    argmin_list_early_0 = everything_dict0["argmin_array_early"]
    argmax_amp_list_early_0 = everything_dict0["max_amp_array_early"]
    argmin_amp_list_early_0 = everything_dict0["min_amp_array_early"]
    argmax_list_late_0 = everything_dict0["argmax_array_late"]
    argmin_list_late_0 = everything_dict0["argmin_array_late"]
    argmax_amp_list_late_0 = everything_dict0["max_amp_array_late"]
    argmin_amp_list_late_0 = everything_dict0["min_amp_array_late"]


    argmax_list_early_1 = everything_dict1["argmax_array_early"]
    argmin_list_early_1 = everything_dict1["argmin_array_early"]
    argmax_amp_list_early_1 = everything_dict1["max_amp_array_early"]
    argmin_amp_list_early_1 = everything_dict1["min_amp_array_early"]
    argmax_list_late_1 = everything_dict1["argmax_array_late"]
    argmin_list_late_1 = everything_dict1["argmin_array_late"]
    argmax_amp_list_late_1 = everything_dict1["max_amp_array_late"]
    argmin_amp_list_late_1 = everything_dict1["min_amp_array_late"]


    plot_butterfly_hist(argmax_list_early_0, argmax_list_late_0, argmin_list_early_0, argmin_list_late_0, ax=axs[0,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 0")
    plot_butterfly_hist(argmax_list_early_1, argmax_list_late_1, argmin_list_early_1, argmin_list_late_1, ax=axs[2,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 1")



    colors_dict = {"Most_0":"orange",
                   "Least_0":"magenta",
                   "Most_1":"purple",
                   "Least_1":"red"}

    axs_list = [
        axs[0,0], axs[0,1],
        axs[1,0], axs[1,1],
        axs[2,0], axs[2,1],
        axs[3,0], axs[3,1],]
    
    plot_no_learn_data(everything_dict0, everything_dict1, color_dict=colors_dict, axs_list=axs_list)
    


    def eval_proportion_two_groups_new(everything_dict0, everything_dict1, axs_expression_list, colors_dict, ymin_most=0.5, ymax_most=0.8, ymin_least=0.0, ymax_least=0.55):
        expression_array_early_pop_maxs0 = everything_dict0["expression_array_early_pop_maxs"]
        expression_array_early_pop_mins0 = everything_dict0["expression_array_early_pop_mins"]
        expression_array_late_pop_maxs0 = everything_dict0["expression_array_late_pop_maxs"]
        expression_array_late_pop_mins0 = everything_dict0["expression_array_late_pop_mins"]

        print(f"expression_array_early_pop_maxs0[0] {expression_array_early_pop_maxs0[0]}")

        mean_expression_array_early_pop_maxs0 = np.mean(expression_array_early_pop_maxs0)
        mean_expression_array_early_pop_mins0 = np.mean(expression_array_early_pop_mins0)
        mean_expression_array_late_pop_maxs0 = np.mean(expression_array_late_pop_maxs0)
        mean_expression_array_late_pop_mins0 = np.mean(expression_array_late_pop_mins0)

        sem_expression_array_early_pop_maxs0 = sem(expression_array_early_pop_maxs0)
        sem_expression_array_early_pop_mins0 = sem(expression_array_early_pop_mins0)
        sem_expression_array_late_pop_maxs0 = sem(expression_array_late_pop_maxs0)
        sem_expression_array_late_pop_mins0 = sem(expression_array_late_pop_mins0)
        

        expression_array_early_pop_maxs1 = everything_dict1["expression_array_early_pop_maxs"]
        expression_array_early_pop_mins1 = everything_dict1["expression_array_early_pop_mins"]
        expression_array_late_pop_maxs1 = everything_dict1["expression_array_late_pop_maxs"]
        expression_array_late_pop_mins1 = everything_dict1["expression_array_late_pop_mins"]


        mean_expression_array_early_pop_maxs1 = np.mean(expression_array_early_pop_maxs1)
        mean_expression_array_early_pop_mins1 = np.mean(expression_array_early_pop_mins1)
        mean_expression_array_late_pop_maxs1 = np.mean(expression_array_late_pop_maxs1)
        mean_expression_array_late_pop_mins1 = np.mean(expression_array_late_pop_mins1)

        sem_expression_array_early_pop_maxs1 = sem(expression_array_early_pop_maxs1)
        sem_expression_array_early_pop_mins1 = sem(expression_array_early_pop_mins1)
        sem_expression_array_late_pop_maxs1 = sem(expression_array_late_pop_maxs1)
        sem_expression_array_late_pop_mins1 = sem(expression_array_late_pop_mins1)


        x_early = 0.1
        x_late  = 0.9

        _, p0_most = ttest_rel(expression_array_early_pop_maxs0, expression_array_late_pop_maxs0)
        _, p1_most = ttest_rel(expression_array_early_pop_maxs1, expression_array_late_pop_maxs1)


        ax0 = axs_expression_list[0]
        ax1 = axs_expression_list[1]


        colors_dict = {"Most_0":"orange",
                "Least_0":"magenta",
                "Most_1":"purple",
                "Least_1":"red"}
        
        group_labels = ["Most Expressed Cell Type 0", "Most Expressed Cell Type 1", "Least Expressed Cell Type 0", "Least Expressed Cell Type 1"]

            
        ax0.plot([x_early, x_late],[mean_expression_array_early_pop_maxs0, mean_expression_array_late_pop_maxs0],
            color=colors_dict["Most_0"],
            marker='o',
            linewidth=2,
            label=f"{group_labels[0]} (p={p0_most:.3f})"
        )
        ax0.errorbar(
            x_early, mean_expression_array_early_pop_maxs0, yerr=sem_expression_array_early_pop_maxs0,
            fmt='none', ecolor=colors_dict["Most_0"], elinewidth=1.5, capsize=4, linestyle='--')
        
        ax0.errorbar(
            x_late, mean_expression_array_late_pop_maxs0, yerr=sem_expression_array_late_pop_maxs0,
            fmt='none', ecolor=colors_dict["Most_0"], elinewidth=1.5, capsize=4, linestyle='--')
        

        ax0.plot([x_early, x_late],[mean_expression_array_early_pop_maxs1, mean_expression_array_late_pop_maxs1],
            color=colors_dict["Most_1"],
            marker='o',
            linewidth=2,
            label=f"{group_labels[1]} (p={p1_most:.3f})"
        )
        ax0.errorbar(
            x_early, mean_expression_array_early_pop_maxs1, yerr=sem_expression_array_early_pop_maxs1,
            fmt='none', ecolor=colors_dict["Most_1"], elinewidth=1.5, capsize=4, linestyle='--')
        
        ax0.errorbar(
            x_late, mean_expression_array_late_pop_maxs1, yerr=sem_expression_array_late_pop_maxs1,
            fmt='none', ecolor=colors_dict["Most_1"], elinewidth=1.5, capsize=4, linestyle='--')
        


        _, p0_least = ttest_rel(expression_array_early_pop_mins0, expression_array_late_pop_mins0)
        _, p1_least = ttest_rel(expression_array_early_pop_mins1, expression_array_late_pop_mins1)

        ax1.plot([x_early, x_late],[mean_expression_array_early_pop_mins0, mean_expression_array_late_pop_mins0],
            color=colors_dict["Least_0"],
            marker='o',
            linewidth=2,
            label=f"{group_labels[2]} (p={p0_least:.3f})"
        )
        ax1.errorbar(
            x_early, mean_expression_array_early_pop_mins0, yerr=sem_expression_array_early_pop_mins0,
            fmt='none', ecolor=colors_dict["Least_0"], elinewidth=1.5, capsize=4, linestyle='--')
        
        ax1.errorbar(
            x_late, mean_expression_array_late_pop_mins0, yerr=sem_expression_array_late_pop_mins0,
            fmt='none', ecolor=colors_dict["Least_0"], elinewidth=1.5, capsize=4, linestyle='--')
        

        ax1.plot([x_early, x_late],[mean_expression_array_early_pop_mins1, mean_expression_array_late_pop_mins1],
            color=colors_dict["Least_1"],
            marker='o',
            linewidth=2,
            label=f"{group_labels[3]} (p={p1_least:.3f})"
        )
        ax1.errorbar(
            x_early, mean_expression_array_early_pop_mins1, yerr=sem_expression_array_early_pop_mins1,
            fmt='none', ecolor=colors_dict["Least_1"], elinewidth=1.5, capsize=4, linestyle='--')
        
        ax1.errorbar(
            x_late, mean_expression_array_late_pop_mins1, yerr=sem_expression_array_late_pop_mins1,
            fmt='none', ecolor=colors_dict["Least_1"], elinewidth=1.5, capsize=4, linestyle='--')
        



        # ---- cosmetics ----
        ax0.set_xticks([0.1, 0.9])
        ax0.set_xlim([0., 1.])
        ax0.set_xticklabels(['Early', 'Late'])
        ax0.set_ylabel("Fraction of Trials")
        ax0.set_ylim(ymin_most,ymax_most)
        ax0.set_title("Most Expressed Trial Types")
        ax0.legend(frameon=False)

         # ---- cosmetics ----
        ax1.set_xticks([0.1, 0.9])
        ax1.set_xlim([0., 1.])
        ax1.set_xticklabels(['Early', 'Late'])
        ax1.set_ylabel("Fraction of Trials")
        ax1.set_ylim(ymin_least,ymax_least)
        ax1.set_title("Least Expressed Trial Types")
        ax1.legend(frameon=False)


    axs_expression_list = [axs[0,2], axs[1,2]]

    eval_proportion_two_groups_new(everything_dict0, everything_dict1, axs_expression_list, colors_dict, ymin_most=0.5, ymax_most=0.7, ymin_least=0.05, ymax_least=0.15)

    # eval_proportion_two_groups(title_fs,
    # least_expressed_label_dict_animal_early_group0,
    # least_expressed_label_dict_animal_late_group0,
    # least_expressed_label_dict_animal_early_group1,
    # least_expressed_label_dict_animal_late_group1, ymin=0.0, ymax=0.55,
    # group_labels=("Cell Type 0", "Cell Type 1"),
    # colors=(colors_dict["Least_0"], colors_dict["Least_1"]),
    # ax=axs[1,2],
    # title="Least expressed: Early vs Late", Most=False)

    ax=axs[2,2]
    ax.hist(np.array(elbow_kmeans_per_cell_list_group_0_list), bins=[2.5,3.5,4.5,5.5], color='red')
    ax.set_xticks([3,4,5])
    ax.set_xlim(2.5,5.5)
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Number of Trial Type Clusters", fontsize=title_fs-1)
    ax.set_ylabel("Number of Cells", fontsize=title_fs-1)

    ax=axs[3,2]
    ax.hist(np.array(elbow_kmeans_per_cell_list_group_1_list), bins=[2.5,3.5,4.5,5.5], color='purple')
    ax.set_xticks([3,4,5])
    ax.set_xlim(2.5,5.5)
    ax.set_title("Cell Type 1")
    ax.set_xlabel("Number of Trial Type Clusters")
    ax.set_ylabel("Number of Cells")

    
    means_list_early_0 = np.mean(argmax_amp_list_early_0)
    sems_list_early_0 = sem(argmax_amp_list_early_0)

    means_list_late_0 = np.mean(argmax_amp_list_late_0)
    sems_list_late_0 = sem(argmax_amp_list_late_0)


    means_list_early_0min = np.mean(argmin_amp_list_early_0)
    sems_list_early_0min = sem(argmin_amp_list_early_0)

    means_list_late_0min = np.mean(argmin_amp_list_late_0)
    sems_list_late_0min = sem(argmin_amp_list_late_0)

    means_list_early_1 = np.mean(argmax_amp_list_early_1)
    sems_list_early_1 = sem(argmax_amp_list_early_1)

    means_list_late_1 = np.mean(argmax_amp_list_late_1)
    sems_list_late_1 = sem(argmax_amp_list_late_1)


    means_list_early_1min = np.mean(argmin_amp_list_early_1)
    sems_list_early_1min = sem(argmin_amp_list_early_1)

    means_list_late_1min = np.mean(argmin_amp_list_late_1)
    sems_list_late_1min = sem(argmin_amp_list_late_1)

    def amps_by_pos(locs, amps, n_pos=50):
        """
        locs: array-like of ints in [0, n_pos-1]
        amps: same length floats
        returns:
        means[n_pos], sems[n_pos], counts[n_pos]
        """
        locs = np.asarray(locs, int)
        amps = np.asarray(amps, float)

        means = np.full(n_pos, np.nan)
        sems  = np.full(n_pos, np.nan)
        counts = np.zeros(n_pos, int)

        for b in range(n_pos):
            vals = amps[locs == b]
            counts[b] = len(vals)
            if counts[b] > 0:
                means[b] = np.nanmean(vals)
                if counts[b] > 1:
                    sems[b] = np.nanstd(vals, ddof=1) / np.sqrt(counts[b])
                else:
                    sems[b] = 0.0  # or np.nan if you prefer

        return means, sems, counts
    

    # Example: Cell type 0, max early amplitude by position
    means_list_early_0, sems_list_early_0, n0 = amps_by_pos(everything_dict0["argmax_array_early"],everything_dict0["max_amp_array_early"])
    means_list_late_0, sems_list_late_0, n0 = amps_by_pos(everything_dict0["argmax_array_late"],everything_dict0["max_amp_array_late"])
    means_list_early_0, sems_list_early_0, n0 = amps_by_pos(everything_dict0["argmax_array_early"],everything_dict0["max_amp_array_early"])
    means_list_late_0, sems_list_late_0, n0 = amps_by_pos(everything_dict0["argmax_array_late"],everything_dict0["max_amp_array_late"])
    means_list_early_1, sems_list_early_1, n1 = amps_by_pos(everything_dict1["argmax_array_early"],everything_dict1["max_amp_array_early"])
    means_list_late_1, sems_list_late_1, n1 = amps_by_pos(everything_dict1["argmax_array_late"],everything_dict1["max_amp_array_late"])
    means_list_early_1, sems_list_early_1, n1 = amps_by_pos(everything_dict1["argmax_array_early"],everything_dict1["max_amp_array_early"])
    means_list_late_1, sems_list_late_1, n1 = amps_by_pos(everything_dict1["argmax_array_late"],everything_dict1["max_amp_array_late"])



    means_list_early_0min, sems_list_early_0min, n0 = amps_by_pos(everything_dict0["argmin_array_early"],everything_dict0["min_amp_array_early"])
    means_list_late_0min, sems_list_late_0min, n0 = amps_by_pos(everything_dict0["argmin_array_late"],everything_dict0["min_amp_array_late"])
    means_list_early_0min, sems_list_early_0min, n0 = amps_by_pos(everything_dict0["argmin_array_early"],everything_dict0["min_amp_array_early"])
    means_list_late_0min, sems_list_late_0min, n0 = amps_by_pos(everything_dict0["argmin_array_late"],everything_dict0["min_amp_array_late"])
    means_list_early_1min, sems_list_early_1min, n1 = amps_by_pos(everything_dict1["argmin_array_early"],everything_dict1["min_amp_array_early"])
    means_list_late_1min, sems_list_late_1min, n1 = amps_by_pos(everything_dict1["argmin_array_late"],everything_dict1["min_amp_array_late"])
    means_list_early_1min, sems_list_early_1min, n1 = amps_by_pos(everything_dict1["argmin_array_early"],everything_dict1["min_amp_array_early"])
    means_list_late_1min, sems_list_late_1min, n1 = amps_by_pos(everything_dict1["argmin_array_late"],everything_dict1["min_amp_array_late"])


        # --- rebin to 10 coarse pos bins (5 original bins per coarse bin) ---
    x10_0_e_max, m10_0_e_max, s10_0_e_max = rebin_means_sems(means_list_early_0,  sems_list_early_0)
    _,         m10_0_l_max, s10_0_l_max   = rebin_means_sems(means_list_late_0,   sems_list_late_0)

    _,         m10_0_e_min, s10_0_e_min   = rebin_means_sems(means_list_early_0min, sems_list_early_0min)
    _,         m10_0_l_min, s10_0_l_min   = rebin_means_sems(means_list_late_0min,  sems_list_late_0min)

    x10_1_e_max, m10_1_e_max, s10_1_e_max = rebin_means_sems(means_list_early_1,  sems_list_early_1)
    _,         m10_1_l_max, s10_1_l_max   = rebin_means_sems(means_list_late_1,   sems_list_late_1)

    _,         m10_1_e_min, s10_1_e_min   = rebin_means_sems(means_list_early_1min, sems_list_early_1min)
    _,         m10_1_l_min, s10_1_l_min   = rebin_means_sems(means_list_late_1min,  sems_list_late_1min)


    axs[1,3].errorbar(x10_0_e_max, m10_0_e_max, yerr=s10_0_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_max, yerr=s10_0_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_e_min, yerr=s10_0_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_min, yerr=s10_0_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[1,3].set_xlabel("Coarse position bin")
    axs[1,3].set_ylabel("dF/F amplitude")
    axs[1,3].set_title("Cell Type 0 Max/Min Amplitude")
    axs[1,3].legend()

   
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_max, yerr=s10_1_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_max, yerr=s10_1_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_min, yerr=s10_1_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_min, yerr=s10_1_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[3,3].set_xlabel("Coarse position bin")
    axs[3,3].set_ylabel("dF/F amplitude")
    axs[3,3].set_title("Cell Type 1 Max/Min Amplitude")
    axs[3,3].legend() 


                 
    # save_path = "/Users/michaelfinch/CA1_interneuron_model/misc/some_things_dict.pkl"

    # with open(save_path, 'wb') as f:
    #     pickle.dump(some_things_dict, f)

    plt.tight_layout()
    plt.show()

    some_things_dict0 = {"overall_activity_list_all":everything_dict0["overall_activity_list_all"],
                    "cells_for_cluster_list":everything_dict0["cells_for_cluster_list"],
                    "expression_list_early_overall" :everything_dict0["expression_list_early_overall"],
                        "expression_list_late_overall" :everything_dict0["expression_list_late_overall"],
                        "activity_list_early_overall" :everything_dict0["activity_list_early_overall"],
                        "activity_list_late_overall" :everything_dict0["activity_list_late_overall"],
                    
                    }
    
    some_things_dict1 = {"overall_activity_list_all":everything_dict1["overall_activity_list_all"],
                "cells_for_cluster_list":everything_dict1["cells_for_cluster_list"],
                "expression_list_early_overall" :everything_dict1["expression_list_early_overall"],
                    "expression_list_late_overall" :everything_dict1["expression_list_late_overall"],
                    "activity_list_early_overall" :everything_dict1["activity_list_early_overall"],
                    "activity_list_late_overall" :everything_dict1["activity_list_late_overall"],
                
                }


    mse_list0 = trial_types_for_given_k(some_things_dict0, k=4, title="Cell Type 0", ymin_act=-0.75, ymax_act=3.75, ymin_percent=0, ymax_percent=0.425)
    mse_list1 = trial_types_for_given_k(some_things_dict1, k=3, title="Cell Type 1", ymin_act=-1.0, ymax_act=7.0, ymin_percent=-0.025, ymax_percent=0.45)
    


    





@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
@click.option('--use_first_or_only/--use_mixed_track', default=True, help="Use the Final NDNF data")
@click.option('--use_all/--use_some', default=True, help="Use the Final NDNF data")

def cli(use_fixed_track, use_first_or_only, use_all):
    run(use_fixed_track, use_first_or_only, use_all, which_celltype="NDNF")

if __name__ == "__main__":
    cli()










    # argmax_list_early_0, argmin_list_early_0, argmax_amp_list_early_0, argmin_amp_list_early_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True)
    # argmax_list_late_0, argmin_list_late_0, argmax_amp_list_late_0, argmin_amp_list_late_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=False)

    # argmax_list_early_1, argmin_list_early_1, argmax_amp_list_early_1, argmin_amp_list_early_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=True)
    # argmax_list_late_1, argmin_list_late_1, argmax_amp_list_late_1, argmin_amp_list_late_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=False)   
    
    # argmax_list_early_0, argmin_list_early_0, argmax_amp_list_early_0, argmin_amp_list_early_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True)
    # argmax_list_late_0, argmin_list_late_0, argmax_amp_list_late_0, argmin_amp_list_late_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=False)

    # argmax_list_early_1, argmin_list_early_1, argmax_amp_list_early_1, argmin_amp_list_early_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=True)
    # argmax_list_late_1, argmin_list_late_1, argmax_amp_list_late_1, argmin_amp_list_late_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=False)    

    # def mean_and_sem(argmax_amp_list):
    #     means_list = []
    #     sems_list = []

    #     for i in range(len(argmax_amp_list)):
    #         pos_bin_vals = argmax_amp_list[i]
    #         if len(pos_bin_vals)>1:
    #             means_list.append(np.mean(pos_bin_vals))
    #             sems_list.append(sem(pos_bin_vals))
    #         elif len(pos_bin_vals)==1:
    #             means_list.append(pos_bin_vals[0])
    #             sems_list.append(0.0)
    #         else:
    #             means_list.append(np.nan)
    #             sems_list.append(np.nan)

    #     return means_list, sems_list

    # means_list_early_0, sems_list_early_0 = mean_and_sem(argmax_amp_list_early_0)
    # means_list_late_0, sems_list_late_0 = mean_and_sem(argmax_amp_list_late_0)
    # means_list_early_1, sems_list_early_1 = mean_and_sem(argmax_amp_list_early_1)
    # means_list_late_1, sems_list_late_1 = mean_and_sem(argmax_amp_list_late_1)

    # means_list_early_0min, sems_list_early_0min = mean_and_sem(argmin_amp_list_early_0)
    # means_list_late_0min, sems_list_late_0min = mean_and_sem(argmin_amp_list_late_0)
    # means_list_early_1min, sems_list_early_1min = mean_and_sem(argmin_amp_list_early_1)
    # means_list_late_1min, sems_list_late_1min = mean_and_sem(argmin_amp_list_late_1)


    
    #     # --- rebin to 10 coarse pos bins (5 original bins per coarse bin) ---
    # x10_0_e_max, m10_0_e_max, s10_0_e_max = rebin_means_sems(means_list_early_0,  sems_list_early_0)
    # _,         m10_0_l_max, s10_0_l_max   = rebin_means_sems(means_list_late_0,   sems_list_late_0)

    # _,         m10_0_e_min, s10_0_e_min   = rebin_means_sems(means_list_early_0min, sems_list_early_0min)
    # _,         m10_0_l_min, s10_0_l_min   = rebin_means_sems(means_list_late_0min,  sems_list_late_0min)

    # x10_1_e_max, m10_1_e_max, s10_1_e_max = rebin_means_sems(means_list_early_1,  sems_list_early_1)
    # _,         m10_1_l_max, s10_1_l_max   = rebin_means_sems(means_list_late_1,   sems_list_late_1)

    # _,         m10_1_e_min, s10_1_e_min   = rebin_means_sems(means_list_early_1min, sems_list_early_1min)
    # _,         m10_1_l_min, s10_1_l_min   = rebin_means_sems(means_list_late_1min,  sems_list_late_1min)





    # axs[1,3].errorbar(x10_0_e_max, m10_0_e_max, yerr=s10_0_e_max,
    #                 label='Early Max', capsize=3, marker='o')
    # axs[1,3].errorbar(x10_0_e_max, m10_0_l_max, yerr=s10_0_l_max,
    #                 label='Late Max', capsize=3, marker='o')
    # axs[1,3].errorbar(x10_0_e_max, m10_0_e_min, yerr=s10_0_e_min,
    #                 label='Early Min', capsize=3, marker='o')
    # axs[1,3].errorbar(x10_0_e_max, m10_0_l_min, yerr=s10_0_l_min,
    #                 label='Late Min', capsize=3, marker='o')
    # axs[1,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    # axs[1,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    # axs[1,3].set_title("Cell Type 0 Max/Min Amplitude", fontsize=title_fs)
    # axs[1,3].legend(fontsize=title_fs-3)

   
    # axs[3,3].errorbar(x10_1_e_max, m10_1_e_max, yerr=s10_1_e_max,
    #                 label='Early Max', capsize=3, marker='o')
    # axs[3,3].errorbar(x10_1_e_max, m10_1_l_max, yerr=s10_1_l_max,
    #                 label='Late Max', capsize=3, marker='o')
    # axs[3,3].errorbar(x10_1_e_max, m10_1_e_min, yerr=s10_1_e_min,
    #                 label='Early Min', capsize=3, marker='o')
    # axs[3,3].errorbar(x10_1_e_max, m10_1_l_min, yerr=s10_1_l_min,
    #                 label='Late Min', capsize=3, marker='o')
    # axs[3,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    # axs[3,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    # axs[3,3].set_title("Cell Type 1 Max/Min Amplitude", fontsize=title_fs)
    # axs[3,3].legend(fontsize=title_fs-3) 
