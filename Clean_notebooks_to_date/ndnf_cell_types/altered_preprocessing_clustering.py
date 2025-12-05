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





def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict, num_clusters=8, reassign_clusters=False, x00=True, umap=True, contiguous=True, ranks=20):
    # tensor_list_by_animal_all_SST = []
    # for animal in residual_activity_dict:
    #     neural_data = ut.get_animal_neural_tensor(residual_activity_dict, animal=animal)
    #     neural_data_tensor = torch.tensor(neural_data)
    #     # Normalize per cell
    #     for i in range(neural_data_tensor.shape[1]):
    #         cell = neural_data_tensor[:, i, :]
    #         min_val = cell.min()
    #         max_val = cell.max()
    #         neural_data_tensor[:, i, :] = (cell - min_val) / (max_val - min_val + 1e-8)
    #     tensor_list_by_animal_all_SST.append(neural_data_tensor)


    internals_per_animal_dict_EC_animal_x00_regkmean = {}
    
    for idx, animal in enumerate(residual_activity_dict):
        internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}
        for idt, cell in enumerate(residual_activity_dict[animal]):

            cell_data = residual_activity_dict[animal][cell].T
            cell_data = ((cell_data-np.min(cell_data)) / np.max(cell_data) - np.min(cell_data))
            cell_data_3d = np.expand_dims(cell_data, axis=1)
            cell_data_3d = torch.from_numpy(cell_data_3d)
            cell_model = NDNF_fixed_model_dict[idx][idt]


    # internals_per_animal_dict_EC_animal_x00_regkmean = {}
    # for animal in animal_EC_model_ranks20_contig_x00[ranks]:
    #     animal_model = animal_EC_model_ranks20_contig_x00[ranks][animal][0]
        # tensor_for_animal = tensor_list_by_animal_all_SST[animal]
            internals_dict = get_animal_model_reconstruction_dict_mod(cell_model, cell_data_3d, max_clusters=num_clusters, display=False, reassign_small_clusters=reassign_clusters, x00=x00, use_umap=umap, use_breakpoints=contiguous)

            internals_per_animal_dict_EC_animal_x00_regkmean_cell[idt] = internals_dict
        
        internals_per_animal_dict_EC_animal_x00_regkmean[idx] = internals_per_animal_dict_EC_animal_x00_regkmean_cell

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



def get_animal_clean_dict_activity(filepath):

    use_final=True
    with h5py.File(filepath, 'r') as f:
        if use_final:
            animal_group = f['animals']
        else:
            animal_group = f['animal']

        shiftR_refs = animal_group['ShiftR'][:]

        shiftRunning_refs = animal_group['ShiftRunning'][:]

        animal_clean_dict_activity = {}


        animal_trials_original = []
        animal_trials_clean = []

        count = 0

        animal_vel_dict = {}

        trials_to_remove_local = []
        for animal_idx in range(len(shiftR_refs)):
            delta_f = np.array(f[shiftR_refs[animal_idx][0]])

            animal_trials_original.append(delta_f.shape[1])


            # nan_trials = np.any(np.isnan(delta_f), axis=(0, 2))

            # delta_f_clean = delta_f[:, ~nan_trials, :]

            # delta_f_clean = delta_f

            vel = f[shiftRunning_refs[animal_idx][0]]

            vel = np.array(vel).T

            # vel_nan_mask = np.isnan(vel)

            # plt.imshow(vel_nan_mask, aspect='auto')
            # plt.title(f"animal_idx {animal_idx} nans={np.any(vel_nan_mask)}")
            # plt.show()


            vel_clean = np.empty(vel.shape)

        
            trials_to_remove_list = []

            delta_f_clean = np.empty(delta_f.shape)


            # print(f"delta_f.shape {delta_f.shape}")

            
            if animal_idx == 22:
                valid_cells = range(1, delta_f.shape[0])
            else:
                valid_cells = range(delta_f.shape[0])

            for cell in valid_cells: #range(delta_f.shape[0]):
                
            
                cell_data = delta_f[cell,:,:].T

                # nan_mask = np.isnan(cell_data)

                # cell_list_num_trials.append(np.sum(nan_mask, axis=0))


                for trial in range(cell_data.shape[1]):

                    trial_data = cell_data[:, trial]

                    vel_data_trial = vel[:,trial]

                    if np.any(np.isnan(trial_data)) or np.any(np.isnan(vel_data_trial)):

                        has_5_nans = has_run_of_n_nans(trial_data, n=5)

                        has_5_nans_vel = has_run_of_n_nans(vel_data_trial, n=5)
                        
                        if has_5_nans or has_5_nans_vel:                            
                            if count == 105:
                                trials_to_remove_local.append(trial)

                            

                            # plt.plot(trial_data)
                            # plt.title(f"Before More than 5 nans cell={count} trial={trial}")
                            # plt.show()

                            # plt.plot(trial_data)
                            # plt.title(f"After More than 5 nans cell={count} trial={trial}")
                            # plt.show()

                            if trial not in trials_to_remove_list:
                                trials_to_remove_list.append(trial)
                        else:

                            # if count==30:

                            #     plt.plot(trial_data)
                            #     plt.title(f"Before Less than 5 nans cell={count} trial={trial}")
                            #     plt.show()

                            trial_data = interp_nans_1d(trial_data)
                            delta_f_clean[cell, trial, :] = trial_data

                            trial_data_vel = interp_nans_1d(vel_data_trial)
                            vel_clean[:,trial] = trial_data_vel




                            # if count==30:
                            #     plt.plot(trial_data)
                            #     plt.title(f"After Less than 5 nans cell={count} trial={trial}")
                            #     plt.show()

                    else:
                        delta_f_clean[cell, trial, :] = trial_data

                        vel_clean[:,trial] = trial_data_vel = vel_data_trial


                count+=1




            trials_to_remove_array = np.array(trials_to_remove_list) 

            if len(trials_to_remove_array) !=0:

                # print(f"trials_to_remove_array {trials_to_remove_array}")
                mask = np.ones(cell_data.shape[1], dtype=bool)
                mask[trials_to_remove_array] = False
                delta_f_clean = delta_f_clean[:, mask,:]

                vel_clean = vel_clean[:, mask]

            animal_trials_clean.append(delta_f_clean.shape[1])

            cell_dict = {}


            for cell in valid_cells:#range(delta_f.shape[0]):

                cell_data = delta_f_clean[cell,:,:]

                cell_data = (cell_data - np.mean(cell_data)) / np.std(cell_data)

                cell_dict[f"cell_{cell+1}"] = cell_data.T
                

            animal_clean_dict_activity[f"animal_{animal_idx+1}"] = cell_dict
            animal_vel_dict[f"animal_{animal_idx+1}"] = {"Velocity":vel_clean}

        return animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local




def get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest):
    min_val = 10000

    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            data = fixed_activity_dict_NDNF_newest[animal][cell]
            if data.shape[1] < min_val:
                min_val = data.shape[1]

    data_truncated_list = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            data_truncated = fixed_activity_dict_NDNF_newest[animal][cell][:,:min_val]
            data_truncated_list.append(data_truncated)


    data_truncated_array = np.array(data_truncated_list)

    return data_truncated_array, min_val

def get_mean_behav_factor_per_cell(residual_activity_dict, factors_dict, min_num_trials, factor:str):

    data_list = []
    for animal in residual_activity_dict:
        for cell in residual_activity_dict[animal]:
            data_list.append(factors_dict[animal][factor][:, :min_num_trials])

    return data_list



def load_data_regular(file_path=r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM", name="NDNFanalC", new_NDNF=True, use_final=False):
    file_path = file_path
    filename = name
    filepath = os.path.join(file_path, "datasets", filename + ".mat")

    activity_dict, factors_dict = preprocess_data2(filepath, normalize=True, new_NDNF=new_NDNF, use_final=use_final)

    filtered_factors_dict = subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"])

    GLM_params, double_predicted_activity_dict_NDNF_new = fit_GLM_population(filtered_factors_dict, activity_dict, quintile=None, regression='linear')
    double_residual_activity_dict_NDNF_new = get_residual_activity_dict(activity_dict, double_predicted_activity_dict_NDNF_new)

    return GLM_params, activity_dict, double_predicted_activity_dict_NDNF_new, factors_dict, filtered_factors_dict, double_residual_activity_dict_NDNF_new


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

def reshape_contig_dict(cued_contig_dict, NDNF_cued_model_dict_clean):
    # Match the old structure: outer key is rank (20)
    cued_contig_final = {20: {}}

    for animal in cued_contig_dict:  # animal index: 0,1,2,...
        cued_contig_final[20][animal] = {}

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


def get_lists_out_of_dicts(fixed_TT_data, fixed_activity_dict_NDNF_newest, cp_dict_NDNF):
    TT_list = []
    for animal in fixed_TT_data:
        for cell in fixed_TT_data[animal]:

            TT_list.append(fixed_TT_data[animal][cell][1][f"cell_{cell}"])


    NDNF_activity_list = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            NDNF_activity_list.append(fixed_activity_dict_NDNF_newest[animal][cell])

    print(len(NDNF_activity_list)) 

    cp_list_NDNF = []

    for animal in cp_dict_NDNF:
        for cell in cp_dict_NDNF[animal]:
            cp_list_NDNF.append(cp_dict_NDNF[animal][cell])

    print(len(cp_list_NDNF)) 

    return TT_list, NDNF_activity_list, cp_list_NDNF



def get_most_expressed_cluster(TT_list, activity_list, cp_list_NDNF, early_late_none="early", to_include=None, most_expressed=True):

    print(f"len(TT_list){len(TT_list)} len(activity_list){len(activity_list)} len(cp_list_NDNF){len(cp_list_NDNF)}")
    
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

        elbow_kmeans_array[j] = elbow_kmeans
        
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




def run(use_fixed_track):


    title_fs = 9


    filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat'

    animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local = get_animal_clean_dict_activity(filepath)

    GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

    residual_activity_dict_NDNF_new = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)



    
    
    ###### cell by cell slice tca models

    


   



    if use_fixed_track:

        clean_resid_activity_dict_NDNF_newest = {}

        clean_velocity_dict_NDNF_newest = {}

        for idx, animal in enumerate(residual_activity_dict_NDNF_new):
            if 14 < idx < 29:
                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]


        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_fixed_model_dict_clean.pkl', 'rb') as f:
            NDNF_model_dict_clean  = pickle.load(f)


        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_fixed_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)
     

        
        
    else:

        clean_resid_activity_dict_NDNF_newest = {}

        clean_velocity_dict_NDNF_newest = {}

        for idx, animal in enumerate(residual_activity_dict_NDNF_new):
            if idx > 28:
                clean_resid_activity_dict_NDNF_newest[f"animal_{idx+1}"] = residual_activity_dict_NDNF_new[animal]
                clean_velocity_dict_NDNF_newest[f"animal_{idx+1}"] = animal_vel_dict[animal]
            

        with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_cued_model_dict_clean.pkl', 'rb') as f:
            NDNF_model_dict_clean = pickle.load(f)

        save_path = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_cells_truncated_cued_model.pkl'
        with open(save_path, 'rb') as f:
            sliceTCA_model = pickle.load(f)

            
    


    
    reassigned_dict = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=8, reassign_clusters=True, x00=True, umap=False, contiguous=False, ranks=20)

    contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)

    

    contig_dict = reshape_contig_dict(contig_dict_all_cell_tca, NDNF_model_dict_clean)



    cp_dict_NDNF = get_cp_dict(contig_dict)

    labels_cells_dict_all_K_NDNF = get_labels_all_different_Ks_single(sliceTCA_model, which_vectors=1)

    cell_type_labels = labels_cells_dict_all_K_NDNF[2]

    cells_group_0 = np.where(cell_type_labels==0)[0]
    cells_group_1 = np.where(cell_type_labels==1)[0]

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

    
    # if use_new_data:
    #     if use_fixed_track:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_new.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_new.pkl')
    #     else:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_new.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_new.pkl')
    # else:
    #     if use_fixed_track:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_old.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_fix_old.pkl')
    #     else:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_old.pkl', 'wb') as f:
    #             pickle.dump(cell_type_labels, f)
    #             print('/Users/michaelfinch/CA1-interneuron-GLM/datasets/save_labels_cue_old.pkl')





    # if use_new_data:

    #     fixed_TT_data = {}

    #     if use_fixed_track:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_fixed_reassign_models_NDNF.pkl', 'rb') as f:
    #             reassigned_dict = pickle.load(f)

    #         trial_type_data = reassigned_dict[20]
    #         for idx, animal in enumerate(trial_type_data):
    #             fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    #     else:
    #         with open('/Users/michaelfinch/CA1-interneuron-GLM/datasets/all_new_cued_reassign_models_NDNF.pkl', 'rb') as f:
    #             reassigned_dict = pickle.load(f)

    #         trial_type_data = reassigned_dict[20]
    #         for idx, animal in enumerate(trial_type_data):
    #             fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    # else:
    #     mse_dir ='/Users/michaelfinch/CA1-interneuron-GLM/datasets/real_final_NDNF_model_ranks20_regkmean_reassign_x00_cell'
    #     reassigned_dict = get_model_data_per_cell(mse_dir)

    

        print(f"reassigned_dict.keys() {reassigned_dict.keys()}")
        trial_type_data = reassigned_dict[20]
        print(f"trial_type_data.keys() {trial_type_data.keys()}")
        fixed_TT_data = {}
        for idx, animal in enumerate(trial_type_data):
            if use_fixed_track:
                if first_an_idx < idx < last_an_idx:
                    fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]
            else:
                if idx > last_an_idx-1:
                    fixed_TT_data[f"animal_{idx+1}"] = trial_type_data[animal]

    print(f"cp_dict_NDNF {cp_dict_NDNF}")
            

    TT_list, NDNF_activity_list, cp_list_NDNF = get_lists_out_of_dicts(fixed_TT_data, clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF)





    # most_expressed_label_dict_animal_early, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", most_expressed=most_expressed)
    # most_expressed_label_dict_animal_late, elbow_kmeans_array = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", most_expressed=most_expressed)


    most_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=True)
    most_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=True)


    most_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=True)
    most_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=True)



    most_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=True)
    most_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_most = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=True)


    least_expressed_label_dict_animal_all_group0, elbow_kmeans_array_group0_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_0, most_expressed=False)
    least_expressed_label_dict_animal_all_group1, elbow_kmeans_array_group1_least = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="none", to_include=cells_group_1, most_expressed=False)

    
    
    least_expressed_label_dict_animal_early_group0, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_0, most_expressed=False)
    least_expressed_label_dict_animal_late_group0, elbow_kmeans_array_group0 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_0, most_expressed=False)


    least_expressed_label_dict_animal_early_group1, _ = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="early", to_include=cells_group_1, most_expressed=False)
    least_expressed_label_dict_animal_late_group1, elbow_kmeans_array_group1 = get_most_expressed_cluster(TT_list, NDNF_activity_list, cp_list_NDNF, early_late_none="late", to_include=cells_group_1, most_expressed=False)




    # print(f"len(elbow_kmeans_array_group0_most) {len(elbow_kmeans_array_group0_most)}")
    # print(f"len(TT_list) {TT_list[0]['labels_dict'].keys()}")
    # print(f"len(TT_list) {TT_list[0]['indices_for_cluster_number']['clusters_chosen_2']}")

    
    argmax_list_early_0, argmin_list_early_0, argmax_amp_list_early_0, argmin_amp_list_early_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=True)
    argmax_list_late_0, argmin_list_late_0, argmax_amp_list_late_0, argmin_amp_list_late_0 = get_argmin_argmax_lists(cells_group_0, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group0_most, use_early=False)

    argmax_list_early_1, argmin_list_early_1, argmax_amp_list_early_1, argmin_amp_list_early_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=True)
    argmax_list_late_1, argmin_list_late_1, argmax_amp_list_late_1, argmin_amp_list_late_1 = get_argmin_argmax_lists(cells_group_1, cp_list_NDNF, NDNF_activity_list, TT_list, elbow_kmeans_array_group1_most, use_early=False)    

    def mean_and_sem(argmax_amp_list):
        means_list = []
        sems_list = []

        for i in range(len(argmax_amp_list)):
            pos_bin_vals = argmax_amp_list[i]
            if len(pos_bin_vals)>1:
                means_list.append(np.mean(pos_bin_vals))
                sems_list.append(sem(pos_bin_vals))
            elif len(pos_bin_vals)==1:
                means_list.append(pos_bin_vals[0])
                sems_list.append(0.0)
            else:
                means_list.append(np.nan)
                sems_list.append(np.nan)

        return means_list, sems_list

    means_list_early_0, sems_list_early_0 = mean_and_sem(argmax_amp_list_early_0)
    means_list_late_0, sems_list_late_0 = mean_and_sem(argmax_amp_list_late_0)
    means_list_early_1, sems_list_early_1 = mean_and_sem(argmax_amp_list_early_1)
    means_list_late_1, sems_list_late_1 = mean_and_sem(argmax_amp_list_late_1)

    means_list_early_0min, sems_list_early_0min = mean_and_sem(argmin_amp_list_early_0)
    means_list_late_0min, sems_list_late_0min = mean_and_sem(argmin_amp_list_late_0)
    means_list_early_1min, sems_list_early_1min = mean_and_sem(argmin_amp_list_early_1)
    means_list_late_1min, sems_list_late_1min = mean_and_sem(argmin_amp_list_late_1)


    
        # --- rebin to 10 coarse pos bins (5 original bins per coarse bin) ---
    x10_0_e_max, m10_0_e_max, s10_0_e_max = rebin_means_sems(means_list_early_0,  sems_list_early_0)
    _,         m10_0_l_max, s10_0_l_max   = rebin_means_sems(means_list_late_0,   sems_list_late_0)

    _,         m10_0_e_min, s10_0_e_min   = rebin_means_sems(means_list_early_0min, sems_list_early_0min)
    _,         m10_0_l_min, s10_0_l_min   = rebin_means_sems(means_list_late_0min,  sems_list_late_0min)

    x10_1_e_max, m10_1_e_max, s10_1_e_max = rebin_means_sems(means_list_early_1,  sems_list_early_1)
    _,         m10_1_l_max, s10_1_l_max   = rebin_means_sems(means_list_late_1,   sems_list_late_1)

    _,         m10_1_e_min, s10_1_e_min   = rebin_means_sems(means_list_early_1min, sems_list_early_1min)
    _,         m10_1_l_min, s10_1_l_min   = rebin_means_sems(means_list_late_1min,  sems_list_late_1min)





    fig, axs = plt.subplots(4, 4, figsize=(10, 12))  



    axs[1,3].errorbar(x10_0_e_max, m10_0_e_max, yerr=s10_0_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_max, yerr=s10_0_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_e_min, yerr=s10_0_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[1,3].errorbar(x10_0_e_max, m10_0_l_min, yerr=s10_0_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[1,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    axs[1,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    axs[1,3].set_title("Cell Type 0 Max/Min Amplitude", fontsize=title_fs)
    axs[1,3].legend(fontsize=title_fs-2)

   
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_max, yerr=s10_1_e_max,
                    label='Early Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_max, yerr=s10_1_l_max,
                    label='Late Max', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_e_min, yerr=s10_1_e_min,
                    label='Early Min', capsize=3, marker='o')
    axs[3,3].errorbar(x10_1_e_max, m10_1_l_min, yerr=s10_1_l_min,
                    label='Late Min', capsize=3, marker='o')
    axs[3,3].set_xlabel("Coarse position bin", fontsize=title_fs-1)
    axs[3,3].set_ylabel("dF/F amplitude", fontsize=title_fs-1)
    axs[3,3].set_title("Cell Type 1 Max/Min Amplitude", fontsize=title_fs)
    axs[3,3].legend(fontsize=title_fs-2)

 


    plot_butterfly_hist(title_fs, argmax_list_early_0, argmax_list_late_0, argmin_list_early_0, argmin_list_late_0, ax=axs[0,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 0")
    plot_butterfly_hist(title_fs, argmax_list_early_1, argmax_list_late_1, argmin_list_early_1, argmin_list_late_1, ax=axs[2,3], colors_list = ["blue", "orange", "green", "red"], title="Cell Type 1")



    if use_new_data:
        if use_fixed_track:
            fig.suptitle("New Data Fixed Track")
        else:
            fig.suptitle("New Data Cued Track")
    else:
        if use_fixed_track:
            fig.suptitle("Old Data Fixed Track")
        else:
            fig.suptitle("Old Data Cued Track")


    colors_dict = {"Most_0":"orange",
                   "Least_0":"magenta",
                   "Most_1":"purple",
                   "Least_1":"red"}

    axs_list = [
        axs[0,0], axs[0,1],
        axs[1,0], axs[1,1],
        axs[2,0], axs[2,1],
        axs[3,0], axs[3,1],]
    
    plot_no_learn_cell_types(title_fs,
        most_expressed_label_dict_animal_all_group0,
        most_expressed_label_dict_animal_all_group1,
        least_expressed_label_dict_animal_all_group0,
        least_expressed_label_dict_animal_all_group1,
        group=None,axs_list=axs_list, color_dict=colors_dict)

    eval_proportion_two_groups(title_fs,
    most_expressed_label_dict_animal_early_group0,
    most_expressed_label_dict_animal_late_group0,
    most_expressed_label_dict_animal_early_group1,
    most_expressed_label_dict_animal_late_group1,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=(colors_dict["Most_0"], colors_dict["Most_1"]),
    ax=axs[0,2],
    title="Most expressed: Early vs Late", Most=True)

    eval_proportion_two_groups(title_fs,
    least_expressed_label_dict_animal_early_group0,
    least_expressed_label_dict_animal_late_group0,
    least_expressed_label_dict_animal_early_group1,
    least_expressed_label_dict_animal_late_group1,
    group_labels=("Cell Type 0", "Cell Type 1"),
    colors=(colors_dict["Least_0"], colors_dict["Least_1"]),
    ax=axs[1,2],
    title="Least expressed: Early vs Late", Most=False)

    ax=axs[2,2]
    ax.hist(elbow_kmeans_array_group0_most, bins=[1.5,2.5,3.5,4.5,5.5], color='red')
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Cell Type 0", fontsize=title_fs)
    ax.set_xlabel("Number of Trial Type Clusters", fontsize=title_fs-1)
    ax.set_ylabel("Number of Cells", fontsize=title_fs-1)

    ax=axs[3,2]
    ax.hist(elbow_kmeans_array_group1_most, bins=[1.5,2.5,3.5,4.5,5.5], color='purple')
    ax.set_xticks([2,3,4,5])
    ax.set_xlim(1.5,5.5)
    ax.set_title("Cell Type 1", fontsize=title_fs)
    ax.set_xlabel("Number of Trial Type Clusters", fontsize=title_fs-1)
    ax.set_ylabel("Number of Cells", fontsize=title_fs-1)


    fixed_residual_list = []

    to_plot_list = []

    min_t = 10000

    for animal in fixed_residual_activity_dict_NDNF_newest:
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            data = fixed_residual_activity_dict_NDNF_newest[animal][cell]
            data_flat = data.flatten()
            if data.shape[1] < min_t:
                min_t = data.shape[1]

    for animal in fixed_residual_activity_dict_NDNF_newest:
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            data_trunc = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,:min_t]
            to_plot_list.append(np.mean(data_trunc, axis=1))
            fixed_residual_list.append(data_trunc.flatten())

    fixed_residual_array = np.array(fixed_residual_list)
    fixed_residual_array_correct_shape = fixed_residual_array.T

    group_1_array = fixed_residual_array_correct_shape[:,cells_group_1]

    group_0_array = fixed_residual_array_correct_shape[:,cells_group_0]


    # plot_LDA1_LDA2_state_space_prepost(title_fs, group_0_array, title="Cell Type 0", ax=axs[2,3])

    # plot_LDA1_LDA2_state_space_prepost(title_fs, group_1_array, title="Cell Type 1", ax=axs[3,3])


    # plot_LDA_hist(title_fs, group_0_array, title="Cell Type 0", color_list=["blue", "orange"], ax=axs[0,3])
    # plot_LDA_hist(title_fs, group_1_array, title="Cell Type 1", color_list=["red", "green"], ax=axs[1,3])

    # lda0, lda1, early0, late0, early1, late1 = plot_LDA_hist_compare_types(title_fs,group0_array=group_0_array,group1_array=group_1_array, title="NDNF Cell Type Comparison",
    # color_dict={
    #     "g0_early": "blue",
    #     "g0_late": "green",
    #     "g1_early": "red",
    #     "g1_late": "magenta",},ax=axs[0,3])

    # print(f"len(lda0[early0]) {len(lda0[early0])} len(lda0[late0]) {len(lda0[late0])}") #, late0, early1, late1


    # def build_clusters_dict():






    plt.tight_layout()
    plt.show()






    fig, axs = plt.subplots(4,4, figsize=(20,20))


@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
# @click.option('--use_new_data/--use_old_data', default=True, help="Use the Final NDNF data")

def cli(use_fixed_track):
    run(use_fixed_track)

if __name__ == "__main__":
    cli()