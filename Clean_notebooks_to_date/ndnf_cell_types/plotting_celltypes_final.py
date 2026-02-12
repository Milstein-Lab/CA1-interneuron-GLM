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
import os
from matplotlib.lines import Line2D


plt.rcParams['axes.titlesize'] = 20       # all titles
plt.rcParams['axes.labelsize'] = 16      # x and y labels
plt.rcParams['xtick.labelsize'] = 16      # tick labels
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["legend.fontsize"] = 12
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.titlepad'] = 8.0



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


def get_mean_sem_lists(binned_data):
    mean_data_list, sem_data_list = [], []
    for bin_data in binned_data:
        n = bin_data.size
        if n == 0:
            mean_data_list.append(np.nan)  # or choose a sentinel
            sem_data_list.append(0.0)
        elif n == 1:
            mean_data_list.append(float(bin_data[0]))
            sem_data_list.append(0.0)
        else:
            mean_data_list.append(float(np.mean(bin_data)))
            sem_data_list.append(float(sem(bin_data)))  # ddof handled by scipy
    return mean_data_list, sem_data_list



def plot_clustered_data_learn(labels_full, activity_early_array, activity_array_late, K=2, title="", ax_list=None, color_list_lists=None):
    # data_good = means_dict_cluster_0x0_raw[K]["labels_loc_dict"]

    # fig, axs = plt.subplots(1,len(data_good), figsize=(4*len(data_good), 4))
    # fig.suptitle(title)

    data_good = np.unique(labels_full)

    for i in data_good:
        color_list = color_list_lists[i]
        axs = ax_list[i]
        # labels = data_good[i]
        labels = np.where(labels_full==i)[0]
        n=len(labels)
        sliced_early = activity_early_array[labels,:]
        mean_sliced_early = np.mean(sliced_early, axis=0)
        sem_sliced_early = sem(sliced_early, axis=0)

        # sliced_data_early_dict[i] = sliced_early
        sliced_late = activity_array_late[labels,:]
        mean_sliced_late = np.mean(sliced_late, axis=0)
        sem_sliced_late = sem(sliced_late, axis=0)

        # sliced_data_late_dict[i] = sliced_late
        axs.plot(mean_sliced_early, label='Early', color=color_list[0])
        axs.fill_between(range(len(mean_sliced_early)), mean_sliced_early-sem_sliced_early, mean_sliced_early+sem_sliced_early, alpha=0.2, color=color_list[0])

        axs.plot(mean_sliced_late, label="Late", color=color_list[1])
        axs.fill_between(range(len(mean_sliced_late)), mean_sliced_late-sem_sliced_late, mean_sliced_late+sem_sliced_late, alpha=0.2, color=color_list[1])
        axs.set_title(f"Cluster {i} n={n}")
        axs.set_ylabel("Z-Scored DF/F")
        axs.set_xlabel("Position bins")
        axs.legend()




def get_activity_cut_learn(fixed_residual_activity_dict_NDNF_newest, cp_dict_NDNF):
    activity_list_early = []
    activity_list_late = []

    cp_early_as_fraction = []
    cp_late_as_fraction = []

    for idx, animal in enumerate(fixed_residual_activity_dict_NDNF_newest):
        for idt, cell in enumerate(fixed_residual_activity_dict_NDNF_newest[animal]):
            data = fixed_residual_activity_dict_NDNF_newest[animal][cell]
            # print(f"animal {animal} cp_dict_NDNF.keys() {cp_dict_NDNF.keys()}")
            cp_early = cp_dict_NDNF[animal][cell][0]
            cp_late = cp_dict_NDNF[animal][cell][1]

            early_data = data[:,:cp_early]
            late_data = data[:,cp_late:]

            mean_early_data = np.mean(early_data, axis=1)
            activity_list_early.append(mean_early_data)
            mean_late_data = np.mean(late_data, axis=1)
            activity_list_late.append(mean_late_data)

            cp_early_as_fraction.append(cp_early/data.shape[1])
            cp_late_as_fraction.append(cp_late/data.shape[1])




    activity_early_array = np.array(activity_list_early)
    activity_array_late = np.array(activity_list_late)

    return activity_early_array, activity_array_late, cp_early_as_fraction, cp_late_as_fraction


# def plot_reconstructions(labels_list, fixed_activity_dict_NDNF_newest, prefix="", plot=False):

#     synthetic_mean_array = {}
#     means_dict_cluster= {}
#     for num_clusters in labels_list:
#     # for num_clusters in range(len(labels_list)):
#         data_truncated_array_NDNF, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
#         labels = labels_list[num_clusters]
#         uniq = np.unique(labels)
#         real_data_mean_array = np.empty(data_truncated_array_NDNF.shape)
#         mean_data_dict = {}

#         r_vel_dict_per_clust = {}
#         r_lick_dict_per_clust = {}

#         cell_ids_per_cluster_dict = {}

#         vel_data_sliced_dict = {}
#         lick_data_sliced_dict = {}

#         # vel_array = r_dict_vel["array_data"]
#         # lick_array = r_dict_licks["array_data"]

#         labels_loc_dict = {}

#         mean_data_list = []
#         fraction_cells_list = []
#         for i in uniq:
#             labels_loc = np.where(labels==i)[0]
#             cell_ids_per_cluster_dict[i] = labels_loc
#             fraction_cells = (len(labels_loc) / data_truncated_array_NDNF.shape[0])*100
#             fraction_cells_list.append(fraction_cells)
#             real_data_array_sliced = data_truncated_array_NDNF[labels_loc,:,:]
#             mean_real_data_array_sliced = np.mean(real_data_array_sliced, axis=0)
#             mean_data_list.append(mean_real_data_array_sliced)
#             real_data_mean_array[labels_loc,:,:] = mean_real_data_array_sliced

#             labels_loc_dict[i] = labels_loc
            
#             # r_list_vel = np.array(r_dict_vel["r_list"])

#             # r_list_licks = np.array(r_dict_licks["r_list"])

            

#             # vel_data_sliced_dict[i] = vel_array[labels_loc,:,:]

#             # lick_data_sliced_dict[i] = lick_array[labels_loc,:,:]

#             # r_vel_dict_per_clust[i] = r_list_vel[labels_loc]
#             # r_lick_dict_per_clust[i] = r_list_licks[labels_loc]



#         MSE_reco_vs_real = np.mean(np.square(data_truncated_array_NDNF-real_data_mean_array))

#         mean_data_dict = {"mean_data_list":mean_data_list,
#                         "fraction_cells":fraction_cells_list,
#                         "MSE_reco_vs_real":MSE_reco_vs_real,
#                         # "r_vel_dict_per_clust":r_vel_dict_per_clust,
#                         # "r_lick_dict_per_clust":r_lick_dict_per_clust,
#                         "cell_ids_per_cluster_dict":cell_ids_per_cluster_dict,
#                         # "vel_data_sliced_dict":vel_data_sliced_dict,
#                         # "lick_data_sliced_dict":lick_data_sliced_dict,
#                         "labels_loc_dict":labels_loc_dict}
        
#         means_dict_cluster[num_clusters] = mean_data_dict
#         synthetic_mean_array[num_clusters] = real_data_mean_array



#     data_truncated_array_NDNF, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
#     data_truncated_array_NDNF_ta = np.mean(data_truncated_array_NDNF, axis=2),
#     if plot:
#         fig, axs = plt.subplots(2,len(synthetic_mean_array), figsize=(30,8))
#         plt.suptitle(f"Reconstructed vs Real Trial Averaged Data {prefix}")
#         for num_clusters in synthetic_mean_array:
#             reconstructed_data = synthetic_mean_array[num_clusters]
#             reconstructed_data_ta = np.mean(reconstructed_data, axis=2)
#             axs[0,num_clusters-1].imshow(reconstructed_data_ta, aspect='auto')
#             axs[0,num_clusters-1].set_ylabel("Cell ID")
#             axs[0,num_clusters-1].set_title(f"Reconstructed K={num_clusters}")
#             axs[0,num_clusters-1].set_xlabel("Position Bin")
#             axs[1,num_clusters-1].imshow(data_truncated_array_NDNF_ta, aspect='auto')
#             axs[1,num_clusters-1].set_title("Real T.A. Data")
#             axs[1,num_clusters-1].set_xlabel("Position Bin")
#         plt.tight_layout()
#         plt.show()

#     return means_dict_cluster



def plot_reconstructions(labels_cells_dict_all_K_NDNF, fixed_activity_dict_NDNF_newest, r_dict_vel, r_dict_licks, prefix="", plot=False):


    synthetic_mean_array = {}
    means_dict_cluster= {}
    for num_clusters in labels_cells_dict_all_K_NDNF:
        data_truncated_array_NDNF, _ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
        print(f"data_truncated_array_NDNF.shape {data_truncated_array_NDNF.shape}")
        labels = labels_cells_dict_all_K_NDNF[num_clusters]
        # print(labels)
        uniq = np.unique(labels)
        real_data_mean_array = np.empty(data_truncated_array_NDNF.shape)
        mean_data_dict = {}

        r_vel_dict_per_clust = {}
        r_lick_dict_per_clust = {}

        cell_ids_per_cluster_dict = {}

        vel_data_sliced_dict = {}
        lick_data_sliced_dict = {}

        vel_array = r_dict_vel["array_data"]
        lick_array = r_dict_licks["array_data"]

        labels_loc_dict = {}

        mean_data_list = []
        fraction_cells_list = []
        for i in uniq:
            labels_loc = np.where(labels==i)[0]
            cell_ids_per_cluster_dict[i] = labels_loc
            fraction_cells = (len(labels_loc) / data_truncated_array_NDNF.shape[0])*100
            fraction_cells_list.append(fraction_cells)
            real_data_array_sliced = data_truncated_array_NDNF[labels_loc,:,:]
            mean_real_data_array_sliced = np.mean(real_data_array_sliced, axis=0)
            mean_data_list.append(mean_real_data_array_sliced)
            real_data_mean_array[labels_loc,:,:] = mean_real_data_array_sliced

            labels_loc_dict[i] = labels_loc
            
            r_list_vel = np.array(r_dict_vel["r_list"])

            r_list_licks = np.array(r_dict_licks["r_list"])

            

            vel_data_sliced_dict[i] = vel_array[labels_loc,:,:]

            lick_data_sliced_dict[i] = lick_array[labels_loc,:,:]

            r_vel_dict_per_clust[i] = r_list_vel[labels_loc]
            r_lick_dict_per_clust[i] = r_list_licks[labels_loc]



        MSE_reco_vs_real = np.mean(np.square(data_truncated_array_NDNF-real_data_mean_array))

        mean_data_dict = {"mean_data_list":mean_data_list,
                        "fraction_cells":fraction_cells_list,
                        "MSE_reco_vs_real":MSE_reco_vs_real,
                        "r_vel_dict_per_clust":r_vel_dict_per_clust,
                        "r_lick_dict_per_clust":r_lick_dict_per_clust,
                        "cell_ids_per_cluster_dict":cell_ids_per_cluster_dict,
                        "vel_data_sliced_dict":vel_data_sliced_dict,
                        "lick_data_sliced_dict":lick_data_sliced_dict,
                        "labels_loc_dict":labels_loc_dict}
        
        means_dict_cluster[num_clusters] = mean_data_dict
        synthetic_mean_array[num_clusters] = real_data_mean_array



    
    if plot:
        data_truncated_array_NDNF,_ = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)
        data_truncated_array_NDNF_ta = np.mean(data_truncated_array_NDNF, axis=2)
        fig, axs = plt.subplots(2,len(synthetic_mean_array), figsize=(30,8))
        plt.suptitle(f"Reconstructed vs Real Trial Averaged Data {prefix}")
        for num_clusters in synthetic_mean_array:
            reconstructed_data = synthetic_mean_array[num_clusters]
            reconstructed_data_ta = np.mean(reconstructed_data, axis=2)
            axs[0,num_clusters-1].imshow(reconstructed_data_ta, aspect='auto')
            axs[0,num_clusters-1].set_ylabel("Cell ID")
            axs[0,num_clusters-1].set_title(f"Reconstructed K={num_clusters}")
            axs[0,num_clusters-1].set_xlabel("Position Bin")
            axs[1,num_clusters-1].imshow(data_truncated_array_NDNF_ta, aspect='auto')
            axs[1,num_clusters-1].set_title("Real T.A. Data")
            axs[1,num_clusters-1].set_xlabel("Position Bin")
        plt.tight_layout()
        plt.show()

    return means_dict_cluster

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

def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict,
                      num_clusters=8, reassign_clusters=False,
                      x00=True, umap=True, contiguous=True, ranks=20):

    internals_per_animal_dict_EC_animal_x00_regkmean = {}

    # iterate over matching animal keys
    for animal_key, cell_dict in residual_activity_dict.items():
        internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}

        for idt, cell in enumerate(cell_dict):
            cell_data = cell_dict[cell].T
            cell_data = ((cell_data - np.min(cell_data)) /
                         (np.max(cell_data) - np.min(cell_data)))
            cell_data_3d = np.expand_dims(cell_data, axis=1)
            cell_data_3d = torch.from_numpy(cell_data_3d)

            # use the same animal_key for the model dict
            cell_model = NDNF_fixed_model_dict[animal_key][cell]

            internals_dict = get_animal_model_reconstruction_dict_mod(
                cell_model,
                cell_data_3d,
                max_clusters=num_clusters,
                display=False,
                reassign_small_clusters=reassign_clusters,
                x00=x00,
                use_umap=umap,
                use_breakpoints=contiguous,
            )

            internals_per_animal_dict_EC_animal_x00_regkmean_cell[cell] = internals_dict

        internals_per_animal_dict_EC_animal_x00_regkmean[animal_key] = \
            internals_per_animal_dict_EC_animal_x00_regkmean_cell

    return internals_per_animal_dict_EC_animal_x00_regkmean


# def preprocess_animal(NDNF_fixed_model_dict, residual_activity_dict, num_clusters=8, reassign_clusters=False, x00=True, umap=True, contiguous=True, ranks=20):

#     internals_per_animal_dict_EC_animal_x00_regkmean = {}
    
#     for idx, animal in enumerate(residual_activity_dict):
#         internals_per_animal_dict_EC_animal_x00_regkmean_cell = {}
#         for idt, cell in enumerate(residual_activity_dict[animal]):

#             cell_data = residual_activity_dict[animal][cell].T
#             cell_data = ((cell_data-np.min(cell_data)) / np.max(cell_data) - np.min(cell_data))
#             cell_data_3d = np.expand_dims(cell_data, axis=1)
#             cell_data_3d = torch.from_numpy(cell_data_3d)
#             cell_model = NDNF_fixed_model_dict[idx][idt]

#             internals_dict = get_animal_model_reconstruction_dict_mod(cell_model, cell_data_3d, max_clusters=num_clusters, display=False, reassign_small_clusters=reassign_clusters, x00=x00, use_umap=umap, use_breakpoints=contiguous)

#             internals_per_animal_dict_EC_animal_x00_regkmean_cell[idt] = internals_dict
        
#         internals_per_animal_dict_EC_animal_x00_regkmean[idx] = internals_per_animal_dict_EC_animal_x00_regkmean_cell

#     return internals_per_animal_dict_EC_animal_x00_regkmean


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


def get_truncated_to_min_data_array(activity_dict):
    # deterministic ordering
    animals = sorted(activity_dict.keys(), key=lambda a: int(a.split("_")[1]))

    # global min trials across all cells
    min_val = min(
        activity_dict[a][c].shape[1]
        for a in animals
        for c in sorted(activity_dict[a].keys(), key=lambda s: int(s.split("_")[1]))
    )

    data_list = []
    idx_to_key = []  # list of (animal, cell_key) in the exact stacking order

    for a in animals:
        cell_keys = sorted(activity_dict[a].keys(), key=lambda s: int(s.split("_")[1]))
        for c in cell_keys:
            data_list.append(activity_dict[a][c][:, :min_val])
            idx_to_key.append((a, c))

    return np.array(data_list), min_val, idx_to_key


# def get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest):
#     min_val = 10000

#     for animal in fixed_activity_dict_NDNF_newest:
#         for cell in fixed_activity_dict_NDNF_newest[animal]:
#             data = fixed_activity_dict_NDNF_newest[animal][cell]
#             if data.shape[1] < min_val:
#                 min_val = data.shape[1]

#     data_truncated_list = []
#     for animal in fixed_activity_dict_NDNF_newest:
#         for cell in fixed_activity_dict_NDNF_newest[animal]:
#             data_truncated = fixed_activity_dict_NDNF_newest[animal][cell][:,:min_val]
#             data_truncated_list.append(data_truncated)


#     data_truncated_array = np.array(data_truncated_list)

#     return data_truncated_array, min_val

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
    

def get_cells_per_animal_dict(fixed_activity_dict_NDNF_newest):
    cells_per_animal_dict = {}

    count=0
    for animal in fixed_activity_dict_NDNF_newest:
        per_animal_list=[]
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            per_animal_list.append(count)
            count+=1
        cells_per_animal_dict[animal] = per_animal_list

    return cells_per_animal_dict



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


# def plot_cluster_traces_by_animal(
#     labels,                 # output from plot_reconstructions (has cell_ids_per_cluster_dict)
#     fixed_activity_dict_NDNF_newest,    # to recompute TA traces
#     cells_per_animal_dict,              # {animal: [global_cell_ids...]}
#     K, ncol=None,                                  # which K to plot
#     ylim=(-1.1, 2.6), spacing=None,  
#     title_prefix="", ax_list=None):

    
#     cell_to_animal = {}
#     for animal, cell_ids in cells_per_animal_dict.items():
#         for cid in cell_ids:
#             cell_to_animal[cid] = animal

#     data, min_val = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
#     ta = data.mean(axis=2)                                                   # (cells, pos)
#     n_cells, n_pos = ta.shape

#     # clust_idx_dict = means_dict_cluster[K]["cell_ids_per_cluster_dict"]      # {cluster_label: array(cell_ids)}
#     # uniq = sorted(clust_idx_dict.keys())

#     uniq = np.unique(labels)

#     label_to_col = {lab: j for j, lab in enumerate(uniq)}

#     print(f"label_to_col {label_to_col}")

#     animals = sorted(set(cell_to_animal.values()))
#     cmap = plt.get_cmap("tab20", len(animals))
#     animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

#     for lab in uniq:
#         ax = ax_list[lab]
#         idx = uniq[lab]
#         traces = ta[idx]                                   # (n_k, n_pos)

#         for cid in idx:
#             a = cell_to_animal.get(int(cid), "unknown")
#             color = animal_to_color.get(a, (0.5,0.5,0.5,0.6))
#             ax.plot(traces[np.where(idx==cid)[0][0], :], lw=1.0, alpha=0.7, color=color)

#         m = traces.mean(axis=0)
#         s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
#         ax.plot(m, lw=2.0, color="k")
#         ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color="k")

#         ax.set_title(f"Cluster {lab} (n={len(idx)})")
#         ax.set_xlabel("Position bins")
#         ax.set_ylim(*ylim)
#     ax.set_ylabel("Z-scored dF/F")

#     fig = ax_list[0].figure
#     handles = [plt.Line2D([0],[0], color=animal_to_color[a], lw=2) for a in animals]
#     labels = [f"{a}" for a in animals]
#     fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False)
#     fig.subplots_adjust(top=0.85, right=0.98, left=0.07, bottom=spacing)


# def plot_cluster_traces_by_animal(
#     labels,                              # 1D array, length = n_cells
#     fixed_activity_dict_NDNF_newest,     # {animal: {cell_x: (pos, trials)}}
#     cells_per_animal_dict,              # kept for API compatibility (not strictly needed)
#     ylim=(-1.1, 2.6),
#     title_prefix="",
#     ncol=None,
#     ax_list=None,
#     spacing=0.15,
# ):
#     """
#     Plot TA traces per cluster, colored by animal.

#     Assumptions:
#     - `labels` is an array of length n_cells giving the cluster index (0..K-1) for each cell.
#     - `get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)` returns
#       data of shape (n_cells, n_pos, n_trials), with cells ordered consistently
#       across animals and cells.
#     """

#     # --- 1) Build TA array: (cells, pos) ---
#     data, min_val = get_truncated_to_min_data_array(fixed_activity_dict_NDNF_newest)  # (cells, pos, trials)
#     ta = data.mean(axis=2)  # (cells, pos)
#     n_cells, n_pos = ta.shape

#     labels = np.asarray(labels)
#     assert labels.shape[0] == n_cells, \
#         f"labels length {labels.shape[0]} does not match n_cells {n_cells}"

#     # --- 2) Build mapping from cell_idx -> animal (based on fixed_activity_dict_NDNF_newest order) ---
#     cell_idx_to_animal = {}
#     idx_counter = 0
#     # We assume get_truncated_to_min_data_array iterates animals and cells
#     # in sorted order; we mirror that here.
#     for animal in sorted(fixed_activity_dict_NDNF_newest.keys()):
#         # sort cell keys numerically: "cell_1", "cell_2", ...
#         cell_keys = sorted(
#             fixed_activity_dict_NDNF_newest[animal].keys(),
#             key=lambda s: int(s.split("_")[1])
#         )
#         for _ in cell_keys:
#             cell_idx_to_animal[idx_counter] = animal
#             idx_counter += 1

#     if idx_counter != n_cells:
#         print(f"⚠️ Warning: built {idx_counter} cell indices but TA has {n_cells} cells")

#     # --- 3) Unique clusters and subplot layout ---
#     uniq_clusters = np.unique(labels)
#     n_clusters = len(uniq_clusters)

#     if ax_list is None:
#         # Create a grid of subplots if not provided
#         if ncol is None:
#             ncol = min(4, n_clusters)
#         nrow = int(np.ceil(n_clusters / ncol))
#         fig, axs = plt.subplots(
#             nrow, ncol,
#             figsize=(4 * ncol, 3 * nrow),
#             sharey=True,
#         )
#         axs = np.atleast_1d(axs).ravel()
#         ax_list = axs
#     else:
#         # make sure we have enough axes
#         ax_list = np.atleast_1d(ax_list)
#         assert len(ax_list) >= n_clusters, \
#             f"Not enough axes ({len(ax_list)}) for {n_clusters} clusters"

#     # --- 4) Color per animal ---
#     animals = sorted(set(cell_idx_to_animal.values()))
#     cmap = plt.get_cmap("tab20", len(animals))
#     animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

#     # --- 5) Plot each cluster ---
#     for j, lab in enumerate(uniq_clusters):
#         ax = ax_list[j]
#         idx = np.where(labels == lab)[0]   # indices of cells in this cluster

#         if idx.size == 0:
#             continue

#         traces = ta[idx, :]  # (n_k, n_pos)

#         # plot individual cell traces
#         for row_i, cell_idx in enumerate(idx):
#             a = cell_idx_to_animal.get(int(cell_idx), "unknown")
#             color = animal_to_color.get(a, (0.5, 0.5, 0.5, 0.6))
#             ax.plot(traces[row_i, :], lw=1.0, alpha=0.7, color=color)

#         # mean ± SEM
#         m = traces.mean(axis=0)
#         s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
#         ax.plot(m, lw=2.0, color="k")
#         ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color="k")

#         ax.set_title(f"{title_prefix}Cluster {lab} (n={len(idx)})")
#         ax.set_xlabel("Position bins")
#         ax.set_ylim(*ylim)

#     # put ylabel on the leftmost used axis
#     ax_list[0].set_ylabel("Z-scored dF/F")

#     # # --- 6) Legend for animals ---
#     # fig = ax_list[0].figure
#     # handles = [plt.Line2D([0], [0], color=animal_to_color[a], lw=2) for a in animals]
#     # legend_labels = [f"{a}" for a in animals]
#     # fig.legend(handles, legend_labels, loc="lower center", ncol=len(animals),
#     #            frameon=False)
#     # fig.subplots_adjust(top=0.85, right=0.98, left=0.07, bottom=spacing)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

def plot_cluster_traces_by_animal_labels(
    labels,                              # 1D array, length = n_cells
    fixed_activity_dict,                 # {animal: {cell_x: (pos, trials)}}
    animal_to_color,
    ylim=(-1.1, 2.6),
    title_prefix="",
    ncol=None,
    ax_list=None,
    spacing=0.15,
):
    """
    Plot TA traces per cluster, colored by animal.

    Correctness guarantee:
    - Uses the SAME cell stacking order for both TA and animal mapping
      via idx_to_key returned by get_truncated_to_min_data_array.
    """

    # --- 1) Build TA array + ordering map ---
    data, min_val, idx_to_key = get_truncated_to_min_data_array(fixed_activity_dict)  # (cells, pos, trials)
    ta = data.mean(axis=2)  # (cells, pos)
    n_cells, n_pos = ta.shape

    labels = np.asarray(labels)
    if labels.shape[0] != n_cells:
        raise ValueError(f"labels length {labels.shape[0]} does not match n_cells {n_cells}")

    # --- 2) Map cell_idx -> animal using idx_to_key (same order as TA) ---
    cell_idx_to_animal = {i: a for i, (a, cell_key) in enumerate(idx_to_key)}

    # --- 3) Unique clusters and subplot layout ---
    uniq_clusters = np.unique(labels)
    n_clusters = len(uniq_clusters)

    if ax_list is None:
        if ncol is None:
            ncol = min(4, n_clusters)
        nrow = int(np.ceil(n_clusters / ncol))
        fig, axs = plt.subplots(
            nrow, ncol,
            figsize=(4 * ncol, 3 * nrow),
            sharey=True,
        )
        ax_list = np.atleast_1d(axs).ravel()
    else:
        ax_list = np.atleast_1d(ax_list)
        if len(ax_list) < n_clusters:
            raise ValueError(f"Not enough axes ({len(ax_list)}) for {n_clusters} clusters")

    # --- 4) Color per animal ---
    animals = list(animal_to_color.keys())
    # cmap = plt.get_cmap("tab20", len(animals))
    # animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}
    colors_list = ["purple", 'r']
    # --- 5) Plot each cluster ---
    for j, lab in enumerate(uniq_clusters):
        ax = ax_list[j]
        idx = np.where(labels == lab)[0]  # indices into TA

        if idx.size == 0:
            ax.set_axis_off()
            continue

        traces = ta[idx, :]  # (n_k, n_pos)

        # individual traces colored by animal
        for row_i, cell_idx in enumerate(idx):
            a = cell_idx_to_animal[cell_idx]
            ax.plot(traces[row_i, :], lw=1.0, alpha=0.7, color=animal_to_color[a])

        # mean ± SEM
        m = traces.mean(axis=0)
        s = sem(traces, axis=0) if traces.shape[0] > 1 else np.zeros_like(m)
        ax.plot(m, lw=2.0, color=colors_list[j])
        ax.fill_between(np.arange(n_pos), m - s, m + s, alpha=0.15, color="k")

        ax.set_title(f"{title_prefix}Cluster {lab} (n={len(idx)})")
        ax.set_xlabel("Position bins")
        ax.set_ylim(*ylim)

        ax.set_ylabel("Z-Scored DF/F")

    # # legend (optional)
    # fig = ax_list[0].figure
    # handles = [plt.Line2D([0], [0], color=animal_to_color[a], lw=2) for a in animals]
    # legend_labels = [f"{a}" for a in animals]
    # fig.legend(handles, legend_labels, loc="lower center", ncol=(ncol or min(6, len(animals))), frameon=False)

    # fig.subplots_adjust(bottom=spacing, top=0.90)


    
def get_r_list(fixed_activity_dict_NDNF_newest, factors_dict_NDNF_newest, data_truncated_array_EC, data_to_corr=None):

    r_list_vel = []
    vel_per_animal = []
    for animal in fixed_activity_dict_NDNF_newest:
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            vel = factors_dict_NDNF_newest[animal][data_to_corr][:,:data_truncated_array_EC.shape[2]]
            vel_per_animal.append(vel)
            data = fixed_activity_dict_NDNF_newest[animal][cell][:,:data_truncated_array_EC.shape[2]]
            r, _ = pearsonr(vel.flatten(), data.flatten())
            r_list_vel.append(r)

    array_data = np.array(vel_per_animal)
    r_dict_vel = {"r_list":r_list_vel,
                  "array_data":array_data}

    return r_dict_vel

# def plot_cluster_animal_composition_stacked_from_index(
#     labels,
#     fixed_activity_dict,   # same dict you pass to get_truncated_to_min_data_array
#     K,
#     title_prefix="",
#     show_percent_labels=False,
#     ax=None,
#     legend="outside",
#     legend_ncol=1
# ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     # Build cell_idx -> animal in the SAME order you stacked data
#     cell_idx_to_animal = {}
#     idx_counter = 0
#     for animal in sorted(fixed_activity_dict.keys()):
#         cell_keys = sorted(fixed_activity_dict[animal].keys(),
#                            key=lambda s: int(s.split("_")[1]))
#         for _ in cell_keys:
#             cell_idx_to_animal[idx_counter] = animal
#             idx_counter += 1

#     labels = np.asarray(labels)
#     n_cells = labels.shape[0]
#     if idx_counter != n_cells:
#         print(f"⚠️ ordering mismatch: built {idx_counter} indices but labels has {n_cells}")

#     clusters = np.unique(labels)
#     animals = sorted(set(cell_idx_to_animal.values()))
#     cmap = plt.get_cmap("tab20", len(animals))
#     animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}

#     counts = np.zeros((len(clusters), len(animals)), dtype=int)
#     for r, clab in enumerate(clusters):
#         cell_indices = np.where(labels == clab)[0]
#         for cell_idx in cell_indices:
#             a = cell_idx_to_animal.get(int(cell_idx), None)
#             if a is None:
#                 continue
#             counts[r, animals.index(a)] += 1

#     totals = counts.sum(axis=1)
#     x = np.arange(len(clusters))

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(1.2 * len(clusters) + 3, 5))

#     bottom = np.zeros_like(totals, dtype=float)
#     for j, a in enumerate(animals):
#         ax.bar(x, counts[:, j], bottom=bottom, width=0.8,
#                color=animal_to_color[a], label=a)
#         bottom += counts[:, j]

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"C{cl}\n(n={totals[i]})" for i, cl in enumerate(clusters)])
#     ax.set_ylabel("# cells")
#     ax.set_title(f"{title_prefix} K={K}")

#     if show_percent_labels:
#         with np.errstate(divide="ignore", invalid="ignore"):
#             props = counts / totals[:, None]
#             props[np.isnan(props)] = 0.0
#         for i in range(len(clusters)):
#             cum = 0.0
#             for j in range(len(animals)):
#                 h = counts[i, j]
#                 if h > 0 and props[i, j] >= 0.07:
#                     ax.text(x[i], cum + h/2, f"{props[i,j]*100:.0f}%",
#                             ha="center", va="center", fontsize=8, color="white")
#                 cum += h

#     if legend == "outside":
#         ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
#                   frameon=False, ncol=legend_ncol)
#     elif legend == "inside":
#         ax.legend(loc="upper right", frameon=False)

#     return ax



def plot_cluster_animal_composition_stacked_from_index(
    labels, animal_to_color,
    fixed_activity_dict,   # same dict you pass to get_truncated_to_min_data_array
    K=None,
    title_prefix="",
    show_percent_labels=False,
    ax=None,
    legend="outside",
    legend_ncol=1
):

    # --- Build cell_idx -> animal using THE SAME stacking order as TA ---
    data, min_val, idx_to_key = get_truncated_to_min_data_array(fixed_activity_dict)
    # idx_to_key is [(animal, cell_key), ...] in the same order as your cell axis
    cell_idx_to_animal = {i: a for i, (a, c) in enumerate(idx_to_key)}

    labels = np.asarray(labels)
    n_cells = labels.shape[0]
    if len(idx_to_key) != n_cells:
        print(f"⚠️ mismatch: idx_to_key has {len(idx_to_key)} cells but labels has {n_cells}")

    clusters = np.unique(labels)

    # animals present (in deterministic order for legend/color stability)
    animals = list(animal_to_color.keys())

    animal_to_j = {a: j for j, a in enumerate(animals)}

    # counts[cluster, animal]
    counts = np.zeros((len(clusters), len(animals)), dtype=int)
    for r, clab in enumerate(clusters):
        cell_indices = np.where(labels == clab)[0]
        for cell_idx in cell_indices:
            a = cell_idx_to_animal.get(int(cell_idx), None)
            if a is None:
                continue
            counts[r, animal_to_j[a]] += 1

    totals = counts.sum(axis=1)
    x = np.arange(len(clusters))

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2 * len(clusters) + 3, 5))

    bottom = np.zeros_like(totals, dtype=float)
    for j, a in enumerate(animals):
        ax.bar(
            x, counts[:, j], bottom=bottom, width=0.8,
            color=animal_to_color[a], label=a
        )
        bottom += counts[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{cl}\n(n={totals[i]})" for i, cl in enumerate(clusters)])
    ax.set_ylabel("# cells")
    k_txt = "" if K is None else f" "
    ax.set_title(f"{title_prefix}{k_txt}")

    if show_percent_labels:
        with np.errstate(divide="ignore", invalid="ignore"):
            props = counts / totals[:, None]
            props[np.isnan(props)] = 0.0

        for i in range(len(clusters)):
            cum = 0.0
            for j in range(len(animals)):
                h = counts[i, j]
                if h > 0 and props[i, j] >= 0.07:
                    ax.text(
                        x[i], cum + h/2, f"{props[i,j]*100:.0f}%",
                        ha="center", va="center", fontsize=8, color="white"
                    )
                cum += h

    if legend == "outside":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, ncol=legend_ncol)
    elif legend == "inside":
        ax.legend(loc="upper right", frameon=False)

    return ax


def plot_lick_vel_data_clust(
    labels,
    clean_velocity_dict_NDNF_newest,
    cells_per_animal_dict,
    num_clusters=None,      # optional, we can infer from labels if None
    use_vel=False,
    title=None,
    ax=None,
    color_list=None):
    """
    Plot mean ± SEM velocity (or lick) traces per cluster.

    Parameters
    ----------
    labels : 1D array-like, shape (n_cells,)
        Cluster label for each global cell ID (0..n_cells-1).
    clean_velocity_dict_NDNF_newest : dict
        {animal_name: {"Velocity": vel_array}} with vel_array shape (pos, trials).
    cells_per_animal_dict : dict
        {animal_name: [global_cell_ids...]} mapping which cells belong to which animal.
    num_clusters : int or None
        If given, only the first num_clusters clusters (sorted) are plotted.
        If None, all unique labels are plotted.
    use_vel : bool
        If True, ylabel = 'Velocity (meters/sec)', else 'Normalized Lick Rate'.
        (You can reuse this for licks by swapping what goes in the dict.)
    title : str
        Plot title.
    ax : matplotlib axis or None
        Axis to plot into. If None, a new fig/ax is created.
    color_list : list of colors
        One color per cluster. If None, a default colormap is used.
    """

    labels = np.asarray(labels)

    # --- set up axis ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True
    else:
        fig = ax.figure

    # --- build cell_id -> animal map ---
    cell_to_animal = {}
    for animal, cell_ids in cells_per_animal_dict.items():
        for cid in cell_ids:
            cell_to_animal[int(cid)] = animal

    # --- determine common # of trials across all animals (for truncation) ---
    # assume vel_array shape is (pos, trials)
    min_num_trials = None
    for animal in clean_velocity_dict_NDNF_newest:
        vel = clean_velocity_dict_NDNF_newest[animal][title]
        n_trials = vel.shape[1]
        if min_num_trials is None or n_trials < min_num_trials:
            min_num_trials = n_trials

    # --- which clusters to plot ---
    uniq = np.unique(labels)
    uniq = np.sort(uniq)
    if num_clusters is not None:
        uniq = uniq[:num_clusters]

    if color_list is None:
        cmap = plt.get_cmap("tab10", len(uniq))
        color_list = [cmap(i) for i in range(len(uniq))]

    # --- loop over clusters ---
    for i_idx, clab in enumerate(uniq):
        labels_loc = np.where(labels == clab)[0]
        if labels_loc.size == 0:
            continue

        # Build a (n_cells_in_cluster, pos, trials_trunc) array
        cell_vels = []
        for cid in labels_loc:
            animal = cell_to_animal.get(int(cid), None)
            if animal is None:
                continue
            vel = clean_velocity_dict_NDNF_newest[animal][title]  # (pos, trials)
            vel_trunc = vel[:, :min_num_trials]                         # (pos, min_trials)
            cell_vels.append(vel_trunc[None, :, :])                    # (1, pos, trials)

        if len(cell_vels) == 0:
            continue

        vel_array = np.concatenate(cell_vels, axis=0)  # (n_cells_in_cluster, pos, trials)

        # average over trials -> (n_cells_in_cluster, pos)
        trial_av_vel_array = np.nanmean(vel_array, axis=2)

        # average over cells
        mean_over_cells = np.nanmean(trial_av_vel_array, axis=0)
        sem_over_cells = sem(trial_av_vel_array, axis=0, nan_policy='omit')

        # plot
        ax.plot(
            mean_over_cells,
            label=f"Cluster {clab}", #(n={vel_array.shape[0]})",
            color=color_list[i_idx],
        )
        ax.fill_between(
            range(len(mean_over_cells)),
            mean_over_cells - sem_over_cells,
            mean_over_cells + sem_over_cells,
            alpha=0.2,
            color=color_list[i_idx],
        )

    # --- labels etc. ---
    ax.set_xlabel("Position bins")
    if use_vel:
        ax.set_ylabel("Velocity (meters/sec)")
    else:
        ax.set_ylabel("Normalized Lick Rate")

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax



def get_cell_features(w1, f1, *,
                      feature_mode="latent_pos_flat",  # "latent_pos_flat" | "latent_trials_flat" | "loadings" | "pos_profile"
                      n_cells_expected=None,
                      n_latents_expected=None):
    """
    Build per-cell feature matrix X (rows=cells).
    Handles arbitrary axis orders from SliceTCA outputs across (x00, 0x0, 00x).
    w1, f1 are torch tensors from model.vectors[which_vectors].

    feature_mode:
      - "latent_pos_flat":   (cells, latents*pos)        e.g., (115, 20*50)
      - "latent_trials_flat":(cells, latents*trials)     e.g., (115, 20*100)
      - "loadings":          (cells, latents)            e.g., (115, 20)
      - "pos_profile":       (cells, pos) via weighted latent templates

    Notes:
      • If a chosen mode can’t be formed from the available axes, it raises a helpful error.
      • Pass n_cells_expected=115 (and n_latents_expected=20) to make detection strict.
    """
    # to numpy
    W = w1.detach().cpu().numpy()
    F = f1.detach().cpu().numpy()

    # small helpers
    def find_axis_by_size(shape, target):
        return shape.index(target) if (target in shape) else None

    def move_front(a, ax):
        return np.moveaxis(a, ax, 0)

    # Try to detect latents axis (often 20)
    Lw = find_axis_by_size(W.shape, n_latents_expected) if n_latents_expected else None
    Lf = find_axis_by_size(F.shape, n_latents_expected) if n_latents_expected else None
    # Detect cells axis (often 115)
    Cw = find_axis_by_size(W.shape, n_cells_expected) if n_cells_expected else None
    Cf = find_axis_by_size(F.shape, n_cells_expected) if n_cells_expected else None

    # ---- Option 1: LOADINGS (cells × latents) ----
    if feature_mode == "0x0":
        # Prefer a (latents, cells) slice if present (W or F that directly has cells)
        if (Lw is not None) and (Cw is not None) and W.ndim == 2:
            # W (latents, cells) -> (cells, latents)
            X = np.moveaxis(W, (Lw, Cw), (1, 0))
            X = X.T  # (cells, latents)
            return X
        # Else try from F by reducing non-latent dims
        if Cf is not None and Lf is not None and F.ndim >= 2:
            # bring cells front, latents second
            order = [Cf, Lf] + [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            G = np.transpose(F, order)     # (cells, latents, ...)
            if G.ndim == 2:
                return G  # already (cells, latents)
            # average remaining dims (e.g., trials/pos) → (cells, latents)
            X = G.mean(axis=tuple(range(2, G.ndim)))
            return X
        raise ValueError("Cannot form 'loadings': no (latents,cells) pairing found in W/F.")

    # ---- Option 2: latent × pos flattened (cells × (L*P)) ----
    if feature_mode == "x00":
        # We need a tensor that contains (latents, cells, pos) in some order
        # Check F first (most common)
        if Cf is not None and Lf is not None and 3 <= F.ndim <= 4:
            # find a pos-like axis: pick the remaining non-cells, non-latents axis
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            if not rem:
                raise ValueError("No pos axis found in F for latent_pos_flat.")
            pos_ax = rem[0]
            # reorder to (cells, latents, pos, [maybe ...])
            order = [Cf, Lf, pos_ax] + [ax for ax in rem[1:]]
            G = np.transpose(F, order)
            # if more dims exist, mean over them
            if G.ndim > 3:
                G = G.mean(axis=tuple(range(3, G.ndim)))
            C, L, P = G.shape
            X = G.reshape(C, L * P)
            return X
        # Else: if W has (latents, pos) and *no* cells, we can’t build per-cell features from W alone.
        raise ValueError("Cannot form 'latent_pos_flat' from current W/F: need latents+cells+pos together (usually in F).")

    # ---- Option 3: latent × trials flattened (cells × (L*T)) ----
    if feature_mode == "00x":
        # Need (latents, trials, cells) in some order
        if Cf is not None and Lf is not None and 3 <= F.ndim <= 4:
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            if not rem:
                raise ValueError("No trials axis found in F for latent_trials_flat.")
            tr_ax = rem[0]
            order = [Cf, Lf, tr_ax] + [ax for ax in rem[1:]]
            G = np.transpose(F, order)  # (cells, latents, trials, [maybe ...])
            if G.ndim > 3:
                G = G.mean(axis=tuple(range(3, G.ndim)))
            C, L, T = G.shape
            X = G.reshape(C, L * T)
            return X
        raise ValueError("Cannot form 'latent_trials_flat': need latents+cells+trials together (usually in F).")

    # ---- Option 4: pos_profile (cells × pos), via weighted templates if possible ----
    if feature_mode == "pos_profile":
        # If we have (latents,cells, pos) in F: average latents → (cells, pos)
        if Cf is not None and Lf is not None and 3 <= F.ndim:
            rem = [ax for ax in range(F.ndim) if ax not in (Cf, Lf)]
            pos_ax = rem[0] if rem else None
            if pos_ax is not None:
                order = [Cf, Lf, pos_ax] + [ax for ax in rem[1:]]
                G = np.transpose(F, order)  # (cells, latents, pos, ...)
                if G.ndim > 3:
                    G = G.mean(axis=tuple(range(3, G.ndim)))
                X = G.mean(axis=1)  # avg over latents -> (cells, pos)
                return X
        raise ValueError("Cannot form 'pos_profile' from current W/F.")

    raise ValueError(f"Unknown feature_mode: {feature_mode}")
    


def lda_with_orthogonal_axis_2d(X, labels, title_prefix=""):
    """
    Returns a 2D embedding:
    axis 1: Fisher LDA direction (max class separation)
    axis 2: top-variance direction orthogonal to LDA (via PCA in orth subspace)
    """
    X = np.asarray(X)
    y = np.asarray(labels)
    n_samples, n_features = X.shape
    if n_features < 2:
        raise ValueError("Need at least 2 features to build an orthogonal second axis.")

    # --- 1) Fit LDA (1 component) and get the discriminant vector w ---
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)

    # For binary classes, coef_ has shape (1, n_features)
    w = lda.coef_[0].astype(float)
    w_norm = np.linalg.norm(w)
    if w_norm == 0:
        # fallback: if degenerate, use mean-difference direction
        classes = np.unique(y)
        mu0 = X[y == classes[0]].mean(axis=0)
        mu1 = X[y == classes[1]].mean(axis=0)
        w = (mu1 - mu0)
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            raise ValueError("Could not determine a discriminant direction (degenerate data).")
    w /= w_norm  # unit vector

    # --- 2) Build the orthogonal subspace and find its top-variance direction ---
    # Project data onto the orthogonal complement of w
    # X_perp = X - (X w) w
    Xw = X @ w
    X_perp = X - np.outer(Xw, w)

    # Center X_perp (important for PCA)
    X_perp_centered = X_perp - X_perp.mean(axis=0, keepdims=True)

    # If all variance orthogonal to w vanishes, fall back to a random orth direction
    if np.allclose(np.var(X_perp_centered, axis=0).sum(), 0.0, atol=1e-12):
        # Random orth direction (Gram-Schmidt)
        rand_vec = np.random.randn(n_features)
        orth = rand_vec - (rand_vec @ w) * w
        if np.linalg.norm(orth) < 1e-12:
            # try again
            rand_vec = np.random.randn(n_features)
            orth = rand_vec - (rand_vec @ w) * w
        orth /= np.linalg.norm(orth)
    else:
        # PCA in the orthogonal subspace to get the most informative 2nd axis
        pca = PCA(n_components=1, svd_solver="full")
        pca.fit(X_perp_centered)
        # PCA component is in the *feature* space of X_perp_centered, already orthogonal to w
        orth = pca.components_[0]
        # Keep it strictly orthogonal (numerical safety)
        orth = orth - (orth @ w) * w
        orth /= np.linalg.norm(orth)

    # --- 3) Project original X onto (w, orth) for 2D embedding ---
    X_2d = np.column_stack((X @ w, X @ orth))
    return X_2d



def get_binned_data_for_CDF(animal_average_selectivity_dict, n_bins=20):
    # collect values
    vals = []
    for animal in animal_average_selectivity_dict:
        for cell in animal_average_selectivity_dict[animal]:
            v = animal_average_selectivity_dict[animal][cell]
            vals.append(v)

    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]  # drop NaN/inf
    if arr.size == 0:
        # return n_bins empty arrays to keep the downstream shape
        return [np.array([]) for _ in range(n_bins)]

    # if fewer points than bins, reduce bins to available size
    n_eff = min(n_bins, arr.size)

    # quantile edges can repeat when many identical values
    edges = np.quantile(arr, np.linspace(0, 1, n_eff + 1))

    binned = []
    for i in range(n_eff):
        low, high = edges[i], edges[i+1]
        if i < n_eff - 1:
            mask = (arr >= low) & (arr < high)
        else:
            mask = (arr >= low) & (arr <= high)
        binned.append(arr[mask])

    # pad if n_eff < n_bins (keeps plotting code unchanged)
    if n_eff < n_bins:
        binned += [np.array([])] * (n_bins - n_eff)

    return binned


def plot_the_CDF_celltypes(binned_data_0, binned_data_1, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
    mean_0, sem_0 = get_mean_sem_lists(binned_data_0)
    mean_1, sem_1 = get_mean_sem_lists(binned_data_1)

    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5

    # Plot: horizontal bars (x=selectivity, y=percentile)
    # plt.figure(figsize=(7, 6))

    ax.errorbar(mean_0, percentiles, xerr=sem_0, fmt='o-', label='Cell Type 0', color='purple', capsize=3)
    ax.errorbar(mean_1, percentiles, xerr=sem_1, fmt='o-', label='Cell Type 1', color='red', capsize=3)

    ax.set_ylabel("Percentile of Cells")
    ax.set_xlabel("Selectivity")
    ax.set_title(title)
    ax.legend()




def get_selectivity_each_trial_early_late_cluster(activity_dict_EC, cp_dict_EC, cell_cluster_list, neg_sel=True, trial_av=False, eml="early", norm=None):
    """
    - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
    returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually
    """

    count=0

    animal_dict_data = {}

    animal_average_selectivity_dict = {}
    for idx, animal in enumerate(activity_dict_EC):
        cell_dict = {}
        cell_dict_data = {}
        for idt, cell in enumerate(activity_dict_EC[animal]):
            
            if count in cell_cluster_list:

                cp = cp_dict_EC[animal][cell]
                early_cut = cp[0]
                late_cut = cp[1]
                cell_data = activity_dict_EC[animal][cell]
                if trial_av:
                    if eml=="early":
                        cell_data_early = cell_data[:,:early_cut]
                        trial_av_activity = np.mean(cell_data_early, axis=1) 
                        cell_dict_data[cell] = trial_av_activity
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict[cell] = selectivity_trial_av
                    elif eml=="middle":
                        cell_data_late = cell_data[:,early_cut:late_cut]
                        trial_av_activity = np.mean(cell_data_late, axis=1) 
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict_data[cell] = trial_av_activity
                        cell_dict[cell] = selectivity_trial_av
                    elif eml=="late":
                        cell_data_late = cell_data[:,-late_cut:]
                        trial_av_activity = np.mean(cell_data_late, axis=1) 
                        selectivity_trial_av = Vinje2000(trial_av_activity, norm=norm, negative_selectivity=neg_sel)
                        cell_dict_data[cell] = trial_av_activity
                        cell_dict[cell] = selectivity_trial_av
                    else:
                        raise ValueError("improper eml")
                else:

                    ####### have to fix this 
                    trial_selectivity_list = []
                    if eml=="early":
                        for trial in range(cell_data.shape[1]):
                            if trial <= early_cut:
                                trial_activity = cell_data[:,trial] 
                                selectivity_trial = Vinje2000(trial_activity, norm=norm, negative_selectivity=neg_sel)
                                trial_selectivity_list.append(selectivity_trial)
                                cell_dict_data[cell] = trial_activity
                    else:
                        for trial in range(cell_data.shape[1]):
                            if trial >= late_cut:
                                trial_activity = cell_data[:,trial] 
                                selectivity_trial = Vinje2000(trial_activity, norm=norm, negative_selectivity=neg_sel)
                                trial_selectivity_list.append(selectivity_trial)
                                cell_dict_data[cell] = trial_activity

                    percentile_average_selectivity = np.mean(trial_selectivity_list)
                    cell_dict[cell] = percentile_average_selectivity
            count+=1
            animal_average_selectivity_dict[animal] = cell_dict
            animal_dict_data[animal] = cell_dict_data
    return animal_average_selectivity_dict, animal_dict_data



def get_selectivity_array(animal_average_selectivity_dict):

    all_cell_selectivity = []

    for animal in animal_average_selectivity_dict:
        for cell in animal_average_selectivity_dict[animal]:
            selectivity_per_bin = animal_average_selectivity_dict[animal][cell]
            if len(selectivity_per_bin) == 10:  # sanity check
                all_cell_selectivity.append(selectivity_per_bin)

    all_cell_selectivity = np.array(all_cell_selectivity)  # shape: [n_cells, 10]

    return all_cell_selectivity


def plot_selectivity_over_trials(group_0_selectivity, group_1_selectivity, color_list=None, ax=None):

    mean_selectivity_0 = np.mean(group_0_selectivity, axis=0)
    sem_selectivity_0 = sem(group_0_selectivity, axis=0)

    mean_selectivity_1 = np.mean(group_1_selectivity, axis=0)
    sem_selectivity_1 = sem(group_1_selectivity, axis=0)

    x = np.arange(1, 11) 

    ax.plot(x, mean_selectivity_0, color=color_list[0], label='Cluster 0')
    ax.fill_between(x, mean_selectivity_0 - sem_selectivity_0, mean_selectivity_0 + sem_selectivity_0, alpha=0.2, color=color_list[0])
    ax.plot(x, mean_selectivity_1, color=color_list[1], label='Cluster 1')
    ax.fill_between(x, mean_selectivity_1 - sem_selectivity_1, mean_selectivity_1 + sem_selectivity_1, alpha=0.2, color=color_list[1])
    
    tick_pos = [1, 5, 10]
    tick_lab = ["0%", "50%", "100%"]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    
    # ax.set_xticks(ticks=x, labels=[f"{int(p)}%" for p in np.linspace(0, 100, 10)])
    ax.set_xlabel("Percentile of Trials")
    ax.set_ylabel("Selectivity")
    ax.set_title("Selectivity Across Trials")
    ax.legend()


def get_percentlie_slices(activity_dict_SST):
    percentile_slices = {}

    for animal in activity_dict_SST:
        percentile_slices_cell = {}
        for cell in activity_dict_SST[animal]:
            data = activity_dict_SST[animal][cell]
            num_trials = data.shape[1]

            cut_indices = [int(p * num_trials / 10) for p in range(1, 10)]  
            cut_indices = [0] + cut_indices + [num_trials] 

            cell_slices = []
            for idx in range(10):
                start = cut_indices[idx]
                end = cut_indices[idx + 1]
                data_slice = data[:, start:end]
                cell_slices.append(data_slice)

            percentile_slices_cell[cell] = cell_slices
        percentile_slices[animal] = percentile_slices_cell

    return percentile_slices


def selectivity_from_percentile_slices(percentile_slices, norm='min_max', neg_sel=False):
    """
    percentile_slices[animal][cell] = list of 10 arrays, each (pos, trials_in_bin)
    Returns array of shape (n_cells, 10) with one scalar selectivity per bin.
    """
    rows = []
    for animal in percentile_slices:
        for cell, list10 in percentile_slices[animal].items():
            if len(list10) != 10:
                continue
            vec = []
            for sl in list10:
                # sl: (pos, trials) – average across trials, then compute selectivity across pos
                if sl.size == 0:
                    vec.append(np.nan)
                    continue
                ta = np.mean(sl, axis=1)  # (pos,)
                vec.append(Vinje2000(ta, norm=norm, negative_selectivity=neg_sel))
            rows.append(vec)
    return np.asarray(rows, dtype=float)


def plot_selectivity_seperated_by_learn_stage(type0, type1, colors_list=None, ax=None):

    NDNF_means0, NDNF_sems0 = get_mean_selelectivity_by_cutpoint(type0)

    NDNF_means1, NDNF_sems1 = get_mean_selelectivity_by_cutpoint(type1)

    x = np.arange(3)
    labels = ["Early", "Middle", "Late"]

    ax.errorbar(x, NDNF_means0, yerr=NDNF_sems0, color=colors_list[0], label="Cell Type 0", capsize=4, fmt='-o')
    ax.errorbar(x, NDNF_means1, yerr=NDNF_sems1, color=colors_list[1], label="Cell Type 1", capsize=4, fmt='-o')

    ax.set_xticks(x, labels)
    ax.set_ylabel("Average Selectivity Across Cells")
    ax.set_xlabel("Contiguous K-Means Learning Stage")
    ax.set_title("Selectivity by Learning Stage")
    ax.legend()
    

def get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_SST):
    early_list_SST = []
    middle_list_SST = []
    late_list_SST = []

    for animal in animal_average_selectivity_dict_SST:
        for cell in animal_average_selectivity_dict_SST[animal]:
            early_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["early_selectivity"])
            middle_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["middle_selectivity"])
            late_list_SST.append(animal_average_selectivity_dict_SST[animal][cell]["late_selectivity"])

    early_mean = np.mean(early_list_SST)
    middle_mean = np.mean(middle_list_SST)
    late_mean = np.mean(late_list_SST)

    early_sem = sem(early_list_SST)
    middle_sem = sem(middle_list_SST)
    late_sem = sem(late_list_SST)

    SST_means = [early_mean, middle_mean, late_mean]
    SST_sems = [early_sem, middle_sem, late_sem]
    return SST_means, SST_sems


def plot_the_CDF_early_late(binned_data_0_early, binned_data_1_early, binned_data_0_late, binned_data_1_late, title="Selectivity Distribution Across Cells +-SEM",  n_bins = None, ax=None):
    mean_0_early, sem_0_early = get_mean_sem_lists(binned_data_0_early)
    mean_1_early, sem_1_early = get_mean_sem_lists(binned_data_1_early)

    mean_0_late, sem_0_late = get_mean_sem_lists(binned_data_0_late)
    mean_1_late, sem_1_late = get_mean_sem_lists(binned_data_1_late)


    percentiles = np.linspace(100 / (2 * n_bins), 100 - (100 / (2 * n_bins)), n_bins)  # e.g., 2.5, 7.5, ..., 97.5


    ax.errorbar(mean_0_early, percentiles, xerr=sem_0_early, fmt='o-', label='Cell Type 0 Early', color='orchid', capsize=3)
    ax.errorbar(mean_0_late, percentiles, xerr=sem_0_late, fmt='o-', label='Cell Type 0 Late', color='purple', capsize=3)
    ax.errorbar(mean_1_early, percentiles, xerr=sem_1_early, fmt='o-', label='Cell Type 1 Early', color='orange', capsize=3)
    ax.errorbar(mean_1_late, percentiles, xerr=sem_1_late, fmt='o-', label='Cell Type 1 Late', color='Crimson', capsize=3)

    ax.set_ylabel("Percentile of Cells")
    ax.set_xlabel("Selectivity")
    ax.set_title(title)
    ax.legend(fontsize=6)


def collect_eml_data(animal_average_selectivity_dict_NDNF_0_early):

    cell_list = []

    for animal in animal_average_selectivity_dict_NDNF_0_early:
        for cell in animal_average_selectivity_dict_NDNF_0_early[animal]:
            cell_list.append(animal_average_selectivity_dict_NDNF_0_early[animal][cell])

    cell_array = np.array(cell_list)

    return cell_array
            


def plot_eml_data(animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=None, color_list=None):

    cell_array_0_early = collect_eml_data(animal_average_selectivity_dict_NDNF_0_early)
    cell_array_1_early = collect_eml_data(animal_average_selectivity_dict_NDNF_1_early)

    cell_array_0_middle = collect_eml_data(animal_average_selectivity_dict_NDNF_0_middle)
    cell_array_1_middle = collect_eml_data(animal_average_selectivity_dict_NDNF_1_middle)

    cell_array_0_late = collect_eml_data(animal_average_selectivity_dict_NDNF_0_late)
    cell_array_1_late = collect_eml_data(animal_average_selectivity_dict_NDNF_1_late)


    cell_array_0_early_mean = np.mean(cell_array_0_early)
    cell_array_0_early_sem = sem(cell_array_0_early)

    cell_array_1_early_mean = np.mean(cell_array_1_early)
    cell_array_1_early_sem = sem(cell_array_1_early)

    cell_array_0_middle_mean = np.mean(cell_array_0_middle)
    cell_array_0_middle_sem = sem(cell_array_0_middle)

    cell_array_1_middle_mean = np.mean(cell_array_1_middle)
    cell_array_1_middle_sem = sem(cell_array_1_middle)

    cell_array_0_late_mean = np.mean(cell_array_0_late)
    cell_array_0_late_sem = sem(cell_array_0_late)

    cell_array_1_late_mean = np.mean(cell_array_1_late)
    cell_array_1_late_sem = sem(cell_array_1_late)

    means0 = [cell_array_0_early_mean,cell_array_0_middle_mean, cell_array_0_late_mean]
    sems0 = [cell_array_0_early_sem,cell_array_0_middle_sem, cell_array_0_late_sem]

    means1 = [cell_array_1_early_mean,cell_array_1_middle_mean, cell_array_1_late_mean]
    sems1 = [cell_array_1_early_sem,cell_array_1_middle_sem, cell_array_1_late_sem]

    ax.errorbar(range(len(means0)), means0, yerr=sems0, label="Cell Type 0", color=color_list[0])
    ax.errorbar(range(len(means1)), means1, yerr=sems1, label="Cell Type 1", color=color_list[1])
    ax.set_xticks(np.arange(3), ["Early", "Middle", "Late"])
    ax.set_ylabel("Selectivity")
    ax.legend()



def debug_lda_orth_geometry(X, y, title=""):
    X = np.asarray(X)
    y = np.asarray(y)

    lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    w = lda.coef_[0].astype(float)
    w /= np.linalg.norm(w)

    # LD1 scores (projection onto w)
    ld1 = X @ w

    # Remove LD1 component to get orthogonal residual cloud
    X_perp = X - np.outer(ld1, w)

    # PCA on residual to get orth axis
    X_perp_centered = X_perp - X_perp.mean(axis=0, keepdims=True)
    pca = PCA(n_components=1).fit(X_perp_centered)
    orth = pca.components_[0]
    orth = orth - (orth @ w) * w
    orth /= np.linalg.norm(orth)

    # Check orthogonality numerically
    print("w·orth =", float(np.dot(w, orth)))

    # 2D embedding
    X2 = np.c_[X @ w, X @ orth]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: LD1 score distribution
    for lab in np.unique(y):
        axs[0].hist(ld1[y == lab], bins=30, alpha=0.5, density=True, label=f"class {lab}")
    axs[0].set_title("LD1 scores (X @ w)")
    axs[0].legend()

    # Panel B: variance left after removing LD1 (norm of residual per sample)
    resid_norm = np.linalg.norm(X_perp_centered, axis=1)
    for lab in np.unique(y):
        axs[1].hist(resid_norm[y == lab], bins=30, alpha=0.5, density=True, label=f"class {lab}")
    axs[1].set_title("Residual magnitude after removing LD1")
    axs[1].legend()

    # Panel C: final 2D embedding
    for lab in np.unique(y):
        idx = (y == lab)
        axs[2].scatter(X2[idx, 0], X2[idx, 1], s=15, alpha=0.7, label=f"class {lab}")
    axs[2].axhline(0, lw=1)
    axs[2].axvline(0, lw=1)
    axs[2].set_xlabel("LD1 (X @ w)")
    axs[2].set_ylabel("Orth-PC1 (X @ orth)")
    axs[2].set_title("LD1 vs Orth-PC1")
    axs[2].legend()

    fig.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

    return X2, w, orth, X_perp


import numpy as np
from matplotlib.lines import Line2D

def _remap_bin_indices(vals, old_n_bins, new_n_bins):
    """Map integer bin indices in [0, old_n_bins-1] -> [0, new_n_bins-1]."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return v.astype(int)

    # Keep only finite values (optional)
    mask = np.isfinite(v)
    v2 = v[mask]

    # Clip to valid old range (optional but safer)
    v2 = np.clip(v2, 0, old_n_bins - 1)

    mapped = np.floor(v2 * new_n_bins / old_n_bins).astype(int)
    mapped = np.clip(mapped, 0, new_n_bins - 1)

    out = np.full(v.shape, np.nan)
    out[mask] = mapped
    return out[mask].astype(int)  # return just finite mapped ints

import numpy as np
from matplotlib.lines import Line2D

def plot_butterfly_hist(
    argmax_list_early, argmax_list_late,
    argmin_list_early, argmin_list_late,
    ax=None,
    colors_list=None,
    title=None,
    n_bins=None,
    old_n_bins=None,
    remap=False,
    mirror_weight=-1.0,
    alpha=1.0,              # alpha for lines (usually 1)
    track_len_cm=180.0,     # for xtick labels
    n_xticks=6,             # how many tick labels across (incl ends)
    lw=1.0,
):
    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

    if colors_list is None:
        colors_list = ["C0", "C1", "C2", "C3"]

    if n_bins is None:
        raise ValueError("Pass n_bins (e.g., 20)")

    if remap:
        if old_n_bins is None:
            raise ValueError("To remap, pass old_n_bins (e.g., 50)")
        argmax_list_early = _remap_bin_indices(argmax_list_early, old_n_bins, n_bins)
        argmax_list_late  = _remap_bin_indices(argmax_list_late,  old_n_bins, n_bins)
        argmin_list_early = _remap_bin_indices(argmin_list_early, old_n_bins, n_bins)
        argmin_list_late  = _remap_bin_indices(argmin_list_late,  old_n_bins, n_bins)

    # histogram bin edges in "bin index units"
    bins = np.arange(0, n_bins + 1)

    # ---- Outline-only hists ----
    ax.hist(np.asarray(argmax_list_early), bins=bins, histtype="step",
            linewidth=lw, alpha=alpha, color=colors_list[0])
    ax.hist(np.asarray(argmax_list_late),  bins=bins, histtype="step",
            linewidth=lw, alpha=alpha, color=colors_list[1])

    weights_early_min = mirror_weight * np.ones(len(argmin_list_early), dtype=float)
    weights_late_min  = mirror_weight * np.ones(len(argmin_list_late),  dtype=float)

    ax.hist(np.asarray(argmin_list_early), bins=bins, histtype="step",
            linewidth=lw, alpha=alpha, color=colors_list[2], linestyle="--",
            weights=weights_early_min)
    ax.hist(np.asarray(argmin_list_late),  bins=bins, histtype="step",
            linewidth=lw, alpha=alpha, color=colors_list[3], linestyle="--",
            weights=weights_late_min)

    # Zero line
    ax.axhline(0, color="k", linewidth=1)

    # ---- X ticks aligned to bin edges, labeled in cm ----
    # positions are in bin-index units [0..n_bins]
    tick_pos = np.linspace(0, n_bins, n_xticks)
    tick_lab = np.linspace(0, track_len_cm, n_xticks)
    ax.set_xlim(0, n_bins)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([f"{v:.0f}" for v in tick_lab])

    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Cluster Count")
    if title is not None:
        ax.set_title(f"{title} Peak / Trough Loc")

    # symmetric y
    y_min, y_max = ax.get_ylim()
    max_val = max(abs(y_min), abs(y_max)) or 1
    ax.set_ylim(-max_val, max_val)

    # y tick labels as absolute values
    n_ticks = 5
    pos_ticks = np.linspace(0, max_val, n_ticks + 1)
    ticks = np.concatenate([-pos_ticks[1:][::-1], [0.0], pos_ticks[1:]])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(abs(t))}" for t in ticks])

    # legends (proxy handles)
    handles_max = [
        Line2D([0], [0], color=colors_list[0], lw=lw, label="Early Peak"),
        Line2D([0], [0], color=colors_list[1], lw=lw, label="Late Peak"),
    ]
    legend_max = ax.legend(handles=handles_max, loc="upper right") #, title="Max Loc")
    ax.add_artist(legend_max)

    handles_min = [
        Line2D([0], [0], color=colors_list[2], lw=lw, linestyle="--", label="Early Trough"),
        Line2D([0], [0], color=colors_list[3], lw=lw, linestyle="--", label="Late Trough"),
    ]
    ax.legend(handles=handles_min, loc="lower right") # title="Min Loc")

    return ax




def get_argmin_argmax_list_learning_celltype(labels, residual_activity_dict_EC, cp_dict_EC):

    early_cp_argmax_list = []
    early_cp_argmin_list = []

    late_cp_argmax_list = []
    late_cp_argmin_list = []

    argmax_amp_list_early = [[] for _ in range(50)]
    argmin_amp_list_early = [[] for _ in range(50)]

    argmax_amp_list_late = [[] for _ in range(50)]
    argmin_amp_list_late = [[] for _ in range(50)]

    count = 0

    for idx, animal in enumerate(residual_activity_dict_EC):
        for idt, cell in enumerate(residual_activity_dict_EC[animal]):

            if count in labels:
                early_cp = cp_dict_EC[animal][cell][0]
                late_cp = cp_dict_EC[animal][cell][1]

                data_early = np.mean(residual_activity_dict_EC[animal][cell][:, :early_cp], axis=1)
                data_late = np.mean(residual_activity_dict_EC[animal][cell][:, late_cp:], axis=1)

                max_loc_early = np.argmax(data_early)
                min_loc_early = np.argmin(data_early)

                early_cp_argmax_list.append(max_loc_early)
                early_cp_argmin_list.append(min_loc_early)

                argmax_amp_list_early[max_loc_early].append(data_early[max_loc_early])
                argmin_amp_list_early[min_loc_early].append(data_early[min_loc_early])

                max_loc_late = np.argmax(data_late)
                min_loc_late = np.argmin(data_late)

                late_cp_argmax_list.append(max_loc_late)
                late_cp_argmin_list.append(min_loc_late)

                argmax_amp_list_late[max_loc_late].append(data_late[max_loc_late])
                argmin_amp_list_late[min_loc_late].append(data_late[min_loc_late])
                

            count+=1



    return early_cp_argmax_list, early_cp_argmin_list, late_cp_argmax_list, late_cp_argmin_list, argmax_amp_list_early, argmin_amp_list_early, argmax_amp_list_late, argmin_amp_list_late


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



def run(use_fixed_track, use_first_or_only, use_all, which_celltype=None):


    fig, axs = plt.subplots(4,4, figsize=(15,13))
    fig.subplots_adjust(hspace=0.9)

    if which_celltype=="NDNF":

        icolore = 'orange'

        filepath = '/Users/michaelfinch/CA1-interneuron-GLM/datasets/NDNF_E0A1B1_251107.mat'

        animal_clean_dict_activity, animal_vel_dict, animal_trials_original, animal_trials_clean, trials_to_remove_local, animal_lick_dict = get_animal_clean_dict_activity(filepath)

        GLM_params, predicted_activity_dict = fit_GLM_population(animal_vel_dict, animal_clean_dict_activity, quintile=None, regression='ridge', alphas=None)

        residual_activity_dict_NDNF_new = get_residual_activity_dict(animal_clean_dict_activity, predicted_activity_dict)



        
        
        ###### cell by cell slice tca models


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

                                fig.suptitle(f"All Fixed Sessions") #{animal_included_list}")
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

                                NDNF_model_dict_clean[animal_key] = NDNF_model_dict[animal] #[idx_for_model_clean]


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

        icolore = 'green'

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

        fig.suptitle(f"EC All Fixed Track Cells")


    elif which_celltype=="SST":

        icolore = 'blue'

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

        fig.suptitle(f"SST All Fixed Track Cells")




    def paired_r_plot(
        ax,
        r_raw_arr,
        r_resid_arr,
        title="",
        left_label="Activity \n vs Velocity",
        right_label="Spatial \n Component \n vs Velocity",
        seed=0,

        # points/jitter
        jitter_scale=0.06,        # <-- smaller jitter keeps points tighter
        color="green",

        # mean/sem bars
        show_mean=True,
        show_sem=True,
        mean_offset=0.42,         # <-- move mean bars farther OUTSIDE
        mean_bar_width=0.18,
        mean_lw=3.0,
        sem_cap_width=0.09,
        sem_lw=2.0,

        # extra safety padding
        bar_gap=0.10,             # <-- additional push outward
    ):
        rng = np.random.default_rng(seed)
        x0, x1 = 0.0, 2.0

        r_raw_arr   = np.asarray(r_raw_arr, dtype=float)
        r_resid_arr = np.asarray(r_resid_arr, dtype=float)

        # jitter around centers
        j0 = rng.normal(0, jitter_scale, size=len(r_raw_arr))
        j1 = rng.normal(0, jitter_scale, size=len(r_resid_arr))

        # paired lines
        for i in range(len(r_raw_arr)):
            ax.plot([x0 + j0[i], x1 + j1[i]],
                    [r_raw_arr[i], r_resid_arr[i]],
                    color="0.75", lw=0.6, alpha=0.6, zorder=1)

        # points
        ax.scatter(x0 + j0, r_raw_arr,
                s=18, color="k", alpha=0.85, edgecolors="none", zorder=2)
        ax.scatter(x1 + j1, r_resid_arr,
                s=18, color=color, alpha=0.85, edgecolors="none", zorder=3)

        if show_mean:
            def _mean_sem(y):
                y = np.asarray(y, float)
                y = y[np.isfinite(y)]
                if len(y) == 0:
                    return np.nan, np.nan
                mu = y.mean()
                sem = 0.0 if len(y) < 2 else y.std(ddof=1) / np.sqrt(len(y))
                return mu, sem

            mu0, sem0 = _mean_sem(r_raw_arr)
            mu1, sem1 = _mean_sem(r_resid_arr)

            # outside x positions (with extra gap)
            xm0 = x0 - (mean_offset + bar_gap)
            xm1 = x1 + (mean_offset + bar_gap)

            # mean bars
            ax.plot([xm0 - mean_bar_width/2, xm0 + mean_bar_width/2], [mu0, mu0],
                    color="k", lw=mean_lw, solid_capstyle="round", zorder=6)
            ax.plot([xm1 - mean_bar_width/2, xm1 + mean_bar_width/2], [mu1, mu1],
                    color=color, lw=mean_lw, solid_capstyle="round", zorder=7)

            # SEM whiskers
            if show_sem:
                # left
                ax.plot([xm0, xm0], [mu0 - sem0, mu0 + sem0], color="k", lw=sem_lw, zorder=6)
                ax.plot([xm0 - sem_cap_width/2, xm0 + sem_cap_width/2], [mu0 - sem0, mu0 - sem0],
                        color="k", lw=sem_lw, zorder=6)
                ax.plot([xm0 - sem_cap_width/2, xm0 + sem_cap_width/2], [mu0 + sem0, mu0 + sem0],
                        color="k", lw=sem_lw, zorder=6)

                # right
                ax.plot([xm1, xm1], [mu1 - sem1, mu1 + sem1], color=color, lw=sem_lw, zorder=7)
                ax.plot([xm1 - sem_cap_width/2, xm1 + sem_cap_width/2], [mu1 - sem1, mu1 - sem1],
                        color=color, lw=sem_lw, zorder=7)
                ax.plot([xm1 - sem_cap_width/2, xm1 + sem_cap_width/2], [mu1 + sem1, mu1 + sem1],
                        color=color, lw=sem_lw, zorder=7)

        # formatting
        ax.set_xticks([x0, x1])
        ax.set_xticklabels([left_label, right_label])
        ax.set_ylabel("R Value")
        ax.axhline(0, color="0.6", lw=1, alpha=0.6)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(title)

        # widen x-limits enough to show the outside summary bars
        pad = mean_offset + bar_gap + 0.35
        ax.set_xlim(x0 - pad, x1 + pad)

        lo = np.nanmin([np.nanmin(r_raw_arr), np.nanmin(r_resid_arr)])
        hi = np.nanmax([np.nanmax(r_raw_arr), np.nanmax(r_resid_arr)])
        ax.set_ylim(lo - 0.05, hi + 0.05)







    animals = sorted(clean_resid_activity_dict_NDNF_newest.keys(), key=lambda a: int(a.split("_")[1]))
    cmap = plt.get_cmap("tab20", len(animals))
    animal_to_color = {a: cmap(i) for i, a in enumerate(animals)}


    print(f"labels {labels}")
    
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

            
    


    
    # reassigned_dict = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=8, reassign_clusters=True, x00=True, umap=False, contiguous=False, ranks=20)


    contig_dict_all_cell_tca = preprocess_animal(NDNF_model_dict_clean, clean_resid_activity_dict_NDNF_newest, num_clusters=5, reassign_clusters=False, x00=True, umap=False, contiguous=True, ranks=20)

    

    contig_dict = reshape_contig_dict(contig_dict_all_cell_tca, NDNF_model_dict_clean)



    cp_dict_NDNF = get_cp_dict(contig_dict)

 
    # data_truncated_array_NDNF, min_num_trials = get_truncated_to_min_data_array(clean_resid_activity_dict_NDNF_newest)

   
    cells_per_animal_dict = get_cells_per_animal_dict(clean_resid_activity_dict_NDNF_newest)
   
    plot_cluster_traces_by_animal_labels(labels,
                                clean_resid_activity_dict_NDNF_newest,
                                animal_to_color, 
                                ncol=5, spacing=0.2,
                                title_prefix="", ax_list=[axs[0,0], axs[0,1]])
    
    # plt.figure()
    activity_early_array, activity_array_late, cp_early_as_fraction, cp_late_as_fraction = get_activity_cut_learn(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF)

    fig, ax = plt.subplots(figsize=(6, 4))

    data = [cp_early_as_fraction, cp_late_as_fraction]
    labels = ["Early", "Late"]

    bp = ax.boxplot(data, labels=labels, showmeans=True, showfliers=False)

    for i, y in enumerate(data, start=1):
        y = np.asarray(y)
        x = np.random.normal(i, 0.04, size=len(y))  # small jitter
        ax.plot(x, y, marker="o", linestyle="", alpha=0.6)

    ax.set_ylabel("CP (fraction)")
    ax.set_title("CP fraction: Early vs Late")
    plt.tight_layout()
    plt.show()


    # mean_activity_early_array = np.mean(activity_early_array, axis=0)
    # mean_activity_array_late = np.mean(activity_array_late, axis=0)

    # sem_activity_early_array = sem(activity_early_array, axis=0)
    # sem_activity_array_late = sem(activity_array_late, axis=0)

    # plt.plot(mean_activity_early_array, label='Early', color='cyan')
    # plt.fill_between(range(len(mean_activity_early_array)), mean_activity_early_array-sem_activity_early_array, mean_activity_early_array+sem_activity_early_array, alpha=0.2, color='cyan')
    # plt.plot(mean_activity_array_late, label='Late', color='blue')
    # plt.fill_between(range(len(mean_activity_array_late)), mean_activity_array_late-sem_activity_array_late, mean_activity_array_late+sem_activity_array_late, alpha=0.2, color='blue')
    # plt.ylabel("Z-Scored DF/F")
    # plt.xlabel("Position Bins")
    # plt.legend()
    # plt.show()


    color_list0 = ["orchid", "purple"]
    color_list1 = ["orange", "Crimson"]

    color_list_lists = [color_list0, color_list1]

    plot_clustered_data_learn(labels, activity_early_array, activity_array_late, K=2, title="Cued Track Residuals NDNF Clustered Changepoint", ax_list=[axs[2,0],axs[2,1]], color_list_lists=color_list_lists)



    color_list = ["purple", "red"]

    # color_list = ["orchid", "purple"]



    print(f"cells_per_animal_dict {cells_per_animal_dict}")

    plot_cluster_animal_composition_stacked_from_index(
    labels,
    animal_to_color, 
    clean_resid_activity_dict_NDNF_newest,
    K=2,
    title_prefix="",
    show_percent_labels=True,
    ax=axs[1,0])


    plot_lick_vel_data_clust(labels, clean_velocity_dict_NDNF_newest, cells_per_animal_dict, use_vel=True, num_clusters=2, title="Velocity", ax=axs[1,2], color_list=color_list)
    plot_lick_vel_data_clust(labels, clean_lick_dict_NDNF_newest, cells_per_animal_dict, use_vel=False, num_clusters=2, title="Licks", ax=axs[1,1], color_list=color_list)

    which_vectors=1

    w1 = sliceTCA_model.vectors[which_vectors][0]
    f1 = sliceTCA_model.vectors[which_vectors][1]

    print(f"w1.shape {w1.shape} f1.shape {f1.shape}")


    w1_subset = w1[:, idx_of_interest]  

    
    # labels = np.asarray(labels_dict_raw_new[2])
    n_latents_expected = 20
    n_cells_expected = len(labels)  

    X = get_cell_features(
        w1_subset, f1,
        feature_mode="0x0",
        n_cells_expected=n_cells_expected,
        n_latents_expected=n_latents_expected,)


    
    if X.shape[0] != len(labels) and X.shape[1] == len(labels):
        X = X.T
    elif X.shape[0] != len(labels):
        raise ValueError(f"Shape mismatch: X.shape={X.shape}, labels={len(labels)}")



    uniq = np.unique(labels)
    X_2d = lda_with_orthogonal_axis_2d(X, labels, title_prefix="")

    # X2, w, orth, X_perp = debug_lda_orth_geometry(X, labels, title="My dataset")

    for i, k in enumerate(uniq):
        m = labels == k
        axs[0,2].scatter(X_2d[m, 0], X_2d[m, 1], s=40, alpha=0.6, color=color_list[i], label=f"C{k} (n={m.sum()})")

    axs[0,2].set_xlabel("LDA Component 1")
    axs[0,2].set_ylabel("PC1 of LDA-subtracted residuals")

    leg = axs[0,2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
    fig.subplots_adjust(right=0.78)

    cells_list_0 = np.where(labels==0)[0]
    cells_list_1 = np.where(labels==1)[0]


    animal_average_selectivity_dict_NDNF_0, animals_dict_data_NDNF_0 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_list_0, neg_sel=False, trial_av=True, norm="min_max")
    animal_average_selectivity_dict_NDNF_1, animals_dict_data_NDNF_1 = get_selectivity_each_trial_cell_type(clean_resid_activity_dict_NDNF_newest, cells_list_1, neg_sel=False, trial_av=True, norm="min_max")


    binned_data_NDNF_0 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0, n_bins=10)
    binned_data_NDNF_1 = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1, n_bins=10)

    plot_the_CDF_celltypes(binned_data_NDNF_0, binned_data_NDNF_1, title="Selectivity Distribution", n_bins = 10, ax=axs[2,2])



    
    animal_average_selectivity_dict_NDNF_0_early,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="early", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_early,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="early", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_late,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="late", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_late,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="late", norm="min_max")

    animal_average_selectivity_dict_NDNF_0_middle,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_0, neg_sel=False, trial_av=True, eml="middle", norm="min_max")
    animal_average_selectivity_dict_NDNF_1_middle,_ = get_selectivity_each_trial_early_late_cluster(clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF, cells_list_1, neg_sel=False, trial_av=True, eml="middle", norm="min_max")

    print(f"animal_average_selectivity_dict_NDNF_1_middle {animal_average_selectivity_dict_NDNF_1_middle}")


    binned_data_NDNF_0_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_early, n_bins=10)
    binned_data_NDNF_1_early = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_early, n_bins=10)

    binned_data_NDNF_0_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_0_late, n_bins=10)
    binned_data_NDNF_1_late = get_binned_data_for_CDF(animal_average_selectivity_dict_NDNF_1_late, n_bins=10)


    plot_the_CDF_early_late(binned_data_NDNF_0_early, binned_data_NDNF_1_early, binned_data_NDNF_0_late, binned_data_NDNF_1_late, title="", n_bins = 10, ax=axs[2,3])



    percentile_slices_NDNF0 = get_percentlie_slices(animals_dict_data_NDNF_0)
    percentile_slices_NDNF1 = get_percentlie_slices(animals_dict_data_NDNF_1)
    

    all_cells_NDNF_0 = selectivity_from_percentile_slices(percentile_slices_NDNF0, norm='min_max', neg_sel=False)
    all_cells_NDNF_1 = selectivity_from_percentile_slices(percentile_slices_NDNF1, norm='min_max', neg_sel=False)

    plot_selectivity_over_trials(all_cells_NDNF_0, all_cells_NDNF_1, color_list=color_list, ax=axs[1,3])


    plot_eml_data(animal_average_selectivity_dict_NDNF_0_early, animal_average_selectivity_dict_NDNF_1_early, animal_average_selectivity_dict_NDNF_0_middle, animal_average_selectivity_dict_NDNF_1_middle, animal_average_selectivity_dict_NDNF_0_late, animal_average_selectivity_dict_NDNF_1_late, ax=axs[0,3], color_list=color_list)

    data, min_val, idx_to_key = get_truncated_to_min_data_array(clean_resid_activity_dict_NDNF_newest)
    print("First 10 rows in TA correspond to:")
    for i in range(10):
        a, c = idx_to_key[i]
        print(i, a, c, "shape", clean_resid_activity_dict_NDNF_newest[a][c].shape)


    early_cp_argmax_list_0, early_cp_argmin_list_0, late_cp_argmax_list_0, late_cp_argmin_list_0, argmax_amp_list_early_0, argmin_amp_list_early_0, argmax_amp_list_late_0, argmin_amp_list_late_0 = get_argmin_argmax_list_learning_celltype(cells_list_0, clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF)
    
    early_cp_argmax_list_1, early_cp_argmin_list_1, late_cp_argmax_list_1, late_cp_argmin_list_1, argmax_amp_list_early_1, argmin_amp_list_early_1, argmax_amp_list_late_1, argmin_amp_list_late_1 = get_argmin_argmax_list_learning_celltype(cells_list_1, clean_resid_activity_dict_NDNF_newest, cp_dict_NDNF)
    
    plot_butterfly_hist(early_cp_argmax_list_0, late_cp_argmax_list_0, early_cp_argmin_list_0, late_cp_argmin_list_0, ax=axs[3,0], n_bins=10, old_n_bins=50, remap=True, colors_list = ["orchid", "purple", "orchid", "purple"], title="Cluster 0")

    plot_butterfly_hist(early_cp_argmax_list_1, late_cp_argmax_list_1, early_cp_argmin_list_1, late_cp_argmin_list_1, ax=axs[3,2], n_bins=10, old_n_bins=50, remap=True, colors_list = ["orange", "red", "orange", "red"], title="Cluster 1")

    bins_per_group=5

    means_list_early_0, sems_list_early_0 = mean_and_sem(argmax_amp_list_early_0)
    means_list_late_0, sems_list_late_0 = mean_and_sem(argmax_amp_list_late_0)
    means_list_early_1, sems_list_early_1 = mean_and_sem(argmax_amp_list_early_1)
    means_list_late_1, sems_list_late_1 = mean_and_sem(argmax_amp_list_late_1)

    means_list_early_0min, sems_list_early_0min = mean_and_sem(argmin_amp_list_early_0)
    means_list_late_0min, sems_list_late_0min = mean_and_sem(argmin_amp_list_late_0)
    means_list_early_1min, sems_list_early_1min = mean_and_sem(argmin_amp_list_early_1)
    means_list_late_1min, sems_list_late_1min = mean_and_sem(argmin_amp_list_late_1)

    x10_0_e_max, m10_0_e_max, s10_0_e_max = rebin_means_sems(means_list_early_0,  sems_list_early_0, bins_per_group=bins_per_group)
    _,         m10_0_l_max, s10_0_l_max   = rebin_means_sems(means_list_late_0,   sems_list_late_0, bins_per_group=bins_per_group)

    _,         m10_0_e_min, s10_0_e_min   = rebin_means_sems(means_list_early_0min, sems_list_early_0min, bins_per_group=bins_per_group)
    _,         m10_0_l_min, s10_0_l_min   = rebin_means_sems(means_list_late_0min,  sems_list_late_0min, bins_per_group=bins_per_group)

    x10_1_e_max, m10_1_e_max, s10_1_e_max = rebin_means_sems(means_list_early_1,  sems_list_early_1, bins_per_group=bins_per_group)
    _,         m10_1_l_max, s10_1_l_max   = rebin_means_sems(means_list_late_1,   sems_list_late_1, bins_per_group=bins_per_group)

    _,         m10_1_e_min, s10_1_e_min   = rebin_means_sems(means_list_early_1min, sems_list_early_1min, bins_per_group=bins_per_group)
    _,         m10_1_l_min, s10_1_l_min   = rebin_means_sems(means_list_late_1min,  sems_list_late_1min, bins_per_group=bins_per_group)


    n = len(m10_0_e_max)

    # If these are 10 coarse bins spanning the full 180 cm, use bin CENTERS:
    x = (np.arange(n) + 0.5) * (180.0 / n)

    
    # axs[3,1].errorbar(x, m10_0_e_max, yerr=s10_0_e_max, label='Early Max', capsize=3, marker='o', color="orchid")
    
    # axs[3,1].errorbar(x, m10_0_l_max, yerr=s10_0_l_max, label='Late Max', capsize=3, marker='o', color="purple")
    
    # axs[3,1].errorbar(x, m10_0_e_min, yerr=s10_0_e_min,label='Early Min', capsize=3, marker='o', color="orchid")
    
    # axs[3,1].errorbar(x, m10_0_l_min, yerr=s10_0_l_min,label='Late Min', capsize=3, marker='o', color="purple")
    
    # axs[3,1].set_xlabel("Position (cm)")
    # x = np.linspace(0, 180, 10)
    # axs[3,1].set_ylabel("dF/F amplitude")
    # axs[3,1].set_title("Cell Type 0 Max/Min Amplitude")
    # axs[3,1].set_xticks(range(len(x)), x)
    # axs[3,1].legend()

    K = len(m10_0_e_max)              # number of coarse bins you ended up with
    track_len = 180.0
    edges = np.linspace(0, track_len, K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])   # e.g. [18,54,90,126,162] if K=5

    axs[3,1].errorbar(centers, m10_0_e_max, yerr=s10_0_e_max, label='Early Max Amp', capsize=3, marker='o', color="orchid")
    axs[3,1].errorbar(centers, m10_0_l_max, yerr=s10_0_l_max, label='Late Max Amp',  capsize=3, marker='o', color="purple")
    axs[3,1].errorbar(centers, m10_0_e_min, yerr=s10_0_e_min, label='Early Min Amp', capsize=3, marker='o', color="orchid")
    axs[3,1].errorbar(centers, m10_0_l_min, yerr=s10_0_l_min, label='Late Min Amp',  capsize=3, marker='o', color="purple")

    axs[3,1].set_xlim(0, track_len)

    # Optional: show coarse bin EDGES as ticks (more “truthful” for rebinned data)
    axs[3,1].set_xticks(edges)
    axs[3,1].set_xticklabels([f"{t:.0f}" for t in edges])

    axs[3,1].set_xlabel("Position (cm)")
    axs[3,1].set_ylabel("dF/F amplitude")
    axs[3,1].set_title("Cell Type 0 Max/Min Amplitude")
    axs[3,1].legend()


   
    axs[3,3].errorbar(centers, m10_1_e_max, yerr=s10_1_e_max,label='Early Max Amp', capsize=3, marker='o', color="orange")
    axs[3,3].errorbar(centers, m10_1_l_max, yerr=s10_1_l_max,label='Late Max Amp', capsize=3, marker='o', color="red")
    axs[3,3].errorbar(centers, m10_1_e_min, yerr=s10_1_e_min,label='Early Min Amp', capsize=3, marker='o', color="orange")
    axs[3,3].errorbar(centers, m10_1_l_min, yerr=s10_1_l_min,label='Late Min Amp', capsize=3, marker='o', color="red")

    # axs[3,3].set_xlabel("Coarse position bin")
    # axs[3,3].set_ylabel("dF/F amplitude")
    # axs[3,3].set_title("Cell Type 1 Max/Min Amplitude")
    # axs[3,3].legend() 

    axs[3,3].set_xlim(0, track_len)

    # Optional: show coarse bin EDGES as ticks (more “truthful” for rebinned data)
    axs[3,3].set_xticks(edges)
    axs[3,3].set_xticklabels([f"{t:.0f}" for t in edges])

    axs[3,3].set_xlabel("Position (cm)")
    axs[3,3].set_ylabel("dF/F amplitude")
    axs[3,3].set_title("Cell Type 1 Max/Min Amplitude")
    axs[3,3].legend()



        
    
    
    plt.tight_layout()

    plt.show()


    animal_for_each_cell_list = []

    for animal in clean_resid_activity_dict_NDNF_newest:
        for cell in clean_resid_activity_dict_NDNF_newest[animal]:
            animal_for_each_cell_list.append(animal)
            

    p = np.ones(50)
    z = np.zeros((10,20))

    print(f"sliceTCA_model.vectors[1][0].shape {sliceTCA_model.vectors[1][0].shape}")

    
    W = sliceTCA_model.vectors[1][0]              # torch.Size([20, 792])

    X = W.detach().cpu().numpy().T                # (792, 20) -> rows=cells, cols=components

    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(X)  

    # # np.concatentate([p,z])

    # fig = plt.figure(figsize=(6, 5))
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=10, alpha=0.8)

    # m0 = labels == 0
    # m1 = labels == 1

    # color_dict = {"EC":['green', 'black'],
    # "SST":['cyan', 'blue'],
    # "NDNF":['orange', 'red']}

    # ax.scatter(Z[m0, 0], Z[m0, 1], Z[m0, 2], s=12, alpha=0.8, label="Cluster 0", color=color_dict[which_celltype][0])
    # ax.scatter(Z[m1, 0], Z[m1, 1], Z[m1, 2], s=12, alpha=0.8, label="Cluster 1", color=color_dict[which_celltype][1])

    # ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    # ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    # ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    # ax.tick_params(axis='x', labelsize=10)
    # ax.tick_params(axis='y', labelsize=10)
    # ax.tick_params(axis='z', labelsize=10)   # optional but usually needed too
    # ax.set_title(which_celltype)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()



    labels = np.asarray(labels).astype(int)
    animals = np.asarray(animal_for_each_cell_list)

    assert len(labels) == len(animals) == Z.shape[0]

    # --- map each animal -> color (categorical colormap) ---
    uniq_animals = np.unique(animals)
    cmap = plt.get_cmap("tab20")  # good categorical map (up to ~20 distinct colors)
    animal_to_color = {a: cmap(i % cmap.N) for i, a in enumerate(uniq_animals)}
    colors = np.array([animal_to_color[a] for a in animals], dtype=object)

    m0 = labels == 0
    m1 = labels == 1

    animals_celltype0_mask = animals[m0]
    animals_celltype1_mask = animals[m1]

    print(f"animals_celltype0_mask {animals_celltype0_mask}")
    print(f"animals_celltype1_mask {animals_celltype1_mask}")

    unique_animals_celltype0_mask = np.unique(animals_celltype0_mask)
    unique_animals_celltype1_mask = np.unique(animals_celltype1_mask)

    overall_num_animals = len(uniq_animals)

    print(f"len(unique_animals_celltype0_mask) {len(unique_animals_celltype0_mask)} len(unique_animals_celltype1_mask) {len(unique_animals_celltype1_mask)}")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # cluster 0 = circles, colored by animal
    # ax.scatter(
    #     Z[m0, 0], Z[m0, 1], Z[m0, 2],
    #     c=colors[m0].tolist(), marker="o", s=30, alpha=0.85,
    #     edgecolors="none"
    # )

    # # cluster 1 = triangles, colored by animal
    # ax.scatter(
    #     Z[m1, 0], Z[m1, 1], Z[m1, 2],
    #     c=colors[m1].tolist(), marker="^", s=43, alpha=0.85,
    #     edgecolors="none"
    # )

    colors_dict = {"EC":["k", 'green'],
    "SST":["cyan", 'blue'],
    "NDNF":["purple", 'red']}

    ax.scatter(
        Z[m0, 0], Z[m0, 1], Z[m0, 2],
        c=colors_dict[which_celltype][0], marker="o", s=30, alpha=0.85,
        edgecolors="none", label=f"Cluster 0: Animals Included {len(unique_animals_celltype0_mask)} / {overall_num_animals}"
    )

    # cluster 1 = triangles, colored by animal
    ax.scatter(
        Z[m1, 0], Z[m1, 1], Z[m1, 2],
        c=colors_dict[which_celltype][1], marker="^", s=43, alpha=0.85,
        edgecolors="none", label=f"Cluster 1: Animals Included {len(unique_animals_celltype1_mask)} / {overall_num_animals}"
    )


    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", labelpad=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", labelpad=10)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", labelpad=10)
    ax.tick_params(labelsize=10, pad=2)
    ax.set_title(which_celltype)
    ax.legend()

    plt.show()





    fig, axs = plt.subplots(2,3, figsize=(10,7))
    fig.suptitle(which_celltype)

    residual_list = []
    vel_prediction_list = []
    raw_activity_list = []

    r_residual_list = []
    r_raw_list = []

    for animal in clean_resid_activity_dict_NDNF_newest:
        for cell in clean_resid_activity_dict_NDNF_newest[animal]:
            residual_list.append(np.mean(clean_resid_activity_dict_NDNF_newest[animal][cell], axis=1))
            vel_prediction_list.append(np.mean(predicted_activity_dict[animal][cell], axis=1))
            raw_activity_list.append(np.mean(animal_clean_dict_activity[animal][cell], axis=1))

            residual_flat = clean_resid_activity_dict_NDNF_newest[animal][cell].flatten()
            vel_flat = clean_velocity_dict_NDNF_newest[animal]["Velocity"].flatten()
            raw_flat = animal_clean_dict_activity[animal][cell].flatten()

            # r_residual, _ = pearsonr(residual_flat,vel_flat)
            # r_residual_list.append(r_residual)
            # r_raw, _ = pearsonr(raw_flat,vel_flat)
            # r_raw_list.append(r_raw)


    r_residual_list = []
    r_raw_list = []
    real_vel_list = []

    for animal in clean_resid_activity_dict_NDNF_newest:
        vel_flat_full = np.asarray(clean_velocity_dict_NDNF_newest[animal]["Velocity"]).flatten()
        real_vel_list.append(np.mean(clean_velocity_dict_NDNF_newest[animal]["Velocity"], axis=1))

        for cell in clean_resid_activity_dict_NDNF_newest[animal]:
            residual_flat_full = np.asarray(clean_resid_activity_dict_NDNF_newest[animal][cell]).flatten()
            raw_flat_full      = np.asarray(animal_clean_dict_activity[animal][cell]).flatten()

            L = min(len(vel_flat_full), len(residual_flat_full), len(raw_flat_full))
            v  = vel_flat_full[:L]
            rr = residual_flat_full[:L]
            rw = raw_flat_full[:L]

            # require BOTH to be valid so they stay paired
            m = np.isfinite(v) & np.isfinite(rw) & np.isfinite(rr)
            if m.sum() < 3:
                continue

            r_raw, _   = pearsonr(rw[m], v[m])
            r_resid, _ = pearsonr(rr[m], v[m])

            r_raw_list.append(r_raw)
            r_residual_list.append(r_resid)



    paired_r_plot(
    axs[1,0],
    r_raw_list,
    r_residual_list,
    title=None,
    color=icolore,
    show_mean=True,
    show_sem=True,      # or False if you only want a mean bar
)



    residual_array = np.array(residual_list)
    vel_prediction_array = np.array(vel_prediction_list)
    raw_activity_array = np.array(raw_activity_list)

    im = axs[0,0].imshow(raw_activity_array, aspect='auto')
    axs[0,0].set_title("Raw")
    axs[0,0].set_ylabel("Cell ID")
    axs[0,0].set_xlabel("Position Bins")
    fig.colorbar(im, ax=axs[0,0], label="Z-Scored DF/F")

    im = axs[0,1].imshow(vel_prediction_array, aspect='auto')
    axs[0,1].set_title("Velocity Contribution")
    axs[0,1].set_ylabel("Cell ID")
    axs[0,1].set_xlabel("Position Bins")
    fig.colorbar(im, ax=axs[0,1], label="Z-Scored DF/F")

    im = axs[0,2].imshow(residual_array, aspect='auto')
    axs[0,2].set_title("Spatial Component")
    axs[0,2].set_ylabel("Cell ID")
    axs[0,2].set_xlabel("Position Bins")
    fig.colorbar(im, ax=axs[0,2], label="Z-Scored DF/F")

    mean_residual_array = np.mean(residual_array, axis=0)
    mean_vel_prediction_array = np.mean(vel_prediction_array, axis=0)
    mean_raw_activity_array = np.mean(raw_activity_array, axis=0)

    sem_residual_array = sem(residual_array, axis=0)
    sem_vel_prediction_array = sem(vel_prediction_array, axis=0)
    sem_raw_activity_array = sem(raw_activity_array, axis=0)

    axs[1,2].plot(mean_raw_activity_array, color='k', label="Raw")
    axs[1,2].fill_between(range(len(mean_raw_activity_array)), mean_raw_activity_array-sem_raw_activity_array, mean_raw_activity_array+sem_raw_activity_array, alpha=0.2, color='k')
    axs[1,2].set_title("Raw Activity")
    axs[1,2].set_ylabel("Z-Scored DF/F")
    axs[1,2].set_xlabel("Position Bins")

    axs[1,2].plot(mean_residual_array, color=icolore, label="Spatial Component")
    axs[1,2].fill_between(range(len(mean_residual_array)), mean_residual_array-sem_residual_array, mean_residual_array+sem_residual_array, alpha=0.2, color=icolore)
    axs[1,2].set_title("Spatial Component")
    axs[1,2].set_ylabel("Z-Scored DF/F")
    axs[1,2].set_xlabel("Position Bins")
    axs[1,2].set_ylim(-0.4, 0.4)
    axs[1,2].legend()

    # axs[1,1].plot(mean_vel_prediction_array, color='r')
    # axs[1,1].fill_between(range(len(mean_vel_prediction_array)), mean_vel_prediction_array-sem_vel_prediction_array, mean_vel_prediction_array+sem_vel_prediction_array, alpha=0.2, color='r')
    # axs[1,1].set_title("Velocity Contribution")
    # axs[1,1].set_ylabel("Z-Scored DF/F")
    # axs[1,1].set_xlabel("Position Bins")
    # axs[1,1].set_ylim(-0.5, 0.25)

    real_vel_array = np.array(real_vel_list)
    mean_real_vel_array = np.mean(real_vel_array, axis=0)
    sem_real_vel_array = sem(real_vel_array, axis=0)

    # for i in range(len(real_vel_list)):
    axs[1,1].plot(mean_real_vel_array, color='r', linewidth=2)
    axs[1,1].fill_between(range(len(mean_real_vel_array)), mean_real_vel_array+sem_real_vel_array, mean_real_vel_array-sem_real_vel_array, alpha=0.2, color='r')
    axs[1,1].set_xlabel("Position Bins")
    axs[1,1].set_ylabel("Meters / Sec")
    axs[1,1].set_title("Velocity")




    
    plt.tight_layout()
    plt.show()









@click.command()
@click.option('--use_fixed_track/--use_cued_track', default=True, help="Use the 'most expressed' scanning logic.")
@click.option('--use_first_or_only/--use_mixed_track', default=True, help="Use the Final NDNF data")
@click.option('--use_all/--use_some', default=True, help="Use the Final NDNF data")


def cli(use_fixed_track, use_first_or_only, use_all):
    run(use_fixed_track, use_first_or_only, use_all, which_celltype="NDNF")

if __name__ == "__main__":
    cli()