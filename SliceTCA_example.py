import sys
from scipy.stats import sem
sys.path.append('/Users/michaelfinch/CA1-interneuron-GLM')

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *


from modelling_to_date_utils import *
from SliceTCA_example import *

def plot_synthetic_sliceTCA_ex(model, ideal_cell):
    weights = torch.abs(model.vectors[0][0].detach().T).cpu().numpy()
    print(weights.shape)
    latents = torch.abs(model.vectors[0][1].detach().T).cpu().numpy()
    latents = np.squeeze(latents, axis=1)
    print(latents.shape)

    plt.figure(figsize=(8,4))
    for i in range(weights.shape[1]):
        plt.plot(weights[:,i])
        plt.title("Weights Over Trials")
    plt.xlabel("Trials")
    plt.show()

    plt.figure(figsize=(8,4))
    for i in range(latents.shape[1]):
        plt.plot(latents[:,i])
        plt.title("Latents Over Position Bins (Fields)")
    plt.xlabel("Posiiton Bins")
    plt.show()

    weight_1 = weights[:,0]
    weight_2 = weights[:,1]

    # Add jitter to the points for better visualization
    rng = np.random.default_rng(0)
    jitter_scale = 0.02  # Adjust as needed
    weight_1_jitter = weight_1 + rng.normal(scale=jitter_scale, size=weight_1.shape)
    weight_2_jitter = weight_2 + rng.normal(scale=jitter_scale, size=weight_2.shape)


    weights_for_kmeans = np.column_stack([weight_1, weight_2])

    k = 4  
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(weights_for_kmeans)

    import matplotlib.patches as mpatches

    # Map each label to a color from tab10, ensuring color order matches cluster label
    tab10 = plt.cm.get_cmap('tab10')
    unique_labels = np.unique(labels)
    label_to_color = {lbl: tab10(int(lbl) % 10) for lbl in unique_labels}
    point_colors = [label_to_color[lbl] for lbl in labels]

    # Plot, colored by cluster
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(weight_1_jitter, weight_2_jitter, c=point_colors, alpha=0.8)

    # Create legend handles for each cluster, matching the scatter colors
    handles = [mpatches.Patch(color=label_to_color[lbl], label=f'Cluster {lbl}') for lbl in unique_labels]

    plt.xlabel("Weight 1 (with jitter)")
    plt.ylabel("Weight 2 (with jitter)")
    plt.title("KMeans Clusters in Weight Space")
    plt.legend(handles=handles, title="Cluster", loc='best')
    plt.show()



    def plot_data(data_3, title=None, ax=None):
        ax.imshow(data_3, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("Position Bins")
        ax.set_ylabel("Trials")

    labels_0 = np.where(labels==0)[0]
    labels_1 = np.where(labels==1)[0]
    labels_2 = np.where(labels==2)[0]
    labels_3 = np.where(labels==3)[0]

    data_0 = ideal_cell[labels_0, :, :].squeeze(axis=1)
    data_1 = ideal_cell[labels_1, :, :].squeeze(axis=1)
    data_2 = ideal_cell[labels_2, :, :].squeeze(axis=1)
    data_3 = ideal_cell[labels_3, :, :].squeeze(axis=1)

    fig, axs = plt.subplots(2, 2)
    plot_data(data_0, title="Cluster 0", ax=axs[1,1])
    plot_data(data_1, title="Cluster 1", ax=axs[0,0])
    plot_data(data_2, title="Cluster 2", ax=axs[0,1])
    plot_data(data_3, title="Cluster 3", ax=axs[1,0])

    plt.tight_layout()
    plt.show()


    labels = np.asarray(labels)
    n = labels.size
    trials = np.arange(n)

    jitter = 0.001

    # small vertical jitter for aesthetics
    y = np.random.uniform(-jitter, jitter, size=n)

    # use existing label_to_color mapping
    point_colors = [label_to_color[lbl] for lbl in labels]

    plt.figure(figsize=(8, 1.4))
    plt.scatter(trials, y, s=18, c=point_colors)
    plt.yticks([])  
    plt.ylim(-0.12, 0.12)
    plt.xlabel("Trial index")
    plt.title("Trials Labelled by Cluster")

    # legend using same mapping
    handles = [mpatches.Patch(color=label_to_color[lbl], label=f'Cluster {lbl}') 
               for lbl in sorted(label_to_color.keys())]
    # plt.legend(handles=handles, title="Cluster", ncols=min(len(label_to_color), 5), 
            #    loc="upper right", bbox_to_anchor=(1.0, 1.55), frameon=False)

    plt.tight_layout()
    plt.show()

def run_slice_tca_ideal_ex(ideal_cell):
  #### Run sliceTCA ####
  device = ('cuda' if torch.cuda.is_available() else 'cpu')
  neural_data_tensor = ideal_cell

  components, model = slicetca.decompose(ideal_cell,
                                        number_components=(2,0,0), # (trials, neurons, time bins)
                                        positive=True,
                                        learning_rate=2*10**-3,
                                        min_std=10**-5,
                                        max_iter=15_000,
                                        seed=0)

  plt.figure(figsize=(4,3), dpi=100)
  plt.plot(np.arange(1000,len(model.losses)), model.losses[1000:], 'k')
  plt.xlabel('iterations')
  plt.ylabel('mean squared error')
  plt.xlim(0,len(model.losses))
  plt.tight_layout()

  axes = slicetca.plot(model,
                variables=('trial', 'neuron', 'time'),
              #   ticks=(None, None, np.linspace(0,50,3)), # we only want to modify the time ticks
              #   tick_labels=(None, None, np.linspace(0,50,3)),
              #   sorting_indices=(None, neuron_sorting_peak_time, None),
                quantile=0.99)
  
  return model 


def get_ideal_random_cell(num_trials=None, num_timebins=None):
    rng = np.random.default_rng(seed=42)
    num_trials = num_trials
    num_timebins = num_timebins
    ideal_cell = torch.zeros(num_trials, 1, num_timebins)

    # Add gaussian place field
    def place_field(x, mu, sigma):
        return torch.exp(-(x - mu)**2 / (2 * sigma**2))

    field1 = place_field(torch.arange(num_timebins), mu=15, sigma=3)
    field2 = place_field(torch.arange(num_timebins), mu=35, sigma=3)

    x = np.linspace(-6, 6, num_trials)

    for i in range(num_trials):
        express_field1 = rng.choice([0, 1])
        express_field2 = rng.choice([0, 1])
        ideal_cell[i] += express_field1*field1 + express_field2*field2


    plt.imshow(ideal_cell[:, 0, :], interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xlabel('time/spatial bin')
    plt.ylabel('trial')
    plt.title("Synthetic Cell 2 Latents")
    plt.show()

    return ideal_cell

def plot_model_vs_reco(example_EC_cell_tensor, model_60, x=60):
    reconstruction = model_60.construct().numpy(force=True).squeeze(axis=1)
    original = example_EC_cell_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1,1, figsize=(5,6))

    im = axs.imshow(original, aspect='auto')
    axs.set_ylabel("Trials")
    axs.set_xlabel("Position Bins")
    axs.set_title("Raw Data", fontsize=14)
    fig.colorbar(im, ax=axs)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,1, figsize=(5,6))

    axs.set_ylabel("Trials")
    axs.set_xlabel("Position Bins")
    axs.set_title(f"SliceTCA {x} Latents Reconstruction", fontsize=14)
    im = axs.imshow(reconstruction, aspect='auto')
    fig.colorbar(im, ax=axs)
    plt.tight_layout()
    plt.show()

def plot_example_weights_latents_real(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=None, cell=None):

    model_ex_cell = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][0]
    latents = torch.abs(model_ex_cell.vectors[0][0].detach().T).cpu().numpy()
    plt.figure(figsize=(10,4))
    for i in range(latents.shape[1]):
        plt.plot(latents[:,i])
    plt.xlabel("Trials")
    plt.title("20 Weights (1/Latent) SliceTCA Example EC Cell")
    plt.show()

    model_ex_cell = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][0]
    latents = torch.abs(model_ex_cell.vectors[0][1].detach().T).cpu().numpy()
    latent_array = latents.squeeze(axis=1)

    plt.figure(figsize=(10,4))
    for i in range(latent_array.shape[1]):
        plt.plot(latent_array[:,:])
    plt.title("20 Latent Factors (Fields)")
    plt.xlabel("Position Bins")
    plt.show()


def plot_real_activity_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal=None, cell=None):

    labels = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][1][f"cell_{cell}"]["indices_for_cluster_number"]["clusters_chosen_4"]

    activity = residual_activity_dict_EC[f'animal_{animal+1}'][f'cell_{cell+1}']

    group_0_act = activity[:, labels[0]]
    group_1_act = activity[:, labels[1]]
    group_2_act = activity[:, labels[2]]
    group_3_act = activity[:, labels[3]]

    plt.plot(np.mean(group_0_act,axis=1), label='Cluster 0')
    plt.plot(np.mean(group_1_act,axis=1), label='Cluster 1')
    plt.plot(np.mean(group_2_act,axis=1), label='Cluster 2')
    plt.plot(np.mean(group_3_act,axis=1), label='Cluster 3')
    plt.ylabel("Z-Scored Activity")
    plt.xlabel("Position Bin")
    plt.title("Trial Averaged Activity For Clusters")
    plt.show()


    fig, axs = plt.subplots(1,2, figsize=(9,4))
    im=axs[0].imshow(group_0_act.T, aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.mean(group_0_act,axis=1), color='blue')
    axs[0].set_ylabel("Trial")
    axs[1].set_ylabel("Z-Scored Activity")
    axs[0].set_xlabel("Position Bins")
    axs[1].set_xlabel("Position Bins")
    axs[0].set_title("Cluster 0 Activity")
    axs[1].set_title("Trial Av. Activity For Cluster")
    axs[1].set_ylim(-0.5, 4.5)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(9,4))
    im=axs[0].imshow(group_1_act.T, aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.mean(group_1_act,axis=1), color='orange')
    axs[0].set_ylabel("Trial")
    axs[1].set_ylabel("Z-Scored Activity")
    axs[0].set_xlabel("Position Bins")
    axs[1].set_xlabel("Position Bins")
    axs[0].set_title("Cluster 1 Activity")
    axs[1].set_title("Trial Av. Activity For Cluster")
    axs[1].set_ylim(-0.5, 4.5)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(9,4))
    im=axs[0].imshow(group_2_act.T, aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.mean(group_2_act,axis=1), color='green')
    axs[0].set_ylabel("Trial")
    axs[1].set_ylabel("Z-Scored Activity")
    axs[0].set_xlabel("Position Bins")
    axs[1].set_xlabel("Position Bins")
    axs[0].set_title("Cluster 2 Activity")
    axs[1].set_title("Trial Av. Activity For Cluster")
    axs[1].set_ylim(-0.5, 4.5)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(9,4))
    im=axs[0].imshow(group_3_act.T, aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].plot(np.mean(group_3_act,axis=1), color='red')
    axs[0].set_ylabel("Trial")
    axs[1].set_ylabel("Z-Scored Activity")
    axs[0].set_xlabel("Position Bins")
    axs[1].set_xlabel("Position Bins")
    axs[0].set_title("Cluster 3 Activity")
    axs[1].set_title("Trial Av. Activity For Cluster")
    axs[1].set_ylim(-0.5, 4.5)
    plt.tight_layout()
    plt.show()


###### every cell will have multiple clusters and the number of those clusters will be chosen by the elbow method -- we take all those and put them togther here, average across trials in each cluster and then plot the means 
def plot_peak_trough_histograms(clusters_dict_NDNF_early, use_argmax=True, title=None, ylim=None, ax=None, color=None, alpha=None):
    all_means_list = []

    for animal in clusters_dict_NDNF_early:
        for cell in clusters_dict_NDNF_early[animal]:
            for i in range(len(clusters_dict_NDNF_early[animal][cell])):
                data = clusters_dict_NDNF_early[animal][cell][i]
                # if np.mean(data != 0):
                mean_data = np.mean(data, axis=0)
                if use_argmax:
                    all_means_list.append(np.argmax(mean_data))
                    ax.set_title(f"Position of Peak Mean Activity in Cluster {title}", fontsize=10)
                else:
                    all_means_list.append(np.argmin(mean_data))
                    ax.set_title(f"Position of Trough Mean Activity in Cluster {title}", fontsize=10)

    if use_argmax:
        color='forestgreen'
    else:
        color='darkred'

    ax.hist(all_means_list, bins=50, alpha=alpha, color=color)
    ax.set_ylabel("Number of Clusters")

    ax.set_xlabel("Position Bins")
    ax.set_ylim(0,ylim)

    return all_means_list



##################### plotting the optimal number of clusters histogram 

def plot_optimal_num_clusters_histogram(clusters_dict_NDNF_early, title=None, ylim=None, ax=None):
    all_means_num_clusters = []

    for animal in clusters_dict_NDNF_early:
        for cell in clusters_dict_NDNF_early[animal]:
            valid_length=0
            for i in range(len(clusters_dict_NDNF_early[animal][cell])):
                # if np.mean(clusters_dict_NDNF_early[animal][cell][i] !=0):
                valid_length+=1
            all_means_num_clusters.append(valid_length)

    data_labels=np.unique(all_means_num_clusters)
               
    counts, bins, patches = ax.hist(all_means_num_clusters, bins=4, edgecolor='k')
    ax.set_xlabel("Optimal Number of Clusters (Elbow)")
    ax.set_title(f"Optimal Number of Clusters Chosen Per Cell {title}")
    ax.set_ylabel("Number of Cells")
    # Place xticks in the middle of each bar
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.set_xticks(bin_centers, data_labels)
    ax.set_ylim(ylim)


############## expression probability over trials we already have --- expression probability in sliding window but now we want to know - each cluster has x number of trials 

def plot_cluster_expression_probs(clusters_dict_NDNF_early, residual_activity_dict_NDNF_newest, title=None, ylim=None, ax=None):
    proportion_trials_per_cluster = []
    for animal in clusters_dict_NDNF_early:
        for cell in clusters_dict_NDNF_early[animal]:
            total_num_trials = residual_activity_dict_NDNF_newest[animal][cell].shape[1]
            for i in range(len(clusters_dict_NDNF_early[animal][cell])):
                data = clusters_dict_NDNF_early[animal][cell][i]
                # if np.mean(data) !=0:
                num_trials = data.shape[0]
                proportion_trials_per_cluster.append(num_trials/total_num_trials)

    
    ax.hist(proportion_trials_per_cluster, bins=50, edgecolor='k')
    ax.set_ylabel("Number of Clusters")
    ax.set_title(f"Cluster Expression Probability of Clusters Across Trials {title}")
    ax.set_xlabel("Fraction of Total Trials in Cluster")
    ax.set_ylim(ylim)

    return proportion_trials_per_cluster




def plot_max_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, title=None, ax=None):
    max_list_per_cell = []
    for animal in testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20]:
        for cell in testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal]:
            max=0
            for i in range(1,8):
                labels = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][1][f"cell_{cell}"]["labels_dict"][f"clusters_chosen_{i}"]
                max=len(np.unique(labels))
            max_list_per_cell.append(max)

    counts, bins, patches = ax.hist(max_list_per_cell, bins=7, edgecolor='k')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.set_xticks(bin_centers, [str(int(center)) for center in bin_centers])
    ax.set_title(f"Max Number Clusters Identified {title}")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Number of Cells")




def get_indices_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, split_learn=None, eln="nothing"):

    """
    - takes in the contiguous cutpoints slice TCA model and the K-Means slice TCA model
    - use elbow kmeans to loop through every number of cluster (up to 8) to get the optimal number of clusters via reconstruction MSE of cluster average reconstruction vs real data for the cell
    - returns a dict where every cell's activity is seperated by its cluster via trial indices for each cluster
    - since we are seperating by trial indices we can ask whether the indices are within the early sliceTCA changepoint or in late and seperate the data by learning
    """

    indices_dict_EC = {}

    for animal_num, animal in enumerate(residual_activity_dict_EC):
        indices_cell_EC = {}
        for cell_num, cell in enumerate(residual_activity_dict_EC[animal]):
            try:
                elbow_kmeans = get_elbow_score_data(
                    testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell,
                    animal=animal_num,
                    cell=cell_num
                )
            except KeyError:
                print(f"⚠️ Skipping animal {animal_num}, cell {cell_num} — not in model dict")
                continue

            clusters = elbow_kmeans + 1
            indices_dict = get_per_cell_clustering_indices(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal_id=animal_num, cell_id=cell_num, num_clusters=clusters, split_learn=split_learn, early_late_nothing=eln)

            indices_cell_EC[cell] = indices_dict

        indices_dict_EC[animal] = indices_cell_EC

    return indices_dict_EC


def get_per_cell_clustering_indices(cell_EC_model_ranks20_contig_x00, residual_activity_dict_NDNF, cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, animal_id=1, cell_id=1, num_clusters=4, split_learn=False, early_late_nothing="nothing"):
    
    indices_dict = cell_NDNF_model_ranks20_kmeans_reassign_umap_x00[20][animal_id][cell_id][1][f"cell_{cell_id}"]["indices_for_cluster_number"][f"clusters_chosen_{num_clusters}"]

    if split_learn:
        indices_dict_new={}
        animal_first_changepoints_list, fraction_first_changepoints_list, animal_second_changepoints_list, fraction_second_changepoints_list = get_changepoints(cell_EC_model_ranks20_contig_x00, residual_activity_dict_NDNF, animal_TCA=False)
        for n in range(num_clusters):
            indices = indices_dict[n]

            if early_late_nothing == "early":
                indices_list = [i for i in indices if i < animal_first_changepoints_list[animal_id][cell_id]]
                indices_dict_new[n] = indices_list
            elif early_late_nothing == "late":
                indices_list = [i for i in indices if i > animal_second_changepoints_list[animal_id][cell_id]]
                indices_dict_new[n] = indices_list
            else:
                raise ValueError(f"Invalid value for early_late_nothing: {early_late_nothing}")
        return indices_dict_new
    else:
        return indices_dict


def get_activity_from_indices(residual_activity_dict_EC, indices_dict_EC_overall):
    animal_dict = {}
    for animal in residual_activity_dict_EC:
        cell_dict = {}
        for cell in residual_activity_dict_EC[animal]:
            activity_per_cluster = {}
            for i in range(len(indices_dict_EC_overall[animal][cell])):
                cluster_indices = indices_dict_EC_overall[animal][cell][i]
                activity_per_cluster[i] = {"activity":residual_activity_dict_EC[animal][cell][:,cluster_indices],
                                        "mean_activity": np.nanmean(residual_activity_dict_EC[animal][cell][:,cluster_indices], axis=1)}
            
            cell_dict[cell] = activity_per_cluster
        animal_dict[animal] = cell_dict

    return animal_dict



# Create a list of lists, one for each possible argmax position (position bin)
# First, determine the maximum possible position bin length
def get_data_by_peak(activity_indices_EC_dict, use_peak=True):
    max_bin = 0
    for animal in activity_indices_EC_dict:
        for cell in activity_indices_EC_dict[animal]:
            for i in activity_indices_EC_dict[animal][cell]:
                mean_data = activity_indices_EC_dict[animal][cell][i]["mean_activity"]
                if len(mean_data) > max_bin:
                    max_bin = len(mean_data)

    # Initialize a list of lists for each possible argmax position
    mean_data_by_peak = [[] for _ in range(max_bin)]

    # Fill the lists
    for animal in activity_indices_EC_dict:
        for cell in activity_indices_EC_dict[animal]:
            for i in activity_indices_EC_dict[animal][cell]:
                mean_data = activity_indices_EC_dict[animal][cell][i]["mean_activity"]
                if use_peak:
                    peak = np.argmax(mean_data)
                else:
                    peak = np.argmin(mean_data)
                mean_data_by_peak[peak].append(mean_data)

    return mean_data_by_peak, max_bin


def mean_sems_peak_trough(mean_data_by_peak, max_bin, use_peak=True):
    per_pos_bin_dict = {}
    for i in range(len(mean_data_by_peak)):
        list_data = mean_data_by_peak[i]
        peak_per_cluster_list = []
        for j in range(len(list_data)):
            if use_peak:
                peak_data_array = np.max(list_data[j])
                peak_per_cluster_list.append(peak_data_array)
            else:
                peak_data_array = np.min(list_data[j])
                peak_per_cluster_list.append(peak_data_array)
        if np.mean(peak_per_cluster_list) !=0:
            per_pos_bin_dict[i] = {"mean":np.mean(peak_per_cluster_list),
                            "sem": sem(peak_per_cluster_list)}

    # Plot mean and SEM per position bin as dots with error bars (with ticks)
    means = [per_pos_bin_dict[i]['mean'] for i in range(max_bin)]
    sems = [per_pos_bin_dict[i]['sem'] for i in range(max_bin)]
    return means, sems

def plot_means_sems_max_min(activity_indices_EC_dict, ax=None):
    mean_data_by_peak, max_bin = get_data_by_peak(activity_indices_EC_dict, use_peak=True)
    mean_data_by_trough, min_bin = get_data_by_peak(activity_indices_EC_dict, use_peak=False)

    means_peak, sems_peak = mean_sems_peak_trough(mean_data_by_peak, max_bin, use_peak=True)
    means_trough, sems_trough = mean_sems_peak_trough(mean_data_by_trough, min_bin, use_peak=False)

    x = np.arange(max_bin)
    ax.plot(means_peak, color="forestgreen", label="Peak")
    ax.errorbar(x, means_peak, yerr=sems_peak, fmt='o', capsize=4, elinewidth=1.5, markeredgewidth=1.5, color='forestgreen')
    ax.plot(means_trough, color="darkred", label="Trough")
    ax.errorbar(x, means_trough, yerr=sems_trough, fmt='o', capsize=4, elinewidth=1.5, markeredgewidth=1.5, color='darkred')
    ax.set_xlabel('Position Bin')
    ax.set_ylabel('Averaged Z-Scored Activity +- SEM Clusters', fontsize=10)
    ax.set_title('Average Amplitude of Clusters with Peak/Trough at Each Position Bin')
    ax.legend(fontsize=10, loc='upper right')


def find_elbow_point(y_vals, min_index=2):
    from scipy.spatial.distance import cdist
    import numpy as np

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

def plot_elbow_cell_example(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=None, cell=None):

    MSE_dict = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][1][f"cell_{cell}"]["MSE_dict"]

    mse_list = []
    for mse in MSE_dict:
        mse_val = MSE_dict[mse]
        mse_list.append(mse_val)

    elbow_mse = find_elbow_point(mse_list, min_index=2)

    plt.plot(mse_list)
    plt.axvline(elbow_mse, color='r', linestyle='--', label=f'Elbow K={elbow_mse+1}')
    plt.xticks(np.arange(len(MSE_dict)), np.arange(1,len(MSE_dict)+1))
    plt.xlabel("Number of K-Means Clusters", fontsize=16)
    plt.ylabel("MSE", fontsize=16)
    plt.title("SliceTCA Reconstruction MSE vs Raw Data \n Animal#2 Cell#4", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()

def plot_synthetic_sliceTCA_ex(model, ideal_cell):
    weights = torch.abs(model.vectors[0][0].detach().T).cpu().numpy()
    print(weights.shape)
    latents = torch.abs(model.vectors[0][1].detach().T).cpu().numpy()
    latents = np.squeeze(latents, axis=1)
    print(latents.shape)

    plt.figure(figsize=(8,4))
    for i in range(weights.shape[1]):
        plt.plot(weights[:,i])
        plt.title("Weights Over Trials")
    plt.xlabel("Trials")
    plt.show()

    plt.figure(figsize=(8,4))
    for i in range(latents.shape[1]):
        plt.plot(latents[:,i])
        plt.title("Latents Over Position Bins (Fields)")
    plt.xlabel("Posiiton Bins")
    plt.show()

    weight_1 = weights[:,0]
    weight_2 = weights[:,1]

    # Add jitter to the points for better visualization
    rng = np.random.default_rng(0)
    jitter_scale = 0.02  # Adjust as needed
    weight_1_jitter = weight_1 + rng.normal(scale=jitter_scale, size=weight_1.shape)
    weight_2_jitter = weight_2 + rng.normal(scale=jitter_scale, size=weight_2.shape)


    weights_for_kmeans = np.column_stack([weight_1, weight_2])

    k = 4  
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(weights_for_kmeans)

    import matplotlib.patches as mpatches

    # Map each label to a color from tab10, ensuring color order matches cluster label
    tab10 = plt.cm.get_cmap('tab10')
    unique_labels = np.unique(labels)
    label_to_color = {lbl: tab10(int(lbl) % 10) for lbl in unique_labels}
    point_colors = [label_to_color[lbl] for lbl in labels]

    # Plot, colored by cluster
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(weight_1_jitter, weight_2_jitter, c=point_colors, alpha=0.8)

    # Create legend handles for each cluster, matching the scatter colors
    handles = [mpatches.Patch(color=label_to_color[lbl], label=f'Cluster {lbl}') for lbl in unique_labels]

    plt.xlabel("Weight 1 (with jitter)")
    plt.ylabel("Weight 2 (with jitter)")
    plt.title("KMeans Clusters in Weight Space")
    plt.legend(handles=handles, title="Cluster", loc='best')
    plt.show()

    return labels


def plot_real_cells_internals(model_2):
    model_ex_cell = model_2
    latents = torch.abs(model_ex_cell.vectors[0][0].detach().T).cpu().numpy()
    plt.figure(figsize=(10,4))

    plt.plot(latents[:,0], color='b')
    plt.plot(latents[:,1], color='orange')
    plt.xlabel("Trials")
    plt.title("2 Weights (1/Latent) SliceTCA Example EC Cell")
    plt.show()

    model_ex_cell = model_2
    latents = torch.abs(model_ex_cell.vectors[0][1].detach().T).cpu().numpy()
    latent_array = latents.squeeze(axis=1)

    plt.figure(figsize=(10,4))
    plt.plot(latent_array[:,0], color='b')
    plt.plot(latent_array[:,1], color='orange')
    plt.title("2 Latent Factors (Fields)")
    plt.xlabel("Position Bins")
    plt.show()
    

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_real_activity_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell,
                                residual_activity_dict_EC, animal=None, cell=None):

    labels = testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell[20][animal][cell][1][f"cell_{cell}"]["indices_for_cluster_number"]["clusters_chosen_4"]
    activity = residual_activity_dict_EC[f'animal_{animal+1}'][f'cell_{cell+1}']  # (pos_bins, trials)
    num_trials = activity.shape[1]

    # ---- make a per-trial label vector (length = num_trials) ----
    n_clusters = len(labels)
    trial_labels = np.full(num_trials, fill_value=-1, dtype=int)
    for k in range(n_clusters):
        trial_labels[np.array(labels[k], dtype=int)] = k

    # safety: check for unassigned trials
    if np.any(trial_labels < 0):
        missing = np.where(trial_labels < 0)[0]
        print(f"Warning: {len(missing)} trials have no cluster assignment:", missing)

    # ---- quick overview: trial strip like your screenshot ----
    cmap = plt.get_cmap('tab10')
    colors = [cmap(k % 10) for k in range(n_clusters)]

    plt.figure(figsize=(9,1.4))
    for k in range(n_clusters):
        idx = np.where(trial_labels == k)[0]
        plt.scatter(idx, np.zeros_like(idx), s=18, color=colors[k])
    plt.yticks([])
    plt.ylim(-0.6, 0.6)
    plt.xlim(-1, num_trials)
    plt.xlabel("Trial index")
    plt.title("Trials Labelled by Cluster")
    # legend
    handles = [Line2D([0],[0], marker='o', linestyle='',
                      color=colors[k], label=f"Cluster {k}") for k in range(n_clusters)]
    # plt.legend(handles=handles, loc='upper right', ncol=min(n_clusters,4), frameon=False)
    plt.tight_layout()
    plt.show()

    # ---- cluster-wise plots (heatmap + mean) in a loop to avoid repetition) ----
    for k in range(n_clusters):
        grp_idx = np.array(labels[k], dtype=int)
        grp_act = activity[:, grp_idx]  # (pos_bins, n_trials_in_cluster)

        # mean traces for all clusters on one plot (only once)
        if k == 0:
            plt.figure(figsize=(7,4))
            for j in range(n_clusters):
                mj_idx = np.array(labels[j], dtype=int)
                plt.plot(np.mean(activity[:, mj_idx], axis=1), label=f"Cluster {j}", color=colors[j])
            plt.ylabel("Z-Scored Activity")
            plt.xlabel("Position Bin")
            plt.title("Trial-Averaged Activity For Clusters")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # per-cluster heatmap + mean
        fig, axs = plt.subplots(1, 2, figsize=(9,4))
        im = axs[0].imshow(grp_act.T, aspect='auto')
        fig.colorbar(im, ax=axs[0])
        axs[0].set_title(f"Cluster {k} Activity")
        axs[0].set_xlabel("Position Bins")
        axs[0].set_ylabel("Trial")

        axs[1].plot(np.mean(grp_act, axis=1), color=colors[k])
        axs[1].set_title(f"Trial Av. Activity\n{len(grp_idx)}/{num_trials} trials ({len(grp_idx)/num_trials*100:.1f}%)")
        axs[1].set_xlabel("Position Bins")
        axs[1].set_ylabel("Z-Scored Activity")
        axs[1].set_ylim(-0.5, 4.5)

        plt.tight_layout()
        plt.show()

    X_trials = activity.T  # shape: (n_trials, n_bins)

    # (optional) if you want to z-score features across trials:
    # X_trials = (X_trials - X_trials.mean(axis=0)) / (X_trials.std(axis=0) + 1e-8)

    pca = PCA(n_components=2)
    PCs = pca.fit_transform(X_trials)  # (n_trials, 2)

    plt.figure(figsize=(5.2, 5.2))
    for k in range(n_clusters):
        idx = np.where(trial_labels == k)[0]
        plt.scatter(PCs[idx, 0], PCs[idx, 1], s=35, alpha=0.9, label=f"Cluster {k}", color=colors[k])

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    exp = pca.explained_variance_ratio_
    plt.title("PC1 vs PC2")
    # legend to match your style
    handles = [Line2D([0],[0], marker='o', linestyle='', color=colors[k], label=f"Cluster {k}")
            for k in range(n_clusters)]
    plt.legend(handles=handles, loc='best', frameon=False)
    plt.tight_layout()
    plt.show()




# ---- your helpers (unchanged) ----
def get_activity_array(activity, ax, i, total_num_trials=None):
    ax.imshow(activity.T, aspect='auto')
    ax.set_xlabel("Positon Bins")
    ax.set_title(f"Example Cell Cluster #{i}")
    ax.set_ylabel("Trials")

def get_mean_activity_array(activity, ax, i, min=None, max=None, total_num_trials=None):
    from scipy.stats import sem
    mean_activity = np.mean(activity, axis=1)
    sem_activity = sem(activity, axis=1)
    ax.plot(mean_activity)
    ax.set_ylim(min, max)
    ax.set_ylabel("Z-Scored Activity")
    ax.set_title(f"Example Cell Mean Cluster #{i} \n Fraction of Total Trials = {activity.shape[1]}/{total_num_trials} or {(activity.shape[1]/total_num_trials)*100:.1f}%")
    ax.set_xlabel("Position Bins")
    ax.fill_between(range(len(mean_activity)),
                    mean_activity + sem_activity,
                    mean_activity - sem_activity,
                    alpha=0.2)

def plot_peak_trough_histograms_wrap(clusters_dict_NDNF_early, use_argmax=True, title=None, ylim=None, ax=None, color=None, alpha=None):
    all_means_list = []

    for animal in clusters_dict_NDNF_early:
        for cell in clusters_dict_NDNF_early[animal]:
            for i in range(len(clusters_dict_NDNF_early[animal][cell])):
                data = clusters_dict_NDNF_early[animal][cell][i]
                # if np.mean(data != 0):
                mean_data = np.mean(data, axis=0)
                if use_argmax:
                    all_means_list.append(np.argmax(mean_data))
                    ax.set_title(f"Position of Peak Mean Activity in Cluster {title}", fontsize=10)
                else:
                    all_means_list.append(np.argmin(mean_data))
                    ax.set_title(f"Position of Trough Mean Activity in Cluster {title}", fontsize=10)

    wrap_count = 0
    else_count = 0
    
    for i in range(len(all_means_list)-1):
        if all_means_list[i] > 45 and all_means_list[i+1] < 45:
            wrap_count+=1
            early_peak = all_means_list[i] 
            late_peak = all_means_list[i+1] 
            if np.random.rand() > 0.5:
                all_means_list[i] = late_peak
            else:
                all_means_list[i] = early_peak
            all_means_list[i+1] = np.nan
        else:
            else_count+=1

    print(f"wrap_count {wrap_count}, else_count, {else_count}")
    
    if use_argmax:
        color='forestgreen'
    else:
        color='darkred'

    ax.hist(all_means_list, bins=50, alpha=alpha, color=color)
    ax.set_ylabel("Number of Clusters")

    ax.set_xlabel("Position Bins")
    ax.set_ylim(0,ylim)

    return all_means_list


def plot_pop_stats(clusters_dict_EC_overall, cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal=None, cell=None, cell_type="EC", ylim=None):

    fig = plt.figure(figsize=(21, 14), constrained_layout=True)

    indices_dict_EC_overall = get_indices_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln="nothing")
    activity_indices_EC_dict = get_activity_from_indices(residual_activity_dict_EC, indices_dict_EC_overall)

    max=0
    min=0
    activity_list = []
    for i in activity_indices_EC_dict[animal][cell]:
        activity = activity_indices_EC_dict[animal][cell][i]['activity']
        activity_list.append(activity)
        if np.max(activity) > max:
            max=np.max(activity)
        if np.min(activity) < min:
            min=np.min(activity)
        

    # --- outer grid: make left column wide, shrink vertical gap between rows ---
    gs = plt.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.6, 1.6, 1.6],
        height_ratios=[1.0, 1.0]
    )

    # reduce global paddings/gaps for constrained_layout
    # (works in recent Matplotlib; fall back to set_constrained_layout_pads if needed)
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.01, hspace=0.04, wspace=0.04)

    # --- nested grid in top-left ---
    subgs = gs[0, 0].subgridspec(4, 2, hspace=0.08, wspace=0.03)

    # make nested axes
    ax00 = fig.add_subplot(subgs[0, 0]); ax01 = fig.add_subplot(subgs[0, 1])
    ax10 = fig.add_subplot(subgs[1, 0]); ax11 = fig.add_subplot(subgs[1, 1])
    ax20 = fig.add_subplot(subgs[2, 0]); ax21 = fig.add_subplot(subgs[2, 1])
    ax30 = fig.add_subplot(subgs[3, 0]); ax31 = fig.add_subplot(subgs[3, 1])

    # your plotting...

    total_num_trials = np.sum([activity_list[0].shape[1] + activity_list[1].shape[1] + activity_list[2].shape[1] + activity_list[3].shape[1]])
    print(f"total_num_trials {total_num_trials}")

    get_activity_array(activity_list[0], ax=ax00, i=0, total_num_trials=total_num_trials); get_mean_activity_array(activity_list[0], ax=ax01, i=0, min=min, max=max, total_num_trials=total_num_trials)
    get_activity_array(activity_list[1], ax=ax10, i=1, total_num_trials=total_num_trials); get_mean_activity_array(activity_list[1], ax=ax11, i=1, min=min, max=max, total_num_trials=total_num_trials)
    get_activity_array(activity_list[2], ax=ax20, i=2, total_num_trials=total_num_trials); get_mean_activity_array(activity_list[2], ax=ax21, i=2, min=min, max=max, total_num_trials=total_num_trials)
    get_activity_array(activity_list[3], ax=ax30, i=3, total_num_trials=total_num_trials); get_mean_activity_array(activity_list[3], ax=ax31, i=3, min=min, max=max, total_num_trials=total_num_trials)

    # ---- tidy nested labels to save vertical space ----
    for ax in [ax00, ax10, ax20]:           # hide repeated x labels on upper rows
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    for ax in [ax01, ax11, ax21]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    # shrink title/label padding inside the nested block
    for ax in [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31]:
        ax.set_title(ax.get_title(), fontsize=11, pad=2)
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        ax.tick_params(labelsize=8, pad=1)

    # --- rest of the outer axes ---
    ax_big_01 = fig.add_subplot(gs[0, 1])
    ax_big_02 = fig.add_subplot(gs[0, 2])
    ax_big_10 = fig.add_subplot(gs[1, 0])
    ax_big_11 = fig.add_subplot(gs[1, 1])
    ax_big_12 = fig.add_subplot(gs[1, 2])

    plot_optimal_num_clusters_histogram(clusters_dict_EC_overall, title=cell_type, ylim=None, ax=ax_big_01)
    proportion_trials_per_cluster = plot_cluster_expression_probs(clusters_dict_EC_overall, residual_activity_dict_EC, title=cell_type, ylim=None, ax=ax_big_02)
    all_means_list_peak = plot_peak_trough_histograms_wrap(clusters_dict_EC_overall, use_argmax=True,  title=cell_type, ylim=ylim, ax=ax_big_10)
    all_means_list_trough = plot_peak_trough_histograms_wrap(clusters_dict_EC_overall, use_argmax=False, title=cell_type, ylim=ylim, ax=ax_big_11)
    plot_means_sems_max_min(activity_indices_EC_dict, ax=ax_big_12)

    # slightly tighten title pads on the outer plots too
    for ax in [ax_big_01, ax_big_02, ax_big_10, ax_big_11, ax_big_12]:
        ax.set_title(ax.get_title(), pad=6)

    plt.show()

    return all_means_list_peak, all_means_list_trough, proportion_trials_per_cluster

def compare_cluster_2_0(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC):
    indices_dict_EC_overall = get_indices_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln="nothing")
    activity_indices_EC_dict = get_activity_from_indices(residual_activity_dict_EC, indices_dict_EC_overall)

    cluster2_data = activity_indices_EC_dict['animal_2']['cell_4'][2]["activity"]
    cluster0_data = activity_indices_EC_dict['animal_2']['cell_4'][0]["activity"]

    fig, axs = plt.subplots(2,3, figsize=(16,6))

    im=axs[0,0].imshow(cluster2_data.T, aspect='auto')
    axs[0,0].set_title("Cluster 2 Example Cell")
    fig.colorbar(im, ax=axs[0,0], label="Z-Scored Activity")
    axs[1,0].set_ylabel("Trials")
    axs[1,0].set_xlabel("Position Bins")

    for i in range(cluster2_data.shape[1]):
        axs[1,0].plot(cluster2_data[:,i])
        axs[1,0].set_title("Mean Activity Per Trial Cluster 2")
        axs[1,0].set_ylabel("Z-Scored DF/F")
        axs[1,0].set_xlabel("Position Bins")

    im=axs[0,1].imshow(cluster0_data.T, aspect='auto')
    axs[0,1].set_title("Cluster 0 Example Cell")
    fig.colorbar(im, ax=axs[0,1], label="Z-Scored Activity")
    axs[0,1].set_ylabel("Trials")
    axs[0,1].set_xlabel("Position Bins")

    for i in range(cluster0_data.shape[1]):
        axs[1,1].plot(cluster0_data[:,i])
        axs[1,1].set_title("Mean Activity Per Trial Cluster 2")
        axs[1,1].set_ylabel("Z-Scored DF/F")
        axs[1,1].set_xlabel("Position Bins")

    trials_of_interest = [4, 6, 7, 23,45, 48]

    for i in trials_of_interest:
        axs[1,2].plot(cluster0_data[:,i], label=f'Trial {i}')
        axs[1,2].set_title("Mean Activity Per Trial Cluster 2")
        axs[1,2].set_ylabel("Z-Scored DF/F")
        axs[1,2].set_xlabel("Position Bins")
        axs[1,2].legend(fontsize=8)

    im=axs[0,2].imshow(cluster0_data.T[trials_of_interest, :], aspect='auto')
    axs[0,2].set_title("Cluster 0 Select Trials")
    fig.colorbar(im, ax=axs[0,2], label="Z-Scored Activity")
    axs[0,2].set_ylabel("Trials")
    axs[0,2].set_xlabel("Position Bins")


    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(10,6))
    trials_of_interest = [4, 6, 7, 23, 45, 48]
    for i in trials_of_interest:
        plt.plot(cluster0_data[:,i], color='r', alpha=0.5)
    for i in range(cluster2_data.shape[1]):
        plt.plot(cluster2_data[:,i], color='k', alpha=0.5)
    plt.ylabel("Z-Scored DF/F")
    plt.xlabel("Position Bins")
    plt.title("Black=Cluster 2, Red=Select Cluster 0")
    plt.show()

def cell_by_cell_variance_values(activity_dict_EC):
    list_cells = []

    for animal in activity_dict_EC:
        for cell in activity_dict_EC[animal]:
            variance = np.var(activity_dict_EC[animal][cell], axis=1)
            list_cells.append(variance)

    array_cells = np.array(list_cells)
    sum_var = np.sum(array_cells, axis=0)
    plt.title("Summed Trial Variance Across EC Cells")
    plt.ylabel("Sum Variance over Cells")
    plt.xlabel("Position Bins")
    plt.bar(range(len(sum_var)), sum_var)
    plt.show()

    mean_var = np.mean(array_cells, axis=0)
    sem_var = sem(array_cells, axis=0)
    plt.title('Mean Trial Variance Across EC Cells')
    plt.ylabel("Mean +- SEM Variance over Cells")
    plt.xlabel("Position Bins")
    plt.bar(range(len(mean_var)), mean_var, yerr=sem_var)
    plt.show()

def sum_ec_then_get_var(activity_dict_EC):
    list_cells_min_max = []

    for animal in activity_dict_EC:
        for cell in activity_dict_EC[animal]:
            activity_trunc = activity_dict_EC[animal][cell][:,:58]
            activity_trunc_norm = ((activity_trunc - np.min(activity_trunc)) / (np.max(activity_trunc) - np.min(activity_trunc)))
            list_cells_min_max.append(activity_trunc_norm)

    list_cells_min_max_array = np.array(list_cells_min_max)
    sum_list_cells_min_max_array = np.sum(list_cells_min_max_array, axis=0)

    plt.imshow(sum_list_cells_min_max_array.T, aspect='auto')
    plt.title("Sum of Min/Max EC Activity")
    plt.colorbar()
    plt.show()

    var_trials = np.var(sum_list_cells_min_max_array, axis=1)
    print(var_trials.shape)
    # sem_var_trials = sem(sum_list_cells_min_max_array, axis=1)

    x = np.arange(len(var_trials)) 
    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(x, var_trials)
    ax.set_ylabel("Variance")
    ax.set_xlabel("Position Bins")
    ax.set_title("Variance Across Trials of Summed Normalized EC Activity")
    plt.tight_layout()
    plt.show()



def plot_cluster_frequency_by_animal(clusters_dict_NDNF_overall, use_argmax=False, title="NDNF"):
    
    per_animal_positions = defaultdict(list)
    n_pos = None

    for animal in clusters_dict_NDNF_overall:
        for cell in clusters_dict_NDNF_overall[animal]:
            for i in range(len(clusters_dict_NDNF_overall[animal][cell])):
                data = clusters_dict_NDNF_overall[animal][cell][i]  
                mean_data = np.mean(data, axis=0)
                if n_pos is None:
                    n_pos = mean_data.size
                pos = np.argmax(mean_data) if use_argmax else np.argmin(mean_data)
                per_animal_positions[animal].append(pos)

    if n_pos is None:
        raise ValueError("No data found.")

    bins = np.arange(n_pos + 1) - 0.5
    bin_centers = np.arange(n_pos)

    animal_names = sorted(per_animal_positions.keys())
    counts_per_animal = []
    for animal in animal_names:
        counts, _ = np.histogram(per_animal_positions[animal], bins=bins)
        counts_per_animal.append(counts)
    counts_per_animal = np.vstack(counts_per_animal) 

    total_counts = counts_per_animal.sum(axis=0)

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(n_pos)

    # palette & style to match your look
    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(animal_names))))[:len(animal_names)]

    for idx, animal in enumerate(animal_names):
        ax.bar(
            bin_centers, counts_per_animal[idx],
            bottom=bottom, width=0.95,
            edgecolor='k', linewidth=0.4, color=palette[idx],
            label=str(animal)
        )
        bottom += counts_per_animal[idx]

    # optional: thin outline of total bar tops to mimic single-color hist edges
    ax.step(bin_centers, total_counts, where='mid', linewidth=0.8, color='k', alpha=0.6)

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_xlabel("Position Bins")
    ax.set_ylabel("Number of Clusters")
    ax.set_title(f"Position of {'Peak' if use_argmax else 'Trough'} Mean Activity in Cluster {title}")

    # lighter legend outside
    ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)
    plt.tight_layout()
    plt.show()

    return per_animal_positions, counts_per_animal, total_counts



def plot_cluster_frequency_by_animal(clusters_dict_NDNF_overall, use_argmax=False, title="NDNF"):
    
    per_animal_positions = defaultdict(list)
    n_pos = None

    for animal in clusters_dict_NDNF_overall:
        for cell in clusters_dict_NDNF_overall[animal]:
            for i in range(len(clusters_dict_NDNF_overall[animal][cell])):
                data = clusters_dict_NDNF_overall[animal][cell][i]  
                mean_data = np.mean(data, axis=0)
                if n_pos is None:
                    n_pos = mean_data.size
                pos = np.argmax(mean_data) if use_argmax else np.argmin(mean_data)
                per_animal_positions[animal].append(pos)

    if n_pos is None:
        raise ValueError("No data found.")

    bins = np.arange(n_pos + 1) - 0.5
    bin_centers = np.arange(n_pos)

    animal_names = sorted(per_animal_positions.keys())
    counts_per_animal = []
    for animal in animal_names:
        counts, _ = np.histogram(per_animal_positions[animal], bins=bins)
        counts_per_animal.append(counts)
    counts_per_animal = np.vstack(counts_per_animal) 

    total_counts = counts_per_animal.sum(axis=0)

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(n_pos)

    # palette & style to match your look
    palette = plt.cm.tab20(np.linspace(0, 1, max(20, len(animal_names))))[:len(animal_names)]

    for idx, animal in enumerate(animal_names):
        ax.bar(
            bin_centers, counts_per_animal[idx],
            bottom=bottom, width=0.95,
            edgecolor='k', linewidth=0.4, color=palette[idx],
            label=str(animal)
        )
        bottom += counts_per_animal[idx]

    # optional: thin outline of total bar tops to mimic single-color hist edges
    ax.step(bin_centers, total_counts, where='mid', linewidth=0.8, color='k', alpha=0.6)

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_xlabel("Position Bins")
    ax.set_ylabel("Number of Clusters")
    ax.set_title(f"Position of {'Peak' if use_argmax else 'Trough'} Mean Activity in Cluster {title}")

    # lighter legend outside
    ax.legend(title="Animal", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)
    plt.tight_layout()
    plt.show()

    return per_animal_positions, counts_per_animal, total_counts


