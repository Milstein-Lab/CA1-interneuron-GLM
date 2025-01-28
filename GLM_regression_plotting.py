from GLM_regression import *

def plot_single_animal_average_trace(animal_mean_list_SST, animal_mean_residuals_SST, activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, cell_type="SST"):
    #### 2 per animal velocity, activity and residual
    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"
    )
    mean_r2_per_animal_raw, sem_r2_per_animal_raw = get_per_animal_mean_r2(r2_variable_activity_dict_SST)
    mean_r2_per_animal_residual, sem_r2_per_animal_residual = get_per_animal_mean_r2(r2_variable_residual_dict_SST)

    # Define the color to plot based on cell type
    color_map = {"SST": "blue", "NDNF": "orange", "EC": "green"}
    color_to_plot = color_map.get(cell_type, "limegreen")

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Velocity Plot
    trial_av_animal_list = []
    for animal in filtered_factors_dict_SST:
        value = filtered_factors_dict_SST[animal]["Velocity"]
        trial_av_animal_list.append(np.mean(value, axis=1))

    velocity_array = np.stack(trial_av_animal_list) * 100
    mean_velocity = np.mean(velocity_array, axis=0)

    for trial in velocity_array:
        axs[0].plot(trial, color="limegreen", alpha=0.5, linewidth=0.8)
    axs[0].plot(mean_velocity, linewidth=3, color='r', alpha=0.9, label="Mean Velocity")
    axs[0].set_title("Velocity Per Animal Trial Averaged", fontsize=14)
    axs[0].set_xlabel("Position Bins", fontsize=12)
    axs[0].set_ylabel("cm Per Second", fontsize=12)
    axs[0].legend(fontsize=10)

    # Raw Activity Plot
    animal_mean_mean_list_SST = np.stack(animal_mean_list_SST)
    animal_mean_mean_list_SST = np.mean(animal_mean_mean_list_SST, axis=0)

    for animal_to_plot in range(len(animal_mean_list_SST)):
        axs[1].plot(animal_mean_list_SST[animal_to_plot], color="limegreen", alpha=0.5, linewidth=0.8)
    axs[1].plot(animal_mean_mean_list_SST, color=color_to_plot, alpha=0.9, linewidth=3, label="Mean Raw Activity")
    axs[1].set_title(f"{cell_type} Raw Activity Across Animals", fontsize=14)
    axs[1].set_xlabel("Position Bin", fontsize=12)
    axs[1].set_ylabel("DF/F", fontsize=12)
    axs[1].set_ylim(-1, 1)
    axs[1].legend(fontsize=10)

    # Residual Activity Plot
    animal_mean_mean_residual_list_SST = np.stack(animal_mean_residuals_SST)
    animal_mean_mean_residual_list_SST = np.mean(animal_mean_mean_residual_list_SST, axis=0)

    for animal_to_plot in range(len(animal_mean_residuals_SST)):
        axs[2].plot(animal_mean_residuals_SST[animal_to_plot], color="limegreen", alpha=0.5)
    axs[2].plot(animal_mean_mean_residual_list_SST, color=color_to_plot, alpha=0.9, linewidth=3, label="Mean Residual Activity")
    axs[2].set_title(f"{cell_type} Residual Activity Across Animals", fontsize=14)
    axs[2].set_xlabel("Position Bin", fontsize=12)
    axs[2].set_ylabel("DF/F", fontsize=12)
    axs[2].set_ylim(-1, 1)
    axs[2].legend(fontsize=10)

    # Finalize layout
    fig.tight_layout()
    plt.show()

    # Plot population correlation
    plot_pop_correlation(
        mean_r2_per_animal_raw, mean_r2_per_animal_residual,
        variable_to_correlate="Velocity", cell_type=cell_type,
        color=color_to_plot, dictionary=False
    )

def plot_cdf_split_r_or_learn(mean_quantiles_list, sem_quantiles_list, title=None, x_title=None, n_bins=None, residual=True, r_or_learn="learn"):

    bin_centers = np.arange(1, n_bins + 1)
    percentile_labels = [f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    cell_types = ["SST", "NDNF", "EC"]
    colors_above = ["blue", "orange", "green"]
    colors_below = ["c", "red", "limegreen"]

    for i, ax in enumerate(axs):
        if r_or_learn=="r":
            ax.errorbar(mean_quantiles_list[i * 2], bin_centers, xerr=sem_quantiles_list[i * 2], fmt='o-', color=colors_above[i], ecolor=colors_above[i], capsize=8, label=f"{cell_types[i]} above 0 R vs Vel")
            ax.errorbar(mean_quantiles_list[i * 2 + 1], bin_centers, xerr=sem_quantiles_list[i * 2 + 1], fmt='o-', color=colors_below[i], ecolor=colors_below[i], capsize=8, label=f"{cell_types[i]} below 0 R vs Vel")
        elif r_or_learn=="learn":
            ax.errorbar(mean_quantiles_list[i * 2], bin_centers, xerr=sem_quantiles_list[i * 2], fmt='o-', color=colors_above[i], ecolor=colors_above[i], capsize=8, label=f"{cell_types[i]} Early Learn (Q1)")
            ax.errorbar(mean_quantiles_list[i * 2 + 1], bin_centers, xerr=sem_quantiles_list[i * 2 + 1], fmt='o-', color=colors_below[i], ecolor=colors_below[i], capsize=8, label=f"{cell_types[i]} Late Learn (Q5)")

        ax.set_xlabel(f"Mean {x_title}")
        ax.set_title(f"{cell_types[i]} {title} residual={residual}")
        ax.legend()

    # Shared y-axis label and formatting
    axs[0].set_ylabel("Percentile of Data")
    axs[0].set_yticks(ticks=bin_centers)
    axs[0].set_yticklabels(percentile_labels)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_frequency_hist_seperate(SST_list_above, SST_list_below, NDNF_list_above, NDNF_list_below, EC_list_above, EC_list_below, selectivity_or_arg="selectivity", name=None, residual=True, r_or_learn="learn"):

    if selectivity_or_arg == "selectivity":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif selectivity_or_arg == "arg":
        bin_edges = np.arange(0, 51, 5)
        bin_centers = bin_edges[:-1] + 2.5
        bin_labels = [f"{start}-{start + 4}" for start in bin_edges[:-1]]

    else:
        raise ValueError("selectivity_or_arg takes either 'selectivity' or 'arg'")

    cell_types = {
        "SST": ("b", "c"),
        "NDNF": ("orange", "red"),
        "EC": ("green", "limegreen")
    }

    data = [
        ("SST", SST_list_above, SST_list_below),
        ("NDNF", NDNF_list_above, NDNF_list_below),
        ("EC", EC_list_above, EC_list_below)
    ]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, (cell_type, list_above, list_below) in enumerate(data):

        hist_above, _ = np.histogram(list_above, bins=bin_edges)
        hist_below, _ = np.histogram(list_below, bins=bin_edges)

        fraction_above = hist_above / np.sum(hist_above)
        fraction_below = hist_below / np.sum(hist_below)

        if r_or_learn == 'r':
            axs[i].plot(bin_centers, fraction_above, marker='o', label=f'{cell_type} above 0 R vs Vel', linestyle='-', color=cell_types[cell_type][0])
            axs[i].plot(bin_centers, fraction_below, marker='o', label=f'{cell_type} below 0 R vs Vel', linestyle='-', color=cell_types[cell_type][1])
        elif r_or_learn == 'learn':
            axs[i].plot(bin_centers, fraction_above, marker='o', label=f'{cell_type} Early Learn (Q1)', linestyle='-', color=cell_types[cell_type][0])
            axs[i].plot(bin_centers, fraction_below, marker='o', label=f'{cell_type} Late Learn (Q5)', linestyle='-', color=cell_types[cell_type][1])
        else:
            raise ValueError("valid options for r_or_learn are r or learn")

        axs[i].set_xlabel(name)
        axs[i].set_ylabel('Fraction of Cells')
        axs[i].set_title(f'{name} residuals={residual}')
        axs[i].set_xticks(bin_centers)
        axs[i].set_xticklabels(bin_labels)
        axs[i].set_ylim(0, 1)
        axs[i].legend()

    plt.tight_layout()
    plt.show()


def plot_activity_by_group_r(mean_SST_above, sem_SST_above, mean_SST_below, sem_SST_below, mean_NDNF_above, sem_NDNF_above, mean_NDNF_below, sem_NDNF_below, mean_EC_above, sem_EC_above, mean_EC_below, sem_EC_below, residual=True):

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    axs[0].errorbar(range(len(mean_SST_above)), mean_SST_above, yerr=sem_SST_above, fmt='o-', color='blue',
             label=f'SST above 0 R vs Vel', markersize=1)
    axs[0].errorbar(range(len(mean_SST_below)), mean_SST_below, yerr=sem_SST_below, fmt='o--', color='cyan',
             label=f'SST below 0 R vs Vel', markersize=1)

    axs[0].set_xlabel("Position Bins")
    axs[0].set_ylabel("Mean Activity")
    axs[0].set_title(f"SST Activity residual={residual}", fontsize=10)
    axs[0].set_ylim(-0.4, 0.4)
    axs[0].legend()


    axs[1].errorbar(range(len(mean_NDNF_above)), mean_NDNF_above, yerr=sem_NDNF_above, fmt='o-', color='orange',
                    label='NDNF above 0 R vs Vel', markersize=1)
    axs[1].errorbar(range(len(mean_NDNF_below)), mean_NDNF_below, yerr=sem_NDNF_below, fmt='o--', color="red",
                    label='NDNF below 0 R vs Vel', markersize=1)

    axs[1].set_xlabel("Position Bins")
    axs[1].set_ylabel("Mean Activity")
    axs[1].set_title(f"NDNF Activity residual={residual}", fontsize=10)
    axs[1].set_ylim(-0.4, 0.4)
    axs[1].legend()


    axs[2].errorbar(range(len(mean_SST_above)), mean_EC_above, yerr=sem_EC_above, fmt='o-', color='green',
                    label='EC above 0 R vs Vel', markersize=1)
    axs[2].errorbar(range(len(mean_EC_below)), mean_EC_below, yerr=sem_EC_below, fmt='o--', color='limegreen',
                   label='EC below 0 R vs Vel', markersize=1)

    axs[2].set_xlabel("Position Bins")
    axs[2].set_ylabel("Mean Activity")
    axs[2].set_title(f"EC Activity residual={residual}", fontsize=10)
    axs[2].set_ylim(-0.4, 0.4)
    axs[2].legend()

    fig.show()
