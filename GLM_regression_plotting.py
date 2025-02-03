from GLM_regression import *

def plot_cell_trial_average_variable_subtraction(activity_dict_SST, activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_SST, predicted_activity_dict_NDNF, predicted_activity_dict_EC):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(activity_dict_EC, predicted_activity_dict_EC)

    trial_av_neuron_activity_list_SST = trial_average(neuron_activity_list_SST)
    trial_av_mean_cell_residual_list_SST = trial_average(cell_residual_list_SST)

    trial_av_neuron_activity_list_NDNF = trial_average(neuron_activity_list_NDNF)
    trial_av_mean_cell_residual_list_NDNF = trial_average(cell_residual_list_NDNF)

    trial_av_neuron_activity_list_EC = trial_average(neuron_activity_list_EC)
    trial_av_mean_cell_residual_list_EC = trial_average(cell_residual_list_EC)

    neuron_activity_list_SST_array = np.stack(trial_av_neuron_activity_list_SST)
    neuron_activity_list_NDNF_array = np.stack(trial_av_neuron_activity_list_NDNF)
    neuron_activity_list_EC_array = np.stack(trial_av_neuron_activity_list_EC)

    neuron_residual_list_SST_array = np.stack(trial_av_mean_cell_residual_list_SST)
    neuron_residual_list_NDNF_array = np.stack(trial_av_mean_cell_residual_list_NDNF)
    neuron_residual_list_EC_array = np.stack(trial_av_mean_cell_residual_list_EC)

    cell_av_neuron_activity_list_SST = np.mean(neuron_activity_list_SST_array, axis=0)
    cell_av_neuron_activity_list_NDNF = np.mean(neuron_activity_list_NDNF_array, axis=0)
    cell_av_neuron_activity_list_EC = np.mean(neuron_activity_list_EC_array, axis=0)

    cell_av_residual_list_SST = np.mean(neuron_residual_list_SST_array, axis=0)
    cell_av_residual_list_NDNF = np.mean(neuron_residual_list_NDNF_array, axis=0)
    cell_av_residual_list_EC = np.mean(neuron_residual_list_EC_array, axis=0)

    sem_neuron_activity_list_SST = np.std(neuron_activity_list_SST_array, axis=0) / np.sqrt(
        neuron_activity_list_SST_array.shape[0])
    sem_residual_list_SST = np.std(neuron_residual_list_SST_array, axis=0) / np.sqrt(
        neuron_residual_list_SST_array.shape[0])

    sem_neuron_activity_list_NDNF = np.std(neuron_activity_list_NDNF_array, axis=0) / np.sqrt(
        neuron_activity_list_NDNF_array.shape[0])
    sem_residual_list_NDNF = np.std(neuron_residual_list_NDNF_array, axis=0) / np.sqrt(
        neuron_residual_list_NDNF_array.shape[0])

    sem_neuron_activity_list_EC = np.std(neuron_activity_list_EC_array, axis=0) / np.sqrt(
        neuron_activity_list_EC_array.shape[0])
    sem_residual_list_EC = np.std(neuron_residual_list_EC_array, axis=0) / np.sqrt(
        neuron_residual_list_EC_array.shape[0])

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_SST, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_SST)),
                     cell_av_neuron_activity_list_SST - sem_neuron_activity_list_SST,
                     cell_av_neuron_activity_list_SST + sem_neuron_activity_list_SST,
                     color='k', alpha=0.2)
    plt.plot(cell_av_residual_list_SST, color='b', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_SST)),
                     cell_av_residual_list_SST - sem_residual_list_SST,
                     cell_av_residual_list_SST + sem_residual_list_SST,
                     color='b', alpha=0.2)
    plt.ylabel("z-scored DF/F")
    plt.xlabel("Position Bins")
    plt.title("SST Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_NDNF, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_NDNF)),
                     cell_av_neuron_activity_list_NDNF - sem_neuron_activity_list_NDNF,
                     cell_av_neuron_activity_list_NDNF + sem_neuron_activity_list_NDNF,
                     color='k', alpha=0.2)

    plt.plot(cell_av_residual_list_NDNF, color='orange', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_NDNF)),
                     cell_av_residual_list_NDNF - sem_residual_list_NDNF,
                     cell_av_residual_list_NDNF + sem_residual_list_NDNF,
                     color='orange', alpha=0.2)
    plt.ylabel("z-scored DF/F")
    plt.xlabel("Position Bins")
    plt.title("NDNF Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(cell_av_neuron_activity_list_EC, color='k', label='Raw Activity')
    plt.fill_between(range(len(cell_av_neuron_activity_list_EC)),
                     cell_av_neuron_activity_list_EC - sem_neuron_activity_list_EC,
                     cell_av_neuron_activity_list_EC + sem_neuron_activity_list_EC,
                     color='k', alpha=0.2)

    plt.plot(cell_av_residual_list_EC, color='g', label='Residual')
    plt.fill_between(range(len(cell_av_residual_list_EC)),
                     cell_av_residual_list_EC - sem_residual_list_EC,
                     cell_av_residual_list_EC + sem_residual_list_EC,
                     color='g', alpha=0.2)
    plt.ylabel("z-scored DF/F")
    plt.xlabel("Position Bins")
    plt.title("EC Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

def plot_single_animal_average_trace(animal_mean_list_SST, animal_mean_residuals_SST, activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, cell_type="SST"):

    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"
    )
    mean_r2_per_animal_raw, sem_r2_per_animal_raw = get_per_animal_mean_r2(r2_variable_activity_dict_SST)
    mean_r2_per_animal_residual, sem_r2_per_animal_residual = get_per_animal_mean_r2(r2_variable_residual_dict_SST)

    # Define the color to plot based on cell type
    color_map = {"SST": "blue", "NDNF": "orange", "EC": "green"}
    color_to_plot = color_map.get(cell_type, "gray")

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
        axs[0].plot(trial, color="gray", alpha=0.5, linewidth=0.8)
    axs[0].plot(mean_velocity, linewidth=3, color='r', alpha=0.9, label="Mean Velocity")
    axs[0].set_title("Velocity Per Animal Trial Averaged", fontsize=14)
    axs[0].set_xlabel("Position Bins", fontsize=12)
    axs[0].set_ylabel("cm Per Second", fontsize=12)
    axs[0].legend(fontsize=10)

    # Raw Activity Plot
    animal_mean_mean_list_SST = np.stack(animal_mean_list_SST)
    animal_mean_mean_list_SST = np.mean(animal_mean_mean_list_SST, axis=0)

    for animal_to_plot in range(len(animal_mean_list_SST)):
        axs[1].plot(animal_mean_list_SST[animal_to_plot], color="gray", alpha=0.5, linewidth=0.8)
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
        axs[2].plot(animal_mean_residuals_SST[animal_to_plot], color="gray", alpha=0.5)
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

def plot_correlations_single_variable_GLM(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict, activity_dict, cell_number, variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"], variable_into_GLM="Velocity"):

    input_variables = {var: [] for var in factors_dict[next(iter(factors_dict))].keys()}
    filtered_input_variables = {var: [] for var in filtered_factors_dict[next(iter(filtered_factors_dict))].keys()}

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            for var in factors_dict[animal]:
                input_variables[var].append(factors_dict[animal][var])
            for var in filtered_factors_dict[animal]:
                filtered_input_variables[var].append(filtered_factors_dict[animal][var])

    neuron_activity_list, predictions_list, cell_residual_list = get_neuron_activity_prediction_residual(
        activity_dict, predicted_activity_dict)

    cell_activity = neuron_activity_list[cell_number]
    cell_prediciton = predictions_list[cell_number]
    cell_residual = cell_residual_list[cell_number]

    flat_neuron_activity = cell_activity.flatten()
    flat_prediction_total = cell_prediciton.flatten()
    flat_residual_total = cell_residual.flatten()

    num_keys = len(variable_to_correlate_list)
    fig, axs = plt.subplots(1, num_keys, figsize=(6 * (num_keys), 5))

    for idx, key in enumerate(variable_to_correlate_list):
        flat_neuron_activity = neuron_activity_list[cell_number].flatten()
        flat_variable = input_variables[key][cell_number].flatten()

        model_activity = LinearRegression()
        model_activity.fit(flat_variable.reshape(-1, 1), flat_neuron_activity)
        y_pred_activity = model_activity.predict(flat_variable.reshape(-1, 1))
        r2_activity, _ = pearsonr(flat_neuron_activity, y_pred_activity)

        axs[idx].scatter(flat_variable, flat_neuron_activity, label="Data", alpha=0.6)
        axs[idx].plot(flat_variable, y_pred_activity, color='r', label="Best Fit", linewidth=2)
        axs[idx].set_title(f"{key} vs Activity\nR value: {r2_activity:.3f}")
        axs[idx].set_xlabel(f"{key} min/max normalized")
        axs[idx].set_ylabel("Activity (z-scored DF/F)")
        axs[idx].legend()

    plt.tight_layout()
    plt.show()

    variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]
    flat_variable_of_interest = variable_of_interest.flatten()

    r2_total, y_pred_total = compute_r_and_model(flat_prediction_total, flat_neuron_activity)
    # r2_residual, y_pred_total_residual = compute_r_and_model(flat_neuron_activity, flat_residual_total)
    r2_variable_residual, y_pred_variable_residual = compute_r_and_model(flat_variable_of_interest, flat_residual_total)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(flat_prediction_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[0].plot(flat_prediction_total, y_pred_total, color='r', label="Best Fit", linewidth=2)
    axs[0].set_title(f"Prediction based on just {variable_into_GLM} vs Activity\nR value: {r2_total:.3f}")
    axs[0].set_xlabel(f"{variable_into_GLM} Prediction (z-scored DF/F)")
    axs[0].set_ylabel("Activity (z-scored DF/F)")
    axs[0].legend()

    axs[1].scatter(flat_variable_of_interest, flat_residual_total, label="Data", alpha=0.6)
    axs[1].plot(flat_variable_of_interest, y_pred_variable_residual, color='r', label="Best Fit", linewidth=2)
    axs[1].set_title(f"Residuals vs Variable: {variable_into_GLM}\nR value: {r2_variable_residual:.3f}")
    axs[1].set_xlabel(f"Variable: {variable_into_GLM} min/max normalized")
    axs[1].set_ylabel(f"Residuals (z-scored DF/F)")
    axs[1].legend()


    flat_residual_total = flat_residual_total.reshape(neuron_activity_list[cell_number].shape)
    flat_neuron_activity = flat_neuron_activity.reshape(neuron_activity_list[cell_number].shape)

    axs[2].plot(np.mean(predictions_list[cell_number], axis=1), color='b', label=f"{variable_into_GLM} Prediction")
    axs[2].plot(np.mean(flat_neuron_activity, axis=1), color='k', label=f"Raw Activity")
    axs[2].plot(np.mean(flat_residual_total, axis=1), color='r', label=f" Residual")
    axs[2].set_title(f"Cell#{cell_number} Trial Averaged")
    axs[2].set_xlabel('Position Bin')
    axs[2].set_ylabel('Activity (z-scored DF/F)')
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_pop_correlation(r2_variable_activity_dict, r2_variable_residual_dict, variable_to_correlate="Velocity", cell_type="SST", color="blue",dictionary=True):

    if dictionary:
        r2_variable_activity_list = []
        r2_variable_residual_list = []

        for animal in r2_variable_activity_dict:
            for neuron in r2_variable_activity_dict[animal]:
                r2_variable_activity_list.append(r2_variable_activity_dict[animal][neuron])
                r2_variable_residual_list.append(r2_variable_residual_dict[animal][neuron])

    else:
        r2_variable_activity_list = r2_variable_activity_dict
        r2_variable_residual_list = r2_variable_residual_dict


    r2_variable_activity_array = np.array(r2_variable_activity_list)
    r2_variable_residual_array = np.array(r2_variable_residual_list)

    mean_activity = np.mean(r2_variable_activity_array)
    sem_activity = np.std(r2_variable_activity_array) / np.sqrt(len(r2_variable_activity_array))

    mean_residuals = np.mean(r2_variable_residual_array)
    sem_residuals = np.std(r2_variable_residual_array) / np.sqrt(len(r2_variable_residual_array))

    x_positions = [0, 1]
    x_jitter_activity = np.random.normal(x_positions[0], 0.05, size=len(r2_variable_activity_array))
    x_jitter_residuals = np.random.normal(x_positions[1], 0.05, size=len(r2_variable_residual_array))

    plt.figure(figsize=(8, 6))

    plt.scatter(x_jitter_activity, r2_variable_activity_array, color='black', alpha=0.8, label=f'Activity vs {variable_to_correlate}', zorder=3)
    plt.scatter(x_jitter_residuals, r2_variable_residual_array, color=color, alpha=0.8, label=f'Residuals vs {variable_to_correlate}', zorder=3)

    for i in range(len(r2_variable_activity_array)):
        plt.plot([x_jitter_activity[i], x_jitter_residuals[i]],
                 [r2_variable_activity_array[i], r2_variable_residual_array[i]],
                 color='gray', alpha=0.6, linewidth=0.8)

    plt.hlines(mean_activity, x_positions[0] - 0.3, x_positions[0] - 0.1, color='black', linewidth=2, label='Mean (Activity)')
    plt.fill_betweenx([mean_activity - sem_activity, mean_activity + sem_activity],
                      x_positions[0] - 0.3, x_positions[0] - 0.1, color='black', alpha=0.2)

    plt.hlines(mean_residuals, x_positions[1] + 0.2, x_positions[1] + 0.3, color=color, linewidth=2, label='Mean (Residuals)')
    plt.fill_betweenx([mean_residuals - sem_residuals, mean_residuals + sem_residuals],
                      x_positions[1] + 0.2, x_positions[1] + 0.3, color=color, alpha=0.2)

    plt.xticks(x_positions, [f'Activity vs {variable_to_correlate}', f'Residuals vs {variable_to_correlate}'])
    plt.ylabel("R Value")
    plt.title(f"R Values for Activity vs {variable_to_correlate} and Residuals vs {variable_to_correlate} for {cell_type}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_cdf(mean_quantiles_list, sem_quantiles_list, pos_neg_selectivity_max_or_min, n_bins=None, residual=True):

    bin_centers = np.arange(1, n_bins + 1)

    if pos_neg_selectivity_max_or_min == "pos":
        fill_in='Positive Selectivity'
    elif pos_neg_selectivity_max_or_min == "neg":
        fill_in='Negative Selectivity'
    elif pos_neg_selectivity_max_or_min == "min":
        fill_in='Position Bin of Minimum Firing'
    elif pos_neg_selectivity_max_or_min == "max":
        fill_in='Position Bin of Maximum Firing'

    plt.figure(figsize=(10, 6))

    plt.errorbar(mean_quantiles_list[0], bin_centers, xerr=sem_quantiles_list[0], fmt='o-', color='blue', ecolor='blue',
                 capsize=8, label="SST")
    plt.errorbar(mean_quantiles_list[1], bin_centers, xerr=sem_quantiles_list[1], fmt='o-', color='orange',
                 ecolor='orange', capsize=8, label="NDNF")
    plt.errorbar(mean_quantiles_list[2], bin_centers, xerr=sem_quantiles_list[2], fmt='o-', color='green',
                 ecolor='green', capsize=8, label="EC")

    plt.ylabel("Percentile of Data")
    plt.yticks(ticks=bin_centers, labels=[f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)])

    if pos_neg_selectivity_max_or_min == "pos" or pos_neg_selectivity_max_or_min == "neg":
        plt.xlabel(f"{fill_in} Index")
    else:
        plt.xlabel(fill_in)
    if residual:
        plt.title(f"{fill_in} Animal Average for Residuals")
    else:
       plt.title(f"{fill_in} Animal Average")
    plt.legend()

    plt.show()

def plot_frequency_dist(SST_list, NDNF_list, EC_list, pos_neg_selectivity_max_or_min=None, residual=True):


    if pos_neg_selectivity_max_or_min == "pos" or pos_neg_selectivity_max_or_min == "neg":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif pos_neg_selectivity_max_or_min == "max" or pos_neg_selectivity_max_or_min == "min":
        bin_edges = np.arange(0, 51, 5)
        bin_centers = bin_edges[:-1] + 2.5
        bin_labels = [f"{start}-{start + 4}" for start in bin_edges[:-1]]

    else:
        raise ValueError("selectivity_or_arg takes either 'pos' for positive selectivity 'neg' negative selecivity or 'min' for argimn or 'max' for argmax")

    SST_hist, _ = np.histogram(SST_list, bins=bin_edges)
    NDNF_hist, _ = np.histogram(NDNF_list, bins=bin_edges)
    EC_hist, _ = np.histogram(EC_list, bins=bin_edges)

    SST_fraction = SST_hist / np.sum(SST_hist)
    NDNF_fraction = NDNF_hist / np.sum(NDNF_hist)
    EC_fraction = EC_hist / np.sum(EC_hist)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, SST_fraction, marker='o', label=f'SST', linestyle='-')
    plt.plot(bin_centers, NDNF_fraction, marker='o', label=f'NDNF', linestyle='-')
    plt.plot(bin_centers, EC_fraction, marker='o', label=f'EC', linestyle='-')

    if pos_neg_selectivity_max_or_min == "pos":
        fill_in='Positive Selectivity'
    elif pos_neg_selectivity_max_or_min == "neg":
        fill_in='Negative Selectivity'
    elif pos_neg_selectivity_max_or_min == "min":
        fill_in='Position Bin of Minimum Firing'
    elif pos_neg_selectivity_max_or_min == "max":
        fill_in='Position Bin of Maximum Firing'

    if pos_neg_selectivity_max_or_min == "neg" or pos_neg_selectivity_max_or_min == "pos":
        plt.xlabel(f"{fill_in} Index")
    else:
        plt.xlabel(fill_in)

    plt.ylabel('Fraction of Cells')
    if residual:
        plt.title(f'{fill_in} Residuals All Cells')
    else:
        plt.title(f'{fill_in} All Cells')
    plt.xticks(bin_centers, bin_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_cdf_split_r_or_learn(mean_quantiles_list, sem_quantiles_list, pos_neg_selectivity_max_or_min="max", n_bins=20, residual=True, r_or_learn="learn"):

    bin_centers = np.arange(1, n_bins + 1)
    percentile_labels = [f"{int(val)}" for val in np.linspace((100 / n_bins), 100, n_bins)]

    if pos_neg_selectivity_max_or_min == "pos":
        fill_in='Positive Selectivity'
    elif pos_neg_selectivity_max_or_min == "neg":
        fill_in='Negative Selectivity'
    elif pos_neg_selectivity_max_or_min == "min":
        fill_in='Position Bin of Minimum Firing'
    elif pos_neg_selectivity_max_or_min == "max":
        fill_in='Position Bin of Maximum Firing'

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

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

        if pos_neg_selectivity_max_or_min=="min":
            ax.set_xlabel(f"Position Bin of Minimum Firing")
            if residual:
                ax.set_title(f"{cell_types[i]} Residuals Animal Average Position of Min Firing")
            else:
                ax.set_title(f"{cell_types[i]} Animal Average Position of Min Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="max":
            ax.set_xlabel(f"Position Bin of Maximum Firing")
            if residual:
                ax.set_title(f"{cell_types[i]} Residuals Animal Average Position of Max Firing")
            elif not residual:
                ax.set_title(f"{cell_types[i]} Animal Average Position of Max Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="pos":
            ax.set_xlabel(f"Positive Selectivity Index")
            if residual:
                ax.set_title(f"{cell_types[i]} Residuals Animal Average Positive Selectivity")
            elif not residual:
                ax.set_title(f"{cell_types[i]} Animal Average Position of Max Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="neg":
            ax.set_xlabel(f"Negative Selectivity Index")
            if residual:
                ax.set_title(f"{cell_types[i]} Residuals Animal Average Negative Selectivity")
            elif not residual:
                ax.set_title(f"{cell_types[i]} Animal Average Negative Selectivity")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        else:
            raise ValueError(f"min_or_max takes either min or max")

    plt.tight_layout()
    plt.show()

def plot_frequency_hist_seperate(SST_list_above, SST_list_below, NDNF_list_above, NDNF_list_below, EC_list_above, EC_list_below, pos_neg_selectivity_max_or_min=None, residual=True, r_or_learn="learn"):

    if pos_neg_selectivity_max_or_min == "pos" or pos_neg_selectivity_max_or_min == "neg":
        bin_edges = np.arange(0, 1.1, 0.1)
        bin_centers = bin_edges[:-1] + 0.05
        bin_labels = [f"{start:.1f}" for start in bin_edges[:-1]]

    elif pos_neg_selectivity_max_or_min == "max" or pos_neg_selectivity_max_or_min == "min":
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
            axs[i].plot(bin_centers, fraction_above, marker='o', label=f'{cell_type} above 0 R vs Velocity', linestyle='-', color=cell_types[cell_type][0])
            axs[i].plot(bin_centers, fraction_below, marker='o', label=f'{cell_type} below 0 R vs Velocity', linestyle='-', color=cell_types[cell_type][1])
        elif r_or_learn == 'learn':
            axs[i].plot(bin_centers, fraction_above, marker='o', label=f'{cell_type} Early Learn (Q1)', linestyle='-', color=cell_types[cell_type][0])
            axs[i].plot(bin_centers, fraction_below, marker='o', label=f'{cell_type} Late Learn (Q5)', linestyle='-', color=cell_types[cell_type][1])
        else:
            raise ValueError("valid options for r_or_learn are r or learn")

        if pos_neg_selectivity_max_or_min == "pos":
            fill_in = 'Positive Selectivity'
        elif pos_neg_selectivity_max_or_min == "neg":
            fill_in = 'Negative Selectivity'
        elif pos_neg_selectivity_max_or_min == "min":
            fill_in = 'Position Bin of Minimum Firing'
        elif pos_neg_selectivity_max_or_min == "max":
            fill_in = 'Position Bin of Maximum Firing'

        if pos_neg_selectivity_max_or_min == "neg" or pos_neg_selectivity_max_or_min == "pos":
            axs[i].set_xlabel(f"{fill_in} Index")
        else:
            axs[i].set_xlabel(fill_in)
        axs[i].set_ylabel('Fraction of Cells')
        if residual:
            axs[i].set_title(f'{fill_in} Residuals All Cells')
        else:
            axs[i].set_title(f'{fill_in} All Cells')
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

    plt.tight_layout()

    fig.show()


def plot_r2_above_below_per_animal(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):
    per_animal_activity_SST_above, per_animal_activity_SST_below, per_animal_activity_NDNF_above, per_animal_activity_NDNF_below, per_animal_activity_EC_above, per_animal_activity_EC_below = get_animal_vel_correlations_activity(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC,
                                                                                                                                                                                                                                    predicted_activity_dict_EC, filtered_factors_dict_EC, residual=residual)

    mean_SST_above, sem_SST_above, mean_SST_below, sem_SST_below, mean_NDNF_above, sem_NDNF_above, mean_NDNF_below, sem_NDNF_below, mean_EC_above, sem_EC_above, mean_EC_below, sem_EC_below = get_activity_by_r2_groups(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC,
                                                                                                                                                                                                                         predicted_activity_dict_EC, filtered_factors_dict_EC, residual=residual)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    axs[0, 0].plot(mean_SST_above, color='blue', label='SST R vs Vel above 0', markersize=1)
    axs[0, 0].fill_between(range(len(mean_SST_above)), mean_SST_above - sem_SST_above, mean_SST_above + sem_SST_above, color='blue', alpha=0.1)
    axs[0, 0].plot(mean_SST_below, color='cyan', label='SST R vs Vel below 0', markersize=1)
    axs[0, 0].fill_between(range(len(mean_SST_below)), mean_SST_below - sem_SST_below, mean_SST_below + sem_SST_below, color='cyan', alpha=0.2)
    axs[0, 0].set_xlabel("Position Bin")
    axs[0, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 0].set_title("SST All Cells Residuals", fontsize=12)
    else:
        axs[0, 0].set_title("SST All Cells", fontsize=12)
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 0].legend(fontsize=10)

    for animal in range(len(per_animal_activity_SST_above)):
        axs[0, 1].plot(per_animal_activity_SST_above[animal], color='blue', alpha=0.5)
    axs[0, 1].plot(np.mean(np.stack(per_animal_activity_SST_above), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean R vs Vel above 0")
    axs[0, 1].set_xlabel("Position Bin")
    axs[0, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 1].set_title("SST Per Animal Residuals", fontsize=12)
    else:
        axs[0, 1].set_title("SST Per Animal", fontsize=12)
    axs[0, 1].set_ylim(-1, 1)
    axs[0, 1].legend(fontsize=10)

    for animal in range(len(per_animal_activity_SST_below)):
        axs[0, 2].plot(per_animal_activity_SST_below[animal], color='cyan', alpha=0.5)
    axs[0, 2].plot(np.mean(np.stack(per_animal_activity_SST_below), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean R vs Vel below 0")
    axs[0, 2].set_xlabel("Position Bin")
    axs[0, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 2].set_title("SST Per Animal Residuals", fontsize=12)
    else:
        axs[0, 2].set_title("SST Per Animal", fontsize=12)
    axs[0, 2].set_ylim(-1, 1)
    axs[0, 2].legend(fontsize=10)

    axs[1, 0].plot(mean_NDNF_above, color='orange', label='NDNF R vs Vel above 0', markersize=1)
    axs[1, 0].fill_between(range(len(mean_NDNF_above)), mean_NDNF_above - sem_NDNF_above, mean_NDNF_above + sem_NDNF_above, color='orange', alpha=0.2)
    axs[1, 0].plot(mean_NDNF_below, color='red', label='NDNF R vs Vel below 0', markersize=1)
    axs[1, 0].fill_between(range(len(mean_NDNF_below)), mean_NDNF_below - sem_NDNF_below, mean_NDNF_below + sem_NDNF_below, color='red', alpha=0.2)
    axs[1, 0].set_xlabel("Position Bin")
    axs[1, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 0].set_title("NDNF All Cells Residuals", fontsize=12)
    else:
        axs[1, 0].set_title("NDNF All Cells Activity", fontsize=12)
    axs[1, 0].set_ylim(-1, 1)
    axs[1, 0].legend(fontsize=10)

    for animal in range(len(per_animal_activity_NDNF_above)):
        axs[1, 1].plot(per_animal_activity_NDNF_above[animal], color='orange', alpha=0.5)
    axs[1, 1].plot(np.mean(np.stack(per_animal_activity_NDNF_above), axis=0), color='red', alpha=0.9, linewidth=4, label="NDNF Mean R vs Vel above 0")
    axs[1, 1].set_xlabel("Position Bin")
    axs[1, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 1].set_title("NDNF Per Animal Residuals", fontsize=12)
    else:
        axs[1, 1].set_title("NDNF Per Animal", fontsize=12)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].legend(fontsize=10)

    for animal in range(len(per_animal_activity_NDNF_below)):
        axs[1, 2].plot(per_animal_activity_NDNF_below[animal], color='red', alpha=0.5)
    axs[1, 2].plot(np.mean(np.stack(per_animal_activity_NDNF_below), axis=0), color='red', alpha=0.9, linewidth=4, label="NDNF Mean R vs Vel below 0")
    axs[1, 2].set_xlabel("Position Bin")
    axs[1, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 2].set_title("NDNF Per Animal Residuals", fontsize=12)
    else:
        axs[1, 2].set_title("NDNF Per Animal", fontsize=12)
    axs[1, 2].set_ylim(-1, 1)
    axs[1, 2].legend(fontsize=10)

    axs[2, 0].plot(mean_EC_above, color='green', label='EC R vs Vel above 0', markersize=1)
    axs[2, 0].fill_between(range(len(mean_EC_above)), mean_EC_above - sem_EC_above, mean_EC_above + sem_EC_above, color='green', alpha=0.2)
    axs[2, 0].plot(mean_EC_below, color='limegreen', label='EC R vs Vel below 0', markersize=1)
    axs[2, 0].fill_between(range(len(mean_EC_below)), mean_EC_below - sem_EC_below, mean_EC_below + sem_EC_below, color='limegreen', alpha=0.2)
    axs[2, 0].set_xlabel("Position Bin")
    axs[2, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 0].set_title("EC All Cells Residuals", fontsize=12)
    else:
        axs[2, 0].set_title("EC All Cells", fontsize=12)
    axs[2, 0].set_ylim(-1, 1)
    axs[2, 0].legend(fontsize=10)

    for animal in range(len(per_animal_activity_EC_above)):
        axs[2, 1].plot(per_animal_activity_EC_above[animal], color='green', alpha=0.5)
    axs[2, 1].plot(np.mean(np.stack(per_animal_activity_EC_above), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean R vs Vel above 0")
    axs[2, 1].set_xlabel("Position Bin")
    axs[2, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 1].set_title("EC Per Animal Residuals", fontsize=12)
    else:
        axs[2, 1].set_title("EC Per Animal", fontsize=12)
    axs[2, 1].set_ylim(-1, 1)
    axs[2, 1].legend(fontsize=10)

    for animal in range(len(per_animal_activity_EC_below)):
        axs[2, 2].plot(per_animal_activity_EC_below[animal], color='limegreen', alpha=0.5)
    axs[2, 2].plot(np.mean(np.stack(per_animal_activity_EC_below), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean R vs Vel below 0")
    axs[2, 2].set_xlabel("Position Bin")
    axs[2, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 2].set_title("EC Per Animal Residuals", fontsize=12)
    else:
        axs[2, 2].set_title("EC Per Animal", fontsize=12)
    axs[2, 2].set_ylim(-1, 1)
    axs[2, 2].legend(fontsize=10)

    fig.tight_layout()
    plt.show()


def plot_first_and_last_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=True):

    first_raw_SST, last_raw_SST, first_mean_SST, first_sem_SST, last_mean_SST, last_sem_SST, animal_mean_residual_first_SST, animal_mean_residual_last_SST, first_quintile_residuals_SST, last_quintile_residuals_SST = compute_mean_and_sem_for_quintiles(activity_dict_SST, predicted_activity_dict_SST)

    first_raw_NDNF, last_raw_NDNF, first_mean_NDNF, first_sem_NDNF, last_mean_NDNF, last_sem_NDNF, animal_mean_residual_first_NDNF, animal_mean_residual_last_NDNF, first_quintile_residuals_NDNF, last_quintile_residuals_NDNF = compute_mean_and_sem_for_quintiles(activity_dict_NDNF, predicted_activity_dict_NDNF)

    first_raw_EC, last_raw_EC, first_mean_EC, first_sem_EC, last_mean_EC, last_sem_EC, animal_mean_residual_first_EC, animal_mean_residual_last_EC, first_quintile_residuals_EC, last_quintile_residuals_EC = compute_mean_and_sem_for_quintiles(activity_dict_EC, predicted_activity_dict_EC)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    axs[0, 0].plot(range(len(first_mean_SST)), first_mean_SST, color='blue', label='SST Mean Cell Q1', markersize=1)
    axs[0, 0].fill_between(range(len(first_mean_SST)), first_mean_SST+first_sem_SST, first_mean_SST-first_sem_SST, color='blue', alpha=0.1)
    axs[0, 0].plot(range(len(last_mean_SST)), last_mean_SST, color='cyan', label='SST Mean Cell Q5', markersize=1)
    axs[0, 0].fill_between(range(len(last_mean_SST)), last_mean_SST + last_sem_SST, last_mean_SST - last_sem_SST, color='cyan', alpha=0.2)
    axs[0, 0].set_xlabel("Position Bin")
    axs[0, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 0].set_title("SST Quintile Activity All Cells Residuals", fontsize=12)
    else:
        axs[0, 0].set_title("SST Quintile Activity All Cells", fontsize=12)
    axs[0, 0].set_ylim(-1, 1)
    axs[0, 0].legend()


    for animal in range(len(animal_mean_residual_first_SST)):
        axs[0, 1].plot(animal_mean_residual_first_SST[animal], color='blue', alpha=0.5)
    axs[0, 1].plot(np.mean(np.stack(animal_mean_residual_first_SST), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean Animal Q1")
    axs[0, 1].set_xlabel("Position Bin")
    axs[0, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 1].set_title("SST Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[0, 1].set_title("SST Quintile Activity Per Animal", fontsize=12)
    axs[0, 1].set_ylim(-1, 1)
    axs[0, 1].legend()

    for animal in range(len(animal_mean_residual_last_SST)):
        axs[0, 2].plot(animal_mean_residual_last_SST[animal], color='cyan', alpha=0.5)
    axs[0, 2].plot(np.mean(np.stack(animal_mean_residual_last_SST), axis=0), color='red', alpha=0.9, linewidth=4, label="SST Mean Animal Q5")
    axs[0, 2].set_xlabel("Position Bin")
    axs[0, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 2].set_title("SST Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[0, 2].set_title("SST Quintile Activity Per Animal", fontsize=12)
    axs[0, 2].set_ylim(-1, 1)
    axs[0, 2].legend()

    axs[1, 0].plot(range(len(first_mean_NDNF)), first_mean_NDNF, color='orange', label='NDNF Mean Cell Q1', markersize=1)
    axs[1, 0].fill_between(range(len(first_mean_NDNF)), first_mean_NDNF + first_sem_NDNF, first_mean_NDNF - first_sem_NDNF, color='orange', alpha=0.3)
    axs[1, 0].plot(range(len(last_mean_NDNF)), last_mean_NDNF, color='red', label='NDNF Mean Cell Q5', markersize=1)
    axs[1, 0].fill_between(range(len(last_mean_NDNF)), last_mean_NDNF + last_sem_NDNF, last_mean_NDNF - last_sem_NDNF, color='red', alpha=0.2)
    axs[1, 0].set_xlabel("Position Bin")
    axs[1, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 0].set_title("NDNF Quintile Activity All Cells Residuals", fontsize=12)
    else:
        axs[1, 0].set_title("NDNF Quintile Activity All Cells", fontsize=12)
    axs[1, 0].set_ylim(-1, 1)
    axs[1, 0].legend()

    for animal in range(len(animal_mean_residual_first_NDNF)):
        axs[1, 1].plot(animal_mean_residual_first_NDNF[animal], color='orange', alpha=0.5)
    axs[1, 1].plot(np.mean(np.stack(animal_mean_residual_first_NDNF), axis=0), color='red', alpha=0.9, linewidth=4,label="NDNF Mean Animal Q1")
    axs[1, 1].set_xlabel("Position Bin")
    axs[1, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 1].set_title("NDNF Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[1, 1].set_title("NDNF Quintile Activity Per Animal", fontsize=12)
    axs[1, 1].set_ylim(-1, 1)
    axs[1, 1].legend()

    for animal in range(len(animal_mean_residual_last_NDNF)):
        axs[1, 2].plot(animal_mean_residual_last_NDNF[animal], color='red', alpha=0.5)
    axs[1, 2].plot(np.mean(np.stack(animal_mean_residual_last_NDNF), axis=0), color='red', alpha=0.9, linewidth=4, label="NDNF Mean Animal Q5")
    axs[1, 2].set_xlabel("Position Bin")
    axs[1, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[1, 2].set_title("NDNF Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[1, 2].set_title("NDNF Quintile Activity Per Animal", fontsize=12)
    axs[1, 2].set_ylim(-1, 1)
    axs[1, 2].legend()

    axs[2, 0].plot(range(len(first_mean_EC)), first_mean_EC, color='green', label='EC Mean Cell Q1', markersize=1)
    axs[2, 0].fill_between(range(len(first_mean_EC)), first_mean_EC + first_sem_EC, first_mean_EC - first_sem_EC, color='green', alpha=0.1)
    axs[2, 0].plot(range(len(last_mean_EC)), last_mean_EC, color='limegreen', label='EC Mean Cell Q5', markersize=1)
    axs[2, 0].fill_between(range(len(last_mean_EC)), last_mean_EC + last_sem_EC, last_mean_EC - last_sem_EC, color='limegreen', alpha=0.1)
    axs[2, 0].set_xlabel("Position Bin")
    axs[2, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 0].set_title("EC Quintile Activity All Cells Residuals", fontsize=12)
    else:
        axs[2, 0].set_title("EC Quintile Activity All Cells", fontsize=12)
    axs[2, 0].set_ylim(-1, 1)
    axs[2, 0].legend()

    for animal in range(len(animal_mean_residual_first_EC)):
        axs[2, 1].plot(animal_mean_residual_first_EC[animal], color='green', alpha=0.5)
    axs[2, 1].plot(np.mean(np.stack(animal_mean_residual_first_EC), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean Animal Q1")
    axs[2, 1].set_xlabel("Position Bin")
    axs[2, 1].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 1].set_title("EC Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[2, 1].set_title("EC Quintile Activity Per Animal", fontsize=12)
    axs[2, 1].set_ylim(-1, 1)
    axs[2, 1].legend()

    for animal in range(len(animal_mean_residual_last_EC)):
        axs[2, 2].plot(animal_mean_residual_last_EC[animal], color='limegreen', alpha=0.5)
    axs[2, 2].plot(np.mean(np.stack(animal_mean_residual_last_EC), axis=0), color='red', alpha=0.9, linewidth=4, label="EC Mean Animal Q5")
    axs[2, 2].set_xlabel("Position Bin")
    axs[2, 2].set_ylabel("z-score DF/F")
    if residual:
        axs[2, 2].set_title("EC Quintile Activity Per Animal Residuals", fontsize=12)
    else:
        axs[2, 2].set_title("EC Quintile Activity Per Animal", fontsize=12)
    axs[2, 2].set_ylim(-1, 1)
    axs[2, 2].legend()

    fig.tight_layout()
    plt.show()

def get_learning_heatmaps(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, max_or_min="min", residual=True):
    first_quintile_trial_av_raw_SST, last_quintile_trial_av_raw_SST, first_mean_residual_SST, first_sem_residual_SST, last_mean_residual_SST, last_sem_residual_SST, animal_mean_first_quintile_residuals_SST, animal_mean_last_quintile_residuals_SST, first_quintile_residuals_SST, last_quintile_residuals_SST = compute_mean_and_sem_for_quintiles(activity_dict_SST, predicted_activity_dict_SST)
    first_quintile_trial_av_raw_NDNF, last_quintile_trial_av_raw_NDNF, first_mean_residual_NDNF, first_sem_residual_NDNF, last_mean_residual_NDNF, last_sem_residual_NDNF, animal_mean_first_quintile_residuals_NDNF, animal_mean_last_quintile_residuals_NDNF, first_quintile_residuals_NDNF, last_quintile_residuals_NDNF = compute_mean_and_sem_for_quintiles(activity_dict_NDNF,
                                                                                                                                                                                                                                                                                                                                                                 predicted_activity_dict_NDNF)
    first_quintile_trial_av_raw_EC, last_quintile_trial_av_raw_EC, first_mean_residual_EC, first_sem_residual_EC, last_mean_residual_EC, last_sem_residual_EC, animal_mean_first_quintile_residuals_EC, animal_mean_last_quintile_residuals_EC, first_quintile_residuals_EC, last_quintile_residuals_EC = compute_mean_and_sem_for_quintiles(activity_dict_EC, predicted_activity_dict_EC)

    if max_or_min == "max":
        first_quintile_residuals_array_SST = np.stack(first_quintile_residuals_SST)
        first_quintile_residuals_array_SST_sorted = np.argsort(np.argmax(first_quintile_residuals_array_SST, axis=1))
        last_quintile_residuals_array_SST = np.stack(last_quintile_residuals_SST)
        last_quintile_residuals_array_SST_sorted = np.argsort(np.argmax(last_quintile_residuals_array_SST, axis=1))
        last_minus_first_array_SST = last_quintile_residuals_array_SST - first_quintile_residuals_array_SST
        last_minus_first_array_SST_sorted = np.argsort(np.argmax(last_minus_first_array_SST, axis=1))

        first_quintile_residuals_array_NDNF = np.stack(first_quintile_residuals_NDNF)
        first_quintile_residuals_array_NDNF_sorted = np.argsort(np.argmax(first_quintile_residuals_array_NDNF, axis=1))
        last_quintile_residuals_array_NDNF = np.stack(last_quintile_residuals_NDNF)
        last_quintile_residuals_array_NDNF_sorted = np.argsort(np.argmax(last_quintile_residuals_array_NDNF, axis=1))
        last_minus_first_array_NDNF = last_quintile_residuals_array_NDNF - first_quintile_residuals_array_NDNF
        last_minus_first_array_NDNF_sorted = np.argsort(np.argmax(last_minus_first_array_NDNF, axis=1))

        first_quintile_residuals_array_EC = np.stack(first_quintile_residuals_EC)
        first_quintile_residuals_array_EC_sorted = np.argsort(np.argmax(first_quintile_residuals_array_EC, axis=1))
        last_quintile_residuals_array_EC = np.stack(last_quintile_residuals_EC)
        last_quintile_residuals_array_EC_sorted = np.argsort(np.argmax(last_quintile_residuals_array_EC, axis=1))
        last_minus_first_array_EC = last_quintile_residuals_array_EC - first_quintile_residuals_array_EC
        last_minus_first_array_EC_sorted = np.argsort(np.argmax(last_minus_first_array_EC, axis=1))

        max_abs_val_SST = np.max(np.abs([first_quintile_residuals_array_SST, last_quintile_residuals_array_SST]))
        max_abs_val_NDNF = np.max(np.abs([first_quintile_residuals_array_NDNF, last_quintile_residuals_array_NDNF]))
        max_abs_val_EC = np.max(np.abs([first_quintile_residuals_array_EC, last_quintile_residuals_array_EC]))

        super_max = np.max(np.abs([max_abs_val_SST, max_abs_val_NDNF, max_abs_val_EC]))

        fig, axs = plt.subplots(3, 3, figsize=(18, 15))

        im1 = axs[0, 0].imshow(first_quintile_residuals_array_SST[first_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 0].set_title("Q1 Peak Sorted SST")
        axs[0, 0].set_ylabel("Cell #")
        axs[0, 0].set_xlabel("Position Bin")
        fig.colorbar(im1)

        im2 = axs[0, 1].imshow(last_quintile_residuals_array_SST[last_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 1].set_title("Q5 Peak Sorted SST")
        axs[0, 1].set_ylabel("Cell #")
        axs[0, 1].set_xlabel("Position Bin")
        fig.colorbar(im2)

        im3 = axs[0, 2].imshow(last_minus_first_array_SST[last_minus_first_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 2].set_title("Q5-Q1 Peak Sorted SST")
        axs[0, 2].set_ylabel("Cell #")
        axs[0, 2].set_xlabel("Position Bin")
        fig.colorbar(im3)

        im4 = axs[1, 0].imshow(first_quintile_residuals_array_NDNF[first_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 0].set_title("Q1 Peak Sorted NDNF")
        axs[1, 0].set_ylabel("Cell #")
        axs[1, 0].set_xlabel("Position Bin")
        fig.colorbar(im4)

        im5 = axs[1, 1].imshow(last_quintile_residuals_array_NDNF[last_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 1].set_title("Q5 Peak Sorted NDNF")
        axs[1, 1].set_ylabel("Cell #")
        axs[1, 1].set_xlabel("Position Bin")
        fig.colorbar(im5)

        im6 = axs[1, 2].imshow(last_minus_first_array_NDNF[last_minus_first_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 2].set_title("Q5-Q1 Peak Sorted NDNF")
        axs[1, 2].set_ylabel("Cell #")
        axs[1, 2].set_xlabel("Position Bin")
        fig.colorbar(im6)

        im7 = axs[2, 0].imshow(first_quintile_residuals_array_EC[first_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 0].set_title("Q1 Peak Sorted EC")
        axs[2, 0].set_ylabel("Cell #")
        axs[2, 0].set_xlabel("Position Bin")
        fig.colorbar(im7)

        im8 = axs[2, 1].imshow(last_quintile_residuals_array_EC[last_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 1].set_title("Q5 Peak Sorted EC")
        axs[2, 1].set_ylabel("Cell #")
        axs[2, 1].set_xlabel("Position Bin")
        fig.colorbar(im8)

        im9 = axs[2, 2].imshow(last_minus_first_array_EC[last_minus_first_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 2].set_title("Q5-Q1 Peak Sorted EC")
        axs[2, 2].set_ylabel("Cell #")
        axs[2, 2].set_xlabel("Position Bin")
        fig.colorbar(im9)

        plt.show()

    elif max_or_min == "min":
        first_quintile_residuals_array_SST = np.stack(first_quintile_residuals_SST)
        first_quintile_residuals_array_SST_sorted = np.argsort(np.argmin(first_quintile_residuals_array_SST, axis=1))
        last_quintile_residuals_array_SST = np.stack(last_quintile_residuals_SST)
        last_quintile_residuals_array_SST_sorted = np.argsort(np.argmin(last_quintile_residuals_array_SST, axis=1))
        last_minus_first_array_SST = last_quintile_residuals_array_SST - first_quintile_residuals_array_SST
        last_minus_first_array_SST_sorted = np.argsort(np.argmin(last_minus_first_array_SST, axis=1))

        first_quintile_residuals_array_NDNF = np.stack(first_quintile_residuals_NDNF)
        first_quintile_residuals_array_NDNF_sorted = np.argsort(np.argmin(first_quintile_residuals_array_NDNF, axis=1))
        last_quintile_residuals_array_NDNF = np.stack(last_quintile_residuals_NDNF)
        last_quintile_residuals_array_NDNF_sorted = np.argsort(np.argmin(last_quintile_residuals_array_NDNF, axis=1))
        last_minus_first_array_NDNF = last_quintile_residuals_array_NDNF - first_quintile_residuals_array_NDNF
        last_minus_first_array_NDNF_sorted = np.argsort(np.argmin(last_minus_first_array_NDNF, axis=1))

        first_quintile_residuals_array_EC = np.stack(first_quintile_residuals_EC)
        first_quintile_residuals_array_EC_sorted = np.argsort(np.argmin(first_quintile_residuals_array_EC, axis=1))
        last_quintile_residuals_array_EC = np.stack(last_quintile_residuals_EC)
        last_quintile_residuals_array_EC_sorted = np.argsort(np.argmin(last_quintile_residuals_array_EC, axis=1))
        last_minus_first_array_EC = last_quintile_residuals_array_EC - first_quintile_residuals_array_EC
        last_minus_first_array_EC_sorted = np.argsort(np.argmin(last_minus_first_array_EC, axis=1))

        max_abs_val_SST = np.max(np.abs([first_quintile_residuals_array_SST, last_quintile_residuals_array_SST]))
        max_abs_val_NDNF = np.max(np.abs([first_quintile_residuals_array_NDNF, last_quintile_residuals_array_NDNF]))
        max_abs_val_EC = np.max(np.abs([first_quintile_residuals_array_EC, last_quintile_residuals_array_EC]))

        super_max = np.max(np.abs([max_abs_val_SST, max_abs_val_NDNF, max_abs_val_EC]))

        fig, axs = plt.subplots(3, 3, figsize=(18, 15))

        im1 = axs[0, 0].imshow(first_quintile_residuals_array_SST[first_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 0].set_title("Q1 Min Sorted SST Residuals")
        axs[0, 0].set_ylabel("Cell #")
        axs[0, 0].set_xlabel("Position Bin")
        fig.colorbar(im1)

        im2 = axs[0, 1].imshow(last_quintile_residuals_array_SST[last_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 1].set_title("Q5 Min Sorted SST Residuals")
        axs[0, 1].set_ylabel("Cell #")
        axs[0, 1].set_xlabel("Position Bin")
        fig.colorbar(im2)

        im3 = axs[0, 2].imshow(last_minus_first_array_SST[last_minus_first_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 2].set_title("Q5-Q1 Min Sorted SST Residuals")
        axs[0, 2].set_ylabel("Cell #")
        axs[0, 2].set_xlabel("Position Bin")
        fig.colorbar(im3)

        im4 = axs[1, 0].imshow(first_quintile_residuals_array_NDNF[first_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 0].set_title("Q1 Min Sorted NDNF Residuals")
        axs[1, 0].set_ylabel("Cell #")
        axs[1, 0].set_xlabel("Position Bin")
        fig.colorbar(im4)

        im5 = axs[1, 1].imshow(last_quintile_residuals_array_NDNF[last_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 1].set_title("Q5 Min Sorted NDNF Residuals")
        axs[1, 1].set_ylabel("Cell #")
        axs[1, 1].set_xlabel("Position Bin")
        fig.colorbar(im5)

        im6 = axs[1, 2].imshow(last_minus_first_array_NDNF[last_minus_first_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 2].set_title("Q5-Q1 Min Sorted NDNF Residuals")
        axs[1, 2].set_ylabel("Cell #")
        axs[1, 2].set_xlabel("Position Bin")
        fig.colorbar(im6)

        im7 = axs[2, 0].imshow(first_quintile_residuals_array_EC[first_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 0].set_title("Q1 Min Sorted EC Residuals")
        axs[2, 0].set_ylabel("Cell #")
        axs[2, 0].set_xlabel("Position Bin")
        fig.colorbar(im7)

        im8 = axs[2, 1].imshow(last_quintile_residuals_array_EC[last_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 1].set_title("Q5 Min Sorted EC Residuals")
        axs[2, 1].set_ylabel("Cell #")
        axs[2, 1].set_xlabel("Position Bin")
        fig.colorbar(im8)

        im9 = axs[2, 2].imshow(last_minus_first_array_EC[last_minus_first_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 2].set_title("Q5-Q1 Min Sorted EC Residuals")
        axs[2, 2].set_ylabel("Cell #")
        axs[2, 2].set_xlabel("Position Bin")
        fig.colorbar(im9)
        plt.show()

    else:
        raise ValueError("max_or_min takes 'max' or 'min'")

