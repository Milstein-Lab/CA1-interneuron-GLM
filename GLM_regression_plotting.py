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
    plt.title("SST Velocity-Subtracted Residuals vs Activity")
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
    plt.title("NDNF Velocity-Subtracted Residuals vs Activity")
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
    plt.title("EC Velocity-Subtracted Residuals vs Activity")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()


def plot_single_animal_average_trace(animal_mean_list_SST, animal_mean_residuals_SST, activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, cell_type="SST"):

    r2_variable_activity_dict_SST, r2_variable_residual_dict_SST = get_pop_correlation_to_variable(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, variable_to_correlate="Velocity"
    )
    mean_r2_per_animal_raw, sem_r2_per_animal_raw = get_per_animal_mean_r2(r2_variable_activity_dict_SST)
    mean_r2_per_animal_residual, sem_r2_per_animal_residual = get_per_animal_mean_r2(r2_variable_residual_dict_SST)

    color_map = {"SST": "blue", "NDNF": "orange", "EC": "green"}
    color_to_plot = color_map.get(cell_type, "gray")

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
    axs[2].set_title(f"{cell_type} Velocity-Subtracted Residual Activity Across Animals", fontsize=14)
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


def plot_correlations_single_variable_GLM(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict, activity_dict, cell_number, variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"], variable_into_GLM="Velocity", cell_type="SST"):

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
        # r2_activity, _ = pearsonr(flat_neuron_activity, y_pred_activity)
        r2_activity, _ = pearsonr(flat_variable, flat_neuron_activity)

        axs[idx].scatter(flat_variable, flat_neuron_activity, label="Data", alpha=0.6)
        axs[idx].plot(flat_variable, y_pred_activity, color='r', label="Best Fit", linewidth=2)
        axs[idx].set_title(f"{cell_type} Cell#{cell_number} {key} vs Activity\nR value: {r2_activity:.3f}")
        axs[idx].set_xlabel(f"{key} min/max normalized")
        axs[idx].set_ylabel("Activity (z-scored DF/F)")
        axs[idx].legend()

    plt.tight_layout()
    plt.show()

    variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]
    flat_variable_of_interest = variable_of_interest.flatten()

    r2_total, y_pred_total = compute_r_and_model(flat_prediction_total, flat_neuron_activity)

    r2_variable_residual, y_pred_variable_residual = compute_r_and_model(flat_variable_of_interest, flat_residual_total)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(flat_prediction_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[0].plot(flat_prediction_total, y_pred_total, color='r', label="Best Fit", linewidth=2)
    axs[0].set_title(f"{cell_type} Cell#{cell_number} Prediction based on just {variable_into_GLM} vs Activity\nR value: {r2_total:.3f}")
    axs[0].set_xlabel(f"{variable_into_GLM} Prediction (z-scored DF/F)")
    axs[0].set_ylabel("Activity (z-scored DF/F)")
    axs[0].legend()

    axs[1].scatter(flat_variable_of_interest, flat_residual_total, label="Data", alpha=0.6)
    axs[1].plot(flat_variable_of_interest, y_pred_variable_residual, color='r', label="Best Fit", linewidth=2)
    axs[1].set_title(f"{cell_type} Cell#{cell_number} {variable_into_GLM}-Subtracted Residuals vs {variable_into_GLM}:\nR value: {r2_variable_residual:.3f}")
    axs[1].set_xlabel(f"{variable_into_GLM} min/max normalized")
    axs[1].set_ylabel(f"Residuals (z-scored DF/F)")
    axs[1].legend()


    flat_residual_total = flat_residual_total.reshape(neuron_activity_list[cell_number].shape)
    flat_neuron_activity = flat_neuron_activity.reshape(neuron_activity_list[cell_number].shape)

    axs[2].plot(np.mean(predictions_list[cell_number], axis=1), color='b', label=f"{variable_into_GLM} Prediction")
    axs[2].plot(np.mean(flat_neuron_activity, axis=1), color='k', label=f"Raw Activity")
    axs[2].plot(np.mean(flat_residual_total, axis=1), color='r', label=f" Residual")
    axs[2].set_title(f"{cell_type} Cell#{cell_number} Trial Averaged")
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

    plt.xticks(x_positions, [f'Activity vs {variable_to_correlate}', f'Velocity-Subtracted Residuals vs {variable_to_correlate}'])
    plt.ylabel("R Value")
    plt.title(f"{cell_type} R Values")
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
        plt.title(f"Velocity-Subtracted Residuals: {fill_in} Animal Average")
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
        plt.title(f'Velocity-Subtracted Residuals: {fill_in} All Cells')
    else:
        plt.title(f'{fill_in} All Cells')
    plt.xticks(bin_centers, bin_labels)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_trial_averages(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(
        activity_dict_EC, predicted_activity_dict_EC)

    if residual:
        trial_av_activity_SST = trial_average(cell_residual_list_SST)
        trial_av_activity_NDNF = trial_average(cell_residual_list_NDNF)
        trial_av_activity_EC = trial_average(cell_residual_list_EC)
    else:
        trial_av_activity_SST = trial_average(neuron_activity_list_SST)
        trial_av_activity_NDNF = trial_average(neuron_activity_list_NDNF)
        trial_av_activity_EC = trial_average(neuron_activity_list_EC)

    trial_av_activity_SST_stack = np.stack(trial_av_activity_SST)
    trial_av_activity_NDNF_stack = np.stack(trial_av_activity_NDNF)
    trial_av_activity_EC_stack = np.stack(trial_av_activity_EC)

    trial_av_activity_SST = normalize(trial_av_activity_SST_stack, norm="z_score", per_cell=True)
    trial_av_activity_NDNF = normalize(trial_av_activity_NDNF_stack, norm="z_score", per_cell=True)
    trial_av_activity_EC = normalize(trial_av_activity_EC_stack, norm="z_score", per_cell=True)

    if which_to_plot == "argmin":
        sorted_indices_SST = np.argsort(np.argmin(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmin(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmin(trial_av_activity_EC, axis=1))

        plot_sorted_activity(trial_av_activity_SST, sorted_indices_SST, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} SST Min Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_NDNF, sorted_indices_NDNF, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} NDNF Min Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_EC, sorted_indices_EC, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} EC Min Sorted", "Cell ID", "Position Bins")


    elif which_to_plot == "argmax":
        sorted_indices_SST = np.argsort(np.argmax(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmax(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmax(trial_av_activity_EC, axis=1))

        plot_sorted_activity(trial_av_activity_SST, sorted_indices_SST, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} SST Peak Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_NDNF, sorted_indices_NDNF, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} NDNF Peak Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_EC, sorted_indices_EC, f"{'Velocity-Subtracted Residuals:' if residual else 'Raw Data'} EC Peak Sorted", "Cell ID", "Position Bins")


    else:
        raise ValueError("Options for which_to_plot are 'argmin' or 'argmax'")


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
                ax.set_title(f"{cell_types[i]} Velocity-Subtracted Residuals: \n Position of Min Firing (Animal)")
            else:
                ax.set_title(f"{cell_types[i]} Per Animal Position of Min Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="max":
            ax.set_xlabel(f"Position Bin of Maximum Firing")
            if residual:
                ax.set_title(f"{cell_types[i]} Velocity-Subtracted Residuals: \n Position of Max Firing (Animal)")
            elif not residual:
                ax.set_title(f"{cell_types[i]} Per Animal Position of Max Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="pos":
            ax.set_xlabel(f"Positive Selectivity Index")
            if residual:
                ax.set_title(f"{cell_types[i]} Velocity-Subtracted Residuals: \n Positive Selectivity (Animal)")
            elif not residual:
                ax.set_title(f"{cell_types[i]} Animal Average Position of Max Firing")
            ax.legend()
            ax.set_ylabel("Percentile of Data")
            ax.set_yticks(ticks=bin_centers)
            ax.set_yticklabels(percentile_labels)

        elif pos_neg_selectivity_max_or_min=="neg":
            ax.set_xlabel(f"Negative Selectivity Index")
            if residual:
                ax.set_title(f"{cell_types[i]} Velocity-Subtracted Residuals: \n Negative Selectivity (Animal)")
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
            fill_in = 'Position Bin of Min Firing'
        elif pos_neg_selectivity_max_or_min == "max":
            fill_in = 'Position Bin of Max Firing'

        if pos_neg_selectivity_max_or_min == "neg" or pos_neg_selectivity_max_or_min == "pos":
            axs[i].set_xlabel(f"{fill_in} Index")
        else:
            axs[i].set_xlabel(fill_in)
        axs[i].set_ylabel('Fraction of Cells')
        if residual:
            axs[i].set_title(f'Velocity-Subtracted Residuals: \n {fill_in}(All Cells)', fontsize=14)
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
    per_animal_activity_SST_above, per_animal_activity_SST_below, per_animal_activity_NDNF_above, per_animal_activity_NDNF_below, per_animal_activity_EC_above, per_animal_activity_EC_below = get_animal_vel_correlations_activity(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=residual)

    mean_SST_above, sem_SST_above, mean_SST_below, sem_SST_below, mean_NDNF_above, sem_NDNF_above, mean_NDNF_below, sem_NDNF_below, mean_EC_above, sem_EC_above, mean_EC_below, sem_EC_below = get_activity_by_r2_groups(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=residual)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    axs[0, 0].plot(mean_SST_above, color='blue', label='SST R vs Vel above 0', markersize=1)
    axs[0, 0].fill_between(range(len(mean_SST_above)), mean_SST_above - sem_SST_above, mean_SST_above + sem_SST_above, color='blue', alpha=0.1)
    axs[0, 0].plot(mean_SST_below, color='cyan', label='SST R vs Vel below 0', markersize=1)
    axs[0, 0].fill_between(range(len(mean_SST_below)), mean_SST_below - sem_SST_below, mean_SST_below + sem_SST_below, color='cyan', alpha=0.2)
    axs[0, 0].set_xlabel("Position Bin")
    axs[0, 0].set_ylabel("z-score DF/F")
    if residual:
        axs[0, 0].set_title("SST Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[0, 1].set_title("SST Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[0, 2].set_title("SST Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[1, 0].set_title("NDNF Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[1, 1].set_title("NDNF Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[1, 2].set_title("NDNF Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[2, 0].set_title("EC Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[2, 1].set_title("EC Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[2, 2].set_title("EC Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[0, 0].set_title("SST Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[0, 1].set_title("SST Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[0, 2].set_title("SST Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[1, 0].set_title("NDNF Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[1, 1].set_title("NDNF Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[1, 2].set_title("NDNF Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[2, 0].set_title("EC Velocity-Subtracted Residuals(All Cells)", fontsize=12)
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
        axs[2, 1].set_title("EC Velocity-Subtracted Residuals(Animal)", fontsize=12)
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
        axs[2, 2].set_title("EC Velocity-Subtracted Residuals(Animal)", fontsize=12)
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

        fig, axs = plt.subplots(3, 2, figsize=(18, 15))

        im1 = axs[0, 0].imshow(first_quintile_residuals_array_SST[first_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 0].set_title("SST Velocity-Subtracted Residuals: Q1 Peak Sorted")
        axs[0, 0].set_ylabel("Cell #")
        axs[0, 0].set_xlabel("Position Bin")
        fig.colorbar(im1)

        im2 = axs[0, 1].imshow(last_quintile_residuals_array_SST[last_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 1].set_title("SST Velocity-Subtracted Residuals: Q5 Peak Sorted")
        axs[0, 1].set_ylabel("Cell #")
        axs[0, 1].set_xlabel("Position Bin")
        fig.colorbar(im2)
        #
        # im3 = axs[0, 2].imshow(last_minus_first_array_SST[last_minus_first_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[0, 2].set_title("SST Velocity-Subtracted Residuals: Q5-Q1 Peak Sorted")
        # axs[0, 2].set_ylabel("Cell #")
        # axs[0, 2].set_xlabel("Position Bin")
        # fig.colorbar(im3)

        im4 = axs[1, 0].imshow(first_quintile_residuals_array_NDNF[first_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 0].set_title("NDNF Velocity-Subtracted Residuals: Q1 Peak Sorted")
        axs[1, 0].set_ylabel("Cell #")
        axs[1, 0].set_xlabel("Position Bin")
        fig.colorbar(im4)

        im5 = axs[1, 1].imshow(last_quintile_residuals_array_NDNF[last_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 1].set_title("NDNF Velocity-Subtracted Residuals: Q5 Peak Sorted")
        axs[1, 1].set_ylabel("Cell #")
        axs[1, 1].set_xlabel("Position Bin")
        fig.colorbar(im5)

        # im6 = axs[1, 2].imshow(last_minus_first_array_NDNF[last_minus_first_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[1, 2].set_title("NDNF Velocity-Subtracted Residuals: Q5-Q1 Peak Sorted")
        # axs[1, 2].set_ylabel("Cell #")
        # axs[1, 2].set_xlabel("Position Bin")
        # fig.colorbar(im6)

        im7 = axs[2, 0].imshow(first_quintile_residuals_array_EC[first_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 0].set_title("EC Velocity-Subtracted Residuals: Q1 Peak Sorted")
        axs[2, 0].set_ylabel("Cell #")
        axs[2, 0].set_xlabel("Position Bin")
        fig.colorbar(im7)

        im8 = axs[2, 1].imshow(last_quintile_residuals_array_EC[last_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 1].set_title("EC Velocity-Subtracted Residuals: Q5 Peak Sorted")
        axs[2, 1].set_ylabel("Cell #")
        axs[2, 1].set_xlabel("Position Bin")
        fig.colorbar(im8)

        # im9 = axs[2, 2].imshow(last_minus_first_array_EC[last_minus_first_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[2, 2].set_title("EC Velocity-Subtracted Residuals: Q5-Q1 Peak Sorted")
        # axs[2, 2].set_ylabel("Cell #")
        # axs[2, 2].set_xlabel("Position Bin")
        # fig.colorbar(im9)

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

        fig, axs = plt.subplots(3, 2, figsize=(14, 15))

        im1 = axs[0, 0].imshow(first_quintile_residuals_array_SST[first_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 0].set_title("SST Velocity-Subtracted Residuals: Q1 Min Sorted")
        axs[0, 0].set_ylabel("Cell #")
        axs[0, 0].set_xlabel("Position Bin")
        fig.colorbar(im1)

        im2 = axs[0, 1].imshow(last_quintile_residuals_array_SST[last_quintile_residuals_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[0, 1].set_title("SST Velocity-Subtracted Residuals: Q5 Min Sorted")
        axs[0, 1].set_ylabel("Cell #")
        axs[0, 1].set_xlabel("Position Bin")
        fig.colorbar(im2)

        # im3 = axs[0, 2].imshow(last_minus_first_array_SST[last_minus_first_array_SST_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[0, 2].set_title("SST Velocity-Subtracted Residuals: Q5-Q1 Min Sorted")
        # axs[0, 2].set_ylabel("Cell #")
        # axs[0, 2].set_xlabel("Position Bin")
        # fig.colorbar(im3)

        im4 = axs[1, 0].imshow(first_quintile_residuals_array_NDNF[first_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 0].set_title("NDNF Velocity-Subtracted Residuals: Q1 Min Sorted")
        axs[1, 0].set_ylabel("Cell #")
        axs[1, 0].set_xlabel("Position Bin")
        fig.colorbar(im4)

        im5 = axs[1, 1].imshow(last_quintile_residuals_array_NDNF[last_quintile_residuals_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[1, 1].set_title("NDNF Velocity-Subtracted Residuals: Q5 Min Sorted")
        axs[1, 1].set_ylabel("Cell #")
        axs[1, 1].set_xlabel("Position Bin")
        fig.colorbar(im5)

        # im6 = axs[1, 2].imshow(last_minus_first_array_NDNF[last_minus_first_array_NDNF_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[1, 2].set_title("NDNF Velocity-Subtracted Residuals: Q5-Q1 Min Sorted")
        # axs[1, 2].set_ylabel("Cell #")
        # axs[1, 2].set_xlabel("Position Bin")
        # fig.colorbar(im6)

        im7 = axs[2, 0].imshow(first_quintile_residuals_array_EC[first_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 0].set_title("EC Velocity-Subtracted Residuals: Q1 Min Sorted")
        axs[2, 0].set_ylabel("Cell #")
        axs[2, 0].set_xlabel("Position Bin")
        fig.colorbar(im7)

        im8 = axs[2, 1].imshow(last_quintile_residuals_array_EC[last_quintile_residuals_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        axs[2, 1].set_title("EC Velocity-Subtracted Residuals: Q5 Min Sorted")
        axs[2, 1].set_ylabel("Cell #")
        axs[2, 1].set_xlabel("Position Bin")
        fig.colorbar(im8)

        # im9 = axs[2, 2].imshow(last_minus_first_array_EC[last_minus_first_array_EC_sorted, :], aspect='auto', cmap="bwr", vmin=-super_max, vmax=super_max)
        # axs[2, 2].set_title("EC Velocity-Subtracted Residuals: Q5-Q1 Min Sorted")
        # axs[2, 2].set_ylabel("Cell #")
        # axs[2, 2].set_xlabel("Position Bin")
        # fig.colorbar(im9)

        plt.show()

    else:
        raise ValueError("max_or_min takes 'max' or 'min'")


def plot_selectivity_vs_peak(activity_dict_SST, predicted_activity_dict_SST, residual=True, cell_type="SST"):
    SST_factor_list, SST_negative_selectivity = get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=True)

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)

    SST_argmax_list = []
    SST_argmin_list = []

    for i in cell_residual_list_SST:
        trial_average = np.mean(i, axis=1)
        SST_argmax_list.append(np.argmax(trial_average))
        SST_argmin_list.append(np.argmin(trial_average))

    if cell_type == "SST":
        color = "blue"
    elif cell_type == "NDNF":
        color = "orange"
    elif cell_type == "EC":
        color = "green"
    else:
        raise ValueError("options cell_type SST, NDNF, EC")

    plt.figure()
    X1 = np.array(SST_factor_list).reshape(-1, 1)
    Y1 = np.array(SST_argmax_list)
    model1 = LinearRegression()
    model1.fit(X1, Y1)
    y_pred1 = model1.predict(X1)

    r_value1, _ = pearsonr(SST_factor_list, SST_argmax_list)

    plt.scatter(Y1, X1, color=color)
    plt.title(f"{cell_type} Velocity-Subtracted Residuals: Positive Selectivity vs. Argmax R={r_value1:.3f}")
    plt.ylabel("Positive Selectivity")
    plt.xlabel("Argmax Position")
    plt.show()

    plt.figure()
    X2 = np.array(SST_negative_selectivity).reshape(-1, 1)
    Y2 = np.array(SST_argmin_list)
    model2 = LinearRegression()
    model2.fit(X2, Y2)
    y_pred2 = model2.predict(X2)

    r_value2, _ = pearsonr(SST_negative_selectivity, SST_argmin_list)

    plt.scatter(Y2, X2, color=color)
    plt.title(f"{cell_type} Velocity-Subtracted Residuals: Negative Selectivity vs. Argmin R={r_value2:.3f}")
    plt.ylabel("Negative Selectivity")
    plt.xlabel("Argmin Position")
    plt.show()


def plot_highest_positive_negative_selectivity(activity_dict_SST, predicted_activity_dict_SST, residual=True, cell_type="SST"):
    SST_factor_list, SST_negative_selectivity = get_selectivity_for_plotting(activity_dict_SST, predicted_activity_dict_SST, residual=residual)
    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(activity_dict_SST, predicted_activity_dict_SST)

    max_selectivity = np.argmax(SST_factor_list)
    max_neg_selectivity = np.argmax(SST_negative_selectivity)

    trial_average_max_residual = np.mean(cell_residual_list_SST[max_selectivity], axis=1)
    trial_average_max_negative_residual = np.mean(cell_residual_list_SST[max_neg_selectivity], axis=1)

    if cell_type == "SST":
        color = "blue"
    elif cell_type == "NDNF":
        color = "orange"
    elif cell_type == "EC":
        color = "green"
    else:
        raise ValueError("options cell_type SST, NDNF, EC")

    plt.figure()
    plt.plot(trial_average_max_residual, color=color)
    plt.title(f"Velocity-Subtracted Residuals: Highest Positive Selectivity {cell_type} Cell#{max_selectivity} Index={SST_factor_list[max_selectivity]:.3f} \n Peak Firing at Position Bin #{np.argmax(trial_average_max_residual)} Min Firing Bin #{np.argmin(trial_average_max_residual)}")
    plt.xlabel("Position Bin")
    plt.ylabel("z-scored DF/F")
    plt.show()

    plt.figure()
    plt.plot(trial_average_max_negative_residual, color=color)
    plt.title(f"Velocity-Subtracted Residuals: Highest Negative Selectivity {cell_type} Cell#{max_neg_selectivity} Index={SST_negative_selectivity[max_neg_selectivity]:.3f} \n Peak Firing at Position Bin #{np.argmax(trial_average_max_negative_residual)} Min Firing Bin #{np.argmin(trial_average_max_negative_residual)}")
    plt.xlabel("Position Bin")
    plt.ylabel("z-scored DF/F")
    plt.show()


def plot_velocity_across_quintiles_per_animal(filtered_factors_dict_SST):
    animal_q1_list = []
    animal_q2_list = []
    animal_q3_list = []
    animal_q4_list = []
    animal_q5_list = []

    for animal in filtered_factors_dict_SST:
        velocity = filtered_factors_dict_SST[animal]["Velocity"]
        sst_quintiles_velocity = split_into_quintiles(velocity)
        q1_velocity = sst_quintiles_velocity[0]
        mean_q1 = np.mean(q1_velocity)
        animal_q1_list.append(mean_q1)
        q2_velocity = sst_quintiles_velocity[1]
        mean_q2 = np.mean(q2_velocity)
        animal_q2_list.append(mean_q2)
        q3_velocity = sst_quintiles_velocity[2]
        mean_q3 = np.mean(q3_velocity)
        animal_q3_list.append(mean_q3)
        q4_velocity = sst_quintiles_velocity[3]
        mean_q4 = np.mean(q4_velocity)
        animal_q4_list.append(mean_q4)
        q5_velocity = sst_quintiles_velocity[4]
        mean_q5 = np.mean(q5_velocity)
        animal_q5_list.append(mean_q5)

    quintile_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    animal_quintile_data = np.vstack([animal_q1_list, animal_q2_list, animal_q3_list, animal_q4_list, animal_q5_list]).T

    plt.figure(figsize=(8, 6))

    for animal_data in animal_quintile_data:
        plt.plot(quintile_labels, animal_data, marker="o", linestyle="-", alpha=0.6)

    plt.xlabel("Quintile")
    plt.ylabel("Mean Velocity")
    plt.title("Mean Velocity Across Quintiles for Each Animal")

    plt.show()


def plot_correlations_single_variable_GLM_quintile(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict, activity_dict, cell_number, variable_to_correlate_list=["Licks", "Reward_loc", "Velocity"], variable_into_GLM="Velocity", cell_type="SST", quintile="Q1"):
    input_variables = {var: [] for var in factors_dict[next(iter(factors_dict))].keys()}
    filtered_input_variables = {var: [] for var in filtered_factors_dict[next(iter(filtered_factors_dict))].keys()}

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            for var in factors_dict[animal]:
                input_variables[var].append(factors_dict[animal][var])
            for var in filtered_factors_dict[animal]:
                filtered_input_variables[var].append(filtered_factors_dict[animal][var])

    neuron_activity_list, predictions_list, cell_residual_list = get_neuron_activity_prediction_residual(activity_dict, predicted_activity_dict)

    cell_activity = neuron_activity_list[cell_number]
    sst_quintiles_activity = split_into_quintiles(cell_activity)
    first_quintile_activity = sst_quintiles_activity[0]
    last_quintile_activity = sst_quintiles_activity[-1]

    cell_prediciton = predictions_list[cell_number]
    sst_quintiles_prediction = split_into_quintiles(cell_prediciton)
    first_quintile_prediction = sst_quintiles_prediction[0]
    last_quintile_prediction = sst_quintiles_prediction[-1]

    cell_residual = cell_residual_list[cell_number]
    sst_quintiles_residual = split_into_quintiles(cell_residual)
    first_quintile_residual = sst_quintiles_residual[0]
    last_quintile_residual = sst_quintiles_residual[-1]

    variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]
    sst_quintiles_velocity = split_into_quintiles(variable_of_interest)
    first_quintile_velocity = sst_quintiles_velocity[0]
    last_quintile_velocity = sst_quintiles_velocity[-1]

    if quintile == "Q1":
        flat_neuron_activity = first_quintile_activity.flatten()
        flat_prediction_total = first_quintile_prediction.flatten()
        flat_residual_total = last_quintile_residual.flatten()
        flat_variable_of_interest = first_quintile_velocity.flatten()

    elif quintile == "Q5":
        flat_neuron_activity = last_quintile_activity.flatten()
        flat_prediction_total = last_quintile_prediction.flatten()
        flat_residual_total = last_quintile_residual.flatten()
        flat_variable_of_interest = last_quintile_velocity.flatten()

    else:
        raise ValueError("options for quintile are Q1 or Q5")

    model_activity = LinearRegression()
    model_activity.fit(flat_variable_of_interest.reshape(-1, 1), flat_neuron_activity)
    y_pred_activity = model_activity.predict(flat_variable_of_interest.reshape(-1, 1))
    r2_activity, _ = pearsonr(flat_variable_of_interest, flat_neuron_activity)

    r2_total, y_pred_total = compute_r_and_model(flat_prediction_total, flat_neuron_activity)

    r2_variable_residual, y_pred_variable_residual = compute_r_and_model(flat_variable_of_interest, flat_residual_total)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(flat_variable_of_interest, flat_neuron_activity, label="Data", alpha=0.6)
    axs[0].plot(flat_variable_of_interest, y_pred_activity, color='r', label="Best Fit", linewidth=2)
    axs[0].set_title(f"{quintile} {cell_type} Cell#{cell_number} Velocity vs Activity\nR value: {r2_activity:.3f}")
    axs[0].set_xlabel(f"Velocity min/max normalized")
    axs[0].set_ylabel("Activity (z-scored DF/F)")
    axs[0].legend()

    axs[1].scatter(flat_prediction_total, flat_neuron_activity, label="Data", alpha=0.6)
    axs[1].plot(flat_prediction_total, y_pred_total, color='r', label="Best Fit", linewidth=2)
    axs[1].set_title(f"{quintile} {cell_type} Cell#{cell_number} Velocity Prediction vs Activity\nR value: {r2_total:.3f}")
    axs[1].set_xlabel(f"{variable_into_GLM} Prediction (z-scored DF/F)")
    axs[1].set_ylabel("Activity (z-scored DF/F)")
    axs[1].legend()

    axs[2].scatter(flat_variable_of_interest, flat_residual_total, label="Data", alpha=0.6)
    axs[2].plot(flat_variable_of_interest, y_pred_variable_residual, color='r', label="Best Fit", linewidth=2)
    axs[2].set_title(f"{quintile} {cell_type} Cell#{cell_number} {variable_into_GLM}-Subtracted Residuals vs {variable_into_GLM}:\nR value: {r2_variable_residual:.3f}")
    axs[2].set_xlabel(f"{variable_into_GLM} min/max normalized")
    axs[2].set_ylabel(f"Residuals (z-scored DF/F)")
    axs[2].legend()

    plt.show()

    if quintile == "Q1":
        residual_total = flat_residual_total.reshape(first_quintile_activity.shape)
        neuron_activity = flat_neuron_activity.reshape(first_quintile_activity.shape)
        prediction_total = flat_prediction_total.reshape(first_quintile_activity.shape)
    elif quintile == "Q5":
        residual_total = flat_residual_total.reshape(last_quintile_activity.shape)
        neuron_activity = flat_neuron_activity.reshape(last_quintile_activity.shape)
        prediction_total = flat_prediction_total.reshape(last_quintile_activity.shape)

    plt.figure()
    plt.plot(np.mean(prediction_total, axis=1), color='b', label=f"{variable_into_GLM} Prediction")
    plt.plot(np.mean(neuron_activity, axis=1), color='k', label=f"Raw Activity")
    plt.plot(np.mean(residual_total, axis=1), color='r', label=f" Residual")
    plt.title(f"{quintile} {cell_type} Cell#{cell_number} Trial Averaged")
    plt.xlabel('Position Bin')
    plt.ylabel('Activity (z-scored DF/F)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_first_last_trials_variable_across_animals(factors_dict_SST, variable_of_interest="Velocity", cell_type="SST"):
    trial_av_animal_first = []
    trial_av_animal_last = []

    for animal in factors_dict_SST:
        velocity = factors_dict_SST[animal][variable_of_interest]

        first_trials = [velocity[:, i] for i in range(10)]

        last_trials = [velocity[:, -i - 1] for i in range(10)]

        first_trials_array = np.stack(first_trials)
        last_trials_array = np.stack(last_trials)

        trial_av_animal_first.append(np.mean(first_trials_array, axis=0))
        trial_av_animal_last.append(np.mean(last_trials_array, axis=0))

    trial_av_animal_first_array = np.stack(trial_av_animal_first)
    trial_av_animal_last_array = np.stack(trial_av_animal_last)

    mean_trial_av_animal_first_array = np.mean(trial_av_animal_first_array, axis=0)
    mean_trial_av_animal_last_array = np.mean(trial_av_animal_last_array, axis=0)

    sem_trial_av_animal_first_array = sem(trial_av_animal_first_array, axis=0)
    sem_trial_av_animal_last_array = sem(trial_av_animal_last_array, axis=0)

    if cell_type == "SST":
        color_first = "blue"
        color_last = "cyan"
    elif cell_type == "NDNF":
        color_first = "orange"
        color_last = "red"
    elif cell_type == "EC":
        color_first = "green"
        color_last = "limegreen"
    else:
        raise ValueError("options cell_type SST, NDNF, EC")

    plt.figure()
    plt.plot(mean_trial_av_animal_first_array, color=color_first, label="first 10 trials")
    plt.fill_between(range(len(mean_trial_av_animal_first_array)), mean_trial_av_animal_first_array + sem_trial_av_animal_first_array, mean_trial_av_animal_first_array - sem_trial_av_animal_first_array, color=color_first, alpha=0.2)
    plt.plot(mean_trial_av_animal_last_array, color=color_last, label="last 10 trials")
    plt.fill_between(range(len(mean_trial_av_animal_last_array)), mean_trial_av_animal_last_array + sem_trial_av_animal_last_array, mean_trial_av_animal_last_array - sem_trial_av_animal_last_array, color=color_last, alpha=0.2)
    plt.title(f"{cell_type} {variable_of_interest} First vs Last Trials")
    plt.xlabel("Position Bin")
    if variable_of_interest == "Velocity":
        plt.ylabel("cm/sec")
    elif variable_of_interest == "Licks":
        plt.ylabel("licks/sec")
    plt.legend()
    plt.show()


def get_q1_q5_residuals_all_cells(GLM_params, factors_dict, filtered_factors_dict, predicted_activity_dict, activity_dict, variable_into_GLM="Velocity"):
    input_variables = {var: [] for var in factors_dict[next(iter(factors_dict))].keys()}
    filtered_input_variables = {var: [] for var in filtered_factors_dict[next(iter(filtered_factors_dict))].keys()}

    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            for var in factors_dict[animal]:
                input_variables[var].append(factors_dict[animal][var])
            for var in filtered_factors_dict[animal]:
                filtered_input_variables[var].append(filtered_factors_dict[animal][var])

    neuron_activity_list, predictions_list, cell_residual_list = get_neuron_activity_prediction_residual(activity_dict, predicted_activity_dict)

    activity_vs_velocity_r = []

    for cell_number in range(len(neuron_activity_list)):
        cell_activity_flat = neuron_activity_list[cell_number].flatten()
        velocity_flat = filtered_input_variables[variable_into_GLM][cell_number].flatten()

        activity_vs_velocity, _ = pearsonr(cell_activity_flat, velocity_flat)
        activity_vs_velocity_r.append(activity_vs_velocity)

    activity_vs_velocity_r_q1_above = []
    activity_vs_velocity_r_q2_above = []
    activity_vs_velocity_r_q3_above = []
    activity_vs_velocity_r_q4_above = []
    activity_vs_velocity_r_q5_above = []

    activity_vs_velocity_r_q1_below = []
    activity_vs_velocity_r_q2_below = []
    activity_vs_velocity_r_q3_below = []
    activity_vs_velocity_r_q4_below = []
    activity_vs_velocity_r_q5_below = []

    for cell_number in range(len(neuron_activity_list)):

        if activity_vs_velocity_r[cell_number] > 0:

            cell_activity = neuron_activity_list[cell_number]
            velocity = filtered_input_variables[variable_into_GLM][cell_number]

            sst_quintiles_activity = split_into_quintiles(cell_activity)
            sst_quintiles_velocity = split_into_quintiles(velocity)

            q1_activity = sst_quintiles_activity[0].flatten()
            q2_activity = sst_quintiles_activity[1].flatten()
            q3_activity = sst_quintiles_activity[2].flatten()
            q4_activity = sst_quintiles_activity[3].flatten()
            q5_activity = sst_quintiles_activity[4].flatten()

            q1_velocity = sst_quintiles_velocity[0].flatten()
            q2_velocity = sst_quintiles_velocity[1].flatten()
            q3_velocity = sst_quintiles_velocity[2].flatten()
            q4_velocity = sst_quintiles_velocity[3].flatten()
            q5_velocity = sst_quintiles_velocity[4].flatten()

            activity_vs_velocity_q1, _ = pearsonr(q1_activity, q1_velocity)
            activity_vs_velocity_q2, _ = pearsonr(q1_activity, q2_velocity)
            activity_vs_velocity_q3, _ = pearsonr(q1_activity, q3_velocity)
            activity_vs_velocity_q4, _ = pearsonr(q1_activity, q4_velocity)
            activity_vs_velocity_q5, _ = pearsonr(q1_activity, q5_velocity)

            activity_vs_velocity_r_q1_above.append(activity_vs_velocity_q1)
            activity_vs_velocity_r_q2_above.append(activity_vs_velocity_q2)
            activity_vs_velocity_r_q3_above.append(activity_vs_velocity_q3)
            activity_vs_velocity_r_q4_above.append(activity_vs_velocity_q4)
            activity_vs_velocity_r_q5_above.append(activity_vs_velocity_q5)

        else:
            cell_activity = neuron_activity_list[cell_number]
            velocity = filtered_input_variables[variable_into_GLM][cell_number]

            sst_quintiles_activity = split_into_quintiles(cell_activity)
            sst_quintiles_velocity = split_into_quintiles(velocity)

            q1_activity = sst_quintiles_activity[0].flatten()
            q2_activity = sst_quintiles_activity[1].flatten()
            q3_activity = sst_quintiles_activity[2].flatten()
            q4_activity = sst_quintiles_activity[3].flatten()
            q5_activity = sst_quintiles_activity[4].flatten()

            q1_velocity = sst_quintiles_velocity[0].flatten()
            q2_velocity = sst_quintiles_velocity[1].flatten()
            q3_velocity = sst_quintiles_velocity[2].flatten()
            q4_velocity = sst_quintiles_velocity[3].flatten()
            q5_velocity = sst_quintiles_velocity[4].flatten()

            activity_vs_velocity_q1, _ = pearsonr(q1_activity, q1_velocity)
            activity_vs_velocity_q2, _ = pearsonr(q1_activity, q2_velocity)
            activity_vs_velocity_q3, _ = pearsonr(q1_activity, q3_velocity)
            activity_vs_velocity_q4, _ = pearsonr(q1_activity, q4_velocity)
            activity_vs_velocity_q5, _ = pearsonr(q1_activity, q5_velocity)

            activity_vs_velocity_r_q1_below.append(activity_vs_velocity_q1)
            activity_vs_velocity_r_q2_below.append(activity_vs_velocity_q2)
            activity_vs_velocity_r_q3_below.append(activity_vs_velocity_q3)
            activity_vs_velocity_r_q4_below.append(activity_vs_velocity_q4)
            activity_vs_velocity_r_q5_below.append(activity_vs_velocity_q5)

    residual_vs_velocity_r_q1_above = []
    residual_vs_velocity_r_q2_above = []
    residual_vs_velocity_r_q3_above = []
    residual_vs_velocity_r_q4_above = []
    residual_vs_velocity_r_q5_above = []

    residual_vs_velocity_r_q1_below = []
    residual_vs_velocity_r_q2_below = []
    residual_vs_velocity_r_q3_below = []
    residual_vs_velocity_r_q4_below = []
    residual_vs_velocity_r_q5_below = []

    for cell_number in range(len(neuron_activity_list)):

        if activity_vs_velocity_r[cell_number] > 0:

            cell_residual = cell_residual_list[cell_number]
            variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]

            sst_quintiles_residual = split_into_quintiles(cell_residual)
            sst_quintiles_velocity = split_into_quintiles(variable_of_interest)

            q1_residual = sst_quintiles_residual[0].flatten()
            q2_residual = sst_quintiles_residual[1].flatten()
            q3_residual = sst_quintiles_residual[2].flatten()
            q4_residual = sst_quintiles_residual[3].flatten()
            q5_residual = sst_quintiles_residual[4].flatten()

            q1_velocity = sst_quintiles_velocity[0].flatten()
            q2_velocity = sst_quintiles_velocity[1].flatten()
            q3_velocity = sst_quintiles_velocity[2].flatten()
            q4_velocity = sst_quintiles_velocity[3].flatten()
            q5_velocity = sst_quintiles_velocity[4].flatten()

            residual_vs_velocity_q1, _ = pearsonr(q1_residual, q1_velocity)
            residual_vs_velocity_q2, _ = pearsonr(q1_residual, q2_velocity)
            residual_vs_velocity_q3, _ = pearsonr(q1_residual, q3_velocity)
            residual_vs_velocity_q4, _ = pearsonr(q1_residual, q4_velocity)
            residual_vs_velocity_q5, _ = pearsonr(q1_residual, q5_velocity)

            residual_vs_velocity_r_q1_above.append(residual_vs_velocity_q1)
            residual_vs_velocity_r_q2_above.append(residual_vs_velocity_q2)
            residual_vs_velocity_r_q3_above.append(residual_vs_velocity_q3)
            residual_vs_velocity_r_q4_above.append(residual_vs_velocity_q4)
            residual_vs_velocity_r_q5_above.append(residual_vs_velocity_q5)

        else:
            cell_residual = cell_residual_list[cell_number]
            variable_of_interest = filtered_input_variables[variable_into_GLM][cell_number]

            sst_quintiles_residual = split_into_quintiles(cell_residual)
            sst_quintiles_velocity = split_into_quintiles(variable_of_interest)

            q1_residual = sst_quintiles_residual[0].flatten()
            q2_residual = sst_quintiles_residual[1].flatten()
            q3_residual = sst_quintiles_residual[2].flatten()
            q4_residual = sst_quintiles_residual[3].flatten()
            q5_residual = sst_quintiles_residual[4].flatten()

            q1_velocity = sst_quintiles_velocity[0].flatten()
            q2_velocity = sst_quintiles_velocity[1].flatten()
            q3_velocity = sst_quintiles_velocity[2].flatten()
            q4_velocity = sst_quintiles_velocity[3].flatten()
            q5_velocity = sst_quintiles_velocity[4].flatten()

            residual_vs_velocity_q1, _ = pearsonr(q1_residual, q1_velocity)
            residual_vs_velocity_q2, _ = pearsonr(q1_residual, q2_velocity)
            residual_vs_velocity_q3, _ = pearsonr(q1_residual, q3_velocity)
            residual_vs_velocity_q4, _ = pearsonr(q1_residual, q4_velocity)
            residual_vs_velocity_q5, _ = pearsonr(q1_residual, q5_velocity)

            residual_vs_velocity_r_q1_below.append(residual_vs_velocity_q1)
            residual_vs_velocity_r_q2_below.append(residual_vs_velocity_q2)
            residual_vs_velocity_r_q3_below.append(residual_vs_velocity_q3)
            residual_vs_velocity_r_q4_below.append(residual_vs_velocity_q4)
            residual_vs_velocity_r_q5_below.append(residual_vs_velocity_q5)

    quintile_labels = ["Q1 Activity vs Velocity", "Q1 Residuals vs Velocity",
                       "Q5 Activity vs Velocity", "Q5 Residuals vs Velocity"]
    x_positions_activity = np.array([1, 3])  # Positions for Activity vs Velocity
    x_positions_residual = np.array([2, 4])  # Positions for Residuals vs Velocity

    # Organizing data
    quintile_data_above_activity = np.array([activity_vs_velocity_r_q1_above, activity_vs_velocity_r_q5_above]).T
    quintile_data_below_activity = np.array([activity_vs_velocity_r_q1_below, activity_vs_velocity_r_q5_below]).T
    quintile_data_above_residual = np.array([residual_vs_velocity_r_q1_above, residual_vs_velocity_r_q5_above]).T
    quintile_data_below_residual = np.array([residual_vs_velocity_r_q1_below, residual_vs_velocity_r_q5_below]).T

    plt.figure(figsize=(10, 6))

    # Plot above 0 correlations
    for i in range(len(residual_vs_velocity_r_q1_above)):
        plt.plot(x_positions_activity, quintile_data_above_activity[i], 'o', color='b', alpha=0.7)
        plt.plot(x_positions_residual, quintile_data_above_residual[i], 'o', color='b', alpha=0.7)
        plt.plot([x_positions_activity, x_positions_residual],
                 [quintile_data_above_activity[i], quintile_data_above_residual[i]], color='b', alpha=0.5)

    for i in range(len(residual_vs_velocity_r_q1_below)):
        plt.plot(x_positions_activity, quintile_data_below_activity[i], 'o', color='r', alpha=0.7)
        plt.plot(x_positions_residual, quintile_data_below_residual[i], 'o', color='r', alpha=0.7)
        plt.plot([x_positions_activity, x_positions_residual],
                 [quintile_data_below_activity[i], quintile_data_below_residual[i]], color='r', alpha=0.5)

    means_above_activity = np.mean(quintile_data_above_activity, axis=0)
    means_above_residual = np.mean(quintile_data_above_residual, axis=0)
    means_below_activity = np.mean(quintile_data_below_activity, axis=0)
    means_below_residual = np.mean(quintile_data_below_residual, axis=0)

    plt.hlines(means_above_activity, x_positions_activity - 0.1, x_positions_activity + 0.1, color='b', linewidth=3)
    plt.hlines(means_above_residual, x_positions_residual - 0.1, x_positions_residual + 0.1, color='b', linewidth=3)
    plt.hlines(means_below_activity, x_positions_activity - 0.1, x_positions_activity + 0.1, color='r', linewidth=3)
    plt.hlines(means_below_residual, x_positions_residual - 0.1, x_positions_residual + 0.1, color='r', linewidth=3)

    plt.xticks(np.arange(1, 5), quintile_labels)
    plt.xlabel("Quintiles")
    plt.ylabel("Pearson Correlation (r)")
    plt.title("Blue=Above, Red=Below 0 Velocity Correlation with All Trials For Cell")
    plt.show()


def plot_r2_above_below_per_animal_with_learning(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):
    mean_above_list, mean_below_list, sem_above_list, sem_below_list, trial_average_activity_list_SST = get_activity_by_r2_groups_plus_learning(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
        activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True)

    labels = ["SST", "NDNF", "EC"]
    colors_q1 = ["blue", "orange", "green"]
    colors_q5 = ["cyan", "red", "limegreen"]

    fig, axs = plt.subplots(3, 2, figsize=(7, 10))  # 3 rows, 2 columns

    # Plot "Above 0" Correlations (Left Column)
    for i in range(3):
        axs[i, 0].plot(mean_above_list[i * 2], color=colors_q1[i], label=f'{labels[i]} R Above 0 Q1', markersize=1)
        axs[i, 0].fill_between(range(len(mean_above_list[i * 2])),
                               mean_above_list[i * 2] - sem_above_list[i * 2],
                               mean_above_list[i * 2] + sem_above_list[i * 2],
                               color=colors_q1[i], alpha=0.1)
        axs[i, 0].plot(mean_above_list[i * 2 + 1], color=colors_q5[i], label=f'{labels[i]} R Above 0 Q5', markersize=1)
        axs[i, 0].fill_between(range(len(mean_above_list[i * 2 + 1])),
                               mean_above_list[i * 2 + 1] - sem_above_list[i * 2 + 1],
                               mean_above_list[i * 2 + 1] + sem_above_list[i * 2 + 1],
                               color=colors_q5[i], alpha=0.2)
        axs[i, 0].set_xlabel("Position Bin")
        axs[i, 0].set_ylabel("z-score DF/F")
        axs[i, 0].set_title(f"{labels[i]} Velocity-Subtracted Residuals ABOVE ZERO", fontsize=9)
        axs[i, 0].set_ylim(-1, 1)
        axs[i, 0].legend(fontsize=10)

    for i in range(3):
        axs[i, 1].plot(mean_below_list[i * 2], color=colors_q1[i], label=f'{labels[i]} R Below 0 Q1', markersize=1)
        axs[i, 1].fill_between(range(len(mean_below_list[i * 2])),
                               mean_below_list[i * 2] - sem_below_list[i * 2],
                               mean_below_list[i * 2] + sem_below_list[i * 2],
                               color=colors_q1[i], alpha=0.1)
        axs[i, 1].plot(mean_below_list[i * 2 + 1], color=colors_q5[i], label=f'{labels[i]} R Below 0 Q5', markersize=1)
        axs[i, 1].fill_between(range(len(mean_below_list[i * 2 + 1])),
                               mean_below_list[i * 2 + 1] - sem_below_list[i * 2 + 1],
                               mean_below_list[i * 2 + 1] + sem_below_list[i * 2 + 1],
                               color=colors_q5[i], alpha=0.2)
        axs[i, 1].set_xlabel("Position Bin")
        axs[i, 1].set_ylabel("z-score DF/F")
        axs[i, 1].set_title(f"{labels[i]} Velocity-Subtracted Residuals BELOW ZERO", fontsize=9)
        axs[i, 1].set_ylim(-1, 1)
        axs[i, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_positive_negative_selectivity_by_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):

    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q5_NDNF = split_activity_and_prediction_into_quintiles(activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, predicted_activity_list_EC_q1, predicted_activity_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(activity_dict_EC, predicted_activity_dict_EC)

    SST_positive_selectivity_q1, SST_negative_selectivity_q1, NDNF_positive_selectivity_q1, NDNF_negative_selectivity_q1, EC_positive_selectivity_q1, EC_negative_selectivity_q1 = get_selectivity_for_plotting_lists(residual_q1_SST, residual_q1_NDNF, residual_q1_EC)

    SST_positive_selectivity_q5, SST_negative_selectivity_q5, NDNF_positive_selectivity_q5, NDNF_negative_selectivity_q5, EC_positive_selectivity_q5, EC_negative_selectivity_q5 = get_selectivity_for_plotting_lists(residual_q5_SST, residual_q5_NDNF, residual_q5_EC)

    animal_ID_list_SST = get_animal_ID_list(activity_dict_SST)
    animal_ID_list_NDNF = get_animal_ID_list(activity_dict_NDNF)
    animal_ID_list_EC = get_animal_ID_list(activity_dict_EC)

    mean_quantiles_SST_q1, sem_quantiles_SST_q1 = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_positive_selectivity_q1, n_bins=20)
    mean_quantiles_SST_q5, sem_quantiles_SST_q5 = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_positive_selectivity_q5, n_bins=20)

    mean_quantiles_NDNF_q1, sem_quantiles_NDNF_q1 = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_positive_selectivity_q1, n_bins=20)
    mean_quantiles_NDNF_q5, sem_quantiles_NDNF_q5 = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_positive_selectivity_q5, n_bins=20)

    mean_quantiles_EC_q1, sem_quantiles_EC_q1 = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_positive_selectivity_q1, n_bins=20)
    mean_quantiles_EC_q5, sem_quantiles_EC_q5 = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_positive_selectivity_q5, n_bins=20)

    mean_quantiles_SST_q1_negative, sem_quantiles_SST_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_negative_selectivity_q1, n_bins=20)
    mean_quantiles_SST_q5_negative, sem_quantiles_SST_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_negative_selectivity_q5, n_bins=20)

    mean_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_negative_selectivity_q1, n_bins=20)
    mean_quantiles_NDNF_q5_negative, sem_quantiles_NDNF_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_negative_selectivity_q5, n_bins=20)

    mean_quantiles_EC_q1_negative, sem_quantiles_EC_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_negative_selectivity_q1, n_bins=20)
    mean_quantiles_EC_q5_negative, sem_quantiles_EC_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_negative_selectivity_q5, n_bins=20)

    positive_mean_list = [mean_quantiles_SST_q1, mean_quantiles_SST_q5, mean_quantiles_NDNF_q1, mean_quantiles_NDNF_q5, mean_quantiles_EC_q1, mean_quantiles_EC_q5]
    positive_sem_list = [sem_quantiles_SST_q1, sem_quantiles_SST_q5, sem_quantiles_NDNF_q1, sem_quantiles_NDNF_q5, sem_quantiles_EC_q1, sem_quantiles_EC_q5]

    negative_mean_list = [mean_quantiles_SST_q1_negative, mean_quantiles_SST_q5_negative, mean_quantiles_NDNF_q1_negative, mean_quantiles_NDNF_q5_negative, mean_quantiles_EC_q1_negative, mean_quantiles_EC_q5_negative]
    negative_sem_list = [sem_quantiles_SST_q1_negative, sem_quantiles_SST_q5_negative, sem_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q5_negative, sem_quantiles_EC_q1_negative, sem_quantiles_EC_q5_negative]

    return positive_mean_list, positive_sem_list, negative_mean_list, negative_sem_list


def plot_positive_negative_selectivity_by_quintile(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC):

    activity_list_SST_q1, activity_list_SST_q5, prediction_list_SST_q1, prediction_list_SST_q5, residual_q1_SST, residual_q5_SST = split_activity_and_prediction_into_quintiles(activity_dict_SST, predicted_activity_dict_SST)
    activity_list_NDNF_q1, activity_list_NDNF_q5, prediction_list_NDNF_q1, prediction_list_NDNF_q5, residual_q1_NDNF, residual_q5_NDNF = split_activity_and_prediction_into_quintiles(activity_dict_NDNF, predicted_activity_dict_NDNF)
    activity_list_EC_q1, activity_list_EC_q5, predicted_activity_list_EC_q1, predicted_activity_list_EC_q5, residual_q1_EC, residual_q5_EC = split_activity_and_prediction_into_quintiles(activity_dict_EC, predicted_activity_dict_EC)

    SST_positive_selectivity_q1, SST_negative_selectivity_q1, NDNF_positive_selectivity_q1, NDNF_negative_selectivity_q1, EC_positive_selectivity_q1, EC_negative_selectivity_q1 = get_selectivity_for_plotting_lists(residual_q1_SST, residual_q1_NDNF, residual_q1_EC)

    SST_positive_selectivity_q5, SST_negative_selectivity_q5, NDNF_positive_selectivity_q5, NDNF_negative_selectivity_q5, EC_positive_selectivity_q5, EC_negative_selectivity_q5 = get_selectivity_for_plotting_lists(residual_q5_SST, residual_q5_NDNF, residual_q5_EC)

    animal_ID_list_SST = get_animal_ID_list(activity_dict_SST)
    animal_ID_list_NDNF = get_animal_ID_list(activity_dict_NDNF)
    animal_ID_list_EC = get_animal_ID_list(activity_dict_EC)

    mean_quantiles_SST_q1, sem_quantiles_SST_q1 = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_positive_selectivity_q1, n_bins=20)
    mean_quantiles_SST_q5, sem_quantiles_SST_q5 = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_positive_selectivity_q5, n_bins=20)

    mean_quantiles_NDNF_q1, sem_quantiles_NDNF_q1 = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_positive_selectivity_q1, n_bins=20)
    mean_quantiles_NDNF_q5, sem_quantiles_NDNF_q5 = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_positive_selectivity_q5, n_bins=20)

    mean_quantiles_EC_q1, sem_quantiles_EC_q1 = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_positive_selectivity_q1, n_bins=20)
    mean_quantiles_EC_q5, sem_quantiles_EC_q5 = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_positive_selectivity_q5, n_bins=20)

    mean_quantiles_SST_q1_negative, sem_quantiles_SST_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_negative_selectivity_q1, n_bins=20)
    mean_quantiles_SST_q5_negative, sem_quantiles_SST_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_SST, SST_negative_selectivity_q5, n_bins=20)

    mean_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_negative_selectivity_q1, n_bins=20)
    mean_quantiles_NDNF_q5_negative, sem_quantiles_NDNF_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_NDNF, NDNF_negative_selectivity_q5, n_bins=20)

    mean_quantiles_EC_q1_negative, sem_quantiles_EC_q1_negative = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_negative_selectivity_q1, n_bins=20)
    mean_quantiles_EC_q5_negative, sem_quantiles_EC_q5_negative = get_quantiles_for_cdf_list(animal_ID_list_EC, EC_negative_selectivity_q5, n_bins=20)

    positive_mean_list = [mean_quantiles_SST_q1, mean_quantiles_SST_q5, mean_quantiles_NDNF_q1, mean_quantiles_NDNF_q5, mean_quantiles_EC_q1, mean_quantiles_EC_q5]
    positive_sem_list = [sem_quantiles_SST_q1, sem_quantiles_SST_q5, sem_quantiles_NDNF_q1, sem_quantiles_NDNF_q5, sem_quantiles_EC_q1, sem_quantiles_EC_q5]

    negative_mean_list = [mean_quantiles_SST_q1_negative, mean_quantiles_SST_q5_negative, mean_quantiles_NDNF_q1_negative, mean_quantiles_NDNF_q5_negative, mean_quantiles_EC_q1_negative, mean_quantiles_EC_q5_negative]
    negative_sem_list = [sem_quantiles_SST_q1_negative, sem_quantiles_SST_q5_negative, sem_quantiles_NDNF_q1_negative, sem_quantiles_NDNF_q5_negative, sem_quantiles_EC_q1_negative, sem_quantiles_EC_q5_negative]

    return positive_mean_list, positive_sem_list, negative_mean_list, negative_sem_list


def plot_r2_above_below_per_animal_with_learning(activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True):
    mean_above_list, mean_below_list, sem_above_list, sem_below_list, trial_average_activity_list_SST = get_activity_by_r2_groups_plus_learning(
        activity_dict_SST, predicted_activity_dict_SST, filtered_factors_dict_SST,
        activity_dict_NDNF, predicted_activity_dict_NDNF, filtered_factors_dict_NDNF,
        activity_dict_EC, predicted_activity_dict_EC, filtered_factors_dict_EC, residual=True)

    labels = ["SST", "NDNF", "EC"]
    colors_q1 = ["blue", "orange", "green"]
    colors_q5 = ["cyan", "red", "limegreen"]

    fig, axs = plt.subplots(3, 2, figsize=(7, 10))  # 3 rows, 2 columns

    # Plot "Above 0" Correlations (Left Column)
    for i in range(3):
        axs[i, 0].plot(mean_above_list[i * 2], color=colors_q1[i], label=f'{labels[i]} R Above 0 Q1', markersize=1)
        axs[i, 0].fill_between(range(len(mean_above_list[i * 2])),
                               mean_above_list[i * 2] - sem_above_list[i * 2],
                               mean_above_list[i * 2] + sem_above_list[i * 2],
                               color=colors_q1[i], alpha=0.1)
        axs[i, 0].plot(mean_above_list[i * 2 + 1], color=colors_q5[i], label=f'{labels[i]} R Above 0 Q5', markersize=1)
        axs[i, 0].fill_between(range(len(mean_above_list[i * 2 + 1])),
                               mean_above_list[i * 2 + 1] - sem_above_list[i * 2 + 1],
                               mean_above_list[i * 2 + 1] + sem_above_list[i * 2 + 1],
                               color=colors_q5[i], alpha=0.2)
        axs[i, 0].set_xlabel("Position Bin")
        axs[i, 0].set_ylabel("z-score DF/F")
        axs[i, 0].set_title(f"{labels[i]} Velocity-Subtracted Residuals ABOVE ZERO", fontsize=9)
        axs[i, 0].set_ylim(-1, 1)
        axs[i, 0].legend(fontsize=10)

    for i in range(3):
        axs[i, 1].plot(mean_below_list[i * 2], color=colors_q1[i], label=f'{labels[i]} R Below 0 Q1', markersize=1)
        axs[i, 1].fill_between(range(len(mean_below_list[i * 2])),
                               mean_below_list[i * 2] - sem_below_list[i * 2],
                               mean_below_list[i * 2] + sem_below_list[i * 2],
                               color=colors_q1[i], alpha=0.1)
        axs[i, 1].plot(mean_below_list[i * 2 + 1], color=colors_q5[i], label=f'{labels[i]} R Below 0 Q5', markersize=1)
        axs[i, 1].fill_between(range(len(mean_below_list[i * 2 + 1])),
                               mean_below_list[i * 2 + 1] - sem_below_list[i * 2 + 1],
                               mean_below_list[i * 2 + 1] + sem_below_list[i * 2 + 1],
                               color=colors_q5[i], alpha=0.2)
        axs[i, 1].set_xlabel("Position Bin")
        axs[i, 1].set_ylabel("z-score DF/F")
        axs[i, 1].set_title(f"{labels[i]} Velocity-Subtracted Residuals BELOW ZERO", fontsize=9)
        axs[i, 1].set_ylim(-1, 1)
        axs[i, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_sorted_activity(data, sorted_indices, title, ylabel, xlabel):
    plt.figure()
    plt.imshow(data[sorted_indices, :], aspect='auto')
    plt.title(title)
    plt.colorbar(label='Activity')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


def plot_trial_averages(activity_dict_SST, predicted_activity_dict_SST, activity_dict_NDNF, predicted_activity_dict_NDNF, activity_dict_EC, predicted_activity_dict_EC, residual=False, which_to_plot="argmin"):

    neuron_activity_list_SST, predictions_list_SST, cell_residual_list_SST = get_neuron_activity_prediction_residual(
        activity_dict_SST, predicted_activity_dict_SST)
    neuron_activity_list_NDNF, predictions_list_NDNF, cell_residual_list_NDNF = get_neuron_activity_prediction_residual(
        activity_dict_NDNF, predicted_activity_dict_NDNF)
    neuron_activity_list_EC, predictions_list_EC, cell_residual_list_EC = get_neuron_activity_prediction_residual(
        activity_dict_EC, predicted_activity_dict_EC)

    if residual:
        trial_av_activity_SST = trial_average(cell_residual_list_SST)
        trial_av_activity_NDNF = trial_average(cell_residual_list_NDNF)
        trial_av_activity_EC = trial_average(cell_residual_list_EC)
    else:
        trial_av_activity_SST = trial_average(neuron_activity_list_SST)
        trial_av_activity_NDNF = trial_average(neuron_activity_list_NDNF)
        trial_av_activity_EC = trial_average(neuron_activity_list_EC)

    trial_av_activity_SST_stack = np.stack(trial_av_activity_SST)
    trial_av_activity_NDNF_stack = np.stack(trial_av_activity_NDNF)
    trial_av_activity_EC_stack = np.stack(trial_av_activity_EC)

    trial_av_activity_SST = normalize(trial_av_activity_SST_stack, norm="z_score", per_cell=True)
    trial_av_activity_NDNF = normalize(trial_av_activity_NDNF_stack, norm="z_score", per_cell=True)
    trial_av_activity_EC = normalize(trial_av_activity_EC_stack, norm="z_score", per_cell=True)

    if which_to_plot == "argmin":
        sorted_indices_SST = np.argsort(np.argmin(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmin(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmin(trial_av_activity_EC, axis=1))

        plot_sorted_activity(trial_av_activity_SST, sorted_indices_SST, f"{'Residuals' if residual else 'Raw Data'} SST Min Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_NDNF, sorted_indices_NDNF, f"{'Residuals' if residual else 'Raw Data'} NDNF Min Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_EC, sorted_indices_EC, f"{'Residuals' if residual else 'Raw Data'} EC Min Sorted", "Cell ID", "Position Bins")


    elif which_to_plot == "argmax":
        sorted_indices_SST = np.argsort(np.argmax(trial_av_activity_SST, axis=1))
        sorted_indices_NDNF = np.argsort(np.argmax(trial_av_activity_NDNF, axis=1))
        sorted_indices_EC = np.argsort(np.argmax(trial_av_activity_EC, axis=1))

        plot_sorted_activity(trial_av_activity_SST, sorted_indices_SST, f"{'Residuals' if residual else 'Raw Data'} SST Peak Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_NDNF, sorted_indices_NDNF, f"{'Residuals' if residual else 'Raw Data'} NDNF Peak Sorted", "Cell ID", "Position Bins")

        plot_sorted_activity(trial_av_activity_EC, sorted_indices_EC, f"{'Residuals' if residual else 'Raw Data'} EC Peak Sorted", "Cell ID", "Position Bins")


    else:
        raise ValueError("Options for which_to_plot are 'argmin' or 'argmax'")


def plot_example_neuron(reorganized_data, GLM_params, variable_list, sort_by='R2', animal=None, cell=None, ax=None):
    if animal is None or cell is None:
        animal, cell = select_neuron(GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell)
        print(f"Best neuron: {cell}, {animal}")

    neuron_data = reorganized_data[animal][cell]
    flattened_data = flatten_data(neuron_data)

    input_variables = neuron_data[:,1:,:]
    neuron_activity = neuron_data[:,0,:]

    if ax is None:
        fig = plt.figure(figsize=(14, 4))
        ax_ = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
        ax_ = ax
    ax_.axis('off')

    axes = gs.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=ax_, wspace=0., width_ratios=[0.3, 0.22, 0.48])
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    avg_variables = np.mean(input_variables, axis=2)
    def plot_example_neuron_variables(variables, variable_list, ax, fig, weights=None):
        variable_list = variable_list[1:]
        height_ratios = np.ones(variables.shape[1])
        height_ratios[-10:] = 0.5
        axes = gs.GridSpecFromSubplotSpec(nrows=variables.shape[1], ncols=1, subplot_spec=ax, hspace=0.5, height_ratios=height_ratios)
        for i in range(variables.shape[1]):
            ax = fig.add_subplot(axes[i])
            ax.plot(variables[:, i], color='k', linewidth=1)
            ax.set_ylabel(variable_list[i], rotation=0, ha='right', va='center', labelpad=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if weights is not None:
                ax.scatter([50],[0.3], c='k', s=abs(weights[i])*20)
        ax.set_xlabel('Position', labelpad=-10)
        ax.set_xticks([0,50])

        # Draw vertical line across all plots (remove axes and make the background transparent)
        ax = fig.add_subplot(axes[:])
        ax.vlines(25, 0, 1, linestyles='--', color='r', alpha=0.7)
        ax.set_xlim([0, 49])
        ax.set_ylim([0, 1])
        ax.axis('off')
        ax.patch.set_alpha(0)
    plot_example_neuron_variables(avg_variables, variable_list, ax, fig)

    ax = fig.add_subplot(axes[1])
    ax.axis('off')
    def plot_weight_lines(GLM_params, animal, cell, ax):
        weights = GLM_params[animal][cell]['weights']
        if len(weights) > 13:
            line_spacing = -np.ones(len(weights)) * 1.4
        else:
            line_spacing = -np.ones(len(weights)) * 1.45
        line_spacing[-10:] *= 0.615
        y = np.cumsum(line_spacing) - 0.4/line_spacing[0]
        w_max = np.max(np.abs(weights))
        for i,w in enumerate(weights):
            if abs(w)<0.05:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='lightgray', linestyle='--', linewidth=1)
            elif w < 0:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='deepskyblue', linewidth=abs(w/w_max)*4)
            else:
                line, = ax.plot([0,1], [y[i],-len(weights)/2], color='black', linewidth=abs(w/w_max)*4)
            line.set_solid_capstyle('round')
        ax.set_ylim([-len(weights),0])

        ax.plot([0,0], [0,0], color='lightgray', linewidth=1.5, linestyle='--', label='Small weights')
        ax.plot([0,0], [0,0], color='deepskyblue', linewidth=1.5, label='Negative weights')
        ax.plot([0,0], [0,0], color='black', linewidth=1.5, label='Positive weights')
        ax.legend(fontsize=10, loc='upper right', frameon=False, handlelength=1.5, handletextpad=0.5, labelspacing=0.2, borderpad=0)
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        ax.text(1, -13, f'Max weight: {max_weight:.2f}', ha='right', va='bottom', fontsize=10)
        ax.text(1, -12, f'Min weight: {min_weight:.2f}', ha='right', va='bottom', fontsize=10)
    plot_weight_lines(GLM_params, animal, cell, ax)

    # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][cell]['model']
    flattened_input_variables = flattened_data[:,1:]
    predicted_activity = glm_model.predict(flattened_input_variables)

    pearson_R = np.corrcoef(predicted_activity, flattened_data[:,0])[0,1]
    print("pearson R2 across all trials:", pearson_R**2)

    predicted_activity = predicted_activity.reshape(neuron_activity.shape)
    avg_predicted_activity = np.mean(predicted_activity, axis=1)
    std_predicted_activity = np.std(predicted_activity, axis=1)
    sem_predicted_activity = std_predicted_activity / np.sqrt(predicted_activity.shape[1])
    avg_neuron_activity = np.mean(neuron_activity, axis=1)
    std_neuron_activity = np.std(neuron_activity, axis=1)
    sem_neuron_activity = std_neuron_activity / np.sqrt(neuron_activity.shape[1])

    pearson_R = np.corrcoef(avg_predicted_activity, avg_neuron_activity)[0,1]
    print("pearson R2 (average prediction vs average activity):", pearson_R**2)

    axes = gs.GridSpecFromSubplotSpec(nrows=3, ncols=3, subplot_spec=ax_, wspace=0., width_ratios=[0.3, 0.3, 0.4], height_ratios=[0.2,1,0.2])
    ax = fig.add_subplot(axes[1,2])
    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-sem_neuron_activity, avg_neuron_activity+sem_neuron_activity, alpha=0.1, color='k')
    ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-sem_predicted_activity, avg_predicted_activity+sem_predicted_activity, alpha=0.1, color='gray')
    ax.set_xlabel("Position")
    ax.set_ylabel("dF/F activity (Z-scored)")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))


def plot_GLM_summary_data(GLM_params, variable_list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    jitter = 0.5
    animal_xoffset = 0.2

    params_all_animals = []
    for animal_id in GLM_params:
        params_all_neurons = []
        for neuron_id in GLM_params[animal_id]:
            weights = GLM_params[animal_id][neuron_id]['weights']
            intercept = GLM_params[animal_id][neuron_id]['intercept']
            all_params = np.concatenate([weights, [intercept]])
            params_all_neurons.append(all_params)
            jittered_x = np.arange(len(all_params)) + np.random.uniform(0.3, jitter, len(all_params))
            ax.scatter(jittered_x, all_params, color='grey', alpha=0.2, s=10)

        params_all_neurons = np.array(params_all_neurons)
        mean_params = np.mean(params_all_neurons, axis=0)
        params_all_animals.append(mean_params)
        ax.scatter(np.arange(len(mean_params))+animal_xoffset, mean_params, color='black', label=f'Animal {animal_id}', s=20)

    params_all_animals = np.array(params_all_animals)
    global_mean = np.mean(params_all_animals, axis=0)
    global_std = np.std(params_all_animals, axis=0)
    ax.errorbar(np.arange(len(global_mean)), global_mean, yerr=global_std, fmt='o', color='red', ecolor='red', capsize=5, label='Average of all animals', markersize=7)

    ax.set_xticks(np.arange(len(global_mean)))
    xtick_labels = variable_list[1:] + ['Intercept']
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel('Weights')

    ax.hlines(0, -0.5, len(variable_list) - 0.5, linestyles='--', color='black', alpha=0.5)
    ax.set_xlim([-0.5, len(global_mean) - 0.4])


def plot_quintile_comparison(reorganized_data, variable_list, filename, quintiles=(1,5), save=False):
    avg_residuals_ls = []
    GLM_params_ls = []
    for quintile in quintiles:
        avg_residuals, GLM_params = compute_velocity_subtracted_residuals(reorganized_data, variable_list, quintile)
        avg_residuals_ls.append(avg_residuals)
        GLM_params_ls.append(GLM_params)

    sorting_idx = np.argsort(np.argmax(avg_residuals_ls[1], axis=1))
    avg_residuals_ls = [avg_residuals[sorting_idx] for avg_residuals in avg_residuals_ls]

    fig = plt.figure(figsize=(14, 14))
    axes = gs.GridSpec(nrows=3, ncols=3, hspace=0.2, wspace=0.4, height_ratios=[3,1,2])
    fig.suptitle(filename, fontsize=15, y=0.93)

    vmax = np.max([np.abs(avg_residuals_ls[0]), np.abs(avg_residuals_ls[1])])
    plot_scale = np.max([np.abs(np.mean(avg_residuals_ls[0], axis=0))+np.std(avg_residuals_ls[0], axis=0), np.abs(np.mean(avg_residuals_ls[1], axis=0))+np.std(avg_residuals_ls[1], axis=0)])

    for col, quintile_data in enumerate(avg_residuals_ls):
        # Plot heatmap
        ax = fig.add_subplot(axes[0,col])
        im = ax.imshow(quintile_data, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
        x,y,w,h = ax.get_position().bounds
        cax = fig.add_axes([x+w*1.02, y, w*0.05, h])
        cbar = plt.colorbar(im, cax=cax)
        ax.set_xlim([0, 50])
        if quintiles[col] == 1:
            ax.set_title(f'First 20% of trials')
        elif quintiles[col] == 5:
            ax.set_title(f'Last 20% of trials')
        else:
            ax.set_title(f'Quintile {quintiles[col]}')
        if col == 0:
            ax.set_ylabel('Neuron #')

        # Plot average activity
        ax = fig.add_subplot(axes[1,col])
        ax.plot(np.mean(quintile_data, axis=0), c='k', lw=2)
        std = np.std(quintile_data, axis=0)
        sem = std / np.sqrt(quintile_data.shape[0])
        ax.fill_between(np.arange(50), np.mean(quintile_data, axis=0)-std, np.mean(quintile_data, axis=0)+std, color='gray', alpha=0.2)
        ax.hlines(0, 0, 50, color='gray', linestyle='--')
        ax.set_xlim([0, 50])
        if col == 0:
            ax.set_ylabel('dF/F residual activity \n(Z-scored)')
        ax.set_xlabel('Position')
        ax.set_ylim([-plot_scale, plot_scale])

    # Plot delta
    delta_residuals = avg_residuals_ls[1] - avg_residuals_ls[0]
    ax = fig.add_subplot(axes[0,2])
    vmax = np.max(np.abs(delta_residuals))
    im = ax.imshow(delta_residuals, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
    x,y,w,h = ax.get_position().bounds
    cax = fig.add_axes([x+w*1.02, y, w*0.05, h])
    cbar = fig.colorbar(im, cax=cax)
    ax.set_xlim([0, 50])
    ax.set_title('Difference')

    ax = fig.add_subplot(axes[1,2])
    ax.plot(np.mean(delta_residuals, axis=0), c='k', lw=2)
    std = np.std(delta_residuals, axis=0)
    sem = std / np.sqrt(delta_residuals.shape[0])
    ax.fill_between(np.arange(50), np.mean(delta_residuals, axis=0)-std, np.mean(delta_residuals, axis=0)+std, color='gray', alpha=0.2)
    ax.set_xlim([0, 50])
    ax.set_xlabel('Position')
    ax.hlines(0, 0, 50, color='gray', linestyle='--')

    ax = fig.add_subplot(axes[2,:])
    delta_weights = calculate_delta_weights(GLM_params_ls[0], GLM_params_ls[1])
    plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=ax)

    if save:
        fig.savefig(f'figures/{filename.split(".")[0]}_residuals.png', dpi=300)
        fig.savefig(f'figures/{filename.split(".")[0]}_residuals.svg', dpi=300)


def plot_R2_distribution(GLM_params, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(4,5))
    else:
        fig = ax.get_figure()

    if isinstance(GLM_params, list):
        model_list = GLM_params
    else:
        model_list = [GLM_params]

    all_R2_values = {}
    for i, model_params in enumerate(model_list, start=1):
        all_R2_values[i] = []
        animal_avg_R2_values = []
        for animal in model_params:
            animal_R2_values = []
            for neuron in model_params[animal]:
                all_R2_values[i].append(model_params[animal][neuron]['R2_trialavg'])
                animal_R2_values.append(model_params[animal][neuron]['R2_trialavg'])
            animal_avg_R2_values.append(np.mean(animal_R2_values))
        all_R2_values[i] = np.array(all_R2_values[i])

        jitter = 0.2
        jittered_x = i*np.ones(all_R2_values[i].shape) + np.random.uniform(0.1, jitter, all_R2_values[i].shape)
        ax.scatter(jittered_x, all_R2_values[i], color='grey', alpha=0.2, s=10)
        ax.scatter(i*np.ones(len(animal_avg_R2_values)), animal_avg_R2_values, color='black', label='Average R2 value', s=20)
        ax.errorbar(i*0.9, np.mean(animal_avg_R2_values), yerr=np.std(animal_avg_R2_values), fmt='o', color='red', ecolor='red',
                    capsize=5, label='Average of all animals', markersize=7)

    ax.set_ylabel("R² value")
    ax.set_xlim([0.8,2*i])
    ax.set_ylim([0,1])
    if len(model_list) > 1:
        # ax.set_xticks([0.8, i+0.2])
        # ax.set_xticklabels(['First quintile', 'Last quintile'])
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['All data', 'No licks', 'No Reward loc'])
        ax.set_xlim([0.8,3.2])
        if title is not None:
            fig.suptitle(title)
            fig.savefig(f"figures/R2_distribution_{title}.png", bbox_inches='tight')
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    # Statistical test
    if len(model_list) == 2:
        t, p = stats.ttest_ind(all_R2_values[1], all_R2_values[2])
        if p < 0.001:
            ax.text(0.1, 0.2, f"p = {p:.2e}", transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.1, 0.2, f"p = {p:.3f}", transform=ax.transAxes, fontsize=12)


def plot_combined_figure(reorganized_data, GLM_params, variable_list, sort_by='R2', animal=None, cell=None, model_name=None, save=False):
    animal, cell = select_neuron(GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell)

    fig = plt.figure(figsize=(10,6))
    axes = gs.GridSpec(nrows=1, ncols=1, top=1, bottom=0.5, left=0, right=1)
    ax = fig.add_subplot(axes[0])
    plot_example_neuron(reorganized_data, GLM_params, variable_list, sort_by=sort_by, animal=animal, cell=cell, ax=ax)

    # Plot R2 distribution
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.4, bottom=0, left=0., right=0.2)
    ax = fig.add_subplot(axes[0])
    plot_R2_distribution(GLM_params, ax=ax)

    # Plot summary data
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.4, bottom=0, left=0.3, right=1)
    ax = fig.add_subplot(axes[0])
    plot_GLM_summary_data(GLM_params, variable_list, ax=ax)

    if save and model_name is not None:
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{cell}.png", bbox_inches='tight', dpi=300)
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{cell}.svg", bbox_inches='tight', dpi=300)


def plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    plot_GLM_summary_data(delta_weights, variable_list, ax=ax)
    ax.set_ylabel('Δ Weights\n(Last - First Quintile)')

    if model_name is not None:
        ax.set_title(model_name)

    if model_name is not None and save:
        fig.savefig(f"figures/{model_name}_delta_weights.png", dpi=300)


def plot_activity_residuals_correlation(factors_dict, predicted_activity_list, neuron_activity_list, residuals_list, cell_number, variable_to_corelate="Velocity"):
    variable_data_list = []
    for animal in activity_dict:
        for neuron in activity_dict[animal]:
            variable_data_list.append(factors_dict[animal][variable_to_corelate])
            print(f"factors_dict[key][variable_to_corelate].shape {factors_dict[animal][variable_to_corelate].shape}")

    trial_average_variable_data_list = []
    for i in variable_data_list:
        trial_average_variable_data_list.append(np.mean(i, axis=1))

    trial_av_prediction_list = []
    for i in predicted_activity_list:
        trial_av_prediction_list.append(np.mean(i, axis=1))

    trial_av_neuron_activity_list = []
    for i in neuron_activity_list:
        trial_av_neuron_activity_list.append(np.mean(i, axis=1))

    trial_av_residuals_list = []
    for i in residuals_list:
        trial_av_residuals_list.append(np.mean(i, axis=1))

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].plot(trial_av_neuron_activity_list[cell_number], color='k', label='neuron activity')
    axs[0, 0].set_title(f"Firing Rate for Cell#{cell_number}", fontsize=6)
    axs[0, 0].set_ylim(-1.5, 1)

    axs[0, 1].plot(trial_average_variable_data_list[cell_number], color='g')
    axs[0, 1].set_title(f"Run Velocity for Animal, Cell#{cell_number}", fontsize=6)
    axs[0, 1].set_ylim(-1.5, 1)

    axs[1, 0].plot(trial_av_residuals_list[cell_number], color='r', label='residuals')
    axs[1, 0].set_title(f"Residuals: Firing Rate - Velocity-Only Prediction of FR for Cell#{cell_number}", fontsize=6)
    axs[1, 0].set_ylim(-1.5, 1)

    axs[1, 1].plot(trial_av_prediction_list[cell_number], color='b', label='velocity prediction')
    axs[1, 1].set_title(f"Prediction of Firing Based on Velocity for Cell#{cell_number}", fontsize=6)
    axs[1, 1].set_ylim(-1.5, 1)

    r2_list_residuals = []
    r2_list_activity = []
    y_pred_activity_list = []
    y_pred_residuals_list = []

    for i in range(len(variable_data_list)):
        velocity_flat = variable_data_list[i].flatten()
        activity_flat = neuron_activity_list[i].flatten()
        residuals_flat = residuals_list[i].flatten()

        model_activity = LinearRegression()
        model_activity.fit(velocity_flat.reshape(-1, 1), activity_flat)
        y_pred_activity = model_activity.predict(velocity_flat.reshape(-1, 1))
        r2_activity = r2_score(activity_flat, y_pred_activity)
        r2_list_activity.append(r2_activity)
        y_pred_activity_list.append(y_pred_activity)

        model_residuals = LinearRegression()
        model_residuals.fit(velocity_flat.reshape(-1, 1), residuals_flat)
        y_pred_residuals = model_residuals.predict(velocity_flat.reshape(-1, 1))
        r2_residuals = r2_score(residuals_flat, y_pred_residuals)
        r2_list_residuals.append(r2_residuals)
        y_pred_residuals_list.append(y_pred_residuals)

    velocity_flat = variable_data_list[cell_number].flatten()
    activity_flat = neuron_activity_list[cell_number].flatten()
    residuals_flat = residuals_list[cell_number].flatten()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].scatter(velocity_flat, activity_flat, color="g", s=10, alpha=0.3, label="Data")
    axs[0].plot(
        velocity_flat, y_pred_activity_list[cell_number], color="r", label=f"R² = {r2_list_activity[cell_number]:.3f}"
    )
    axs[0].set_title(f"Activity vs Velocity (Cell #{cell_number})")
    axs[0].set_xlabel("Velocity")
    axs[0].set_ylabel("Activity")
    axs[0].legend()

    axs[1].scatter(velocity_flat, residuals_flat, color="b", s=10, alpha=0.3, label="Data")
    axs[1].plot(
        velocity_flat, y_pred_residuals_list[cell_number], color="r", label=f"R² = {r2_list_residuals[cell_number]:.3f}"
    )
    axs[1].set_title(f"Residuals vs Velocity (Cell #{cell_number})")
    axs[1].set_xlabel("Velocity")
    axs[1].set_ylabel("Residuals")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return r2_list_residuals, r2_list_activity


def plot_r_difference_vs_variable(r2_list_activity, r2_list_residuals, variable_to_compare="Velocity"):
    r2_list_activity = np.array(r2_list_activity)
    r2_list_residuals = np.array(r2_list_residuals)

    mean_activity = np.mean(r2_list_activity)
    mean_residuals = np.mean(r2_list_residuals)
    sem_activity = sem(r2_list_activity)
    sem_residuals = sem(r2_list_residuals)

    positions = [0, 1]

    plt.figure(figsize=(8, 6))

    # Scatter points
    x_jitter_activity = np.random.normal(positions[0], 0.05, size=len(r2_list_activity))
    x_jitter_residuals = np.random.normal(positions[1], 0.05, size=len(r2_list_residuals))
    plt.scatter(x_jitter_activity, r2_list_activity, color='black', alpha=0.8, label='R² Activity', zorder=3)
    plt.scatter(x_jitter_residuals, r2_list_residuals, color='red', alpha=0.8, label='R² Residuals', zorder=3)

    # Connecting lines
    for i in range(len(r2_list_activity)):
        plt.plot([x_jitter_activity[i], x_jitter_residuals[i]],
                 [r2_list_activity[i], r2_list_residuals[i]],
                 color='gray', alpha=0.6, linewidth=0.8)

    # Mean and SEM lines for Activity
    plt.hlines(mean_activity, positions[0] - 0.2, positions[0] + 0.2, color='black', linewidth=2, zorder=4)
    plt.vlines(positions[0], mean_activity - sem_activity, mean_activity + sem_activity, color='black', linewidth=2)

    # Mean and SEM lines for Residuals
    plt.hlines(mean_residuals, positions[1] - 0.2, positions[1] + 0.2, color='red', linewidth=2, zorder=4)
    plt.vlines(positions[1], mean_residuals - sem_residuals, mean_residuals + sem_residuals, color='red', linewidth=2)

    # Formatting
    plt.xticks(positions, [f'Activity vs {variable_to_compare}', f'Residuals vs {variable_to_compare}'])
    plt.ylabel("R Value")
    plt.title(f"Activity vs {variable_to_compare} and Residuals vs {variable_to_compare}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_temporal_correlation(filepaths, output_to_plot='r', data_to_plot='all'):
    overall_r_per_dataset = {}
    per_animal_r_per_dataset = {}

    for label, filepath in filepaths.items():
        if data_to_plot == 'all' or data_to_plot.lower() == label.lower():
            reorganized_data, _ = preprocess_data(filepath)
            r_all_neurons, r_per_animal, _ = compute_temporal_correlation(reorganized_data)

            if output_to_plot.lower() == 'r2':
                r_all_neurons = np.array(r_all_neurons) ** 2
                r_per_animal = np.array(r_per_animal) ** 2

            overall_r_per_dataset[label] = r_all_neurons
            per_animal_r_per_dataset[label] = r_per_animal

    plot_results(overall_r_per_dataset, per_animal_r_per_dataset, output_to_plot)


def plot_activity_residuals_correlation(reorganized_data, predicted_activity_list, neuron_activity_list, residuals_list, cell_number, variable_to_corelate=["Velocity"]):
    velocity_list = []
    for key, value in reorganized_data.items():
        for key2, value2 in value.items():
            velocity = value2["Velocity"]
            velocity_list.append(velocity)

