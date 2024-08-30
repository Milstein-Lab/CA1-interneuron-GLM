import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

plt.rcParams.update({'font.size': 10,
                     'axes.spines.right': False,
                     'axes.spines.top': False,
                     'legend.frameon': False, })


# plt.rcParams.update({'font.size': 10,
#                     'axes.spines.right': False,
#                     'axes.spines.top':   False,
#                     'legend.frameon':       False,
#                     'font.sans-serif': 'Helvetica',
#                     'svg.fonttype': 'none'})


def load_data(filepath):
    data_dict = mat73.loadmat(filepath)

    # Define new position variables to use as input for the GLM
    num_spatial_bins = 5
    position_matrix = np.zeros((50, num_spatial_bins))
    bin_size = 50 // num_spatial_bins
    for i in range(num_spatial_bins):
        position_matrix[i * bin_size:(i + 1) * bin_size, i] = 1

    reorganized_data = {}
    variable_list = []
    num_animals = len(data_dict['animal']['ShiftLrate'])

    for animal_idx in range(num_animals):
        neuron_list = []

        num_neurons = data_dict['animal']['ShiftR'][animal_idx].shape[2]

        for neuron_idx in range(1, num_neurons):
            neuron_data = []

            # Neuron activity
            neuron_data.append(data_dict['animal']['ShiftR'][animal_idx][:, :, neuron_idx])
            variable_list.append('Activity') if neuron_idx == 1 and animal_idx == 0 else None

            # Lick rate
            neuron_data.append(data_dict['animal']['ShiftLrate'][animal_idx])
            variable_list.append('Licks') if neuron_idx == 1 and animal_idx == 0 else None

            # Reward location (valve opening)
            neuron_data.append(data_dict['animal']['ShiftV'][animal_idx])
            variable_list.append('R_loc') if neuron_idx == 1 and animal_idx == 0 else None

            # Running speed
            neuron_data.append(data_dict['animal']['ShiftRunning'][animal_idx])
            variable_list.append('Speed') if neuron_idx == 1 and animal_idx == 0 else None

            ####################################################
            # TODO: EC_GLM.mat has a data mismatch (missing trial in neuron activity). Check with Christine
            num_trials = np.min([neuron_data[i].shape[1] for i in range(len(neuron_data))])
            for i in range(len(neuron_data)):
                neuron_data[i] = neuron_data[i][:, :num_trials]
            ####################################################

            combined_matrix = np.stack(neuron_data, axis=1)

            # # Add variable for actual reward delivered (licks at reward location)
            # licking_on_reward = data_dict['animal']['ShiftLrate'][animal_idx] * data_dict['animal']['ShiftV'][animal_idx]
            # licking_on_reward_expanded = licking_on_reward[:, np.newaxis, :num_trials]
            # combined_matrix = np.concatenate((combined_matrix, licking_on_reward_expanded), axis=1)
            # variable_list.append('R+Lick') if neuron_idx == 1 and animal_idx == 0 else None

            # Add position variables to the data matrix
            expanded_position_matrix = np.repeat(position_matrix[:, :, np.newaxis], combined_matrix.shape[2],
                                                 axis=2)  # Copy along the 'trials' dimension
            expanded_position_matrix = np.roll(expanded_position_matrix, 5, axis=0)
            combined_matrix = np.concatenate((combined_matrix, expanded_position_matrix), axis=1)
            variable_list.extend(['#1', '#2', '#3', '#4', '#5']) if neuron_idx == 1 and animal_idx == 0 else None
#,'6', '#7', '#8', '#9', '#10'
            neuron_list.append(combined_matrix)

        reorganized_data[f'animal_{animal_idx + 1}'] = neuron_list

    return reorganized_data, variable_list


def fit_GLM(reorganized_data, quintile=None, regression='ridge', alphas=None):
    GLM_params = {}
    for animal in reorganized_data:
        GLM_params[animal] = {}
        for i, neuron_data in enumerate(reorganized_data[animal]):
            num_trials = neuron_data.shape[2]
            quintile_indices = [(i * num_trials) // 5 for i in range(6)]
            if quintile is not None:
                start_idx = quintile_indices[quintile - 1]
                end_idx = quintile_indices[quintile]
                neuron_data = neuron_data[:, :, start_idx:end_idx]

            neuron_data = neuron_data[:, :, ~np.isnan(neuron_data).any(axis=(0, 1))]
            flattened_data = []
            for var_idx in range(neuron_data.shape[1]):
                if var_idx == 0:  # Z-score the neuron activity (df/f)
                    neuron_data[:, var_idx] = (neuron_data[:, var_idx] - np.mean(neuron_data[:, var_idx])) / np.std(
                        neuron_data[:, var_idx])
                else:  # Normalize the other variables to [0,1]
                    neuron_data[:, var_idx] = (neuron_data[:, var_idx] - np.min(neuron_data[:, var_idx])) / (
                                np.max(neuron_data[:, var_idx]) - np.min(neuron_data[:, var_idx]))
                flattened_data.append(neuron_data[:, var_idx].flatten())
            flattened_data = np.stack(flattened_data, axis=1)
            design_matrix_X = flattened_data[:, 1:]
            neuron_activity = flattened_data[:, 0]

            # flattened_data = []
            # for var_idx in range(neuron_data.shape[1]):
            #     if var_idx == 0: # Z-score the neuron activity (df/f)
            #         neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmean(neuron_data[:,var_idx])) / np.nanstd(neuron_data[:,var_idx])
            #     else: # Normalize the other variables to [0,1]
            #         neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmin(neuron_data[:,var_idx])) / (np.nanmax(neuron_data[:,var_idx]) - np.nanmin(neuron_data[:,var_idx]))
            #     flattened_data.append(neuron_data[:,var_idx].flatten())
            # flattened_data = np.stack(flattened_data, axis=1)
            # flattened_data = flattened_data[~np.isnan(flattened_data).any(axis=1)]
            # design_matrix_X = flattened_data[:, 1:]
            # neuron_activity = flattened_data[:, 0]

            if regression == 'lasso':
                model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
            elif regression == 'ridge':
                model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
            elif regression == 'elastic':
                l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
                model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000],
                                     l1_ratio=l1_ratio, cv=None)
            else:
                raise ValueError("Regression type must be 'lasso' or 'ridge'")

            model.fit(design_matrix_X, neuron_activity)

            predicted_activity = model.predict(design_matrix_X)
            trialavg_neuron_activity = np.mean(neuron_data[:, 0, :], axis=1)
            trialavg_predicted_activity = np.mean(predicted_activity.reshape(neuron_data[:, 0, :].shape), axis=1)
            pearson_R = np.corrcoef(trialavg_predicted_activity, trialavg_neuron_activity)[0, 1]

            GLM_params[animal][i] = {
                'weights': model.coef_,
                'intercept': model.intercept_,
                'alpha': model.alpha_ if regression == 'ridge' else model.alpha_,
                'l1_ratio': model.l1_ratio_ if regression == 'elastic' else None,
                'R2': model.score(design_matrix_X, neuron_activity),
                'R2_trialavg': pearson_R ** 2,
                'model': model
            }

    return GLM_params


def fit_GLM_quintile_comparison(reorganized_data, regression='ridge', alphas=None):
    GLM_params_comparison = {}

    for animal in reorganized_data:
        GLM_params_comparison[animal] = {'R2_first': [], 'R2_last': []}

        for i, neuron_data in enumerate(reorganized_data[animal]):
            num_trials = neuron_data.shape[2]
            quintile_indices = [(i * num_trials) // 5 for i in range(6)]

            # Extract the first quintile
            start_idx_first = quintile_indices[0]
            end_idx_first = quintile_indices[1]
            neuron_data_first = neuron_data[:, :, start_idx_first:end_idx_first]
            neuron_data_first = neuron_data_first[:, :, ~np.isnan(neuron_data_first).any(axis=(0, 1))]

            # Extract the last quintile
            start_idx_last = quintile_indices[-2]
            end_idx_last = quintile_indices[-1]
            neuron_data_last = neuron_data[:, :, start_idx_last:end_idx_last]
            neuron_data_last = neuron_data_last[:, :, ~np.isnan(neuron_data_last).any(axis=(0, 1))]

            # Normalize and flatten data for GLM
            def normalize_and_flatten(neuron_data):
                flattened_data = []
                for var_idx in range(neuron_data.shape[1]):
                    for var_idx in range(neuron_data.shape[1]):
                        if var_idx == 0: # Z-score the neuron activity (df/f)
                                neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmean(neuron_data[:,var_idx])) / np.nanstd(neuron_data[:,var_idx])
                        else: # Normalize the other variables to [0,1]
                            neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmin(neuron_data[:,var_idx])) / (np.nanmax(neuron_data[:,var_idx]) - np.nanmin(neuron_data[:,var_idx]))
                    # neuron_data[:, var_idx] = (neuron_data[:, var_idx] - np.min(neuron_data[:, var_idx])) / (
                    #         np.max(neuron_data[:, var_idx]) - np.min(neuron_data[:, var_idx]))
                        flattened_data.append(neuron_data[:, var_idx].flatten())
                flattened_data = np.stack(flattened_data, axis=1)
                return flattened_data[:, 1:], flattened_data[:, 0]

            # First quintile GLM fitting
            design_matrix_X_first, neuron_activity_first = normalize_and_flatten(neuron_data_first)
            if regression == 'lasso':
                model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
            elif regression == 'ridge':
                model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
            else:
                raise ValueError("Regression type must be 'lasso' or 'ridge'")
            model.fit(design_matrix_X_first, neuron_activity_first)
            R2_first = model.score(design_matrix_X_first, neuron_activity_first)

            # Last quintile GLM fitting
            design_matrix_X_last, neuron_activity_last = normalize_and_flatten(neuron_data_last)
            model.fit(design_matrix_X_last, neuron_activity_last)
            R2_last = model.score(design_matrix_X_last, neuron_activity_last)

            # Store the R² values for first and last quintiles
            GLM_params_comparison[animal]['R2_first'].append(R2_first)
            GLM_params_comparison[animal]['R2_last'].append(R2_last)

    return GLM_params_comparison


# def compare_neural_activity(reorganized_data):
#     activity_comparison = {}
#
#     for animal in reorganized_data:
#         activity_comparison[animal] = {'delta_activity': []}
#
#
#         for neuron_data in reorganized_data[animal]:
#             num_trials = neuron_data.shape[2]
#             quintile_indices = [(i * num_trials) // 5 for i in range(6)]
#
#             # Extract the first quintile
#             start_idx_first = quintile_indices[0]
#             end_idx_first = quintile_indices[1]
#             neuron_data_first = neuron_data[:, :, start_idx_first:end_idx_first]
#             neuron_data_first = neuron_data_first[:, :, ~np.isnan(neuron_data_first).any(axis=(0, 1))]
#
#             # Extract the last quintile
#             start_idx_last = quintile_indices[-2]
#             end_idx_last = quintile_indices[-1]
#             neuron_data_last = neuron_data[:, :, start_idx_last:end_idx_last]
#             neuron_data_last = neuron_data_last[:, :, ~np.isnan(neuron_data_last).any(axis=(0, 1))]
#
#             # Calculate the mean activity for the first and last quintiles
#             mean_activity_first = np.mean(neuron_data_first, axis=(1, 2))
#             mean_activity_last = np.mean(neuron_data_last, axis=(1, 2))
#
#             # Calculate the delta activity (mean of last quintile - mean of first quintile)
#             delta_activity = mean_activity_last - mean_activity_first
#
#             # Append the calculated delta activity to the list for this animal
#             activity_comparison[animal]['delta_activity'].append(delta_activity)
#
#     mean_delta_activity_per_neuron = np.mean(reorganized_data[animal]['delta_activity'])
#
#     return activity_comparison

def compare_neural_activity(reorganized_data):
    activity_comparison = {}

    for animal in reorganized_data:
        activity_comparison[animal] = {'delta_activity': []}

        for neuron_data in reorganized_data[animal]:
            num_trials = neuron_data.shape[2]
            quintile_indices = [(i * num_trials) // 5 for i in range(6)]

            start_idx_first = quintile_indices[0]
            end_idx_first = quintile_indices[1]
            neuron_data_first = neuron_data[:, :, start_idx_first:end_idx_first]
            neuron_data_first = neuron_data_first[:, :, ~np.isnan(neuron_data_first).any(axis=(0, 1))]

            start_idx_last = quintile_indices[-2]
            end_idx_last = quintile_indices[-1]
            neuron_data_last = neuron_data[:, :, start_idx_last:end_idx_last]
            neuron_data_last = neuron_data_last[:, :, ~np.isnan(neuron_data_last).any(axis=(0, 1))]

            # Calculate the mean activity for the first and last quintiles across all trials
            mean_activity_first = np.mean(neuron_data_first, axis=(0, 2))
            mean_activity_last = np.mean(neuron_data_last, axis=(0, 2))

            # Calculate the delta activity (mean of last quintile - mean of first quintile)
            delta_activity = mean_activity_last - mean_activity_first

            # Append the calculated delta activity to the list for this animal
            activity_comparison[animal]['delta_activity'].append(delta_activity)

    return activity_comparison


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_neural_activity_delta(activity_comparison, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    animals = list(activity_comparison.keys())

    # Plot delta activity for each animal
    for i, animal in enumerate(animals):
        delta_activity = np.array(activity_comparison[animal]['delta_activity'])

        # Print the mean and SEM for debugging
        print(f"Animal: {animal}")
        print(f"Delta Activity: {delta_activity}")

        # Calculate mean and standard error for the animal
        mean_delta = np.mean(delta_activity)
        sem_delta = np.std(delta_activity) / np.sqrt(len(delta_activity))  # Standard Error of the Mean (SEM)

        # Print the mean and SEM for debugging
        print(f"Mean Delta: {mean_delta}, SEM: {sem_delta}")

        # Jittered scatter plot for individual neuron delta activities
        jittered_x = np.ones(len(delta_activity)) * i + np.random.uniform(-0.1, 0.1, len(delta_activity))
        ax.scatter(jittered_x, delta_activity, color='blue', alpha=0.5, s=20)

        # Plot the mean dot and error bars
        ax.errorbar(i, mean_delta, yerr=sem_delta, fmt='o', color='red', capsize=5, markersize=8)

    # General plot formatting
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Δ Neural Activity (Mean Last Quintile - Mean First Quintile)")
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha='right')
    ax.set_xlim([-0.5, len(animals) - 0.5])
    ax.set_title("Delta Neural Activity Across Animals")
    plt.show()


# def compare_neural_activity(reorganized_data):
#     activity_comparison = {}
#
#     for animal in reorganized_data:
#         activity_comparison[animal] = {'delta_activity': []}
#
#         for neuron_data in reorganized_data[animal]:
#             num_trials = neuron_data.shape[2]
#             quintile_indices = [(i * num_trials) // 5 for i in range(6)]
#
#             # Extract the first quintile
#             start_idx_first = quintile_indices[0]
#             end_idx_first = quintile_indices[1]
#             neuron_data_first = neuron_data[:, :, start_idx_first:end_idx_first]
#             neuron_data_first = neuron_data_first[:, :, ~np.isnan(neuron_data_first).any(axis=(0, 1))]
#
#             # Extract the last quintile
#             start_idx_last = quintile_indices[-2]
#             end_idx_last = quintile_indices[-1]
#             neuron_data_last = neuron_data[:, :, start_idx_last:end_idx_last]
#             neuron_data_last = neuron_data_last[:, :, ~np.isnan(neuron_data_last).any(axis=(0, 1))]
#
#             # Align trial counts by taking the minimum number of trials available
#             min_trials = min(neuron_data_first.shape[2], neuron_data_last.shape[2])
#             neuron_data_first = neuron_data_first[:, :, :min_trials]
#             neuron_data_last = neuron_data_last[:, :, :min_trials]
#
#             # Z-score the neural activity across all trials
#             activity_first_z = (neuron_data_first - np.mean(neuron_data_first, axis=1, keepdims=True)) / np.std(
#                 neuron_data_first, axis=1, keepdims=True)
#             activity_last_z = (neuron_data_last - np.mean(neuron_data_last, axis=1, keepdims=True)) / np.std(
#                 neuron_data_last, axis=1, keepdims=True)
#
#             # Calculate the difference in average activity between the first and last quintiles
#             delta_activity = np.mean(activity_last_z, axis=1) - np.mean(activity_first_z, axis=1)
#
#             # Append the calculated delta activity to the list for this animal
#             activity_comparison[animal]['delta_activity'].append(delta_activity)
#
#     return activity_comparison


def plot_R2_delta(GLM_params_comparison, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    delta_R2 = {}

    # Calculate delta R² for each animal
    for animal in GLM_params_comparison:
        R2_first = np.array(GLM_params_comparison[animal]['R2_first'])
        R2_last = np.array(GLM_params_comparison[animal]['R2_last'])
        delta_R2[animal] = R2_last - R2_first

    # Plot delta R² values for each animal
    animals = list(delta_R2.keys())
    for i, animal in enumerate(animals):
        jittered_x = np.ones(len(delta_R2[animal])) * i + np.random.uniform(-0.1, 0.1, len(delta_R2[animal]))
        ax.scatter(jittered_x, delta_R2[animal], color='blue', alpha=0.5, s=20)

    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("ΔR² (Last - First)")
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha='right')
    ax.set_xlim([-0.5, len(animals) - 0.5])
    ax.set_title("Delta R² Values Across Animals")
    plt.show()


def plot_neural_activity_delta(activity_comparison, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    animals = list(activity_comparison.keys())

    # Plot delta activity for each animal
    for i, animal in enumerate(animals):
        delta_activity = np.concatenate(activity_comparison[animal]['delta_activity'])
        jittered_x = np.ones(len(delta_activity)) * i + np.random.uniform(-0.1, 0.1, len(delta_activity))
        ax.scatter(jittered_x, delta_activity, color='blue', alpha=0.5, s=20)

    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Δ Neural Activity (Z-scored)")
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha='right')
    ax.set_xlim([-0.5, len(animals) - 0.5])
    ax.set_title("Delta Neural Activity Across Animals")
    plt.show()


def plot_example_neuron_variables(example_variables, variable_list, weights, ax):
    variable_list = variable_list[1:]

    fig = ax.get_figure()
    height_ratios = np.ones(example_variables.shape[1])
    height_ratios[-5:] = 0.5
    axes = gs.GridSpecFromSubplotSpec(nrows=example_variables.shape[1], ncols=1, subplot_spec=ax, hspace=0.5,
                                      height_ratios=height_ratios)
    for i in range(example_variables.shape[1]):
        ax = fig.add_subplot(axes[i])
        ax.plot(example_variables[:, i])
        ax.set_ylabel(variable_list[i], rotation=0, ha='right', va='center')
        ax.set_xticks([])
        # ax.scatter([50],[0.3], c='k', s=abs(weights[i])*20)
    ax.set_xlabel('Position', labelpad=-10)
    ax.set_xticks([0, 50])

    # Draw vertical line across all plots (remove axes and make the background transparent)
    ax = fig.add_subplot(axes[:])
    ax.vlines(24.5, 0, 1, linestyles='--', color='r')
    ax.set_xlim([0, 49])
    ax.set_ylim([0, 1])
    ax.axis('off')
    ax.patch.set_alpha(0)


def plot_example_neuron(animal, reorganized_data, GLM_params, variable_list, neuron='best', model_name=None):
    # Pick neuron with the highest R2 value
    if neuron == 'best':
        R2_values = [GLM_params[animal][i]['R2'] for i in GLM_params[animal]]
        neuron = np.argmax(R2_values)
    print("Best neuron:", neuron)
    print("R2:", GLM_params[animal][neuron]['R2'])
    print("alpha:", GLM_params[animal][neuron]['alpha'])
    weights = GLM_params[animal][neuron]['weights']

    neuron_data = reorganized_data[animal][neuron][:, :, 1:]
    neuron_data = neuron_data[:, :, ~np.isnan(neuron_data).any(axis=(0, 1))]

    flattened_data = []
    for i in range(neuron_data.shape[1]):
        if i == 0:  # Z-score the neuron activity (df/f)
            neuron_data[:, i] = (neuron_data[:, i] - np.mean(neuron_data[:, i])) / np.std(neuron_data[:, i])
        else:  # Normalize the other variables to [0,1]
            neuron_data[:, i] = (neuron_data[:, i] - np.min(neuron_data[:, i])) / (
                        np.max(neuron_data[:, i]) - np.min(neuron_data[:, i]))
        flattened_data.append(neuron_data[:, i].flatten())
    flattened_data = np.stack(flattened_data, axis=1)

    input_variables = neuron_data[:, 1:, :]
    neuron_activity = neuron_data[:, 0, :]

    fig = plt.figure(figsize=(10, 8))

    # Plot input variables
    axes = gs.GridSpec(nrows=1, ncols=1, left=0, right=0.3, bottom=0.55)
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    avg_variables = np.mean(input_variables, axis=2)
    plot_example_neuron_variables(avg_variables, variable_list, weights, ax=ax)

    # Plot weights as lines across the figure
    axes = gs.GridSpec(nrows=1, ncols=1, left=0.31, right=0.53, bottom=0.55)
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    y1 = np.linspace(-0.8, -2.8, 3).tolist()
    y2 = np.linspace(-3.45, -6, 5).tolist()
    y = y1 + y2
    w_max = np.max(np.abs(weights))
    for i, w in enumerate(weights):
        if w == 0:
            line, = ax.plot([0, 1], [y[i], -3.4], color='lightgray', linestyle='--', linewidth=1)
        elif w < 0:
            line, = ax.plot([0, 1], [y[i], -3.4], color='deepskyblue', linewidth=abs(w / w_max) * 4)
        else:
            line, = ax.plot([0, 1], [y[i], -3.4], color='black', linewidth=abs(w / w_max) * 4)
        line.set_solid_capstyle('round')
    ax.set_ylim([-6, 0])

    # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][neuron]['model']
    flattened_input_variables = flattened_data[:, 1:]
    predicted_activity = glm_model.predict(flattened_input_variables)

    pearson_R = np.corrcoef(predicted_activity, flattened_data[:, 0])[0, 1]
    print("pearson R2 overall:", pearson_R ** 2)

    predicted_activity = predicted_activity.reshape(neuron_activity.shape)
    avg_predicted_activity = np.mean(predicted_activity, axis=1)
    std_predicted_activity = np.std(predicted_activity, axis=1)
    avg_neuron_activity = np.mean(neuron_activity, axis=1)
    std_neuron_activity = np.std(neuron_activity, axis=1)

    pearson_R = np.corrcoef(avg_predicted_activity, avg_neuron_activity)[0, 1]
    print("pearson R2 average:", pearson_R ** 2)

    axes = gs.GridSpec(nrows=1, ncols=1, left=0.61, right=1, top=0.8, bottom=0.55)
    ax = fig.add_subplot(axes[0])
    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    # ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-std_predicted_activity, avg_predicted_activity+std_predicted_activity, alpha=0.1, color='gray')
    # ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-std_neuron_activity, avg_neuron_activity+std_neuron_activity, alpha=0.1, color='k')
    ax.set_xlabel("Position")
    ax.set_xticks([0, 50])
    ax.set_ylabel("dF/F activity (Z-scored)")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))

    # Plot summary data
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.45, bottom=0.2, left=0.3, right=1)
    ax = fig.add_subplot(axes[0])
    plot_GLM_summary_data(GLM_params, variable_list, model_name="tests", save=False, ax=ax)

    axes = gs.GridSpec(nrows=1, ncols=1, top=0.45, bottom=0.2, left=0., right=0.2)
    ax = fig.add_subplot(axes[0])
    plot_R2_distribution(GLM_params, ax=ax)

    if model_name is not None:
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{neuron}.png", bbox_inches='tight', dpi=300)


def plot_GLM_summary_data(GLM_params, variable_list, model_name, save=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    animal_averages = []
    animal_stds = []
    jitter = 0.25

    for animal_key in GLM_params:
        neuron_weights = []
        for neuron_nr in range(len(GLM_params[animal_key])):
            neuron_weights.append(GLM_params[animal_key][neuron_nr]['weights'])
            jittered_x = np.arange(len(variable_list[1:])) + np.random.uniform(0.1, jitter, len(variable_list[1:]))
            ax.scatter(jittered_x, GLM_params[animal_key][neuron_nr]['weights'], color='grey', alpha=0.2, s=10)

        neuron_weights = np.array(neuron_weights)
        mean_weights = np.mean(neuron_weights, axis=0)
        std_weights = np.std(neuron_weights, axis=0)
        animal_averages.append(mean_weights)
        animal_stds.append(std_weights)
        ax.scatter(range(len(variable_list[1:])), mean_weights, color='black', label=f'Animal {animal_key}', s=20)

    animal_averages = np.array(animal_averages)
    animal_stds = np.array(animal_stds)

    global_mean = np.mean(animal_averages, axis=0)
    global_std = np.std(animal_averages, axis=0)
    ax.errorbar(np.arange(len(variable_list[1:])) - 0.15, global_mean, yerr=global_std, fmt='o', color='red',
                ecolor='red',
                capsize=5, label='Average of all animals', markersize=7)

    ax.set_xticks(range(len(variable_list[1:])), variable_list[1:], rotation=45, ha='right')
    ax.set_ylabel('Weights')
    ax.hlines(0, 0, len(variable_list[1:]), linestyles='--', color='black', alpha=0.5)
    ax.set_xlim([-0.5, len(variable_list[1:]) - 0.5])


def plot_R2_distribution(GLM_params, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    all_R2_values = []
    animal_avg_R2_values = []
    for animal in GLM_params:
        animal_R2_values = []
        for neuron in GLM_params[animal]:
            all_R2_values.append(GLM_params[animal][neuron]['R2_trialavg'])
            animal_R2_values.append(GLM_params[animal][neuron]['R2_trialavg'])
        animal_avg_R2_values.append(np.mean(animal_R2_values))
    all_R2_values = np.array(all_R2_values)

    jitter = 0.2
    jittered_x = np.ones(all_R2_values.shape) + np.random.uniform(0.1, jitter, all_R2_values.shape)
    ax.scatter(jittered_x, all_R2_values, color='grey', alpha=0.2, s=10)
    ax.scatter(np.ones(len(animal_avg_R2_values)), animal_avg_R2_values, color='black', label='Average R2 value', s=20)
    ax.errorbar(0.9, np.mean(animal_avg_R2_values), yerr=np.std(animal_avg_R2_values), fmt='o', color='red',
                ecolor='red',
                capsize=5, label='Average of all animals', markersize=7)
    ax.set_ylabel("R² value")
    ax.set_xlim([0.8, 2])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)


if __name__ == "__main__":

    datasets = ["SSTindivsomata_GLM.mat"]  # , "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    for dataset_path in datasets:
        reorganized_data, variable_list = load_data(dataset_path)
        GLM_params = fit_GLM(reorganized_data, quintile=1, regression=lasso)
        GLM_params_comparison = fit_GLM_quintile_comparison(reorganized_data, regression='ridge')

        fig = plt.figure(figsize=(10, 5))
        axes = gs.GridSpec(nrows=1, ncols=3)
        fig.suptitle(dataset_path)

        ax = fig.add_subplot(axes[0, 0])
        plot_example_neuron(reorganized_data, variable_list, model_name=dataset_path, fig=fig, ax=ax)

        ax = fig.add_subplot(axes[0, 2])
        plot_GLM_summary_data(GLM_params, variable_list, dataset_path, ax=ax)

        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(f'Change in R² Values - {dataset_path}')
        ax = fig.add_subplot(1, 1, 1)
        plot_R2_delta(GLM_params_comparison, ax=ax)

        activity_comparison = compare_neural_activity(reorganized_data)
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle(f'Delta Neural Activity Across Animals - {dataset_path}')
        ax = fig.add_subplot(1, 1, 1)
        plot_neural_activity_delta(activity_comparison, ax=ax)

        plt.show()

        fig.savefig(f"figures/{dataset_path[:-4]}.png", dpi=300)