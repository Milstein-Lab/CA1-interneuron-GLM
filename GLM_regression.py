import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
import scipy.stats as stats
import pandas as pd
import copy

plt.rcParams.update({'font.size': 10,
                    'axes.spines.right': False,
                    'axes.spines.top':   False,
                    'legend.frameon':       False,})


# plt.rcParams.update({'font.size': 10,
#                     'axes.spines.right': False,
#                     'axes.spines.top':   False,
#                     'legend.frameon':       False,
#                     'font.sans-serif': 'Helvetica',
#                     'svg.fonttype': 'none'})



def preprocess_data(filepath, normalize=True):
    data_dict = mat73.loadmat(filepath)

    # Define new position variables to use as input for the GLM
    num_spatial_bins = 10
    position_matrix = np.zeros((50, num_spatial_bins))
    bin_size = 50//num_spatial_bins
    for i in range(num_spatial_bins):
        position_matrix[i*bin_size:(i+1)*bin_size,i] = 1

    reorganized_data = {}
    variable_list = []

    for animal_idx, (delta_f,velocity,lick_rate,reward_loc) in enumerate(zip(data_dict['animal']['ShiftR'], data_dict['animal']['ShiftRunning'], data_dict['animal']['ShiftLrate'], data_dict['animal']['ShiftV'])):
        reorganized_data[f'animal_{animal_idx + 1}'] = {}

        for i, neuron_idx in enumerate(range(delta_f.shape[2])):
            if np.all(np.isnan(delta_f[:,:,neuron_idx])) or np.all(delta_f[:,:,neuron_idx] == 0):
                continue

            neuron_data = []
            
            # Neuron activity
            neuron_data.append(delta_f[:,:,neuron_idx]) 
            variable_list.append('Activity') if 'Activity' not in variable_list else None
            
            # Lick rate
            neuron_data.append(lick_rate)  
            variable_list.append('Licks') if 'Licks' not in variable_list else None

            # Reward location (valve opening)
            neuron_data.append(reward_loc)
            variable_list.append('Reward') if 'Reward' not in variable_list else None

            # Running speed
            neuron_data.append(velocity)
            variable_list.append('Velocity') if 'Velocity' not in variable_list else None

            ####################################################
            # TODO: EC_GLM.mat has a data mismatch (missing trial in neuron activity). Check with Christine
            num_trials = np.min([neuron_data[i].shape[1] for i in range(len(neuron_data))])
            for i in range(len(neuron_data)):
                neuron_data[i] = neuron_data[i][:, :num_trials]
            ####################################################

            neuron_data = np.stack(neuron_data, axis=1)

            # Add position variables to the data matrix
            expanded_position_matrix = np.repeat(position_matrix[:, :, np.newaxis], neuron_data.shape[2], axis=2) # Copy along the 'trials' dimension
            neuron_data = np.concatenate((neuron_data, expanded_position_matrix), axis=1)
            variable_list.extend([f'#{i}' for i in range(1, num_spatial_bins+1)]) if f'#{i}' not in variable_list else None

            # Filter out NaN trials
            neuron_data = neuron_data[:, :, ~np.isnan(neuron_data).any(axis=(0, 1))]

            if normalize:
                neuron_data = normalize_data(neuron_data)

            reorganized_data[f'animal_{animal_idx + 1}'][f'cell_{neuron_idx + 1}'] = neuron_data

    return reorganized_data, variable_list


def normalize_data(neuron_data):
    for var_idx in range(neuron_data.shape[1]):
        if var_idx == 0: # Z-score the neuron activity (df/f)
            neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.mean(neuron_data[:,var_idx])) / np.std(neuron_data[:,var_idx])
        else: # Normalize the other variables to [0,1]
            neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.min(neuron_data[:,var_idx])) / (np.max(neuron_data[:,var_idx]) - np.min(neuron_data[:,var_idx]))
    return neuron_data


def flatten_data(neuron_data):
    flattened_data = []
    for var_idx in range(neuron_data.shape[1]):
        flattened_data.append(neuron_data[:,var_idx].flatten())
    flattened_data = np.stack(flattened_data, axis=1)
    return flattened_data


def get_quintile_indices(num_trials, quintile=None):
    quintile_indices = [(i * num_trials) // 5 for i in range(6)]
    start_idx = quintile_indices[quintile - 1]
    end_idx = quintile_indices[quintile]
    return start_idx,end_idx


def filter_neurons_by_metric(reorganized_data, GLM_params, variable_list, metric, threshold, scale_type='value', keep='top'):
    GLM_r2 = get_GLM_R2(GLM_params)
    weights = get_GLM_weights(GLM_params, variable_list)
    consequtive_trials_correlations, _,_ = compute_temporal_correlation(reorganized_data)

    metrics_dict = {**weights, 'R2': GLM_r2, 'trial correlations': consequtive_trials_correlations}
    if metric not in metrics_dict:
        raise ValueError(f"Invalid metric: {metric}. Choose from {list(metrics_dict.keys())}")
    metric_values = metrics_dict[metric]
    
    filtered_GLM = copy.deepcopy(GLM_params)
    filtered_data = copy.deepcopy(reorganized_data)
    neuron_idx = 0
    for animal in filtered_GLM:
        neurons_to_remove = []
        for neuron in filtered_GLM[animal]:
            metric_value = metric_values[neuron_idx]

            # Define thresholds based on scale type
            if scale_type == 'std':
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                upper_bound = mean_val + threshold * std_val
                lower_bound = mean_val - threshold * std_val

            elif scale_type == 'percentile':
                upper_bound = np.percentile(metric_values, 100 - threshold)
                lower_bound = np.percentile(metric_values, threshold)

            elif scale_type == 'value':
                upper_bound = threshold
                lower_bound = threshold

            # Apply filtering based on filter type
            if (keep == 'top' and metric_value < upper_bound) or \
               (keep == 'bottom' and metric_value > lower_bound):
                neurons_to_remove.append(neuron)

            neuron_idx += 1

        for neuron in neurons_to_remove:
            filtered_GLM[animal].pop(neuron)
            filtered_data[animal].pop(neuron)
        
    filtered_GLM = {animal: filtered_GLM[animal] for animal in filtered_GLM if filtered_GLM[animal] != {}}
    filtered_data = {animal: filtered_data[animal] for animal in filtered_data if filtered_data[animal] != {}}

    return filtered_data, filtered_GLM


def compute_temporal_correlation(reorganized_data):
    r_all_neurons = []
    r_per_animal = []
    r_dict = {}

    def compute_r_consecutive_pairs(neuron_activity):
        r_values = []
        for col in range(neuron_activity.shape[1] - 1):
            X = neuron_activity[:, col]
            Y = neuron_activity[:, col + 1]
            if np.std(X) > 0 and np.std(Y) > 0:
                r = np.corrcoef(X, Y)[0, 1]
            else:
                r = np.nan  # Handle cases with no variability
            r_values.append(r)
        return r_values

    for animal_key in reorganized_data:
        r_animal_neurons = []
        r_dict[animal_key] = {}

        for neuron_key in reorganized_data[animal_key]:
            neuron_data = reorganized_data[animal_key][neuron_key]
            neuron_activity = neuron_data[:, 0, :]

            # Remove trials with NaNs
            valid_trials_mask = ~np.isnan(neuron_activity).any(axis=0)
            neuron_activity = neuron_activity[:, valid_trials_mask]

            # Remove bins with NaNs
            valid_bins_mask = ~np.isnan(neuron_activity).any(axis=1)
            neuron_activity = neuron_activity[valid_bins_mask, :]

            if neuron_activity.shape[1] > 1:
                r_values = compute_r_consecutive_pairs(neuron_activity)
                mean_r_value = np.nanmean(r_values)
                r_animal_neurons.append(mean_r_value)
                r_all_neurons.append(mean_r_value)
                r_dict[animal_key][neuron_key] = mean_r_value

        r_per_animal.append(np.nanmean(r_animal_neurons))

    return r_all_neurons, r_per_animal, r_dict


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


def plot_results(overall_r_per_dataset, per_animal_r_per_dataset, output_to_plot):
    dataset_labels = list(overall_r_per_dataset.keys())
    plt.figure(figsize=(10, 5))
    x_positions = np.arange(len(dataset_labels))
    jitter_strength = 0.05

    for i, label in enumerate(dataset_labels):
        all_neurons_r = overall_r_per_dataset[label]
        per_animal_r = per_animal_r_per_dataset[label]

        jitter_neuron = np.random.normal(0, jitter_strength, size=len(all_neurons_r))
        jitter_animal = np.random.normal(0, jitter_strength, size=len(per_animal_r))

        plt.scatter(np.full(len(all_neurons_r), i) + jitter_neuron, all_neurons_r, color='grey', alpha=0.5,
                    label='Neurons' if i == 0 else "")

        plt.scatter(np.full(len(per_animal_r), i) + jitter_animal, per_animal_r, color='k',
                    label='Per Animal' if i == 0 else "")

        mean_r = np.mean(all_neurons_r)
        error = sem(all_neurons_r)
        plt.errorbar(i, mean_r, yerr=error, fmt='o', color='r', label='Overall' if i == 0 else "")

    plt.xticks(x_positions, dataset_labels)
    plt.ylabel(f'{output_to_plot.upper()} value')
    plt.title(f'Average {output_to_plot.upper()} for NDNF, EC, and SST Neurons with SEM')
    plt.legend()
    plt.grid(True)
    plt.show()


def fit_GLM(reorganized_data, quintile=None, regression='ridge', renormalize=True, alphas=None):
    GLM_params = {}
    for animal in reorganized_data:
        GLM_params[animal] = {}
        for neuron, neuron_data in reorganized_data[animal].items():
            neuron_data = neuron_data[:,:,~np.isnan(neuron_data).any(axis=(0,1))]

            if quintile is not None:
                num_trials = neuron_data.shape[2]
                start_idx,end_idx = get_quintile_indices(num_trials, quintile)
                neuron_data = neuron_data[:, :, start_idx:end_idx]

            if renormalize:
                neuron_data = normalize_data(neuron_data)

            flattened_data = flatten_data(neuron_data)
            design_matrix_X = flattened_data[:,1:]
            neuron_activity = flattened_data[:,0]

            if regression == 'lasso':
                model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
            elif regression == 'ridge':
                model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
            elif regression == 'elastic':
                l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
                model = ElasticNetCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], l1_ratio=l1_ratio, cv=None)
            else:
                raise ValueError("Regression type must be 'lasso' or 'ridge'")

            model.fit(design_matrix_X, neuron_activity)

            predicted_activity = model.predict(design_matrix_X)
            trialavg_neuron_activity = np.mean(neuron_data[:,0,:], axis=1)
            trialavg_predicted_activity = np.mean(predicted_activity.reshape(neuron_data[:,0,:].shape), axis=1)
            pearson_R = np.corrcoef(trialavg_predicted_activity, trialavg_neuron_activity)[0,1]

            GLM_params[animal][neuron] = {
                'weights': model.coef_,
                'intercept': model.intercept_,
                'alpha': model.alpha_ if regression == 'ridge' else model.alpha_,
                'l1_ratio': model.l1_ratio_ if regression == 'elastic' else None,
                'R2': model.score(design_matrix_X, neuron_activity),
                'R2_trialavg': pearson_R**2,
                'model': model
            }

    return GLM_params


def compute_residual_activity(GLM_params, reorganized_data, quintile=None):
    residual_activity = {}
    avg_residuals = []
    for animal in GLM_params:
        residual_activity[animal] = {}
        for neuron in GLM_params[animal]:
            # Pre-process the data
            neuron_data = reorganized_data[animal][neuron]
            if quintile is not None:
                num_trials = neuron_data.shape[2]
                start_idx,end_idx = get_quintile_indices(num_trials, quintile)
                neuron_data = neuron_data[:, :, start_idx:end_idx]

            flattened_data = flatten_data(neuron_data)
            neuron_activity = neuron_data[:,0,:]

            # Predict activity and compute residuals
            glm_model = GLM_params[animal][neuron]['model']
            flattened_input_variables = flattened_data[:,1:]
            predicted_activity = glm_model.predict(flattened_input_variables)
            predicted_activity = predicted_activity.reshape(neuron_activity.shape)
            residual_activity[animal][neuron] = neuron_activity - predicted_activity

            avg_residuals.append(np.mean(residual_activity[animal][neuron], axis=1))

    avg_residuals = np.array(avg_residuals)

    return residual_activity, avg_residuals


def remove_variables_from_glm(GLM_params, vars_to_remove, variable_list):
    GLM_params_ = copy.deepcopy(GLM_params)

    idx_to_remove = [variable_list[1:].index(var) for var in vars_to_remove if var in variable_list[1:]]

    modified_GLM_params = {}
    for animal_key, neurons in GLM_params_.items():
        modified_GLM_params[animal_key] = {}
        for neuron_idx, params in neurons.items():
            params['model'].coef_[idx_to_remove] = 0
            if "intercept" in vars_to_remove:
                params['model'].intercept_ = 0
                params['intercept'] = 0
            modified_GLM_params[animal_key][neuron_idx] = params

    return modified_GLM_params


def select_neuron(GLM_params, variable_list, sort_by='R2', animal=None, cell=None):

    variable_list = variable_list[1:]

    flattened_data = []

    if animal is not None:
        if animal not in GLM_params:
            raise KeyError(f"{animal} not found in GLM_params")
        if cell is not None and cell not in GLM_params[animal]:
            raise KeyError(f"{cell} not found in GLM_params for {animal}")
    
    for animal_id in GLM_params:
        for neuron_id in GLM_params[animal_id]:
            metrics = GLM_params[animal_id][neuron_id]
            row = {'animal': animal_id, 'neuron': neuron_id}
            row.update(metrics)
            flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    if sort_by in variable_list: # If sort_by is a variable, sort by the magnitude of the specified weight
        var_idx = variable_list.index(sort_by)
        df[f'{sort_by}_weight_magnitude'] = df['weights'].apply(lambda w: np.abs(w[var_idx])) # Add a column for the magnitude of the specified weight
        df_sorted = df.sort_values(by=[f'{sort_by}_weight_magnitude', 'R2'], ascending=[False, False])
        top_neuron = df_sorted.iloc[0]
        top_neuron_weight = top_neuron['weights'][var_idx]
        print(f"Top neuron for {sort_by}, with weight: {top_neuron_weight}")
    else:
        if sort_by != 'R2':
            raise ValueError(f"Invalid sort_by value: {sort_by}. Must be one of {variable_list} or 'R2'")
        df_sorted = df.sort_values(by=sort_by, ascending=False)

    if animal is not None:
        df_sorted = df_sorted[df_sorted['animal'] == animal]

    if cell is not None:
        df_sorted = df_sorted[df_sorted['neuron'] == cell]

    top_neuron = df_sorted.iloc[0]
    animal, cell = top_neuron['animal'], top_neuron['neuron']

    return animal, cell


def plot_example_neuron(reorganized_data, GLM_params, variable_list, sort_by='R2', animal=None, cell=None, ax=None):
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
        ax.vlines(24.5, 0, 1, linestyles='--', color='r')
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

    # # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][cell]['model']
    flattened_input_variables = flattened_data[:,1:]
    predicted_activity = glm_model.predict(flattened_input_variables)

    pearson_R = np.corrcoef(predicted_activity, flattened_data[:,0])[0,1]
    print("pearson R2 overall:", pearson_R**2)

    predicted_activity = predicted_activity.reshape(neuron_activity.shape)
    avg_predicted_activity = np.mean(predicted_activity, axis=1)
    std_predicted_activity = np.std(predicted_activity, axis=1)
    sem_predicted_activity = std_predicted_activity / np.sqrt(predicted_activity.shape[1])
    avg_neuron_activity = np.mean(neuron_activity, axis=1)
    std_neuron_activity = np.std(neuron_activity, axis=1)
    sem_neuron_activity = std_neuron_activity / np.sqrt(neuron_activity.shape[1])

    pearson_R = np.corrcoef(avg_predicted_activity, avg_neuron_activity)[0,1]
    print("pearson R2 average:", pearson_R**2)

    axes = gs.GridSpecFromSubplotSpec(nrows=3, ncols=3, subplot_spec=ax_, wspace=0., width_ratios=[0.3, 0.3, 0.4], height_ratios=[0.2,1,0.2])
    ax = fig.add_subplot(axes[1,2])
    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-sem_neuron_activity, avg_neuron_activity+sem_neuron_activity, alpha=0.1, color='k')
    ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-sem_predicted_activity, avg_predicted_activity+sem_predicted_activity, alpha=0.1, color='gray')
    ax.set_xlabel("Position")
    ax.set_xticks([0,50])
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
        GLM_params = fit_GLM(reorganized_data, quintile=quintile, regression='ridge', renormalize=False)
        GLM_params_ls.append(GLM_params)
        vars_to_remove = variable_list.copy()[1:] + ['intercept']
        vars_to_remove.remove('Velocity')
        filtered_GLM_params = remove_variables_from_glm(GLM_params, vars_to_remove, variable_list)
        residual_activity, avg_residuals = compute_residual_activity(filtered_GLM_params, reorganized_data, quintile=quintile)
        avg_residuals_ls.append(avg_residuals)

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


def get_GLM_R2(GLM_params):
    all_R2_values = np.array([GLM_params[animal][neuron]['R2_trialavg'] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_R2_values


def get_GLM_weights(GLM_params, variable_list):
    all_weights = {}
    for i,var_name in enumerate(variable_list[1:]):
        all_weights[var_name] = np.array([GLM_params[animal][neuron]['weights'][i] for animal in GLM_params for neuron in GLM_params[animal]])
    return all_weights


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


def calculate_delta_weights(GLM_params_first, GLM_params_last):
    assert GLM_params_first.keys() == GLM_params_last.keys(), "Animal keys do not match between the two GLM parameters dictionaries."

    delta_params = {}
    for animal in GLM_params_first:
        delta_params[animal] = {}
        for cell in GLM_params_first[animal]:
            weights_first = GLM_params_first[animal][cell]['weights']
            weights_last = GLM_params_last[animal][cell]['weights']
            intercept_first = GLM_params_first[animal][cell]['intercept']
            intercept_last = GLM_params_last[animal][cell]['intercept']

            delta_weights = weights_last - weights_first
            delta_intercept = intercept_last - intercept_first
            delta_params[animal][cell] = {'weights':delta_weights, 'intercept':delta_intercept}
    return delta_params


def plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    plot_GLM_summary_data(delta_weights, variable_list, ax=ax)
    ax.set_ylabel('Δ Weights\n(Last - First Quintile)')

    if model_name is not None:
        ax.set_title(model_name)

    if model_name is not None and save:
        fig.savefig(f"figures/{model_name}_delta_weights.png", dpi=300)




if __name__ == "__main__":

    datasets = ["SSTindivsomata_GLM.mat"]#, "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    filepath_list = ["SSTindivsomata_GLM.mat", "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    filepaths = {
        'NDNF': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\NDNFindivsomata_GLM.mat",
        'EC': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\EC_GLM.mat",
        'SST': r"C:\Users\Msfin\cloned_repositories\CA1-interneuron-GLM\datasets\SSTindivsomata_GLM.mat"
    }

    get_delta_weights_and_plot(filepath_list)

    # variable_list: ['Activity', 'Licks', 'R_loc', 'Speed', '#1', '#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9', '#10']
    vars_to_remove = [2]  # speed

    filtered_GLM_params = remove_variables_from_glm(GLM_params, vars_to_remove)

    for dataset_path in datasets:
        reorganized_data, variable_list = load_data(dataset_path)
        GLM_params = fit_GLM(reorganized_data, quintile=1, regression=lasso)

        fig = plt.figure(figsize=(10,5))
        axes = gs.GridSpec(nrows=1, ncols=3)
        fig.suptitle(dataset_path)

        ax = fig.add_subplot(axes[0,0])
        plot_example_neuron(reorganized_data, variable_list, model_name=dataset_path, fig=fig, ax=ax)

        ax = fig.add_subplot(axes[0,2])
        plot_GLM_summary_data(GLM_params, variable_list, dataset_path, ax=ax)

        fig.savefig(f"figures/{dataset_path[:-4]}.png", dpi=300)