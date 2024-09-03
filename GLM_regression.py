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



def load_data(filepath):
    data_dict = mat73.loadmat(filepath)

    # Define new position variables to use as input for the GLM
    num_spatial_bins = 10
    position_matrix = np.zeros((50, num_spatial_bins))
    bin_size = 50//num_spatial_bins
    for i in range(num_spatial_bins):
        position_matrix[i*bin_size:(i+1)*bin_size,i] = 1

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
            expanded_position_matrix = np.repeat(position_matrix[:, :, np.newaxis], combined_matrix.shape[2], axis=2) # Copy along the 'trials' dimension
            combined_matrix = np.concatenate((combined_matrix, expanded_position_matrix), axis=1)
            variable_list.extend([f'#{i}' for i in range(1, num_spatial_bins+1)]) if neuron_idx == 1 and animal_idx == 0 else None
            neuron_list.append(combined_matrix)

        reorganized_data[f'animal_{animal_idx + 1}'] = neuron_list

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

            neuron_data = neuron_data[:,:,~np.isnan(neuron_data).any(axis=(0,1))]
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

            GLM_params[animal][i] = {
                'weights': model.coef_,
                'intercept': model.intercept_,
                'alpha': model.alpha_ if regression == 'ridge' else model.alpha_,
                'l1_ratio': model.l1_ratio_ if regression == 'elastic' else None,
                'R2': model.score(design_matrix_X, neuron_activity),
                'R2_trialavg': pearson_R**2,
                'model': model
            }

    return GLM_params


def compute_residual_activity(GLM_params, reorganized_data):
    residual_activity = {}
    for animal in GLM_params:
        residual_activity[animal] = {}
        for neuron in GLM_params[animal]:
            # Pre-process the data
            neuron_data = reorganized_data[animal][neuron]
            neuron_data = neuron_data[:, :, ~np.isnan(neuron_data).any(axis=(0, 1))]
            neuron_data = normalize_data(neuron_data)
            flattened_data = flatten_data(neuron_data)
            neuron_activity = neuron_data[:,0,:]

            # Predict activity and compute residuals
            glm_model = GLM_params[animal][neuron]['model']
            flattened_input_variables = flattened_data[:,1:]
            predicted_activity = glm_model.predict(flattened_input_variables)
            predicted_activity = predicted_activity.reshape(neuron_activity.shape)
            residual_activity[animal][neuron] = neuron_activity - predicted_activity

    return residual_activity


def remove_variables_from_glm(GLM_params, vars_to_remove, variable_list):
    GLM_params = copy.deepcopy(GLM_params)

    vars_to_remove = [variable_list[1:].index(var) for var in vars_to_remove]

    modified_GLM_params = {}

    for animal_key, neurons in GLM_params.items():
        modified_GLM_params[animal_key] = {}
        for neuron_idx, params in neurons.items():
            new_params = params.copy()

            for var in vars_to_remove:
                new_params['weights'][var] = 0

            new_params['model'].coef_[vars_to_remove] = 0

            modified_GLM_params[animal_key][neuron_idx] = new_params

    return modified_GLM_params


def plot_example_neuron_variables(example_variables, variable_list, ax, weights=None):
    variable_list = variable_list[1:]

    fig = ax.get_figure()
    height_ratios = np.ones(example_variables.shape[1])
    height_ratios[-10:] = 0.5
    axes = gs.GridSpecFromSubplotSpec(nrows=example_variables.shape[1], ncols=1, subplot_spec=ax, hspace=0.5, height_ratios=height_ratios)
    for i in range(example_variables.shape[1]):
        ax = fig.add_subplot(axes[i])
        ax.plot(example_variables[:, i], color='k')
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


def select_neuron(GLM_params, variable_list, sort_by='R2', animal=None, neuron=None):

    variable_list = variable_list[1:]

    # Flatten the dictionary and convert to a DataFrame
    flattened_data = []

    for animal_id, results in GLM_params.items():
        for neuron_id, metrics in results.items():
            row = {'animal': animal_id, 'neuron': neuron_id}
            row.update(metrics)
            flattened_data.append(row)

    df = pd.DataFrame(flattened_data)

    if sort_by in variable_list: # Sort by R2 and weight magnitude
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

    if neuron is not None:
        df_sorted = df_sorted[df_sorted['neuron'] == neuron]

    top_neuron = df_sorted.iloc[0]
    animal, neuron = top_neuron['animal'], top_neuron['neuron']

    return animal, neuron


def plot_example_neuron(reorganized_data, GLM_params, variable_list, animal=None, neuron=None, model_name=None, sort_by='R2'):

    animal, neuron = select_neuron(GLM_params, variable_list, sort_by=sort_by, animal=animal, neuron=neuron)
    print(f"Best neuron: {neuron}, {animal}")

    neuron_data = reorganized_data[animal][neuron]
    neuron_data = neuron_data[:,:,~np.isnan(neuron_data).any(axis=(0, 1))]
    neuron_data = normalize_data(neuron_data)
    flattened_data = flatten_data(neuron_data)

    input_variables = neuron_data[:,1:,:]
    neuron_activity = neuron_data[:,0,:]

    fig = plt.figure(figsize=(10, 8))

    axes = gs.GridSpec(nrows=1, ncols=1, left=0, right=0.3, bottom=0.5)
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    avg_variables = np.mean(input_variables, axis=2)
    plot_example_neuron_variables(avg_variables, variable_list, ax=ax)


    # Plot weights as lines across the figure
    axes = gs.GridSpec(nrows=1, ncols=1, left=0.3, right=0.54, bottom=0.5)
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    y1 = np.linspace(-0.54, -1.86, 3).tolist()
    y2 = np.linspace(-2.28, -6, 10).tolist()
    y = y1 + y2
    weights = GLM_params[animal][neuron]['weights']
    w_max = np.max(np.abs(weights))
    for i,w in enumerate(weights):
        if abs(w)<0.05:
            line, = ax.plot([0,1], [y[i],-3.4], color='lightgray', linestyle='--', linewidth=1)
        elif w < 0:
            line, = ax.plot([0,1], [y[i],-3.4], color='deepskyblue', linewidth=abs(w/w_max)*4)
        else:
            line, = ax.plot([0,1], [y[i],-3.4], color='black', linewidth=abs(w/w_max)*4)
        line.set_solid_capstyle('round')
    ax.set_ylim([-6,0])
    ax.plot([0,0], [0,0], color='lightgray', linewidth=1.5, linestyle='--', label='Small weights')
    ax.plot([0,0], [0,0], color='deepskyblue', linewidth=1.5, label='Negative weights')
    ax.plot([0,0], [0,0], color='black', linewidth=1.5, label='Positive weights')
    ax.legend(fontsize=10, loc='upper right', frameon=False, handlelength=1.5, handletextpad=0.5, labelspacing=0.2, borderpad=0)
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    ax.text(1, -5.6, f'Max weight: {max_weight:.2f}', ha='right', va='bottom', fontsize=10)
    ax.text(1, -6, f'Min weight: {min_weight:.2f}', ha='right', va='bottom', fontsize=10)

    # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][neuron]['model']
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

    axes = gs.GridSpec(nrows=1, ncols=1, left=0.61, right=1, top=0.8, bottom=0.55)
    ax = fig.add_subplot(axes[0])
    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-sem_neuron_activity, avg_neuron_activity+sem_neuron_activity, alpha=0.1, color='k')
    ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-sem_predicted_activity, avg_predicted_activity+sem_predicted_activity, alpha=0.1, color='gray')
    ax.set_xlabel("Position")
    ax.set_xticks([0,50])
    ax.set_ylabel("dF/F activity (Z-scored)")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))

    # Plot summary data
    axes = gs.GridSpec(nrows=1, ncols=1, top=0.45, bottom=0.2, left=0.3, right=1)
    ax = fig.add_subplot(axes[0])
    plot_GLM_summary_data(GLM_params, variable_list, ax=ax)

    axes = gs.GridSpec(nrows=1, ncols=1, top=0.45, bottom=0.2, left=0., right=0.2)
    ax = fig.add_subplot(axes[0])
    plot_R2_distribution(GLM_params, ax=ax)

    if model_name is not None:
        fig.savefig(f"figures/GLM_regression_{model_name}_{animal}_{neuron}.png", bbox_inches='tight', dpi=300)


def plot_GLM_summary_data(GLM_params, variable_list, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)

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
    ax.errorbar(np.arange(len(variable_list[1:]))-0.15, global_mean, yerr=global_std, fmt='o', color='red', ecolor='red', 
                capsize=5, label='Average of all animals', markersize=7)

    ax.set_xticks(range(len(variable_list[1:])), variable_list[1:], rotation=45, ha='right')
    ax.set_ylabel('Weights')
    ax.hlines(0, 0, len(variable_list[1:]), linestyles='--', color='black', alpha=0.5)
    ax.set_xlim([-0.5,len(variable_list[1:])-0.5])


def plot_R2_distribution(GLM_params, GLM_params2=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()

    model_list = [GLM_params]
    if GLM_params2 is not None:
        model_list.append(GLM_params2)

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
    if GLM_params2 is not None:
        ax.set_xticks([0.8, i+0.2])
        ax.set_xticklabels(['First quintile', 'Last quintile'])
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


def calculate_delta_weights(reorganized_data, GLM_params_first, GLM_params_last):
    delta_weights = {}

    for animal in reorganized_data:
        delta_weights[animal] = []
        for i in range(len(GLM_params_first[animal])):
            weights_first = GLM_params_first[animal][i]['weights']
            weights_last = GLM_params_last[animal][i]['weights']

            delta = weights_last - weights_first
            delta_weights[animal].append(delta)

    return delta_weights


def plot_delta_weights_summary(delta_weights, variable_list, model_name=None, save=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    animal_averages = []
    jitter = 0.25
    variable_list = variable_list[1:]

    for animal_key in delta_weights:
        neuron_weights = []
        for neuron_nr in range(len(delta_weights[animal_key])):
            neuron_weights.append(delta_weights[animal_key][neuron_nr])
            jittered_x = np.arange(len(variable_list)) + np.random.uniform(0.1, jitter, len(variable_list))
            ax.scatter(jittered_x, delta_weights[animal_key][neuron_nr], color='grey', alpha=0.2, s=10)

        neuron_weights = np.array(neuron_weights)
        mean_weights = np.mean(neuron_weights, axis=0)
        animal_averages.append(mean_weights)
        ax.scatter(range(len(variable_list)), mean_weights, color='black', s=20)

    animal_averages = np.array(animal_averages)
    global_mean = np.mean(animal_averages, axis=0)
    global_std = np.std(animal_averages, axis=0)
    global_sem = global_std / np.sqrt(len(animal_averages))
    ax.errorbar(np.arange(len(variable_list)) - 0.15, global_mean, yerr=global_std, fmt='o', color='red', ecolor='red',
                capsize=5, markersize=7)

    absolute_distances = np.abs(global_mean)
    max_index = np.argmax(absolute_distances)
    second_max_index = np.argsort(absolute_distances)[-2]

    ax.set_xticks(range(len(variable_list)), variable_list, rotation=45, ha='right')
    ax.set_ylabel('Δ Weights\n(Last - First Quintile)')
    ax.hlines(0, -0.5, len(variable_list) - 0.5, linestyles='--', color='black', alpha=0.5)
    ax.set_xlim([-0.5, len(variable_list) - 0.5])

    if model_name is not None:
        ax.set_title(model_name)

    if model_name is not None and save:
        fig.savefig(f"{model_name}_delta_weights.png", dpi=300)

    return max_index, second_max_index


def get_delta_weights_and_plot(filepath_list):
    for filepath in filepath_list:
        reorganized_data, variable_list = load_data(filepath)

        GLM_params_first = fit_GLM(reorganized_data, quintile=1, regression='ridge')
        GLM_params_last = fit_GLM(reorganized_data, quintile=5, regression='ridge')

        delta_weights = calculate_delta_weights(reorganized_data, GLM_params_first, GLM_params_last)


        fig, ax = plt.subplots(figsize=(10, 5))
        max_index, second_max_index = plot_delta_weights_summary(delta_weights, variable_list,
                                                                 model_name=filepath.split('.')[0], ax=ax)


        x_data = np.concatenate([np.array(delta_weights[animal])[:, max_index] for animal in delta_weights])
        y_data = np.concatenate([np.array(delta_weights[animal])[:, second_max_index] for animal in delta_weights])


        fig, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(x_data, y_data, color='blue', alpha=0.6)
        ax2.set_xlabel(f"Δ {variable_list[1:][max_index]}")
        ax2.set_ylabel(f"Δ {variable_list[1:][second_max_index]}")
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.axvline(0, color='gray', linestyle='--')


        m, b = np.polyfit(x_data, y_data, 1)
        ax2.plot(x_data, m * x_data + b, color='red', linestyle='--')
        ax2.set_title(f'2D Scatter Plot of Most Differing Variables - {filepath}')

        plt.show()


if __name__ == "__main__":

    datasets = ["SSTindivsomata_GLM.mat"]#, "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

    filepath_list = ["SSTindivsomata_GLM.mat", "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]
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