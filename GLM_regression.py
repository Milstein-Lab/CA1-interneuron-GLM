import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import LassoCV, RidgeCV

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
    position_matrix = np.zeros((50, 5))
    bin_slices = [slice(i, i + 10) for i in range(0, 50, 10)]
    for i, sl in enumerate(bin_slices):
        position_matrix[sl, i] = 1

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
            expanded_position_matrix = np.roll(expanded_position_matrix, 5, axis=0)
            combined_matrix = np.concatenate((combined_matrix, expanded_position_matrix), axis=1)
            variable_list.extend(['#1', '#2', '#3', '#4', '#5']) if neuron_idx == 1 and animal_idx == 0 else None
            
            neuron_list.append(combined_matrix)

        reorganized_data[f'animal_{animal_idx + 1}'] = neuron_list

    return reorganized_data, variable_list


def fit_GLM(reorganized_data, quintile=None, regression='lasso', alphas=None):
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

            flattened_data = []
            for var_idx in range(neuron_data.shape[1]):
                if var_idx == 0: # Z-score the neuron activity (df/f)
                    neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmean(neuron_data[:,var_idx])) / np.nanstd(neuron_data[:,var_idx])
                else: # Normalize the other variables to [0,1]
                    neuron_data[:,var_idx] = (neuron_data[:,var_idx] - np.nanmin(neuron_data[:,var_idx])) / (np.nanmax(neuron_data[:,var_idx]) - np.nanmin(neuron_data[:,var_idx]))
                flattened_data.append(neuron_data[:,var_idx].flatten())
            flattened_data = np.stack(flattened_data, axis=1)
            flattened_data = flattened_data[~np.isnan(flattened_data).any(axis=1)]

            design_matrix_X = flattened_data[:, 1:]
            neuron_activity = flattened_data[:, 0]

            if regression == 'lasso':
                model = LassoCV(alphas=alphas, cv=None) if alphas is not None else LassoCV(cv=None)
            elif regression == 'ridge':
                model = RidgeCV(alphas=alphas if alphas is not None else [0.1, 1, 10, 100, 1000, 5000], cv=None)
            else:
                raise ValueError("Regression type must be 'lasso' or 'ridge'")

            model.fit(design_matrix_X, neuron_activity)

            GLM_params[animal][i] = {
                'weights': model.coef_,
                'intercept': model.intercept_,
                'alpha': model.alpha_ if regression == 'ridge' else model.alpha_,
                'R2': model.score(design_matrix_X, neuron_activity),
                'model': model
            }

    return GLM_params


def plot_example_neuron_variables(example_variables, variable_list, weights, ax):
    variable_list = variable_list[1:]

    fig = ax.get_figure()
    height_ratios = np.ones(example_variables.shape[1])
    height_ratios[-5:] = 0.5
    axes = gs.GridSpecFromSubplotSpec(nrows=example_variables.shape[1], ncols=1, subplot_spec=ax, hspace=0.5, height_ratios=height_ratios)
    for i in range(example_variables.shape[1]):
        ax = fig.add_subplot(axes[i])
        ax.plot(example_variables[:, i])
        ax.set_ylabel(variable_list[i])
        ax.set_xticks([])
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


def plot_example_neuron(animal, reorganized_data, GLM_params, variable_list, neuron=None):

    # Pick neuron with the highest R2 value
    if neuron is None:
        R2_values = [GLM_params[animal][i]['R2'] for i in GLM_params[animal]]
        neuron = np.argmax(R2_values)
    print("R2:", GLM_params[animal][neuron]['R2'])
    print("alpha:", GLM_params[animal][neuron]['alpha'])
    weights = GLM_params[animal][neuron]['weights']
    
    neuron_data = reorganized_data[animal][neuron][:,:,1:]  
    neuron_data = neuron_data[:,:,~np.isnan(neuron_data).any(axis=(0,1))]

    flattened_data = []
    for i in range(neuron_data.shape[1]):
        if i == 0: # Z-score the neuron activity (df/f)
            neuron_data[:,i] = (neuron_data[:,i] - np.mean(neuron_data[:,i])) / np.std(neuron_data[:,i])
        else: # Normalize the other variables to [0,1]
            neuron_data[:,i] = (neuron_data[:,i] - np.min(neuron_data[:,i])) / (np.max(neuron_data[:,i]) - np.min(neuron_data[:,i]))
        flattened_data.append(neuron_data[:,i].flatten())
    flattened_data = np.stack(flattened_data, axis=1)
        
    input_variables = neuron_data[:,1:,:]
    neuron_activity = neuron_data[:,0 ,:]

    fig = plt.figure(figsize=(10,5))
    
    # Plot input variables
    axes = gs.GridSpec(nrows=1, ncols=3)
    ax = fig.add_subplot(axes[0,0])
    ax.axis('off')
    avg_variables = np.mean(input_variables, axis=2)
    plot_example_neuron_variables(avg_variables, variable_list, weights, ax=ax)


    # Plot weights as lines across the figure
    axes = gs.GridSpec(nrows=1, ncols=1, left=0.36, right=0.54, top=0.82, bottom=0.08)
    ax = fig.add_subplot(axes[0])
    ax.axis('off')
    y1 = np.linspace(-1, -2.8, 3).tolist()
    y2 = np.linspace(-3.3, -5.5, 5).tolist()
    y = y1 + y2
    for i,w in enumerate(weights):
        if w > 0:
            ax.plot([0,1], [y[i],-3.5], color='blue', linewidth=abs(w))
        else:
            ax.plot([0,1], [y[i],-3.5], color='black', linewidth=abs(w))

    # Plot prediction vs actual neuron activity
    glm_model = GLM_params[animal][neuron]['model']
    flattened_input_variables = flattened_data[:,1:]
    predicted_activity = glm_model.predict(flattened_input_variables)
    predicted_activity = predicted_activity.reshape(neuron_activity.shape)

    avg_predicted_activity = np.mean(predicted_activity, axis=1)
    std_predicted_activity = np.std(predicted_activity, axis=1)
    avg_neuron_activity = np.mean(neuron_activity, axis=1)
    std_neuron_activity = np.std(neuron_activity, axis=1)

    axes = gs.GridSpec(nrows=1, ncols=1, left=0.6, right=1, top=0.7, bottom=0.2)
    ax = fig.add_subplot(axes[0])

    # pred_norm = (predicted_activity - np.min(predicted_activity)) / (np.max(predicted_activity) - np.min(predicted_activity))
    # actual_norm = (neuron_activity - np.min(neuron_activity)) / (np.max(neuron_activity) - np.min(neuron_activity))
    # ax.plot(pred_norm, label='GLM prediction', c='gray', linestyle='--')
    # ax.plot(actual_norm, label='Actual activity', c='k')

    ax.plot(avg_predicted_activity, label='GLM prediction', c='gray', linestyle='--')
    ax.fill_between(np.arange(avg_predicted_activity.shape[0]), avg_predicted_activity-std_predicted_activity, avg_predicted_activity+std_predicted_activity, alpha=0.2, color='gray')
    ax.plot(avg_neuron_activity, label='Actual activity', c='k')
    ax.fill_between(np.arange(avg_neuron_activity.shape[0]), avg_neuron_activity-std_neuron_activity, avg_neuron_activity+std_neuron_activity, alpha=0.2, color='k')
    ax.set_xlabel("Position")
    ax.set_xticks([0,50])
    ax.set_ylabel("dF/F activity")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.2))



def plot_GLM_summary_data(GLM_params, variable_list, model_name, save=False, ax=None):

    if ax is None:
        fig, ax = plt.subplots(1,1)

    animal_averages = []
    animal_stds = []
    jitter = 0.28

    for animal_key in GLM_params:
        neuron_weights = []
        for neuron_nr in range(len(GLM_params[animal_key])):
            neuron_weights.append(GLM_params[animal_key][neuron_nr]['weights'])        
            jittered_x = np.arange(len(variable_list[1:])) + np.random.uniform(-jitter, jitter, len(variable_list[1:]))
            ax.scatter(jittered_x, GLM_params[animal_key][neuron_nr]['weights'], color='grey', alpha=0.5)
        
        neuron_weights = np.array(neuron_weights)
        mean_weights = np.mean(neuron_weights, axis=0)
        std_weights = np.std(neuron_weights, axis=0)  # Use standard deviation
        animal_averages.append(mean_weights)
        animal_stds.append(std_weights)
        ax.scatter(range(len(variable_list[1:])), mean_weights, color='black', label=f'Animal {animal_key}', s=100)

    animal_averages = np.array(animal_averages)
    animal_stds = np.array(animal_stds)

    global_mean = np.mean(animal_averages, axis=0)
    global_std = np.std(animal_averages, axis=0)
    ax.errorbar(range(len(variable_list[1:])), global_mean, yerr=global_std, fmt='o', color='red', ecolor='red', 
                capsize=5, label='Average of all animals', markersize=10)

    ax.set_xticks(range(len(variable_list[1:])), variable_list[1:], rotation=45, ha='right')
    ax.set_xlabel('Variables')
    ax.set_ylabel('Weights')
    ax.set_title('Neuron Weights Scatter Plot')

    ax.hlines(0, 0, len(variable_list[1:]), linestyles='--', color='black', alpha=0.5)

    if save:
        fig.savefig(f"figures/{model_name[:-4]}_GLM_summary_data.png", dpi=300)




if __name__ == "__main__":

    datasets = ["SSTindivsomata_GLM.mat"]#, "NDNFindivsomata_GLM.mat", "EC_GLM.mat"]

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