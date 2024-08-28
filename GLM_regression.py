import mat73
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from sklearn.linear_model import RidgeCV


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


def fit_GLM(reorganized_data):
    GLM_params = {}

    for animal in reorganized_data:
        GLM_params[animal] = {}
        for i,neuron in enumerate(reorganized_data[animal]):
            flattened_data = neuron.reshape(neuron.shape[0]*neuron.shape[2], neuron.shape[1]) # Combine trials and spatial bins into a single dimension
            flattened_data = flattened_data[~np.isnan(flattened_data).any(axis=1)] # Remove rows with NaNs
            flattened_data = (flattened_data - np.min(flattened_data, axis=0)) / (np.max(flattened_data, axis=0) - np.min(flattened_data, axis=0)) # Normalize between 0 and 1 

            design_matrix_X = flattened_data[:, 1:]
            neuron_activity = flattened_data[:, 0]
            ridge_cv = RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 5000], store_cv_results=True)
            ridge_cv.fit(design_matrix_X, neuron_activity)

            GLM_params[animal][i] = {'weights': ridge_cv.coef_, 
                                    'intercept': ridge_cv.intercept_, 
                                    'alpha': ridge_cv.alpha_, 
                                    'cv_results': np.mean(ridge_cv.cv_results_, axis=0),
                                    'R2': ridge_cv.score(design_matrix_X, neuron_activity),
                                    'model': ridge_cv}
            
    return GLM_params
            

def plot_example_neuron(reorganized_data, variable_list, model_name, save=False, fig=None, ax=None):
    example_neuron = 0
    example_animal = 'animal_1'

    example_data = np.nanmean(reorganized_data[example_animal][example_neuron][:,:,1:], axis=2)
    example_data = (example_data - np.min(example_data, axis=0)) / (np.max(example_data, axis=0) - np.min(example_data, axis=0))
    example_neuron_activity = example_data[:, 0]
    example_variables = example_data[:, 1:]

    if ax:
        assert fig is not None, "Must specify both fig and ax"
        axes1 = gs.GridSpecFromSubplotSpec(nrows=example_variables.shape[1]-5, ncols=1, subplot_spec=ax, bottom=0.5)
        axes2 = gs.GridSpecFromSubplotSpec(nrows=5, ncols=1, subplot_spec=ax, top=0.5)
    else:
        fig = plt.figure(figsize=(6, 8))
        axes1 = gs.GridSpec(nrows=example_variables.shape[1]-5, ncols=1, figure=fig, bottom=0.5)
        axes2 = gs.GridSpec(nrows=5, ncols=1, figure=fig, top=0.5)

    for i in range(example_variables.shape[1]):
        if i < example_variables.shape[1]-5:
            ax = fig.add_subplot(axes1[i])
        else:
            ax = fig.add_subplot(axes2[i])
        ax.plot(example_variables[:, i])
        ax.set_ylabel(variable_list[i+1])
        ax.vlines(25, 0, 1, linestyles='--', color='r')
        if i == example_variables.shape[1] - 1:
            ax.set_xlabel('Position')
            ax.set_xticks([0,50])
        else:
            ax.set_xticks([])

    if save:
        fig.savefig(f"figures/{model_name[:-4]}_example_{example_animal}_neuron{example_neuron}.png", dpi=300)


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
        GLM_params = fit_GLM(reorganized_data)
        
        fig = plt.figure(figsize=(10,5))
        axes = gs.GridSpec(nrows=1, ncols=3)
        fig.suptitle(dataset_path)

        ax = fig.add_subplot(axes[0,0])
        plot_example_neuron(reorganized_data, variable_list, model_name=dataset_path, fig=fig, ax=ax)
        
        ax = fig.add_subplot(axes[0,2])
        plot_GLM_summary_data(GLM_params, variable_list, dataset_path, ax=ax)

        fig.savefig(f"figures/{dataset_path[:-4]}.png", dpi=300)