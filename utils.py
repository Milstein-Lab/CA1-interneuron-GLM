import mat73
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
from scipy.stats import norm
import random
# from pygam import LinearGAM

###############################################################################################
# Data Processing
###############################################################################################

def preprocess_data(filepath, normalize=True):
    data_dict = mat73.loadmat(filepath)

    # Define new position variables to use as input for the GLM
    num_spatial_bins = 10
    position_matrix = np.zeros((50, num_spatial_bins))
    bin_size = 50 // num_spatial_bins
    for i in range(num_spatial_bins):
        position_matrix[i * bin_size:(i + 1) * bin_size, i] = 1

    factors_dict = {}
    activity_dict = {}
    for animal_idx, (delta_f, velocity, lick_rate, reward_loc) in enumerate(zip(data_dict['animal']['ShiftR'], data_dict['animal']['ShiftRunning'], data_dict['animal']['ShiftLrate'],data_dict['animal']['ShiftV'])):
        num_trials = min(delta_f.shape[1], lick_rate.shape[1], reward_loc.shape[1], velocity.shape[1])
        lick_rate = lick_rate[:, :num_trials]
        reward_loc = reward_loc[:, :num_trials]
        velocity = velocity[:, :num_trials]
        delta_f = delta_f[:, :num_trials, :]

        # Exclude trials with NaNs
        nan_trials_licks = np.any(np.isnan(lick_rate), axis=0)
        nan_trials_reward = np.any(np.isnan(reward_loc), axis=0)
        nan_trials_velocity = np.any(np.isnan(velocity), axis=0)
        nan_trials_activity = np.any(np.isnan(delta_f), axis=(0, 2))
        nan_trials = nan_trials_licks | nan_trials_reward | nan_trials_velocity | nan_trials_activity

        animal_key = f'animal_{animal_idx + 1}'
        factors_dict[animal_key] = {"Licks": lick_rate[:, ~nan_trials],
                                    "Reward_loc": reward_loc[:, ~nan_trials],
                                    "Velocity": velocity[:, ~nan_trials]}

        # Add a factor for each position variable (from 1 to num_spatial_bins)
        num_trials = factors_dict[animal_key]["Velocity"].shape[1]
        for bin_idx in range(num_spatial_bins):
            bin_key = f"Position_{bin_idx + 1}"
            factors_dict[animal_key][bin_key] = np.tile(position_matrix[:, bin_idx][:, np.newaxis], num_trials) # Copy the position variable for each trial

        if normalize: # Normalize behavioral factors to [0,1]
            for var_name in factors_dict[animal_key]:
                factors_dict[animal_key][var_name] = (factors_dict[animal_key][var_name] - np.min(factors_dict[animal_key][var_name])) / (np.max(factors_dict[animal_key][var_name]) - np.min(factors_dict[animal_key][var_name]))

        activity_dict[animal_key] = {}
        for neuron_idx in range(delta_f.shape[2]):
            neuron_activity = delta_f[:, :, neuron_idx]
            if np.all(np.isnan(neuron_activity)) or np.all(neuron_activity == 0): # Don't save empty/silent neurons
                continue
            cleaned_activity = neuron_activity[:, ~nan_trials]

            if normalize:  # Z-score the neuron activity (df/f)
                cleaned_activity = (cleaned_activity - np.mean(cleaned_activity)) / np.std(cleaned_activity)

            neuron_key = f'cell_{neuron_idx + 1}'
            activity_dict[animal_key][neuron_key] = cleaned_activity

    return activity_dict, factors_dict


def subset_variables_from_data(factors_dict, variables_to_keep=["Velocity"]):
    filtered_factors_dict = {}
    for animal in factors_dict:
        filtered_factors_dict[animal] = {}
        for variable in variables_to_keep:
            filtered_factors_dict[animal][variable] = factors_dict[animal][variable]
    return filtered_factors_dict


def normalize_data(neuron_dict):
    for var_name in neuron_dict:
        if var_name == "Activity": # Z-score the neuron activity (df/f)
            neuron_dict[var_name] = (neuron_dict[var_name] - np.mean(neuron_dict[var_name])) / np.std(neuron_dict[var_name])
        else: # Normalize the other variables to [0,1]
            neuron_dict[var_name] = (neuron_dict[var_name] - np.min(neuron_dict[var_name])) / (np.max(neuron_dict[var_name]) - np.min(neuron_dict[var_name]))
        

def get_residual_activity_dict(activity_dict, predicted_activity_dict):
    residual_activity_dict = {}
    for animal in activity_dict:
        residual_activity_dict[animal] = {}
        for neuron in activity_dict[animal]:
            residual_activity_dict[animal][neuron] = activity_dict[animal][neuron] - predicted_activity_dict[animal][neuron]
    return residual_activity_dict


def flatten_data(neuron_dict):
    flattened_data = {}
    for var in neuron_dict:
        flattened_data[var] = neuron_dict[var].flatten()
    return flattened_data


def create_RNN_data_sequences(data_x, data_y, seq_length, split='trial'):
    X_seq, y_seq = [], []

    if split == 'continuous': # Assuming data_x and data_y are flat numpy arrays
        assert len(data_x.shape)==1 and len(data_y.shape)==1, "Data must be 1D for continuous splitting"        
        for i in range(len(data_x) - seq_length):
            X_seq.append(data_x[i : i + seq_length])  # Input sequence
            y_seq.append(data_y[i + seq_length])      # Target at next step

    elif split == 'trial': # Assuming data_x and data_y are 2D numpy arrays (trials, time_steps)
        assert len(data_x.shape)==2 and len(data_y.shape)==2, "Data must be 2D for trial splitting"
        for trial in range(data_x.shape[0]):
            for i in range(data_x.shape[1] - seq_length):
                X_seq.append(data_x[trial, i : i + seq_length])  # Input sequence
                y_seq.append(data_y[trial, i + seq_length])      # Target at next step

    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(y_seq), dtype=torch.float32)



def example_EC_cell(velocity):
    length = 50
    num_trials = velocity.shape[1]
    x = np.linspace(0, length - 1, length)

    mean1 = 15
    std_dev1 = 5
    original_gaussian1 = norm.pdf(x, mean1, std_dev1)

    mean2 = 35
    std_dev2 = 5
    original_gaussian2 = norm.pdf(x, mean2, std_dev2)

    gaussian_list = []

    for i in range(num_trials):
        appear_1 = np.random.choice([0, 1])
        appear_2 = np.random.choice([0, 1])

        gaussian1 = original_gaussian1 * appear_1
        gaussian2 = original_gaussian2 * appear_2

        combined_gaussian = gaussian1 + gaussian2

        bimodal_gaussian = combined_gaussian / np.max(combined_gaussian) if np.max(combined_gaussian) != 0 else combined_gaussian

        gaussian_list.append(bimodal_gaussian)

    pf = np.stack(gaussian_list)

    return pf



def BTSP_field(num_trials):
    BTSP_trial = random.randint(0, num_trials)

    trial_weights = np.zeros(num_trials)

    trial_weights[BTSP_trial:] = 1

    return trial_weights



def get_synthetic_data(activity_dict, velocity, place_field_type='flat', place_field_scale=1, place_field_shift=0, velocity_weight_type='flat', velocity_weight=1, velocity_power=1, noise_scale=1):
    # 1. Make "ground truth" place field
    def get_average_cell_profile(activity_dict):
        all_cells_average = []
        for animal in activity_dict:
            for neuron in activity_dict[animal]:
                cell_trial_average = activity_dict[animal][neuron].mean(axis=1)
                all_cells_average.append(cell_trial_average)
        all_cells_average = np.stack(all_cells_average, axis=0).mean(axis=0)
        return all_cells_average

    place_field_profile = get_average_cell_profile(activity_dict)
    num_trials = velocity.shape[1]
    place_field = np.tile(place_field_profile, (num_trials,1)).T
    def staircase_vector(start, stop, num_steps, length):
        steps = np.linspace(start, stop, num_steps)  # Generate step levels
        step_counts = np.full(num_steps, length // num_steps)  # Base count per step
        step_counts[:length % num_steps] += 1  # Distribute remainder among first steps
        return np.repeat(steps, step_counts)  # Repeat steps with adjusted counts
    match place_field_type:
        case "flat":
            place_field_scale = np.ones(num_trials)
            place_field *= place_field_scale
        case "positive_ramp":
            place_field_scale = np.linspace(0, place_field_scale, num_trials)
            place_field *= place_field_scale
        case "negative_ramp":
            place_field_scale = np.linspace(place_field_scale, 0, num_trials)
            place_field *= place_field_scale
        case "step":
            place_field_scale = staircase_vector(0, place_field_scale, num_steps=2, length=num_trials)
            place_field *= place_field_scale
        case "BTSP":
            place_field_scale = BTSP_field(num_trials)
            place_field *= place_field_scale
        case "EC":
            place_field = example_EC_cell(velocity)
            place_field = place_field.T

    place_field = np.roll(place_field, shift=place_field_shift, axis=0)


    # 2. Combine the synthetic place field with velocity
    match velocity_weight_type:
        case "flat":
            velocity_weight = velocity_weight * np.ones(num_trials)
        case "positive_ramp":
            velocity_weight = np.linspace(0, velocity_weight, num_trials)
        case "negative_ramp":
            velocity_weight = np.linspace(velocity_weight, 0, num_trials)
        case "step":
            velocity_weight = staircase_vector(0, velocity_weight, num_steps=5, length=num_trials)

    velocity_component = velocity_weight * (velocity**velocity_power)

    noise = np.random.normal(0, noise_scale, size=(len(place_field_profile), num_trials))
    combined_activity = place_field + velocity_component + noise

    return combined_activity, place_field, velocity_component, noise


###############################################################################################
# Model fitting
###############################################################################################

############### Simple models ###############

def fit_GLM_population(factors_dict, activity_dict, quintile=None, regression='ridge', renormalize=True, alphas=None):
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

        if renormalize:
            normalize_data(animal_factors_dict)

        for neuron_idx in activity_dict[animal]:
            neuron_activity = activity_dict[animal][neuron_idx]
            neuron_GLM_params, neuron_predicted_activity = fit_GLM(animal_factors_dict, neuron_activity, regression, alphas)
            GLM_params[animal][neuron_idx] = neuron_GLM_params
            predicted_activity_dict[animal][neuron_idx] = neuron_predicted_activity.reshape(
                activity_dict[animal][neuron_idx].shape)
                
    return GLM_params, predicted_activity_dict


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


def fit_behavior_GLM(animal_activity_dict, behavior_data, regression='ridge', alphas=None):
    neural_data = []
    for cell, data in animal_activity_dict.items():
        neural_data.append(data.flatten())

    design_matrix_X = np.stack(neural_data, axis=1)
    behavior_data_flattened = behavior_data.flatten()

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

    model.fit(design_matrix_X, behavior_data_flattened)

    predicted_behavior = model.predict(design_matrix_X)
    pearson_R = np.corrcoef(predicted_behavior, behavior_data_flattened)[0, 1]

    animal_GLM_params = {}
    animal_GLM_params['alpha'] = model.alpha_ if regression == 'ridge' else None
    animal_GLM_params['l1_ratio'] = model.l1_ratio_ if regression == 'elastic' else None
    animal_GLM_params['R2'] = model.score(design_matrix_X, behavior_data_flattened)
    animal_GLM_params['pearson_R'] = pearson_R
    animal_GLM_params['model'] = model

    return animal_GLM_params, predicted_behavior.reshape(behavior_data.shape)


# def fit_GAM(velocity, activity, regression='linear', alphas=None):
#     """
#     Fit a Generalized Additive Model (GAM)
#     """
#     velocity_flat = velocity.flatten()
#     activity_flat = activity.flatten()

#     model = LinearGAM()
#     model.fit(velocity_flat, activity_flat)
#     neuron_predicted_activity = model.predict(flattened_data)
#     return model, neuron_predicted_activity



################ RNN models ################
class VelocityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, (hn,cn) = self.lstm(x, (h0, c0)) # Out shape: (batch_size, seq_length, hidden_size). hn and cn are the final hidden and cell states.
        out = self.fc(out[:, -1, :]) # Use the last time step output for regression
        return out


class VelocityGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, hn = self.gru(x, h0)  # GRU only returns hidden state
        out = self.fc(out[:, -1, :])  # Take the last time step output for regression
        return out


class VelocityRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity='relu', bias=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, hn = self.rnn(x, h0)  # RNN only returns hidden state
        out = self.fc(out[:, -1, :])  # Take the last time step output for regression
        return out


########### Convolutional models ###########
class ConvModel1D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=self.padding)

    def forward(self, x):
        out = self.conv1(x)
        return out

    def fit(self, velocity, activity, learning_rate=0.0005, num_iterations = 10_000, plot=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        velocity_flat = torch.tensor(velocity.flatten()).float().unsqueeze(0).unsqueeze(0).to(device)
        activity_flat = torch.tensor(activity.flatten()).float().unsqueeze(0).unsqueeze(0).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        for epoch in tqdm(range(num_iterations)): 
            optimizer.zero_grad()
            activity_pred = self.forward(velocity_flat)
            loss = criterion(activity_pred, activity_flat)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        self.train_loss = train_loss

        if plot:
            # Plot Training Loss
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train Loss", color="k")
            ax.plot(train_loss, label="Train Loss", color="k")
            ax.tick_params(axis="y", labelcolor="k")
            plt.title("Training Loss")
            fig.tight_layout()
            plt.show()

        return activity_pred[0,0].detach().cpu().numpy()


class ConvModel2D(nn.Module):
    def __init__(self, kernel_width, kernel_height):
        super().__init__()
        self.kernel_size = (kernel_height, kernel_width)
        self.padding = (kernel_height//2, kernel_width//2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, x):
        out = self.conv1(x)
        return out

    def fit(self, velocity, activity, learning_rate=0.0005, num_iterations = 10_000, plot=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        velocity_tensor = torch.tensor(velocity).float().unsqueeze(0).unsqueeze(0).to(device)
        activity_tensor = torch.tensor(activity).float().unsqueeze(0).unsqueeze(0).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        train_loss = []
        for epoch in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            activity_pred = self.forward(velocity_tensor)
            loss = criterion(activity_pred, activity_tensor)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        self.train_loss = train_loss

        if plot:
            # Plot Training Loss
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train Loss", color="k")
            ax.plot(train_loss, label="Train Loss", color="k")
            ax.tick_params(axis="y", labelcolor="k")
            plt.title("Training Loss")
            fig.tight_layout()
            plt.show()
        
        return activity_pred[0,0].detach().cpu().numpy()


class VelocityCNN1D(nn.Module):
    def __init__(self, kernel_size=51):
        super().__init__()
        self.hidden_size = hidden_size
        self.padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=self.padding)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = out.view(out.size(0), -1)  # Flatten the output for the fully connected layer
        out = self.fc(out)
        return out


###############################################################################################
# Other
###############################################################################################

def get_quintile_indices(num_trials, quintile=None):
    quintile_indices = [(i * num_trials) // 5 for i in range(6)]
    start_idx = quintile_indices[quintile - 1]
    end_idx = quintile_indices[quintile]
    return start_idx,end_idx
