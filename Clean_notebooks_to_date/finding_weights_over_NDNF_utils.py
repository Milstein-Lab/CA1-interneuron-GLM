import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils_TCA_clustering_scratchpad import *
from GLM_regression_plotting import *


from modelling_to_date_utils import *




def subset_factors(factors_dict_NDNF_newest):
    fixed_factors_dict_NDNF_newest = {}
    for idx, animal in enumerate(factors_dict_NDNF_newest):
        if 17 < idx < 31:
            fixed_factors_dict_NDNF_newest[f"animal_{idx+1}"] = factors_dict_NDNF_newest[animal]

    return fixed_factors_dict_NDNF_newest

def generate_spatial_rate_maps(x, n=200, peak_rate=1., field_width=90., track_length=180.):
    """
    Return a list of spatial rate maps with peak locations that span the track. Return firing rate vs. location
    computed at the resolution of the provided x array.
    :param x: array
    :param n: int
    :param peak_rate: float
    :param field_width: float
    :param track_length: float
    :return: list of array, array
    """
    gauss_sigma = field_width / 3. / np.sqrt(2.)  # contains 99.7% gaussian area
    d_peak_locs = track_length / float(n)
    peak_locs = np.arange(d_peak_locs / 2., track_length, d_peak_locs)
    spatial_rate_maps = []
    extended_x = np.concatenate([x - track_length, x, x + track_length])
    for peak_loc in peak_locs:
        gauss_force = peak_rate * np.exp(-((extended_x - peak_loc) / gauss_sigma) ** 2.)
        gauss_force = wrap_around_and_compress(gauss_force, x)
        spatial_rate_maps.append(gauss_force)
    return spatial_rate_maps, peak_locs

def wrap_around_and_compress(waveform, interp_x):
    before = np.array(waveform[:len(interp_x)])
    after = np.array(waveform[2 * len(interp_x):])
    within = np.array(waveform[len(interp_x):2 * len(interp_x)])
    compressed_waveform = within[:len(interp_x)] + before[:len(interp_x)] + after[:len(interp_x)]
    return compressed_waveform

def get_CA3_data(track_length = 180, num_cells=200):
    binned_dx = track_length / 49

    binned_x = np.arange(0., track_length + binned_dx / 2., binned_dx, dtype=np.float32)[:100] + binned_dx / 2.

    ca3 = generate_spatial_rate_maps(binned_x, n=num_cells, peak_rate=1., field_width=90., track_length=180.)

    ca3_vs_position_all_cells = ca3[0]

    ca3_vs_position_all_cells_array = np.array(ca3_vs_position_all_cells)
    
    return ca3_vs_position_all_cells_array

def plot_CA3(ca3_vs_position_all_cells_array):

    plt.imshow(ca3_vs_position_all_cells_array, aspect='auto')
    plt.title("CA3 Activity")
    plt.ylabel("Cell ID")
    plt.xlabel("Position Bin")
    plt.colorbar()
    plt.show()

    position_bins = 50

    dx = 180/position_bins

    visualization_trials = [20, 40]

    p = np.linspace(0, 180, 20)
    x = []
    for i in (p):
        x.append(int(i))

    for j in x:
        plt.plot(ca3_vs_position_all_cells_array[j, :])
    plt.title("Subset of CA3 Neurons")
    plt.xlabel("Position Bins")
    plt.ylabel("Activity")
    plt.show()
    
def get_plot_EC(residual_activity_dict_EC): 
    EC_residuals_list = []
    for animal in residual_activity_dict_EC:
        for cell in residual_activity_dict_EC[animal]:
            EC_residuals_list.append(np.mean(residual_activity_dict_EC[animal][cell], axis=1))

    EC_residuals_array = np.array(EC_residuals_list)

    sorted_EC_residuals_array = np.argsort(np.argmax(EC_residuals_array, axis=1))
    
    plt.imshow(EC_residuals_array[sorted_EC_residuals_array, :], aspect='auto')
    plt.title("EC Activity")
    plt.ylabel("Cell ID")
    plt.xlabel("Position Bin")
    plt.colorbar
    plt.show()
    
    return EC_residuals_array

def get_synthetic_NDNF(EC_residuals_array, ca3_vs_position_all_cells_array, dist_type = "normal"):

    n_ndnf = 110
    n_EC = EC_residuals_array.shape[0]
    n_CA3 = ca3_vs_position_all_cells_array.shape[0]

    EC_norm = EC_residuals_array / np.max(EC_residuals_array, axis=0)
    CA3_norm = ca3_vs_position_all_cells_array / np.max(ca3_vs_position_all_cells_array, axis=0)

    NDNF_activity_list = []

    for i in range(n_ndnf):

        if dist_type == "normal":
            weights_EC = np.random.normal(loc=0, scale=1, size=n_EC)
            weights_CA3 = np.random.normal(loc=0, scale=1, size=n_CA3)
        elif dist_type == "lognormal":
    #         weights = np.random.lognormal(mean=0, sigma=1, size=50)
            weights_EC = np.random.lognormal(mean=0, sigma=1, size=n_EC)
            weights_CA3 = np.random.lognormal(mean=0, sigma=1, size=n_CA3)
        elif dist_type == "uniform":
    #         weights = np.random.uniform(low=0, high=1, size=50)
            weights_EC = np.random.uniform(low=0, high=1, size=n_EC)
            weights_CA3 = np.random.uniform(low=0, high=1, size=n_CA3)

        ndnf_activity = (weights_EC @ EC_norm + weights_CA3 @ CA3_norm)

        NDNF_activity_list.append(ndnf_activity)

    return NDNF_activity_list

def fit_GLM2(EC_data_array, neuron_activity_flat, regression='linear', alphas=None):
    
    design_matrix_X = EC_data_array.T
    
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
    
    return model 

def get_EC_data_array(residual_activity_dict_EC):
    min_val = 1000
    for animal in residual_activity_dict_EC:
        for cell in residual_activity_dict_EC[animal]:
            if residual_activity_dict_EC[animal][cell].shape[1] < min_val:
                min_val = residual_activity_dict_EC[animal][cell].shape[1] 

    EC_data_list = []

    for animal in residual_activity_dict_EC:
        for cell in residual_activity_dict_EC[animal]:
            EC_truncated = residual_activity_dict_EC[animal][cell][:,:min_val]
            EC_data_list.append(EC_truncated.flatten())

    EC_data_array = np.array(EC_data_list)

    return EC_data_array

def get_CA3_data_array(ca3_vs_position_all_cells_array):
    num_trials = 58
    CA3_data_list = []

    for i in range(ca3_vs_position_all_cells_array.shape[0]):
        data = ca3_vs_position_all_cells_array[i, :] 
        
        tiled_data = np.tile(data[:, np.newaxis], (1, num_trials))
        
    #     plt.imshow(tiled_data.T, aspect='auto')
    #     plt.show()
        
        CA3_data_list.append(tiled_data.flatten())
        
    CA3_data_array = np.array(CA3_data_list)

    return CA3_data_array

def get_SST_data_array(activity_dict_SST, residual_activity_dict_SST):

    # min_val = 1000
    # for animal in activity_dict_SST:
    #     for cell in activity_dict_SST[animal]:
    #         if activity_dict_SST[animal][cell].shape[1] < min_val:
    #             min_val = activity_dict_SST[animal][cell].shape[1] 

    SST_data_list = []

    for animal in residual_activity_dict_SST:
        for cell in residual_activity_dict_SST[animal]:
            SST_data_list.append(residual_activity_dict_SST[animal][cell][:,:58].flatten())
            
    SST_data_array = np.array(SST_data_list)
    return SST_data_array

def get_MSE_cell_type(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array):
    MSE_list = []
    coefficients_list = []
    for animal in model_EC_just_SST:
        for cell in model_EC_just_SST[animal]:
            model = model_EC_just_SST[animal][cell]
            neuron_activity = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,:58]
            coefficients = model.coef_
            coefficients_list.append(coefficients)
            neuron_predicted_activity = model.predict(SST_data_array.T)
            predicted_NDNF = neuron_predicted_activity.reshape(neuron_activity.shape)
            MSE = np.mean(np.square(predicted_NDNF - neuron_activity))
            MSE_list.append(MSE) 
            
    return MSE_list, coefficients_list

def plot_MSE_for_NDNF_pred_by_other_celltype_input(MSE_list_just_SST, MSE_list_just_EC, MSE_list_just_CA3, MSE_list_EC_CA3, MSE_list_EC_SST, MSE_list_CA3_SST, MSE_list_EC_CA3_SST, inputs_list, output_cell_type="NDNF Output"):
    df = pd.DataFrame({
        "MSE": (
            MSE_list_just_SST +
            MSE_list_just_EC +
            MSE_list_just_CA3 +
            MSE_list_EC_CA3 +
            MSE_list_EC_SST +
            MSE_list_CA3_SST +
            MSE_list_EC_CA3_SST
        ),
        "Input": (
            [inputs_list[0]] * len(MSE_list_just_SST) +
            [inputs_list[1]] * len(MSE_list_just_EC) +
            [inputs_list[2]] * len(MSE_list_just_CA3) +
            [inputs_list[3]] * len(MSE_list_EC_CA3) +
            [inputs_list[4]] * len(MSE_list_EC_SST) +
            [inputs_list[5]] * len(MSE_list_CA3_SST) +
            [inputs_list[6]] * len(MSE_list_EC_CA3_SST)
        )
    })

    # Step 2: Define custom color palette
    custom_palette = {
        inputs_list[0]: "blue",
        inputs_list[1]: "green",
        inputs_list[2]: "red",
        inputs_list[3]: "yellow",
        inputs_list[4]: "cyan",
        inputs_list[5]: "purple",
        inputs_list[6]: "orange"
    }

    # Step 3: Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Input", y="MSE", data=df, inner="box", cut=0, palette=custom_palette)
    plt.title(f"{output_cell_type} Cell MSE by Input Cell Type")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Input Type")
    plt.xticks(rotation=20)
    plt.ylim(0,1.75)
    plt.tight_layout()
    plt.show()

def get_model_dict(EC_data_array, fixed_residual_activity_dict_NDNF_newest, reg_type="ridge"):
    model_EC = {}
    for animal in fixed_residual_activity_dict_NDNF_newest:
        cell_model_EC = {}
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            neuron_activity = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,:58]
            neuron_activity_flat = neuron_activity.flatten()
            model_NDNF = fit_GLM2(EC_data_array, neuron_activity_flat, regression=reg_type, alphas=None)
            cell_model_EC[cell] = model_NDNF
        model_EC[animal] = cell_model_EC
        
    return model_EC

def get_data_array_learning_split(activity_dict_EC):

    data_list = []
    for animal in activity_dict_EC:
        for cell in activity_dict_EC[animal]:
            data_list.append(activity_dict_EC[animal][cell].shape[1]//5)


    mean_quint = np.mean(data_list)
    mean_quint_int = int(mean_quint)

    early_cp_dict={}
    late_cp_dict={}

    data_early_list = []
    data_late_list = []

    for animal in activity_dict_EC:
        early_cell_cp_dict={}
        late_cell_cp_dict={}
        for cell in activity_dict_EC[animal]:
            data = activity_dict_EC[animal][cell]
            data_early = data[:,:mean_quint_int]
            data_early_list.append(data_early)
            data_late = data[:,-mean_quint_int:]
            data_late_list.append(data_late)

            early_cell_cp_dict[cell] = data_early
            late_cell_cp_dict[cell] = data_late

        early_cp_dict[animal] = early_cell_cp_dict
        late_cp_dict[animal] = late_cell_cp_dict

    data_early_array = np.array(data_early_list)
    data_late_array = np.array(data_late_list)

    return data_early_array, data_late_array


def get_MSE_cell_type2(model_EC_dict_early, fixed_activity_dict_NDNF_newest, data_late_array_EC, predict_early_or_late="early"):

    """"
    inputs - the models that were trained on early data from EC and is trained to predict early NDNF, testing on the neuron activity of EC late which will give a prediction of NDNF late
    input order - the models that were trained just on the  
    """

    MSE_list = []
    coefficients_list = []
    for animal in model_EC_dict_early:
        for cell in model_EC_dict_early[animal]:
            model = model_EC_dict_early[animal][cell]
            index = data_late_array_EC.shape[2]
            if predict_early_or_late == "early":
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,:index]
            else:
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,-index:]
            coefficients = model.coef_
            coefficients_list.append(coefficients)
            data_late_array_EC_flat = data_late_array_EC.reshape(data_late_array_EC.shape[0], -1)
            neuron_predicted_activity = model.predict(data_late_array_EC_flat.T)
            predicted_NDNF = neuron_predicted_activity.reshape(neuron_activity.shape)
            MSE = np.mean(np.square(predicted_NDNF - neuron_activity))
            MSE_list.append(MSE) 
            
    return MSE_list, coefficients_list

def get_model_dict_early_celltype_late_another_celltype(EC_data_array, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early"):

    """"
    train on the set of all possible inputs early in learn (first few trials for them) then test the output cell population (NDNF)
    """

    model_EC_dict = {}
    for animal in fixed_activity_dict_NDNF_newest:
        cell_model_EC = {}
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            num_trials = EC_data_array[0].shape[1]

            data_early_array_EC_flat = EC_data_array.reshape(EC_data_array.shape[0], -1)
            
            if early_or_late == "early":
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,:num_trials]
            elif early_or_late == "late":
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,-num_trials:]


            neuron_activity_flat = neuron_activity.flatten()
            model_EC = fit_GLM2(data_early_array_EC_flat, neuron_activity_flat, regression=reg_type, alphas=None)
            cell_model_EC[cell] = model_EC
        model_EC_dict[animal] = cell_model_EC
        
    return model_EC_dict

def plot_early_late_split_all_input_types(activity_dict_SST, activity_dict_EC, fixed_activity_dict_NDNF_newest, ca3_vs_position_all_cells_array, cp_dict_EC, cp_dict_SST, cp_dict_NDNF, ymax=1.55):


    SST_data_array_late_TA = get_activity_late(activity_dict_SST, cp_dict_SST)
    SST_data_array_early_TA = get_activity_early(activity_dict_SST, cp_dict_SST)

    EC_data_array_late_TA = get_activity_late(activity_dict_EC, cp_dict_EC)
    EC_data_array_early_TA = get_activity_early(activity_dict_EC, cp_dict_EC)

    NDNF_data_array_late_TA = get_activity_late(fixed_activity_dict_NDNF_newest, cp_dict_NDNF)
    NDNF_data_array_early_TA = get_activity_early(fixed_activity_dict_NDNF_newest, cp_dict_NDNF)

    EC_CA3_data_array_TA_early = np.concatenate([EC_data_array_early_TA, ca3_vs_position_all_cells_array], axis=0)
    EC_CA3_data_array_TA_late = np.concatenate([EC_data_array_late_TA, ca3_vs_position_all_cells_array], axis=0)

    EC_SST_data_array_TA_early = np.concatenate([EC_data_array_early_TA, SST_data_array_early_TA], axis=0)
    EC_SST_data_array_TA_late = np.concatenate([EC_data_array_late_TA, SST_data_array_late_TA], axis=0)

    CA3_SST_data_array_TA_early = np.concatenate([ca3_vs_position_all_cells_array, SST_data_array_early_TA], axis=0)
    CA3_SST_data_array_TA_late = np.concatenate([ca3_vs_position_all_cells_array, SST_data_array_late_TA], axis=0)



    model_EC_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(EC_data_array_early_TA, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_EC_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_data_array_late_TA)
    MSE_list_EC_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_data_array_early_TA)



    model_SST_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(SST_data_array_early_TA, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_SST_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, SST_data_array_late_TA)
    MSE_list_SST_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, SST_data_array_early_TA)



    model_CA3_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(ca3_vs_position_all_cells_array, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_CA3_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_CA3_dict_early_TA, fixed_activity_dict_NDNF_newest, ca3_vs_position_all_cells_array)
    MSE_list_CA3_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_CA3_dict_early_TA, fixed_activity_dict_NDNF_newest, ca3_vs_position_all_cells_array)



    model_EC_CA3_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(EC_CA3_data_array_TA_early, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_EC_CA3_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_CA3_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_CA3_data_array_TA_late)
    MSE_list_EC_CA3_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_CA3_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_CA3_data_array_TA_early)


    model_EC_SST_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(EC_SST_data_array_TA_early, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_EC_SST_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_SST_data_array_TA_late)
    MSE_list_EC_SST_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_EC_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, EC_SST_data_array_TA_early)


    model_CA3_SST_dict_early_TA = get_model_dict_early_celltype_late_another_celltype_TA(CA3_SST_data_array_TA_early, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=0, end=20)
    MSE_list_CA3_SST_early_predicting_NDNF_late_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_CA3_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, CA3_SST_data_array_TA_late)
    MSE_list_CA3_SST_early_predicting_NDNF_early_TA, coefficients_list_EC_early = get_MSE_cell_type_TA(model_CA3_SST_dict_early_TA, fixed_activity_dict_NDNF_newest, CA3_SST_data_array_TA_early)



    data_list = [MSE_list_EC_early_predicting_NDNF_late_TA, MSE_list_EC_early_predicting_NDNF_early_TA, MSE_list_SST_early_predicting_NDNF_late_TA, MSE_list_SST_early_predicting_NDNF_early_TA, MSE_list_SST_early_predicting_NDNF_late_TA, MSE_list_SST_early_predicting_NDNF_early_TA, MSE_list_CA3_early_predicting_NDNF_late_TA, MSE_list_CA3_early_predicting_NDNF_early_TA, MSE_list_EC_CA3_early_predicting_NDNF_late_TA, MSE_list_EC_CA3_early_predicting_NDNF_early_TA, MSE_list_EC_SST_early_predicting_NDNF_late_TA, MSE_list_EC_SST_early_predicting_NDNF_early_TA, MSE_list_CA3_SST_early_predicting_NDNF_late_TA, MSE_list_CA3_SST_early_predicting_NDNF_early_TA]
    input_titles = ["EC_early:NDNF_late", "EC_early:NDNF_early", "SST_early:NDNF_late", "SST_early:NDNF_early", "SST_early:NDNF_late", "SST_early:NDNF_early", "CA3_early:NDNF_late", "CA3_early:NDNF_early", "EC_CA3_early:NDNF_late", "EC_CA3_early:NDNF_early", "EC_SST_early:NDNF_late", "EC_SST_early:NDNF_early", "CA3_SST_early:NDNF_late", "CA3_SST_early:NDNF_early"]


    reordered_data_list, reordered_input_titles = reorder_early_late_pairs(data_list, input_titles)

    plot_models_trained_early_late(reordered_data_list, reordered_input_titles, ymax=ymax, title="Train Timepoint : Test Timepoint)")

def plot_models_trained_early_late(data_list, input_titles, title="Model MSEs", ylabel="MSE"):
    assert len(data_list) == len(input_titles), "Mismatch: data_list and input_titles must be same length"

    # Flatten MSE values and their labels
    mse_values = [val for sublist in data_list for val in sublist]
    mse_labels = [label for label, sublist in zip(input_titles, data_list) for _ in sublist]

    # Create DataFrame
    df = pd.DataFrame({"MSE": mse_values, "Input": mse_labels})

    plt.figure(figsize=(10, 6))

    # Plot individual data points in black
    sns.stripplot(x="Input", y="MSE", data=df, 
                jitter=True, color="black", alpha=0.6, zorder=1)

    # Compute mean and SEM
    group_stats = df.groupby("Input")["MSE"].agg(["mean", "sem"]).reset_index()

    # Determine category order as used by seaborn
    input_order = df["Input"].unique()
    x_positions = np.arange(len(input_order))

    # Plot red diamonds with error bars
    for i, input_type in enumerate(input_order):
        mean = group_stats.loc[group_stats["Input"] == input_type, "mean"].values[0]
        sem = group_stats.loc[group_stats["Input"] == input_type, "sem"].values[0]
        plt.errorbar(i, mean, yerr=sem, fmt='D', color='red', 
                    capsize=5, markersize=8, elinewidth=2, label="Mean ± SEM" if i == 0 else "", zorder=2)

    # Final polish
    plt.xticks(ticks=x_positions, labels=input_order, rotation=20, fontsize=7)
    plt.xlabel("Input Type")
    plt.ylabel("Mean Squared Error")
    # plt.title("Train on Early, Test on Late (Same Cell Type)")
    plt.ylim(0, 3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_model_dict_early_celltype_late_another_celltype_TA(EC_data_array, fixed_activity_dict_NDNF_newest, reg_type="ridge", early_or_late="early", start=20, end=30):

    """"
    train on the set of all possible inputs early in learn (first few trials for them) then test the output cell population (NDNF)
    """

    model_EC_dict = {}
    for animal in fixed_activity_dict_NDNF_newest:
        cell_model_EC = {}
        for cell in fixed_activity_dict_NDNF_newest[animal]:
            if early_or_late == "early":
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,:end]
            elif early_or_late == "late":
                neuron_activity = fixed_activity_dict_NDNF_newest[animal][cell][:,start:end]

            mean_neuron_activity = np.mean(neuron_activity, axis=1)

            model_EC = fit_GLM2(EC_data_array, mean_neuron_activity, regression=reg_type, alphas=None)
            cell_model_EC[cell] = model_EC
        model_EC_dict[animal] = cell_model_EC
        
    return model_EC_dict

def get_MSE_cell_type_TA(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array):
    MSE_list = []
    coefficients_list = []
    for animal in model_EC_just_SST:
        for cell in model_EC_just_SST[animal]:
            model = model_EC_just_SST[animal][cell]
            neuron_activity = np.mean(fixed_residual_activity_dict_NDNF_newest[animal][cell], axis=1)
            # print(f"neuron_activity.shape {neuron_activity.shape}")
            coefficients = model.coef_
            coefficients_list.append(coefficients)
            neuron_predicted_activity = model.predict(SST_data_array.T)
            # print(f"neuron_predicted_activity {neuron_predicted_activity.shape}")
            # predicted_NDNF = neuron_predicted_activity.reshape(neuron_activity.shape)
            MSE = np.mean(np.square(neuron_predicted_activity - neuron_activity))
            MSE_list.append(MSE) 
            
    return MSE_list, coefficients_list

def plot_models_trained_early_late(data_list, input_titles, ymax=3, title="Labels=Train_timepoint:Test_timepoint)"):
    assert len(data_list) == len(input_titles), "Mismatch: data_list and input_titles must be same length"

    # Flatten MSE values and their labels
    mse_values = [val for sublist in data_list for val in sublist]
    mse_labels = [label for label, sublist in zip(input_titles, data_list) for _ in sublist]

    # Create DataFrame
    df = pd.DataFrame({"MSE": mse_values, "Input": mse_labels})

    plt.figure(figsize=(10, 6))

    # Plot individual data points in black
    sns.stripplot(x="Input", y="MSE", data=df, 
                jitter=True, color="black", alpha=0.6, zorder=1)

    # Compute mean and SEM
    group_stats = df.groupby("Input")["MSE"].agg(["mean", "sem"]).reset_index()

    # Determine category order as used by seaborn
    input_order = df["Input"].unique()
    x_positions = np.arange(len(input_order))

    # Plot red diamonds with error bars
    for i, input_type in enumerate(input_order):
        mean = group_stats.loc[group_stats["Input"] == input_type, "mean"].values[0]
        sem = group_stats.loc[group_stats["Input"] == input_type, "sem"].values[0]
        plt.errorbar(i, mean, yerr=sem, fmt='D', color='red', 
                    capsize=5, markersize=8, elinewidth=2, label="Mean ± SEM" if i == 0 else "", zorder=2)

    # Final polish
    plt.xticks(ticks=x_positions, labels=input_order, rotation=20, fontsize=7)
    plt.xlabel(title)
    plt.ylabel("Mean Squared Error")
    # plt.title(title)
    plt.ylim(0, ymax)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_fixed_model_dict_NDNF_newest(cell_NDNF_model_ranks20_contig_x00):
    fixed_model_dict_NDNF_newest = {20:{}}
    for animal in cell_NDNF_model_ranks20_contig_x00[20]:
        if 17 < animal < 31:
            fixed_model_dict_NDNF_newest[20][animal-18] = cell_NDNF_model_ranks20_contig_x00[20][animal]
    return fixed_model_dict_NDNF_newest

def reorder_early_late_pairs(data_list, input_titles):
    assert len(data_list) == len(input_titles), "Mismatch in data and title lengths"

    paired_data = list(zip(input_titles, data_list))

    # Group into consecutive pairs
    reordered_titles = []
    reordered_data = []

    for i in range(0, len(paired_data), 2):
        pair = paired_data[i:i+2]

        # If both titles are in the pair
        if len(pair) == 2:
            t1, d1 = pair[0]
            t2, d2 = pair[1]

            # Put the "early" one first
            if "early" in t1 and "late" in t2:
                reordered_titles.extend([t1, t2])
                reordered_data.extend([d1, d2])
            elif "late" in t1 and "early" in t2:
                reordered_titles.extend([t2, t1])
                reordered_data.extend([d2, d1])
            else:
                # If neither or both have same type, keep as-is
                reordered_titles.extend([t1, t2])
                reordered_data.extend([d1, d2])
        else:
            # Just append the last one if it's an odd-length list
            reordered_titles.append(pair[0][0])
            reordered_data.append(pair[0][1])

    return reordered_data, reordered_titles

def get_activity_av_all_trials(activity_dict_SST):
    SST_data_list = []
    for animal in activity_dict_SST:
        for cell in activity_dict_SST[animal]:
            # SST_data_list.append(activity_dict_SST[animal][cell][:,:end])
            SST_data_list.append(np.mean(activity_dict_SST[animal][cell], axis=1))
    SST_data_array_early = np.array(SST_data_list)
    return SST_data_array_early

def get_cp_dict(cell_SST_model_ranks20_contig_x00):
    changepoints_dict = {}
    for animal in cell_SST_model_ranks20_contig_x00[20]:
        changepoints_cell_dict = {}
        for cell in cell_SST_model_ranks20_contig_x00[20][animal]:
            labels = cell_SST_model_ranks20_contig_x00[20][animal][cell][1][f'cell_{cell}']["labels_dict"]["clusters_chosen_3"]
            changepoints = np.where(np.diff(labels) != 0)[0]
            changepoints_cell_dict[cell] = changepoints
        changepoints_dict[animal] = changepoints_cell_dict

    return changepoints_dict

def get_activity_early(activity_dict_SST, cp_dict_SST):
    SST_data_list = []
    for idx, animal in enumerate(activity_dict_SST):
        for idt, cell in enumerate(activity_dict_SST[animal]):
            cp_early = cp_dict_SST[idx][idt][0]
            SST_data_list.append(np.mean(activity_dict_SST[animal][cell][:,:cp_early], axis=1))
            # SST_data_list.append(np.mean(activity_dict_SST[animal][cell], axis=1))

    SST_data_array_early = np.array(SST_data_list)
    return SST_data_array_early

def get_activity_late(activity_dict_SST, cp_dict_SST):
    SST_data_list = []
    for idx, animal in enumerate(activity_dict_SST):
        for idt, cell in enumerate(activity_dict_SST[animal]):
            cp_late = cp_dict_SST[idx][idt][1]
            SST_data_list.append(np.mean(activity_dict_SST[animal][cell][:,-cp_late:], axis=1))
            # SST_data_list.append(np.mean(activity_dict_SST[animal][cell], axis=1))

    SST_data_array_late = np.array(SST_data_list)
    return SST_data_array_late

def get_MSE_lists(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before):
  indices_array = list(np.arange(4))
  MSE_list_all = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  indices_array = list(indices_array[1:])
  MSE_list_minusV = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  indices_array = [0,2,3,4]
  MSE_list_minusL = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  indices_array = [0,1,3,4]
  MSE_list_minusC = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  indices_array = [0,1,2,4]
  MSE_list_minusR = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  indices_array = list(np.arange(3))
  MSE_list_minusT = remove_behaviors_GLM(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before, indices_array)

  return MSE_list_all, MSE_list_minusV, MSE_list_minusL, MSE_list_minusC, MSE_list_minusR, MSE_list_minusT

def remove_behaviors_GLM(activity_dict_NDNF, NDNF_GLM_models, design_matrix_dict_NDNF, indices_array):
  MSE_list = []
  prediction_list = []
  weights_list = []

  for animal in NDNF_GLM_models:
    for cell in NDNF_GLM_models[animal]:
    
        neuron_data = activity_dict_NDNF[animal][cell]
        weights = NDNF_GLM_models[animal][cell].coef_
        intercept = NDNF_GLM_models[animal][cell].intercept_

        behavioral_correlates_before_slice = design_matrix_dict_NDNF[animal][cell][indices_array,:]

        prediction = (weights[indices_array] @ behavioral_correlates_before_slice) + intercept
        prediction_reshaped = prediction.reshape(neuron_data.shape)
        prediction_list.append(prediction_reshaped)

        mean_neuron_data = np.mean(neuron_data, axis=1)
        mean_prediction_data = np.mean(prediction_reshaped, axis=1)

        MSE = np.mean(np.square(mean_neuron_data - mean_prediction_data))
        MSE_list.append(MSE)

        weights_list.append(weights)

  return MSE_list, prediction_list, weights_list




def get_model_dict_split(EC_data_array, fixed_residual_activity_dict_NDNF_newest, start=20, end=30, reg_type="ridge", early=True):
    model_EC = {}
    for animal in fixed_residual_activity_dict_NDNF_newest:
        cell_model_EC = {}
        for cell in fixed_residual_activity_dict_NDNF_newest[animal]:
            # if early:
            #     neuron_activity = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,:end]
            # else:
            #     neuron_activity = fixed_residual_activity_dict_NDNF_newest[animal][cell][:,start:end]

            neuron_activity = fixed_residual_activity_dict_NDNF_newest[animal][cell]
            mean_neuron_activity = np.mean(neuron_activity, axis=1)
            neuron_activity_flat = mean_neuron_activity.flatten()

            print(EC_data_array.shape)
            print(neuron_activity_flat.shape)

            # EC_data_array_flat = EC_data_array.flatten()
            model_NDNF = fit_GLM2(EC_data_array, neuron_activity_flat, regression=reg_type, alphas=None)
            cell_model_EC[cell] = model_NDNF
        model_EC[animal] = cell_model_EC
        
    return model_EC

def get_MSE_cell_type_TA(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array):
    MSE_list = []
    coefficients_list = []
    for animal in model_EC_just_SST:
        for cell in model_EC_just_SST[animal]:
            model = model_EC_just_SST[animal][cell]
            neuron_activity = np.mean(fixed_residual_activity_dict_NDNF_newest[animal][cell], axis=1)
            # print(f"neuron_activity.shape {neuron_activity.shape}")
            coefficients = model.coef_
            coefficients_list.append(coefficients)
            neuron_predicted_activity = model.predict(SST_data_array.T)
            # print(f"neuron_predicted_activity {neuron_predicted_activity.shape}")
            # predicted_NDNF = neuron_predicted_activity.reshape(neuron_activity.shape)
            MSE = np.mean(np.square(neuron_predicted_activity - neuron_activity))
            MSE_list.append(MSE) 
            
    return MSE_list, coefficients_list

def plot_MSE_for_NDNF_pred_by_other_celltype_input_random(MSE_list_just_SST, MSE_list_just_EC, MSE_list_just_CA3, MSE_list_EC_CA3, MSE_list_EC_SST, MSE_list_CA3_SST, MSE_list_EC_CA3_SST, MSE_list_random, inputs_list, output_cell_type="NDNF Output", ymax=0.03):
    df = pd.DataFrame({
        "MSE": (
            MSE_list_just_SST +
            MSE_list_just_EC +
            MSE_list_just_CA3 +
            MSE_list_EC_CA3 +
            MSE_list_EC_SST +
            MSE_list_CA3_SST +
            MSE_list_EC_CA3_SST + 
            MSE_list_random
        ),
        "Input": (
            [inputs_list[0]] * len(MSE_list_just_SST) +
            [inputs_list[1]] * len(MSE_list_just_EC) +
            [inputs_list[2]] * len(MSE_list_just_CA3) +
            [inputs_list[3]] * len(MSE_list_EC_CA3) +
            [inputs_list[4]] * len(MSE_list_EC_SST) +
            [inputs_list[5]] * len(MSE_list_CA3_SST) +
            [inputs_list[6]] * len(MSE_list_EC_CA3_SST) +
            [inputs_list[7]] * len(MSE_list_random)
        )
    })

    # Step 2: Define custom color palette
    custom_palette = {
        inputs_list[0]: "blue",
        inputs_list[1]: "green",
        inputs_list[2]: "red",
        inputs_list[3]: "yellow",
        inputs_list[4]: "cyan",
        inputs_list[5]: "purple",
        inputs_list[6]: "orange",
        inputs_list[7]: "cyan"
    }

    # Step 3: Plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Input", y="MSE", data=df, inner="box", cut=0, palette=custom_palette)
    plt.title(f"{output_cell_type} Cell MSE by Input Cell Type")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Input Type")
    plt.xticks(rotation=20)
    plt.ylim(0,ymax)
    plt.tight_layout()
    plt.show()

def plot_NDNF_prediction_from_other_celltypes_input_GLM(activity_dict_SST, activity_dict_EC, fixed_activity_dict_NDNF_newest, ymax=0.10):
    # SST_data_array_late = get_activity_late(activity_dict_SST, start=40, end=60)
    SST_data_array_TA = get_activity_av_all_trials(activity_dict_SST)

    # EC_data_array_late = get_activity_late(activity_dict_EC, start=30, end=50)
    EC_data_array_TA = get_activity_av_all_trials(activity_dict_EC)

    # NDNF_data_array_late = get_activity_late(fixed_activity_dict_NDNF_newest, start=40, end=60)
    NDNF_data_array_TA = get_activity_av_all_trials(fixed_activity_dict_NDNF_newest)

    EC_CA3_data_array_TA = np.concatenate([EC_data_array_TA, ca3_vs_position_all_cells_array], axis=0)
    EC_SST_data_array_TA = np.concatenate([EC_data_array_TA, SST_data_array_TA], axis=0)
    CA3_SST_data_array_TA = np.concatenate([ca3_vs_position_all_cells_array, SST_data_array_TA], axis=0)
    EC_CA3_SST_data_array_TA = np.concatenate([EC_data_array_TA, ca3_vs_position_all_cells_array, SST_data_array_TA], axis=0)

    rand_tens = np.random.rand(792, 58, 50)
    rand_tens_TA = np.mean(rand_tens, axis=1)

    model_just_CA3 = get_model_dict_split(ca3_vs_position_all_cells_array, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)
    model_just_SST = get_model_dict_split(SST_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)
    model_just_EC = get_model_dict_split(EC_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)

    model_EC_CA3 = get_model_dict_split(EC_CA3_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)
    model_EC_SST = get_model_dict_split(EC_SST_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)
    model_CA3_SST = get_model_dict_split(CA3_SST_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)
    model_EC_CA3_SST = get_model_dict_split(EC_CA3_SST_data_array_TA, fixed_residual_activity_dict_NDNF_newest, start=20, end=58, reg_type="ridge", early=True)

    model_random = get_model_dict_split(rand_tens_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)

    MSE_list_just_SST, coefficients_list_just_SST = get_MSE_cell_type_TA(model_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array_TA)
    MSE_list_just_EC, coefficients_list_just_EC = get_MSE_cell_type_TA(model_just_EC, fixed_residual_activity_dict_NDNF_newest, EC_data_array_TA)
    MSE_list_just_CA3, coefficients_list_just_CA3 = get_MSE_cell_type_TA(model_just_CA3, fixed_residual_activity_dict_NDNF_newest, ca3_vs_position_all_cells_array)

    MSE_list_EC_CA3, coefficients_list_EC_CA3 = get_MSE_cell_type_TA(model_EC_CA3, fixed_residual_activity_dict_NDNF_newest, EC_CA3_data_array_TA)
    MSE_list_EC_SST, coefficients_list_EC_SST = get_MSE_cell_type_TA(model_EC_SST, fixed_residual_activity_dict_NDNF_newest, EC_SST_data_array_TA)
    MSE_list_CA3_SST, coefficients_list_CA3_SST = get_MSE_cell_type_TA(model_CA3_SST, fixed_residual_activity_dict_NDNF_newest, CA3_SST_data_array_TA)
    MSE_list_EC_CA3_SST, coefficients_list_CA3_SST = get_MSE_cell_type_TA(model_EC_CA3_SST, fixed_residual_activity_dict_NDNF_newest, EC_CA3_SST_data_array_TA)
    MSE_list_random_TA, coefficients_list_random = get_MSE_cell_type_TA(model_random, activity_dict_SST, rand_tens_TA)

    inputs_list = ["SST", "EC", "CA3", "EC + CA3", "EC + SST", "CA3 + SST", "EC + CA3 + SST", "Random Control"]

    plot_MSE_for_NDNF_pred_by_other_celltype_input_random(MSE_list_just_SST, MSE_list_just_EC, MSE_list_just_CA3, MSE_list_EC_CA3, MSE_list_EC_SST, MSE_list_CA3_SST, MSE_list_EC_CA3_SST, MSE_list_random_TA, inputs_list, output_cell_type="NDNF Output Trial Averaged Input", ymax=ymax)

def plot_SST_prediction_from_other_celltypes_input_GLM(activity_dict_SST, activity_dict_EC, fixed_activity_dict_NDNF_newest, ymax=0.10):

    # SST_data_array_late = get_activity_late(activity_dict_SST, start=40, end=60)
    SST_data_array_TA = get_activity_av_all_trials(activity_dict_SST)

    # EC_data_array_late = get_activity_late(activity_dict_EC, start=30, end=50)
    EC_data_array_TA = get_activity_av_all_trials(activity_dict_EC)

    # NDNF_data_array_late = get_activity_late(fixed_activity_dict_NDNF_newest, start=40, end=60)
    NDNF_data_array_TA = get_activity_av_all_trials(fixed_activity_dict_NDNF_newest)

    EC_CA3_data_array_TA = np.concatenate([EC_data_array_TA, ca3_vs_position_all_cells_array], axis=0)
    EC_NDNF_data_array_TA = np.concatenate([EC_data_array_TA, NDNF_data_array_TA], axis=0)
    CA3_NDNF_data_array_TA = np.concatenate([ca3_vs_position_all_cells_array, NDNF_data_array_TA], axis=0)
    EC_CA3_NDNF_data_array_TA = np.concatenate([EC_data_array_TA, ca3_vs_position_all_cells_array, NDNF_data_array_TA], axis=0)

    np.random.seed(42)
    rand_tens = np.random.rand(792, 58, 50)
    rand_tens_TA = np.mean(rand_tens, axis=1)

    ######### models trained on the trial averaged data 
    model_just_CA3_TA = get_model_dict_split(ca3_vs_position_all_cells_array, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_just_NDNF_TA = get_model_dict_split(NDNF_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_just_EC_TA = get_model_dict_split(EC_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)

    model_EC_CA3_TA = get_model_dict_split(EC_CA3_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_EC_NDNF_TA = get_model_dict_split(EC_NDNF_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_CA3_NDNF_TA = get_model_dict_split(CA3_NDNF_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_EC_CA3_NDNF_TA = get_model_dict_split(EC_CA3_NDNF_data_array_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)
    model_random = get_model_dict_split(rand_tens_TA, activity_dict_SST, start=20, end=58, reg_type="ridge", early=True)

    MSE_list_just_NDNF_TA, coefficients_list_just_NDNF = get_MSE_cell_type_TA(model_just_NDNF_TA, activity_dict_SST, NDNF_data_array_TA)
    MSE_list_just_EC_TA, coefficients_list_just_EC = get_MSE_cell_type_TA(model_just_EC_TA, activity_dict_SST, EC_data_array_TA)
    MSE_list_just_CA3_TA, coefficients_list_just_CA3 = get_MSE_cell_type_TA(model_just_CA3_TA, activity_dict_SST, ca3_vs_position_all_cells_array)

    MSE_list_EC_CA3_TA, coefficients_list_EC_CA3 = get_MSE_cell_type_TA(model_EC_CA3_TA, activity_dict_SST, EC_CA3_data_array_TA)
    MSE_list_EC_NDNF_TA, coefficients_list_EC_NDNF = get_MSE_cell_type_TA(model_EC_NDNF_TA, activity_dict_SST, EC_NDNF_data_array_TA)
    MSE_list_CA3_NDNF_TA, coefficients_list_CA3_NDNF = get_MSE_cell_type_TA(model_CA3_NDNF_TA, activity_dict_SST, CA3_NDNF_data_array_TA)
    MSE_list_EC_CA3_NDNF_TA, coefficients_list_CA3_NDNF = get_MSE_cell_type_TA(model_EC_CA3_NDNF_TA, activity_dict_SST, EC_CA3_NDNF_data_array_TA)

    MSE_list_random_TA, coefficients_list_random = get_MSE_cell_type_TA(model_random, activity_dict_SST, rand_tens_TA)


    inputs_list = ["NDNF", "EC", "CA3", "EC + CA3", "EC + NDNF", "CA3 + NDNF", "EC + CA3 + NDNF", "Random Control"]

    plot_MSE_for_NDNF_pred_by_other_celltype_input_random(MSE_list_just_NDNF_TA, MSE_list_just_EC_TA, MSE_list_just_CA3_TA, MSE_list_EC_CA3_TA, MSE_list_EC_NDNF_TA, MSE_list_CA3_NDNF_TA, MSE_list_EC_CA3_NDNF_TA, MSE_list_random_TA, inputs_list, output_cell_type="SST Output Trial Averaged Input", ymax=ymax)

def get_velocities(factors_dict_SST):
    SST_data_list = []
    for animal in factors_dict_SST:
        velocity = factors_dict_SST[animal]["Velocity"]
            # SST_data_list.append(np.mean(activity_dict_SST[animal][cell][:,:end], axis=1))
        SST_data_list.append(np.mean(velocity, axis=1))

    SST_data_array_early = np.array(SST_data_list)
    return SST_data_array_early

def plot_datar(SST_datas, title="Velocity Across All Animals", color='b'):
    SST_data_mean = np.mean(SST_datas, axis=0)
    SST_data_sem = sem(SST_datas, axis=0)  

    for i in range(SST_datas.shape[0]):
        plt.plot(SST_datas[i,:], color='grey', alpha=0.2)

    plt.plot(SST_data_mean, color=color, label='Raw')
    plt.fill_between(range(len(SST_data_mean)), SST_data_mean+SST_data_sem, SST_data_mean - SST_data_sem, alpha=0.2, color=color)

    plt.title(title)
    plt.ylabel("Z-Scored Velocity")
    plt.xlabel("Position Bins")
    # plt.ylim(-0.5,0.5)
    plt.show()

def plot_data(SST_datas, SST_datas_r, title="SST Activity Across All Cells", color='b'):
    SST_data_mean = np.mean(SST_datas, axis=0)
    SST_data_sem = sem(SST_datas, axis=0)  

    SST_data_mean_r = np.mean(SST_datas_r, axis=0)
    SST_data_sem_r = sem(SST_datas_r, axis=0)  

    plt.plot(SST_data_mean, color=color, label='Raw')
    plt.fill_between(range(len(SST_data_mean)), SST_data_mean+SST_data_sem, SST_data_mean - SST_data_sem, alpha=0.2, color=color)
    plt.plot(SST_data_mean_r, color='k', label="Vel. Sub. Residual")
    plt.fill_between(range(len(SST_data_mean_r)), SST_data_mean_r+SST_data_sem_r, SST_data_mean_r - SST_data_sem_r, alpha=0.2, color='k')
    plt.title(title)
    plt.ylabel("Z-Scored DF/F")
    plt.xlabel("Position Bins")
    plt.ylim(-0.5,0.5)
    plt.legend()
    plt.show()

def plot_coefficients_cell_type(weights_list_EC, cell_type="EC"):
    lick_weights = []
    reward_weights = []
    velocity_weights = []

    for i in weights_list_EC:
        lick_weights.append(i[0])
        reward_weights.append(i[1])
        velocity_weights.append(i[2])

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import numpy as np
    # import pandas as pd

    # Combine into a DataFrame for easy plotting
    df = pd.DataFrame({
        'Lick': lick_weights,
        'Reward': reward_weights,
        'Velocity': velocity_weights
    })

    # Melt for long-form plotting
    df_melted = df.melt(var_name='Variable', value_name='Weight')

    # Plot
    plt.figure(figsize=(6, 5))
    # sns.pointplot(data=df_melted, x='Variable', y='Weight', color='red', join=False, errorbar='sd', markers='d', err_kws={'linewidth': 1.5})
    sns.stripplot(data=df_melted, x='Variable', y='Weight', color='black', jitter=True, alpha=0.5)

    offset=0.2

    variable_names = ['Lick', 'Reward', 'Velocity']
    for i, var in enumerate(variable_names):
        values = df[var]
        mean = np.mean(values)
        se = np.std(values) / np.sqrt(len(values))
        plt.errorbar(i + offset, mean, yerr=se, fmt='d', color='red',
                     capsize=5, elinewidth=1.5, markeredgewidth=1.5)

    plt.axhline(0, linestyle='--', color='grey')
    plt.title(f'GLM Coefficients {cell_type}')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.show()

def extract_weight_lists(weights_list):
    lick = [w[0] for w in weights_list]
    reward = [w[1] for w in weights_list]
    velocity = [w[2] for w in weights_list]
    return lick, reward, velocity

def plot_coefficients_all_celltypes_together(weights_list_NDNF, weights_list_SST, weights_list_EC, title="GLM Coefficients Across Cell Types"):
    # Extract weight values per group
    lick_NDNF, reward_NDNF, velocity_NDNF = extract_weight_lists(weights_list_NDNF)
    lick_SST, reward_SST, velocity_SST = extract_weight_lists(weights_list_SST)
    lick_EC, reward_EC, velocity_EC = extract_weight_lists(weights_list_EC)

    # Combine all groups
    data_means = [
        np.mean(lick_NDNF), np.mean(lick_SST), np.mean(lick_EC),
        np.mean(reward_NDNF), np.mean(reward_SST), np.mean(reward_EC),
        np.mean(velocity_NDNF), np.mean(velocity_SST), np.mean(velocity_EC)
    ]

    data_sems = [
        np.std(lick_NDNF)/np.sqrt(len(lick_NDNF)), np.std(lick_SST)/np.sqrt(len(lick_SST)), np.std(lick_EC)/np.sqrt(len(lick_EC)),
        np.std(reward_NDNF)/np.sqrt(len(reward_NDNF)), np.std(reward_SST)/np.sqrt(len(reward_SST)), np.std(reward_EC)/np.sqrt(len(reward_EC)),
        np.std(velocity_NDNF)/np.sqrt(len(velocity_NDNF)), np.std(velocity_SST)/np.sqrt(len(velocity_SST)), np.std(velocity_EC)/np.sqrt(len(velocity_EC)),
    ]

    # Labels
    x_labels = ['Lick\nNDNF', 'Lick\nSST', 'Lick\nEC',
                'Reward\nNDNF', 'Reward\nSST', 'Reward\nEC',
                'Velocity\nNDNF', 'Velocity\nSST', 'Velocity\nEC']

    # Plot
    plt.figure(figsize=(10, 5))
    x = np.arange(len(data_means))
    colors = ['orange', 'blue', 'green'] * 3

    plt.bar(x, data_means, yerr=data_sems, capsize=5, color=colors, edgecolor='black', alpha=0.9)
    plt.xticks(x, x_labels, fontsize=10)
    plt.ylabel('GLM Coefficient (Mean ± SEM)')
    plt.axhline(0, linestyle='--', color='grey')
    plt.title(title)
    plt.tight_layout()
    plt.ylim(-0.6, 0.9)
    plt.show()