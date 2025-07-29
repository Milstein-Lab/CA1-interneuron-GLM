import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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

def plot_models_trained_early_late(data_list, input_titles):
# Create the DataFrame
    df = pd.DataFrame({
        "MSE": (
            data_list[0] +
            data_list[1] +
            data_list[2] +
            data_list[3] +
            data_list[4] +
            data_list[5] +
            data_list[6] +
            data_list[7]
        ),
        "Input": (
            [input_titles[0]] * len(data_list[0]) +
            [input_titles[1]]  * len(data_list[1]) +
            [input_titles[2]]  * len(data_list[2]) +
            [input_titles[3]]  * len(data_list[3]) +
            [input_titles[4]] * len(data_list[4]) +
            [input_titles[5]]  * len(data_list[5]) +
            [input_titles[6]]  * len(data_list[6]) +
            [input_titles[7]]  * len(data_list[7]) 
            )

    })

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
