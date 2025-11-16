
#  Clean_notebooks_to_date % mpiexec -n 6 python -m mpi4py.futures -m nested.optimize \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=mpi \
#   --num-workers=5 \
#   --pop_size=6 --path_length=1 --max_iter=1 \
#   --num_network_seeds=2 \
#   --disp --network_start_seed=0 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True --debug=True

# python -m nested.analyze \
#   --config-file-path=config/ec_nested.yaml \
#   --framework=serial \
#   --num_network_seeds=2 \
#   --disp --network_start_seed=0 --vel_applied='real' --animal_by_animal=False --constant_vel=False --include_beta=False --flat_input=True --debug=True --optimization_time=False --plot --param-file-path=model_key_yaml_opt.yaml --model-key=oct21_justec_5seeds_serial_number1_velTbetaFinputT

from nested.utils import Context, param_array_to_dict

from spiking_model_utils import load_data_regular
from build_a_model_object_per_animal import *


import pickle

context = Context()



def config_worker():

    start = int(context.network_start_seed)
    end   = start + int(context.num_network_seeds)
    seeds_array = list(range(start, end))
    
    residuals_activity_dict = {}
    GLM_params_dict = {}
    behav_factors_dict = {}
    
    GLM_params_SST, _, _, factors_dict_SST, filtered_factors_dict_SST, residual_activity_dict_SST = load_data_regular(
        file_path=context.data_root, name="SSTindivsomata_GLM", new_NDNF=False)
    GLM_params_EC, _, _, factors_dict_EC, filtered_factors_dict_EC, residual_activity_dict_EC = load_data_regular(
        file_path=context.data_root, name="EC_GLM", new_NDNF=False)
    GLM_params_NDNF_newest, _, _, factors_dict_NDNF_newest, filtered_factors_dict_NDNF_newest, residual_activity_dict_NDNF_newest = load_data_regular(
        file_path=context.data_root, name="NDNF_E1A1B", new_NDNF=True)

    fixed_residual_activity_dict_NDNF_newest = {f"animal_{idx+1}": residual_activity_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(residual_activity_dict_NDNF_newest)
                                               if 17 < idx < 31}
    fixed_filtered_factors_dict_NDNF_newest = {f"animal_{idx+1}": filtered_factors_dict_NDNF_newest[animal]
                                               for idx, animal in enumerate(filtered_factors_dict_NDNF_newest)
                                               if 17 < idx < 31}
    
    
    residuals_activity_dict['EC'] = residual_activity_dict_EC
    residuals_activity_dict['SST'] = residual_activity_dict_SST
    residuals_activity_dict['NDNF'] = fixed_residual_activity_dict_NDNF_newest
    
    GLM_params_dict['EC'] = GLM_params_EC
    GLM_params_dict['SST'] = GLM_params_SST
    GLM_params_dict['NDNF'] = GLM_params_NDNF_newest
    
    behav_factors_dict['EC'] = factors_dict_EC
    behav_factors_dict['SST'] = factors_dict_SST
    behav_factors_dict['NDNF'] = factors_dict_NDNF_newest
    
    christine_save_path = (f"{context.data_root}/datasets/christines_overrepresentation_pkl.pkl")
    with open(christine_save_path, 'rb') as f:
        christine_overrepresentation_array = pickle.load(f)

    christine_field_build_path = (f"{context.data_root}/datasets/christines_percent_ca1pc.pkl")
    with open(christine_field_build_path, 'rb') as f:
        christine_field_building_array = pickle.load(f)

    dx = 180. / 50.
    
    context.update(locals())


def get_args():
    seed_list = [context.seeds_array]
    return seed_list


def compute_features(params, *pos, model_id=None, export=False, plot=False, **kw):
    if not pos:
        raise ValueError("No stage args; expected a (seed,) group as first positional.")
    group = pos[0]                         # your arg-group from get_args()
    network_seed = int(group[0]) if isinstance(group, (tuple, list)) else int(group)

    rank = MPI.COMM_WORLD.Get_rank()
    pid  = os.getpid()
    print(f"[rank={rank} pid={pid}] seed={network_seed} extra_pos={len(pos)-1}", flush=True)


    param_dict = param_array_to_dict(params, context.param_names)

    
    constant_vel = str_true_false_to_bool(context.constant_vel)
    include_beta = str_true_false_to_bool(context.include_beta)
    flat_input = str_true_false_to_bool(context.flat_input)
    store_intermediates = str_true_false_to_bool(context.store_intermediates)
    debug = str_true_false_to_bool(context.debug)

    ######## no longer searching weight mean 
    
    if 'EC' in context.weight_config_dict:
        if not all([param_name in param_dict for param_name in ['EC_weights_std']]):
            raise Exception('missing EC weight mean and/or std in param_dict')
        # context.weight_config_dict['EC']['mean'] = param_dict['EC_weights_mean']
        context.weight_config_dict['EC']['std'] = param_dict['EC_weights_std']

    if 'SST' in context.weight_config_dict:
        if not all([param_name in param_dict for param_name in ['SST_sf', 'SST_weights_std']]):
            raise Exception('missing SST weight mean and/or std in param_dict')
        context.weight_config_dict['SST']['sf'] = param_dict['SST_sf']
        context.weight_config_dict['SST']['std'] = param_dict['SST_weights_std']

    if 'NDNF' in context.weight_config_dict:
        if not all([param_name in param_dict for param_name in ['NDNF_sf', 'NDNF_weights_std']]):
            raise Exception('missing NDNF weight mean and/or std in param_dict')
        context.weight_config_dict['NDNF']['sf'] = param_dict['NDNF_sf']
        context.weight_config_dict['NDNF']['std'] = param_dict['NDNF_weights_std']

    
    dt_ms   = context.dt * 1000.0
    AMP     = 1.0
    MODE    = "peak"
    
    kernel  = exp_kernel(param_dict["tau_ms"], dt_ms, n_taus=5, norm=MODE, target=AMP)
    
  
    sim_config = SimConfig(
    dt_constant=getattr(context, "dt", 0.001),
    dx=getattr(context, "dx", None),
    multiple_dendrites=True,
    make_it_spike=True,
    num_dendrites=int(getattr(context, "num_dendrites", 100)),
    seed=int(getattr(context, "network_start_seed", 42)),
    dist="Lognormal")

    model = SpikeSimModel(kernel=kernel, weight_config_dict=context.weight_config_dict, dt=context.dt, track_len=context.track_len,
                        store_intermediates=store_intermediates, plot=plot, debug=debug,
                        residuals_activity_dict=context.residuals_activity_dict,
                        GLM_params_dict=context.GLM_params_dict, behav_factors_dict=context.behav_factors_dict, 
                        constant_vel=constant_vel, include_beta=include_beta, flat_input=flat_input, dend_threshold=param_dict['dend_threshold'], tau_ms=param_dict['tau_ms'], 
                        GLM_param_dict=context.GLM_params_dict, n_cells_dict = context.n_cells_dict, sim_config=sim_config, 
                        christine_overrep_array = context.christine_overrepresentation_array, christine_field_building_array = context.christine_field_building_array)
    


    if context.debug:
        log_mem("B: after building empty model")
    
    t0 = time.time()
    

    if context.debug:
        log_mem("C: after attaching attrs, pre simulate")

    model_histogram, frac_dends_cum, christine_overrepresentation_array_scaled = model.simulate(int(network_seed), debug=context.debug)

    if context.interactive:
    
        model.plot_summary()

    features_dict = {"model_histogram":model_histogram, 
                     "frac_dends_cum":frac_dends_cum,
                     "christine_overrepresentation_array_scaled":christine_overrepresentation_array_scaled}

    return features_dict


def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):

    features = {}

    christine_overrepresentation_array_scaled = features_dict_list[0]["christine_overrepresentation_array_scaled"]

    histogram_list = []
    pc_list = []

    for seed in range(len(features_dict_list)):
        features_dict = features_dict_list[seed]
        seed_model_histogram = features_dict["model_histogram"]
        histogram_list.append(seed_model_histogram)
        pc_list.append(features_dict["frac_dends_cum"])

    histogram_array = np.array(histogram_list)
    mean_histogram_array = np.mean(histogram_array, axis=0)
    sem_histogram_array = sem(histogram_array, axis=0)

    pc_array = np.array(pc_list)
    print(f"pc_array.shape {pc_array.shape}")
    mean_pc_array = np.mean(pc_array, axis=0)
    sem_pc_array = sem(pc_array, axis=0)

    loss = np.mean(np.square(christine_overrepresentation_array_scaled - mean_histogram_array))

    if context.interactive:
        plot_seed_average(context.christine_field_building_array, christine_overrepresentation_array_scaled, mean_histogram_array, sem_histogram_array, mean_pc_array, sem_pc_array, loss)

    

    features['total_error'] = loss

    return features

def get_objectives(features, model_id=None, export=False, plot=False):

    objectives = {}
    objectives['total_error'] = features["total_error"]

    return features, objectives

