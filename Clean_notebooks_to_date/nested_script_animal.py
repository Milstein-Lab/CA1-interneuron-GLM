import sys
from nested.utils import Context, param_array_to_dict, print_param_dict_like_yaml, write_to_yaml
from CRCNS_BTSP.utils import *

context = Context()


def config_worker():
    
    if 'export_network_config_file_path' not in context():
        context.export_network_config_file_path = context.network_config_file_path.rsplit('.', 1)[0]
        context.export_network_config_file_path = context.export_network_config_file_path + '_exported.yaml'
    
    legacy_network = load_network('%s/%s' % (context.output_dir, context.legacy_network_file_name))

    with open(context.network_config_file_path, 'r') as f:
        context.network_config_dict = yaml.safe_load(f)

    targets_from_legacy = {
        "novel_explore_plateaus": legacy_network.get_plateau_counts(context.novel_explore_lap)['plateau_count_outside_reward'],
        "fixed_reward_in_reward_plateaus": legacy_network.get_plateau_counts(context.fixed_reward_lap)['plateau_count_inside_reward'],
        "fixed_reward_out_reward_plateaus": legacy_network.get_plateau_counts(context.fixed_reward_lap)['plateau_count_outside_reward'],
        "familiar_explore_plateaus": legacy_network.get_plateau_counts(context.familiar_explore_lap)['plateau_count_outside_reward']
    }
    
    if 'debug' not in context():
        context.debug = False

    if 'store_history' not in context():
        context.store_history = False

    context.phase_laps = int(context.phase_laps)
    
    if 'debug' not in context():
        context.debug = False

    context.update(locals())


def get_args():
    seed_range = list(range(int(context.network_start_seed), int(context.network_start_seed) +
                            int(context.num_network_seeds)))

    return [seed_range]


def compute_features(params, network_seed, model_id=None, export=False, plot=False):

    store_history = context.store_history
    phase_laps = context.phase_laps
    elife_weights = context.elife_weights
    
    laps_list = get_laps_list(context.phase_laps, novel_explore_start=0, fixed_reward_start=10, familiar_explore_start=20)

    start_time = time.time()
    
    paramsdict = param_array_to_dict(params, context.param_names)

    context.network_config_dict['network_seed'] = network_seed

    modify_network_config(context.network_config_dict, paramsdict)

    with open(context.elife_data_dict, 'rb') as f:
        elife_data = pickle.load(f)

    network = simulate_network_quick(elife_data, context.network_config_dict, laps_list, store_history, elife_weights, plot=plot, debug=context.debug)

    master_plateaus_dict_new = get_plateau_counts_by_phase(network, laps_list)

    target_elife_data_file_path = context.target_elife_data_file_path

    transformed_features = setup_transformed_features(network, master_plateaus_dict_new, elife_data)

    features = setup_transformed_features(network, master_plateaus_dict_new, elife_data)

    if plot:
        # inside_reward_legacy, outside_reward_legacy = get_legacy_plateaus_from_network()

        inside_reward_elife = elife_data['in_reward_plateau_list_elife']
        outside_reward_elife = elife_data['out_reward_plateau_list_elife']

        plot_plateau_counts(master_plateaus_dict_new, inside_reward_elife, outside_reward_elife, laps_list, quick_sim=True)

        if context.plot_intermediates:
            for lap in laps_list:
                print(f"Plotting data for lap {lap}")
                plot_network_state_for_lap_separate(network, lap)

        if context.plot_phase_summary:
            network.plot_network_state()
            data_dir = 'data'
            network.plot_delta_rates(data_dir=data_dir)
    
    if context.disp and context.debug:
        print('compute_features took %.2f s (model_id: %s)' % (time.time() - start_time, model_id))
        print_param_dict_like_yaml(paramsdict)
        print('seed: %i; plateau counts: %s' % (network_seed, str(master_plateaus_dict_new)))
        sys.stdout.flush()
    
    if context.interactive:
        context.update(locals())
    
    if export:
        write_to_yaml(context.export_network_config_file_path, context.network_config_dict, convert_scalars=True)
        print('exported modified network_config to path: %s' % context.export_network_config_file_path)
        sys.stdout.flush()
    
    return features


def filter_features(features_dict_list, previous_features, model_id=None, export=False, plot=False):

    features = {}
    first_feature_dict = features_dict_list[0]
    for key in first_feature_dict:
        val_list = []
        for feature_dict in features_dict_list:
            val_list.append(feature_dict[key])
        features[key] = np.mean(val_list)

    return features


def get_objectives_multi(features, model_id=None, export=False, plot=False):

    new_features = dict()
    objectives = dict()

    for key, value in features.items():
        if 'residual' in key:
            objectives[key] = value
        else:
            new_features[key] = value

    # print(f"new_features {new_features}")
    # print(f"objectives {objectives}")
    #
    # if len(objectives) > 0:
    #     raise ValueError("Objectives were not")

    return new_features, objectives


def get_objectives_single(features, model_id=None, export=False, plot=False):

    features, multi_objectives = get_objectives_multi(features, model_id, export, plot)
    objectives = {}
    objectives['total_error'] = np.sum(list(multi_objectives.values()))

    return features, objectives



# every lap as a target version using elife data new pickle

def setup_transformed_features(new_network, master_plateaus_dict_new, elife_data):
    transformed_features = dict()

    phases = new_network.track_phases
    phase_lap_ranges = dict()
    cumulative_lap = 0

    for phase in phases:
        phase_lap_ranges[phase['label']] = list(np.arange(cumulative_lap, cumulative_lap + phase['num_laps']))
        cumulative_lap += phase['num_laps']

    elife_outside_reward = elife_data['out_reward_plateau_list_elife']
    elife_inside_reward = elife_data['in_reward_plateau_list_elife']

    elife_plateau_dict = dict()
    for lap in range(len(elife_outside_reward)):
        elife_plateau_dict[lap] = dict()
        if lap in phase_lap_ranges['Novel Explore']:
            elife_plateau_dict[lap]['plateau_count_outside_reward'] = elife_outside_reward[lap]
        if lap in phase_lap_ranges['Fixed Reward']:
            elife_plateau_dict[lap]['plateau_count_outside_reward'] = elife_outside_reward[lap]
            elife_plateau_dict[lap]['plateau_count_inside_reward'] = elife_inside_reward[lap]
        if lap in phase_lap_ranges['Familiar Explore']:
            elife_plateau_dict[lap]['plateau_count_outside_reward'] = elife_outside_reward[lap]

    new_inside_reward = []
    new_outside_reward = []

    for key, counts in master_plateaus_dict_new.items():
        if 'inside' in key:
            new_inside_reward.append(counts)
        elif 'outside' in key:
            new_outside_reward.append(counts)

    new_plateau_dict = dict()
    for lap in range(len(elife_outside_reward)):
        new_plateau_dict[lap] = dict()
        if lap in phase_lap_ranges['Novel Explore']:
            new_plateau_dict[lap]['plateau_count_outside_reward'] = new_outside_reward[lap]
        if lap in phase_lap_ranges['Fixed Reward']:
            new_plateau_dict[lap]['plateau_count_outside_reward'] = new_outside_reward[lap]
            new_plateau_dict[lap]['plateau_count_inside_reward'] = new_inside_reward[lap]
        if lap in phase_lap_ranges['Familiar Explore']:
            new_plateau_dict[lap]['plateau_count_outside_reward'] = new_outside_reward[lap]

    novel_explore_list = []
    fixed_reward_in_list = []
    fixed_reward_out_list = []
    familiar_explore_list = []

    for lap, sub_dict in new_plateau_dict.items():
        if lap in phase_lap_ranges['Novel Explore']:
            novel_explore_list.append(sub_dict['plateau_count_outside_reward'])
        elif lap in phase_lap_ranges['Fixed Reward']:
            if 'plateau_count_inside_reward' in sub_dict:
                fixed_reward_in_list.append(sub_dict['plateau_count_inside_reward'])
            if 'plateau_count_outside_reward' in sub_dict:
                fixed_reward_out_list.append(sub_dict['plateau_count_outside_reward'])
        elif lap in phase_lap_ranges['Familiar Explore']:
            familiar_explore_list.append(sub_dict['plateau_count_outside_reward'])


    transformed_features['mean_novel_explore_plateaus_outside_reward'] = np.mean(novel_explore_list)
    transformed_features['mean_fixed_reward_plateaus_inside_reward'] = np.mean(fixed_reward_in_list)
    transformed_features['mean_fixed_reward_plateaus_outside_reward'] = np.mean(fixed_reward_out_list)
    transformed_features['mean_familiar_explore_plateaus_outside_reward'] = np.mean(familiar_explore_list)


    targets_dict = dict()
    for lap in range(len(elife_outside_reward)):
        targets_dict[lap] = dict()
        if lap in phase_lap_ranges['Novel Explore']:
            targets_dict[lap]['plateau_count_outside_reward'] = ((new_plateau_dict[lap][
                                                                      'plateau_count_outside_reward'] -
                                                                  elife_plateau_dict[lap][
                                                                      'plateau_count_outside_reward']) ** 2)
        elif lap in phase_lap_ranges['Fixed Reward']:
            if 'plateau_count_outside_reward' in new_plateau_dict[lap]:
                targets_dict[lap]['plateau_count_outside_reward'] = ((new_plateau_dict[lap][
                                                                          'plateau_count_outside_reward'] -
                                                                      elife_plateau_dict[lap][
                                                                          'plateau_count_outside_reward']) ** 2)
            if 'plateau_count_inside_reward' in new_plateau_dict[lap]:
                targets_dict[lap]['plateau_count_inside_reward'] = ((new_plateau_dict[lap][
                                                                         'plateau_count_inside_reward'] -
                                                                     elife_plateau_dict[lap][
                                                                         'plateau_count_inside_reward']) ** 2)
        elif lap in phase_lap_ranges['Familiar Explore']:
            targets_dict[lap]['plateau_count_outside_reward'] = ((new_plateau_dict[lap][
                                                                      'plateau_count_outside_reward'] -
                                                                  elife_plateau_dict[lap][
                                                                      'plateau_count_outside_reward']) ** 2)

    mean_plateau_residual_list_in_reward = []
    mean_plateau_residual_list_out_reward = []

    #     plateau_count_list = []
    #     for lap, sub_dict in targets_dict.items():
    #         for value in sub_dict.values():
    #             plateau_count_list.append(value)
    #     mean_plateaus = np.mean(plateau_count_list)
    #     print(f"mean_plateaus {mean_plateaus}")

    #     transformed_features[f'plateaus_residuals'] = mean_plateaus

    for lap, sub_dict in targets_dict.items():
        if 'plateau_count_outside_reward' in targets_dict[lap]:
            mean_plateau_residual_list_out_reward.append(targets_dict[lap]['plateau_count_outside_reward'])
        if 'plateau_count_inside_reward' in targets_dict[lap]:
            mean_plateau_residual_list_in_reward.append((targets_dict[lap]['plateau_count_inside_reward']))

    transformed_features[f'plateaus_in_reward_mean_residuals'] = np.mean(mean_plateau_residual_list_in_reward)
    transformed_features[f'plateaus_out_reward_mean_residuals'] = np.mean(mean_plateau_residual_list_out_reward)

    interpolated_data_elife = elife_data['CA1_firing_rate_array']

    interpolated_data_new = get_interpolated_ca1_summed_activity(new_network)

    diff = interpolated_data_new - interpolated_data_elife

    diff_sq = diff ** 2

    sum_sq_diff = np.sum(diff_sq)

    transformed_features[f'CA1_mean_activity_residuals'] = sum_sq_diff

    return transformed_features

