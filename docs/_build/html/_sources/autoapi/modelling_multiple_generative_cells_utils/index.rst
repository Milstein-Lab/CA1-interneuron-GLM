modelling_multiple_generative_cells_utils
=========================================

.. py:module:: modelling_multiple_generative_cells_utils


Functions
---------

.. autoapisummary::

   modelling_multiple_generative_cells_utils.get_means_list
   modelling_multiple_generative_cells_utils.get_clusters_dict
   modelling_multiple_generative_cells_utils.get_indices_TCA
   modelling_multiple_generative_cells_utils.get_indices_dict
   modelling_multiple_generative_cells_utils.get_clusters_dict_field_type
   modelling_multiple_generative_cells_utils.plot_per_cell_clustering_internals_single_cluster
   modelling_multiple_generative_cells_utils.get_count
   modelling_multiple_generative_cells_utils.plot_cluster_peaks
   modelling_multiple_generative_cells_utils.plot_clustered_averages
   modelling_multiple_generative_cells_utils.plot_clustered_averages_deterministic
   modelling_multiple_generative_cells_utils.generate_gaussian
   modelling_multiple_generative_cells_utils.random_timeseries
   modelling_multiple_generative_cells_utils.remove_duplicate_trials
   modelling_multiple_generative_cells_utils.remove_all_duplicates
   modelling_multiple_generative_cells_utils.reconstruct_activity_from_clusters5
   modelling_multiple_generative_cells_utils.get_early_vs_late_activity
   modelling_multiple_generative_cells_utils.min_max_my_data
   modelling_multiple_generative_cells_utils.compare_within_subtype
   modelling_multiple_generative_cells_utils.split_data
   modelling_multiple_generative_cells_utils.plot_clustered_averages2
   modelling_multiple_generative_cells_utils.get_field_type_percents2
   modelling_multiple_generative_cells_utils.get_cells_percents
   modelling_multiple_generative_cells_utils.random_timeseries2
   modelling_multiple_generative_cells_utils.get_means_array
   modelling_multiple_generative_cells_utils.reconstruct_activity_from_clusters_every_cell_diff_latent
   modelling_multiple_generative_cells_utils.get_weighted_activityies_per_cell
   modelling_multiple_generative_cells_utils.plot_cluster_peaks2
   modelling_multiple_generative_cells_utils.reconstruct_activity_from_clusters_every_cell_diff_latent_split
   modelling_multiple_generative_cells_utils.plot_generative_model
   modelling_multiple_generative_cells_utils.slot_trials_into_spots
   modelling_multiple_generative_cells_utils.get_trial_indices
   modelling_multiple_generative_cells_utils.get_cell_array
   modelling_multiple_generative_cells_utils.randomly_pick_cells


Module Contents
---------------

.. py:function:: get_means_list(residual_activity_dict_EC)

.. py:function:: get_clusters_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln='nothing')

   - takes in the contiguous cutpoints slice TCA model and the K-Means slice TCA model
   - use elbow kmeans to loop through every number of cluster (up to 8) to get the optimal number of clusters via reconstruction MSE of cluster average reconstruction vs real data for the cell
   - returns a dict where every cell's activity is seperated by its cluster via trial indices for each cluster
   - since we are seperating by trial indices we can ask whether the indices are within the early sliceTCA changepoint or in late and seperate the data by learning


.. py:function:: get_indices_TCA(cell_EC_model_ranks20_contig_x00, cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, residual_activity_dict_NDNF, animal_id=1, cell_id=1, num_clusters=4, early_late_nothing='nothing')

.. py:function:: get_indices_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln='nothing')

   - takes in the contiguous cutpoints slice TCA model and the K-Means slice TCA model
   - use elbow kmeans to loop through every number of cluster (up to 8) to get the optimal number of clusters via reconstruction MSE of cluster average reconstruction vs real data for the cell
   - returns a dict where every cell's activity is seperated by its cluster via trial indices for each cluster
   - since we are seperating by trial indices we can ask whether the indices are within the early sliceTCA changepoint or in late and seperate the data by learning


.. py:function:: get_clusters_dict_field_type(means_list, clusters_dict_EC, residual_activity_dict_EC, use_peak=True)

   -finds the argmax or argmin of each clusters' average activity from the cell dict of activity seperated into clusters and then labells that cluster as a before, near(reward), after or wraparound if the peak was near the end of the trial and there was a peak near the start (within 5 position bins) of the next trial


.. py:function:: plot_per_cell_clustering_internals_single_cluster(cell_EC_model_ranks20_contig_x00, cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, residual_activity_dict_NDNF, animal_id=1, cell_id=1, num_clusters=4, plot=True, early_late_nothing='nothing')

.. py:function:: get_count(animal_percent_dict_EC_early_peak, field_type='before_percent')

.. py:function:: plot_cluster_peaks(clusters_dict_labelled_field_type_early_SST_trough, field_type='near_field', plot=False)

.. py:function:: plot_clustered_averages(clusters_dict_labelled_field_type_early_SST_trough, clusters_dict_labelled_field_type_late_SST_trough, clusters_dict_labelled_field_type_early_SST_peak, clusters_dict_labelled_field_type_late_SST_peak, use_trough=True, plot=False, cell_type='SST')

   - uses plot cluster peaks to give you a average of the trials in the cluster for every cluster grouping - roughtly 1 of each per cell - does so for our 4 types of clusters/trial types - before, near, after and wraparound
   - then gets the mean and sem across cells
   - plots the array which is the trial average of every cluster across the whole dataset and the mean of them


.. py:function:: plot_clustered_averages_deterministic(clusters_dict_labelled_field_type_early_SST_trough, clusters_dict_labelled_field_type_late_SST_trough, plot=False, cell_type='SST')

.. py:function:: generate_gaussian(length=50, peak_position=25, std=5, amplitude=1.0)

   Generate a Gaussian array of given length, peaking at `peak_position`.

   Parameters:
   - length: total number of bins (default 50)
   - peak_position: where the Gaussian peaks (can be float)
   - std: standard deviation of the Gaussian
   - amplitude: height of the peak

   Returns:
   - 1D numpy array of shape (length,)


.. py:function:: random_timeseries(initial_value: float, volatility: float, count: int, rng)

.. py:function:: remove_duplicate_trials(cells_with_A, cells_with_B, trials_A_dict, trials_B_dict, rng)

.. py:function:: remove_all_duplicates(trial_dicts, cells_with_types, cell_array_early_list, rng)

   trial_dicts: dict of field_type -> {cell_id -> np.array of trial indices}
   cells_with_types: dict of field_type -> np.array of cell ids


.. py:function:: reconstruct_activity_from_clusters5(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, example_before_field_early, example_near_field_early, example_after_field_early, example_wraparound_field_early, example_before_field_late, example_near_field_late, example_after_field_late, example_wraparound_field_late, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, inits, vol, cell_type='EC Trough', plot=False)

   - consult the percent of cells that have the given trial type and then random choice select cells to express that field type
   - within each cell that gets a field get a number of trials that will express based on the proportion and then randomly select which trials out of all possible trials for that cell will be assigned that field types activity



.. py:function:: get_early_vs_late_activity(cell_SST_model_ranks20_contig_x00)

.. py:function:: min_max_my_data(array_data)

.. py:function:: compare_within_subtype(cell_SST_model_ranks20_contig_x00, trial_averaged_cell_array_early_SST_trough, trial_averaged_cell_array_late_SST_trough, full_activity_trial_av_array_SST_trough, residual_activity_dict_SST, cell_type='SST Peak')

.. py:function:: split_data(testing_cell_super_new_NDNF_model_ranks20_reassign_regkmean_x00_cell)

.. py:function:: plot_clustered_averages2(clusters_dict_labelled_field_type_early_SST_peak, clusters_dict_labelled_field_type_late_SST_peak, clusters_dict_labelled_field_type_middle_SST_peak, use_trough=True, plot=False, cell_type='SST', return_array=True)

   - params: early, middle and late cluster dicts cut by contiguous kmeans
   - returns: either dicts seperated by cell and animal for the cluster mean activity or arrays where all the clusters trial averages are clumped togther in a 2d array - dependent on the return_array flag


.. py:function:: get_field_type_percents2(clusters_dict_labelled_field_type, cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, e_or_l='e')

   - get a count of the total number of each of our 4 categories appearing across trials for each cell and then divide by total num trials for that cell to get a probability of trial appearance


.. py:function:: get_cells_percents(animal_percent_dict_EC_early_peak, animal_percent_dict_EC_late_peak, animal_percent_dict_EC_middle_peak, animal_percent_dict_EC_early_trough, animal_percent_dict_EC_late_trough, animal_percent_dict_EC_middle_trough)

.. py:function:: random_timeseries2(initial_value: float, volatility: float, count: int, rng)

.. py:function:: get_means_array(final_weighted_means_per_cell_before_early)

.. py:function:: reconstruct_activity_from_clusters_every_cell_diff_latent(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, before_field_array_early_trough_EC_dict, near_field_array_early_trough_EC_dict, after_field_array_early_trough_EC_dict, wraparound_field_array_early_trough_EC_dict, before_field_array_late_trough_EC_dict, near_field_array_late_trough_EC_dict, after_field_array_late_trough_EC_dict, wraparound_field_array_late_trough_EC_dict, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, inits, vol, cell_type='EC Trough', plot=False, seed=42)

   - same function as the other reconstruction except we are no longer reusing the same across-cell average trace slotted into every trial across all cells, now we are using a different average trace for each cell's trials but it is just a single trace per trial
   - uses the function get_weighted_activityies_per_cell in case there were multiple of the same type of field per cell (ex / 2 clusters that were both before fields with different kinetics) and gets a weighted average of them to hand into the before trials for that given cell


.. py:function:: get_weighted_activityies_per_cell(field_dict, field_type='before_field')

.. py:function:: plot_cluster_peaks2(clusters_dict_labelled_field_type_early_SST_trough, field_type='near_field', plot=False)

.. py:function:: reconstruct_activity_from_clusters_every_cell_diff_latent_split(cell_EC_model_ranks20_contig_x00, residual_activity_dict_EC, early_data_list_dict, late_data_list_dict, middle_data_list_dict, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_lists_EC_middle_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, percent_of_cells_list_EC_middle_trough, inits, vol, cell_type='EC Trough', plot=False, seed=42, num_t_e=20, num_t_m=40, num_t_l=20)

   - same function as the other reconstruction except we are no longer reusing the same across-cell average trace slotted into every trial across all cells, now we are using a different average trace for each cell's trials but it is just a single trace per trial
   - uses the function get_weighted_activityies_per_cell in case there were multiple of the same type of field per cell (ex / 2 clusters that were both before fields with different kinetics) and gets a weighted average of them to hand into the before trials for that given cell


.. py:function:: plot_generative_model(final_weights_early, final_weights_late, percent_lists_EC_early_peak, percent_lists_EC_late_peak, percent_of_cells_list_EC_early_trough, percent_of_cells_list_EC_late_trough, full_activity_trial_av_list, trial_averaged_cell_array_early, trial_averaged_cell_array_late, trial_averaged_cell_array_middle, cell_type='SST')

.. py:function:: slot_trials_into_spots(animal_first_changepoints_list, cells_lists, trials_lists, field_array_lists, inits, vol, rng=None, num_t=20)

.. py:function:: get_trial_indices(animal_first_changepoints_list, before_percent_array, near_percent_array, after_percent_array, wraparound_percent_array, cells_with_before_field_early, cells_with_near_field_early, cells_with_after_field_early, cells_with_wraparound_field_early, rng=None, num_trials=20)

.. py:function:: get_cell_array(animal_first_changepoints_list, inits, vol, rng=None, num_trials_early=20)

.. py:function:: randomly_pick_cells(percent_of_cells_list_EC_early_trough, animal_first_changepoints_list, num_cells, rng)

