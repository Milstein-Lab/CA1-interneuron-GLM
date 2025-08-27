SliceTCA_example
================

.. py:module:: SliceTCA_example


Functions
---------

.. autoapisummary::

   SliceTCA_example.plot_synthetic_sliceTCA_ex
   SliceTCA_example.run_slice_tca_ideal_ex
   SliceTCA_example.get_ideal_random_cell
   SliceTCA_example.plot_model_vs_reco
   SliceTCA_example.plot_example_weights_latents_real
   SliceTCA_example.plot_real_activity_clusters
   SliceTCA_example.plot_peak_trough_histograms
   SliceTCA_example.plot_optimal_num_clusters_histogram
   SliceTCA_example.plot_cluster_expression_probs
   SliceTCA_example.plot_max_clusters
   SliceTCA_example.get_indices_dict
   SliceTCA_example.get_per_cell_clustering_indices
   SliceTCA_example.get_activity_from_indices
   SliceTCA_example.get_data_by_peak
   SliceTCA_example.mean_sems_peak_trough
   SliceTCA_example.plot_means_sems_max_min
   SliceTCA_example.find_elbow_point
   SliceTCA_example.plot_elbow_cell_example
   SliceTCA_example.plot_synthetic_sliceTCA_ex
   SliceTCA_example.plot_real_cells_internals
   SliceTCA_example.plot_real_activity_clusters
   SliceTCA_example.get_activity_array
   SliceTCA_example.get_mean_activity_array
   SliceTCA_example.plot_pop_stats
   SliceTCA_example.compare_cluster_2_0
   SliceTCA_example.cell_by_cell_variance_values
   SliceTCA_example.sum_ec_then_get_var
   SliceTCA_example.plot_cluster_frequency_by_animal
   SliceTCA_example.plot_cluster_frequency_by_animal


Module Contents
---------------

.. py:function:: plot_synthetic_sliceTCA_ex(model, ideal_cell)

.. py:function:: run_slice_tca_ideal_ex(ideal_cell)

.. py:function:: get_ideal_random_cell(num_trials=None, num_timebins=None)

.. py:function:: plot_model_vs_reco(example_EC_cell_tensor, model_60, x=60)

.. py:function:: plot_example_weights_latents_real(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=None, cell=None)

.. py:function:: plot_real_activity_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal=None, cell=None)

.. py:function:: plot_peak_trough_histograms(clusters_dict_NDNF_early, use_argmax=True, title=None, ylim=None, ax=None)

.. py:function:: plot_optimal_num_clusters_histogram(clusters_dict_NDNF_early, title=None, ylim=None, ax=None)

.. py:function:: plot_cluster_expression_probs(clusters_dict_NDNF_early, residual_activity_dict_NDNF_newest, title=None, ylim=None, ax=None)

.. py:function:: plot_max_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, title=None, ax=None)

.. py:function:: get_indices_dict(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, eln='nothing')

   - takes in the contiguous cutpoints slice TCA model and the K-Means slice TCA model
   - use elbow kmeans to loop through every number of cluster (up to 8) to get the optimal number of clusters via reconstruction MSE of cluster average reconstruction vs real data for the cell
   - returns a dict where every cell's activity is seperated by its cluster via trial indices for each cluster
   - since we are seperating by trial indices we can ask whether the indices are within the early sliceTCA changepoint or in late and seperate the data by learning


.. py:function:: get_per_cell_clustering_indices(cell_EC_model_ranks20_contig_x00, residual_activity_dict_NDNF, cell_NDNF_model_ranks20_kmeans_reassign_umap_x00, animal_id=1, cell_id=1, num_clusters=4, split_learn=False, early_late_nothing='nothing')

.. py:function:: get_activity_from_indices(residual_activity_dict_EC, indices_dict_EC_overall)

.. py:function:: get_data_by_peak(activity_indices_EC_dict, use_peak=True)

.. py:function:: mean_sems_peak_trough(mean_data_by_peak, max_bin, use_peak=True)

.. py:function:: plot_means_sems_max_min(activity_indices_EC_dict, ax=None)

.. py:function:: find_elbow_point(y_vals, min_index=2)

.. py:function:: plot_elbow_cell_example(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, animal=None, cell=None)

.. py:function:: plot_synthetic_sliceTCA_ex(model, ideal_cell)

.. py:function:: plot_real_cells_internals(model_2)

.. py:function:: plot_real_activity_clusters(testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal=None, cell=None)

.. py:function:: get_activity_array(activity, ax, i, total_num_trials=None)

.. py:function:: get_mean_activity_array(activity, ax, i, min=None, max=None, total_num_trials=None)

.. py:function:: plot_pop_stats(clusters_dict_EC_overall, cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC, animal=None, cell=None, cell_type='EC', ylim=None)

.. py:function:: compare_cluster_2_0(cell_EC_model_ranks20_contig_x00, testing_cell_EC_model_ranks20_reassign_regkmean_x00_cell, residual_activity_dict_EC)

.. py:function:: cell_by_cell_variance_values(activity_dict_EC)

.. py:function:: sum_ec_then_get_var(activity_dict_EC)

.. py:function:: plot_cluster_frequency_by_animal(clusters_dict_NDNF_overall, use_argmax=False, title='NDNF')

.. py:function:: plot_cluster_frequency_by_animal(clusters_dict_NDNF_overall, use_argmax=False, title='NDNF')

