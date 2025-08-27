finding_weights_over_NDNF_utils
===============================

.. py:module:: finding_weights_over_NDNF_utils


Functions
---------

.. autoapisummary::

   finding_weights_over_NDNF_utils.subset_factors
   finding_weights_over_NDNF_utils.generate_spatial_rate_maps
   finding_weights_over_NDNF_utils.wrap_around_and_compress
   finding_weights_over_NDNF_utils.get_CA3_data
   finding_weights_over_NDNF_utils.plot_CA3
   finding_weights_over_NDNF_utils.get_plot_EC
   finding_weights_over_NDNF_utils.get_synthetic_NDNF
   finding_weights_over_NDNF_utils.fit_GLM2
   finding_weights_over_NDNF_utils.get_EC_data_array
   finding_weights_over_NDNF_utils.get_CA3_data_array
   finding_weights_over_NDNF_utils.get_SST_data_array
   finding_weights_over_NDNF_utils.get_MSE_cell_type
   finding_weights_over_NDNF_utils.plot_MSE_for_NDNF_pred_by_other_celltype_input
   finding_weights_over_NDNF_utils.get_model_dict
   finding_weights_over_NDNF_utils.get_data_array_learning_split
   finding_weights_over_NDNF_utils.get_MSE_cell_type2
   finding_weights_over_NDNF_utils.get_model_dict_early_celltype_late_another_celltype
   finding_weights_over_NDNF_utils.plot_early_late_split_all_input_types
   finding_weights_over_NDNF_utils.plot_models_trained_early_late
   finding_weights_over_NDNF_utils.get_model_dict_early_celltype_late_another_celltype_TA
   finding_weights_over_NDNF_utils.get_MSE_cell_type_TA
   finding_weights_over_NDNF_utils.plot_models_trained_early_late
   finding_weights_over_NDNF_utils.get_fixed_model_dict_NDNF_newest
   finding_weights_over_NDNF_utils.reorder_early_late_pairs
   finding_weights_over_NDNF_utils.get_activity_av_all_trials
   finding_weights_over_NDNF_utils.get_cp_dict
   finding_weights_over_NDNF_utils.get_activity_early
   finding_weights_over_NDNF_utils.get_activity_late
   finding_weights_over_NDNF_utils.get_MSE_lists
   finding_weights_over_NDNF_utils.remove_behaviors_GLM
   finding_weights_over_NDNF_utils.get_model_dict_split
   finding_weights_over_NDNF_utils.get_MSE_cell_type_TA
   finding_weights_over_NDNF_utils.plot_MSE_for_NDNF_pred_by_other_celltype_input_random
   finding_weights_over_NDNF_utils.plot_SST_prediction_from_other_celltypes_input_GLM
   finding_weights_over_NDNF_utils.get_velocities
   finding_weights_over_NDNF_utils.plot_datar
   finding_weights_over_NDNF_utils.plot_data
   finding_weights_over_NDNF_utils.plot_coefficients_cell_type
   finding_weights_over_NDNF_utils.extract_weight_lists
   finding_weights_over_NDNF_utils.plot_coefficients_all_celltypes_together


Module Contents
---------------

.. py:function:: subset_factors(factors_dict_NDNF_newest)

.. py:function:: generate_spatial_rate_maps(x, n=200, peak_rate=1.0, field_width=90.0, track_length=180.0)

   Return a list of spatial rate maps with peak locations that span the track. Return firing rate vs. location
   computed at the resolution of the provided x array.
   :param x: array
   :param n: int
   :param peak_rate: float
   :param field_width: float
   :param track_length: float
   :return: list of array, array


.. py:function:: wrap_around_and_compress(waveform, interp_x)

.. py:function:: get_CA3_data(track_length=180, num_cells=200)

.. py:function:: plot_CA3(ca3_vs_position_all_cells_array)

.. py:function:: get_plot_EC(residual_activity_dict_EC)

.. py:function:: get_synthetic_NDNF(EC_residuals_array, ca3_vs_position_all_cells_array, dist_type='normal')

.. py:function:: fit_GLM2(EC_data_array, neuron_activity_flat, regression='linear', alphas=None)

.. py:function:: get_EC_data_array(residual_activity_dict_EC)

.. py:function:: get_CA3_data_array(ca3_vs_position_all_cells_array)

.. py:function:: get_SST_data_array(activity_dict_SST, residual_activity_dict_SST)

.. py:function:: get_MSE_cell_type(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array)

.. py:function:: plot_MSE_for_NDNF_pred_by_other_celltype_input(MSE_list_just_SST, MSE_list_just_EC, MSE_list_just_CA3, MSE_list_EC_CA3, MSE_list_EC_SST, MSE_list_CA3_SST, MSE_list_EC_CA3_SST, inputs_list, output_cell_type='NDNF Output')

.. py:function:: get_model_dict(EC_data_array, fixed_residual_activity_dict_NDNF_newest, reg_type='ridge')

.. py:function:: get_data_array_learning_split(activity_dict_EC)

.. py:function:: get_MSE_cell_type2(model_EC_dict_early, fixed_activity_dict_NDNF_newest, data_late_array_EC, predict_early_or_late='early')

   "
   inputs - the models that were trained on early data from EC and is trained to predict early NDNF, testing on the neuron activity of EC late which will give a prediction of NDNF late
   input order - the models that were trained just on the


.. py:function:: get_model_dict_early_celltype_late_another_celltype(EC_data_array, fixed_activity_dict_NDNF_newest, reg_type='ridge', early_or_late='early')

   "
   train on the set of all possible inputs early in learn (first few trials for them) then test the output cell population (NDNF)


.. py:function:: plot_early_late_split_all_input_types(activity_dict_SST, activity_dict_EC, fixed_activity_dict_NDNF_newest, ca3_vs_position_all_cells_array, cp_dict_EC, cp_dict_SST, cp_dict_NDNF, ymax=1.55)

.. py:function:: plot_models_trained_early_late(data_list, input_titles, title='Model MSEs', ylabel='MSE')

.. py:function:: get_model_dict_early_celltype_late_another_celltype_TA(EC_data_array, fixed_activity_dict_NDNF_newest, reg_type='ridge', early_or_late='early', start=20, end=30)

   "
   train on the set of all possible inputs early in learn (first few trials for them) then test the output cell population (NDNF)


.. py:function:: get_MSE_cell_type_TA(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array)

.. py:function:: plot_models_trained_early_late(data_list, input_titles, ymax=3, title='Labels=Train_timepoint:Test_timepoint)')

.. py:function:: get_fixed_model_dict_NDNF_newest(cell_NDNF_model_ranks20_contig_x00)

.. py:function:: reorder_early_late_pairs(data_list, input_titles)

.. py:function:: get_activity_av_all_trials(activity_dict_SST)

.. py:function:: get_cp_dict(cell_SST_model_ranks20_contig_x00)

.. py:function:: get_activity_early(activity_dict_SST, cp_dict_SST)

.. py:function:: get_activity_late(activity_dict_SST, cp_dict_SST)

.. py:function:: get_MSE_lists(neural_activity_sup_bef_V1, GLM_data_dict_sup_bef_V1, behavioral_correlates_before)

.. py:function:: remove_behaviors_GLM(activity_dict_NDNF, NDNF_GLM_models, design_matrix_dict_NDNF, indices_array)

.. py:function:: get_model_dict_split(EC_data_array, fixed_residual_activity_dict_NDNF_newest, start=20, end=30, reg_type='ridge', early=True)

.. py:function:: get_MSE_cell_type_TA(model_EC_just_SST, fixed_residual_activity_dict_NDNF_newest, SST_data_array)

.. py:function:: plot_MSE_for_NDNF_pred_by_other_celltype_input_random(MSE_list_just_SST, MSE_list_just_EC, MSE_list_just_CA3, MSE_list_EC_CA3, MSE_list_EC_SST, MSE_list_CA3_SST, MSE_list_EC_CA3_SST, MSE_list_random, inputs_list, output_cell_type='NDNF Output', ymax=0.03)

.. py:function:: plot_SST_prediction_from_other_celltypes_input_GLM(activity_dict_SST, activity_dict_EC, fixed_activity_dict_NDNF_newest, ymax=0.1)

.. py:function:: get_velocities(factors_dict_SST)

.. py:function:: plot_datar(SST_datas, title='Velocity Across All Animals', color='b')

.. py:function:: plot_data(SST_datas, SST_datas_r, title='SST Activity Across All Cells', color='b')

.. py:function:: plot_coefficients_cell_type(weights_list_EC, cell_type='EC')

.. py:function:: extract_weight_lists(weights_list)

.. py:function:: plot_coefficients_all_celltypes_together(weights_list_NDNF, weights_list_SST, weights_list_EC, title='GLM Coefficients Across Cell Types')

