selectivity_over_learning_utils
===============================

.. py:module:: selectivity_over_learning_utils


Attributes
----------

.. autoapisummary::

   selectivity_over_learning_utils.file_path


Functions
---------

.. autoapisummary::

   selectivity_over_learning_utils.get_fixed_model_dict_NDNF_newest
   selectivity_over_learning_utils.subset_factors
   selectivity_over_learning_utils.Vinje2000
   selectivity_over_learning_utils.get_selectivity_each_trial
   selectivity_over_learning_utils.get_selectivity_each_trial_early_late
   selectivity_over_learning_utils.get_binned_data_for_CDF
   selectivity_over_learning_utils.get_mean_sem_lists
   selectivity_over_learning_utils.plot_the_CDF
   selectivity_over_learning_utils.plot_the_CDF_early_late
   selectivity_over_learning_utils.get_animal_average_selectivity_dict_eml
   selectivity_over_learning_utils.get_mean_selelectivity_by_cutpoint
   selectivity_over_learning_utils.plot_selectivity_seperated_by_learn_stage
   selectivity_over_learning_utils.plot_selectivity_over_trials
   selectivity_over_learning_utils.get_selectivity_array
   selectivity_over_learning_utils.get_animal_average_selectivity_dict
   selectivity_over_learning_utils.get_percentlie_slices


Module Contents
---------------

.. py:data:: file_path
   :value: '/Users/michaelfinch/CA1-interneuron-GLM'


.. py:function:: get_fixed_model_dict_NDNF_newest(cell_NDNF_model_ranks20_contig_x00)

.. py:function:: subset_factors(factors_dict_NDNF_newest)

.. py:function:: Vinje2000(tuning_curve, norm='None', negative_selectivity=False)

.. py:function:: get_selectivity_each_trial(activity_dict_EC, neg_sel=True, trial_av=False)

   - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
   returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually


.. py:function:: get_selectivity_each_trial_early_late(activity_dict_EC, cp_dict_EC, neg_sel=True, trial_av=False, use_early=True)

   - get a selectivity for every trial of every cell and then average it to be the selectivity for that cell or trial_av will trial average first and then run the selectivity on that trial averaged trace  if the trial_av flag is False
   returns: animal_average_selectivity_dict - every cell gets a single value from either the selectivity of the trial averaged trace or the average of all selectivity metrics quantified for every trial individually


.. py:function:: get_binned_data_for_CDF(animal_average_selectivity_dict_SST_r, n_bins=20)

   params animal_average_selectivity_dict_SST_r: selectivity value for every cell in a dict
   returns binned_data: the cells with selectiivty values that fit within each bin (percent of data)


.. py:function:: get_mean_sem_lists(binned_data)

.. py:function:: plot_the_CDF(binned_data_SST, binned_data_EC, binned_data_NDNF, title='Selectivity Distribution Across Cells +-SEM')

.. py:function:: plot_the_CDF_early_late(binned_data_SST_e, binned_data_EC_e, binned_data_NDNF_e, binned_data_SST_l, binned_data_EC_l, binned_data_NDNF_l, title='Selectivity Distribution Across Cells +-SEM')

.. py:function:: get_animal_average_selectivity_dict_eml(residual_activity_dict_SST, cp_dict_SST, neg_sel=True, trial_av=False)

.. py:function:: get_mean_selelectivity_by_cutpoint(animal_average_selectivity_dict_SST)

.. py:function:: plot_selectivity_seperated_by_learn_stage(animal_average_selectivity_dict_SST, animal_average_selectivity_dict_NDNF, animal_average_selectivity_dict_EC)

.. py:function:: plot_selectivity_over_trials(all_cell_selectivity_SST, all_cell_selectivity_EC, all_cell_selectivity_NDNF)

.. py:function:: get_selectivity_array(animal_average_selectivity_dict)

.. py:function:: get_animal_average_selectivity_dict(percentile_slices, neg_sel=True, trial_av=False)

.. py:function:: get_percentlie_slices(activity_dict_SST)

