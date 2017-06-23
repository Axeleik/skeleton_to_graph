
import os

import sys
sys.path.append(
    '/export/home/amatskev/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings, load_dataset

from workflow.pipeline import resolve_false_merges, project_new_result

if __name__ == '__main__':

    # TODO Change here
    from init_exp_splB_z1 import meta_folder, experiment_folder
    from init_exp_splB_z1 import test_name
    from run_mc_splB_z1 import rf_cache_folder
    from detect_merges_splB_z1 import test_paths_cache_folder

    # These are the parameters as used for the initial mc
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = 8
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    # Parameters for the resolving algorithm
    ExperimentSettings().min_nh_range = 5
    ExperimentSettings().max_sample_size = 20

    # TODO Change here
    rf_cache_name = 'rf_merges_ds_train_splB_z0/'
    min_prob_thresh = 0.3
    max_prob_thresh = 1.
    exclude_objs_with_larger_thresh = False
    pre_seg_filepath = os.path.join(experiment_folder, 'result.h5')

    # TODO Change here when switching half
    pre_seg_key = 'z/1/data'
    # TODO Change here
    weight_filepath = os.path.join(meta_folder, test_name,
                                   'probs_to_energies_0_z_16.0_0.5_rawprobreg.h5')
    lifted_filepath = os.path.join(meta_folder, test_name,
                                   'lifted_probs_to_energies_0_3_0.5_2.0.h5')

    # Global resolving -------------------
    new_nodes_filepath = os.path.join(meta_folder, 'new_ones_global.pkl')

    resolve_false_merges(
        test_name, meta_folder, test_paths_cache_folder, rf_cache_folder,
        new_nodes_filepath,
        pre_seg_filepath, pre_seg_key,
        weight_filepath, lifted_filepath,
        rf_cache_name,
        min_prob_thresh, max_prob_thresh,
        exclude_objs_with_larger_thresh,
        global_resolve=True
    )

    result_filepath = os.path.join(experiment_folder, 'result_resolved_global.h5')

    project_new_result(
        test_name, meta_folder,
        new_nodes_filepath,
        result_filepath, pre_seg_key
    )

    # # Local resolving ---------------------
    # new_nodes_filepath = os.path.join(meta_folder, 'new_ones_local.pkl')
    #
    # resolve_false_merges(
    #     test_name, meta_folder, test_paths_cache_folder, rf_cache_folder,
    #     new_nodes_filepath,
    #     pre_seg_filepath, pre_seg_key,
    #     weight_filepath, lifted_filepath,
    #     rf_cache_name,
    #     min_prob_thresh, max_prob_thresh,
    #     exclude_objs_with_larger_thresh,
    #     global_resolve=False
    # )
    #
    # result_filepath = os.path.join(experiment_folder, 'result_resolved_local.h5')
    #
    # project_new_result(
    #     test_name, meta_folder,
    #     new_nodes_filepath,
    #     result_filepath, pre_seg_key
    # )
