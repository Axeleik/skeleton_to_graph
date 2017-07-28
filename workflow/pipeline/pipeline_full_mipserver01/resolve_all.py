
import os

import sys
sys.path.append(
    '/export/home/amatskev/Bachelor/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings, load_dataset

from pipeline import resolve_false_merges, project_new_result, project_resolved_objects_to_segmentation

if __name__ == '__main__':

    from init_datasets import meta_folder, project_folder
    from init_datasets import ds_names, result_keys, experiment_ids,computer_cores
    from run_mc_all import rf_cache_folder
    # assert (1==0)
    # These are the parameters as used for the initial mc
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = computer_cores
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    # Parameters for the resolving algorithm
    ExperimentSettings().min_nh_range = 5
    ExperimentSettings().max_sample_size = 20

    # Parameters deciding which objects to resolve
    min_prob_thresh = 0.3
    max_prob_thresh = 1.
    exclude_objs_with_larger_thresh = False

    for ds_id in experiment_ids:
        ds_name = ds_names[ds_id]

        result_key = result_keys[ds_id]

        test_seg_path = os.path.join(project_folder, ds_name, 'result.h5')
        test_seg_key = result_keys[ds_id]

        # Local resolving ---------------------
        # TODO: Change here when adding result
        new_nodes_filepath = os.path.join(meta_folder, ds_name, 'new_ones_local.pkl')
        # TODO: Change here when adding result
        result_filepath = os.path.join(project_folder, ds_name, 'result_resolved_local.h5')

        if not os.path.isfile(result_filepath):

            resolve_false_merges(
                ds_name, ds_names,
                meta_folder, rf_cache_folder,
                new_nodes_filepath,
                test_seg_path, test_seg_key,
                min_prob_thresh, max_prob_thresh,
                exclude_objs_with_larger_thresh,
                global_resolve=False
            )

            project_resolved_objects_to_segmentation(
                meta_folder, ds_name,
                test_seg_path, test_seg_key,
                new_nodes_filepath,
                result_filepath, test_seg_key
            )

        else:

            print 'Nothing to do'