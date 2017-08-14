
import os

import sys
sys.path.append(
    '/export/home/amatskev/Bachelor/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import run_lifted_mc

from init_datasets import meta_folder,computer_cores
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

experiment_sets = ['splB_z0', 'splB_z1']

if __name__ == '__main__':

    # TODO Change here
    from init_datasets import ds_names, project_folder, result_keys, experiment_ids

    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = computer_cores
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    for ds_id in experiment_ids:

        result_key = result_keys[ds_id]
        ds_name = ds_names[ds_id]

        result_folder = os.path.join(project_folder, ds_name)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        run_lifted_mc(
            meta_folder,
            meta_folder,
            ds_names,
            ds_name,
            os.path.join(result_folder, 'result.h5'),
            result_key,
            pre_save_path=os.path.join(result_folder, 'pre_result.h5')
        )