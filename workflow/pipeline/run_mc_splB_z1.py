
import os

import sys
sys.path.append(
    '/export/home/amatskev/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from workflow.pipeline import run_lifted_mc

# TODO Change here
from init_exp_splB_z1 import meta_folder, experiment_folder
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

# TODO Change here when switching half
result_key = 'z/1/data'


if __name__ == '__main__':

    # TODO Change here
    from init_exp_splB_z1 import test_name, train_name

    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = 8
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    # TODO Change here for changing half
    run_lifted_mc(
        meta_folder,
        train_name,
        test_name,
        experiment_folder + 'result.h5',
        result_key
    )
