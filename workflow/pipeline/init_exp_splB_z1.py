
import os

import sys
sys.path.append(
    '/export/home/amatskev/nature_methods_multicut_pipeline/software/')


from multicut_src import DataSet
from workflow.pipeline import init_dataset

# The following locations should be importable by downstream scripts
source_folder = '/mnt/localdata03/amatskev/neuraldata/'
project_folder = '/mnt/localdata03/amatskev/neuraldata/results/'
experiment_folder = project_folder + 'splB_z1/'
meta_folder = project_folder + 'splB_z1/cache/'
test_name = 'splB_z1'
train_name = 'splB_z0'

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    raw_path = source_folder
    raw_file = 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    probs_path = source_folder
    probs_file = 'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    seg_path = source_folder
    seg_file = 'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'
    gt_path = source_folder
    gt_file = 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'

    # Init test set
    init_dataset(
        meta_folder, test_name,
        raw_path + raw_file, 'z/1/raw',
        probs_path + probs_file, 'z/1/data',
        seg_path + seg_file, 'z/1/labels'
    )

    # Init train set
    init_dataset(
        meta_folder, train_name,
        raw_path + raw_file, 'z/0/raw',
        probs_path + probs_file, 'z/0/data',
        seg_path + seg_file, 'z/0/labels',
        gt_filepath=gt_path + gt_file, gt_name='z/0/neuron_ids',
        make_cutouts=True
    )
