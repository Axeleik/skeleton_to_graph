
import os

import sys
sys.path.append(
    '/export/home/amatskev/nature_methods_multicut_pipeline/software/')

# The following locations should be importable by downstream scripts
# TODO Change here when switching server
source_folder = '/mnt/localdata03/amatskev/neuraldata/cremi_data/'
project_folder = '/mnt/localdata03/amatskev/neuraldata/results/'

ds_names = ['splA_z0', 'splA_z1', 'splB_z0', 'splB_z1', 'splC_z0', 'splC_z1']
experiment_ids = [3]
result_keys = ['z/0/data', 'z/1/data'] * 3

meta_folder = os.path.join(project_folder, 'cache')

from pipeline import init_train_sets

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    train_raw_files = [
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_raw_names = ['z/0/raw', 'z/1/raw'] * 3
    train_probs_files = [
        'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_probs_names = ['z/0/data', 'z/1/data'] * 3
    train_seg_files = [
        'cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_seg_names = ['z/0/labels', 'z/1/labels'] * 3
    gt_files = [
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    gt_names = ['z/0/neuron_ids', 'z/1/neuron_ids'] * 3

    # Init train sets
    init_train_sets(
        meta_folder, ds_names,
        source_folder, train_raw_files, train_raw_names,
        source_folder, train_probs_files, train_probs_names,
        source_folder, train_seg_files, train_seg_names,
        source_folder, gt_files, gt_names
    )