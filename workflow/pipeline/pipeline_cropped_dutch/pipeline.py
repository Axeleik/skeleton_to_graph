
import os
import vigra
import cPickle as pickle
import numpy as np

import sys
sys.path.append(
    '/export/home/amatskev/Bachelor/nature_methods_multicut_pipeline/software/')

from multicut_src import DataSet
from multicut_src import lifted_multicut_workflow
from multicut_src import load_dataset
from multicut_src import compute_false_merges
from multicut_src import resolve_merges_with_lifted_edges_global, resolve_merges_with_lifted_edges
from multicut_src import RandomForest
from multicut_src import ExperimentSettings
from multicut_src import merge_small_segments

def init_dataset(
        meta_folder, name,
        raw_filepath, raw_name,
        probs_filepath, probs_name,
        seg_filepath, seg_name,
        gt_filepath=None, gt_name=None,
        make_cutouts=False
):

    # Init the dataset
    ds = DataSet(meta_folder, name)

    # Add data
    ds.add_raw(raw_filepath, raw_name)
    ds.add_input(probs_filepath, probs_name)
    ds.add_seg(seg_filepath, seg_name)
    if gt_filepath is not None:
        ds.add_gt(gt_filepath, gt_name)

    # add cutouts for lifted multicut training
    if make_cutouts:
        shape = ds.shape
        z_offset = 10
        ds.make_cutout([0, 0, 0], [shape[0], shape[1], z_offset])
        ds.make_cutout([0, 0, z_offset], [shape[0], shape[1], shape[2] - z_offset])
        ds.make_cutout([0, 0, shape[2] - z_offset], [shape[0], shape[1], shape[2]])


def init_train_sets(
        meta_folder, names,
        raw_path, raw_files, raw_names,
        probs_path, probs_files, probs_names,
        seg_path, seg_files, seg_names,
        gt_path, gt_files, gt_names
):

    for idx, train_name in enumerate(names):

        if not os.path.exists(os.path.join(meta_folder, train_name)):

            print 'Training set {} is being created ...'.format(train_name)

            raw_file = raw_files[idx]
            probs_file = probs_files[idx]
            seg_file = seg_files[idx]
            gt_file = gt_files[idx]
            raw_name = raw_names[idx]
            probs_name = probs_names[idx]
            seg_name = seg_names[idx]
            gt_name = gt_names[idx]

            init_dataset(
                meta_folder, train_name,
                raw_path + raw_file, raw_name,
                probs_path + probs_file, probs_name,
                seg_path + seg_file, seg_name,
                gt_filepath=gt_path + gt_file, gt_name=gt_name,
                make_cutouts=True
            )

        else:

            print 'Training set {} exists, nothing to do.'.format(train_name)


def run_lifted_mc(
        meta_folder,
        train_folder,
        ds_train_names,
        ds_test_name,
        save_path,
        results_name,
        pre_save_path=None
):
    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    merge_segments = True
    compute_mc = True
    if os.path.isfile(save_path):  # Nothing to do
        compute_mc = False
        merge_segments = False
    else:                          # Final result needs to be computed
        if pre_save_path is not None:
            if os.path.isfile(pre_save_path):
                compute_mc = False

    if compute_mc:

        seg_id = 0

        feature_list = ['raw', 'prob', 'reg']
        feature_list_lifted = ['cluster', 'reg']

        gamma = 2.

        ds_train = [load_dataset(train_folder, name) for name in ds_train_names if name != ds_test_name]
        ds_test = load_dataset(meta_folder, ds_test_name)

        mc_nodes, _, _, _ = lifted_multicut_workflow(
            ds_train, ds_test,
            seg_id, seg_id,
            feature_list, feature_list_lifted,
            gamma=gamma
        )

        segmentation = ds_test.project_mc_result(seg_id, mc_nodes)

        # Save in case sth in the following function goes wrong
        if pre_save_path is not None:
            vigra.writeHDF5(segmentation, pre_save_path, results_name, compression='gzip')

    if merge_segments:

        if not compute_mc:
            assert pre_save_path is not None, 'Investigate code, this must not happen!'
            segmentation = vigra.readHDF5(pre_save_path, results_name)

        # # Relabel with connected components
        # segmentation = vigra.analysis.labelVolume(segmentation.astype('uint32'))
        # Merge small segments
        segmentation = merge_small_segments(segmentation.astype('uint32'), 100)

        # Store the final result
        vigra.writeHDF5(segmentation, save_path, results_name, compression = 'gzip')


def find_false_merges(
        ds_test_name,
        ds_train_names,
        meta_folder, rf_cache_folder,
        test_seg_path, test_seg_key,
        train_segs_paths, train_segs_keys
):

    ds_train = [load_dataset(meta_folder, name) for name in ds_train_names if name != ds_test_name]
    ds_test = load_dataset(meta_folder, ds_test_name)

    # Path folders
    test_paths_cache_folder = os.path.join(meta_folder, ds_test_name, 'path_data')
    train_paths_cache_folder = os.path.join(meta_folder, 'train_path_data')

    _, false_merge_probs, _ = compute_false_merges(
        ds_train, ds_test,
        train_segs_paths, train_segs_keys,
        test_seg_path, test_seg_key,
        rf_cache_folder,
        test_paths_cache_folder,
        train_paths_cache_folder
    )

    with open(os.path.join(test_paths_cache_folder, 'false_paths_predictions.pkl'), 'w') as f:
        pickle.dump(false_merge_probs, f)


def resolve_false_merges(
        ds_name, ds_names,
        meta_folder, rf_cache_folder,
        new_nodes_filepath,
        pre_seg_filepath, pre_seg_key,
        min_prob_thresh, max_prob_thresh,
        exclude_objs_with_larger_thresh,
        global_resolve=True
):

    # Path folders
    paths_cache_folder = os.path.join(meta_folder, ds_name, 'path_data')

    # TODO Change here
    weight_filepath = os.path.join(meta_folder, ds_name,
                                   'probs_to_energies_0_z_16.0_0.5_rawprobreg.h5')
    lifted_filepath = os.path.join(meta_folder, ds_name,
                                   'lifted_probs_to_energies_0_3_0.5_2.0.h5')

    ds_train = [load_dataset(meta_folder, name) for name in ds_names if name != ds_name]
    rf_cache_name = 'rf_merges_%s' % '_'.join([ds.ds_name for ds in ds_train])

    ds = load_dataset(meta_folder, ds_name)
    seg_id = 0

    path_data_filepath = os.path.join(paths_cache_folder, 'paths_ds_{}.h5'.format(ds_name))
    # with open(os.path.join(paths_cache_folder, 'paths_ds_{}.pkl'.format(ds_name))) as f:
    #     path_data = pickle.load(f)
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(paths_cache_folder, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    # Find objects where probability >= min_prob_thresh and <= max_prob_thresh
    objs_with_prob_greater_thresh = np.unique(
        np.array(paths_to_objs)[
            np.logical_and(
                false_merge_probs >= min_prob_thresh,
                false_merge_probs <= max_prob_thresh
            )
        ]
    )
    if exclude_objs_with_larger_thresh:
        objs_to_exclude = np.unique(
            np.array(paths_to_objs)[
                false_merge_probs > max_prob_thresh
            ]
        )
        objs_with_prob_greater_thresh = np.setdiff1d(objs_with_prob_greater_thresh, objs_to_exclude)

    # Extract all paths for each of the found objects
    false_paths = {}
    for obj in objs_with_prob_greater_thresh:
        # print paths_to_objs == obj
        false_paths[obj] = np.array(paths)[paths_to_objs == obj]

    rf_filepath = os.path.join(rf_cache_folder, rf_cache_name)
    # with open(rf_filepath) as f:
    #     path_rf = pickle.load(f)
    path_rf = RandomForest.load_from_file(rf_filepath, 'rf', ExperimentSettings().n_threads)

    mc_segmentation = vigra.readHDF5(pre_seg_filepath, pre_seg_key)
    mc_weights_all = vigra.readHDF5(weight_filepath, "data")
    lifted_weights_all = vigra.readHDF5(lifted_filepath, "data")

    if global_resolve:
        new_node_labels = resolve_merges_with_lifted_edges_global(
            ds, seg_id,
            false_paths,
            path_rf,
            mc_segmentation,
            mc_weights_all,
            paths_cache_folder=paths_cache_folder,
            lifted_weights_all=lifted_weights_all
        )
    else:
        new_node_labels = resolve_merges_with_lifted_edges(
            ds, seg_id,
            false_paths,
            path_rf,
            mc_segmentation,
            mc_weights_all,
            paths_cache_folder=paths_cache_folder,
            lifted_weights_all=lifted_weights_all
        )

    with open(new_nodes_filepath, 'w') as f:
        pickle.dump(new_node_labels, f)


def project_new_result(
        ds_name, meta_folder,
        new_nodes_filepath,
        save_path, results_name
):

    ds = load_dataset(meta_folder, ds_name)
    seg_id = 0

    # Load resolving result
    with open(new_nodes_filepath) as f:
        new_node_labels = pickle.load(f)

    # project the result back to the volume
    mc_seg = ds.project_mc_result(seg_id, new_node_labels)

    # Write the result
    vigra.writeHDF5(mc_seg, save_path, results_name, compression = 'gzip')


import nifty_with_cplex.graph.rag as nrag

def project_resolved_objects_to_segmentation(
        meta_folder, ds_name,
        mc_seg_filepath, mc_seg_key,
        new_nodes_filepath,
        save_path, results_name
):

    ds = load_dataset(meta_folder, ds_name)
    seg_id = 0

    mc_segmentation = vigra.readHDF5(mc_seg_filepath, mc_seg_key)

    # Load resolving result
    with open(new_nodes_filepath) as f:
        resolved_objs = pickle.load(f)

    rag = ds.rag(seg_id)
    mc_labeling = nrag.gridRagAccumulateLabels(rag, mc_segmentation)
    new_label_offset = np.max(mc_labeling) + 1
    for obj in resolved_objs:
        resolved_nodes = resolved_objs[obj]
        for node_id in resolved_nodes:
            mc_labeling[node_id] = new_label_offset + resolved_nodes[node_id]
        new_label_offset += np.max(np.array(resolved_nodes.values())) + 1
    mc_segmentation = nrag.projectScalarNodeDataToPixels(rag, mc_labeling, ExperimentSettings().n_threads)


    # Write the result
    vigra.writeHDF5(mc_segmentation, save_path, results_name, compression = 'gzip')