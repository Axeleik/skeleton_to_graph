
import vigra
import cPickle as pickle
import os
import re
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import sys
sys.path.append('/home/jhennies/src/cremi_python/cremi_python')
from cremi import Volume, NeuronIds

def roi_and_rand_general(
        sample, half, defect_correct, project_folder,
        source_folder,
        result_file, caching=False, debug=False
):
    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print 'Result file: {}'.format(result_file)

    experiment_folder = os.path.join(project_folder, 'spl{}_z{}/'.format(sample, half))

    if caching:
        cache_filepath = os.path.join(
            experiment_folder,
            re.sub('.h5$', '', result_file) + '_roi_and_rand_cache.pkl'
        )
    else:
        cache_filepath = None

    if caching and os.path.isfile(cache_filepath):
        with open(cache_filepath, mode='r') as f:
            voi_split, voi_merge, adapted_rand = pickle.load(f)

    else:

        if defect_correct:
            defect_correct_str = '_defect_correct'
        else:
            defect_correct_str = ''

        mc_result_key = 'z/{}/data'.format(half)

        # # Load stuff
        # source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

        # # TODO: Change here when switching sample
        # ref_result_filepath = os.path.join(source_folder, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(sample))
        # # TODO: Change here when switching half
        # ref_result_key = 'z/{}/beta_0.5'.format(half)

        gt_filepath = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defect_correct_str))
        gt_key = 'z/{}/neuron_ids'.format(half)

        # ref_result, _, _ = vigra.analysis.relabelConsecutive(vigra.readHDF5(ref_result_filepath, ref_result_key), start_label=1, keep_zeros=True)
        if not debug:
            gt = vigra.readHDF5(gt_filepath, gt_key)
            vol_gt = Volume(gt)
            neuron_ids_evaluation = NeuronIds(vol_gt)

        mc_result_filepath = os.path.join(experiment_folder, result_file)

        if not debug:
            # Evaluate baseline
            mc_result = vigra.readHDF5(mc_result_filepath, mc_result_key)
            vol_mc_result = Volume(mc_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_mc_result)
            adapted_rand = neuron_ids_evaluation.adapted_rand(vol_mc_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            adapted_rand = 0.23

        if caching:
            with open(cache_filepath, mode='w') as f:
                pickle.dump((voi_split, voi_merge, adapted_rand), f)

    print "\tvoi split   : " + str(voi_split)
    print "\tvoi merge   : " + str(voi_merge)
    print "\tadapted RAND: " + str(adapted_rand)

    return (voi_split, voi_merge, adapted_rand)


from evaluation import resolve_merges_error_rate_path_level


def eval_obj_measures(
        spl, half,
        project_folder,
        seg_file, seg_key,
        resolved_file, resolved_key,
        thresh_range=None,
        resolved_only=False,
        defect_correct=False
):

    if thresh_range is None:
        thresh_range = [0.3]

    experiment_folder = os.path.join(
        project_folder,
        'spl{}_z{}/'.format(spl, half)
    )
    path_data_folder = os.path.join(
        project_folder,
        'cache/spl{}_z{}/path_data/'.format(spl, half)
    )

    path_data_filepath = os.path.join(
        path_data_folder,
        'paths_ds_spl{}_z{}.h5'.format(spl, half)
    )

    if defect_correct:
        defect_correct_str = '_defect_correct'
    else:
        defect_correct_str = ''

    mc_result_key = 'z/{}/data'.format(half)

    # Load stuff
    # source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
    source_folder = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/'

    gt_filepath = os.path.join(
        source_folder,
        'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(spl, defect_correct_str))
    gt_key = 'z/{}/neuron_ids'.format(half)

    # Load paths
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(path_data_folder, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    # Load images
    resolved = vigra.readHDF5(os.path.join(experiment_folder, resolved_file), resolved_key)
    gt = vigra.readHDF5(gt_filepath, gt_key)
    seg = vigra.readHDF5(os.path.join(experiment_folder, seg_file), seg_key)

    # Determine merge error rate (path level)
    errors_seg, errors_rsd, errors_to_obj = resolve_merges_error_rate_path_level(
        paths, paths_to_objs, resolved, gt, seg,
        false_merge_probs,
        thresh_range=thresh_range,
        resolved_only=resolved_only
    )

    return errors_seg, errors_rsd, errors_to_obj


def eval_obj_measures_readable(
        spl, half,
        project_folder,
        seg_file, seg_key,
        resolved_files, resolved_key,
        thresh_range=None,
        resolved_only=False,
        defect_correct=False
):

    if thresh_range == None:
        thresh_range = [0.3]

    result = []

    files_shape = None
    if np.array(resolved_files).ndim > 1:
        files_shape = np.array(resolved_files).shape
        resolved_files = np.array(resolved_files).flatten()

    for resolved_file in resolved_files:

        errors_seg, errors_rsd, errors_to_obj = eval_obj_measures(
            spl, half,
            project_folder,
            seg_file, seg_key,
            resolved_file, resolved_key,
            thresh_range=thresh_range,
            resolved_only=resolved_only,
            defect_correct=defect_correct
        )

        result.append([])

        for thresh in thresh_range:
            print 'Evaluation for thresh = {}'.format(thresh)

            # TP
            obj_mask_with_all_true_seg = np.array([x.all() for x in errors_seg[thresh]])
            tp = sum(np.logical_not(obj_mask_with_all_true_seg))

            # TP & fully resolved
            obj_mask_with_all_true_rsd = np.array([(x != 1).all() for x in errors_rsd[thresh]])
            false_merge_rsd = sum(np.logical_not(obj_mask_with_all_true_rsd))
            tp_fully_resolved = tp - false_merge_rsd

            # TP & falsely split
            obj_mask_at_least_one_false_split = np.array([(x == 2).any() for x in errors_rsd[thresh]])
            tp_falsely_split = sum(
                np.logical_and(
                    obj_mask_at_least_one_false_split, np.logical_not(obj_mask_with_all_true_seg)
                )
            )

            # FP
            errors_seg_with_all_true = np.array(errors_seg[thresh])[obj_mask_with_all_true_seg]
            fp = len(errors_seg_with_all_true)

            # FP & falsely split
            obj_mask_at_least_one_false_split = np.array([(x == 2).any() for x in errors_rsd[thresh]])
            fp_falsely_split = sum(
                np.logical_and(
                    obj_mask_at_least_one_false_split, obj_mask_with_all_true_seg
                )
            )

            # Like this we can find the erroneous and correct pairs with information on
            #   the error type:
            #   3: gt == True, rs == True -> correct merge
            #   2: gt == True, rs == False -> false split
            #   1: gt == False, rs == True -> false merge
            #   0: gt == False, rs == False -> correct split
            # The original performance with respect to merged obj is in gt_equal:
            #   True -> correct merge
            #   False -> false merge

            # Object count
            number_of_objs = len(errors_seg[thresh])

            result[-1].append((number_of_objs, tp, fp, tp_fully_resolved, tp_falsely_split, fp_falsely_split))

    if files_shape is not None:
        result = np.array(result)
        result = result.reshape(np.concatenate((files_shape, result.shape[1:])))

    return result


def path_eval_on_sample(sample, half, defect_correct, project_folder, thresh_range):

    from evaluation import compute_path_error_rates

    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print '--------------------'

    if defect_correct:
        defect_correct_str = '_defect_correct'
    else:
        defect_correct_str = ''

    # Load stuff
    source_folder = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/'
    # TODO: Change here
    experiment_folder = os.path.join(
        project_folder, 'spl{}_z{}/'.format(sample, half)
    )
    meta_folder = os.path.join(project_folder, 'cache/')

    test_name = 'spl{}_z{}'.format(sample, half)

    path_data_path = os.path.join(meta_folder,
                                  'spl{}_z{}/path_data'.format(sample, half))
    path_data_filepath = os.path.join(path_data_path, 'paths_ds_{}.h5'.format(test_name))

    # TODO Change here when switching sample
    gt_file = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5').format(sample, defect_correct_str)
    # TODO Change here when switching half
    gt_key = 'z/{}/neuron_ids'.format(half)
    gt = vigra.readHDF5(gt_file, gt_key)

    # Load paths
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(path_data_path, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    print 'Number of paths = {}'.format(len(paths_to_objs))
    print 'Number of objects = {}'.format(len(np.unique(paths_to_objs)))

    # Determine path error rates
    result_path, result_obj = compute_path_error_rates(
        paths_to_objs, paths, gt, false_merge_probs, thresh_range=thresh_range
    )

    return result_path, result_obj


def all_sample_path_eval(project_folder, thresh_range, samples, halves, defect_corrects, measures=None):

    if measures is None:
        measures = ['precision', 'recall', 'accuracy', 'f1']

    results_path = {}
    results_obj = {}
    for measure in measures:
        results_path[measure] = []
        results_obj[measure] = []

    for idx, sample in enumerate(samples):
        result_path, result_obj = path_eval_on_sample(sample, halves[idx], defect_corrects[idx], project_folder,
                                                      thresh_range)

        sorted_keys = sorted(result_path.keys())
        for key in results_path.keys():
            results_path[key].append(np.array([result_path[k][key] for k in sorted_keys])[:, None])
            results_obj[key].append(np.array([result_obj[k][key] for k in sorted_keys])[:, None])

    return results_path, results_obj


def plot_all_sample_path_eval(project_folder, thresh_range, halves, defect_corrects, samples,
                              measures=['f1', 'precision', 'recall', 'accuracy']):

    results_path, results_obj = all_sample_path_eval(
        project_folder, thresh_range, samples, halves, defect_corrects,
        measures=measures
    )

    plt.figure()

    for key in results_path.keys():

        measures_path = np.concatenate(results_path[key], axis=1)

        means_path = np.mean(measures_path, axis=1)
        std_path = np.std(measures_path, axis=1)

        plt.errorbar(thresh_range, means_path, yerr=std_path, fmt='-o', label=key)

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()

    plt.figure()

    for key in results_obj.keys():

        measures_obj = np.concatenate(results_obj[key], axis=1)

        means_obj = np.mean(measures_obj, axis=1)
        std_obj = np.std(measures_obj, axis=1)

        plt.errorbar(thresh_range, means_obj, yerr=std_obj, fmt='-o', label=key)

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()

    plt.show()


def plot_all_sample_path_eval_split_samples(
        project_folder, thresh_range, halves, defect_corrects, samples,
        measures=['f1', 'precision', 'recall', 'accuracy']
):

        results_path, results_obj = all_sample_path_eval(
            project_folder, thresh_range, samples, halves, defect_corrects,
            measures=measures
        )

        plt.figure()

        colors = {'f1': 'C0', 'precision': 'C1', 'recall': 'C2'}
        opacity = 1
        error_config = {'ecolor': '0.3'}

        for key, val in results_path.iteritems():
            # measures_path = np.concatenate(results_path[key], axis=1)

            # means_path = np.mean(measures_path, axis=1)
            # std_path = np.std(measures_path, axis=1)

            plt.errorbar(thresh_range, val[0], yerr=[0] * len(val[0]), fmt='-o', label=key,
                         color=colors[key], alpha=opacity)
            plt.errorbar(thresh_range, val[1], yerr=[0] * len(val[1]), fmt='--o',
                         color=colors[key], alpha=opacity)

        plt.xlabel('$t_m$')
        plt.title("")
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([-0.05, 1.05])

        plt.figure()

        for key, val in results_obj.iteritems():
            # measures_obj = np.concatenate(results_obj[key], axis=1)
            #
            # means_obj = np.mean(measures_obj, axis=1)
            # std_obj = np.std(measures_obj, axis=1)

            plt.errorbar(thresh_range, val[0], yerr=[0] * len(val[0]), fmt='-o', label=key, color=colors[key])
            plt.errorbar(thresh_range, val[1], yerr=[0] * len(val[1]), fmt='--o', color=colors[key])

        plt.xlabel('$t_m$')
        plt.title("")
        plt.legend()
        axes = plt.gca()
        axes.set_xlim([-0.05, 1.05])

        plt.show()


if __name__ == '__main__':

    pass