import numpy as np
import vigra
import h5py
from copy import deepcopy
from path_computation_for_tests import compute_graph_and_paths,cut_off
import logging
logger = logging.getLogger(__name__)


def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None, anisotropy=[1,1,10]):
    """
        extract paths from segmentation, for pipeline
    """

    if False:
        pass
    # if paths_cache_folder is not None:
    #     if not os.path.exists(paths_cache_folder):
    #         os.mkdir(paths_cache_folder)
    #     paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s.h5' % ds.ds_name)
    # else:
    #     paths_save_file = ''
    #
    # # if the cache exists, load paths from cache
    # if os.path.exists(paths_save_file):
    #     all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
    #     # we need to reshape the paths again to revover the coordinates
    #     if all_paths.size:
    #         all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])
    #     paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')

    # otherwise compute the paths
    else:

        seg = vigra.readHDF5(seg_path, key)
        dt = ds.inp(ds.n_inp - 1)
        img = deepcopy(seg)
        all_paths = []
        paths_to_objs = []

        len_uniq=len(np.unique(seg))-1
        for idx,label in enumerate(np.unique(seg)):
            print "Number ", idx, " without labels of ",len_uniq-1

            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1


            paths=compute_graph_and_paths(img, dt, anisotropy)

            if len(paths)==0:
                continue

            [all_paths.extend([np.array(path)]) for path in paths]
            [paths_to_objs.extend([label]) for path in paths]

        # all_paths=np.array(all_paths)
        paths_to_objs=np.array(paths_to_objs, dtype="float64")

        # if paths_cache_folder is not None:
        #     # need to write paths with vlen and flatten before writing to properly save this
        #     all_paths_save = np.array([pp.flatten() for pp in all_paths])
        #     # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
        #     # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
        #     # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
        #     # see also the following issue (https://github.com/h5py/h5py/issues/875)
        #     try:
        #         with h5py.File(paths_save_file) as f:
        #             dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
        #             f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
        #     except (TypeError, IndexError):
        #         vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
        #     # if len(all_paths_save) < 2:
        #     #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
        #     # else:
        #     #     with h5py.File(paths_save_file) as f:
        #     #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
        #     #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
        #     vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')

    return all_paths, paths_to_objs


def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None, anisotropy=[1,1,10]):

    """
        extract paths from segmentation, for learning
    """
    logger.debug('Extracting paths and labels from segmentation ...')

    if False:
        pass
    # if paths_cache_folder is not None:
    #     if not os.path.exists(paths_cache_folder):
    #         os.mkdir(paths_cache_folder)
    #     paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s_seg_%i.h5' % (ds.ds_name, seg_id))
    # else:
    #     paths_save_file = ''
    #
    # # if the cache exists, load paths from cache
    # if os.path.exists(paths_save_file):
    #     all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
    #     # we need to reshape the paths again to revover the coordinates
    #     if all_paths.size:
    #         all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])
    #     paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')
    #     path_classes = vigra.readHDF5(paths_save_file, 'path_classes')
    #     correspondence_list = vigra.readHDF5(paths_save_file, 'correspondence_list').tolist()

    # otherwise compute paths
    else:
        assert seg.shape == gt.shape
        dt = ds.inp(ds.n_inp - 1)  # we assume that the last input is the distance transform

        img = deepcopy(seg)
        dt = ds.inp(ds.n_inp - 1)
        all_paths = []
        paths_to_objs = []
        path_classes=[]

        len_uniq=len(np.unique(seg))-1
        for label in np.unique(seg):
            print "Label ", label, " of ",len_uniq

            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1

            paths=compute_graph_and_paths(img, dt, anisotropy)

            if len(paths)==0:
                continue

            all_paths, paths_to_objs, path_classes=\
                cut_off(all_paths, paths_to_objs,
                            path_classes, label, paths, gt, anisotropy)

        all_paths=np.array(all_paths)
        paths_to_objs=np.array(paths_to_objs, dtype="float64")
        path_classes=np.array(path_classes)

        # # if caching is enabled, write the results to cache
        # if paths_cache_folder is not None:
        #     # need to write paths with vlen and flatten before writing to properly save this
        #     all_paths_save = np.array([pp.flatten() for pp in all_paths])
        #     # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
        #     # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
        #     # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
        #     # see also the following issue (https://github.com/h5py/h5py/issues/875)
        #     try:
        #         logger.info('Saving paths in {}'.format(paths_save_file))
        #         with h5py.File(paths_save_file) as f:
        #             dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
        #             f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
        #     except (TypeError, IndexError):
        #         vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
        #     # if len(all_paths_save) < 2:
        #     #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
        #     # else:
        #     #     with h5py.File(paths_save_file) as f:
        #     #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
        #     #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
        #     vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')
        #     vigra.writeHDF5(path_classes, paths_save_file, 'path_classes')
        #     vigra.writeHDF5(correspondence_list, paths_save_file, 'correspondence_list')

    logger.debug('... done extracting paths and labels from segmentation!')
    return all_paths, paths_to_objs, path_classes, correspondence_list