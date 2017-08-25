import numpy as np
import vigra
import h5py
from copy import deepcopy
from path_computation_for_tests \
    import compute_graph_and_paths,cut_off, parallel_wrapper
import logging
logger = logging.getLogger(__name__)
from joblib import Parallel,delayed
import vigra


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

    else:

        seg = vigra.readHDF5(seg_path, key)
        dt = ds.inp(ds.n_inp - 1)
        all_paths = []
        paths_to_objs = []

        #creating distance transform of whole volume for border near paths
        volume_expanded = np.ones((dt.shape[0]+2,dt.shape[1]+2,dt.shape[1]+2))
        volume_expanded[1:-1, 1:-1, 1:-1] = 0
        volume_dt = vigra.filters.distanceTransform(
            volume_expanded.astype("uint32"), background=True,
            pixel_pitch=[10, 1, 1])[1:-1, 1:-1, 1:-1]

        #threshhold for distance transform for picking terminal
        #points near boundary
        threshhold_boundary=30
        volume_where_threshhold = np.where(volume_dt > threshhold_boundary)
        volume_dt_boundaries = np.s_[min(volume_where_threshhold[0]):max(volume_where_threshhold[0]),
                               min(volume_where_threshhold[1]):max(volume_where_threshhold[1]),
                               min(volume_where_threshhold[2]):max(volume_where_threshhold[2])]

        #for counting and debugging purposes
        len_uniq=len(np.unique(seg))-1

        #parallelized path computation
        parallel_array = Parallel(n_jobs=-1)\
            (delayed(parallel_wrapper)(seg, dt, [],
                                       anisotropy, label,
                                       len_uniq, volume_dt_boundaries,
                                       "only_paths")
             for label in np.unique(seg))


        [[all_paths.append(path)
           for path in seg_array[0] if seg_array!=[]]
            for seg_array in parallel_array]

        [[paths_to_objs.append(path_to_obj)
          for path_to_obj in seg_array[1] if seg_array!=[]]
            for seg_array in parallel_array]

        # all_paths=np.array(all_paths)
        paths_to_objs=np.array(paths_to_objs, dtype="float64")


    return all_paths, paths_to_objs


def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None, anisotropy=[1, 1, 10]):

    """
        extract paths from segmentation, for learning
    """

    logger.debug('Extracting paths and labels from segmentation ...')


    if False:
        pass
    # otherwise compute paths
    else:


        assert seg.shape == gt.shape
        dt = ds.inp(ds.n_inp - 1)
        all_paths = []
        paths_to_objs = []
        path_classes = []

        # creating distance transform of whole volume for border near paths
        volume_expanded = np.ones((dt.shape[0] + 2, dt.shape[1] + 2, dt.shape[1] + 2))
        volume_expanded[1:-1, 1:-1, 1:-1] = 0
        volume_dt = vigra.filters.distanceTransform(
            volume_expanded.astype("uint32"), background=True,
            pixel_pitch=[10, 1, 1])[1:-1, 1:-1, 1:-1]

        # threshhold for distance transform for picking terminal
        # points near boundary
        threshhold_boundary=30
        volume_where_threshhold = np.where(volume_dt > threshhold_boundary)
        volume_dt_boundaries = np.s_[min(volume_where_threshhold[0]):max(volume_where_threshhold[0]),
                               min(volume_where_threshhold[1]):max(volume_where_threshhold[1]),
                               min(volume_where_threshhold[2]):max(volume_where_threshhold[2])]

        # for counting and debugging purposes
        len_uniq = len(np.unique(seg)) - 1

        #parallelized path computation
        parallel_array = Parallel(n_jobs=-1)\
            (delayed(parallel_wrapper)(seg, dt, gt,
                                       anisotropy, label, len_uniq,
                                       volume_dt_boundaries)
             for label in np.unique(seg))
        # parallel_array=[parallel_wrapper(seg, dt, gt,
        #                                anisotropy, label, len_uniq,
        #                                volume_dt_boundaries)
        #      for label in np.unique(seg)]


        [[all_paths.append(path)
           for path in seg_array[0]]
            for seg_array in parallel_array]

        [[paths_to_objs.append(path_to_obj)
          for path_to_obj in seg_array[1]]
            for seg_array in parallel_array]

        [[path_classes.append(path_class)
           for path_class in seg_array[2]]
            for seg_array in parallel_array]


        print "finished appending"
        all_paths = np.array(all_paths)
        paths_to_objs = np.array(paths_to_objs, dtype="float64")
        path_classes = np.array(path_classes)
        print "finished numpying"

    logger.debug('... done extracting paths and labels from segmentation!')

    return all_paths, paths_to_objs, \
            path_classes, correspondence_list


