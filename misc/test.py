import numpy as np
# from mesh_test import plot_mesh
import test_functions as tf
# from neuro_seg_plot import NeuroSegPlot as nsp
from skimage.measure import label
# import nifty_with_cplex as nifty
# import nifty_with_cplex.graph.rag as nrag
# from concurrent import futures
from skimage.morphology import skeletonize_3d
from time import time
from copy import deepcopy,copy
import sys
import math
sys.path.append(
    '/export/home/amatskev/Bachelor/nature_methods_multicut_pipeline/software/')
sys.path.append(
    '/export/home/amatskev/Bachelor/skeleton_to_graph/')
from workflow.methods.functions_for_workflow \
    import extract_paths_and_labels_from_segmentation
# from multicut_src.false_merges.compute_paths_and_features import path_feature_aggregator
from multicut_src.false_merges import compute_border_contacts,false_merges_workflow
# import vigra
# from joblib import parallel,delayed
from multiprocessing import Process
from Queue import Queue
# import platform
# from joblib import Parallel, delayed
import cPickle as pickle
import scipy
# from joblib import Parallel,delayed
import multiprocessing
# from path_computation_for_tests import parallel_wrapper



def close_cavities(volume):
    test_vol=np.zeros(volume.shape)
    test_vol[volume==0]=2
    test_vol[volume==1]=1
    test_vol=np.pad(test_vol, 1, "constant", constant_values=2)

    lab=label(test_vol,connectivity=1)
    if len(np.unique(lab))==2:
        return volume
    count,what=0,0
    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    test_vol[lab==what]=0
    test_vol[lab != what] = 1
    return test_vol[1:-1,1:-1,1:-1]

def example_rag():
    x = np.zeros((25, 25), dtype='uint32')
    x[:13, :13] = 1
    x[12:, 12:] = 2
    rag = nrag.gridRag(x)
    return rag


    # iterate over all the nodes in the rag
    # for each, print its adjacent nodes and the corresponding edge
def node_iteration(rag):

    # iterate over the nodes
    for node_id in xrange(rag.numberOfNodes):

        # iterate over the node adjacency of this node
        print "Node:", node_id, "is adjacent to:"
        for adj_node, adj_edge in rag.nodeAdjacency(node_id):
            print "Node:", adj_node
            print "via Edge", adj_edge

def skeletonization_parallel(seg,seg_copy,skel_img,seg_id):

    seg_copy[seg == seg_id] = 1
    seg_copy[seg != seg_id] = 0

    skel_dump=skeletonize_3d(seg_copy)

    skel_img[skel_dump==1]=1


def skeletonization_linear(seg,seg_id):

    seg_copy=np.zeros(seg.shape)
    seg_copy[seg == seg_id] = 1


    skel_img=skeletonize_3d(seg_copy)

    return skel_img



def parallel(seg):

    time1=time()
    skel_img=np.zeros(seg.shape)

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        [executor.submit(skeletonization_parallel, seg,copy(seg),skel_img,seg_id) for seg_id in np.unique(seg)]

    time2=time()
    print "parallel took ", time2-time1, " secs"

    return skel_img


def linear(seg):

    time1=time()
    skel_img=np.zeros(seg.shape)

    for seg_id in np.unique(seg):
        single_skel=skeletonization_linear(seg,seg_id)
        skel_img[single_skel==1]=1

    time2 = time()
    print "linear took ", time2 - time1, " secs"

    return skel_img

def func(hi):
    for i in xrange(0,10):
        hi.extend([i])


def worker(img_orig,label):
    """thread worker function"""
    img = np.zeros(img_orig.shape)
    img[img_orig==label]=1

    skel_img=skeletonize_3d(img)
    if label==60:
        return []
    return skel_img

def mp_skeletonize(seg):

    def worker(img_orig, label,out_q):
        """thread worker function"""
        img = np.zeros(img_orig.shape)
        img[img_orig == label] = 1

        skel_img = skeletonize_3d(img)

        out_q.put(skel_img)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    out_q = Queue()
    # chunksize = int(math.ceil(len(nums) / float(nprocs)))
    procs = []

    for label in np.unique(seg):
        p = Process(
                target=worker,
                args=(seg,label,
                      out_q))
        procs.append(p)
        p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    final=np.zeros(seg.shape)

    print "1"
    while out_q.size():
        final[out_q.get()==1]=1
    print "2"

    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    print "3"
    return final


def extract_region_features(
        feature_volume,
        path_image,
        ignoreLabel=0,
        features=''
):

    extractor = vigra.analysis.extractRegionFeatures(
        feature_volume,
        path_image,
        ignoreLabel=ignoreLabel,
        features=features
    )

    return extractor


def python_region_features_extractor_2_mc(single_vals):
    path_features = []

    [path_features.extend([np.mean(vals), np.var(vals), sum(vals),
                           max(vals), min(vals),
                           scipy.stats.kurtosis(vals),
                           (((vals - vals.mean()) / vals.std(ddof=0)) ** 3).mean()]) for vals in single_vals]

    return np.array(path_features)



def python_region_features_extractor_sc(path,feature_volumes):

    pixel_values = []

    for feature_volume in feature_volumes:
        for c in range(feature_volume.shape[-1]):

            pixel_values.extend([feature_volume[path[:, 0], path[:, 1], path[:, 2]][:,c]])


    return np.array(pixel_values)

def python_region_features_extractor_2_mc(single_vals,idx):

    path_features=[]

    [path_features.extend([np.mean(vals),np.var(vals),sum(vals),
         max(vals),min(vals),
    scipy.stats.kurtosis(vals),scipy.stats.skew(vals)])
                          for vals in single_vals]

    # print idx ," done"

    if idx%100==0:

        print idx

    return np.array(path_features)

def python_region_features_extractor_2_mc_test(single_vals,idx):

    path_features=[]

    [path_features.extend([
                           (((vals - vals.mean()) / vals.std(ddof=0)) ** 3).mean()])
     for vals in single_vals]

    # print idx ," done"

    # if idx%100==0:

    print idx

    return np.array(path_features)


def extract_features_for_path(path,feature_volumes,stats,idx):

        print "start feature ",idx
        # calculate the local path bounding box
        min_coords  = np.min(path, axis=0)
        max_coords = np.max(path, axis=0)
        max_coords += 1
        shape = tuple(max_coords - min_coords)
        path_image = np.zeros(shape, dtype='uint32')
        path -= min_coords
        # we swapaxes to properly index the image properly
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        path_features = []
        for feature_volume in feature_volumes:
            for c in range(feature_volume.shape[-1]):
                path_roi = np.s_[
                    min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2],
                    c  # wee need to also add the channel to the slicing
                ]
                # extractor = vigra.analysis.extractRegionFeatures(
                #     feature_volume[path_roi],
                #     path_image,
                #     ignoreLabel=0,
                #     features=stats
                # )
                extractor = extract_region_features(
                    feature_volume[path_roi],
                    path_image,
                    ignoreLabel=0,
                    features=stats
                )
                # TODO make sure that dimensions match for more that 1d stats!
                path_features.extend(
                    [extractor[stat][1] for stat in stats]
                )
                hi=[extractor[stat][1] for stat in stats]
        # ret = np.array(path_features)[:,None]
        # print ret.shape
        print "path_features: ",idx
        return np.array(path_features)[None, :]

def read(path):

    with open(path, mode='r') as f:
        file = pickle.load(f)

    return file


def test_func(a,b):
        # put to Queue instead of returning
        return np.array([a,b])



def python_region_features_extractor_2_mc(single_vals):

    path_features = []

    [path_features.extend([np.mean(vals), np.var(vals), sum(vals),
                            max(vals), min(vals),
                            scipy.stats.kurtosis(vals),
                           (((vals - vals.mean()) / vals.std(ddof=0)) ** 3).mean()
])
     for vals in single_vals]

    return np.array(path_features)

def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        anisotropy_factor):

    # FIXME for now we don't use fastfilters here
    feat_paths = ds.make_filters(inp_id, anisotropy_factor)
    # print feat_paths
    # TODO sort the feat_path correctly
    # load the feature images ->
    # FIXME this might be too memory hungry if we have a large global bounding box

    # compute the global bounding box
    global_min = np.min(
        np.concatenate([np.min(path, axis=0)[None, :] for path in paths], axis=0),
        axis=0
    )
    global_max = np.max(
        np.concatenate([np.max(path, axis=0)[None, :] for path in paths], axis=0),
        axis=0
    ) + 1
    # substract min coords from all paths to bring them to new coordinates
    paths_in_roi = [path - global_min for path in paths]
    roi = np.s_[
        global_min[0]:global_max[0],
        global_min[1]:global_max[1],
        global_min[2]:global_max[2]
    ]

    # load features in global boundng box
    feature_volumes = []
    import h5py
    for path in feat_paths:
        with h5py.File(path) as f:
            feat_shape = f['data'].shape
            # we add a singleton dimension to single channel features to loop over channel later
            if len(feat_shape) == 3:
                feature_volumes.append(f['data'][roi][..., None])
            else:
                feature_volumes.append(f['data'][roi])
    stats = ExperimentSettings().feature_stats

    def extract_features_for_path(path):

        # calculate the local path bounding box
        min_coords  = np.min(path, axis=0)
        max_coords = np.max(path, axis=0)
        max_coords += 1
        shape = tuple(max_coords - min_coords)
        path_image = np.zeros(shape, dtype='uint32')
        path -= min_coords
        # we swapaxes to properly index the image properly
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        path_features = []
        for feature_volume in feature_volumes:
            for c in range(feature_volume.shape[-1]):
                path_roi = np.s_[
                    min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2],
                    c  # wee need to also add the channel to the slicing
                ]
                extractor = vigra.analysis.extractRegionFeatures(
                    feature_volume[path_roi],
                    path_image,
                    ignoreLabel=0,
                    features=stats
                )
                # TODO make sure that dimensions match for more that 1d stats!
                path_features.extend(
                    [extractor[stat][1] for stat in stats]
                )
        # ret = np.array(path_features)[:,None]
        # print ret.shape
        return np.array(path_features)[None, :]

    if len(paths) > 1:

        # We parallelize over the paths for now.
        # TODO parallelizing over filters might in fact be much faster, because
        # we avoid the single threaded i/o in the beginning!
        # it also lessens memory requirements if we have less threads than filters
        # parallel
        with futures.ThreadPoolExecutor(max_workers=ExperimentSettings().n_threads) as executor:
            tasks = []
            for p_id, path in enumerate(paths_in_roi):
                tasks.append(executor.submit(extract_features_for_path, path))
            out = np.concatenate([t.result() for t in tasks], axis=0)

    else:

        out = np.concatenate([extract_features_for_path(path) for path in paths_in_roi])

    # serial for debugging
    # out = []
    # for p_id, path in enumerate(paths_in_roi):
    #     out.append( extract_features_for_path(path) )
    # out = np.concatenate(out, axis = 0)

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out

if __name__ == '__main__':
    import h5py
    filename = '/export/home/amatskev/Downloads/groundtruth.h5'
    f = h5py.File(filename, 'r')






    feature_stats = ["Mean", "Variance", "Sum", "Maximum", "Minimum", "Kurtosis", "Skewness"]





    # ds, seg, seg_id, gt, correspondence_list = \
    #     np.load("/mnt/localdata01/amatskev/"
    #             "debugging/border_term_points/first_seg_paths_and_labels_all.npy")
    # dt = \
    #         np.load("/mnt/localdata01/amatskev/debugging/border_term_points//"
    #             "first_dt.npy")
    # centres_dict=np.load("/mnt/localdata01/amatskev/debugging/"
    #         "border_term_points/centres_array.npy").tolist()
    #
    # extract_paths_and_labels_from_segmentation(
    #     dt, seg, seg_id, gt, correspondence_list,centres_dict)

    # _,_, _, _, _, paths_cache_folder = \
    #     np.load("/mnt/ssd/amatskev/debugging/border_term_points/for_cut_off.npy")
    #
    #
    # ds, seg, seg_id, gt, correspondence_list = \
    #     np.load("/mnt/ssd/amatskev/debugging/border_term_points/first_seg_paths_and_labels_all.npy")
    #
    # false_merges_workflow.extract_paths_and_labels_from_segmentation(ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)

    # seg= \
    #     np.load("/mnt/localdata01/amatskev/debugging/border_term_points/"
    #             "first_seg.npy")
    # dt= \
    #     np.load("/mnt/localdata01/amatskev/debugging/border_term_points//"
    #             "first_dt.npy")
    # time1=time()
    # centres_array =compute_border_contacts.compute_border_contacts_old(seg,dt)
    # time2=time()
    # print "time: ",time2-time1
    #
    # np.save("/mnt/localdata01/amatskev/debugging/"
    #         "border_term_points/centres_array.npy",centres_array)
    #
    # where_centres=np.where(centres_array)[0].transpose()
    #
    # centre_dict={}
    #
    # for centre_coords in where_centres:
    #     pass

    # seg = np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_seg.npy")
    # volume_dt = np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_volume_dt.npy")
    # # seg = np.array(f["z/1/data"])

    # threshhold_boundary = 20
    # volume_where_threshhold = np.where(volume_dt > threshhold_boundary)
    # volume_dt_boundaries = np.s_[min(volume_where_threshhold[0]):max(volume_where_threshhold[0]),
    #                        min(volume_where_threshhold[1]):max(volume_where_threshhold[1]),
    #                        min(volume_where_threshhold[2]):max(volume_where_threshhold[2])]
    # where = np.unique(seg)
    #
    # for label_number in where:
    #     img = np.zeros(seg.shape)
    #     img[seg == label_number] = 1
    #     img[volume_dt_boundaries] = 0
    #     unique = label(img)
    #     if len(np.unique(img)) != 2:
    #         print "label: ", label_number, " of ", len(where) - 1

    # feature_volumes=[]
    # for i in xrange(0, 9):
    #     feature_volumes.append \
    #         (np.load("/net/hci-storage02/userfolders/amatskev/debugging/debugging_feature_volumes_for_NaN{}.npy".format(i)))
    #     print "volume ", i, " loaded"
    # all_paths=np.load(
    #     "/mnt/ssd/amatskev/debugging/debugging_paths_seg__6.npy")
    #
    # float_feature_volumes=[]
    # for i in feature_volumes:
    #     float_feature_volumes.append(np.float64(i))
    #
    #
    # pixel_values_all = [python_region_features_extractor_sc(path,float_feature_volumes)
    #                     for idx, path in enumerate(all_paths[6151:6152])]
    # print "pixel_values "
    # out2 = np.concatenate([extract_features_for_path_orig(path,feature_volumes,idx)
    #                        for idx,path in enumerate(all_paths[6151:6152])])
    #
    #
    # out1 = np.array([python_region_features_extractor_2_mc_test(single_vals,idx)
    #                 for idx,single_vals in enumerate(pixel_values_all)])
    #
    #
    # print "hi"


    # out2 = np.array([python_region_features_extractor_2_mc_test(single_vals,idx)
    #                 for idx,single_vals in enumerate(pixel_values_all)])
    # print "1: ",out1[309][6],", ",out1[309][34]
    # print "2: ", out2[309][6], ", ", out2[309][34]
    # all_paths = vigra.readHDF5("/mnt/localdata01/amatskev/debugging/paths_ds_splC_z1_seg_8.h5", 'all_paths')  #slimchicken
    # all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])
    # #     "/mnt/ssd/amatskev/debugging/border_term_points/"
    # #     "comparison_pruning_4.npy")
    # out2=   np.load("/mnt/ssd/amatskev/debugging/border_term_points/features_train.npy")

    # with open("/mnt/ssd/amatskev/debugging/border_term_points/pixel_values_full.pkl", mode='r') as f:
    #     pixel_values_all = pickle.load(f)
    #
    # out1 = np.array(Parallel(n_jobs=-1) \
    #                     (delayed(python_region_features_extractor_2_mc)(single_vals)
    #                      for single_vals in pixel_values_all))


    #
    # boundaries=(slice(3, 58, None), slice(30, 1219, None), slice(30, 1219, None))
    #
    # seg=np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_seg.npy")
    # gt=np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_gt.npy")
    # img=np.zeros((seg.shape))
    # img[seg==2]=1
    # img[boundaries] = 0
    #
    # # result1=extract_paths_and_labels_from_segmentation(ds,seg,seg_id,gt,correspondence_list)
    # pass
    # with open("/mnt/ssd/amatskev/debugging/border_term_points/pixel_values.pkl", mode='w') as f:
    #     pickle.dump(pixel_values_all, f)
    # testing_dt(200,seg)



    # factor=70
    # print "factor: ",factor
    # np.save("/mnt/ssd/amatskev/debugging/border_term_points/"
    #     "first_result_border_term_40.npy",result1)
    # #
    # print "len",factor,": ", len(result1[0])



    # ds, seg, seg_id, gt, correspondence_list, paths_cache_folder = \
    #     np.load("/mnt/localdata01/amatskev/misc/debugging/for_cut_off.npy")
    #
    # len_uniq = len(np.unique(seg)) - 1
    # dt = ds.inp(ds.n_inp - 1)
    #
    # # creating distance transform of whole volume for border near paths
    # volume_expanded = np.ones((dt.shape[0] + 2, dt.shape[1] + 2, dt.shape[1] + 2))
    # volume_expanded[1:-1, 1:-1, 1:-1] = 0
    # volume_dt = vigra.filters.distanceTransform(
    #     volume_expanded.astype("uint32"), background=True,
    #     pixel_pitch=[10., 1., 1.])[1:-1, 1:-1, 1:-1]
    # # we assume that the last input is the distance transform
    # threshhold_boundary=30
    # volume_where_threshhold=np.where(volume_dt<threshhold_boundary)
    # volume_dt_boundaries=np.s_[min(volume_where_threshhold[0]):max(volume_where_threshhold[0]),
    #                         min(volume_where_threshhold[1]):max(volume_where_threshhold[1]),
    #                         min(volume_where_threshhold[2]):max(volume_where_threshhold[2])]
    #
    #
    #
    # dummy_volume=np.ones(volume_dt.shape)
    # bla2=[parallel_wrapper(seg,dt,gt,[10,1,1],label,len_uniq,dummy_volume,"only_paths") for label in np.unique(seg)]
    #
    # bla1=[parallel_wrapper(seg,dt,gt,[10,1,1],label,len_uniq,volume_dt,"only_paths") for label in np.unique(seg)]
    # i1=0
    # i2=0
    # for idx in xrange(0,len(np.unique(seg))):
    #     i1=i1+len(bla1[idx][0])
    #     print "len i1: ",len(bla1[idx][0])
    #     i2 = i2 + len(bla2[idx][0])
    #     print "len i2: ",len(bla2[idx][0])
    # print "i1: ",i1
    # print "i2: ",i2
    #
    # np.save("/mnt/localdata01/amatskev/misc/debugging/border_paths.npy",(bla1,bla2))


    # print "hi"



    # dummy = np.ones((5, 5, 5),dtype="uint32")
    #
    #     a=vigra.filters.distanceTransform(
    #         dummy.astype("uint32"),background=False)
    #
    #
    #
    #     feature_volumes = []
    #
    #     for i in xrange(0, 9):
    #         feature_volumes.append \
    #             (np.load("/mnt/ssd/amatskev/debugging/debugging_feature_volumes{}.npy".format(i)))
    #         print "volume ", i, " loaded"
    #
    #     print "feature volumes loaded"
    #     stats = read("/mnt/ssd/amatskev/debugging/debuggingstats.pkl")
    #     print "stats loaded"
    #
    #     paths_in_roi = np.load("/mnt/ssd/amatskev/debugging/debugging_paths_in_roi.npy")
    #     print "paths in roi loaded"
    #
    #     # a_test=[x for x in range(0,100000)]
    #
    #     print "start 1000"
    #     # time0=time()
    #     # # out1 = np.concatenate([extract_features_for_path(path,feature_volumes,stats,idx)
    #     # #                       for idx,path in enumerate(paths_in_roi[:1000])])
    #     time1=time()
    #
    #     ##########################################
    #     pixel_values_all = [python_region_features_extractor_sc(path,feature_volumes)
    #                           for idx,path in enumerate(paths_in_roi[:3])]
    #
    #     parallel_array = np.array(Parallel(n_jobs=-1) \
    #         (delayed(python_region_features_extractor_2_mc)(single_vals)
    #          for single_vals in pixel_values_all ))
    #
    #     out=np.array([python_region_features_extractor_2_mc(single_vals)
    #                         for single_vals in pixel_values_all])
    #
    #
    #
    #     ##########################################
    #     time2=time()
    #     # with futures.ThreadPoolExecutor(max_workers=32) as executor:
    #     #     tasks = []
    #     #     for idx,single_vals in enumerate(out2):
    #     #         tasks.append(executor.submit(python_region_features_extractor_2_mc, single_vals,idx))
    #     #     out = np.concatenate([t.result() for t in tasks], axis=0)
    #     #
    #     # print "first: ", time1-time0
    #     # print "second: ", time2-time1
    #
    #     # with futures.ThreadPoolExecutor(max_workers=32) as executor:
    #     #     tasks = []
    #     #     for idx, path in enumerate(paths_in_roi):
    #     #         tasks.append(executor.submit(extract_features_for_path,
    #     #                                      path, feature_volumes,stats,idx))
    #     #     out = np.concatenate([t.result() for t in tasks], axis=0)
    #     time3=time()
    #     print "time mine: ",time2-time1
    #     print "time before: ",time3-time2

    # with futures.ThreadPoolExecutor(max_workers=8) as executor:
        #     tasks = []
        #     for p_id, a in enumerate(a_test):
        #         tasks.append(executor.submit(test_func, a, a * a))
        #     out1 = np.concatenate([t.result() for t in tasks], axis=0)

        # with futures.ThreadPoolExecutor(max_workers=24) as executor:
        #     tasks = []
        #     for idx, path in enumerate(paths_in_roi):
        #         tasks.append(executor.submit(extract_features_for_path, path,feature_volumes,stats,idx))
        #     out = np.concatenate([t.result() for t in tasks], axis=0)
        #
        #
        # # parallel_array = Parallel(n_jobs=-1) \
        # #     (delayed(extract_features_for_path)( path,feature_volumes,stats,p_id)
        # #      for p_id, path in enumerate(paths_in_roi[0:50]) )
        #
        #
        # print len(out)

        # print "start"
        #
        # feature_volumes = []
        #
        # for i in xrange(0, 5):
        #     feature_volumes.append \
        #         (np.load("/mnt/localdata01/amatskev/debugging/debugging_feature_volumes{}.npy".format(i)))
        #     print "volume ", i, " loaded"
        #
        # print "feature volumes loaded"
        # stats = read("/mnt/localdata01/amatskev/debugging/debuggingstats.pkl")
        # print "stats loaded"
        #
        # paths_in_roi = np.load("/mnt/localdata01/amatskev/debugging/debugging_paths_in_roi.npy")
        # print "paths in roi loaded"
        #
        # parallel_array = Parallel(n_jobs=-1) \
        #     (delayed(python_region_features_extractor)( path, feature_volumes,idx)
        #      for idx,path in enumerate(paths_in_roi[:500]))
        #
        #
        # print len(parallel_array)

    # time0=time()

    #
    # print "finished first"
    # time1=time()
    #
    # with futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     tasks = []
    #     for p_id, path in enumerate(paths_in_roi[:500]):
    #         tasks.append(executor.submit(extract_features_for_path, path, feature_volumes, stats))
    #     out2 = np.concatenate([t.result() for t in tasks], axis=0)
    # print "finished second"
    #
    # time2=time()
    #
    # parallel_array = Parallel(n_jobs=-1) \
    #     (delayed(python_region_features_extractor)( path, feature_volumes)
    #      for path in paths_in_roi[:500])
    #
    # time3=time()
    #
    # print "New took ",time1-time0," secs"
    # print "Old took ",time2-time1," secs"
    # print "Newpar took ",time3-time2," secs"

    # parallel_array = Parallel(n_jobs=-1) \
    #     (delayed(python_region_features_extractor)(path, feature_volumes)
    #      for path in paths_in_roi[:500])













    # seg1=deepcopy(seg)
    # seg2=deepcopy(seg)
    # time1=time()
    # procs = []
    #
    # out_q = Queue()
    # uniq = np.unique(seg)
    # for label in uniq:
    #     p = Process(
    #         target=skeletonize_3d,
    #         args=(seg,label
    #                 ))
    #     procs.append(p)
    #     p.start()
    #
    # result1=np.zeros(seg.shape)
    #
    # print "1: ", out_q.size()
    # print "2: ", len(uniq)
    #
    # while out_q.size():
    #     result1[out_q.get()==1]=1

    # result1_unfinished=Parallel(n_jobs=-1)(delayed(worker)(seg1,label) for label in np.unique(seg))
    #
    # result1=np.zeros(seg.shape)
    #
    # for i in result1_unfinished:
    #     result1[i==1]=1
    #
    #
    # time2=time()
    #
    # print "parallel took ", time2-time1, " sec"
    #
    #
    # result2 = linear(seg2)
    #
    # assert (result1==result2).all()
    # time1=time()
    # # result1=extract_paths_and_labels_from_segmentation_single\
    # #     (ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)
    #
    # time2=time()
    #
    # result2=extract_paths_and_labels_from_segmentation\
    #     (ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)
    #
    # time3=time()
    #
    # print "single took ", time2-time1," secs"
    # print "parallel took ", time3-time2," secs"

    #
    # all_paths, paths_to_objs, path_classes, correspondence_list=np.load("/export/home/amatskev/Bachelor/misc/times_test/result.npy")
    # print "starting tests..."
    #
    # result1 = parallel(seg)
    # result2 = linear(seg)
    #
    # assert (result1==result2).all()
    #
    #
    #
    #
    # print "hi"
    # print "hi"























    # all_paths = vigra.readHDF5('/export/home/amatskev/Bachelor/'
    #                                'data/debugging/paths_ds_splB_z1.h5',
    #                                'all_paths')
    #
    # paths_to_objs = vigra.readHDF5('/export/home/amatskev/Bachelor/'
    #                                'data/debugging/paths_ds_splB_z1.h5',
    #                                'paths_to_objs')



    # all_paths_unfinished, paths_to_objs_unfinished,cut_off_array=\
    #     np.load("/export/home/amatskev/Bachelor/data/"
    #             "debugging/cut_off3.npy")
    # #
    # # all_paths_unfinished=all_paths_unfinished[:3]
    # # paths_to_objs_unfinished[:3]
    # # cut_off_array={label:cut_off_array[label] for label in cut_off_array.keys()[:3]}
    #
    #
    #
    #
    #
    #
    #
    #
    # ds, seg, seg_id, gt, correspondence_list, paths_cache_folder = \
    #     np.load("/export/home/amatskev/Bachelor/data/"
    #             "testing_speeds/all_of_them.npy")



    # time5=time()
    #
    #
    # time6=time()
    #
    #
    # time7=time()
    #
    # print "time 1 = ", time6-time5
    # print "time 1 = ", time7 - time6
    #
    # hi=[3,3,3,3,3]
    # func(hi)
    # print hi
    #
    # volume = np.load(
    #     "/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    #
    #
    # time0 = time()
    #
    # results0=false_merges_workflow. \
    #     extract_paths_and_labels_from_segmentation(ds, seg,
    #                                                     seg_id, gt,
    #                                                     correspondence_list,
    #                                                paths_cache_folder)
    # # inputs=range(0,1000)
    #
    # time1=time()
    # print "I took   ", time1 - time0, " secs before"
    #
    # results2=false_merges_workflow.\
    #     extract_paths_and_labels_from_segmentation_julian(ds, seg,
    #                                                seg_id, gt,
    #                                                correspondence_list,
    #                                                       "/export/home/amatskev/Bachelor/data/testing_speeds/paths/julian")
    #
    # time2=time()
    # print "Julian took ", time2-time1, " secs"
    #
    # time0=time()

    # results1l=extract_paths_and_labels_from_segmentation_linear(ds, seg,
    #                                                seg_id, gt,
    #                                                correspondence_list,
    #                                                "/export/home/amatskev/Bachelor/"
    #                                                "data/testing_speeds/paths/alex/linear")
    #
    # time1=time()

    # results1p=extract_paths_and_labels_from_segmentation_parallel(ds, seg,
    #                                                seg_id, gt,
    #                                                correspondence_list,
    #                                                "/export/home/amatskev/Bachelor/"
    #                                                "data/testing_speeds/paths/alex/parallel")
    #
    # time2=time()
    #

    # time3=time()
    # print "I took   ", time1 - time0, " secs before"
    # print "I took   ", time2-time1, " secs now"
    # print "I took   ", time3-time2, " secs now"
    # np.save("/export/home/amatskev/Bachelor/misc/times_test/results0.npy",results0)

    # np.save("/export/home/amatskev/Bachelor/misc/times_test/results1l.npy",results1l)
    # np.save("/export/home/amatskev/Bachelor/misc/times_test/results1p.npy",results1p)
    # np.save("/export/home/amatskev/Bachelor/misc/times_test/results2.npy",results2)

    # assert (results0==results1).all()









    # volume=np.load(
    #     "/export/home/amatskev/Link to neuraldata/test/test_label.npy")

















    # rag = example_rag()
    # node_iteration(rag)











    # volume=np.zeros((250,250,250))
    # volume[75:175,75:175,75:175]=1
    # volume[100:150,100:150,100:150] = 0
    #
    # _,skel=tf.img_to_skel(volume)
    #
    # tf.plot_figure_and_path(volume,skel)
    # diff_vol=np.load("/export/home/amatskev/Link to neuraldata/test/difficult_volume.npy")
    # diff_vol=close_cavities(diff_vol)
    #
    # vol=np.load("/export/home/amatskev/Link to neuraldata/test/difficult_volume_relabeled.npy")
    # vol[vol==2]=0
    # _,skel=tf.img_to_skel(vol)
    #
    # tf.plot_figure_and_path(vol, skel)
    #
    # print "test"

    # faces_unfinished = np.loadtxt("/export/home/amatskev/Bachelor/data/cgal/elephant/elephant_faces.txt")
    #
    # vertices = np.loadtxt("/export/home/amatskev/Bachelor/data/cgal/elephant/elephant_vertices.txt")
    #
    # faces = np.array([[int(e1), int(e2), int(e3)] for (e0, e1, e2, e3) in faces_unfinished])
    #
    #
    # vertices1, faces1_unfinished, normals, values = np.load("/export/home/amatskev/Bachelor/data/first_try/mesh_data.npy")
    #
    # faces1 = np.array([[int(3), int(e1), int(e2), int(e3)] for (e1, e2, e3) in faces1_unfinished])
    #
    #
    #
    # plot_mesh(vertices,faces)

