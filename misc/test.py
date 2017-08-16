import numpy as np
# from mesh_test import plot_mesh
# import test_functions as tf
# from neuro_seg_plot import NeuroSegPlot as nsp
from skimage.measure import label
import nifty_with_cplex as nifty
import nifty_with_cplex.graph.rag as nrag
from concurrent import futures
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
    import extract_paths_and_labels_from_segmentation,\
    extract_paths_and_labels_from_segmentation_single
# from multicut_src.false_merges import false_merges_workflow
import vigra
from joblib import parallel,delayed
from multiprocessing import Process
from Queue import Queue
import platform
from joblib import Parallel, delayed
import cPickle as pickle



def close_cavities(volume):
    volume[volume==0]=2
    lab=label(volume)
    if len(np.unique(lab))==2:
        return volume
    count,what=0,0
    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    volume[lab==what]=0
    volume[lab != what] = 1

    return volume

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

if __name__ == '__main__':




    ds, seg, seg_id, gt, correspondence_list, paths_cache_folder = \
        np.load("/mnt/localdata01/amatskev/misc/debugging/for_cut_off.npy")

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
    time1=time()
    # result1=extract_paths_and_labels_from_segmentation_single\
    #     (ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)

    time2=time()

    result2=extract_paths_and_labels_from_segmentation\
        (ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)

    time3=time()

    print "single took ", time2-time1," secs"
    print "parallel took ", time3-time2," secs"

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
    print "hi"
    print "hi"























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

