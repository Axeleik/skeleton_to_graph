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
from workflow.methods.functions_for_workflow import extract_paths_and_labels_from_segmentation
# from multicut_src.false_merges import false_merges_workflow
import vigra




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

def skeletonization(seg,seg_copy,skel_img,seg_id):

    seg_copy[seg == seg_id] = 1
    seg_copy[seg != seg_id] = 0

    skel_dump=skeletonize_3d(seg_copy)

    skel_img[skel_dump==1]=1


def parallel(seg):

    skel_img=np.zeros(seg.shape)

    with futures.ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [executor.submit(skeletonization, seg,copy(seg),skel_img,seg_id) for seg_id in np.unique(seg)]
        results = [t.result() for t in tasks]
    return skel_img


def linear(seg):

    skel_img=np.zeros(seg.shape)

    for seg_id in np.unique(seg):

        skeletonization(seg,copy(seg),skel_img,seg_id)

    return skel_img

def func(hi):
    for i in xrange(0,10):
        hi.extend([i])


if __name__ == "__main__":

    # ds, seg, seg_id, gt, correspondence_list, paths_cache_folder=\
    #     np.load("/mnt/localdata01/amatskev/misc/debugging/for_cut_off.npy")
    #
    # result=extract_paths_and_labels_from_segmentation(ds, seg, seg_id, gt, correspondence_list, paths_cache_folder)

    all_paths, paths_to_objs, path_classes, correspondence_list=np.load("/export/home/amatskev/Bachelor/misc/times_test/result.npy")

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

    time0=time()

    # results1l=extract_paths_and_labels_from_segmentation_linear(ds, seg,
    #                                                seg_id, gt,
    #                                                correspondence_list,
    #                                                "/export/home/amatskev/Bachelor/"
    #                                                "data/testing_speeds/paths/alex/linear")

    time1=time()

    # results1p=extract_paths_and_labels_from_segmentation_parallel(ds, seg,
    #                                                seg_id, gt,
    #                                                correspondence_list,
    #                                                "/export/home/amatskev/Bachelor/"
    #                                                "data/testing_speeds/paths/alex/parallel")

    time2=time()


    time3=time()
    print "I took   ", time1 - time0, " secs before"
    print "I took   ", time2-time1, " secs now"
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

