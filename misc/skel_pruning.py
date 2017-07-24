from copy import deepcopy
from time import time

import nifty_with_cplex as nifty
import numpy as np
import vigra
from workflow.methods.skel_contraction import graph_pruning
from skimage.morphology import skeletonize_3d

from workflow.methods.skel_graph import compute_graph_and_paths


def serialize_graph(graph):

    graph_path = '/export/home/amatskev/Bachelor/data/graph_pruning/graph_tmp.h5'

    def write_graph():
        # write the graph to hdf5
        serialization = graph.serialize()
        vigra.writeHDF5(serialization, graph_path, 'data')

    def read_graph():
        serialization = vigra.readHDF5(graph_path, 'data')
        graph_ = nifty.graph.UndirectedGraph()
        graph_.deserialize(serialization)
        return graph_
    write_graph()
    graph_ = read_graph()
    assert graph_.numberOfNodes == graph.numberOfNodes
    assert graph_.numberOfEdges == graph.numberOfEdges
    print "Success"




def read_graph(graph_path):
    serialization = vigra.readHDF5(graph_path, 'data')
    graph_ = nifty.graph.UndirectedGraph()
    graph_.deserialize(serialization)
    return graph_

def adj(g,val):
    for adj_node,adj_edge in g.nodeAdjacency(val):
        print "Node: ",adj_node
        print "Edge: ", adj_edge





                # # if the other branches (except one) are not already at this node
        # if current_node in node_dict.keys():
        #     if len(node_dict[current_node]) != 1:
        #         queue.put([current_node, label])
        #         continue
        #



def bla(seg,label,test):
    test[seg==label]=1
    test[seg!=label]=0
    _,skel=img_to_skel(test)

    plot_figure_and_path(test, skel, True, [1, 1, 10])

def extract_from_seg(seg,label):


    test=deepcopy(seg)
    test[seg!=label]=0
    test[seg == label] = 1
    volume=test

    return volume

def plot_pruned(label):

    print "-----------------------------------------------------------"
    print "Label: ",label
    # seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    # volume = extract_from_seg(seg, label)

    finished=np.load("/export/home/amatskev/Bachelor/"
            "data/graph_pruning/finished_label_{0}.npy".format(label))

    finished=finished.tolist()

    # for_plotting =[finished[key][1]/finished[key][3] for key in finished.keys()]
    #
    # range = xrange(0, len(for_plotting))
    #
    # for_plotting=sorted(for_plotting)
    #
    # plt.figure()
    # plt.bar(range, for_plotting)
    # plt.show()
    while True:

        threshhold=input("What is the threshhold for the pruning? ")

        finished_pruned = np.array([finished[key][2] for key in finished.keys() if finished[key][1] / finished[key][4] > threshhold])

        # a=finished_pruned[0]
        # if len(finished_pruned)>1:
        #     for i in finished_pruned[1:]:
        #         a=np.concatenate([a,i])

        # a=np.array(a)
        plot_figure_and_path(volume,finished_pruned,anisotropy_input=[1,1,10])

def get_finished_paths_for_pruning(label):

    print "-----------------------------------------------------------"
    print "Label: ",label
    seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    dt = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/dt_seg_0.npy")

    volume = extract_from_seg(seg, label)


    time_before_skel = time()
    skel_img = skeletonize_3d(volume)
    time_after_skel = time()
    print "skeletonize_3d took ", time_after_skel \
                                            - time_before_skel, " secs"

    time_before_graph=time()
    term_list, edges, g, nodes = compute_graph_and_paths(skel_img,dt, "testing")
    time_after_graph=time()
    print "compute_graph_and_paths took ", time_after_graph \
                                            - time_before_graph, " secs"


    time_before_pruning=time()
    finished = graph_pruning(g, term_list, edges, dt, nodes)
    time_after_pruning=time()
    print "graph_pruning took ", time_after_pruning-time_before_pruning," secs"
    print "-----------------------------------------------------------"

    np.save("/export/home/amatskev/Bachelor/"
            "data/graph_pruning/finished_label_{0}.npy".format(label),finished)

if __name__ == "__main__":

    # get_finished_paths_for_pruning(140)
    get_finished_paths_for_pruning(281)
    # get_finished_paths_for_pruning(86)
    # get_finished_paths_for_pruning(67)
    # get_finished_paths_for_pruning(71)
    # get_finished_paths_for_pruning(41)



    # plot_pruned(140)
    plot_pruned(281)
    # plot_pruned(86)
    # plot_pruned(67)
    # plot_pruned(71)
    # plot_pruned(41)
    # print "hi"
    # print "hi"

    # a=np.zeros((500,500,500))
    # a[50:450,200:250,100:150]=1
    # a[50:450, 200:250, 300:350] = 1
    # a[100:130, 210:240, 50:400] = 1
    # a[250:280, 210:240, 50:400] = 1
    # a[350:370, 210:240, 280:380] = 1
    # a[390:450, 200:250, 2:280] = 1
    # a[250:280, 210:240, 200:230] = 0
    # img,skel=img_to_skel(a)
    # volume=np.load("/export/home/amatskev/"
    #                       "Bachelor/data/test_volume.npy")
    #

    # test=deepcopy(seg)
    # a = np.array([[len(np.where(seg == label)[0]), label]
    #               for label in np.unique(seg)])

    # skel_img=np.load("/export/home/amatskev/"
    #         "Bachelor/data/test_volume_skel_img.npy")
    #
    # skel=np.load("/export/home/amatskev/"
    #         "Bachelor/data/test_volume_skel.npy")
    #
    # edges=np.load('/export/home/amatskev/Bachelor/data/test_edges.npy')
    # term_list=np.load('/export/home/amatskev/Bachelor/data/test_term_list.npy')
    # nodes=np.load('/export/home/amatskev/Bachelor/data/test_nodes.npy')
    # g=read_graph('/export/home/amatskev/Bachelor/data/test_graph.h5')

    # skel = np.array([[array for array in nodes.tolist()[key]]
    #               for key in (nodes.tolist()).keys()])
    # plt.figure()
    # plt.imshow(skel_img[:,225,:])
    # a=np.array([[358, 225, 325],[359, 225, 326],[359, 225, 325]])
    # plot_figure_and_path(volume,a)
    #
    # #
    # volume=np.load("/export/home/amatskev/"
    #                "Link to neuraldata/test/first_try_volume.npy")
    #
    # skel_img=np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/skel_img.npy")
    #
    # volume=close_cavities(volume)

    # skel_img = skeletonize_3d(volume)
    # #
    # volume = extract_from_seg(seg,86)
    # skel_img=skeletonize_3d(volume)
    # term_list, edges, g=compute_graph_and_paths(skel_img,"testing")
    # #
    # skel = np.array(
    #     [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0],
    #                                         np.where(skel_img)[1],
    #                                         np.where(skel_img)[2])])


    # with open('/export/home/amatskev/Link to neuraldata/'
    #           'test/extract_paths_from_segmentation/input_ds.pkl', mode='r') as f:
    #     input_ds_example=pickle.load(f)
    #

    # term_list   = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/term_list.npy")
    # # edge_lens   = np.load("/export/home/amatskev/"
    # #                       "Bachelor/data/graph_pruning/edge_lens.npy")
    # edges       = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/edges.npy")
    # nodes       = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/nodes.npy")
    # is_node_map = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/is_node_map.npy")
    #
    # g=read_graph('/export/home/amatskev/Bachelor/data/graph_pruning/graph_tmp.h5')
    # finished=np.load("/export/home/amatskev/Bachelor/data/"
    #                  "graph_pruning/finished_without_unique_label_86.npy" )
    # skel=np.load("/export/home/amatskev/Bachelor/data/"
    #              "graph_pruning/skel_label_86.npy")
    # volume=np.load("/export/home/amatskev/Bachelor/"
    #                "data/graph_pruning/volume_label_86.npy")

    # skel_img=np.load("/export/home/amatskev/"
    #                  "Bachelor/data/graph_pruning/skel_img_label_86.npy")



    # finished=np.load("/export/"
    #                         "home/amatskev/Bachelor/data/graph_pruning/"
    #                         "finished_with_ordered_paths_label_86.npy")
    #
    # skel=np.load("/export/"
    #         "home/amatskev/Bachelor/data/"
    #         "graph_pruning/skeleton_paths_seg_0_label_86.npy")

