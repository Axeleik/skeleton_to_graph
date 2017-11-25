from copy import deepcopy
from time import time

# import nifty_with_cplex as nifty
import numpy as np
import vigra
# from workflow.methods.skel_contraction import graph_pruning
from skimage.morphology import skeletonize_3d
from test_functions import plot_figure_and_path
# from workflow.methods.skel_contraction import graph_pruning
#
# from workflow.methods.skel_graph import compute_graph_and_paths


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

def plot_pruned(id,threshhold,finished):
    volume=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    volume[volume!=id]=0
    volume[volume>0]=1

    finished=finished.tolist()

    for_plotting =[finished[key][1]/finished[key][3] for key in finished.keys()]

    range = xrange(0, len(for_plotting))

    for_plotting=sorted(for_plotting)

    # plt.figure()
    # plt.bar(range, for_plotting)
    # plt.show()


    finished_pruned = [finished[key][2] for key in finished.keys() if finished[key][1] / finished[key][3] > threshhold]

    a=finished_pruned[0]
    if len(finished_pruned)>1:
        for i in finished_pruned[1:]:
            a=np.concatenate([a,i])

    a=np.array(a)
    plot_figure_and_path(volume,a)


def pruning_without_space(label):

    print "-----------------------------------------------------------"
    print "Label: ", label
    seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    volume = extract_from_seg(seg, label)

    finished = np.load("/export/home/amatskev/Bachelor/"
                       "data/graph_pruning/finished_{0}.npy".format(label))
    finished = finished.tolist()

    # for_plotting =[finished[key][1]/finished[key][3] for key in finished.keys()]
    #
    # range = xrange(0, len(for_plotting))
    #
    # for_plotting=sorted(for_plotting)
    #
    # plt.figure()
    # plt.bar(range, for_plotting)
    # plt.show()
    threshhold = input("What is the threshhold for the pruning? ")

    # TODO maybe faster ?
    pre_plotting_dict = \
        {key: deepcopy(finished[key][5]) for key in finished.keys()
         if finished[key][1] / finished[key][4] > threshhold}


    counter = 1
    while counter != 0:
        counter = 0
        print pre_plotting_dict.keys()
        for key in pre_plotting_dict.keys():
            print key
            if pre_plotting_dict[key] not in pre_plotting_dict.keys():
                counter = 1

                del pre_plotting_dict[key]

    finished_pruned = np.array([finished[key][2] for key in pre_plotting_dict.keys()])
    terms=[key for key in pre_plotting_dict.keys()]
    a = finished_pruned[0]
    if len(finished_pruned) > 1:
        for i in finished_pruned[1:]:
            a = np.concatenate([a, i])

    a = np.array(a)
    plot_figure_and_path(volume, a)
    return finished_pruned

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

def compute_graph_and_pruning_for_label(where,label):
    seg_test,dt_test=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/seg_and_dt_test.npy")
    img=np.zeros(seg_test.shape)
    img[seg_test==1]=1
    term_list, edges_and_lens, g, nodes=compute_graph_and_paths(img,dt_test,"run",[10,1,1])
    finished_pruning=graph_pruning(g,term_list,edges_and_lens,nodes)

    np.save("/export/home/amatskev/Bachelor/misc/86/finished_test.npy",finished_pruning)

def actual_pruning_and_plotting(volume,finished,threshhold):
    # TODO maybe faster ?
    finished=finished.tolist()
    pre_plotting_dict = \
        {key: deepcopy(finished[key][5]) for key in finished.keys()
         if finished[key][1] / finished[key][4] > threshhold}
    print "Max factor: ", np.max([finished[key][1] / finished[key][4] for key in finished.keys()])
    counter = 1
    while counter != 0:
        counter = 0

        for key in pre_plotting_dict.keys():

            if pre_plotting_dict[key] not in pre_plotting_dict.keys():
                counter = 1

                del pre_plotting_dict[key]

    finished_pruned = np.array([finished[key][2] for key in pre_plotting_dict.keys()])
    if len(finished_pruned)>0:
        a = finished_pruned[0]
        if len(finished_pruned) > 1:
            for i in finished_pruned[1:]:
                a = np.concatenate([a, i])

        a = np.array(a)
        plot_figure_and_path(volume, a)

    else:
        plot_figure_and_path(volume, [],False)

if __name__ == "__main__":
    # compute_graph_and_pruning_for_label(1, 2)
    # seg_test,dt_test=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/seg_and_dt_test.npy")
    # img=np.zeros(seg_test.shape)
    # img[seg_test==1]=1
    #
    # where="/export/home/amatskev/Bachelor/data/graph_pruning/"
    # dict=np.load(where + "finished_86.npy").tolist()
    # # label=86
    # seg_0=np.load(where+"seg_0.npy")
    seg_0,dt_test=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/seg_and_dt_test.npy")

    volume = np.zeros(seg_0.shape)
    volume[seg_0 == 41] = 1
    # a=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/finished_paths_86_test.npy")
    #
    # plot_figure_and_path(volume, a)
    skel_img=skeletonize_3d(volume)

    wher_skel=np.where(skel_img)

    skel_arr=np.zeros((len(wher_skel[0]),3))
    skel_arr[:,0]=wher_skel[0]
    skel_arr[:, 1] = wher_skel[1]
    skel_arr[:, 2] = wher_skel[2]


    plot_figure_and_path(volume, skel_arr,anisotropy_input=[10,1,1])

    # pruning_without_space(86)

    # finished=np.load("/export/home/amatskev/Bachelor/misc/86/finished_86.npy")
    # seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    # volume = extract_from_seg(seg, 86)
    #
    #
    #
    # actual_pruning_and_plotting(volume,finished,56)
    # actual_pruning_and_plotting(volume,finished,40)
    # actual_pruning_and_plotting(volume,finished,44)


# compute_graph_and_pruning_for_label(where,86)