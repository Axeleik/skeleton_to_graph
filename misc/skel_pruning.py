from copy import deepcopy
from time import time

# import nifty_with_cplex as nifty
import numpy as np
import vigra
import skimage
# from workflow.methods.skel_contraction import graph_pruning
from skimage.morphology import skeletonize_3d
from test_functions import plot_figure_and_path
# from workflow.methods.skel_contraction import graph_pruning,actual_pruning
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

# calculate the distance transform for the given segmentation
def distance_transform(segmentation, anisotropy):
    edge_volume = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(segmentation[z])[None, :]
         for z in xrange(segmentation.shape[0])],
        axis=0
    )
    dt = vigra.filters.distanceTransform(edge_volume.astype('uint32'), pixel_pitch=anisotropy, background=True)
    return dt


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

def compute_graph_and_pruning_for_label():

    seg_test=vigra.readHDF5('/mnt/localdata1/amatskev/debugging/result.h5', "z/1/data")
    for label in np.unique(seg_test)[18:]:
        print "label: ", label
        img=np.zeros(seg_test.shape)
        img[seg_test==label]=1
        img=np.uint64(img)
        dt = distance_transform(img, [10., 1., 1.])
        a=compute_graph_and_paths(img,dt,"run",[10,1,1])
        if len(a)==0:
            continue
        term_list, edges_and_lens, g, nodes=a
        if len(term_list)<3:
            continue
        finished_dict, intersecting_node_dict, nodes_list=\
            graph_pruning(g,term_list,edges_and_lens,nodes)

    np.save("/export/home/amatskev/Bachelor/misc/finished_question.npy",(finished_dict, intersecting_node_dict, nodes_list))

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

def close_cavities(volume):
    test_vol=np.zeros(volume.shape)
    test_vol[volume==0]=2
    test_vol[volume==1]=1
    test_vol=np.pad(test_vol, 1, "constant", constant_values=2)

    lab=label(test_vol)
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

if __name__ == "__main__":
    # compute_graph_and_pruning_for_label()
    # finished_dict, intersecting_node_dict, nodes_list=np.load("/export/home/amatskev/Bachelor/misc/finished_question.npy")
    # finished_dict
    # intersecting_node_dict=intersecting_node_dict.tolist()
    # actual_pruning(finished_dict, intersecting_node_dict, nodes_list)
    # seg_test,dt_test=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/seg_and_dt_test.npy")
    # img=np.zeros(seg_test.shape)
    # img[seg_test==1]=1
    # a=6
    # seg_0=np.zeros((a*100,a*100,a*100))
    # seg_0[  a*10:a*20,    a*10:a*90,a*47:a*-47]=1
    # seg_0[a*80:a*90,  a*10:a*90,a*47:a*-47]=1
    # seg_0[a*40:a*60, a*10:a*90, a*40:a*60] = 1
    # # seg_0[a*45:a*55,a*45:a*55,a*45:a*55]=0
    # seg_0[a*10:a*90,  a*10:a*20, a* 47:a*-47]=1
    # seg_0[a*10:a*90, a*80:a*90, a*47:a*-47] = 1
    # # seg_0[40:-40, 40:-40, 10:-10] = 1
    from skimage.measure import label
    # volume=seg_0
    #
    where="/mnt/localdata1/amatskev/debugging/data/graph_pruning/"
    # seg_0=np.load(where+"seg_0.npy")
    seg_0 = vigra.readHDF5('/mnt/localdata1/amatskev/neuraldata/results2/'
                           'splC_z1/result.h5', "z/1/data")
    volume = np.zeros(seg_0.shape)
    volume[seg_0 == 141] = 1
    # volume[:,:,400:]=0
    # volume[:, :, :250] = 0
    # volume[:,500:, : ] = 0
    # volume[:, :250, :] = 0
    # volume[7:, :, :] = 0
    # a=label(volume)

    volume=close_cavities(volume)

    #
    #
    # seg_0=np.load("/mnt/localdata1/amatskev/debugging/data/"
    #                  "graph_pruning/debugging/seg_and_dt_test.npy")
    # skel_img=np.int64(np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/spooky/skel_seg.npy"))


    # a=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/finished_paths_86_test.npy")
    # wher_vol=np.where(volume)
    # print "min: ",min(wher_vol[0]),"max: ",max(wher_vol[0])
    # print "len: " ,len(wher_vol[0])

    # seg_0 = np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/seg_0.npy")

    # volume = np.zeros(seg_0.shape)
    # volume[seg_0 == 0] = 1
    # # a=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/finished_paths_86_test.npy")
    # wher_vol = np.where(volume)
    # print "min: ", min(wher_vol[2]), "max: ", max(wher_vol[2])
    # print "len: ", len(wher_vol[2])
    # # plot_figure_and_path(volume, a)
    # expand image as the skeletonization sees the image border as a border
    expand_number = 10
    # skeletonize
    skel_img = skeletonize_3d(volume)

    # skel_img[1000:,:,:]=0
    # volume[1000:,:,:]=0

    wher_skel=np.where(skel_img)

    skel_arr=np.zeros((len(wher_skel[0]),3))
    skel_arr[:,0]=wher_skel[0]
    skel_arr[:, 1] = wher_skel[1]
    skel_arr[:, 2] = wher_skel[2]

    plot_figure_and_path(volume, skel_arr,anisotropy_input=[10,1,1],opacity=0.3)

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