from copy import deepcopy
from time import time
import h5py

from neuro_seg_plot import NeuroSegPlot as nsp
# import nifty_with_cplex as nifty
import numpy as np
import vigra
import skimage
# from workflow.methods.skel_contraction import graph_pruning
from skimage.morphology import skeletonize_3d
from test_functions import plot_figure_and_path
# from workflow.methods.skel_contraction import graph_pruning
# from workflow.methods.skel_graph import compute_graph_and_paths
from skimage.measure import label
# from joblib import Parallel,delayed
# import matplotlib.pyplot as plt
from math import sqrt

# FIXME AAAAAAHHH THE HORROR
def get_faces_with_neighbors(image):

    # --- XY ---
    # w = x + 2*z, h = y + 2*z
    shpxy = (image.shape[0] + 2 * image.shape[2], image.shape[1] + 2 * image.shape[2])
    xy0 = (0, 0)
    xy1 = (image.shape[2],) * 2
    xy2 = (image.shape[2] + image.shape[0], image.shape[2] + image.shape[1])
    print shpxy, xy0, xy1, xy2

    # xy front face
    xyf = np.zeros(shpxy)
    xyf[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, 0]
    xyf[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[0, :, :]), 0, 1)
    xyf[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(image[-1, :, :], 0, 1)
    xyf[xy1[0]:xy2[0], 0:xy1[1]] = np.fliplr(image[:, 0, :])
    xyf[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = image[:, -1, :]

    # xy back face
    xyb = np.zeros(shpxy)
    xyb[xy1[0]:xy2[0], xy1[1]:xy2[1]] = image[:, :, -1]
    xyb[0:xy1[0], xy1[1]:xy2[1]] = np.swapaxes(image[0, :, :], 0, 1)
    xyb[xy2[0]:shpxy[0], xy1[1]:xy2[1]] = np.swapaxes(np.fliplr(image[-1, :, :]), 0, 1)
    xyb[xy1[0]:xy2[0], 0:xy1[1]] = image[:, 0, :]
    xyb[xy1[0]:xy2[0], xy2[1]:shpxy[1]] = np.fliplr(image[:, -1, :])

    # --- XZ ---
    # w = x + 2*y, h = z + 2*y
    shpxz = (image.shape[0] + 2 * image.shape[1], image.shape[2] + 2 * image.shape[1])
    xz0 = (0, 0)
    xz1 = (image.shape[1],) * 2
    xz2 = (image.shape[1] + image.shape[0], image.shape[1] + image.shape[2])
    print shpxz, xz0, xz1, xz2

    # xz front face
    xzf = np.zeros(shpxz)
    xzf[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, 0, :]
    xzf[0:xz1[0], xz1[1]:xz2[1]] = np.flipud(image[0, :, :])
    xzf[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = image[-1, :, :]
    xzf[xz1[0]:xz2[0], 0:xz1[1]] = np.fliplr(image[:, :, 0])
    xzf[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = image[:, :, -1]

    # xz back face
    xzb = np.zeros(shpxz)
    xzb[xz1[0]:xz2[0], xz1[1]:xz2[1]] = image[:, -1, :]
    xzb[0:xz1[0], xz1[1]:xz2[1]] = image[0, :, :]
    xzb[xz2[0]:shpxz[0], xz1[1]:xz2[1]] = np.flipud(image[-1, :, :])
    xzb[xz1[0]:xz2[0], 0:xz1[1]] = image[:, :, 0]
    xzb[xz1[0]:xz2[0], xz2[1]:shpxz[1]] = np.fliplr(image[:, :, -1])

    # --- YZ ---
    # w = y + 2*x, h = z + 2*x
    shpyz = (image.shape[1] + 2 * image.shape[0], image.shape[2] + 2 * image.shape[0])
    yz0 = (0, 0)
    yz1 = (image.shape[0],) * 2
    yz2 = (image.shape[0] + image.shape[1], image.shape[0] + image.shape[2])
    print shpyz, yz0, yz1, yz2

    # yz front face
    yzf = np.zeros(shpyz)
    yzf[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[0, :, :]
    yzf[0:yz1[0], yz1[1]:yz2[1]] = np.flipud(image[:, 0, :])
    yzf[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = image[:, -1, :]
    yzf[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(np.flipud(image[:, :, 0]), 0, 1)
    yzf[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(image[:, :, -1], 0, 1)

    # yz back face
    yzb = np.zeros(shpyz)
    yzb[yz1[0]:yz2[0], yz1[1]:yz2[1]] = image[-1, :, :]
    yzb[0:yz1[0], yz1[1]:yz2[1]] = image[:, 0, :]
    yzb[yz2[0]:shpyz[0], yz1[1]:yz2[1]] = np.flipud(image[:, -1, :])
    yzb[yz1[0]:yz2[0], 0:yz1[1]] = np.swapaxes(image[:, :, 0], 0, 1)
    yzb[yz1[0]:yz2[0], yz2[1]:shpyz[1]] = np.swapaxes(np.flipud(image[:, :, -1]), 0, 1)

    faces = {
        'xyf': xyf,
        'xyb': xyb,
        'xzf': xzf,
        'xzb': xzb,
        'yzf': yzf,
        'yzb': yzb
    }

    shp = image.shape
    bounds = {
        'xyf': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xyb': np.s_[shp[2]:shp[2] + shp[0], shp[2]:shp[2] + shp[1]],
        'xzf': np.s_[shp[1]:shp[1] + shp[0], shp[1] + 1:shp[1] + shp[2] - 1],
        'xzb': np.s_[shp[1]:shp[1] + shp[0], shp[1] + 1:shp[1] + shp[2] - 1],
        'yzf': np.s_[shp[0] + 1:shp[0] + shp[1] - 1, shp[0] + 1:shp[0] + shp[2] - 1],
        'yzb': np.s_[shp[0] + 1:shp[0] + shp[1] - 1, shp[0] + 1:shp[0] + shp[2] - 1]
    }

    return faces, bounds


def find_centroids(seg, dt, bounds):

    centroids = {}

    for lbl in np.unique(seg[bounds])[1:]:

        # Mask the segmentation
        mask = seg == lbl

        # Connected component analysis to detect when a label touches the border multiple times
        conncomp = vigra.analysis.labelImageWithBackground(
            mask.astype(np.uint32),
            neighborhood=8,
            background_value=0
        )

        # Only these labels will be used for further processing
        # FIXME expose radius as parameter
        opened_labels = np.unique(vigra.filters.discOpening(conncomp.astype(np.uint8), 2))

        # unopened_labels = np.unique(conncomp)
        # print 'opened_labels = {}'.format(opened_labels)
        # print 'unopened_labels = {}'.format(unopened_labels)

        # FIXME why do we use the dt here ?
        # as far as I can see, this only takes the mean of each coordinate anyway
        # -> probably best to take the eccentricity centers instead
        for l in opened_labels[1:]:

            # Get the current label object
            curobj = conncomp == l

            # Get disttancetransf of the object
            cur_dt = dt.copy()
            cur_dt[curobj == False] = 0

            # Detect the global maximum of this object
            amax = np.amax(cur_dt)
            cur_dt[cur_dt < amax] = 0
            cur_dt[cur_dt > 0] = lbl

            # Get the coordinates of the maximum pixel(s)
            coords = np.where(cur_dt[bounds])

            # If something was found
            if coords[0].any():

                # Only one pixel is allowed to be selected
                # FIXME: This may cause a bug if two maximum pixels exist that are not adjacent (although it is very unlikely)
                coords = [int(np.mean(x)) for x in coords]

                if lbl in centroids.keys():
                    centroids[lbl].append(coords)
                else:
                    centroids[lbl] = [coords]

    return centroids


def translate_centroids_to_volume(centroids, volume_shape):
    rtrn_centers = {}

    for orientation, centers in centroids.iteritems():

        if orientation == 'xyf':
            centers = {
                lbl: [center + [0] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xyb':
            centers = {
                lbl: [center + [volume_shape[2] - 1] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xzf':
            centers = {
                lbl: [[center[0], 0, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'xzb':
            centers = {
                lbl: [[center[0], volume_shape[1] - 1, center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'yzf':
            centers = {
                lbl: [[0, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }
        elif orientation == 'yzb':
            centers = {
                lbl: [[volume_shape[0] - 1, center[0], center[1]] for center in centers_in_lbl]
                for lbl, centers_in_lbl in centers.iteritems()
            }

        for key, val in centers.iteritems():
            if key in rtrn_centers:
                rtrn_centers[key].extend(val)
            else:
                rtrn_centers[key] = val

    return rtrn_centers


def compute_border_contacts_old(
        segmentation,
        disttransf,
        return_image=True,
        return_coordinates=False
):
    print "starting computing_border_contacts_old..."
    faces_seg, bounds = get_faces_with_neighbors(segmentation)
    print "get_faces_with_neighbors(segmentation) finished"
    faces_dt, _ = get_faces_with_neighbors(disttransf)
    print "get_faces_with_neighbors(disttransf) finished"

    centroids_coords = {key: find_centroids(val, faces_dt[key], bounds[key]) for key, val in faces_seg.iteritems()}
    print "finished computing_border_contacts_old"

    if return_coordinates and return_image==False:
        return centroids_coords

    centroids = translate_centroids_to_volume(centroids_coords, segmentation.shape)

    if return_image and return_coordinates:
        return centroids,centroids_coords

    else:
        return centroids


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

def compute_graph_and_pruning_for_label(volume=np.array([]),uniq_label=0,anisotropy=[10,1,1]):

    print "Label: ",uniq_label
    img=np.zeros(volume.shape)
    img[volume==uniq_label]=1
    img=close_cavities(img)
    volume=np.uint64(img)
    dt=distance_transform(volume,anisotropy)



    a=compute_graph_and_paths(volume,dt,"run",anisotropy)
    if len(a)==0:
        return [[],[]]
    del dt
    term_list, edges_and_lens, g, nodes=a



    return graph_pruning(g, term_list, edges_and_lens, nodes,mode="not_debug")


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

def norm3d(point1,point2):

    anisotropy = [10, 1, 1]
    return sqrt(((point1[0] - point2[0])* anisotropy[0])*((point1[0] - point2[0])* anisotropy[0])+
                 ((point1[1] - point2[1])*anisotropy[1])*((point1[1] - point2[1])*anisotropy[1])+
                 ((point1[2] - point2[2]) * anisotropy[2])*((point1[2] - point2[2]) * anisotropy[2]))

def shorten_paths(paths_full_raw):
    """for shortening paths so they dont end exactly at the border """

    shortage_ratio=0.2
    if shortage_ratio==0:
        return paths_full_raw

    assert(shortage_ratio<=0.5)

    paths_full=[path for path in paths_full_raw if len(path)>5]

    if len(paths_full)==0:
        return paths_full_raw

    # collects the "length" for every point of every path
    lens_paths = [np.concatenate(([norm3d(path[0], path[1]) / 2],
                                 [norm3d(path[idx - 1], path[idx]) / 2 +
                                  norm3d(path[idx], path[idx + 1]) / 2
                                  for idx, point in enumerate(path)
                                  if idx != 0 and idx != len(path) - 1],
                                 [norm3d(path[-2], path[-1]) / 2]))
                 for path in paths_full]

    overall_lengths=[sum(val) for val in lens_paths]

    shortage_lengths=[overall_length*shortage_ratio for overall_length in overall_lengths]

    indexes_front_back_array=[path_index_shortage_pointss_return(lens_paths[idx],shortage_lengths[idx]) for idx,val in enumerate(lens_paths)]

    paths_cut=[paths_full[idx][indexes_front_back_array[idx][0]:indexes_front_back_array[idx][1]] for idx,val in enumerate(paths_full)]

    return paths_cut


def path_index_shortage_pointss_return(lens_path,shortage_length):
    """returns index point where we cut off our paths for shortage"""


    sum_front=0
    sum_back=0


    for idx,val in enumerate(lens_path):

        sum_front=sum_front+val

        if sum_front >= shortage_length:
            index_front = idx
            break


    for idx, val in enumerate(lens_path[::-1]):

        sum_back = sum_back + val

        if sum_back >= shortage_length:
            index_back = len(lens_path)- 1 - idx
            break


    assert (index_front <= index_back)

    # assuring we dont
    if index_back-index_front<2:
        index_front=index_front-1
        index_back=index_back+1

    assert (index_back < len(lens_path))
    assert (index_front >= 0)

    return index_front,index_back



def close_cavities(volume):
    test_vol=np.zeros(volume.shape)
    test_vol[volume==0]=2
    test_vol[volume==1]=1
    # test_vol=np.pad(test_vol, 1, "constant", constant_values=2)

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


def actual_pruning(uniq_label,volume=[], finished_dict={}, intersecting_node_dict={},
                   nodes_list=[],return_term_list=False,anisotropy=[10,1,1], plot=True, threshhold=None):
    # filename = '/export/home/amatskev/Downloads/groundtruth.h5'
    # f = h5py.File(filename, 'r')
    #
    # test_vol = np.array(f["stack"])
    # if len(test_vol)==0:
    #     test_vol = np.load("/mnt/localdata1/amatskev/"
    #                        "debugging/data/graph_pruning/seg_0.npy")
    #
    # volume=np.zeros(test_vol.shape)
    # volume[test_vol==uniq_label]=1
    if threshhold==None:
        threshhold=input("What is the threshhold? ")
    #Here begins the actual pruning
    # TODO maybe faster ?
    pre_plotting_dict = \
        {key: deepcopy(finished_dict[key][5]) for key in finished_dict.keys()
         if finished_dict[key][1] / finished_dict[key][4] > threshhold}




    counter = 1
    while counter != 0:
        counter = 0
        for key in pre_plotting_dict.keys():
            if pre_plotting_dict[key] not in pre_plotting_dict.keys():
                counter = 1

                del pre_plotting_dict[key]

    pruned_term_list = [key for key in pre_plotting_dict.keys()]



    # if len(pruned_term_list)==1:
    #
    #     key=pruned_term_list[0]
    #
    #     if len(nodes_list.keys())==4:
    #         node = finished_dict[key][0][-2]
    #     else:
    #
    #         node=finished_dict[key][0][-1]
    #     list=np.array(intersecting_node_dict[node])[np.array(intersecting_node_dict[node])!=key]
    #     count=[0,0]
    #     for possible_lbl in list:
    #         possible_ratio=(finished_dict[possible_lbl][1]+finished_dict[key][1])/\
    #                        max(finished_dict[possible_lbl][4],finished_dict[key][4])
    #         if possible_ratio>count[1]:
    #             count[0]=possible_lbl
    #             count[1]=possible_ratio
    #
    #     pruned_term_list.append(count[0])

    if return_term_list==True:
        return pruned_term_list


    finished_pruned = np.array([finished_dict[key][2] for key in pruned_term_list])

    # finished_pruned=shorten_paths(finished_pruned)

    if len(finished_pruned) > 0:
        a = finished_pruned[0]
        if len(finished_pruned) > 1:
            for i in finished_pruned[1:]:
                a = np.concatenate([a, i])

        a = np.array(a)

        if plot==True:

            plot_figure_and_path(volume, a,anisotropy_input=anisotropy)
        else:
            print "path_size: ",len(a)
            return volume,a
    else:
        if plot==True:

            plot_figure_and_path(volume, [], False,anisotropy_input=anisotropy)
        else:
            print "path_size: ",0



def plot_statistics():
    ratio_array=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/statistics.npy")
    arr=deepcopy(np.array(ratio_array))
    # for idx,single_label in enumerate(ratio_array):
    #     max=np.max(single_label)
    #     min=np.min(single_label)
    #     diff=max-min
    #     arr[idx]=arr[idx]-min
    #     arr[idx]=np.divide(arr[idx],diff)
    a=arr[0]
    if len(arr)>1:
        for bla in arr[1:]:
            a=np.concatenate([a,bla])

    b=[val[0] for val in a]
    b=np.array(b)
    plt.hist(b, bins=100, normed=True)



    print "hi"



# Just keep the biggest branch
def largest_lbls(im, n):
    im_unique, im_counts = np.unique(im, return_counts=True)
    largest_ids = np.argsort(im_counts)[-n - 1:-1]
    return im_unique[largest_ids][::-1]


def compute_path_length(paths):
        # pathlen = 0.
        # for i in xrange(1, len(path)):
        #    add2pathlen = 0.
        #    for j in xrange(0, len(path[0, :])):
        #        add2pathlen += (anisotropy[j] * (path[i, j] - path[i - 1, j])) ** 2

        #    pathlen += add2pathlen ** (1. / 2)
        # TODO check that this actually agrees
        len_paths = [np.concatenate(([norm3d(path[0], path[1]) / 2],
                                     [norm3d(path[idx - 1], path[idx]) / 2 +
                                      norm3d(path[idx], path[idx + 1]) / 2
                                      for idx, point in enumerate(path)
                                      if idx != 0 and idx != len(path) - 1],
                                     [norm3d(path[-2], path[-1]) / 2]))
                     for path in paths]
        return len_paths




def extract_graph_and_distance_features(distance_map):
    shape = distance_map.shape
    grid_graph = nifty.graph.undirectedGridGraph(shape)
    distance_features = grid_graph.imageToEdgeMap(distance_map.astype('float32'), 'max')

    # if you want to use the graph in the shortest paths functions,
    # you might need to transfer it to a `normal` undirected graph, like so
    graph = nifty.graph.UndirectedGraph(grid_graph.numberOfNodes)
    graph.insertEdges(grid_graph.uvIds())

    # just a sanity check
    assert graph.numberOfEdges == len(distance_features)
    return graph, distance_features




import numpy as np
import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




if __name__ == "__main__":

    # paths_bla=vigra.readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
    #                          "renamed_and_stiched/paths_block1/paths_ds_block1.h5","all_paths")
    # paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths_bla])
    #
    # paths_to_objs=vigra.readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
    #                          "renamed_and_stiched/paths_block1/paths_ds_block1.h5","paths_to_objs")
    #
    #
    #
    # np.save("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
    #                          "renamed_and_stiched/paths_block1/paths.npy",(paths,paths_to_objs))






    paths,paths_to_objs=np.load("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
                             "renamed_and_stiched/paths_block1/paths.npy")


    gt=vigra.readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
                      "renamed_and_stiched/gt/gt_x_5000_5520_y_2000_2520_z_3480_4000.h5","data")


    init_seg=vigra.readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
                      "renamed_and_stiched/init_seg_alba/initResult_x_5000_5520_y_2000_2520_z_3480_4000.h5","data")

    res_seg=vigra.readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/"
                           "renamed_and_stiched/oracle_local_alba/oracleLocal_x_5000_5520_y_2000_2520_z_3480_4000.h5","data")
    #
    # fig, ax = plt.subplots(1, 1)
    #
    #
    # tracker = IndexTracker(ax, init_seg)
    #
    # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    # plt.show()
    # #
    #
    #
    # assert 1==0
    label=25

    res_seg[init_seg!=label]=0
    gt[init_seg!=label]=0
    init_seg[init_seg!=label]=0

    gt_relabeled=np.zeros(init_seg.shape)
    res_relabeled=np.zeros(init_seg.shape)

    for idx,l in enumerate(largest_lbls(res_seg, 5)):
        print "l: ",l
        res_relabeled[res_seg == l] = idx+1

    for idx,l in enumerate(largest_lbls(gt, 5)):
        print "l: ",l
        gt_relabeled[gt == l] = idx+1


    nsp.start_figure()
    nsp.add_iso_surfaces(init_seg, anisotropy=[1, 1, 1], vmin=0, vmax=np.max(init_seg), opacity=0.3)
    # nsp.add_path(paths.swapaxes(0, 1), anisotropy=[10, 1, 1], representation="points", line_width=100)
    nsp.show()

    nsp.start_figure()
    nsp.add_iso_surfaces(gt_relabeled, anisotropy=[1, 1, 1], vmin=0, vmax=np.max(gt_relabeled), opacity=0.3)
    nsp.show()

    nsp.start_figure()
    nsp.add_iso_surfaces(res_relabeled, anisotropy=[1, 1, 1], vmin=0, vmax=np.max(res_relabeled), opacity=0.3)
    nsp.show()






    #
    # img=np.load("/mnt/localdata1/amatskev/debugging/testimg.npy")
    #
    # skel_img=skeletonize_3d(img)
    # wher_skel=np.where(skel_img)
    # skel_arr=np.zeros((len(wher_skel[0]),3))
    # skel_arr[:,0]  = wher_skel[0]
    # skel_arr[:, 1] = wher_skel[1]
    # skel_arr[:, 2] = wher_skel[2]
    # plot_figure_and_path(img,skel_arr,anisotropy_input=[1,1,1])
    #
    #
    #



    assert 1==2
    nsp.start_figure()
    nsp.add_iso_surfaces(img, anisotropy=[1, 1, 1], vmin=0, vmax=np.max(img), opacity=0.3)
    # nsp.add_path(paths.swapaxes(0, 1), anisotropy=[10, 1, 1], representation="points", line_width=100)
    nsp.show()

    # a = 6
    # seg_0 = np.zeros((a * 100, a * 100, a * 100))
    # seg_0[a * 10:a * 20, a * 10:a * 90, a * 47:a * -47] = 1
    # seg_0[a * 80:a * 90, a * 10:a * 90, a * 47:a * -47] = 1
    # seg_0[a * 40:a * 60, a * 10:a * 90, a * 40:a * 60] = 1
    # seg_0[a*45:a*55,a*45:a*55,a*45:a*55]=0
    # seg_0[a * 10:a * 90, a * 10:a * 20, a * 47:a * -47] = 1
    # seg_0[a * 10:a * 90, a * 80:a * 90, a * 47:a * -47] = 1
    # # seg_0[40:-40, 40:-40, 10:-10] = 1
    # test_vol=np.load("/mnt/localdata1/amatskev/"
    #                  "debugging/data/graph_pruning/seg_0.npy")
    # # filename = '/export/home/amatskev/Downloads/groundtruth.h5'
    # # f = h5py.File(filename, 'r')
    # #
    # # test_vol = np.array(f["stack"])
    # # a=np.unique(test_vol,return_counts=True)
    # uniq_label=2148
    # test_vol[test_vol!=uniq_label]=0
    # print np.unique(test_vol,return_counts=True)
    #
    # # #
    # # # wher=np.where(a[0]==uniq_label)
    # # # print a[1][wher]
    # # #
    # finished_dict, intersecting_node_dict, nodes_list=np.load("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(uniq_label))

    filename = '/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/splB_z1/'

    finished_dict, intersecting_node_dict, nodes_list=np.load(filename+"finished_44.npy")



    ref_seg=np.load(filename+"ref_seg.npy")
    res_seg=np.load(filename+"res_seg.npy")
    gt_saved=np.load(filename+"res_seg.npy")

    mask=ref_seg!=44


    ref_seg[mask]=0
    res_seg[mask]=0
    gt_saved[mask]=0

    ref_seg[ref_seg==44]=1

    # volume,paths=actual_pruning(1,[],finished_dict,intersecting_node_dict,
    #                     nodes_list,return_term_list=False,anisotropy=[10,1,1],plot=False)

    a=compute_graph_and_pruning_for_label(ref_seg,1,anisotropy=[10,1,1])
    np.load(filename+"finished_44.npy")
    res_unified=np.zeros(res_seg.shape)
    gt_unified=np.zeros(res_seg.shape)

    for idx,l in enumerate(largest_lbls(res_seg, 6)):
        print "l: ",l
        res_unified[res_seg == l] = idx+1

    for idx,l in enumerate(largest_lbls(gt_saved, 6)):
        print "l: ",l
        gt_unified[gt_saved == l] = idx+1

    nsp.start_figure()
    nsp.add_iso_surfaces(ref_seg, anisotropy=[10, 1, 1], vmin=0, vmax=np.max(ref_seg), opacity=0.3)
    # nsp.add_path(paths.swapaxes(0, 1), anisotropy=[10, 1, 1], representation="points", line_width=100)
    nsp.show()

    nsp.start_figure()
    nsp.add_iso_surfaces(res_unified, anisotropy=[10, 1, 1], vmin=0, vmax=np.max(res_unified), opacity=0.3)
    nsp.show()

    nsp.start_figure()
    nsp.add_iso_surfaces(gt_unified, anisotropy=[10, 1, 1], vmin=0, vmax=np.max(gt_unified), opacity=0.3)
    nsp.show()













    # f = h5py.File(filename, 'r')
    #
    # test_vol = np.array(f["stack"])
    #
    # test_vol[test_vol!=uniq_label]=0
    # test_vol[test_vol == uniq_label] = 1
    #
    # volume,paths=actual_pruning(uniq_label,test_vol,finished_dict,intersecting_node_dict,
    #                     nodes_list,return_term_list=False,anisotropy=[1,1,1],plot=False)
    #
    #
    #
    # nsp.start_figure()
    # nsp.add_iso_surfaces(volume, anisotropy=[1, 1, 1], vmin=0, vmax=1, opacity=0.3)
    # nsp.add_path(paths.swapaxes(0, 1), anisotropy=[1, 1, 1], representation="points", line_width=100)
    # nsp.movie_show()
    #
    # x=input("FINISHED")

    #
    #
    #
    # for uniq_label in [1957]:
    #     print "uniq_label: ",uniq_label
    #     # wher = np.where(a[0] == uniq_label)
    #     # print "size:", a[1][wher][0]
    #     finished_dict, intersecting_node_dict, nodes_list=np.load("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(uniq_label))
    #
    #     for threshhold in [6.5]:
    #         print "threshold: ",threshhold
    #         actual_pruning(uniq_label,finished_dict,intersecting_node_dict,
    #                        nodes_list,return_term_list=False,anisotropy=[1,1,1],plot=True,threshhold=threshhold)
    # # #

    # f = h5py.File("/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5", 'r')
    # results = np.array(f["z/1/labels"])
    # # uniq=np.unique(results,return_counts=True)
    #
    # length_half=int(len(uniq[0])/18)
    # print length_half
    # while True:
    #     ids_sorted=uniq[0][np.argsort(uniq[1])[::-1]]
    #     counts_sorted=uniq[1][np.argsort(uniq[1])[::-1]]
    #     print ids_sorted[:20]
    #     print counts_sorted[:20]
    #
    #     label= int(input("Which label?"))
    #
    #     to_show=np.zeros(results.shape,dtype="uint8")
    #     to_show[results==label]=1
    #
    #     skel_img=skeletonize_3d(to_show)
    #     wher_skel=np.where(skel_img)
    #     skel_arr=np.zeros((len(wher_skel[0]),3))
    #     skel_arr[:,0]  = wher_skel[0]
    #     skel_arr[:, 1] = wher_skel[1]
    #     skel_arr[:, 2] = wher_skel[2]
    #
    #     plot_figure_and_path(to_show, skel_arr)
    #
    #
    # key_for_plot = 230
    # dataset="splC_z0"
    #
    #
    # # finished_dict, intersecting_node_dict, nodes_list=\
    # #     np.load("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(key))
    # #
    # # filename1 = "/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/splB_z1/result.h5"
    # filename2 = "/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/"+ dataset+"/gt_saved.npy"
    #
    # # filename3 = "/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/splB_z1/result_resolved_local.h5"
    # # actual_pruning(uniq_label, finished_dict, intersecting_node_dict,nodes_list)
    #                # #
    # #
    # #
    # #
    # project_folder="/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/"
    # wichtig=np.load("/mnt/localdata1/amatskev/debugging/bac/pruning_4_0.2/a.npy")
    # a=np.concatenate(wichtig[:,0])
    # b=np.concatenate(wichtig[:,1])
    #
    # a=a[np.argsort(b)[::-1]]
    # b=b[np.argsort(b)[::-1]]
    #
    # # f = h5py.File(filename1, 'r')
    # # results = np.array(f["z/1/data"])
    # # h = h5py.File(filename1, 'r')[0][:3916]
    # # results_res = np.array(h["z/1/data"])
    # results=np.load(project_folder + dataset+"/ref_seg.npy")
    # results_res=np.load(project_folder + dataset+ "/res_seg.npy")
    # map_key=465
    #
    # # a=compute_graph_and_pruning_for_label(results,map_key,anisotropy=[10,1,1])
    # # np.save("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(20),a)
    # # finished_dict, intersecting_node_dict, nodes_list=\
    # #     np.load("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(20))
    # results[results!=map_key]=0
    # results[results==map_key]=1
    # # actual_pruning(map_key,results,finished_dict,intersecting_node_dict,
    # #                        nodes_list,return_term_list=False,anisotropy=[10,1,1],plot=True)
    #
    #
    #
    # mapping=np.load(project_folder + dataset+"/mapping.npy").tolist()
    # map_key=41
    # paths,objs,preds=np.load(project_folder + dataset+"/paths_saved.npy")
    #
    # a=np.array(mapping.values())
    # wher=np.where(a==map_key)[0][0]
    # key=mapping.keys()[wher]
    #
    # paths=paths[objs==key]
    #
    # preds=preds[objs==key]
    #
    # aniso_temp = np.array([10,1,1])
    # path_lengths_u=np.array(compute_path_length(paths))
    # path_lengths=[sum(test) for test in path_lengths_u]
    # sorted=np.sort(path_lengths,axis=0)
    # import matplotlib.pyplot as plt
    # plt.plot(sorted)
    # font = {'family': 'serif',
    #         'color': 'black',
    #         'weight': 'normal',
    #         'size': 22,
    #         }
    # # add some text for labels, title and axes ticks
    # plt.ylabel('Length of paths', fontdict=font)
    # plt.xlabel('Path number', fontdict=font)
    # plt.tick_params(axis='both', which='major', labelsize=18)
    #
    # plt.title("Object 41, dataset 3", font)
    # plt.show()
    #
    #
    # paths=paths[preds>0.4]
    # preds=preds[preds>0.4]
    # # # paths = paths[preds > 0.3]
    # # # preds = preds[preds > 0.3]
    # # paths=paths[[37,51,124,126,194,206,215,245,271,296,320,338,365,371]]
    # print len(paths)
    # if len(paths) > 0:
    #     a = paths[0]
    #     if len(paths) > 1:
    #         for i in paths[1:]:
    #             a = np.concatenate([a, i])
    #
    #     a = np.array(a)
    # # results,_,_=vigra.analysis.relabelConsecutive(results, start_label=0, keep_zeros=False)
    # gt = np.load(filename2)
    #
    # # results[results!=map_key]=0
    # # results[results==map_key]=1
    #
    # results_res[results!=1]=0
    # gt_cop=gt
    # gt_cop[results!=1]=0
    #
    # gt_new=np.zeros(gt_cop.shape)
    #
    # for idx,l in enumerate(largest_lbls(gt_cop, 12)):
    #     gt_new[gt_cop == l] = idx + 1
    # gt_new[gt_new==2]=15
    # # actual_pruning(20,results_res,finished_dict,intersecting_node_dict,nodes_list)
    #
    #
    # # results,_,_=vigra.analysis.relabelConsecutive(results, start_label=0, keep_zeros=False)
    # # a=compute_graph_and_pruning_for_label(results,20,anisotropy=[10,1,1])
    # # np.save("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(20),a)
    #
    # # gt = np.load(filename2)
    # #
    # # results[results!=key]=0
    # # results[results==key]=1
    #
    # # gt_cop=gt
    # # gt_cop[results!=1]=0
    # #
    # # gt_new=np.zeros(gt_cop.shape)
    # #
    # # for idx,l in enumerate(largest_lbls(gt_cop, 10)):
    # #     print l
    # #     gt_new[gt_cop == l] = idx + 1
    # #
    # #
    # plot_figure_and_path(results,[],plot_path=False)
    # plot_figure_and_path(gt_new,[],plot_path=False)
    # #
    # plot_figure_and_path(results_res,[],plot_path=False)
    # #
    # #
    # # print "hi"
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # # sorted_labels=a[0][np.argsort(a[1])[::-1]]
    # # sorted_counts=a[1][np.argsort(a[1])[::-1]]
    # # # seg_0=np.zeros(test_vol.shape)
    # # uniq_label=sorted_labels[701]
    # # # print uniq_label
    # # seg_0[test_vol==uniq_label]=1
    # # a=compute_graph_and_pruning_for_label(test_vol,uniq_label,anisotropy=[1,1,1])
    # # np.save("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(uniq_label),a)
    #
    #
    #
    # # skel_img=skeletonize_3d(seg_0)
    # # wher_skel=np.where(skel_img)
    # # skel_arr=np.zeros((len(wher_skel[0]),3))
    # # skel_arr[:,0]=wher_skel[0]
    # # skel_arr[:, 1] = wher_skel[1]
    # # skel_arr[:, 2] = wher_skel[2]
    # # #
    # # plot_figure_and_path(seg_0,skel_arr,anisotropy_input=[1,1,1])
    # #
    # #
    # # uniq_label=2148
    #
    #
    # # filename = '/export/home/amatskev/Downloads/groundtruth.h5'
    # # f = h5py.File(filename, 'r')
    # #
    # # test_vol = np.array(f["stack"])
    #
    # # test_vol=np.load("/mnt/localdata1/amatskev/"
    # #                  "debugging/data/graph_pruning/seg_0.npy")
    #
    # #
    # # # a=np.unique(test_vol,return_counts=True)
    # # # sorted_labels=a[0][np.argsort(a[1])[::-1]]
    # # # print sorted_labels
    # #
    # # a=compute_graph_and_pruning_for_label(test_vol,uniq_label,anisotropy=[1,1,10])
    # # np.save("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(uniq_label),a)
    # # finished_dict, intersecting_node_dict, nodes_list=np.load("/export/home/amatskev/Bachelor/misc/objects/label_{}_finished.npy".format(uniq_label))
    # #
    # # actual_pruning(uniq_label,finished_dict,intersecting_node_dict,nodes_list,return_term_list=False,anisotropy=[1,1,1])
    #
    #
    #





















    # from diced import DicedStore
    #
    # store = DicedStore("gs://flyem-public-connectome")
    # # open repo with version id or repo name
    # repo = store.open_repo("3630f935d9944fe098770c57c462427f")
    # my_array = repo.get_array("pb1-groundtruth")
    # plot_statistics()
    # seg = np.array(vigra.readHDF5('/mnt/localdata0/amatskev/neuraldata/results/'
    #                           'splA_z0/result.h5', "z/0/data"))
    # dt = distance_transform(seg, [10., 1., 1.])
    #
    # # centres_dict = compute_border_contacts_old(seg, dt)
    # # uniq_border_contacts = [key for key in centres_dict.keys()
    # #                         if len(centres_dict[key]) > 1]
    # uniq_border_contacts=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/statistics_border_uniq.npy")
    #
    # for uniq_label in uniq_border_contacts:
    #     mask=seg==uniq_label
    #     print uniq_label
    #     a=dt[mask]
    #     print "mean: ",np.mean(a)
    #     print "variance: ", np.var(a)
    #     print "std: ", np.std(a)
    #     plt.hist(a, bins=100, normed=True)
    #     plt.show()
    #
    # print "hi"

    # # test=deepcopy(seg)
    # uniq_len=len(uniq_border_contacts)-1
    # # a = compute_graph_and_pruning_for_label(test, 0)
    # # parallel_array=[compute_graph_and_pruning_for_label(seg,uniq_label,idx,uniq_len) for idx,uniq_label in enumerate(uniq)]
    #
    # parallel_array = Parallel(n_jobs=32) \
    #     (delayed(compute_graph_and_pruning_for_label)(seg,dt,uniq_label,idx,uniq_len)
    #      for idx, uniq_label in enumerate(uniq_border_contacts))
    #
    #
    # np.save("/mnt/localdata1/amatskev/debugging/data/graph_pruning/statistics.npy",parallel_array)
    # a=np.unique(seg)[:5]
    # arr=[]
    # img = np.zeros(seg.shape)
    # img[seg == 1] = 1
    # img[seg != 1] = 2
    # b = label(img)
    # for uniq_label in a:
    #     print "label: ", label
    #     img = np.zeros(seg.shape)
    #     img[seg == 1] = 1
    #     img[seg != 1] = 2
    #     b = label(img)
    #     img = np.zeros(seg.shape)
    #     img[seg == uniq_label] = 1
    #     img[seg!=uniq_label]=2
    #     a=label(img)
    #     img = close_cavities(seg)
    #     volume = np.uint64(seg)
    #
    #     dt = distance_transform(img, [10., 1., 1.])
    #     a = compute_graph_and_paths(img, dt, "run", [10, 1, 1])
    #     if len(a) == 0:
    #         continue
    #     del dt
    #     term_list, edges_and_lens, g, nodes = a
    #
    #     arr.append(graph_pruning(g, term_list, edges_and_lens, nodes))
    # np.save("/mnt/localdata1/amatskev/debugging/data/graph_pruning/wtf.npy",arr)

        # actual_pruning_and_plotting
    #
    # label = 239
    # print "label: ", label
    # img = np.zeros(seg_test.shape)
    # img[seg_test == label] = 1
    #
    # # compute_graph_and_pruning_for_label()
    # finished_dict, intersecting_node_dict, nodes_list=np.load("/export/home/amatskev/Bachelor/misc/finished_question.npy")
    # # # finished_dict
    # # # intersecting_node_dict=intersecting_node_dict.tolist()
    # term_list=actual_pruning(finished_dict, intersecting_node_dict, nodes_list)
    # # # seg_test,dt_test=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/seg_and_dt_test.npy")
    # # # img=np.zeros(seg_test.shape)
    # # # img[seg_test==1]=1
    # finished_pruned = np.array([finished_dict[key][2] for key in term_list])
    # if len(finished_pruned)>0:
    #     a = finished_pruned[0]
    #     if len(finished_pruned) > 1:
    #         for i in finished_pruned[1:]:
    #             a = np.concatenate([a, i])
    #
    # a = np.array(a)


    # skel_img=skeletonize_3d(img)
    # wher_skel=np.where(skel_img)
    # skel_arr=np.zeros((len(wher_skel[0]),3))
    # skel_arr[:,0]=wher_skel[0]
    # skel_arr[:, 1] = wher_skel[1]
    # skel_arr[:, 2] = wher_skel[2]
    #
    # plot_figure_and_path(img,skel_arr, anisotropy_input=[1,1,1])
    #
    # a=6
    # seg_0=np.zeros((a*100,a*100,a*100))
    # seg_0[  a*10:a*20,    a*10:a*90,a*47:a*-47]=1
    # seg_0[a*80:a*90,  a*10:a*90,a*47:a*-47]=1
    # seg_0[a*40:a*60, a*10:a*90, a*40:a*60] = 1
    # # seg_0[a*45:a*55,a*45:a*55,a*45:a*55]=0
    # seg_0[a*10:a*90,  a*10:a*20, a* 47:a*-47]=1
    # seg_0[a*10:a*90, a*80:a*90, a*47:a*-47] = 1
    # seg_0[40:-40, 40:-40, 10:-10] = 1
    # # from skimage.measure import label
    # # # volume=seg_0
    # # #
    # # # seg1=np.load("/mnt/localdata1/amatskev/debugging/funny_shit1.npy" )
    # # # seg=np.load("/mnt/localdata1/amatskev/debugging/funny_shit.npy" )
    # #
    # where="/mnt/localdata1/amatskev/debugging/data/graph_pruning/"
    # # seg_0=np.load(where+"seg_0.npy")
    # seg_0 = vigra.readHDF5('/mnt/localdata1/amatskev/neuraldata/results2/'
    #                        'splA_z0/result.h5', "z/0/data")
    # volume = np.zeros(seg_0.shape)
    # volume[seg_0 == 239] = 1
    #
    # volume=np.load("/mnt/localdata1/amatskev/debugging/funny_shit.npy")
    #
    # # volume[:,:,:500]=0
    # # # volume[:, :, :250] = 0
    # # volume[:,230:, : ] = 0
    # # volume[:, :130, :] = 0
    # # volume[15:, :, :] = 0
    # # a=label(volume)
    #
    # """239.0
    # min:  0 max:  61
    # len:  1540645
    # Label nr.  200  of  204
    # no paths found for label  790.0
    # min:  33 max:  61
    # len:  94654
    # no paths found for label  657.0
    # min:  26 max:  61
    # len:  342958
    # no paths found for label  1124.0
    # """
    #
    #
    # # volume=close_cavities(volume)
    #
    # #
    # #
    # # seg_0=np.load("/mnt/localdata1/amatskev/debugging/data/"
    # #                  "graph_pruning/debugging/seg_and_dt_test.npy")
    # # skel_img=np.int64(np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/debugging/spooky/skel_seg.npy"))
    #
    #
    # # a=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/finished_paths_86_test.npy")
    # # wher_vol=np.where(volume)
    # # print "min: ",min(wher_vol[0]),"max: ",max(wher_vol[0])
    # # print "len: " ,len(wher_vol[0])
    #
    # # seg_0 = np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/seg_0.npy")
    #
    # # volume = np.zeros(seg_0.shape)
    # # volume[seg_0 == 0] = 1
    # # # a=np.load("/export/home/amatskev/Bachelor/data/graph_pruning/finished_paths_86_test.npy")
    # # wher_vol = np.where(volume)
    # # print "min: ", min(wher_vol[2]), "max: ", max(wher_vol[2])
    # # print "len: ", len(wher_vol[2])
    # plot_figure_and_path(volume,plot_path=False)
    # # expand image as the skeletonization sees the image border as a border
    # expand_number = 10
    # # skeletonize
    # skel_img = skeletonize_3d(volume)
    #
    # # skel_img[1000:,:,:]=0
    # # volume[1000:,:,:]=0
    #
    # wher_skel=np.where(skel_img)
    #
    # skel_arr=np.zeros((len(wher_skel[0]),3))
    # skel_arr[:,0]=wher_skel[0]
    # skel_arr[:, 1] = wher_skel[1]
    # skel_arr[:, 2] = wher_skel[2]
    #
    # plot_figure_and_path(volume, skel_arr,anisotropy_input=[10,1,1],opacity=0.3)

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