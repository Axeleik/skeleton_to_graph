from copy import deepcopy
from time import time

# import nifty_with_cplex as nifty
import numpy as np
import vigra
import skimage
# from workflow.methods.skel_contraction import graph_pruning
from skimage.morphology import skeletonize_3d
# from test_functions import plot_figure_and_path
from workflow.methods.skel_contraction import graph_pruning
from skimage.measure import label
from workflow.methods.skel_graph import compute_graph_and_paths
from joblib import Parallel,delayed
import matplotlib.pyplot as plt


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

def compute_graph_and_pruning_for_label(volume=np.array([]),dt=[],uniq_label=0,idx=0,uniq_len=0):


    print "idx: ", idx," of ", uniq_len
    img=np.zeros(volume.shape)
    img[volume==uniq_label]=1
    img=close_cavities(img)
    volume=np.uint64(img)



    a=compute_graph_and_paths(volume,dt,"run",[10,1,1])
    if len(a)==0:
        return [[],[]]
    del dt
    term_list, edges_and_lens, g, nodes=a



    return graph_pruning(g, term_list, edges_and_lens, nodes)

    # np.save("/export/home/amatskev/Bachelor/misc/finished_question.npy",(finished_dict, intersecting_node_dict, nodes_list))

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


def actual_pruning(finished_dict, intersecting_node_dict, nodes_list):

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




    if len(pruned_term_list)==1:

        key=pruned_term_list[0]

        if len(nodes_list.keys())==4:
            node = finished_dict[key][0][-2]
        else:

            node=finished_dict[key][0][-1]
        list=np.array(intersecting_node_dict[node])[np.array(intersecting_node_dict[node])!=key]
        count=[0,0]
        for possible_lbl in list:
            possible_ratio=(finished_dict[possible_lbl][1]+finished_dict[key][1])/\
                           max(finished_dict[possible_lbl][4],finished_dict[key][4])
            if possible_ratio>count[1]:
                count[0]=possible_lbl
                count[1]=possible_ratio

        pruned_term_list.append(count[0])

    return np.array(pruned_term_list)


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










if __name__ == "__main__":
    # plot_statistics()
    seg = np.array(vigra.readHDF5('/mnt/localdata0/amatskev/neuraldata/results/'
                              'splA_z0/result.h5', "z/0/data"))
    dt = distance_transform(seg, [10., 1., 1.])

    # centres_dict = compute_border_contacts_old(seg, dt)
    # uniq_border_contacts = [key for key in centres_dict.keys()
    #                         if len(centres_dict[key]) > 1]
    uniq_border_contacts=np.load("/mnt/localdata1/amatskev/debugging/data/graph_pruning/statistics_border_uniq.npy")

    for uniq_label in uniq_border_contacts:
        mask=seg==uniq_label
        print uniq_label
        a=dt[mask]
        print "mean: ",np.mean(a)
        print "variance: ", np.var(a)
        print "std: ", np.std(a)
        plt.hist(a, bins=100, normed=True)
        plt.show()

    print "hi"

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
    # plot_figure_and_path(img, a)
    #
    # a=6
    # # seg_0=np.zeros((a*100,a*100,a*100))
    # # seg_0[  a*10:a*20,    a*10:a*90,a*47:a*-47]=1
    # # seg_0[a*80:a*90,  a*10:a*90,a*47:a*-47]=1
    # # seg_0[a*40:a*60, a*10:a*90, a*40:a*60] = 1
    # # # seg_0[a*45:a*55,a*45:a*55,a*45:a*55]=0
    # # seg_0[a*10:a*90,  a*10:a*20, a* 47:a*-47]=1
    # # seg_0[a*10:a*90, a*80:a*90, a*47:a*-47] = 1
    # # # seg_0[40:-40, 40:-40, 10:-10] = 1
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
    # # volume=np.load("/mnt/localdata1/amatskev/debugging/funny_shit.npy")
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
    # # # plot_figure_and_path(volume, a)
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