import numpy as np
import vigra
import os
import cPickle as pickle
import shutil
import itertools
import h5py
import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue
from time import time
import h5py
import nifty_with_cplex as nifty
from skimage.measure import label


def close_cavities(init_volume):
    """close cavities in segments so skeletonization don't bugs"""

    print "looking for open cavities inside the object..."
    volume=deepcopy(init_volume)
    volume[volume==0]=2
    lab=label(volume)

    if len(np.unique(lab))==2:
        print "No cavities to close!"
        return init_volume

    count,what=0,0

    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    volume[lab==what]=0
    volume[lab != what] = 1

    print "cavities closed"

    return volume


def check_box(volume,point,is_queued_map,is_visited_map,stage=1):
    """checks the Box around the point for points which are 1,
    but were not already put in the queue and returns them in a list"""
    list_not_visited=[]
    list_not_queued = []
    list_are_near = []
    list_is_node = []



    for x in xrange(-1, 2):

        # Edgecase for x
        if point[0] + x < 0 or point[0] + x > volume.shape[0] - 1:
            continue

        for y in xrange(-1, 2):

            # Edgecase for y
            if point[1] + y < 0 or point[1] + y > volume.shape[1] - 1:
                continue

            for z in xrange(-1, 2):

                # Edgecase for z
                if point[2] + z < 0 or point[2] + z > volume.shape[2] - 1:
                    continue

                # Dont look at the middle point
                if x == 0 and y == 0 and z == 0:
                    continue

                if volume[point[0] + x, point[1] + y, point[2] + z] > 0:

                    list_are_near.extend([[point[0] + x, point[1] + y, point[2] + z]])


                    if is_queued_map[point[0] + x, point[1] + y, point[2] + z]==0:
                        list_not_queued.extend([[point[0] + x, point[1] + y, point[2] + z]])


                    #leftover, maybe i`ll need it sometime
                    if is_visited_map[point[0] + x, point[1] + y, point[2] + z] == 0:
                        list_not_visited.extend([[point[0] + x, point[1] + y, point[2] + z]])

    is_visited_map[point[0],point[1],point[2]]=1
    return list_not_queued,list_not_visited,is_visited_map,list_are_near



def init(volume):
    """searches for the first node to start with"""
    if len(np.where(volume)[0])==0:
        return np.array([-1,-1,-1])
    point = np.array((np.where(volume)[:][0][0], np.where(volume)[:][1][0], np.where(volume)[:][2][0]))

    is_visited_map = np.zeros(volume.shape, dtype=int)
    is_visited_map[point[0], point[1], point[2]]=1

    is_queued_map =np.zeros(volume.shape, dtype=int)
    is_queued_map[point[0], point[1], point[2]]=1

    not_queued,_,_,_ = check_box(volume,point,is_queued_map,np.zeros(volume.shape, dtype=int))

    if len(not_queued)==2:
        while True:
            point = np.array(not_queued[0])
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            not_queued,_,_,_ = check_box(volume, point, is_queued_map, np.zeros(volume.shape, dtype=int))

            if len(not_queued)!=1:
                break


    return point


def stage_one(img):
    """stage one, finds all nodes and edges, except for loops"""


    #initializing
    volume = deepcopy(img)
    is_visited_map = np.zeros(volume.shape, dtype=int)
    is_queued_map = np.zeros(volume.shape, dtype=int)
    is_node_map = np.zeros(volume.shape, dtype=int)
    is_term_map  = np.zeros(volume.shape, dtype=int)
    is_branch_map  = np.zeros(volume.shape, dtype=int)
    is_standart_map = np.zeros(volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node = 1
    current_node = 1
    queue = LifoQueue()
    point=init(volume)
    leftover_list= []
    branch_point_list=[]
    node_list = []
    length=0
    edge_list=[]
    if (point == np.array([-1,-1,-1])).all():
        return is_node_map, is_term_map, is_branch_map, nodes, edges

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,not_visited,is_visited_map,are_near=check_box(volume, point, is_queued_map, is_visited_map)
    nodes[current_node]=point

    while len(not_queued)==0:
        volume[point[0], point[1], point[2]] = 0
        is_queued_map[point[0], point[1], point[2]] = 0
        is_visited_map[point[0], point[1], point[2]] = 0
        nodes = {}
        point = init(volume)
        if (point == np.array([-1, -1, -1])).all():
            return is_node_map, is_term_map, is_branch_map, nodes, edges
        is_queued_map[point[0], point[1], point[2]] = 1
        not_queued, not_visited, is_visited_map, are_near = check_box(volume, point, is_queued_map, is_visited_map)
        nodes[current_node] = point


    for i in xrange(0,len(not_queued)):
        queue.put(np.array([not_queued[i],current_node,length,[[point[0], point[1], point[2]]]]))
        is_queued_map[not_queued[i][0], not_queued[i][1], not_queued[i][2]] = 1



    if len(not_queued)==1:
        is_term_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    else:
        is_branch_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node


    while queue.qsize():

        #pull item from queue
        point,current_node,length,edge_list=queue.get()

        not_queued,not_visited,is_visited_map,are_near = check_box(volume, point, is_queued_map, is_visited_map)


        #standart point
        if len(not_queued)==1:
            edge_list.extend([[point[0], point[1], point[2]]])
            length = length + np.linalg.norm([point[0] - not_queued[0][0], point[1] - not_queued[0][1], (point[2] - not_queued[0][2]) * 10])
            queue.put(np.array([not_queued[0],current_node,length,edge_list]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])
            is_standart_map[point[0], point[1], point[2]] = 1



        #terminating point
        elif len(not_queued)==0 and len(are_near)==1:
            last_node=last_node+1
            nodes[last_node] = point
            edge_list.extend([[point[0], point[1], point[2]]])
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[np.array([current_node, last_node]),length,edge_list]])
            is_term_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node




        # branch point
        elif len(not_queued)>1:
            edge_list.extend([[point[0], point[1], point[2]]])
            last_node = last_node + 1
            nodes[last_node ] = point
            #build edge
            edges.extend([[np.array([current_node, last_node]),length,edge_list]]) #build edge
            node_list.extend([[point[0], point[1], point[2]]])
            #putting node branches in the queue
            for x in not_queued:
                length = np.linalg.norm([point[0] - x[0], point[1] - x[1], (point[2] - x[2]) * 10])
                queue.put(np.array([x, last_node,length,[[point[0], point[1], point[2]]]]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node





    # if (len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))!=0:
    #     pass
    #     print "assert"
    # else:
    #     print "no assert"

    #assert((len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))==0), "too few points were looked at/some were looked at twice !"



    return  is_node_map,is_term_map,is_branch_map,nodes,edges




def stage_two(is_node_map, is_term_map, edges,nodes):
    """finds edges for loops"""


    list_term = np.array(np.where(is_term_map)).transpose()

    for point in list_term:

        _,_,_,list_near_nodes = check_box(is_node_map, point, np.zeros(is_node_map.shape, dtype=int), np.zeros(is_node_map.shape, dtype=int),2 )

        if len(list_near_nodes) != 0:

            is_term_map[point[0], point[1], point[2]]=0
            print "hi"

        for i in list_near_nodes:
            edge_list = []
            edge_list.extend([[point[0], point[1], point[2]]])
            edge_list.extend([[i[0], i[1], i[2]]])
            edges.extend([[np.array([is_node_map[point[0],point[1],point[2]], is_node_map[i[0],i[1],i[2]]]),np.linalg.norm([point[0] - i[0], point[1] - i[1], (point[2] - i[2]) * 10]),edge_list]]) #build edge


    return edges,is_term_map



def form_term_list(is_term_map):
    """returns list of terminal points taken from an image"""


    term_where = np.array(np.where(is_term_map)).transpose()
    term_list=[]
    for point in term_where:
        term_list.extend([is_term_map[point[0],point[1],point[2]]])
    term_list = np.array([term for term in term_list])

    return term_list


def skeleton_to_graph(img):
    """main function, wraps up stage one and two"""

    time_before_stage_one_1=time()
    is_node_map, is_term_map, is_branch_map, nodes, edges = stage_one(img)
    if len(nodes)==0:
        return nodes, np.array(edges), [], is_node_map

    edges,is_term_map = stage_two(is_node_map, is_term_map, edges,nodes)



    term_list = form_term_list(is_term_map)
    term_list -= 1
    return nodes,np.array(edges),term_list,is_node_map


def get_unique_rows(array, return_index=False):
    """ make the rows of array unique
        see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
    """

    array_view = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    _, idx = np.unique(array_view, return_index=True)
    unique_rows = array[idx]
    if return_index:
        return unique_rows, idx
    else:
        return unique_rows


def unique_rows(a):
    """same same but different"""

    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))




def graph_and_edge_weights(nodes,edges_and_lens):
    """creates graph from edges and nodes, length of edges included"""

    edges = []
    edge_lens = []
    edges.extend(edges_and_lens[:, 0])
    edges = np.array(edges,dtype="uint32")
    edge_lens.extend(edges_and_lens[:, 1])
    edge_lens = np.array([edge for edge in edge_lens])
    edges = np.sort(edges, axis=1)
    # remove duplicates from edges and edge-lens
    edges, unique_idx = get_unique_rows(edges, return_index=True)
    edge_lens = edge_lens[unique_idx]
    edges_and_lens = edges_and_lens[unique_idx]
    edges_and_lens[:,0]-=1

    assert len(edges) == len(edge_lens)
    assert edges.shape[1] == 2
    node_list = np.array(nodes.keys())
    edges = np.array(edges, dtype='uint32')
    edges = np.sort(edges, axis=1)
    edges -= 1
    n_nodes = edges.max() + 1
    assert len(node_list) == n_nodes
    g = nifty.graph.UndirectedGraph(n_nodes)
    g.insertEdges(edges)
    assert g.numberOfEdges == len(edge_lens), '%i, %i' % (g.numberOfEdges, len(edge_lens))
    return g, edge_lens,edges_and_lens





#
def check_connected_components(g):
    """check that we have the correct number of connected components"""

    cc = nifty.graph.components(g)
    cc.build()

    components = cc.componentLabels()

    #print components
    n_ccs = len(np.unique(components))
    assert n_ccs == 1, str(n_ccs)


def edge_paths_and_counts_for_nodes(g, weights, node_list, n_threads=8):
    """
    Returns the path of edges for all pairs of nodes in node list as
    well as the number of times each edge is included in a shortest path.
    @params:
    g         : nifty.graph.UndirectedGraph, the underlying graph
    weights   : list[float], the edge weights for shortest paths
    node_list : list[int],   the list of nodes that will be considered for the shortest paths
    n_threads : int, number of threads used
    @returns:
    edge_paths: dict[tuple(int,int),list[int]] : Dictionary with list of edge-ids corresponding to the
                shortest path for each pair in node_list
    edge_counts: np.array[int] : Array with number of times each edge was visited in a shortest path
    """

    edge_paths = {}
    edge_counts = np.zeros(g.numberOfEdges, dtype='uint32')

    # single threaded implementation
    if n_threads < 2:

        # build the nifty shortest path object
        path_finder = nifty.graph.ShortestPathDijkstra(g)

        # iterate over the source nodes
        # we don't need to go to the last node, because it won't have any more targets
        for ii, u in enumerate(node_list[:-1]):

            # target all nodes in node list tat we have not visited as source
            # already (for these the path is already present)
            target_nodes = node_list[ii + 1:]

            # find the shortest path from source node u to the target nodes
            shortest_paths = path_finder.runSingleSourceMultiTarget(
                weights.tolist(),
                u,
                target_nodes,
                returnNodes=False
            )
            assert len(shortest_paths) == len(target_nodes)

            # extract the shortest path for each node pair and
            # increase the edge counts
            for jj, sp in enumerate(shortest_paths):
                v = target_nodes[jj]
                edge_paths[(u, v)] = sp
                edge_counts[sp] += 1

    # multi-threaded implementation
    # this might be quite memory hungry!
    else:

        # construct the target nodes for all source nodes and run shortest paths
        # in parallel, don't need last node !
        all_target_nodes = [node_list[ii + 1:] for ii in xrange(len(node_list[:-1]))]
        all_shortest_paths = nifty.graph.shortestPathMultiTargetParallel(
            g,
            weights.tolist(),
            node_list[:-1],
            all_target_nodes,
            returnNodes=False,
            numberOfThreads=n_threads
        )
        assert len(all_shortest_paths) == len(node_list) - 1, "%i, %i" % (len(all_shortest_paths), len(node_list) - 1)

        # TODO this is still quite some serial computation overhead.
        # for good paralleliztion, this should also be parallelized

        # extract the shortest paths for all node pairs and edge counts
        for ii, shortest_paths in enumerate(all_shortest_paths):

            u = node_list[ii]
            target_nodes = all_target_nodes[ii]
            for jj, sp in enumerate(shortest_paths):
                v = target_nodes[jj]
                edge_paths[(u, v)] = sp
                edge_counts[sp] += 1

    return edge_paths, edge_counts



def check_edge_paths(edge_paths, node_list):
    """checks edge paths (constantin)"""

    from itertools import combinations
    pairs = combinations(node_list, 2)
    pair_list = [pair for pair in pairs]

    # make sure that we have all combination in the edge_paths
    for pair in pair_list:
        assert pair in edge_paths

    # make sure that we don't have any spurious pairs in edge_paths
    for pair in edge_paths:
        assert pair in pair_list

    print "passed"





def compute_graph_and_paths(img):
    """ overall wrapper for all functions, input: label image; output: paths
        sampled from skeleton
    """

    nodes, edges, term_list, is_node_map = skeleton_to_graph(img)
    if len(term_list)==0:
        return []
    g, edge_lens, edges = graph_and_edge_weights(nodes, edges)

    check_connected_components(g)

    edge_paths, edge_counts = edge_paths_and_counts_for_nodes(g,
                                                              edge_lens,
                                                              term_list[:30],8)
    check_edge_paths(edge_paths, term_list[:30])
    edge_paths_julian = {}

    for pair in edge_paths.keys():
        edge_paths_julian[pair] = []

        for idx in edge_paths[pair]:
            edge_paths_julian[pair].extend(edges[idx][2])


    final_edge_paths = {}

    for pair in edge_paths_julian.keys():
        final_edge_paths[pair] = unique_rows(edge_paths_julian[pair])

    workflow_paths = []

    # for workflow functions
    for pair in final_edge_paths.keys():
        workflow_paths.extend([final_edge_paths[pair]])

    return workflow_paths


def cut_off(all_paths_unfinished,paths_to_objs_unfinished,
            cut_off_array,ratio_true=0.13,ratio_false=0.4):
    """ cuts array so that all paths with a ratio between ratio_true% and
        ratio_false% are not in the paths as they are not clearly to identify
        as false paths or true paths
    """

    print "start cutting off array..."
    test_label = []
    con_label = {}
    test_length = []
    con_len = {}

    for label in cut_off_array.keys():
        #FIXME  in cut_off conc=np.concatenate(con_label[label]).tolist() ValueError: need at least one array to concatenate
        if len(cut_off_array[label])==0:
            continue
        con_label[label]=[]
        con_len[label]=[]
        for path in cut_off_array[label]:
            test_label.append(path[0])
            con_label[label].append(path[0])
            test_length.append(path[1])
            con_len[label].append(path[1])

    help_array=[]
    for label in con_label.keys():

        conc=np.concatenate(con_label[label]).tolist()
        counter=[0,0]
        for number in np.unique(conc):
            many = conc.count(number)

            if counter[1] < many:
                counter[0] = number
                counter[1] = many
        #TODO con label len =0
        for i in xrange(0,len(con_label[label])):
            help_array.extend([counter[0]])

    end = []

    for idx, path in enumerate(test_label):


        overall_length = 0
        for i in test_length[idx]:
            overall_length = overall_length + i

        less_length = 0
        for u in np.where(np.array(path) != help_array[idx]):
            for index in u:
                less_length = less_length + test_length[idx][index]

        end.extend([less_length/ overall_length])



    path_classes=[]
    all_paths=[]
    paths_to_objs=[]
    for idx,ratio in enumerate(end):
        if ratio<ratio_true:
            path_classes.extend([True])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])

        elif ratio>ratio_false:
            path_classes.extend([False])
            all_paths.extend([all_paths_unfinished[idx]])
            paths_to_objs.extend([paths_to_objs_unfinished[idx]])


    print "finished cutting of"

    return np.array(all_paths), np.array(paths_to_objs, dtype="float64"),\
           np.array(path_classes)


def extract_paths_from_segmentation(
        ds,
        seg_path,
        key,
        paths_cache_folder=None):
    """
        extract paths from segmentation, for pipeline
    """


    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s.h5' % ds.ds_name)
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')

    # otherwise compute the paths

    else:

        seg = vigra.readHDF5(seg_path, key)
        gt = deepcopy(seg)
        img = deepcopy(seg)
        all_paths = []
        paths_to_objs = []


        cut_off_array = {}
        len_uniq=len(np.unique(seg))-1
        for idx,label in enumerate(np.unique(seg)):
            print "Number ", idx, " without labels of ",len_uniq-1
            if label == 0:
                continue


            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1

            # no skeletons too close to the borders No.2
            #img[dt == 0] = 0

            # skeletonize
            skel_img = skeletonize_3d(img)

            paths=compute_graph_and_paths(skel_img)

            percentage = []
            len_path=len(paths)
            for idx,path in enumerate(paths):
                print idx ,". path of ",len_path-1
                #TODO better workaround
                # workaround till tomorrow
                workaround_array=[]
                length_array=[]
                last_point=path[0]
                for i in path:
                    workaround_array.extend([gt[i[0],i[1],i[2]]])
                    length_array.extend([np.linalg.norm([last_point[0] - i[0],
                                                      last_point[1] - i[1], (last_point[2] - i[2]) * 10])])
                    last_point=i


                all_paths.extend([path])
                paths_to_objs.extend([label])

                half_length_array=[]

                for idx,obj in enumerate(length_array[:-1]):
                    half_length_array.extend([length_array[idx]/2+length_array[idx+1]/2])
                half_length_array.extend([length_array[-1] / 2])


                percentage.extend([[workaround_array, half_length_array]])

            cut_off_array[label] = percentage

        all_paths, paths_to_objs,_ =cut_off(all_paths,paths_to_objs,cut_off_array)

        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')

    return all_paths, paths_to_objs



def extract_paths_and_labels_from_segmentation(
        ds,
        seg,
        seg_id,
        gt,
        correspondence_list,
        paths_cache_folder=None):

    """
        extract paths from segmentation, for learning
    """

    if paths_cache_folder is not None:
        if not os.path.exists(paths_cache_folder):
            os.mkdir(paths_cache_folder)
        paths_save_file = os.path.join(paths_cache_folder, 'paths_ds_%s_seg_%i.h5' % (ds.ds_name, seg_id))
    else:
        paths_save_file = ''

    # if the cache exists, load paths from cache
    if os.path.exists(paths_save_file):
        all_paths = vigra.readHDF5(paths_save_file, 'all_paths')
        # we need to reshape the paths again to revover the coordinates
        if all_paths.size:
            all_paths = np.array( [ path.reshape( (len(path)/3, 3) ) for path in all_paths ] )
        paths_to_objs = vigra.readHDF5(paths_save_file, 'paths_to_objs')
        path_classes = vigra.readHDF5(paths_save_file, 'path_classes')
        correspondence_list = vigra.readHDF5(paths_save_file, 'correspondence_list').tolist()

    # otherwise compute paths
    else:

        img = deepcopy(seg)
        all_paths = []
        paths_to_objs = []


        cut_off_array = {}
        len_uniq=len(np.unique(seg))-1
        for label in np.unique(seg):
            print "Label ", label, " of ",len_uniq
            if label == 0:
                continue

            # masking volume
            img[seg != label] = 0
            img[seg == label] = 1


            # skeletonize
            skel_img = skeletonize_3d(img)

            paths=compute_graph_and_paths(skel_img)

            percentage = []

            for path in paths:

                #TODO better workaround
                # workaround till tomorrow
                workaround_array=[]
                length_array=[]
                last_point=path[0]
                for i in path:
                    workaround_array.extend([gt[i[0],i[1],i[2]]])
                    length_array.extend([np.linalg.norm([last_point[0] - i[0],
                                                      last_point[1] - i[1], (last_point[2] - i[2]) * 10])])
                    last_point=i


                all_paths.extend([path])
                paths_to_objs.extend([label])

                half_length_array=[]

                for idx,obj in enumerate(length_array[:-1]):
                    half_length_array.extend([length_array[idx]/2+length_array[idx+1]/2])
                half_length_array.extend([length_array[-1] / 2])


                percentage.extend([[workaround_array, half_length_array]])

            cut_off_array[label] = percentage

        all_paths,paths_to_objs,path_classes = cut_off(all_paths, paths_to_objs, cut_off_array)

        # if caching is enabled, write the results to cache
        if paths_cache_folder is not None:
            # need to write paths with vlen and flatten before writing to properly save this
            all_paths_save = np.array([pp.flatten() for pp in all_paths])
            # TODO this is kind of a dirty hack, because write vlen fails if the vlen objects have the same lengths
            # -> this fails if we have only 0 or 1 paths, beacause these trivially have the same lengths
            # -> in the edge case that we have more than 1 paths with same lengths, this will still fail
            # see also the following issue (https://github.com/h5py/h5py/issues/875)
            try:
                print 'Saving paths in {}'.format(paths_save_file)
                with h5py.File(paths_save_file) as f:
                    dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
                    f.create_dataset('all_paths', data=all_paths_save, dtype=dt)
            except (TypeError, IndexError):
                vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # if len(all_paths_save) < 2:
            #     vigra.writeHDF5(all_paths_save, paths_save_file, 'all_paths')
            # else:
            #     with h5py.File(paths_save_file) as f:
            #         dt = h5py.special_dtype(vlen=np.dtype(all_paths_save[0].dtype))
            #         f.create_dataset('all_paths', data = all_paths_save, dtype = dt)
            vigra.writeHDF5(paths_to_objs, paths_save_file, 'paths_to_objs')
            vigra.writeHDF5(path_classes, paths_save_file, 'path_classes')
            vigra.writeHDF5(correspondence_list, paths_save_file, 'correspondence_list')

    return all_paths, paths_to_objs, path_classes, correspondence_list




