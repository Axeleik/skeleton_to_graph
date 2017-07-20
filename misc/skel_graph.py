import numpy as np
from copy import deepcopy
from Queue import LifoQueue
from time import time
import nifty_with_cplex as nifty


def check_box(volume,point,is_queued_map,is_node_map,stage=1):
    """checks the Box around the point for points which are 1,
    but were not already put in the queue and returns them in a list"""
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
                    if is_node_map[point[0] + x, point[1] + y, point[2] + z] !=0:
                        list_is_node.extend([[point[0] + x, point[1] + y, point[2] + z]])

    return list_not_queued,list_is_node,list_are_near



def init(volume):
    """searches for the first node to start with"""
    if len(np.where(volume)[0])==0:
        return np.array([-1,-1,-1])
    point = np.array((np.where(volume)[:][0][0], np.where(volume)[:][1][0], np.where(volume)[:][2][0]))

    is_queued_map =np.zeros(volume.shape, dtype=int)
    is_queued_map[point[0], point[1], point[2]]=1

    not_queued,_,_ = check_box(volume,point,is_queued_map,np.zeros(volume.shape, dtype=int))

    if len(not_queued)==2:
        while True:
            point = np.array(not_queued[0])
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            not_queued,_,_ = check_box(volume, point, is_queued_map, np.zeros(volume.shape, dtype=int))

            if len(not_queued)!=1:
                break


    return point


def stage_one(img,dt):
    """stage one, finds all nodes and edges, except for loops"""


    #initializing
    volume = deepcopy(img)
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
    loop_list= []
    branch_point_list=[]
    node_list = []
    length=0
    if (point == np.array([-1,-1,-1])).all():
        return is_node_map, is_term_map, is_branch_map, nodes, edges

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,is_node_list,are_near=check_box(volume, point, is_queued_map, is_node_map)
    nodes[current_node]=point

    while len(not_queued)==0:
        volume[point[0], point[1], point[2]] = 0
        is_queued_map[point[0], point[1], point[2]] = 0
        nodes = {}
        point = init(volume)
        if (point == np.array([-1, -1, -1])).all():
            return is_node_map, is_term_map, is_branch_map, nodes, edges
        is_queued_map[point[0], point[1], point[2]] = 1
        not_queued, is_node_list, are_near = check_box(volume, point, is_queued_map, is_node_map)
        nodes[current_node] = point


    for i in not_queued:
        queue.put(np.array([i,current_node,length,
                            [[point[0], point[1], point[2]]],
                            [dt[point[0], point[1], point[2]]]]))
        is_queued_map[i[0], i[1], i[2]] = 1



    if len(not_queued)==1:
        is_term_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    else:
        is_branch_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node


    while queue.qsize():


        #pull item from queue
        point, current_node, length, edge_list, dt_list = queue.get()

        not_queued,is_node_list,are_near = check_box(volume, point, is_queued_map, is_node_map)

        # if current_node==531:
        #     print "hi"
        #     print "hi"

        #standart point
        if len(not_queued)==1:
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            length = length + np.linalg.norm([point[0] - not_queued[0][0], point[1] - not_queued[0][1], (point[2] - not_queued[0][2]) * 10])
            queue.put(np.array([not_queued[0],current_node,length,edge_list,dt_list]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])
            is_standart_map[point[0], point[1], point[2]] = 1

        elif len(not_queued)==0 and (len(are_near)>1 or len(is_node_list)>0):
            loop_list.extend([current_node])


        #terminating point
        elif len(not_queued)==0 and len(are_near)==1 and len(is_node_list)==0:
            last_node=last_node+1
            nodes[last_node] = point
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[np.array([current_node, last_node]),length,edge_list,dt_list]])
            is_term_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node




        # branch point
        elif len(not_queued)>1:
            dt_list.extend([dt[point[0], point[1], point[2]]])
            edge_list.extend([[point[0], point[1], point[2]]])
            last_node = last_node + 1
            nodes[last_node ] = point
            #build edge
            edges.extend([[np.array([current_node, last_node]),length,edge_list,dt_list]]) #build edge
            node_list.extend([[point[0], point[1], point[2]]])
            #putting node branches in the queue
            for x in not_queued:
                length = np.linalg.norm([point[0] - x[0], point[1] - x[1], (point[2] - x[2]) * 10])
                queue.put(np.array([x, last_node,length,
                                    [[point[0], point[1], point[2]]],
                                    [dt[point[0], point[1], point[2]]]]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node





    # if (len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))!=0:
    #     pass
    #     print "assert"
    # else:
    #     print "no assert"

    #assert((len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))==0), "too few points were looked at/some were looked at twice !"



    return  is_node_map,is_term_map,is_branch_map,nodes,edges,loop_list




def stage_two(is_node_map, is_term_map, edges,dt):
    """finds edges for loops"""

    list_term = np.array(np.where(is_term_map)).transpose()

    for point in list_term:

        _,_,list_near_nodes = check_box(is_node_map, point, np.zeros(is_node_map.shape, dtype=int), np.zeros(is_node_map.shape, dtype=int),2 )

        assert (len(list_near_nodes) == 0)

    #     if len(list_near_nodes) != 0:
    #
    #         assert()
    #         node_number=is_term_map[point[0], point[1], point[2]]
    #         is_term_map[point[0], point[1], point[2]]=0
    #         print "hi"
    #
    #     for i in list_near_nodes:
    #         edge_list = []
    #         edge_list.extend([[point[0], point[1], point[2]]])
    #         edge_list.extend([[i[0], i[1], i[2]]])
    #         dt_list = []
    #         dt_list.extend([dt[point[0], point[1], point[2]]])
    #         dt_list.extend([dt[i[0], i[1], i[2]]])
    #         edges.extend([[np.array([is_node_map[point[0],point[1],point[2]],
    #                                  is_node_map[i[0],i[1],i[2]]]),
    #                        np.linalg.norm([point[0] - i[0], point[1] - i[1],
    #                                        (point[2] - i[2]) * 10]),
    #                        edge_list,
    #                        dt_list]]) #build edge
    #
    #
    # return edges,is_term_map



def form_term_list(is_term_map):
    """returns list of terminal points taken from an image"""


    term_where = np.array(np.where(is_term_map)).transpose()
    term_list=[]
    for point in term_where:
        term_list.extend([is_term_map[point[0],point[1],point[2]]])
    term_list = np.array([term for term in term_list])

    return term_list


def skeleton_to_graph(img,dt):
    """main function, wraps up stage one and two"""

    time_before_stage_one_1=time()
    is_node_map, is_term_map, is_branch_map, nodes, edges,loop_list = stage_one(img,dt)
    if len(nodes)==0:
        return nodes, np.array(edges), [], is_node_map

    edges,is_term_map = stage_two(is_node_map, is_term_map, edges,dt)

    edges=[[a,b,c,max(d)] for a,b,c,d in edges]

    term_list = form_term_list(is_term_map)
    term_list -= 1
    # loop_list -= 1
    return nodes,np.array(edges),term_list,is_node_map,loop_list


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



def compute_graph_and_paths(img,dt,modus="run"):
    """ overall wrapper for all functions, input: label image; output: paths
        sampled from skeleton
    """

    nodes, edges, term_list, is_node_map,loop_list = skeleton_to_graph(img,dt)
    if len(term_list)==0:
        return []
    g, edge_lens, edges = graph_and_edge_weights(nodes, edges)

    check_connected_components(g)

    loop_uniq,loop_nr=np.unique(loop_list, return_counts=True)


    for where in np.where(loop_nr>1)[0]:

        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(loop_uniq[where]-1)])

        if (len(adjacency))==1:
            term_list = np.append(term_list, loop_uniq[where]-1)


    # if modus=="testing":
    #     return term_list,edges,g,nodes


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



    if modus=="testing":
        return term_list,edges,g,nodes

    return workflow_paths