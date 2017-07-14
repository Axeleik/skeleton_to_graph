import nifty_with_cplex as nifty
import numpy as np
from workflow.methods.functions_for_workflow import \
    compute_graph_and_paths,close_cavities,get_unique_rows
# from test_functions import plot_figure_and_path,img_to_skel
from skimage.morphology import skeletonize_3d
import cPickle as pickle
import vigra
from Queue import Queue
from copy import deepcopy
import matplotlib.pyplot as plt


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

def terminal_func(start_queue,g,finished_dict,node_dict,main_dict,edges):

    print "---------------------------------------------"
    print "---------------------------------------------"
    print "Starting terminal_func..."


    queue = Queue()

    while start_queue.qsize():

        # draw from queue
        current_node, label = start_queue.get()

        print "---------------------------------------------"
        print "current node: ", current_node
        print "label: ", label

        #check the adjacency
        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(current_node)])

        # assert(len(adjacency)<3), "terminal points can not have 3 adjacent neighbors," \
        #                           " only a maximum of 2 in loops"
        #

        # #for
        # if current_node==7:
        #     print "hi"
        #     print "hi"
        #
        #     adjacency=np.array([neighbor for neighbor in adjacency
        #                if neighbor[0] not in term_list])

        assert(len(adjacency) == 1)

        # for terminating points
        if len(adjacency) == 1:

            main_dict[current_node] = [[current_node, adjacency[0][0]],
                                       edges[adjacency[0][1]][1],
                                       edges[adjacency[0][1]][2],adjacency[0][1]]

            # if adjacent node was already visited
            if adjacency[0][0] in node_dict.keys():

                node_dict[adjacency[0][0]][0].remove(current_node)
                node_dict[adjacency[0][0]][2].remove(adjacency[0][1])

                # if this edge is longer than already written edge
                if edges[adjacency[0][1]][1] >= \
                        main_dict[node_dict[adjacency[0][0]][1]][1]:

                    print current_node," is longer than ",node_dict[adjacency[0][0]][1]

                    finished_dict[node_dict[adjacency[0][0]][1]] \
                        = deepcopy(main_dict[node_dict[adjacency[0][0]][1]])

                    #get unique rows
                    finished_dict[node_dict[adjacency[0][0]][1]][2]= \
                        get_unique_rows(np.array
                                        (finished_dict[node_dict[adjacency[0][0]][1]][2]))
                    del main_dict[node_dict[adjacency[0][0]][1]]
                    node_dict[adjacency[0][0]][1] = current_node

                else:

                    print current_node," is shorter than ",node_dict[adjacency[0][0]][1]

                    finished_dict[current_node] = deepcopy(main_dict[current_node])

                    # get unique rows
                    finished_dict[current_node][2]=\
                        get_unique_rows(np.array(finished_dict[current_node][2]))
                    del main_dict[current_node]

            # create new dict.key for adjacent node
            else:

                node_dict[adjacency[0][0]] = [[adj_node for adj_node, adj_edge
                                               in g.nodeAdjacency(adjacency[0][0])
                                               if adj_node != current_node],
                                              current_node,
                                              [adj_edge for adj_node, adj_edge
                                               in g.nodeAdjacency(adjacency[0][0])
                                               if adj_edge != adjacency[0][1]]]

            # if all except one branches reached the adjacent node
            if len(node_dict[adjacency[0][0]][0]) == 1:

                print "Winner at node ", adjacency[0][0]," is label ",\
                node_dict[adjacency[0][0]][1]

                # writing new node to label
                main_dict[node_dict[adjacency[0][0]][1]][0].\
                    extend([node_dict[adjacency[0][0]][0][0]])

                # adding length to label
                main_dict[node_dict[adjacency[0][0]][1]][1] += \
                    edges[node_dict[adjacency[0][0]][2][0]][1]

                # adding path to next node to label
                main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                    edges[node_dict[adjacency[0][0]][2][0]][2])

                #adding edge number to label
                main_dict[node_dict[adjacency[0][0]][1]][3]=\
                    node_dict[adjacency[0][0]][2][0]

                # putting next
                queue.put([node_dict[adjacency[0][0]][0][0],
                           node_dict[adjacency[0][0]][1]])

                # deleting node from dict
                del node_dict[adjacency[0][0]]


    return queue,finished_dict,node_dict,main_dict





#TODO check whether edgelist and termlist is ok (because of -1)
def graph_pruning(g,term_list,edges,dt):

    finished_dict={}
    node_dict={}
    main_dict={}
    start_queue=Queue()
    last_dict={}

    for term_point in term_list:
        start_queue.put([term_point,term_point])


    queue,finished_dict,node_dict,main_dict = \
        terminal_func (start_queue, g, finished_dict, node_dict, main_dict, edges)

    print "---------------------------------------------"
    print "---------------------------------------------"
    print "Starting main function... "


    while queue.qsize():
        test_len1=len(main_dict.keys())

        # draw from queue
        current_node, label = queue.get()

        print "---------------------------------------------"
        print "current node: ", current_node
        print "label: ", label


        # if current node was already visited at least once
        if current_node in node_dict.keys():


            if len(node_dict[current_node][0])==1:
                last_dict[label] = deepcopy(main_dict[label])
                continue

            # remove previous node from adjacency
            node_dict[current_node][0].remove(main_dict[label][0][-2])

            # remove previous edge from adjacency
            node_dict[current_node][2].remove(main_dict[label][3])


            # if current label is longer than longest in node
            if main_dict[label][1] >= \
                    main_dict[node_dict[current_node][1]][1]:

                print label, " is longer than ", \
                    node_dict[current_node][1]


                # finishing previous longest label
                finished_dict[node_dict[current_node][1]] \
                    = deepcopy(main_dict[node_dict[current_node][1]])
                del main_dict[node_dict[current_node][1]]

                # get unique rows
                finished_dict[node_dict[current_node][1]][2]\
                    =get_unique_rows(np.array
                                     (finished_dict[node_dict[current_node][1]][2]))

                # writing new label to longest in node
                node_dict[current_node][1]=label


            else:

                #finishing this label
                finished_dict[label] = deepcopy(main_dict[label])

                # get unique rows
                finished_dict[label][2]=\
                    get_unique_rows(np.array(finished_dict[label][2]))

                del main_dict[label]



        else:

            #create new entry for this node
            node_dict[current_node] = [[adj_node for adj_node, adj_edge
                                           in g.nodeAdjacency(current_node)
                                           if adj_node != main_dict[label][0][-2]],
                                          label,
                                          [adj_edge for adj_node, adj_edge
                                           in g.nodeAdjacency(current_node)
                                           if adj_edge != main_dict[label][3]]]

        if len(main_dict.keys())==2:
            for key in main_dict.keys():
                finished_dict[key]=deepcopy(main_dict[key])
                finished_dict[key][2]=get_unique_rows(np.array(finished_dict[key][2]))
                del main_dict[key]
            # deleting node from dict
            del node_dict[current_node]
            break


        # if all except one branches reached the adjacent node
        if len(node_dict[current_node][0]) == 1:

            print "Winner at node ", current_node, " is label ", \
                node_dict[current_node][1]

            # writing new node to label
            main_dict[node_dict[current_node][1]][0]. \
                extend([node_dict[current_node][0][0]])

            # adding length to label
            main_dict[node_dict[current_node][1]][1] += \
                edges[node_dict[current_node][2][0]][1]

            # adding path to next node to label
            main_dict[node_dict[current_node][1]][2].extend(
                edges[node_dict[current_node][2][0]][2])

            # adding edge number to label
            main_dict[node_dict[current_node][1]][3] = \
                node_dict[current_node][2][0]

            # putting next
            queue.put([node_dict[current_node][0][0],
                        node_dict[current_node][1]])

            # deleting node from dict
            del node_dict[current_node]

        test_len2=len(main_dict.keys())








    for key in finished_dict.keys():
        finished_dict[key][3]=max([dt[val[0],val[1],val[2]]
                                   for val in finished_dict[key][2]])

    return finished_dict












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

if __name__ == "__main__":


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
    # seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    dt = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/dt_seg_0.npy")
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
    skel_img=np.load("/export/home/amatskev/"
                          "Bachelor/data/graph_pruning/skel_img.npy")
    #
    # volume=close_cavities(volume)

    # skel_img = skeletonize_3d(volume)
    # #
    compute_graph_and_paths(skel_img)

    # with open('/export/home/amatskev/Link to neuraldata/'
    #           'test/extract_paths_from_segmentation/input_ds.pkl', mode='r') as f:
    #     input_ds_example=pickle.load(f)
    #

    term_list   = np.load("/export/home/amatskev/"
                          "Bachelor/data/graph_pruning/term_list.npy")
    # # edge_lens   = np.load("/export/home/amatskev/"
    # #                       "Bachelor/data/graph_pruning/edge_lens.npy")
    edges       = np.load("/export/home/amatskev/"
                          "Bachelor/data/graph_pruning/edges.npy")
    # nodes       = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/nodes.npy")
    # is_node_map = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/is_node_map.npy")
    #
    g=read_graph('/export/home/amatskev/Bachelor/data/graph_pruning/graph_tmp.h5')


    finished=graph_pruning(g, term_list, edges,dt)

    print "hi"
    print "hi"

