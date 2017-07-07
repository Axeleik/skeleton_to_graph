import nifty_with_cplex as nifty
import numpy as np
from workflow.methods.functions_for_workflow import compute_graph_and_paths,close_cavities
# from test_functions import img_to_skel
from skimage.morphology import skeletonize_3d
import cPickle as pickle
import vigra
from Queue import Queue
from copy import deepcopy

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



#TODO check whether edgelist and termlist is ok (because of -1)
def graph_pruning(g,term_list,edges):

    finished_dict={}
    node_dict={}
    main_dict={}
    queue=Queue()
    for term_point in term_list:
        queue.put([term_point,term_point])



    while queue.qsize():

        #draw from queue
        current_node,label=queue.get()

        if current_node==0:
            print current_node
            pass

        # if the other branches (except one) are not already at this node
        if current_node in node_dict.keys():
            if len(node_dict[current_node])!=1:
                queue.put([current_node,label])
                continue



        adjacency=np.array([[adj_node,adj_edge] for adj_node, adj_edge
                            in g.nodeAdjacency(current_node)])


        # for terminating points
        if len(adjacency)==1:

            main_dict[current_node]=[[current_node, adjacency[0][0]],
                                   edges[adjacency[0][1]][1],
                                     edges[adjacency[0][1]][2]]

            # if adjacent node was already visited
            if adjacency[0][0] in node_dict.keys():

                node_dict[adjacency[0][0]][0].remove(current_node)
                node_dict[adjacency[0][0]][2].remove(adjacency[0][1])


                # if this edge is longer than already written edge
                if edges[adjacency[0][1]][1] >= main_dict[node_dict[adjacency[0][0]][1]][1]:

                    finished_dict[node_dict[adjacency[0][0]][1]]\
                        = deepcopy(main_dict[node_dict[adjacency[0][0]][1]])
                    del main_dict[node_dict[adjacency[0][0]][1]]
                    node_dict[adjacency[0][0]][1]=current_node

                else:

                    finished_dict[current_node]=deepcopy(main_dict[current_node])
                    del main_dict[current_node]

            #create new dict.key for adjacent node
            else:

                node_dict[adjacency[0][0]] = [[adj_node for adj_node, adj_edge
                                        in g.nodeAdjacency(adjacency[0][0])
                                               if adj_node != current_node],
                                              current_node,
                                              [adj_edge for adj_node, adj_edge
                                        in g.nodeAdjacency(adjacency[0][0])
                                               if adj_edge != adjacency[0][1]]]


            #if all except one branches reached the adjacent node
            if len(node_dict[adjacency[0][0]][0])==1:

                #writing new node to label
                main_dict[node_dict[adjacency[0][0]][1]][0].extend([node_dict[adjacency[0][0]][0]])

                #adding length to label
                main_dict[node_dict[adjacency[0][0]][1]][1]+=edges[node_dict[adjacency[0][0]][2][0]][1]

                #adding path to next node to label
                main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                    edges[node_dict[adjacency[0][0]][2][0]][2])

                #putting next
                queue.put([node_dict[adjacency[0][0]][0],
                           node_dict[adjacency[0][0]][1]])

                #deleting node from dict
                del node_dict[adjacency[0][0]]






if __name__ == "__main__":


    # volume=np.load("/export/home/amatskev/"
    #                "Link to neuraldata/test/first_try_Volume.npy")

    # with open('/export/home/amatskev/Link to neuraldata/'
    #           'test/extract_paths_from_segmentation/input_ds.pkl', mode='r') as f:
    #     input_ds_example=pickle.load(f)
    #

    term_list   = np.load("/export/home/amatskev/"
                          "Bachelor/data/graph_pruning/term_list.npy")
    # edge_lens   = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/edge_lens.npy")
    edges       = np.load("/export/home/amatskev/"
                          "Bachelor/data/graph_pruning/edges.npy")
    # nodes       = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/nodes.npy")
    # is_node_map = np.load("/export/home/amatskev/"
    #                       "Bachelor/data/graph_pruning/is_node_map.npy")

    g=read_graph('/export/home/amatskev/Bachelor/data/graph_pruning/graph_tmp.h5')


    graph_pruning(g, term_list, edges)

# volume=close_cavities(volume)
    # skel_img = skeletonize_3d(volume)
    #
    # compute_graph_and_paths(skel_img)
