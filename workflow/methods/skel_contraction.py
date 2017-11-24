import numpy as np
from Queue import Queue
from copy import deepcopy
# from misc.skel_pruning import extract_from_seg

def adj(g,val):
    for adj_node,adj_edge in g.nodeAdjacency(val):
        print "Node: ",adj_node
        print "Edge: ", adj_edge



def terminal_func(start_queue,g,finished_dict,node_dict,main_dict,edges,nodes_list):



    queue = Queue()

    if start_queue.qsize()==2:

        current_node, label = start_queue.get()

        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(current_node)])

        finished_dict[current_node] = [[current_node, adjacency[0][0]],
                                   edges[adjacency[0][1]][1],
                                   edges[adjacency[0][1]][2],
                                   adjacency[0][1],
                                   edges[adjacency[0][1]][3],
                                       current_node]
        _ = start_queue.get()


    while start_queue.qsize():

        # draw from queue
        current_node, label = start_queue.get()


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

            if (edges[adjacency[0][1]][2][0]==np.array(nodes_list[current_node+1])).all():

                main_dict[current_node] = [[current_node, adjacency[0][0]],
                                           edges[adjacency[0][1]][1],
                                           edges[adjacency[0][1]][2],
                                           adjacency[0][1],
                                           edges[adjacency[0][1]][3]]

            else:

                main_dict[current_node] = [[current_node, adjacency[0][0]],
                                           edges[adjacency[0][1]][1],
                                           edges[adjacency[0][1]][2][::-1],
                                           adjacency[0][1],
                                           edges[adjacency[0][1]][3]]

            # if adjacent node was already visited
            if adjacency[0][0] in node_dict.keys():

                node_dict[adjacency[0][0]][0].remove(current_node)
                node_dict[adjacency[0][0]][2].remove(adjacency[0][1])

                # if this edge is longer than already written edge
                if (edges[adjacency[0][1]][1]/edges[adjacency[0][1]][3]) >= \
                        (main_dict[node_dict[adjacency[0][0]][1]][1]/
                             main_dict[node_dict[adjacency[0][0]][1]][4]):

                    finished_dict[node_dict[adjacency[0][0]][1]] \
                        = deepcopy(main_dict[node_dict[adjacency[0][0]][1]])

                    #get unique rows
                    # finished_dict[node_dict[adjacency[0][0]][1]][2]= \
                        # get_unique_rows(np.array
                        #                 (finished_dict[node_dict[adjacency[0][0]][1]][2]))
                    del main_dict[node_dict[adjacency[0][0]][1]]
                    node_dict[adjacency[0][0]][1] = current_node

                else:

                    finished_dict[current_node] = deepcopy(main_dict[current_node])

                    # get unique rows
                    # finished_dict[current_node][2]=\
                    #     get_unique_rows(np.array(finished_dict[current_node][2]))
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


            # we finish when we have only two open labels left
            if len(main_dict.keys()) == 2 and queue.qsize()==1:


                # writing longest label at node to the others,
                # so we dont cut in half later
                for finished_label in [adj_node for adj_node,
                                           adj_edge in g.nodeAdjacency(adjacency[0][0])
                              if adj_node!=node_dict[adjacency[0][0]][0][0]
                              and adj_node!=node_dict[adjacency[0][0]][1]]:

                    finished_dict[finished_label].\
                        append(finished_label)

                for key in main_dict.keys():

                    finished_dict[key] = deepcopy(main_dict[key])

                    del main_dict[key]

                    finished_dict[key].append(key)

                # deleting node from dict
                # del node_dict[current_node]
                _=queue.get()
                break




            # if all except one branches reached the adjacent node
            if len(node_dict[adjacency[0][0]][0]) == 1:


                # writing longest label at node to the others,
                # so we dont cut in half later
                # noblankspace
                for finished_label in [adj_node for adj_node,
                                           adj_edge in g.nodeAdjacency(adjacency[0][0])
                              if adj_node!=node_dict[adjacency[0][0]][0][0]
                              and adj_node!=node_dict[adjacency[0][0]][1]]:

                    finished_dict[finished_label].\
                        append(node_dict[adjacency[0][0]][1])

                # writing new node to label
                main_dict[node_dict[adjacency[0][0]][1]][0].\
                    extend([node_dict[adjacency[0][0]][0][0]])

                #comparing maximum of dt
                if main_dict[node_dict[adjacency[0][0]][1]][4] <\
                        edges[node_dict[adjacency[0][0]][2][0]][3]:

                    main_dict[node_dict[adjacency[0][0]][1]][4]=\
                        edges[node_dict[adjacency[0][0]][2][0]][3]

                # adding length to label
                main_dict[node_dict[adjacency[0][0]][1]][1] += \
                    edges[node_dict[adjacency[0][0]][2][0]][1]

                # adding path to next node to label
                if main_dict[node_dict[adjacency[0][0]][1]][2][-1]==\
                        edges[node_dict[adjacency[0][0]][2][0]][2][-1]:

                    main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                        edges[node_dict[adjacency[0][0]][2][0]][2][-2::-1])

                # adding path to next node to label
                else:

                    main_dict[node_dict[adjacency[0][0]][1]][2].extend(
                        edges[node_dict[adjacency[0][0]][2][0]][2][1:])

                #adding edge number to label
                main_dict[node_dict[adjacency[0][0]][1]][3]=\
                    node_dict[adjacency[0][0]][2][0]

                # putting next
                queue.put([node_dict[adjacency[0][0]][0][0],
                           node_dict[adjacency[0][0]][1]])

                # # deleting node from dict
                # del node_dict[adjacency[0][0]]


    return queue,finished_dict,node_dict,main_dict





#TODO check whether edgelist and termlist is ok (because of -1)
def graph_pruning(g,term_list,edges,nodes_list):

    finished_dict={}
    node_dict={}
    main_dict={}
    start_queue=Queue()
    last_dict={}

    for term_point in term_list:
        start_queue.put([term_point,term_point])


    queue,finished_dict,node_dict,main_dict = \
        terminal_func (start_queue, g, finished_dict,
                       node_dict, main_dict, edges,nodes_list)



    while queue.qsize():


        # draw from queue
        current_node, label = queue.get()


        # if current node was already visited at least once
        if current_node in node_dict.keys():


            # remove previous node from adjacency
            node_dict[current_node][0].remove(main_dict[label][0][-2])

            # remove previous edge from adjacency
            node_dict[current_node][2].remove(main_dict[label][3])


            # if current label is longer than longest in node
            if (main_dict[label][1]/main_dict[label][4]) >= \
                    (main_dict[node_dict[current_node][1]][1]/
                         main_dict[node_dict[current_node][1]][4]):


                # writing winner node in loser node
                #noblankspace
                main_dict[node_dict[current_node][1]].append(label)

                # finishing previous longest label
                finished_dict[node_dict[current_node][1]] \
                    = deepcopy(main_dict[node_dict[current_node][1]])
                del main_dict[node_dict[current_node][1]]

                # get unique rows
                # finished_dict[node_dict[current_node][1]][2]\
                #     =get_unique_rows(np.array
                #                      (finished_dict[node_dict[current_node][1]][2]))

                # writing new label to longest in node
                node_dict[current_node][1]=label


            else:

                #finishing this label
                finished_dict[label] = deepcopy(main_dict[label])

                # get unique rows
                # finished_dict[label][2]=\
                #     get_unique_rows(np.array(finished_dict[label][2]))

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

        # we finish when we have only two open labels left
        if len(main_dict.keys())<3:


            for finished_label in [adj_node for adj_node,
                                                adj_edge in g.nodeAdjacency(current_node)
                                   if adj_node!=node_dict[current_node][0][0] and
                                                   adj_node!=main_dict[node_dict[current_node][1]][0][-2]]:

                finished_dict[finished_label].append(finished_label)

            for key in main_dict.keys():

                finished_dict[key]=deepcopy(main_dict[key])

                del main_dict[key]

                finished_dict[key].append(key)

            # # deleting node from dict
            # del node_dict[current_node]

            break


        # if all except one branches reached the adjacent node
        if len(node_dict[current_node][0]) == 1:

            # writing longest label at node to the others,
            # so we dont cut in half later
            for finished_label in [adj_node for adj_node,
                                                adj_edge in g.nodeAdjacency(current_node)
                                   if adj_node!=node_dict[current_node][0][0] and
                                                   adj_node!=main_dict[node_dict[current_node][1]][0][-2]]:

                if finished_label in finished_dict.keys():

                    finished_dict[finished_label] \
                        .append(node_dict[current_node][1])

                else:

                    finished_dict[node_dict[finished_label][1]]\
                        .append(node_dict[current_node][1])

            # writing new node to label
            main_dict[node_dict[current_node][1]][0]. \
                extend([node_dict[current_node][0][0]])

            # comparing maximum of dt
            if main_dict[node_dict[current_node][1]][4] < \
                    edges[node_dict[current_node][2][0]][3]:

                main_dict[node_dict[current_node][1]][4]=\
                    edges[node_dict[current_node][2][0]][3]

            # adding length to label
            main_dict[node_dict[current_node][1]][1] += \
                edges[node_dict[current_node][2][0]][1]

            # adding path to next node to label
            if main_dict[node_dict[current_node][1]][2][-1]==\
                    edges[node_dict[current_node][2][0]][2][-1]:

                main_dict[node_dict[current_node][1]][2].extend(
                    edges[node_dict[current_node][2][0]][2][-2::-1])

            # adding path to next node to label
            else:

                main_dict[node_dict[current_node][1]][2].extend(
                    edges[node_dict[current_node][2][0]][2][1:])

            # adding edge number to label
            main_dict[node_dict[current_node][1]][3] = \
                node_dict[current_node][2][0]

            # putting next
            queue.put([node_dict[current_node][0][0],
                        node_dict[current_node][1]])

            # # deleting node from dict
            # del node_dict[current_node]

        assert(queue.qsize()>0),"contraction finished before all the nodes were seen"





    return finished_dict


def pruning_without_space(label):

    print "-----------------------------------------------------------"
    print "Label: ", label
    seg = np.load("/export/home/amatskev/Bachelor/data/graph_pruning/seg_0.npy")
    volume = extract_from_seg(seg, label)

    finished = np.load("/export/home/amatskev/Bachelor/"
                       "data/graph_pruning/finished_label_{0}.npy".format(label))
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

        for key in pre_plotting_dict.keys():

            if pre_plotting_dict[key] not in pre_plotting_dict.keys():
                counter = 1

                pre_plotting_dict[pre_plotting_dict[key]] = \
                    deepcopy(finished[pre_plotting_dict[key]][5])

    finished_pruned = np.array([finished[key][2] for key in pre_plotting_dict.keys()])

    return finished_pruned