import numpy as np
from Queue import Queue
from copy import deepcopy
# from misc.skel_pruning import extract_from_seg
from copy import copy

def adj(g,val):
    for adj_node,adj_edge in g.nodeAdjacency(val):
        print "Node: ",adj_node
        print "Edge: ", adj_edge



def terminal_func(start_queue,g,finished_dict,node_dict,main_dict,edges,nodes_list,intersecting_node_dict):

    queue = Queue()


    while start_queue.qsize():

        # draw from queue
        current_node, label = start_queue.get()


        #check the adjacency
        adjacency = np.array([[adj_node, adj_edge] for adj_node, adj_edge
                              in g.nodeAdjacency(current_node)])



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
                if (edges[adjacency[0][1]][1]) >= \
                        (main_dict[node_dict[adjacency[0][0]][1]][1]):
                    #writing to intersection_node so we can connect later on
                    if adjacency[0][0] in intersecting_node_dict.keys():
                        intersecting_node_dict[adjacency[0][0]].append(node_dict[adjacency[0][0]][1])
                    else:
                        intersecting_node_dict[adjacency[0][0]]=[node_dict[adjacency[0][0]][1]]

                    finished_dict[node_dict[adjacency[0][0]][1]] \
                        = deepcopy(main_dict[node_dict[adjacency[0][0]][1]])

                    del main_dict[node_dict[adjacency[0][0]][1]]
                    node_dict[adjacency[0][0]][1] = current_node

                else:
                    #writing to intersection_node so we can connect later on
                    if adjacency[0][0] in intersecting_node_dict.keys():
                        intersecting_node_dict[adjacency[0][0]].append(current_node)
                    else:
                        intersecting_node_dict[adjacency[0][0]]=[current_node]

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
            if start_queue.qsize() == 0 and len(nodes_list.keys())==6:

                # writing longest label at node to the others,
                # so we dont cut in half later
                for finished_label in [adj_node for adj_node,
                                                    adj_edge in g.nodeAdjacency(adjacency[0][0])
                                       if adj_node != node_dict[adjacency[0][0]][0][0]
                                       and adj_node != node_dict[adjacency[0][0]][1]]:
                    finished_dict[finished_label]. \
                        append(finished_label)

                for key in main_dict.keys():

                    # writing to intersection_node so we can connect later on
                    if main_dict[key][0][-1] in intersecting_node_dict.keys():
                        intersecting_node_dict[main_dict[key][0][-1]].append(key)
                    else:
                        intersecting_node_dict[main_dict[key][0][-1]] = [key]

                    finished_dict[key] = deepcopy(main_dict[key])

                    del main_dict[key]

                    finished_dict[key].append(key)

                # deleting node from dict
                # del node_dict[current_node]
                _ = queue.get()
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

    if (len(main_dict.keys()))==1 and len(nodes_list.keys())==4:
        key=main_dict.keys()[0]
        # writing to intersection_node so we can connect later on
        if main_dict[key][0][-2] in intersecting_node_dict.keys():
            intersecting_node_dict[main_dict[key][0][-2]].append(key)
        else:
            intersecting_node_dict[main_dict[key][0][-2]] = [key]

        for possible_unfinished_key in finished_dict.keys():
            if len(finished_dict[possible_unfinished_key])<6:
                finished_dict[possible_unfinished_key].append(main_dict.keys()[0])

        for key in main_dict.keys():
            finished_dict[key] = deepcopy(main_dict[key])

            del main_dict[key]

            finished_dict[key].append(key)
        _ = queue.get()

    return queue,finished_dict,node_dict,main_dict,intersecting_node_dict,min_array


def pruning_2_nodes(edges,term_list,pruning_factor):

    assert len(edges)==1, "if we have two nodes we should have only one edge"
    assert len(term_list)==2,"if we have two nodes we should have two term points"

    ratio=(edges[0][1]/2)/edges[0][3]

    if ratio>pruning_factor:
        return np.array(term_list)
    else:
        return []





#TODO check whether edgelist and termlist is ok (because of -1)
def graph_pruning(g,term_list,edges,nodes_list, dict_border_points=None):

    finished_dict={}
    node_dict={}
    main_dict={}
    start_queue=Queue()
    intersecting_node_dict={}
    last_dict={}
    level_dict={}
    # edges_and_lens=deepcopy(edges)

    pruning_factor=4

    for term_point in term_list:
        start_queue.put([term_point,term_point])

    assert start_queue.qsize()!=1
    #TODO implement clean case for 3 or 2 term_points
    if start_queue.qsize()==2:
        return pruning_2_nodes(edges,term_list,pruning_factor)

    queue,finished_dict,node_dict,main_dict,intersecting_node_dict,min_array = \
        terminal_func (start_queue, g, finished_dict,
                       node_dict, main_dict, copy(edges),
                       nodes_list,intersecting_node_dict)

    assert queue.qsize()!=1

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
            if (main_dict[label][1]) >= \
                    (main_dict[node_dict[current_node][1]][1]):

                # writing to intersection_node so we can connect later on
                if current_node in intersecting_node_dict.keys():
                    intersecting_node_dict[current_node].append(node_dict[current_node][1])
                else:
                    intersecting_node_dict[current_node] = [node_dict[current_node][1]]


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

                if current_node in intersecting_node_dict.keys():
                    intersecting_node_dict[current_node].append(label)
                else:
                    intersecting_node_dict[current_node] = [label]


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

        #finishing contraction
        if len(main_dict.keys())<3:


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


            for key in main_dict.keys():

                # writing to intersection_node so we can connect later on
                if main_dict[key][0][-1] in intersecting_node_dict.keys():
                    intersecting_node_dict[main_dict[key][0][-1]].append(key)
                else:
                    intersecting_node_dict[main_dict[key][0][-1]] = [key]

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





    return finished_dict,intersecting_node_dict,nodes_list


def actual_pruning(finished_dict, intersecting_node_dict, nodes_list):

    #Here begins the actual pruning
    # TODO maybe faster ?
    pre_plotting_dict = \
        {key: deepcopy(finished_dict[key][5]) for key in finished_dict.keys()
         if finished_dict[key][1] / finished_dict[key][4] > 6}




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