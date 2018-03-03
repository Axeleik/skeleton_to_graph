import numpy as np
import nifty



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




if __name__ == "__main__":

    dt, skel_img=np.load("/mnt/localdata1/amatskev/debugging/bac/splC_z0/dt_skel-img.npy")

    graph,dt_features= extract_graph_and_distance_features(dt)

    print "debug"