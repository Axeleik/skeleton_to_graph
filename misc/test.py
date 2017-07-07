import numpy as np
# from mesh_test import plot_mesh
# import test_functions as tf
# from neuro_seg_plot import NeuroSegPlot as nsp
from skimage.measure import label
import nifty_with_cplex as nifty
import nifty_with_cplex.graph.rag as nrag


def close_cavities(volume):
    volume[volume==0]=2
    lab=label(volume)
    if len(np.unique(lab))==2:
        return volume
    count,what=0,0
    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    volume[lab==what]=0
    volume[lab != what] = 1

    return volume

def example_rag():
    x = np.zeros((25, 25), dtype='uint32')
    x[:13, :13] = 1
    x[12:, 12:] = 2
    rag = nrag.gridRag(x)
    return rag


    # iterate over all the nodes in the rag
    # for each, print its adjacent nodes and the corresponding edge
def node_iteration(rag):

    # iterate over the nodes
    for node_id in xrange(rag.numberOfNodes):

        # iterate over the node adjacency of this node
        print "Node:", node_id, "is adjacent to:"
        for adj_node, adj_edge in rag.nodeAdjacency(node_id):
            print "Node:", adj_node
            print "via Edge", adj_edge


if __name__ == "__main__":










    volume=np.load(
        "/export/home/amatskev/Link to neuraldata/test/first_try_Volume.npy")

















    rag = example_rag()
    node_iteration(rag)











    # volume=np.zeros((250,250,250))
    # volume[75:175,75:175,75:175]=1
    # volume[100:150,100:150,100:150] = 0
    #
    # _,skel=tf.img_to_skel(volume)
    #
    # tf.plot_figure_and_path(volume,skel)
    diff_vol=np.load("/export/home/amatskev/Link to neuraldata/test/difficult_volume.npy")
    diff_vol=close_cavities(diff_vol)

    vol=np.load("/export/home/amatskev/Link to neuraldata/test/difficult_volume_relabeled.npy")
    vol[vol==2]=0
    _,skel=tf.img_to_skel(vol)

    tf.plot_figure_and_path(vol, skel)

    print "test"

    # faces_unfinished = np.loadtxt("/export/home/amatskev/Bachelor/data/cgal/elephant/elephant_faces.txt")
    #
    # vertices = np.loadtxt("/export/home/amatskev/Bachelor/data/cgal/elephant/elephant_vertices.txt")
    #
    # faces = np.array([[int(e1), int(e2), int(e3)] for (e0, e1, e2, e3) in faces_unfinished])
    #
    #
    # vertices1, faces1_unfinished, normals, values = np.load("/export/home/amatskev/Bachelor/data/first_try/mesh_data.npy")
    #
    # faces1 = np.array([[int(3), int(e1), int(e2), int(e3)] for (e1, e2, e3) in faces1_unfinished])
    #
    #
    #
    # plot_mesh(vertices,faces)

