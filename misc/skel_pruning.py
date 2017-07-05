import nifty_with_cplex as nifty
import numpy as np
from workflow.methods.functions_for_workflow import compute_graph_and_paths,close_cavities
import test_functions as tf



if __name__ == "__main__":

    volume=np.load(
        "/export/home/amatskev/Link to neuraldata/test/first_try_Volume.npy")

    volume=close_cavities(volume)
    skel_img,skel=tf.img_to_skel(volume)
    tf.plot_figure_and_path(volume,skel)
