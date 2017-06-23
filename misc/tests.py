from neuro_seg_plot import NeuroSegPlot as nsp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from copy import deepcopy
import cPickle as pickle
from skimage.morphology import skeletonize_3d
import scipy
from scipy import interpolate



def interpolation(data,param=5000):
    """interpolates the "stairlike" path"""

    tck, u = interpolate.splprep(data, s=param,k=3)
    new = interpolate.splev(np.linspace(0, 1, len(data[0])), tck)

    return new


def draw_path(data):
    """Draws 3D path given in form of numpy array"""

    data = np.array([(elem1 * 10, elem2, elem3) for elem1, elem2, elem3 in data])
    data = data.transpose()

    #new=interpolation(data)

    fig = plt.figure(1)
    ax = Axes3D(fig)
    #ax.set_axis_off()

    ax.plot(data[0], data[1], data[2], label='original_true', lw=2, c='Black')  # gezackt
    #ax.scatter(data2[0], data2[1], data2[2], label='original_true', lw=0, c='Dodgerblue')  # gezackt

    ax.legend()
    plt.show()


def printname(name):
    """shows all directories and files in h5py file"""

    print name


def extract_from_h5py():
    "extracts data from h5py"

    f = h5py.File("/mnt/localdata03/amatskev/cremi.paths.crop.split_z.h5", mode='r')
    #f.visit(printname)

    data = np.array(f["z_predict0/truepaths/z/0/beta_0.5/216/0"])

    return data


def plot_figure_and_path(figure,path):
    """plots figure and path"""

    nsp.start_figure()
    nsp.add_iso_surfaces(figure, [1, 1, 10])
    nsp.add_path(np.array(path.transpose()), anisotropy=[1, 1, 10])
    nsp.show()

def img_to_skel(img):
    """transforms image with skeleton to a numpy array of skeleton"""

    skel_img = skeletonize_3d(img)

    skel = np.array(
        [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0],
                                            np.where(skel_img)[1],
                                            np.where(skel_img)[2])])

    return skel_img,skel

if __name__ == "__main__":


    data = extract_from_h5py()

    draw_path(data)