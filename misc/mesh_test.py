#from marching_cubes import march
import numpy as np
# from mayavi import mlab
import cPickle as pickle
from copy import deepcopy
from pyqtgraph.opengl import GLViewWidget, MeshData
from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem
from PyQt4 import QtGui
#from nilearn import datasets, plotting, image
#from neuro_seg_plot import NeuroSegPlot as nsp
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import skeletonize_3d
from skimage import measure
# from stl import mesh
# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
# import trimesh



def plot_mesh(vertices, faces, normals, values):
    """plot mesh """

    app = QtGui.QApplication([])
    view = GLViewWidget()

    mesh = MeshData(vertices / 100, faces)  # scale down - because camera is at a fixed position
    mesh._vertexNormals = normals

    item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

    view.addItem(item)
    view.show()
    app.exec_()


vertices, faces, normals, values = np.load("/export/home/amatskev/Bachelor/data/first_try/mesh_data.npy")

plot_mesh(vertices, faces, normals, values)




#
# volume = np.load("/export/home/amatskev/Bachelor/data/first_try/first_try_Volume.npy")
# testvolume=np.load("/export/home/amatskev/Bachelor/marching_cubes/test/sample.npy")
# vertices, faces, normals, values = measure.marching_cubes_lewiner(volume,
#                                                      spacing=(1,1,10))

