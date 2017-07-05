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



def plot_mesh(vertices, faces, normals=False, values=False):
    """plot mesh """

    app = QtGui.QApplication([])
    view = GLViewWidget()

    mesh = MeshData(vertices, faces)  # scale down - because camera is at a fixed position

    if type(normals)!=bool:
        mesh._vertexNormals = normals

    item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

    view.addItem(item)
    view.show()
    app.exec_()



if __name__ == "__main__":

    vertices, faces, normals, values = np.load("/export/home/amatskev/Bachelor/data/first_try/mesh_data.npy")

    plot_mesh(vertices, faces)




#
# volume = np.load("/export/home/amatskev/Bachelor/data/first_try/first_try_Volume.npy")
# testvolume=np.load("/export/home/amatskev/Bachelor/marching_cubes/test/sample.npy")
# vertices, faces, normals, values = measure.marching_cubes_lewiner(volume,
#                                                      spacing=(1,1,10))
















# volume = np.load("/export/home/amatskev/Bachelor/data/first_try/first_try_Volume.npy")
# where=np.where(volume)
# volume=volume[where[0][0]:where[0][-1],where[1][0]:where[1][-1],where[2][0]:where[2][-1]]
# volume = gaussian_filter(volume,(2,2,0.2))
# skel_img = skeletonize_3d(volume)
# skel = np.array(
#         [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0],
#                                             np.where(skel_img)[1],
#                                             np.where(skel_img)[2])])
# nsp.start_figure()
# nsp.add_iso_surfaces(volume, [1, 1, 10],opacity=0.3)
# nsp.add_path(np.array(skel.transpose()), anisotropy=[1, 1, 10])
#
# nsp.show()

# testvolume=np.load("/export/home/amatskev/Bachelor/marching_cubes/test/sample.npy")

# new=np.zeros((volume.shape[0],volume.shape[1],volume.shape[2]*10))
#
# for slice in xrange(0,volume.shape[2]):
#     for z in xrange(0,10):
#         new[:, :, slice*10 + z] = volume[:, :, slice]
#
# new.dtype="int32"
# print "hi"
#
# np.save("/export/home/amatskev/Bachelor/data/first_try/mesh_data.npy",(vertices, faces, normals, values))
#
# your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         your_mesh.vectors[i][j] = vertices[f[j],:]
#
# your_mesh.save('/export/home/amatskev/Link to neuraldata/test/first_try_sample.stl')
# '/export/home/amatskev/Link to neuraldata/test/first_try_mesh.stl'
#
# print your_mesh.trimesh.is_watertight
# your_mesh = trimesh.load_mesh('/export/home/amatskev/Link to neuraldata/test/first_try_mesh.stl')
# print "blabla ", your_mesh.is_watertight
#
#
# # Create a new plot
# figure = pyplot.figure()
# axes = mplot3d.Axes3D(figure)
#
# # Load the STL files and add the vectors to the plot
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
#
# # Auto scale to the mesh size
# scale = your_mesh.points.flatten(-1)
# axes.auto_scale_xyz(scale, scale, scale)
# is_watertight
# # Show the plot to the screen
# pyplot.show()
#
#
#
#
#
#
#
#
#
# app = QtGui.QApplication([])
# view = GLViewWidget()
#
# mesh = MeshData(vertices/100, faces)  # scale down - because camera is at a fixed position
# mesh._vertexNormals = normals
#
# item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")
#
# view.addItem(item)
# view.show()
# app.exec_()
#
#
# mlab.figure()
#
#
# mlab.triangular_mesh(vertices.transpose()[0],
#                      vertices.transpose()[1],
#                      vertices.transpose()[2],
#                      faces)
# mlab.show()