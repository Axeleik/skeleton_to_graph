
# Do the following for objects ['merged_split'][3]
#                          and ['merged_split'][8]
#
# 1. Iso surface of mc segmentation
# 2. Iso surface of mc segmentation + Path of merge detection + raw data + xyz planes
#    Make sure it is indicated that the path runs from border to border
# 3. Iso surface of mc segmentation + sampled paths
# 4. Iso surface of resolved result (+ xyz planes ?)
#    Iso surface of gt (mask with segmentation) (+ xyz planes ?)

from neuro_seg_plot import NeuroSegPlot as nsp
import vigra
import numpy as np
import pickle
from mayavi import mlab
from skimage.morphology import skeletonize_3d
from copy import deepcopy

def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]


def synchronize_labels(gt, newseg):

    newseg = newseg + np.amax(np.unique(gt))
    newseg[newseg == np.amax(np.unique(gt))] = 0

    z1, z2 = np.unique(gt, return_counts=True)
    z1 = z1[np.argsort(z2)][::-1]
    list = [0]
    for i in z1:
        print "i: ", i

        if i == 0:
            continue

        a, b = np.unique(newseg[gt == i], return_counts=True)
        a = a[np.argsort(b)][::-1]
        c = 0

        if a[0] == 0 and len(a) == 1:
            continue

        else:
            for idx, elem in enumerate(a):
                if elem in list:
                    continue
                c = elem
                newseg[newseg == c] = i
                print "C: ", c
                break

        print "np.unique(newseg):", np.unique(newseg)
        list.append(i)

    for idx, elem in enumerate(np.unique(newseg)[np.unique(newseg) > np.amax(np.unique(gt))]):
        newseg[newseg == elem] = np.amax(np.unique(gt)) + idx + 1

# Just keep the biggest branch
def largest_lbls(im, n):
    im_unique, im_counts = np.unique(im, return_counts=True)
    largest_ids = np.argsort(im_counts)[-n - 1:-1]
    return im_unique[largest_ids][::-1]



# with open(
#         '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/evaluation.pkl',
#         mode='r'
# ) as f:
#     eval_data = pickle.load(f)
#
# cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/cache/'
folder="/mnt/localdata1/amatskev/debugging/vi_scores/sanity_pruning_12/splB_z1/"
# Load segmentation
print 'Loading segmentation...'
seg=np.load(folder + "ref_seg.npy")
print 'Loading resolved segmentation...'
newseg=np.load(folder + "res_seg.npy")
print 'Loading gt...'
gt=np.load(folder + "gt.npy")

print 'Loading raw data...'
raw = vigra.readHDF5(
    '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
    'z/0/raw'
)

paths=np.load(folder + "all_paths.npy")
paths_to_objs=np.load(folder + "paths_to_objs.npy")


idx = 140
print 'ID = {}'.format(idx)

number_of_objects_to_plot = 1

plot_type = 1

mask_gt = gt != 455722
seg_idx=deepcopy(gt)
seg_idx[mask_gt] = 0
crop = find_bounding_rect(seg_idx)
seg_idx = seg_idx[crop]
seg_idx[seg_idx > 0] = 1
skel_img=skeletonize_3d(seg_idx)
where_np=np.where(skel_img)
skel_path=np.zeros((len(where_np[0]),3))
skel_path[:,0]=where_np[0]
skel_path[:,1]=where_np[1]
skel_path[:,2]=where_np[2]


nsp.start_figure()
nsp.add_iso_surfaces(seg_idx, anisotropy=[10, 1, 1], vmin=0, vmax=1, opacity=0.4)
nsp.add_path(skel_path.swapaxes(0, 1), anisotropy=[10, 1, 1],representation="points",line_width=100)
mlab.view(azimuth=135)
nsp.show()


mask = seg != idx
seg[mask] = 0
crop = find_bounding_rect(seg)
seg = seg[crop]
gt[mask]=0
seg[seg > 0] = 1
# seg = seg[:200, :200, :100]
nsp.add_iso_surfaces(seg, anisotropy=[10, 1, 1], vmin=0, vmax=3, opacity=0.8)
mlab.view(azimuth=135)


# nsp.start_figure()

if plot_type == 1:
    # ---------------------------------
    # 1. Iso surface of mc segmentation
    seg[seg > 0] = 1
    # seg = seg[:200, :200, :100]
    nsp.add_iso_surfaces(seg, anisotropy=[10, 1, 1], vmin=0, vmax=3, opacity=0.8)
    mlab.view(azimuth=135)

elif plot_type == 2:

    # ---------------------------------
    # 2. Iso surface of mc segmentation + Path of merge detection + raw data as xyz planes

    # Load path
    paths = np.load(folder + "all_paths.npy")
    paths_to_objs = np.load(folder + "paths_to_objs.npy")

    current_paths = np.array(paths)[paths_to_objs == 12]

    path_id = 0
    for idx,path in enumerate(current_paths):
        if gt[path[0][0],path[0][1],path[0][2]]!=gt[path[-1][0],path[-1][1],path[-1][2]] and\
                        newseg[path[0][0],path[0][1],path[0][2]]==newseg[path[-1][0],path[-1][1],path[-1][2]]:
            path_id=idx
            print path_id
    path_id = 7
    # Get specific path from selected idx
    path = current_paths[path_id]
    # path[:, 0] = path[:, 0] - (crop[0]).start
    # path[:, 1] = path[:, 1] - (crop[1]).start
    # path[:, 2] = path[:, 2] - (crop[2]).start



    # gt=gt[crop]
    uniq_gt=np.unique(gt)
    projected_gt=np.zeros(gt.shape)
    # for idx,l in enumerate(largest_lbls(gt, number_of_objects_to_plot)):
    #     print "l: ",l
    #     projected_gt[gt == l] = idx+1

    newseg = newseg[crop]
    uniq_newseg = np.unique(newseg)
    print "unique_gt: ",len(uniq_gt)
    projected_newseg = np.zeros(newseg.shape)
    for idx,l in enumerate(largest_lbls(newseg, number_of_objects_to_plot)):
        print "l: ",l
        projected_newseg[newseg == l] = idx + 1
    for idx,l in enumerate(largest_lbls(gt, number_of_objects_to_plot)):
        print "l: ",l
        projected_newseg[gt == l] = idx+5
        projected_gt[gt == l] = idx+5
    raw = raw[crop]
    for idx,l in enumerate(largest_lbls(newseg, number_of_objects_to_plot)):
        print "l: ",l
        projected_gt[newseg == l] = idx + 1

    nsp.start_figure()

    nsp.add_iso_surfaces(projected_newseg, anisotropy=[10, 1, 1], vmin=np.amin(projected_newseg), vmax=np.amax(projected_newseg), opacity=0.6)


    nsp.add_path(path.swapaxes(0, 1), anisotropy=[10, 1, 1])
    # nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    mlab.view(azimuth=135)

    nsp.start_figure()

    nsp.add_iso_surfaces(projected_gt, anisotropy=[10, 1, 1], vmin=np.amin(projected_gt), vmax=np.amax(projected_gt), opacity=1)


    nsp.add_path(path.swapaxes(0, 1), anisotropy=[10, 1, 1])
    # nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    mlab.view(azimuth=135)

elif plot_type ==3:

    # ---------------------------------
    # 3. Iso surface of mc segmentation + sampled paths

    raw = raw[crop]

    # Load paths
    with open(cache_folder + 'path_data/resolve_paths_{}.pkl'.format(idx), mode='r') as f:
        paths = pickle.load(f)
    with open(cache_folder + 'path_data/resolve_paths_probs_{}.pkl'.format(idx), mode='r') as f:
        path_weights = pickle.load(f)

    # Adjust path position to the cropped images
    for i in xrange(0, len(paths)):
        path = paths[i]
        path = np.swapaxes(path, 0, 1)
        path[0] = path[0] - crop[0].start
        path[1] = path[1] - crop[1].start
        path[2] = path[2] - crop[2].start
        path = np.swapaxes(path, 0, 1)
        paths[i] = path

    # Create custom colormap dictionary
    cdict_BlYlRd = {'red': ((0.0, 0.0, 0.0),  # Very likely merge position
                            (0.5, 1.0, 1.0),  #
                            (1.0, 1.0, 1.0)),  # Very likely non-merge position
                    #
                    'green': ((0.0, 1.0, 1.0),  # No green at the first stop
                              (0.5, 1.0, 1.0),
                              (1.0, 0.0, 0.0)),  # No green for final stop
                    #
                    'blue': ((0.0, 0.0, 0.0),
                             (0.5, 0.0, 0.0),
                             (1.0, 0.0, 0.0))}

    lut = nsp.lut_from_colormap(cdict_BlYlRd, 256)

    path_vmin = min(path_weights)
    path_vmax = max(path_weights)


    nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(seg), opacity=0.5)
    nsp.plot_multiple_paths_with_mean_class(
        paths, path_weights,
        custom_lut=lut,
        anisotropy=[1, 1, 10],
        vmin=path_vmin, vmax=path_vmax
    )
    nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    mlab.view(azimuth=135)

elif plot_type == 4:

    # ---------------------------------
    # 4. Iso surface of resolved result (+ xyz planes ?)
    #    Iso surface of gt (mask with segmentation) (+ xyz planes ?)

    newseg[mask] = 0
    newseg = newseg[crop]
    raw = raw[crop]
    gt[mask] = 0
    gt = gt[crop]



    # from copy import deepcopy
    tgt = np.zeros(gt.shape, dtype=gt.dtype)
    tnewseg = np.zeros(newseg.shape, dtype=newseg.dtype)
    for l in largest_lbls(gt, number_of_objects_to_plot):
        tgt[gt == l] = l
    for l in largest_lbls(newseg, number_of_objects_to_plot):
        tnewseg[newseg == l] = l
    gt = tgt
    newseg = tnewseg

    # newseg, _, _ = vigra.analysis.relabelConsecutive(newseg, start_label=0)
    # gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0)

    synchronize_labels(gt, newseg)

    # nsp.add_xyz_planes(raw, anisotropy=[1, 1, 10], xpos=1250)
    nsp.add_iso_surfaces(gt, anisotropy=[10, 1, 1], vmin=np.amin(gt), vmax=np.amax(gt), opacity=0.8)
    # mlab.view(azimuth=135)
    #
    # nsp.start_figure()

    # nsp.add_xyz_planes(raw, anisotropy=[10, 1, 1], xpos=1250)
    # nsp.add_iso_surfaces(gt, anisotropy=[10, 1, 1], vmin=0, vmax=np.amax(gt), opacity=0.8)
    # mlab.view(azimuth=135)

nsp.show()
