from neuro_seg_plot import NeuroSegPlot as nsp
import pickle
import vigra
import numpy as np
import mayavi

# # Load path
# path_file = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1/cache/path_data/path_splB_z1.pkl'
# with open(path_file, mode='r') as f:
#     path_data = pickle.load(f)
#
# paths = path_data['paths']
# paths_to_objs = path_data['paths_to_objs']


def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]


# with open(
#         '/media/axeleik/EA62ECB562EC8821/data/plotted_figure/170331_splB_z1_defcor/evaluation.pkl',
#         mode='r'
# ) as f:
#     eval_data = pickle.load(f)

# Load segmentation
print 'Loading segmentation...'
seg=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/ref_seg.npy")
print 'Loading resolved segmentation...'
newseg=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/res_seg.npy")
print 'Loading gt...'
gt=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/gt.npy")

#seg = seg[0:200, 0]

id = 1131
print 'ID = {}'.format(id)

number_of_objects_to_plot_newseg = 5
number_of_objects_to_plot_gt = 8
# id = 101.0
# path_to_obj = paths_to_objs[id]
# print 'path_to_obj = {}'.format(path_to_obj)
# seg[seg != path_to_obj] = 0

mask = seg != id

seg[mask] = 0
newseg[mask] = 0
gt[mask] = 0

crop = find_bounding_rect(seg)
seg = seg[crop]
newseg = newseg[crop]
gt = gt[crop]

print seg.shape

print 'unique in seg: {}'.format(np.unique(seg))
print 'unique in newseg: {}'.format(np.unique(newseg))
print 'unique in gt: {}'.format(np.unique(gt))

# gt_unique, gt_counts = np.unique(gt, return_counts=True)
# for l in gt_unique[gt_counts < 1000]:
#     gt[gt == l] = 0


# Just keep the biggest branch
def largest_lbls(im, n):
    im_unique, im_counts = np.unique(im, return_counts=True)
    largest_ids = np.argsort(im_counts)[-n-1:-1]
    return im_unique[largest_ids]

# from copy import deepcopy
tgt = np.zeros(gt.shape, dtype=gt.dtype)
tnewseg = np.zeros(newseg.shape, dtype=newseg.dtype)
for l in largest_lbls(gt, number_of_objects_to_plot_gt):
    tgt[gt == l] = l
for l in largest_lbls(newseg, number_of_objects_to_plot_newseg):
    tnewseg[newseg == l] = l
gt = tgt
newseg = tnewseg

# sorted_ids = np.argsort(gt_counts)
# sorted_uniques = gt_unique[sorted_ids]

seg, _, _ = vigra.analysis.relabelConsecutive(seg, start_label = 0,keep_zeros=False)
newseg, _, _ = vigra.analysis.relabelConsecutive(newseg, start_label=0,keep_zeros=False)
gt, _, _ = vigra.analysis.relabelConsecutive(gt, start_label=0,keep_zeros=False)

newseg = newseg + np.amax(np.unique(gt))
newseg[newseg == np.amax(np.unique(gt))] = 0


z1,z2=np.unique(gt,return_counts=True)
z1=z1[np.argsort(z2)][::-1]
list=[0]
for i in z1:
    print "i: ", i


    if i ==0:
        continue


    a, b = np.unique(newseg[gt == i], return_counts=True)
    a = a[np.argsort(b)][::-1]
    c = 0

    if a[0] == 0 and len(a)==1:
        continue

    else:
        for idx,elem in enumerate(a):
            if elem in list:
                continue
            c = elem
            newseg[newseg == c] = i
            print "C: ",c
            break







    print "np.unique(newseg):", np.unique(newseg)
    list.append(i)

for idx,elem in enumerate(np.unique(newseg)[np.unique(newseg)>np.amax(np.unique(gt))]):
    newseg[newseg==elem]=np.amax(np.unique(gt))+idx+1


print 'Starting to plot...'

nsp.start_figure()

#nsp.add_path(paths[id].swapaxes(0, 1), anisotropy=[1, 1, 10])
#nsp.add_iso_surfaces(seg, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax(seg), opacity=0.5)
#nsp.start_figure()

nsp.add_iso_surfaces(gt, anisotropy=[10, 1, 1], vmin=0, vmax=np.amax((np.amax(newseg),np.amax(gt))), opacity=0.5)

#nsp.start_figure()

#nsp.add_iso_surfaces(gt, anisotropy=[1, 1, 10], vmin=0, vmax=np.amax((np.amax(newseg),np.amax(gt))), opacity=0.5)
nsp.show()





