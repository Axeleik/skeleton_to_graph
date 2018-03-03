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
import vigra
# from joblib import Parallel,delayed
#import volumina_viewer
# from concurrent import futures
# from multicut_src.false_merges.path_computation import parallel_wrapper
from math import sqrt

def close_cavities(volume):
    test_vol=np.zeros(volume.shape)
    test_vol[volume==0]=2
    test_vol[volume==1]=1
    test_vol=np.pad(test_vol, 1, "constant", constant_values=2)

    lab=label(test_vol,connectivity=1)
    if len(np.unique(lab))==2:
        return volume
    count,what=0,0
    for uniq in np.unique(lab):
        if len(np.where(lab == uniq)[0])> count:
            count=len(np.where(lab == uniq)[0])
            what=uniq

    test_vol[lab==what]=0
    test_vol[lab != what] = 1
    return test_vol[1:-1,1:-1,1:-1]

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

def find_bounding_rect(image):

    # Bands
    bnds = np.flatnonzero(image.sum(axis=0).sum(axis=0))
    # Rows
    rows = np.flatnonzero(image.sum(axis=1).sum(axis=1))
    # Columns
    cols = np.flatnonzero(image.sum(axis=2).sum(axis=0))

    return np.s_[rows.min():rows.max()+1, cols.min():cols.max()+1, bnds.min():bnds.max()+1]


def plot_figure_and_path(figure,paths=[],plot_path=True,anisotropy_input=[10,1,1],opacity=0.3):
    """plots figure and path"""
    from mayavi import mlab

    # sub_paths = nsp.multiple_paths_for_plotting(paths)
    print len(paths)
    seg_figure = nsp()
    # sub_paths = seg_figure.multiple_paths_for_plotting(paths)
    seg_figure.start_figure()

    seg_figure.add_iso_surfaces(figure, anisotropy_input,vmin=np.amin(figure),vmax=np.amax(figure),opacity=opacity)
    # path=np.array(paths[0])
    # sub_paths = seg_figure.multiple_paths_for_plotting(paths)
    if plot_path==True:
            seg_figure.add_path(paths.swapaxes(0, 1),anisotropy=anisotropy_input,representation="points",line_width=10)
    mlab.view(azimuth=38)


    seg_figure.show()

def img_to_skel(img):
    """transforms image with skeleton to a numpy array of skeleton"""

    skel_img = skeletonize_3d(img)

    skel = np.array(
        [[e1, e2, e3] for e1, e2, e3 in zip(np.where(skel_img)[0],
                                            np.where(skel_img)[1],
                                            np.where(skel_img)[2])])

    return skel_img,skel

def write(file,path):

    with open(path, mode='w') as f:
        pickle.dump(file, f)


def read(path):

    with open(path, mode='r') as f:
        file = pickle.load(f)

    return file


def view(filepaths, filekeys, names=None, types=None, swapaxes=None, crop=None):
    inputs = []
    this_type = None
    swp = None
    if crop is None:
        crop = np.s_[:, :, :]
    for idx, filepath in enumerate(filepaths):
        if types is not None:
            this_type = types[idx]
        if swapaxes is not None:
            swp = swapaxes[idx]
        if this_type is not None:
            inputs.append(vigra.readHDF5(filepath, filekeys[idx]).astype(this_type)[crop])
        else:
            inputs.append(vigra.readHDF5(filepath, filekeys[idx])[crop])
        if swp is not None:
            inputs[-1] = inputs[-1].swapaxes(*swp)
        print inputs[-1].shape

        inputs[0][inputs[0] != 25] = 0
        # inputs[0][inputs[0] == 25] = 1

    volumina_viewer.volumina_n_layer(
        inputs,
        names
    )

def python_region_features_extractor_2_mc(single_vals):

    path_features = []

    [path_features.extend([np.mean(vals), np.var(vals), sum(vals),
                            max(vals), min(vals),
                            scipy.stats.kurtosis(vals),
                           (((vals - vals.mean()) / vals.std(ddof=0)) ** 3).mean()
])
     for vals in single_vals]

    return np.array(path_features)





def path_features_from_feature_images(
        ds,
        inp_id,
        paths,
        anisotropy_factor):

    # FIXME for now we don't use fastfilters here
    feat_paths = ds.make_filters(inp_id, anisotropy_factor)
    # print feat_paths
    # TODO sort the feat_path correctly
    # load the feature images ->
    # FIXME this might be too memory hungry if we have a large global bounding box

    # compute the global bounding box
    global_min = np.min(
        np.concatenate([np.min(path, axis=0)[None, :] for path in paths], axis=0),
        axis=0
    )
    global_max = np.max(
        np.concatenate([np.max(path, axis=0)[None, :] for path in paths], axis=0),
        axis=0
    ) + 1
    # substract min coords from all paths to bring them to new coordinates
    paths_in_roi = [path - global_min for path in paths]
    roi = np.s_[
        global_min[0]:global_max[0],
        global_min[1]:global_max[1],
        global_min[2]:global_max[2]
    ]

    # load features in global boundng box
    feature_volumes = []
    import h5py
    for path in feat_paths:
        with h5py.File(path) as f:
            feat_shape = f['data'].shape
            # we add a singleton dimension to single channel features to loop over channel later
            if len(feat_shape) == 3:
                feature_volumes.append(f['data'][roi][..., None])
            else:
                feature_volumes.append(f['data'][roi])
    stats = ExperimentSettings().feature_stats

    def extract_features_for_path(path):

        # calculate the local path bounding box
        min_coords  = np.min(path, axis=0)
        max_coords = np.max(path, axis=0)
        max_coords += 1
        shape = tuple(max_coords - min_coords)
        path_image = np.zeros(shape, dtype='uint32')
        path -= min_coords
        # we swapaxes to properly index the image properly
        path_sa = np.swapaxes(path, 0, 1)
        path_image[path_sa[0], path_sa[1], path_sa[2]] = 1

        path_features = []
        for feature_volume in feature_volumes:
            for c in range(feature_volume.shape[-1]):
                path_roi = np.s_[
                    min_coords[0]:max_coords[0],
                    min_coords[1]:max_coords[1],
                    min_coords[2]:max_coords[2],
                    c  # wee need to also add the channel to the slicing
                ]
                extractor = vigra.analysis.extractRegionFeatures(
                    feature_volume[path_roi],
                    path_image,
                    ignoreLabel=0,
                    features=stats
                )
                # TODO make sure that dimensions match for more that 1d stats!
                path_features.extend(
                    [extractor[stat][1] for stat in stats]
                )
        # ret = np.array(path_features)[:,None]
        # print ret.shape
        return np.array(path_features)[None, :]

    if len(paths) > 1:

        # We parallelize over the paths for now.
        # TODO parallelizing over filters might in fact be much faster, because
        # we avoid the single threaded i/o in the beginning!
        # it also lessens memory requirements if we have less threads than filters
        # parallel
        with futures.ThreadPoolExecutor(max_workers=ExperimentSettings().n_threads) as executor:
            tasks = []
            for p_id, path in enumerate(paths_in_roi):
                tasks.append(executor.submit(extract_features_for_path, path))
            out = np.concatenate([t.result() for t in tasks], axis=0)

    else:

        out = np.concatenate([extract_features_for_path(path) for path in paths_in_roi])

    # serial for debugging
    # out = []
    # for p_id, path in enumerate(paths_in_roi):
    #     out.append( extract_features_for_path(path) )
    # out = np.concatenate(out, axis = 0)

    assert out.ndim == 2, str(out.shape)
    assert out.shape[0] == len(paths), str(out.shape)
    # TODO checkfor correct number of features

    return out

def bl(a):
    del a
    return 3

def true_false_indexing(label,
                paths, gt):

    all_paths_single=[]
    paths_to_objs_single=[]
    path_classes_single=[]


    for path in paths:

        if len(np.unique(gt[np.array(path).transpose()[0],
               np.array(path).transpose()[1],
               np.array(path).transpose()[2]]))>1:
            bool_crossing=True

        else:
            bool_crossing=False


        if gt[path[0][0], path[0][1], path[0][2]] ==\
                gt[path[-1][0], path[-1][1], path[-1][2]]:

            if bool_crossing:
                continue

            else:
                all_paths_single.append(np.array(path))
                paths_to_objs_single.append(label)
                path_classes_single.append(False)

        else:
            all_paths_single.append(np.array(path))
            paths_to_objs_single.append(label)
            path_classes_single.append(True)


    return all_paths_single,paths_to_objs_single,path_classes_single

def norm3d(point1,point2):

    anisotropy = [10., 1, 1]
    return sqrt(((point1[0] - point2[0])* anisotropy[0])*((point1[0] - point2[0])* anisotropy[0])+
                 ((point1[1] - point2[1])*anisotropy[1])*((point1[1] - point2[1])*anisotropy[1])+
                 ((point1[2] - point2[2]) * anisotropy[2])*((point1[2] - point2[2]) * anisotropy[2]))


def path_index_shortage_pointss_return(lens_path,shortage_length):
    """returns index point where we cut off our paths for shortage"""


    sum_front=0
    sum_back=0


    for idx,val in enumerate(lens_path):

        sum_front=sum_front+val

        if sum_front >= shortage_length:
            index_front = idx
            break


    for idx, val in enumerate(lens_path[::-1]):

        sum_back = sum_back + val

        if sum_back >= shortage_length:
            index_back = len(lens_path)- 1 - idx
            break


    assert (index_front <= index_back)

    # assuring we dont
    if index_back-index_front<2:
        index_front=index_front-1
        index_back=index_back+1

    assert (index_back < len(lens_path))
    assert (index_front >= 0)

    return index_front,index_back





def shorten_paths(paths_full_raw,shortage_ratio=0.1):
    """for shortening paths so they dont end exactly at the border """



    assert(shortage_ratio<=0.5)

    paths_full=[path for path in paths_full_raw if len(path)>5]

    if len(paths_full)==0:
        return paths_full_raw

    # collects the "length" for every point of every path
    lens_paths = [np.concatenate(([norm3d(path[0], path[1]) / 2],
                                 [norm3d(path[idx - 1], path[idx]) / 2 +
                                  norm3d(path[idx], path[idx + 1]) / 2
                                  for idx, point in enumerate(path)
                                  if idx != 0 and idx != len(path) - 1],
                                 [norm3d(path[-2], path[-1]) / 2]))
                 for path in paths_full]

    overall_lengths=[sum(val) for val in lens_paths]

    shortage_lengths=[overall_length*shortage_ratio for overall_length in overall_lengths]

    indexes_front_back_array=[path_index_shortage_pointss_return(lens_paths[idx],shortage_lengths[idx]) for idx,val in enumerate(lens_paths)]

    paths_cut=[paths_full[idx][indexes_front_back_array[idx][0]:indexes_front_back_array[idx][1]] for idx,val in enumerate(paths_full)]

    return paths_cut

if __name__ == "__main__":
    import h5py
    filename = '/export/home/amatskev/Downloads/groundtruth.h5'
    f = h5py.File(filename, 'r')

    test_vol=np.array(f["stack"])



    ref_seg=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/ref_seg.npy")
    res_seg=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/res_seg.npy")
    gt=np.load("/mnt/localdata1/amatskev/debugging/vi_scores/splA_z0/gt.npy")
    gt_to_shange=deepcopy(gt)
    gt_to_shange[:32,387:460,559:700]=1
    mask = gt != 9299
    gt[mask]=0
    # to_show_ref=deepcopy(ref_seg)
    bounds=find_bounding_rect(gt)


    # to_show_ref=to_show_ref[bounds]
    # plot_figure_and_path(to_show_ref,[],False)

    # to_show_res=deepcopy(res_seg)
    # to_show_res[mask]=0
    # to_show_res=to_show_res[bounds]
    # plot_figure_and_path(to_show_res,[],False)
    # #
    # to_show_gt=deepcopy(gt)
    gt_to_shange[mask]=0
    gt_to_shange=gt_to_shange[bounds]

    plot_figure_and_path(gt_to_shange,[],False)



    #
    # probs=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/probs.npy")
    #
    #
    #
    # gt=np.load("/mnt/localdata1/amatskev/debugging/features/labels/gt_C1.npy")
    #
    # paths=np.load("/mnt/localdata1/amatskev/debugging/features/labels/paths_4_nice.npy")
    # path_classes = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                        for path in paths]
    # print "True: ",len(np.where(np.array(path_classes)==True)[0])
    # print "False: ",len(np.where(np.array(path_classes)==False)[0])
    # paths_shorten_005=shorten_paths(paths,0.05)
    # path_classes_shorten_005 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                        for path in paths_shorten_005]
    # print "True_shorten_005: ",len(np.where(np.array(path_classes_shorten_005)==True)[0])
    # print "False_shorten_005: ",len(np.where(np.array(path_classes_shorten_005)==False)[0])
    # paths_shorten_01=shorten_paths(paths,0.1)
    # path_classes_shorten_01 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                        for path in paths_shorten_01]
    # print "True_shorten_01: ",len(np.where(np.array(path_classes_shorten_01)==True)[0])
    #
    # print "False_shorten_01: ",len(np.where(np.array(path_classes_shorten_01)==False)[0])
    #
    # paths = np.load("/mnt/localdata1/amatskev/debugging/features/labels/paths_8_nice.npy")
    # path_classes = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                          gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                 for path in paths]
    # print "True: ", len(np.where(np.array(path_classes) == True)[0])
    # print "False: ", len(np.where(np.array(path_classes) == False)[0])
    # paths_shorten_005 = shorten_paths(paths, 0.05)
    # path_classes_shorten_005 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                      gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                             for path in paths_shorten_005]
    # print "True_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == True)[0])
    # print "False_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == False)[0])
    # paths_shorten_01 = shorten_paths(paths, 0.1)
    # path_classes_shorten_01 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                     gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                            for path in paths_shorten_01]
    # print "True_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == True)[0])
    #
    # print "False_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == False)[0])
    #
    # paths = np.load("/mnt/localdata1/amatskev/debugging/features/labels/paths_11_nice.npy")
    # path_classes = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                          gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                 for path in paths]
    # print "True: ", len(np.where(np.array(path_classes) == True)[0])
    # print "False: ", len(np.where(np.array(path_classes) == False)[0])
    # paths_shorten_005 = shorten_paths(paths, 0.05)
    # path_classes_shorten_005 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                      gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                             for path in paths_shorten_005]
    # print "True_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == True)[0])
    # print "False_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == False)[0])
    # paths_shorten_01 = shorten_paths(paths, 0.1)
    # path_classes_shorten_01 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                     gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                            for path in paths_shorten_01]
    # print "True_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == True)[0])
    #
    # print "False_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == False)[0])
    #
    # paths = np.load("/mnt/localdata1/amatskev/debugging/features/labels/paths_14_nice.npy")
    # path_classes = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                          gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                 for path in paths]
    # print "True: ", len(np.where(np.array(path_classes) == True)[0])
    # print "False: ", len(np.where(np.array(path_classes) == False)[0])
    # paths_shorten_005 = shorten_paths(paths, 0.05)
    # path_classes_shorten_005 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                      gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                             for path in paths_shorten_005]
    # print "True_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == True)[0])
    # print "False_shorten_005: ", len(np.where(np.array(path_classes_shorten_005) == False)[0])
    # paths_shorten_01 = shorten_paths(paths, 0.1)
    # path_classes_shorten_01 = [False if gt[path[0][0], path[0][1], path[0][2]] ==
    #                                     gt[path[-1][0], path[-1][1], path[-1][2]] else True
    #                            for path in paths_shorten_01]
    # print "True_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == True)[0])
    #
    # print "False_shorten_01: ", len(np.where(np.array(path_classes_shorten_01) == False)[0])




    # probs_julian=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/probs_julian.npy")
    #
    #
    #
    #
    # paths_to_objs_test=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/paths_to_objs_test.npy")
    # paths_test=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/paths_test.npy")
    # probs=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/probs.npy")
    # seg_test=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/seg_test.npy")
    # paths_50=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/paths_50.npy")
    # seg,dt,gt=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/seg_dt_gt.npy")
    # paths_81=np.load("/mnt/localdata1/amatskev/debugging/features/test_for_paths/paths_81.npy")
    #
    # seg_81=np.zeros(seg_test.shape)
    # seg_81[seg_test==81]=1
    # gt_labels_for_81=[]
    # gt_81=[gt[path.transpose()[0],path.transpose()[1],path.transpose()[2]] for path in paths_81]
    # gt_50=[gt[path.transpose()[0],path.transpose()[1],path.transpose()[2]] for path in paths_50]
    #
    # gt_true_false_50=[val[0]==val[-1] for val in gt_50]
    #
    # for val in gt_81:
    #         for label in np.unique(val):
    #             gt_labels_for_81.append(label)
    # # _,skel=img_to_skel(seg_50)
    # gt_81_labels=np.unique(gt_labels_for_81)
    # # plot_figure_and_path(seg_81,[paths_81[0]],True)
    # gt_labels_image=np.zeros(gt.shape)
    # for label in gt_81_labels:
    #     gt_labels_image[gt==label]=label
    # # gt_81=[gt[path.transpose()[0],path.transpose()[1],path.transpose()[2]] for path in paths_81]
    # plot_figure_and_path(gt_labels_image,[paths_81[0]],False)

    # parallel_wrapper(seg,dt,gt,[10.,1.,1.],50,1,[])

    # hi=5
    # b=bl(hi)
    # print hi
    #
    # f = h5py.File("/net/hci-storage02/userfolders/amatskev/"
    #               "neuraldata/results/splA_z1/result.h5", mode='r')
    #
    #
    # feat_paths=read("/mnt/localdata1/amatskev/debugging/features/"
    #                 "test_for_right_order/feat_paths.pkl")
    # print (out_single==out_parallel).all()

    # from time import time
    # time1=time()
    #
    # pixel_values_all=read("/mnt/ssd/amatskev/debugging/features/test_for_right_order/pixel_values_all.pkl")
    # print "loading took ",time()-time1," secs"
    # time1=time()
    # out1 = np.array(Parallel(n_jobs=32) \
    #                    (delayed(python_region_features_extractor_2_mc)(single_vals)
    #                     for single_vals in pixel_values_all))
    # print "first took ",time()-time1," secs"
    # np.save("/mnt/ssd/amatskev/debugging/features/test_for_right_order/out_parallel.npy",out1)
    # # write("/mnt/ssd/amatskev/debugging/features/test_for_right_order/out_parallel.pkl",out1)
    # print "written 1"
    # time1=time()
    # out2 = np.array([python_region_features_extractor_2_mc(single_vals)
    #                 for single_vals in pixel_values_all])
    # print "second took ",time()-time1," secs"
    # np.save("/mnt/ssd/amatskev/debugging/features/test_for_right_order/out_single.npy",out2)
    #
    # # write("/mnt/ssd/amatskev/debugging/features/test_for_right_order/out_single.pkl",out1)
    #
    # print "written 2"
    #
    #
    # print "written npy"
    # assert 1==2
    # seg=np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_seg.npy")
    # volume_dt=np.load(
    #     "/mnt/localdata01/amatskev/debugging/border_term_points/"
    #     "first_volume_dt.npy")
    # # seg = np.array(f["z/1/data"])
    #
    # threshhold_boundary = 20
    # volume_where_threshhold = np.where(volume_dt > threshhold_boundary)
    # volume_dt_boundaries = np.s_[min(volume_where_threshhold[0]):max(volume_where_threshhold[0]),
    #                        min(volume_where_threshhold[1]):max(volume_where_threshhold[1]),
    #                        min(volume_where_threshhold[2]):max(volume_where_threshhold[2])]
    # where=np.unique(seg)

    # for label in [329]:
    #     img = np.zeros(seg.shape)
    #     img[seg==label]=1
    #     # unique=label(img)
    #
    #     # if len(np.unique(img))!=2:
    #     #     print "label: ", label, " of ", len(where)-1
    #     img = [img]
    #     volumina_viewer.volumina_n_layer(
    #         img,
    #         "bla"
    #     )
    #     [0, 24, 54, 90, 105, 118, 124, 144, 187, 202, 209,
    #      239, 250, 254, 283, 285, 289, 291, 293, 294, 296, 301, 305, 307, 308,
    #      311, 318, 327, 329]

    # img=np.zeros(seg.shape)
    # boundary_img=np.ones(seg.shape)
    # boundary_img[boundaries]=0
    # img[seg==2]=1
    # img[boundaries]=0
    # img=[img]
    # boundary_img=[boundary_img]
    # volumina_viewer.volumina_n_layer(
    #     boundary_img,
    #     "bla"
    # )
    #
    #
    #
    # view(
    #     [
    #         "/net/hci-storage02/userfolders/amatskev/neuraldata/results/splA_z1/result.h5",
    #     ], ['z/1/data'],
    #     types=['uint32']
    # )
    #
    #
    #
    #
    #
    #
    #
    # img,_ = np.load("/export/home/amatskev/Bachelor/"
    #                        "data/graph_pruning/debugging/"
    #                        "spooky/skel_seg.npy")
    #
    # skel_img,skel = img_to_skel(img)
    #
    #
    # plot_figure_and_path(img,skel)