
from mayavi import mlab
import numpy as np
import vigra
import sys
from collections import OrderedDict as Odict




class NeuroSegPlot():



    def __init__(self):
        pass



    def path_in_segmentation(self):
        pass

    @staticmethod
    def start_figure():
        mlab.figure(bgcolor=(0, 0, 0))


    @staticmethod
    def add_path(path, s=None, anisotropy=[1, 1, 1],
                 color=(1, 0, 0),
                 colormap='Spectral',
                 representation='wireframe',
                 line_width=100,
                 vmax=None, vmin=None,
                 custom_lut=None
                 ):
        """
        :param path: np.ndarray of the form:
            path = [[x_0, ..., x_n], [y_0, ..., y_n], [z_0, ..., z_n]]
        :param s: array of len=n containing a scalar for each path position (scalars
            define colors according to colormap
        :param color: Color of the path (when s is set to None)
        :param colormap:
        :param representation
        :param anisotropy:
        :return:
        """

        if len(path) > 3:
            s = path[3]

        # FIXME Why does plotting not work for np.int64?
        if path.dtype == np.int64:
            path = path.astype(np.uint32)

        # Plot the path -----------------------------------------------------------
        if s is not None:

            plot = mlab.plot3d(
                path[0] * anisotropy[0],
                path[1] * anisotropy[1],
                path[2] * anisotropy[2],
                s,
                representation=representation,
                colormap=colormap,
                line_width=line_width,
                vmax=vmax,
                vmin=vmin
            )
            # Set custom look up table as colormap
            if custom_lut is not None:
                plot.module_manager.scalar_lut_manager.lut.table = custom_lut

        else:
            mlab.plot3d(
                path[0] * anisotropy[0],
                path[1] * anisotropy[1],
                path[2] * anisotropy[2],
                representation=representation,
                color=color,
                line_width=line_width
            )

    @staticmethod
    def add_xyz_planes(image, anisotropy=[1, 1, 1], threshold=None, colormap='black-white'):
        # Raw data planes ---------------------------------------------------------
        src = mlab.pipeline.scalar_field(image)

        # Our data is not equally spaced in all directions:
        src.spacing = anisotropy
        src.update_image_data = True

        if threshold is not None:
            src = mlab.pipeline.threshold(src, **threshold)

        cut_plane = mlab.pipeline.scalar_cut_plane(src,
                                                   plane_orientation='x_axes',
                                                   colormap=colormap,
                                                   vmin=None,
                                                   vmax=None)
        cut_plane.implicit_plane.origin = (5, 0, 0)
        # cut_plane.implicit_plane.widget.enabled = False
        #
        cut_plane2 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='y_axes',
                                                    colormap=colormap,
                                                    vmin=None,
                                                    vmax=None)
        cut_plane2.implicit_plane.origin = (0, 2, 0)
        # cut_plane2.implicit_plane.widget.enabled = False

        cut_plane3 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='z_axes',
                                                    colormap=colormap,
                                                    vmin=None,
                                                    vmax=None)
        cut_plane3.implicit_plane.origin = (0, 0, 2)
        # cut_plane3.implicit_plane.widget.enabled = False

    @staticmethod
    def add_iso_surfaces(image, anisotropy=[1, 1, 1], colormap='Spectral', color=None,
                         opacity=0.3, vmin=None, vmax=None):

        for i in np.unique(image):

            if i == 0:
                continue

            # The surface of the selected segmentation object
            t_image = np.array(image)
            t_image[image != i] = 0
            obj = mlab.pipeline.scalar_field(t_image)
            t_image = None

            # Resize possibility no. two
            obj.spacing = anisotropy
            obj.update_image_data = True

            mlab.pipeline.iso_surface(
                obj, contours=[i],
                colormap=colormap,
                color=color,
                opacity=opacity,
                vmin=vmin, vmax=vmax
            )

    @staticmethod
    def show():

        mlab.show()

    @staticmethod
    @mlab.animate(delay=10)
    def anim():
        f = mlab.gcf()
        while 1:
            f.scene.camera.azimuth(0.1)
            f.scene.render()
            yield

    @staticmethod
    def movie_show():

        a = NeuroSegPlot.anim()
        mlab.show(stop=True)


    @staticmethod
    def lut_from_colormap(cmap_dict, color_resolution=256):

        import matplotlib as mpl
        from matplotlib.cm import get_cmap

        colormap = mpl.colors.LinearSegmentedColormap('Name', cmap_dict, color_resolution)
        lut = (get_cmap(colormap)(np.arange(0, color_resolution)) * (color_resolution-1)).astype(np.uint8)

        return lut

    @staticmethod
    def multiple_paths_for_plotting(paths, classes=None, image=None):
        """
        Prepares multiple paths for plotting using add_multiple_paths()
        :param paths: np.array of the shape
            paths = [[[x_0, y_0, z_0], ..., [x_n, y_n, z_n]], ...]
                     |<------------ one path ------------->|
                    |<------------ N paths --------------------->|
        :param classes: np.ndarray with shape denoting a class to each path
            classes = [class_0, ..., class_N]
        :param image: values for 's' are obtained from the image at the respective
            coordinate of each path
        :return: The multiple path dictionary
        """

        def create_sub_paths(paths, classes=None, image=None):

            paths_at_position = Odict()

            for p_id in xrange(0, len(paths)):

                for pos in list(paths[p_id]):

                    if tuple(pos) in paths_at_position.keys():
                        if classes is not None:
                            paths_at_position[tuple(pos)].append((p_id, classes[p_id]))
                        elif image is not None:
                            paths_at_position[tuple(list(pos) + [image[tuple(pos)]])].append((p_id,))
                        else:
                            paths_at_position[tuple(pos)].append((p_id,))
                    else:
                        if classes is not None:
                            paths_at_position[tuple(pos)] = [(p_id, classes[p_id])]
                        elif image is not None:
                            paths_at_position[tuple(list(pos) + [image[tuple(pos)]])] = [(p_id,)]
                        else:
                            paths_at_position[tuple(pos)] = [(p_id,)]

                sub_paths = Odict()
                for key, p_ids in paths_at_position.iteritems():

                    if tuple(p_ids) in sub_paths.keys():
                        sub_paths[tuple(p_ids)].append(key)
                    else:
                        sub_paths[tuple(p_ids)] = [key]

            return sub_paths

        def split_for_consecutive_path_sections(path):

            prev_pos = path[0]
            current_sub_path = 0
            new_paths = [[path[0]]]
            for i in xrange(1, len(path)):
                pos = path[i]
                if abs(sum(np.array(pos) - np.array(prev_pos))) > 1:
                    current_sub_path += 1
                    new_paths.append([])
                new_paths[current_sub_path].append(pos)
                prev_pos = pos

            return new_paths

        # Split the paths into common sections
        # i.e. consider two paths A and B where A and B have a common section.
        # Split the paths into three sub paths: only A, only B, A and B
        sub_paths = create_sub_paths(paths, classes, image)
        # Split the sub paths into consecutive sections
        for key, path in sub_paths.iteritems():
            sub_paths[key] = split_for_consecutive_path_sections(path)

        # Swap axes for correct input format
        for key, spaths in sub_paths.iteritems():
            for i in xrange(0, len(spaths)):
                path = spaths[i]
                path = np.swapaxes(path, 0, 1)
                spaths[i] = path
            sub_paths[key] = spaths

        return sub_paths

    @staticmethod
    def add_multiple_paths(paths, s=None, anisotropy=[1, 1, 1],
                           color=(1, 0, 0),
                           colormap='Spectral',
                           representation='wireframe',
                           line_width=5,
                           vmax=None, vmin=None,
                           custom_lut=None
                           ):

        for key, spaths in paths.iteritems():

            if len(key[0]) > 1:
                mean_class = np.mean([x[1] for x in key])

            for i in xrange(0, len(spaths)):

                path = spaths[i]

                s_in = None
                if s is not None:
                    s_in = s[key][i]

                if len(key[0]) > 1:
                    s_in = [mean_class] * len(path[0])

                NeuroSegPlot.add_path(
                    path,
                    s=s_in,
                    anisotropy=anisotropy,
                    colormap=colormap,
                    color=color,
                    representation=representation,
                    line_width=line_width,
                    vmin=vmin, vmax=vmax,
                    custom_lut=custom_lut
                )

    @staticmethod
    def plot_multiple_paths_with_mean_class(
            paths, classes,
            custom_lut=None,
            colormap='Spectral',
            representation='wireframe',
            line_width=5,
            vmin=0, vmax=1,
            anisotropy=[1, 1, 1],
            method='mean'
    ):

        sub_paths = NeuroSegPlot.multiple_paths_for_plotting(paths, classes)

        for key, spaths in sub_paths.iteritems():

            if method == 'mean':
                mean_class = np.mean([x[1] for x in key])
            else:
                mean_class = method([x[1] for x in key])

            for path in spaths:

                NeuroSegPlot.add_path(
                    path,
                    s=[mean_class] * len(path[0]),
                    anisotropy=anisotropy,
                    colormap=colormap,
                    representation=representation,
                    line_width=line_width,
                    vmin=vmin, vmax=vmax,
                    custom_lut=custom_lut
                )

    def path_in_segmentation_data_bg(self, raw_image, seg_image, path):
        """
        :param raw_image: np.ndarray data used for background
        :param seg_image: np.ndarray image segmentation used for iso-surface plot
        :param path: np.ndarray of the form:
            path = [[x_0, ..., x_n], [y_0, ..., y_n], [z_0, ..., z_n]]
        """

        from scipy.signal import medfilt

        mlab.figure(bgcolor=(0, 0, 0), size=(400, 400))

        # Plot the path -----------------------------------------------------------
        mlab.plot3d(
            path[0] * 10, path[1], path[2],
            representation='wireframe',
            color=(1, 0, 0),
            line_width=5
        )

        # Objects of the segmentation ----------------------------------------------
        print (np.unique(seg_image))

        for i in np.unique(seg_image):

            if i == 0:
                continue

            # seg_image = vigra.filters.gaussianSmoothing(seg_image.astype(np.uint8), sigma=3)
            # seg_image = (seg_image > 0.4).astype(np.uint8)

            # # Resize possibility no. one
            # # Takes longer but like this I can smooth the surface of my segmentation
            # seg_image = lib.resize(seg_image, [10, 1, 1], mode='nearest')
            # seg_image = medfilt(seg_image, kernel_size=5)

            # The surface of the selected segmentation object
            t_seg_image = np.array(seg_image)
            t_seg_image[seg_image != i] = 0
            obj = mlab.pipeline.scalar_field(t_seg_image)
            t_seg_image = None

            # Resize possibility no. two
            obj.spacing = [10, 1, 1]
            obj.update_image_data = True

            mlab.pipeline.iso_surface(
                obj, contours=[i],
                colormap='Spectral',
                opacity=0.3,
                vmin=0, vmax=np.unique(seg_image)[-1]
            )

        # Raw data planes ---------------------------------------------------------
        src = mlab.pipeline.scalar_field(raw_image)
        # Our data is not equally spaced in all directions:
        src.spacing = [10, 1, 1]
        src.update_image_data = True

        cut_plane = mlab.pipeline.scalar_cut_plane(src,
                                                   plane_orientation='x_axes',
                                                   colormap='black-white',
                                                   vmin=None,
                                                   vmax=None)
        cut_plane.implicit_plane.origin = (5, 0, 0)
        # cut_plane.implicit_plane.widget.enabled = False
        #
        cut_plane2 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='y_axes',
                                                    colormap='black-white',
                                                    vmin=None,
                                                    vmax=None)
        cut_plane2.implicit_plane.origin = (0, 2, 0)
        # cut_plane2.implicit_plane.widget.enabled = False

        cut_plane3 = mlab.pipeline.scalar_cut_plane(src,
                                                    plane_orientation='z_axes',
                                                    colormap='black-white',
                                                    vmin=None,
                                                    vmax=None)
        cut_plane3.implicit_plane.origin = (0, 0, 2)
        # cut_plane3.implicit_plane.widget.enabled = False

        # mlab.show()
        self.show()


if __name__ == '__main__':

    # Just some developmental stuff >>

    from hdf5_slim_processing import Hdf5Processing as hp

    # Parameters:
    anisotropy = [10, 1, 1]
    interpolation_mode = 'nearest'
    transparent = True
    opacity = 0.25
    # label = '118'
    label = '456'
    pathid = '1'
    surface_source = 'gt'

    # Specify the files
    raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    raw_file = 'cremi.splC.train.raw_neurons.crop.split_xyz.h5'
    raw_skey = ['z', '1', 'raw']

    seg_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170217_crossvalidation_spl_c_slc_z_train01_pred10/intermed/'
    seg_file = 'cremi.segmlarge.crop.split_z.h5'
    seg_skey = ['z', '1', 'beta_0.5']

    gt_path = raw_path
    gt_file = 'cremi.splC.train.raw_neurons.crop.split_xyz.h5'
    gt_skey = ['z', '1', 'neuron_ids']

    paths_path = '/mnt/localdata01/jhennies/neuraldata/results/cremi_2016/170217_crossvalidation_spl_c_slc_z_train01_pred10_recompute/intermed/'
    paths_file = 'cremi.paths.crop.split_z.h5'
    pathlist_file = 'cremi.pathlist.crop.split_z.pkl'
    paths_skey = ['z_predict1', 'falsepaths', 'z', '1', 'beta_0.5']

    crop = (slice(0, 47, None), slice(698, 1250, None), slice(962, 1250, None))

    # # DEVELOP ----
    # # Specify the files
    # raw_path = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/'
    # raw_file = 'cremi.splA.train.raw_neurons.crop.crop_x10_110_y200_712_z200_712.split_xyz.h5'
    # raw_skey = ['x', '1', 'raw']
    #
    # seg_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # seg_file = 'cremi.splA.train.segmlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # seg_skey = ['x', '1', 'beta_0.5']
    #
    # gt_path = seg_path
    # gt_file = 'cremi.splA.train.gtlarge.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # gt_skey = ['x', '1', 'neuron_ids']
    #
    # paths_path = '/mnt/localdata02/jhennies/neuraldata/results/cremi_2016/170124_neurobioseg_x_cropped_avoid_duplicates_develop/intermed/'
    # paths_file = 'cremi.splA.train.paths.crop.crop_x10_110_y200_712_z200_712.split_x.h5'
    # pathlist_file = 'cremi.splA.train.pathlist.crop.crop_x10_110_y200_712_z200_712.split_x.pkl'
    # paths_skey = ['predict', 'truepaths', 'x', '1', 'beta_0.5']

    # crop = np.s_[0:10, 0:100, 0:100]

    # Load path
    paths = hp(filepath=paths_path + paths_file, nodata=True, skeys=[paths_skey])[paths_skey]
    print paths.keys()
    if label == 'random':
        paths_list = []
        for d, k, v, kl in paths.data_iterator(leaves_only=True):
            paths_list.append(kl)
        import random

        random.seed()
        chosen_path = random.choice(paths_list)
        label = chosen_path[0]
        pathid = chosen_path[1]
    # label = paths.keys()[1]
    print 'Selected label = {}'.format(label)
    print 'Selected pathid = {}'.format(pathid)
    path = np.array(paths[label, pathid])

    import processing_lib as lib

    if surface_source == 'seg':
        # Load segmentation
        seg_image = np.array(hp(filepath=seg_path + seg_file, nodata=True, skeys=[seg_skey])[seg_skey])
        seg_image[seg_image != int(label)] = 0

    elif surface_source == 'gt':
        # Load ground truth
        seg_image = np.array(hp(filepath=gt_path + gt_file, nodata=True, skeys=[gt_skey])[gt_skey])
        gt_labels = np.unique(lib.getvaluesfromcoords(seg_image, path))
        t_seg_image = np.array(seg_image)
        for l in gt_labels:
            t_seg_image[t_seg_image == l] = 0
        seg_image[t_seg_image > 0] = 0
        t_seg_image = None

    else:
        sys.exit()

    if crop is None:
        crop = lib.find_bounding_rect(seg_image, s_=True)

    print 'crop = {}'.format(crop)
    seg_image = lib.crop_bounding_rect(seg_image, crop)

    path = np.swapaxes(path, 0, 1)
    path[0] = path[0] - crop[0].start
    path[1] = path[1] - crop[1].start
    path[2] = path[2] - crop[2].start

    # Load raw image
    raw_image = np.array(hp(filepath=raw_path + raw_file, nodata=True, skeys=[raw_skey])[raw_skey])

    raw_image = lib.crop_bounding_rect(raw_image, crop)


    # DEVELOP

    # pathim = lib.positions2value(np.zeros(seg_image.shape), path, 10)

    # s = range(0, len(path[0]))



    NeuroSegPlot.start_figure()
    # NeuroSegPlot.add_iso_surfaces(
    #     pathim, anisotropy=anisotropy, colormap='Spectral', opacity=1
    # )
    NeuroSegPlot.add_path(path, anisotropy=anisotropy)
    NeuroSegPlot.add_iso_surfaces(seg_image, anisotropy=anisotropy, colormap='Spectral')
    NeuroSegPlot.add_xyz_planes(raw_image, anisotropy=anisotropy)
    NeuroSegPlot.show()
