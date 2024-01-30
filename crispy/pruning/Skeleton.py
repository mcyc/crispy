__author__ = 'mcychen'

import numpy as np
from astropy.io import fits
import time
from datetime import timedelta
from skimage.morphology import skeletonize, skeletonize_3d, label, remove_small_objects
from importlib import reload

from . import pruning
from .fil_finder import length as ff_length

reload(pruning)
reload(ff_length)

########################################################################################################################

class Skeleton(object):

    def __init__(self, skeleton_raw, header=None, img = None, min_size=3):
        '''
        :param skeleton_raw:
            [ndarray] One pixel wide filament skeleton
        :param header:
            The fits header corresponding ot the provided skeleton
        :param img:
            [ndarray] A reference image/cube for intensity weighted pruning
        :param min_size:
            [int] the minize size of skeleton allowed in the final output
        :return:
        '''
        # skeletonize_3d is used to ensure the input skeleton is indeed one pixel wide

        if isinstance(skeleton_raw, str):
            skeleton_raw, hdr = fits.getdata(skeleton_raw, header=True)
            if hdr is None:
                header = hdr

        # skeletonize and removing object that's less than 1 pixel in size to ensuer it's compitable with pruning

        self.ndim = skeleton_raw.ndim

        if self.ndim ==2:
            self.skeleton_raw = skeletonize(skeleton_raw).astype(bool)
        else:
            self.skeleton_raw = skeletonize_3d(skeleton_raw).astype(bool)

        self.skeleton_raw = remove_small_objects(self.skeleton_raw, min_size=min_size, connectivity=self.ndim)
        self.skeleton_full = self.skeleton_raw.copy()
        self.header = header
        if img is not None:
            self.intensity_img = img


    def prune_from_scratch(self, use_skylength=True, remove_bad_ppv=True):
        # the run everything in one go
        start = time.time()

        if self.ndim ==3 and remove_bad_ppv:
            self.remove_bad_branches()
        self.classify_structure()
        self.init_branch_properties(use_skylength=use_skylength)
        self.pre_graph()
        self.longest_path()
        self.prune_graph()
        self.main_length()

        end = time.time()
        delta_time = end - start
        delta_time = timedelta(seconds=delta_time)
        print("total time used to prune the branches: {0}".format(delta_time))

    def save_pruned_skel(self, outpath, overwrite=True):
        if not hasattr(self, 'length_thresh'):
            self.main_length()
        # labeling the skeletons can be an useful feature in the future
        data = self.spines
        fits.writeto(outpath, data, self.header, overwrite=overwrite)

    def remove_bad_branches(self, v2pp_ratio=1.0):
        # remove branches that has "unphysical" aspect-ratio in the ppv space (i.e., unphysical velocity gradient)
        start = time.time()
        print("removing bad ppv branches...")
        intPts = pruning.branchedPoints(self.skeleton_full, endpt = None)
        branches = np.logical_and(self.skeleton_full, ~intPts)
        labBodyPtAry, num_lab = label(branches, connectivity=2, return_num=True)

        self.skeleton_full = pruning.remove_bad_ppv_branches(labBodyPtAry, num_lab, refStructure=self.skeleton_full,
                                                             max_pp_length=30.0, v2pp_ratio=v2pp_ratio, method="full")
        self.skeleton_full = skeletonize_3d(self.skeleton_full).astype(bool)
        end = time.time()
        delta_time = int(end - start)
        delta_time = timedelta(seconds=delta_time)
        print("time took to remove the bad branches: {0}".format(delta_time))


    def classify_structure(self):
        self.labelisofil, self.interpts, self.ends = pruning.classify_structure(self.skeleton_full)


    def init_branch_properties(self, img = None, use_skylength=True):
        if img is not None:
            self.intensity_img = img

        if not hasattr(self, 'labelisofil'):
            self.classify_structure()

        if not hasattr(self, 'intensity_img'):
            self.branch_properties = pruning.init_branch_properties(self.labelisofil, use_skylength=use_skylength,
                                                                    ndim=self.ndim)
        else:
            self.branch_properties = pruning.init_branch_properties(self.labelisofil, self.intensity_img, use_skylength,
                                                                    ndim=self.ndim)


    def pre_graph(self):
        if not hasattr(self, 'branch_properties'):
            self.init_branch_properties()

        # for 2D (currently not implemented)
        if self.ndim == 2:
            self.edge_list, self.nodes, self.loop_edges =\
                ff_length.pre_graph(self.labelisofil, self.branch_properties, self.interpts, self.ends)

        # for 3D skeleton
        elif self.ndim == 3:
            self.edge_list, self.nodes, self.loop_edges =\
                pruning.pre_graph_3D(self.labelisofil, self.branch_properties, self.interpts, self.ends)
        else:
            print("[ERROR]: the number of dimension for the data is incorrect.")


    def longest_path(self):
        if not hasattr(self, 'edge_list'):
            self.pre_graph()
        self.max_path, self.extremum, self.graphs = ff_length.longest_path(edge_list=self.edge_list, nodes=self.nodes) #ff.


    def prune_graph(self, length_thresh=0.5):
        self.length_thresh = length_thresh

        if not hasattr(self, 'graphs'):
            self.longest_path()

        # note: the current implementation only works with prune_criteria='length'
        self.labelisofil, self.edge_list, self.nodes, self.branch_properties = ff_length.prune_graph(self.graphs, self.nodes, self.edge_list, self.max_path, self.labelisofil,
                                  self.branch_properties, self.loop_edges, prune_criteria='length',
                                  length_thresh=self.length_thresh) #ff.


    def main_length(self):
        if not hasattr(self, 'length_thresh'):
            self.prune_graph()

        # for 2D
        if self.ndim == 2:
            self.main_lengths, self.spines = \
                ff_length.main_length(self.max_path, self.edge_list, self.labelisofil, self.interpts,
                                      self.branch_properties['length'], img_scale=1.0, verbose=False,
                                      save_png=False, save_name=None)
            # collapse all the spines onto a single image
            self.spines = np.sum(self.spines, axis=0)

        # for 3D
        elif self.ndim == 3:
            self.main_lengths, self.spines =\
                pruning.main_length_3D(self.max_path, self.edge_list, self.labelisofil, self.interpts,
                                       self.branch_properties['length'], img_scale=1.0)


