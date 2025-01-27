"""
Provides tools for processing, pruning, and analyzing filament skeleton structures in multidimensional data.
"""

__author__ = 'mcychen'

import numpy as np
from astropy.io import fits
import time
from datetime import timedelta
from skimage.morphology import skeletonize, label, remove_small_objects

from . import pruning
from . import _filfinder_length as ff_length

########################################################################################################################

class Skeleton(object):
    """
    Represents a skeletonized structure with tools for pruning, analyzing, and processing filament data.
    """

    def __init__(self, skeleton_raw, header=None, img=None, min_size=3):
        """
        Initializes the Skeleton object with a raw skeleton, header, and optional reference image.

        Parameters
        ----------
        skeleton_raw : ndarray or str
            One-pixel-wide filament skeleton as an array or the file path to a FITS file containing the skeleton data.
        header : fits.Header, optional
            The FITS header corresponding to the provided skeleton. If `skeleton_raw` is a file path and a header
            exists,
            it will be loaded automatically.
        img : ndarray, optional
            A reference image or cube used for intensity-weighted pruning. Defaults to None.
        min_size : int, optional
            The minimum size (in pixels) of skeleton components allowed in the final output. Defaults to 3.

        Notes
        -----
        The `skeletonize` function ensures the input skeleton is one-pixel wide. Small objects below the size threshold
        specified by `min_size` are removed to ensure compatibility with pruning operations.
        """
        if isinstance(skeleton_raw, str):
            skeleton_raw, hdr = fits.getdata(skeleton_raw, header=True)
            if hdr is None:
                header = hdr

        self.ndim = skeleton_raw.ndim
        # skeletonize and removing object that's less than 1 pixel in size to ensuer it's compitable with pruning
        self.skeleton_raw = skeletonize(skeleton_raw, method='lee').astype(bool)
        self.skeleton_raw = remove_small_objects(self.skeleton_raw, min_size=min_size, connectivity=self.ndim)
        self.skeleton_full = self.skeleton_raw.copy()
        self.header = header
        if img is not None:
            self.intensity_img = img

    def prune_from_scratch(self, use_skylength=True, remove_bad_ppv=True):
        """
        Executes the full pruning process on the skeleton, including classification,
        branch property initialization, graph pruning, and length determination.

        Parameters
        ----------
        use_skylength : bool, optional
            Whether to use sky-projected length for pruning and branch properties. Defaults to True.
        remove_bad_ppv : bool, optional
            If True, removes branches with unphysical aspect ratios in position-position-velocity (PPV) space.
            This is only applicable for 3D skeletons. Defaults to True.

        Notes
        -----
        This method performs the following sequential steps:
        1. Removes bad branches if `remove_bad_ppv` is True (3D only).
        2. Classifies the skeleton structure into branches and intersections.
        3. Initializes branch properties based on intensity or geometry.
        4. Prepares the graph representation of the skeleton.
        5. Finds the longest path within the skeleton graph.
        6. Prunes the graph based on specified criteria.
        7. Computes the main length of the pruned skeleton.

        Timing information for the pruning process is printed upon completion.
        """
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
        """
        Saves the pruned skeleton data to a FITS file.

        Parameters
        ----------
        outpath : str
            The file path where the pruned skeleton will be saved.
        overwrite : bool, optional
            If True, overwrites the existing file at `outpath`. Defaults to True.

        Notes
        -----
        If the `length_thresh` attribute has not been set prior to calling this method,
        it will automatically compute the main length of the pruned skeleton before saving.
        The pruned skeleton data is saved in a labeled format, which can be extended for
        future analysis or visualization purposes.
        """
        if not hasattr(self, 'length_thresh'):
            self.main_length()
        # labeling the skeletons can be an useful feature in the future
        data = self.spines
        fits.writeto(outpath, data, self.header, overwrite=overwrite)

    def remove_bad_branches(self, v2pp_ratio=1.0):
        """
        Removes branches with unphysical aspect ratios in position-position-velocity (PPV) space.

        Parameters
        ----------
        v2pp_ratio : float, optional
            The maximum allowable ratio of velocity to positional gradients. Branches exceeding
            this ratio are considered "bad" and are removed. Defaults to 1.0.

        Notes
        -----
        This method identifies and removes branches that have unphysical velocity gradients
        in the PPV space. The skeleton is re-skeletonized after removal to ensure consistency.

        Timing information for the branch removal process is printed upon completion.
        """
        # remove branches that has "unphysical" aspect-ratio in the ppv space (i.e., unphysical velocity gradient)
        start = time.time()
        print("removing bad ppv branches...")
        intPts = pruning.branchedPoints(self.skeleton_full, endpt = None)
        branches = np.logical_and(self.skeleton_full, ~intPts)
        labBodyPtAry, num_lab = label(branches, connectivity=2, return_num=True)

        self.skeleton_full = pruning.remove_bad_ppv_branches(labBodyPtAry, num_lab, refStructure=self.skeleton_full,
                                                             max_pp_length=30.0, v2pp_ratio=v2pp_ratio, method="full")
        self.skeleton_full = skeletonize(self.skeleton_full).astype(bool)
        end = time.time()
        delta_time = int(end - start)
        delta_time = timedelta(seconds=delta_time)
        print("time took to remove the bad branches: {0}".format(delta_time))

    def classify_structure(self):
        """
        Classify the skeleton into labeled branches, intersection points, and endpoints.

        This method processes the skeleton to identify its components and assigns unique
        labels to individual branches while detecting intersections and endpoints.
        """
        self.labelisofil, self.interpts, self.ends = pruning.classify_structure(self.skeleton_full)

    def init_branch_properties(self, img=None, use_skylength=True):
        """
        Initialize the properties of branches in the skeleton.

        Parameters
        ----------
        img : ndarray, optional
            A reference image used to calculate intensity-based properties. Defaults to None.
        use_skylength : bool, optional
            Whether to use sky-projected length for branch calculations. Defaults to True.
        """
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
        """
        Prepare the graph representation of the skeleton, defining nodes and edges.

        This method generates the graph structure based on labeled branches, intersection points,
        and endpoints in the skeleton.
        """
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
        """
        Identify the longest path within the graph representation of the skeleton.

        This method calculates the maximum-length path through the skeleton's graph.
        """
        if not hasattr(self, 'edge_list'):
            self.pre_graph()
        self.max_path, self.extremum, self.graphs = ff_length.longest_path(edge_list=self.edge_list, nodes=self.nodes) #ff.

    def prune_graph(self, length_thresh=0.5):
        """
        Prune the skeleton graph based on a specified length threshold.

        Parameters
        ----------
        length_thresh : float, optional
            The minimum branch length to retain in the pruned graph. Defaults to 0.5.
        """
        self.length_thresh = length_thresh

        if not hasattr(self, 'graphs'):
            self.longest_path()

        # note: the current implementation only works with prune_criteria='length'
        self.labelisofil, self.edge_list, self.nodes, self.branch_properties = ff_length.prune_graph(self.graphs, self.nodes, self.edge_list, self.max_path, self.labelisofil,
                                  self.branch_properties, self.loop_edges, prune_criteria='length',
                                  length_thresh=self.length_thresh) #ff.

    def main_length(self):
        """
        Compute the main lengths of the pruned skeleton and generate labeled spines.

        This method calculates the overall length of the skeleton and identifies the main branches.
        """
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