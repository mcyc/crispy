"""
Utilities for skeleton processing, branch property initialization, and pruning in 2D and 3D structures.
"""

__author__ = 'mcychen'

from scipy import ndimage
from skimage import morphology
import numpy as np
from astropy.utils.console import ProgressBar
import copy
import string
from ._filfinder_length import product_gen
from ._filfinder_length import init_lengths as init_lengths_2D
from .structures import get_base_block, two_con_3D


# ==============================================================================================

def branchedPoints(skel, endpt=None):
    """
    Identify branch points in a skeletonized structure.

    Detects branch points in a 2D or 3D skeleton. Branch points are defined as
    skeleton pixels that are not endpoints or body points. If no endpoints
    are provided, they are computed automatically.

    Parameters
    ----------
    skel : ndarray
        Binary array representing the skeletonized structure. Non-zero values represent
        skeleton points, and zero values represent the background.

    endpt : ndarray, optional
        Precomputed binary array of endpoints in the skeleton. If `None`, the function
        calculates the endpoints internally.

    Returns
    -------
    pt : ndarray
        Binary array with the same shape as `skel`, where branch points are set to `True`.

    Notes
    -----
    - Branch points are identified by excluding body points and endpoints from the skeleton.
    - The function automatically adjusts for 2D or 3D skeletons using appropriate connectivity rules.
    - This function relies on `bodyPoints` to determine body points and `endPoints` to calculate endpoints
      if `endpt` is not provided.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch

    Examples
    --------
    Detect branch points in a 2D skeleton:

    >>> import numpy as np
    >>> from crispy import pruning
    >>> skel = np.zeros((5, 5), dtype=bool)
    >>> skel[2, 1:4] = True
    >>> skel[1, 2] = True
    >>> branches = pruning.branchedPoints(skel)
    >>> print(branches)
    [[False False False False False]
     [False False  True False False]
     [False False False False False]
     [False False False False False]
     [False False False False False]]
    """
    pt = bodyPoints(skel)
    pt = np.logical_and(skel, np.logical_not(pt))

    # if no end-points are defined, find the end-points first and remove them
    if endpt is None:
        print("calculating end points...")
        endpt = endPoints(skel)

    pt = np.logical_and(pt, np.logical_not(endpt))

    return pt


# identify body points (points with only two neighbour by 3-connectivity)
def bodyPoints(skel):
    """
    Identify body points in a skeletonized structure.

    Detects body points in a 2D or 3D skeleton. Body points are defined as
    pixels with exactly two neighbors in the skeleton, based on 2-connectivity in ND.

    Parameters
    ----------
    skel : ndarray
        Binary array representing the skeletonized structure. Non-zero values represent
        skeleton points, and zero values represent the background.

    Returns
    -------
    pt : ndarray
        Binary array with the same shape as `skel`, where body points are set to `True`.

    Notes
    -----
    - Body points are computed by identifying skeleton points with exactly two neighbors
      under the specified connectivity rules.
    - This function supports both 2D and 3D skeletons, adjusting connectivity checks
      based on the dimensionality.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    Examples
    --------
    Detect body points in a 2D skeleton:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> skel = np.zeros((5, 5), dtype=bool)
    >>> skel[2, 1:4] = True
    >>> body_pts = grid_ridge.bodyPoints(skel)
    >>> print(body_pts)
    [[False False False False False]
     [False False False False False]
     [False  True  True  True False]
     [False False False False False]
     [False False False False False]]
    """
    base_block = get_base_block(skel.ndim, return_cent_idx=False)
    ptList = []

    # iterate over the "top" layer
    i = 0
    for idx_top, v_top in np.ndenumerate(base_block[0]):
        # for each cell in the "top" layer, iterate over teh "bottom" layer
        for idx_bottom, v_bottom in np.ndenumerate(base_block[2]):
            str_block = base_block.copy()
            # populate the two neighbours
            str_block[(0,) + idx_top] = 1
            str_block[(2,) + idx_bottom] = 1
            ptList.append(str_block)

    # now add the permutations that are rotationally symmetric to the above list
    ptListOri = copy.deepcopy(ptList)
    for i in ptListOri:
        ptList.append(np.swapaxes(i, 0, 1))

    # again, for a 3D skeleton
    if np.size(skel.shape) == 3:
        for i in ptListOri:
            ptList.append(np.swapaxes(i, 0, 2))

    # remove the redundant elements
    ptList = np.unique(np.array(ptList), axis=0)

    pt = np.full(np.shape(skel), False, dtype=bool)

    for pt_i in ptList:
        pt = pt + ndimage.binary_hit_or_miss(skel, structure1=pt_i)

    return pt


# identify end points (points with only two neighbour by 3-connectivity)
# (only works if the skeleton is on 1-pixel in width by 3-connectivity and not 1-connectivity)
def endPoints(skel):
    """
    Identify endpoints in a skeletonized structure.

    Detects endpoints in a 2D or 3D skeleton. Endpoints are defined as
    pixels in the skeleton with exactly one neighbor, based on 2-connectivity in ND.

    Parameters
    ----------
    skel : ndarray
        Binary array representing the skeletonized structure. Non-zero values represent
        skeleton points, and zero values represent the background.

    Returns
    -------
    ep : ndarray
        Binary array with the same shape as `skel`, where endpoints are set to `True`.

    Notes
    -----
    - Endpoints are determined using hit-or-miss morphology with connectivity rules that
      detect pixels with only one neighbor.
    - This function supports both 2D and 3D skeletons, adjusting connectivity checks
      based on the dimensionality.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    Examples
    --------
    Detect endpoints in a 2D skeleton:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> skel = np.zeros((5, 5), dtype=bool)
    >>> skel[2, 1:4] = True
    >>> skel[1, 2] = True
    >>> endpoints = grid_ridge.endPoints(skel)
    >>> print(endpoints)
    [[False False False False False]
     [False False  True False False]
     [False  True False  True False]
     [False False False False False]
     [False False False False False]]
    """
    base_block, cent_idx = get_base_block(skel.ndim, return_cent_idx=True)
    epList = []

    # iterate over all permutation of endpoints
    # Note: this does not account for "end points" that are only a pixel long
    for index, value in np.ndenumerate(base_block):
        if index != cent_idx:
            str_block = base_block.copy()
            str_block[index] = 1
            epList.append(str_block)

    ep = np.full(np.shape(skel), False, dtype=bool)

    for ep_i in epList:
        ep = ep + ndimage.binary_hit_or_miss(skel, structure1=ep_i)

    return ep


def walk_through_segment_3D(segment):
    """
    Traverse a 3D skeleton segment to obtain an ordered list of pixel coordinates.

    This function processes a skeleton segment that does not contain branches or intersections
    and returns an ordered list of pixel coordinates. The traversal starts from the endpoint
    closest to the origin.

    Parameters
    ----------
    segment : ndarray
        A binary 3D array representing the skeleton segment. Non-zero values indicate
        skeleton pixels, and zero values represent the background.

    Returns
    -------
    idx_list : list of tuple
        A list of 3D coordinate tuples ordered by their position along the segment.
        The traversal begins from the endpoint nearest to the origin.

    Notes
    -----
    - This function assumes that the segment has exactly two endpoints and does not touch the
      edges of the array.
    - The traversal may fail if the segment width exceeds one pixel at any point due to
      imperfect skeletonization.
    - Endpoints are detected using the `endPoints` function.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    Raises
    ------
    ValueError
        If the segment has less than two pixels or more than one pixel-wide connectivity.
    """
    # note: this only works if the endpoints does not touch the edge

    segment = copy.copy(segment)

    # in case the segment contains less than 2 pixels
    num_pix = len(segment[segment >= 1])
    if num_pix < 1:
        print("[ERROR]: the total number of pixels in the segment is less than 1!")
        return None
    if num_pix == 1:
        z, y, x = np.argwhere(segment >= 1)[0]
        return [(z, y, x)]

    # find indicies of the endpoints
    ept = endPoints(segment)
    ept_idx = np.argwhere(ept)

    # find the endpoint that is closest to the origin
    if np.sum(ept_idx[0] ** 2) < np.sum(ept_idx[1] ** 2):
        idx = ept_idx[0]
    else:
        idx = ept_idx[1]

    z, y, x = idx
    idx_list = [(z, y, x)]

    block = segment[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
    block[1, 1, 1] = 0

    # "walk through" the pixels in the segment
    while len(block[block > 0]) == 1:
        k, j, i = np.argwhere(block > 0)[0]
        z, y, x = z + k - 1, y + j - 1, x + i - 1
        idx_list.append((z, y, x))
        block = segment[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
        block[1, 1, 1] = 0

    # in case the walk was terminated due to imperfect skeletonization
    if len(block[block > 0]) > 1:
        print("[ERROR]: the skeleton segment is more than a pixel wide by 3-connectivity")
        return None

    return idx_list


def init_lengths_3D(labelisofil, array_offsets=None, img=None, use_skylength=True):
    """
    Compute lengths and intensities for branches in 3D skeletons.

    This function calculates the lengths and average intensities of branches
    in labeled skeletons, accounting for both the full 3D length and the
    sky-projected length if specified.

    Parameters
    ----------
    labelisofil : list of ndarray
        A list of 3D labeled skeleton arrays. Each array contains skeleton branches
        where intersections have been removed, and branches are labeled with unique integers.

    array_offsets : list of ndarray, optional, default=None
        Indices specifying where each skeleton array fits in the original image.
        If None, offsets default to ones.

    img : ndarray, optional, default=None
        The original 3D intensity image. If provided, the average intensity along
        each branch is computed. If not provided, the intensity is assumed to be uniform.

    use_skylength : bool, optional, default=True
        If True, calculates the sky-projected length (ignoring the velocity axis).
        If False, calculates the full 3D length in PPV space.

    Returns
    -------
    branch_properties : dict
        A dictionary containing the following keys:
        - `length`: A list of branch lengths for each skeleton.
        - `intensity`: A list of average intensities for each branch.
        - `pixels`: A list of pixel coordinates for each branch.

    Notes
    -----
    - Branch lengths are calculated using the `walk_through_segment_3D` function.
    - Sky-projected lengths are computed by ignoring the velocity axis during length calculation.
    - The function pads branch arrays to prevent edge-related errors during traversal.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    Raises
    ------
    ValueError
        If the shape of `img` does not match the shape of the skeleton arrays in `labelisofil`.

    """
    print("getting branch_properties")

    num = len(labelisofil)

    if img is None:
        img = np.ones(labelisofil[0].shape)
    else:
        if img.shape != labelisofil[0].shape:
            print("[ERROR]: the shape of the intensity image does that match that of the skeleton")
            return None

    if array_offsets is None:
        array_offsets = np.ones((num, 1, 3), dtype=int)

    # Initialize Lists
    lengths = []
    av_branch_intensity = []
    all_branch_pts = []

    for n in range(num):
        leng = []
        av_intensity = []
        branch_pix = []

        label_copy = copy.copy(labelisofil[n])
        objects = ndimage.find_objects(label_copy)
        for i, obj in enumerate(objects):
            # Scale the branch array to the branch size
            branch_array = np.zeros_like(label_copy[obj])

            # Find the skeleton points and set those to 1
            branch_pts = np.where(label_copy[obj] == i + 1)
            branch_array[branch_pts] = 1

            # pad the edges to so it's compatiable with walk_through_segment_3D (i.e., skeleton does not touch the edge)
            branch_array = np.pad(branch_array, 1, mode='constant', constant_values=0)

            # Now find the length on the branch
            if branch_array.sum() == 1:
                # if single pixel. No need to find length
                # For use in the longest path algorithm, will be set to zero for final analysis
                branch_length = 0.5
            else:
                wlk_idx = walk_through_segment_3D(branch_array)

                if use_skylength:
                    # calculate the sky-projected length
                    branch_length = segment_len(wlk_idx, remove_axis=0)
                else:
                    # calculate the ppv length
                    branch_length = segment_len(wlk_idx)

            leng.append(branch_length)

            # Find the average intensity along each branch
            # Get the offsets from the original array and add on the offset the branch array introduces.
            x_offset = obj[0].start + array_offsets[n][0][0]
            y_offset = obj[1].start + array_offsets[n][0][1]
            z_offset = obj[2].start + array_offsets[n][0][2]
            av_intensity.append(np.nanmean([img[x + x_offset, y + y_offset, z + z_offset]
                                            for x, y, z in zip(*branch_pts)
                                            if np.isfinite(img[x + x_offset, y + y_offset, z + z_offset]) and
                                            not img[x + x_offset, y + y_offset, z + z_offset] < 0.0]))
            branch_pix.append(np.array([(x + x_offset, y + y_offset, z + z_offset)
                                        for x, y, z in zip(*branch_pts)]))

        lengths.append(leng)
        av_branch_intensity.append(av_intensity)
        all_branch_pts.append(branch_pix)

        branch_properties = {"length": lengths,
                             "intensity": av_branch_intensity,
                             "pixels": all_branch_pts}

    return branch_properties


def init_branch_properties(labelisofil, ndim, img=None, use_skylength=True):
    """
    Initialize branch properties for 2D or 3D skeletons.

    Computes lengths and intensities of branches in skeletons, supporting both
    2D and 3D skeleton structures. For 2D skeletons, the lengths are initialized
    using `init_lengths_2D`, while for 3D skeletons, `init_lengths_3D` is used.

    Parameters
    ----------
    labelisofil : list of ndarray
        A list of labeled skeleton arrays, where branches are labeled with unique integers,
        and intersections have been removed.

    ndim : int
        The number of dimensions of the skeletons. Must be either 2 or 3.

    img : ndarray, optional
        Intensity image associated with the skeletons. If provided, the average intensity
        along each branch is calculated. Defaults to None.

    use_skylength : bool, optional
        If True, calculates the sky-projected length for each branch (ignoring the velocity axis).
        If False, calculates the full 3D length in PPV space for 3D skeletons. Defaults to True.

    Returns
    -------
    branch_properties : dict
        A dictionary containing the following keys:
        - `length`: List of branch lengths.
        - `intensity`: List of average intensities for each branch.
        - `pixels`: List of pixel coordinates for each branch.

    Notes
    -----
    - The function dispatches to different implementations depending on the dimensionality (`ndim`).
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch, which as been
      depreciated

    Examples
    --------
    Initialize branch properties for a 2D skeleton:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> skel = np.zeros((5, 5), dtype=bool)
    >>> skel[2, 1:4] = True
    >>> labelisofil = [skel.astype(int)]
    >>> props = grid_ridge.init_branch_properties(labelisofil, ndim=2)
    >>> print(props["length"])
    [[2.0]]
    """
    if ndim == 2:
        num = len(labelisofil)
        array_offsets = np.ones((num, 1, 2), dtype=int)
        if img is None:
            img = np.ones(labelisofil[0].shape)
        return init_lengths_2D(labelisofil, array_offsets, img=img)
    else:
        return init_lengths_3D(labelisofil, img=img, use_skylength=use_skylength)


def segment_len(wlk_idx, remove_axis=None):
    """
    Calculate the length of a skeleton segment.

    This function computes the length of a skeleton segment from an ordered list
    of its pixel coordinates. The length is calculated as the sum of Euclidean distances
    between consecutive pixels. Optionally, a specified axis can be excluded from the
    calculation, which is useful for computing sky-projected lengths.

    Parameters
    ----------
    wlk_idx : list of tuple
        Ordered list of pixel coordinates representing the skeleton segment.

    remove_axis : int, optional, default=None
        Axis to exclude from the length calculation. If None, the full length is computed
        in all dimensions.

    Returns
    -------
    length : float
        The computed length of the skeleton segment.

    Notes
    -----
    - The calculated length may underestimate the actual length by approximately one pixel
      due to measuring from the center of each pixel.
    - Excluding an axis (e.g., velocity in PPV space) computes the sky-projected length.
    """
    crd_diff = np.diff(np.swapaxes(wlk_idx, 0, 1) * 1.0)
    if remove_axis is not None:
        crd_diff[remove_axis, :] = 0.0

    dst = np.sum(np.sqrt(np.sum(crd_diff ** 2, axis=0)))
    return dst


def remove_bad_ppv_branches(labBodyPtAry, num_lab, refStructure=None, max_pp_length=9.0, v2pp_ratio=1.5, method="full"):
    """
    Remove unphysical branches from a labeled 3D skeleton in PPV space.

    This function identifies and removes branches that are likely unphysical, such as
    those with small projected lengths in the position-position (PP) plane or with
    high velocity-to-length ratios. Optionally, a faster approximation method can be
    used for branch filtering.

    Parameters
    ----------
    labBodyPtAry : ndarray
        A 3D array of the skeleton with body points removed, where branches are labeled
        with unique integers.

    num_lab : int
        Number of labeled branches in the array.

    refStructure : ndarray, optional, default=None
        The reference structure array (full skeleton). If None, it is derived from
        `labBodyPtAry`.

    max_pp_length : float, optional, default=9.0
        Maximum allowed length of a branch in the PP plane. Branches shorter than this
        threshold are evaluated for removal.

    v2pp_ratio : float, optional, default=1.5
        Minimum allowed velocity-to-length ratio. Branches exceeding this ratio are
        removed.

    method : {"full", "quick"}, optional, default="full"
        Method for branch filtering:
        - "full": Performs a detailed analysis using branch traversal and length calculations.
        - "quick": Uses a faster, approximate method for filtering based on pixel counts.

    Returns
    -------
    filtered_structure : ndarray
        A binary array of the reference structure with unphysical branches removed.

    Notes
    -----
    - The "full" method uses `walk_through_segment_3D` to accurately calculate branch
      lengths and velocity ratios.
    - The "quick" method approximates branch lengths by counting pixels, which may
      be less accurate for longer branches.
    - Branch removal may fail if `labBodyPtAry` and `refStructure` have mismatched shapes.

    Raises
    ------
    ValueError
        If the shapes of `labBodyPtAry` and `refStructure` do not match.

    """
    if refStructure is None:
        refStructure = labBodyPtAry.copy()
        refStructure[refStructure != 0] = 1
    else:
        refStructure = refStructure.copy()
        if labBodyPtAry.shape != refStructure.shape:
            print("[ERROR]: the shape fo labBodyPtAry and refStructure are not the same!")
            return None

    if method == "full":
        objects = ndimage.find_objects(labBodyPtAry)

        for i, obj in enumerate(ProgressBar(objects)):
            # Scale the branch array to the branch size
            branch = np.zeros_like(labBodyPtAry[obj])
            # Find the skeleton points and set those to 1
            branch_pts = np.where(labBodyPtAry[obj] == i + 1)
            branch[branch_pts] = 1

            # pad the edges to so it's compatiable with walk_through_segment_3D (i.e., skeleton does not touch the edge)
            branch = np.pad(branch, 1, mode='constant', constant_values=0)

            if len(branch == 1) > 1:
                wlk_idx = walk_through_segment_3D(branch)
                skylength = segment_len(wlk_idx, remove_axis=0)
                fulllength = segment_len(wlk_idx)
                vlength = np.sqrt(fulllength ** 2 - skylength ** 2)

                if skylength <= max_pp_length:
                    ratio = vlength / skylength
                    if ratio >= v2pp_ratio:
                        refStructure[labBodyPtAry == i + 1] = 0

    elif method == "quick":
        # a quick way to approximate the ratio between the total length and the on-sky length (the accuracy decreases
        # as the total length increases)
        for n in ProgressBar(list(range(num_lab))):
            # count the number of pixels in an on-sky projection of a branch
            # (i.e., a proxy for the on-sky length of a filament)
            branch = labBodyPtAry == n + 1
            size_pp = (np.logical_or.reduce(branch, axis=0)).sum()
            if size_pp <= max_pp_length:
                v_length = (np.logical_or.reduce(branch, axis=(1, 2))).sum()
                # remove branches that have high velocity-to-length ratio
                ratio = 1.0 * v_length / size_pp
                if ratio >= v2pp_ratio:
                    refStructure[branch] = 0
    else:
        print(("[ERROR]: the entered method {0} is not recongnized.".format(method)))
        return None

    return morphology.remove_small_objects(refStructure, min_size=2, connectivity=3)


def pre_graph_3D(labelisofil, branch_properties, interpts, ends, w=0.0):
    """
    Convert 3D skeletons into graph representations with weighted edges.

    This function generates graph representations of 3D skeletons where nodes represent
    end points and intersection points, and edges represent branches. Edge weights are
    calculated using branch lengths and intensities.

    Parameters
    ----------
    labelisofil : list of ndarray
        A list of 3D labeled skeleton arrays, where branches are labeled with unique
        integers, and intersection points are removed.

    branch_properties : dict
        A dictionary containing properties of the branches, with the following keys:
        - `length`: List of branch lengths.
        - `intensity`: List of average intensities for each branch.

    interpts : list of list of ndarray
        Intersection points for each skeleton, with each entry containing the coordinates
        of pixels belonging to an intersection.

    ends : list of ndarray
        Endpoints for each skeleton.

    w : float, optional, default=0.0
        Weighting factor for branch lengths and intensities in edge weight calculation.
        Must be between 0.0 (length-only weighting) and 1.0 (intensity-only weighting).

    Returns
    -------
    edge_list : list
        List of edges in the graph. Each edge is represented as a tuple:
        `(node_1, node_2, edge_properties)`, where `edge_properties` includes
        branch length and intensity.

    nodes : list
        List of all nodes in the graph, including endpoints and intersection points.

    loop_edges : list
        List of loop edges (edges connecting two intersection nodes through multiple
        branches).

    Notes
    -----
    - Nodes corresponding to intersection points are labeled alphabetically. For graphs
      with more than 26 intersections, labels extend to AA, AB, etc.
    - The `path_weighting` function calculates edge weights using both length and
      intensity, with the relative contribution controlled by `w`.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    Raises
    ------
    ValueError
        If `w` is not between 0.0 and 1.0.

    """
    num = len(labelisofil)

    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []
    loop_edges = []

    def path_weighting(idx, length, intensity, w=0.5):
        """
        Relative weighting for the shortest path algorithm using the branch lengths and the average intensity
        along the branch.
        MC note: this weighting scheme may be potentially flawed
        """
        if w > 1.0 or w < 0.0:
            raise ValueError(
                "Relative weighting w must be between 0.0 and 1.0.")
        return (1 - w) * (length[idx] / np.sum(length)) + \
            w * (intensity[idx] / np.sum(intensity))

    lengths = branch_properties["length"]
    branch_intensity = branch_properties["intensity"]

    for n in range(num):
        inter_nodes_temp = []
        # Create end_nodes, which contains lengths, and nodes, which we will later add in the intersections
        end_nodes.append([(labelisofil[n][i],
                           path_weighting(int(labelisofil[n][i] - 1),
                                          lengths[n],
                                          branch_intensity[n],
                                          w),
                           lengths[n][int(labelisofil[n][i] - 1)],
                           branch_intensity[n][int(labelisofil[n][i] - 1)])
                          for i in ends[n]])
        nodes.append([labelisofil[n][i] for i in ends[n]])

        # Intersection nodes are given by the intersections points of the filament.
        for intersec in interpts[n]:
            uniqs = []
            for i in intersec:  # Intersections can contain multiple pixels

                z, y, x = i

                int_arr = labelisofil[n][z - 1: z + 2, y - 1: y + 2, x - 1: x + 2]
                int_arr = int_arr.astype(int)
                int_arr[1, 1, 1] = 0

                for x in np.unique(int_arr[np.nonzero(int_arr)]):
                    uniqs.append((x,
                                  path_weighting(x - 1,
                                                 lengths[n],
                                                 branch_intensity[n],
                                                 w),
                                  lengths[n][x - 1],
                                  branch_intensity[n][x - 1]))
            # Intersections with multiple pixels can give the same branches.
            # Get rid of duplicates
            uniqs = list(set(uniqs))
            inter_nodes_temp.append(uniqs)

        # Add the intersection labels. Also append those to nodes
        inter_nodes.append(list(zip(product_gen(string.ascii_uppercase),
                                    inter_nodes_temp)))
        for alpha, node in zip(product_gen(string.ascii_uppercase),
                               inter_nodes_temp):
            nodes[n].append(alpha)
        # Edges are created from the information contained in the nodes.
        edge_list_temp = []
        loops_temp = []
        for i, inters in enumerate(inter_nodes[n]):
            end_match = list(set(inters[1]) & set(end_nodes[n]))
            for k in end_match:
                edge_list_temp.append((inters[0], k[0], k))

            for j, inters_2 in enumerate(inter_nodes[n]):
                if i != j:
                    match = list(set(inters[1]) & set(inters_2[1]))
                    new_edge = None
                    if len(match) == 1:
                        new_edge = (inters[0], inters_2[0], match[0])
                    elif len(match) > 1:
                        # Multiple connections (a loop)
                        multi = [match[l][1] for l in range(len(match))]
                        keep = multi.index(min(multi))
                        new_edge = (inters[0], inters_2[0], match[keep])

                        # Keep the other edges information in another list
                        for jj in range(len(multi)):
                            if jj == keep:
                                continue
                            loop_edge = (inters[0], inters_2[0], match[jj])
                            dup_check = loop_edge not in loops_temp and \
                                        (loop_edge[1], loop_edge[0], loop_edge[2]) \
                                        not in loops_temp
                            if dup_check:
                                loops_temp.append(loop_edge)

                    if new_edge is not None:
                        dup_check = (new_edge[1], new_edge[0], new_edge[2]) \
                                    not in edge_list_temp \
                                    and new_edge not in edge_list_temp
                        if dup_check:
                            edge_list_temp.append(new_edge)

        # Remove duplicated edges between intersections

        edge_list.append(edge_list_temp)
        loop_edges.append(loops_temp)

    return edge_list, nodes, loop_edges


def main_length_3D(max_path, edge_list, labelisofil, interpts, branch_lengths, img_scale, verbose=False, save_png=False,
                   save_name=None):
    """
    Compute the main lengths of 3D skeletons and generate longest path arrays.

    This function calculates the overall lengths of skeletons in a 3D image by identifying
    and preserving the longest paths. Intersections are added back to the skeletons, and
    extraneous pixels introduced by intersections are removed.

    Parameters
    ----------
    max_path : list
        List of paths corresponding to the longest lengths for each skeleton.

    edge_list : list
        List of edges representing connectivity information for the skeleton graphs.

    labelisofil : list of ndarray
        List of 3D labeled skeleton arrays. Each array contains skeleton branches
        with unique integer labels and no intersection points.

    interpts : list of list of ndarray
        Intersection points for each skeleton, with each entry containing the
        coordinates of pixels belonging to an intersection.

    branch_lengths : list
        Lengths of individual branches in each skeleton.

    img_scale : float
        Conversion factor from pixel units to physical units.

    verbose : bool, optional, default=False
        If True, prints detailed information about the process (currently disabled for 3D).

    save_png : bool, optional, default=False
        If True, saves 2D visualizations of the skeletons (disabled for 3D).

    save_name : str, optional, default=None
        Name for saving output PNGs (currently unused for 3D).

    Returns
    -------
    main_lengths : list
        List of overall lengths for each skeleton, in physical units.

    longpath_arrays : ndarray
        Binary 3D array representing the longest paths for all skeletons. Non-zero
        values indicate pixels belonging to the longest paths.

    Notes
    -----
    - Intersections are added back to the skeleton, and extraneous pixels are removed
      using a recursive pruning process.
    - The `max_path` input determines the longest path in each skeleton.
    - This function is adapted from the 2D `main_length` function in FilFinder and
      includes modifications for 3D structures.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.
    """
    main_lengths = []
    longpath_cube = np.zeros(labelisofil[0].shape, dtype=bool)

    for num, (path, edges, inters, skel_arr, lengths) in \
            enumerate(zip(max_path, edge_list, interpts, labelisofil,
                          branch_lengths)):

        if not lengths:
            # if the lengths list is empty
            print("empty!")
            print(("lengths: {}".format(lengths)))
            # may want to check not doing anything does not cause problem

        elif len(path) == 1:
            main_lengths.append(lengths[0] * img_scale)
            skeleton = skel_arr  # for viewing purposes when verbose
        else:
            skeleton = np.zeros(skel_arr.shape)

            # Add edges along longest path
            good_edge_list = [(path[i], path[i + 1])
                              for i in range(len(path) - 1)]
            # Find the branches along the longest path.
            for i in good_edge_list:
                for j in edges:
                    if (i[0] == j[0] and i[1] == j[1]) or \
                            (i[0] == j[1] and i[1] == j[0]):
                        label = j[2][0]
                        skeleton[np.where(skel_arr == label)] = 1

            # Add intersections along longest path
            intersec_pts = []
            for label in path:
                try:
                    label = int(label)
                except ValueError:
                    pass
                if not isinstance(label, int):
                    k = 1
                    while list(zip(product_gen(string.ascii_uppercase),
                                   [1] * k))[-1][0] != label:
                        k += 1
                    intersec_pts.extend(inters[k - 1])
                    skeleton[tuple(zip(*inters[k - 1]))] = 2
            # Remove unnecessary pixels

            count = 0
            while True:
                for pt in intersec_pts:
                    # If we have already eliminated the point, continue
                    if skeleton[pt] == 0:
                        continue
                    skeleton[pt] = 0
                    lab_try, n = ndimage.label(skeleton, two_con_3D)
                    if n > 1:
                        skeleton[pt] = 1
                    else:
                        count += 1
                if count == 0:
                    break
                count = 0

            # main_lengths.append(skeleton_length(skeleton) * img_scale)
            # This is a place holding hack at the moment and main_lengths does not actually hold the lengths of the
            # longest paths
            main_lengths.append(1.0 * img_scale)

        longpath_cube[skeleton.astype(bool)] = True

    return main_lengths, longpath_cube.astype(int)


def classify_structure(skeleton):
    """
    Classify the components of a skeleton into labeled branches, intersections, and endpoints.

    This function processes a binary skeleton array, identifies its endpoints and
    intersection points, and removes intersections to separate individual branches.
    It returns labeled arrays for the branches and lists of intersection points and endpoints.

    Parameters
    ----------
    skeleton : ndarray
        Binary array representing the skeletonized structure. Non-zero values indicate
        skeleton points, and zero values represent the background.

    Returns
    -------
    labelisofil : list of ndarray
        List of labeled arrays, where each array corresponds to a skeleton with intersections
        removed, and branches are labeled with unique integers.

    interpts : list of list of tuple
        List of intersection points for each skeleton, with each intersection containing
        the coordinates of pixels belonging to it.

    ends : list of list of tuple
        List of endpoint coordinates for each skeleton.

    Notes
    -----
    - The function uses 8-connectivity for 2D skeletons and maximum connectivity for
      higher dimensions to label individual structures.
    - Endpoints are identified using the `endPoints` function.
    - Intersection points are labeled separately, and their coordinates are stored
      in `interpts`.
    - Branches are labeled after removing intersections from the skeleton.
    - This code is based on the 2D version seen in FilFinder (v1.7.2) by Eric Koch.

    """

    def labCrdList(labelled, num, refStructure):
        '''
        Place the coordinates of individual, labelled structure a list
        '''
        crd_list = []
        for n in range(num):
            crd = np.argwhere(np.logical_and(labelled == n + 1, refStructure != 0))
            crd = list(map(tuple, crd))
            if crd:
                crd_list.append(crd)
        return crd_list

    # label the skeletons
    connectivity = skeleton.ndim  # use maximum connectivity for the dimensions
    SkLb, SkNum = morphology.label(skeleton, connectivity=connectivity, return_num=True)

    # acquire end-points, label them, and place them into a coordinate list
    print("getting end-points")
    EpFk = endPoints(skeleton)
    ends = labCrdList(labelled=SkLb, num=SkNum, refStructure=EpFk)

    # acquire the branched-points (i.e., intersection points), label them, and place them into a coordinate list
    print("getting branched-points")
    BpFk = branchedPoints(skeleton, endpt=EpFk)

    # Not exactly an elegant implementation below, but it'll have to do for now
    interpts = []
    for n in range(SkNum):
        BpFk_temp = BpFk.copy()
        BpFk_temp[SkLb != n + 1] = 0
        Lb, Num = morphology.label(BpFk_temp, connectivity=connectivity, return_num=True)
        crdList = labCrdList(labelled=Lb, num=Num, refStructure=BpFk)
        interpts.append(crdList)

    # remove the intersection points from the original skeleton
    SkFk_bpRemoved = skeleton.copy()
    SkFk_bpRemoved[BpFk != 0] = 0

    # for each skeleton with the intersections removed, label each branch
    labelisofil = []
    for n in range(SkNum):
        skl = SkFk_bpRemoved.copy()
        skl[SkLb != n + 1] = 0
        labelisofil.append(morphology.label(skl, connectivity=connectivity, return_num=False))

    return labelisofil, interpts, ends
