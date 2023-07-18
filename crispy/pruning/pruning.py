__author__ = 'mcychen'

from scipy import ndimage
from skimage import morphology
import numpy as np
from astropy.io import fits
from astropy.utils.console import ProgressBar
import copy
import string

from .fil_finder import utilities as ff_util

########################################################################################################################

def branchedPoints(skel, endpt=None):
    # note: it's more efficient to find the body-points than to find branch-points
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
    # for 2D skeleton
    if np.size(skel.shape) == 2:
        base_block = np.zeros((3, 3))
        base_block[1, 1] = 1

    # for 3D skeleton
    elif np.size(skel.shape) == 3:
        base_block = np.zeros((3, 3, 3))
        base_block[1, 1, 1] = 1

    else:
        print("[ERROR] the skeleton is neither 2 or 3 dimensions in size!")
        return None

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
    # for 2D skeleton
    if np.size(skel.shape) == 2:
        base_block = np.zeros((3, 3))
        base_block[1, 1] = 1
        cent_idx = (1, 1)

    # for 3D skeleton
    elif np.size(skel.shape) == 3:
        base_block = np.zeros((3, 3, 3))
        base_block[1, 1, 1] = 1
        cent_idx = (1, 1, 1)

    else:
        print("[ERROR] the skeleton is neither 2 or 3 dimensions in size!")
        return None

    epList = []

    # iterate over all permutation of endpoints
    # Note: this does not account for "end points" that are only a pixel long
    i = 0
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
    '''
    :param segment: <ndarray>
        a skeleton segment that does not contain intersections (i.e. branches)
    :return:
        a list of coordinate indices ordered by their relative position along the segment, start from the end closest
        to the origin
    '''
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


def init_lengths(labelisofil, array_offsets=None, img=None, use_skylength=True):
    '''
    3D version of the same function borrowed from Koch's FilFinder, with some modifications and hacks
    This is a wrapper on fil_length for running on the branches of the
    skeletons.
    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    filbranches : list
        Contains the number of branches in each skeleton.
    array_offsets : List
        The indices of where each filament array fits in the
        original image.
    img : numpy.ndarray
        Original image.
    Returns
    -------
    branch_properties: dict
        Contains the lengths and intensities of the branches.
        Keys are *length* and *intensity*.
    '''

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
        # print("...calculating length for segment {0}/{1}...".format(n + 1, num))

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
                # For use in longest path algorithm, will be set to zero for final analysis
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


def init_branch_properties(labelisofil, img=None, use_skylength=True):
    # a quick hack to integrate the fil_finder updates (version 2.0.dev887) into the code
    # note: fil_finder no longer uses init_branch_properties function
    return init_lengths(labelisofil, img=img, use_skylength=use_skylength)


def segment_len(wlk_idx, remove_axis=None):
    '''
    Take the ordered indices of a skeleton segment, with no intersections, and calculate its length
    Note: the distance being calculated may be short by ~1 pixel in length, as the distance is calculated from the
    centre of each pixel
    :param wlk_idx:
    :param remove_axis: <int>
        if not None, the axis to mask out when calculate the segment length (useful if the only on-sky length, rather
        than the ppv length, is interested
    :return:
    '''
    crd_diff = np.diff(np.swapaxes(wlk_idx, 0, 1) * 1.0)
    if remove_axis is not None:
        crd_diff[remove_axis, :] = 0.0

    dst = np.sum(np.sqrt(np.sum(crd_diff ** 2, axis=0)))
    return dst


def coord_list(skel_list):
    '''
    Return the coordinate of any none-zero pixels in a list
    :param skel_list:
    :return:
    crd_list <>
    '''
    crd_list = []
    for skel in skel_list:
        crd = np.argwhere(skel != 0)
        crd = list(map(tuple, crd))
        crd_list.append(crd)
    return crd_list


def remove_bad_ppv_branches(labBodyPtAry, num_lab, refStructure=None, max_pp_length=9.0, v2pp_ratio=1.5, method="full"):
    '''
    Take a 3D labelled body-point array (i.e. with the intersections removed) and remove branches that are likely
    unphysical. Specifically, branches that are on the order of a beam width or on the plane of the sky
    (i.e., pp-projection) and long aspect ratio

    Warning: this method may not be very effective if the said branch segment is actually part of a longer, contineous,
    intersection free segment.

    :param labBodyPtAry: <array>
        Array of the skeletons with the body-points removed and branches labelled.
    :param num_lab: <int>
        number of branches in the labelled array
    :param refStructure: <array>
        Array of the reference structure (i.e., full skeleton)
    :param v2pp_ratio: <array>
        The minimum aspect ratio between the project branch length in v-space and in pp-space
    :return: <array>
        The reference structure with the bad ppv branches removed
    '''

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
                        #print(("v size: {0}, pp size: {1}, ratio: {2}".format(vlength, skylength, ratio)))
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
                    #print(("v size: {0}, pp size: {1}, ratio: {2}".format(v_length, size_pp, ratio)))
                    refStructure[branch] = 0
    else:
        print(("[ERROR]: the entered method {0} is not recongnized.".format(method)))
        return None

    return morphology.remove_small_objects(refStructure, min_size=2, connectivity=3)


def pre_graph_3D(labelisofil, branch_properties, interpts, ends, w=0.0):
    '''
    This function converts the skeletons into a graph object compatible with
    networkx. The graphs have nodes corresponding to end and
    intersection points and edges defining the connectivity as the branches
    with the weights set to the branch length.
    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    branch_properties : dict
        Contains the lengths and intensities of all branches.
    interpts : list
        Contains the pixels which belong to each intersection.
    ends : list
        Contains the end pixels for each skeleton.
    Returns
    -------
    end_nodes : list
        Contains the nodes corresponding to end points.
    inter_nodes : list
        Contains the nodes corresponding to intersection points.
    edge_list : list
        Contains the connectivity information for the graphs.
    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.
    '''

    num = len(labelisofil)

    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []
    loop_edges = []

    def path_weighting(idx, length, intensity, w=0.5):
        '''
        Relative weighting for the shortest path algorithm using the branch lengths and the average intensity
        along the branch.
        MC note: this weighting scheme may be potentially flawed
        '''
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
        # They are labeled alphabetically (if len(interpts[n])>26, subsequent labels are AA,AB,...).
        # The branch labels attached to each intersection are included for future use.
        for intersec in interpts[n]:
            # print(intersec)
            # print("branch point size: {0}".format(len(intersec[intersec!=0])))
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
        inter_nodes.append(list(zip(ff_util.product_gen(string.ascii_uppercase),
                                    inter_nodes_temp)))
        for alpha, node in zip(ff_util.product_gen(string.ascii_uppercase),
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


def pre_graph_3D_old(labelisofil, branch_properties, interpts, ends, w=0.5):
    '''
    The 3D version of Eric Koch's pre_graph function in FilFinder
    This function converts the skeletons into a graph object compatible with
    networkx. The graphs have nodes corresponding to end and
    intersection points and edges defining the connectivity as the branches
    with the weights set to the branch length.
    Parameters
    ----------
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    branch_properties : dict
        Contains the lengths and intensities of all branches.
    interpts : list
        Contains the pixels which belong to each intersection.
    ends : list
        Contains the end pixels for each skeleton.
    Returns
    -------
    end_nodes : list
        Contains the nodes corresponding to end points.
    inter_nodes : list
        Contains the nodes corresponding to intersection points.
    edge_list : list
        Contains the connectivity information for the graphs.
    nodes : list
        A complete list of all of the nodes. The other nodes lists have
        been separated as they are labeled differently.
    '''

    num = len(labelisofil)

    end_nodes = []
    inter_nodes = []
    nodes = []
    edge_list = []
    loop_edges = []

    def path_weighting(idx, length, intensity, w=0.5):
        print(("intensity weighting fraction: {0}".format(w)))
        '''
        Relative weighting for the shortest path algorithm using the branch
        lengths and the average intensity along the branch.
        '''
        print(("index is: {0}".format(idx)))
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
        # They are labeled alphabetically (if len(interpts[n])>26, subsequent labels are AA,AB,...).
        # The branch labels attached to each intersection are included for future use.
        for intersec in interpts[n]:
            uniqs = []
            for i in intersec:  # Intersections can contain multiple pixels

                z, y, x = i
                int_arr = labelisofil[n][z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]
                int_arr[1, 1, 1] = 0

                for x in np.unique(int_arr[np.nonzero(int_arr)]):
                    uniqs.append((x,
                                  path_weighting(x - 1, lengths[n],
                                                 branch_intensity[n],
                                                 w),
                                  lengths[n][x - 1],
                                  branch_intensity[n][x - 1]))
            # Intersections with multiple pixels can give the same branches.
            # Get rid of duplicates
            uniqs = list(set(uniqs))
            inter_nodes_temp.append(uniqs)

        # Add the intersection labels. Also append those to nodes
        inter_nodes.append(list(zip(ff_util.product_gen(string.ascii_uppercase),
                                    inter_nodes_temp)))
        for alpha, node in zip(ff_util.product_gen(string.ascii_uppercase),
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
                        multi = [match[l][1] for l in range(len(match))]
                        keep = multi.index(min(multi))
                        new_edge = (inters[0], inters_2[0], match[keep])

                        # Keep the other edges information in another list (for loops)
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
                        if not (new_edge[1], new_edge[0], new_edge[2]) in edge_list_temp \
                                and new_edge not in edge_list_temp:
                            edge_list_temp.append(new_edge)

        # Remove duplicated edges between intersections

        edge_list.append(edge_list_temp)
        loop_edges.append(loops_temp)

    return edge_list, nodes, loop_edges


def get_furthest_nodes(ends, return_dist=False):
    '''
    Take a list of end pixels for each skeleton, and return a list of coordinates of endpoints that are furtherest from
     each other : list

    :param ends: <List>
        Contains the end pixels for each skeleton.
    :param return_dist: <Boolean>
        Indicate whether or not to return a list of maximum distances
    :return maxdst_ends_list <List>:
        A list of coordinates corresponding to the endpoint pairs in each skeleton that are furthest separated from
        each others
    :return maxdst_ends_list <List, optional>:
        A list of Ecludian distances between the endpoint pairs found in maxdst_ends_list


    '''
    return None


def main_length_3D(max_path, edge_list, labelisofil, interpts, branch_lengths,
                   img_scale, verbose=False, save_png=False, save_name=None):
    '''
    3D version of the main_length() function from fil_finder/length.py, with the verbose, save_png, and save_name
    disabled given we are dealing with a 3D structure

    Wraps previous functionality together for all of the skeletons in the
    image. To find the overall length for each skeleton, intersections are
    added back in, and any extraneous pixels they bring with them are deleted.
    Parameters
    ----------
    max_path : list
        Contains the paths corresponding to the longest lengths for
        each skeleton.
    edge_list : list
        Contains the connectivity information for the graphs.
    labelisofil : list
        Contains individual arrays for each skeleton where the
        branches are labeled and the intersections have been removed.
    interpts : list
        Contains the pixels which belong to each intersection.
    branch_lengths : list
        Lengths of individual branches in each skeleton.
    img_scale : float
        Conversion from pixel to physical units.

    Returns
    -------
    main_lengths : list
        Lengths of the skeletons.
    longpath_arrays : list
        Arrays of the longest paths in the skeletons.
    '''

    def eight_con():
        '''
        3D version of eight_con() from fil_finder/utilities.py
        '''
        return np.ones((3, 3, 3))

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
            #print("spine {}".format(num))
            #print("lengths: {}".format(lengths))
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
                    while list(zip(ff_util.product_gen(string.ascii_uppercase),
                                   [1] * k))[-1][0] != label:
                        k += 1
                    intersec_pts.extend(inters[k - 1])
                    skeleton[list(zip(*inters[k - 1]))] = 2
            # Remove unnecessary pixels

            count = 0
            while True:
                for pt in intersec_pts:
                    # If we have already eliminated the point, continue
                    if skeleton[pt] == 0:
                        continue
                    skeleton[pt] = 0
                    lab_try, n = ndimage.label(skeleton, eight_con())
                    if n > 1:
                        skeleton[pt] = 1
                    else:
                        count += 1
                if count == 0:
                    break
                count = 0

            # main_lengths.append(skeleton_length(skeleton) * img_scale)
            # This is a place holding hack at the moment and main_lengths does not actually hold the lengths of the longest paths
            main_lengths.append(1.0 * img_scale)

        longpath_cube[skeleton.astype(bool)] = True
        print('spine {} added'.format(num))

    return main_lengths, longpath_cube.astype(int)


def save_labskel2fits(labelisofil, outpath, header):
    '''
    Saving a list of labelled skeletons into a single fits file
    :param labelisofil:
    :param outpath:
    :param header:
    :return:
    '''

    labelisofil = np.array(labelisofil)
    data = np.nansum(labelisofil, axis=0)

    fits.writeto(outpath, data, header, overwrite=True)


def classify_structure(skeleton):
    '''
    :param skeleton:
    :return:
    '''

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
    SkLb, SkNum = morphology.label(skeleton, connectivity=3, return_num=True)

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
        Lb, Num = morphology.label(BpFk_temp, connectivity=3, return_num=True)
        crdList = labCrdList(labelled=Lb, num=Num, refStructure=BpFk)
        if crdList:
            interpts.append(crdList)

    # remove the intersection points from the original skeleton
    SkFk_bpRemoved = skeleton.copy()
    SkFk_bpRemoved[BpFk != 0] = 0

    # for each skeleton with the intersections removed, label each branch
    labelisofil = []
    for n in range(SkNum):
        skl = SkFk_bpRemoved.copy()
        skl[SkLb != n + 1] = 0
        labelisofil.append(morphology.label(skl, connectivity=3, return_num=False))

    return labelisofil, interpts, ends






