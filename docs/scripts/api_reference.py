# This file is auto-generated. Edit descriptions and structure as needed.

API_REFERENCE = {
    'crispy.grid_ridge': {
        'module': 'crispy.grid_ridge',
        'description': '''Functions for gridding and processing CRISPy results on reference images.''',
        'members': [
            {'name': 'bodyPoints', 'type': 'function', 'description': '''Identify body points in a skeletonized structure.'''},
            {'name': 'branchedPoints', 'type': 'function', 'description': '''Identify branch points in a skeletonized structure.'''},
            {'name': 'clean_grid', 'type': 'function', 'description': '''Process and grid CRISPy coordinates onto a reference image, labeling and cleaning skeleton structures.'''},
            {'name': 'clean_grid_ppv', 'type': 'function', 'description': '''Process and grid CRISPy coordinates in PPV space onto a reference image, labeling and cleaning skeleton structures.'''},
            {'name': 'endPoints', 'type': 'function', 'description': '''Identify endpoints in a skeletonized structure.'''},
            {'name': 'get_2d_length', 'type': 'function', 'description': '''Calculate the sky-projected length of a 3D skeleton.'''},
            {'name': 'grid_skel', 'type': 'function', 'description': '''Map raw CRISPy results onto a reference image grid and save the gridded results.'''},
            {'name': 'grid_skeleton', 'type': 'function', 'description': '''Map CRISPy skeleton coordinates onto a reference image grid.'''},
            {'name': 'label_ridge', 'type': 'function', 'description': '''Label unconnected ridges using DBSCAN clustering.'''},
            {'name': 'make_skeleton', 'type': 'function', 'description': '''Map CRISPy skeleton coordinates onto a reference grid and clean the skeleton.'''},
            {'name': 'read_table', 'type': 'function', 'description': '''Read filament skeleton data from a file.'''},
            {'name': 'uniq_per_pix', 'type': 'function', 'description': '''Reduce a list of ridge coordinates to one unique point per pixel.'''},
            {'name': 'write_skel', 'type': 'function', 'description': '''Undocumented'''}
        ]
    },
    'crispy.image_ridge_find': {
        'module': 'crispy.image_ridge_find',
        'description': '''Functions for density ridge identification in gridded images.''',
        'members': [
            {'name': 'image2data', 'type': 'function', 'description': '''Convert an image into a format compatible with the SCMS algorithm.'''},
            {'name': 'read_output', 'type': 'function', 'description': '''Read SCMS output files and retrieve walker coordinates.'''},
            {'name': 'run', 'type': 'function', 'description': '''Identify density ridges in a gridded image using the SCMS algorithm.'''},
            {'name': 'write_output', 'type': 'function', 'description': '''Write SCMS output coordinates to a file.'''}
        ]
    },
    'crispy.pruning': {
        'module': 'crispy.pruning',
        'description': '''Undocumented''',
        'members': [
            {'name': 'crispy.pruning.Skeleton', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.fil_finder', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.pruning', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'crispy.pruning.Skeleton': {
        'module': 'crispy.pruning.Skeleton',
        'description': '''Undocumented''',
        'members': [
            {'name': 'Skeleton', 'type': 'class', 'description': '''Undocumented'''}
        ]
    },
    'crispy.pruning.fil_finder': {
        'module': 'crispy.pruning.fil_finder',
        'description': '''Undocumented''',
        'members': [
            {'name': 'crispy.pruning.fil_finder.length', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.fil_finder.pixel_ident', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.fil_finder.utilities', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'crispy.pruning.fil_finder.length': {
        'module': 'crispy.pruning.fil_finder.length',
        'description': '''Taken from FilFinder (v1.''',
        'members': [
            {'name': 'init_lengths', 'type': 'function', 'description': '''This is a wrapper on fil_length for running on the branches of the.'''},
            {'name': 'longest_path', 'type': 'function', 'description': '''Takes the output of pre_graph and runs the shortest path algorithm.'''},
            {'name': 'main_length', 'type': 'function', 'description': '''Wraps previous functionality together for all of the skeletons in the.'''},
            {'name': 'pre_graph', 'type': 'function', 'description': '''This function converts the skeletons into a graph object compatible with.'''},
            {'name': 'prune_graph', 'type': 'function', 'description': '''Function to remove unnecessary branches, while maintaining connectivity.'''},
            {'name': 'skeleton_length', 'type': 'function', 'description': '''Length finding via morphological operators.'''}
        ]
    },
    'crispy.pruning.fil_finder.pixel_ident': {
        'module': 'crispy.pruning.fil_finder.pixel_ident',
        'description': '''Taken from FilFinder (v1.''',
        'members': [
            {'name': 'extremum_pts', 'type': 'function', 'description': '''This function returns the the farthest extents of each filament.'''},
            {'name': 'find_extran', 'type': 'function', 'description': '''Identify pixels that are not necessary to keep the connectivity of the.'''},
            {'name': 'find_filpix', 'type': 'function', 'description': '''Identifies the types of pixels in the given skeletons.'''},
            {'name': 'is_blockpoint', 'type': 'function', 'description': '''Determine if point is part of a block:.'''},
            {'name': 'is_tpoint', 'type': 'function', 'description': '''Determine if point is part of a block:.'''},
            {'name': 'isolateregions', 'type': 'function', 'description': '''Labels regions in a boolean array and returns individual arrays for each.'''},
            {'name': 'make_final_skeletons', 'type': 'function', 'description': '''Creates the final skeletons outputted by the algorithm.'''},
            {'name': 'merge_nodes', 'type': 'function', 'description': '''Combine a node into its neighbors.'''},
            {'name': 'pix_identify', 'type': 'function', 'description': '''This function is essentially a wrapper on find_filpix.'''},
            {'name': 'recombine_skeletons', 'type': 'function', 'description': '''Takes a list of skeleton arrays and combines them back into.'''}
        ]
    },
    'crispy.pruning.fil_finder.utilities': {
        'module': 'crispy.pruning.fil_finder.utilities',
        'description': '''Utility functions for fil-finder package (v1.''',
        'members': [
            {'name': 'distance', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'eight_con', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'find_nearest', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'in_ipynb', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'pad_image', 'type': 'function', 'description': '''Figure out where an image needs to be padded based on pixel extents.'''},
            {'name': 'product_gen', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'red_chisq', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'round_to_odd', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'shifter', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'threshold_local', 'type': 'function', 'description': '''skimage changed threshold_adaptive to threshold_local.'''}
        ]
    },
    'crispy.pruning.pruning': {
        'module': 'crispy.pruning.pruning',
        'description': '''Undocumented''',
        'members': [
            {'name': 'bodyPoints', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'branchedPoints', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'classify_structure', 'type': 'function', 'description': '''Classify the components of a skeleton into labeled branches, intersections, and endpoints.'''},
            {'name': 'coord_list', 'type': 'function', 'description': '''Extract coordinates of non-zero pixels from a list of skeleton arrays.'''},
            {'name': 'endPoints', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_furthest_nodes', 'type': 'function', 'description': '''Take a list of end pixels for each skeleton, and return a list of coordinates of endpoints that are furtherest from.'''},
            {'name': 'init_branch_properties', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'init_lengths', 'type': 'function', 'description': '''Compute lengths and intensities for branches in 3D skeletons.'''},
            {'name': 'main_length_3D', 'type': 'function', 'description': '''Compute the main lengths of 3D skeletons and generate longest path arrays.'''},
            {'name': 'pre_graph_3D', 'type': 'function', 'description': '''Convert 3D skeletons into graph representations with weighted edges.'''},
            {'name': 'pre_graph_3D_old', 'type': 'function', 'description': '''The 3D version of Eric Koch's pre_graph function in FilFinder.'''},
            {'name': 'remove_bad_ppv_branches', 'type': 'function', 'description': '''Remove unphysical branches from a labeled 3D skeleton in PPV space.'''},
            {'name': 'save_labskel2fits', 'type': 'function', 'description': '''Save labeled skeletons into a single FITS file.'''},
            {'name': 'segment_len', 'type': 'function', 'description': '''Calculate the length of a skeleton segment.'''},
            {'name': 'walk_through_segment_2D', 'type': 'function', 'description': '''Traverse a 2D skeleton segment to obtain an ordered list of pixel coordinates.'''},
            {'name': 'walk_through_segment_3D', 'type': 'function', 'description': '''Traverse a 3D skeleton segment to obtain an ordered list of pixel coordinates.'''}
        ]
    },
    'crispy.scms': {
        'module': 'crispy.scms',
        'description': '''Subspace Constrained Mean Shift (SCMS) algorithm for density ridge estimation.''',
        'members': [
            {'name': 'chunk_data', 'type': 'function', 'description': '''Divide data into chunks for multiprocessing.'''},
            {'name': 'euclidean_dist', 'type': 'function', 'description': '''Compute the Euclidean distances and differences between data points and walkers.'''},
            {'name': 'find_ridge', 'type': 'function', 'description': '''Identify density ridges in data using the Subspace Constrained Mean Shift (SCMS) algorithm.'''},
            {'name': 'shift_particles', 'type': 'function', 'description': '''Shift walkers toward density ridges using the Subspace Constrained Mean Shift (SCMS) algorithm.'''},
            {'name': 'shift_wakers_multiproc', 'type': 'function', 'description': '''Shift walkers towards density ridges using the SCMS algorithm with multiprocessing.'''},
            {'name': 'shift_walkers', 'type': 'function', 'description': '''Shift walkers towards density ridges using the Subspace Constrained Mean Shift (SCMS) algorithm.'''},
            {'name': 'vectorized_gaussian', 'type': 'function', 'description': '''Compute Gaussian kernel values for data points relative to walker positions.'''},
            {'name': 'wgauss_n_filtered_points', 'type': 'function', 'description': '''Compute weighted Gaussian values for data points relative to walker positions,.'''},
            {'name': 'wgauss_n_filtered_points_multiproc', 'type': 'function', 'description': '''Compute weighted Gaussian values for data points relative to walker positions.'''}
        ]
    },
}
