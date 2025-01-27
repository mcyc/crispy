# This file is auto-generated. Edit descriptions and structure as needed.

API_REFERENCE = {
    'crispy.grid_ridge': {
        'module': 'crispy.grid_ridge',
        'description': '''Functions for gridding and processing CRISPy results on reference images.''',
        'members': [
            {'name': 'clean_grid', 'type': 'function', 'description': '''Process and grid CRISPy coordinates onto a reference image, labeling and cleaning skeleton structures.'''},
            {'name': 'clean_grid_ppv', 'type': 'function', 'description': '''Process and grid CRISPy coordinates in PPV space onto a reference image, labeling and cleaning skeleton structures.'''},
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
            {'name': 'threshold_local', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'write_output', 'type': 'function', 'description': '''Write SCMS output coordinates to a file.'''}
        ]
    },
    'crispy.pruning': {
        'module': 'crispy.pruning',
        'description': '''Undocumented''',
        'members': [
            {'name': 'crispy.pruning.Skeleton', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.pruning', 'type': 'module', 'description': '''No description available.'''},
            {'name': 'crispy.pruning.structures', 'type': 'module', 'description': '''No description available.'''}
        ]
    },
    'crispy.pruning.Skeleton': {
        'module': 'crispy.pruning.Skeleton',
        'description': '''Undocumented''',
        'members': [
            {'name': 'Skeleton', 'type': 'class', 'description': '''Undocumented'''}
        ]
    },
    'crispy.pruning.pruning': {
        'module': 'crispy.pruning.pruning',
        'description': '''Undocumented''',
        'members': [
            {'name': 'bodyPoints', 'type': 'function', 'description': '''Identify body points in a skeletonized structure.'''},
            {'name': 'branchedPoints', 'type': 'function', 'description': '''Identify branch points in a skeletonized structure.'''},
            {'name': 'classify_structure', 'type': 'function', 'description': '''Classify the components of a skeleton into labeled branches, intersections, and endpoints.'''},
            {'name': 'endPoints', 'type': 'function', 'description': '''Identify endpoints in a skeletonized structure.'''},
            {'name': 'init_branch_properties', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'init_lengths_3D', 'type': 'function', 'description': '''Compute lengths and intensities for branches in 3D skeletons.'''},
            {'name': 'main_length_3D', 'type': 'function', 'description': '''Compute the main lengths of 3D skeletons and generate longest path arrays.'''},
            {'name': 'pre_graph_3D', 'type': 'function', 'description': '''Convert 3D skeletons into graph representations with weighted edges.'''},
            {'name': 'remove_bad_ppv_branches', 'type': 'function', 'description': '''Remove unphysical branches from a labeled 3D skeleton in PPV space.'''},
            {'name': 'segment_len', 'type': 'function', 'description': '''Calculate the length of a skeleton segment.'''},
            {'name': 'walk_through_segment_3D', 'type': 'function', 'description': '''Traverse a 3D skeleton segment to obtain an ordered list of pixel coordinates.'''}
        ]
    },
    'crispy.pruning.structures': {
        'module': 'crispy.pruning.structures',
        'description': '''A place host specialized binary strucutres.''',
        'members': [
            {'name': 'two_con', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_base_block', 'type': 'function', 'description': '''Undocumented'''},
            {'name': 'get_footprints', 'type': 'function', 'description': '''Undocumented'''}
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
