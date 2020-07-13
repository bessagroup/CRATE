#
# Perform Clustering Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the clustering-based domain decomposition of the representative
# volume element (RVE) into the cluster-reduced representative volume element (CRVE).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Unsupervised clustering algorithms
import sklearn.cluster
# Python object serialization
import pickle
# Inspect file name and line
import inspect
# Generate efficient iterators
import itertools as it
# Display messages
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                                         Perform clustering
# ==========================================================================================
# Perform clustering processes according to the selected clustering strategy by employing
# the selected clustering method
def clustering(dirs_dict, mat_dict, rg_dict, clst_dict):
    # Get directories and paths data
    cluster_file_path = dirs_dict['cluster_file_path']
    # Get material data
    material_phases = mat_dict['material_phases']
    # Get regular grid data
    rve_dims = rg_dict['rve_dims']
    n_voxels_dims = rg_dict['n_voxels_dims']
    n_voxels = np.prod(n_voxels_dims)
    phase_voxel_flatidx = rg_dict['phase_voxel_flatidx']
    # Get clustering data
    clustering_method = clst_dict['clustering_method']
    clustering_strategy = clst_dict['clustering_strategy']
    phase_n_clusters = clst_dict['phase_n_clusters']
    clst_dataidxs = clst_dict['clst_dataidxs']
    clst_quantities = clst_dict['clst_quantities']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize clustering processes labels list
    clst_processes = list()
    # Initialize label correction (avoid that different material phases as well as different
    # clustering processes share the same labels)
    label_correction = 0
    # Perform clustering processes according to the selected clustering method
    if clustering_method == 1:
        info.displayinfo('5', 'Performing K-Means clustering...')
        # Loop over clustering processes (each with associated data indexes)
        for iclst in range(len(clst_dataidxs)):
            info.displayinfo('6', 'progress', iclst + 1, len(clst_dataidxs))
            # Initialize current clustering process labels
            clst_process_lbls_flat = np.full(n_voxels, -1, dtype=int)
            # Get current clustering process data indexes
            data_indexes = clst_dataidxs[iclst]
            # Loop over material phases
            for mat_phase in material_phases:
                # Set number of clusters
                n_clusters = phase_n_clusters[mat_phase]
                # Set clustering training dataset
                dataset = mop.getcondmatrix(clst_quantities,
                                            phase_voxel_flatidx[mat_phase], data_indexes)
                # Perform kmeans clustering (Lloyd's algorithm)
                kmeans = sklearn.cluster.KMeans(n_clusters, init = 'k-means++', n_init = 10,
                                                max_iter = 300, tol = 1e-4,
                                                algorithm = 'auto').fit(dataset)
                # Store current material phase cluster labels
                clst_process_lbls_flat[phase_voxel_flatidx[mat_phase]] = \
                    kmeans.labels_ + label_correction
                # Update label correction
                label_correction = label_correction + n_clusters
            # Check if all the training dataset points have been labeled
            if np.any(clst_process_lbls_flat == -1):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00036', location.filename, location.lineno + 1,
                                    iclst + 1)
            # Store current clustering process labels list
            clst_processes.append(list(clst_process_lbls_flat))
        info.displayinfo('6', 'completed')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform RVE clustering discretization according to the selected clustering strategy
    if clustering_strategy == 1:
        # Build cluster labels from the unique clustering process (regular grid shape)
        voxels_clusters = np.array(clst_processes[0], dtype=int).reshape(n_voxels_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize mapping dictionary to sort the cluster labels in asceding order of material
    # phase
    sort_dict = dict()
    # Initialize material phase initial cluster label
    lbl_init = 0
    # Loop over material phases sorted in ascending order
    sorted_mat_phases = list(np.sort(list(material_phases)))
    for mat_phase in sorted_mat_phases:
        # Get old cluster labels
        old_clusters = np.unique(voxels_clusters.flatten()[phase_voxel_flatidx[mat_phase]])
        # Set new cluster labels
        new_clusters = list(range(lbl_init, lbl_init + phase_n_clusters[mat_phase]))
        # Set next material phase initial cluster label
        lbl_init = lbl_init + phase_n_clusters[mat_phase]
        # Build mapping dictionary to sort the cluster labels
        for i in range(phase_n_clusters[mat_phase]):
            if old_clusters[i] in sort_dict.keys():
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00038', location.filename, location.lineno + 1)
            else:
                sort_dict[old_clusters[i]] = new_clusters[i]
    # Check mapping dictionary
    if np.any(np.sort(list(sort_dict.keys())) != range(sum(phase_n_clusters.values()))):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00039', location.filename, location.lineno + 1)
    elif np.any(np.sort([sort_dict[key] for key in sort_dict.keys()]) != \
            range(sum(phase_n_clusters.values()))):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00039', location.filename, location.lineno + 1)
    # Sort cluster labels in ascending order of material phase
    for voxel_idx in it.product(*[list(range(n_voxels_dims[i])) \
            for i in range(len(n_voxels_dims))]):
        voxels_clusters[voxel_idx] = sort_dict[voxels_clusters[voxel_idx]]
    # Store cluster labels
    clst_dict['voxels_clusters'] = voxels_clusters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store material clusters belonging to each material phase
    for mat_phase in material_phases:
        clst_dict['phase_clusters'][mat_phase] = \
            np.unique(voxels_clusters.flatten()[phase_voxel_flatidx[mat_phase]])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute voxel volume
    voxel_vol = np.prod([float(rve_dims[i])/n_voxels_dims[i] for i in range(len(rve_dims))])
    # Compute RVE volume
    rve_vol = np.prod(rve_dims)
    # Compute volume fraction associated to each material cluster
    for cluster in np.unique(voxels_clusters):
        clst_dict['clusters_f'][str(cluster)] = \
            (np.sum(voxels_clusters == cluster)*voxel_vol)/rve_vol
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open file which contains all the required information associated to the clustering
    # discretization
    try:
        cluster_file = open(cluster_file_path, 'wb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1, message)
    # Dump clustering data
    info.displayinfo('5', 'Storing clustering file (.clusters)...')
    pickle.dump(clst_dict, cluster_file)
    # Close file
    cluster_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                                                Perform compatibility check
#                                                (loading previously computed offline stage)
# ==========================================================================================
# Perform a compatibility check between the clustering parameters read from the input data
# file and the previously computed offline stage loaded data
def checkclstcompat(problem_dict, rg_dict, clst_dict_read, clst_dict):
    # Check clustering method, clustering strategy and clustering solution method
    keys = ['clustering_method', 'clustering_strategy', 'clustering_solution_method']
    for key in keys:
        if clst_dict[key] != clst_dict_read[key]:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00044', location.filename, location.lineno + 1, key,
                                clst_dict_read[key],clst_dict[key])
    # Check number of clusters associated to each material phase
    if clst_dict['phase_n_clusters'] != clst_dict_read['phase_n_clusters']:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00045', location.filename, location.lineno + 1,
                            clst_dict_read['phase_n_clusters'],
                            clst_dict['phase_n_clusters'])
    # Check spatial discretization
    elif list(clst_dict['voxels_clusters'].shape) != rg_dict['n_voxels_dims']:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00046', location.filename, location.lineno + 1,
                            rg_dict['n_voxels_dims'],
                            list(clst_dict['voxels_clusters'].shape))
