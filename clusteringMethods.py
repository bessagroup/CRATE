#
# Perform Clustering Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Working with arrays
import numpy as np
# Unsupervised clustering algorithms
import sklearn.cluster
# Python object serialization
import pickle
# Inspect file name and line
import inspect
# Display messages
import info
# Display errors, warnings and built-in exceptions
import errors
# Tensorial operations
import tensorOperations as top
#
#                                                                         Perform clustering
# ==========================================================================================
# Perform clustering processes according to the selected clustering strategy by employing
# the selected clustering method
def performClustering(dirs_dict,mat_dict,rg_dict,clst_dict):
    # Get directories and paths data
    cluster_file_path = dirs_dict['cluster_file_path']
    # Get material data
    n_material_phases = mat_dict['n_material_phases']
    material_properties = mat_dict['material_properties']
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    n_voxels = np.prod(n_voxels_dims)
    phase_voxel_flatidx = rg_dict['phase_voxel_flatidx']
    # Get clustering data
    clustering_method = clst_dict['clustering_method']
    clustering_strategy = clst_dict['clustering_strategy']
    phase_nclusters = clst_dict['phase_nclusters']
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
        info.displayInfo('5','Performing K-Means clustering...')
        # Loop over clustering processes (each with associated data indexes)
        for iclst in range(len(clst_dataidxs)):
            info.displayInfo('6','progress',iclst+1,len(clst_dataidxs))
            # Initialize current clustering process labels
            clst_process_lbls_flat = np.full(n_voxels,-1,dtype=int)
            # Get current clustering process data indexes
            data_indexes = clst_dataidxs[iclst]
            # Loop over material phases
            for mat_phase in material_properties.keys():
                # Set number of clusters
                n_clusters = phase_nclusters[mat_phase]
                # Set clustering training dataset
                dataset = top.getCondensedMatrix(clst_quantities,\
                                                phase_voxel_flatidx[mat_phase],data_indexes)
                # Perform kmeans clustering (Lloyd's algorithm)
                kmeans = sklearn.cluster.KMeans(n_clusters,init = 'k-means++',n_init = 10,
                                  max_iter = 300,tol = 1e-4,algorithm = 'auto').fit(dataset)
                # Store current material phase cluster labels
                clst_process_lbls_flat[phase_voxel_flatidx[mat_phase]] = \
                                                           kmeans.labels_ + label_correction
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Validation
                if False:
                    # Set functions being validated
                    val_functions = ['performClustering()',]
                    # Display validation header
                    print('\nValidation: ',
                            (len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
                    print(92*'-')
                    # Display validation
                    np.set_printoptions(linewidth=np.inf)
                    print('\nClustering process: ', iclst + 1)
                    print('\nData indexes: ', data_indexes)
                    print('\nMaterial phase: ', mat_phase)
                    print('\nNumber of clusters: ', n_clusters)
                    print('\nClustering quantities:')
                    print(clst_quantities)
                    print('\nMaterial phase indexes:',phase_voxel_flatidx[mat_phase])
                    print('\nMaterial phase dataset:')
                    print(dataset)
                    print('\nLabels (-1 means not labeled):')
                    print(clst_process_lbls_flat)
                    # Display validation footer
                    print('\n' + 92*'-' + '\n')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update label correction
                label_correction = label_correction + n_clusters
            # Check if all the training dataset points have been labeled
            if np.any(clst_process_lbls_flat == -1):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00036',location.filename,location.lineno+1,iclst+1)
            # Store current clustering process labels list
            clst_processes.append(list(clst_process_lbls_flat))
        info.displayInfo('6','completed')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform RVE clustering discretization according to the selected clustering strategy
    if clustering_strategy == 1:
        # Get the cluster labels from the unique clustering process
        clst_dict['voxels_clstlbl_flat'] = clst_processes[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open file which contains all the required information associated to the clustering
    # discretization
    try:
        cluster_file = open(cluster_file_path,'wb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
    # Store clustering data
    pickle.dump(clst_dict,cluster_file)
    # Close file
    cluster_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return None
