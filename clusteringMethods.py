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
# Display messages
import info
# Tensorial operations
import tensorOperations as top
#
#                                                                         Perform clustering
# ==========================================================================================
# ...
def performClustering(n_material_phases,rg_dict,clst_dict):
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
    # Initialize clustering processes labels list
    clst_processes = list()
    # Perform clustering processes according to the selected clustering method
    if clustering_method == 1:
        info.displayInfo('5','Performing K-Means clustering...')
        # Loop over clustering processes (each with associated data indexes)
        for iclst in range(len(clst_dataidxs)):
            info.displayInfo('6','progress',iclst+1,len(clst_dataidxs))
            # Initialize current clustering process labels
            clst_process_lbls_flat = np.full(n_voxels,-1,dtype=int)
            label_correction = 0
            # Get current clustering process data indexes
            data_indexes = clst_dataidxs[iclst]
            # Loop over material phases
            for iphase in range(n_material_phases):
                # Set number of clusters
                n_clusters = phase_nclusters[str(iphase+1)]
                # Set clustering training dataset
                dataset = top.getCondensedMatrix(clst_quantities,\
                                            phase_voxel_flatidx[str(iphase+1)],data_indexes)
                # Perform kmeans clustering (Lloyd's algorithm)
                kmeans = sklearn.cluster.KMeans(n_clusters,init = 'k-means++',n_init = 10,
                                  max_iter = 300,tol = 1e-4,algorithm = 'auto').fit(dataset)
                # Store current material phase cluster labels
                clst_process_lbls_flat[phase_voxel_flatidx[str(iphase+1)]] = \
                                                           kmeans.labels_ + label_correction
                # Update label correction so that different material phases do not share the
                # same label
                label_correction = label_correction + n_clusters
            # Store current clustering process labels list
            clst_processes.append(list(clst_process_lbls_flat))
        info.displayInfo('6','completed')
    # Perform RVE clustering discretization according to the selected clustering strategy
    if clustering_strategy == 1:
        # Get the cluster labels from the unique clustering process
        clst_dict['voxels_clstlbl_flat'] = clst_processes[0]
    # Return
    return None
