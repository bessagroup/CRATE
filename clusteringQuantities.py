#
# Cluster-defining Quantities Computation Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Display messages
import info
#
#                                                        Compute cluster-defining quantities
# ==========================================================================================
# The discretization of the RVE into clusters requires one or more quantities in which to
# base the classification of different points into groups of similar points. According to
# the adopted clustering strategy, all the required quantities (usually based on the
# solution of a microscale problem through computational homogenization) are computed and
# stored in a format dependent on the type of spatial discretization:
#
# A. Spatial discretization in a regular grid of pixels/voxels
#
#    Consider the general case where the clustering is based on a scalar variable 'a', a
#    first-order tensorial variable b = [b1 b2] and a second-order tensorial variable
#    c = [[c11 c12],[c21 c22]]. These quantities are stored in a array(n_voxels,7),
#    with n_voxels = d1xd2 (2D) or n_voxels = d1xd2xd3 (3D), where di is the number of
#    voxels in the dimension i, as
#
#                         _                       _
#                        | a b1 b2 c11 c21 c12 c22 | > voxel 1
#                array = | a b1 b2 c11 c21 c12 c22 | > voxel 2
#                        | . .. .. ... ... ... ... | > ...
#                        |_a b1 b2 c11 c21 c12 c22_| > voxel n_voxels
#
#   Note: Second-order tensorial variables (or higher-order tensorial variables stored in
#         matrix form) are to be stored columnwise
#
def computeClusteringQuantities(clustering_strategy,clustering_solution_method,discret_file_path):
    # Extract the required data from the spatial discretization file(s) according to the
    # chosen solution method to compute the cluster-defining quantities
    info.displayInfo('5','Reading discretization file...')
    if clustering_solution_method == 1:
        # [WIP - Get the regular grid mesh file]
        # Read the regular grid mesh
        regular_grid = np.loadtxt(discret_file_path)
        print(regular_grid)
        # Set number of pixels/voxels in each dimension and total number of pixels/voxels
        n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        n_voxels = np.prod(n_voxels_dir)
        print(n_voxels_dim)
        print(n_voxels)
    # Compute the required cluster-defining quantities according to the adopted clustering
    # strategy
    if clustering_strategy == 1:
        # Clustering based solely on the strain concentration tensor. The total number of
        # (scalar) cluster-defining quantities is thus equal to the number of independent
        # strain components (according to the problem type and strain formulation)
        n_clustering_var = n_strain
        # Initialize clustering quantities array
        clustering_quantities = np.zeros((n_voxels,n_clustering_var))
        # Loop over independent strain components
        for i in range(n_strain):
            # Set macroscopic strain loading
            mac_strain = np.zeros((n_strain))
            mac_strain[i] = 1.0
            # Call FFT method
            # ...
            # Assemble strain concentration tensor components in the clustering quantities
            # array
            for j in range(n_strain):
                pass


        pass
    # Return the clustering quantities array
    return clustering_array
