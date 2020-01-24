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
    # Compute the required cluster-defining quantities according to the adopted clustering
    # strategy
    if clustering_strategy == 1:
        pass
