#
# Cluster-defining Data Computation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the computation of the physical metrics serving as a basis to
# perform the clustering-based domain decomposition, as well as to the different strategies
# available to perform the model reduction.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Extract information from path
import ntpath
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# FFT-Based Homogenization Method (Moulinec, H. and Suquet, P., 1998)
import clustering.solution.ffthombasicscheme as ffthom
# Links related procedures
import links.ioput.genlinksinputdatafile as linksglif
import links.execution.linksexec as linksexec
import links.postprocess.linkspostprocess as linkspp
#
#                                                        Compute cluster-defining quantities
# ==========================================================================================
# The discretization of the RVE into clusters requires a given quantity (evaluated at each
# domain point) in which to base the classification of different points into groups of
# similar points. Moreover, several clustering processes may be performed (each one based
# on a different quantity) and then merged in some way to obtain a unique RVE clustering
# discretization.
# According to the adopted clustering strategy, all the required quantities are computed
# by solving a microscale equilibrium problem through a given method. The storage of such
# quantities is described below for the cases where the microscale problem is solved with
# an FFT-based homogenization method (spatial discretization in a regular grid of pixels/
# voxels) or with the FEM-based homogenization method (spatial discretization in a regular
# finite element mesh). A list is also build where the quantities to be used in each
# required clustering discretization are specified. The storage is performed for each type
# of solution method as follows:
#
# A. FFT-based homogenization (spatial discretization in a regular grid of pixels/voxels):
#
#    Consider the case where one desires to perform three clustering processes, the first
#    one based on a scalar variable 'a', the second based on a first-order tensorial
#    variable b = [b1 b2] and the last one based on a second-order tensorial variable
#    c = [[c11 c12],[c21 c22]]. These quantities are stored in a array(n_voxels,7),
#    with n_voxels = d1xd2 (2D) or n_voxels = d1xd2xd3 (3D), where di is the number of
#    voxels in the dimension i, as
#                         _                       _
#                        | a b1 b2 c11 c21 c12 c22 | > voxel 0
#                array = | a b1 b2 c11 c21 c12 c22 | > voxel 1
#                        | . .. .. ... ... ... ... | > ...
#                        |_a b1 b2 c11 c21 c12 c22_| > voxel n_voxels - 1
#
#    The quantities associated to each clustering process (referring to columns of the
#    previous array) are then specified in a list as
#
#                        list = [ 0 , [1,2] , [3,4,5,6]]
#
#    Note: When the clustering quantity is a symmetric second-order tensor or a higher-order
#          tensor with major or minor simmetries, only the independent components may be
#          stored in the clustering array
#
# B. FEM-based homogenization (spatial discretization in a regular finite element mesh)
#
#   If the microscale equilibrium problem is to be solved with the Finite Element Method
#   based on computational homogenization, then a regular mesh of quadrilateral (2D) /
#   hexahedral (3D) finite elements (linear or quadratic) shall be generated in total
#   consistency with the regular grid of pixels/voxels (i.e. there is a perfect spatial
#   match quadrilateral finite element - pixel or hexahedral finite element - voxel). In
#   this way, by averaging the value of any given quantity over the finite element Gauss
#   sampling points, the storage previously described for the regular grid of pixels/voxels
#   applies.
#
def compclusteringdata(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict):
    # Get problem data
    strain_formulation = problem_dict['strain_formulation']
    n_dim = problem_dict['n_dim']
    comp_order_sym = problem_dict['comp_order_sym']
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    rg_file_name = rg_dict['rg_file_name']
    # Get clustering data
    clustering_solution_method = clst_dict['clustering_solution_method']
    clustering_strategy = clst_dict['clustering_strategy']
    # Compute total number of voxels
    n_voxels = np.prod(n_voxels_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the required cluster-defining quantities according to the adopted clustering
    # strategy and set clustering processes quantities list
    if clustering_strategy == 1:
        # In this clustering strategy, only one clustering process is performed based on the
        # strain concentration fourth-order tensor. Initialize the clustering quantities
        # array according to the number of independent strain components
        if strain_formulation == 1:
            n_clustering_var = len(comp_order_sym)**2
        clst_quantities = np.zeros((n_voxels, n_clustering_var))
        clst_dataidxs = [list(range(n_clustering_var)),]
        # Small strain formulation
        if strain_formulation == 1:
            info.displayinfo('5','Computing strain concentration tensors...')
            # Loop over independent strain components
            for i in range(len(comp_order_sym)):
                compi = comp_order_sym[i]
                so_idx = tuple([int(x) - 1 for x in list(comp_order_sym[i])])
                # Set macroscopic strain loading
                mac_strain = np.zeros((n_dim, n_dim))
                if compi[0] == compi[1]:
                    mac_strain[so_idx] = 1.0
                else:
                    mac_strain[so_idx] = 1.0
                    mac_strain[so_idx[::-1]] = 1.0
                # Solve the microscale equilibrium problem through a given homogenization
                # method and get the strain concentation tensor components associated to the
                # imposed macroscale strain loading component
                if clustering_solution_method == 1:
                    # Run Moulinec and Suquet FFT-based homogenization method and get the
                    # strain concentration tensor components
                    strain_vox = ffthom.ffthombasicscheme(problem_dict, rg_dict, mat_dict,
                                                          mac_strain)
                elif clustering_solution_method == 2:
                    # Generate microscale problem Links input data file
                    Links_file_name = rg_file_name + '_SCT_' + compi
                    links_file_path = linksglif.writelinksinputdatafile(
                        Links_file_name, dirs_dict, problem_dict, mat_dict, rg_dict,
                        clst_dict, mac_strain)
                    # Run Links (FEM-based homogenization method)
                    links_bin_path = clst_dict['links_dict']['links_bin_path']
                    linksexec.runlinks(links_bin_path, links_file_path)
                    # Get the strain concentration tensor components
                    strain_vox = linkspp.getlinksstrainvox(links_file_path, n_dim,
                                                           comp_order_sym, n_voxels_dims)
                # Assemble strain concentration tensor components associated to the imposed
                # macroscale strain loading component
                for j in range(len(comp_order_sym)):
                    compj = comp_order_sym[j]
                    clst_quantities[:, i*len(comp_order_sym) + j] = \
                        strain_vox[compj].flatten()
            # Add clustering data to clustering dictionary
            clst_dict['clst_quantities'] = clst_quantities
            clst_dict['clst_dataidxs'] = clst_dataidxs
