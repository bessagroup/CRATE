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
# Operating system related functions
import os
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Extract information from path
import ntpath
# Generate efficient iterators
import itertools as it
# Display messages
import info
# Read user input data file
import readInputData as rid
# FFT-Based Homogenization Method (Moulinec, H. and Suquet, P., 1998)
import FFTHomogenizationBasicScheme
#
#                                                        Compute cluster-defining quantities
# ==========================================================================================
# The discretization of the RVE into clusters requires a given quantity (evaluated at each
# domain point) in which to base the classification of different points into groups of
# similar points. Moreover, several clustering processes may be performed (each one based
# on a different quantity) and then merged in some way to obtain a unique RVE clustering
# discretization.
# According to the adopted clustering strategy, all the required quantities (usually based
# on the solution of a microscale equilibrium problem) are computed and stored in a format
# dependent on the type of spatial discretization. A list is also build where the quantities
# to be used in each required clustering discretization are specified. The storage is
# performed for each type of spatial discretization as detailed below:
#
# A. Spatial discretization in a regular grid of pixels/voxels
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
def computeClusteringQuantities(problem_dict,mat_dict,rg_dict,clst_dict):
    # Get problem data
    strain_formulation = problem_dict['strain_formulation']
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order_sym = problem_dict['comp_order_sym']
    # Get clustering data
    clustering_solution_method = clst_dict['clustering_solution_method']
    clustering_strategy = clst_dict['clustering_strategy']
    # Get the required data from the spatial discretization file(s) according to the
    # chosen solution method to compute the cluster-defining quantities
    if clustering_solution_method == 1:
        # Get the spatial discretization file (regular grid of pixels/voxels)
        regular_grid = rg_dict['regular_grid']
        # Get number of pixels/voxels in each dimension and total number of pixels/voxels
        n_voxels_dims = rg_dict['n_voxels_dims']
        n_voxels = np.prod(n_voxels_dims)
        # Get RVE dimensions
        rve_dims = rg_dict['rve_dims']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the required cluster-defining quantities according to the adopted clustering
    # strategy and set clustering processes quantities list
    if clustering_strategy == 1:
        # In this clustering strategy, only one clustering process is performed based on the
        # strain concentration fourth-order tensor. Initialize the clustering quantities
        # array according to the number of independent strain components
        if strain_formulation == 1:
            n_clustering_var = len(comp_order_sym)**2
        clst_quantities = np.zeros((n_voxels,n_clustering_var))
        clst_dataidxs = [list(range(n_clustering_var)),]
        # Small strain formulation
        if strain_formulation == 1:
            info.displayInfo('5','Computing strain concentration tensors...')
            # Loop over independent strain components
            for i in range(len(comp_order_sym)):
                compi = comp_order_sym[i]
                so_idx = tuple([int(x)-1 for x in list(comp_order_sym[i])])
                # Set macroscopic strain loading
                mac_strain = np.zeros((n_dim,n_dim))
                if compi[0] == compi[1]:
                    mac_strain[so_idx] = 1.0
                else:
                    mac_strain[so_idx] = 1.0
                    mac_strain[so_idx[::-1]] = 1.0
                # Solve RVE static equilibrium problem
                strain_vox = FFTHomogenizationBasicScheme.FFTHomogenizationBasicScheme(
                                         copy.deepcopy(problem_dict),copy.deepcopy(rg_dict),
                                         copy.deepcopy(mat_dict),mac_strain)
                # Assemble strain concentration tensor components associated to the imposed
                # macroscale strain loading component
                for j in range(len(comp_order_sym)):
                    compj = comp_order_sym[j]
                    clst_quantities[:,i*len(comp_order_sym)+j] = strain_vox[compj].flatten()
                # --------------------------------------------------------------------------
                # Validation:
                if __name__ == '__main__':
                    if n_dim == 2:
                        val_voxel_idx = (2,1)
                        val_voxel_row = val_voxel_idx[0]*n_voxels_dims[1] + val_voxel_idx[1]
                    else:
                        val_voxel_idx = (2,1,3)
                        val_voxel_row = \
                                    val_voxel_idx[0]*(n_voxels_dims[1]*n_voxels_dims[2]) + \
                                        val_voxel_idx[1]*n_voxels_dims[2] + val_voxel_idx[2]
                    print('\nPerturbed strain component: ' + compi)
                    for j in range(len(comp_order_sym)):
                        compj = comp_order_sym[j]
                        print('  Strain (' + compj + '): ', \
                        '{:>11.4e}'.format(strain_vox[compj][val_voxel_idx]))
                # --------------------------------------------------------------------------
            # Add clustering data to clustering dictionary
            clst_dict['clst_quantities'] = clst_quantities
            clst_dict['clst_dataidxs'] = clst_dataidxs
            # ------------------------------------------------------------------------------
            # Validation:
            if __name__ == '__main__':
                print('\nClustering quantities array row - Voxel ', val_voxel_idx, ':')
                print(clst_quantities[val_voxel_row,:])
                print('\nClustering processes list:')
                print(clst_dataidxs)
            # ------------------------------------------------------------------------------
    # Return
    return None
#
#                                                                     Validation (temporary)
# ==========================================================================================
if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['computeClusteringQuantities()',]
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set functions arguments
    strain_formulation = 1
    problem_type = 4
    n_dim, comp_order_sym, comp_order_nsym = rid.setProblemTypeParameters(problem_type)
    problem_dict()
    problem_dict['strain_formulation'] = strain_formulation
    problem_dict['problem_type'] = problem_type
    problem_dict['n_dim'] = n_dim
    problem_dict['comp_order_sym'] = comp_order_sym
    problem_dict['comp_order_nsym'] = comp_order_nsym
    n_material_phases = 2
    material_properties = dict()
    material_properties['1'] = dict()
    material_properties['1']['E'] = 210e6
    material_properties['1']['v'] = 0.3
    material_properties['2'] = dict()
    material_properties['2']['E'] = 70e6
    material_properties['2']['v'] = 0.33
    mat_dict = dict()
    mat_dict['n_material_phases'] = n_material_phases
    mat_dict['material_properties'] = material_properties
    if problem_type == 1:
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
        'debug/FFT_Homogenization_Method/RVE_2D_2Phases_5x5.rgmsh.npy'
        rve_dims = [1.0,1.0]
    else:
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
        'debug/FFT_Homogenization_Method/RVE_3D_2Phases_5x5x5.rgmsh.npy'
        rve_dims = [1.0,1.0,1.0]
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
    else:
        regular_grid = np.loadtxt(discret_file_path)
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    rg_dict = dict()
    rg_dict['rve_dims'] = rve_dims
    rg_dict['regular_grid'] = regular_grid
    rg_dict['n_voxels_dims'] = n_voxels_dims
    clustering_strategy = 1
    clustering_solution_method = 1
    clst_dict = dict()
    clst_dict['clustering_strategy'] = clustering_strategy
    clst_dict['clustering_solution_method'] = clustering_solution_method
    # Call function
    computeClusteringQuantities(problem_dict,mat_dict,rg_dict,clst_dict)
    # Display validation footer
    print('\n' + 92*'-' + '\n')
