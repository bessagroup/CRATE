#
# Cluster-defining Quantities Computation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the computation of the physical metrics serving
# as a basis to perform the clustering-based domain decomposition, as well as to the
# different strategies available to perform the model reduction.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Extract information from path
import ntpath
# Display messages
import ioput.info as info
# Read user input data file
import ioput.readinputdata as rid
# FFT-Based Homogenization Method (Moulinec, H. and Suquet, P., 1998)
import FFTHomogenizationBasicScheme
# Links related procedures
import Links.ioput.genLinksInputFile as LinksGLIF
import Links.execution.LinksExecution as LinksExec
import Links.postprocess.LinksPostProcess as LinksPP
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
def computeClusteringQuantities(dirs_dict,problem_dict,mat_dict,rg_dict,clst_dict):
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
        clst_quantities = np.zeros((n_voxels,n_clustering_var))
        clst_dataidxs = [list(range(n_clustering_var)),]
        # Small strain formulation
        if strain_formulation == 1:
            info.displayinfo('5','Computing strain concentration tensors...')
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
                # Solve the microscale equilibrium problem through a given homogenization
                # method and get the strain concentation tensor components associated to the
                # imposed macroscale strain loading component
                if clustering_solution_method == 1:
                    # Run Moulinec and Suquet FFT-based homogenization method and get the
                    # strain concentration tensor components
                    strain_vox = FFTHomogenizationBasicScheme.FFTHomogenizationBasicScheme(
                                         copy.deepcopy(problem_dict),copy.deepcopy(rg_dict),
                                                         copy.deepcopy(mat_dict),mac_strain)
                elif clustering_solution_method == 2:
                    # Generate microscale problem Links input data file
                    Links_file_name = rg_file_name + '_SCT_' + compi
                    Links_file_path = \
                        LinksGLIF.writeLinksInputDataFile(Links_file_name,dirs_dict,
                                                          problem_dict,mat_dict,rg_dict,
                                                          clst_dict,mac_strain)
                    # Run Links (FEM-based homogenization method)
                    Links_bin_path = clst_dict['Links_dict']['Links_bin_path']
                    LinksExec.runLinks(Links_bin_path,Links_file_path)
                    # Get the strain concentration tensor components
                    strain_vox = LinksPP.getLinksStrainVox(Links_file_path,n_dim,
                                                      comp_order_sym,n_voxels_dims)
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
    n_dim, comp_order_sym, comp_order_nsym = mop.getproblemtypeparam(problem_type)
    problem_dict = dict()
    problem_dict['strain_formulation'] = strain_formulation
    problem_dict['problem_type'] = problem_type
    problem_dict['n_dim'] = n_dim
    problem_dict['comp_order_sym'] = comp_order_sym
    problem_dict['comp_order_nsym'] = comp_order_nsym
    material_properties = dict()
    material_properties['1'] = dict()
    material_properties['1']['E'] = 210e6
    material_properties['1']['v'] = 0.3
    material_properties['2'] = dict()
    material_properties['2']['E'] = 70e6
    material_properties['2']['v'] = 0.33
    mat_dict = dict()
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
