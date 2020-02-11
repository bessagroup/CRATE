#
# Packager Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Extract information from path
import ntpath
# Generate efficient iterators
import itertools as it
#
#                                                                          Package functions
# ==========================================================================================
# Package data associated to a regular grid of pixels/voxels
def packageRegularGrid(discret_file_path,rve_dims,n_material_phases,n_dim):
    # Initialize regular grid dictionary
    rg_dict = dict()
    # Read the spatial discretization file (regular grid of pixels/voxels)
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
    else:
        regular_grid = np.loadtxt(discret_file_path)
    # Set number of pixels/voxels in each dimension
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    n_voxels = np.prod(n_voxels_dims)
    # Flatten the regular grid array such that:
    #
    # 2D Problem (swipe 2-1)   - voxel(i,j) is stored in index = i*d2 + j, where d2 is the
    #                                       the number of voxels along dimension 2
    #
    # 3D Problem (swipe 3-2-1) - voxel(i,j,k) is stored in index = i*(d2*d3) + j*d3 + k,
    #                                         where d2 and d3 are the number of voxels along
    #                                         dimensions 2 and 3 respectively
    #
    regular_grid_flat = list(regular_grid.flatten())
    # Build flattened list with the voxels indexes (consistent with the flat regular grid)
    voxels_idx_flat = list()
    shape = tuple([n_voxels_dims[i] for i in range(n_dim)])
    voxels_idx_flat = [np.unravel_index(i,shape) for i in range(n_voxels)]
    # Set voxel flattened indexes associated to each material phase
    phase_voxel_flatidx = list()
    for phase_idx in range(n_material_phases):
        is_phase_list = (regular_grid.flatten() - 1) == phase_idx
        phase_voxel_flatidx.append(\
                                 list(it.compress(range(len(is_phase_list)),is_phase_list)))
    # Build regular grid dictionary
    rg_dict['rve_dims'] = rve_dims
    rg_dict['regular_grid'] = regular_grid
    rg_dict['n_voxels_dims'] = n_voxels_dims
    rg_dict['regular_grid_flat'] = regular_grid_flat
    rg_dict['voxels_idx_flat'] = voxels_idx_flat
    rg_dict['phase_voxel_flatidx'] = phase_voxel_flatidx
    # Return regular grid dictionary
    return rg_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the clustering on a regular grid of pixels/voxels
def packageRGClustering(clustering_method,clustering_strategy,clustering_solution_method,\
                                                             phase_nclusters,n_voxels_dims):
    # Initialize clustering dictionary
    clst_dict = dict()
    # Initialize flattened list with voxels cluster labels
    n_voxels = np.prod(n_voxels_dims)
    voxels_clstlbl_flat = n_voxels*[-1]
    # Build clustering dictionary
    clst_dict['clustering_method'] = clustering_method
    clst_dict['clustering_strategy'] = clustering_strategy
    clst_dict['clustering_solution_method'] = clustering_solution_method
    clst_dict['phase_nclusters'] = phase_nclusters
    clst_dict['voxels_clstlbl_flat'] = voxels_clstlbl_flat
    # Return clustering dictionary
    return clst_dict
#
#                                                                     Validation (temporary)
# ==========================================================================================
if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['packageRegularGrid()','packageRGClustering()']
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set functions arguments
    problem_type = 1
    if problem_type == 1:
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
                            'debug/FFT_Homogenization_Method/RVE_2D_2Phases_5x5.rgmsh.npy'
        rve_dims = [1.0,1.0]
    else:
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
                            'debug/FFT_Homogenization_Method/RVE_3D_2Phases_5x5x5.rgmsh.npy'
        rve_dims = [1.0,1.0,1.0]
    n_material_phases = 2
    import readInputData as rid
    n_dim, _ = rid.setProblemTypeParameters(problem_type)
    # Call function
    rg_dict = packageRegularGrid(discret_file_path,rve_dims,n_material_phases,n_dim)
    # Display validation
    print('\nrve_dims:')
    print(rg_dict['rve_dims'])
    print('\nregular_grid:')
    print(rg_dict['regular_grid'])
    print('\nn_voxels_dims:')
    print(rg_dict['n_voxels_dims'])
    print('\nregular_grid_flat:')
    print(rg_dict['regular_grid_flat'])
    print('\nvoxels_idx_flat:')
    print(rg_dict['voxels_idx_flat'])
    print('\nphase_voxel_flatidx:')
    print(rg_dict['phase_voxel_flatidx'])
    # Set functions arguments
    clustering_method = 1
    clustering_strategy = 1
    clustering_solution_method = 1
    phase_nclusters = {'1':10,'2':20}
    # Call function
    clst_dict = packageRGClustering(clustering_method,clustering_strategy,\
                        clustering_solution_method,phase_nclusters,rg_dict['n_voxels_dims'])
    # Display validation
    print('\nclustering_method: ', clustering_method)
    print('\nclustering_strategy: ', clustering_strategy)
    print('\nclustering_solution_method: ', clustering_solution_method)
    print('\nphase_nclusters:')
    print(clst_dict['phase_nclusters'])
    print('\nvoxels_clstlbl_flat:')
    print(clst_dict['voxels_clstlbl_flat'])
    # Display validation footer
    print('\n' + 92*'-' + '\n')
