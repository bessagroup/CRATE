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
# Package directories and paths
def packageDirsPaths(input_file_name,input_file_path,input_file_dir,problem_name,
            problem_dir,offline_stage_dir,postprocess_dir,cluster_file_path,hres_file_path):
    # Initialize directories and paths dictionary
    dirs_dict = dict()
    # Build directories and paths dictionary
    dirs_dict['input_file_name'] = input_file_name
    dirs_dict['input_file_path'] = input_file_path
    dirs_dict['input_file_dir'] = input_file_dir
    dirs_dict['problem_name'] = problem_name
    dirs_dict['problem_dir'] = problem_dir
    dirs_dict['offline_stage_dir'] = offline_stage_dir
    dirs_dict['cluster_file_path'] = cluster_file_path
    dirs_dict['hres_file_path'] = hres_file_path
    dirs_dict['postprocess_dir'] = postprocess_dir
    # Return
    return dirs_dict
# ------------------------------------------------------------------------------------------
# Package problem general data
def packageProblem(strain_formulation,problem_type,n_dim,comp_order_sym,comp_order_nsym):
    # Initialize problem dictionary
    problem_dict = dict()
    # Build problem dictionary
    problem_dict['strain_formulation'] = strain_formulation
    problem_dict['problem_type'] = problem_type
    problem_dict['n_dim'] = n_dim
    problem_dict['comp_order_sym'] = comp_order_sym
    problem_dict['comp_order_nsym'] = comp_order_nsym
    # Return
    return problem_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the material phases
def packageMaterialPhases(n_material_phases,material_properties):
    # Initialize material phases dictionary
    mat_dict = dict()
    # Build material phases dictionary
    mat_dict['n_material_phases'] = n_material_phases
    mat_dict['material_properties'] = material_properties
    # Return
    return mat_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the macroscale loading
def packageMacroscaleLoading(mac_load_type,mac_load,mac_load_typeidxs):
    # Initialize macroscale loading dictionary
    macload_dict = dict()
    # Build macroscale loading dictionary
    macload_dict['mac_load_type'] = mac_load_type
    macload_dict['mac_load'] = mac_load
    macload_dict['mac_load_typeidxs'] = mac_load_typeidxs
    # Return
    return macload_dict
# ------------------------------------------------------------------------------------------
# Package data associated to a regular grid of pixels/voxels
def packageRegularGrid(discret_file_path,rve_dims,mat_dict,problem_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Get material data
    n_material_phases = mat_dict['n_material_phases']
    material_properties = mat_dict['material_properties']
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
    phase_voxel_flatidx = dict()
    for mat_phase in material_properties.keys():
        is_phase_list = regular_grid.flatten() == int(mat_phase)
        phase_voxel_flatidx[mat_phase] = \
                                  list(it.compress(range(len(is_phase_list)),is_phase_list))
    # Initialize regular grid dictionary
    rg_dict = dict()
    # Build regular grid dictionary
    rg_dict['rve_dims'] = rve_dims
    rg_dict['regular_grid'] = regular_grid
    rg_dict['n_voxels_dims'] = n_voxels_dims
    rg_dict['regular_grid_flat'] = regular_grid_flat
    rg_dict['voxels_idx_flat'] = voxels_idx_flat
    rg_dict['phase_voxel_flatidx'] = phase_voxel_flatidx
    # Return
    return rg_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the clustering on a regular grid of pixels/voxels
def packageRGClustering(clustering_method,clustering_strategy,clustering_solution_method,\
                                                                   phase_nclusters,rg_dict):
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Initialize array with voxels cluster labels
    voxels_clusters = np.full(n_voxels_dims,-1,dtype=int)
    # Initialize array with each material phase clusters
    phase_clusters = dict()
    # Initialize array with clusters volume fractions
    clusters_f = dict()
    # Initialize clustering dictionary
    clst_dict = dict()
    # Build clustering dictionary
    clst_dict['clustering_method'] = clustering_method
    clst_dict['clustering_strategy'] = clustering_strategy
    clst_dict['clustering_solution_method'] = clustering_solution_method
    clst_dict['phase_nclusters'] = phase_nclusters
    clst_dict['phase_clusters'] = phase_clusters
    clst_dict['voxels_clusters'] = voxels_clusters
    clst_dict['clusters_f'] = clusters_f
    # Return
    return clst_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the VTK output
def packageVTK():
    # Initialize VTK dictionary
    vtk_dict = dict()
    # Build VTK dictionary
    vtk_dict['format'] = 'ascii'
    vtk_dict['precision'] = 'SinglePrecision'
    # Return
    return vtk_dict
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
    n_dim,_,_ = rid.setProblemTypeParameters(problem_type)
    problem_dict = dict()
    problem_dict['n_dim'] = n_dim
    # Call function
    rg_dict = packageRegularGrid(discret_file_path,rve_dims,n_material_phases,problem_dict)
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
    print('\nvoxels_clusters:')
    print(clst_dict['voxels_clusters'])
    # Display validation footer
    print('\n' + 92*'-' + '\n')
