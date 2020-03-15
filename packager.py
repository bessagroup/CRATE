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
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Generate efficient iterators
import itertools as it
# Display messages
import info
# Display errors, warnings and built-in exceptions
import errors
#
#                                                                          Package functions
# ==========================================================================================
# Package directories and paths
def packageDirsPaths(input_file_name,input_file_path,input_file_dir,problem_name,
              problem_dir,offline_stage_dir,postprocess_dir,cluster_file_path,cit_file_path,
                                                                            hres_file_path):
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
    dirs_dict['cit_file_path'] = cit_file_path
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
def packageMaterialPhases(n_material_phases,material_phases_models,material_properties):
    # Initialize list with existent material phases in the microstructure
    material_phases = list()
    # Initialize dictionary with existent material phases volume fraction
    material_phases_f = dict()
    # Initialize material phases dictionary
    mat_dict = dict()
    # Build material phases dictionary
    mat_dict['n_material_phases'] = n_material_phases
    mat_dict['material_phases_models'] = material_phases_models
    mat_dict['material_properties'] = material_properties
    mat_dict['material_phases'] = material_phases
    mat_dict['material_phases_f'] = material_phases_f
    # Return
    return mat_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the macroscale loading
def packageMacroscaleLoading(mac_load_type,mac_load,mac_load_presctype,n_load_increments):
    # Initialize macroscale loading dictionary
    macload_dict = dict()
    # Build macroscale loading dictionary
    macload_dict['mac_load_type'] = mac_load_type
    macload_dict['mac_load'] = mac_load
    macload_dict['mac_load_presctype'] = mac_load_presctype
    macload_dict['n_load_increments'] = n_load_increments
    # Return
    return macload_dict
# ------------------------------------------------------------------------------------------
# Package data associated to a regular grid of pixels/voxels
def packageRegularGrid(discret_file_path,rve_dims,mat_dict,problem_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Get material data
    material_properties = mat_dict['material_properties']
    # Read the spatial discretization file (regular grid of pixels/voxels)
    info.displayInfo('5','Reading discretization file...')
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
        rg_file_name = \
                ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[-2])[-2]
    else:
        regular_grid = np.loadtxt(discret_file_path)
        rg_file_name = ntpath.splitext(ntpath.basename(discret_file_path))[-2]
    # Check validity of regular grid of pixels/voxels
    if len(regular_grid.shape) not in [2,3]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00042',location.filename,location.lineno+1)
    elif np.any([str(phase) not in material_properties.keys() \
                                                     for phase in np.unique(regular_grid)]):
        idf_phases = list(np.sort([int(key) for key in material_properties.keys()]))
        rg_phases = list(np.unique(regular_grid))
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00043',location.filename,location.lineno+1,idf_phases,
                                                                                  rg_phases)
    # Set number of pixels/voxels in each dimension
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    n_voxels = np.prod(n_voxels_dims)
    # Set material phases that are actually present in the microstructure
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    mat_dict['material_phases'] = material_phases
    mat_dict['n_material_phases'] = len(mat_dict['material_phases'])
    # Display warning if all the material phases that have been specified in the input data
    # file are not present in the microstructure
    if any([phase not in material_phases for phase in material_properties.keys()]):
        idf_phases = list(np.sort([int(key) for key in material_properties.keys()]))
        rg_phases = list(np.unique(regular_grid))
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayWarning('W00002',location.filename,location.lineno+1,idf_phases,                                                                                  rg_phases)
    # Compute voxel volume
    voxel_vol = np.prod([float(rve_dims[i])/n_voxels_dims[i] for i in range(len(rve_dims))])
    # Compute RVE volume
    rve_vol = np.prod(rve_dims)
    # Compute volume fraction associated to each material phase existent in the
    # microstructure
    for phase in material_phases:
        mat_dict['material_phases_f'][phase] = \
                                      (np.sum(regular_grid == int(phase))*voxel_vol)/rve_vol
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
    for mat_phase in material_phases:
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
    rg_dict['rg_file_name'] = rg_file_name
    # Return
    return rg_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the clustering on a regular grid of pixels/voxels
def packageRGClustering(clustering_method,clustering_strategy,clustering_solution_method,
                                                       Links_dict,phase_n_clusters,rg_dict):
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Initialize array with voxels cluster labels
    voxels_clusters = np.full(n_voxels_dims,-1,dtype=int)
    # Initialize dictionary with each material phase clusters
    phase_clusters = dict()
    # Initialize dictionary with clusters volume fractions
    clusters_f = dict()
    # Initialize clustering dictionary
    clst_dict = dict()
    # Build clustering dictionary
    clst_dict['clustering_method'] = clustering_method
    clst_dict['clustering_strategy'] = clustering_strategy
    clst_dict['clustering_solution_method'] = clustering_solution_method
    if clustering_solution_method == 2:
        clst_dict['Links_dict'] = Links_dict
    clst_dict['phase_n_clusters'] = phase_n_clusters
    clst_dict['phase_clusters'] = phase_clusters
    clst_dict['voxels_clusters'] = voxels_clusters
    clst_dict['clusters_f'] = clusters_f
    # Return
    return clst_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the self-consistent scheme
def packageSCS(self_consistent_scheme,scs_max_n_iterations,scs_conv_tol):
    # Initialize self-consistent scheme dictionary
    scs_dict = dict()
    # Build self-consistent scheme dictionary
    scs_dict['self_consistent_scheme'] = self_consistent_scheme
    scs_dict['scs_max_n_iterations'] = scs_max_n_iterations
    scs_dict['scs_conv_tol'] = scs_conv_tol
    # Return
    return scs_dict
# ------------------------------------------------------------------------------------------
# Package data associated to algorithmic parameters related to the solution procedure
def packageAlgorithmicParameters(max_n_iterations,conv_tol,max_subincrem_level,
                                                           su_max_n_iterations,su_conv_tol):
    # Initialize algorithmic parameters dictionary
    algpar_dict = dict()
    # Build algorithmic parameters dictionary
    algpar_dict['max_n_iterations'] = max_n_iterations
    algpar_dict['conv_tol'] = conv_tol
    algpar_dict['max_subincrem_level'] = max_subincrem_level
    algpar_dict['su_max_n_iterations'] = su_max_n_iterations
    algpar_dict['su_conv_tol'] = su_conv_tol
    # Return
    return algpar_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the VTK output
def packageVTK(is_VTK_output,*args):
    # Initialize VTK dictionary
    vtk_dict = dict()
    # Build VTK dictionary
    vtk_dict['is_VTK_output'] = is_VTK_output
    if is_VTK_output:
        # vtk_format = args[0]
        vtk_inc_div = args[1]
        vtk_vars = args[2]
        vtk_dict['vtk_format'] = 'ascii'   # Change to vtk_format when binary is implemented
        vtk_dict['vtk_inc_div'] = vtk_inc_div
        vtk_dict['vtk_vars'] = vtk_vars
        vtk_dict['vtk_precision'] = 'SinglePrecision'
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
    phase_n_clusters = {'1':10,'2':20}
    # Call function
    clst_dict = packageRGClustering(clustering_method,clustering_strategy,\
                       clustering_solution_method,phase_n_clusters,rg_dict['n_voxels_dims'])
    # Display validation
    print('\nclustering_method: ', clustering_method)
    print('\nclustering_strategy: ', clustering_strategy)
    print('\nclustering_solution_method: ', clustering_solution_method)
    print('\nphase_nclusters:')
    print(clst_dict['phase_n_clusters'])
    print('\nvoxels_clusters:')
    print(clst_dict['voxels_clusters'])
    # Display validation footer
    print('\n' + 92*'-' + '\n')
