#
# Packager Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the packaging of objects in specific dictionaries.
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
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
#
#                                                                          Package functions
# ==========================================================================================
# Package directories and paths
def packdirpaths(input_file_name, input_file_path, input_file_dir, problem_name,
                 problem_dir, offline_stage_dir, postprocess_dir, cluster_file_path,
                 cit_file_path, hres_file_path):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # input_file_name             Name of the user input data file                  str
    # input_file_path             Path of the user input data file                  str
    # input_file_dir              Directory of the user input data file             str
    # problem_name                Name of problem under analysis                    str
    # problem_dir                 Directory of the problem results                  str
    # offline_stage_dir           Directory of the offline stage associated files   str
    # cluster_file_path           Path of the .clusters file                        str
    # cit_file_path               Path of the .cit file                             str
    # hres_file_path              Path of the .hres file                            str
    # postprocess_dir             Directory of the post-processing files            str
    #
    # Note: The meaning of the previous directories and files is detailed in the module
    #       fileoperations (see function setproblemdirs documentation)
    #
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
def packproblem(strain_formulation, problem_type, n_dim, comp_order_sym, comp_order_nsym):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # strain_formulation          Strain formulation                                int
    # problem_type                Problem type                                      int
    # n_dim                       Number of problem dimensions                      int
    # comp_order_sym              Symmetric strain/stress components order          list
    # comp_order_nsym             Nonsymmetric strain/stress components order       list
    #
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
def packmaterialphases(n_material_phases, material_phases_models, material_properties):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # n_material_phases           Number of material phases                         int
    # material_phases_models      Material phases constitutive models               dict
    #                             key: material phase id (str)
    # material_properties         Material phases properties                        dict
    #                             key: material phase id (str)
    # material_phases             Material phases ids                               list
    # material_phases_f           Material phases volume fractions                  dict
    #                             key: material phase id (str)
    #
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
def packmacroscaleloading(mac_load_type, mac_load, mac_load_presctype, mac_load_increm):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # mac_load_type               Macroscale loading constraint type                int
    # mac_load                    Macroscale loading constraint values              dict
    #                             key: 'strain' and/or 'stress'
    # mac_load_presctype          Macroscale loading component type                 ndarray
    # mac_load_increm             Macroscale loading subpaths incrementation        dict
    #                             key: loading subpath (str)
    #
    # Initialize macroscale loading dictionary
    macload_dict = dict()
    # Build macroscale loading dictionary
    macload_dict['mac_load_type'] = mac_load_type
    macload_dict['mac_load'] = mac_load
    macload_dict['mac_load_presctype'] = mac_load_presctype
    macload_dict['mac_load_increm'] = mac_load_increm
    # Return
    return macload_dict
# ------------------------------------------------------------------------------------------
# Package data associated to a regular grid of pixels/voxels
def packregulargrid(discret_file_path, rve_dims, mat_dict, problem_dict):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # rve_dims                    Dimensions of the RVE                             list
    # regular_grid                Regular grid of pixels/voxels                     ndarray
    # n_voxels_dims               Number of voxels on each dimension                list
    # regular_grid_flat           Flattened regular grid of pixels/voxels           ndarray
    # voxels_idx_flat             Flattened voxels indexes                          ndarray
    # phase_voxel_flatidx         Flattened voxels indexes associated to each
    #                             material phase                                    dict
    #                             key: material phase id (str)
    # rg_file_name                Name of regular grid discretization file          str
    #
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Get material data
    material_properties = mat_dict['material_properties']
    # Read the spatial discretization file (regular grid of pixels/voxels)
    info.displayinfo('5', 'Reading discretization file...')
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
        rg_file_name = ntpath.splitext(ntpath.splitext(
            ntpath.basename(discret_file_path))[-2])[-2]
    else:
        regular_grid = np.loadtxt(discret_file_path)
        rg_file_name = ntpath.splitext(ntpath.basename(discret_file_path))[-2]
    # Check validity of regular grid of pixels/voxels
    if len(regular_grid.shape) not in [2, 3]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00042', location.filename, location.lineno + 1)
    elif np.any([str(phase) not in material_properties.keys() \
            for phase in np.unique(regular_grid)]):
        idf_phases = list(np.sort([int(key) for key in material_properties.keys()]))
        rg_phases = list(np.unique(regular_grid))
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00043', location.filename, location.lineno + 1, idf_phases,
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
        errors.displaywarning('W00002', location.filename, location.lineno + 1, idf_phases,
                              rg_phases)
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
    voxels_idx_flat = [np.unravel_index(i, shape) for i in range(n_voxels)]
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
def packrgclustering(clustering_method, clustering_strategy, clustering_solution_method,
                     links_dict, phase_n_clusters, rg_dict):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # clustering_method           Clustering algorithm                              int
    # clustering_strategy         Clustering strategy                               int
    # clustering_solution_method  Clustering solution method                        list
    # phase_n_clusters            Number of clusters of each material phase         dict
    #                             key: material phase id (str)
    # phase_clusters              Clusters associated to each material phase        dict
    #                             key: material phase id (str)
    # voxels_clusters             Regular grid of pixels/voxels with the            ndarray
    #                             cluster labels
    # clusters_f                  Clusters volume fraction                          dict
    #                             key: material cluster label (str)
    # links_dict                  Links related variables                           dict
    #
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Initialize array with voxels cluster labels
    voxels_clusters = np.full(n_voxels_dims, -1, dtype=int)
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
    if len(links_dict.keys()) > 0:
        clst_dict['links_dict'] = links_dict
    clst_dict['phase_n_clusters'] = phase_n_clusters
    clst_dict['phase_clusters'] = phase_clusters
    clst_dict['voxels_clusters'] = voxels_clusters
    clst_dict['clusters_f'] = clusters_f
    # Return
    return clst_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the self-consistent scheme
def packagescs(self_consistent_scheme, scs_max_n_iterations, scs_conv_tol):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # self_consistent_scheme  Self-consistent scheme                                int
    # scs_max_n_iterations    Maximum number of self-consistent
    #                         scheme iterations to convergence                      int
    # scs_conv_tol            Self-consistent scheme convergence
    #                         tolerance                                             float
    #
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
def packalgparam(max_n_iterations, conv_tol, max_subincrem_level, su_max_n_iterations,
                 su_conv_tol):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # max_n_iterations     Maximum number of iterations to achieve
    #                      equilibrium convergence                                  int
    # conv_tol             Equilibrium convergence tolerance                        float
    # max_subincrem_level  Maximum subincrementation level                          int
    # su_max_n_iterations  Maximum number of iterations to achieve
    #                      the state update convergence                             int
    # su_conv_tol          State update convergence tolerance                       float
    #
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
def packvtk(is_VTK_output, *args):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # vtk_format           VTK file format                                          str
    # vtk_inc_div          VTK increment output divisor                             int
    # vtk_vars             VTK state variables output                               str
    # vtk_precision        VTK file precision                                       str
    #
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
