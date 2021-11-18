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
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
#
#                                                                          Package functions
# ==========================================================================================
# Package directories and paths
def packdirpaths(input_file_name, input_file_path, input_file_dir, problem_name,
                 problem_dir, offline_stage_dir, postprocess_dir, crve_file_path,
                 hres_file_path, refm_file_path, adapt_file_path):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # input_file_name             Name of the user input data file                  str
    # input_file_path             Path of the user input data file                  str
    # input_file_dir              Directory of the user input data file             str
    # problem_name                Name of problem under analysis                    str
    # problem_dir                 Directory of the problem results                  str
    # offline_stage_dir           Directory of the offline stage associated files   str
    # crve_file_path              Path of the .crve file                            str
    # hres_file_path              Path of the .hres file                            str
    # refm_file_path              Path of the .refm file                            str
    # adapt_file_path             Path of the .adapt file                           str
    # postprocess_dir             Directory of the post-processing files            str
    # cbsvar_file_path            Path of the .voxout file                          str
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
    dirs_dict['crve_file_path'] = crve_file_path
    dirs_dict['hres_file_path'] = hres_file_path
    dirs_dict['refm_file_path'] = refm_file_path
    dirs_dict['adapt_file_path'] = adapt_file_path
    dirs_dict['postprocess_dir'] = postprocess_dir
    dirs_dict['voxout_file_path'] = None
    # Return
    return dirs_dict
# ------------------------------------------------------------------------------------------
# Package problem general data
def packproblem(strain_formulation, problem_type, n_dim, comp_order_sym, comp_order_nsym):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # strain_formulation          Strain formulation                                str
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
def packmaterialphases(material_phases, material_phases_data, material_phases_properties,
                       material_phases_vf):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # n_material_phases           Number of material phases                         int
    # material_phases             Material phases ids                               list
    # material_phases_data        Material phases constitutive models data          dict
    #                             key: material phase id (str)
    # material_phases_properties  Material phases properties                        dict
    #                             key: material phase id (str)
    # material_phases_vf          Material phases volume fractions                  dict
    #                             key: material phase id (str)
    #
    # Initialize material phases dictionary
    mat_dict = dict()
    # Build material phases dictionary
    mat_dict['n_material_phases'] = len(material_phases)
    mat_dict['material_phases'] = material_phases
    mat_dict['material_phases_data'] = material_phases_data
    mat_dict['material_phases_properties'] = material_phases_properties
    mat_dict['material_phases_vf'] = material_phases_vf
    # Return
    return mat_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the macroscale loading
def packmacroscaleloading(mac_load_type, mac_load, mac_load_presctype, mac_load_increm,
                          is_solution_rewinding, rewind_state_criterion=None,
                          rewinding_criterion=None, max_n_rewinds=None):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # mac_load_type               Macroscale loading constraint type                int
    # mac_load                    Macroscale loading constraint values              dict
    #                             key: 'strain' and/or 'stress'
    # mac_load_presctype          Macroscale loading component type                 ndarray
    # mac_load_increm             Macroscale loading subpaths incrementation        dict
    #                             key: loading subpath (str)
    # is_solution_rewinding       Analysis rewinding flag                           bool
    # rewind_state_criterion      Rewind state criterion and parameter              tuple
    # rewinding_criterion         Rewinding criterion and parameter                 tuple
    # max_n_rewinds               Maximum number of solution rewinds                int
    #
    # Initialize macroscale loading dictionary
    macload_dict = dict()
    # Build macroscale loading dictionary
    macload_dict['mac_load_type'] = mac_load_type
    macload_dict['mac_load'] = mac_load
    macload_dict['mac_load_presctype'] = mac_load_presctype
    macload_dict['mac_load_increm'] = mac_load_increm
    macload_dict['is_solution_rewinding'] = is_solution_rewinding
    if is_solution_rewinding:
        macload_dict['rewind_state_criterion'] = rewind_state_criterion
        macload_dict['rewinding_criterion'] = rewinding_criterion
        macload_dict['max_n_rewinds'] = max_n_rewinds
    # Return
    return macload_dict
# ------------------------------------------------------------------------------------------
# Package data associated to a regular grid of pixels/voxels
def packregulargrid(discret_file_path, regular_grid, rve_dims, problem_dict):
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
    #
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Set number of pixels/voxels in each dimension
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    n_voxels = np.prod(n_voxels_dims)
    # Get material phases present in the microstructure
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
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
    # Return
    return rg_dict
# ------------------------------------------------------------------------------------------
# Package data associated to the clustering on a regular grid of pixels/voxels
def packrgclustering(clustering_solution_method, standardization_method, links_data,
                     phase_n_clusters, rg_dict, clustering_type, base_clustering_scheme,
                     adaptive_clustering_scheme, adapt_criterion_data, adaptivity_type,
                     adaptivity_control_feature, clust_adapt_freq, is_clust_adapt_output,
                     is_store_final_clustering):
    #
    # Object                       Meaning                                         Type
    # ------------------------------------------------------------------------------------
    # phase_n_clusters             Number of clusters of each material phase       dict
    #                              key: material phase id (str)
    # phase_clusters               Clusters associated to each material phase      dict
    #                              key: material phase id (str)
    # voxels_clusters              Regular grid of pixels/voxels with the          ndarray
    #                              cluster labels
    # clusters_vf                  Clusters volume fraction                        dict
    #                              key: material cluster label (str)
    # links_data                   Links related variables                         dict
    #                              key: links parameter (str)
    # clustering_type              Clustering type, {'static', 'adaptive'}         dict
    #                              key: material phase id (str)
    # base_clustering_scheme       Base clustering scheme                          dict
    #                              key: material phase id (str)
    # adaptive_clustering_scheme   Adaptive clustering scheme                      dict
    #                              key: material phase id (str)
    # adapt_criterion_data         Adaptivity criterion parameters                 dict
    #                              key: material phase id (str)
    # adaptivity_type              Adaptivity type parameters                      dict
    #                              key: material phase id (str)
    # adaptivity_control_feature   Adaptivity control feature                      dict
    #                              key: material phase id (str)
    # clust_adapt_freq             Clustering adaptivity frequency                 dict
    #                              key: material phase id (str)
    # is_clust_adapt_output        Adaptivity output                               bool
    # is_store_final_clustering    Final clustering state storage                  bool
    #
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Initialize array with voxels cluster labels
    voxels_clusters = np.full(n_voxels_dims, -1, dtype=int)
    # Initialize dictionary with each material phase clusters
    phase_clusters = dict()
    # Initialize dictionary with clusters volume fractions
    clusters_vf = dict()
    # Initialize clustering dictionary
    clst_dict = dict()
    # Build clustering dictionary
    clst_dict['clustering_solution_method'] = clustering_solution_method
    clst_dict['standardization_method'] = standardization_method
    if len(links_data.keys()) > 0:
        clst_dict['links_data'] = links_data
    clst_dict['phase_n_clusters'] = phase_n_clusters
    clst_dict['phase_clusters'] = phase_clusters
    clst_dict['voxels_clusters'] = voxels_clusters
    clst_dict['clusters_vf'] = clusters_vf
    clst_dict['clustering_type'] = clustering_type
    clst_dict['base_clustering_scheme'] = base_clustering_scheme
    clst_dict['adaptive_clustering_scheme'] = adaptive_clustering_scheme
    clst_dict['adapt_criterion_data'] = adapt_criterion_data
    clst_dict['adaptivity_type'] = adaptivity_type
    clst_dict['adaptivity_control_feature'] = adaptivity_control_feature
    clst_dict['clust_adapt_freq'] = clust_adapt_freq
    clst_dict['is_clust_adapt_output'] = is_clust_adapt_output
    clst_dict['is_store_final_clustering'] = is_store_final_clustering
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
def packalgparam(max_n_iterations, conv_tol, max_subinc_level, max_cinc_cuts,
                 su_max_n_iterations, su_conv_tol):
    #
    # Object                      Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # max_n_iterations     Maximum number of iterations to achieve
    #                      equilibrium convergence                                  int
    # conv_tol             Equilibrium convergence tolerance                        float
    # max_subinc_level     Maximum macroscale loading subincrementation level       int
    # max_cinc_cuts        Maximum number of consecutive macroscale load
    #                      increment cuts                                           int
    # su_max_n_iterations  Maximum number of iterations to achieve
    #                      the state update convergence                             int
    # su_conv_tol          State update convergence tolerance                       float
    #
    # Initialize algorithmic parameters dictionary
    algpar_dict = dict()
    # Build algorithmic parameters dictionary
    algpar_dict['max_n_iterations'] = max_n_iterations
    algpar_dict['conv_tol'] = conv_tol
    algpar_dict['max_subinc_level'] = max_subinc_level
    algpar_dict['max_cinc_cuts'] = max_cinc_cuts
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
    # is_VTK_output        VTK output flag                                          bool
    # vtk_format           VTK file format                                          str
    # vtk_inc_div          VTK increment output divisor                             int
    # vtk_vars             VTK state variables output                               str
    # vtk_precision        VTK file precision                                       str
    # vtk_byte_order       VTK file byte order                                      str
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
        if sys.byteorder == 'little':
            vtk_dict['vtk_byte_order'] = 'LittleEndian'
        else:
            vtk_dict['vtk_byte_order'] = 'BigEndian'
    # Return
    return vtk_dict
# ------------------------------------------------------------------------------------------
# Package data associated to general output files
def packoutputfiles(is_voxels_output, *args):
    #
    # Object                 Meaning                                           Type
    # -------------------------------------------------------------------------------------
    # is_voxels_output       Voxels material-related quantities                bool
    #
    # Initialize output dictionary
    output_dict = dict()
    # Build output dictionary
    output_dict['is_voxels_output'] = is_voxels_output
    # Return
    return output_dict
