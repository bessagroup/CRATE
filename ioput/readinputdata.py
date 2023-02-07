"""Read input data file and store data.

This module includes a function that reads the input data file and stores all
the relevant data in suitable containers.

Functions
---------
read_input_data_file
    Read input data file and store data in suitable containers.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
import linecache
import inspect
import ntpath
# Third-party
import numpy as np
# Local
import ioput.info as info
import ioput.errors as errors
import ioput.fileoperations as filop
import ioput.packager as packager
import ioput.readprocedures as rproc
import tensor.matrixoperations as mop
from clustering.clusteringdata import get_available_clustering_features
from material.materialmodeling import MaterialState
from links.inputdatareader import read_links_input_parameters
from links.stateupdate import LinksConstitutiveModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira',]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def read_input_data_file(input_file, dirs_dict):
    """Read input data file and store data in suitable containers.

    Parameters
    ----------
    input_file : file
        Input data file.
    dirs_dict : dict
        Container of output directories and files paths.

    Returns
    -------
    problem_dict : dict
        Container of data associated with the problem type.
    mat_dict : dict
        Container of data associated with the material phases.
    macload_dict : dict
        Container of data associated with the macroscale loading.
    rg_dict : dict
        Container of data associated with the spatial discretization in a
        regular grid of voxels.
    clst_dict : dict
        Container of data associated with the clustering-based domain
        decomposition.
    scs_dict : dict
        Container of data associated with the self-consistent scheme.
    algpar_dict : dict
        Container of data associated with algorithmic parameters underlying the
        equilibrium problem solution procedure.
    vtk_dict : dict
        Container of data associated with the VTK output files.
    output_dict : dict
        Container of data associated with general output files.
    material_state : MaterialState
        CRVE material constitutive state.
    """
    # Get input data file path and output directories paths
    input_file_name = dirs_dict['input_file_name']
    input_file_path = dirs_dict['input_file_path']
    problem_dir = dirs_dict['problem_dir']
    offline_stage_dir = dirs_dict['offline_stage_dir']
    postprocess_dir = dirs_dict['postprocess_dir']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read strain formulation (mandatory)
    keyword = 'Strain_Formulation'
    max_val = 2
    strain_formulation_code = rproc.readtypeAkeyword(
        input_file, input_file_path, keyword, max_val)
    if strain_formulation_code == 1:
        strain_formulation = 'infinitesimal'
    elif strain_formulation_code == 2:
        strain_formulation = 'finite'
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read problem type (mandatory)
    keyword = 'Problem_Type'
    max_val = 4
    problem_type = rproc.readtypeAkeyword(
        input_file, input_file_path, keyword, max_val)
    # Get parameters dependent on the problem type
    n_dim, comp_order_sym, comp_order_nsym = \
        mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read RVE dimensions (mandatory)
    keyword = 'RVE_Dimensions'
    rve_dims = rproc.readrvedimensions(
        input_file, input_file_path, keyword, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of material phases, constitutive models and associated
    # properties (mandatory)
    keyword = 'Material_Phases'
    n_material_phases, material_phases_data, material_phases_properties = \
        rproc.read_material_properties(input_file, input_file_path, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the Links python binary absolute path if at least one material phase
    # has the associated constitutive model source set as Links
    is_links_python_bin = False
    for mat_phase in material_phases_properties.keys():
        if material_phases_data[mat_phase]['source'] == 'links':
            # Read the Links python binary absolute path
            keyword = 'Links_Python_bin'
            line_number = rproc.searchkeywordline(input_file, keyword) + 1
            Links_python_bin_path = \
                linecache.getline(input_file_path, line_number).strip()
            if not os.path.isabs(Links_python_bin_path):
                raise RuntimeError('Input data error: Links python binary '
                                   'path must be absolute.')
            elif not os.path.isfile(Links_python_bin_path):
                raise RuntimeError('Input data error: Links python binary '
                                   'path has not been found.')
            else:
                is_links_python_bin = True
                break
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read macroscale loading (mandatory)
    keyword = 'Macroscale_Loading'
    max_val = 3
    mac_load_type = rproc.readtypeAkeyword(
        input_file, input_file_path, keyword, max_val)
    mac_load, mac_load_presctype = rproc.readmacroscaleloading(
        input_file, input_file_path, mac_load_type, strain_formulation, n_dim,
        comp_order_nsym)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read analysis rewinding procedure (optional)
    keyword = 'Analysis_Rewinding'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_solution_rewinding = True
        # Read rewind state criterion
        keyword = 'Analysis_Rewind_State_Criterion'
        rewind_state_criterion = rproc.read_rewind_state_parameters(
            input_file, input_file_path, keyword)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read rewinding criterion
        keyword = 'Analysis_Rewinding_Criterion'
        rewinding_criterion = rproc.read_rewinding_criterion_parameters(
            input_file, input_file_path, keyword)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Read maximum number of rewinds
        keyword = 'Max_Number_of_Rewinds'
        is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
        if is_found:
            max_val = '~'
            max_n_rewinds = rproc.readtypeAkeyword(
                input_file, input_file_path, keyword, max_val)
        else:
            max_n_rewinds = 1
    else:
        is_solution_rewinding = False
        rewind_state_criterion = None
        rewinding_criterion = None
        max_n_rewinds = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self-consistent scheme (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Self_Consistent_Scheme'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = 3
        self_consistent_scheme_id = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        self_consistent_scheme_id = 1
    # Set self-consistent scheme according to the identifier
    if self_consistent_scheme_id == 1:
        self_consistent_scheme = 'regression'
    elif self_consistent_scheme_id == 3:
        self_consistent_scheme = 'self_consistent_optimization'
    else:
        raise RuntimeError('Unknown self-consistent scheme.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme maximum number of iterations (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'SCS_Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        scs_max_n_iterations = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        scs_max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme convergence tolerance (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'SCS_Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        scs_conv_tol = rproc.readtypeBkeyword(
            input_file, input_file_path, keyword)
    else:
        scs_conv_tol = 1e-4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get available clustering features
    clustering_features = list(get_available_clustering_features(
        strain_formulation, problem_type).keys())
    # Read cluster analysis scheme (mandatory)
    keyword = 'Cluster_Analysis_Scheme'
    clustering_type, base_clustering_scheme, adaptive_clustering_scheme, \
        adapt_criterion_data, adaptivity_type, adaptivity_control_feature = \
            rproc.read_cluster_analysis_scheme(
                input_file, input_file_path, keyword,
                material_phases_properties.keys(), clustering_features)
    # Get adaptive material phases
    adapt_material_phases = [x for x in clustering_type.keys()
                             if clustering_type[x] == 'adaptive']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of cluster associated with each material phase (mandatory)
    keyword = 'Number_of_Clusters'
    phase_n_clusters = rproc.readphaseclustering(
        input_file, input_file_path, keyword, n_material_phases,
        material_phases_properties)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering adaptivity frequency (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Clustering_Adaptivity_Frequency'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        clust_adapt_freq = rproc.read_adaptivity_frequency(
            input_file, input_file_path, keyword, adapt_material_phases)
    else:
        clust_adapt_freq = {mat_phase: 1
                            for mat_phase in adapt_material_phases}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering solution method (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Clustering_Solution_Method'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = 2
        clustering_solution_method = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        clustering_solution_method = 1
    # Read clustering solution method parameters
    links_data = {}
    if clustering_solution_method == 2:
        # Check if all the material phases have the associated constitutive
        # model source set as Links
        for mat_phase in material_phases_data.keys():
            if material_phases_data[mat_phase]['source'] != 'links':
                raise RuntimeError(
                    'Input data error: To perform the offline-stage DNS '
                    'computations with Links, all material constitutive '
                    'models must have the corresponding source.')
        # Build Links parameters dictionary
        links_data = read_links_input_parameters(
            input_file, input_file_path, problem_type)
        # Set directory where Links offline-stage simulation files are stored.
        links_offline_dir =  offline_stage_dir + 'Links' + '/'
        if not os.path.exists(links_offline_dir):
            filop.makedirectory(links_offline_dir)
        links_data['links_offline_dir'] = links_offline_dir
    # If at least one material phase has the associated constitutive model
    # source set as Links, add Links python binary to Links dictionary
    if is_links_python_bin:
        LinksConstitutiveModel.Links_python_bin_path = Links_python_bin_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read macroscale loading incrementation parameters (mandatory)
    keyword_1 = 'Number_of_Load_Increments'
    is_found_1, _ = rproc.searchoptkeywordline(input_file, keyword_1)
    keyword_2 = 'Increment_List'
    is_found_2, _ = rproc.searchoptkeywordline(input_file, keyword_2)
    if not (is_found_1 or is_found_2):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00089', location.filename, location.lineno + 1)
    elif is_found_1 and is_found_2:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00090', location.filename, location.lineno + 1)
    else:
        # Get number of loading subpaths
        n_load_subpaths = mac_load_presctype.shape[1]
        # Read macroscale loading incrementation
        keyword = keyword_1 if is_found_1 else keyword_2
        mac_load_increm = rproc.readmacloadincrem(
            input_file, input_file_path, keyword, n_load_subpaths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum number of iterations to solve each load increment (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        max_n_iterations = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        max_n_iterations = 12
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read convergence tolerance to solve each load increment (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        conv_tol = rproc.readtypeBkeyword(input_file, input_file_path, keyword)
    else:
        conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum level of macroscale loading subincrementation (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Max_SubInc_Level'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        max_subinc_level = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        max_subinc_level = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum number of consecutive macroscale loading increment cuts
    # (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Max_Consecutive_Inc_Cuts'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        max_cinc_cuts = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        max_cinc_cuts = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update maximum number of iterations (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'SU_Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = '~'
        su_max_n_iterations = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        su_max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update convergence tolerance (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'SU_Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        su_conv_tol = rproc.readtypeBkeyword(
            input_file, input_file_path, keyword)
    else:
        su_conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the spatial discretization file absolute path (mandatory)
    keyword = 'Discretization_File'
    valid_exts = ['.rgmsh']
    discret_file_path = rproc.readdiscretizationfilepath(
        input_file, input_file_path, keyword, valid_exts)
    # Copy the spatial discretization file to the problem directory and update
    # the absolute path to the copied file
    try:
        shutil.copy2(discret_file_path,
                     problem_dir + ntpath.basename(discret_file_path))
        discret_file_path = problem_dir + ntpath.basename(discret_file_path)
    except IOError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1,
                                message)
    # Store spatial discretization file absolute path
    dirs_dict['discret_file_path'] = discret_file_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering global matrix data standardization method (optional)
    # If the associated keyword is not found, then a default option is adopted
    keyword = 'Standardization_Method'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max_val = 2
        standardization_method = rproc.readtypeAkeyword(
            input_file, input_file_path, keyword, max_val)
    else:
        standardization_method = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read VTK output options (optional)
    keyword = 'VTK_Output'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_vtk_output = True
        vtk_format, vtk_inc_div, vtk_vars = rproc.readvtkoptions(
            input_file, input_file_path, keyword, keyword_line_number)
        # Create VTK folder in post processing directory
        filop.makedirectory(postprocess_dir + 'VTK/', 'overwrite')
    else:
        is_vtk_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read voxels material-related quantities output options (optional)
    keyword = 'Voxels_Output'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_voxels_output = True
        # Set voxels material-related quantities output file path
        voxout_file_path = postprocess_dir + input_file_name + '.voxout'
        # Store output file path in directories and paths dictionary
        dirs_dict['voxout_file_path'] = voxout_file_path
    else:
        is_voxels_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering adaptivity output (optional)
    keyword = 'Clustering_Adaptivity_Output'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_clust_adapt_output = True
    else:
        is_clust_adapt_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read reference material output (optional)
    keyword = 'Reference_Material_Output'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_ref_material_output = True
    else:
        is_ref_material_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read final clustering state storage option (optional)
    keyword = 'Store_Final_Clustering_State'
    is_found, keyword_line_number = rproc.searchoptkeywordline(
        input_file, keyword)
    if is_found:
        is_store_final_clustering = True
    else:
        is_store_final_clustering = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the spatial discretization file (regular grid of voxels)
    info.displayinfo('5', 'Reading discretization file...')
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
    else:
        regular_grid = np.loadtxt(discret_file_path)
    # Check validity of regular grid of voxels
    if len(regular_grid.shape) not in [2, 3]:
        raise RuntimeError('Input data error: Number of dimensions of regular '
                           'grid of voxels must be either 2 (2D problem) or '
                           '3 (3D problem).')
    elif np.any([str(phase) not in material_phases_properties.keys() \
            for phase in np.unique(regular_grid)]):
        raise RuntimeError('Input data error: At least one material phase '
                           'present in the regular grid of voxels has not '
                           'been specified.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material phases that are actually present in the microstructure
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    # Initialize material phases volume fractions
    material_phases_vf = {}
    # Loop over material phases
    for mat_phase in material_phases:
        # Compute material phase volume fraction
        material_phases_vf[mat_phase] = \
            np.sum(regular_grid == int(mat_phase))/np.prod(regular_grid.size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantiate CRVE material constitutive state
    material_state = MaterialState(strain_formulation, problem_type,
                                   material_phases, material_phases_properties,
                                   material_phases_vf)
    # Loop over material phases
    for mat_phase in material_phases:
        # Get material phase constitutive model keyword and source
        model_keyword = material_phases_data[mat_phase]['keyword']
        model_source = material_phases_data[mat_phase]['source']
        # Initialize material phase constitutive model
        material_state.init_constitutive_model(mat_phase, model_keyword,
                                               model_source)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store problem general data
    info.displayinfo('5', 'Storing problem general data...')
    problem_dict = packager.packproblem(
        strain_formulation, problem_type, n_dim, comp_order_sym,
        comp_order_nsym)
    # Store data associated with the material phases
    info.displayinfo('5', 'Storing material data...')
    mat_dict = packager.packmaterialphases(
        material_phases, material_phases_data, material_phases_properties,
        material_phases_vf)
    # Store data associated with the macroscale loading
    info.displayinfo('5', 'Storing macroscale loading data...')
    macload_dict = packager.packmacroscaleloading(
        mac_load_type, mac_load, mac_load_presctype, mac_load_increm,
        is_solution_rewinding, rewind_state_criterion, rewinding_criterion,
        max_n_rewinds)
    # Store data associated with the spatial discretization file(s)
    info.displayinfo('5', 'Storing regular grid data...')
    rg_dict = packager.packregulargrid(
        discret_file_path, regular_grid, rve_dims, problem_dict)
    # Store data associated with the clustering
    info.displayinfo('5', 'Storing clustering data...')
    clst_dict = packager.packrgclustering(
        clustering_solution_method, standardization_method, links_data,
        phase_n_clusters, rg_dict, clustering_type, base_clustering_scheme,
        adaptive_clustering_scheme, adapt_criterion_data, adaptivity_type,
        adaptivity_control_feature, clust_adapt_freq, is_clust_adapt_output,
        is_store_final_clustering)
    # Store data associated with the self-consistent scheme
    info.displayinfo('5', 'Storing self-consistent scheme data...')
    scs_dict = packager.packagescs(
        self_consistent_scheme, scs_max_n_iterations, scs_conv_tol)
    # Store data associated with the self-consistent scheme
    info.displayinfo('5', 'Storing algorithmic data...')
    algpar_dict = packager.packalgparam(
        max_n_iterations, conv_tol, max_subinc_level, max_cinc_cuts,
        su_max_n_iterations, su_conv_tol)
    # Store data associated with the VTK output
    info.displayinfo('5', 'Storing VTK output data...')
    if is_vtk_output:
        vtk_dict = packager.packvtk(
            is_vtk_output, vtk_format, vtk_inc_div, vtk_vars)
    else:
        vtk_dict = packager.packvtk(is_vtk_output)
    # Store data associated with the output files
    info.displayinfo('5', 'Storing general output files data...')
    output_dict = packager.packoutputfiles(
        is_ref_material_output, is_voxels_output)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return problem_dict, mat_dict, macload_dict, rg_dict, clst_dict, \
        scs_dict, algpar_dict, vtk_dict, output_dict, material_state
