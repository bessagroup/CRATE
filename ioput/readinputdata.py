#
# Input Data Reader Module (CRATE Program)
# ==========================================================================================
# Summary:
# Read and extraction of data from the user provided input data file. All the relevant data
# is stored in suitable dictionaries.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Operations on files and directories
import shutil
# Display messages
import ioput.info as info
# Read specific lines from file
import linecache
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Manage files and directories
import ioput.fileoperations as filop
# Packager
import ioput.packager as packager
# Links related procedures
import links.ioput.readlinksinputdatafile as linksrlid
# Reading procedures
import ioput.readprocedures as rproc
# I/O utilities
import ioput.ioutilities as ioutil
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                                       Read input data file
# ==========================================================================================
# Read the input data file
def readinputdatafile(input_file,dirs_dict):
    # Get input file and problem directory and path data
    input_file_path = dirs_dict['input_file_path']
    problem_dir = dirs_dict['problem_dir']
    postprocess_dir = dirs_dict['postprocess_dir']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read strain formulation
    keyword = 'Strain_Formulation'
    max = 2
    strain_formulation = rproc.readtypeAkeyword(input_file, input_file_path, keyword, max)
    # Large strain formulation has not been implemented yet
    if strain_formulation == 2:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00016', location.filename, location.lineno + 1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read problem type and set problem dimensions
    keyword = 'Problem_Type'
    max = 4
    problem_type = rproc.readtypeAkeyword(input_file, input_file_path, keyword, max)
    n_dim, comp_order_sym, comp_order_nsym = mop.getproblemtypeparam(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read RVE dimensions
    keyword = 'RVE_Dimensions'
    rve_dims = rproc.readrvedimensions(input_file, input_file_path, keyword, n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of material phases and the associated constitutive models and properties
    keyword = 'Material_Phases'
    n_material_phases, material_phases_models, material_properties = \
        rproc.readmaterialproperties(input_file, input_file_path, keyword)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the Links python binary absolute path if at least one material phase has the
    # associated constitutive model source set as Links
    is_Links_python_bin = False
    for mat_phase in material_properties.keys():
        if material_phases_models[mat_phase]['source'] == 2:
            # Read the Links python binary absolute path
            keyword = 'Links_Python_bin'
            line_number = rproc.searchkeywordline(input_file, keyword) + 1
            Links_python_bin_path = linecache.getline(input_file_path, line_number).strip()
            if not os.path.isabs(Links_python_bin_path):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00087', location.filename, location.lineno + 1,
                                    keyword, Links_python_bin_path)
            elif not os.path.isfile(Links_python_bin_path):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00087', location.filename, location.lineno + 1,
                                    keyword, Links_python_bin_path)
            else:
                is_Links_python_bin = True
                break
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read macroscale loading
    keyword = 'Macroscale_Loading'
    max = 3
    mac_load_type = rproc.readtypeAkeyword(input_file, input_file_path, keyword, max)
    mac_load, mac_load_presctype = rproc.readmacroscaleloading(
        input_file, input_file_path, mac_load_type, strain_formulation, n_dim,
        comp_order_nsym)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme (optional). If the associated keyword is not found, then
    # a default specification is assumed
    keyword = 'Self_Consistent_Scheme'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = 2
        self_consistent_scheme = rproc.readtypeAkeyword(input_file, input_file_path,
                                                        keyword, max)
    else:
        self_consistent_scheme = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme maximum number of iterations (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SCS_Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = '~'
        scs_max_n_iterations = rproc.readtypeAkeyword(input_file, input_file_path,
                                                      keyword, max)
    else:
        scs_max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme convergence tolerance (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SCS_Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = '~'
        scs_conv_tol = rproc.readtypeBkeyword(input_file, input_file_path, keyword)
    else:
        scs_conv_tol = 1e-4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering method
    keyword = 'Clustering_Method'
    max = 1
    clustering_method = rproc.readtypeAkeyword(input_file, input_file_path, keyword, max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering strategy (optional). If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Clustering_Strategy'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = 1
        clustering_strategy = rproc.readtypeAkeyword(input_file, input_file_path, keyword,
                                                     max)
    else:
        clustering_strategy = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering solution method (optional). If the associated keyword is not found,
    # then a default specification is assumed
    keyword = 'Clustering_Solution_Method'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = 2
        clustering_solution_method = rproc.readtypeAkeyword(input_file, input_file_path,
                                                            keyword, max)
    else:
        clustering_solution_method = 1
    # Read clustering solution method parameters
    links_dict = dict()
    if clustering_solution_method == 2:
        # Check if all the material phases have the associated constitutive model source set
        # as Links
        for mat_phase in material_properties.keys():
            if material_phases_models[mat_phase]['source'] != 2:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00067', location.filename, location.lineno + 1)
        # Build Links dictionary
        links_dict = linksrlid.readlinksinputdatafile(
            input_file, input_file_path, problem_type, ioutil.checknumber,
            ioutil.checkposint, rproc.searchkeywordline, rproc.searchoptkeywordline)
    # If at least one material phase has the associated constitutive model source set as
    # Links add Links python binary to Links dictionary
    if is_Links_python_bin:
        links_dict['Links_python_bin_path'] = Links_python_bin_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of cluster associated to each material phase
    keyword = 'Number_of_Clusters'
    phase_n_clusters = rproc.readphaseclustering(input_file, input_file_path, keyword,
                                                 n_material_phases, material_properties)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read macroscale loading incrementation parameters
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
        mac_load_increm = rproc.readmacloadincrem(input_file, input_file_path, keyword,
                                                  n_load_subpaths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum number of iterations to solve each load increment (optional). If the
    # associated keyword is not found, then a default specification is assumed
    keyword = 'Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = '~'
        max_n_iterations = rproc.readtypeAkeyword(input_file, input_file_path, keyword, max)
    else:
        max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read convergence tolerance to solve each load increment (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        conv_tol = rproc.readtypeBkeyword(input_file, input_file_path, keyword)
    else:
        conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum level of subincrementation allowed (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'Max_SubIncrem_Level'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = '~'
        max_subincrem_level = rproc.readtypeAkeyword(input_file, input_file_path, keyword,
                                                     max)
    else:
        max_subincrem_level = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update maximum number of iterations (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SU_Max_Number_of_Iterations'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        max = '~'
        su_max_n_iterations = rproc.readtypeAkeyword(input_file, input_file_path, keyword,
                                                     max)
    else:
        su_max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update convergence tolerance (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SU_Convergence_Tolerance'
    is_found, _ = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        su_conv_tol = rproc.readtypeBkeyword(input_file, input_file_path, keyword)
    else:
        su_conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the spatial discretization file absolute path
    keyword = 'Discretization_File'
    valid_exts = ['.rgmsh']
    discret_file_path = rproc.readdiscretizationfilepath(input_file, input_file_path,
                                                         keyword, valid_exts)
    # Copy the spatial discretization file to the problem directory and update the absolute
    # path to the copied file
    try:
        shutil.copy2(discret_file_path, problem_dir + ntpath.basename(discret_file_path))
        discret_file_path = problem_dir + ntpath.basename(discret_file_path)
    except IOError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1, message)
    # Store spatial discretization file absolute path
    dirs_dict['discret_file_path'] = discret_file_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read VTK output options
    keyword = 'VTK_Output'
    is_found, keyword_line_number = rproc.searchoptkeywordline(input_file, keyword)
    if is_found:
        is_VTK_output = True
        vtk_format, vtk_inc_div, vtk_vars = \
            rproc.readvtkoptions(input_file, input_file_path, keyword, keyword_line_number)
        # Create VTK folder in post processing directory
        filop.makedirectory(postprocess_dir + 'VTK/', 'overwrite')
    else:
        is_VTK_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Package problem general data
    info.displayinfo('5', 'Packaging problem general data...')
    problem_dict = packager.packproblem(strain_formulation, problem_type, n_dim,
                                        comp_order_sym, comp_order_nsym)
    # Package data associated to the material phases
    info.displayinfo('5', 'Packaging material data...')
    mat_dict = packager.packmaterialphases(n_material_phases, material_phases_models,
                                           material_properties)
    # Package data associated to the macroscale loading
    info.displayinfo('5', 'Packaging macroscale loading data...')
    macload_dict = packager.packmacroscaleloading(mac_load_type, mac_load,
                                                  mac_load_presctype, mac_load_increm)
    # Package data associated to the spatial discretization file(s)
    info.displayinfo('5', 'Packaging regular grid data...')
    rg_dict = packager.packregulargrid(discret_file_path, rve_dims, mat_dict,
                                       problem_dict)
    # Package data associated to the clustering
    info.displayinfo('5', 'Packaging clustering data...')
    clst_dict = packager.packrgclustering(clustering_method, clustering_strategy,
                                          clustering_solution_method, links_dict,
                                          phase_n_clusters, rg_dict)
    # Package data associated to the self-consistent scheme
    info.displayinfo('5', 'Packaging self-consistent scheme data...')
    scs_dict = packager.packagescs(self_consistent_scheme, scs_max_n_iterations,
                                   scs_conv_tol)
    # Package data associated to the self-consistent scheme
    info.displayinfo('5', 'Packaging algorithmic data...')
    algpar_dict = packager.packalgparam(
        max_n_iterations, conv_tol, max_subincrem_level, su_max_n_iterations, su_conv_tol)
    # Package data associated to the VTK output
    info.displayinfo('5', 'Packaging VTK output data...')
    if is_VTK_output:
        vtk_dict = packager.packvtk(is_VTK_output, vtk_format, vtk_inc_div, vtk_vars)
    else:
        vtk_dict = packager.packvtk(is_VTK_output)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [problem_dict, mat_dict, macload_dict, rg_dict, clst_dict, scs_dict, algpar_dict,
            vtk_dict]
