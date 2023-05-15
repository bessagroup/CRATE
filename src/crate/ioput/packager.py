"""Store data in suitable containers.

This module includes a set of functions that store data in suitable
category-related containers.

Functions
---------
store_paths_data
    Store problem directories and files paths.
store_problem_data
    Store data associated with the problem formulation and type.
store_material_data
    Store data associated with the material phases.
store_loading_path_data
    Store data associated with the macroscale loading path.
store_regular_grid_data
    Store data associated with the RVE spatial discretization.
store_clustering_data
    Store data associated with the clustering-based domain decomposition.
store_scs_data
    Store data associated with the self-consistent scheme.
store_algorithmic_data
    Store data associated with the problem solution algorithmic parameters.
store_vtk_data
    Store data associated with the VTK output.
store_output_data
    Store data associated with general output files.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import itertools as it
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def store_paths_data(input_file_name, input_file_path, input_file_dir,
                     problem_name, problem_dir, offline_stage_dir,
                     postprocess_dir, crve_file_path, discret_file_dir=None):
    """Store problem directories and files paths.

    Parameters
    ----------
    input_file_name : str
        Input data file name.
    input_file_path : str
        Input data file path.
    input_file_dir : str
        Input data file directory path.
    problem_name : str
        Problem name.
    problem_dir : str
        Problem output directory path.
    offline_stage_dir : str
        Problem output offline-stage subdirectory path.
    postprocess_dir : str
        Problem output post-processing subdirectory path.
    crve_file_path : str
        Problem '.crve' output file path.
    discret_file_dir : str, default=None
        Spatial discretization file directory path.

    Returns
    -------
    dirs_dict : dict
        Container.
    """
    # Initialize directories and paths dictionary
    dirs_dict = dict()
    # Build directories and paths dictionary
    dirs_dict['input_file_name'] = input_file_name
    dirs_dict['input_file_path'] = input_file_path
    dirs_dict['input_file_dir'] = input_file_dir
    dirs_dict['discret_file_dir'] = discret_file_dir
    dirs_dict['problem_name'] = problem_name
    dirs_dict['problem_dir'] = problem_dir
    dirs_dict['offline_stage_dir'] = offline_stage_dir
    dirs_dict['crve_file_path'] = crve_file_path
    dirs_dict['postprocess_dir'] = postprocess_dir
    # Return
    return dirs_dict
# =============================================================================
def store_problem_data(strain_formulation, problem_type, n_dim, comp_order_sym,
                       comp_order_nsym):
    """Store data associated with the problem formulation and type.

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.

    Returns
    -------
    problem_dict : dict
        Container.
    """
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
# =============================================================================
def store_material_data(material_phases, material_phases_data,
                        material_phases_properties, material_phases_vf):
    """Store data associated with the material phases.

    Parameters
    ----------
    material_phases : list[str]
        RVE material phases labels (str).
    material_phases_data : dict
        Material phase data (item, dict) associated with each material phase
        (key, str).
    material_phases_properties : dict
        Constitutive model material properties (item, dict) associated with
        each material phase (key, str).
    material_phases_vf : dict
        Volume fraction (item, float) associated to each material phase
        (key, str).

    Returns
    -------
    mat_dict : dict
        Container.
    """
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
# =============================================================================
def store_loading_path_data(mac_load_type, mac_load, mac_load_presctype,
                            mac_load_increm, is_solution_rewinding,
                            rewind_state_criterion=None,
                            rewinding_criterion=None, max_n_rewinds=None):
    """Store data associated with the macroscale loading path.

    Parameters
    ----------
    mac_load_type : {1, 2, 3}
        Loading type:

        * 1 : Macroscale strain constraint
        * 2 : Macroscale stress constraint
        * 3 : Macroscale strain and stress constraint
    mac_load : dict
        For each loading nature type (key, {'strain', 'stress'}), stores
        the loading constraints for each loading subpath in a
        numpy.ndarray (2d), where the i-th row is associated with the i-th
        strain/stress component and the j-th column is associated with the
        j-th loading subpath.
    mac_load_presctype : numpy.ndarray (2d)
        Loading nature type ({'strain', 'stress'}) associated with each
        loading constraint (ndarray of shape (n_comps, n_load_subpaths)),
        where the i-th row is associated with the i-th strain/stress
        component and the j-th column is associated with the j-th loading
        subpath.
    mac_load_increm : dict
        For each loading subpath id (key, str), stores a numpy.ndarray of shape
        (n_load_increments, 2) where each row is associated with a prescribed
        loading increment, and the columns 0 and 1 contain the corresponding
        incremental load factor and incremental time, respectively.
    is_solution_rewinding : bool, default=False
        Problem solution rewinding flag.
    rewind_state_criterion : tuple, default=None
        Rewind state storage criterion [0] and associated parameter [1].
    rewinding_criterion : tuple, default=None
        Rewinding criterion [0] and associated parameter [1].
    max_n_rewinds : int, default=None
        Maximum number of rewind operations.

    Returns
    -------
    macload_dict : dict
        Container.
    """
    # Initialize macroscale loading dictionary
    macload_dict = dict()
    # Build macroscale loading dictionary
    macload_dict['mac_load_type'] = mac_load_type
    macload_dict['mac_load'] = mac_load
    macload_dict['mac_load_presctype'] = mac_load_presctype
    macload_dict['mac_load_increm'] = mac_load_increm
    macload_dict['is_solution_rewinding'] = is_solution_rewinding
    macload_dict['rewind_state_criterion'] = rewind_state_criterion
    macload_dict['rewinding_criterion'] = rewinding_criterion
    macload_dict['max_n_rewinds'] = max_n_rewinds
    # Return
    return macload_dict
# =============================================================================
def store_regular_grid_data(discret_file_path, regular_grid, rve_dims,
                            problem_dict):
    """Store data associated with the RVE spatial discretization.

    Parameters
    ----------
    discret_file_path : str
        Spatial discretization file path.
    regular_grid : numpy.ndarray (2d or 3d)
        Regular grid of voxels (spatial discretization of the RVE), where
        each entry contains the material phase label (int) assigned to the
        corresponding voxel.
    rve_dims : list
        RVE size in each dimension.
    problem_dict : dict
        Container.

    Returns
    -------
    rg_dict : dict
        Container.
    """
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Set number of pixels/voxels in each dimension
    n_voxels_dims = [regular_grid.shape[i]
                     for i in range(len(regular_grid.shape))]
    n_voxels = np.prod(n_voxels_dims)
    # Get material phases present in the microstructure
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    # Flatten the regular grid array such that:
    #
    # 2D Problem (swipe 2-1):
    #   voxel(i,j) is stored in index = i*d2 + j, where d2 is the
    #   the number of voxels along dimension 2
    #
    # 3D Problem (swipe 3-2-1):
    #   voxel(i,j,k) is stored in index = i*(d2*d3) + j*d3 + k,
    #   where d2 and d3 are the number of voxels along dimensions 2 and 3
    #   respectively
    #
    regular_grid_flat = list(regular_grid.flatten())
    # Build flattened list with the voxels indexes (consistent with the flat
    # regular grid)
    voxels_idx_flat = list()
    shape = tuple([n_voxels_dims[i] for i in range(n_dim)])
    voxels_idx_flat = [np.unravel_index(i, shape) for i in range(n_voxels)]
    # Set voxel flattened indexes associated to each material phase
    phase_voxel_flatidx = dict()
    for mat_phase in material_phases:
        is_phase_list = regular_grid.flatten() == int(mat_phase)
        phase_voxel_flatidx[mat_phase] = \
            list(it.compress(range(len(is_phase_list)), is_phase_list))
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
# =============================================================================
def store_clustering_data(clustering_solution_method, standardization_method,
                          phase_n_clusters, rg_dict, clustering_type,
                          base_clustering_scheme, adaptive_clustering_scheme,
                          adapt_criterion_data, adaptivity_type,
                          adaptivity_control_feature, clust_adapt_freq,
                          is_clust_adapt_output, is_store_final_clustering):
    """Store data associated with the clustering-based domain decomposition.

    Parameters
    ----------
    clustering_solution_method : int
        Identifier of DNS homogenization-based multi-scale DNS method to
        compute the clustering features data.
    standardization_method : int
        Identifier of global cluster analysis data standardization algorithm.
    phase_n_clusters : dict
        Number of clusters (item, int) prescribed for each material phase
        (key, str).
    rg_dict : dict
        Container.
    clustering_type : str
        Type of cluster-reduced material phase.
    base_clustering_scheme : dict
        Prescribed base clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).
    adaptive_clustering_scheme : dict
        Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).
    adapt_criterion_data : dict
        Clustering adaptivity criterion (item, dict) associated with each
        material phase (key, str). This dictionary contains the adaptivity
        criterion to be used and the required parameters.
    adaptivity_type : dict
        Clustering adaptivity type (item, dict) associated with each material
        phase (key, str). This dictionary contains the adaptivity type to be
        used and the required parameters.
    adaptivity_control_feature : dict
        Clustering adaptivity control feature (item, str) associated with each
        material phase (key, str).
    clust_adapt_freq : dict
        Clustering adaptivity frequency (relative to the macroscale
        loading) (item, int, default=1) associated with each adaptive
        cluster-reduced material phase (key, str).
    is_clust_adapt_output : bool
        Clustering adaptivity output flag.
    is_store_final_clustering : bool
        `True` to store CRVE final clustering state into file, `False`
        otherwise.

    Returns
    -------
    clst_dict : dict
        Container.
    """
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
# =============================================================================
def store_scs_data(self_consistent_scheme, scs_parameters,
                   scs_max_n_iterations, scs_conv_tol):
    """Store data associated with the self-consistent scheme.

    Parameters
    ----------
    self_consistent_scheme : {'none', 'regression',}
        Self-consistent scheme to update the elastic reference material
        properties.
    scs_parameters : {dict, None}
        Self-consistent scheme parameters (key, str; item, {int, float, bool}).
    scs_max_n_iterations : int
        Self-consistent scheme maximum number of iterations.
    scs_conv_tol : float
        Self-consistent scheme convergence tolerance.

    Returns
    -------
    scs_dict : dict
        Container.
    """
    # Initialize self-consistent scheme dictionary
    scs_dict = dict()
    # Build self-consistent scheme dictionary
    scs_dict['self_consistent_scheme'] = self_consistent_scheme
    scs_dict['scs_parameters'] = scs_parameters
    scs_dict['scs_max_n_iterations'] = scs_max_n_iterations
    scs_dict['scs_conv_tol'] = scs_conv_tol
    # Return
    return scs_dict
# =============================================================================
def store_algorithmic_data(max_n_iterations, conv_tol, max_subinc_level,
                           max_cinc_cuts, su_max_n_iterations, su_conv_tol):
    """Store data associated with the problem solution algorithmic parameters.

    Parameters
    ----------
    max_n_iterations : int
        Newton-Raphson maximum number of iterations.
    conv_tol : float
        Newton-Raphson convergence tolerance.
    max_subinc_level : int
        Maximum level of loading subincrementation.
    max_cinc_cuts : int
        Maximum number of consecutive increment cuts.
    su_max_n_iterations : int
        State update maximum number of iterations.
    su_conv_tol : float
        State update convergence tolerance.

    Returns
    -------
    algpar_dict : dict
        Container.
    """
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
# =============================================================================
def store_vtk_data(is_vtk_output, *args):
    """Store data associated with the VTK output.

    Parameters
    ----------
    is_vtk_output : bool
        VTK output flag.
    *args :
        * vtk_inc_div (int): VTK output increment divider.
        * vtk_vars {'all', 'common'}: VTK output constitutive state variables.

    Returns
    -------
    vtk_dict : dict
        Container.
    """
    # Initialize VTK dictionary
    vtk_dict = dict()
    # Build VTK dictionary
    vtk_dict['is_vtk_output'] = is_vtk_output
    if is_vtk_output:
        vtk_format = args[0]
        vtk_inc_div = args[1]
        vtk_vars = args[2]
        vtk_dict['vtk_format'] = vtk_format
        vtk_dict['vtk_inc_div'] = vtk_inc_div
        vtk_dict['vtk_vars'] = vtk_vars
        vtk_dict['vtk_precision'] = 'SinglePrecision'
        if sys.byteorder == 'little':
            vtk_dict['vtk_byte_order'] = 'LittleEndian'
        else:
            vtk_dict['vtk_byte_order'] = 'BigEndian'
    # Return
    return vtk_dict
# =============================================================================
def store_output_data(is_ref_material_output, is_voxels_output):
    """Store data associated with general output files.

    Parameters
    ----------
    is_ref_material_output : bool, default=False
        Reference material output flag.
    is_voxels_output : bool
        Voxels output file flag.

    Returns
    -------
    output_dict : dict
        Container.
    """
    # Initialize output dictionary
    output_dict = dict()
    # Build output dictionary
    output_dict['is_ref_material_output'] = is_ref_material_output
    output_dict['is_voxels_output'] = is_voxels_output
    # Return
    return output_dict
