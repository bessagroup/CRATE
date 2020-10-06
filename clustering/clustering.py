#
# Perform Clustering Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the clustering-based domain decomposition of the representative
# volume element (RVE) into the Cluster-reduced Representative Volume Element (CRVE).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Python object serialization
import pickle
# Inspect file name and line
import inspect
# Display messages
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# CRVE generation
from clustering.crve import CRVE
#
#                                   Set Cluster-reduced Representative Volume Element (CRVE)
# ==========================================================================================
# Perform prescribed clustering scheme, compute the Cluster-reduced Representative Element
# (CRVE) and set all the associated descriptors
def set_crve_data(dirs_dict, mat_dict, rg_dict, clst_dict):
    # Get directories and paths data
    cluster_file_path = dirs_dict['cluster_file_path']
    # Get material data
    material_phases = mat_dict['material_phases']
    # Get regular grid data
    regular_grid = rg_dict['regular_grid']
    rve_dims = rg_dict['rve_dims']
    # Get clustering data
    clustering_scheme = clst_dict['clustering_scheme']
    clustering_ensemble_strategy = clst_dict['clustering_ensemble_strategy']
    phase_n_clusters = clst_dict['phase_n_clusters']
    clst_quantities = clst_dict['clst_quantities']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing Cluster-reduced Representative Volume Element ' +
                          '(CRVE)...')
    # Instatiate Cluster-reduced Representative Volume Element (CRVE)
    crve = CRVE(clustering_scheme, clustering_ensemble_strategy, phase_n_clusters, rve_dims,
                regular_grid, material_phases)
    # Perform prescribed clustering scheme to generate the CRVE
    crve.get_crve(clst_quantities)
    # Store CRVE descriptors
    clst_dict['voxels_clusters'] = crve.voxels_clusters
    clst_dict['phase_clusters'] = crve.phase_clusters
    clst_dict['clusters_f'] = crve.clusters_f
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open file which contains all the required information associated to the
    # clustering scheme and the Cluster-reduced Representative Volume Element (CRVE)
    try:
        cluster_file = open(cluster_file_path, 'wb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1, message)
    # Dump clustering data
    info.displayinfo('5', 'Storing clustering file (.clusters)...')
    pickle.dump(clst_dict, cluster_file)
    # Close file
    cluster_file.close()
#
#                                                                Perform compatibility check
#                                                (loading previously computed offline stage)
# ==========================================================================================
# Perform a compatibility check between the clustering parameters read from the input data
# file and the previously computed offline stage loaded data
def checkclstcompat(problem_dict, rg_dict, clst_dict_read, clst_dict):
    # Check clustering method, clustering strategy and clustering solution method
    keys = ['clustering_method', 'clustering_strategy', 'clustering_solution_method']
    for key in keys:
        if clst_dict[key] != clst_dict_read[key]:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00044', location.filename, location.lineno + 1, key,
                                clst_dict_read[key],clst_dict[key])
    # Check number of clusters associated to each material phase
    if clst_dict['phase_n_clusters'] != clst_dict_read['phase_n_clusters']:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00045', location.filename, location.lineno + 1,
                            clst_dict_read['phase_n_clusters'],
                            clst_dict['phase_n_clusters'])
    # Check spatial discretization
    elif list(clst_dict['voxels_clusters'].shape) != rg_dict['n_voxels_dims']:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00046', location.filename, location.lineno + 1,
                            rg_dict['n_voxels_dims'],
                            list(clst_dict['voxels_clusters'].shape))
