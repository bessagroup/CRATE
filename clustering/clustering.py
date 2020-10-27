#
# Cluster Analysis Module (CRATE Program)
# ==========================================================================================
# Summary:
# Interface to the computation of the Cluster-reduced Representative Volume Element (CRVE).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# Bernardo P. Ferreira |  October 2020 | Refactored as Clustering class.
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
from clustering.crve import SCRVE, HACRVE
#
#                                                                           Cluster Analysis
# ==========================================================================================
class Clustering:
    '''The Clustering class provides a simple interface to generate a CRVE.

    The Clustering class provides a simple interface to perform the computation of the
    Cluster-reduced Representative Volume Element (CRVE), including the clustering-based
    domain decomposition and the computation of the cluster interaction tensors.
    '''
    def __init__(self, rve_dims, material_phases, regular_grid, comp_order,
                 phase_n_clusters, crve_type, clustering_options, cluster_data_matrix):
        '''Clustering class constructor.

        Parameters
        ----------
        rve_dims : list
            RVE size in each dimension.
        material_phases : list
            RVE material phases labels (str).
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        comp_order : list
            Strain/Stress components (str) order.
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        cluster_type : str, {'static', 'hierarchical-adaptive'}
            Type of Cluster-reduced Representative Volume Element (CRVE).
        clustering_options : dict
            To be defined...
        cluster_data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing all the required data to perform the RVE clustering-based
            domain decomposition.
        '''
        self._rve_dims = rve_dims
        self._material_phases = material_phases
        self._regular_grid = regular_grid
        self._comp_order = comp_order
        self._phase_n_clusters = phase_n_clusters
        self._crve_type = crve_type
        self._clustering_options = clustering_options
        self._cluster_data_matrix = cluster_data_matrix
    # --------------------------------------------------------------------------------------
    def compute_crve(self):
        '''Compute Cluster-reduced Representative Volume Element (CRVE).

        Returns
        -------
        crve : CRVE
            Cluster-reduced Representative Volume Element.
        '''
        if self._crve_type == 'static':
            info.displayinfo('5', 'Computing Static Cluster-reduced Representative ' +
                                  'Volume Element (S-CRVE)...')
            # Get S-CRVE required clustering parameters
            try:
                clustering_scheme = self._clustering_options['clustering_scheme']
                clustering_ensemble_strategy = \
                    self._clustering_options['clustering_ensemble_strategy']
            except KeyError:
                print('Missing S-CRVE required clustering parameter.')
                raise
            # Instatiate Static Cluster-reduced Representative Volume Element (S-CRVE)
            crve = SCRVE(self._phase_n_clusters, self._rve_dims, self._regular_grid,
                self._material_phases, self._comp_order, clustering_scheme,
                clustering_ensemble_strategy)
            # Perform prescribed clustering scheme to generate the CRVE
            crve.get_scrve(self._cluster_data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._crve_type == 'hierarchical-adaptive':
            info.displayinfo('5', 'Computing Hierarchical Adaptive Cluster-reduced ' +
                                  'Representative Volume Element (HA-CRVE)...')
            # Get HA-CRVE required clustering parameters
            pass
            # Instatiate Hierarchical Adaptive Cluster-reduced Representative Volume Element
            # (HA-CRVE)
            crve = HACRVE(self._phase_n_clusters, self._rve_dims, self._regular_grid,
                          self._material_phases, self._comp_order, split_greed=0.5)
            # Perform HA-CRVE base clustering
            crve.get_base_clustering(self._cluster_data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown type of cluster-reduced representative volume ' +
                               'element.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return CRVE instance
        return crve
    # --------------------------------------------------------------------------------------
    def compute_cit(self, crve, mode='full'):
        '''Compute CRVE cluster interaction tensors.

        Parameters
        ----------
        crve : CRVE
            Cluster-reduced Representative Volume Element.
        mode : str, {'full', 'adaptive'}, default='full'
            The default 'full' mode performs the complete computation of all cluster
            interaction tensors, irrespective of the type of CRVE. The 'adaptive' mode
            speeds up the computation of the new cluster interaction tensors resulting
            from an adaptation step of an adaptive CRVE.

        Notes
        -----
        The cluster interaction tensors 'adaptive' computation mode can only be performed
        after at least one 'full' computation has been performed.
        '''
        # Check parameters
        if mode not in ['full', 'adaptive']:
            raise RuntimeError('Unknown mode to compute cluster interaction tensors.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute CRVE cluster interaction tensors
        if mode == 'full':
            crve.compute_cit(mode)
        elif mode == 'adaptive':
            if crve._adaptive_clustering_map is None:
                raise RuntimeError('Adaptive clustering map must be first build in a ' +
                                   '\'full\' computation in order to use the ' +
                                   '\'adaptive\' mode.')
            else:
                crve.compute_cit(mode,
                                 adaptive_clustering_map=self._adaptive_clustering_map)
    # --------------------------------------------------------------------------------------
    def save_crve_file(self, crve, crve_file_path):
        '''Dump CRVE into file.

        Parameters
        ----------
        crve : CRVE
            Cluster-reduced Representative Volume Element.
        crve_file_path : str
            Path of file where the CRVE's instance is dumped.
        '''
        # Dump CRVE instance into file
        info.displayinfo('5', 'Dumping CRVE into file (.crve)...')
        with open(crve_file_path, 'wb') as crve_file:
            pickle.dump(crve, crve_file)
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
