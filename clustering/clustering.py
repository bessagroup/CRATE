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
# Display messages
import ioput.info as info
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
                 phase_n_clusters, clustering_scheme, crve_type, crve_type_options,
                 cluster_data_matrix):
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
        clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed global clustering scheme to generate the CRVE. Each row is associated
            with a unique RVE clustering, characterized by a clustering algorithm
            (col 1, int), a list of features (col 2, list of int) and a list of the feature
            data matrix' indexes (col 3, list of int).
        cluster_type : str, {'static', 'hierarchical-adaptive'}
            Type of Cluster-reduced Representative Volume Element (CRVE).
        crve_type_options : dict
            CRVE type specific options.
        cluster_data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing all the required data to perform the RVE clustering-based
            domain decomposition.
        '''
        self._rve_dims = rve_dims
        self._material_phases = material_phases
        self._regular_grid = regular_grid
        self._comp_order = comp_order
        self._phase_n_clusters = phase_n_clusters
        self._clustering_scheme = clustering_scheme
        self._crve_type = crve_type
        self._crve_type_options = crve_type_options
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
                clustering_ensemble_strategy = \
                    self._crve_type_options['clustering_ensemble_strategy']
            except KeyError:
                print('Missing S-CRVE required clustering parameter.')
                raise
            # Instatiate Static Cluster-reduced Representative Volume Element (S-CRVE)
            crve = SCRVE(self._phase_n_clusters, self._rve_dims, self._regular_grid,
                self._material_phases, self._comp_order, self._clustering_scheme,
                clustering_ensemble_strategy)
            # Perform prescribed clustering scheme to generate the CRVE
            crve.get_scrve(self._cluster_data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._crve_type == 'hierarchical-adaptive':
            info.displayinfo('5', 'Computing Hierarchical Adaptive Cluster-reduced ' +
                                  'Representative Volume Element (HA-CRVE)...')
            # Get HA-CRVE required clustering parameters
            adaptive_split_factor = self._crve_type_options['adaptive_split_factor']
            # Instatiate Hierarchical Adaptive Cluster-reduced Representative Volume Element
            # (HA-CRVE)
            crve = HACRVE(self._phase_n_clusters, self._rve_dims, self._regular_grid,
                          self._material_phases, self._comp_order, adaptive_split_factor)
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
        with open(crve_file_path, 'wb') as crve_file:
            pickle.dump(crve, crve_file)
#
#                                                                       Available CRVE types
# ==========================================================================================
def get_available_crve_types():
    '''Get available CRVE types in CRATE.

    Available CRVE types:
    1- Static ('static')
    2- Hierarchical Adaptive ('hierarchical-adaptive')

    Returns
    -------
    available_crve_types : dict
        Available CRVE types (item, str) and associated identifiers (key, str).
    '''
    available_crve_types = {'1': 'static',
                            '2': 'hierarchical-adaptive'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return available_crve_types
