#
# Cluster Analysis Module (CRATE Program)
# ==========================================================================================
# Summary:
# Interface to the computation of the Cluster-Reduced Representative Volume Element (CRVE).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2020 | Refactored as Clustering class.
# Bernardo P. Ferreira | Dec 2020 | Removed CRVE types.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Python object serialization
import pickle
# CRVE generation
from clustering.crve import CRVE
#
#                                                                           Cluster Analysis
# ==========================================================================================
class Clustering:
    '''The Clustering class provides a simple interface to generate a CRVE.

    The Clustering class provides a simple interface to perform the computation of the
    Cluster-Reduced Representative Volume Element (CRVE), including the clustering-based
    domain decomposition and the computation of the cluster interaction tensors.
    '''
    def __init__(self, rve_dims, material_phases, regular_grid, comp_order,
                 cluster_data_matrix, clustering_type, phase_n_clusters,
                 base_clustering_scheme, adaptive_clustering_scheme=None,
                 adaptivity_criterion=None, adaptivity_type=None,
                 adaptivity_control_feature=None):
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
        cluster_data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing all the required data to perform the RVE clustering-based
            domain decomposition.
        clustering_type : dict
            Clustering type (item, {'static', 'adaptive'}) of each material phase
            (key, str).
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, ndarry of shape (n_clusterings, 3)) for
            each material phase (key, str). Each row is associated with a unique clustering
            characterized by a clustering algorithm (col 1, int), a list of features
            (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        adaptive_clustering_scheme : dict, default=None
            Prescribed adaptive clustering scheme (item, ndarry of shape (n_clusterings, 3))
            for each material phase (key, str). Each row is associated with a unique
            clustering characterized by a clustering algorithm (col 1, int), a list of
            features (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        adaptivity_criterion : dict, default=None
            Clustering adaptivity criterion (item, dict) associated to each material phase
            (key, str). This dictionary contains the adaptivity criterion to be used and the
            required parameters.
        adaptivity_type : dict, default=None
            Clustering adaptivity type (item, dict) associated to each material phase
            (key, str). This dictionary contains the adaptivity type to be used and the
            required parameters.
        adaptivity_control_feature : dict, default=None
            Clustering adaptivity control feature (item, str) associated to each material
            phase (key, str).
        '''
        self._rve_dims = rve_dims
        self._material_phases = material_phases
        self._regular_grid = regular_grid
        self._comp_order = comp_order
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = clustering_type
        self._phase_n_clusters = phase_n_clusters
        self._base_clustering_scheme = base_clustering_scheme
        self._adaptive_clustering_scheme = adaptive_clustering_scheme
        self._adaptivity_criterion = adaptivity_criterion
        self._adaptivity_type = adaptivity_type
        self._adaptivity_control_feature = adaptivity_control_feature
    # --------------------------------------------------------------------------------------
    def compute_crve(self):
        '''Compute Cluster-Reduced Representative Volume Element (CRVE).

        Returns
        -------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        '''
        # Instatiate Cluster-Reduced Representative Volume Element (CRVE)
        crve = CRVE(self._rve_dims, self._regular_grid, self._material_phases,
                    self._comp_order, self._cluster_data_matrix, self._clustering_type,
                    self._phase_n_clusters, self._base_clustering_scheme,
                    self._adaptive_clustering_scheme, self._adaptivity_criterion,
                    self._adaptivity_type, self._adaptivity_control_feature)
        # Perform CRVE base clustering
        crve.compute_base_crve()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return crve
    # --------------------------------------------------------------------------------------
    def compute_cit(self, crve, mode='full'):
        '''Compute CRVE cluster interaction tensors.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
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
            Cluster-Reduced Representative Volume Element.
        crve_file_path : str
            Path of file where the CRVE's instance is dumped.
        '''
        # Dump CRVE instance into file
        with open(crve_file_path, 'wb') as crve_file:
            pickle.dump(crve, crve_file)
