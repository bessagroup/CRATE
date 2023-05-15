"""Cluster-Reduced Material Phase.

This module includes the interface to implement any Cluster-Reduced Material
Phase and several cluster-reduced material phases, namely the most basic Static
Cluster-Reduced Material Phase (SCRMP). Each class contains all the required
methods to manage the material phase clustering.

The concept of Cluster-Reduced Material Phase was coined by
Ferreira et. al (2022) [#]_ and arises in the context of clustering-based
reduced order modeling (see Chapter 4 of Ferreira (2022) [#]_).

.. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
       *Adaptivity for clustering-based reduced-order modeling of
       localized history-dependent phenomena.* Comp Methods Appl M, 393
       (see `here <https://www.sciencedirect.com/science/article/pii/
       S0045782522000895?via%3Dihub>`_)

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <https://repositorio-aberto.up.pt/handle/10216/
       146900?locale=en>`_)

Classes
-------
CRMP
    Cluster-Reduced Material Phase interface.
ACRMP
    Adaptive Cluster-Reduced Material Phase interface.
SCRMP
    Static Cluster-Reduced Material Phase (SCRMP).
GACRMP
    Generalized Adaptive Cluster-Reduced Material Phase.
HAACRMP
    Hierarchical Agglomerative Adaptive Cluster-Reduced Material Phase.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import time
import copy
# Third-party
import numpy as np
import scipy.cluster.hierarchy as sciclst
import anytree
# import anytree.exporter
# Local
import clustering.clusteringalgs as clstalgs
from clustering.clusteringalgs import ClusterAnalysis
import tensor.matrixoperations as mop
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                     Interface: Cluster-reduced material phase
# =============================================================================
class CRMP(ABC):
    """Cluster-Reduced Material Phase interface.

    Methods
    -------
    perform_base_clustering(self)
        *abstract*: Perform CRMP base clustering.
    get_n_clusters(self)
        *abstract*: Get current number of clusters.
    get_clustering_type(self)
        *abstract*: Get cluster-reduced material phase adaptivity type.
    get_valid_clust_algs()
        *abstract*: Get valid clustering algorithms to compute the CRMP.
    _update_cluster_labels(labels, min_label=0)
        Update cluster labels starting with the provided minimum label.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def perform_base_clustering(self):
        """Perform CRMP base clustering."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_n_clusters(self):
        """Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_clustering_type(self):
        """Get cluster-reduced material phase adaptivity type.

        Returns
        -------
        clustering_type : str
            Type of cluster-reduced material phase.
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_valid_clust_algs():
        """Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list[str]
            Clustering algorithms identifiers (str).
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    def _update_cluster_labels(labels, min_label=0):
        """Update cluster labels starting with the provided minimum label.

        Parameters
        ----------
        labels : numpy.ndarray (1d)
            Cluster labels (numpy.ndarray[int] of shape (n_items,)).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        new_labels : numpy.ndarray (1d)
            Updated cluster labels (numpy.ndarray[int] of shape (n_items,)).
        max_label : int
            Maximum cluster label.
        """
        # Get sorted set of original labels
        unique_labels = sorted(list(set(labels)))
        # Initialize updated labels array
        new_labels = np.full(len(labels), -1, dtype=int)
        # Initialize new label
        new_label = min_label
        # Loop over sorted original labels
        for label in unique_labels:
            # Get original labels indexes
            indexes = np.where(labels == label)
            # Set updated labels
            new_labels[indexes] = new_label
            # Increment new label
            new_label += 1
        # Get maximum cluster label
        max_label = max(new_labels)
        # Check cluster updated labels
        if len(unique_labels) != len(set(new_labels)):
            raise RuntimeError('Number of clusters differs between original '
                               'and updated labels.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return new_labels, max_label
# =============================================================================
class ACRMP(CRMP):
    """Adaptive Cluster-Reduced Material Phase interface.

    Methods
    -------
    perform_adaptive_clustering(self, target_clusters, target_clusters_data)
        *abstract*: Perform ACRMP adaptive clustering step.
    get_adaptive_output(self)
        *abstract*: Get adaptivity metrics for clustering adaptivity output.
    _check_adaptivity_lock(self)
        *abstract*: Check ACRMP adaptivity locking conditions.
    get_adaptivity_type_parameters()
        *abstract*: Get ACRMP mandatory and optional adaptivity type
        parameters.
    _set_adaptivity_type_parameters(self, adaptivity_type)
        *abstract*: Set clustering adaptivity parameters.
    _dynamic_split_factor(ref_split_factor, adapt_trigger_ratio, magnitude, \
                          dynamic_amp=0)
        Compute dynamic adaptive clustering split factor.
    """
    @abstractmethod
    def perform_adaptive_clustering(self, target_clusters,
                                    target_clusters_data):
        """Perform ACRMP adaptive clustering step.

        Parameters
        ----------
        target_clusters : list[int]
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict)
            containing cluster associated parameters relevant for the adaptive
            procedures.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list[int] resulting from the
            adaptivity of each target cluster (key, str).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_adaptive_output(self):
        """Get adaptivity metrics for clustering adaptivity output.

        Returns
        -------
        adaptivity_output : list[int or float]
            List containing the adaptivity metrics associated with the
            clustering adaptivity output file.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def _check_adaptivity_lock(self):
        """Check ACRMP adaptivity locking conditions.

        Check conditions that may deactivate the adaptive procedures in the
        ACRMP. Once the ACRMP adaptivity is locked, it is treated as a SCRMP
        for the remainder of the problem numerical solution.
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_adaptivity_type_parameters():
        """Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type
        parameters, this method establishes the default values for the optional
        parameters.

        Returns
        ----------
        adapt_type_man_parameters : list[str]
            Mandatory adaptivity type parameters (str).
        adapt_type_opt_parameters : dict
            Optional adaptivity type parameters (key, str) and associated
            default value (item).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        """Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    def _dynamic_split_factor(ref_split_factor, adapt_trigger_ratio, magnitude,
                              dynamic_amp=0):
        """Compute dynamic adaptive clustering split factor.

        A detailed description of the dynamic adaptive clustering split factor
        can be found in Ferreira et. al (2022) [#]_.

        .. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
               *Adaptivity for clustering-based reduced-order modeling of
               localized history-dependent phenomena.* Comp Methods Appl M, 393
               (see `here <https://www.sciencedirect.com/science/article/pii/
               S0045782522000895?via%3Dihub>`_)

        ----

        Parameters
        ----------
        ref_split_factor : float
            Reference (centered) adaptive clustering split factor. The adaptive
            clustering split factor must be contained between 0 and 1
            (included). The lower bound (0) enforces a single split, while the
            upper bound (1) performs the maximum number splits of each cluster
            (leading to single-voxel clusters).
        adapt_trigger_ratio : float
            Threshold associated with the adaptivity trigger condition.
        magnitude : float
            Difference between cluster ratio and adaptive trigger ratio. Given
            that the cluster ratio ranges between 0 and 1 and only clusters
            with a ratio greater or equal than the adaptive trigger ratio are
            targeted, the magnitude ranges between 0 and 1 - trigger ratio.
        dynamic_amp : float, default=0
            Dynamic split factor amplitude centered around the reference
            adaptive clustering split factor.

        Returns
        -------
        adapt_split_factor : float
            Adaptive clustering split factor. The adaptive clustering split
            factor must be contained between 0 and 1 (included). The lower
            bound (0) enforces a single split, i.e., 2 new clusters, while the
            upper bound (1) is associated with a maximum defined number of new
            voxels.
        """
        # Check provided parameters
        if not (ref_split_factor >= 0 and ref_split_factor <= 1):
            raise RuntimeError('Invalid reference adaptive clustering split '
                               'factor.')
        if not (adapt_trigger_ratio >= 0 and adapt_trigger_ratio <= 1):
            raise RuntimeError('Invalid adaptive trigger ratio.')
        if not (dynamic_amp >= 0):
            raise RuntimeError('Invalid dynamic split factor amplitude.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the dynamic split factor amplitude is null, skip computations and
        # return reference adaptive clustering split factor. Otherwise, compute
        # adaptive clustering split factor lower and upper bounds
        if abs(dynamic_amp) < 1e-10:
            return ref_split_factor
        else:
            lower_bound = max(0, ref_split_factor - 0.5*dynamic_amp)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Choose dynamic function type
        dynamic_type = 'power'
        # Set dynamic function
        if dynamic_type == 'power':
            # Set dynamic function power
            n = 1.0
            # Check power admissibility
            if n < 0:
                raise RuntimeError('Dynamic function power must be greater or '
                                   'equal than zero.')
            # Power dynamic function
            dynamic_function = \
                lambda magnitude: (1/((1 - adapt_trigger_ratio)**n))*(
                    magnitude**n)
        else:
            # Linear dynamic function
            dynamic_function = \
                lambda magnitude: (1/(1 - adapt_trigger_ratio))*magnitude
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute adaptive clustering split factor
        adapt_split_factor = min(1, lower_bound
                                 + dynamic_function(magnitude)*dynamic_amp)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adapt_split_factor
#
#                                               Cluster-Reduced Material Phases
# =============================================================================
class SCRMP(CRMP):
    """Static Cluster-Reduced Material Phase (SCRMP).

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    max_label : int
        Clustering maximum label.
    cluster_labels : numpy.ndarray (1d)
        Material phase cluster labels (numpy.ndarray[int] of shape
        (n_phase_voxels,)).

    Methods
    -------
    perform_base_clustering(self, base_clustering_scheme, min_label=0)
        Perform SCRMP base clustering.
    get_n_clusters(self)
        Get current number of clusters.
    get_clustering_type(self)
        Get cluster-reduced material phase adaptivity type.
    get_valid_clust_algs()
        Get valid clustering algorithms to compute the CRMP.
    """
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters):
        """Constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: numpy.ndarray (2d)
            Data matrix (numpy.ndarray of shape
            (n_phase_voxels, n_features_dims)) containing the clustering
            features data to perform the material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        """
        self._mat_phase = mat_phase
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = 'static'
        self._n_clusters = n_clusters
        self.max_label = 0
        self.cluster_labels = None
    # -------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        """Perform SCRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        min_label : int, default=0
            Minimum cluster label.
        """
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize collection of clustering solutions
        clustering_solutions = []
        # Loop over prescribed clustering solutions
        for i in range(n_clusterings):
            # Get base clustering algorithm and check validity
            clust_alg_id = str(base_clustering_scheme[i, 0])
            if clust_alg_id not in self.get_valid_clust_algs():
                raise RuntimeError('Invalid base clustering algorithm.')
            # Get clustering features' column indexes
            indexes = base_clustering_scheme[i, 2]
            # Get base clustering data matrix
            data_matrix = mop.get_condensed_matrix(
                self._cluster_data_matrix, list(range(n_phase_voxels)),
                indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform cluster analysis
            cluster_labels, _, is_n_clusters_satisfied = \
                ClusterAnalysis().get_fitted_estimator(data_matrix,
                                                       clust_alg_id,
                                                       self._n_clusters)
            # Check if prescribed number of clusters is satisfied
            if not is_n_clusters_satisfied:
                raise RuntimeError('The number of clusters ('
                                   + str(len(set(cluster_labels)))
                                   + ') obtained is different from the '
                                   + 'prescribed number of clusters ('
                                   + str(self._n_clusters) + ').')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add clustering to collection of clustering solutions
            clustering_solutions.append(cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get consensus clustering
        if n_clusterings > 1:
            raise RuntimeError('No clustering ensemble method has been '
                               'implemented yet.')
        else:
            self.cluster_labels = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
    # -------------------------------------------------------------------------
    def get_n_clusters(self):
        """Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        """
        return self._n_clusters
    # -------------------------------------------------------------------------
    def get_clustering_type(self):
        """Get cluster-reduced material phase adaptivity type.

        Returns
        -------
        clustering_type : str
            Type of cluster-reduced material phase.
        """
        return self._clustering_type
    # -------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        """Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        """
        return list(ClusterAnalysis.available_clustering_alg.keys())
# =============================================================================
class GACRMP(ACRMP):
    """Generalized Adaptive Cluster-Reduced Material Phase.

    A detailed description of a Generalized Adaptive Cluster-Reduced Material
    Phase can be found in Ferreira et. al (2022) [#]_.

    .. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
           *Adaptivity for clustering-based reduced-order modeling of
           localized history-dependent phenomena.* Comp Methods Appl M, 393
           (see `here <https://www.sciencedirect.com/science/article/pii/
           S0045782522000895?via%3Dihub>`_)

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base
        clustering.
    _adapt_split_factor : float
        Adaptive clustering split factor. The adaptive clustering split factor
        must be contained between 0 and 1 (included). The lower bound (0)
        enforces a single split, i.e., 2 new clusters, while the upper bound
        (1) is associated with a maximum defined number of new voxels.
    _threshold_n_clusters : int
        Threshold number of adaptive material phase number of clusters. Once
        this threshold is surpassed, the adaptive procedures of the adaptive
        material phase are deactivated.
    _is_dynamic_split_factor : bool
        True if adaptive clustering split factor is to be computed dynamically.
        Otherwise, the adaptive clustering split factor is always set equal to
        `_adapt_split_factor`.
    _clustering_tree_nodes : dict
        Clustering tree node (item, anytree.Node) associated with each material
        cluster (key, str).
    _root_cluster_node : anytree.Node
        Clustering tree root node.
    max_label : int
        Clustering maximum label.
    cluster_labels : numpy.ndarray (1d)
        Material phase cluster labels (numpy.ndarray[int] of shape
        (n_phase_voxels,)).
    adaptive_time : float
        Total amount of time (s) spent in the adaptive procedures.
    adaptivity_lock : bool
        True if the adaptive procedures are deactivated, False otherwise.

    Methods
    -------
    perform_base_clustering(self, base_clustering_scheme, min_label=0)
        Perform GACRMP base clustering.
    get_valid_clust_algs():
        Get valid clustering algorithms to compute the CRMP.
    perform_adaptive_clustering(self, target_clusters, target_clusters_data, \
                                adaptive_clustering_scheme=None, min_label=0)
        Perform GACRMP adaptive clustering step.
    _check_adaptivity_lock(self)
        Check ACRMP adaptivity locking conditions.
    get_n_clusters(self)
        Get current number of clusters.
    get_clustering_type(self)
        Get cluster-reduced material phase adaptivity type.
    get_clustering_tree_nodes(self)
        Get clustering tree nodes.
    get_adaptivity_type_parameters()
        Get ACRMP mandatory and optional adaptivity type parameters.
    _set_adaptivity_type_parameters(self, adaptivity_type)
        Set clustering adaptivity parameters.
    get_adaptive_output(self)
        Get adaptivity metrics for clustering adaptivity output.
    reset_adaptive_parameters(self)
        Reset clustering adaptive progress parameters.
    update_adaptivity_type(self, adaptivity_type)
        Update clustering adaptivity parameters.
    """
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters,
                 adaptivity_type):
        """Constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: numpy.ndarray (2d)
            Data matrix (numpy.ndarray of shape
            (n_phase_voxels, n_features_dims)) containing the clustering
            features data to perform the material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        self._mat_phase = mat_phase
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = 'adaptive'
        self._n_clusters = n_clusters
        self._adaptivity_type = adaptivity_type
        self._adaptive_step = 0
        self._clustering_tree_nodes = {}
        self.max_label = 0
        self.cluster_labels = None
        self.adaptive_time = 0
        self.adaptivity_lock = False
        # Set clustering adaptivity parameters
        self._set_adaptivity_type_parameters(self._adaptivity_type)
        # Set dynamic adaptive split factor
        if abs(self._dynamic_split_factor_amp) < 1e-10:
            self._is_dynamic_split_factor = False
        else:
            self._is_dynamic_split_factor = True
        # Set clustering tree root node
        root_cluster = -1
        self._root_cluster_node = anytree.Node(-1)
        self._clustering_tree_nodes[str(root_cluster)] = \
            self._root_cluster_node
    # -------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        """Perform GACRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        min_label : int, default=0
            Minimum cluster label.
        """
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize collection of clustering solutions
        clustering_solutions = []
        # Loop over prescribed clustering solutions
        for i in range(n_clusterings):
            # Get base clustering algorithm and check validity
            clust_alg_id = str(base_clustering_scheme[i, 0])
            if clust_alg_id not in self.get_valid_clust_algs():
                raise RuntimeError('Invalid base clustering algorithm.')
            # Get clustering features' column indexes
            indexes = base_clustering_scheme[i, 2]
            # Get base clustering data matrix
            data_matrix = mop.get_condensed_matrix(
                self._cluster_data_matrix, list(range(n_phase_voxels)),
                indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform cluster analysis
            cluster_labels, _, is_n_clusters_satisfied = \
                ClusterAnalysis().get_fitted_estimator(data_matrix,
                                                       clust_alg_id,
                                                       self._n_clusters)
            # Check if prescribed number of clusters is satisfied
            if not is_n_clusters_satisfied:
                raise RuntimeError('The number of clusters ('
                                   + str(len(set(cluster_labels)))
                                   + ') obtained is different from the '
                                   + 'prescribed number of clusters ('
                                   + str(self._n_clusters) + ').')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add clustering to collection of clustering solutions
            clustering_solutions.append(cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get consensus clustering
        if n_clusterings > 1:
            raise RuntimeError('No clustering ensemble method has been '
                               'implemented yet.')
        else:
            self.cluster_labels = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over base clustering clusters
        for cluster in set(self.cluster_labels):
            # Update clustering tree
            self._clustering_tree_nodes[str(cluster)] = \
                anytree.Node(cluster, parent=self._root_cluster_node)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        """Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        """
        return list(ClusterAnalysis.available_clustering_alg.keys())
    # -------------------------------------------------------------------------
    def perform_adaptive_clustering(
        self, target_clusters, target_clusters_data,
            adaptive_clustering_scheme=None, min_label=0):
        """Perform GACRMP adaptive clustering step.

        Refine the provided target clusters by splitting them according to the
        prescribed adaptive clustering scheme.

        ----

        Parameters
        ----------
        target_clusters : list[int]
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict)
            containing cluster associated parameters required for the adaptive
            procedures.
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list[int]) resulting from the
            adaptivity of each target cluster (key, str).
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated '
                               'labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            if target_cluster not in self.cluster_labels:
                raise RuntimeError('Target cluster ' + str(target_cluster)
                                   + ' does not exist in material phase '
                                   + str(self._mat_phase))
        # Check adaptive clustering scheme
        if adaptive_clustering_scheme is None:
            raise RuntimeError('An adaptive clustering scheme must be '
                               'prescribed to perform the GACRMP clustering '
                               'adaptivity.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary
        adaptive_clustering_map = {}
        # Get material phase original clustering
        original_cluster_labels = copy.deepcopy(self.cluster_labels)
        # Initialize new cluster label
        new_cluster_label = min_label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize sorted target clusters flag
        is_sorted_target_clusters = False
        # Check if target clusters magnitude is available
        if 'max_magnitude' in \
                target_clusters_data[str(target_clusters[0])].keys():
            # Set sorted target clusters flag
            is_sorted_target_clusters = True
            # Get target clusters magnitude
            target_clusters_magnitude = {
                str(cluster):
                target_clusters_data[str(cluster)]['max_magnitude']
                for cluster in target_clusters}
            # Get target clusters in descending order of magnitude
            target_clusters_sorted = \
                [int(x[0]) for x in sorted(target_clusters_magnitude.items(),
                                           key=lambda x: x[1], reverse=True)]
            # Set sorted target clusters
            if set(target_clusters) == set(target_clusters_sorted):
                target_clusters = copy.deepcopy(target_clusters_sorted)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for i in range(len(target_clusters)):
            # Get target cluster label
            target_cluster = target_clusters[i]
            # Get total number of voxels associated with target cluster.
            # If target cluster is already single-voxeled, skip to the next
            # target cluster. Otherwise compute the number of child clusters.
            # If the target cluster number of voxels is lower or equal than the
            # number of child clusters, skip to the next target cluster.
            n_cluster_voxels = \
                np.count_nonzero(original_cluster_labels == target_cluster)
            if n_cluster_voxels == 1:
                continue
            else:
                # Get referece adaptive clustering split factor
                ref_split_factor = self._adapt_split_factor
                # Get target cluster dynamic split factor ability
                is_cluster_dynamic_split_factor = target_clusters_data[
                    str(target_cluster)]['is_dynamic_split_factor']
                # Set adaptive clustering split factor
                if self._is_dynamic_split_factor \
                        and is_cluster_dynamic_split_factor:
                    # Get adaptive trigger ratio and magnitude
                    adapt_trigger_ratio = target_clusters_data[
                        str(target_cluster)]['adapt_trigger_ratio']
                    magnitude = target_clusters_data[
                        str(target_cluster)]['max_magnitude']
                    # Compute dynamic adaptive clustering split factor
                    adapt_split_factor = super()._dynamic_split_factor(
                        ref_split_factor, adapt_trigger_ratio, magnitude,
                        dynamic_amp=self._dynamic_split_factor_amp)
                else:
                    adapt_split_factor = ref_split_factor
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute number of child clusters, enforcing at least two
                # clusters
                n_new_clusters = max(
                    2, int(round(adapt_split_factor*int(
                        round(1.0/self._child_cluster_vol_fraction)))))
                # Compare number of child clusters and number of target cluster
                # number of voxels
                if n_cluster_voxels <= n_new_clusters:
                    continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize target cluster mapping
            adaptive_clustering_map[str(target_cluster)] = []
            # Get target cluster indexes
            target_cluster_idxs = \
                list(*np.nonzero(original_cluster_labels == target_cluster))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get number of prescribed clusterings
            n_clusterings = adaptive_clustering_scheme.shape[0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize collection of clustering solutions
            clustering_solutions = []
            # Loop over prescribed clustering solutions
            for i in range(n_clusterings):
                # Get adaptive clustering algorithm and check validity
                clust_alg_id = str(adaptive_clustering_scheme[i, 0])
                if clust_alg_id not in self.get_valid_clust_algs():
                    raise RuntimeError('Invalid adaptive clustering '
                                       'algorithm.')
                # Get clustering features' column indexes
                indexes = adaptive_clustering_scheme[i, 2]
                # Get adaptive clustering data matrix
                data_matrix = mop.get_condensed_matrix(
                    self._cluster_data_matrix, target_cluster_idxs, indexes)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform cluster analysis
                cluster_labels, _, is_n_clusters_satisfied = \
                    ClusterAnalysis().get_fitted_estimator(data_matrix,
                                                           clust_alg_id,
                                                           n_new_clusters)
                # Check if prescribed number of clusters is satisfied
                if not is_n_clusters_satisfied:
                    # If the prescribed number of clusters is not satisfied,
                    # proceed with the number of clusters obtained
                    pass
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add clustering to collection of clustering solutions
                clustering_solutions.append(cluster_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get consensus clustering
            if n_clusterings > 1:
                raise RuntimeError('No clustering ensemble method has been '
                                   'implemented yet.')
            else:
                child_cluster_labels = cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update cluster labels
            child_cluster_labels, max_label = self._update_cluster_labels(
                child_cluster_labels, new_cluster_label)
            # Update new cluster label
            new_cluster_label = max_label + 1
            # Add new clusters to target cluster mapping
            adaptive_clustering_map[str(target_cluster)] += \
                list(set(child_cluster_labels))
            # Update material phase clustering
            self.cluster_labels[target_cluster_idxs] = child_cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check if number of clusters threshold has been surpassed
            if is_sorted_target_clusters:
                if len(set(self.cluster_labels)) > self._threshold_n_clusters:
                    break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update number of material phase clusters
        self._n_clusters = len(set(self.cluster_labels))
        # Update material phase maximum cluster label
        self.max_label = max(self.cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for target_cluster in adaptive_clustering_map.keys():
            # Get target cluster node
            parent_node = self._clustering_tree_nodes[str(target_cluster)]
            # Loop over child clusters
            for child_cluster in adaptive_clustering_map[target_cluster]:
                # Set child cluster tree node
                self._clustering_tree_nodes[str(child_cluster)] = \
                    anytree.Node(child_cluster, parent=parent_node)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check adaptivity threshold conditions
        self._check_adaptivity_lock()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # -------------------------------------------------------------------------
    def _check_adaptivity_lock(self):
        """Check ACRMP adaptivity locking conditions.

        Check conditions that may deactivate the adaptive procedures in the
        ACRMP. Once the ACRMP adaptivity is locked, it is treated as a SCRMP
        for the remainder of the problem numerical solution.
        """
        # Check if the number of clusters threshold as been surpassed
        if self._n_clusters > self._threshold_n_clusters:
            self.adaptivity_lock = True
    # -------------------------------------------------------------------------
    def get_n_clusters(self):
        """Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        """
        return self._n_clusters
    # -------------------------------------------------------------------------
    def get_clustering_type(self):
        """Get cluster-reduced material phase adaptivity type.

        Returns
        -------
        clustering_type : str
            Type of cluster-reduced material phase.
        """
        return self._clustering_type
    # -------------------------------------------------------------------------
    def get_clustering_tree_nodes(self):
        """Get clustering tree nodes.

        Returns
        -------
        clustering_tree_nodes : dict
            Clustering tree node (item, anytree.Node) associated with each
            material cluster (key, str).
        root_cluster_node : anytree.Node
            Clustering tree root node.
        """
        # Output clustering tree
        # anytree.exporter.DotExporter(self._root_cluster_node).to_picture(
        #    'clustering_tree_nodes_phase_' + self._mat_phase + '.png')
        return self._clustering_tree_nodes, self._root_cluster_node
    # -------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_type_parameters():
        """Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type
        parameters, this method establishes the default values for the optional
        parameters.

        ----

        Returns
        ----------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type
            (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated
            default value (item).
        """
        # Set mandatory adaptivity type parameters
        mandatory_parameters = {}
        # Set optional adaptivity type parameters and associated default values
        optional_parameters = {'adapt_split_factor': 0.01,
                               'child_cluster_vol_fraction': 0.5,
                               'dynamic_split_factor_amp': 0.0,
                               'threshold_n_clusters': 10**6}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return mandatory_parameters, optional_parameters
    # -------------------------------------------------------------------------
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        """Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        # Set mandatory adaptivity type parameters
        pass
        # Set optional adaptivity type parameters
        self._adapt_split_factor = adaptivity_type['adapt_split_factor']
        self._child_cluster_vol_fraction = \
            adaptivity_type['child_cluster_vol_fraction']
        self._dynamic_split_factor_amp = \
            adaptivity_type['dynamic_split_factor_amp']
        self._threshold_n_clusters = adaptivity_type['threshold_n_clusters']
        # Set dynamic adaptive split factor
        if abs(self._dynamic_split_factor_amp) < 1e-10:
            self._is_dynamic_split_factor = False
        else:
            self._is_dynamic_split_factor = True
    # -------------------------------------------------------------------------
    def get_adaptive_output(self):
        """Get adaptivity metrics for clustering adaptivity output.

        Returns
        -------
        adaptivity_output : list
            List containing the adaptivity metrics associated with the
            clustering adaptivity output file.
        """
        # Build adaptivity output
        adaptivity_output = [self._n_clusters, self._adaptive_step,
                             self.adaptive_time]
        # Return
        return adaptivity_output
    # -------------------------------------------------------------------------
    def reset_adaptive_parameters(self):
        """Reset clustering adaptive progress parameters."""
        # Reset counter of adaptive clustering steps
        self._adaptive_step = 0
        # Reset time spent in adaptive procedures
        self.adaptive_time = 0
    # -------------------------------------------------------------------------
    def update_adaptivity_type(self, adaptivity_type):
        """Update clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        # Update clustering adaptivity parameters
        self._adaptivity_type = adaptivity_type
        # Set clustering adaptivity parameters
        self._set_adaptivity_type_parameters(self._adaptivity_type)
# =============================================================================
class HAACRMP(ACRMP):
    """Hierarchical Agglomerative Adaptive Cluster-Reduced Material Phase.

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    _linkage_matrix : numpy.ndarray (2d)
        Linkage matrix associated with the hierarchical agglomerative
        clustering (numpy.ndarray of shape (n_phase_voxels - 1, 4)).
    _cluster_node_map : dict
        Tree node id (item, int) associated with each cluster label (key, str).
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base
        clustering.
    _adapt_split_factor : float
        Adaptive clustering split factor. The adaptive clustering split factor
        must be contained between 0 and 1 (included). The lower bound (0)
        enforces a single split, i.e., 2 new clusters, while the upper bound
        (1) is associated with a maximum defined number of new voxels.
    _threshold_n_clusters : int
        Threshold number of adaptive material phase number of clusters. Once
        this threshold is surpassed, the adaptive procedures of the adaptive
        material phase are deactivated.
    _is_dynamic_split_factor : bool
        True if adaptive clustering split factor is to be computed dynamically.
        Otherwise, the adaptive clustering split factor is always set equal to
        `_adapt_split_factor`.
    max_label : int
        Clustering maximum label.
    cluster_labels : numpy.ndarray (1d)
        Material phase cluster labels (numpy.ndarray[int] of shape
        (n_phase_voxels,)).
    adaptive_time : float
        Total amount of time (s) spent in the adaptive procedures.
    adaptivity_lock : bool
        True if the adaptive procedures are deactivated, False otherwise.

    Methods
    -------
    perform_base_clustering(self, base_clustering_scheme, min_label=0)
        Perform HAACRMP base clustering.
    perform_adaptive_clustering(self, target_clusters, target_clusters_data, \
                                adaptive_clustering_scheme=None, \
                                min_label=0)
        Perform HAACRMP adaptive clustering step.
    add_to_tree_node_list(node_list, node)
        Add node to tree node list and sort by descending linkage distance.
    _check_adaptivity_lock(self)
        Check ACRMP adaptivity locking conditions.
    print_adaptive_clustering(self, adaptive_clustering_map, \
                              adaptive_tree_node_map)
    get_valid_clust_algs()
        Get valid clustering algorithms to compute the CRMP.
    get_n_clusters(self)
        Get current number of clusters.
    get_clustering_type(self)
        Get cluster-reduced material phase adaptivity type.
    get_adaptivity_type_parameters()
        Get ACRMP mandatory and optional adaptivity type parameters.
    _set_adaptivity_type_parameters(self, adaptivity_type)
        Set clustering adaptivity parameters.
    get_adaptive_output(self)
        Get adaptivity metrics for clustering adaptivity output.
    """
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters,
                 adaptivity_type):
        """Constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: numpy.ndarray (2d)
            Data matrix (numpy.ndarray of shape
            (n_phase_voxels, n_features_dims)) containing the clustering
            features data to perform the material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        self._mat_phase = mat_phase
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = 'adaptive'
        self._n_clusters = n_clusters
        self._adaptivity_type = adaptivity_type
        self._linkage_matrix = None
        self._cluster_node_map = None
        self._adaptive_step = 0
        self.max_label = 0
        self.cluster_labels = None
        self.adaptive_time = 0
        self.adaptivity_lock = False
        # Set clustering adaptivity parameters
        self._set_adaptivity_type_parameters(self._adaptivity_type)
        # Set dynamic adaptive split factor
        if abs(self._dynamic_split_factor_amp) < 1e-10:
            self._is_dynamic_split_factor = False
        else:
            self._is_dynamic_split_factor = True
    # -------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        """Perform HAACRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        min_label : int, default=0
            Minimum cluster label.
        """
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        if n_clusterings > 1:
            raise RuntimeError('A HAACRMP only accepts a single '
                               'hierarchical agglomerative prescribed '
                               'clustering.')
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get clustering algorithm and check validity
        clust_alg_id = str(base_clustering_scheme[0, 0])
        if clust_alg_id not in self.get_valid_clust_algs():
            raise RuntimeError('An invalid clustering algorithm has been '
                               'prescribed.')
        # Get base clustering features' column indexes
        indexes = base_clustering_scheme[0, 2]
        # Get base clustering data matrix
        data_matrix = mop.get_condensed_matrix(self._cluster_data_matrix,
                                               list(range(n_phase_voxels)),
                                               indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform cluster analysis
        cluster_analysis = clstalgs.ClusterAnalysis()
        cluster_labels, clust_alg, is_n_clusters_satisfied = \
            cluster_analysis.get_fitted_estimator(data_matrix, clust_alg_id,
                                                  self._n_clusters)
        # Check if prescribed number of clusters is satisfied
        if not is_n_clusters_satisfied:
            raise RuntimeError('The number of clusters ('
                               + str(len(set(cluster_labels)))
                               + ') obtained is different from the '
                               'prescribed number of clusters ('
                               + str(self._n_clusters) + ').')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set cluster labels
        self.cluster_labels = cluster_labels
        # Get hierarchical agglomerative base clustering linkage matrix
        self._linkage_matrix = clust_alg.get_linkage_matrix()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
    # -------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters,
                                    target_clusters_data,
                                    adaptive_clustering_scheme=None,
                                    min_label=0):
        """Perform HAACRMP adaptive clustering step.

        Refine the provided target clusters by splitting them according to the
        hierarchical agglomerative tree, prioritizing child nodes by descending
        order of linkage distance.

        ----

        Parameters
        ----------
        target_clusters : list[int]
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict)
            containing cluster associated parameters relevant for the adaptive
            procedures.
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list[int]) resulting from the
            adaptivity of each target cluster (key, str).
        adaptive_tree_node_map : dict
            List of new cluster tree node ids (item, list[int]) resulting from
            the split of each target cluster tree node id (key, str).
            Validation purposes only (not returned otherwise).
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated '
                               'labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            if target_cluster not in self.cluster_labels:
                raise RuntimeError('Target cluster ' + str(target_cluster)
                                   + ' does not exist in material phase '
                                   + str(self._mat_phase))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary
        adaptive_clustering_map = {}
        # Initialize adaptive tree node mapping dictionary (validation purposes
        # only)
        adaptive_tree_node_map = {}
        # Get current hierarchical agglomerative clustering
        cluster_labels = copy.deepcopy(self.cluster_labels)
        # Initialize new cluster label
        new_cluster_label = min_label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get cluster labels (conversion to int32 is required to avoid raising
        # a TypeError in scipy hierarchy leaders function)
        labels = cluster_labels.astype('int32')
        # Convert hierarchical agglomerative base clustering linkage matrix
        # into tree object
        rootnode, nodelist = sciclst.to_tree(self._linkage_matrix, rd=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initial cluster-node mapping between cluster labels and tree
        # nodes associated with the hierarchical agglomerative base clustering
        if self._cluster_node_map is None:
            # Get root nodes of hierarchical clustering corresponding to an
            # horizontal cut defined by a flat clustering assignment vector.
            # L contains the tree nodes ids while M contains the corresponding
            # cluster labels
            L, M = sciclst.leaders(self._linkage_matrix, labels)
            # Build initial cluster-node mapping
            self._cluster_node_map = dict(zip([str(x) for x in M], L))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for i in range(len(target_clusters)):
            # Get target cluster label
            target_cluster = target_clusters[i]
            # Get target cluster tree node instance
            target_node = nodelist[self._cluster_node_map[str(target_cluster)]]
            # Get total number of leaf nodes associated with target node. If
            # target node is a leaf itself (not splitable), skip to the next
            # target cluster
            if target_node.is_leaf():
                continue
            # else:
            #     n_leaves = target_node.get_count()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get referece adaptive clustering split factor
            ref_split_factor = self._adapt_split_factor
            # Get target cluster dynamic split factor ability
            is_cluster_dynamic_split_factor = target_clusters_data[
                str(target_cluster)]['is_dynamic_split_factor']
            # Set adaptive clustering split factor
            if self._is_dynamic_split_factor \
                    and is_cluster_dynamic_split_factor:
                # Get adaptive trigger ratio and magnitude
                adapt_trigger_ratio = target_clusters_data[
                    str(target_cluster)]['adapt_trigger_ratio']
                magnitude = target_clusters_data[
                    str(target_cluster)]['max_magnitude']
                # Compute dynamic adaptive clustering split factor
                adapt_split_factor = super()._dynamic_split_factor(
                    ref_split_factor, adapt_trigger_ratio, magnitude,
                    dynamic_amp=self._dynamic_split_factor_amp)
            else:
                adapt_split_factor = ref_split_factor
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute total number of tree node splits, enforcing at least one
            # split
            n_splits = max(
                1, int(round(adapt_split_factor*int(int(
                    round(1.0/self._child_cluster_vol_fraction)) - 1))))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize child nodes list
            child_nodes = []
            # Initialize child target nodes list
            child_target_nodes = []
            # Split loop
            for i_split in range(n_splits):
                # Set node to be splitted
                if i_split == 0:
                    # In the first split operation, the node to be splitted is
                    # the target cluster tree node
                    node_to_split = target_node
                else:
                    # Get maximum linkage distance child target node and remove
                    # it from the child target nodes list
                    node_to_split = child_target_nodes[0]
                    child_target_nodes.pop(0)
                # Loop over child target node's left and right child nodes
                for node in [node_to_split.get_left(),
                             node_to_split.get_right()]:
                    if node.is_leaf():
                        # Append to child nodes list if leaf node
                        child_nodes.append(node)
                    else:
                        # Append to child target nodes list if non-leaf node
                        child_target_nodes = self.add_to_tree_node_list(
                            child_target_nodes, node)
            # Add remaining child target nodes to child nodes list
            child_nodes += child_target_nodes
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize target cluster mapping
            adaptive_clustering_map[str(target_cluster)] = []
            # Remove target cluster from cluster node mapping
            self._cluster_node_map.pop(str(target_cluster))
            # Update target cluster mapping and flat clustering labels
            for node in child_nodes:
                # Add new cluster to target cluster mapping
                adaptive_clustering_map[str(target_cluster)].append(
                    new_cluster_label)
                # Update flat clustering labels
                labels[node.pre_order()] = new_cluster_label
                # Update cluster-node mapping
                self._cluster_node_map[str(new_cluster_label)] = node.id
                # Increment new cluster label
                new_cluster_label += 1
            # Update adaptive tree node mapping dictionary (validation purposes
            # only)
            adaptive_tree_node_map[str(target_node.id)] = \
                [x.id for x in child_nodes]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update RVE hierarchical agglomerative clustering
        self.cluster_labels = labels
        # Update number of material phase clusters
        self._n_clusters = len(set(self.cluster_labels))
        # Update clustering maximum label
        self.max_label = max(self.cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check adaptivity threshold conditions
        self._check_adaptivity_lock()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # -------------------------------------------------------------------------
    @staticmethod
    def add_to_tree_node_list(node_list, node):
        """Add node to tree node list and sort by descending linkage distance.

        Parameters
        ----------
        node_list : list[ClusterNode]
            List of ClusterNode instances.
        node : ClusterNode
            ClusterNode to be added to list[ClusterNode].
        """
        # Check parameters
        if not isinstance(node, sciclst.ClusterNode):
            raise TypeError('Node must be of type ClusterNode, not '
                            + str(type(node)) + '.')
        if any([not isinstance(node, sciclst.ClusterNode)
                for node in node_list]):
            raise TypeError('Node list can only contain elements of the type '
                            'ClusterNode.')
        # Append tree node to node list
        node_list.append(node)
        # Sort tree node list by descending order of linkage distance
        node_list = sorted(node_list, reverse=True, key=lambda x: x.dist)
        # Return sorted tree node list
        return node_list
    # -------------------------------------------------------------------------
    def _check_adaptivity_lock(self):
        """Check ACRMP adaptivity locking conditions.

        Check conditions that may deactivate the adaptive procedures in the
        ACRMP. Once the ACRMP adaptivity is locked, it is treated as a SCRMP
        for the remainder of the problem numerical solution.
        """
        # Check if the number of clusters threshold as been surpassed
        if self._n_clusters > self._threshold_n_clusters:
            self.adaptivity_lock = True
    # -------------------------------------------------------------------------
    def print_adaptive_clustering(self, adaptive_clustering_map,
                                  adaptive_tree_node_map):
        """Print hierarchical adaptive clustering report (validation)."""
        # Print report header
        print(3*'\n' + 'Hierarchical adaptive clustering report\n' + 80*'-'
              + '\n\n' + 'Material phase: ' + str(self._mat_phase))
        # Print adaptive clustering adaptive step
        print('\nAdaptive refinement step: ', self._adaptive_step)
        # Print hierarchical adaptive CRVE
        print('\n\n' + 'Adaptive clustering: ' + '('
              + str(len(np.unique(self.cluster_labels))) + ' clusters)'
              + '\n\n', self.cluster_labels)
        # Print adaptive clustering mapping
        print('\n\n' + 'Adaptive cluster mapping: ')
        for old_cluster in adaptive_clustering_map.keys():
            print('    Old cluster: ' + '{:>4s}'.format(old_cluster)
                  + '  ->  '
                  + 'New clusters: ',
                  adaptive_clustering_map[str(old_cluster)])
        # Print adaptive tree node mapping
        print('\n\n' + 'Adaptive tree node mapping (validation): ')
        for old_node in adaptive_tree_node_map.keys():
            print('  Old node: ' + '{:>4s}'.format(old_node)
                  + '  ->  '
                  + 'New nodes: ', adaptive_tree_node_map[str(old_node)])
        # Print cluster-node mapping
        print('\n\n' + 'Cluster-Node mapping: ')
        for new_cluster in self._cluster_node_map.keys():
            print('    Cluster: ' + '{:>4s}'.format(new_cluster)
                  + '  ->  '
                  + 'Tree node: ',
                  self._cluster_node_map[str(new_cluster)])
    # -------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        """Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list[str]
            Clustering algorithms identifiers (str).
        """
        return ['3', ]
    # -------------------------------------------------------------------------
    def get_n_clusters(self):
        """Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        """
        return self._n_clusters
    # -------------------------------------------------------------------------
    def get_clustering_type(self):
        """Get cluster-reduced material phase adaptivity type.

        Returns
        -------
        clustering_type : str
            Type of cluster-reduced material phase.
        """
        return self._clustering_type
    # -------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_type_parameters():
        """Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type
        parameters, this method establishes the default values for the optional
        parameters.

        ----

        Returns
        ----------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type
            (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated
            default value (item).
        """
        # Set mandatory adaptivity type parameters
        mandatory_parameters = {}
        # Set optional adaptivity type parameters and associated default values
        optional_parameters = {'adapt_split_factor': 0.01,
                               'child_cluster_vol_fraction': 0.5,
                               'dynamic_split_factor_amp': 0.0,
                               'threshold_n_clusters': 10**6}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return mandatory_parameters, optional_parameters
    # -------------------------------------------------------------------------
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        """Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        """
        # Set mandatory adaptivity type parameters
        pass
        # Set optional adaptivity type parameters
        self._adapt_split_factor = adaptivity_type['adapt_split_factor']
        self._child_cluster_vol_fraction = \
            adaptivity_type['child_cluster_vol_fraction']
        self._dynamic_split_factor_amp = \
            adaptivity_type['dynamic_split_factor_amp']
        self._threshold_n_clusters = adaptivity_type['threshold_n_clusters']
    # -------------------------------------------------------------------------
    def get_adaptive_output(self):
        """Get adaptivity metrics for clustering adaptivity output.

        Returns
        -------
        adaptivity_output : list
            List containing the adaptivity metrics associated with the
            clustering adaptivity output file.
        """
        # Build adaptivity output
        adaptivity_output = [self._n_clusters, self._adaptive_step,
                             self.adaptive_time]
        # Return
        return adaptivity_output
