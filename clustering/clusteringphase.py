#
# Cluster-Reduced Material Phases Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the cluster analysis of each material phase.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Dec 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Defining abstract base classes
from abc import ABC, abstractmethod
# Date and time
import time
# Shallow and deep copy operations
import copy
# Unsupervised clustering algorithms
import clustering.clusteringalgs as clstalgs
from clustering.clusteringalgs import ClusterAnalysis
import scipy.cluster.hierarchy as sciclst
# Matricial operations
import tensor.matrixoperations as mop
#                                                   Cluster-Reduced material phase interface
# ==========================================================================================
class CRMP(ABC):
    '''Cluster-Reduced Material Phase interface.'''
    @abstractmethod
    def __init__(self):
        '''Cluster-Reduced Material Phase constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def perform_base_clustering(self):
        '''Perform CRMP base clustering.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_n_clusters(self):
        '''Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_valid_clust_algs():
        '''Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _update_cluster_labels(labels, min_label=0):
        '''Update cluster labels starting with the provided minimum label.

        Parameters
        ----------
        labels : ndarray of shape (n_items,)
            List of cluster labels (int).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        new_labels : ndarray of shape (n_items,)
            Updated cluster labels (int).
        max_label : int
            Maximum cluster label.
        '''
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
            raise RuntimeError('Number of clusters differs between original and updated '
                               'labels.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [new_labels, max_label]
# ------------------------------------------------------------------------------------------
class ACRMP(CRMP):
    '''Adaptive Cluster-Reduced Material Phase interface.'''
    @abstractmethod
    def perform_adaptive_clustering(self, target_clusters, target_clusters_data):
        '''Perform ACRMP adaptive clustering step.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters relevant for the adaptive procedures.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the adaptivity of
            each target cluster (key, str).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_adaptive_output(self):
        '''Get adaptivity metrics for clustering adaptivity output.

        Returns
        -------
        adaptivity_output : list
            List containing the adaptivity metrics associated to the clustering adaptivity
            output file.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def _check_adaptivity_lock(self):
        '''Check ACRMP adaptivity locking conditions.

        Check conditions that may deactivate the adaptive procedures in the ACRMP. Once the
        ACRMP adaptivity is locked, it is treated as a SCRMP for the remainder of the
        problem numerical solution.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_adaptivity_type_parameters():
        '''Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type parameters, this
        method establishes the default values for the optional parameters.

        Returns
        ----------
        adapt_type_man_parameters : list
            Mandatory adaptivity type parameters (str).
        adapt_type_opt_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        '''Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        '''
        pass
#
#                                                      Static Cluster-Reduced Material Phase
# ==========================================================================================
class SCRMP(CRMP):
    '''Static Cluster-Reduced Material Phase.

    This class provides all the required attributes and methods associated with the
    generation of a Static Cluster-Reduced Material Phase (SCRMP).

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    max_label : int
        Clustering maximum label.
    cluster_labels : ndarray of shape (n_phase_voxels,)
        Material phase cluster labels.
    '''
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters):
        '''Static Cluster-Reduced Representative Volume Element constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: ndarray of shape (n_phase_voxels, n_features_dims)
            Data matrix containing the required clustering features' data to perform the
            material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        '''
        self._mat_phase = mat_phase
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = 'static'
        self._n_clusters = n_clusters
        self.max_label = 0
        self.cluster_labels = None
    # --------------------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        '''Perform SCRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed clustering scheme to perform the material phase base cluster
            analysis. Each row is associated with a unique clustering, characterized by a
            clustering algorithm (col 1, int), a list of features (col 2, list of int) and a
            list of the cluster data matrix' indexes (col 3, list of int).
        min_label : int, default=0
            Minimum cluster label.
        '''
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            data_matrix = mop.getcondmatrix(self._cluster_data_matrix,
                                            list(range(n_phase_voxels)), indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform cluster analysis
            cluster_labels, _ = ClusterAnalysis().get_fitted_estimator(data_matrix,
                                                                       clust_alg_id,
                                                                       self._n_clusters)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add clustering to collection of clustering solutions
            clustering_solutions.append(cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get consensus clustering
        if n_clusterings > 1:
            raise RuntimeError('No clustering ensemble method has been implemented yet.')
        else:
            self.cluster_labels = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
    # --------------------------------------------------------------------------------------
    def get_n_clusters(self):
        '''Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        '''
        return self._n_clusters
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        '''Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        '''
        return list(ClusterAnalysis.available_clustering_alg.keys())
#
#                                        Generalized Adaptive Cluster-Reduced Material Phase
# ==========================================================================================
class GACRMP(ACRMP):
    '''Generalized Adaptive Cluster-Reduced Material Phase.

    This class provides all the required attributes and methods associated with the
    generation and update of a Generalized Adaptive Cluster-Reduced Material Phase
    (GACRMP).

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base clustering.
    _adapt_split_factor : float
        Adaptive clustering split factor. The adaptive clustering split factor must be
        contained between 0 and 1 (included). The lower bound (0) enforces a single
        split, while the upper bound (1) performs the maximum number splits of
        each cluster (leading to single-voxel clusters).
    _is_dynamic_split_factor : bool
        True if adaptive clustering split factor is to be computed dynamically. Otherwise,
        the adaptive clustering split factor is always set equal to _adapt_split_factor.
    max_label : int
        Clustering maximum label.
    cluster_labels : ndarray of shape (n_phase_voxels,)
        Material phase cluster labels.
    adaptive_time : float
        Total amount of time (s) spent in the adaptive procedures.
    adaptivity_lock : bool
        True if the adaptive procedures are deactivated, False otherwise.
    '''
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters, adaptivity_type):
        '''Generalized Adaptive Cluster-Reduced Representative Volume Element constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: ndarray of shape (n_phase_voxels, n_features_dims)
            Data matrix containing the required clustering features' data to perform the
            material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        adaptivity_type : dict
            Clustering adaptivity parameters.
        '''
        self._mat_phase = mat_phase
        self._cluster_data_matrix = cluster_data_matrix
        self._clustering_type = 'adaptive'
        self._n_clusters = n_clusters
        self._adaptivity_type = adaptivity_type
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
    # --------------------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        '''Perform GACRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed clustering scheme to perform the material phase base cluster
            analysis. Each row is associated with a unique clustering, characterized by a
            clustering algorithm (col 1, int), a list of features (col 2, list of int) and a
            list of the cluster data matrix' indexes (col 3, list of int).
        min_label : int, default=0
            Minimum cluster label.
        '''
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            data_matrix = mop.getcondmatrix(self._cluster_data_matrix,
                                            list(range(n_phase_voxels)), indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform cluster analysis
            cluster_labels, _ = ClusterAnalysis().get_fitted_estimator(data_matrix,
                                                                       clust_alg_id,
                                                                       self._n_clusters)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add clustering to collection of clustering solutions
            clustering_solutions.append(cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get consensus clustering
        if n_clusterings > 1:
            raise RuntimeError('No clustering ensemble method has been implemented yet.')
        else:
            self.cluster_labels = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        '''Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        '''
        return list(ClusterAnalysis.available_clustering_alg.keys())
    # --------------------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters, target_clusters_data,
                                    adaptive_clustering_scheme=None, min_label=0):
        '''Perform GACRMP adaptive clustering step.

        Refine the provided target clusters by splitting them according to the prescribed
        adaptive clustering scheme.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters required for the adaptive procedures.
        adaptive_clustering_scheme : ndarray of shape (n_clusterings, 3), default=None
            Prescribed adaptive clustering scheme to perform the material phase adaptive
            cluster analysis. Each row is associated with a unique clustering, characterized
            by a clustering algorithm (col 1, int), a list of features (col 2, list of int)
            and a list of the cluster data matrix' indexes (col 3, list of int).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the adaptivity of
            each target cluster (key, str).
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            if target_cluster not in self.cluster_labels:
                raise RuntimeError('Target cluster ' + str(target_cluster) + ' does not ' +
                                   'exist in material phase ' + str(self._mat_phase))
        # Check adaptive clustering scheme
        if adaptive_clustering_scheme is None:
            raise RuntimeError('An adaptive clustering scheme must be prescribed to ' +
                               'perform the GACRMP clustering adaptivity.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary
        adaptive_clustering_map = {}
        # Get material phase original clustering
        original_cluster_labels = copy.deepcopy(self.cluster_labels)
        # Initialize new cluster label
        new_cluster_label = min_label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for i in range(len(target_clusters)):
            # Get target cluster label
            target_cluster = target_clusters[i]
            # Get total number of voxels associated to target cluster. If target cluster is
            # already single-voxeled, skip to the next target cluster. Otherwise compute the
            # number of child clusters. If the target cluster number of voxels is lower or
            # equal than the number of child clusters, skip to the next target cluster.
            n_cluster_voxels = np.count_nonzero(original_cluster_labels == target_cluster)
            if n_cluster_voxels == 1:
                continue
            else:
                # Get referece adaptive clustering split factor
                ref_split_factor = self._adapt_split_factor
                # Set adaptive clustering split factor
                if self._is_dynamic_split_factor:
                    # Get adaptive trigger ratio and magnitude
                    adapt_trigger_ratio = \
                        target_clusters_data[str(target_cluster)]['adapt_trigger_ratio']
                    magnitude = \
                        target_clusters_data[str(target_cluster)]['max_magnitude']
                    # Compute dynamic adaptive clustering split factor
                    adapt_split_factor = self._dynamic_split_factor(ref_split_factor,
                        adapt_trigger_ratio, magnitude,
                        dynamic_amp=self._dynamic_split_factor_amp)
                else:
                    adapt_split_factor = ref_split_factor
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute number of child clusters, enforcing at least two clusters
                n_new_clusters = \
                    max(2, int(np.round(adapt_split_factor*n_cluster_voxels)))
                # Compare number of child clusters and number of target cluster number of
                # voxels
                if n_cluster_voxels <= n_new_clusters:
                    continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize target cluster mapping
            adaptive_clustering_map[str(target_cluster)] = []
            # Get target cluster indexes
            target_cluster_idxs = \
                list(*np.nonzero(original_cluster_labels == target_cluster))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get number of prescribed clusterings
            n_clusterings = adaptive_clustering_scheme.shape[0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize collection of clustering solutions
            clustering_solutions = []
            # Loop over prescribed clustering solutions
            for i in range(n_clusterings):
                # Get adaptive clustering algorithm and check validity
                clust_alg_id = str(adaptive_clustering_scheme[i, 0])
                if clust_alg_id not in self.get_valid_clust_algs():
                    raise RuntimeError('Invalid adaptive clustering algorithm.')
                # Get clustering features' column indexes
                indexes = adaptive_clustering_scheme[i, 2]
                # Get adaptive clustering data matrix
                data_matrix = mop.getcondmatrix(self._cluster_data_matrix,
                                                target_cluster_idxs, indexes)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform cluster analysis
                cluster_labels, _ = \
                    ClusterAnalysis().get_fitted_estimator(data_matrix, clust_alg_id,
                                                           n_new_clusters)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add clustering to collection of clustering solutions
                clustering_solutions.append(cluster_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get consensus clustering
            if n_clusterings > 1:
                raise RuntimeError('No clustering ensemble method has been implemented yet.')
            else:
                child_cluster_labels = cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update cluster labels
            child_cluster_labels, max_label = \
                self._update_cluster_labels(child_cluster_labels, new_cluster_label)
            # Update new cluster label
            new_cluster_label = max_label + 1
            # Add new clusters to target cluster mapping
            adaptive_clustering_map[str(target_cluster)] += list(set(child_cluster_labels))
            # Update material phase clustering
            self.cluster_labels[target_cluster_idxs] = child_cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update number of material phase clusters
        self._n_clusters = len(set(self.cluster_labels))
        # Update material phase maximum cluster label
        self.max_label = max(self.cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check adaptivity threshold conditions
        self._check_adaptivity_lock()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _dynamic_split_factor(ref_split_factor, adapt_trigger_ratio, magnitude,
                              dynamic_amp=0):
        '''Compute dynamic adaptive clustering split factor.

        Parameters
        ----------
        ref_split_factor : float
            Reference (centered) adaptive clustering split factor. The adaptive clustering
            split factor must be contained between 0 and 1 (included). The lower bound (0)
            enforces a single split, while the upper bound (1) performs the maximum number
            splits of each cluster (leading to single-voxel clusters).
        adapt_trigger_ratio : float
            Threshold associated to the adaptivity trigger condition.
        magnitude : float
            Difference between cluster ratio and adaptive trigger ratio. Given that the
            cluster ratio ranges between 0 and 1 and only clusters with a ratio greater or
            equal than the adaptive trigger ratio are targeted, the magnitude ranges between
            0 and 1 - trigger ratio.
        dynamic_amp : float, default=0
            Dynamic split factor amplitude centered around the reference adaptive clustering
            split factor.

        Returns
        -------
        adapt_split_factor : float
            Adaptive clustering split factor. The adaptive clustering split factor must be
            contained between 0 and 1 (included). The lower bound (0) enforces a single
            split, while the upper bound (1) performs the maximum number splits of
            each cluster (leading to single-voxel clusters).
        '''
        # Check provided parameters
        if not (ref_split_factor >= 0 and ref_split_factor <= 1):
            raise RuntimeError('Invalid reference adaptive clustering split factor.')
        if not (adapt_trigger_ratio >= 0 and adapt_trigger_ratio <= 1):
            raise RuntimeError('Invalid adaptive trigger ratio.')
        if not (dynamic_amp >= 0):
            raise RuntimeError('Invalid dynamic split factor amplitude.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # If the dynamic split factor amplitude is null, skip computations and return
        # reference adaptive clustering split factor. Otherwise, compute adaptive clustering
        # split factor lower bound
        if abs(dynamic_amp) < 1e-10:
            return ref_split_factor
        else:
            lower_bound = max(0, ref_split_factor - 0.5*dynamic_amp)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Choose dynamic function type
        dynamic_type = 'linear'
        # Set dynamic function
        if dynamic_type == 'linear':
            # Linear dynamic function
            dynamic_function = lambda magnitude : (1/(1 - adapt_trigger_ratio))*magnitude
        elif dynamic_type == 'quadratic':
            # Quadratic dynamic function
            dynamic_function = \
                lambda magnitude : (1/((1 - adapt_trigger_ratio)**2))*(magnitude**2)
        else:
            raise RuntimeError('Unknown dynamic function type.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute adaptive clustering split factor
        adapt_split_factor = lower_bound + dynamic_function(magnitude)*dynamic_amp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adapt_split_factor
    # --------------------------------------------------------------------------------------
    def _check_adaptivity_lock(self):
        '''Check ACRMP adaptivity locking conditions.

        Check conditions that may deactivate the adaptive procedures in the ACRMP. Once the
        ACRMP adaptivity is locked, it is treated as a SCRMP for the remainder of the
        problem numerical solution.
        '''
        # Check if the number of clusters threshold as been surpassed
        if self._n_clusters > self._threshold_n_clusters:
            self.adaptivity_lock = True
    # --------------------------------------------------------------------------------------
    def get_n_clusters(self):
        '''Get current number of clusters.

        Returns
        -------
        n_clusters : int
            Number of material phase clusters.
        '''
        return self._n_clusters
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_type_parameters():
        '''Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type parameters, this
        method establishes the default values for the optional parameters.

        Returns
        ----------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        # Set mandatory adaptivity type parameters
        mandatory_parameters = {}
        # Set optional adaptivity type parameters and associated default values
        optional_parameters = {'adapt_split_factor': 0.01,
                               'dynamic_split_factor_amp': 0.0,
                               'threshold_n_clusters': 10**6}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [mandatory_parameters, optional_parameters]
    # --------------------------------------------------------------------------------------
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        '''Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        '''
        # Set mandatory adaptivity type parameters
        pass
        # Set optional adaptivity type parameters
        self._adapt_split_factor = adaptivity_type['adapt_split_factor']
        self._dynamic_split_factor_amp = adaptivity_type['dynamic_split_factor_amp']
        self._threshold_n_clusters = adaptivity_type['threshold_n_clusters']
    # --------------------------------------------------------------------------------------
    def get_adaptive_output(self):
        '''Get adaptivity metrics for clustering adaptivity output.

        Returns
        -------
        adaptivity_output : list
            List containing the adaptivity metrics associated to the clustering adaptivity
            output file.
        '''
        # Build adaptivity output
        adaptivity_output = [self._n_clusters, self._adaptive_step, self.adaptive_time]
        # Return
        return adaptivity_output
#
#                         Hierarchical Agglomerative Adaptive Cluster-Reduced Material Phase
# ==========================================================================================
class HAACRMP(CRMP):
    '''Hierarchical Agglomerative Adaptive Cluster-Reduced Material Phase.

    This class provides all the required attributes and methods associated with the
    generation and update of a Hierarchical Agglomerative Adaptive Cluster-Reduced Material
    Phase (HAACRMP).

    Attributes
    ----------
    _clustering_type : str
        Type of cluster-reduced material phase.
    _linkage_matrix : ndarray of shape (n_phase_voxels-1, 4)
        Linkage matrix associated with the hierarchical agglomerative clustering.
    _cluster_node_map : dict
        Tree node id (item, int) associated to each cluster label (key, str).
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base clustering.
    _adapt_split_factor : float
        Adaptive clustering split factor. The adaptive clustering split factor must be
        contained between 0 and 1 (included). The lower bound (0) enforces a single
        split, while the upper bound (1) performs the maximum number splits of
        each cluster (leading to single-voxel clusters).
    max_label : int
        Clustering maximum label.
    cluster_labels : ndarray of shape (n_phase_voxels,)
        Material phase cluster labels.
    adaptive_time : float
        Total amount of time (s) spent in the adaptive procedures.
    '''
    def __init__(self, mat_phase, cluster_data_matrix, n_clusters, adaptivity_type):
        '''Hierarchical Agglomerative Adaptive Cluster-Reduced Material Phase constructor.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        cluster_data_matrix: ndarray of shape (n_phase_voxels, n_features_dims)
            Data matrix containing the required clustering features' data to perform the
            material phase cluster analyses.
        n_clusters : int
            Number of material phase clusters.
        adaptivity_type : dict
            Clustering adaptivity parameters.
        '''
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
        # Set clustering adaptivity parameters
        self._set_adaptivity_type_parameters(self._adaptivity_type)
    # --------------------------------------------------------------------------------------
    def perform_base_clustering(self, base_clustering_scheme, min_label=0):
        '''Perform HAACRMP base clustering.

        Parameters
        ----------
        base_clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed clustering scheme to perform the material phase base cluster
            analysis. Each row is associated with a unique clustering, characterized by a
            clustering algorithm (col 1, int), a list of features (col 2, list of int) and a
            list of the cluster data matrix' indexes (col 3, list of int).
        min_label : int, default=0
            Minimum cluster label.
        '''
        # Get number of prescribed clusterings
        n_clusterings = base_clustering_scheme.shape[0]
        if n_clusterings > 1:
            raise RuntimeError('A HAA-CRMP only accepts a single hierarchical ' +
                               'agglomerative prescribed clustering.')
        # Get number of material phase voxels
        n_phase_voxels = self._cluster_data_matrix.shape[0]
        # Get clustering algorithm and check validity
        clust_alg_id = str(base_clustering_scheme[0, 0])
        if clust_alg_id not in self.get_valid_clust_algs():
            raise RuntimeError('An invalid clustering algorithm has been prescribed.')
        # Get base clustering features' column indexes
        indexes = base_clustering_scheme[0, 2]
        # Get base clustering data matrix
        data_matrix = mop.getcondmatrix(self._cluster_data_matrix,
                                        list(range(n_phase_voxels)), indexes)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform cluster analysis
        cluster_analysis = clstalgs.ClusterAnalysis()
        cluster_labels, clust_alg = cluster_analysis.get_fitted_estimator(data_matrix,
                                                                          clust_alg_id,
                                                                          self._n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set cluster labels
        self.cluster_labels = cluster_labels
        # Get hierarchical agglomerative base clustering linkage matrix
        self._linkage_matrix = clust_alg.get_linkage_matrix()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster labels
        self.cluster_labels, self.max_label = \
            self._update_cluster_labels(self.cluster_labels, min_label)
    # --------------------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters, adaptive_clustering_scheme=None,
                                    min_label=0):
        '''Perform HAACRMP adaptive clustering step.

        Refine the provided target clusters by splitting them according to the hierarchical
        agglomerative tree, prioritizing child nodes by descending order of linkage
        distance.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be adapted.
        adaptive_clustering_scheme : ndarray of shape (n_clusterings, 3), default=None
            Prescribed adaptive clustering scheme to perform the material phase adaptive
            cluster analysis. Each row is associated with a unique clustering, characterized
            by a clustering algorithm (col 1, int), a list of features (col 2, list of int)
            and a list of the cluster data matrix' indexes (col 3, list of int).
        min_label : int, default=0
            Minimum cluster label.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the adaptivity of
            each target cluster (key, str).
        adaptive_tree_node_map : dict
            List of new cluster tree node ids (item, list of int) resulting from the split
            of each target cluster tree node id (key, str).
            Validation purposes only (not returned otherwise).
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            if target_cluster not in self.cluster_labels:
                raise RuntimeError('Target cluster ' + str(target_cluster) + ' does not ' +
                                   'exist in material phase ' + str(self._mat_phase))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary
        adaptive_clustering_map = {}
        # Initialize adaptive tree node mapping dictionary (validation purposes only)
        adaptive_tree_node_map = {}
        # Get current hierarchical agglomerative clustering
        cluster_labels = copy.deepcopy(self.cluster_labels)
        # Initialize new cluster label
        new_cluster_label = min_label
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get cluster labels (conversion to int32 is required to avoid raising a TypeError
        # in scipy hierarchy leaders function)
        labels = cluster_labels.astype('int32')
        # Convert hierarchical agglomerative base clustering linkage matrix into tree object
        rootnode, nodelist = sciclst.to_tree(self._linkage_matrix, rd=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build initial cluster-node mapping between cluster labels and tree nodes
        # associated with the hierarchical agglomerative base clustering
        if self._cluster_node_map == None:
            # Get root nodes of hierarchical clustering corresponding to an horizontal cut
            # defined by a flat clustering assignment vector. L contains the tree nodes ids
            # while M contains the corresponding cluster labels
            L, M = sciclst.leaders(self._linkage_matrix, labels)
            # Build initial cluster-node mapping
            self._cluster_node_map = dict(zip([str(x) for x in M], L))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for i in range(len(target_clusters)):
            # Get target cluster label
            target_cluster = target_clusters[i]
            # Get target cluster tree node instance
            target_node = nodelist[self._cluster_node_map[str(target_cluster)]]
            # Get total number of leaf nodes associated to target node. If target node is a
            # leaf itself (not splitable), skip to the next target cluster
            if target_node.is_leaf():
                continue
            else:
                n_leaves = target_node.get_count()
            # Compute total number of tree node splits, enforcing at least one split
            n_splits = max(1, int(np.round(self._adapt_split_factor*(n_leaves - 1))))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize child nodes list
            child_nodes = []
            # Initialize child target nodes list
            child_target_nodes = []
            # Split loop
            for i_split in range(n_splits):
                # Set node to be splitted
                if i_split == 0:
                    # In the first split operation, the node to be splitted is the target
                    # cluster tree node
                    node_to_split = target_node
                else:
                    # Get maximum linkage distance child target node and remove it from the
                    # child target nodes list
                    node_to_split = child_target_nodes[0]
                    child_target_nodes.pop(0)
                # Loop over child target node's left and right child nodes
                for node in [node_to_split.get_left(), node_to_split.get_right()]:
                    if node.is_leaf():
                        # Append to child nodes list if leaf node
                        child_nodes.append(node)
                    else:
                        # Append to child target nodes list if non-leaf node
                        child_target_nodes = \
                            self.add_to_tree_node_list(child_target_nodes, node)
            # Add remaining child target nodes to child nodes list
            child_nodes += child_target_nodes
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize target cluster mapping
            adaptive_clustering_map[str(target_cluster)] = []
            # Remove target cluster from cluster node mapping
            self._cluster_node_map.pop(str(target_cluster))
            # Update target cluster mapping and flat clustering labels
            for node in child_nodes:
                # Add new cluster to target cluster mapping
                adaptive_clustering_map[str(target_cluster)].append(new_cluster_label)
                # Update flat clustering labels
                labels[node.pre_order()] = new_cluster_label
                # Update cluster-node mapping
                self._cluster_node_map[str(new_cluster_label)] = node.id
                # Increment new cluster label
                new_cluster_label += 1
            # Update adaptive tree node mapping dictionary (validation purposes only)
            adaptive_tree_node_map[str(target_node.id)] = [x.id for x in child_nodes]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update RVE hierarchical agglomerative clustering
        self.cluster_labels = labels
        # Update clustering maximum label
        self.max_label = max(self.cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # --------------------------------------------------------------------------------------
    @staticmethod
    def add_to_tree_node_list(node_list, node):
        '''Add node to tree node list and sort it by descending order of linkage distance.

        Parameters
        ----------
        node_list : list of ClusterNode
            List of ClusterNode instances.
        node : ClusterNode
            ClusterNode to be added to list of ClusterNode.
        '''
        # Check parameters
        if not isinstance(node, sciclst.ClusterNode):
            raise TypeError('Node must be of type ClusterNode, not ' + str(type(node)) +
                            '.')
        if any([not isinstance(node, sciclst.ClusterNode) for node in node_list]):
            raise TypeError('Node list can only contain elements of the type ClusterNode.')
        # Append tree node to node list
        node_list.append(node)
        # Sort tree node list by descending order of linkage distance
        node_list = sorted(node_list, reverse=True, key=lambda x: x.dist)
        # Return sorted tree node list
        return node_list
    # --------------------------------------------------------------------------------------
    def print_adaptive_clustering(self, adaptive_clustering_map, adaptive_tree_node_map):
        '''Print hierarchical adaptive clustering refinement descriptors (validation).'''
        # Print report header
        print(3*'\n' + 'Hierarchical adaptive clustering report\n' + 80*'-' + '\n\n' +
              'Material phase: ' + str(self._mat_phase))
        # Print adaptive clustering adaptive step
        print('\nAdaptive refinement step: ', self._adaptive_step)
        # Print hierarchical adaptive CRVE
        print('\n\n' + 'Adaptive clustering: ' +
              '(' + str(len(np.unique(self.cluster_labels))) + ' clusters)' + '\n\n',
              self.cluster_labels)
        # Print adaptive clustering mapping
        print('\n\n' + 'Adaptive cluster mapping: ')
        for old_cluster in adaptive_clustering_map.keys():
            print('    Old cluster: ' + '{:>4s}'.format(old_cluster) +
                  '  ->  ' +
                  'New clusters: ',
                  adaptive_clustering_map[str(old_cluster)])
        # Print adaptive tree node mapping
        print('\n\n' + 'Adaptive tree node mapping (validation): ')
        for old_node in adaptive_tree_node_map.keys():
            print('  Old node: ' + '{:>4s}'.format(old_node) +
                  '  ->  ' +
                  'New nodes: ', adaptive_tree_node_map[str(old_node)])
        # Print cluster-node mapping
        print('\n\n' + 'Cluster-Node mapping: ')
        for new_cluster in self._cluster_node_map.keys():
            print('    Cluster: ' + '{:>4s}'.format(new_cluster) +
                  '  ->  ' +
                  'Tree node: ',
                  self._cluster_node_map[str(new_cluster)])
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_valid_clust_algs():
        '''Get valid clustering algorithms to compute the CRMP.

        Returns
        ----------
        clust_algs : list
            Clustering algorithms identifiers (str).
        '''
        return ['5',]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_type_parameters():
        '''Get ACRMP mandatory and optional adaptivity type parameters.

        Besides returning the ACRMP mandatory and optional adaptivity type parameters, this
        method establishes the default values for the optional parameters.

        Returns
        ----------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        # Set mandatory adaptivity type parameters
        mandatory_parameters = {}
        # Set optional adaptivity type parameters and associated default values
        optional_parameters = {'adapt_split_factor': 0.01}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [mandatory_parameters, optional_parameters]
    # --------------------------------------------------------------------------------------
    def _set_adaptivity_type_parameters(self, adaptivity_type):
        '''Set clustering adaptivity parameters.

        Parameters
        ----------
        adaptivity_type : dict
            Clustering adaptivity parameters.
        '''
        # Set mandatory adaptivity type parameters
        pass
        # Set optional adaptivity type parameters
        self._adapt_split_factor = adaptivity_type['adapt_split_factor']
