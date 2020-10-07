#
# Cluster-reduced Representative Volume Element Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the generation of the Cluster-reduced Representative Volume Element
# (CRVE), a key step in the so called clustering-based reduced order models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | October 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Unsupervised clustering algorithms
import sklearn.cluster as skclst
# Defining abstract base classes
from abc import ABC, abstractmethod
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                            Available clustering algorithms
# ==========================================================================================
def get_available_clustering_algorithms():
    '''Get available clustering algorithms in CRATE.

    Clustering algorithms identifiers:
    1- K-Means (source: scikit-learn)
    2- Mini-Batch K-Means (source: scikit-learn)
    3- Birch (source: scikit-learn)

    Returns
    -------
    available_clustering_alg : dict
        Available clustering algorithms (item, str) and associated identifiers (key, str).
    '''

    available_clustering_alg = {'1': 'K-Means (scikit-learn)',
                                '2': 'Mini-Batch K-Means (scikit-learn)',
                                '3': 'Birch (scikit-learn)'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return available_clustering_alg
#
#                                                                                 CRVE class
# ==========================================================================================
class CRVE:
    '''Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation of a Cluster-reduced Representative Volume Element (CRVE).

    Attributes
    ----------
    _clustering_solutions : list
        List containing one or more RVE clustering solutions (ndarray of shape
        (n_clusters,)).
    _n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).
    _n_voxels : int
        Total number of voxels of the regular grid (spatial discretization of the RVE).
    _phase_voxel_flatidx : dict
        Flat (1D) voxels' indexes (item, list) associated to each material phase (key, str).
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material phase (key, str).
    '''

    def __init__(self, clustering_scheme, clustering_ensemble_strategy, phase_n_clusters,
                 rve_dims, regular_grid, material_phases):
        '''Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed global clustering scheme to generate the CRVE. Each row is associated
            with a unique RVE clustering, characterized by a clustering algorithm
            (col 1, int), a list of features (col 2, list of int) and a list of the feature
            data matrix' indexes (col 3, list of int).
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        '''
        self._clustering_scheme = clustering_scheme
        self._clustering_ensemble_strategy = clustering_ensemble_strategy
        self._material_phases = material_phases
        self._phase_n_clusters = phase_n_clusters
        self._rve_dims = rve_dims
        self.clustering_solutions = None
        self.voxels_clusters = None
        self.phase_clusters = None
        self.clusters_f = None
        # Get number of voxels on each dimension and total number of voxels
        self._n_voxels_dims = \
            [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        self._n_voxels = np.prod(self._n_voxels_dims)
        # Get material phases' voxels' 1D flat indexes
        self._phase_voxel_flatidx = \
            type(self)._get_phase_idxs(regular_grid, material_phases)
    # --------------------------------------------------------------------------------------
    def get_crve(self, global_data_matrix):
        '''Generate CRVE from one or more RVE clustering solutions.

        Main method commanding the generation of the Cluster-Reduced Representative Volume
        Element (CRVE): (1) performs the prescribed clustering scheme on the provided global
        data matrix, acquiring one or more RVE clustering solutions; (2) obtains a unique
        clustering solution (consensus solution) that materializes the CRVE; (3) computes
        several descriptors of the CRVE.

        Parameters
        ----------
        global_data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform all the RVE clusterings
            prescribed in the clustering scheme.
        '''
        # Loop over prescribed RVE clustering solutions
        for i_clst in range(self._clustering_scheme.shape[0]):
            # Get clustering algorithm
            clustering_method = self._clustering_scheme[i_clst, 0]
            # Get clustering features' columns
            feature_cols = self._clustering_scheme[i_clst, 2]
            # Get RVE clustering data matrix
            rve_data_matrix = mop.getcondmatrix(global_data_matrix,
                                                list(range(global_data_matrix.shape[0])),
                                                feature_cols)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            info.displayinfo('5', 'RVE clustering (' + str(i_clst + 1) + ' of ' +
                             str(self._clustering_scheme.shape[0]) + ')...', 2)
            # Instantiate RVE clustering
            rve_clustering = RVEClustering(clustering_method, self._phase_n_clusters,
                                           self._n_voxels, self._material_phases,
                                           self._phase_voxel_flatidx)
            # Perform RVE clustering
            rve_clustering.perform_rve_clustering(rve_data_matrix)
            # Assemble RVE clustering
            self._add_new_clustering(rve_clustering.labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE consensus clustering solution
        info.displayinfo('5', 'Building CRVE (RVE consensus clustering)...', 2)
        self._set_consensus_clustering()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing CRVE descriptors...', 2)
        # Reassign and sort RVE clustering labels
        self._sort_cluster_labels()
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_phase_idxs(regular_grid, material_phases):
        '''Get flat indexes of each material phase's voxels.

        Parameters
        ----------
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).

        Returns
        -------
        phase_voxel_flat_idx : dict
            Flat voxels' indexes (item, list of int) associated to each material phase
            (key, str).
        '''
        phase_voxel_flat_idx = dict()
        # Loop over material phases
        for mat_phase in material_phases:
            # Build boolean 'belongs to material phase' list
            is_phase_list = regular_grid.flatten() == int(mat_phase)
            # Get material phase's voxels' indexes
            phase_voxel_flat_idx[mat_phase] = list(it.compress(range(len(is_phase_list)),
                                                              is_phase_list))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return phase_voxel_flat_idx
    # --------------------------------------------------------------------------------------
    def _set_phase_clusters(self):
        '''Set CRVE cluster labels associated to each material phase.'''
        self.phase_clusters = {}
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get cluster labels
            self.phase_clusters[mat_phase] = \
                np.unique(self.voxels_clusters.flatten()[
                          self._phase_voxel_flatidx[mat_phase]])
    # --------------------------------------------------------------------------------------
    def _set_clusters_vf(self):
        '''Set CRVE clusters' volume fractions.'''
        # Compute voxel volume
        voxel_vol = np.prod([float(self._rve_dims[i])/self._n_voxels_dims[i]
                             for i in range(len(self._rve_dims))])
        # Compute RVE volume
        rve_vol = np.prod(self._rve_dims)
        # Compute volume fraction associated to each material cluster
        self.clusters_f = {}
        for cluster in np.unique(self.voxels_clusters):
            n_voxels_cluster = np.sum(self.voxels_clusters == cluster)
            self.clusters_f[str(cluster)] = (n_voxels_cluster*voxel_vol)/rve_vol
    # --------------------------------------------------------------------------------------
    def _add_new_clustering(self, rve_clustering):
        '''Add new RVE clustering to collection of clustering solutions.

        Parameters
        ----------
        rve_clustering : ndarray of shape (n_clusters,)
            Cluster label (int) assigned to each RVE voxel.
        '''
        self.clustering_solutions = []
        self.clustering_solutions.append(rve_clustering)
    # --------------------------------------------------------------------------------------
    def _set_consensus_clustering(self):
        '''Set a unique RVE clustering solution (consensus solution).

        Notes
        -----
        Even if the clustering scheme only accounts for a single RVE clustering solution,
        this method must be called in order to the set unique CRVE clustering solution.
        '''
        # Get RVE consensus clustering solution according to the prescribed clustering
        # ensemble strategy
        if self._clustering_ensemble_strategy == 0:
            # Build CRVE from the single RVE clustering solution
            self.voxels_clusters = np.reshape(np.array(self.clustering_solutions[0],
                                              dtype=int),self._n_voxels_dims)
    # --------------------------------------------------------------------------------------
    def _sort_cluster_labels(self):
        '''Reassign and sort CRVE cluster labels material phasewise.

        Reassign CRVE cluster labels in the range (0, n_clusters) and sort them in
        ascending order of material phase's labels.

        Notes
        -----
        Why is this required?
        '''
        # Initialize material phase initial cluster label
        lbl_init = 0
        # Initialize old cluster labels
        old_clusters = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize mapping dictionary to sort the cluster labels in asceding order of
        # material phase
        sort_dict = dict()
        # Loop over material phases sorted in ascending order
        sorted_mat_phases = list(np.sort(list(self._material_phases)))
        for mat_phase in sorted_mat_phases:
            # Get material phase old cluster labels
            phase_old_clusters = np.unique(
                self.voxels_clusters.flatten()[self._phase_voxel_flatidx[mat_phase]])
            # Set material phase new cluster labels
            phase_new_clusters = list(range(lbl_init,
                                      lbl_init + self._phase_n_clusters[mat_phase]))
            # Build mapping dictionary to sort the cluster labels
            for i in range(self._phase_n_clusters[mat_phase]):
                if phase_old_clusters[i] in sort_dict.keys():
                    raise RuntimeError('Cluster label (key) already exists in cluster' +
                                       'labels mapping dictionary.')
                else:
                    sort_dict[phase_old_clusters[i]] = phase_new_clusters[i]
            # Set next material phase initial cluster label
            lbl_init = lbl_init + self._phase_n_clusters[mat_phase]
            # Append old cluster labels
            old_clusters += list(phase_old_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check cluster labels mapping dictionary
        new_clusters = [sort_dict[key] for key in sort_dict.keys()]
        if set(sort_dict.keys()) != set(old_clusters) or \
                set(new_clusters) != set(old_clusters):
            raise RuntimeError('Invalid cluster labels mapping dictionary.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort cluster labels in ascending order of material phase
        for voxel_idx in it.product(*[list(range(self._n_voxels_dims[i])) \
                for i in range(len(self._n_voxels_dims))]):
            self.voxels_clusters[voxel_idx] = sort_dict[self.voxels_clusters[voxel_idx]]
#
#                                                                             RVE clustering
# ==========================================================================================
class RVEClustering:
    '''RVE clustering class.

    RVE clustering-based domain decomposition based on a given clustering algorithm.

    Atributes
    ---------
    labels : ndarray of shape (n_clusters,)
        Cluster label (int) assigned to each RVE voxel.
    '''
    def __init__(self, clustering_method, phase_n_clusters, n_voxels, material_phases,
                 phase_voxel_flat_idx):
        '''RVE clustering constructor.

        Parameters
        ----------
        _clustering_method : int
            Clustering algorithm identifier.
        _phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        _n_voxels : int
            Total number of voxels of the RVE spatial discretization.
        _material_phases : list
            RVE material phases labels (str).
        _phase_voxel_flatidx : dict
            Flat (1D) voxels' indexes (item, list) associated to each material phase
            (key, str).
        '''
        self._clustering_method = clustering_method
        self._phase_n_clusters = phase_n_clusters
        self._n_voxels = n_voxels
        self._material_phases = material_phases
        self._phase_voxel_flatidx = phase_voxel_flat_idx
        self.labels = None
    # --------------------------------------------------------------------------------------
    def perform_rve_clustering(self, data_matrix):
        '''Perform the RVE clustering-based domain decomposition.

        Instantiates a given clustering algorithm and performs the RVE clustering-based
        domain decomposition based on the provided data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform the RVE clustering.

        Notes
        -----
        The clustering is performed independently for each RVE's material phase. With the
        exception of the number of clusters, which is in general different for each material
        phase, the clustering algorithm and associated parameters remain the same for all
        material phases.
        '''
        # Initialize RVE clustering
        self.labels = np.full(self._n_voxels, -1, dtype=int)
        # Instantiate clustering algorithm
        if self._clustering_method == 1:
            # Set number of full batch K-Means clusterings (with different initializations)
            n_init = 10
            # Instantiate K-Means
            clst_alg = KMeans(init='k-means++', n_init=n_init, max_iter=300, tol=1e-4,
                              random_state=None, algorithm='auto')
        elif self._clustering_method == 2:
            # Set size of the mini-batches
            batch_size = 100
            # Set number of random initializations
            n_init = 3
            # Intantiate Mini-Batch K-Means
            clst_alg = MiniBatchKMeans(init='k-means++', max_iter=100, tol=0.0,
                                       random_state=None, batch_size=batch_size,
                                       max_no_improvement=10, init_size=None,
                                       n_init=n_init, reassignment_ratio=0.01)
        elif self._clustering_method == 3:
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clst_alg = Birch(threshold=threshold, branching_factor=branching_factor)
        else:
            raise RuntimeError('Unknown clustering algorithm.')
        # Initialize label offset (avoid that different material phases share the same
        # labels)
        label_offset = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase's voxels indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # Set material phase number of clusters
            if hasattr(clst_alg, 'n_clusters'):
                clst_alg.n_clusters = self._phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase data matrix
            phase_data_matrix = mop.getcondmatrix(data_matrix, voxels_idxs,
                                                  list(range(data_matrix.shape[1])))
            # Perform material phase clustering
            cluster_labels = clst_alg.perform_clustering(phase_data_matrix)
            # Check number of clusters formed
            if len(set(cluster_labels)) != self._phase_n_clusters[mat_phase]:
                raise RuntimeError('The number of clusters (' +
                                   str(len(set(cluster_labels))) +
                                   ') obtained for material phase ' + mat_phase + ' is ' +
                                   'different from the prescribed number of clusters (' +
                                   str(self._phase_n_clusters[mat_phase]) + ').')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material phase cluster labels
            self.labels[voxels_idxs] = label_offset + cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update label offset
            if not ioutil.checkposint(clst_alg.n_clusters):
                raise RuntimeError('Invalid number of clusters.')
            else:
                label_offset = label_offset + clst_alg.n_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the dataset items have been labeled
        if np.any(self.labels == -1):
            raise RuntimeError('At least one RVE domain point has not been labeled' +
                               'during the cluster analysis.')
#
#                                                                      Clustering algorithms
# ==========================================================================================
class ClusteringAlgorithm(ABC):
    '''Clustering algorithm interface.'''
    @abstractmethod
    def __init__(self):
        '''Clustering algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis on a given data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels : ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        pass
# ------------------------------------------------------------------------------------------
class KMeans(ClusteringAlgorithm):
    '''K-Means clustering algorithm.

    Notes
    -----
    The K-Means clustering algorithm is taken from scikit-learn (https://scikit-learn.org).
    Additional parameters and further information can be found in there.
    '''
    def __init__(self, n_clusters=None, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, algorithm='auto'):
        '''K-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        init : {‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’
            Method for centroid initialization.
        n_init : int, default=10
            Number of times K-Means is run with different centroid seeds.
        max_iter : int, default=300
            Maximum number of iterations.
        tol : float, default=1e-4
            Convergence tolerance (based on Frobenius norm of the different in the cluster
            centers of two consecutive iterations).
        random_state : int, RandomState instance, default=None
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        algorithm : {'auto', 'full', 'elkan'}, default='auto'
            K-Means algorithm to use. 'full' is the classical EM-style algorithm, 'elkan'
            uses the triangle inequality to speed up convergence. 'auto' currently chooses
            'elkan' (scikit-learn 0.23.2).

        Notes
        -----
        Validation of parameters is performed upon instantiation of scikit-learn K-Means
        clustering algorithm.
        '''
        self.n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._algorithm = algorithm
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Compute cluster centers and predict cluster label for each dataset item.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels : ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instantiate scikit-learn K-Means clustering algorithm
        self._clst_alg = skclst.KMeans(n_clusters=self.n_clusters, init=self._init,
                                       n_init=self._n_init, max_iter=self._max_iter,
                                       tol=self._tol, random_state=self._random_state,
                                       algorithm=self._algorithm)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute cluster centers (fitted estimator) and predict cluster label (prediction)
        # for each dataset item
        cluster_labels = self._clst_alg.fit_predict(data_matrix, sample_weight=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class MiniBatchKMeans(ClusteringAlgorithm):
    '''Mini-Batch K-Means clustering algorithm.

    Notes
    -----
    The Mini-Batch K-Means clustering algorithm is taken from scikit-learn
    (https://scikit-learn.org). Additional parameters and further information can be found
    in there.
    '''
    def __init__(self, n_clusters=None, init='k-means++', max_iter=100, tol=0.0,
                 random_state=None, batch_size=100, max_no_improvement=10, init_size=None,
                 n_init=3, reassignment_ratio=0.01):
        '''Mini-Batch K-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        init: {‘k-means++’, ‘random’, ndarray, callable}, default=’k-means++’
            Method for centroid initialization.
        n_init : int, default=10
            Number of times K-Means is run with different centroid seeds.
        max_iter : int, default=300
            Maximum number of iterations.
        tol : float, default=1e-4
            Convergence tolerance (based on Frobenius norm of the different in the cluster
            centers of two consecutive iterations).
        random_state : int, RandomState instance, default=None
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        init_size : int, default=None
            Number of samples to randomly sample for speeding up the initialization
            (sometimes at the expense of accuracy): the only algorithm is initialized by
            running a batch KMeans on a random subset of the data.
        n_init : int, default=3
            Number of random initializations that are tried (best of initializations is
            used to run the algorithm).
        reassignment_ratio : float, default=0.01
            Control the fraction of the maximum number of counts for a center to be
            reassigned.

        Notes
        -----
        Validation of parameters is performed upon instantiation of scikit-learn Mini-Batch
        K-Means clustering algorithm.
        '''
        self.n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._init_size = init_size
        self._reassignment_ratio = reassignment_ratio
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Compute cluster centers and predict cluster label for each dataset item.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels : ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instantiate scikit-learn Mini-Batch K-Means clustering algorithm
        self._clst_alg = skclst.MiniBatchKMeans(n_clusters=self.n_clusters, init=self._init,
            n_init=self._n_init, max_iter=self._max_iter, tol=self._tol,
            random_state=self._random_state, init_size=self._init_size,
            reassignment_ratio=self._reassignment_ratio)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute cluster centers (fitted estimator) and predict cluster label (prediction)
        # for each dataset item
        cluster_labels = self._clst_alg.fit_predict(data_matrix, sample_weight=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class Birch(ClusteringAlgorithm):
    '''Birch clustering algorithm.

    Notes
    -----
    The Birch clustering algorithm is taken from scikit-learn (https://scikit-learn.org).
    Additional parameters and further information can be found in there.
    '''
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=None):
        '''Birch clustering algorithm constructor.

        Parameters
        ----------
        threshold : float, default=0.5
            The radius of the subcluster obtained by merging a new sample and the closest
            subcluster should be lesser than the threshold. Otherwise a new subcluster is
            started. Setting this value to be very low promotes splitting and vice-versa.
        branching_factor : int, default=50
            Maximum number of CF subclusters in each node. If a new samples enters such that
            the number of subclusters exceed the branching_factor then that node is split
            into two nodes with the subclusters redistributed in each. The parent subcluster
            of that node is removed and two new subclusters are added as parents of the 2
            split nodes.
        n_clusters : int, instance of sklearn.cluster model, default=3
            Number of clusters after the final clustering step, which treats the subclusters
            from the leaves as new samples.
            - `None` : the final clustering step is not performed and the subclusters are
               returned as they are.
            - `sklearn.cluster` Estimator : If a model is provided, the model is fit
               treating the subclusters as new samples and the initial data is mapped to the
               label of the closest subcluster.
            - `int` : the model fit is `AgglomerativeClustering` with `n_clusters` set to be
               equal to the int.

        Notes
        -----
        Validation of parameters is performed upon instantiation of scikit-learn Birch
        clustering algorithm.
        '''
        self.n_clusters = n_clusters
        self._threshold = threshold
        self._branching_factor = branching_factor
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform clustering and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instantiate scikit-learn Birch clustering algorithm
        self._clst_alg = skclst.Birch(threshold=self._threshold,
                                      branching_factor=self._branching_factor,
                                      n_clusters=self.n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering and return cluster labels
        cluster_labels = self._clst_alg.fit_predict(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
