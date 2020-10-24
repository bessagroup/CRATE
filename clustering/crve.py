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
import scipy.cluster.hierarchy as sciclst
import pyclustering.cluster.kmeans as pykmeans
import pyclustering.cluster.birch as pybirch
import pyclustering.cluster.cure as pycure
import pyclustering.cluster.xmeans as pyxmeans
import pyclustering.container.cftree as pycftree
import pyclustering.cluster.encoder as pyencoder
import pyclustering.utils.metric as pymetric
import pyclustering.cluster.center_initializer as pycenterinit
import fastcluster as fastclst
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
    2- K-Means (source: pyclustering)
    3- Mini-Batch K-Means (source: scikit-learn)
    4- Agglomerative (source: scikit-learn)
    5- Agglomerative (source: scipy)
    6- Agglomerative (source: fastcluster)
    7- Birch (source: scikit-learn)
    8- Birch (source: pyclustering)
    9- Cure (source: pyclustering)
    10- X-Means (source: pyclustering)

    Returns
    -------
    available_clustering_alg : dict
        Available clustering algorithms (item, str) and associated identifiers (key, str).
    '''
    available_clustering_alg = {'1': 'K-Means (scikit-learn)',
                                '2': 'K-Means (pyclustering)',
                                '3': 'Mini-Batch K-Means (scikit-learn)',
                                '4': 'Agglomerative (scikit-learn)',
                                '5': 'Agglomerative (scipy)',
                                '6': 'Agglomerative (fastcluster)',
                                '7': 'Birch (scikit-learn)',
                                '8': 'Birch (pyclustering)',
                                '9': 'Cure (pyclustering)',
                                '10': 'X-Means (pyclustering)'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return available_clustering_alg


#
#                                                                                 CRVE class
# ==========================================================================================
class CRVE:
    '''Cluster-reduced Representative Volume Element.

    Base class of a Cluster-reduced Representative Volume Element (CRVE) from which
    Static CRVE (S-CRVE) and Adaptive CRVE (XA-CRVE) are to be derived from.

    Attributes
    ----------
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
    def __new__(cls, *args, **kwargs):
        if cls is CRVE:
            raise TypeError("CRVE base class may not be instantiated")
        return super().__new__(cls)
    # --------------------------------------------------------------------------------------
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases):
        '''Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
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
        self._material_phases = material_phases
        self._phase_n_clusters = phase_n_clusters
        self._rve_dims = rve_dims
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
                len(set(new_clusters)) != len(set(old_clusters)):
            raise RuntimeError('Invalid cluster labels mapping dictionary.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort cluster labels in ascending order of material phase
        for voxel_idx in it.product(*[list(range(self._n_voxels_dims[i])) \
                for i in range(len(self._n_voxels_dims))]):
            self.voxels_clusters[voxel_idx] = sort_dict[self.voxels_clusters[voxel_idx]]
#
#                                                                               S-CRVE class
# ==========================================================================================
class SCRVE(CRVE):
    '''Static Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation of a Static Cluster-reduced Representative Volume Element (S-CRVE).

    Attributes
    ----------
    _n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).
    _n_voxels : int
        Total number of voxels of the regular grid (spatial discretization of the RVE).
    _phase_voxel_flatidx : dict
        Flat (1D) voxels' indexes (item, list) associated to each material phase (key, str).
    _clustering_solutions : list
        List containing one or more RVE clustering solutions (ndarray of shape
        (n_clusters,)).
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material phase (key, str).
    '''
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 clustering_scheme, clustering_ensemble_strategy):
        '''Static Cluster-reduced Representative Volume Element constructor.

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
        super().__init__(phase_n_clusters, rve_dims, regular_grid, material_phases)
        self._clustering_scheme = clustering_scheme
        self._clustering_ensemble_strategy = clustering_ensemble_strategy
        self._clustering_solutions = []
    # --------------------------------------------------------------------------------------
    def get_scrve(self, global_data_matrix):
        '''Generate S-CRVE from one or more RVE clustering solutions.

        Main method commanding the generation of the Static Cluster-Reduced Representative
        Volume Element (S-CRVE): (1) performs the prescribed clustering scheme on the
        provided global data matrix, acquiring one or more RVE clustering solutions;
        (2) obtains a unique clustering solution (consensus solution) that materializes the
        S-CRVE; (3) computes several descriptors of the S-CRVE.

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
        info.displayinfo('5', 'Building S-CRVE (RVE consensus clustering)...', 2)
        self._set_consensus_clustering()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing S-CRVE descriptors...', 2)
        # Reassign and sort RVE clustering labels
        self._sort_cluster_labels()
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def _add_new_clustering(self, rve_clustering):
        '''Add new RVE clustering to collection of clustering solutions.

        Parameters
        ----------
        rve_clustering : ndarray of shape (n_clusters,)
            Cluster label (int) assigned to each RVE voxel.
        '''
        self._clustering_solutions.append(rve_clustering)
    # --------------------------------------------------------------------------------------
    def _set_consensus_clustering(self):
        '''Set a unique RVE clustering solution (consensus solution).

        Notes
        -----
        Even if the clustering scheme only accounts for a single RVE clustering solution,
        this method must be called in order to the set unique S-CRVE clustering solution.
        '''
        # Get RVE consensus clustering solution according to the prescribed clustering
        # ensemble strategy
        if self._clustering_ensemble_strategy == 0:
            # Build S-CRVE from the single RVE clustering solution
            self.voxels_clusters = np.reshape(np.array(self._clustering_solutions[0],
                                              dtype=int),self._n_voxels_dims)
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate clustering algorithm
        if self._clustering_method == 1:
            # Set number of full batch K-Means clusterings (with different initializations)
            n_init = 10
            # Instantiate K-Means
            clst_alg = KMeansSK(init='k-means++', n_init=n_init, max_iter=300, tol=1e-4,
                                random_state=None, algorithm='auto')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 2:
            # Instatiante K-Means
            clst_alg = KMeansPC(tolerance=1e-03, itermax=200)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 3:
            # Set size of the mini-batches
            batch_size = 100
            # Set number of random initializations
            n_init = 3
            # Intantiate Mini-Batch K-Means
            clst_alg = MiniBatchKMeansSK(init='k-means++', max_iter=100, tol=0.0,
                                         random_state=None, batch_size=batch_size,
                                         max_no_improvement=10, init_size=None,
                                         n_init=n_init, reassignment_ratio=0.01)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 4:
            # Instantiate Agglomerative clustering
            clst_alg = AgglomerativeSK(n_clusters=None, affinity='euclidean', memory=None,
                                       connectivity=None, compute_full_tree='auto',
                                       linkage='ward', distance_threshold=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 5:
            # Instatiate Agglomerative clustering
            clst_alg = AgglomerativeSP(0, n_clusters=None, method='ward',
                                       metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 6:
            # Instatiate Agglomerative clustering
            clst_alg = AgglomerativeFC(0, n_clusters=None, method='ward',
                                       metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 7:
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clst_alg = BirchSK(threshold=threshold, branching_factor=branching_factor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 8:
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clst_alg = BirchPC(threshold=threshold, branching_factor=branching_factor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 9:
            # Instantiate Cure
            clst_alg = CurePC(number_represent_points=5, compression=0.5)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 10:
            # Instantiate X-Means
            clst_alg = XMeansPC(tolerance=2.5e-2, repeat=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown clustering algorithm.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize label offset (avoid that different material phases share the same
        # labels)
        label_offset = 0
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
        '''Perform cluster analysis and return cluster label for each dataset item.

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
class AgglomerativeAlgorithm(ClusteringAlgorithm):
    '''Hierarchical agglomerative interface.'''
    @abstractmethod
    def get_linkage_matrix(self):
        '''Get hierarchical agglomerative clustering linkage matrix.

        Returns
        -------
        linkage_matrix : ndarray of shape (n-1, 4)
                         Linkage matrix associated with the hierarchical agglomerative
                         clustering. At the i-th iteration the clusterings with indices
                         Z[i, 0] and Z[i, 1], with distance Z[i, 2], are merged, forming a
                         new cluster that contains Z[i, 3] original dataset items. All
                         cluster indices j >= n refer to the cluster formed in Z[j-n, :].

        Notes
        -----
        The hierarchical agglomerative clustering linkage matrix follows the definition of
        scipy (https://docs.scipy.org/) agglomerative clustering algorithm.
        '''
        pass
# ------------------------------------------------------------------------------------------
class KMeansSK(ClusteringAlgorithm):
    '''K-Means clustering algorithm.

    Notes
    -----
    The K-Means clustering algorithm is taken from scikit-learn (https://scikit-learn.org).
    Further information can be found in there.
    '''
    def __init__(self, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, algorithm='auto', n_clusters=None):
        '''K-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
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
        '''Perform cluster analysis and return cluster label for each dataset item.

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
class MiniBatchKMeansSK(ClusteringAlgorithm):
    '''Mini-Batch K-Means clustering algorithm.

    Notes
    -----
    The Mini-Batch K-Means clustering algorithm is taken from scikit-learn
    (https://scikit-learn.org). Further information can be found in there.
    '''
    def __init__(self, init='k-means++', max_iter=100, tol=0.0, random_state=None,
                 batch_size=100, max_no_improvement=10, init_size=None, n_init=3,
                 reassignment_ratio=0.01, n_clusters=None):
        '''Mini-Batch K-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
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
        '''Perform cluster analysis and return cluster label for each dataset item.

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
class BirchSK(ClusteringAlgorithm):
    '''Birch clustering algorithm.

    Notes
    -----
    The Birch clustering algorithm is taken from scikit-learn (https://scikit-learn.org).
    Further information can be found in there.
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
        n_clusters : int, instance of sklearn.cluster model, default=None
            Number of clusters to find after the final clustering step, which treats the
            subclusters from the leaves as new samples.
            - `None` : the final clustering step is not performed and the subclusters are
               returned as they are.
            - `sklearn.cluster` Estimator : If a model is provided, the model is fit
               treating the subclusters as new samples and the initial data is mapped to the
               label of the closest subcluster.
            - `int` : the model fit is `AgglomerativeClustering` with `n_clusters` set to be
               equal to the int.
        '''
        self.n_clusters = n_clusters
        self._threshold = threshold
        self._branching_factor = branching_factor
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

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
# ------------------------------------------------------------------------------------------
class AgglomerativeSK(ClusteringAlgorithm):
    '''Agglomerative clustering algorithm.

    Notes
    -----
    The Agglomerative clustering algorithm is taken from scikit-learn
    (https://scikit-learn.org). Further information can be found in there.
    '''
    def __init__(self, affinity='euclidean', memory=None, connectivity=None,
                 compute_full_tree='auto', linkage='ward', distance_threshold=None,
                 n_clusters=None):
        '''Agglomerative clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find. It must be ``None`` if ``distance_threshold`` is
            not ``None``.
        affinity : str or callable, default='euclidean'
            Metric used to compute the linkage. Can be "euclidean", "l1", "l2", "manhattan",
            "cosine", or "precomputed". If linkage is "ward", only "euclidean" is accepted.
            If "precomputed", a distance matrix (instead of a similarity matrix) is needed
            as input for the fit method.
        memory : str or object with the joblib.Memory interface, default=None
            Used to cache the output of the computation of the tree. By default, no caching
            is done. If a string is given, it is the path to the caching directory.
        connectivity : array-like or callable, default=None
            Connectivity matrix. Defines for each sample the neighboring samples following a
            given structure of the data. This can be a connectivity matrix itself or a
            callable that transforms the data into a connectivity matrix, such as derived
            from kneighbors_graph. Default is None, i.e, the hierarchical clustering
            algorithm is unstructured.
        compute_full_tree : 'auto' or bool, default='auto'
            Stop early the construction of the tree at n_clusters. This is useful to
            decrease computation time if the number of clusters is not small compared to the
            number of samples. This option is useful only when specifying a connectivity
            matrix. Note also that when varying the number of clusters and using caching, it
            may be advantageous to compute the full tree. It must be ``True`` if
            ``distance_threshold`` is not ``None``. By default `compute_full_tree` is
            "auto", which is equivalent to `True` when `distance_threshold` is not `None` or
            that `n_clusters` is inferior to the maximum between 100 or `0.02 * n_samples`.
            Otherwise, "auto" is equivalent to `False`.
        linkage : {"ward", "complete", "average", "single"}, default="ward"
            Which linkage criterion to use. The linkage criterion determines which distance
            to use between sets of observation. The algorithm will merge the pairs of
            cluster that minimize this criterion.
            - ward minimizes the variance of the clusters being merged.
            - average uses the average of the distances of each observation of the two sets.
            - complete or maximum linkage uses the maximum distances between all
              observations of the two sets.
            - single uses the minimum of the distances between all observations of the two
              sets.
        distance_threshold : float, default=None
            The linkage distance threshold above which, clusters will not be merged. If not
            ``None``, ``n_clusters`` must be ``None`` and ``compute_full_tree`` must be
            ``True``.
        '''
        self.n_clusters = n_clusters
        self._affinity = affinity
        self._memory = memory
        self._connectivity = connectivity
        self._compute_full_tree = compute_full_tree
        self._linkage = linkage
        self._distance_threshold = distance_threshold
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

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
        self._clst_alg = skclst.AgglomerativeClustering(n_clusters=self.n_clusters,
            affinity=self._affinity, memory=self._memory, connectivity=self._connectivity,
            compute_full_tree=self._compute_full_tree, linkage=self._linkage,
            distance_threshold=self._distance_threshold)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit the hierarchical clustering and return cluster labels
        cluster_labels = self._clst_alg.fit_predict(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class AgglomerativeSP(AgglomerativeAlgorithm):
    '''Agglomerative clustering algorithm.

    Attributes
    ----------
    _linkage_matrix : ndarray of shape (n-1, 4)
                      Linkage matrix associated with the hierarchical clustering. At the
                      i-th iteration the clusterings with indices Z[i, 0] and Z[i, 1], with
                      distance Z[i, 2], are merged, forming a new cluster that contains
                      Z[i, 3] original dataset items. All cluster indices j >= n refer to
                      the cluster formed in Z[j-n, :].

    Notes
    -----
    The Agglomerative clustering algorithm is taken from scipy (https://docs.scipy.org/)
    Further information can be found in there.
    '''
    def __init__(self, t, method='ward', metric='euclidean', criterion='maxclust',
                 n_clusters=None):
        '''Agglomerative clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find.
        t : int or float
            Scalar parameter associated to the criterion used to form a flat clustering.
            Threshold (float) with criterion in {'inconsistent', 'distance', 'monocrit'} or
            maximum number of clusters with criterion in {'maxclust', 'maxclust_monocrit'}.
        method : str, {'single', 'complete', 'average', 'weighted', 'centroid', 'median',
                'ward'}, default='ward'
            Linkage criterion.
        metric : str or function, default='euclidean'
            Distance metric to use when the input data matrix is a ndarray of observation
            vectors, otherwise ignored. Options: {'cityblock', 'euclidean', 'cosine', ...}.
        criterion : str, {'inconsistent', 'distance', 'maxclust', 'monocrit',
            'maxclust_monocrit'}, default='maxclust'
            Criterion used to form a flat clustering (i.e., perform a horizontal cut in the
            hierarchical tree).
        '''
        self._t = t
        self.n_clusters = n_clusters
        self._method = method
        self._metric = metric
        self._criterion = criterion
        self._linkage_matrix = None
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Perform hierarchical clustering and encode it in a linkage matrix
        self._linkage_matrix = sciclst.linkage(data_matrix, method=self._method,
                                               metric=self._metric)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform horizontal cut in hierarchical tree and return cluster labels (form a flat
        # clustering)
        cluster_labels = sciclst.fcluster(self._linkage_matrix, self.n_clusters,
                                          criterion=self._criterion)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
    # --------------------------------------------------------------------------------------
    def get_linkage_matrix(self):
        '''Get hierarchical agglomerative clustering linkage matrix.

        Returns
        -------
        linkage_matrix : ndarray of shape (n-1, 4)
                         Linkage matrix associated with the hierarchical agglomerative
                         clustering. At the i-th iteration the clusterings with indices
                         Z[i, 0] and Z[i, 1], with distance Z[i, 2], are merged, forming a
                         new cluster that contains Z[i, 3] original dataset items. All
                         cluster indices j >= n refer to the cluster formed in Z[j-n, :].

        Notes
        -----
        The hierarchical agglomerative clustering linkage matrix follows the definition of
        scipy (https://docs.scipy.org/) agglomerative clustering algorithm.
        '''
        if self._linkage_matrix is None:
            raise ValueError('Hierarchical agglomerative clustering linkage matrix has' +
                             'not been computed yet.')
        return self._linkage_matrix
# ------------------------------------------------------------------------------------------
class BirchPC(ClusteringAlgorithm):
    '''Birch clustering algorithm.

    Notes
    -----
    The Birch clustering algorithm is taken from pyclustering (https://pypi.org/).
    Further information can be found in there.
    '''
    def __init__(self, threshold=0.5, branching_factor=50, max_node_entries=200,
                 type_measurement=pycftree.measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
                 entry_size_limit=500, threshold_multiplier=1.5, n_clusters=None):
        '''Birch clustering algorithm constructor.

        Parameters
        ----------
        threshold : float, default=0.5
            CF-entry diameter that is used for CF-Tree construction (might increase if
            `entry_size_limit` is exceeded).
        branching_factor : int, default=50
            Maximum number of successor that might be contained by each non-leaf node in
            CF-Tree.
        max_node_entries : int, default=200
            Maximum number of entries that might be contained by each leaf node in CF-Tree.
        type_measurement : measurement type, default=CENTROID_EUCLIDEAN_DISTANCE
            Type of measurement used for calculation of distance metrics.
        entry_size_limit : int, default=500
            Maximum number of entries that can be stored in CF-Tree (if exceeded during
            creation of CF-Tree, then threshold is increased and CF-Tree is rebuilt).
        threshold_multiplier : float, default=1.5
            Multiplier used to increase the threshold when `entry_size_limit` is exceeded.
        n_clusters : int, default=None
            Number of clusters to find.
        '''
        self.n_clusters = n_clusters
        self._threshold = threshold
        self._branching_factor = branching_factor
        self._max_node_entries = max_node_entries
        self._type_measurement = type_measurement
        self._entry_size_limit = entry_size_limit
        self._threshold_multiplier = threshold_multiplier
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instantiate pyclustering Birch clustering algorithm
        self._clst_alg = pybirch.birch(data_matrix.tolist(), self.n_clusters,
                                       diameter=self._threshold,
                                       branching_factor=self._branching_factor,
                                       max_node_entries=self._max_node_entries,
                                       entry_size_limit=self._entry_size_limit,
                                       diameter_multiplier=self._threshold_multiplier)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters, data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class CurePC(ClusteringAlgorithm):
    '''Cure clustering algorithm.

    Notes
    -----
    The Cure clustering algorithm is taken from pyclustering (https://pypi.org/).
    Further information can be found in there.
    '''
    def __init__(self, number_represent_points=5, compression=0.5, n_clusters=None):
        '''Cure clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
        number_represent_points : int, default=5
            Number of representative points for each cluster.
        compression : float, default=0.5
            Coefficient that defines the level of shrinking of representation points toward
            the mean of the new created cluster after merging on each step (usually set
            between 0 and 1).
        '''
        self.n_clusters = n_clusters
        self._number_represent_points = number_represent_points
        self._compression = compression
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instantiate pyclustering Cure clustering algorithm
        self._clst_alg = pycure.cure(data_matrix.tolist(), self.n_clusters,
                                     number_represent_points=self._number_represent_points,
                                     compression=self._compression)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters, data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class KMeansPC(ClusteringAlgorithm):
    '''K-Means clustering algorithm.

    Notes
    -----
    The K-Means clustering algorithm is taken from pyclustering (https://pypi.org/).
    Further information can be found in there.
    '''
    def __init__(self, tolerance=1e-03, itermax=200,
                 metric=pymetric.distance_metric(pymetric.type_metric.EUCLIDEAN_SQUARE),
                 n_clusters=None):
        '''K-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
        tolerance : float, default=1e-03
            Convergence tolerance (based on the maximum value of change of cluster centers
            of two consecutive iterations).
        itermax : int, default=200
            Maximum number of iterations.
        metric : distance_metric, default=EUCLIDEAN_SQUARE
            Metric used for distance calculation between samples.
        '''
        self.n_clusters = n_clusters
        self._tolerance = tolerance
        self._itermax = itermax
        self._metric = metric
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Instatiante cluster centers seeds using K-Means++
        amount_candidates = \
            pycenterinit.kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
        initial_centers = pycenterinit.kmeans_plusplus_initializer(data_matrix.tolist(),
            self.n_clusters, amount_candidates=amount_candidates).initialize()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate pyclustering K-Means clustering algorithm
        self._clst_alg = pykmeans.kmeans(data_matrix.tolist(), initial_centers,
                                         tolerance=self._tolerance, itermax=self._itermax,
                                         metric=self._metric)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters, data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class XMeansPC(ClusteringAlgorithm):
    '''X-Means clustering algorithm.

    Notes
    -----
    The X-Means clustering algorithm is taken from pyclustering (https://pypi.org/).
    Further information can be found in there.
    '''
    def __init__(self, tolerance=2.5e-2,
                 criterion=pyxmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION, repeat=1,
                 n_clusters=None):
        '''X-Means clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Maximum number of clusters than can be found.
        tolerance : float, default=2.5e-2
            Convergence tolerance (based on the maximum value of change of cluster centers
            of two consecutive iterations).
        criterion : splitting_type, BAYESIAN_INFORMATION_CRITERION
            Criterion to perform cluster splitting.
        repeat : int, default=1
            How many times K-Means should be run to improve parameters. Larger values
            increase the probability of finding global optimum.
        '''
        self.n_clusters = n_clusters
        self._tolerance = tolerance
        self._criterion = criterion
        self._repeat = repeat
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Set initial numbers of clusters
        amount_initial_centers = max(1, int(0.1*self.n_clusters))
        # Instatiante cluster centers seeds using K-Means++
        initial_centers = pycenterinit.kmeans_plusplus_initializer(data_matrix.tolist(),
            amount_initial_centers).initialize()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate pyclustering X-Means clustering algorithm
        self._clst_alg = pyxmeans.xmeans(data_matrix.tolist(), initial_centers,
                                         kmax=self.n_clusters, tolerance=self._tolerance,
                                         criterion=self._criterion, repeat=self._repeat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters, data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# ------------------------------------------------------------------------------------------
class AgglomerativeFC(ClusteringAlgorithm):
    '''Agglomerative clustering algorithm.

    Attributes
    ----------
    Z : ndarray of shape (n-1, 4)
        Linkage matrix associated with the hierarchical clustering. At the i-th iteration
        the clusterings with indices Z[i, 0] and Z[i, 1], with distance Z[i, 2], are merged,
        forming a new cluster that contains Z[i, 3] original dataset items. All cluster
        indices j >= n refer to the cluster formed in Z[j-n, :].

    Notes
    -----
    The Agglomerative clustering algorithm is taken from Daniel Mullner fastcluster package
    (http://danifold.net/fastcluster). Apart from one optional argument (`preserve_input`),
    fastcluster implements the scipy Agglomerative clustering algorithm in a more efficient
    way, being the associated classes and methods the same (https://docs.scipy.org/).
    Further information can be found in both domains.
    '''
    def __init__(self, t, method='ward', metric='euclidean', criterion='maxclust',
                 n_clusters=None):
        '''Agglomerative clustering algorithm constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find.
        t : int or float
            Scalar parameter associated to the criterion used to form a flat clustering.
            Threshold (float) with criterion in {'inconsistent', 'distance', 'monocrit'} or
            maximum number of clusters with criterion in {'maxclust', 'maxclust_monocrit'}.
        method : str, {'single', 'complete', 'average', 'weighted', 'centroid', 'median',
                'ward'}, default='ward'
            Linkage criterion.
        metric : str or function, default='euclidean'
            Distance metric to use when the input data matrix is a ndarray of observation
            vectors, otherwise ignored. Options: {'cityblock', 'euclidean', 'cosine', ...}.
        criterion : str, {'inconsistent', 'distance', 'maxclust', 'monocrit',
            'maxclust_monocrit'}, default='maxclust'
            Criterion used to form a flat clustering (i.e., perform a horizontal cut in the
            hierarchical tree).
        '''
        self._t = t
        self.n_clusters = n_clusters
        self._method = method
        self._metric = metric
        self._criterion = criterion
        self._Z = None
    # --------------------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        '''Perform cluster analysis and return cluster label for each dataset item.

        Parameters
        ----------
        data_matrix: ndarray of shape (n_items, n_features)
            Data matrix containing the required data to perform cluster analysis.

        Returns
        -------
        cluster_labels: ndarray of shape (n_items,)
            Cluster label (int) assigned to each dataset item.
        '''
        # Perform hierarchical clustering and encode it in a linkage matrix
        self.Z = fastclst.linkage(data_matrix, method=self._method, metric=self._metric,
                                  preserve_input=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform horizontal cut in hierarchical tree and return cluster labels (form a flat
        # clustering)
        cluster_labels = sciclst.fcluster(self.Z, self.n_clusters,
                                          criterion=self._criterion)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
#
#                                                                              HA-CRVE class
# ==========================================================================================
class HACRVE(CRVE):
    '''Hierarchical Adaptive Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation and update of a Hierarchical Adaptive Cluster-reduced Representative Volume
    Element (HA-CRVE).

    Attributes
    ----------
    _n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).
    _n_voxels : int
        Total number of voxels of the regular grid (spatial discretization of the RVE).
    _phase_voxel_flatidx : dict
        Flat (1D) voxels' indexes (item, list) associated to each material phase (key, str).
    _phase_linkage_matrix : dict
        Linkage matrix (item, ndarray of shape (n_voxels-1, 4)) associated with the
        hierarchical agglomerative clustering of each material phase (key, str).
    _phase_map_cluster_node : dict
        Cluster-node mapping (item, dict with tree node id (item, int) associated to each
        cluster label (item, str)) for each material phase (key, str).
    _adaptive_step : int
        Counter of hierarchical adaptive clustering steps, with 0 associated with the
        hierarchical agglomerative base clustering.
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material phase (key, str).
    '''
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 split_greed=0.5):
        '''Hierarchical Adaptive Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        split_greed : float, default=0.5
            Cluster splitting greediness parameter contained between 0 and 1 (included).
            The lower bound (0) prevents any cluster to be splitted, while the upper bound
            (1) performs the maximum number splits of each cluster (single-voxel clusters).
        '''
        super().__init__(phase_n_clusters, rve_dims, regular_grid, material_phases)
        self._split_greed = split_greed
        self._phase_linkage_matrix = None
        self._phase_map_cluster_node = {}
        self._adaptive_step = 0
    # --------------------------------------------------------------------------------------
    def get_base_clustering(self, data_matrix):
        '''Perform the RVE hierarchical agglomerative base clustering-based domain
        decomposition.

        Instantiates a given hierarchical agglomerative clustering algorithm and performs
        the RVE hierarchical agglomerative base clustering-based domain decomposition
        (hierarchical tree building and horizontal cut) based on the provided data matrix.

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
        # Instatiate Agglomerative clustering
        clst_alg = AgglomerativeSP(0, n_clusters=None, method='ward',
                                   metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize linkage matrices dictionary
        self._phase_linkage_matrix = {}
        # Initialize RVE hierarchical agglomerative base clustering
        labels = np.full(self._n_voxels, -1, dtype=int)
        # Initialize label offset (avoid that different material phases share the same
        # labels)
        label_offset = 0
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase's voxels indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # Set material phase base number of clusters
            clst_alg.n_clusters = self._phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase data matrix
            phase_data_matrix = mop.getcondmatrix(data_matrix, voxels_idxs,
                                                  list(range(data_matrix.shape[1])))
            # Perform material phase hierarchical agglomerative base clustering
            # (hierarchical tree horizontal cut)
            cluster_labels = clst_alg.perform_clustering(phase_data_matrix)
            # Check number of clusters formed
            if len(set(cluster_labels)) != self._phase_n_clusters[mat_phase]:
                raise RuntimeError('The number of clusters (' +
                                   str(len(set(cluster_labels))) +
                                   ') obtained for material phase ' + mat_phase + ' is ' +
                                   'different from the prescribed number of clusters (' +
                                   str(self._phase_n_clusters[mat_phase]) + ').')
            # Store material phase hierarchical agglomerative base clustering linkage matrix
            self._phase_linkage_matrix[mat_phase] = clst_alg.get_linkage_matrix()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material phase cluster labels
            labels[voxels_idxs] = label_offset + cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update label offset
            if not ioutil.checkposint(clst_alg.n_clusters):
                raise RuntimeError('Invalid number of clusters.')
            else:
                label_offset = label_offset + clst_alg.n_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the voxels have been labeled
        if np.any(labels == -1):
            raise RuntimeError('At least one RVE domain point has not been labeled' +
                               'during the cluster analysis.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE from the hierarchical agglomerative base clustering
        info.displayinfo('5', 'Building HA-CRVE base clustering...', 2)
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing HA-CRVE descriptors...', 2)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters):
        '''Perform a hierarchical adaptive clustering refinement step.

        Refine the provided target clusters by splitting them according to the hierarchical
        agglomerative tree, prioritizing child nodes by descending order of linkage
        distance.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be refined.

        Returns
        -------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the refinement of
            each target cluster (key, str).
        adaptive_tree_node_map : dict
            List of new cluster tree node ids (item, list of int) resulting from the split
            of each target cluster tree node id (key, str).

        Notes
        -----
        Given that the hierarchical agglomerative base clustering is performed independently
        for each RVE's material phase, so is the adaptive clustering refinement, i.e., each
        material phase has an associated hierarchical agglomerative tree.
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            is_exist = False
            for mat_phase in self._material_phases:
                if target_cluster in self.phase_clusters[mat_phase]:
                    is_exist = True
                    break
            if not is_exist:
                raise RuntimeError('Target cluster ' + str(target_cluster) + ' does not ' +
                                   'exist in the current CRVE clustering.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment adaptive clustering refinement step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary
        adaptive_clustering_map = {str(old_cluster): [] for old_cluster in target_clusters}
        # Initialize adaptive tree node mapping dictionary (only validation purposes)
        adaptive_tree_node_map = {}
        # Get RVE hierarchical agglomerative base clustering
        labels = self.voxels_clusters.flatten()
        # Get maximum cluster label in RVE hierarchical agglomerative base clustering (avoid
        # that new cluster labels override existing ones)
        new_cluster_label = max(labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase cluster labels (conversion to int32 is required to avoid
            # raising a TypeError in scipy hierarchy leaders function)
            cluster_labels = labels[self._phase_voxel_flatidx[mat_phase]].astype('int32')
            # Convert hierarchical original clustering linkage matrix into tree object
            rootnode, nodelist = sciclst.to_tree(self._phase_linkage_matrix[mat_phase],
                                                 rd=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build initial cluster-node mapping between cluster labels and tree nodes
            # associated with the hierarchical agglomerative base clustering
            if not mat_phase in self._phase_map_cluster_node.keys():
                # Get root nodes of hierarchical clustering corresponding to an horizontal
                # cut defined by a flat clustering assignment vector. L contains the tree
                # nodes ids while M contains the corresponding cluster labels
                L, M = sciclst.leaders(self._phase_linkage_matrix[mat_phase],
                                       cluster_labels)
                # Build initial cluster-node mapping
                self._phase_map_cluster_node[mat_phase] = dict(zip([str(x) for x in M], L))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over target clusters
            for i in range(len(target_clusters)):
                # Get target cluster label
                target_cluster = target_clusters[i]
                # If target cluster label belongs to the current material phase, then get
                # target cluster tree node instance. Otherwise skip to the next target
                # cluster
                if str(target_cluster) in self._phase_map_cluster_node[mat_phase].keys():
                    target_node = nodelist[
                        self._phase_map_cluster_node[mat_phase][str(target_cluster)]]
                else:
                    continue
                # Get total number of leaf nodes associated to target node
                n_leaves = target_node.get_count()
                # Compute total number of tree node splits
                n_splits = int(np.round(self._split_greed*(n_leaves - 1)))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize child nodes list
                child_nodes = []
                # Initialize child target nodes list
                child_target_nodes = []
                # Split loop
                for i_split in range(n_splits):
                    # Set node to be splitted
                    if i_split == 0:
                        # In the first split operation, the node to be splitted is the
                        # target cluster tree node
                        node_to_split = target_node
                    else:
                        # Get maximum linkage distance child target node and remove it from
                        # the child target nodes list
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
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize target cluster mapping
                adaptive_clustering_map[str(target_cluster)] = []
                # Remove target cluster from cluster node mapping
                self._phase_map_cluster_node[mat_phase].pop(str(target_cluster))
                # Update target cluster mapping and flat clustering labels
                for node in child_nodes:
                    # Increment new cluster label
                    new_cluster_label += 1
                    # Add new cluster to target cluster mapping
                    adaptive_clustering_map[str(target_cluster)].append(new_cluster_label)
                    # Update flat clustering labels
                    cluster_labels[node.pre_order()] = new_cluster_label
                    # Update cluster-node mapping
                    self._phase_map_cluster_node[mat_phase][str(new_cluster_label)] = \
                        node.id
                # Update adaptive tree node mapping dictionary (only validation purposes)
                adaptive_tree_node_map[str(target_node.id)] = [x.id for x in child_nodes]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update RVE hierarchical agglomerative clustering
            labels[self._phase_voxel_flatidx[mat_phase]] = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build hiearchical adaptive CRVE from the hierarchical agglomerative adaptive
        # clustering
        info.displayinfo('5', 'Building HA-CRVE clustering (adaptive step ' +
                              str(self._adaptive_step) + ')...', 2)
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing HA-CRVE descriptors...', 2)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [adaptive_clustering_map, adaptive_tree_node_map]
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
        print(3*'\n' + 'Hierarchical adaptive clustering report\n' + 80*'-')
        # Print adaptive clustering refinement step
        print('\nAdaptive refinement step: ', self._adaptive_step)
        # Print hiearchical adaptive CRVE
        print('\n\n' + 'Adaptive clustering: ' + '(' +
              str(len(np.unique(self.voxels_clusters))) + ' clusters)' + '\n\n',
              self.voxels_clusters)
        # Print adaptive clustering mapping
        print('\n\n' + 'Adaptive cluster mapping: ')
        for mat_phase in self._material_phases:
            print('\n  Material phase ' + mat_phase + ':\n')
            for old_cluster in adaptive_clustering_map.keys():
                if adaptive_clustering_map[str(old_cluster)][0] in \
                    self.phase_clusters[mat_phase]:
                    print('    Old cluster: ' + '{:>4s}'.format(old_cluster) +
                          '  ->  ' +
                          'New clusters: ', adaptive_clustering_map[str(old_cluster)])
        # Print adaptive tree node mapping
        print('\n\n' + 'Adaptive tree node mapping (validation): ' + '\n')
        for old_node in adaptive_tree_node_map.keys():
            print('  Old node: ' + '{:>4s}'.format(old_node) +
                  '  ->  ' +
                  'New nodes: ', adaptive_tree_node_map[str(old_node)])
        # Print cluster-node mapping
        print('\n\n' + 'Cluster-Node mapping: ')
        for mat_phase in self._material_phases:
            print('\n  Material phase ' + mat_phase + ':\n')
            for new_cluster in self._phase_map_cluster_node[mat_phase].keys():
                print('    Cluster: ' + '{:>4s}'.format(new_cluster) +
                      '  ->  ' +
                      'Tree node: ',
                      self._phase_map_cluster_node[mat_phase][str(new_cluster)])
