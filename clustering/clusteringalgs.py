"""Clustering analysis and algorithms.

This module includes the interface to perform a cluster analysis as well as
the interface to implement any clustering algorithm. It also includes several
wrappers over clustering algorithms available on open-source libraries (e.g.,
`SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html#>`_,
`Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_) and
others.

Classes
-------
ClusterAnalysis
    Interface to perform a cluster analysis.
ClusteringAlgorithm
    Clustering algorithm interface.
AgglomerativeAlgorithm
    Hierarchical agglomerative interface.
KMeansSK
    K-Means clustering algorithm (wrapper).
MiniBatchKMeansSK
    Mini-Batch K-Means clustering algorithm (wrapper).
BirchSK
    Birch clustering algorithm (wrapper).
AgglomerativeSK
    Agglomerative clustering algorithm (wrapper).
AgglomerativeSP
    Agglomerative clustering algorithm (wrapper).
BirchPC
    Birch clustering algorithm (wrapper).
CurePC
    Cure clustering algorithm (wrapper).
KMeansPC
    K-Means clustering algorithm (wrapper).
XMeansPC
    X-Means clustering algorithm (wrapper).
AgglomerativeFC
    Agglomerative clustering algorithm (wrapper).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
# Third-party
import numpy as np
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
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira',]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                              Cluster Analysis
# =============================================================================
class ClusterAnalysis:
    """Interface to perform a cluster analysis.

    Attributes
    ----------
    available_clust_alg : dict
        Available clustering algorithms (item, str) and associated identifiers
        (key, str).

    Methods
    -------
    get_fitted_estimator(self, data_matrix, clust_alg_id, n_clusters):
        Get cluster labels and clustering fitted estimator.
    """
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
    # -------------------------------------------------------------------------
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    def get_fitted_estimator(self, data_matrix, clust_alg_id, n_clusters):
        """Get cluster labels and clustering fitted estimator.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).
        clust_alg_id : str
            Clustering algorithm identifier.
        n_clusters : int
            The number of clusters to find.

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        clust_alg : function
            Clustering fitted estimator.
        is_n_clusters_satisfied : bool
            True if the number of clusters obtained from the cluster analysis
            matches the prescribed number of clusters, False otherwise.
        """
        # Get number of dataset items
        n_items = data_matrix.shape[0]
        # Initialize clustering
        cluster_labels = np.full(n_items, -1, dtype=int)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate clustering algorithm
        if clust_alg_id == '1':
            # Set number of full batch K-Means clusterings (with different
            # initializations)
            n_init = 10
            # Instantiate K-Means
            clust_alg = KMeansSK(init='k-means++', n_init=n_init, max_iter=300,
                                 tol=1e-4, random_state=None, algorithm='auto',
                                 n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '2':
            # Instatiante K-Means
            clust_alg = KMeansPC(tolerance=1e-03, itermax=200,
                                 n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '3':
            # Set size of the mini-batches
            batch_size = 100
            # Set number of random initializations
            n_init = 3
            # Intantiate Mini-Batch K-Means
            clust_alg = MiniBatchKMeansSK(init='k-means++', max_iter=100,
                                          tol=0.0, random_state=None,
                                          batch_size=batch_size,
                                          max_no_improvement=10,
                                          init_size=None, n_init=n_init,
                                          reassignment_ratio=0.01,
                                          n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '4':
            # Instantiate Agglomerative clustering
            clust_alg = AgglomerativeSK(affinity='euclidean', memory=None,
                                        connectivity=None,
                                        compute_full_tree='auto',
                                        linkage='ward',
                                        distance_threshold=None,
                                        n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '5':
            # Instatiate Agglomerative clustering
            clust_alg = AgglomerativeSP(0, method='ward', metric='euclidean',
                                        criterion='maxclust',
                                        n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '6':
            # Instatiate Agglomerative clustering
            clust_alg = AgglomerativeFC(0, method='ward', metric='euclidean',
                                        criterion='maxclust',
                                        n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '7':
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clust_alg = BirchSK(threshold=threshold,
                                branching_factor=branching_factor,
                                n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '8':
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clust_alg = BirchPC(threshold=threshold,
                                branching_factor=branching_factor,
                                n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '9':
            # Instantiate Cure
            clust_alg = CurePC(number_represent_points=5, compression=0.5,
                               n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '10':
            # Instantiate X-Means
            clust_alg = XMeansPC(tolerance=2.5e-2, repeat=1,
                                 n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown clustering algorithm.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform cluster analysis
        cluster_labels = clust_alg.perform_clustering(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the dataset items have been labeled
        if np.any(cluster_labels == -1):
            raise RuntimeError('At least one dataset item has not been '
                               'labeled during the cluster analysis.')
        # Check number of clusters formed
        if len(set(cluster_labels)) != n_clusters:
            is_n_clusters_satisfied = False
        else:
            is_n_clusters_satisfied = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return [cluster_labels, clust_alg, is_n_clusters_satisfied]
#
#                                               Interface: Clustering algorithm
# =============================================================================
class ClusteringAlgorithm(ABC):
    """Clustering algorithm interface.

    Methods
    -------
    perform_clustering(self, data_matrix):
        *abstract*: Perform cluster analysis and get cluster label of each
        dataset item.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        pass
# -----------------------------------------------------------------------------
class AgglomerativeAlgorithm(ClusteringAlgorithm):
    """Hierarchical agglomerative interface.

    Methods
    -------
    get_linkage_matrix(self):
        *abstract*: Get hierarchical agglomerative clustering linkage matrix.
    """
    @abstractmethod
    def get_linkage_matrix(self):
        """Get hierarchical agglomerative clustering linkage matrix.

        Returns
        -------
        linkage_matrix : numpy.ndarray (2d)
            Linkage matrix associated with the hierarchical agglomerative
            clustering (numpy.ndarray of shape (n-1, 4)). At the i-th iteration
            the clusterings with indices Z[i, 0] and Z[i, 1], with distance
            Z[i, 2], are merged, forming a new cluster that contains Z[i, 3]
            original dataset items. All cluster indices j >= n refer to the
            cluster formed in Z[j-n, :].

        Notes
        -----
        The hierarchical agglomerative clustering linkage matrix follows the
        definition of `SciPy <https://docs.scipy.org/>`_ agglomerative
        clustering algorithm (see `here <https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.
        cluster.hierarchy.linkage>`_).
        """
        pass
#
#                                                         Clustering algorithms
# =============================================================================
class KMeansSK(ClusteringAlgorithm):
    """K-Means clustering algorithm (wrapper).

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(self, init='k-means++', n_init=10, max_iter=300, tol=1e-4,
                 random_state=None, algorithm='auto', n_clusters=None):
        """Constructor.

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
            Convergence tolerance (based on Frobenius norm of the different in
            the cluster centers of two consecutive iterations).
        random_state : {int, RandomState}, default=None
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        algorithm : {'auto', 'full', 'elkan'}, default='auto'
            K-Means algorithm to use. 'full' is the classical EM-style
            algorithm, 'elkan' uses the triangle inequality to speed up
            convergence. 'auto' currently chooses 'elkan'
            (scikit-learn 0.23.2).
        """
        self.n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._algorithm = algorithm
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate scikit-learn K-Means clustering algorithm
        self._clst_alg = skclst.KMeans(n_clusters=self.n_clusters,
                                       init=self._init, n_init=self._n_init,
                                       max_iter=self._max_iter, tol=self._tol,
                                       random_state=self._random_state,
                                       algorithm=self._algorithm)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute cluster centers (fitted estimator) and predict cluster label
        # (prediction) for each dataset item
        cluster_labels = self._clst_alg.fit_predict(data_matrix,
                                                    sample_weight=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class MiniBatchKMeansSK(ClusteringAlgorithm):
    """Mini-Batch K-Means clustering algorithm (wrapper).

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.
    MiniBatchKMeans>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.

    Notes
    -----
    The Mini-Batch K-Means clustering algorithm is taken from scikit-learn
    (https://scikit-learn.org). Further information can be found in there.
    """
    def __init__(self, init='k-means++', max_iter=100, tol=0.0,
                 random_state=None, batch_size=100, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01,
                 n_clusters=None):
        """Constructor.

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
            Convergence tolerance (based on Frobenius norm of the different in
            the cluster centers of two consecutive iterations).
        random_state : int, RandomState instance, default=None
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic.
        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy): the only
            algorithm is initialized by running a batch KMeans on a random
            subset of the data.
        n_init : int, default=3
            Number of random initializations that are tried (best of
            initializations is used to run the algorithm).
        reassignment_ratio : float, default=0.01
            Control the fraction of the maximum number of counts for a center
            to be reassigned.
        """
        self.n_clusters = n_clusters
        self._init = init
        self._n_init = n_init
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._init_size = init_size
        self._reassignment_ratio = reassignment_ratio
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate scikit-learn Mini-Batch K-Means clustering algorithm
        self._clst_alg = skclst.MiniBatchKMeans(n_clusters=self.n_clusters,
            init=self._init, n_init=self._n_init, max_iter=self._max_iter,
            tol=self._tol, random_state=self._random_state,
            init_size=self._init_size,
            reassignment_ratio=self._reassignment_ratio)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute cluster centers (fitted estimator) and predict cluster label
        # (prediction) for each dataset item
        cluster_labels = self._clst_alg.fit_predict(data_matrix,
                                                    sample_weight=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class BirchSK(ClusteringAlgorithm):
    """Birch clustering algorithm (wrapper).

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=None):
        """Constructor.

        Parameters
        ----------
        threshold : float, default=0.5
            The radius of the subcluster obtained by merging a new sample and
            the closest subcluster should be lesser than the threshold.
            Otherwise a new subcluster is started. Setting this value to be
            very low promotes splitting and vice-versa.
        branching_factor : int, default=50
            Maximum number of CF subclusters in each node. If a new samples
            enters such that the number of subclusters exceed the
            branching_factor then that node is split
            into two nodes with the subclusters redistributed in each. The
            parent subcluster of that node is removed and two new subclusters
            are added as parents of the 2 split nodes.
        n_clusters : {int, sklearn.cluster model}, default=None
            Number of clusters to find after the final clustering step, which
            treats the subclusters from the leaves as new samples.

            * `None`: the final clustering step is not performed and the \
              subclusters are returned as they are.

            * `sklearn.cluster estimator`: if a model is provided, the model \
              is fit treating the subclusters as new samples and the initial \
              data is mapped to the label of the closest subcluster.

            * `int`: the model fit is `AgglomerativeClustering` with \
              `n_clusters` set to be equal to the int.
        """
        self.n_clusters = n_clusters
        self._threshold = threshold
        self._branching_factor = branching_factor
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate scikit-learn Birch clustering algorithm
        self._clst_alg = skclst.Birch(threshold=self._threshold,
                                      branching_factor=self._branching_factor,
                                      n_clusters=self.n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering and return cluster labels
        cluster_labels = self._clst_alg.fit_predict(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class AgglomerativeSK(ClusteringAlgorithm):
    """Agglomerative clustering algorithm (wrapper).

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.
    AgglomerativeClustering>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(self, affinity='euclidean', memory=None, connectivity=None,
                 compute_full_tree='auto', linkage='ward',
                 distance_threshold=None, n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find. It must be `None` if
            `distance_threshold` is not `None`.
        affinity : {str, callable}, default='euclidean'
            Metric used to compute the linkage. Can be 'euclidean', 'l1', 'l2',
            'manhattan', 'cosine', or 'precomputed'. If linkage is 'ward', only
            'euclidean' is accepted. If 'precomputed', a distance matrix
            (instead of a similarity matrix) is needed as input for the fit
            method.
        memory : {str, joblib.Memory interface}, default=None
            Used to cache the output of the computation of the tree. By
            default, no caching is done. If a string is given, it is the path
            to the caching directory.
        connectivity : {array-like, callable}, default=None
            Connectivity matrix. Defines for each sample the neighboring
            samples following a given structure of the data. This can be a
            connectivity matrix itself or a callable that transforms the data
            into a connectivity matrix, such as derived from kneighbors_graph.
            Default is `None`, i.e, the hierarchical clustering algorithm is
            unstructured.
        compute_full_tree : {'auto', bool}, default='auto'
            Stop early the construction of the tree at n_clusters. This is
            useful to decrease computation time if the number of clusters is
            not small compared to the number of samples. This option is useful
            only when specifying a connectivity matrix. Note also that when
            varying the number of clusters and using caching, it may be
            advantageous to compute the full tree. It must be `True` if
            `distance_threshold` is not `None`. By default `compute_full_tree`
            is 'auto', which is equivalent to `True` when `distance_threshold`
            is not `None` or that `n_clusters` is inferior to the maximum
            between 100 or `0.02 * n_samples`. Otherwise, 'auto' is equivalent
            to `False`.
        linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
            Which linkage criterion to use. The linkage criterion determines
            which distance to use between sets of observation. The algorithm
            will merge the pairs of cluster that minimize this criterion.

            * 'ward' minimizes the variance of the clusters being merged.

            * 'average' uses the average of the distances of each observation \
              of the two sets.

            * 'complete' linkage uses the maximum distances between all \
              observations of the two sets.

            * 'single' uses the minimum of the distances between all \
              observations of the two sets.
        distance_threshold : float, default=None
            The linkage distance threshold above which, clusters will not be
            merged. If not `None`, `n_clusters` must be `None` and
            `compute_full_tree` must be `True`.
        """
        self.n_clusters = n_clusters
        self._affinity = affinity
        self._memory = memory
        self._connectivity = connectivity
        self._compute_full_tree = compute_full_tree
        self._linkage = linkage
        self._distance_threshold = distance_threshold
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate scikit-learn Birch clustering algorithm
        self._clst_alg = skclst.AgglomerativeClustering(
            n_clusters=self.n_clusters, affinity=self._affinity,
            memory=self._memory, connectivity=self._connectivity,
            compute_full_tree=self._compute_full_tree, linkage=self._linkage,
            distance_threshold=self._distance_threshold)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fit the hierarchical clustering and return cluster labels
        cluster_labels = self._clst_alg.fit_predict(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class AgglomerativeSP(AgglomerativeAlgorithm):
    """Agglomerative clustering algorithm (wrapper).

    Documentation: see `here <https://docs.scipy.org/doc/scipy/reference/
    cluster.hierarchy.html>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.

    Attributes
    ----------
    _linkage_matrix : numpy.ndarray (2d)
        Linkage matrix associated with the hierarchical agglomerative
        clustering (numpy.ndarray of shape (n-1, 4)). At the i-th iteration
        the clusterings with indices Z[i, 0] and Z[i, 1], with distance
        Z[i, 2], are merged, forming a new cluster that contains Z[i, 3]
        original dataset items. All cluster indices j >= n refer to the
        cluster formed in Z[j-n, :].
    """
    def __init__(self, t, method='ward', metric='euclidean',
                 criterion='maxclust', n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find.
        t : {int, float}
            Scalar parameter associated to the criterion used to form a flat
            clustering. Threshold (float) with criterion in {'inconsistent',
            'distance', 'monocrit'} or maximum number of clusters with
            criterion in {'maxclust', 'maxclust_monocrit'}.
        method : {'single', 'complete', 'average', 'weighted', 'centroid', \
                  'median', 'ward'}, default='ward'
            Linkage criterion.
        metric : {str, function}, default='euclidean'
            Distance metric to use when the input data matrix is a
            numpy.ndarray of observation vectors, otherwise ignored. Options:
            {'cityblock', 'euclidean', 'cosine', ...}.
        criterion : str, {'inconsistent', 'distance', 'maxclust', 'monocrit', \
                          'maxclust_monocrit'}, default='maxclust'
            Criterion used to form a flat clustering (i.e., perform a
            horizontal cut in the hierarchical tree).
        """
        self._t = t
        self.n_clusters = n_clusters
        self._method = method
        self._metric = metric
        self._criterion = criterion
        self._linkage_matrix = None
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Perform hierarchical clustering and encode it in a linkage matrix
        self._linkage_matrix = sciclst.linkage(data_matrix,
                                               method=self._method,
                                               metric=self._metric)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform horizontal cut in hierarchical tree and return cluster labels
        # (form a flat clustering)
        cluster_labels = sciclst.fcluster(self._linkage_matrix,
                                          self.n_clusters,
                                          criterion=self._criterion)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
    # -------------------------------------------------------------------------
    def get_linkage_matrix(self):
        """Get hierarchical agglomerative clustering linkage matrix.

        Returns
        -------
        linkage_matrix : numpy.ndarray (2d)
            Linkage matrix associated with the hierarchical agglomerative
            clustering (numpy.ndarray of shape (n-1, 4)). At the i-th iteration
            the clusterings with indices Z[i, 0] and Z[i, 1], with distance
            Z[i, 2], are merged, forming a new cluster that contains Z[i, 3]
            original dataset items. All cluster indices j >= n refer to the
            cluster formed in Z[j-n, :].

        Notes
        -----
        The hierarchical agglomerative clustering linkage matrix follows the
        definition of `SciPy <https://docs.scipy.org/>`_ agglomerative
        clustering algorithm (see `here <https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.
        cluster.hierarchy.linkage>`_).
        """
        if self._linkage_matrix is None:
            raise ValueError('Hierarchical agglomerative clustering linkage '
                             'matrix has not been computed yet.')
        return self._linkage_matrix
# =============================================================================
class BirchPC(ClusteringAlgorithm):
    """Birch clustering algorithm (wrapper).

    Documentation: see `here <https://pyclustering.github.io/docs/0.8.2/html/
    d6/d00/classpyclustering_1_1cluster_1_1birch_1_1birch.html>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(
        self, threshold=0.5, branching_factor=50, max_node_entries=200,
        type_measurement=pycftree.measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
        entry_size_limit=500, threshold_multiplier=1.5, n_clusters=None):
        """Constructor.

        Parameters
        ----------
        threshold : float, default=0.5
            CF-entry diameter that is used for CF-Tree construction (might
            increase if `entry_size_limit` is exceeded).
        branching_factor : int, default=50
            Maximum number of successor that might be contained by each
            non-leaf node in CF-Tree.
        max_node_entries : int, default=200
            Maximum number of entries that might be contained by each leaf node
            in CF-Tree.
        type_measurement : measurement type, \
                           default=CENTROID_EUCLIDEAN_DISTANCE
            Type of measurement used for calculation of distance metrics.
        entry_size_limit : int, default=500
            Maximum number of entries that can be stored in CF-Tree (if
            exceeded during creation of CF-Tree, then threshold is increased
            and CF-Tree is rebuilt).
        threshold_multiplier : float, default=1.5
            Multiplier used to increase the threshold when `entry_size_limit`
            is exceeded.
        n_clusters : int, default=None
            Number of clusters to find.
        """
        self.n_clusters = n_clusters
        self._threshold = threshold
        self._branching_factor = branching_factor
        self._max_node_entries = max_node_entries
        self._type_measurement = type_measurement
        self._entry_size_limit = entry_size_limit
        self._threshold_multiplier = threshold_multiplier
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate pyclustering Birch clustering algorithm
        self._clst_alg = pybirch.birch(
            data_matrix.tolist(), self.n_clusters, diameter=self._threshold,
            branching_factor=self._branching_factor,
            max_node_entries=self._max_node_entries,
            entry_size_limit=self._entry_size_limit,
            diameter_multiplier=self._threshold_multiplier)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters,
                                            data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class CurePC(ClusteringAlgorithm):
    """Cure clustering algorithm (wrapper).

    Documentation: see `here <https://pyclustering.github.io/docs/0.8.2/html/
    dc/d6d/classpyclustering_1_1cluster_1_1cure_1_1cure.html>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(self, number_represent_points=5, compression=0.5,
                 n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
        number_represent_points : int, default=5
            Number of representative points for each cluster.
        compression : float, default=0.5
            Coefficient that defines the level of shrinking of representation
            points toward the mean of the new created cluster after merging on
            each step (usually set between 0 and 1).
        """
        self.n_clusters = n_clusters
        self._number_represent_points = number_represent_points
        self._compression = compression
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instantiate pyclustering Cure clustering algorithm
        self._clst_alg = pycure.cure(
            data_matrix.tolist(), self.n_clusters,
            number_represent_points=self._number_represent_points,
            compression=self._compression)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters,
                                            data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class KMeansPC(ClusteringAlgorithm):
    """K-Means clustering algorithm (wrapper).

    Documentation: see `here <https://pyclustering.github.io/docs/0.8.2/html/
    da/d22/classpyclustering_1_1cluster_1_1kmeans_1_1kmeans.html>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(
        self, tolerance=1e-03, itermax=200,
        metric=pymetric.distance_metric(pymetric.type_metric.EUCLIDEAN_SQUARE),
        n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Number of clusters to find.
        tolerance : float, default=1e-03
            Convergence tolerance (based on the maximum value of change of
            cluster centers of two consecutive iterations).
        itermax : int, default=200
            Maximum number of iterations.
        metric : distance_metric, default=EUCLIDEAN_SQUARE
            Metric used for distance calculation between samples.
        """
        self.n_clusters = n_clusters
        self._tolerance = tolerance
        self._itermax = itermax
        self._metric = metric
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Instatiante cluster centers seeds using K-Means++
        amount_candidates = \
            pycenterinit.kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
        initial_centers = pycenterinit.kmeans_plusplus_initializer(
            data_matrix.tolist(), self.n_clusters,
            amount_candidates=amount_candidates).initialize()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate pyclustering K-Means clustering algorithm
        self._clst_alg = pykmeans.kmeans(data_matrix.tolist(), initial_centers,
                                         tolerance=self._tolerance,
                                         itermax=self._itermax,
                                         metric=self._metric)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters,
                                            data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class XMeansPC(ClusteringAlgorithm):
    """X-Means clustering algorithm (wrapper).

    Documentation: see `here <https://pyclustering.github.io/docs/0.8.2/html/
    dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.
    """
    def __init__(
        self, tolerance=2.5e-2,
        criterion=pyxmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION,
        repeat=1, n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            Maximum number of clusters than can be found.
        tolerance : float, default=2.5e-2
            Convergence tolerance (based on the maximum value of change of
            cluster centers of two consecutive iterations).
        criterion : splitting_type, BAYESIAN_INFORMATION_CRITERION
            Criterion to perform cluster splitting.
        repeat : int, default=1
            How many times K-Means should be run to improve parameters. Larger
            values increase the probability of finding global optimum.
        """
        self.n_clusters = n_clusters
        self._tolerance = tolerance
        self._criterion = criterion
        self._repeat = repeat
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Set initial numbers of clusters
        amount_initial_centers = max(1, int(0.1*self.n_clusters))
        # Instatiante cluster centers seeds using K-Means++
        initial_centers = pycenterinit.kmeans_plusplus_initializer(
            data_matrix.tolist(), amount_initial_centers).initialize()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate pyclustering X-Means clustering algorithm
        self._clst_alg = pyxmeans.xmeans(data_matrix.tolist(), initial_centers,
                                         kmax=self.n_clusters,
                                         tolerance=self._tolerance,
                                         criterion=self._criterion,
                                         repeat=self._repeat)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform clustering
        self._clst_alg.process()
        clusters = self._clst_alg.get_clusters()
        # Get type of cluster encoding (index list separation by default)
        type_clusters = self._clst_alg.get_cluster_encoding()
        # Instantiate cluster encoder (clustering result representor)
        encoder = pyencoder.cluster_encoder(type_clusters, clusters,
                                            data_matrix)
        # Change cluster encoding to index labeling
        encoder.set_encoding(pyencoder.type_encoding.CLUSTER_INDEX_LABELING)
        # Return cluster labels
        cluster_labels = np.array(encoder.get_clusters())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
# =============================================================================
class AgglomerativeFC(ClusteringAlgorithm):
    """Agglomerative clustering algorithm (wrapper).

    Documentation: see `here <http://danifold.net/fastcluster>`_.

    Methods
    -------
    perform_clustering(self, data_matrix):
        Perform cluster analysis and get cluster label of each dataset item.

    Attributes
    ----------
    Z : ndarray of shape (n-1, 4)
        Linkage matrix associated with the hierarchical clustering. At the
        i-th iteration the clusterings with indices Z[i, 0] and Z[i, 1], with
        distance Z[i, 2], are merged, forming a new cluster that contains
        Z[i, 3] original dataset items. All cluster indices j >= n refer to the
        cluster formed in Z[j-n, :].
    """
    def __init__(self, t, method='ward', metric='euclidean',
                 criterion='maxclust', n_clusters=None):
        """Constructor.

        Parameters
        ----------
        n_clusters : int, default=None
            The number of clusters to find.
        t : {int, float}
            Scalar parameter associated to the criterion used to form a flat
            clustering. Threshold (float) with criterion in {'inconsistent',
            'distance', 'monocrit'} or maximum number of clusters with
            criterion in {'maxclust', 'maxclust_monocrit'}.
        method : {'single', 'complete', 'average', 'weighted', 'centroid', \
                  'median', 'ward'}, default='ward'
            Linkage criterion.
        metric : {str, function}, default='euclidean'
            Distance metric to use when the input data matrix is a
            numpy.ndarray of observation vectors, otherwise ignored. Options:
            {'cityblock', 'euclidean', 'cosine', ...}.
        criterion : {'inconsistent', 'distance', 'maxclust', 'monocrit', \
                     'maxclust_monocrit'}, default='maxclust'
            Criterion used to form a flat clustering (i.e., perform a
            horizontal cut in the hierarchical tree).
        """
        self._t = t
        self.n_clusters = n_clusters
        self._method = method
        self._metric = metric
        self._criterion = criterion
        self._Z = None
    # -------------------------------------------------------------------------
    def perform_clustering(self, data_matrix):
        """Perform cluster analysis and get cluster label of each dataset item.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix containing the required data to perform the cluster
            analysis (numpy.ndarray of shape (n_items, n_features)).

        Returns
        -------
        cluster_labels : numpy.ndarray (1d)
            Cluster label (int) assigned to each dataset item.
        """
        # Perform hierarchical clustering and encode it in a linkage matrix
        self.Z = fastclst.linkage(data_matrix, method=self._method,
                                  metric=self._metric, preserve_input=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform horizontal cut in hierarchical tree and return cluster labels
        # (form a flat clustering)
        cluster_labels = sciclst.fcluster(self.Z, self.n_clusters,
                                          criterion=self._criterion)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_labels
