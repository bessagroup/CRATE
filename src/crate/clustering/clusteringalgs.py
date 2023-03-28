"""Clustering analysis and algorithms.

This module includes the interface to perform a cluster analysis as well as
the interface to implement any clustering algorithm. It also includes several
wrappers over clustering algorithms available on open-source libraries (e.g.,
`SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html#>`_,
`Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_).

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
AgglomerativeSP
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
                                '2': 'Mini-Batch K-Means (scikit-learn)',
                                '3': 'Agglomerative (scipy)', }
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
                                 tol=1e-4, random_state=None,
                                 algorithm='lloyd', n_clusters=n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif clust_alg_id == '2':
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
        elif clust_alg_id == '3':
            # Instatiate Agglomerative clustering
            clust_alg = AgglomerativeSP(0, method='ward', metric='euclidean',
                                        criterion='maxclust',
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
        return cluster_labels, clust_alg, is_n_clusters_satisfied
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
                 random_state=None, algorithm='lloyd', n_clusters=None):
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
        algorithm : {'lloyd', 'elkan'}, default='lloyd'
            K-Means algorithm to use. 'lloyd' is the classical EM-style
            algorithm, 'elkan' uses the triangle inequality to speed up
            convergence.
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
        init: {‘k-means++’, ‘random’, numpy.ndarray, callable}, \
              default=’k-means++’
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
        self._clst_alg = skclst.MiniBatchKMeans(
            n_clusters=self.n_clusters, init=self._init, n_init=self._n_init,
            max_iter=self._max_iter, tol=self._tol,
            random_state=self._random_state, init_size=self._init_size,
            reassignment_ratio=self._reassignment_ratio)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute cluster centers (fitted estimator) and predict cluster label
        # (prediction) for each dataset item
        cluster_labels = self._clst_alg.fit_predict(data_matrix,
                                                    sample_weight=None)
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
