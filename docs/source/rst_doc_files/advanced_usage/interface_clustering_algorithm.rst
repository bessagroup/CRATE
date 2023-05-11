
Interface: Clustering algorithm
===============================

Context
-------
One of the key ingredients of any clustering-based reduced-order model is the "compression" of the material RVE into a cluster-reduced RVE (CRVE) by means of a clustering-based domain decomposition, i.e., a cluster analysis that decomposes the spatial domain into a given number of material clusters. A material cluster can be defined as a group of domain points that exhibit some sort of similarity according to a given set of clustering features or attributes available at the point level. To take advantage of prior knowledge about such a similarity, the cluster analysis is performed independently for each material phase of the RVE.

.. note::
   For a fundamental background on clustering-based reduced-order modeling, the interested reader is referred to `Ferreira (2022) <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_ (see Chapter 4 and Appendix C) and references therein.

To perform the cluster analysis of the RVE it is necessary to select (at least) one suitable clustering algorithm from the vast amount of algorithms available on the literature on unsupervised machine learning (e.g., `Aggarwal, C.C., and Reddy C.K. (2018) <https://www.google.com/books/edition/Data_Clustering/cH50DwAAQBAJ?>`_, `Gan et al. (2020) <https://www.google.com/books/edition/Data_Clustering_Theory_Algorithms_and_Ap/r4wIEAAAQBAJ?>`_). For instance, one well-known and widely applied centroid-based clustering algorithm is called K-Means clustering, for which several implementation algorithms can often be found in open-source libraries.

Therefore, given a particular application and the constitutive behavior of the RVE material phases, it may be of interest to implement new clustering algorithms that result in the definition of a more suitable CRVE. It is also worth noticing that there are several Python open-source libraries (e.g., `SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html>`_, `Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_) that provide many ready-to-use implementations of well-established clustering algorithms.

----

Implementation steps
--------------------

The implementation of a new clustering feature in CRATE involves **five fundamental steps**:

* **Step 1** - Create a Python module with the name of the new clustering algorithm (e.g., :code:`new_clustering_algorithm.py`) in the directory :code:`crate.clustering.algorithms` (create directory if not existent);

* **Step 2** - In :code:`new_clustering_algorithm.py`, import the clustering algorithm interface (:py:class:`~crate.clustering.clusteringalgs.ClusteringAlgorithm`) and derive a class for the new clustering algorithm (e.g., :code:`NewClusteringAlgorithm`):

    .. code-block:: python

       # clustering.algorithms.new_clustering_algorithm.py

       from clustering.clusteringalgs import ClusteringAlgorithm

       class NewClusteringAlgorithm(ClusteringAlgorithm):
           """New clustering algorithm."""

* **Step 3** - In :py:mod:`crate.clustering.clusteringalgs`, check the already existent clustering algorithms (:py:attr:`~crate.clustering.clusteringalgs.ClusterAnalysis.available_clustering_alg`), choose a unique integer identifier :code:`id` for the new clustering algorithm and add it as a new item of :py:attr:`~crate.clustering.clusteringalgs.ClusterAnalysis.available_clustering_alg`:

    .. code-block:: python

       # clustering.clusteringalgs.py

       class ClusterAnalysis:
           """Interface to perform a cluster analysis."""

           available_clustering_alg = {'< id >': 'New clustering algorithm', }

* **Step 4** - In :py:mod:`crate.clustering.clusteringalgs`, import and add the initialization of the new clustering algorithm in the :py:meth:`~crate.clustering.clusteringals.ClusterAnalysis.get_fitted_estimator` method of class :py:class:`~crate.clustering.clusteringals.ClusterAnalysis`:

    .. code-block:: python
       :emphasize-lines: 3, 12-14

       # clustering.clusteringals.py

       from clustering.algorithms.new_clustering_algorithm import NewClusteringAlgorithm

       class ClusterAnalysis:
           """Interface to perform a cluster analysis."""

           def get_fitted_estimator(self, data_matrix, clust_alg_id, n_clusters):
               """Get cluster labels and clustering fitted estimator."""

               # Instantiate clustering algorithm
               if clust_alg_id == '< id >':
                   # Instantiate New Clustering Algorithm
                   clust_alg = NewClusteringAlgorithm()
               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               else:
                   raise RuntimeError('Unknown clustering algorithm.')

* **Step 5** - Perform the complete implementation of the new clustering algorithm in :code:`new_clustering_algorithm.py` by developing the class :code:`NewClusteringAlgorithm` and implementing the abstract methods (look for the @abstractmethod decorator) established by the clustering algorithm interface (:py:class:`~crate.clustering.clusteringalgs.ClusteringAlgorithm`).

----

Recommendations
---------------

* If you are not familiar with the implementation of a clustering algorithm in CRATE, it is recommended that you first take a look into the implementation of the clustering algorithms already available (:py:mod:`crate.clustering.clusteringalgs`). Despite being embedded directly in :py:mod:`crate.clustering.clusteringalgs`, the fundamental implementation steps of these clustering algorithms follows the steps previously outlined and are fully documented;

* In the particular case of a hierarchical agglomerative clustering algorithm and when access to the linkage matrix is required, derive a class for the new clustering algorithm (e.g., :code:`NewClusteringAlgorithm`) from the hierarchical agglomerative clustering algorithm interface (:py:class:`~crate.clustering.clusteringalgs.AgglomerativeAlgorithm`) instead;

* Take advantage of the several Python open-source libraries (e.g., `SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html>`_, `Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_) that provide many well-established clustering algorithms and implement a simple wrapper class compliant with the clustering algorithm interface (:py:class:`~crate.clustering.clusteringalgs.ClusteringAlgorithm`).
