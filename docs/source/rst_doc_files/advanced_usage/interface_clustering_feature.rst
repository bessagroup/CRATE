
Interface: Clustering feature
=============================

Context
-------
One of the key ingredients of any clustering-based reduced-order model is the "compression" of the material RVE into a cluster-reduced RVE (CRVE) by means of a clustering-based domain decomposition, i.e., a cluster analysis that decomposes the spatial domain into a given number of material clusters. A material cluster can be defined as a group of domain points that exhibit some sort of similarity according to a given set of clustering features or attributes available at the point level. To take advantage of prior knowledge about such a similarity, the cluster analysis is performed independently for each material phase of the RVE.

.. note::
   For a fundamental background on clustering-based reduced-order modeling, the interested reader is referred to `Ferreira (2022) <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_ (see Chapter 4 and Appendix C) and references therein.

One simple (multi-dimensional) clustering feature could be the vector of Cartesian coordinates, from which the cluster analysis would result in a clustering-based domain decomposition resembling a Voronoi diagram. However, given that the goal is to reduce the computational cost of the material RVE analysis without losing the ability to accurately capture the material mechanical behavior, it makes more sense to seek grouping points with similar mechanical behavior instead. Not surprisingly, computing such mechanical-based clustering features may involve performing some direct numerical simulation (DNS) analyses of the RVE under given macro-scale loading conditions. For instance, this is the case when adopting the fourth-order local elastic strain concentration tensor as the (multi-dimensional) clustering feature.

Therefore, given a particular application and the constitutive behavior of the RVE material phases, it may be of interest to implement new clustering features that result in the definition of a more suitable CRVE.

----

Implementation steps
--------------------

The implementation of a new clustering feature in CRATE involves **two different stages**:

* **Stage 1** - Implement the new clustering feature algorithm

    * **Step 1.1** - Create a Python module with the name of the clustering feature (e.g., :code:`new_feature_algorithm.py`) in the directory :code:`crate.clustering.features` (create directory if not existent);

    * **Step 1.2** - In :code:`new_feature_algorithm.py`, import the clustering feature algorithm interface (:py:class:`~crate.clustering.clusteringdata.FeatureAlgorithm`), derive a class for the new clustering feature algorithm (e.g., :code:`NewFeatureAlgorithm`), and implement the abstract methods (look for the @abstractmethod decorator) established by the clustering feature algorithm interface:

        .. code-block:: python

           # clustering.features.new_feature_algorithm.py

           from clustering.clusteringdata import FeatureAlgorithm

           class NewFeatureAlgorithm(FeatureAlgorithm):
               """New clustering feature algorithm."""

               def __init__(self):
                   """Constructor."""

                   pass

* **Stage 2** - Setup the new clustering feature descriptors

    * **Step 2.1** - In :py:mod:`crate.clustering.clusteringdata`, import the new clustering feature algorithm (:code:`NewFeatureAlgorithm`) implemented during Stage 1 in :code:`new_feature_algorithm.py`;

    * **Step 2.2** - In :py:mod:`crate.clustering.clusteringdata`, check the already existent clustering features (read the documentation of :py:func:`~crate.clustering.clusteringdata.get_available_clustering_features`) and choose a unique integer identifier :code:`id` for the new clustering feature;

    * **Step 2.3** - In :py:mod:`crate.clustering.clusteringdata`, implement the descriptors of the new clustering feature (e.g., number of dimensions, feature algorithm, macro-scale loadings) in the function :py:func:`~crate.clustering.clusteringdata.get_available_clustering_features`:

        .. code-block:: python

           # crate.clustering.clusteringdata.py

           from clustering.features.new_feature_algorithm import NewFeatureAlgorithm

           def get_available_clustering_features(strain_formulation, problem_type):
               """Get available clustering features and corresponding descriptors."""

               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               # New clustering feature:
               # Set number of feature dimensions
               n_feature_dim = ...
               # Set feature computation algorithm
               feature_algorithm = NewFeatureAlgorithm()
               # Set macroscale strain loadings required to compute feature
               mac_strains = ...
               # Set macroscale strain magnitude factor
               strain_magnitude_factor = ...
               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               # Assemble to available clustering features
               features_descriptors['< id >'] = (n_feature_dim, feature_algorithm,
                                                 mac_strains, strain_magnitude_factor)
               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * **Step 2.4** - In :py:mod:`crate.clustering.clusteringdata`, add the documentation of the new clustering feature in the docstring of :py:func:`~crate.clustering.clusteringdata.get_available_clustering_features`.

----

Recommendations
---------------

* If you are not familiar with the implementation of a clustering feature in CRATE, it is recommended that you first take a look into the implementation of the clustering features already available (:py:func:`~crate.clustering.clusteringdata.get_available_clustering_features`). Despite being embedded directly in :py:mod:`crate.clustering.clusteringdata`, the fundamental implementation steps of these clustering features follows the steps previously outlined and are fully documented.
