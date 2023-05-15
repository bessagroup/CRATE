
Interface: DNS solver
=====================

Context
-------
One of the key ingredients of any clustering-based reduced-order model is the "compression" of the material RVE into a cluster-reduced RVE (CRVE) by means of a clustering-based domain decomposition, i.e., a cluster analysis that decomposes the spatial domain into a given number of material clusters. A material cluster can be defined as a group of domain points that exhibit some sort of similarity according to a given set of **clustering features or attributes** available at the point level. To take advantage of prior knowledge about such a similarity, the cluster analysis is performed independently for each material phase of the RVE.

.. note::
   For a fundamental background on clustering-based reduced-order modeling, the interested reader is referred to `Ferreira (2022) <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_ (see Chapter 4 and Appendix C) and references therein.

One simple (multi-dimensional) clustering feature could be the vector of Cartesian coordinates, from which the cluster analysis would result in a clustering-based domain decomposition resembling a Voronoi diagram. However, given that the goal is to reduce the computational cost of the material RVE analysis without losing the ability to accurately capture the material mechanical behavior, it makes more sense seeking to group points with **similar mechanical behavior** instead. Not surprisingly, computing such **mechanical-based clustering features** may involve performing some **direct numerical simulation (DNS)** analyses of the RVE under given macro-scale loading conditions. For instance, this is the case when adopting the fourth-order local elastic strain concentration tensor as the (multi-dimensional) clustering feature.

Therefore, when selecting clustering features that require data based on the mechanical behavior of the RVE, it is necessary to implement a direct numerical simulation (DNS) multi-scale method to compute such data.

.. note::
   For a fundamental background on multi-scale modeling, the interested reader is referred to `Ferreira (2022) <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_ (see Chapter 2: 2.10-2.14, Chapter 4: 4.6, and Appendix B) and references therein.

----

Implementation steps
--------------------
The implementation of a **new direct numerical simulation (DNS) multi-scale method** in CRATE involves **five fundamental steps**:

* **Step 1** - Create a Python module with the name of the new DNS multi-scale method (e.g., :code:`new_dns_method.py`) in the directory :py:mod:`cratepy.clustering.solution`;

* **Step 2** - In :code:`new_dns_method.py`, import the DNS multi-scale method interface (:py:class:`~cratepy.clustering.solution.dnshomogenization.DNSHomogenizationMethod`) and derive a class for the new DNS multi-scale method (e.g., :code:`NewDNSMethod`):

    .. code-block:: python

       # clustering.solution.new_dns_method.py

       from clustering.solution import DNSHomogenizationMethod

       class NewDNSMethod(DNSHomogenizationMethod):
           """New DNS multi-scale method."""

* **Step 3** - Choose a unique identifier :code:`id` for the new DNS multi-scale method;

* **Step 4** - In :py:mod:`cratepy.clustering.rveelasticdatabase`, import and add the initialization of the new DNS multi-scale method in the :py:meth:`~cratepy.clustering.rveelasticdatabase.RVEElasticDatabase.compute_rve_response_database` method of class :py:class:`~cratepy.clustering.rveelasticdatabase.RVEElasticDatabase`:

    .. code-block:: python
       :emphasize-lines: 3, 13-14

       # clustering.rveelasticdatabase.py

       from clustering.solution.new_dns_method import NewDNSMethod

       class RVEElasticDatabase:
           """RVE local elastic response database class."""

           def compute_rve_response_database(self, dns_method, dns_method_data,
                                             mac_strains, is_strain_sym):
               """Compute RVE's local elastic strain response database."""

               # Instantiate homogenization-based multi-scale method
               if dns_method == '< id >':
                   homogenization_method = NewDNSMethod()
               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               else:
                   raise RuntimeError('Unknown homogenization-based multi-scale '
                                      'method.')

* **Step 5** - Perform the complete implementation of the new DNS multi-scale method in :code:`new_dns_method.py` by developing the class :code:`NewDNSMethod` and implementing the abstract methods (look for the @abstractmethod decorator) established by the DNS multi-scale method interface (:py:class:`~cratepy.clustering.solution.dnshomogenization.DNSHomogenizationMethod`).

----

Recommendations
---------------

* If you are not familiar with the implementation of a DNS multi-scale method in CRATE, it is **recommended** that you first take a look into the implementation of the DNS multi-scale methods already available (:py:mod:`cratepy.clustering.solution`). The implementation of these DNS multi-scale methods follows the steps previously outlined and are fully documented;
