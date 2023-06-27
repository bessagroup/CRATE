
CRATE
=====

.. image:: ../media/logo/CRATE_logo_horizontal_long.png
   :width: 70 %
   :align: center

|

Summary
-------
**CRATE** (Clustering-based Nonlinear Analysis of Materials) is a Python project (package `cratepy <https://pypi.org/project/cratepy/>`_) developed in the context of computational mechanics to aid the design and development of new materials. Its main purpose is **performing multi-scale nonlinear analyses of heterogeneous materials** through a suitable coupling between first-order computational homogenization and clustering-based reduced-order modeling: given a representative volume element of the material microstructure and the corresponding material phase properties, CRATE computes the material's effective mechanical response when subject to a prescribed macro-scale loading path.

----

Statement of Need
-----------------
`cratepy <https://pypi.org/project/cratepy/>`_ is essentially a numerical tool for any application that requires material multi-scale simulations. Given the intrinsic clustering-based reduced-order modeling approach (e.g., `SCA <https://www.sciencedirect.com/science/article/pii/S0045782516301499>`_, `ASCA <https://www.sciencedirect.com/science/article/pii/S0045782522000895?via%3Dihub>`_), CRATE is mostly useful in applications where the computational cost of standard simulation methods is prohibitive, namely to solve lower-scales in coupled hierarchical multi-scale simulations (e.g., `B.P. Ferreira (2022) <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_) and to generate large material response databases for data-driven frameworks based on machine learning (e.g., `Bessa et al. (2017) <https://www.sciencedirect.com/science/article/pii/S0045782516314803>`_). Clustering-based reduced-order models achieve a **striking balance between accuracy and computational cost** by first performing a clustering-based domain decomposition of the material model and then solving the equilibrium problem formulated over the resulting reduced model.

In the particular case of a **research environment**, `cratepy <https://pypi.org/project/cratepy/>`_ is designed to easily accommodate further developments, either by improving the already implemented methods or by including new numerical models and techniques. It also provides all the fundamental means to perform comparisons with alternative methods, both in terms of accuracy and computational cost. In a **teaching environment**, `cratepy <https://pypi.org/project/cratepy/>`_ is a readily available tool for demonstrative purposes and/or academic work proposals in solid mechanics and material-related courses.

----

Authorship & Citation
---------------------
CRATE was originally developed by Bernardo P. Ferreira [#]_ in the context of his PhD Thesis [#]_ .

If you use CRATE in a scientific publication, it is appreciated that you cite this PhD Thesis:

.. code-block:: python

    @phdthesis{ferreira:2022a,
      title = {Towards Data-driven Multi-scale Optimization of Thermoplastic Blends: Microstructural Generation, Constitutive Development and Clustering-based Reduced-Order Modeling},
      author = {Ferreira, B.P.},
      year = {2022},
      langid = {english},
      school = {University of Porto},
      url={https://hdl.handle.net/10216/146900}
    }

.. [#] `LinkedIn <https://www.linkedin.com/in/bpferreira/>`_ , `ORCID <https://orcid.org/0000-0001-5956-3877>`_, `ResearchGate <https://www.researchgate.net/profile/Bernardo-Ferreira-11?ev=hdr_xprf>`_

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_)

----

Getting started
---------------
To get a quick idea of what CRATE is all about, take a look into :doc:`GETTING STARTED <../rst_doc_files/getting_started/overview>` and go through :doc:`Overview <../rst_doc_files/getting_started/overview>` > :doc:`Installation <../rst_doc_files/getting_started/installation>` > :doc:`Run a benchmark <../rst_doc_files/getting_started/run_benchmark>`!


----

Useful Links
------------

* `GitHub repository <https://github.com/bessagroup/CRATE>`_ (source code);

* `PyPI <https://pypi.org/project/cratepy/>`_ (distribution package).

----

Community Support
-----------------

If you find any **issues**, **bugs** or **problems** with CRATE, please use the `GitHub issue tracker <https://github.com/bessagroup/CRATE/issues>`_ to report them. Provide a clear description of the problem, as well as a complete report on the underlying details, so that it can be easily reproduced and (hopefully) fixed!

You are also welcome to post any **questions**, **comments** or **suggestions** for improvement in the `GitHub discussions <https://github.com/bessagroup/CRATE/discussions>`_ space!

.. note::
    Bear in mind that CRATE is a program developed in an academic environment and that I'm currently the only developer as a side project. This means that I'll do my best to address all the issues, questions and suggestions, but do expect a reasonable time frame! ~ Bernardo P. Ferreira

----

Credits
-------
* Bernardo P. Ferreira is deeply thankful to `Francisco Pires <https://sigarra.up.pt/feup/pt/func_geral.formview?p_codigo=240385>`_ and `Miguel Bessa <https://github.com/mabessa>`_ for supervising the PhD Thesis that motivated the development of CRATE.

* Bernardo P. Ferreira acknowledges the pioneering development of clustering-based reduced-order models by `Zeliang Liu <https://zeliangliu.com/>`_ , namely by proposing the `Self-Consistent Clustering Analysis (SCA) <https://www.sciencedirect.com/science/article/pii/S0045782516301499>`_ , that established the conceptual framework and foundations for the development of CRATE.
----

License
-------
Copyright 2020, Bernardo Ferreira

All rights reserved.

CRATE is a free and open-source software published under a :doc:`BSD 3-Clause License <../license>`.



.. toctree::
   :name: gettingstartedtoc
   :caption: Getting started
   :maxdepth: 3
   :hidden:
   :includehidden:

   rst_doc_files/getting_started/overview.rst
   rst_doc_files/getting_started/installation.rst
   rst_doc_files/getting_started/run_benchmark.rst

.. toctree::
   :name: basicusagetoc
   :caption: Basic usage
   :maxdepth: 3
   :hidden:
   :includehidden:

   rst_doc_files/basic_usage/general_workflow.rst
   rst_doc_files/basic_usage/step1_material_model.rst
   rst_doc_files/basic_usage/step2_input_data.rst
   rst_doc_files/basic_usage/step3_simulation.rst
   rst_doc_files/basic_usage/step4_post_processing.rst
   rst_doc_files/basic_usage/available_features.rst

.. toctree::
   :name: validationtoc
   :caption: Validation
   :maxdepth: 3
   :hidden:
   :includehidden:

   rst_doc_files/validation/benchmarks.rst

.. toctree::
   :name: advancedusagetoc
   :caption: Advanced usage
   :maxdepth: 3
   :hidden:
   :includehidden:

   rst_doc_files/advanced_usage/customization.rst
   rst_doc_files/advanced_usage/interface_dns_solver.rst
   rst_doc_files/advanced_usage/interface_clustering_feature.rst
   rst_doc_files/advanced_usage/interface_clustering_algorithm.rst
   rst_doc_files/advanced_usage/interface_constitutive_model.rst

.. toctree::
   :name: apitoc
   :caption: API
   :hidden:

   rst_doc_files/reference/index.rst
   Code <_autosummary/cratepy>

.. toctree::
   :name: licensetoc
   :caption: License
   :hidden:

   license.rst
