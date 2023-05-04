
Available features
******************

Below is a summary of the **main features** that CRATE has to offer regarding the computational simulation of materials.

General formulation:
~~~~~~~~~~~~~~~~~~~~
* Quasi-static deformation process;
* Infinitesimal and finite strains;
* Implicit time integration.

Macro-scale loading path:
~~~~~~~~~~~~~~~~~~~~~~~~~
* General monotonic and non-monotonic macro-scale loading paths;
* Enforcement of macro-scale strain and/or stress constraints;
* General prescription of macro-scale loading incrementation;
* Dynamic macro-scale loading subincrementation.

Material constitutive modeling:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* General nonlinear material constitutive behavior;
* Interface to implement a new constitutive model;
* Admits three different families of constitutive models:

  - Infinitesimal strains constitutive models;
  - Finite strains constitutive models;
  - Finite strains constitutive models whose implementation stems from a purely kinematical extension of their infinitesimal counterpart;
* Available computational solid mechanics common procedures;
* Suitable toolkit of tensorial and matricial operations;
* Out-of-the-box constitutive models include:

  - General anisotropic linear elastic constitutive model (infinitesimal strains);
  - von Mises elasto-plastic constitutive model with isotropic strain hardening (infinitesimal and finite strains);
  - General anisotropic Hencky hyperelastic constitutive model (finite strains);
  - General anisotropic St.Venant-Kirchhoff hyperelastic constitutive model (finite strains).

Offline-stage DNS methods:
~~~~~~~~~~~~~~~~~~~~~~~~~~
* Interface to implement any direct numerical simulation (DNS) homogenization-based multi-scale method;
* FFT-based homogenization basic scheme (`article 1 <https://www.sciencedirect.com/science/article/pii/S0045782597002181>`_, `article 2 <https://link.springer.com/article/10.1007/s00466-014-1071-8>`_).

Offline-stage clustering methods:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Interface to implement any clustering algorithm;
* Wrappers over clustering algorithms available from third-party libraries (e.g., `SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html>`_, `Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_, ...).

Online-stage clustering-based reduced-order models:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Self-Consistent Clustering Analysis (SCA) (`article <https://www.sciencedirect.com/science/article/pii/S0045782516301499>`_);
* Adaptive Self-Consistent Clustering Analysis (ASCA) (`article <https://www.sciencedirect.com/science/article/pii/S0045782522000895>`_).

Post-processing:
~~~~~~~~~~~~~~~~
* VTK (XML format) output files allowing the visualization of data associated to the material microstructure (topology, material phases, material clusters) and micro-scale physical fields (strain, stress, internal variables, ...).

   
