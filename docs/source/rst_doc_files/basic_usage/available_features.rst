
Available features
==================

Below is a summary of the **main features** and **current limitations** of CRATE in the computational multi-scale simulation of heterogeneous materials.

**General formulation:**

* Quasi-static deformation processes;
* Infinitesimal and finite strains;
* Implicit time integration.

.. note ::

   A **limitation under finite strains** is under investigation, namely the development of a suitable self-consistent scheme for the clustering-based reduced-order models SCA and ASCA. Therefore, enforcing **constant reference material properties** is currently the only option available to simulate with the previous models under finite strains.

----

**Macro-scale loading path:**

* General monotonic and non-monotonic macro-scale loading paths;
* Enforcement of macro-scale strain and/or stress constraints;
* General prescription of macro-scale loading incrementation;
* Dynamic macro-scale loading subincrementation.

----

**Material constitutive modeling:**

* General nonlinear material constitutive behavior;
* Interface to implement a new constitutive model;
* Out-of-the-box constitutive models include:

  - General anisotropic linear elastic constitutive model (infinitesimal strains);
  - von Mises elasto-plastic constitutive model with isotropic strain hardening (infinitesimal and finite strains);
  - General anisotropic Hencky hyperelastic constitutive model (finite strains);
  - General anisotropic St.Venant-Kirchhoff hyperelastic constitutive model (finite strains).

.. note ::
   Besides the constitutive models themselves, CRATE also makes available a complete and validated **set of computational solid mechanics common procedures** as well as a **toolkit of tensorial and matricial operations**!

----

**Offline-stage DNS methods:**

* Interface to implement a new direct numerical simulation (DNS) multi-scale method;
* FFT-based homogenization basic scheme (`article 1 <https://www.sciencedirect.com/science/article/pii/S0045782597002181>`_, `article 2 <https://link.springer.com/article/10.1007/s00466-014-1071-8>`_).

.. note::

   Despite a highly efficient implementation of the FFT-based homogenization basic scheme, the convergence of this method is **limited to moderate stiffness ratios** between different material phases. Variants of this method or different methods should be implemented to handle some cases of engineering interest (e.g., microstructures with voids or rigid inclusions).

----

**Offline-stage clustering methods:**

* Interface to implement a new clustering algorithm;
* Wrappers over clustering algorithms available from third-party libraries (e.g., `SciPy <https://docs.scipy.org/doc/scipy/reference/cluster.html>`_, `Scikit-Learn <https://scikit-learn.org/stable/modules/clustering.html>`_).

----

**Online-stage clustering-based reduced-order models:**

* Self-Consistent Clustering Analysis (SCA) (`article <https://www.sciencedirect.com/science/article/pii/S0045782516301499>`_);
* Adaptive Self-Consistent Clustering Analysis (ASCA) (`article <https://www.sciencedirect.com/science/article/pii/S0045782522000895>`_).
