---
title: 'CRATE: A Python package to perform fast material simulations'
tags:
  - Python
  - computational mechanics
  - material simulation
  - homogenization
  - clustering
authors:
  - name: Bernardo P. Ferreira
    orcid: 0000-0001-5956-3877
    affiliation: "1, 2"
  - name: F. M. Andrade Pires
    orcid: 0000-0002-4802-6360
    affiliation: 2
  - name: Miguel A. Bessa
    orcid: 0000-0002-6216-0355
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Engineering, Brown University, United States of America
   index: 1
 - name: Faculty of Engineering, University of Porto, Portugal
   index: 2
date: 15 April 2023
bibliography: references.bib

---

# Summary

[CRATE](https://github.com/bessagroup/CRATE) (Clustering-based Nonlinear Analysis of Materials) is a Python project (package [`cratepy`](https://pypi.org/project/cratepy/)) developed in the context of computational mechanics [@ferreira:2022a] to aid the design and development of new materials. Its main purpose is performing multi-scale nonlinear analyses of heterogeneous materials through a suitable coupling between first-order computational homogenization and clustering-based reduced-order modeling. This means that, given a representative volume element of the material micro-structure and the corresponding material phase properties, [`cratepy`](https://pypi.org/project/cratepy/) computes the material's effective mechanical response under a given loading by leveraging a so-called clustering-based reduced-order model (CROM).

![Logo of CRATE ([`cratepy`](https://pypi.org/project/cratepy/)). \label{fig:crate_logo_horizontal_long}](crate_logo_horizontal_long.png)

# Statement of need

CRATE (Clustering-based Nonlinear Analysis of Materials) is a Python project (package [`cratepy`](https://pypi.org/project/cratepy/)) developed in the field of computational mechanics and material science. To the best of the authors' knowledge, it is a first-of-its-kind open-source software that allows any material development enthusiast to perform multi-scale analyses of materials by taking advantage of the recent family of clustering-based reduced-order models (CROMs). \autoref{fig:crate_conceptual_scheme} provides a simple illustration of a CRATE simulation. It is worth remarking that CRATE is supported by a rich documentation that provides a conceptual overview of the project, clear installation instructions, a step-by-step basic workflow description, and detailed guidelines for advanced customized developments. Moreover, [`cratepy`](https://pypi.org/project/cratepy/) relies solely on a few well-established third-party Python scientific computing packages, such as [`numpy`](https://pypi.org/project/numpy/) [@harris2020array] and [`scipy`](https://pypi.org/project/scipy/) [@2020SciPy-NMeth], [`cratepy`](https://pypi.org/project/cratepy/)'s modules are extensively documented, and the automatically generated API provides a complete and updated description of the underlying object-oriented implementation, including LaTeX rendered formulae to improve comprehension.

`cratepy` is essentially a numerical tool for any application that requires material multi-scale simulations. Given the intrinsic clustering-based reduced-order modeling approach (e.g., SCA [@liu:2016], ASCA [@ferreira:2022b]), CRATE is mostly useful in applications where the computational cost of standard simulation methods is prohibitive, namely to solve lower-scales in coupled hierarchical multi-scale simulations (e.g., @ferreira:2022a) and to generate large material response databases for data-driven frameworks based on machine learning (e.g., @bessa:2017). CROMs achieve a striking balance between accuracy and computational cost by first performing a clustering-based domain decomposition of the material model and then solving the equilibrium problem formulated over the resulting reduced model. In a similar scope, it is worth mentioning the projects [`fibergen`](https://github.com/fospald/fibergen) (@Ospald2019) and [`FFTHomPy`](https://github.com/vondrejc/FFTHomPy) that, although not relying on a clustering-based reduced-order modeling approach, implement homogenization methods to extract effective material parameters.


In the particular case of a research environment, [`cratepy`](https://pypi.org/project/cratepy/) is designed to easily accommodate further developments, either by improving the already implemented methods or by including new numerical models and techniques. It also provides all the fundamental means to perform comparisons with alternative methods, both in terms of accuracy and computational cost. In a teaching environment, [`cratepy`](https://pypi.org/project/cratepy/) is a readily available tool for demonstrative purposes and/or academic work proposals in solid mechanics and material-related courses.

We hope that CRATE contributes effectively to the development of new materials and encourages other researchers to share their own projects.

![Schematic illustration of CRATE ([`cratepy`](https://pypi.org/project/cratepy/)) simulation.\label{fig:crate_conceptual_scheme}](crate_conceptual_scheme.png)


# Acknowledgements

Bernardo P. Ferreira acknowledges the support provided by Fundação para a Ciência e a Tecnologia (FCT, Portugal) through the scholarship with reference [SFRH/BD/130593/2017](https://www.sciencedirect.com/science/article/pii/S0045782522000895?via%3Dihub#GS1). This research has also been supported by Instituto de Ciência e Inovação em Engenharia Mecânica e Engenharia Industrial (INEGI, Portugal). Miguel A. Bessa acknowledges the support from the project ‘Artificial intelligence towards a sustainable future: ecodesign of recycled polymers and composites’ (with project number 17260 of the research programme Applied and Engineering Sciences) which is financed by the Dutch Research Council (NWO), The Netherlands.

# References
