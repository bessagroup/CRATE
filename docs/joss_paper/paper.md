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
 - name: School of Engineering, Brown University, USA
   index: 1
 - name: Faculty of Engineering, University of Porto, Portugal
   index: 2
date: 15 April 2023
bibliography: references.bib

---

# Summary

`crate`  is a Python package developed in the context of computational mechanics [@ferreira:2022a] to aid the design and development of new materials. Its main purpose is performing multi-scale nonlinear analyses of heterogeneous materials through a suitable coupling between first-order computational homogenization and clustering-based reduced-order modeling. This means that, given a representative volume element of the material micro-structure and the corresponding material phase properties, `crate` computes the material's effective mechanical response when subject to prescribed loading conditions.

![Logo of `crate`. \label{fig:crate_logo_horizontal_long}](crate_logo_horizontal_long.png)

# Statement of need

`crate` is a Python package developed in the field of computational mechanics and material science. To the best of the authors' knowledge, it is a first-of-its-kind open-source software that allows any material development enthusiast to perform multi-scale analyses of materials seamlessly. \autoref{fig:crate_conceptual_scheme} provides a simple illustration of a `crate` simulation. It is also worth remarking that `crate` relies solely on a few well-established third-party Python scientific computing packages, such as `numpy` [@harris2020array] and `scipy` [@2020SciPy-NMeth], and includes several interfaces that allow a straightforward integration with other Python packages (e.g., `sklearn.cluster` clustering algorithms from `scikit-learn` [@scikit-learn]). Moreover, `crate`'s modules are extensively documented and the automatically generated API provides a complete and updated description of the underlying object-oriented implementation, including LaTeX rendered formulae to improve comprehension.

`crate` is essentially a numerical tool for any application that requires material multi-scale simulations. Given the intrinsic clustering-based reduced-order modeling approach (e.g., @liu:2016, @ferreira:2022b), `crate` is mostly useful in applications where the computational cost of standard simulation methods is prohibitive, namely to solve lower-scales in coupled hierarchical multi-scale simulations (e.g., @ferreira:2022a) and to generate large material response databases for data-driven frameworks based on machine learning (e.g.,  @bessa:2017). In the particular case of a research environment,  `crate` is designed to easily accomodate further developments, either by improving the already implemented methods or by including new numerical models and techniques. It also provides all the fundamental means to perform comparisons with alternative methods, both in terms of accuracy and computational cost. In a teaching environment, `crate` is a readily available tool for demonstrative purposes and/or academic work proposals in solid mechanics and material-related courses.

We hope that `crate` contributes efffectively to the development of new materials and encourages other researchers to share their own projects.

![Schematic illustration of `crate` simulation.\label{fig:crate_conceptual_scheme}](crate_conceptual_scheme.png)


# Acknowledgements

Bernardo P. Ferreira acknowledges the support provided by Fundação para a Ciência e a Tecnologia (FCT, Portugal) through the scholarship with reference [SFRH/BD/130593/2017](https://www.sciencedirect.com/science/article/pii/S0045782522000895?via%3Dihub#GS1). This research has also been supported by Instituto de Ciência e Inovação em Engenharia Mecânica e Engenharia Industrial (INEGI, Portugal).

# References
