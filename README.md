

<p align="center">
  <a href=""><img alt="logo" src="https://user-images.githubusercontent.com/25851824/254047444-74e630b0-85fb-4746-bdf6-df504fda5912.png" width="80%"></a>
</p>

# What is CRATE?

[**Docs**](https://bessagroup.github.io/CRATE/)
| [**Installation**](https://bessagroup.github.io/CRATE/rst_doc_files/getting_started/installation.html)
| [**GitHub**](https://github.com/bessagroup/CRATE)
| [**PyPI**](https://pypi.org/project/cratepy/)

### Summary
**CRATE** (Clustering-based Nonlinear Analysis of Materials) is a Python project (package [cratepy](https://pypi.org/project/cratepy/)) developed in the context of computational mechanics to aid the design and development of new materials. Its main purpose is **performing multi-scale nonlinear analyses of heterogeneous materials** through a suitable coupling between first-order computational homogenization and clustering-based reduced-order modeling: given a representative volume element of the material microstructure and the corresponding material phase properties, CRATE computes the material's effective mechanical response when subject to a prescribed macro-scale loading path.

### Statement of need
[cratepy](https://pypi.org/project/cratepy/) is essentially a numerical tool for any application that requires material multi-scale simulations. Given the intrinsic clustering-based reduced-order modeling approach (e.g., [SCA](https://www.sciencedirect.com/science/article/pii/S0045782516301499), [ASCA](https://www.sciencedirect.com/science/article/pii/S0045782522000895?via%3Dihub)), CRATE is mostly useful in applications where the computational cost of standard simulation methods is prohibitive, namely to solve lower-scales in coupled hierarchical multi-scale simulations (e.g., [B.P. Ferreira (2022)](http://dx.doi.org/10.13140/RG.2.2.33940.17289)) and to generate large material response databases for data-driven frameworks based on machine learning (e.g., [Bessa et al. (2017)](https://www.sciencedirect.com/science/article/pii/S0045782516314803)). Clustering-based reduced-order models achieve a **striking balance between accuracy and computational cost** by first performing a clustering-based domain decomposition of the material model and then solving the equilibrium problem formulated over the resulting reduced model.

In the particular case of a **research environment**, [cratepy](https://pypi.org/project/cratepy/) is designed to easily accommodate further developments, either by improving the already implemented methods or by including new numerical models and techniques. It also provides all the fundamental means to perform comparisons with alternative methods, both in terms of accuracy and computational cost. In a **teaching environment**, [cratepy](https://pypi.org/project/cratepy/) is a readily available tool for demonstrative purposes and/or academic work proposals in solid mechanics and material-related courses.

Consider leaving a star if you think CRATE is useful for the research community!

### Authorship & Citation
CRATE was originally developed by Bernardo P. Ferreira<sup>[1](#f1)</sup> in the context of his PhD Thesis<sup>[2](#f2)</sup>.

<sup id="f1"> 1 </sup> Profile: [LinkedIN](https://www.linkedin.com/in/bpferreira/), [ORCID](https://orcid.org/0000-0001-5956-3877), [ResearchGate](https://www.researchgate.net/profile/Bernardo-Ferreira-11?ev=hdr_xprf)

<sup id="f2"> 2 </sup> Ferreira, B.P. (2022). *Towards Data-driven Multi-scale Optimization of Thermoplastic Blends: Microstructural Generation, Constitutive Development and Clustering-based Reduced-Order Modeling.* PhD Thesis, University of Porto (see [here](http://dx.doi.org/10.13140/RG.2.2.33940.17289))

If you use CRATE in your research or in a scientific publication, it is appreciated that you cite the two papers below.

**Journal of Open Source Software** ([paper](https://doi.org/10.21105/joss.05594)):
```
@article{Ferreira2023,
  title = {CRATE: A Python package to perform fast material simulations},
  author = {Bernardo P. Ferreira and F. M. Andrade Pires and Miguel A. Bessa}
  doi = {10.21105/joss.05594},
  url = {https://doi.org/10.21105/joss.05594},
  year = {2023},
  publisher = {The Open Journal},
  volume = {8},
  number = {87},
  pages = {5594},
  journal = {Journal of Open Source Software}
}
```

**Computer Methods in Applied Mechanics and Engineering** ([paper](http://dx.doi.org/10.1016/j.cma.2022.114726)):
```
@article{Ferreira2022,
  title = {Adaptivity for clustering-based reduced-order modeling of localized history-dependent phenomena},
  author = {Ferreira, B.P., and Andrade Pires, F.M., and Bessa, M.A.},
  doi = {10.1016/j.cma.2022.114726},
  url = {https://www.sciencedirect.com/science/article/pii/S0045782522000895},
  year = {2022},
  volume = {393},
  pages = {114726},
  issn = {0045-7825},
  journal = {Computer Methods in Applied Mechanics and Engineering},
}
```

----

# Getting started

You can find everything you need to know in [CRATE documentation](https://bessagroup.github.io/CRATE/)!

<p align="center">
  <a href=""><img alt="logo" src="https://user-images.githubusercontent.com/25851824/238440445-48811f19-8131-4161-8eeb-108197221986.png" width="80%"></a>
</p>


# Community Support

If you find any **issues**, **bugs** or **problems** with CRATE, please use the [GitHub issue tracker](https://github.com/bessagroup/CRATE/issues) to report them. Provide a clear description of the problem, as well as a complete report on the underlying details, so that it can be easily reproduced and (hopefully) fixed!

You are also welcome to post there any **questions**, **comments** or **suggestions** for improvement in the [GitHub discussions](https://github.com/bessagroup/CRATE/discussions) space!

Please refer to CRATE's [Code of Conduct](https://github.com/bessagroup/CRATE/blob/master/CODE_OF_CONDUCT.md).

>**Note:**  
>Bear in mind that CRATE is a program developed in an academic environment and that I'm currently the only developer as a side project. This means that I'll do my best to address all the issues, questions and suggestions, but do expect a reasonable time frame! ~ *Bernardo P. Ferreira*


# Credits

* Bernardo P. Ferreira is deeply thankful to [Francisco Pires](https://sigarra.up.pt/feup/pt/func_geral.formview?p_codigo=240385) and [Miguel Bessa](https://github.com/mabessa) for supervising the PhD Thesis that motivated the development of CRATE.

* Bernardo P. Ferreira acknowledges the pioneering development of clustering-based reduced-order models by [Zeliang Liu](https://zeliangliu.com/), namely by proposing the [Self-Consistent Clustering Analysis (SCA)](https://www.sciencedirect.com/science/article/pii/S0045782516301499), that established the conceptual framework and foundations for the development of CRATE.


# License

Copyright 2020, Bernardo Ferreira

All rights reserved.

CRATE is a free and open-source software published under a BSD 3-Clause License.
