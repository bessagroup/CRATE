

<p align="center">
  <a href=""><img alt="logo" src="doc/media/CRATE_logo_horizontal_long.png" width="70%"></a>
</p>

# CRATE

## Overview

#### Summary
CRATE is a numerical tool developed in the context of computational mechanics to aid the
design and development of advanced materials. Through a suitable coupling between first-order
computational homogenization and a clustering-based reduced order approach, CRATE is able
to perform **accurate** and **efficient** multi-scale nonlinear analyses of heterogeneous materials.

#### Authors
This program initial version was fully coded and documented by Bernardo P. Ferreira<sup>[1](#f1)</sup> and developed
in colaboration with Miguel A. Bessa<sup>[2](#f2)</sup> and Francisco M. Andrade Pires<sup>[3](#f3)</sup>.

<sup id="f1"> 1 </sup> [bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt), CM2S and Bessa research groups, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2"> 2 </sup> [m.a.bessa@tudelft.nl](mailto:m.a.bessa@tudelft.nl), Bessa research group, Faculty of Mechanical, Maritime and Materials Engineering, Delft University of Technology  
<sup id="f3"> 3 </sup> [fpires@fe.up.pt](mailto:fpires@fe.up.pt), CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

#### Description
CRATE has been designed with the main purpose of performing accurate and efficient
multi-scale analyses of nonlinear heterogeneous materials, a crucial task in the development of
new materials with innovative and enhanced properties. This goal is achieved through the
coupling between first-order computational homogenization and a clustering-based reduced
order modeling approach, allowing the efficient solution of a given microscale equilibrium problem
generally formulated as:
1. Definiton of a representative volume element (RVE) of the heterogeneous material under analysis;
2. Enforcement of first-order macroscale strain and/or stress loading constraints;
3. Solution of the microscale equilibrium problem with suitable boundary conditions;
4. Computation of the heterogeneous material first-order homogenized response through computational homogenization.

The clustering-based reduced order modeling approach comes into play by compressing the RVE into a **cluster-reduced
representative volume element (CRVE)**, aiming to reduce the overall computational cost of
the analysis at the expense of an acceptable decrease of accuracy.

Besides its direct application in the analysis of a given material's effective response, CRATE can be, for instance, properly embedded
in a first-order hierarchical coupled multi-scale scheme<sup>[4](#f4)</sup> or in a data-driven material design framework<sup>[5](#f5)</sup>.

> **Note:** The heterogeneous material representative volume element (RVE) is part of the input data that must be provided to CRATE (see here),
i.e., CRATE does **not** perform the computational generation of microstructures. 

#### Computational framework
CRATE is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with
other softwares implemented in different programming languages and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known NumPy and SciPy scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.


****

<sup id="f4"> 4 </sup> Feyel, F. (1998). Application Du Calcul Parallèle Aux Modèles à Grand Nombre de Variables Internes. PhD Thesis, École Nationale Supérieure des Mines de Paris.  
<sup id="f5"> 5 </sup> See e.g. Bessa, M., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., and Liu, W. K.
(2017). A framework for data-driven analysis of materials under uncertainty: Countering the
curse of dimensionality. Computer Methods in Applied Mechanics and Engineering, 320:633–
667.



## Main features

#### General formulation:
* Quasi-static loading conditions;
* Monotonic loading paths;
* Infinitesimal strains;
* Nonlinear material constitutive behavior (elasticity and plasticity);

#### Material constitutive modeling:
* CRATE's embedded material model database includes:  
  * Isotropic linear elastic model;
  * Von Mises elastoplastic model with isotropic strain hardening.
* Interface with the constitutive model database of LINKS<sup>[5](#f5)</sup>, accounting for several infinitesimal and finite strain hyperelastic, elastoplastic and elastoviscoplastic constitutive models;

#### Methods:
* Self-Consistent Clustering Analysis (SCA) clustering-based reduced order model<sup>[6](#f6)</sup> proposed by Zeliang Liu and coworkers;
* FFT-based homogenization basic scheme<sup>[7](#f7)</sup> proposed by H. Moulinec and P. Suquet;
* FEM-based homogenization through suitable interface with LINKS<sup>[5](#f5)</sup> (includes FE mesh generation and post-processing);
* Clustering algorithms imported from [scikit-learn](https://scikit-learn.org/stable/index.html) python machine learning package;

#### Post-processing:
* VTK (XML format) output files allowing the visualization of data associated to the material microstructure (material phases, material clusters, ...) and response local fields (strain, stress, internal variables, ...);

***
<sup id="f5"> 5 </sup> LINKS (Large Strain Implicit Non-linear Analysis of Solids Linking Scales) is a multi-scale finite element code developed by CM2S research group at Faculty of Engineering of University of Porto.  
<sup id="f6"> 6 </sup> Liu, Z., Bessa, M., and Liu, W. K. (2016a). Self-consistent clustering analysis: An efficient multi-
scale scheme for inelastic heterogeneous materials. Computer Methods in Applied Mechanics
and Engineering, 306:319–341.  
<sup id="f7"> 7 </sup> Moulinec, H. and Suquet, P. (1994). A fast numerical method for computing the linear and
nonlinear mechanical properties of composites. A fast numerical method for computing the
linear and nonlinear mechanical properties of composites, 318(11):1417–1423.

## Quick guide

#### Requirements
Some software must be installed in order to successfully run CRATE:
* Python 3.X (see [here](https://www.python.org/downloads/)) - Required to compile (byte code) and run (Python Virtual Machine) CRATE;

  > In Linux/UNIX operative systems, python can be simply installed from apt library by executing the following command:  
  `sudo apt install python3.X`  

* PyPi pip (see [here](https://pypi.org/project/pip/)) - Required to install Python 3 packages (learn [here](https://docs.python.org/3/installing/));

  > In Linux/UNIX operative systems, pip can be simply installed from apt library by executing the following command:  
  `sudo apt install python3-pip`
  
* ParaView (see [here](https://www.paraview.org/download/)) - Required to visualize the data contained in the VTK output files (learn [here](https://www.paraview.org/resources/));  

  > In Linux/UNIX operative systems, ParaView can be installed by placing the tarball in the installation directory and extracting it by executing the following command:  
  `sudo tar -xvf ParaView-< version >.tar.gz`

#### Run CRATE (general workflow)
