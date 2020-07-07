

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
in colaboration with Dr. Miguel A. Bessa<sup>[2](#f2)</sup> and Dr. Francisco M. Andrade Pires<sup>[3](#f3)</sup>.

<sup id="f1"> 1 </sup> bpferreira@fe.up.pt, CM2S and Bessa research groups, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2"> 2 </sup> m.a.bessa@tudelft.nl, Bessa research group, Faculty of Mechanical, Maritime and Materials Engineering, Delft University of Technology  
<sup id="f3"> 3 </sup> fpires@fe.up.pt, CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

#### Description
CRATE has been designed with the main purpose of performing accurate and efficient
multi-scale analyses of nonlinear heterogeneous materials, a crucial task in the development of
new materials with innovative and enhanced properties. This goal is achieved through the
coupling between first-order computational homogenization and a clustering-based reduced
order modeling approach, allowing the efficient solution of a given microscale equilibrium problem
generally formulated as:
1. Define the representative volume element (RVE) of the heterogeneous material under analysis;
2. Enforce first-order macroscale strain and/or stress loading constraints;
3. Solve the microscale equilibrium problem with suitable boundary conditions;
4. Compute the heterogeneous material first-order homogenized response through computational homogenization.

The clustering-based reduced order modeling approach comes into play by compressing the RVE into a **cluster-reduced
representative volume element (CRVE)**, aiming to reduce the overall computational cost of
the analysis at the expense of an acceptable decrease of accuracy.

Besides its direct application in the analysis of a given material's effective response, CRATE can be, for instance, properly embedded
in a first-order hierarchical coupled multi-scale scheme<sup>[4](#f4)</sup> or in a data-driven material design framework<sup>[5](#f5)</sup>.

> **Note:** The heterogeneous material representative volume element (RVE) is part of the input data that must be provided to CRATE. 

***

<sup id="f4"> 4 </sup> Feyel, F. (1998). Application Du Calcul Parallèle Aux Modèles à Grand Nombre de Variables Internes. PhD Thesis, École Nationale Supérieure des Mines de Paris.  
<sup id="f5"> 5 </sup> e.g. Bessa, M., Bostanabad, R., Liu, Z., Hu, A., Apley, D. W., Brinson, C., Chen, W., and Liu, W. K.
(2017). A framework for data-driven analysis of materials under uncertainty: Countering the
curse of dimensionality. Computer Methods in Applied Mechanics and Engineering, 320:633–
667.

#### Computational framework





## Main features

## Usage guide
