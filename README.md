

<p align="center">
  <a href=""><img alt="logo" src="doc/media/CRATE_logo_horizontal_long.png" width="80%"></a>
</p>

# Overview

### Summary
CRATE is a numerical tool developed in the context of computational mechanics to aid the design and development of advanced materials. Through a suitable coupling between first-order computational homogenization and a clustering-based reduced order approach, CRATE aims to perform **accurate** and **efficient** multi-scale nonlinear analyses of heterogeneous materials.

### Authors
This program initial version was documented and fully coded by Bernardo P. Ferreira<sup>[1](#f1)</sup> ([bpferreira@fe.up.pt](mailto:bpferreira@fe.up.pt)) and developed in colaboration with Miguel A. Bessa<sup>[2](#f2)</sup> ([m.a.bessa@tudelft.nl](mailto:m.a.bessa@tudelft.nl)) and Francisco M. Andrade Pires<sup>[3](#f3)</sup> ([fpires@fe.up.pt](mailto:fpires@fe.up.pt)).

<sup id="f1"> 1 </sup> Member of CM2S and Bessa research groups, Department of Mechanical Engineering, Faculty of Engineering, University of Porto  
<sup id="f2"> 2 </sup> Leader of Bessa research group, Faculty of Mechanical, Maritime and Materials Engineering, Delft University of Technology  
<sup id="f3"> 3 </sup> Leader of CM2S research group, Department of Mechanical Engineering, Faculty of Engineering, University of Porto

### Description
CRATE has been designed with the main purpose of performing accurate and efficient multi-scale analyses of nonlinear heterogeneous materials, a crucial task in the development of new materials with innovative and enhanced properties. This goal is achieved through the coupling between **first-order computational homogenization** and a **clustering-based reduced order modeling** approach, allowing the efficient solution of a given microscale equilibrium problem that essentially involves four main steps:
1. Definition of a **representative volume element (RVE)** of the heterogeneous material under analysis;
2. Enforcement of first-order macroscale strain and/or stress loading constraints;
3. Solution of the microscale equilibrium problem with suitable boundary conditions;
4. Computation of the heterogeneous material first-order homogenized response through computational homogenization.

The clustering-based reduced order modeling approach comes into play by compressing the RVE into a **cluster-reduced representative volume element (CRVE)**, aiming to reduce the overall computational cost of the analysis by reducing the problem dimension. Of course that there is no free lunch but rather a **tradeoff between accuracy and computational cost** - in general, a greater (lower) degree of model compression, associated to a greater (lower) computational cost, leads to more (less) accurate results.

Although nothing prevents the use of CRATE as a standalone program to analyse a given material's behavior, it is in applications where the overall computational cost emerges as a bottleneck (both in terms of computational time and/or memory footprint) that its utility really shines. Among such applications, one can mention for instance: heavy parametric material studies, where the computational cost stems from a large number of microscale analyses under different parameters ranges (microstructure topology, material properties, loading conditions, ...); first-order hierarchical coupled multi-scale schemes (so-called FE<sup>2</sup>), where a high computational cost arises from the nested solution of equilibrium problems at different scales; and the more recent data-driven material design frameworks, usually requiring large material response databases to train the underlying machine learning models. 

> **Note:** The heterogeneous material representative volume element (RVE) is part of the input data that must be provided to CRATE,
i.e., CRATE does **not** perform the computational generation of microstructures. 

### Computational framework
CRATE is designed and implemented in Python (Python 3 release), making it easily portable between all major computer platforms, easily integrated with
other softwares implemented in different programming languages and benefiting from an extensive collection of prebuilt (standard library) and third-party libraries. Given the extensive numerical nature of the program, its implementation relies heavily on the well-known [NumPy](https://numpy.org/devdocs/index.html) and [SciPy](https://www.scipy.org/) scientific computing packages, being most numerical tasks dispatched to compiled C code inside the Python interpreter.


# Main features

### General formulation:
* Quasi-static loading conditions;
* Monotonic loading paths;
* Infinitesimal strains;
* Nonlinear material constitutive behavior (elasticity and plasticity).

### Material constitutive modeling:
* CRATE's embedded material model database includes:  
  * Isotropic linear elastic model;
  * Von Mises elastoplastic model with isotropic strain hardening.
* Interface with the constitutive model database of LINKS<sup>[5](#f5)</sup>, accounting for several infinitesimal and finite strain hyperelastic, elastoplastic and elastoviscoplastic constitutive models.

### Methods:
* Self-Consistent Clustering Analysis (SCA) clustering-based reduced order model<sup>[6](#f6)</sup> proposed by Zeliang Liu and coworkers;
* FFT-based homogenization basic scheme<sup>[7](#f7)</sup> proposed by H. Moulinec and P. Suquet;
* FEM-based homogenization through suitable interface with LINKS<sup>[5](#f5)</sup> (includes FE mesh generation and post-processing);
* Clustering algorithms imported from [scikit-learn](https://scikit-learn.org/stable/index.html) python machine learning package.

### Post-processing:
* VTK (XML format) output files allowing the visualization of data associated to the material microstructure (material phases, material clusters, ...) and response local fields (strain, stress, internal variables, ...);

***
<sup id="f5"> 5 </sup> LINKS (Large Strain Implicit Non-linear Analysis of Solids Linking Scales) is a multi-scale finite element code developed by CM2S research group at Faculty of Engineering of University of Porto.  
<sup id="f6"> 6 </sup> Liu, Z., Bessa, M., and Liu, W. K. (2016a). Self-consistent clustering analysis: An efficient multi-
scale scheme for inelastic heterogeneous materials. Computer Methods in Applied Mechanics
and Engineering, 306:319–341.  
<sup id="f7"> 7 </sup> Moulinec, H. and Suquet, P. (1994). A fast numerical method for computing the linear and
nonlinear mechanical properties of composites. A fast numerical method for computing the
linear and nonlinear mechanical properties of composites, 318(11):1417–1423.

# Quick guide

### Requirements
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

> **Note:** When trying to run CRATE for the first couple times, it is expected that Python's ImportError and ModuleNotFoundError are raised depending on the required packages that are not installed. Install them in turn and rerun CRATE until these exceptions are no longer raised, meaning that all required packages are properly installed and accessed.

### CRATE workflow
In what follows, the general workflow of CRATE in the solution of a microscale equilibrium problem is described in a step-by-step manner:  
1. **Generate microstructure.** The first step consists in the computational generation of a representative volume element (RVE) of the heterogeneous material under analysis, being mandatory that the RVE is quadrilateral (2D) or paralelepipedic (3D). The generation of the RVE can be made with any external software suitable for this purpose, **but** the discretization file that is ultimately supplied to CRATE **must** be generated within a Python script:
  
    * The generated microstructure must be discretized in a regular grid of pixels (2D) or voxels (3D), something that can be conveniently done by storing it in a [NumPy](https://numpy.org/devdocs/index.html) 2darray or 3darray named, for instance, `regular_grid`. Each entry of this integer arrays contains the label (id) associated to a given material phase (properly defined in the input data file);  
    * The discretization file (which **must** must have extension `.rgmsh`) should then be generated by saving the `regular_grid` array in binary format through the [NumPy](https://numpy.org/devdocs/index.html) function [save](https://numpy.org/doc/stable/reference/generated/numpy.save.html);
    <br/><br/>
    
    ```python
    # Import NumPy module
    import numpy as np

    # Generate the microstructure and store it in a numpy ndarray
    regular_grid = ...

    # Generate the discretization file in binary format
    # (this will append the .npy extension to the filename)
    np.save('filename.rgmsh', regular_grid)
    ```
2. **Write input data file.** After generating the microstructure RVE discretization file, the next step consists in writing the actual CRATE's input data file. This file contains all the required information about the problem itself (problem type, material properties, loading conditions, discretization file, ...) and about the solution procedure (load incrementation, clustering domain decomposition, convergence tolerances, ...). A complete CRATE input data file where each parameter specification (either mandatory or optional) is fully documented (meaning, syntax, available options) can be found in the `doc/` directory (or [here](https://github.com/BernardoFerreira/CRATE/blob/master/doc/CRATE_input_data_file.dat)). This file can be copied to a given directory and be readily used by replacing the `[insert here]` boxes with the suitable specification.

3. **Run CRATE.** In order to run CRATE, one must simply execute the main file (`crate.py`) with Python 3.X and provide the input data file (argument parsing).
    > In Linux/UNIX operative systems, open a terminal console window and execute the following command:  
    `python3.X crate.py input_data_file.dat`
     <br/><br/>
    The program execution can be followed in the terminal console window, where the data associated to the program launch, to the progress of the main execution phases and to the program end is output. 
  
4. **Get results.** As soon as CRATE is executed over a given input data file (lets say, `input_data_file.dat`), a folder with the same name is created in the same directory (`input_data_file/`). This folder contains all the output data related to the problem solution, namely: a log file (`input_data_file.screen`), where all data printed to the default standard ouput is stored; a homogenized results file (`input_data_file.hres`), where the homogenized results are stored; and one or more VTK output files (`.vti`) that can be read with a suitable software (e.g. [ParaView](https://www.paraview.org/)) to visualize and analyse the problem data.
