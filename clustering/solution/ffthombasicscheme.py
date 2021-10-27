#
# Moulinec and Suquet FFT-Based Homogenization Method Module (CRATE Program)
# ==========================================================================================
# Summary:
# Implementation of the FFT-based homogenization basic scheme proposed by H. Moulinec
# and P. Suquet ("A numerical method for computing the overall response of nonlinear
# composites with complex microstructure" Comp Methods Appl M 157 (1998):69-94) for the
# solution of microscale equilibrium problems of linear elastic heterogeneous materials.
# Finite strains extension following M. Kabel, T. Bohlke and M. Schneider ("Efficient fixed
# point and Newton-Krylov solver for FFT-based homogenization of elasticity at large
# deformations" Comput Mech 54 (2014):1497-1514) and adopting the Hencky hyperelastic
# constitutive model.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Jan 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Refactoring and OOP implementation.
# Bernardo P. Ferreira | Oct 2021 | Finite strains extension.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Parse command-line options and arguments
import sys
# Date and time
import time
# Shallow and deep copy operations
import copy
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Terminal colors
import colorama
# I/O utilities
import ioput.ioutilities as ioutil
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# Cluster interaction tensors operations
import clustering.citoperations as citop
# DNS homogenization-based multi-scale methods interface
from clustering.solution.dnshomogenization import DNSHomogenizationMethod
#
#                                                      FFT-based homogenization basic scheme
# ==========================================================================================
class FFTBasicScheme(DNSHomogenizationMethod):
    '''FFT-based homogenization basic scheme.

    FFT-based homogenization basic scheme proposed by H. Moulinec and P. Suquet
    ("A numerical method for computing the overall response of nonlinear composites with
    complex microstructure" Comp Methods Appl M 157 (1998):69-94). For a given RVE
    discretized in a regular grid of voxels, this method solves the microscale static
    equilibrium problem when the RVE is subjected to a macroscale strain and is constrained
    by periodic boundary conditions. Restricted to linear elastic (infinitesimal strains) or
    Hencky hyperelastic (finite strains) constitutive behavior.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _max_n_iterations : int
        Maximum number of iterations to convergence.
    _conv_criterion : str, {'stress_div', 'avg_stress', 'avg_strain'}
        Convergence criterion: 'stress_div' is the original convergence criterion based on
        the evaluation of the divergence of the stress tensor; 'avg_stress' is based on the
        evolution of the average stress norm; 'avg_strain' is based on the evolution of the
        average stress norm.
    _conv_tol : float
        Convergence tolerance.
    _max_subinc_level : int
        Maximum level of macroscale loading subincrementation.
    _max_cinc_cuts : int
        Maximum number of consecutive macroscale loading increment cuts.
    '''
    def __init__(self, strain_formulation, problem_type, rve_dims, n_voxels_dims,
                 regular_grid, material_phases, material_properties):
        '''FFT-based homogenization basic scheme constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        rve_dims : list
            RVE size in each dimension.
        n_voxels_dims : list
            Number of voxels in each dimension of the regular grid (spatial discretization
            of the RVE).
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        material_properties : dict
            Constitutive model material properties (key, str) values (item, int/float/bool).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_properties = material_properties
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Set maximum number of iterations
        self._max_n_iterations = 100
        # Set convergence criterion and tolerance
        self._conv_criterion = 'avg_stress'
        self._conv_tol = 1e-6
        # Set macroscale loading subincrementation parameters
        self._max_subinc_level = 5
        self._max_cinc_cuts = 5
    # --------------------------------------------------------------------------------------
    def compute_rve_local_response(self, mac_strain, verbose=False):
        '''Compute RVE local elastic strain response.

        Compute the local response of the material's representative volume element (RVE)
        subjected to a given macroscale strain loading: infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains). It is assumed that
        the RVE is spatially discretized in a regular grid of voxels.

        Parameters
        ----------
        mac_strain : 2darray
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str). Infinitesimal strain
            tensor (infinitesimal strains) or material logarithmic strain tensor (finite
            strains).
        '''
        # Set initial time
        init_time = time.time()
        # Display greetings
        type(self)._display_greetings()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store total macroscale strain tensor
        mac_strain_total = copy.deepcopy(mac_strain)
        # Initialize macroscale strain increment cut flag
        is_inc_cut = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        #
        #                                                 Material phases elasticity tensors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elastic properties-related optimized variables
        evar1 = np.zeros(tuple(self._n_voxels_dims))
        evar2 = np.zeros(tuple(self._n_voxels_dims))
        for mat_phase in material_phases:
            # Get material phase elastic properties
            E = material_properties[mat_phase]['E']
            v = material_properties[mat_phase]['v']
            # Build optimized variables
            evar1[regular_grid == int(mat_phase)] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
            evar2[regular_grid == int(mat_phase)] = np.multiply(2,E/(2.0*(1.0 + v)))
        evar3 = np.add(evar1, evar2)
        #
        #                                              Reference material elastic properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material elastic properties as the mean between the minimum and
        # maximum values existent among the microstructure's material phases
        # (proposed in Moulinec, H. and Suquet, P., 1998)
        mat_prop_ref = dict()
        mat_prop_ref['E'] = \
            0.5*(min([material_properties[phase]['E'] for phase in material_phases]) + \
                 max([material_properties[phase]['E'] for phase in material_phases]))
        mat_prop_ref['v'] = \
            0.5*(min([material_properties[phase]['v'] for phase in material_phases]) + \
                 max([material_properties[phase]['v'] for phase in material_phases]))
        #
        #                                                           Frequency discretization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set discrete frequencies (rad/m) for each dimension
        freqs_dims = list()
        for i in range(self._n_dim):
            # Set sampling spatial period
            sampling_period = self._rve_dims[i]/self._n_voxels_dims[i]
            # Set discrete frequencies
            freqs_dims.append(2*np.pi*np.fft.fftfreq(self._n_voxels_dims[i],
                                                     sampling_period))
        #
        #                                                  Reference material Green operator
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get reference material Young modulus and Poisson coeficient
        E_ref = mat_prop_ref['E']
        v_ref = mat_prop_ref['v']
        # Compute reference material Lamé parameters
        lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
        miu_ref = E_ref/(2.0*(1.0 + v_ref))
        # Compute Green operator reference material related constants
        if self._strain_formulation == 'infinitesimal':
            c1 = 1.0/(4.0*miu_ref)
            c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
        else:
            c1 = 1.0/(miu_ref)
            c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Green operator material independent terms
        gop_1_dft_vox, gop_2_dft_vox, _ = citop.gop_material_independent_terms(
            self._strain_formulation, self._n_dim, self._rve_dims, self._n_voxels_dims,
                self._comp_order_sym, self._comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Green operator matricial form components
        comps = list(it.product(comp_order, comp_order))
        # Set mapping between Green operator fourth-order tensor and matricial form
        # components
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in \
                [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Green operator
        gop_dft_vox = {''.join([str(x + 1) for x in idx]): \
            np.zeros(tuple(self._n_voxels_dims)) for idx in fo_indexes}
        # Compute Green operator matricial form components
        for i in range(len(mf_indexes)):
            # Get fourth-order tensor indexes
            fo_idx = fo_indexes[i]
            # Get Green operator component
            comp = ''.join([str(x+1) for x in fo_idx])
            # Compute Green operator matricial form component
            gop_dft_vox[comp] = c1*gop_1_dft_vox[comp] + c2*gop_2_dft_vox[comp]
        #
        #                                           Macroscale strain loading incrementation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize macroscale strain incrementer
        mac_strain_incrementer = MacroscaleStrainIncrementer(self._strain_formulation,
                                                             self._problem_type,
                                                             mac_strain_total,
                                                             inc_lfacts=[0.5, 0.5])
        # Set first macroscale strain increment
        mac_strain_incrementer.update_inc()
        # Get current macroscale strain tensor
        mac_strain = mac_strain_incrementer.get_current_mac_strain()
        # Display increment data
        if verbose:
            type(self)._display_increment_init(
                *mac_strain_incrementer.get_inc_output_data())
        # Set increment initial time
        inc_init_time = time.time()
        # Set iteration initial time
        iter_init_time = time.time()
        #
        #                                                Macroscale loading incremental loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Start macroscale loading incremental loop
        while True:
            #
            #                                                        Initial iterative guess
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set initial iterative guess
            if mac_strain_incrementer.get_inc() == 1:
                # Initialize strain tensor
                strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                              for comp in comp_order}
                # Set strain initial iterative guess
                for comp in comp_order:
                    # Get strain component indexes
                    so_idx = tuple([int(x) - 1 for x in comp])
                    # Initial guess: Macroscale strain tensor
                    strain_vox[comp] = np.full(self._regular_grid.shape, mac_strain[so_idx])
                # Initialize last converged strain tensor
                strain_old_vox = copy.deepcopy(strain_vox)
            else:
                # Initial guess: Last converged strain field
                strain_vox = copy.deepcopy(strain_old_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute stress initial iterative guess
            stress_vox = self._elastic_constitutive_model(strain_vox, evar1, evar2, evar3)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute average strain/stress norm
            if self._conv_criterion == 'avg_stress':
                # Compute initial guess average stress norm
                avg_stress_norm = self._compute_avg_state_vox(stress_vox)
                # Initialize last iteration average stress norm
                avg_stress_norm_old = 0.0
            elif self._conv_criterion == 'avg_strain':
                # Compute initial guess average strain norm
                avg_strain_norm = self._compute_avg_state_vox(strain_vox)
                # Initialize last iteration average strain norm
                avg_strain_norm_old = 0.0
            #
            #                                        Strain Discrete Fourier Transform (DFT)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute strain Discrete Fourier Transform (DFT)
            strain_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims), dtype=complex) \
                              for comp in comp_order}
            # Loop over strain components
            for comp in comp_order:
                # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
                strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize macroscale strain voxelwise tensor
            mac_strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                              for comp in comp_order}
            mac_strain_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims), dtype=complex)\
                                  for comp in comp_order}
            # Enforce macroscale strain DFT at the zero-frequency
            freq_0_idx = self._n_dim*(0,)
            mac_strain_DFT_0 = {}
            for comp in comp_order:
                # Get strain component indexes
                so_idx = tuple([int(x) - 1 for x in comp])
                # Compute macroscale strain DFT
                mac_strain_vox[comp] = np.full(self._regular_grid.shape, mac_strain[so_idx])
                mac_strain_DFT_vox[comp] = np.fft.fftn(mac_strain_vox[comp])
                # Enforce macroscale strain DFT at the zero-frequency
                mac_strain_DFT_0[comp] = mac_strain_DFT_vox[comp][freq_0_idx]
            #
            #                                                   Fixed-point iterative scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize iteration counter:
            iter = 0
            # Start iterative loop
            while True:
                #
                #                                    Stress Discrete Fourier Transform (DFT)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute stress Discrete Fourier Transform (DFT)
                stress_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims), dtype=complex)
                                  for comp in comp_order}
                for comp in comp_order:
                    # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform
                    # (FFT)
                    stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
                #
                #                                                     Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Evaluate convergence criterion
                if self._conv_criterion == 'stress_div':
                    # Compute discrete error
                    discrete_error = self._stress_div_conv_criterion(freqs_dims,
                                                                    stress_DFT_vox)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif self._conv_criterion == 'avg_stress':
                    # Compute discrete error
                    discrete_error = \
                        abs(avg_stress_norm - avg_stress_norm_old)/avg_stress_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif self._conv_criterion == 'avg_strain':
                    # Compute discrete error
                    discrete_error = \
                        abs(avg_strain_norm - avg_strain_norm_old)/avg_strain_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Display iteration data
                if verbose:
                    type(self)._display_iteration(iter, time.time() - iter_init_time,
                                                  discrete_error)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check solution convergence and iteration counter
                if discrete_error <= self._conv_tol:
                    # Leave fixed-point iterative scheme
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif iter == self._max_n_iterations:
                    # Raise macroscale increment cut procedure
                    is_inc_cut = True
                    # Display increment cut (maximum number of iterations)
                    type(self)._display_increment_cut(self._max_n_iterations)
                    # Leave fixed-point iterative scheme
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    # Increment iteration counter
                    iter += 1
                    # Set iteration initial time
                    iter_init_time = time.time()
                #
                #                                                              Update strain
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over strain components
                for i in range(len(comp_order)):
                    # Get strain component
                    comp_i = comp_order[i]
                    # Initialize auxiliar variable
                    aux = 0.0
                    # Loop over strain components
                    for j in range(len(comp_order)):
                        # Get strain component
                        comp_j = comp_order[j]
                        # Compute product between Green operator and stress DFT
                        idx1 = [comp_order.index(comp_i), comp_order.index(comp_j)]
                        idx2 = comp_order.index(comp_j)
                        aux = np.add(aux,np.multiply(
                            mop.kelvin_factor(idx1, comp_order)*gop_dft_vox[comp_i +
                                                                            comp_j],
                            mop.kelvin_factor(idx2, comp_order)*stress_DFT_vox[comp_j]))
                    # Update strain DFT
                    strain_DFT_vox[comp_i] = np.subtract(strain_DFT_vox[comp_i],
                        (1.0/mop.kelvin_factor(i, comp_order))*aux)
                    # Enforce macroscale strain DFT at the zero-frequency
                    freq_0_idx = n_dim*(0,)
                    strain_DFT_vox[comp_i][freq_0_idx] = mac_strain_DFT_0[comp_i]
                #
                #                           Strain Inverse Discrete Fourier Transform (IDFT)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute strain Inverse Discrete Fourier Transform (IDFT)
                for comp in comp_order:
                    # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
                    # Transform (FFT)
                    strain_vox[comp] = np.real(np.fft.ifftn(strain_DFT_vox[comp]))
                #
                #                                                              Stress update
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update stress
                stress_vox = \
                    self._elastic_constitutive_model(strain_vox, evar1, evar2, evar3)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute average strain/stress norm
                if self._conv_criterion == 'avg_stress':
                    # Update last iteration average stress norm
                    avg_stress_norm_old = avg_stress_norm
                    # Compute average stress norm
                    avg_stress_norm = self._compute_avg_state_vox(stress_vox)
                elif self._conv_criterion == 'avg_strain':
                    # Update last iteration average strain norm
                    avg_strain_norm_old = avg_strain_norm
                    # Compute average strain norm
                    avg_strain_norm = self._compute_avg_state_vox(strain_vox)
            #
            #                                                Macroscale strain increment cut
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_inc_cut:
                # Reset macroscale strain increment cut flag
                is_inc_cut = False
                # Perform macroscale strain increment cut
                mac_strain_incrementer.increment_cut()
                # Get current macroscale strain tensor
                mac_strain = mac_strain_incrementer.get_current_mac_strain()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Display increment data
                if verbose:
                    type(self)._display_increment_init(
                        *mac_strain_incrementer.get_inc_output_data())
                # Start new macroscale strain increment solution procedure
                continue
            #
            #                                        Macroscale strain increment convergence
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute homogenized strain and stress tensor
            hom_strain = self._compute_homogenized_field(strain_vox)
            hom_stress = self._compute_homogenized_field(stress_vox)
            # Display increment data
            if verbose:
                type(self)._display_increment_end(self._strain_formulation,
                                                  hom_strain, hom_stress,
                                                  time.time() - inc_init_time,
                                                  time.time() - init_time)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return if last macroscale loading increment
            if mac_strain_incrementer.get_is_last_inc():
                # Compute material logarithmic strain tensor from deformation gradient
                if self._strain_formulation == 'finite':
                    # Loop over voxels
                    for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                        # Initialize deformation gradient
                        def_gradient = np.zeros((self._n_dim, self._n_dim))
                        # Loop over deformation gradient components
                        for comp in self._comp_order_nsym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Get voxel deformation gradient component
                            def_gradient[so_idx] = strain_vox[comp][voxel]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute material logarithmic strain tensor
                        log_strain = 0.5*top.isotropic_tensor('log',
                            np.matmul(np.transpose(def_gradient), def_gradient))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over material logarithmic strain tensor components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store material logarithmic strain tensor
                            strain_vox[comp][voxel] = log_strain[so_idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Return local strain field
                return strain_vox
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update last converged strain tensor
            strain_old_vox = copy.deepcopy(strain_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Setup new macroscale strain increment
            mac_strain_incrementer.update_inc()
            # Get current macroscale strain tensor
            mac_strain = mac_strain_incrementer.get_current_mac_strain()
            # Set increment initial time
            inc_init_time = time.time()
            # Display increment data
            if verbose:
                type(self)._display_increment_init(
                    *mac_strain_incrementer.get_inc_output_data())
    # --------------------------------------------------------------------------------------
    def _elastic_constitutive_model(self, strain_vox, evar1, evar2, evar3):
        '''Material elastic or hyperelastic constitutive model.

        Infinitesimal strains: standard isotropic linear elastic constitutive model
        Finite strains: Hencky hyperelastic constitutive model

        Parameters
        ----------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str). Infinitesimal strain
            tensor (infinitesimal strains) or deformation gradient (finite strains).
        evar1 : ndarray of shape equal to RVE regular grid discretization
            Auxiliar elastic properties array containing an elastic properties-related
            quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)), where
            E and v are the Young's Modulus and Poisson's ratio, respectively.
        evar2 : ndarray of shape equal to RVE regular grid discretization
            Auxiliar elastic properties array containing an elastic properties-related
            quantity associated to each voxel: 2.0*(E/(2.0*(1.0 + v)), where E and v are
            the Young's Modulus and Poisson's ratio, respectively.
        evar3 : ndarray of shape equal to RVE regular grid discretization
            Auxiliar elastic properties array containing an elastic properties-related
            quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)) +
            2.0*(E/(2.0*(1.0 + v)), where E and v are the Young's Modulus and Poisson's
            ratio, respectively.

        Returns
        -------
        stress_vox: dict
            Local stress response (item, ndarray of shape equal to RVE regular grid
            discretization) for each stress component (key, str). Cauchy stress tensor
            (infinitesimal strains) or First Piola-Kirchhoff stress tensor (finite strains).
        '''
        # Initialize Cauchy stress tensor (infinitesimal strains) or First Piola-Kirchhoff
        # stress tensor (finite strains)
        if self._strain_formulation == 'infinitesimal':
            stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                          for comp in self._comp_order_sym}
        elif self._strain_formulation == 'finite':
            stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                          for comp in self._comp_order_nsym}
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute spatial logarithmic strain tensor
        if self._strain_formulation == 'finite':
            # Initialize spatial logarithmic strain tensor
            log_strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                              for comp in self._comp_order_sym}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over voxels
            for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                # Initialize deformation gradient
                def_gradient = np.zeros((self._n_dim, self._n_dim))
                # Loop over deformation gradient components
                for comp in self._comp_order_nsym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Get voxel deformation gradient component
                    def_gradient[so_idx] = strain_vox[comp][voxel]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute spatial logarithmic strain tensor
                log_strain = 0.5*top.isotropic_tensor('log',
                    np.matmul(def_gradient, np.transpose(def_gradient)))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over spatial logarithmic strain tensor components
                for comp in self._comp_order_sym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Store spatial logarithmic strain tensor
                    log_strain_vox[comp][voxel] = log_strain[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set spatial logarithmic strain tensor
            strain_vox = log_strain_vox
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Cauchy stress tensor from infinitesimal strain tensor (infinitesimal
        # strains) or Kirchhoff stress tensor from spatial logarithmic strain tensor
        # (finite strains)
        if self._problem_type == 1:
            stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                      np.multiply(evar1, strain_vox['22']))
            stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                      np.multiply(evar1, strain_vox['11']))
            stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
        else:
            stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                      np.multiply(evar1, np.add(strain_vox['22'],
                                                               strain_vox['33'])))
            stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                      np.multiply(evar1, np.add(strain_vox['11'],
                                                               strain_vox['33'])))
            stress_vox['33'] = np.add(np.multiply(evar3, strain_vox['33']),
                                      np.multiply(evar1, np.add(strain_vox['11'],
                                                               strain_vox['22'])))
            stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
            stress_vox['23'] = np.multiply(evar2, strain_vox['23'])
            stress_vox['13'] = np.multiply(evar2, strain_vox['13'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute First Piola-Kirchhoff stress tensor
        if self._strain_formulation == 'finite':
            # Initialize First Piola-Kirchhoff stress tensor
            piola_stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                                for comp in self._comp_order_nsym}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over voxels
            for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                # Initialize Kirchhoff stress tensor
                kirchhoff_stress = np.zeros((self._n_dim, self._n_dim))
                # Loop over Kirchhoff stress tensor components
                for comp in self._comp_order_sym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Get voxel Kirchhoff stress tensor component
                    kirchhoff_stress[so_idx] = stress_vox[comp][voxel]
                    if so_idx[0] != so_idx[1]:
                        kirchhoff_stress[so_idx[::-1]] = kirchhoff_stress[so_idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute First Piola-Kirchhoff stress tensor
                piola_stress = np.matmul(kirchhoff_stress,
                                         np.transpose(np.linalg.inv(def_gradient)))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over First Piola-Kirchhoff stress tensor components
                for comp in self._comp_order_nsym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Store First Piola-Kirchhoff stress tensor
                    piola_stress_vox[comp][voxel] = piola_stress[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set First Piola-Kirchhoff stress tensor
            stress_vox = piola_stress_vox
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return stress_vox
    # --------------------------------------------------------------------------------------
    def _stress_div_conv_criterion(self, freqs_dims, stress_DFT_vox):
        '''Convergence criterion based on the divergence of the stress tensor.

        Convergence criterion proposed by H. Moulinec and P. Suquet ("A numerical method for
        computing the overall response of nonlinear composites with complex microstructure"
        Comp Methods Appl M 157 (1998):69-94).

        Parameters
        ----------
        freqs_dims : list
            List of discrete frequencies (1darray) associated to each spatial dimension.
        stress_DFT_vox : dict
            Discrete Fourier Transform of local stress response (item, ndarray of shape
            equal to RVE regular grid discretization) for each stress component (key, str).
            Cauchy stress tensor (infinitesimal strains) or First Piola-Kirchhoff stress
            tensor (finite strains).

        Returns
        -------
        discrete_error : float
            Discrete error associated to the convergence criterion.
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order == self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # Compute total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize discrete error sum
        error_sum = 0.0
        # Initialize stress DFT at the zero-frequency
        stress_DFT_0_mf = np.zeros(len(comp_order), dtype=complex)
        # Initialize stress divergence DFT
        div_stress_DFT = \
            {str(comp + 1): np.zeros(tuple(self._n_voxels_dims), dtype=complex)
             for comp in range(self._n_dim)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get discrete frequency index
            freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) \
                              for x in range(self._n_dim)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize stress tensor DFT matricial form
            stress_DFT_mf = np.zeros(len(comp_order), dtype=complex)
            # Loop over stress components
            for i in range(len(comp_order)):
                # Get stress component
                comp = comp_order[i]
                # Build stress tensor DFT matricial form
                stress_DFT_mf[i] = mop.kelvin_factor(i, comp_order)*\
                    stress_DFT_vox[comp][freq_idx]
                # Store stress tensor DFT matricial form for zero-frequency
                if freq_idx == self._n_dim*(0,):
                    stress_DFT_0_mf[i] = mop.kelvin_factor(i, comp_order)*\
                        stress_DFT_vox[comp][freq_idx]
            # Build stress tensor DFT
            stress_DFT = mop.get_tensor_from_mf(stress_DFT_mf, self._n_dim, comp_order)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add discrete frequency contribution to discrete error sum
            error_sum = error_sum + np.linalg.norm(
                top.dot12_1(1j*np.asarray(freq_coord), stress_DFT))**2
            # Compute stress divergence Discrete Fourier Transform (DFT)
            for i in range(self._n_dim):
                div_stress_DFT[str(i + 1)][freq_idx] = \
                    top.dot12_1(1j*np.asarray(freq_coord), stress_DFT)[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute discrete error
        discrete_error = \
            np.sqrt(error_sum/n_voxels)/np.linalg.norm(stress_DFT_0_mf)
        # Compute stress divergence Inverse Discrete Fourier Transform (IDFT)
        div_stress = {str(comp + 1): np.zeros(tuple(self._n_voxels_dims)) \
                      for comp in range(self._n_dim)}
        for i in range(self._n_dim):
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            div_stress[str(i + 1)] = \
                np.real(np.fft.ifftn(div_stress_DFT[str(i + 1)]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return discrete_error
    # --------------------------------------------------------------------------------------
    def _compute_avg_state_vox(self, state_vox):
        '''Compute average norm of strain or stress local field.

        Parameters
        ----------
        state_vox : dict
            Local strain or stress response (item, ndarray of shape equal to RVE regular
            grid discretization) for each strain or stress component (key, str).

        Returns
        -------
        avg_state_norm : float
            Average norm of strain or stress local field.
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        # Compute total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize average strain or stress norm
        avg_state_norm = 0
        # Loop over strain or stress components
        for i in range(len(comp_order)):
            # Get component
            comp = comp_order[i]
            # Add contribution to average norm
            if self._strain_formulation == 'infinitesimal' and comp[0] != comp[1]:
                # Account for symmetric component
                avg_state_norm = avg_state_norm + 2.0*np.square(state_vox[comp])
            else:
                avg_state_norm = avg_state_norm + np.square(state_vox[comp])
        # Compute average norm
        avg_state_norm = np.sum(np.sqrt(avg_state_norm))/n_voxels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return avg_state_norm
    # --------------------------------------------------------------------------------------
    def _compute_homogenized_field(self, state_vox):
        '''Perform strain or stress homogenization over regular grid spatial discretization.

        Parameters
        ----------
        state_vox : dict
            Local strain or stress response (item, ndarray of shape equal to RVE regular
            grid discretization) for each strain or stress component (key, str).

        Returns
        -------
        hom_state : 2darray
            Homogenized strain or stress tensor.
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        # Compute total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized strain or stress tensor
        hom_state = np.zeros((self._n_dim, self._n_dim))
        # Loop over strain or stress tensor components
        for comp in comp_order:
            # Get second-order array index
            so_idx = tuple([int(i) - 1 for i in comp])
            # Assemble strain or stress component
            hom_state[so_idx] = np.sum(state_vox[comp])
            # Account for symmetric component
            if self._strain_formulation == 'infinitesimal' and comp[0] != comp[1]:
                hom_state[so_idx[::-1]] = np.sum(state_vox[comp])
        # Complete field homogenization
        hom_state = (1.0/n_voxels)*hom_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return hom_state
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_greetings():
        '''Output homogenization-based multiscale DNS method method greetings.'''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        tilde_line, _ = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = ('FFT-Based Homogenization Method (H. Moulinec and P. Suquet)',
                'Implemented by Bernardo P. Ferreira (bpferreira@fe.up.pt)',
                'Last version: October 2021')
        # Set output template
        template = '\n' + colorama.Fore.WHITE + tilde_line + colorama.Style.RESET_ALL + \
                   colorama.Fore.WHITE + '\n{:^{width}}\n' + '\n{:^{width}}\n' + \
                   '\n{:^{width}}\n' + colorama.Fore.WHITE + \
                   tilde_line + colorama.Style.RESET_ALL + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        #ioutil.print2(template.format(*info, width=output_width))
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_increment_init(inc, subinc_level, total_lfact, inc_lfact):
        '''Output increment initial data.

        Parameters
        ----------
        inc : int
            Increment number.
        subinc_level : int
            Subincrementation level.
        total_lfact : float
            Total load factor.
        inc_lfact : float
            Incremental load factor.
        '''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, _, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if subinc_level == 0:
            # Set output data
            info = (inc, total_lfact, inc_lfact)
            # Set output template
            template = colorama.Fore.CYAN + '\n' + \
                       indent + 'Increment number: {:3d}' + '\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 60*' ' + 'Load factor | Total = {:8.1e}' + 7*' ' + '\n' + \
                       indent + 72*' ' + '| Incr. = {:8.1e}' + \
                       colorama.Style.RESET_ALL + '\n'
        else:
            # Set output data
            info = (inc, subinc_level, total_lfact, inc_lfact)
            # Set output template
            template = colorama.Fore.CYAN + '\n' + \
                       indent + 'Increment number: {:3d}' + 3*' ' + \
                       '(Sub-inc. level: {:3d})' + '\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 60*' ' + 'Load factor | Total = {:8.1e}' + 7*' ' + '\n' + \
                       indent + 72*' ' + '| Incr. = {:8.1e}' + \
                       colorama.Style.RESET_ALL  + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        #ioutil.print2(template.format(*info, width=output_width))
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_increment_end(strain_formulation, hom_strain, hom_stress, inc_time,
                               total_time):
        '''Output increment end data.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        hom_strain : 2darray
            Homogenized strain tensor.
        hom_stress : 2darray
            Homogenized stress tensor.
        inc_time : float
            Increment running time.
        total_time : float
            Total running time.
        '''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        space_1 = (output_width - 84)*' '
        space_2 = (output_width - (len('Homogenized strain tensor') + 48))*' '
        space_3 = (output_width - (len('Increment run time (s): ') + 44))*' '
        space_4 = (output_width - 72)*' '
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain and stress nomenclature
        if strain_formulation == 'infinitesimal':
            strain_header = 'Homogenized strain tensor (\u03B5)'
            stress_header = 'Homogenized stress tensor (\u03C3)'
        else:
            strain_header = 'Homogenized strain tensor (F)'
            stress_header = 'Homogenized stress tensor (P)'
        # Get problem number of spatial dimensions
        n_dim = hom_strain.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build strain and stress components list
        comps = list()
        for i in range(n_dim):
            for j in range(n_dim):
                comps.append(hom_strain[i, j])
            for j in range(n_dim):
                comps.append(hom_stress[i, j])
        # Get output data
        info = tuple(comps + [inc_time, total_time])
        # Set output template
        if n_dim == 2:
            template = indent + dashed_line[:-len(indent)] + '\n\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 7*' ' + strain_header + space_2 + stress_header + '\n\n' + \
                       indent + 6*' ' + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_4 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + 6*' ' + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_4 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n'
        else:
            template = indent + dashed_line[:-len(indent)] + '\n\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 7*' ' + strain_header + space_2 + stress_header + '\n\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n'

        template += '\n' + indent + equal_line[:-len(indent)] + '\n' + \
                    indent + 'Increment run time (s): {:>11.4e}' + space_3 + \
                    'Total run time (s): {:>11.4e}' + '\n\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        #ioutil.print2(template.format(*info, width=output_width))
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_iteration(iter, iter_time, discrete_error):
        '''Output iteration data.

        Parameters
        ----------
        iter : int
            Iteration number.
        iter_time : float
            Iteration running time.
        discrete_error : float
            Discrete error associated to convergence criterion.
        '''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        space_1 = (output_width - 29)*' '
        space_2 = (output_width - 35)*' '
        space_3 = (output_width - 38)*' '
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        template = ''
        # Set iteration output header
        if iter == 0:
            template += indent + 5*' ' + 'Iteration' + space_1 + 'Convergence' '\n' + \
                        indent + ' Number    Run time (s)' + space_2 + \
                        'Error' + '\n' + \
                        indent + dashed_line[:-len(indent)] + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = (iter, iter_time, discrete_error)
        # Set output template
        template += indent + ' {:^6d}    {:^12.4e}' + space_3 + '{:>11.4e}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        #ioutil.print2(template.format(*info, width=output_width))
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_increment_cut(max_n_iterations):
        '''Output increment cut data.

        Parameters
        ----------
        max_n_iterations : int
            Maximum number of iterations to convergence.
        '''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, _, indent, asterisk_line = display_features[0:4]
        _, _ = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set increment cut cause
        cut_msg = 'Maximum number of iterations ({}) reached without convergence.'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = (max_n_iterations, )
        # Set output template
        template = '\n\n' + colorama.Fore.RED + indent + asterisk_line[:-len(indent)] + \
                   '\n' + \
                   indent + 'Increment cut: ' + colorama.Style.RESET_ALL + cut_msg + \
                   '\n' + colorama.Fore.RED + indent + asterisk_line[:-len(indent)] + \
                   colorama.Style.RESET_ALL + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        #ioutil.print2(template.format(*info, width=output_width))
# ------------------------------------------------------------------------------------------
class MacroscaleStrainIncrementer:
    '''Macroscale strain loading incrementer.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _inc : int
        Increment counter.
    _total_lfact : float
        Total load factor.
    _inc_mac_strain_total : 2darray
        Total incremental macroscale strain second-order tensor. Infinitesimal strain
        tensor (infinitesimal strains) or deformation gradient (finite strains).
    _mac_strain : 2darray
        Current macroscale strain second-order tensor. Infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains).
    _mac_strain_old : 2darray
        Last converged macroscale strain tensor. Infinitesimal strain tensor (infinitesimal
        strains) or deformation gradient (finite strains).
    _is_last_inc : bool
        Last increment flag.
    _sub_inc_levels : list
        List of increments subincrementation level.
    _n_cinc_cuts : int
        Consecutive increment cuts counter.
    '''
    def __init__(self, strain_formulation, problem_type, mac_strain_total,
                 mac_strain_init=None, inc_lfacts=[1.0,], max_subinc_level=5,
                 max_cinc_cuts=5):
        '''Macroscale strain loading incrementer constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        mac_strain_total : 2darray
            Total macroscale strain tensor. Infinitesimal strain tensor (infinitesimal
            strains) or deformation gradient (finite strains).
        mac_strain_init : 2darray, default=None
            Initial macroscale strain tensor. Infinitesimal strain tensor (infinitesimal
            strains) or deformation gradient (finite strains).
        inc_lfacts : list, default=[1.0,]
            List of incremental load factors (float). Default applies the total macroscale
            strain tensor in a single increment.
        max_subinc_level : int, default=5
            Maximum level of macroscale loading subincrementation.
        max_cinc_cuts : int, default=5
            Maximum number of consecutive macroscale loading increment cuts.
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set total macroscale strain tensor
        self._mac_strain_total = mac_strain_total
        # Set initial macroscale strain tensor
        if mac_strain_init != None:
            self._mac_strain_init = mac_strain_init
        else:
            if self._strain_formulation == 'infinitesimal':
                self._mac_strain_init = np.zeros((self._n_dim, self._n_dim))
            else:
                self._mac_strain_init = np.eye(self._n_dim)
        # Set total incremental macroscale strain tensor
        if self._strain_formulation == 'infinitesimal':
            # Additive decomposition of infinitesimal strain tensor
            self._inc_mac_strain_total = self._mac_strain_total - self._mac_strain_init
        else:
            # Multiplicative decomposition of deformation gradient
            self._inc_mac_strain_total = np.matmul(self._mac_strain_total,
                                                   np.linalg.inv(self._mac_strain_init))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize increment counter
        self._inc = 0
        # Initialize current macroscale strain tensor
        self._mac_strain = copy.deepcopy(self._mac_strain_init)
        # Initialize last converged macroscale strain
        self._mac_strain_old = copy.deepcopy(self._mac_strain)
        # Set list of incremental load factors
        self._inc_lfacts = copy.deepcopy(inc_lfacts)
        # Initialize subincrementation levels
        self._sub_inc_levels = [0,]*len(self._inc_lfacts)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._max_subinc_level = max_subinc_level
        self._max_cinc_cuts = max_cinc_cuts
        # Initialize consecutive increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize last increment flag
        self._is_last_inc = False
    # --------------------------------------------------------------------------------------
    def get_inc(self):
        '''Get current increment counter.

        Returns
        -------
        inc : int
            Increment counter.
        '''
        return self._inc
    # --------------------------------------------------------------------------------------
    def get_current_mac_strain(self):
        '''Get current macroscale strain.

        Returns
        -------
        mac_strain : 2darray
            Current macroscale strain tensor. Infinitesimal strain tensor (infinitesimal
            strains) or deformation gradient (finite strains).
        '''
        return copy.deepcopy(self._mac_strain)
    # --------------------------------------------------------------------------------------
    def get_is_last_inc(self):
        '''Get last increment flag.

        Returns
        -------
        is_last_inc : bool
            Last increment flag.
        '''
        return self._is_last_inc
    # --------------------------------------------------------------------------------------
    def get_inc_output_data(self):
        '''Get increment output data.

        Returns
        -------
        inc_data : tuple
            Increment output data.
        '''
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return (self._inc, self._sub_inc_levels[inc_idx], self._total_lfact,
                self._inc_lfacts[inc_idx])
    # --------------------------------------------------------------------------------------
    def update_inc(self):
        '''Update increment counter, total load factor and current macroscale strain.'''
        # Update last converged macroscale strain
        self._mac_strain_old = copy.deepcopy(self._mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment increment counter
        self._inc += 1
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset consecutive increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Procedure related with the macroscale strain subincrementation: upon convergence
        # of a given increment, guarantee that the following increment magnitude is at most
        # one (subincrementation) level above. The increment cut procedure is performed the
        # required number of times in order to ensure this progressive recovery towards the
        # prescribed incrementation
        if self._inc > 1:
            while self._sub_inc_levels[inc_idx - 1] - self._sub_inc_levels[inc_idx] >= 2:
                self.increment_cut()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        # Update current macroscale strain
        self._update_mac_strain()
        # Check if last increment
        if self._inc == len(self._inc_lfacts):
            self._is_last_inc = True
    # --------------------------------------------------------------------------------------
    def increment_cut(self):
        '''Perform macroscale strain increment cut.'''
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment consecutive increment cuts counter
        self._n_cinc_cuts += 1
        # Check if maximum number of consecutive increment cuts is surpassed
        if self._n_cinc_cuts > self._max_cinc_cuts:
            raise RuntimeError('Maximum number of consecutive increments cuts reached '
                               'without convergence.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update subincrementation level
        self._sub_inc_levels[inc_idx] += 1
        self._sub_inc_levels.insert(inc_idx + 1, self._sub_inc_levels[inc_idx])
        # Check if maximum subincrementation level is surpassed
        if self._sub_inc_levels[inc_idx] > self._max_subinc_level:
            raise RuntimeError('Maximum subincrementation level reached without '
                               'convergence.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get current incremental load factor
        inc_lfact = self._inc_lfacts[inc_idx]
        # Cut macroscale strain increment in half
        self._inc_lfacts[inc_idx] = inc_lfact/2.0
        self._inc_lfacts.insert(inc_idx + 1, self._inc_lfacts[inc_idx])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update current macroscale strain
        self._update_mac_strain()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset last increment flag
        self._is_last_inc = False
    # --------------------------------------------------------------------------------------
    def _update_mac_strain(self):
        '''Update current macroscale strain loading.'''
        # Get increment index
        inc_idx = self._inc - 1
        # Get current incremental load factor
        inc_lfact = self._inc_lfacts[inc_idx]
        # Compute current macroscale strain loading according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            # Compute incremental macroscale infinitesimal strain tensor
            inc_mac_strain = inc_lfact*self._inc_mac_strain_total
            # Compute current macroscale infinitesimal strain tensor
            self._mac_strain = self._mac_strain_old + inc_mac_strain
        else:
            # Compute incremental macroscale deformation gradient
            inc_mac_strain = mop.matrix_root(self._inc_mac_strain_total, inc_lfact)
            # Compute current macroscale deformation gradient
            self._mac_strain = np.matmul(inc_mac_strain, self._mac_strain_old)


# OLD CODE BELOW (TO BE REMOVED LATER)
#
#                                                      FFT-Based Homogenization Basic Scheme
# ==========================================================================================
# This function is the implementation of the FFT-based homogenization method proposed in
# "A numerical method for computing the overall response of nonlinear composites with
# complex microstructure. Comp Methods Appl M 157 (1998):69-94 (Moulinec, H. and
# Suquet, P.)". For a given RVE discretized in a regular grid of pixels (2D) / voxels (3D)
# (with a total of n_voxels pixels/voxels) and where each pixel/voxel is associated to a
# given material phase, this method solves the microscale static equilibrium problem when
# the RVE is subjected to a given macroscale strain and constrained by periodic boundary
# conditions.
#
# Method scope: | Small strains
#
# Implementation scope: | 2D problems (plain strain) and 3D problems
#                       | Linear elastic constitutive behavior
#                       | Material isotropy
#
# Note 1: At the end of article's 2.4 section, Moulinec and Suquet propose a modification of
#         the Green operator at the higher frequencies in order to overcome limitations of
#         the FFT packages (1998...) for low spatial resolutions. Such modification is not
#         implemented here.
#
# Note 2: Besides the original convergence criterion proposed by Moulinec and Suquet, a
#         second convergence criterion based on the average stress tensor norm has been
#         implemented.
#
# The function returns the strain tensor in every pixel/voxel stored componentwise as:
#
# A. 2D problem (plane strain):
#
#   strain_vox[comp] = array(d1,d2), where | di is the number of pixels in dimension i
#                                          | comp is the strain component that would be
#                                          | stored in matricial form ('11', '22', '12')
#
# B. 3D problem:
#
#   strain_vox[comp] = array(d1,d2,d3), where | di is the number of pixels in dimension i
#                                             | comp is the strain component that would be
#                                             |     stored in matricial form ('11', '22',
#                                             |     '33', '12', '23', '13')
#
# Note 1: All the strain or stress related tensors stored componentwise in dictionaries (for
#         every pixels/voxels) do not follow the Kelvin notation, i.e. the stored component
#         values are the real ones.
#
# Note 2: The suffix '_mf' is employed to denote tensorial quantities stored in matricial
#         form. The matricial form follows the Kelvin notation when symmetry conditions
#         exist and is performed columnwise otherwise.
#
def ffthombasicscheme(problem_dict, rg_dict, mat_dict, mac_strain):
    # --------------------------------------------------------------------------------------
    # Time profile
    is_time_profile = False
    is_conv_output = False
    is_validation_return = False
    if is_time_profile:
        # Date and time
        import time
        # Initialize time profile variables
        start_time_s = time.time()
        phase_names = ['']
        phase_times = np.zeros((1, 2))
        phase_names[0] = 'Total'
        phase_times[0, :] = [start_time_s, 0.0]
        # Set validation return flag
        is_validation_return = True
        # Set convergence output flag
        is_conv_output = True
    # --------------------------------------------------------------------------------------
    #
    #                                                                             Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence criterion
    # 1. Original convergence criterion (Moulinec & Suquet, 1998) - Non-optimized!
    # 2. Average stress norm based convergence criterion
    conv_criterion = 2
    # Set maximum number of iterations
    max_n_iterations = 100
    # Set convergence tolerance
    conv_tol = 1e-6
    #
    #                                                                        Input arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Get problem type and dimensions
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get the spatial discretization file (regular grid of pixels/voxels)
    regular_grid = rg_dict['regular_grid']
    # Get number of pixels/voxels in each dimension and total number of pixels/voxels
    n_voxels_dims = rg_dict['n_voxels_dims']
    n_voxels = np.prod(n_voxels_dims)
    # Get RVE dimensions
    rve_dims = rg_dict['rve_dims']
    # Get material phases
    material_phases = mat_dict['material_phases']
    # Get material properties
    material_properties = mat_dict['material_properties']
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Input arguments')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set optimized variables
    var6 = np.zeros(tuple(n_voxels_dims))
    var7 = np.zeros(tuple(n_voxels_dims))
    for mat_phase in material_phases:
        E = material_properties[mat_phase]['E']
        v = material_properties[mat_phase]['v']
        var6[regular_grid == int(mat_phase)] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        var7[regular_grid == int(mat_phase)] = np.multiply(2,E/(2.0*(1.0 + v)))
    var8 = np.add(var6,var7)
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Material phases elasticity tensors')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                  Reference material elastic properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set reference material elastic properties as the mean between the minimum and maximum
    # values existent in the microstructure material phases (this is reported in
    # Moulinec, H. and Suquet, P., 1998 as the choice that leads to the best rate of
    # convergence)
    mat_prop_ref = dict()
    mat_prop_ref['E'] = \
        0.5*(min([material_properties[phase]['E'] for phase in material_phases]) + \
             max([material_properties[phase]['E'] for phase in material_phases]))
    mat_prop_ref['v'] = \
        0.5*(min([material_properties[phase]['v'] for phase in material_phases]) + \
             max([material_properties[phase]['v'] for phase in material_phases]))
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Reference material elastic properties')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                               Frequency discretization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i], sampling_period))
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Frequency discretization')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                      Reference material Green operator
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # The fourth-order Green operator for the reference material is computed for every
    # pixel/voxel and stored as follows:
    #
    # A. 2D problem (plane strain):
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |      form ('1111', '2211', '1211', '1122', ...)
    #
    # B. 3D problem:
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2,d3),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |     form ('1111', '2211', '3311', '1211', ...)
    #
    # Get reference material Young modulus and Poisson coeficient
    E_ref = mat_prop_ref['E']
    v_ref = mat_prop_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator reference material related constants
    c1 = 1.0/(4.0*miu_ref)
    c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    # Set Green operator matricial form components
    comps = list(it.product(comp_order, comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
        mf_indexes.append([x for x in \
                          [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
    # Set optimized variables
    var1 = [*np.meshgrid(*freqs_dims, indexing = 'ij')]
    var2 = dict()
    for fo_idx in fo_indexes:
        if str(fo_idx[1]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[1]], var1[fo_idx[3]])
        if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[1]], var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[0]], var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[0]], var1[fo_idx[3]])
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]], var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]], var1[fo_idx[3]]))
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]), np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]), np.square(var1[1])),
                              np.square(var1[2])))
    # Initialize Green operator
    Gop_DFT_vox = {''.join([str(x+1) for x in idx]): \
        np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Set optimized variables
        var4 = [fo_idx[0] == fo_idx[2], fo_idx[0] == fo_idx[3],
                fo_idx[1] == fo_idx[3], fo_idx[1] == fo_idx[2]]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]),str(fo_idx[1]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[2]),str(fo_idx[0]) + str(fo_idx[3])]
        # Compute first material independent term of Green operator
        first_term = np.zeros(tuple(n_voxels_dims))
        for j in range(len(var4)):
            if var4[j]:
                first_term = np.add(first_term, var2[var5[j]])
        first_term = np.divide(first_term, np.square(var3), where = abs(var3) > 1e-10)
        # Compute second material independent term of Green operator
        second_term = -1.0*np.divide(var2[''.join([str(x) for x in fo_idx])],
                                     np.square(np.square(var3)), where = abs(var3) > 1e-10)
        # Compute Green operator matricial form component
        Gop_DFT_vox[comp] = c1*first_term + c2*second_term
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Reference material Green operator')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Initialize strain and stress tensors
    strain_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    stress_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Strain and stress tensors initialization')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                Initial iterative guess
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set strain initial iterative guess
    for comp in comp_order:
        so_idx = tuple([int(x) - 1 for x in comp])
        strain_vox[comp] = np.full(regular_grid.shape, mac_strain[so_idx])
    # Compute stress initial iterative guess
    if problem_type == 1:
        stress_vox['11'] = np.add(np.multiply(var8, strain_vox['11']),
                                  np.multiply(var6, strain_vox['22']))
        stress_vox['22'] = np.add(np.multiply(var8, strain_vox['22']),
                                  np.multiply(var6, strain_vox['11']))
        stress_vox['12'] = np.multiply(var7, strain_vox['12'])
    else:
        stress_vox['11'] = np.add(np.multiply(var8, strain_vox['11']),
                                  np.multiply(var6, np.add(strain_vox['22'],
                                                           strain_vox['33'])))
        stress_vox['22'] = np.add(np.multiply(var8, strain_vox['22']),
                                  np.multiply(var6, np.add(strain_vox['11'],
                                                           strain_vox['33'])))
        stress_vox['33'] = np.add(np.multiply(var8, strain_vox['33']),
                                  np.multiply(var6, np.add(strain_vox['11'],
                                                           strain_vox['22'])))
        stress_vox['12'] = np.multiply(var7, strain_vox['12'])
        stress_vox['23'] = np.multiply(var7, strain_vox['23'])
        stress_vox['13'] = np.multiply(var7, strain_vox['13'])
    # Compute average stress norm (convergence criterion)
    avg_stress_norm = 0
    for i in range(len(comp_order)):
        comp = comp_order[i]
        if comp[0] == comp[1]:
            avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
        else:
            avg_stress_norm = avg_stress_norm + 2.0*np.square(stress_vox[comp])
    avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
    avg_stress_norm_old = 0
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Initial iterative guess')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                Strain Discrete Fourier Transform (DFT)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Compute strain Discrete Fourier Transform (DFT)
    strain_DFT_vox = {comp: np.zeros(tuple(n_voxels_dims), dtype=complex) \
                      for comp in comp_order}
    for comp in comp_order:
        # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
        strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
    # Store macroscale strain DFT at the zero-frequency
    freq_0_idx = n_dim*(0,)
    mac_strain_DFT_0 = np.zeros((n_dim, n_dim))
    mac_strain_DFT_0 = np.array([strain_DFT_vox[comp][freq_0_idx] for comp in comp_order])
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Strain Discrete Fourier Transform (DFT)')
        phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize iteration counter:
    iter = 0
    # Start iterative loop
    while True:
        #
        #                                            Stress Discrete Fourier Transform (DFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Compute stress Discrete Fourier Transform (DFT)
        stress_DFT_vox = {comp: np.zeros(tuple(n_voxels_dims), dtype=complex) \
                          for comp in comp_order}
        for comp in comp_order:
            # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
            stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Stress Discrete Fourier Transform (DFT)')
            phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],
                                    axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                             Convergence evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Convergence criterion 1:
        if conv_criterion == 1:
            # Discrete error as proposed in (Moulinec, H. and Suquet, P., 1998)
            error_sum = 0
            stress_DFT_0_mf = np.zeros(len(comp_order), dtype=complex)
            div_stress_DFT = {str(comp+1): np.zeros(tuple(n_voxels_dims), dtype=complex) \
                              for comp in range(n_dim)}
            # Loop over discrete frequencies
            for freq_coord in it.product(*freqs_dims):
                # Get discrete frequency index
                freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) \
                                  for x in range(n_dim)])
                # Initialize stress tensor Discrete Fourier Transform (DFT) matricial form
                stress_DFT_mf = np.zeros(len(comp_order), dtype=complex)
                for i in range(len(comp_order)):
                    comp = comp_order[i]
                    # Build stress tensor Discrete Fourier Transform (DFT) matricial form
                    stress_DFT_mf[i] = \
                        mop.kelvin_factor(i,comp_order)*stress_DFT_vox[comp][freq_idx]
                    # Store stress tensor Discrete Fourier Transform (DFT) matricial form
                    # for zero-frequency
                    if freq_idx == n_dim*(0,):
                        stress_DFT_0_mf[i] = \
                            mop.kelvin_factor(i,comp_order)*stress_DFT_vox[comp][freq_idx]
                # Build stress tensor Discrete Fourier Transform (DFT)
                stress_DFT = np.zeros((n_dim, n_dim), dtype=complex)
                stress_DFT = mop.get_tensor_from_mf(stress_DFT_mf, n_dim, comp_order)
                # Add discrete frequency contribution to discrete error required sum
                error_sum = error_sum + \
                    np.linalg.norm(top.dot12_1(1j*np.asarray(freq_coord), stress_DFT))**2
                # Compute stress divergence Discrete Fourier Transform (DFT)
                for i in range(n_dim):
                    div_stress_DFT[str(i + 1)][freq_idx] = \
                        top.dot12_1(1j*np.asarray(freq_coord), stress_DFT)[i]
            # Compute discrete error serving to check convergence
            discrete_error_1 = np.sqrt(error_sum/n_voxels)/np.linalg.norm(stress_DFT_0_mf)
            # Compute stress divergence Inverse Discrete Fourier Transform (IDFT)
            div_stress = {str(comp + 1): np.zeros(tuple(n_voxels_dims)) \
                          for comp in range(n_dim)}
            for i in range(n_dim):
                # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
                # Transform (FFT)
                div_stress[str(i + 1)] = np.real(np.fft.ifftn(div_stress_DFT[str(i + 1)]))
        # ----------------------------------------------------------------------------------
        # Convergence criterion 2:
        if conv_criterion == 2:
            # Discrete error based on the average stress norm
            discrete_error_2 = abs(avg_stress_norm - avg_stress_norm_old)/avg_stress_norm
        # ----------------------------------------------------------------------------------
        print('\nIteration', iter, '- Convergence evaluation:\n')
        print('Average stress norm     = ', '{:>11.4e}'.format(avg_stress_norm))
        print('Average stress norm old = ', '{:>11.4e}'.format(avg_stress_norm_old))
        print('Discrete error          = ', '{:>11.4e}'.format(discrete_error_2))
        # Validation:
        if is_conv_output:
            print('\nIteration', iter, '- Convergence evaluation:\n')
            print('Average stress norm     = ', '{:>11.4e}'.format(avg_stress_norm))
            print('Average stress norm old = ', '{:>11.4e}'.format(avg_stress_norm_old))
            if conv_criterion == 1:
                print('Discrete error          = ', '{:>11.4e}'.format(discrete_error_1))
            else:
                print('Discrete error          = ', '{:>11.4e}'.format(discrete_error_2))
            # Print iterative stress convergence for file
            if n_dim == 2:
                conv_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                                 'offline_stage/main/2D/FFT_NEW/' + \
                                 'Disk_50_0.3_100_100_uniaxial/convergence_table.dat'
            else:
                conv_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                                 'offline_stage/main/3D/FFT_NEW/' + \
                                 'Sphere_20_0.2_30_30_30_uniaxial/convergence_table.dat'
            if conv_criterion == 1:
                writeIterationConvergence(conv_file_path, 'iteration', iter,
                                          discrete_error_1, 0.0)
            elif conv_criterion == 2:
                writeIterationConvergence(conv_file_path,'iteration', iter, 0.0,
                                          discrete_error_2)
            else:
                writeIterationConvergence(conv_file_path,'iteration', iter,
                                          discrete_error_1, discrete_error_2)
        # ----------------------------------------------------------------------------------
        # Check if the solution converged (return) and if the maximum number of iterations
        # was reached (stop execution)
        if conv_criterion == 1 and discrete_error_1 < conv_tol:
            # ------------------------------------------------------------------------------
            # Time profile
            if is_validation_return:
                return [strain_vox, stress_vox, phase_names, phase_times]
            # ------------------------------------------------------------------------------
            # Return strain
            return strain_vox
        elif conv_criterion == 2 and discrete_error_2 < conv_tol:
            # ------------------------------------------------------------------------------
            # Time profile
            if is_validation_return:
                return [strain_vox, stress_vox, phase_names, phase_times]
            # ------------------------------------------------------------------------------
            # Return strain
            return strain_vox
        if iter == max_n_iterations:
            # Stop execution
            print('\nAbort: The maximum number of iterations was reached before ' + \
                  'solution convergence \n(ffthombasicscheme.py).')
            sys.exit(1)
        # Increment iteration counter
        iter = iter + 1
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Convergence evaluation')
            phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],
                                    axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                                      Strain update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        for i in range(len(comp_order)):
            compi = comp_order[i]
            # Update strain
            aux = 0
            for j in range(len(comp_order)):
                compj = comp_order[j]
                idx1 = [comp_order.index(compi), comp_order.index(compj)]
                idx2 = comp_order.index(compj)
                aux = np.add(aux,np.multiply(
                    mop.kelvin_factor(idx1, comp_order)*Gop_DFT_vox[compi + compj],
                    mop.kelvin_factor(idx2, comp_order)*stress_DFT_vox[compj]))
            strain_DFT_vox[compi] = np.subtract(strain_DFT_vox[compi],
                (1.0/mop.kelvin_factor(i, comp_order))*aux)
            # Enforce macroscopic strain at the zero-frequency strain component
            freq_0_idx = n_dim*(0,)
            strain_DFT_vox[compi][freq_0_idx] = mac_strain_DFT_0[i]
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Update strain')
            phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],
                                    axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                   Strain Inverse Discrete Fourier Transform (IDFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Compute strain Inverse Discrete Fourier Transform (IDFT)
        for comp in comp_order:
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            strain_vox[comp] = np.real(np.fft.ifftn(strain_DFT_vox[comp]))
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Strain Inverse Discrete Fourier Transform (IDFT)')
            phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],
                                    axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                                      Stress update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Update stress
        if problem_type == 1:
            stress_vox['11'] = np.add(np.multiply(var8, strain_vox['11']),
                                      np.multiply(var6, strain_vox['22']))
            stress_vox['22'] = np.add(np.multiply(var8, strain_vox['22']),
                                      np.multiply(var6, strain_vox['11']))
            stress_vox['12'] = np.multiply(var7, strain_vox['12'])
        else:
            stress_vox['11'] = np.add(np.multiply(var8, strain_vox['11']),
                                      np.multiply(var6, np.add(strain_vox['22'],
                                                               strain_vox['33'])))
            stress_vox['22'] = np.add(np.multiply(var8, strain_vox['22']),
                                      np.multiply(var6, np.add(strain_vox['11'],
                                                               strain_vox['33'])))
            stress_vox['33'] = np.add(np.multiply(var8, strain_vox['33']),
                                      np.multiply(var6, np.add(strain_vox['11'],
                                                               strain_vox['22'])))
            stress_vox['12'] = np.multiply(var7, strain_vox['12'])
            stress_vox['23'] = np.multiply(var7, strain_vox['23'])
            stress_vox['13'] = np.multiply(var7, strain_vox['13'])
        # Compute average stress norm (convergence criterion)
        avg_stress_norm_old = avg_stress_norm
        avg_stress_norm = 0
        for i in range(len(comp_order)):
            comp = comp_order[i]
            if comp[0] == comp[1]:
                avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
            else:
                avg_stress_norm = avg_stress_norm + 2.0*np.square(stress_vox[comp])
        avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Update stress')
            phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]],
                                    axis=0)
            end_time_s = time.time()
            phase_times[0, 1] = end_time_s
            is_time_profile = False
        # ----------------------------------------------------------------------------------
#
#                                                                    Complementary functions
# ==========================================================================================
# Compute the small strain elasticity tensor according to the problem type and material
# constitutive model. Then store it in matricial form following Kelvin notation.
#
# The elasticity tensor is described as follows:
#
# A. Small strain:
#
#   A.1 Isotropic material:
#
#      General Hooke's Law: De = lambda*(IdyadI) + 2*miu*IIsym
#
#                           where | lambda, miu denote the Lamé parameters
#                                 | I denotes the second-order identity tensor
#                                 | IIsym denotes the fourth-order symmetric project. tensor
#
#      A.1.1 2D problem (plane strain) - Kelvin notation:
#                            _                _
#                           | 1-v   v      0   | (11)
#          De =  E/(2*(1+v))|  v   1-v     0   | (22)
#                           |_ 0    0   (1-2v)_| (12)
#
#      A.1.2 2D problem (plane stress) - Kelvin notation:
#                           _              _
#                          |  1   v     0   | (11)
#          De =  E/(1-v**2)|  v   1     0   | (22)
#                          |_ 0   0   (1-v)_| (12)
#
#     A.1.3 2D problem (axisymmetric) - Kelvin notation:
#                            _                     _
#                           | 1-v   v      0     v  | (11)
#          De =  E/(2*(1+v))|  v   1-v     0     v  | (22)
#                           |  0    0   (1-2v)   0  | (12)
#                           |_ v    v      0    1-v_| (33)
#
#     A.1.4 3D problem - Kelvin notation:
#                            _                                    _
#                           | 1-v   v    v     0       0       0   | (11)
#                           |  v   1-v   v     0       0       0   | (22)
#          De =  E/(2*(1+v))|  v    v  1-v     0       0       0   | (33)
#                           |  0    0    0  (1-2v)     0       0   | (12)
#                           |  0    0    0     0    (1-2v)     0   | (23)
#                           |_ 0    0    0     0       0    (1-2v)_| (13)
#
#    Note: E and v denote the Young modulus and the Poisson ratio respectively.
#
def getElasticityTensor(problem_type, n_dim, comp_order, properties):
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0 + v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0 + v))
    # Set required fourth-order tensors
    _, _, _, fosym, fodiagtrace, _, _ = top.get_id_operators(n_dim)
    # 2D problem (plane strain)
    if problem_type == 1:
        De_tensor = lam*fodiagtrace + 2.0*miu*fosym
        De_tensor_mf = mop.get_tensor_mf(De_tensor, n_dim, comp_order)
    # 3D problem
    elif problem_type == 4:
        De_tensor = lam*fodiagtrace + 2.0*miu*fosym
        De_tensor_mf = mop.get_tensor_mf(De_tensor, n_dim, comp_order)
    # Return
    return De_tensor_mf
#
#                                                                     Validation (temporary)
# ==========================================================================================
def writeVoxelStressDivergence(file_path,mode,voxel_idx,*args):
    if mode == 'header':
        # Open file where the stress divergence tensor will be written
        open(file_path,'w').close()
        file = open(file_path,'a')
        # Write file header
        print('\nStress divergence tensor evaluation',file=file)
        print('-----------------------------------',file=file)
        # Write voxel indexes
        print('\nVoxel indexes:',voxel_idx,file=file)
        # Write stress divergence tensor header
        print('\nIter'+ 9*' ' + 'x' + 12*' ' + 'y' + 12*' ' + 'z', file=file)
        # Close file where the stress divergence tensor are written
        file.close()
    else:
        # Open file where the stress divergence tensor are written
        file = open(file_path,'a')
        # Get iteration and stress divergence tensor
        iter = args[0]
        div_stress = args[1]
        # Write iterative value of stress divergence tensor
        print('{:>4d}  '.format(iter),\
              (n_dim*'{:>11.4e}  ').format(*[div_stress[str(i+1)][voxel_idx] \
                                                          for i in range(n_dim)]),file=file)
# ------------------------------------------------------------------------------------------
def writeIterationConvergence(file_path, mode, *args):
    if mode == 'header':
        # Open file where the iteration convergence metrics are written
        open(file_path, 'w').close()
        file = open(file_path, 'a')
        # Write file header
        print('\nFFT Homogenization Basic Scheme Convergence', file=file)
        print('-------------------------------------------', file=file)
        # Write stress divergence tensor header
        print('\nIter' + 5*' ' + 'error 1' +  7*' ' + 'error 2', file=file)
        # Close file where the stress divergence tensor will be written
        file.close()
    else:
        # Open file where the stress divergence tensor are written
        file = open(file_path, 'a')
        # Get iteration and stress divergence tensor
        iter = args[0]
        error1 = args[1]
        error2 = args[2]
        # Write iterative value of stress divergence tensor
        print('{:>4d} '.format(iter), '{:>11.4e}'.format(error1),
              '  {:>11.4e}'.format(error2),file=file)
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Insert this at import modules section
    #p = os.path.abspath('/home/bernardoferreira/Documents/CRATE/src')
    #sys.path.insert(1, p)
    from ioput.vtkoutput import XMLGenerator, VTKOutput
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set working directory
    working_dir = '/home/bernardoferreira/Documents/CRATE/developments/finite_strains/3d/'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 4
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set RVE spatial discretization file absolute path
    if problem_type == 1:
        discrete_file_path = working_dir + 'Disk_50_0.3_100_100.rgmsh.npy'
    else:
        discrete_file_path = working_dir + 'Sphere_10_0.2_30_30_30.rgmsh.npy'
    # Read spatial discretization file and set regular grid data
    regular_grid = np.load(discrete_file_path)
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set material phases
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    # Set material phases properties
    material_properties = dict()
    material_properties['1'] = {'E': 100.0e6, 'v': 0.30}
    material_properties['2'] = {'E': 500.0e6, 'v': 0.19}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale strain loading
    if strain_formulation == 'infinitesimal':
        # Set macroscale infinitesimal strain tensor
        if n_dim == 2:
            mac_strain = np.array([[5.000e-3, 0.000e+0],
                                   [0.000e+0, 0.000e+0]])
        else:
            mac_strain = np.array([[5.000e-3, 0.000e+0, 0.000+0],
                                   [0.000e+0, 0.000e+0, 0.000+0],
                                   [0.000e+0, 0.000e+0, 0.000+0]])
        mac_strain_inf = mac_strain
    else:
        # Set macroscale deformation gradient tensor
        if n_dim == 2:
            mac_strain = np.array([[1.005e+0, 0.000e+0],
                                   [0.000e+0, 1.000e+0]])
        else:
            mac_strain = np.array([[1.005e+0, 0.000e+0, 0.000+0],
                                   [0.000e+0, 1.000e+0, 0.000+0],
                                   [0.000e+0, 0.000e+0, 1.000+0]])
        # Compute gradient of displacement field
        disp_grad = mac_strain - np.eye(n_dim)
        # Compute infinitesimal strain tensor
        mac_strain_inf = 0.5*(disp_grad + np.transpose(disp_grad))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Old version: FFT-based homogenization basic scheme:
    strain_vox_old = ffthombasicscheme(
        {'problem_type': problem_type, 'n_dim': n_dim, 'comp_order_sym': comp_order_sym},
        {'regular_grid': regular_grid, 'rve_dims': rve_dims, 'n_voxels_dims': n_voxels_dims},
        {'material_phases': material_phases, 'material_properties': material_properties},
        mac_strain_inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # New version: FFT-based homogenization basic scheme:
    homogenization_method = FFTBasicScheme(strain_formulation, problem_type, rve_dims,
                                           n_voxels_dims, regular_grid, material_phases,
                                           material_properties)
    strain_vox_new = homogenization_method.compute_rve_local_response(mac_strain,
                                                                      verbose=True)
    # --------------------------------------------------------------------------------------
    # MATPLOTLIB OUTPUT
    # Import modules
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Set plot font
    plt.rc('text',usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage[widespace]{fourier}'
                                  r'\usepackage{amsmath}'
                                  r'\usepackage{amssymb}'
                                  r'\usepackage{bm}'
                                  r'\usepackage{physics}'
                                  r'\usepackage[clock]{ifsym}')
    # Set figure
    figure1 = plt.figure()
    axes = figure1.add_subplot(111)
    #axes.set_title('2D Particle-reinforced Composite')
    figure1.set_figheight(8, forward=True)
    figure1.set_figwidth(8, forward=True)
    # Set figure axes
    axes.set_frame_on(True)
    axes.set_xlabel('$x_{1}$', fontsize=12, labelpad=10)
    axes.set_ylabel('$\\varepsilon_{11}$', fontsize=12, labelpad=8)
    axes.ticklabel_format(axis='x', style='plain', scilimits=(3,4))
    axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes.xaxis.set_minor_formatter(ticker.NullFormatter())
    axes.yaxis.set_major_locator(ticker.AutoLocator())
    axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes.tick_params(which='major', width=1.0, length=10, labelcolor='0.0', labelsize=12)
    axes.tick_params(which='minor', width=1.0, length=5, labelsize=12)
    axes.grid(linestyle='-',linewidth=0.5, color='0.5',zorder=15)
    axes.set_xlim(xmin=0.0, xmax=1.0)
    #axes.set_ylim(ymin=0, ymax=0.008)
    # Set plot data
    x = np.array(range(n_voxels_dims[0]))*(rve_dims[0]/n_voxels_dims[0])
    range_init = int(np.nonzero(x >= 0.0)[0][0])
    range_end = int(np.nonzero(x <= 1.0)[-1][-1]) + 1
    x = x[range_init:range_end]
    if n_dim == 2:
        y1 = [strain_vox_old['11'][(i, int(n_voxels_dims[1]/2))]
             for i in range(range_init, range_end)]
        y2 = [strain_vox_new['11'][(i, int(n_voxels_dims[1]/2))]
             for i in range(range_init, range_end)]
    else:
        y1 = [strain_vox_old['11'][(i, int(n_voxels_dims[1]/2), int(n_voxels_dims[2]/2))]
             for i in range(range_init, range_end)]
        y2 = [strain_vox_new['11'][(i, int(n_voxels_dims[1]/2), int(n_voxels_dims[2]/2))]
             for i in range(range_init, range_end)]
    # Plot data
    axes.plot(x, y1, label='original', color='r', linewidth=2, linestyle='-', clip_on=False)
    axes.plot(x, y2, label='OOP', color='b', linewidth=1, linestyle='--', clip_on=False)
    # Legend
    axes.legend(loc='center',ncol=3, numpoints=1, frameon=True, fancybox=True,
             facecolor='inherit', edgecolor='inherit', fontsize=10, framealpha=1.0,
             bbox_to_anchor=(0, 1.1, 1.0, 0.1), borderaxespad=0.0, markerscale=0.0)
    # Save plot
    figure1.set_figheight(3.6, forward=False)
    figure1.set_figwidth(3.6, forward=False)
    fig_path = working_dir + strain_formulation + '_strain_plot.pdf'
    figure1.savefig(fig_path, transparent=False, dpi=300, bbox_inches='tight')
    # --------------------------------------------------------------------------------------
    # VTK OUTPUT
    # Set clustering VTK file path
    vtk_file_path = working_dir + strain_formulation + '_fft_local_fields.vti'
    # Open clustering VTK file (append mode)
    if os.path.isfile(vtk_file_path):
        os.remove(vtk_file_path)
    vtk_file = open(vtk_file_path, 'a')
    # Instantiate VTK XML generator
    xml = XMLGenerator('ImageData', '1.0', 'LittleEndian', 'ascii', 'SinglePrecision',
                       'UInt64')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file header
    xml.write_file_header(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    dataset_parameters, piece_parameters = \
        VTKOutput._set_image_data_parameters(rve_dims, n_voxels_dims)
    xml.write_open_dataset_elem(vtk_file, dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    xml.write_open_dataset_piece(vtk_file, piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    xml.write_open_cell_data(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain/stress components order according to problem strain formulation
    if strain_formulation == 'infinitesimal':
        # Infinitesimal strain tensor and Cauchy stress tensor
        comp_order = comp_order_sym
    elif strain_formulation == 'finite':
        # Deformation gradient and First Piola-Kirchhoff stress tensor
        comp_order = comp_order_nsym
    # Loop over strain components
    for comp in comp_order:
        # Set data name
        data_name = 'strain_' + comp
        # Set data
        data_list = list(strain_vox_new[comp].flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name': data_name, 'format': 'ascii',
                           'RangeMin': min_val, 'RangeMax': max_val}
        # Write cell data array
        xml.write_cell_data_array(vtk_file, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element cell data
    xml.write_close_cell_data(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    xml.write_close_dataset_piece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    xml.write_close_dataset_elem(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    xml.write_file_footer(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close clustering VTK file
    vtk_file.close()
