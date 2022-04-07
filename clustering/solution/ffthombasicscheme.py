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
# Scientific computation
import scipy.linalg
# Warnings management
import warnings
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
    _conv_criterion : str, {'stress_div', 'avg_stress_norm'}
        Convergence criterion: 'stress_div' is the original convergence criterion based on
        the evaluation of the divergence of the stress tensor; 'avg_stress_norm' is based on
        the iterative change of the average stress norm.
    _conv_tol : float
        Convergence tolerance.
    _max_subinc_level : int
        Maximum level of macroscale loading subincrementation.
    _max_cinc_cuts : int
        Maximum number of consecutive macroscale loading increment cuts.
    _hom_stress_strain : 2darray
        Homogenized stress-strain material response. The homogenized strain and homogenized
        stress tensor components of the i-th loading increment are stored columnwise
        in the i-th row, sorted respectively. Infinitesimal strains: Cauchy stress tensor -
        infinitesimal strains tensor. Finite strains: first Piola-Kirchhoff stress tensor -
        deformation gradient.
    '''
    def __init__(self, strain_formulation, problem_type, rve_dims, n_voxels_dims,
                 regular_grid, material_phases, material_phases_properties):
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
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to each material
            phase (key, str).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_phases_properties = material_phases_properties
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Set maximum number of iterations
        self._max_n_iterations = 100
        # Set convergence criterion and tolerance
        self._conv_criterion = 'avg_stress_norm'
        self._conv_tol = 1e-4
        # Set macroscale loading subincrementation parameters
        self._max_subinc_level = 5
        self._max_cinc_cuts = 5
        # Initialize homogenized strain-stress response
        self._hom_stress_strain = np.zeros((1, 2*n_dim**2))
        if self._strain_formulation == 'finite':
            self._hom_stress_strain[0, 0] = 1.0
    # --------------------------------------------------------------------------------------
    def compute_rve_local_response(self, mac_strain_id, mac_strain, verbose=False):
        '''Compute RVE local elastic strain response.

        Compute the local response of the material's representative volume element (RVE)
        subjected to a given macroscale strain loading: infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains). It is assumed that
        the RVE is spatially discretized in a regular grid of voxels.

        Parameters
        ----------
        mac_strain_id : int
            Macroscale strain second-order tensor identifier.
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
        if verbose:
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized strain-stress response
        self._hom_stress_strain = np.zeros((1, 2*self._n_dim**2))
        if self._strain_formulation == 'finite':
            self._hom_stress_strain[0, 0] = 1.0
        #
        #                                                 Material phases elasticity tensors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elastic properties-related optimized variables
        evar1 = np.zeros(tuple(self._n_voxels_dims))
        evar2 = np.zeros(tuple(self._n_voxels_dims))
        for mat_phase in self._material_phases:
            # Get material phase elastic properties
            E = self._material_phases_properties[mat_phase]['E']
            v = self._material_phases_properties[mat_phase]['v']
            # Build optimized variables
            evar1[self._regular_grid == int(mat_phase)] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
            evar2[self._regular_grid == int(mat_phase)] = np.multiply(2,E/(2.0*(1.0 + v)))
        evar3 = np.add(evar1, evar2)
        #
        #                                              Reference material elastic properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material elastic properties as the mean between the minimum and
        # maximum values existent among the microstructure's material phases
        # (proposed in Moulinec, H. and Suquet, P., 1998)
        mat_prop_ref = dict()
        mat_prop_ref['E'] = \
            0.5*(min([self._material_phases_properties[phase]['E']
                 for phase in self._material_phases]) + \
                 max([self._material_phases_properties[phase]['E']
                 for phase in self._material_phases]))
        mat_prop_ref['v'] = \
            0.5*(min([self._material_phases_properties[phase]['v']
                 for phase in self._material_phases]) + \
                 max([self._material_phases_properties[phase]['v']
                 for phase in self._material_phases]))
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
            # Symmetrized isotropic reference material elasticity tensor
            c1 = 1.0/(4.0*miu_ref)
            c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
        else:
            # Non-symmetrized isotropic reference material elasticity tensor
            c1 = 1.0/(2.0*miu_ref)
            c2 = lam_ref/(2.0*miu_ref*(lam_ref + 2.0*miu_ref))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Green operator material independent terms
        gop_1_dft_vox, gop_2_dft_vox, _ = \
            citop.gop_material_independent_terms(self._strain_formulation,
                self._problem_type, self._rve_dims, self._n_voxels_dims)
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
        # Set number of increments
        if self._strain_formulation == 'infinitesimal':
            n_incs = 1
        else:
            n_incs = 1
        # Set incremental load factors
        inc_lfacts = n_incs*[1.0/n_incs,]
        # Initialize macroscale strain incrementer
        mac_strain_incrementer = MacroscaleStrainIncrementer(self._strain_formulation,
            self._problem_type, mac_strain_total, inc_lfacts=inc_lfacts,
                max_subinc_level=self._max_subinc_level, max_cinc_cuts=self._max_cinc_cuts)
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
            if self._conv_criterion == 'avg_stress_norm':
                # Compute initial guess average stress norm
                avg_stress_norm = self._compute_avg_state_vox(stress_vox)
                # Initialize last iteration average stress norm
                avg_stress_norm_itold = 0.0
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
                elif self._conv_criterion == 'avg_stress_norm':
                    # Compute discrete error
                    discrete_error = \
                        abs(avg_stress_norm - avg_stress_norm_itold)/avg_stress_norm
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
                elif iter == self._max_n_iterations or not np.isfinite(discrete_error):
                    # Set increment cut output
                    if not np.isfinite(discrete_error):
                        cut_msg = 'Solution diverged.'
                    else:
                        cut_msg = 'Maximum number of iterations reached without ' + \
                                  'convergence.'
                    # Raise macroscale increment cut procedure
                    is_inc_cut = True
                    # Display increment cut (maximum number of iterations)
                    type(self)._display_increment_cut(cut_msg)
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
                    freq_0_idx = self._n_dim*(0,)
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
                if self._conv_criterion == 'avg_stress_norm':
                    # Update last iteration average stress norm
                    avg_stress_norm_itold = avg_stress_norm
                    # Compute average stress norm
                    avg_stress_norm = self._compute_avg_state_vox(stress_vox)
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
            # Append to homogenized strain-stress response
            self._hom_stress_strain = np.vstack([self._hom_stress_strain,
                np.concatenate((hom_strain.flatten('F'), hom_stress.flatten('F')))])
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
                        mat_log_strain = 0.5*top.isotropic_tensor('log',
                            np.matmul(np.transpose(def_gradient), def_gradient))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over material logarithmic strain tensor components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store material logarithmic strain tensor
                            strain_vox[comp][voxel] = mat_log_strain[so_idx]
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
    def _elastic_constitutive_model(self, strain_vox, evar1, evar2, evar3,
                                    finite_strains_model='stvenant-kirchhoff',
                                    is_optimized=True):
        '''Material elastic or hyperelastic constitutive model.

        Infinitesimal strains: standard isotropic linear elastic constitutive model
        Finite strains: Hencky hyperelastic isotropic constitutive model
                        Saint Venant-Kirchhoff hyperlastic isotropic constitutive model

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
        finite_strains_model : bool, {'hencky', 'stvenant-kirchhoff'}, default='hencky'
            Finite strains hyperelastic isotropic constitutive model.
        is_optimized : bool
            Optimization flag (minimizes loops over spatial discretization voxels).

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
        # Check hyperlastic constitutive model
        if self._strain_formulation == 'finite':
            if finite_strains_model not in ('hencky', 'stvenant-kirchhoff'):
                raise RuntimeError('Unknown hyperelastic constitutive model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute finite strains strain tensor
        if self._strain_formulation == 'finite':
            # Save deformation gradient
            def_gradient_vox = copy.deepcopy(strain_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize symmetric finite strains strain tensor
            finite_sym_strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                                     for comp in self._comp_order_sym}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute finite strains strain tensor
            if is_optimized:
                # Compute finite strains strain tensor according to hyperelastic
                # constitutive model
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Compute voxelwise material Green-Lagrange strain tensor
                    if self._n_dim == 2:
                        finite_sym_strain_vox['11'] = \
                            0.5*(np.add(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['11']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['21'])) - 1.0)
                        finite_sym_strain_vox['22'] = \
                            0.5*(np.add(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['12']),
                                        np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['22'])) - 1.0)
                        finite_sym_strain_vox['12'] = \
                            0.5*np.add(np.multiply(def_gradient_vox['11'],
                                                   def_gradient_vox['12']),
                                       np.multiply(def_gradient_vox['21'],
                                                   def_gradient_vox['22']))
                    else:
                        finite_sym_strain_vox['11'] = \
                            0.5*(np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                           def_gradient_vox['11']),
                                               np.multiply(def_gradient_vox['21'],
                                                           def_gradient_vox['21'])),
                                        np.multiply(def_gradient_vox['31'],
                                                    def_gradient_vox['31'])) - 1.0)
                        finite_sym_strain_vox['12'] = \
                            0.5*np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                          def_gradient_vox['12']),
                                              np.multiply(def_gradient_vox['21'],
                                                          def_gradient_vox['22'])),
                                       np.multiply(def_gradient_vox['31'],
                                                   def_gradient_vox['32']))
                        finite_sym_strain_vox['13'] = \
                            0.5*np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                          def_gradient_vox['13']),
                                              np.multiply(def_gradient_vox['21'],
                                                          def_gradient_vox['23'])),
                                       np.multiply(def_gradient_vox['31'],
                                                   def_gradient_vox['33']))
                        finite_sym_strain_vox['22'] = \
                            0.5*(np.add(np.add(np.multiply(def_gradient_vox['12'],
                                                           def_gradient_vox['12']),
                                               np.multiply(def_gradient_vox['22'],
                                                           def_gradient_vox['22'])),
                                        np.multiply(def_gradient_vox['32'],
                                                    def_gradient_vox['32'])) - 1.0)
                        finite_sym_strain_vox['23'] = \
                            0.5*np.add(np.add(np.multiply(def_gradient_vox['12'],
                                                          def_gradient_vox['13']),
                                              np.multiply(def_gradient_vox['22'],
                                                          def_gradient_vox['23'])),
                                       np.multiply(def_gradient_vox['32'],
                                                   def_gradient_vox['33']))
                        finite_sym_strain_vox['33'] = \
                            0.5*(np.add(np.add(np.multiply(def_gradient_vox['13'],
                                                           def_gradient_vox['13']),
                                               np.multiply(def_gradient_vox['23'],
                                                           def_gradient_vox['23'])),
                                        np.multiply(def_gradient_vox['33'],
                                                    def_gradient_vox['33'])) - 1.0)
                else:
                    # Compute voxelwise left Cauchy-Green strain tensor
                    if self._n_dim == 2:
                        ftfvar11 = np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['11']),
                                          np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['12']))
                        ftfvar22 = np.add(np.multiply(def_gradient_vox['21'],
                                                      def_gradient_vox['21']),
                                          np.multiply(def_gradient_vox['22'],
                                                      def_gradient_vox['22']))
                        ftfvar12 = np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['21']),
                                          np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['22']))
                    else:
                        ftfvar11 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                             def_gradient_vox['11']),
                                                 np.multiply(def_gradient_vox['12'],
                                                             def_gradient_vox['12'])),
                                          np.multiply(def_gradient_vox['13'],
                                                      def_gradient_vox['13']))
                        ftfvar12 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                             def_gradient_vox['21']),
                                                 np.multiply(def_gradient_vox['12'],
                                                             def_gradient_vox['22'])),
                                          np.multiply(def_gradient_vox['13'],
                                                      def_gradient_vox['23']))
                        ftfvar13 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                             def_gradient_vox['31']),
                                                 np.multiply(def_gradient_vox['12'],
                                                             def_gradient_vox['32'])),
                                          np.multiply(def_gradient_vox['13'],
                                                      def_gradient_vox['33']))
                        ftfvar22 = np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                             def_gradient_vox['21']),
                                                 np.multiply(def_gradient_vox['22'],
                                                             def_gradient_vox['22'])),
                                          np.multiply(def_gradient_vox['23'],
                                                      def_gradient_vox['23']))
                        ftfvar23 = np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                             def_gradient_vox['31']),
                                                 np.multiply(def_gradient_vox['22'],
                                                             def_gradient_vox['32'])),
                                          np.multiply(def_gradient_vox['23'],
                                                      def_gradient_vox['33']))
                        ftfvar33 = np.add(np.add(np.multiply(def_gradient_vox['31'],
                                                             def_gradient_vox['31']),
                                                 np.multiply(def_gradient_vox['32'],
                                                             def_gradient_vox['32'])),
                                          np.multiply(def_gradient_vox['33'],
                                                      def_gradient_vox['33']))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over voxels
                    for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                        # Build left Cauchy-Green strain tensor
                        if self._n_dim == 2:
                            left_cauchy_green = np.reshape(
                                np.array([ftfvar11[voxel], ftfvar12[voxel],
                                          ftfvar12[voxel], ftfvar22[voxel]]),
                                (self._n_dim, self._n_dim), 'F')
                        else:
                            left_cauchy_green = np.reshape(
                                np.array([ftfvar11[voxel], ftfvar12[voxel], ftfvar13[voxel],
                                          ftfvar12[voxel], ftfvar22[voxel], ftfvar23[voxel],
                                          ftfvar13[voxel], ftfvar23[voxel], ftfvar33[voxel]]
                                         ), (self._n_dim, self._n_dim), 'F')
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute spatial logarithmic strain tensor
                        with warnings.catch_warnings():
                            # Supress warnings
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            # Compute spatial logarithmic strain tensor
                            spatial_log_strain = 0.5*top.isotropic_tensor('log',
                                                                          left_cauchy_green)
                            if np.any(np.logical_not(np.isfinite(spatial_log_strain))):
                                spatial_log_strain = \
                                    0.5*scipy.linalg.logm(left_cauchy_green)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over spatial logarithmic strain tensor components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store spatial logarithmic strain tensor
                            finite_sym_strain_vox[comp][voxel] = spatial_log_strain[so_idx]
            else:
                # Compute finite strains strain tensor according to hyperelastic
                # constitutive model
                for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                    # Initialize deformation gradient
                    def_gradient = np.zeros((self._n_dim, self._n_dim))
                    # Loop over deformation gradient components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel deformation gradient component
                        def_gradient[so_idx] = strain_vox[comp][voxel]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute finite strains strain tensor according to hyperelastic
                    # constitutive model
                    if finite_strains_model == 'stvenant-kirchhoff':
                        # Compute material Green-Lagrange strain tensor
                        mat_green_lagr_strain = \
                            0.5*(np.matmul(np.transpose(def_gradient), def_gradient) -
                                 np.eye(self._n_dim))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over symmetric strain components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store material Green-Lagrange strain tensor
                            finite_sym_strain_vox[comp][voxel] = \
                                mat_green_lagr_strain[so_idx]
                    else:
                        # Compute spatial logarithmic strain tensor
                        with warnings.catch_warnings():
                            # Supress warnings
                            warnings.simplefilter('ignore', category=RuntimeWarning)
                            # Compute spatial logarithmic strain tensor
                            spatial_log_strain = 0.5*top.isotropic_tensor('log',
                                np.matmul(def_gradient, np.transpose(def_gradient)))
                            if np.any(np.logical_not(np.isfinite(spatial_log_strain))):
                                spatial_log_strain = 0.5*scipy.linalg.logm(
                                    np.matmul(def_gradient, np.transpose(def_gradient)))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over symmetric strain components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store spatial logarithmic strain tensor
                            finite_sym_strain_vox[comp][voxel] = spatial_log_strain[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store symmetric finite strains strain tensor
            strain_vox = finite_sym_strain_vox
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
            first_piola_stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                                      for comp in self._comp_order_nsym}
            # Compute First Piola-Kirchhoff stress tensor
            if is_optimized:
                # Compute First Piola-Kirchhoff stress tensor according to hyperelastic
                # constitutive model
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Compute voxelwise First Piola-Kirchhoff stress tensor
                    if self._n_dim == 2:
                        first_piola_stress_vox['11'] = \
                            np.add(np.multiply(def_gradient_vox['11'], stress_vox['11']),
                                   np.multiply(def_gradient_vox['12'], stress_vox['12']))
                        first_piola_stress_vox['21'] = \
                            np.add(np.multiply(def_gradient_vox['21'], stress_vox['11']),
                                   np.multiply(def_gradient_vox['22'], stress_vox['12']))
                        first_piola_stress_vox['12'] = \
                            np.add(np.multiply(def_gradient_vox['11'], stress_vox['12']),
                                   np.multiply(def_gradient_vox['12'], stress_vox['22']))
                        first_piola_stress_vox['22'] = \
                            np.add(np.multiply(def_gradient_vox['21'], stress_vox['12']),
                                   np.multiply(def_gradient_vox['22'], stress_vox['22']))
                    else:
                        first_piola_stress_vox['11'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'], stress_vox['11']),
                                   np.multiply(def_gradient_vox['12'], stress_vox['12'])),
                            np.multiply(def_gradient_vox['13'], stress_vox['13']))
                        first_piola_stress_vox['21'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'], stress_vox['11']),
                                   np.multiply(def_gradient_vox['22'], stress_vox['12'])),
                            np.multiply(def_gradient_vox['23'], stress_vox['13']))
                        first_piola_stress_vox['31'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'], stress_vox['11']),
                                   np.multiply(def_gradient_vox['32'], stress_vox['12'])),
                            np.multiply(def_gradient_vox['33'], stress_vox['13']))
                        first_piola_stress_vox['12'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'], stress_vox['12']),
                                   np.multiply(def_gradient_vox['12'], stress_vox['22'])),
                            np.multiply(def_gradient_vox['13'], stress_vox['23']))
                        first_piola_stress_vox['22'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'], stress_vox['12']),
                                   np.multiply(def_gradient_vox['22'], stress_vox['22'])),
                            np.multiply(def_gradient_vox['23'], stress_vox['23']))
                        first_piola_stress_vox['32'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'], stress_vox['12']),
                                   np.multiply(def_gradient_vox['32'], stress_vox['22'])),
                            np.multiply(def_gradient_vox['33'], stress_vox['23']))
                        first_piola_stress_vox['13'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'], stress_vox['13']),
                                   np.multiply(def_gradient_vox['12'], stress_vox['23'])),
                            np.multiply(def_gradient_vox['13'], stress_vox['33']))
                        first_piola_stress_vox['23'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'], stress_vox['13']),
                                   np.multiply(def_gradient_vox['22'], stress_vox['23'])),
                            np.multiply(def_gradient_vox['23'], stress_vox['33']))
                        first_piola_stress_vox['33'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'], stress_vox['13']),
                                   np.multiply(def_gradient_vox['32'], stress_vox['23'])),
                            np.multiply(def_gradient_vox['33'], stress_vox['33']))
                else:
                    if self._n_dim == 2:
                        # Compute voxelwise determinant of deformation gradient
                        jvar = np.reciprocal(
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['12'])))
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox['11'] = np.multiply(jvar,
                            np.subtract(np.multiply(stress_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(stress_vox['12'],
                                                    def_gradient_vox['12'])))
                        first_piola_stress_vox['21'] = np.multiply(jvar,
                            np.subtract(np.multiply(stress_vox['12'],
                                                    def_gradient_vox['22']),
                                        np.multiply(stress_vox['22'],
                                                    def_gradient_vox['12'])))
                        first_piola_stress_vox['12'] = np.multiply(jvar,
                            np.subtract(np.multiply(stress_vox['12'],
                                                    def_gradient_vox['11']),
                                        np.multiply(stress_vox['11'],
                                                    def_gradient_vox['21'])))
                        first_piola_stress_vox['22'] = np.multiply(jvar,
                            np.subtract(np.multiply(stress_vox['22'],
                                                    def_gradient_vox['11']),
                                        np.multiply(stress_vox['12'],
                                                    def_gradient_vox['21'])))
                    else:
                        # Compute voxelwise determinant of deformation gradient
                        jvar = np.reciprocal(np.add(np.subtract(
                            np.multiply(def_gradient_vox['11'],
                                        np.subtract(np.multiply(def_gradient_vox['22'],
                                                                def_gradient_vox['33']),
                                                    np.multiply(def_gradient_vox['23'],
                                                                def_gradient_vox['32']))),
                            np.multiply(def_gradient_vox['12'],
                                        np.subtract(np.multiply(def_gradient_vox['21'],
                                                                def_gradient_vox['33']),
                                                    np.multiply(def_gradient_vox['23'],
                                                                def_gradient_vox['31'])))),
                            np.multiply(def_gradient_vox['13'],
                                        np.subtract(np.multiply(def_gradient_vox['21'],
                                                                def_gradient_vox['32']),
                                                    np.multiply(def_gradient_vox['22'],
                                                                def_gradient_vox['31'])))))
                        # Compute voxelwise transpose of inverse of deformationg gradient
                        fitvar11 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['33']),
                                        np.multiply(def_gradient_vox['23'],
                                                    def_gradient_vox['32'])))
                        fitvar21 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['32']),
                                        np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['33'])))
                        fitvar31 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['23']),
                                        np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['22'])))
                        fitvar12 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['23'],
                                                    def_gradient_vox['31']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['33'])))
                        fitvar22 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['33']),
                                        np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['31'])))
                        fitvar32 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['21']),
                                        np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['23'])))
                        fitvar13 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['32']),
                                        np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['31'])))
                        fitvar23 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['31']),
                                        np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['32'])))
                        fitvar33 = np.multiply(jvar,
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['21'])))
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox['11'] = \
                            np.add(np.add(np.multiply(stress_vox['11'], fitvar11),
                                          np.multiply(stress_vox['12'], fitvar21)),
                                   np.multiply(stress_vox['13'], fitvar31))
                        first_piola_stress_vox['21'] = \
                            np.add(np.add(np.multiply(stress_vox['12'], fitvar11),
                                          np.multiply(stress_vox['22'], fitvar21)),
                                   np.multiply(stress_vox['23'], fitvar31))
                        first_piola_stress_vox['31'] = \
                            np.add(np.add(np.multiply(stress_vox['13'], fitvar11),
                                          np.multiply(stress_vox['23'], fitvar21)),
                                   np.multiply(stress_vox['33'], fitvar31))
                        first_piola_stress_vox['12'] = \
                            np.add(np.add(np.multiply(stress_vox['11'], fitvar12),
                                          np.multiply(stress_vox['12'], fitvar22)),
                                   np.multiply(stress_vox['13'], fitvar32))
                        first_piola_stress_vox['22'] = \
                            np.add(np.add(np.multiply(stress_vox['12'], fitvar12),
                                          np.multiply(stress_vox['22'], fitvar22)),
                                   np.multiply(stress_vox['23'], fitvar32))
                        first_piola_stress_vox['32'] = \
                            np.add(np.add(np.multiply(stress_vox['13'], fitvar12),
                                          np.multiply(stress_vox['23'], fitvar22)),
                                   np.multiply(stress_vox['33'], fitvar32))
                        first_piola_stress_vox['13'] = \
                            np.add(np.add(np.multiply(stress_vox['11'], fitvar13),
                                          np.multiply(stress_vox['12'], fitvar23)),
                                   np.multiply(stress_vox['13'], fitvar33))
                        first_piola_stress_vox['23'] = \
                            np.add(np.add(np.multiply(stress_vox['12'], fitvar13),
                                          np.multiply(stress_vox['22'], fitvar23)),
                                   np.multiply(stress_vox['23'], fitvar33))
                        first_piola_stress_vox['33'] = \
                            np.add(np.add(np.multiply(stress_vox['13'], fitvar13),
                                          np.multiply(stress_vox['23'], fitvar23)),
                                   np.multiply(stress_vox['33'], fitvar33))
            else:
                # Compute First Piola-Kirchhoff stress tensor according to hyperelastic
                # constitutive model
                for voxel in it.product(*[list(range(n)) for n in self._n_voxels_dims]):
                    # Initialize deformation gradient
                    def_gradient = np.zeros((self._n_dim, self._n_dim))
                    # Loop over deformation gradient components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel deformation gradient component
                        def_gradient[so_idx] = def_gradient_vox[comp][voxel]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute First Piola-Kirchhoff stress tensor
                    if finite_strains_model == 'stvenant-kirchhoff':
                        # Initialize Second Piola-Kirchhoff stress tensor
                        second_piola_stress = np.zeros((self._n_dim, self._n_dim))
                        # Loop over Second Piola-Kirchhoff stress tensor components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Get voxel Second Piola-Kirchhoff stress tensor component
                            second_piola_stress[so_idx] = stress_vox[comp][voxel]
                            if so_idx[0] != so_idx[1]:
                                second_piola_stress[so_idx[::-1]] = \
                                    second_piola_stress[so_idx]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress = np.matmul(def_gradient, second_piola_stress)
                    else:
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
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress = np.matmul(kirchhoff_stress,
                            np.transpose(np.linalg.inv(def_gradient)))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over First Piola-Kirchhoff stress tensor components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Store First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox[comp][voxel] = first_piola_stress[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set First Piola-Kirchhoff stress tensor
            stress_vox = first_piola_stress_vox
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
            comp_order = self._comp_order_nsym
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
    def get_hom_stress_strain(self):
        '''Get homogenized strain-stress material response.

        Returns
        -------
        _hom_stress_strain : 2darray
            Homogenized stress-strain material response. The homogenized strain and
            homogenized stress tensor components of the i-th loading increment are stored
            columnwise in the i-th row, sorted respectively. Infinitesimal strains: Cauchy
            stress tensor - infinitesimal strains tensor. Finite strains: first
            Piola-Kirchhoff stress tensor - deformation gradient.
        '''
        return copy.deepcopy(self._hom_stress_strain)
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _display_greetings():
        '''Output homogenization-based multi-scale DNS method method greetings.'''
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
    def _display_increment_cut(cut_msg=''):
        '''Output increment cut data.

        Parameters
        ----------
        cut_msg : str
            Increment cut output message.
        '''
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, _, indent, asterisk_line = display_features[0:4]
        _, _ = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = ()
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
#
#                                                                     Validation (temporary)
# ==========================================================================================
if __name__ == '__main__':
    # Insert this at import modules section
    #p = os.path.abspath('/home/bernardoferreira/Documents/CRATE/src')
    #sys.path.insert(1, p)
    import pickle
    import tensor.matrixoperations as mop
    from ioput.vtkoutput import XMLGenerator, VTKOutput
    from clustering.clusteringdata import def_gradient_from_log_strain
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set working directory
    working_dir = '/home/bernardoferreira/Documents/CRATE/developments/finite_strains/2d/'
    # Set plots output flag
    is_output_plots = False
    # Set VTK output flag
    is_vtk_output = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem strain formulation
    strain_formulation = 'finite'
    # Set problem type
    problem_type = 1
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set RVE spatial discretization file absolute path
    if problem_type == 1:
        discrete_file_path = working_dir + 'Disk_50_0.3_400_400.rgmsh.npy'
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
        mac_strain_is = mac_strain
    else:
        # Set macroscale deformation gradient tensor
        if n_dim == 2:
            mac_strain = np.eye(2) + np.array([[0.300e+0,  0.000e+0],
                                               [0.000e+0,  0.000e+0]])*1.0
        else:
            mac_strain = np.eye(3) + np.array([[0.100e+0, -0.100e+0, 0.050+0],
                                               [0.050e+0, -0.050e+0, 0.100+0],
                                               [-0.100e+0, 0.050e+0, 0.050+0]])
        # Compute gradient of displacement field
        disp_grad = mac_strain - np.eye(n_dim)
        # Compute infinitesimal strain tensor
        mac_strain_is = 0.5*(disp_grad + np.transpose(disp_grad))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # Strain concentration tensor study
    # ---------------------------------
    #
    # Define method to compute strain concentration tensor at a given voxel
    def compute_sct(strain_formulation, n_dim, n_voxels_dims, comp_order_sym, target_voxel,
                    strain_magnitude_factor=1.0):
        '''Compute strain concentration tensor at target voxel.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        n_dim : int
            Problem number of spatial dimensions.
        n_voxels_dims : list
            Number of voxels in each dimension of the regular grid (spatial discretization
            of the RVE).
        comp_order_sym : list
            Strain/Stress components symmetric order.
        target_voxel : tuple
            Voxel where the strain concentration tensor is computed.
        strain_magnitude_factor : float, default=1.0
            Macroscale strain magnitude factor.

        Returns
        -------
        sct : 2darray
            Strain concentration tensor at target voxel.
        mac_strains : list
            Macroscale strain (orthogonal) loadings.
        local_strains : list
            Local strain tensor (target voxel) associated to each macroscale strain loading.
        '''
        # Set macroscale strain loadings required to compute strain concentration tensor
        mac_strains = []
        for i in range(len(comp_order_sym)):
            # Get strain component and associated indexes
            comp = comp_order_sym[i]
            so_idx = tuple([int(x) - 1 for x in list(comp_order_sym[i])])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set orthogonal infinitesimal strain tensor (infinitesimal strains) or material
            # logarithmic strain tensor (finite strains) according with Kelvin notation
            mac_strain = np.zeros((n_dim, n_dim))
            mac_strain[so_idx] = \
                strain_magnitude_factor*(1.0/mop.kelvin_factor(i, comp_order_sym))*1.0
            if comp[0] != comp[1]:
                mac_strain[so_idx[::-1]] = mac_strain[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store macroscale strain loading
            mac_strains.append(mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize FFT-based homogenization basic scheme
        homogenization_method = FFTBasicScheme(strain_formulation=strain_formulation,
            problem_type=problem_type, rve_dims=rve_dims, n_voxels_dims=n_voxels_dims,
                regular_grid=regular_grid, material_phases=material_phases,
                    material_phases_properties=material_properties)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain concentration tensor
        sct = np.zeros((len(comp_order_sym), len(comp_order_sym)))
        # Initialize local strain tensor associated to each macroscale strain loading
        local_strains = []
        # Loop over macroscale strain loadings
        for j in range(len(mac_strains)):
            # Get macroscale strain loading
            mac_strain = mac_strains[j]
            # Get Kelvin factor associated with macroscale strain loading strain component
            kf_j = mop.kelvin_factor(j, comp_order_sym)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute deformation gradient associated to the material logarithmic strain
            # tensor
            if strain_formulation == 'finite':
                mac_strain_imposed = def_gradient_from_log_strain(mac_strain)
            else:
                mac_strain_imposed = mac_strain
            # Compute local strain field
            strain_vox = \
                homogenization_method.compute_rve_local_response(j, mac_strain_imposed,
                                                                 verbose=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize local strain tensor
            local_strain = np.zeros(len(comp_order_sym))
            # Loop over strain components
            for i in range(len(comp_order_sym)):
                # Get strain component
                comp = comp_order_sym[i]
                # Get Kelvin factor associated with strain component
                kf_i = mop.kelvin_factor(i, comp_order_sym)
                # Assemble fourth-order elastic strain concentration tensor component
                # (accounting for the Kelvin notation coefficients)
                sct[i, j] = kf_i*strain_vox[comp][target_voxel]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove Kelvin coefficient
                sct[i, j] = (1.0/kf_j)*(1.0/kf_i)*sct[i, j]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Build local strain tensor (validation purposes only) (Kelvin notation)
                local_strain[i] = kf_i*strain_vox[comp][target_voxel]
            # Store local strain tensor (validation purposes only)
            local_strains.append(local_strain)
        # Normalize strain concentration tensor
        sct = (1.0/strain_magnitude_factor)*sct
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain concentration tensor (Kelvin notation)
        sct_kelvin = copy.deepcopy(sct)
        # Loop over macroscale strain loadings
        for j in range(len(mac_strains)):
            # Get Kelvin factor associated with macroscale strain loading strain component
            kf_j = mop.kelvin_factor(j, comp_order_sym)
            # Get macroscale strain loading
            mac_strain = mac_strains[j]
            # Build macroscale strain vector
            mac_strain_vec = np.zeros(len(comp_order_sym))
            for k in range(len(comp_order_sym)):
                # Get Kelvin factor associated with strain component
                kf_k = mop.kelvin_factor(k, comp_order_sym)
                # Get strain component and associated indexes
                comp = comp_order_sym[k]
                so_idx = tuple([int(x) - 1 for x in list(comp)])
                # Assemble macroscale strain vector (Kelvin notation)
                mac_strain_vec[k] = kf_k*mac_strain[so_idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add Kelvin coefficients to strain concentration tensor
                sct_kelvin[k, j] = kf_j*kf_k*sct_kelvin[k, j]
            # Compute local strain tensor from strain concentration tensor (Kelvin notation)
            local_strain = np.matmul(sct_kelvin, mac_strain_vec)
            # Validate strain concentration tensor computation
            if not np.allclose(local_strain, local_strains[j]):
                raise RuntimeError('Error in SCT computation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return sct, mac_strains, local_strains
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain concentration tensor data file path
    sct_data_path = working_dir + 'sca_data.dat'
    # Set pickle flag
    is_pickle_sct = True
    # Set voxels under analysis
    target_voxels = [(200, 375), (375, 260), (300, 100), (50, 275)]
    # Set macroscale strain magnitude factors
    strain_magnitude_factors = [1.0e-4, 3.1628e-4, 1.0e-3, 3.1628e-3, 1.0e-2, 3.1628e-2,
                                1.0e-1]
    # Build or recover strain concentration data array
    if is_pickle_sct:
        # Get strain concentration data array
        with open(sct_data_path, 'rb') as sct_file:
            data_array = pickle.load(sct_file)
    else:
        # Initialize strain concentration data array
        data_array = np.zeros((len(strain_magnitude_factors), 2*2*len(target_voxels)))
        # Loop over target voxels
        for j in range(len(target_voxels)):
            # Get target_voxel
            target_voxel = target_voxels[j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over macroscale strain magnitude factors
            for i in range(len(strain_magnitude_factors)):
                # Get macroscale strain magnitude factor
                strain_magnitude_factor = strain_magnitude_factors[i]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over
                for strain_formulation in ('infinitesimal', 'finite'):
                    # Compute strain concentration tensor
                    sct, mac_strains, local_strains = \
                        compute_sct(strain_formulation, n_dim, n_voxels_dims,
                                    comp_order_sym, target_voxel,
                                    strain_magnitude_factor=strain_magnitude_factor)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute norm of strain concentration tensor
                    sct_norm = np.linalg.norm(sct)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble to plot data array
                    if strain_formulation == 'infinitesimal':
                        data_array[i, 2*j] = strain_magnitude_factor
                        data_array[i, 2*j + 1] = sct_norm
                    else:
                        offset = 2*len(target_voxels)
                        data_array[i, offset + 2*j] = strain_magnitude_factor
                        data_array[i, offset + 2*j + 1] = sct_norm
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dump strain concentration tensor data array
        with open(sct_data_path, 'wb') as sct_file:
            pickle.dump(data_array, sct_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output results:
    if False:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\nMacroscale strain and local strain:')
        # Initialize macroscale strain vectors
        mac_strains_vec = []
        # Loop over macroscale strain loadings
        for j in range(len(mac_strains)):
            # Get macroscale strain loading
            mac_strain = mac_strains[j]
            # Build macroscale strain vector
            mac_strain_vec = np.zeros(len(comp_order_sym))
            for k in range(len(comp_order_sym)):
                # Get strain component and associated indexes
                comp = comp_order_sym[k]
                so_idx = tuple([int(x) - 1 for x in list(comp)])
                # Assemble macroscale strain vector
                mac_strain_vec[k] = mac_strain[so_idx]
            mac_strains_vec.append(mac_strain_vec)
            # Get local strain tensor
            local_strain = local_strains[j]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Output macroscale strain loading and local strain tensor
            print('  mac_strain: ', mac_strain_vec, ' -> local_strain: ', local_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 'Strain concentration tensor ( normalization factor = ',
              strain_magnitude_factor, '):\n', sct)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n' + 'Strain concentration tensor norms:')
        print('  global norm   = ', np.linalg.norm(sct), '\n')
        # Loop over macroscale strain loadings
        for j in range(len(mac_strains)):
            print('  column ' + str(j) + ' norm = ', np.linalg.norm(sct[:, j]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:
        # New version: FFT-based homogenization basic scheme
        homogenization_method = FFTBasicScheme(strain_formulation=strain_formulation,
            problem_type=problem_type, rve_dims=rve_dims, n_voxels_dims=n_voxels_dims,
                regular_grid=regular_grid, material_phases=material_phases,
                    material_phases_properties=material_properties)
        # Compute local strain field
        strain_vox = homogenization_method.compute_rve_local_response(0, mac_strain,
                                                                      verbose=True)
        # Get homogenized stress-strain response
        hom_stress_strain = homogenization_method.get_hom_stress_strain()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute infinitesimal strains response for comparison with finite strains case
    is_is_fs_comparison = False
    if strain_formulation == 'finite' and is_is_fs_comparison:
        # New version: FFT-based homogenization basic scheme (infinitesimal strains)
        homogenization_method = FFTBasicScheme(strain_formulation='infinitesimal',
            problem_type=problem_type, rve_dims=rve_dims, n_voxels_dims=n_voxels_dims,
                regular_grid=regular_grid, material_phases=material_phases,
                    material_phases_properties=material_properties)
        # Compute local strain field
        strain_vox_is = homogenization_method.compute_rve_local_response(0, mac_strain_is,
                                                                         verbose=False)
        # Get homogenized stress-strain response
        hom_stress_strain_is = homogenization_method.get_hom_stress_strain()
    #
    # --------------------------------------------------------------------------------------
    # MATPLOTLIB OUTPUT
    #
    # Get NEWGATE line plots method
    def newgate_line_plots(plots_dir, fig_name, data_array, data_labels=None, x_label=None,
                           y_label=None, x_min=None, x_max=None, y_min=None, y_max=None,
                           is_marker=False, xticklabels=None):
        # Import modules
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import cycler
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute number of plot lines
        if data_labels != None:
            # Compute number of plot lines
            n_plot_lines = len(data_labels)
            # Check if plot data is conform with the provided data labels
            if data_array.shape[1] != 2*n_plot_lines:
                print('Abort: The plot data is not conform with the number of data labels.')
                sys.exit(1)
        else:
            # Check if plot data has valid format
            if data_array.shape[1]%2 != 0:
                print('Abort: The plot data must have an even number of columns (xi,yi).')
                sys.exit(1)
            # Compute number of plot lines
            n_plot_lines = int(data_array.shape[1]/2)
        #
        #                                                                     Data structure
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize data dictionary
        data_dict = dict()
        # Loop over plot lines
        for i in range(n_plot_lines):
            # Set plot line key
            data_key = 'data_' + str(i)
            # Initialize plot line dictionary
            data_dict[data_key] = dict()
            # Set plot line label
            if data_labels != None:
                data_dict[data_key]['label'] = data_labels[i]
            # Set plot line data
            data_dict[data_key]['x'] = data_array[:,2*i]
            data_dict[data_key]['y'] = data_array[:,2*i+1]
        #
        #                                                                   LaTeX formatting
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # LaTeX Fourier
        plt.rc('text',usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage[widespace]{fourier}'
                                      r'\usepackage{amsmath}'
                                      r'\usepackage{amssymb}'
                                      r'\usepackage{bm}'
                                      r'\usepackage{physics}'
                                      r'\usepackage[clock]{ifsym}')
        #
        #                                                              Default style cyclers
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set default color cycle
        cycler_color = cycler.cycler('color',['k','b','r','g'])
        # Set default linestyle cycle
        cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
        # Set default marker cycle
        cycler_marker = cycler.cycler('marker',['s','o','*'])
        # Get Paul Tol's color scheme cycler:
        # Set color scheme ('bright')
        color_list = ['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB']
        # Set colors to be used
        use_colors = {'1':[0], '2':[0,4], '3':[0,4,3], '4':[0,1,2,4], '5':[0,1,2,4,3],
                      '6':[0,1,2,4,3,5], '7':list(range(7))}
        # Set color scheme cycler
        if n_plot_lines > len(color_list):
            cycler_color = cycler.cycler('color',color_list)
        else:
            cycler_color = cycler.cycler('color',
                [color_list[i] for i in use_colors[str(n_plot_lines)]])


        cycler_linestyle = cycler.cycler('linestyle',4*['--','-'])
        cycler_color = cycler.cycler('color', [color_list[i] for i in [0,0,2,2,4,4,3,3]])


        # Set default cycler
        if is_marker:
            default_cycler = cycler_marker*cycler_linestyle*cycler_color
        else:
            default_cycler = cycler_linestyle*cycler_color

            default_cycler = cycler_linestyle+cycler_color
        plt.rc('axes', prop_cycle = default_cycler)
        #
        #                                                                    Figure and axes
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create figure
        figure = plt.figure()
        # Set figure size (inches) - stdout print purpose
        figure.set_figheight(8, forward=True)
        figure.set_figwidth(8, forward=True)
        # Create axes
        axes = figure.add_subplot(1,1,1)
        # Set axes patch visibility
        axes.set_frame_on(True)
        # Set axes labels
        if x_label != None:
            axes.set_xlabel(x_label, fontsize=12, labelpad=10)
        if y_label != None:
            axes.set_ylabel(y_label, fontsize=12, labelpad=10)
        #
        #                                                                               Axis
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Configure axis scales sources
        # 1. Log scale:
        #    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#
        #    matplotlib.axes.Axes.set_xscale
        #
        # Set scale option
        # 0. Default (linear x - linear y)
        # 1. log x - linear y
        # 2. linear x - log y
        # 3. log x - log y
        scale_option = 1
        #
        # Configure axes scales
        if scale_option == 1:
            # Set log scale in x axis
            axes.set_xscale('log')
            is_x_tick_format = False
            is_y_tick_format = True
        elif scale_option == 2:
            # Set log scale in y axis
            axes.set_yscale('log')
            is_x_tick_format = True
            is_y_tick_format = False
        elif scale_option == 3:
            # Set log scale in both x and y axes
            axes.set_xscale('log')
            axes.set_yscale('log')
            is_x_tick_format = False
            is_y_tick_format = False
        else:
            is_x_tick_format = True
            is_y_tick_format = True
        # Set tick formatting option
        # 0. Default formatting
        # 1. Scalar formatting
        # 2. User-defined formatting
        # 3. No tick labels
        tick_option = 1
        #
        # Configure ticks format
        if tick_option == 1:
            # Use a function which simply changes the default ScalarFormatter
            if is_x_tick_format:
                axes.ticklabel_format(axis='x', style='sci', scilimits=(3,4))
            if is_y_tick_format:
                axes.ticklabel_format(axis='y', style='sci', scilimits=(3,4))
        elif tick_option == 2:
            # Set user-defined functions to format tickers
            def intTickFormat(x,pos):
                return '${:2d}$'.format(int(x))
            def floatTickFormat(x,pos):
                return '${:3.1f}$'.format(x)
            def expTickFormat(x,pos):
                return '${:7.2e}$'.format(x)
            # Set ticks format
            if is_x_tick_format:
                axes.xaxis.set_major_formatter(ticker.FuncFormatter(floatTickFormat))
            if is_y_tick_format:
                axes.yaxis.set_major_formatter(ticker.FuncFormatter(floatTickFormat))
        elif tick_option == 3:
            # Set ticks format
            axes.xaxis.set_major_formatter(ticker.NullFormatter())
            axes.yaxis.set_major_formatter(ticker.NullFormatter())
        # Configure ticks locations
        if is_x_tick_format:
            if xticklabels != None:
                if len(xticklabels) != data_array.shape[0]:
                    raise RuntimeError('Invalid number of user-defined x-tick labels.')
                axes.set_xticks(data_array[:, 0])
                axes.set_xticklabels(['$' + str(int(x)) + '$' for x in xticklabels])
            else:
                axes.xaxis.set_major_locator(ticker.AutoLocator())
                axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if is_y_tick_format:
            axes.yaxis.set_major_locator(ticker.AutoLocator())
            axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        # Configure ticks appearance
        axes.tick_params(which='major', width=1.0, length=10, labelcolor='0.0',
                         labelsize=12)
        axes.tick_params(which='minor', width=1.0, length=5, labelsize=12)
        # Configure grid
        axes.grid(linestyle='-', linewidth=0.5, color='0.5', zorder=-20)
        #
        #                                                                 Special Parameters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set DNS line option
        is_DNS_line = False
        if is_DNS_line:
            # Set DNS line
            DNS_plot_line = n_plot_lines - 1
        else:
            DNS_plot_line = -1
        # Set markers at specific points
        is_special_markers = False
        if is_special_markers:
            # Adaptivity steps
            marker_symbol = 'd'
            marker_size = 5
            markers_on = {i: [] for i in range(n_plot_lines)}
            #markers_on[2] = [1,]
            markers_on[3] = [33,]
            markers_on[4] = [33,]
        #
        #                                                                               Plot
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set line width
        line_width = 2
        # Set marker type, size and frequency
        if is_special_markers:
            mark_every = markers_on
        else:
            marker_symbol = 'o'
            marker_size = 5
            mark_every = {i: 1 for i in range(n_plot_lines)}

            marker_symbols = int(n_plot_lines)*['v', '^']
        # Loop over plot lines
        for i in range(n_plot_lines):
            # Get plot line key
            data_key = 'data_' + str(i)
            # Get plot line label
            if data_labels != None:
                data_label = data_dict[data_key]['label']
            else:
                data_label = None
            # Get plot line data
            x = data_dict[data_key]['x']
            y = data_dict[data_key]['y']
            # Set plot line layer
            layer = 10 + i

            if np.mod(i, 2) == 0:
                layer = 10 - i
            else:
                layer = 10 + i

            if i == DNS_plot_line:
                axes.plot(x, y, label=data_label, linewidth=line_width, clip_on=False,
                                marker=marker_symbol, markersize=marker_size,
                                markevery=mark_every[i], zorder=layer,
                                color='k', linestyle='dashed')
                break

            # Plot line
            axes.plot(x, y, label=data_label, linewidth=line_width, clip_on=False,
                            marker=marker_symbols[i], markersize=marker_size,
                            markevery=mark_every[i], zorder=layer)
        #
        #                                                                        Axis limits
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set axis limits
        axes.set_xlim(xmin=x_min, xmax=x_max)
        axes.set_ylim(ymin=y_min, ymax=y_max)
        #
        #                                                                             Legend
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set plot legend
        if data_labels != None:
            axes.legend(loc='center',ncol=4, numpoints=2, frameon=True, fancybox=True,
                        facecolor='inherit', edgecolor='inherit', fontsize=10,
                        framealpha=1.0, bbox_to_anchor=(0, 1.1, 1.0, 0.1),
                        borderaxespad=0.0, markerscale=1.0, handlelength=2.5,
                        handletextpad=0.5)
        #
        #                                                                              Print
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Print plot to stdout
        #plt.show()
        #
        #                                                                        Figure file
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set figure size (inches)
        figure.set_figheight(3.6, forward=False)
        figure.set_figwidth(3.6, forward=False)
        # Save figure file in the desired format
        fig_path = plots_dir + fig_name
        # figure.savefig(fig_name + '.png', transparent=False, dpi=300, bbox_inches='tight')
        # figure.savefig(fig_name + '.eps', transparent=False, dpi=300, bbox_inches='tight')
        figure.savefig(fig_path + '.pdf', transparent=False, dpi=300, bbox_inches='tight')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output plots
    if is_output_plots:
        # Set plots directory
        plots_dir = working_dir
        # Set strain plot component
        plot_strain = '11'
        # Set stress plot component
        plot_stress = '11'
        # Get strain plot component (nonsymmetric) index
        plot_strain_idx = comp_order_nsym.index(plot_strain)
        plot_strain_so_idx = tuple([int(i) - 1 for i in plot_strain])
        # Get stress plot component (nonsymmetric) index
        plot_stress_idx = len(comp_order_nsym) + comp_order_nsym.index(plot_stress)
        plot_stress_so_idx = tuple([int(i) - 1 for i in plot_stress])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Strain mid-plane profile:
        # Infinitesimal strains: Infinitesimal strain tensor
        # Finite strains: Material logarithmic strain tensor
        #
        # Set figure name
        fig_name = strain_formulation + '_strain_' + plot_strain + '_midplane_plot'
        # Set plot data
        x = np.array(range(n_voxels_dims[0]))*(rve_dims[0]/n_voxels_dims[0])
        range_init = int(np.nonzero(x >= 0.0)[0][0])
        range_end = int(np.nonzero(x <= 1.0)[-1][-1]) + 1
        x = x[range_init:range_end]
        if n_dim == 2:
            y = [strain_vox[plot_strain][(i, int(n_voxels_dims[1]/2))]
                for i in range(range_init, range_end)]
        else:
            y = [strain_vox[plot_strain][(i, int(n_voxels_dims[1]/2),
                int(n_voxels_dims[2]/2))] for i in range(range_init, range_end)]
        # Build plot data array
        data_array = np.zeros((x.shape[0], 2))
        data_array[:, 0] = x
        data_array[:, 1] = y
        # Set axes labels
        x_label = '$x_{1}$'
        if strain_formulation == 'infinitesimal':
            y_label = '$\\varepsilon_{' + plot_strain + '}$'
        else:
            y_label = '$E_{' + plot_strain + '}$'
        # Output plot
        newgate_line_plots(plots_dir, fig_name, data_array,
                           x_label=x_label, y_label=y_label)
        # Comparison between infinitesimal and finite strains
        if strain_formulation == 'finite' and is_is_fs_comparison:
            # Set figure name
            fig_name = 'is_vs_fs' + '_strain_' + plot_strain + '_midplane_plot'
            # Set plot data
            x = np.array(range(n_voxels_dims[0]))*(rve_dims[0]/n_voxels_dims[0])
            range_init = int(np.nonzero(x >= 0.0)[0][0])
            range_end = int(np.nonzero(x <= 1.0)[-1][-1]) + 1
            x = x[range_init:range_end]
            if n_dim == 2:
                y1 = [strain_vox_is[plot_strain][(i, int(n_voxels_dims[1]/2))]
                     for i in range(range_init, range_end)]
                y2 = [strain_vox[plot_strain][(i, int(n_voxels_dims[1]/2))]
                     for i in range(range_init, range_end)]
            else:
                y1 = [strain_vox_is[plot_strain][(i, int(n_voxels_dims[1]/2),
                                                  int(n_voxels_dims[2]/2))]
                     for i in range(range_init, range_end)]
                y2 = [strain_vox[plot_strain][(i, int(n_voxels_dims[1]/2),
                                               int(n_voxels_dims[2]/2))]
                     for i in range(range_init, range_end)]
            # Build plot data array
            data_array = np.zeros((x.shape[0], 4))
            data_array[:, 0] = x
            data_array[:, 1] = y1
            data_array[:, 2] = x
            data_array[:, 3] = y2
            # Set data labels
            data_labels = ['$\mathrm{FFT-Infinitesimal}$',
                           '$\mathrm{FFT-Finite}$']
            # Set axes labels
            x_label = '$x_{1}$'
            y_label = '$\\varepsilon_{' + plot_strain + '} \, \mathrm{(IS)}$ / ' + \
                      '$E_{' + plot_strain + '} \, \mathrm{(FS)}$'
            # Set data labels
            data_labels = ['$\mathrm{Infinitesimal \; strains}$',
                           '$\mathrm{Finite \; strains}$']
            # Output plot
            newgate_line_plots(plots_dir, fig_name, data_array, x_label=x_label,
                               y_label=y_label, data_labels=data_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Homogenized stress-strain response:
        # Infinitesimal strains:
        #   Cauchy stress tensor- Infinitesimal strains tensor
        # Finite strains:
        #   First Piola-Kirchhoff stress tensor - Material logarithmic strain tensor
        #
        # Set figure name
        fig_name = strain_formulation + '_hom_stress_' + plot_stress + '_strain_' + \
            plot_strain + '_plot'
        # Build plot data array
        n_incs = hom_stress_strain.shape[0]
        data_array = np.zeros((n_incs, 2))
        data_array[:, 0] = hom_stress_strain[:, plot_strain_idx]
        data_array[:, 1] = hom_stress_strain[:, plot_stress_idx]
        # Set axes labels
        if strain_formulation == 'infinitesimal':
            x_label = '$\\varepsilon_{' + plot_strain + '}$'
            y_label = '$\\sigma_{' + plot_stress + '}$'
        else:
            x_label = '$F_{' + plot_strain + '}$'
            y_label = '$P_{' + plot_stress + '}$'
        # Output plot
        newgate_line_plots(plots_dir, fig_name, data_array,
                           x_label=x_label, y_label=y_label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Comparison of homogenized stress or homogenized strain response between
        # infinitesimal and finite strains:
        # Homogenized strain:
        #   Infinitesimal strains tensor/Material logarithmic strain tensor - Increment
        # Homogenized stress:
        #   Cauchy stress tensor - Increment
        #
        if strain_formulation == 'finite' and is_is_fs_comparison:
            # Set figure name
            fig_name = 'is_vs_fs_hom_strain_' + plot_strain + '_plot'
            # Build plot data array
            n_incs = hom_stress_strain.shape[0]
            data_array = np.zeros((n_incs, 4))
            data_array[:, 0] = np.array([x for x in range(n_incs)])
            data_array[:, 2] = np.array([x for x in range(n_incs)])
            data_array[:, 1] = hom_stress_strain_is[:, plot_strain_idx]
            # Loop over increments
            for inc in range(1, n_incs):
                # Get deformation gradient
                def_gradient = np.reshape(hom_stress_strain[inc, 0:n_dim**2],
                                          (n_dim, n_dim), order='F')
                # Compute material logarithmic strain tensor
                mat_log_strain = \
                    0.5*top.isotropic_tensor('log', np.matmul(np.transpose(def_gradient),
                                                              def_gradient))
                # Store material logarithmic strain tensor
                data_array[inc, 3] = mat_log_strain[plot_strain_so_idx]
            # Set axes labels
            x_label = '$\mathrm{Increment}$'
            y_label = '$\\varepsilon_{11} \, \mathrm{(IS)}$ / $E_{11} \, \mathrm{(FS)}$'
            # Set data labels
            data_labels = ['$\mathrm{Infinitesimal \; strains}$',
                           '$\mathrm{Finite \; strains}$']
            # Output plot
            newgate_line_plots(plots_dir, fig_name, data_array, x_label=x_label,
                               y_label=y_label, data_labels=data_labels)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set figure name
            fig_name = 'is_vs_fs_hom_stress_' + plot_stress + '_plot'
            # Build plot data array
            n_incs = hom_stress_strain.shape[0]
            data_array = np.zeros((n_incs, 4))
            data_array[:, 0] = np.array([x for x in range(n_incs)])
            data_array[:, 2] = np.array([x for x in range(n_incs)])
            data_array[:, 1] = hom_stress_strain_is[:, plot_stress_idx]
            # Loop over increments
            for inc in range(1, n_incs):
                # Get deformation gradient
                def_gradient = np.reshape(hom_stress_strain[inc, 0:n_dim**2],
                                          (n_dim, n_dim), order='F')
                # Get first Piola-Kirchhoff stress tensor
                first_piola_kirchhoff = \
                    np.reshape(hom_stress_strain[inc, n_dim**2:2*n_dim**2],
                               (n_dim, n_dim), order='F')
                # Compute Cauchy stress tensor
                cauchy = \
                    (1.0/np.linalg.det(def_gradient))*np.matmul(first_piola_kirchhoff,
                                                                np.transpose(def_gradient))
                # Store Cauchy stress tensor
                data_array[inc, 3] = cauchy[plot_stress_so_idx]
            # Set axes labels
            x_label = '$\mathrm{Increment}$'
            y_label = '$\\sigma_{' + plot_stress + '}$'
            # Set data labels
            data_labels = ['$\mathrm{Infinitesimal \; strains}$',
                           '$\mathrm{Finite \; strains}$']
            # Output plot
            newgate_line_plots(plots_dir, fig_name, data_array, x_label=x_label,
                               y_label=y_label, data_labels=data_labels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if True:
        # Set plots directory
        plots_dir = working_dir
        # Set figure name
        fig_name = strain_formulation + '_strain_magnitude_factor'
        # Set axes labels
        x_label = '$\chi$'
        y_label = '$\\norm{\mathbf{H}^{e}}(\\bm{Y}_{s})$'
        # Set axes limits
        x_min = 7.94328e-5
        x_max = 1.258925e-1
        y_min = None
        y_max = None
        # Set data labels
        point_labels = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
        data_labels = ['$\\bm{Y}_' + point_labels[x] + '$'
                       for x in range(2*len(target_voxels))]
        # Build plot data array
        data_array_complete = copy.deepcopy(data_array)
        data_array[:, 0:2] = data_array_complete[:, 0:2]
        data_array[:, 2:4] = data_array_complete[:, 8:10]
        data_array[:, 4:6] = data_array_complete[:, 2:4]
        data_array[:, 6:8] = data_array_complete[:, 10:12]
        data_array[:, 8:10] = data_array_complete[:, 4:6]
        data_array[:, 10:12] = data_array_complete[:, 12:14]
        data_array[:, 12:14] = data_array_complete[:, 6:8]
        data_array[:, 14:16] = data_array_complete[:, 14:16]
        # Output plot
        newgate_line_plots(plots_dir, fig_name, data_array, data_labels=data_labels,
                           x_label=x_label, y_label=y_label, x_min=x_min, x_max=x_max,
                           y_min=y_min, y_max=y_max)
    #
    # --------------------------------------------------------------------------------------
    # VTK OUTPUT
    if is_vtk_output:
        # Set clustering VTK file path
        vtk_file_path = working_dir + strain_formulation + '_fft_local_fields.vti'
        # Open clustering VTK file (append mode)
        if os.path.isfile(vtk_file_path):
            os.remove(vtk_file_path)
        vtk_file = open(vtk_file_path, 'a')
        # Instantiate VTK XML generator
        xml = XMLGenerator('ImageData', '1.0', 'LittleEndian', 'ascii', 'SinglePrecision',
                           'UInt64')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file header
        xml.write_file_header(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK dataset element
        dataset_parameters, piece_parameters = \
            VTKOutput._set_image_data_parameters(rve_dims, n_voxels_dims)
        xml.write_open_dataset_elem(vtk_file, dataset_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece
        xml.write_open_dataset_piece(vtk_file, piece_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open VTK dataset element piece cell data
        xml.write_open_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            data_list = list(strain_vox[comp].flatten('F'))
            min_val = min(data_list)
            max_val = max(data_list)
            data_parameters = {'Name': data_name, 'format': 'ascii',
                               'RangeMin': min_val, 'RangeMax': max_val}
            # Write cell data array
            xml.write_cell_data_array(vtk_file, data_list, data_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element cell data
        xml.write_close_cell_data(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element piece
        xml.write_close_dataset_piece(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close VTK dataset element
        xml.write_close_dataset_elem(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write VTK file footer
        xml.write_file_footer(vtk_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Close clustering VTK file
        vtk_file.close()
