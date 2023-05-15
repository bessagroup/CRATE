"""FFT-based homogenization basic scheme multi-scale method.

This module includes the implementation of the FFT-based homogenization basic
scheme proposed by Moulinec and Suquet (1998) [1]_ for the solution of
micro-scale equilibrium problems of linear elastic heterogeneous materials. The
finite strain extension of this method is also implemented by following the
formulation of Kabel and coworkers (2022) [2]_ and adopting the Hencky
hyperelastic constitutive model. It also includes a class that handles a
general incrementation of the macroscale strain loading and a dynamic
subincrementation scheme.

.. [1] Moulinec, H. and Suquet, P. (1998). *A numerical method for computing
       the overall response of nonlinear composites with complex
       microstructure.* Comp Methods Appl M, 157:69-94 (see `here
       <https://www.sciencedirect.com/science/article/pii/S0045782597002181>`_)

.. [2] Kabel, M., Bohlke, T., and Schneider, M. (2014). *Efficient fixed point
       and Newton-Krylov solver for FFT-based homogenization of elasticity
       at large deformations.* Comp Methods Appl M, 54:1497-1514 (see `here
       <https://link.springer.com/article/10.1007/s00466-014-1071-8>`_)

Classes
-------
FFTBasicScheme
    FFT-based homogenization basic scheme.
MacroscaleStrainIncrementer
    Macroscale strain loading incrementer.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import time
import copy
import warnings
import itertools as it
# Third-party
import numpy as np
import scipy.linalg
import colorama
# Local
import ioput.ioutilities as ioutil
import tensor.tensoroperations as top
import tensor.matrixoperations as mop
import clustering.citoperations as citop
from clustering.solution.dnshomogenization import DNSHomogenizationMethod
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class FFTBasicScheme(DNSHomogenizationMethod):
    """FFT-based homogenization basic scheme.

    FFT-based homogenization basic scheme proposed by Moulinec and Suquet
    (1998) [#]_ for the solution of micro-scale equilibrium problems of linear
    elastic heterogeneous materials. In particular, for a given RVE discretized
    in a regular grid of voxels, the method solves the microscale equilibrium
    problem when the RVE is subjected to a macroscale strain tensor and is
    constrained by periodic boundary conditions. Finite strain extension
    is also available, following the formulation of Kabel and coworkers
    (2022) [#]_ and adopting the Hencky hyperelastic constitutive model.

    A detailed description of the computational implementation can be found
    in Appendix B (infinitesimal strains) and Section 4.6 (finite strains)
    of Ferreira (2022) [#]_.

    .. [#] Moulinec, H. and Suquet, P. (1998). *A numerical method for
           computing the overall response of nonlinear composites with complex
           microstructure.* Comp Methods Appl M, 157:69-94 (see `here
           <https://www.sciencedirect.com/science/article/pii/
           S0045782597002181>`_)

    .. [#] Kabel, M., Bohlke, T., and Schneider, M. (2014). *Efficient fixed
           point and Newton-Krylov solver for FFT-based homogenization of
           elasticity at large deformations.* Comp Methods Appl M, 54:1497-1514
           (see `here <https://link.springer.com/article/10.1007/
           s00466-014-1071-8>`_)

    .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale Optimization
           of Thermoplastic Blends: Microstructural Generation, Constitutive
           Development and Clustering-based Reduced-Order Modeling.*
           PhD Thesis, University of Porto (see `here <https://
           repositorio-aberto.up.pt/handle/10216/146900?locale=en>`_)

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _max_n_iterations : int
        Maximum number of iterations to convergence.
    _conv_criterion : {'stress_div', 'avg_stress_norm'}
        Convergence criterion: 'stress_div' is the original convergence
        criterion based on the evaluation of the divergence of the stress
        tensor; 'avg_stress_norm' is based on the iterative change of the
        average stress norm.
    _conv_tol : float
        Convergence tolerance.
    _max_subinc_level : int
        Maximum level of macroscale loading subincrementation.
    _max_cinc_cuts : int
        Maximum number of consecutive macroscale loading increment cuts.
    _hom_stress_strain : numpy.ndarray (2d)
        Homogenized stress-strain material response. The homogenized strain and
        homogenized stress tensor components of the i-th loading increment are
        stored columnwise in the i-th row, sorted respectively. Infinitesimal
        strain tensor and Cauchy stress tensor (infinitesimal strains) or
        Deformation gradient and first Piola-Kirchhoff stress tensor (finite
        strains).

    Methods
    -------
    compute_rve_local_response(self, mac_strain_id, mac_strain, verbose=False)
        Compute RVE local elastic strain response.
    _elastic_constitutive_model(self, strain_vox, evar1, evar2, evar3, \
                                finite_strains_model='stvenant-kirchhoff', \
                                is_optimized=True)
        Elastic or hyperelastic material constitutive model.
    stress_div_conv_criterion(self, freqs_dims, stress_DFT_vox)
        Convergence criterion based on the divergence of the stress tensor.
    compute_avg_state_vox(self, state_vox)
        Compute average norm of strain or stress local field.
    _compute_homogenized_field(self, state_vox)
        Perform homogenization over regular grid spatial discretization.
    get_hom_stress_strain(self)
        Get homogenized strain-stress material response.
    _display_greetings()
        Output greetings.
    _display_increment_init(inc, subinc_level, total_lfact, inc_lfact)
        Output increment initial data.
    _display_increment_end(strain_formulation, hom_strain, hom_stress, \
                           inc_time, total_time)
        Output increment end data.
    _display_iteration(iter, iter_time, discrete_error)
        Output iteration data.
    _display_increment_cut(cut_msg='')
        Output increment cut data.
    """
    def __init__(self, strain_formulation, problem_type, rve_dims,
                 n_voxels_dims, regular_grid, material_phases,
                 material_phases_properties):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        regular_grid : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the material phase label (int) assigned to the
            corresponding voxel.
        material_phases : list[str]
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to
            each material phase (key, str).
        """
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
    # -------------------------------------------------------------------------
    def compute_rve_local_response(self, mac_strain_id, mac_strain,
                                   verbose=False):
        """Compute RVE local elastic strain response.

        Compute the RVE local strain response (solution of microscale
        equilibrium problem) when subjected to a given macroscale strain
        loading, namely a macroscale infinitesimal strain tensor (infinitesimal
        strains) or a macroscale deformation gradient (finite strains). It is
        assumed that the RVE is spatially discretized in a regular grid of
        voxels.

        ----

        Parameters
        ----------
        mac_strain_id : int
            Macroscale strain second-order tensor identifier.
        mac_strain : numpy.ndarray (2d)
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        strain_vox: dict
            RVE local strain response (item, numpy.ndarray of shape equal to
            RVE regular grid discretization) for each strain component
            (key, str). Infinitesimal strain tensor (infinitesimal strains) or
            material logarithmic strain tensor (finite strains).
        """
        # Set initial time
        init_time = time.time()
        # Display greetings
        if verbose:
            type(self)._display_greetings()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store total macroscale strain tensor
        mac_strain_total = copy.deepcopy(mac_strain)
        # Initialize macroscale strain increment cut flag
        is_inc_cut = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized strain-stress response
        self._hom_stress_strain = np.zeros((1, 2*self._n_dim**2))
        if self._strain_formulation == 'finite':
            self._hom_stress_strain[0, 0] = 1.0
        #
        #                                    Material phases elasticity tensors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set elastic properties-related optimized variables
        evar1 = np.zeros(tuple(self._n_voxels_dims))
        evar2 = np.zeros(tuple(self._n_voxels_dims))
        for mat_phase in self._material_phases:
            # Get material phase elastic properties
            E = self._material_phases_properties[mat_phase]['E']
            v = self._material_phases_properties[mat_phase]['v']
            # Build optimized variables
            evar1[self._regular_grid == int(mat_phase)] = \
                (E*v)/((1.0 + v)*(1.0 - 2.0*v))
            evar2[self._regular_grid == int(mat_phase)] = \
                np.multiply(2, E/(2.0*(1.0 + v)))
        evar3 = np.add(evar1, evar2)
        #
        #                                 Reference material elastic properties
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material elastic properties as the mean between the
        # minimum and maximum values existent among the microstructure's
        # material phases (proposed by Moulinec and Suquet (1998))
        mat_prop_ref = dict()
        mat_prop_ref['E'] = \
            0.5*(min([self._material_phases_properties[phase]['E']
                      for phase in self._material_phases])
                 + max([self._material_phases_properties[phase]['E']
                        for phase in self._material_phases]))
        mat_prop_ref['v'] = \
            0.5*(min([self._material_phases_properties[phase]['v']
                      for phase in self._material_phases])
                 + max([self._material_phases_properties[phase]['v']
                        for phase in self._material_phases]))
        #
        #                                              Frequency discretization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set discrete frequencies (rad/m) for each dimension
        freqs_dims = list()
        for i in range(self._n_dim):
            # Set sampling spatial period
            sampling_period = self._rve_dims[i]/self._n_voxels_dims[i]
            # Set discrete frequencies
            freqs_dims.append(2*np.pi*np.fft.fftfreq(self._n_voxels_dims[i],
                                                     sampling_period))
        #
        #                                     Reference material Green operator
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Green operator material independent terms
        gop_1_dft_vox, gop_2_dft_vox, _ = \
            citop.gop_material_independent_terms(
                self._strain_formulation, self._problem_type, self._rve_dims,
                self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Green operator matricial form components
        comps = list(it.product(comp_order, comp_order))
        # Set mapping between Green operator fourth-order tensor and matricial
        # form components
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1
                               for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in [comp_order.index(comps[i][0]),
                                           comp_order.index(comps[i][1])]])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Green operator
        gop_dft_vox = {''.join([str(x + 1) for x in idx]):
                       np.zeros(tuple(self._n_voxels_dims))
                       for idx in fo_indexes}
        # Compute Green operator matricial form components
        for i in range(len(mf_indexes)):
            # Get fourth-order tensor indexes
            fo_idx = fo_indexes[i]
            # Get Green operator component
            comp = ''.join([str(x+1) for x in fo_idx])
            # Compute Green operator matricial form component
            gop_dft_vox[comp] = c1*gop_1_dft_vox[comp] + c2*gop_2_dft_vox[comp]
        #
        #                              Macroscale strain loading incrementation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of increments
        if self._strain_formulation == 'infinitesimal':
            n_incs = 1
        else:
            n_incs = 1
        # Set incremental load factors
        inc_lfacts = n_incs*[1.0/n_incs, ]
        # Initialize macroscale strain incrementer
        mac_strain_incrementer = MacroscaleStrainIncrementer(
            self._strain_formulation, self._problem_type, mac_strain_total,
            inc_lfacts=inc_lfacts, max_subinc_level=self._max_subinc_level,
            max_cinc_cuts=self._max_cinc_cuts)
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
        #                                   Macroscale loading incremental loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Start macroscale loading incremental loop
        while True:
            #
            #                                           Initial iterative guess
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                    strain_vox[comp] = np.full(self._regular_grid.shape,
                                               mac_strain[so_idx])
                # Initialize last converged strain tensor
                strain_old_vox = copy.deepcopy(strain_vox)
            else:
                # Initial guess: Last converged strain field
                strain_vox = copy.deepcopy(strain_old_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute stress initial iterative guess
            stress_vox = self._elastic_constitutive_model(strain_vox, evar1,
                                                          evar2, evar3)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute average strain/stress norm
            if self._conv_criterion == 'avg_stress_norm':
                # Compute initial guess average stress norm
                avg_stress_norm = self._compute_avg_state_vox(stress_vox)
                # Initialize last iteration average stress norm
                avg_stress_norm_itold = 0.0
            #
            #                           Strain Discrete Fourier Transform (DFT)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute strain Discrete Fourier Transform (DFT)
            strain_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims),
                                             dtype=complex)
                              for comp in comp_order}
            # Loop over strain components
            for comp in comp_order:
                # Discrete Fourier Transform (DFT) by means of Fast Fourier
                # Transform (FFT)
                strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize macroscale strain voxelwise tensor
            mac_strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                              for comp in comp_order}
            mac_strain_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims),
                                                 dtype=complex)
                                  for comp in comp_order}
            # Enforce macroscale strain DFT at the zero-frequency
            freq_0_idx = self._n_dim*(0,)
            mac_strain_DFT_0 = {}
            for comp in comp_order:
                # Get strain component indexes
                so_idx = tuple([int(x) - 1 for x in comp])
                # Compute macroscale strain DFT
                mac_strain_vox[comp] = np.full(self._regular_grid.shape,
                                               mac_strain[so_idx])
                mac_strain_DFT_vox[comp] = np.fft.fftn(mac_strain_vox[comp])
                # Enforce macroscale strain DFT at the zero-frequency
                mac_strain_DFT_0[comp] = mac_strain_DFT_vox[comp][freq_0_idx]
            #
            #                                      Fixed-point iterative scheme
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize iteration counter:
            iter = 0
            # Start iterative loop
            while True:
                #
                #                       Stress Discrete Fourier Transform (DFT)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute stress Discrete Fourier Transform (DFT)
                stress_DFT_vox = {comp: np.zeros(tuple(self._n_voxels_dims),
                                                 dtype=complex)
                                  for comp in comp_order}
                for comp in comp_order:
                    # Discrete Fourier Transform (DFT) by means of Fast Fourier
                    # Transform (FFT)
                    stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
                #
                #                                        Convergence evaluation
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Evaluate convergence criterion
                if self._conv_criterion == 'stress_div':
                    # Compute discrete error
                    discrete_error = self._stress_div_conv_criterion(
                        freqs_dims, stress_DFT_vox)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif self._conv_criterion == 'avg_stress_norm':
                    # Compute discrete error
                    discrete_error = \
                        abs(avg_stress_norm - avg_stress_norm_itold) \
                        / avg_stress_norm
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Display iteration data
                if verbose:
                    type(self)._display_iteration(
                        iter, time.time() - iter_init_time, discrete_error)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check solution convergence and iteration counter
                if discrete_error <= self._conv_tol:
                    # Leave fixed-point iterative scheme
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif iter == self._max_n_iterations or \
                        not np.isfinite(discrete_error):
                    # Set increment cut output
                    if not np.isfinite(discrete_error):
                        cut_msg = 'Solution diverged.'
                    else:
                        cut_msg = 'Maximum number of iterations reached ' + \
                                  'without convergence.'
                    # Raise macroscale increment cut procedure
                    is_inc_cut = True
                    # Display increment cut (maximum number of iterations)
                    type(self)._display_increment_cut(cut_msg)
                    # Leave fixed-point iterative scheme
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    # Increment iteration counter
                    iter += 1
                    # Set iteration initial time
                    iter_init_time = time.time()
                #
                #                                                 Update strain
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                        idx1 = [comp_order.index(comp_i),
                                comp_order.index(comp_j)]
                        idx2 = comp_order.index(comp_j)
                        aux = np.add(
                            aux, np.multiply(
                                mop.kelvin_factor(idx1, comp_order)
                                * gop_dft_vox[comp_i + comp_j],
                                mop.kelvin_factor(idx2, comp_order)
                                * stress_DFT_vox[comp_j]))
                    # Update strain DFT
                    strain_DFT_vox[comp_i] = np.subtract(
                        strain_DFT_vox[comp_i],
                        (1.0/mop.kelvin_factor(i, comp_order))*aux)
                    # Enforce macroscale strain DFT at the zero-frequency
                    freq_0_idx = self._n_dim*(0,)
                    strain_DFT_vox[comp_i][freq_0_idx] = \
                        mac_strain_DFT_0[comp_i]
                #
                #              Strain Inverse Discrete Fourier Transform (IDFT)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute strain Inverse Discrete Fourier Transform (IDFT)
                for comp in comp_order:
                    # Inverse Discrete Fourier Transform (IDFT) by means of
                    # Fast Fourier Transform (FFT)
                    strain_vox[comp] = \
                        np.real(np.fft.ifftn(strain_DFT_vox[comp]))
                #
                #                                                 Stress update
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update stress
                stress_vox = self._elastic_constitutive_model(strain_vox,
                                                              evar1, evar2,
                                                              evar3)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute average strain/stress norm
                if self._conv_criterion == 'avg_stress_norm':
                    # Update last iteration average stress norm
                    avg_stress_norm_itold = avg_stress_norm
                    # Compute average stress norm
                    avg_stress_norm = self._compute_avg_state_vox(stress_vox)
            #
            #                                   Macroscale strain increment cut
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_inc_cut:
                # Reset macroscale strain increment cut flag
                is_inc_cut = False
                # Perform macroscale strain increment cut
                mac_strain_incrementer.increment_cut()
                # Get current macroscale strain tensor
                mac_strain = mac_strain_incrementer.get_current_mac_strain()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Display increment data
                if verbose:
                    type(self)._display_increment_init(
                        *mac_strain_incrementer.get_inc_output_data())
                # Start new macroscale strain increment solution procedure
                continue
            #
            #                           Macroscale strain increment convergence
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute homogenized strain and stress tensor
            hom_strain = self._compute_homogenized_field(strain_vox)
            hom_stress = self._compute_homogenized_field(stress_vox)
            # Append to homogenized strain-stress response
            self._hom_stress_strain = np.vstack(
                [self._hom_stress_strain, np.concatenate(
                    (hom_strain.flatten('F'), hom_stress.flatten('F')))])
            # Display increment data
            if verbose:
                type(self)._display_increment_end(self._strain_formulation,
                                                  hom_strain, hom_stress,
                                                  time.time() - inc_init_time,
                                                  time.time() - init_time)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return if last macroscale loading increment
            if mac_strain_incrementer.get_is_last_inc():
                # Compute material logarithmic strain tensor from deformation
                # gradient
                if self._strain_formulation == 'finite':
                    # Loop over voxels
                    for voxel in it.product(*[list(range(n))
                                              for n in self._n_voxels_dims]):
                        # Initialize deformation gradient
                        def_gradient = np.zeros((self._n_dim, self._n_dim))
                        # Loop over deformation gradient components
                        for comp in self._comp_order_nsym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Get voxel deformation gradient component
                            def_gradient[so_idx] = strain_vox[comp][voxel]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute material logarithmic strain tensor
                        mat_log_strain = 0.5*top.isotropic_tensor(
                            'log', np.matmul(np.transpose(def_gradient),
                                             def_gradient))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over material logarithmic strain tensor
                        # components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store material logarithmic strain tensor
                            strain_vox[comp][voxel] = mat_log_strain[so_idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Return local strain field
                return strain_vox
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update last converged strain tensor
            strain_old_vox = copy.deepcopy(strain_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # -------------------------------------------------------------------------
    def _elastic_constitutive_model(self, strain_vox, evar1, evar2, evar3,
                                    finite_strains_model='stvenant-kirchhoff',
                                    is_optimized=True):
        """Elastic or hyperelastic material constitutive model.

        Available constitutive models:

        *Infinitesimal strains*:

        * Isotropic linear elastic constitutive model:

            .. math::

               \\boldsymbol{\\sigma} = \\boldsymbol{\\mathsf{D}}^{e} :
                                       \\boldsymbol{\\varepsilon}

          where :math:`\\boldsymbol{\\sigma}` is the Cauchyevar1 stress tensor,
          :math:`\\boldsymbol{\\mathsf{D}}^{e}` is the elasticity tensor, and
          :math:`\\boldsymbol{\\varepsilon}` is the infinitesimal strain
          tensor.

        ----

        *Finite strains*:

        * Hencky hyperelastic (isotropic) constitutive model:

            .. math::

               \\boldsymbol{\\tau} = \\boldsymbol{\\mathsf{D}}^{e} :
                                     \\boldsymbol{\\varepsilon}

          where :math:`\\boldsymbol{\\tau}` is the Kirchhoff stress tensor,
          :math:`\\boldsymbol{\\mathsf{D}}^{e}` is the elasticity tensor, and
          :math:`\\boldsymbol{\\varepsilon}` is the spatial logarithmic strain
          tensor.

        * St.Venant-Kirchhoff hyperelastic (isotropic) constitutive model:

            .. math::

               \\boldsymbol{S} = \\boldsymbol{\\mathsf{D}}^{e} :
                                 \\boldsymbol{E}^{(2)}

          where :math:`\\boldsymbol{S}` is the second Piola-Kirchhoff stress
          tensor, :math:`\\boldsymbol{\\mathsf{D}}^{e}` is the elasticity
          tensor, and :math:`\\boldsymbol{E}^{(2)}` is the Green-Lagrange
          strain tensor.

        A detailed description of the computational implementation based on
        Hadamard (element-wise) operations can be found in Section 4.6 of
        Ferreira (2022) [#]_.

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        strain_vox: dict
            Local strain response (item, numpy.ndarray of shape equal to RVE
            regular grid discretization) for each strain component (key, str).
            Infinitesimal strain tensor (infinitesimal strains) or deformation
            gradient (finite strains).
        evar1 : numpy.ndarray (2d or 3d)
            Auxiliar elastic properties array (numpy.ndarray of shape equal to
            RVE regular grid discretization) containing an elastic
            properties-related quantity associated to each voxel,

            .. math ::

               \\dfrac{E \\nu}{(1 + \\nu)(1 - 2 \\nu)} \\, ,

            where :math:`E` and :math:`\\nu` are the Young's Modulus and
            Poisson's ratio, respectively.
        evar2 : numpy.ndarray (2d or 3d)
            Auxiliar elastic properties array (numpy.ndarray of shape equal to
            RVE regular grid discretization) containing an elastic
            properties-related quantity associated to each voxel,

            .. math ::

               \\dfrac{E}{1 + \\nu} \\, ,

            where :math:`E` and :math:`\\nu` are the Young's Modulus and
            Poisson's ratio, respectively.
        evar3 : numpy.ndarray (2d or 3d)
            Auxiliar elastic properties array (numpy.ndarray of shape equal to
            RVE regular grid discretization) containing an elastic
            properties-related quantity associated to each voxel,

            .. math ::

               \\dfrac{E \\nu}{(1 + \\nu)(1 - 2 \\nu)} -
               \\dfrac{E}{1 + \\nu} \\, ,

            where :math:`E` and :math:`\\nu` are the Young's Modulus and
            Poisson's ratio, respectively.
        finite_strains_model : {'hencky', 'stvenant-kirchhoff'}, \
                               default='hencky'
            Finite strains hyperelastic isotropic constitutive model.
        is_optimized : bool
            Optimization flag (minimizes loops over spatial discretization
            voxels).

        Returns
        -------
        stress_vox: dict
            Local stress response (item, numpy.ndarray of shape equal to RVE
            regular grid discretization) for each stress component (key, str).
            Cauchy stress tensor (infinitesimal strains) or first
            Piola-Kirchhoff stress tensor (finite strains).
        """
        # Initialize Cauchy stress tensor (infinitesimal strains) or first
        # Piola-Kirchhoff stress tensor (finite strains)
        if self._strain_formulation == 'infinitesimal':
            stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                          for comp in self._comp_order_sym}
        elif self._strain_formulation == 'finite':
            stress_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                          for comp in self._comp_order_nsym}
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check hyperlastic constitutive model
        if self._strain_formulation == 'finite':
            if finite_strains_model not in ('hencky', 'stvenant-kirchhoff'):
                raise RuntimeError('Unknown hyperelastic constitutive model.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute finite strains strain tensor
        if self._strain_formulation == 'finite':
            # Save deformation gradient
            def_gradient_vox = copy.deepcopy(strain_vox)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize symmetric finite strains strain tensor
            finite_sym_strain_vox = {comp: np.zeros(tuple(self._n_voxels_dims))
                                     for comp in self._comp_order_sym}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute finite strains strain tensor
            if is_optimized:
                # Compute finite strains strain tensor according to
                # hyperelastic constitutive model
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Compute voxelwise material Green-Lagrange strain tensor
                    if self._n_dim == 2:
                        finite_sym_strain_vox['11'] = \
                            0.5*(np.add(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['11']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['21']))
                                 - 1.0)
                        finite_sym_strain_vox['22'] = \
                            0.5*(np.add(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['12']),
                                        np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['22']))
                                 - 1.0)
                        finite_sym_strain_vox['12'] = \
                            0.5*np.add(np.multiply(def_gradient_vox['11'],
                                                   def_gradient_vox['12']),
                                       np.multiply(def_gradient_vox['21'],
                                                   def_gradient_vox['22']))
                    else:
                        finite_sym_strain_vox['11'] = \
                            0.5*(np.add(
                                np.add(np.multiply(def_gradient_vox['11'],
                                                   def_gradient_vox['11']),
                                       np.multiply(def_gradient_vox['21'],
                                                   def_gradient_vox['21'])),
                                np.multiply(def_gradient_vox['31'],
                                            def_gradient_vox['31'])) - 1.0)
                        finite_sym_strain_vox['12'] = \
                            0.5*np.add(
                                np.add(np.multiply(def_gradient_vox['11'],
                                                   def_gradient_vox['12']),
                                       np.multiply(def_gradient_vox['21'],
                                                   def_gradient_vox['22'])),
                                np.multiply(def_gradient_vox['31'],
                                            def_gradient_vox['32']))
                        finite_sym_strain_vox['13'] = \
                            0.5*np.add(
                                np.add(np.multiply(def_gradient_vox['11'],
                                                   def_gradient_vox['13']),
                                       np.multiply(def_gradient_vox['21'],
                                                   def_gradient_vox['23'])),
                                np.multiply(def_gradient_vox['31'],
                                            def_gradient_vox['33']))
                        finite_sym_strain_vox['22'] = \
                            0.5*(np.add(
                                np.add(np.multiply(def_gradient_vox['12'],
                                                   def_gradient_vox['12']),
                                       np.multiply(def_gradient_vox['22'],
                                                   def_gradient_vox['22'])),
                                np.multiply(def_gradient_vox['32'],
                                            def_gradient_vox['32'])) - 1.0)
                        finite_sym_strain_vox['23'] = \
                            0.5*np.add(
                                np.add(np.multiply(def_gradient_vox['12'],
                                                   def_gradient_vox['13']),
                                       np.multiply(def_gradient_vox['22'],
                                                   def_gradient_vox['23'])),
                                np.multiply(def_gradient_vox['32'],
                                            def_gradient_vox['33']))
                        finite_sym_strain_vox['33'] = \
                            0.5*(np.add(
                                np.add(np.multiply(def_gradient_vox['13'],
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
                        ftfvar11 = \
                            np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['11']),
                                          np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['12'])),
                                   np.multiply(def_gradient_vox['13'],
                                               def_gradient_vox['13']))
                        ftfvar12 = \
                            np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['21']),
                                          np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['22'])),
                                   np.multiply(def_gradient_vox['13'],
                                               def_gradient_vox['23']))
                        ftfvar13 = \
                            np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['31']),
                                          np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['32'])),
                                   np.multiply(def_gradient_vox['13'],
                                               def_gradient_vox['33']))
                        ftfvar22 = \
                            np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                      def_gradient_vox['21']),
                                          np.multiply(def_gradient_vox['22'],
                                                      def_gradient_vox['22'])),
                                   np.multiply(def_gradient_vox['23'],
                                               def_gradient_vox['23']))
                        ftfvar23 = \
                            np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                      def_gradient_vox['31']),
                                          np.multiply(def_gradient_vox['22'],
                                                      def_gradient_vox['32'])),
                                   np.multiply(def_gradient_vox['23'],
                                               def_gradient_vox['33']))
                        ftfvar33 = \
                            np.add(np.add(np.multiply(def_gradient_vox['31'],
                                                      def_gradient_vox['31']),
                                          np.multiply(def_gradient_vox['32'],
                                                      def_gradient_vox['32'])),
                                   np.multiply(def_gradient_vox['33'],
                                               def_gradient_vox['33']))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over voxels
                    for voxel in it.product(*[list(range(n))
                                              for n in self._n_voxels_dims]):
                        # Build left Cauchy-Green strain tensor
                        if self._n_dim == 2:
                            left_cauchy_green = np.reshape(
                                np.array([ftfvar11[voxel], ftfvar12[voxel],
                                          ftfvar12[voxel], ftfvar22[voxel]]),
                                (self._n_dim, self._n_dim), 'F')
                        else:
                            left_cauchy_green = np.reshape(
                                np.array([ftfvar11[voxel], ftfvar12[voxel],
                                          ftfvar13[voxel], ftfvar12[voxel],
                                          ftfvar22[voxel], ftfvar23[voxel],
                                          ftfvar13[voxel], ftfvar23[voxel],
                                          ftfvar33[voxel]]),
                                (self._n_dim, self._n_dim), 'F')
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute spatial logarithmic strain tensor
                        with warnings.catch_warnings():
                            # Supress warnings
                            warnings.simplefilter(
                                'ignore', category=RuntimeWarning)
                            # Compute spatial logarithmic strain tensor
                            spatial_log_strain = 0.5*top.isotropic_tensor(
                                'log', left_cauchy_green)
                            if np.any(np.logical_not(
                                    np.isfinite(spatial_log_strain))):
                                spatial_log_strain = \
                                    0.5*scipy.linalg.logm(left_cauchy_green)
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over spatial logarithmic strain tensor
                        # components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store spatial logarithmic strain tensor
                            finite_sym_strain_vox[comp][voxel] = \
                                spatial_log_strain[so_idx]
            else:
                # Compute finite strains strain tensor according to
                # hyperelastic constitutive model
                for voxel in it.product(*[list(range(n))
                                          for n in self._n_voxels_dims]):
                    # Initialize deformation gradient
                    def_gradient = np.zeros((self._n_dim, self._n_dim))
                    # Loop over deformation gradient components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel deformation gradient component
                        def_gradient[so_idx] = strain_vox[comp][voxel]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute finite strains strain tensor according to
                    # hyperelastic constitutive model
                    if finite_strains_model == 'stvenant-kirchhoff':
                        # Compute material Green-Lagrange strain tensor
                        mat_green_lagr_strain = \
                            0.5*(np.matmul(np.transpose(def_gradient),
                                           def_gradient) - np.eye(self._n_dim))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                            warnings.simplefilter(
                                'ignore', category=RuntimeWarning)
                            # Compute spatial logarithmic strain tensor
                            spatial_log_strain = 0.5*top.isotropic_tensor(
                                'log', np.matmul(def_gradient,
                                                 np.transpose(def_gradient)))
                            if np.any(np.logical_not(
                                    np.isfinite(spatial_log_strain))):
                                spatial_log_strain = 0.5*scipy.linalg.logm(
                                    np.matmul(def_gradient,
                                              np.transpose(def_gradient)))
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Loop over symmetric strain components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Store spatial logarithmic strain tensor
                            finite_sym_strain_vox[comp][voxel] = \
                                spatial_log_strain[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Store symmetric finite strains strain tensor
            strain_vox = finite_sym_strain_vox
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Cauchy stress tensor from infinitesimal strain tensor
        # (infinitesimal strains) or Kirchhoff stress tensor from spatial
        # logarithmic strain tensor (finite strains)
        if self._problem_type == 1:
            stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                      np.multiply(evar1, strain_vox['22']))
            stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                      np.multiply(evar1, strain_vox['11']))
            stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
        else:
            stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                      np.multiply(evar1,
                                                  np.add(strain_vox['22'],
                                                         strain_vox['33'])))
            stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                      np.multiply(evar1,
                                                  np.add(strain_vox['11'],
                                                         strain_vox['33'])))
            stress_vox['33'] = np.add(np.multiply(evar3, strain_vox['33']),
                                      np.multiply(evar1,
                                                  np.add(strain_vox['11'],
                                                         strain_vox['22'])))
            stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
            stress_vox['23'] = np.multiply(evar2, strain_vox['23'])
            stress_vox['13'] = np.multiply(evar2, strain_vox['13'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute First Piola-Kirchhoff stress tensor
        if self._strain_formulation == 'finite':
            # Initialize First Piola-Kirchhoff stress tensor
            first_piola_stress_vox = {comp:
                                      np.zeros(tuple(self._n_voxels_dims))
                                      for comp in self._comp_order_nsym}
            # Compute First Piola-Kirchhoff stress tensor
            if is_optimized:
                # Compute First Piola-Kirchhoff stress tensor according to
                # hyperelastic constitutive model
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Compute voxelwise First Piola-Kirchhoff stress tensor
                    if self._n_dim == 2:
                        first_piola_stress_vox['11'] = \
                            np.add(np.multiply(def_gradient_vox['11'],
                                               stress_vox['11']),
                                   np.multiply(def_gradient_vox['12'],
                                               stress_vox['12']))
                        first_piola_stress_vox['21'] = \
                            np.add(np.multiply(def_gradient_vox['21'],
                                               stress_vox['11']),
                                   np.multiply(def_gradient_vox['22'],
                                               stress_vox['12']))
                        first_piola_stress_vox['12'] = \
                            np.add(np.multiply(def_gradient_vox['11'],
                                               stress_vox['12']),
                                   np.multiply(def_gradient_vox['12'],
                                               stress_vox['22']))
                        first_piola_stress_vox['22'] = \
                            np.add(np.multiply(def_gradient_vox['21'],
                                               stress_vox['12']),
                                   np.multiply(def_gradient_vox['22'],
                                               stress_vox['22']))
                    else:
                        first_piola_stress_vox['11'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'],
                                               stress_vox['11']),
                                   np.multiply(def_gradient_vox['12'],
                                               stress_vox['12'])),
                            np.multiply(def_gradient_vox['13'],
                                        stress_vox['13']))
                        first_piola_stress_vox['21'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'],
                                               stress_vox['11']),
                                   np.multiply(def_gradient_vox['22'],
                                               stress_vox['12'])),
                            np.multiply(def_gradient_vox['23'],
                                        stress_vox['13']))
                        first_piola_stress_vox['31'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'],
                                               stress_vox['11']),
                                   np.multiply(def_gradient_vox['32'],
                                               stress_vox['12'])),
                            np.multiply(def_gradient_vox['33'],
                                        stress_vox['13']))
                        first_piola_stress_vox['12'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'],
                                               stress_vox['12']),
                                   np.multiply(def_gradient_vox['12'],
                                               stress_vox['22'])),
                            np.multiply(def_gradient_vox['13'],
                                        stress_vox['23']))
                        first_piola_stress_vox['22'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'],
                                               stress_vox['12']),
                                   np.multiply(def_gradient_vox['22'],
                                               stress_vox['22'])),
                            np.multiply(def_gradient_vox['23'],
                                        stress_vox['23']))
                        first_piola_stress_vox['32'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'],
                                               stress_vox['12']),
                                   np.multiply(def_gradient_vox['32'],
                                               stress_vox['22'])),
                            np.multiply(def_gradient_vox['33'],
                                        stress_vox['23']))
                        first_piola_stress_vox['13'] = np.add(
                            np.add(np.multiply(def_gradient_vox['11'],
                                               stress_vox['13']),
                                   np.multiply(def_gradient_vox['12'],
                                               stress_vox['23'])),
                            np.multiply(def_gradient_vox['13'],
                                        stress_vox['33']))
                        first_piola_stress_vox['23'] = np.add(
                            np.add(np.multiply(def_gradient_vox['21'],
                                               stress_vox['13']),
                                   np.multiply(def_gradient_vox['22'],
                                               stress_vox['23'])),
                            np.multiply(def_gradient_vox['23'],
                                        stress_vox['33']))
                        first_piola_stress_vox['33'] = np.add(
                            np.add(np.multiply(def_gradient_vox['31'],
                                               stress_vox['13']),
                                   np.multiply(def_gradient_vox['32'],
                                               stress_vox['23'])),
                            np.multiply(def_gradient_vox['33'],
                                        stress_vox['33']))
                else:
                    if self._n_dim == 2:
                        # Compute voxelwise determinant of deformation gradient
                        jvar = np.reciprocal(
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['12'])))
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox['11'] = np.multiply(
                            jvar,
                            np.subtract(np.multiply(stress_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(stress_vox['12'],
                                                    def_gradient_vox['12'])))
                        first_piola_stress_vox['21'] = np.multiply(
                            jvar,
                            np.subtract(np.multiply(stress_vox['12'],
                                                    def_gradient_vox['22']),
                                        np.multiply(stress_vox['22'],
                                                    def_gradient_vox['12'])))
                        first_piola_stress_vox['12'] = np.multiply(
                            jvar,
                            np.subtract(np.multiply(stress_vox['12'],
                                                    def_gradient_vox['11']),
                                        np.multiply(stress_vox['11'],
                                                    def_gradient_vox['21'])))
                        first_piola_stress_vox['22'] = np.multiply(
                            jvar,
                            np.subtract(np.multiply(stress_vox['22'],
                                                    def_gradient_vox['11']),
                                        np.multiply(stress_vox['12'],
                                                    def_gradient_vox['21'])))
                    else:
                        # Compute voxelwise determinant of deformation gradient
                        jvar = np.reciprocal(np.add(np.subtract(
                            np.multiply(
                                def_gradient_vox['11'],
                                np.subtract(
                                    np.multiply(def_gradient_vox['22'],
                                                def_gradient_vox['33']),
                                    np.multiply(def_gradient_vox['23'],
                                                def_gradient_vox['32']))),
                            np.multiply(
                                def_gradient_vox['12'],
                                np.subtract(
                                    np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['33']),
                                    np.multiply(def_gradient_vox['23'],
                                                def_gradient_vox['31'])))),
                            np.multiply(
                                def_gradient_vox['13'],
                                np.subtract(
                                    np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['32']),
                                    np.multiply(def_gradient_vox['22'],
                                                def_gradient_vox['31'])))))
                        # Compute voxelwise transpose of inverse of deformation
                        # gradient
                        fitvar11 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['33']),
                                        np.multiply(def_gradient_vox['23'],
                                                    def_gradient_vox['32'])))
                        fitvar21 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['32']),
                                        np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['33'])))
                        fitvar31 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['23']),
                                        np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['22'])))
                        fitvar12 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['23'],
                                                    def_gradient_vox['31']),
                                        np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['33'])))
                        fitvar22 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['33']),
                                        np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['31'])))
                        fitvar32 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['13'],
                                                    def_gradient_vox['21']),
                                        np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['23'])))
                        fitvar13 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['21'],
                                                    def_gradient_vox['32']),
                                        np.multiply(def_gradient_vox['22'],
                                                    def_gradient_vox['31'])))
                        fitvar23 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['31']),
                                        np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['32'])))
                        fitvar33 = np.multiply(
                            jvar,
                            np.subtract(np.multiply(def_gradient_vox['11'],
                                                    def_gradient_vox['22']),
                                        np.multiply(def_gradient_vox['12'],
                                                    def_gradient_vox['21'])))
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox['11'] = np.add(
                            np.add(np.multiply(stress_vox['11'], fitvar11),
                                   np.multiply(stress_vox['12'], fitvar21)),
                            np.multiply(stress_vox['13'], fitvar31))
                        first_piola_stress_vox['21'] = np.add(
                            np.add(np.multiply(stress_vox['12'], fitvar11),
                                   np.multiply(stress_vox['22'], fitvar21)),
                            np.multiply(stress_vox['23'], fitvar31))
                        first_piola_stress_vox['31'] = np.add(
                            np.add(np.multiply(stress_vox['13'], fitvar11),
                                   np.multiply(stress_vox['23'], fitvar21)),
                            np.multiply(stress_vox['33'], fitvar31))
                        first_piola_stress_vox['12'] = np.add(
                            np.add(np.multiply(stress_vox['11'], fitvar12),
                                   np.multiply(stress_vox['12'], fitvar22)),
                            np.multiply(stress_vox['13'], fitvar32))
                        first_piola_stress_vox['22'] = np.add(
                            np.add(np.multiply(stress_vox['12'], fitvar12),
                                   np.multiply(stress_vox['22'], fitvar22)),
                            np.multiply(stress_vox['23'], fitvar32))
                        first_piola_stress_vox['32'] = np.add(
                            np.add(np.multiply(stress_vox['13'], fitvar12),
                                   np.multiply(stress_vox['23'], fitvar22)),
                            np.multiply(stress_vox['33'], fitvar32))
                        first_piola_stress_vox['13'] = np.add(
                            np.add(np.multiply(stress_vox['11'], fitvar13),
                                   np.multiply(stress_vox['12'], fitvar23)),
                            np.multiply(stress_vox['13'], fitvar33))
                        first_piola_stress_vox['23'] = np.add(
                            np.add(np.multiply(stress_vox['12'], fitvar13),
                                   np.multiply(stress_vox['22'], fitvar23)),
                            np.multiply(stress_vox['23'], fitvar33))
                        first_piola_stress_vox['33'] = np.add(
                            np.add(np.multiply(stress_vox['13'], fitvar13),
                                   np.multiply(stress_vox['23'], fitvar23)),
                            np.multiply(stress_vox['33'], fitvar33))
            else:
                # Compute first Piola-Kirchhoff stress tensor according to
                # hyperelastic constitutive model
                for voxel in it.product(*[list(range(n))
                                          for n in self._n_voxels_dims]):
                    # Initialize deformation gradient
                    def_gradient = np.zeros((self._n_dim, self._n_dim))
                    # Loop over deformation gradient components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel deformation gradient component
                        def_gradient[so_idx] = def_gradient_vox[comp][voxel]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute first Piola-Kirchhoff stress tensor
                    if finite_strains_model == 'stvenant-kirchhoff':
                        # Initialize second Piola-Kirchhoff stress tensor
                        second_piola_stress = np.zeros((self._n_dim,
                                                        self._n_dim))
                        # Loop over second Piola-Kirchhoff stress tensor
                        # components
                        for comp in self._comp_order_sym:
                            # Get second-order array index
                            so_idx = tuple([int(i) - 1 for i in comp])
                            # Get voxel second Piola-Kirchhoff stress tensor
                            # component
                            second_piola_stress[so_idx] = \
                                stress_vox[comp][voxel]
                            if so_idx[0] != so_idx[1]:
                                second_piola_stress[so_idx[::-1]] = \
                                    second_piola_stress[so_idx]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute first Piola-Kirchhoff stress tensor
                        first_piola_stress = np.matmul(def_gradient,
                                                       second_piola_stress)
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
                                kirchhoff_stress[so_idx[::-1]] = \
                                    kirchhoff_stress[so_idx]
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Compute First Piola-Kirchhoff stress tensor
                        first_piola_stress = np.matmul(
                            kirchhoff_stress,
                            np.transpose(np.linalg.inv(def_gradient)))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over First Piola-Kirchhoff stress tensor components
                    for comp in self._comp_order_nsym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Store First Piola-Kirchhoff stress tensor
                        first_piola_stress_vox[comp][voxel] = \
                            first_piola_stress[so_idx]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set First Piola-Kirchhoff stress tensor
            stress_vox = first_piola_stress_vox
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return stress_vox
    # -------------------------------------------------------------------------
    def _stress_div_conv_criterion(self, freqs_dims, stress_DFT_vox):
        """Convergence criterion based on the divergence of the stress tensor.

        Convergence criterion proposed by Moulinec and Suquet (1998) [#]_.

        .. [#] Moulinec, H. and Suquet, P. (1998). *A numerical method for
               computing the overall response of nonlinear composites with
               complex microstructure.* Comp Methods Appl M, 157:69-94 (see
               `here <https://www.sciencedirect.com/science/article/pii/
               S0045782597002181>`_)

        ----

        Parameters
        ----------
        freqs_dims : list[numpy.ndarray (1d)]
            List of discrete frequencies (numpy.ndarray (1d)) associated to
            each spatial dimension.
        stress_DFT_vox : dict
            Discrete Fourier Transform of local stress response (item,
            numpy.ndarray of shape equal to RVE regular grid discretization)
            for each stress component (key, str). Cauchy stress tensor
            (infinitesimal strains) or First Piola-Kirchhoff stress tensor
            (finite strains).

        Returns
        -------
        discrete_error : float
            Discrete error associated to the convergence criterion.
        """
        # Set strain/stress components order according to problem strain
        # formulation
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize discrete error sum
        error_sum = 0.0
        # Initialize stress DFT at the zero-frequency
        stress_DFT_0_mf = np.zeros(len(comp_order), dtype=complex)
        # Initialize stress divergence DFT
        div_stress_DFT = \
            {str(comp + 1): np.zeros(tuple(self._n_voxels_dims), dtype=complex)
             for comp in range(self._n_dim)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get discrete frequency index
            freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x])
                              for x in range(self._n_dim)])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize stress tensor DFT matricial form
            stress_DFT_mf = np.zeros(len(comp_order), dtype=complex)
            # Loop over stress components
            for i in range(len(comp_order)):
                # Get stress component
                comp = comp_order[i]
                # Build stress tensor DFT matricial form
                stress_DFT_mf[i] = mop.kelvin_factor(i, comp_order) \
                    * stress_DFT_vox[comp][freq_idx]
                # Store stress tensor DFT matricial form for zero-frequency
                if freq_idx == self._n_dim*(0,):
                    stress_DFT_0_mf[i] = mop.kelvin_factor(i, comp_order) \
                        * stress_DFT_vox[comp][freq_idx]
            # Build stress tensor DFT
            stress_DFT = mop.get_tensor_from_mf(stress_DFT_mf, self._n_dim,
                                                comp_order)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Add discrete frequency contribution to discrete error sum
            error_sum = error_sum + np.linalg.norm(
                top.dot12_1(1j*np.asarray(freq_coord), stress_DFT))**2
            # Compute stress divergence Discrete Fourier Transform (DFT)
            for i in range(self._n_dim):
                div_stress_DFT[str(i + 1)][freq_idx] = \
                    top.dot12_1(1j*np.asarray(freq_coord), stress_DFT)[i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute discrete error
        discrete_error = \
            np.sqrt(error_sum/n_voxels)/np.linalg.norm(stress_DFT_0_mf)
        # Compute stress divergence Inverse Discrete Fourier Transform (IDFT)
        div_stress = {str(comp + 1): np.zeros(tuple(self._n_voxels_dims))
                      for comp in range(self._n_dim)}
        for i in range(self._n_dim):
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast
            # Fourier Transform (FFT)
            div_stress[str(i + 1)] = \
                np.real(np.fft.ifftn(div_stress_DFT[str(i + 1)]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return discrete_error
    # -------------------------------------------------------------------------
    def _compute_avg_state_vox(self, state_vox):
        """Compute average norm of strain or stress local field.

        Parameters
        ----------
        state_vox : dict
            Local strain or stress response (item, numpy.ndarray of shape equal
            to RVE regular grid discretization) for each strain or stress
            component (key, str).

        Returns
        -------
        avg_state_norm : float
            Average norm of strain or stress local field.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        # Compute total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize average strain or stress norm
        avg_state_norm = 0
        # Loop over strain or stress components
        for i in range(len(comp_order)):
            # Get component
            comp = comp_order[i]
            # Add contribution to average norm
            if self._strain_formulation == 'infinitesimal' \
                    and comp[0] != comp[1]:
                # Account for symmetric component
                avg_state_norm = avg_state_norm \
                    + 2.0*np.square(state_vox[comp])
            else:
                avg_state_norm = avg_state_norm + np.square(state_vox[comp])
        # Compute average norm
        avg_state_norm = np.sum(np.sqrt(avg_state_norm))/n_voxels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return avg_state_norm
    # -------------------------------------------------------------------------
    def _compute_homogenized_field(self, state_vox):
        """Perform homogenization over regular grid spatial discretization.

        Parameters
        ----------
        state_vox : dict
            Local strain or stress response (item, numpy.ndarray of shape equal
            to RVE regular grid discretization) for each strain or stress
            component (key, str).

        Returns
        -------
        hom_state : numpy.ndarray (2d)
            Homogenized strain or stress tensor.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            # Infinitesimal strain tensor and Cauchy stress tensor
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            # Deformation gradient and First Piola-Kirchhoff stress tensor
            comp_order = self._comp_order_nsym
        # Compute total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize homogenized strain or stress tensor
        hom_state = np.zeros((self._n_dim, self._n_dim))
        # Loop over strain or stress tensor components
        for comp in comp_order:
            # Get second-order array index
            so_idx = tuple([int(i) - 1 for i in comp])
            # Assemble strain or stress component
            hom_state[so_idx] = np.sum(state_vox[comp])
            # Account for symmetric component
            if self._strain_formulation == 'infinitesimal' \
                    and comp[0] != comp[1]:
                hom_state[so_idx[::-1]] = np.sum(state_vox[comp])
        # Complete field homogenization
        hom_state = (1.0/n_voxels)*hom_state
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return hom_state
    # -------------------------------------------------------------------------
    def get_hom_stress_strain(self):
        """Get homogenized strain-stress material response.

        Returns
        -------
        _hom_stress_strain : numpy.ndarray (2d)
            RVE homogenized stress-strain response (item, numpy.ndarray (2d))
            for each macroscale strain loading identifier (key, int). The
            homogenized strain and homogenized stress tensor components of the
            i-th loading increment are stored columnwise in the i-th row,
            sorted respectively. Infinitesimal strain tensor and Cauchy stress
            tensor (infinitesimal strains) or Deformation gradient and first
            Piola-Kirchhoff stress tensor (finite strains).
        """
        return copy.deepcopy(self._hom_stress_strain)
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_greetings():
        """Output greetings."""
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        tilde_line, _ = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = ('FFT-Based Homogenization Method (H. Moulinec and P. Suquet)',
                'Implemented by Bernardo Ferreira (bpferreira@fe.up.pt)',
                'Last version: October 2021')
        # Set output template
        template = '\n' + colorama.Fore.WHITE + tilde_line + \
                   colorama.Style.RESET_ALL + colorama.Fore.WHITE + \
                   '\n{:^{width}}\n' + '\n{:^{width}}\n' + \
                   '\n{:^{width}}\n' + colorama.Fore.WHITE + \
                   tilde_line + colorama.Style.RESET_ALL + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        # ioutil.print2(template.format(*info, width=output_width))
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_increment_init(inc, subinc_level, total_lfact, inc_lfact):
        """Output increment initial data.

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
        """
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, _, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if subinc_level == 0:
            # Set output data
            info = (inc, total_lfact, inc_lfact)
            # Set output template
            template = colorama.Fore.CYAN + '\n' \
                + indent + 'Increment number: {:3d}' + '\n' \
                + indent + equal_line[:-len(indent)] + '\n' \
                + indent + 60*' ' + 'Load factor | Total = {:8.1e}' \
                + 7*' ' + '\n' \
                + indent + 72*' ' + '| Incr. = {:8.1e}' \
                + colorama.Style.RESET_ALL + '\n'
        else:
            # Set output data
            info = (inc, subinc_level, total_lfact, inc_lfact)
            # Set output template
            template = colorama.Fore.CYAN + '\n' \
                + indent + 'Increment number: {:3d}' + 3*' ' \
                + '(Sub-inc. level: {:3d})' + '\n' \
                + indent + equal_line[:-len(indent)] + '\n' \
                + indent + 60*' ' + 'Load factor | Total = {:8.1e}' \
                + 7*' ' + '\n' \
                + indent + 72*' ' + '| Incr. = {:8.1e}' \
                + colorama.Style.RESET_ALL + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        # ioutil.print2(template.format(*info, width=output_width))
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_increment_end(strain_formulation, hom_strain, hom_stress,
                               inc_time, total_time):
        """Output increment end data.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        hom_strain : numpy.ndarray (2d)
            Homogenized strain tensor.
        hom_stress : numpy.ndarray (2d)
            Homogenized stress tensor.
        inc_time : float
            Increment running time.
        total_time : float
            Total running time.
        """
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        space_1 = (output_width - 84)*' '
        space_2 = (output_width - (len('Homogenized strain tensor') + 48))*' '
        space_3 = (output_width - (len('Increment run time (s): ') + 44))*' '
        space_4 = (output_width - 72)*' '
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain and stress nomenclature
        if strain_formulation == 'infinitesimal':
            strain_header = 'Homogenized strain tensor (\u03B5)'
            stress_header = 'Homogenized stress tensor (\u03C3)'
        else:
            strain_header = 'Homogenized strain tensor (F)'
            stress_header = 'Homogenized stress tensor (P)'
        # Get problem number of spatial dimensions
        n_dim = hom_strain.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
                       indent + 7*' ' + strain_header + space_2 \
                              + stress_header + '\n\n' + \
                       indent + 6*' ' + ' [' + n_dim*'{:>12.4e}' + '  ]' \
                              + space_4 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + 6*' ' + ' [' + n_dim*'{:>12.4e}' + '  ]' \
                              + space_4 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n'
        else:
            template = indent + dashed_line[:-len(indent)] + '\n\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 7*' ' + strain_header + space_2 \
                              + stress_header + '\n\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + n_dim*'{:>12.4e}' + '  ]' + space_1 + \
                       '[' + n_dim*'{:>12.4e}' + '  ]' + '\n'

        template += '\n' + indent + equal_line[:-len(indent)] + '\n' + \
                    indent + 'Increment run time (s): {:>11.4e}' + space_3 + \
                    'Total run time (s): {:>11.4e}' + '\n\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        # ioutil.print2(template.format(*info, width=output_width))
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_iteration(iter, iter_time, discrete_error):
        """Output iteration data.

        Parameters
        ----------
        iter : int
            Iteration number.
        iter_time : float
            Iteration running time.
        discrete_error : float
            Discrete error associated to convergence criterion.
        """
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, dashed_line, indent, _ = display_features[0:4]
        _, equal_line = display_features[4:6]
        space_1 = (output_width - 29)*' '
        space_2 = (output_width - 35)*' '
        space_3 = (output_width - 38)*' '
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        template = ''
        # Set iteration output header
        if iter == 0:
            template += indent + 5*' ' + 'Iteration' + space_1 \
                               + 'Convergence' '\n' + \
                        indent + ' Number    Run time (s)' + space_2 + \
                        'Error' + '\n' + \
                        indent + dashed_line[:-len(indent)] + '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = (iter, iter_time, discrete_error)
        # Set output template
        template += indent + ' {:^6d}    {:^12.4e}' + space_3 + '{:>11.4e}'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        # ioutil.print2(template.format(*info, width=output_width))
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_increment_cut(cut_msg=''):
        """Output increment cut data.

        Parameters
        ----------
        cut_msg : str
            Increment cut output message.
        """
        # Get display features
        display_features = ioutil.setdisplayfeatures()
        output_width, _, indent, asterisk_line = display_features[0:4]
        _, _ = display_features[4:6]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output data
        info = ()
        # Set output template
        template = '\n\n' + colorama.Fore.RED + indent + \
                   asterisk_line[:-len(indent)] + '\n' + \
                   indent + 'Increment cut: ' + colorama.Style.RESET_ALL + \
                   cut_msg + '\n' + colorama.Fore.RED + indent + \
                   asterisk_line[:-len(indent)] + colorama.Style.RESET_ALL + \
                   '\n'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output data
        print(template.format(*info, width=output_width))
        # ioutil.print2(template.format(*info, width=output_width))
# =============================================================================
class MacroscaleStrainIncrementer:
    """Macroscale strain loading incrementer.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _inc : int
        Increment counter.
    _total_lfact : float
        Total load factor.
    _inc_mac_strain_total : numpy.ndarray (2d)
        Total incremental macroscale strain second-order tensor. Infinitesimal
        strain tensor (infinitesimal strains) or deformation gradient (finite
        strains).
    _mac_strain : numpy.ndarray (2d)
        Current macroscale strain second-order tensor. Infinitesimal strain
        tensor (infinitesimal strains) or deformation gradient (finite
        strains).
    _mac_strain_old : numpy.ndarray (2d)
        Last converged macroscale strain tensor. Infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains).
    _is_last_inc : bool
        Last increment flag.
    _sub_inc_levels : list
        List of increments subincrementation level.
    _n_cinc_cuts : int
        Consecutive increment cuts counter.

    Methods
    -------
    get_inc(self)
        Get current increment counter.
    get_current_mac_strain(self)
        Get current macroscale strain.
    get_is_last_inc(self)
        Get last increment flag.
    get_inc_output_data(self)
        Get increment output data.
    update_inc(self)
        Update increment counter, total load factor and current loading.
    increment_cut(self)
        Perform macroscale strain increment cut.
    _update_mac_strain(self)
        Update current macroscale strain loading.
    """
    def __init__(self, strain_formulation, problem_type, mac_strain_total,
                 mac_strain_init=None, inc_lfacts=[1.0, ], max_subinc_level=5,
                 max_cinc_cuts=5):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        mac_strain_total : numpy.ndarray (2d)
            Total macroscale strain tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).
        mac_strain_init : numpy.ndarray (2d), default=None
            Initial macroscale strain tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).
        inc_lfacts : list[float], default=[1.0,]
            List of incremental load factors (float). Default applies the total
            macroscale strain tensor in a single increment.
        max_subinc_level : int, default=5
            Maximum level of macroscale loading subincrementation.
        max_cinc_cuts : int, default=5
            Maximum number of consecutive macroscale loading increment cuts.
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set total macroscale strain tensor
        self._mac_strain_total = mac_strain_total
        # Set initial macroscale strain tensor
        if mac_strain_init is not None:
            self._mac_strain_init = mac_strain_init
        else:
            if self._strain_formulation == 'infinitesimal':
                self._mac_strain_init = np.zeros((self._n_dim, self._n_dim))
            else:
                self._mac_strain_init = np.eye(self._n_dim)
        # Set total incremental macroscale strain tensor
        if self._strain_formulation == 'infinitesimal':
            # Additive decomposition of infinitesimal strain tensor
            self._inc_mac_strain_total = \
                self._mac_strain_total - self._mac_strain_init
        else:
            # Multiplicative decomposition of deformation gradient
            self._inc_mac_strain_total = np.matmul(
                self._mac_strain_total, np.linalg.inv(self._mac_strain_init))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize increment counter
        self._inc = 0
        # Initialize current macroscale strain tensor
        self._mac_strain = copy.deepcopy(self._mac_strain_init)
        # Initialize last converged macroscale strain
        self._mac_strain_old = copy.deepcopy(self._mac_strain)
        # Set list of incremental load factors
        self._inc_lfacts = copy.deepcopy(inc_lfacts)
        # Initialize subincrementation levels
        self._sub_inc_levels = [0, ]*len(self._inc_lfacts)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._max_subinc_level = max_subinc_level
        self._max_cinc_cuts = max_cinc_cuts
        # Initialize consecutive increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize last increment flag
        self._is_last_inc = False
    # -------------------------------------------------------------------------
    def get_inc(self):
        """Get current increment counter.

        Returns
        -------
        inc : int
            Increment counter.
        """
        return self._inc
    # -------------------------------------------------------------------------
    def get_current_mac_strain(self):
        """Get current macroscale strain.

        Returns
        -------
        mac_strain : 2darray
            Current macroscale strain tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).
        """
        return copy.deepcopy(self._mac_strain)
    # -------------------------------------------------------------------------
    def get_is_last_inc(self):
        """Get last increment flag.

        Returns
        -------
        is_last_inc : bool
            Last increment flag.
        """
        return self._is_last_inc
    # -------------------------------------------------------------------------
    def get_inc_output_data(self):
        """Get increment output data.

        Returns
        -------
        inc_data : tuple
            Increment output data.
        """
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return (self._inc, self._sub_inc_levels[inc_idx], self._total_lfact,
                self._inc_lfacts[inc_idx])
    # -------------------------------------------------------------------------
    def update_inc(self):
        """Update increment counter, total load factor and current loading."""
        # Update last converged macroscale strain
        self._mac_strain_old = copy.deepcopy(self._mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment increment counter
        self._inc += 1
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset consecutive increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Procedure related with the macroscale strain subincrementation: upon
        # convergence of a given increment, guarantee that the following
        # increment magnitude is at most one (subincrementation) level above.
        # The increment cut procedure is performed the required number of times
        # in order to ensure this progressive recovery towards the prescribed
        # incrementation
        if self._inc > 1:
            while self._sub_inc_levels[inc_idx - 1] \
                    - self._sub_inc_levels[inc_idx] >= 2:
                self.increment_cut()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        # Update current macroscale strain
        self._update_mac_strain()
        # Check if last increment
        if self._inc == len(self._inc_lfacts):
            self._is_last_inc = True
    # -------------------------------------------------------------------------
    def increment_cut(self):
        """Perform macroscale strain increment cut."""
        # Get increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment consecutive increment cuts counter
        self._n_cinc_cuts += 1
        # Check if maximum number of consecutive increment cuts is surpassed
        if self._n_cinc_cuts > self._max_cinc_cuts:
            raise RuntimeError('Maximum number of consecutive increments cuts '
                               'reached without convergence.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update subincrementation level
        self._sub_inc_levels[inc_idx] += 1
        self._sub_inc_levels.insert(inc_idx + 1, self._sub_inc_levels[inc_idx])
        # Check if maximum subincrementation level is surpassed
        if self._sub_inc_levels[inc_idx] > self._max_subinc_level:
            raise RuntimeError('Maximum subincrementation level reached '
                               'without convergence.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get current incremental load factor
        inc_lfact = self._inc_lfacts[inc_idx]
        # Cut macroscale strain increment in half
        self._inc_lfacts[inc_idx] = inc_lfact/2.0
        self._inc_lfacts.insert(inc_idx + 1, self._inc_lfacts[inc_idx])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update current macroscale strain
        self._update_mac_strain()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset last increment flag
        self._is_last_inc = False
    # -------------------------------------------------------------------------
    def _update_mac_strain(self):
        """Update current macroscale strain loading."""
        # Get increment index
        inc_idx = self._inc - 1
        # Get current incremental load factor
        inc_lfact = self._inc_lfacts[inc_idx]
        # Compute current macroscale strain loading according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            # Compute incremental macroscale infinitesimal strain tensor
            inc_mac_strain = inc_lfact*self._inc_mac_strain_total
            # Compute current macroscale infinitesimal strain tensor
            self._mac_strain = self._mac_strain_old + inc_mac_strain
        else:
            # Compute incremental macroscale deformation gradient
            inc_mac_strain = mop.matrix_root(self._inc_mac_strain_total,
                                             inc_lfact)
            # Compute current macroscale deformation gradient
            self._mac_strain = np.matmul(inc_mac_strain, self._mac_strain_old)
