"""Adaptive Self-consistent Clustering Analysis (ASCA).

This module includes the implementation of the Adaptive Self-consistent
Clustering Analysis (ASCA), a clustering-based reduced-order model proposed by
Ferreira et. al (2022) [#]_. Under infinitesimal strains, the Self-consistent
Clustering Analysis (SCA) proposed by Liu et. al (2016) [#]_ is recovered in
the absence of clustering adaptivity.

.. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
       *Adaptivity for clustering-based reduced-order modeling of
       localized history-dependent phenomena.* Comp Methods Appl M, 393
       (see `here <https://www.sciencedirect.com/science/article/pii/
       S0045782522000895?via%3Dihub>`_)

.. [#] Liu, Z., Bessa, M., and Liu, W.K. (2016).
       *Self-consistent clustering analysis: An efficient multi-scale scheme
       for inelastic heterogeneous materials.* Comp Methods Appl M, 396:319-341
       (see `here <https://www.sciencedirect.com/science/article/pii/
       S0045782516301499>`_)

The finite strain extension compatible with multiplicative kinematics of both
SCA and ASCA methods is also available as proposed by Ferreira (2022) [#]_
(see Sections 4.7-4.9). However, the development of an accurate self-consistent
scheme compatible with such as a formulation is still under investigation (see
Section 4.9).

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <https://repositorio-aberto.up.pt/handle/10216/
       146900?locale=en>`_)

Besides the main class that implements the aforementioned methods, this
module also includes a class associated with the reference (fictitious)
homogeneous material, which arises in the formulation of the SCA and ASCA,
and an interface to implement any self-consistent scheme, required to determine
the properties of such a reference material.

Classes
-------
ASCA
    Adaptive Self-Consistent Clustering Analysis (ASCA).
ReferenceMaterialOptimizer(ABC)
    Elastic reference material properties optimizer interface.
InfinitesimalRegressionSCS(ReferenceMaterialOptimizer)
    Infinitesimal strains format regression-based self-consistent scheme.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import time
import copy
# Third-party
import numpy as np
import numpy.matlib
import scipy.linalg
# Local
import ioput.info as info
import tensor.matrixoperations as mop
import tensor.tensoroperations as top
from clustering.citoperations import assemble_cit
from online.loading.macloadincrem import LoadingPath, IncrementRewinder, \
                                         RewindManager
from clustering.adaptivity.crve_adaptivity import AdaptivityManager, \
                                                  ClusteringAdaptivityOutput
from ioput.incoutputfiles.homresoutput import HomResOutput
from ioput.incoutputfiles.efftanoutput import EffTanOutput
from ioput.incoutputfiles.refmatoutput import RefMatOutput
from ioput.miscoutputfiles.vtkoutput import VTKOutput
from ioput.miscoutputfiles.voxelsoutput import VoxelsOutput
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class ASCA:
    """Adaptive Self-Consistent Clustering Analysis (ASCA).

    The detailed formulation of this method can be found in
    Ferreira et. al (2022) [#]_ and also in Ferreira (2022) [#]_.

    .. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
           *Adaptivity for clustering-based reduced-order modeling of
           localized history-dependent phenomena.* Comp Methods Appl M, 393
           (see `here <https://www.sciencedirect.com/science/article/pii/
           S0045782522000895?via%3Dihub>`_)

    .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
           Optimization of Thermoplastic Blends: Microstructural
           Generation, Constitutive Development and Clustering-based
           Reduced-Order Modeling.* PhD Thesis, University of Porto
           (see `here <https://repositorio-aberto.up.pt/handle/10216/
           146900?locale=en>`_)

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _global_strain_old_mf : numpy.ndarray (1d)
        Last converged global vector of clusters strains stored in matricial
        form.
    _farfield_strain_old_mf : numpy.ndarray (1d), default=None
        Last converged far-field strain tensor (matricial form).
    _total_time : float
        Total time (s) associated with online-stage.
    _effective_time : float
        Total time (s) associated with the solution of the equilibrium problem.
    _post_process_time : float
        Total time (s) associated with post-processing operations.

    Methods
    -------
    get_time_profile(self)
        Get time profile of online-stage.
    solve_equilibrium_problem(self, crve, material_state, mac_load, \
                              mac_load_presctype, mac_load_increm, \
                              output_dir, problem_name='problem', \
                              clust_adapt_freq=None, \
                              is_solution_rewinding=False, \
                              rewind_state_criterion=None, \
                              rewinding_criterion=None, max_n_rewinds=1, \
                              is_clust_adapt_output=False, \
                              is_ref_material_output=False, \
                              is_vtk_output=False, \
                              vtk_data=None, is_voxels_output=False)
        Solve clustering-based reduced-order equilibrium problem.
    _init_global_strain_mf(self, crve, material_state, mode='last_converged')
        Set clusters strains initial iterative guess.
    _init_global_inc_strain_mf(self, n_total_clusters, mode='last_converged')
        Set clusters incremental strains initial iterative guess.
    _init_farfield_strain_mf(self, mode='last_converged')
        Set far-field strain initial iterative guess.
    _init_inc_farfield_strain_mf(self, mode='last_converged')
        Set incremental far-field strain initial iterative guess.
    _build_residual(self, crve, material_state, presc_strain_idxs, \
                    presc_stress_idxs, applied_mac_load_mf, ref_material, \
                    global_cit_mf, global_strain_mf, farfield_strain_mf=None, \
                    applied_mix_strain_mf=None, applied_mix_stress_mf=None)
        Build Lippmann-Schwinger equilibrium residuals.
    _build_jacobian(self, crve, material_state, presc_strain_idxs, \
                    presc_stress_idxs, global_cit_diff_tangent_mf)
        Build Lippmann-Schwinger equilibrium Jacobian matrix.
    _build_global_cit_diff_tangent_mf(self, crve, global_cit_mf, \
                                      material_state, ref_material)
        Build global cluster interaction - tangent modulus matrix.
    _check_convergence(self, crve, material_state, presc_strain_idxs, \
                       presc_stress_idxs, applied_mac_load_mf, residual, \
                       applied_mix_strain_mf=None)
        Check Lippmann-Schwinger equilibrium convergence.
    _crve_effective_tangent_modulus(self, crve, material_state, \
                                    global_cit_diff_tangent_mf, \
                                    global_strain_mf=None, \
                                    farfield_strain_mf=None)
        CRVE tangent modulus and clusters strain concentration tensors.
    _validate_csct(self, material_phases, phase_clusters, global_csct_mf, \
                   global_strain_mf, farfield_strain_mf)
        Validate clusters strain concentration tensors computation.
    _init_clusters_sct(self, material_phases, phase_clusters)
        Initialize cluster strain concentration tensors.
    _build_clusters_residuals(self, material_phases, phase_clusters, residual)
        Build clusters equilibrium residuals dictionary.
    _display_inc_data(mac_load_path)
        Display loading increment data.
    _display_scs_iter_data(ref_material, is_lock_prop_ref, mode='init', \
                           scs_iter_time=None)
        Display reference material self-consistent scheme iteration data.
    _display_nr_iter_data(mode='init', nr_iter=None, nr_iter_time=None, \
                          errors=[])
        Display Newton-Raphson iteration data.
    _set_output_files(self, output_dir, crve, problem_name='problem', \
                      is_clust_adapt_output=False, \
                      is_ref_material_output=None, \
                      is_vtk_output=False, vtk_data=None, \
                      is_voxels_output=None)
        Create and initialize output files.
    """
    def __init__(self, strain_formulation, problem_type,
                 self_consistent_scheme='regression', scs_parameters=None,
                 scs_max_n_iterations=20, scs_conv_tol=1e-4,
                 max_n_iterations=12, conv_tol=1e-6, max_subinc_level=5,
                 max_cinc_cuts=5, is_adapt_repeat_inc=True):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        self_consistent_scheme : {'regression',}, default='regression'
            Self-consistent scheme to update the elastic reference material
            properties.
        scs_parameters : {dict, None}, default=None
            Self-consistent scheme parameters (key, str; item,
            {int, float, bool}).
        scs_max_n_iterations : int, default=20
            Self-consistent scheme maximum number of iterations.
        scs_conv_tol : float, default=1e-4
            Self-consistent scheme convergence tolerance.
        max_n_iterations : int, default=12
            Newton-Raphson maximum number of iterations.
        conv_tol : float, default=1e-6
            Newton-Raphson convergence tolerance.
        max_subinc_level : int, default=5
            Maximum level of loading subincrementation.
        max_cinc_cuts : int, default=5
            Maximum number of consecutive increment cuts.
        is_adapt_repeat_inc : bool, default=False
            True if loading increment is to be repeated after a clustering
            adaptivity step, False otherwise.
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._self_consistent_scheme = self_consistent_scheme
        self._scs_parameters = scs_parameters
        self._scs_max_n_iterations = scs_max_n_iterations
        self._scs_conv_tol = scs_conv_tol
        self._max_n_iterations = max_n_iterations
        self._conv_tol = conv_tol
        self._max_subinc_level = max_subinc_level
        self._max_cinc_cuts = max_cinc_cuts
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Initialize last converged algorithmic variables
        self._global_strain_old_mf = None
        self._farfield_strain_old_mf = None
        # Initialize times
        self._total_time = 0.0
        self._effective_time = 0.0
        self._post_process_time = 0.0
    # -------------------------------------------------------------------------
    def get_time_profile(self):
        """Get time profile of online-stage.

        Returns
        -------
        total_time : float
            Total time (s) associated with online-stage.
        effective_time : float
            Total time (s) associated with the solution of the equilibrium
            problem.
        post_process_time : float
            Total time (s) associated with post-processing operations.
        """
        return self._total_time, self._effective_time, self._post_process_time
    # -------------------------------------------------------------------------
    def solve_equilibrium_problem(self, crve, material_state, mac_load,
                                  mac_load_presctype, mac_load_increm,
                                  output_dir, problem_name='problem',
                                  clust_adapt_freq=None,
                                  is_solution_rewinding=False,
                                  rewind_state_criterion=None,
                                  rewinding_criterion=None, max_n_rewinds=1,
                                  is_clust_adapt_output=False,
                                  is_ref_material_output=False,
                                  is_vtk_output=False,
                                  vtk_data=None, is_voxels_output=False):
        """Solve clustering-based reduced-order equilibrium problem.

        The overall solution procedure of the Self-consistent Clustering
        Analysis (SCA) under infinitesimal strains is summarized in Boxes C.2
        (Newton-Raphson iterative scheme) and C.3 (self-consistent iterative
        scheme) of Ferreira (2022) [#]_. The finite strain extension compatible
        with multiplicative kinematics can be found in Box 4.2 (Newton-Raphson
        iterative scheme) and the enrichement with clustering adaptivity
        (Adaptive Self-consistent Clustering Analysis (ASCA)) is described in
        Section 4.4.

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        mac_load : dict
            For each loading nature type (key, {'strain', 'stress'}), stores
            the loading constraints for each loading subpath in a
            numpy.ndarray (2d), where the i-th row is associated with the i-th
            strain/stress component and the j-th column is associated with the
            j-th loading subpath.
        mac_load_presctype : numpy.ndarray (2d)
            Loading nature type ({'strain', 'stress'}) associated with each
            loading constraint (numpy.ndarray of shape
            (n_comps, n_load_subpaths)), where the i-th row is associated with
            the i-th strain/stress component and the j-th column is associated
            with the j-th loading subpath.
        mac_load_increm : dict
            For each loading subpath id (key, str), stores a numpy.ndarray of
            shape (n_load_increments, 2) where each row is associated with a
            prescribed loading increment, and the columns 0 and 1 contain the
            corresponding incremental load factor and incremental time,
            respectively.
        output_dir : str
            Absolute directory path of output files.
        problem_name : str, default='problem'
            Problem name.
        clust_adapt_freq : dict, default=None
            Clustering adaptivity frequency (relative to loading
            incrementation) (item, int) associated with each adaptive
            cluster-reduced material phase (key, str).
        is_solution_rewinding : bool, default=False
            Problem solution rewinding flag.
        rewind_state_criterion : tuple, default=None
            Rewind state storage criterion [0] and associated parameter [1].
        rewinding_criterion : tuple, default=None
            Rewinding criterion [0] and associated parameter [1].
        max_n_rewinds : int, default=1
            Maximum number of rewind operations.
        is_clust_adapt_output : bool, default=False
            Clustering adaptivity output flag.
        is_ref_material_output : bool, default=False
            Reference material output flag.
        is_vtk_output : bool, default=False
            VTK output flag.
        vtk_data : dict, default=None
            VTK output file parameters.
        is_voxels_output : bool
            Voxels output file flag.
        """
        #
        #                                                       Initializations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set online-stage initial time
        init_time = time.time()
        # Initialize online-stage post-processing time
        self._post_process_time = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set default strain/stress components order according to problem
        # strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize loading increment cut flag
        is_inc_cut = False
        # Initialize improved cluster incremental strains initial iterative
        # guess flag
        is_improved_init_guess = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize global vector of clusters strain tensors and far-field
        # strain tensor
        if self._strain_formulation == 'infinitesimal':
            # Set clusters infinitesimal strain tensors
            global_strain_mf = np.zeros(
                (crve.get_n_total_clusters()*len(self._comp_order_sym)))
            # Set far-field strain tensor
            farfield_strain_mf = np.zeros(len(self._comp_order_sym))
        else:
            # Set initialized deformation gradient (matricial form)
            def_gradient_mf = np.array([1.0 if x[0] == x[1] else 0.0
                                        for x in self._comp_order_nsym])
            # Build clusters deformation gradients
            global_strain_mf = np.tile(def_gradient_mf,
                                       crve.get_n_total_clusters())
            # Set far-field strain tensor
            farfield_strain_mf = copy.deepcopy(def_gradient_mf)
        # Initialize last converged algorithmic variables
        self._global_strain_old_mf = copy.deepcopy(global_strain_mf)
        self._farfield_strain_old_mf = copy.deepcopy(farfield_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering adaptivity flag
        is_crve_adaptivity = False
        adaptivity_manager = None
        if len(crve.get_adapt_material_phases()) > 0:
            # Switch on clustering adaptivity flag
            is_crve_adaptivity = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set flag that controls if the macroscale loading increment where
            # the clustering adaptivity is triggered is to be repeated
            # considering the new clustering
            is_adapt_repeat_inc = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set clustering adaptivity frequency default
            if clust_adapt_freq is None:
                clust_adapt_freq = {mat_phase: 1 for mat_phase
                                    in crve.get_adapt_material_phases()}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize clustering adaptivity manager
            adaptivity_manager = AdaptivityManager(
                self._strain_formulation, self._problem_type,
                crve.get_adapt_material_phases(), crve.get_phase_clusters(),
                crve.get_adaptivity_control_feature(),
                crve.get_adapt_criterion_data(), clust_adapt_freq)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize increment rewinder manager
        if is_solution_rewinding:
            # Initialize increment rewinder manager
            rewind_manager = RewindManager(
                rewind_state_criterion=rewind_state_criterion,
                rewinding_criterion=rewinding_criterion,
                max_n_rewinds=max_n_rewinds)
            # Initialize increment rewinder flag
            is_inc_rewinder = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clusters state variables
        material_state.init_clusters_state()
        # Initialize clusters strain concentration tensors
        clusters_sct_mf = self._init_clusters_sct(
            material_state.get_material_phases(), crve.get_phase_clusters())
        clusters_sct_old_mf = copy.deepcopy(clusters_sct_mf)
        #
        #                                                          Output files
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create and initialize output files
        hres_output, efftan_output, ref_mat_output, voxels_output, \
            adapt_output, vtk_output = self._set_output_files(
                output_dir, crve, problem_name=problem_name,
                is_clust_adapt_output=is_clust_adapt_output,
                is_ref_material_output=is_ref_material_output,
                is_vtk_output=is_vtk_output, vtk_data=vtk_data,
                is_voxels_output=is_voxels_output)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_vtk_output:
            # Write VTK file associated with the initial state
            info.displayinfo('18')
            vtk_output.write_vtk_file_time_step(
                0, self._strain_formulation, self._problem_type, crve,
                material_state, vtk_vars=vtk_data['vtk_vars'],
                adaptivity_manager=adaptivity_manager)
            # Get VTK output increment divider
            vtk_inc_div = vtk_data['vtk_inc_div']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment post-processing time
        self._post_process_time += time.time() - procedure_init_time
        #
        #                                            Elastic reference material
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate elastic reference material
        ref_material = ElasticReferenceMaterial(self._strain_formulation,
                                                self._problem_type,
                                                self._self_consistent_scheme,
                                                self._scs_conv_tol)
        # Initialize initial value of elastic reference material properties
        scs_init_properties = None
        # Get initial value of elastic reference material properties
        if (self._scs_parameters is not None) \
                and {'E_init', 'v_init'}.issubset(self._scs_parameters.keys()):
            # Get properties specifications
            spec_1 = self._scs_parameters['E_init']
            spec_2 = self._scs_parameters['v_init']
            # Get properties
            if (spec_1 == 'init_eff_tangent' and spec_2 == 'init_eff_tangent'):
                scs_init_properties = \
                    crve.get_eff_isotropic_elastic_constants()
            else:
                scs_init_properties = {}
                scs_init_properties['E'] = self._scs_parameters['E_init']
                scs_init_properties['v'] = self._scs_parameters['v_init']
        # Set initial value of elastic reference material properties
        ref_material.init_material_properties(
            material_state.get_material_phases(),
            material_state.get_material_phases_properties(),
            material_state.get_material_phases_vf(),
            properties=scs_init_properties)
        # Initialize reference material elastic properties locking flag
        if self._self_consistent_scheme == 'none':
            is_lock_prop_ref = True
        else:
            is_lock_prop_ref = False
        #
        #                                                          Loading path
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate macroscale loading path
        mac_load_path = LoadingPath(
            self._strain_formulation, self._problem_type, mac_load,
            mac_load_presctype, mac_load_increm,
            max_subinc_level=self._max_subinc_level,
            max_cinc_cuts=self._max_cinc_cuts)
        # Set initial homogenized state
        mac_load_path.update_hom_state(material_state.get_hom_strain_mf(),
                                       material_state.get_hom_stress_mf())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get increment counter
        inc = mac_load_path.get_increm_state()['inc']
        # Save macroscale loading increment (converged) state
        if is_solution_rewinding and rewind_manager.is_rewind_available() \
                and rewind_manager.is_save_rewind_state(inc):
            # Set reference rewind time
            rewind_manager.update_rewind_time(mode='init')
            # Instantiate increment rewinder
            inc_rewinder = IncrementRewinder(
                rewind_inc=inc, phase_clusters=crve.get_phase_clusters())
            # Save loading path state
            inc_rewinder.save_loading_path(loading_path=mac_load_path)
            # Save material constitutive state
            inc_rewinder.save_material_state(material_state)
            # Save elastic reference material
            inc_rewinder.save_reference_material(ref_material)
            # Save clusters strain concentration tensors
            inc_rewinder.save_clusters_sct(clusters_sct_mf)
            # Save algorithmic variables
            inc_rewinder.save_asca_algorithmic_variables(global_strain_mf,
                                                         farfield_strain_mf)
            # Set increment rewinder flag
            is_inc_rewinder = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Setup first macroscale loading increment
        applied_mac_load_mf, inc_mac_load_mf, n_presc_strain, \
            presc_strain_idxs, n_presc_stress, presc_stress_idxs, \
            is_last_inc = mac_load_path.new_load_increment()
        # Get increment counter
        inc = mac_load_path.get_increm_state()['inc']
        # Display increment data
        type(self)._display_inc_data(mac_load_path)
        # Set increment initial time
        inc_init_time = time.time()
        # ---------------------------------------------------------------------
        # Start incremental loading loop
        while True:
            #
            #                           Clusters strain initial iterative guess
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set clusters strain initial iterative guess
            global_strain_mf = \
                self._init_global_strain_mf(crve, material_state)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set far-field strain initial iterative guess
            farfield_strain_mf = self._init_farfield_strain_mf()
            #
            #                             Self-consistent scheme iterative loop
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize self-consistent scheme iteration counter
            ref_material.init_scs_iter()
            # Display reference material self-consistent scheme iteration data
            type(self)._display_scs_iter_data(ref_material, is_lock_prop_ref,
                                              mode='init')
            # Set self-consistent scheme iteration initial time
            scs_iter_init_time = time.time()
            # -----------------------------------------------------------------
            # Start self-consistent scheme iterative loop
            while True:
                #
                #                    Global cluster interaction matrix assembly
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update cluster interaction tensors with elastic reference
                # material properties and assemble global cluster interaction
                # matrix
                global_cit_mf = assemble_cit(
                    self._strain_formulation, self._problem_type,
                    ref_material.get_material_properties(),
                    material_state.get_material_phases(),
                    crve.get_phase_n_clusters(), crve.get_phase_clusters(),
                    crve.get_cit_x_mf()[0], crve.get_cit_x_mf()[1],
                    crve.get_cit_x_mf()[2])
                #
                #                                 Newton-Raphson iterative loop
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize Newton-Raphson iteration counter
                nr_iter = 0
                # Display Newton-Raphson iteration header
                type(self)._display_nr_iter_data(mode='init')
                # Set Newton-Raphson iteration initial time
                nr_iter_init_time = time.time()
                # -------------------------------------------------------------
                # Start Newton-Raphson iterative loop
                while True:
                    #
                    #                         Cluster material state update and
                    #                                consistent tangent modulus
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get clusters incremental strain in matricial form
                    clusters_inc_strain_mf = \
                        material_state.get_clusters_inc_strain_mf(
                            global_strain_mf)
                    # Perform clusters material state update and compute
                    # associated consistent tangent modulus
                    su_fail_state = material_state.update_clusters_state(
                        clusters_inc_strain_mf)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Raise increment cut procedure if material cluster state
                    # update failed
                    if su_fail_state['is_su_fail']:
                        # Set increment cut flag
                        is_inc_cut = True
                        # Display increment cut
                        info.displayinfo('11', 'su_fail', su_fail_state)
                        # Leave Newton-Raphson equilibrium iterative loop
                        break
                    #
                    #                     Homogenized strain and stress tensors
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update homogenized strain and stress tensors
                    material_state.update_state_homogenization()
                    #
                    #        Global cluster interaction - tangent moduli matrix
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute global cluster interaction - tangent moduli
                    # matrix
                    global_cit_diff_tangent_mf = \
                        self._build_global_cit_diff_tangent_mf(
                            crve, global_cit_mf, material_state, ref_material)
                    #
                    #                  Lippmann-Schwinger equilibrium residuals
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Build Lippmann-Schwinger equilibrium residuals
                    residual = self._build_residual(
                        crve, material_state,
                        presc_strain_idxs, presc_stress_idxs,
                        applied_mac_load_mf, ref_material, global_cit_mf,
                        global_strain_mf,
                        inc_mac_load_mf=inc_mac_load_mf,
                        farfield_strain_mf=farfield_strain_mf,
                        farfield_strain_old_mf=self._farfield_strain_old_mf)
                    #
                    #                                    Convergence evaluation
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Evaluate Lippmann-Schwinger equilibrium solution
                    # convergence
                    is_converged, conv_errors = self._check_convergence(
                        crve, material_state, presc_strain_idxs,
                        presc_stress_idxs, applied_mac_load_mf, residual)
                    # Display Newton-Raphson iteration header
                    type(self)._display_nr_iter_data(
                        mode='iter', nr_iter=nr_iter,
                        nr_iter_time=time.time()-nr_iter_init_time,
                        errors=conv_errors)
                    #
                    #                             Newton-Raphson iterative flow
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Control Newton-Raphson iteration loop flow
                    if is_converged:
                        # Leave Newton-Raphson iterative loop (converged
                        # solution)
                        break
                    elif nr_iter == self._max_n_iterations:
                        # Raise macroscale increment cut procedure
                        is_inc_cut = True
                        # Display increment cut
                        info.displayinfo('11', 'max_iter',
                                         self._max_n_iterations)
                        # Leave Newton-Raphson equilibrium iterative loop
                        break
                    else:
                        # Increment iteration counter
                        nr_iter = nr_iter + 1
                        # Set Newton-Raphson iteration initial time
                        nr_iter_init_time = time.time()
                    #
                    #            Lippmann-Schwinger equilibrium Jacobian matrix
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Build Lippmann-Schwinger equilibrium Jacobian matrix
                    jacobian = self._build_jacobian(crve, material_state,
                                                    presc_strain_idxs,
                                                    presc_stress_idxs,
                                                    global_cit_diff_tangent_mf)
                    #
                    #                   Lippmann-Schwinger equilibrium solution
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Solve Lippmann-Schwinger equilibrium system of linearized
                    # equilibrium equations
                    d_iter = numpy.linalg.solve(jacobian, -residual)
                    #
                    #                                  Strains iterative update
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update clusters incremental strain
                    global_strain_mf = global_strain_mf + \
                        d_iter[0:crve.get_n_total_clusters()*len(comp_order)]
                    # Update far-field strain
                    farfield_strain_mf = farfield_strain_mf + d_iter[
                        crve.get_n_total_clusters()*len(comp_order):]
                #
                #                                        Self-consistent scheme
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # If raising a loading increment cut, leave self-consistent
                # iterative loop
                if is_inc_cut:
                    break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute CRVE effective tangent modulus and clusters strain
                # concentration tensors
                eff_tangent_mf, clusters_sct_mf = \
                    self._crve_effective_tangent_modulus(
                        crve, material_state, global_cit_diff_tangent_mf)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Set post-processing procedure initial time
                procedure_init_time = time.time()
                # Output reference material associated quantities (.refm file)
                if is_ref_material_output:
                    ref_mat_output.write_file(
                        inc, ref_material,
                        material_state.get_hom_strain_mf(),
                        material_state.get_hom_stress_mf(),
                        farfield_strain_mf, applied_mac_load_mf['strain'],
                        eff_tangent_mf=eff_tangent_mf)
                    # Increment post-processing time
                    self._post_process_time += \
                        time.time() - procedure_init_time
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Solve self-consistent minimization problem
                if is_lock_prop_ref:
                    # Skip update of reference material elastic properties
                    E_ref = ref_material.get_material_properties()['E']
                    v_ref = ref_material.get_material_properties()['v']
                else:
                    # Compute reference material elastic properties through the
                    # solution of a self-consistent minimization problem
                    is_scs_admissible, E_ref, v_ref = \
                        ref_material.self_consistent_update(
                            material_state.get_hom_strain_mf(),
                            material_state.get_hom_strain_old_mf(),
                            material_state.get_hom_stress_mf(),
                            material_state.get_hom_stress_old_mf(),
                            eff_tangent_mf)
                    # If self-consistent scheme iterative solution is not
                    # admissible, either accept the current solution (first
                    # self-consistent scheme iteration) or perform one last
                    # self-consistent scheme iteration with the last converged
                    # increment reference material elastic properties
                    if not is_scs_admissible:
                        # Display reference material self-consistent scheme
                        # iteration footer
                        type(self)._display_scs_iter_data(
                            ref_material, is_lock_prop_ref, mode='end',
                            scs_iter_time=time.time() - scs_iter_init_time)
                        # Elastic reference material properties locking
                        if ref_material.get_scs_iter() == 0:
                            # Display locking of reference material elastic
                            # properties
                            info.displayinfo('14', 'locked_scs_solution')
                            # Leave self-consistent scheme iterative loop
                            # (accepted solution)
                            break
                        else:
                            # Display locking of reference material elastic
                            # properties
                            info.displayinfo('14', 'inadmissible_scs_solution')
                            # If inadmissible self-consistent scheme solution,
                            # reset reference material elastic properties to
                            # the last converged increment values
                            ref_material.reset_material_properties()
                            # Lock reference material elastic properties
                            is_lock_prop_ref = True
                            # Perform one last self-consistent scheme iteration
                            # with the last converged increment reference
                            # material elastic properties
                            type(self)._display_scs_iter_data(
                                ref_material, is_lock_prop_ref, mode='init')
                            # Proceed to last self-consistent scheme iteration
                            continue
                #
                #                                        Convergence evaluation
                #                                      (self-consistent scheme)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check self-consistent scheme iterative solution convergence
                is_scs_converged = \
                    ref_material.check_scs_convergence(E_ref, v_ref)
                # Display reference material self-consistent scheme iteration
                # footer
                type(self)._display_scs_iter_data(
                    ref_material, is_lock_prop_ref, mode='end',
                    scs_iter_time=time.time()-scs_iter_init_time)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Self-consistent scheme iterative flow
                if is_scs_converged:
                    # Reset flag that locks reference material elastic
                    # properties
                    if self._self_consistent_scheme != 'none':
                        is_lock_prop_ref = False
                    # Leave self-consistent scheme iterative loop (converged
                    # solution)
                    break
                elif ref_material.get_scs_iter() == self._scs_max_n_iterations:
                    # Display locking of reference material elastic properties
                    info.displayinfo('14', 'max_scs_iter',
                                     self._scs_max_n_iterations)
                    # If the maximum number of self-consistent scheme
                    # iterations is reached without convergence, reset elastic
                    # reference material properties to last loading increment
                    # values
                    ref_material.reset_material_properties()
                    # Lock reference material elastic properties
                    is_lock_prop_ref = True
                    # Perform one last self-consistent scheme iteration with
                    # the last converged increment reference material elastic
                    # properties
                    type(self)._display_scs_iter_data(
                        ref_material, is_lock_prop_ref, mode='init')
                else:
                    # Update reference material elastic properties
                    ref_material.update_material_properties(E_ref, v_ref)
                    # Increment self-consistent scheme iteration counter
                    ref_material.update_scs_iter()
                    # Display reference material self-consistent scheme
                    # iteration data
                    type(self)._display_scs_iter_data(
                        ref_material, is_lock_prop_ref, mode='init')
                    # Set self-consistent scheme iteration initial time
                    scs_iter_init_time = time.time()
            #
            #                                             Loading increment cut
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_inc_cut:
                # Reset loading increment cut flag
                is_inc_cut = False
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Reset elastic reference material properties to last loading
                # increment values
                ref_material.reset_material_properties()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform loading increment cut and setup new loading increment
                applied_mac_load_mf, inc_mac_load_mf, n_presc_strain, \
                    presc_strain_idxs, n_presc_stress, presc_stress_idxs, \
                    is_last_inc = mac_load_path.increment_cut(self._n_dim,
                                                              comp_order)
                # Get increment counter
                inc = mac_load_path.get_increm_state()['inc']
                # Display increment data
                type(self)._display_inc_data(mac_load_path)
                # Set increment initial time
                inc_init_time = time.time()
                # Start new loading increment solution procedures
                continue
            #
            #                                             Clustering adaptivity
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # This section should only be executed if the loading increment
            # where the clustering adaptivity condition is triggered is to be
            # repeated considering the new clustering
            if is_crve_adaptivity and is_adapt_repeat_inc \
                    and adaptivity_manager.check_inc_adaptive_steps(inc):
                # Display increment data
                if is_clust_adapt_output:
                    info.displayinfo('12', crve.get_adaptive_step() + 1)
                # Build clusters equilibrium residuals
                clusters_residuals_mf = self._build_clusters_residuals(
                    material_state.get_material_phases(),
                    crve.get_phase_clusters(), residual)
                # Get clustering adaptivity trigger condition and target
                # clusters
                is_trigger, target_clusters, target_clusters_data = \
                    adaptivity_manager.get_target_clusters(
                        crve.get_phase_clusters(), crve.get_voxels_clusters(),
                        material_state.get_clusters_state(),
                        material_state.get_clusters_def_gradient_mf(),
                        material_state.get_clusters_def_gradient_old_mf(),
                        material_state.get_clusters_state_old(),
                        clusters_sct_mf, clusters_sct_old_mf,
                        clusters_residuals_mf, inc,
                        verbose=is_clust_adapt_output)
                # Perform clustering adaptivity if adaptivity condition is
                # triggered
                if is_trigger:
                    # Display clustering adaptivity
                    info.displayinfo('16', 'repeat', inc)
                    # Set improved initial iterative guess for the clusters
                    # strain global vector (matricial form) after the
                    # clustering adaptivity
                    is_improved_init_guess = True
                    improved_init_guess = \
                        [is_improved_init_guess, global_strain_mf]
                    # Perform clustering adaptivity
                    adaptivity_manager.adaptive_refinement(
                        crve, material_state, target_clusters,
                        target_clusters_data, inc,
                        improved_init_guess=improved_init_guess,
                        verbose=is_clust_adapt_output)
                    # Get improved initial iterative guess for the clusters
                    # strain global vector (matricial form)
                    if is_improved_init_guess:
                        global_strain_mf = improved_init_guess[1]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get increment counter
                    inc = mac_load_path.get_increm_state()['inc']
                    # Display increment data
                    type(self)._display_inc_data(mac_load_path)
                    # Set increment initial time
                    inc_init_time = time.time()
                    # Start new loading increment solution
                    continue
            #
            #                                                Solution rewinding
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check rewind operations availability
            if is_solution_rewinding and is_inc_rewinder \
                    and rewind_manager.is_rewind_available():
                # Check analysis rewind criteria
                is_rewind = rewind_manager.is_rewinding_criteria(
                    inc, material_state.get_material_phases(),
                    crve.get_phase_clusters(),
                    material_state.get_clusters_state())
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Rewind analysis if criteria are met
                if is_rewind:
                    info.displayinfo('17', inc_rewinder.get_rewind_inc())
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rewind loading path
                    mac_load_path = inc_rewinder.get_loading_path()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rewind material constitutive state
                    material_state = inc_rewinder.get_material_state(crve)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rewind elastic reference material
                    ref_material = inc_rewinder.get_reference_material()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rewind clusters strain concentration tensors
                    clusters_sct_old_mf = inc_rewinder.get_clusters_sct()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Rewind algorithmic variables
                    self._global_strain_old_mf, \
                        self._farfield_strain_old_mf = \
                        inc_rewinder.get_asca_algorithmic_variables()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Set post-processing procedure initial time
                    procedure_init_time = time.time()
                    # Rewind output files
                    inc_rewinder.rewind_output_files(
                        hres_output, efftan_output, ref_mat_output,
                        voxels_output, adapt_output, vtk_output)
                    # Increment post-processing time
                    self._post_process_time += \
                        time.time() - procedure_init_time
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Reset clustering adaptive steps
                    if is_crve_adaptivity:
                        adaptivity_manager.clear_inc_adaptive_steps(
                            inc_threshold=inc_rewinder.get_rewind_inc())
                    # Update total rewind time
                    rewind_manager.update_rewind_time(mode='update')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Setup new loading increment
                    applied_mac_load_mf, inc_mac_load_mf, n_presc_strain, \
                        presc_strain_idxs, n_presc_stress, presc_stress_idxs, \
                        is_last_inc = mac_load_path.new_load_increment()
                    # Get increment counter
                    inc = mac_load_path.get_increm_state()['inc']
                    # Display increment data
                    type(self)._display_inc_data(mac_load_path)
                    # Set increment initial time
                    inc_init_time = time.time()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Start new loading increment solution
                    continue
            #
            #                             Homogenized strain and stress tensors
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get homogenized strain and stress tensors (matricial form)
            hom_strain_mf = material_state.get_hom_strain_mf()
            hom_stress_mf = material_state.get_hom_stress_mf()
            # Build homogenized strain and stress tensors
            hom_strain = mop.get_tensor_from_mf(hom_strain_mf, self._n_dim,
                                                comp_order)
            hom_stress = mop.get_tensor_from_mf(hom_stress_mf, self._n_dim,
                                                comp_order)
            # Get homogenized strain or stress tensor out-of-plane component
            # (output only)
            if self._problem_type == 1:
                hom_stress_33 = material_state.get_oop_hom_comp()
            #
            #                        Increment homogenized results (.hres file)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize homogenized results dictionary
            hom_results = dict()
            # Build homogenized results dictionary
            hom_results = {'hom_strain': hom_strain, 'hom_stress': hom_stress}
            if self._problem_type == 1:
                hom_results['hom_stress_33'] = hom_stress_33
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Write increment homogenized results (.hres)
            hres_output.write_file(
                self._strain_formulation, self._problem_type, mac_load_path,
                hom_results, time.time() - init_time - self._post_process_time)
            # Write increment CRVE effective tangent modulus (.efftan)
            efftan_output.write_file(
                self._strain_formulation, self._problem_type, mac_load_path,
                eff_tangent_mf)
            # Increment post-processing time
            self._post_process_time += time.time() - procedure_init_time
            #
            #                                                Increment VTK file
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Write VTK file associated with the loading increment
            if is_vtk_output and inc % vtk_inc_div == 0:
                # Set post-processing procedure initial time
                procedure_init_time = time.time()
                # Write VTK file associated with the converged increment
                info.displayinfo('18')
                vtk_output.write_vtk_file_time_step(
                    inc, self._strain_formulation, self._problem_type, crve,
                    material_state, vtk_vars=vtk_data['vtk_vars'],
                    adaptivity_manager=adaptivity_manager)
                # Increment post-processing time
                self._post_process_time += time.time() - procedure_init_time
            #
            #                                Converged material state variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update last converged material state variables
            material_state.update_converged_state()
            # Update last converged clusters strain concentration tensors
            clusters_sct_old_mf = copy.deepcopy(clusters_sct_mf)
            #
            #           Converged elastic reference material elastic properties
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set reference material properties converged in the first loading
            # increment
            if inc == 1:
                ref_material.set_material_properties_scs_init()
            # Update converged elastic reference material elastic properties
            ref_material.update_converged_material_properties()
            #
            #                                   Converged algorithmic variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update last converged global vector of clusters strain tensors
            self._global_strain_old_mf = copy.deepcopy(global_strain_mf)
            # Update last converged far-field strain tensor
            self._farfield_strain_old_mf = copy.deepcopy(farfield_strain_mf)
            #
            #                        Clustering adaptivity output (.adapt file)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update clustering adaptivity output file with converged increment
            # data
            if is_crve_adaptivity:
                # Set post-processing procedure initial time
                procedure_init_time = time.time()
                # Update clustering adaptivity output file
                adapt_output.write_adapt_file(inc, adaptivity_manager, crve,
                                              mode='increment')
                # Increment post-processing time
                self._post_process_time += time.time() - procedure_init_time
            #
            #       Voxels cluster state based quantities output (.voxout file)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update clustering adaptivity output file with converged increment
            # data
            if is_voxels_output:
                # Set post-processing procedure initial time
                procedure_init_time = time.time()
                # Update clustering adaptivity output file
                voxels_output.write_voxels_output_file(
                    self._n_dim, comp_order, crve,
                    material_state.get_clusters_state(),
                    material_state.get_clusters_def_gradient_mf())
                # Increment post-processing time
                self._post_process_time += time.time() - procedure_init_time
            #
            #                               Incremental macroscale loading flow
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update converged macroscale (homogenized) state
            mac_load_path.update_hom_state(material_state.get_hom_strain_mf(),
                                           material_state.get_hom_stress_mf())
            # Display converged increment data
            if self._problem_type == 1:
                info.displayinfo(
                    '7', 'end', self._strain_formulation, self._problem_type,
                    hom_strain, hom_stress, time.time() - inc_init_time,
                    time.time() - init_time, hom_stress_33)
            else:
                info.displayinfo(
                    '7', 'end', self._strain_formulation, self._problem_type,
                    hom_strain, hom_stress, time.time() - inc_init_time,
                    time.time() - init_time)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Save macroscale loading increment (converged) state
            if is_solution_rewinding and rewind_manager.is_rewind_available() \
                    and rewind_manager.is_save_rewind_state(inc):
                # Set reference rewind time
                rewind_manager.update_rewind_time(mode='init')
                # Instantiate increment rewinder
                inc_rewinder = IncrementRewinder(
                    rewind_inc=inc, phase_clusters=crve.get_phase_clusters())
                # Save loading path state
                inc_rewinder.save_loading_path(loading_path=mac_load_path)
                # Save material constitutive state
                inc_rewinder.save_material_state(material_state)
                # Save elastic reference material
                inc_rewinder.save_reference_material(ref_material)
                # Save clusters strain concentration tensors
                inc_rewinder.save_clusters_sct(clusters_sct_mf)
                # Save algorithmic variables
                inc_rewinder.save_asca_algorithmic_variables(
                    global_strain_mf, farfield_strain_mf=farfield_strain_mf)
                # Set increment rewinder flag
                is_inc_rewinder = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return if last loading increment, otherwise setup new loading
            # increment
            if is_last_inc:
                # Set total time associated with online-stage
                self._total_time = time.time() - init_time
                # Set total time associated with the solution of the
                # equilibrium problem
                self._effective_time = self._total_time \
                    - self._post_process_time
                # Output clustering adaptivity summary
                if is_crve_adaptivity:
                    info.displayinfo('15', adaptivity_manager, crve,
                                     self._effective_time)
                # Finish solution of clustering-based reduced order equilibrium
                # problem
                return
            else:
                # Setup new loading increment
                applied_mac_load_mf, inc_mac_load_mf, n_presc_strain, \
                    presc_strain_idxs, n_presc_stress, presc_stress_idxs, \
                    is_last_inc = mac_load_path.new_load_increment()
                # Get increment counter
                inc = mac_load_path.get_increm_state()['inc']
                # Display increment data
                type(self)._display_inc_data(mac_load_path)
                # Set increment initial time
                inc_init_time = time.time()
    # -------------------------------------------------------------------------
    def _init_global_strain_mf(self, crve, material_state,
                               mode='last_converged'):
        """Set clusters strains initial iterative guess.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state at rewind state.
        mode : {'last_converged',}, default='last_converged'
            Strategy to set clusters incremental strains initial iterative
            guess.

        Returns
        -------
        global_strain_mf : numpy.ndarray (1d)
            Global vector of clusters strains stored in matricial form.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = material_state.get_material_phases()
        # Get clusters associated with each material phase
        phase_clusters = crve.get_phase_clusters()
        # Get total number of clusters
        n_total_clusters = crve.get_n_total_clusters()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set clusters strain initial iterative guess associated with the last
        # converged solution
        if mode == 'last_converged':
            # Initialize initial iterative guess
            global_strain_mf = np.zeros((n_total_clusters*len(comp_order)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get last converged clusters state variables
            clusters_state_old = material_state.get_clusters_state_old()
            # Get last converged clusters deformation gradient
            clusters_def_gradient_old_mf = \
                material_state.get_clusters_def_gradient_old_mf()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize material cluster strain range indexes
            i_init = 0
            i_end = i_init + len(comp_order)
            # Loop over material phases
            for mat_phase in material_phases:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Get last converged material cluster infinitesimal strain
                    # tensor (infinitesimal strains) or deformation gradient
                    # (finite strains)
                    if self._strain_formulation == 'infinitesimal':
                        strain_old_mf = \
                            clusters_state_old[str(cluster)]['strain_mf']
                    else:
                        strain_old_mf = \
                            clusters_def_gradient_old_mf[str(cluster)]      
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
                    # Assemble to initial iterative guess
                    global_strain_mf[i_init:i_end] = strain_old_mf
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update cluster strain range indexes
                    i_init = i_init + len(comp_order)
                    i_end = i_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unavailable strategy to set clusters strains '
                               'initial iterative guess.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return global_strain_mf
    # -------------------------------------------------------------------------
    def _init_global_inc_strain_mf(self, n_total_clusters,
                                   mode='last_converged'):
        """Set clusters incremental strains initial iterative guess.

        Parameters
        ----------
        n_total_clusters : int
            Total number of clusters.
        mode : {'last_converged',}, default='last_converged'
            Strategy to set clusters incremental strains initial iterative
            guess.

        Returns
        -------
        global_inc_strain_mf : numpy.ndarray (1d)
            Global vector of clusters incremental strains stored in matricial
            form.
        """
        if mode == 'last_converged':
            # Set incremental initial iterative guess associated with the last
            # converged solution
            if self._strain_formulation == 'infinitesimal':
                # Set clusters infinitesimal strain tensors
                global_inc_strain_mf = \
                    np.zeros((n_total_clusters*len(self._comp_order_sym)))
            else:
                # Set initialized deformation gradient (matricial form)
                def_gradient_mf = np.array([1.0 if x[0] == x[1] else 0.0
                                            for x in self._comp_order_nsym])
                # Build clusters deformation gradients
                global_inc_strain_mf = np.tile(def_gradient_mf,
                                               n_total_clusters)
        else:
            raise RuntimeError('Unavailable strategy to set clusters '
                               'incremental strains initial iterative guess.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return global_inc_strain_mf
    # -------------------------------------------------------------------------
    def _init_farfield_strain_mf(self, mode='last_converged'):
        """Set far-field strain initial iterative guess.

        Parameters
        ----------
        mode : {'last_converged',}, default='last_converged'
            Strategy to set incremental far-field strain initial iterative
            guess.

        Returns
        -------
        farfield_strain_mf : 1darray, default=None
            Incremental far-field strain tensor (matricial form).
        """
        if mode == 'last_converged':
            # Set initial iterative guess associated with the last converged
            # solution
            farfield_strain_mf = copy.deepcopy(self._farfield_strain_old_mf)
        else:
            raise RuntimeError('Unavailable strategy to set far-field strain '
                               'initial iterative guess.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return farfield_strain_mf
    # -------------------------------------------------------------------------
    def _init_inc_farfield_strain_mf(self, mode='last_converged'):
        """Set incremental far-field strain initial iterative guess.

        Parameters
        ----------
        mode : {'last_converged',}, default='last_converged'
            Strategy to set incremental far-field strain initial iterative
            guess.

        Returns
        -------
        inc_farfield_strain_mf : numpy.ndarray (1d), default=None
            Incremental far-field strain tensor (matricial form).
        """
        if mode == 'last_converged':
            # Set incremental initial iterative guess associated with the last
            # converged solution
            if self._strain_formulation == 'infinitesimal':
                # Set far-field infinitesimal strain tensor
                inc_farfield_strain_mf = np.zeros(len(self._comp_order_sym))
            else:
                # Set far-field deformation gradient
                inc_farfield_strain_mf = \
                    np.array([1.0 if x[0] == x[1] else 0.0
                              for x in self._comp_order_nsym])
        else:
            raise RuntimeError('Unavailable strategy to set incremental '
                               'far-field strain initial iterative guess.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_farfield_strain_mf
    # -------------------------------------------------------------------------
    def _build_residual(self, crve, material_state, presc_strain_idxs,
                        presc_stress_idxs, applied_mac_load_mf, ref_material,
                        global_cit_mf, global_strain_mf, inc_mac_load_mf=None,
                        farfield_strain_mf=None, farfield_strain_old_mf=None):
        """Build Lippmann-Schwinger equilibrium residuals.

        **Global residual function:**

            .. math::

               \\boldsymbol{R}_{n+1} =
               \\begin{bmatrix}
               \\boldsymbol{R}^{(I)}_{n+1} \\\\[5pt]
               \\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}
               \\end{bmatrix} =
               \\begin{bmatrix} \\boldsymbol{0} \\\\[5pt]
               \\boldsymbol{0}  \\end{bmatrix} \\, , \\qquad
               \\forall I = 1,2, \\, \\dots, \\, n_{\\text{c}} \\, ,

            where :math:`\\boldsymbol{R}_{n+1}` is the global residual
            function, :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th
            material cluster equilibrium residual function,
            :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the strain
            and/or stress loading constraints residual function,
            :math:`n_{c}` is the number of material clusters,
            and :math:`n+1` denotes the current increment.

        ----

        **Infinitesimal strains (incremental equilibrium formulation, \
                                 incremental primary unknowns):**

            *Equilibrium residuals*

                .. math::

                   \\boldsymbol{R}^{(I)}_{n+1} =
                   \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}
                   + \\sum^{n_{\\text{c}}}_{J=1}
                   \\boldsymbol{\\mathsf{T}}^{(I)(J)} : \\left(
                   \\Delta \\hat{\\boldsymbol{\\sigma}}_{\\mu,\\,n+1}^{(J)}
                   - \\boldsymbol{\\mathsf{D}}^{e,\\, 0}:
                   \\Delta\\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(J)}
                   \\right)
                   - \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0}\\, ,

                .. math::

                   \\forall I = 1, \\, \\dots, \\, n_{\\text{c}} \\, ,

                where :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th
                material cluster equilibrium residual function,
                :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}`
                is the :math:`I` th material cluster incremental infinitesimal
                strain tensor, :math:`\\boldsymbol{\\mathsf{T}}^{(I)(J)}` is
                the cluster interaction tensor (fourth-order tensor) between
                the :math:`I` th and :math:`J` th material clusters,
                :math:`\\Delta \\boldsymbol{\\sigma}_{\\mu,\\,n+1}^{(J)}` is
                the :math:`J` th material cluster incremental Cauchy stress
                tensor (:math:`\\hat{(\\cdot)}` denotes the incremental nature
                of the constitutive function),
                :math:`\\boldsymbol{\\mathsf{D}}^{e,\\, 0}` is the elastic
                tangent modulus of the reference homogeneous material,
                :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0}` is
                the incremental far-field infinitesimal strain tensor,
                :math:`n_{c}` is the number of material clusters, and
                :math:`n+1` denotes the current increment.


            *Loading (homogenization-based) strain and/or stress constraints*

                .. math::

                   \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                   \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                   \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}
                   - \\Delta \\boldsymbol{\\varepsilon}_{n+1} \\, ,

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                strain loading constraint residual function, :math:`f^{(I)}` is
                the volume fraction of the :math:`I` th material cluster,
                :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}`
                is the :math:`I` th material cluster incremental infinitesimal
                strain tensor, :math:`\\Delta \\boldsymbol{\\varepsilon}_{n+1}`
                is the macroscale incremental Cauchy stress tensor,
                :math:`n_{c}` is the number of material clusters, and
                :math:`n+1` denotes the current increment.

                .. math::

                    \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                    \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                    \\Delta \\hat{\\boldsymbol{\\sigma}}_{\\mu,\\,n+1}^{(I)}
                    - \\Delta \\boldsymbol{\\sigma}_{n+1} \\, ,

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                stress loading constraint residual function,
                :math:`f^{(I)}` is the volume fraction of the :math:`I` th
                material cluster,
                :math:`\\Delta \\boldsymbol{\\sigma}_{\\mu,\\,n+1}^{(I)}` is
                the :math:`I` th material cluster incremental Cauchy stress
                tensor (:math:`\\hat{(\\cdot)}` denotes the incremental nature
                of the constitutive function),
                :math:`\\Delta \\boldsymbol{\\sigma}_{n+1}` is the macroscale
                incremental Cauchy stress tensor, :math:`n_{c}` is the number
                of material clusters, and :math:`n+1` denotes the current
                increment.

        ----

        **Infinitesimal strains (incremental equilibrium formulation, \
                                 total primary unknowns):**

            This formulation is mathematically equivalent to the incremental
            formulation of the equilibrium problem. Based on the additive
            nature of both infinitesimal strain and Cauchy stress tensors, this
            functional format is suitable for a computational implementation
            where total primary unknowns are adopted.

            *Equilibrium residuals*

                .. math::

                   \\begin{multline}
                        \\boldsymbol{R}^{(I)}_{n+1} =
                        \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}
                        + \\sum^{n_{\\text{c}}}_{J=1}
                        \\boldsymbol{\\mathsf{T}}^{(I)(J)} : \\left(
                        \\hat{\\boldsymbol{\\sigma}}_{\\mu,\\,n+1}^{(J)}
                        - \\boldsymbol{\\mathsf{D}}^{e,\\, 0}:
                        \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(J)} \\right)
                        -  \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0} \\\\
                        - \\left( \\boldsymbol{\\varepsilon}_{\\mu,\\,n}^{(I)}
                        + \\sum^{n_{\\text{c}}}_{J=1}
                        \\boldsymbol{\\mathsf{T}}^{(I)(J)} : \\left(
                        \\boldsymbol{\\sigma}_{\\mu,\\,n}^{(J)}
                        - \\boldsymbol{\\mathsf{D}}^{e,\\, 0}:
                        \\boldsymbol{\\varepsilon}_{\\mu,\\,n}^{(J)} \\right)
                        - \\boldsymbol{\\varepsilon}_{\\mu,\\,n}^{0} \\right)
                        \\, ,
                    \\end{multline}

                .. math::

                   \\forall I = 1, \\, \\dots, \\, n_{\\text{c}} \\, ,

                where :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th
                material cluster equilibrium residual function,
                :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{(I)}` is the
                :math:`I` th material cluster infinitesimal strain tensor,
                :math:`\\boldsymbol{\\mathsf{T}}^{(I)(J)}` is the cluster
                interaction tensor (fourth-order tensor) between the
                :math:`I` th and :math:`J` th material clusters,
                :math:`\\boldsymbol{\\sigma}_{\\mu}^{(J)}` is the :math:`J` th
                material cluster Cauchy stress tensor (:math:`\\hat{(\\cdot)}`
                denotes the incremental nature of the constitutive function),
                :math:`\\boldsymbol{\\mathsf{D}}^{e,\\, 0}` is the elastic
                tangent modulus of the reference homogeneous material,
                :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{0}` is the far-field
                infinitesimal strain tensor, :math:`n_{c}` is the number of
                material clusters, :math:`n+1` denotes the current increment,
                and :math:`n` denotes the last converged increment.


            *Loading (homogenization-based) strain and/or stress constraints*

                .. math::

                   \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                   \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                   \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(I)}
                   - \\boldsymbol{\\varepsilon}_{n+1}
                   - \\left( \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                   \\boldsymbol{\\varepsilon}_{\\mu,\\,n}^{(I)}
                   - \\boldsymbol{\\varepsilon}_{n}  \\right) \\, ,

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                strain loading constraint residual function, :math:`f^{(I)}` is
                the volume fraction of the :math:`I` th material cluster,
                :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{(I)}` is the
                :math:`I` th material cluster infinitesimal strain tensor,
                :math:`\\boldsymbol{\\varepsilon}` is the macroscale
                infinitesimal strain tensor, :math:`n_{c}` is the number of
                material clusters, :math:`n+1` denotes the current increment,
                and :math:`n` denotes the last converged increment.

                .. math::

                    \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                    \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                    \\hat{\\boldsymbol{\\sigma}}_{\\mu,\\,n+1}^{(I)}
                    - \\boldsymbol{\\sigma}_{n+1} - \\left(
                    \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                    \\boldsymbol{\\sigma}_{\\mu,\\,n}^{(I)}
                    - \\boldsymbol{\\sigma}_{n}  \\right) \\, ,

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                stress loading constraint residual function,
                :math:`f^{(I)}` is the volume fraction of the :math:`I` th
                material cluster, :math:`\\boldsymbol{\\sigma}_{\\mu}^{(I)}` is
                the :math:`I` th material cluster Cauchy stress tensor
                (:math:`\\hat{(\\cdot)}` denotes the incremental nature of the
                constitutive function), :math:`\\boldsymbol{\\sigma}` is the
                macroscale incremental Cauchy stress tensor, :math:`n_{c}` is
                the number of material clusters, :math:`n+1` denotes the
                current increment, and :math:`n` denotes the last converged
                increment.

        ----

        **Finite strains (total equilibrium formulation, \
                          total primary unknowns):**

            *Equilibrium residuals*

                .. math::

                   \\boldsymbol{R}^{(I)}_{n+1} =
                   \\boldsymbol{F}_{\\mu,\\,n+1}^{(I)}
                   + \\sum^{n_{\\text{c}}}_{J=1}
                   \\boldsymbol{\\mathsf{T}}^{(I)(J)} : \\left(
                   \\hat{\\boldsymbol{P}}_{\\mu,\\,n+1}^{(J)}
                   - \\boldsymbol{\\mathsf{A}}^{e,\\, 0}:
                   \\boldsymbol{F}_{\\mu,\\,n+1}^{(J)} \\right)
                   - \\boldsymbol{F}_{\\mu,\\,n+1}^{0} \\, ,

                .. math::

                   \\forall I = 1, \\, \\dots, \\, n_{\\text{c}} \\, ,

                where :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th
                material cluster equilibrium residual function,
                :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{(I)}` is the :math:`I` th
                material cluster deformation gradient,
                :math:`\\boldsymbol{\\mathsf{T}}^{(I)(J)}` is the cluster
                interaction tensor (fourth-order tensor) between the
                :math:`I` th and :math:`J` th material clusters,
                :math:`\\boldsymbol{P}_{\\mu,\\,n+1}^{(J)}` is the :math:`J` th
                material cluster first Piola-Kirchhoff stress tensor
                (:math:`\\hat{(\\cdot)}` denotes the incremental nature of the
                constitutive function),
                :math:`\\boldsymbol{\\mathsf{A}}^{e,\\, 0}` is the elastic
                tangent modulus of the reference homogeneous material,
                :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{0}` is the far-field
                deformation gradient, :math:`n_{c}` is the number of material
                clusters, and :math:`n+1` denotes the current increment.


            *Loading (homogenization-based) strain and/or stress constraints*

                .. math::

                   \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                   \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                   \\boldsymbol{F}_{\\mu,\\,n+1}^{(I)}
                   - \\boldsymbol{F}_{n+1}  \\, ,

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                strain loading constraint residual function,
                :math:`f^{(I)}` is the volume fraction of the :math:`I` th
                material cluster, :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{(I)}`
                is the :math:`I` th material cluster deformation gradient,
                :math:`\\boldsymbol{F}_{n+1}` is the macroscale deformation
                gradient, :math:`n_{c}` is the number of material clusters, and
                :math:`n+1` denotes the current increment.

                .. math::

                    \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1} =
                    \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
                    \\hat{\\boldsymbol{P}}_{\\mu,\\,n+1}^{(I)}
                    - \\boldsymbol{P}_{n+1}

                where :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the
                stress loading constraint residual function, :math:`f^{(I)}` is
                the volume fraction of the :math:`I` th material cluster,
                :math:`\\boldsymbol{P}_{\\mu, \\,n+1}^{(I)}` is the
                :math:`I` th material cluster first Piola-Kirchhoff stress
                tensor (:math:`\\hat{(\\cdot)}` denotes the incremental nature
                of the constitutive function), :math:`\\boldsymbol{P}_{n+1}` is
                the macroscale first Piola-Kirchhoff stress tensor,
                :math:`n_{c}` is the number of material clusters, and
                :math:`n+1` denotes the current increment.

        ----

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        presc_strain_idxs : list[int]
            Prescribed macroscale loading strain components indexes.
        presc_stress_idxs : list[int]
            Prescribed macroscale loading stress components indexes.
        applied_mac_load_mf : dict
            For each prescribed loading nature type
            (key, {'strain', 'stress'}), stores the current applied macroscale
            loading constraints in a numpy.ndarray of shape (n_comps,).
        ref_material : ElasticReferenceMaterial
            Elastic reference material.
        global_cit_mf : numpy.ndarray (2d)
            Global cluster interaction matrix. Assembly positions are assigned
            according to the order of material_phases (1st) and phase_clusters
            (2nd).
        global_strain_mf : numpy.ndarray (1d)
            Global vector of clusters strain tensors (matricial form).
        inc_mac_load_mf : dict, default=None
            For each loading nature type (key, {'strain', 'stress'}), stores
            the incremental loading constraint matricial form in a
            numpy.ndarray of shape (n_comps,).
        farfield_strain_mf : numpy.ndarray (1d), default=None
            Far-field strain tensor (matricial form).
        farfield_strain_old_mf : numpy.ndarray (1d), default=None
            Last converged far-field strain tensor (matricial form).

        Returns
        -------
        residual : numpy.ndarray (1d)
            Lippmann-Schwinger equilibrium residual vector.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = material_state.get_material_phases()
        # Get clusters associated with each material phase
        phase_clusters = crve.get_phase_clusters()
        # Get total number of clusters
        n_total_clusters = crve.get_n_total_clusters()
        # Get clusters state variables
        clusters_state = material_state.get_clusters_state()
        # Get elastic reference material tangent modulus (matricial form)
        ref_elastic_tangent_mf = ref_material.get_elastic_tangent_mf()
        # Get homogenized strain and stress tensors (matricial form)
        hom_strain_mf = material_state.get_hom_strain_mf()
        hom_stress_mf = material_state.get_hom_stress_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialiatize and get last converged variables
        if self._strain_formulation == 'infinitesimal':
            # Get incremental homogenized strain and stress tensors
            # (matricial form)
            inc_hom_strain_mf = material_state.get_inc_hom_strain_mf()
            inc_hom_stress_mf = material_state.get_inc_hom_stress_mf()
            # Initialize last converged global vector of clusters strain
            # tensors
            global_strain_old_mf = np.zeros_like(global_strain_mf)
            # Initialize last converged clusters polarization stress
            global_pol_stress_old_mf = np.zeros_like(global_strain_mf)
            # Get last converged clusters state variables
            clusters_state_old = material_state.get_clusters_state_old()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clusters polarization stress
        global_pol_stress_mf = np.zeros_like(global_strain_mf)
        # Initialize material cluster strain range indexes
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Compute material cluster stress (matricial form)
                stress_mf = clusters_state[str(cluster)]['stress_mf']
                # Get material cluster strain (matricial form)
                strain_mf = global_strain_mf[i_init:i_end]
                # Add cluster polarization stress to global array
                global_pol_stress_mf[i_init:i_end] = stress_mf - \
                    np.matmul(ref_elastic_tangent_mf, strain_mf)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute last converged polarization stress
                if self._strain_formulation == 'infinitesimal':
                    # Compute material cluster last converged stress
                    # (matricial form)
                    stress_old_mf = \
                        clusters_state_old[str(cluster)]['stress_mf']
                    # Get last converged material cluster strain
                    # (matricial form)
                    strain_old_mf = \
                        clusters_state_old[str(cluster)]['strain_mf']
                    global_strain_old_mf[i_init:i_end] = strain_old_mf
                    # Add last converged cluster polarization stress to global
                    # array
                    global_pol_stress_old_mf[i_init:i_end] = stress_old_mf \
                        - np.matmul(ref_elastic_tangent_mf, strain_old_mf)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update cluster strain range indexes
                i_init = i_init + len(comp_order)
                i_end = i_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize residual vector
        residual = np.zeros(n_total_clusters*len(comp_order)
                            + len(comp_order))
        # Compute clusters equilibrium residuals
        residual[0:n_total_clusters*len(comp_order)] = \
            np.subtract(
                np.add(global_strain_mf,
                       np.matmul(global_cit_mf, global_pol_stress_mf)),
                numpy.matlib.repmat(farfield_strain_mf, 1,
                                    n_total_clusters))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add residual term for compatibility with incremental formulation
        # under infinitesimal strains
        if self._strain_formulation == 'infinitesimal':
            residual[0:n_total_clusters*len(comp_order)] += np.reshape(
                -np.subtract(
                    np.add(global_strain_old_mf,
                           np.matmul(global_cit_mf,
                                     global_pol_stress_old_mf)),
                    numpy.matlib.repmat(farfield_strain_old_mf, 1,
                                        n_total_clusters)),
                (n_total_clusters*len(comp_order),))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute homogenization constraints residuals
        for i in range(len(comp_order)):
            if i in presc_strain_idxs:
                if self._strain_formulation == 'infinitesimal':
                    # Constraint compatible with incremental formulation
                    # under infinitesimal strains
                    residual[n_total_clusters*len(comp_order) + i] = \
                        inc_hom_strain_mf[i] - inc_mac_load_mf['strain'][i]
                else:
                    residual[n_total_clusters*len(comp_order) + i] = \
                        hom_strain_mf[i] - applied_mac_load_mf['strain'][i]
            else:
                if self._strain_formulation == 'infinitesimal':
                    # Constraint compatible with incremental formulation
                    # under infinitesimal strains
                    residual[n_total_clusters*len(comp_order) + i] = \
                        inc_hom_stress_mf[i] - inc_mac_load_mf['stress'][i]
                else:
                    residual[n_total_clusters*len(comp_order) + i] = \
                        hom_stress_mf[i] - applied_mac_load_mf['stress'][i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return residual
    # -------------------------------------------------------------------------
    def _build_jacobian(self, crve, material_state, presc_strain_idxs,
                        presc_stress_idxs, global_cit_diff_tangent_mf):
        """Build Lippmann-Schwinger equilibrium Jacobian matrix.

        *Infinitesimal strains:*

        .. math::

           \\boldsymbol{J}_{n+1} =
           \\dfrac{\\partial \\boldsymbol{R}_{n+1}}{\\partial \\Delta
           \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}} = \\begin{bmatrix}
           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(K)}} &
           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0}} \\\\[8pt]
           \\dfrac{\\partial \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}
           }{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(K)}} &
           \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0}}
           \\end{bmatrix} \\, , \\qquad \\forall I, \\, K=1, \\, \\dots,
           \\, n_{\\text{c}} \\, ,

        where :math:`\\boldsymbol{R}_{n+1}` is the global residual function
        (assuming incremental equilibrium formulation and incremental
        primary unknowns), :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the
        :math:`I` th material cluster equilibrium residual function,
        :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the strain
        and/or stress loading constraints residual function,
        :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(K)}` is
        the :math:`K` th material cluster incremental infinitesimal strain
        tensor,
        :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{0}` is the
        incremental far-field infinitesimal strain tensor, :math:`n_{c}` is
        the number of material clusters, and :math:`n+1` denotes the
        current increment.

        The partial derivatives are defined as

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{(K)}} =
           \\delta_{(I)(K)} \\boldsymbol{\\mathsf{I}} +
           \\boldsymbol{\\mathsf{T}}^{(I)(K)} : \\left(
           \\boldsymbol{\\mathsf{D}}^{(K)}_{n+1}
           - \\boldsymbol{\\mathsf{D}}^{e,\\, 0} \\right) \\, ,

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{0}} =
           - \\boldsymbol{\\mathsf{I}} \\, ,

        .. math::

           \\text{strain:} \\; \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}}{\\partial \\Delta
           \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{(K)}} =  f^{(K)} \\,
           \\boldsymbol{\\mathsf{I}}   \\, ,  \\quad
           \\text{stress:} \\; \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}}{\\partial \\Delta
           \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{(K)}} =  f^{(K)} \\,
           \\boldsymbol{\\mathsf{D}}^{(K)}_{n+1} \\, ,

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}
           }{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{0}} =
           \\mathbf{0} \\, ,

        where :math:`\\delta_{(I)(K)}` is the Kronecker delta,
        :math:`\\boldsymbol{\\mathsf{I}}` is the fourth-order identity
        tensor, :math:`\\boldsymbol{\\mathsf{T}}^{(I)(K)}` is the cluster
        interaction tensor (fourth-order tensor) between the :math:`I` th
        and :math:`K` th material clusters,
        :math:`\\boldsymbol{\\mathsf{D}}^{(K)}_{n+1}` is the consistent
        tangent modulus of the :math:`K` th material cluster,
        :math:`\\boldsymbol{\\mathsf{D}}^{e,\\, 0}` is the elastic tangent
        modulus of the reference homogeneous material, :math:`f^{(K)}` is
        the volume fraction of the :math:`K` th material cluster.

        **Remark:** The Jacobian matrix is the same when the residual
        functions are derived with respect to the total strains (assuming
        incremental equilibrium formulation and total primary unknowns).

        ----

        *Finite strains:*

        .. math::

           \\boldsymbol{J}_{n+1} =
           \\dfrac{\\partial \\boldsymbol{R}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu,\\,n+1}} = \\begin{bmatrix}
           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu,\\,n+1}^{(K)}} &  \\dfrac{\\partial
           \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu,\\,n+1}^{0}} \\\\[8pt]
           \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu,\\,n+1}^{(K)}} &  \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu,\\,n+1}^{0}}
           \\end{bmatrix} \\, , \\qquad \\forall I, \\, K=1, \\, \\dots,
           \\, n_{\\text{c}} \\, ,

        where :math:`\\boldsymbol{R}_{n+1}` is the global residual
        function, :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th
        material cluster equilibrium residual function,
        :math:`\\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}` is the strain
        and/or stress loading constraints residual function,
        :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{(K)}` is the :math:`K` th
        material cluster deformation gradient,
        :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{0}` is the far-field
        deformation gradient, :math:`n_{c}` is the number of material
        clusters, and :math:`n+1` denotes the current increment.

        The partial derivatives are defined as

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu, \\, n+1}^{(K)}} = \\delta_{(I)(K)}
           \\boldsymbol{\\mathsf{I}} +  \\boldsymbol{\\mathsf{T}}^{(I)(K)}
           : \\left( \\boldsymbol{\\mathsf{A}}^{(K)}_{n+1}
           - \\boldsymbol{\\mathsf{A}}^{e,\\, 0}   \\right) \\, ,

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu, \\, n+1}^{0}} =
           - \\boldsymbol{\\mathsf{I}} \\, ,

        .. math::

           \\text{strain:} \\; \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu, \\, n+1}^{(K)}} =  f^{(K)} \\,
           \\boldsymbol{\\mathsf{I}}   \\, ,  \\quad
           \\text{stress:} \\; \\dfrac{\\partial
           \\boldsymbol{R}^{(n_{\\text{c}}+1)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu, \\, n+1}^{(K)}} =  f^{(K)} \\,
           \\boldsymbol{\\mathsf{A}}^{(K)}_{n+1} \\, ,

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(n_{\\text{c}} + 1)}_{n+1}
           }{\\partial \\boldsymbol{F}_{\\mu, \\, n+1}^{0}} = \\mathbf{0}
           \\, ,

        where :math:`\\delta_{(I)(K)}` is the Kronecker delta,
        :math:`\\boldsymbol{\\mathsf{I}}` is the fourth-order identity
        tensor, :math:`\\boldsymbol{\\mathsf{T}}^{(I)(K)}` is the cluster
        interaction tensor (fourth-order tensor) between the :math:`I` th
        and :math:`K` th material clusters,
        :math:`\\boldsymbol{\\mathsf{A}}^{(K)}_{n+1}` is the material
        consistent tangent modulus of the :math:`K` th material cluster,
        :math:`\\boldsymbol{\\mathsf{A}}^{e,\\, 0}` is the elastic tangent
        modulus of the reference homogeneous material, :math:`f^{(K)}` is
        the volume fraction of the :math:`K` th material cluster.

        ----

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        global_cit_diff_tangent_mf : numpy.ndarray (2d)
            Global matrix similar to global cluster interaction matrix but
            where each cluster interaction tensor is double contracted with the
            difference between the associated material cluster consistent
            tangent modulus and the elastic reference material tangent modulus.

        Returns
        -------
        jacobian : numpy.ndarray (2d)
            Lippmann-Schwinger equilibrium Jacobian matrix.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        _, foid, _, fosym, _, _, _ = top.get_id_operators(self._n_dim)
        if self._strain_formulation == 'infinitesimal':
            # Set fourth-order symmetric projection tensor (matricial form)
            fosym_mf = mop.get_tensor_mf(fosym, self._n_dim, comp_order)
        else:
            # Set fourth-order identity tensor (matricial form)
            foid_mf = mop.get_tensor_mf(foid, self._n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = material_state.get_material_phases()
        # Get clusters associated with each material phase
        phase_clusters = crve.get_phase_clusters()
        # Get total number of clusters
        n_total_clusters = crve.get_n_total_clusters()
        # Get clusters volume fraction
        clusters_vf = crve.get_clusters_vf()
        # Get material consistent tangent modulus associated with each material
        # cluster
        clusters_tangent_mf = material_state.get_clusters_tangent_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Jacobian matrix
        jacobian = np.zeros(2*(n_total_clusters*len(comp_order)
                               + len(comp_order),))
        # Compute Jacobian matrix component 11
        i_init = 0
        i_end = n_total_clusters*len(comp_order)
        j_init = 0
        j_end = n_total_clusters*len(comp_order)
        if self._strain_formulation == 'infinitesimal':
            jacobian[i_init:i_end, j_init:j_end] = \
                scipy.linalg.block_diag(*(n_total_clusters*[fosym_mf, ])) \
                + global_cit_diff_tangent_mf
        else:
            jacobian[i_init:i_end, j_init:j_end] = \
                scipy.linalg.block_diag(*(n_total_clusters*[foid_mf, ])) \
                + global_cit_diff_tangent_mf
        # Compute Jacobian matrix component 12
        i_init = 0
        i_end = n_total_clusters*len(comp_order)
        j_init = n_total_clusters*len(comp_order)
        j_end = n_total_clusters*len(comp_order) + len(comp_order)
        if self._strain_formulation == 'infinitesimal':
            jacobian[i_init:i_end, j_init:j_end] = \
                numpy.matlib.repmat(-1.0*fosym_mf, n_total_clusters, 1)
        else:
            jacobian[i_init:i_end, j_init:j_end] = \
                numpy.matlib.repmat(-1.0*foid_mf, n_total_clusters, 1)
        # Compute Jacobian matrix component 21
        for k in range(len(comp_order)):
            i = n_total_clusters*len(comp_order) + k
            jclst = 0
            for mat_phase in material_phases:
                for cluster in phase_clusters[mat_phase]:
                    if k in presc_strain_idxs:
                        if self._strain_formulation == 'infinitesimal':
                            f_foid_mf = clusters_vf[str(cluster)]*fosym_mf
                        else:
                            f_foid_mf = clusters_vf[str(cluster)]*foid_mf
                        j_init = jclst*len(comp_order)
                        j_end = j_init + len(comp_order)
                        jacobian[i, j_init:j_end] = f_foid_mf[k, :]
                    else:
                        vf_tangent_mf = clusters_vf[str(cluster)]\
                            * clusters_tangent_mf[str(cluster)]
                        j_init = jclst*len(comp_order)
                        j_end = j_init + len(comp_order)
                        jacobian[i, j_init:j_end] = vf_tangent_mf[k, :]
                    # Increment column cluster index
                    jclst = jclst + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return jacobian
    # -------------------------------------------------------------------------
    def _build_global_cit_diff_tangent_mf(self, crve, global_cit_mf,
                                          material_state, ref_material):
        """Build global cluster interaction - tangent modulus matrix.

        *Infinitesimal strains:*

        .. math::

           \\boldsymbol{\\mathsf{T}}^{(I)(K)} : \\left(
           \\boldsymbol{\\mathsf{D}}^{(K)}_{n+1}
           - \\boldsymbol{\\mathsf{D}}^{e,\\, 0} \\right) \\, ,
           \\qquad \\forall I, \\, K=1, \\, \\dots, \\, n_{\\text{c}} \\, ,

        where :math:`\\boldsymbol{\\mathsf{T}}^{(I)(K)}` is the cluster
        interaction tensor (fourth-order tensor) between the :math:`I` th and
        :math:`K` th material clusters,
        :math:`\\boldsymbol{\\mathsf{D}}^{(K)}_{n+1}` is the consistent tangent
        modulus of the :math:`K` th material cluster,
        :math:`\\boldsymbol{\\mathsf{D}}^{e,\\, 0}` is the elastic tangent
        modulus of the reference homogeneous material, :math:`n_{c}` is the
        number of material clusters. and :math:`n+1` denotes the current
        increment.

        ----

        *Finite strains:*

        .. math::

           \\boldsymbol{\\mathsf{T}}^{(I)(K)} : \\left(
           \\boldsymbol{\\mathsf{A}}^{(K)}_{n+1}
           - \\boldsymbol{\\mathsf{A}}^{e,\\, 0} \\right) \\, ,
           \\qquad \\forall I, \\, K=1, \\, \\dots, \\, n_{\\text{c}} \\, ,

        where :math:`\\boldsymbol{\\mathsf{T}}^{(I)(K)}` is the cluster
        interaction tensor (fourth-order tensor) between the :math:`I` th and
        :math:`K` th material clusters,
        :math:`\\boldsymbol{\\mathsf{A}}^{(K)}_{n+1}` is the material
        consistent tangent modulus of the :math:`K` th material cluster,
        :math:`\\boldsymbol{\\mathsf{A}}^{e,\\, 0}` is the elastic tangent
        modulus of the reference homogeneous material, :math:`n_{c}` is the
        number of material clusters. and :math:`n+1` denotes the current
        increment.

        ----

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        global_cit_mf : numpy.ndarray (2d)
            Global cluster interaction matrix. Assembly positions are assigned
            according to the order of material_phases (1st) and phase_clusters
            (2nd).
        material_state : MaterialState
            CRVE material constitutive state.
        ref_material : ElasticReferenceMaterial
            Elastic reference material.

        Returns
        -------
        global_cit_diff_tangent_mf : numpy.ndarray (2d)
            Global matrix similar to the global cluster interaction matrix but
            where each cluster interaction tensor is double contracted with the
            difference between the associated material cluster consistent
            tangent modulus and the reference material elastic tangent modulus.
        """
        # Get material consistent tangent modulus associated with each material
        # cluster
        clusters_tangent_mf = material_state.get_clusters_tangent_mf()
        # Get elastic reference material tangent modulus
        ref_elastic_tangent_mf = ref_material.get_elastic_tangent_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build list which stores the difference between the material cluster
        # consistent tangent modulus (matricial form) and the reference
        # material elastic tangent modulus (matricial form)
        diff_tangent_mf = list()
        # Loop over material phases
        for mat_phase in material_state.get_material_phases():
            # Loop over material clusters
            for cluster in crve.get_phase_clusters()[mat_phase]:
                diff_tangent_mf.append(clusters_tangent_mf[str(cluster)]
                                       - ref_elastic_tangent_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build global matrix similar to the global cluster interaction matrix
        # but where each cluster interaction tensor is double contracted with
        # the difference between the associated material cluster consistent
        # tangent modulus and the reference material elastic tangent modulus
        global_cit_diff_tangent_mf = np.matmul(
            global_cit_mf, scipy.linalg.block_diag(*diff_tangent_mf))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return global_cit_diff_tangent_mf
    # -------------------------------------------------------------------------
    def _check_convergence(self, crve, material_state, presc_strain_idxs,
                           presc_stress_idxs, applied_mac_load_mf, residual,
                           applied_mix_strain_mf=None):
        """Check Lippmann-Schwinger equilibrium convergence.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        presc_strain_idxs : list[int]
            Prescribed macroscale loading strain components indexes.
        presc_stress_idxs : list[int]
            Prescribed macroscale loading stress components indexes.
        applied_mac_load_mf : dict
            For each prescribed loading nature type
            (key, {'strain', 'stress'}), stores the current applied loading
            constraints in a numpy.ndarray of shape (n_comps,).
        residual : numpy.ndarray (1d)
            Lippmann-Schwinger equilibrium residual vector.
        applied_mix_strain_mf : numpy.ndarray (1d), default=None
            Strain tensor (matricial form) that contains prescribed strain
            components and (non-prescribed) homogenized strain components.

        Returns
        -------
        is_converged : bool
            True if Lippmann-Schwinger equilibrium iterative solution
            converged, False otherwise.
        errors : list[float]
            List of errors associated with the Lippmann-Schwinger equilibrium
            convergence evaluation.
        """
        # Initialize convergence flag
        is_converged = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get total number of clusters
        n_total_clusters = crve.get_n_total_clusters()
        # Compute number of prescribed loading strain components
        n_presc_strain = len(presc_strain_idxs)
        # Compute number of prescribed loading stress components
        n_presc_stress = len(presc_stress_idxs)
        # Get homogenized strain and stress tensors (matricial form)
        hom_strain_mf = material_state.get_hom_strain_mf()
        hom_stress_mf = material_state.get_hom_stress_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain and stress normalization factors
        if n_presc_strain > 0 and not np.allclose(
            applied_mac_load_mf['strain'][tuple([presc_strain_idxs])],
            np.zeros(applied_mac_load_mf['strain'][tuple([
                     presc_strain_idxs])].shape), atol=1e-10):
            strain_norm_factor = np.linalg.norm(
                applied_mac_load_mf['strain'][tuple([presc_strain_idxs])])
        elif not np.allclose(hom_strain_mf, np.zeros(hom_strain_mf.shape),
                             atol=1e-10):
            strain_norm_factor = np.linalg.norm(hom_strain_mf)
        else:
            strain_norm_factor = 1.0
        if n_presc_stress > 0 and not np.allclose(
            applied_mac_load_mf['stress'][tuple([presc_stress_idxs])],
            np.zeros(applied_mac_load_mf['stress'][tuple([
                     presc_stress_idxs])].shape), atol=1e-10):
            stress_norm_factor = np.linalg.norm(
                applied_mac_load_mf['stress'][tuple([presc_stress_idxs])])
        elif not np.allclose(hom_stress_mf, np.zeros(hom_stress_mf.shape),
                             atol=1e-10):
            stress_norm_factor = np.linalg.norm(hom_stress_mf)
        else:
            stress_norm_factor = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute error associated with the clusters equilibrium residuals
        error_1 = \
            np.linalg.norm(residual[0:n_total_clusters*len(comp_order)]) \
            / strain_norm_factor
        # Compute error associated with the homogenization constraints
        # residuals
        aux = residual[n_total_clusters*len(comp_order):]
        if n_presc_strain > 0:
            error_2 = \
                np.linalg.norm(aux[presc_strain_idxs])/strain_norm_factor
        if n_presc_stress > 0:
            error_3 = \
                np.linalg.norm(aux[presc_stress_idxs])/stress_norm_factor
        # Criterion convergence flag is True if all residual errors
        # converged according to the defined convergence tolerance
        if n_presc_strain == 0:
            error_2 = None
            is_converged = (error_1 < self._conv_tol) \
                and (error_3 < self._conv_tol)
        elif n_presc_stress == 0:
            error_3 = None
            is_converged = (error_1 < self._conv_tol) \
                and (error_2 < self._conv_tol)
        else:
            is_converged = (error_1 < self._conv_tol) \
                and (error_2 < self._conv_tol) \
                and (error_3 < self._conv_tol)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build convergence evaluation errors list
        errors = [error_1, error_2, error_3]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return is_converged, errors
    # -------------------------------------------------------------------------
    def _crve_effective_tangent_modulus(self, crve, material_state,
                                        global_cit_diff_tangent_mf,
                                        global_strain_mf=None,
                                        farfield_strain_mf=None):
        """CRVE tangent modulus and clusters strain concentration tensors.

        *Infinitesimal strains:*

        .. math::

           \\overline{\\boldsymbol{\\mathsf{D}}}_{n+1} =
           \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
           \\boldsymbol{\\mathsf{D}}^{(I)}_{n+1} :
           \\boldsymbol{\\mathsf{H}}^{(I)}_{n+1} \\, ,

        .. math::

           \\mathbf{H}^{(I)}_{n+1} = \\sum_{K=1}^{ n_{\\text{c}}}
           \\left( \\mathbf{M}^{-1} \\right)_{(I)(K)} \\, ,
           \\quad \\forall I = 1,2, \\, \\dots, \\, n_{\\text{c}} \\, ,

        .. math::

            \\mathbf{M} = \\begin{bmatrix}
            \\dfrac{\\partial \\boldsymbol{R}^{(1)}_{n+1}}{\\partial
            \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(1)}} & \\dots &
            \\dfrac{\\partial \\boldsymbol{R}^{(1)}_{n+1}}{\\partial
            \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(n_{\\text{c}})}}
            \\\\[10pt] \\vdots & \\ddots & \\vdots \\\\[5pt]
            \\dfrac{\\partial \\boldsymbol{R}^{(n_{\\mathrm{c}})}_{n+1}}{
            \\partial \\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(1)}} &
            \\dots & \\dfrac{\\partial
            \\boldsymbol{R}^{(n_{\\text{c}})}_{n+1}}{\\partial \\Delta
            \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(n_{\\text{c}})}}
            \\end{bmatrix} \\, ,

        where :math:`\\overline{\\boldsymbol{\\mathsf{D}}}_{n+1}` is the CRVE
        homogenized consistent tangent modulus, :math:`f^{(I)}` is the volume
        fraction of the :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{D}}^{(I)}_{n+1}` is the consistent tangent
        modulus of the :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{H}}^{(I)}_{n+1}` is the :math:`I` th
        material cluster strain concentration tensor,
        :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th material
        cluster equilibrium residual function,
        :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu,\\,n+1}^{(K)}` is the
        :math:`K` th material cluster incremental infinitesimal strain tensor,
        :math:`n_{c}` is the number of material clusters, and :math:`n+1`
        denotes the current increment.

        The residual derivatives are defined as

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\Delta \\boldsymbol{\\varepsilon}_{\\mu, \\, n+1}^{(K)}} =
           \\delta_{(I)(K)} \\boldsymbol{\\mathsf{I}} +
           \\boldsymbol{\\mathsf{T}}^{(I)(K)} : \\left(
           \\boldsymbol{\\mathsf{D}}^{(K)}_{n+1} -
           \\boldsymbol{\\mathsf{D}}^{e,\\, 0} \\right) \\, ,

        .. math::

           \\forall I, K = 1, \\, \\dots, \\, n_{\\text{c}} \\, .

        **Remark:** The residual derivatives are the same when the residual
        functions are derived with respect to the total strains.

        ----

        *Finite strains:*

        .. math::

           \\overline{\\boldsymbol{\\mathsf{A}}}_{n+1} =
           \\sum^{n_{\\text{c}}}_{I=1} f^{(I)}
           \\boldsymbol{\\mathsf{A}}^{(I)}_{n+1} :
           \\boldsymbol{\\mathsf{H}}^{(I)}_{n+1} \\, ,

        .. math::

           \\mathbf{H}^{(I)}_{n+1} = \\sum_{K=1}^{ n_{\\text{c}}} \\left(
           \\mathbf{M}^{-1} \\right)_{(I)(K)} \\, , \\quad
           \\forall I = 1,2, \\, \\dots, \\, n_{\\text{c}} \\, ,

        .. math::

            \\mathbf{M} = \\begin{bmatrix}
            \\dfrac{\\partial \\boldsymbol{R}^{(1)}_{n+1}}{\\partial
            \\boldsymbol{F}_{\\mu,\\,n+1}^{(1)}} & \\dots & \\dfrac{\\partial
            \\boldsymbol{R}^{(1)}_{n+1}}{\\partial
            \\boldsymbol{F}_{\\mu,\\,n+1}^{(n_{\\text{c}})}} \\\\[10pt]
            \\vdots & \\ddots & \\vdots \\\\[5pt] \\dfrac{\\partial
            \\boldsymbol{R}^{(n_{\\mathrm{c}})}_{n+1}}{\\partial
            \\boldsymbol{F}_{\\mu,\\,n+1}^{(1)}} & \\dots &
            \\dfrac{\\partial \\boldsymbol{R}^{(n_{\\text{c}})}_{n+1}}{
            \\partial \\boldsymbol{F}_{\\mu,\\,n+1}^{(n_{\\text{c}})}}
            \\end{bmatrix} \\, ,

        where :math:`\\overline{\\boldsymbol{\\mathsf{A}}}_{n+1}` is the CRVE
        homogenized material consistent tangent modulus, :math:`f^{(I)}` is the
        volume fraction of the :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{A}}^{(I)}_{n+1}` is the material
        consistent tangent modulus of the :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{H}}^{(I)}_{n+1}` is the :math:`I` th
        material cluster strain concentration tensor,
        :math:`\\boldsymbol{R}^{(I)}_{n+1}` is the :math:`I` th material
        cluster equilibrium residual function,
        :math:`\\boldsymbol{F}_{\\mu,\\,n+1}^{(K)}` is the :math:`K` th
        material cluster deformation gradient, :math:`n_{c}` is the number of
        material clusters, and :math:`n+1` denotes the current increment.

        The residual derivatives are defined as

        .. math::

           \\dfrac{\\partial \\boldsymbol{R}^{(I)}_{n+1}}{\\partial
           \\boldsymbol{F}_{\\mu, \\, n+1}^{(K)}} = \\delta_{(I)(K)}
           \\boldsymbol{\\mathsf{I}} + \\boldsymbol{\\mathsf{T}}^{(I)(K)} :
           \\left( \\boldsymbol{\\mathsf{A}}^{(K)}_{n+1} -
           \\boldsymbol{\\mathsf{A}}^{e,\\, 0} \\right) \\, ,

        .. math::

           \\forall I, K = 1, \\, \\dots, \\, n_{\\text{c}} \\, .

        ----

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state.
        global_cit_diff_tangent_mf : numpy.ndarray (2d)
            Global matrix similar to global cluster interaction matrix but
            where each cluster interaction tensor is double contracted with the
            difference between the associated material cluster consistent
            tangent modulus and the elastic reference material tangent modulus.
        global_strain_mf : numpy.ndarray (1d), default=None
            Global vector of clusters strains stored in matricial form. Only
            required for validation of cluster strain concentration tensors
            computation.
        farfield_strain_mf : numpy.ndarray (1d), default=None
            Far-field strain tensor (matricial form). Only required for
            validation of cluster strain concentration tensors computation.

        Returns
        -------
        eff_tangent_mf : numpy.ndarray (2d)
            CRVE effective material tangent modulus (matricial form).
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form)
            (item, numpy.ndarray (2d)) associated with each material cluster
            (key, str).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set second-order identity tensor
        _, foid, _, fosym, _, _, _ = top.get_id_operators(self._n_dim)
        fosym_mf = mop.get_tensor_mf(fosym, self._n_dim, comp_order)
        foid_mf = mop.get_tensor_mf(foid, self._n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = material_state.get_material_phases()
        # Get clusters associated with each material phase
        phase_clusters = crve.get_phase_clusters()
        # Get total number of clusters
        n_total_clusters = crve.get_n_total_clusters()
        # Get clusters volume fraction
        clusters_vf = crve.get_clusters_vf()
        # Get material consistent tangent modulus associated with each material
        # cluster
        clusters_tangent_mf = material_state.get_clusters_tangent_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute equilibrium jacobian matrix (cluster strain concentration
        # tensors system of linear equations coefficient matrix)
        if self._strain_formulation == 'infinitesimal':
            csct_matrix = \
                scipy.linalg.block_diag(*(n_total_clusters*[fosym_mf, ])) \
                + global_cit_diff_tangent_mf
        else:
            csct_matrix = \
                scipy.linalg.block_diag(*(n_total_clusters*[foid_mf, ])) \
                + global_cit_diff_tangent_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Select clusters strain concentration tensors computation option:
        #
        # Option 1 - Solve linear system of equations
        #
        # Option 2 - Direct computation from inverse of equilibrium Jacobian
        #            matrix
        #
        option = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if option == 1:
            # Compute cluster strain concentration tensors system of linear
            # equations right-hand side
            if self._strain_formulation == 'infinitesimal':
                csct_rhs = numpy.matlib.repmat(fosym_mf, n_total_clusters, 1)
            else:
                csct_rhs = numpy.matlib.repmat(foid_mf, n_total_clusters, 1)
            # Initialize system solution matrix (containing clusters strain
            # concentration tensors)
            global_csct_mf = np.zeros((n_total_clusters*len(comp_order),
                                       len(comp_order)))
            # Solve cluster strain concentration tensors system of linear
            # equations
            for i in range(len(comp_order)):
                global_csct_mf[:, i] = numpy.linalg.solve(csct_matrix,
                                                          csct_rhs[:, i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif option == 2:
            # Compute inverse of equilibrium jacobian matrix
            csct_matrix_inv = numpy.linalg.inv(csct_matrix)
            # Initialize system solution matrix (containing clusters strain
            # concentration tensors)
            global_csct_mf = np.zeros((n_total_clusters*len(comp_order),
                                       len(comp_order)))
            # Initialize cluster indexes
            i_init = 0
            i_end = i_init + len(comp_order)
            j_init = 0
            j_end = j_init + len(comp_order)
            # Loop over material phases
            for mat_phase_I in material_phases:
                # Loop over material phase clusters
                for cluster_I in phase_clusters[mat_phase_I]:
                    # Loop over material phases
                    for mat_phase_J in material_phases:
                        # Loop over material phase clusters
                        for cluster_J in phase_clusters[mat_phase_J]:
                            # Add cluster J contribution to cluster I strain
                            # concentration tensor
                            global_csct_mf[i_init:i_end, :] += \
                                csct_matrix_inv[i_init:i_end, j_init:j_end]
                            # Increment cluster index
                            j_init = j_init + len(comp_order)
                            j_end = j_init + len(comp_order)
                    # Increment cluster indexes
                    i_init = i_init + len(comp_order)
                    i_end = i_init + len(comp_order)
                    j_init = 0
                    j_end = j_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validate cluster strain concentration tensors computation
        is_csct_validation = False
        if is_csct_validation:
            self._validate_csct(material_phases, phase_clusters,
                                global_csct_mf, global_strain_mf,
                                farfield_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize effective tangent modulus
        eff_tangent_mf = np.zeros((len(comp_order), len(comp_order)))
        # Initialize clusters strain concentration tensors dictionary
        clusters_sct_mf = {}
        # Initialize cluster index
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Get material cluster volume fraction
                cluster_vf = clusters_vf[str(cluster)]
                # Get material cluster consistent tangent (matricial form)
                cluster_tangent_mf = clusters_tangent_mf[str(cluster)]
                # Get material cluster strain concentration tensor
                cluster_sct_mf = global_csct_mf[i_init:i_end, :]
                # Store material cluster strain concentration tensor (matricial
                # form)
                clusters_sct_mf[str(cluster)] = cluster_sct_mf
                # Add material cluster contribution to effective tangent
                # modulus
                eff_tangent_mf = eff_tangent_mf + \
                    cluster_vf*np.matmul(cluster_tangent_mf, cluster_sct_mf)
                # Increment cluster index
                i_init = i_init + len(comp_order)
                i_end = i_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return eff_tangent_mf, clusters_sct_mf
    # -------------------------------------------------------------------------
    def _validate_csct(self, material_phases, phase_clusters, global_csct_mf,
                       global_strain_mf, farfield_strain_mf):
        """Validate clusters strain concentration tensors computation.

        This validation procedure requires the homogenized strain tensor
        instead of the far-field strain tensor in the SCA formulation without
        the far-field strain tensor.

        ----

        Parameters
        ----------
        material_phases : list[str]
            RVE material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        global_csct_mf : numpy.ndarray (2d)
            Global matrix of cluster strain concentration tensors (matricial
            form).
        global_strain_mf : numpy.ndarray (1d)
            Global vector of clusters strains stored in matricial form.
        farfield_strain_mf : numpy.ndarray (1d)
            Far-field strain tensor (matricial form).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cluster index
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Get material cluster strain concentration tensor
                cluster_sct_mf = global_csct_mf[i_init:i_end, :]
                # Compute cluster strain from strain concentration tensor
                strain_mf = np.matmul(cluster_sct_mf, farfield_strain_mf)
                # Compare cluster strain computed from strain concentration
                # tensor with actual cluster strain. Raise error if equality
                # comparison fails
                if not np.allclose(strain_mf, global_strain_mf[i_init:i_end],
                                   rtol=1e-05, atol=1e-08):
                    raise RuntimeError('Wrong computation of cluster strain '
                                       'concentration tensor.')
                # Increment cluster index
                i_init = i_init + len(comp_order)
                i_end = i_init + len(comp_order)
    # -------------------------------------------------------------------------
    def _init_clusters_sct(self, material_phases, phase_clusters):
        """Initialize cluster strain concentration tensors.

        Parameters
        ----------
        material_phases : list[str]
            RVE material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).

        Returns
        -------
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form)
            (item, numpy.ndarray (2d)) associated with each material cluster
            (key, str).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cluster strain concentration tensors dictionary
        cluster_sct_mf = {}
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Initialize cluster strain concentration tensor (matricial
                # form)
                cluster_sct_mf[str(cluster)] = np.zeros((len(comp_order),
                                                         len(comp_order)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_sct_mf
    # -------------------------------------------------------------------------
    def _build_clusters_residuals(self, material_phases, phase_clusters,
                                  residual):
        """Build clusters equilibrium residuals dictionary.

        This procedure is only carried out so that clusters equilibrium
        residuals are conveniently stored to perform post-processing
        operations.

        ----

        Parameters
        ----------
        material_phases : list[str]
            RVE material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        residual : numpy.ndarray (1d)
            Lippmann-Schwinger equilibrium residual vector.

        Returns
        -------
        clusters_residuals_mf : dict
            Equilibrium residual second-order tensor (matricial form)
            (item, numpy.ndarray (1d)) associated with each material cluster
            (key, str).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clusters equilibrium residuals dictionary
        clusters_residuals_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize cluster strain range indexes
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Store cluster equilibrium residual
                clusters_residuals_mf[str(cluster)] = residual[i_init:i_end]
                # Update cluster strain range indexes
                i_init = i_init + len(comp_order)
                i_end = i_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return clusters_residuals_mf
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_inc_data(mac_load_path):
        """Display loading increment data.

        Parameters
        ----------
        mac_load_path : LoadingPath
            Macroscale loading path.
        """
        # Get increment counter
        inc = mac_load_path.get_increm_state()['inc']
        # Get loading subpath data
        sp_id, sp_inc, sp_total_lfact, sp_inc_lfact, sp_total_time, \
            sp_inc_time, subinc_level = mac_load_path.get_subpath_state()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display loading increment data
        info.displayinfo('7', 'init', inc, subinc_level, sp_id + 1,
                         sp_total_lfact, sp_total_time, sp_inc, sp_inc_lfact,
                         sp_inc_time)
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_scs_iter_data(ref_material, is_lock_prop_ref, mode='init',
                               scs_iter_time=None):
        """Display reference material self-consistent scheme iteration data.

        Parameters
        ----------
        ref_material : ElasticReferenceMaterial
            Elastic reference material.
        is_lock_prop_ref : bool
            True if elastic reference material properties are locked, False
            otherwise.
        mode : {'init', 'end'}
            Output mode: Self-consistent scheme iteration header (`init`) or
            footer (`end`).
        scs_iter_time : float
            Total self-consistent scheme time (s).
        """
        # Get elastic reference material properties
        material_properties = ref_material.get_material_properties()
        # Get self-consistent scheme iteration counter
        scs_iter = ref_material.get_scs_iter()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display reference material self-consistent scheme iteration data
        if mode == 'end':
            # Display reference material self-consistent scheme iteration
            # footer
            info.displayinfo('8', mode, scs_iter_time)
        else:
            # Display reference material self-consistent scheme iteration
            # header
            if is_lock_prop_ref:
                info.displayinfo('13', 'init', 0, material_properties['E'],
                                 material_properties['v'])
            else:
                if scs_iter == 0:
                    info.displayinfo('8', 'init', scs_iter,
                                     material_properties['E'],
                                     material_properties['v'])
                else:
                    # Get normalized iterative changes of elastic reference
                    # material properties associated with the last
                    # self-consistent iteration convergence evaluation
                    norm_dE = ref_material.get_norm_dE()
                    norm_dv = ref_material.get_norm_dv()
                    # Display reference material self-consistent scheme
                    # iteration header
                    info.displayinfo('8', 'init', scs_iter,
                                     material_properties['E'], norm_dE,
                                     material_properties['v'], norm_dv)
    # -------------------------------------------------------------------------
    @staticmethod
    def _display_nr_iter_data(mode='init', nr_iter=None, nr_iter_time=None,
                              errors=[]):
        """Display Newton-Raphson iteration data.

        Parameters
        ----------
        mode : {'init', 'iter'}
            Output mode: Newton-Raphson iteration header ('init') or solution
            related metrics ('iter').
        nr_iter : int
            Newton-Raphson iteration counter.
        nr_iter_time : float
            Total Newton-Raphson iteration time (s).
        errors : list[float]
            List of errors associated with the Newton-Raphson convergence
            evaluation.
        """
        if mode == 'iter':
            info.displayinfo('9', mode, nr_iter, nr_iter_time, *errors)
        else:
            info.displayinfo('9', mode)
    # -------------------------------------------------------------------------
    def _set_output_files(self, output_dir, crve, problem_name='problem',
                          is_clust_adapt_output=False,
                          is_ref_material_output=None, is_vtk_output=False,
                          vtk_data=None, is_voxels_output=None):
        """Create and initialize output files.

        Parameters
        ----------
        output_dir : str
            Absolute directory path of output files.
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        problem_name : str, default='problem'
            Problem name.
        is_clust_adapt_output : bool, default=False
            Clustering adaptivity output flag.
        is_ref_material_output : bool, default=False
            Reference material output flag.
        is_vtk_output : bool, default=False
            VTK output flag.
        vtk_data : dict, default=None
            VTK output file parameters.
        is_voxels_output : bool
            Voxels output flag.

        Returns
        -------
        hres_output : HomResOutput
            Output associated with the homogenized results.
        efftan_output : EffTanOutput
            Output associated with the CRVE effective tangent modulus.
        ref_mat_output : RefMatOutput
            Output associated with the reference material.
        voxels_output : VoxelsOutput
            Output associated with voxels material-related quantities.
        adapt_output : ClusteringAdaptivityOutput
            Output associated with the clustering adaptivity procedures.
        vtk_output : VTKOutput
            Output associated with the VTK files.
        """
        if output_dir[-1] != '/':
            output_dir += '/'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set file where homogenized strain/stress results are stored
        hres_file_path = output_dir + problem_name + '.hres'
        # Instantiate homogenized results output
        hres_output = HomResOutput(hres_file_path)
        # Write homogenized results output file header
        hres_output.init_file(self._strain_formulation)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set file where CRVE effective material tangent modulus is stored
        efftan_file_path = output_dir + problem_name + '.efftan'
        # Instantiate CRVE effective material tangent modulus output
        efftan_output = EffTanOutput(efftan_file_path)
        # Write homogenized results output file header
        efftan_output.init_file(self._strain_formulation)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ref_mat_output = None
        if is_ref_material_output:
            # Set file where reference material data is stored
            refm_file_path = output_dir + problem_name + '.refm'
            # Instantiate reference material output
            ref_mat_output = RefMatOutput(
                refm_file_path, self._strain_formulation, self._problem_type,
                self._self_consistent_scheme)
            # Write reference material output file header
            ref_mat_output.init_file()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        voxels_output = None
        if is_voxels_output:
            # Set voxels material-related output file path
            voxout_file_path = output_dir + problem_name + '.voxout'
            # Instantiate voxels material-related output
            voxels_output = VoxelsOutput(
                voxout_file_path, self._strain_formulation, self._problem_type)
            # Write voxels material-related output file header
            voxels_output.init_voxels_output_file(crve)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        vtk_output = None
        if is_vtk_output:
            # Set VTK output directories paths
            pvd_dir = output_dir + 'post_processing/'
            vtk_dir = pvd_dir + 'VTK/'
            # Instantiante VTK output
            vtk_output = \
                VTKOutput(type='ImageData', version='1.0',
                          byte_order=vtk_data['vtk_byte_order'],
                          format=vtk_data['vtk_format'],
                          precision=vtk_data['vtk_precision'],
                          header_type='UInt64', base_name=problem_name,
                          vtk_dir=vtk_dir, pvd_dir=pvd_dir)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        adapt_output = None
        if is_clust_adapt_output:
            # Set file where clustering adaptivity data is stored
            adapt_file_path = output_dir + problem_name + '.adapt'
            # Instantiate clustering adaptivity output
            adapt_output = ClusteringAdaptivityOutput(
                adapt_file_path, crve.get_adapt_material_phases())
            # Write clustering adaptivity output file header
            adapt_output.init_adapt_file(crve)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return hres_output, efftan_output, ref_mat_output, voxels_output, \
            adapt_output, vtk_output
#
#                                   Reference (fictitious) homogeneous material
# =============================================================================
class ElasticReferenceMaterial:
    """Elastic reference material.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _material_properties : dict
        Elastic material properties (key, str) values
        (item, {int, float, bool}).
    _material_properties_old : dict
        Last loading increment converged elastic material properties (key, str)
        values (item, {int, float, bool}).
    _material_properties_init : dict
        Elastic material properties (key, str) values
        (item, {int, float, bool}) initial guess.
    _material_properties_scs_init : dict
        Elastic material properties (key, str) values
        (item, {int, float, bool}) converged in the first loading increment.
    _elastic_tangent_mf : numpy.ndarray (2d)
        Elastic tangent modulus in matricial form.
    _elastic_compliance_matrix : numpy.ndarray (2d)
        Elastic compliance in matrix form.
    _scs_iter : int
        Self-consistent scheme iteration counter.
    _norm_dE : float
        Normalized iterative change of Young modulus associated with the last
        self-consistent iteration convergence evaluation.
    _norm_dv : float
        Normalized iterative change of Poisson ratio associated with the last
        self-consistent iteration convergence evaluation.

    Methods
    -------
    init_material_properties(self, material_phases, \
                             material_phases_properties, material_phases_vf, \
                             properties=None)
        Set initial guess of elastic reference material properties.
    update_material_properties(self, E, v)
        Update elastic reference material properties.
    update_converged_material_properties(self)
        Update converged elastic reference material properties.
    reset_material_properties(self)
        Reset material properties to last loading increment values.
    set_material_properties_scs_init(self)
        Set material properties converged in the first loading increment.
    get_material_properties(self)
        Get elastic reference material properties.
    get_elastic_tangent_mf(self)
        Get elastic tangent modulus in matricial form.
    get_elastic_compliance_matrix(self)
        Get elastic compliance in matrix form.
    init_scs_iter(self)
        Initialize self-consistent scheme iteration counter.
    update_scs_iter(self)
        Update self-consistent scheme iteration counter.
    get_scs_iter(self)
        Get self-consistent scheme iteration counter.
    get_norm_dE(self)
        Get normalized iterative change of Young modulus.
    get_norm_dv(self)
        Get normalized iterative change of Poisson ratio.
    self_consistent_update(self, strain_mf, strain_old_mf, stress_mf, \
                           stress_old_mf, eff_tangent_mf)
        Compute reference elastic properties through self-consistent scheme.
    _update_elastic_tangent(self)
        Update reference material elastic tangent modulus and compliance.
    _check_scs_solution(self, E, v)
        Check admissibility of self-consistent scheme iterative solution.
    check_scs_convergence(self, E, v)
        Check self-consistent scheme iterative solution convergence.
    get_available_scs(strain_formulation)
        Get available self-consistent schemes.
    lame_from_technical(E, v)
        Get Lamé parameters from Young modulus and Poisson ratio.
    technical_from_lame(lam, miu)
        Get Young modulus and Poisson ratio from Lamé parameters.
    """
    def __init__(self, strain_formulation, problem_type,
                 self_consistent_scheme, conv_tol=1e-4):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        self_consistent_scheme : {'regression',}, default='regression'
            Self-consistent scheme to update the elastic reference material
            properties.
        conv_tol : float, default=1e-4
            Self-consistent scheme convergence tolerance.
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._self_consistent_scheme = self_consistent_scheme
        self._conv_tol = conv_tol
        self._material_properties = None
        self._material_properties_old = None
        self._material_properties_init = None
        self._material_properties_scs_init = None
        self._elastic_tangent_mf = None
        self._scs_iter = 0
        self._elastic_compliance_matrix = None
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # -------------------------------------------------------------------------
    def init_material_properties(self, material_phases,
                                 material_phases_properties,
                                 material_phases_vf, properties=None):
        """Set initial guess of elastic reference material properties.

        Parameters
        ----------
        material_phases : list[str]
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated with
            each material phase (key, str).
        material_phases_vf : dict
            Volume fraction (item, float) associated with each material phase
            (key, str).
        properties : dict, default=None
            Initial guess (item, float) of elastic reference material
            properties (key, str). Expecting Young's modulus ('E') and
            Poisson's coefficient ('v') for an isotropic elastic reference
            material.
        """
        if properties is None:
            # If a initial guess of the elastic reference material properties
            # is not provided, set them from the volume average of the actual
            # material phases elastic properties
            E = sum([material_phases_vf[phase]
                     * material_phases_properties[phase]['E']
                     for phase in material_phases])
            v = sum([material_phases_vf[phase]
                     * material_phases_properties[phase]['v']
                     for phase in material_phases])
        else:
            # Set initial guess of elastic reference material properties
            E = properties['E']
            v = properties['v']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elastic reference material properties
        self._material_properties_init = {'E': E, 'v': v}
        self._material_properties = \
            copy.deepcopy(self._material_properties_init)
        self._material_properties_old = \
            copy.deepcopy(self._material_properties_init)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic tangent modulus and elastic compliance matrix
        self._update_elastic_tangent()
    # -------------------------------------------------------------------------
    def update_material_properties(self, E, v):
        """Update elastic reference material properties.

        Parameters
        ----------
        E : float
            Young modulus of elastic reference material.
        v : float
            Poisson ratio of elastic reference material.
        """
        # Update elastic reference material properties
        self._material_properties['E'] = E
        self._material_properties['v'] = v
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic tangent modulus and elastic compliance matrix
        self._update_elastic_tangent()
    # -------------------------------------------------------------------------
    def update_converged_material_properties(self):
        """Update converged elastic reference material properties."""
        self._material_properties_old = \
            copy.deepcopy(self._material_properties)
    # -------------------------------------------------------------------------
    def reset_material_properties(self):
        """Reset material properties to last loading increment values."""
        # Reset elastic reference material properties to last loading increment
        # values
        self._material_properties = \
            copy.deepcopy(self._material_properties_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic tangent modulus and elastic compliance matrix
        self._update_elastic_tangent()
    # -------------------------------------------------------------------------
    def set_material_properties_scs_init(self):
        """Set material properties converged in the first loading increment."""
        self._material_properties_scs_init = \
            copy.deepcopy(self._material_properties)
    # -------------------------------------------------------------------------
    def get_material_properties(self):
        """Get elastic reference material properties.

        Returns
        -------
        material_properties : dict
            Elastic material properties (key, str) values
            (item, {int, float, bool}).
        """
        return copy.deepcopy(self._material_properties)
    # -------------------------------------------------------------------------
    def get_elastic_tangent_mf(self):
        """Get elastic tangent modulus in matricial form.

        Returns
        -------
        elastic_tangent_mf : numpy.ndarray (2d)
            Elastic tangent modulus in matricial form.
        """
        return copy.deepcopy(self._elastic_tangent_mf)
    # -------------------------------------------------------------------------
    def get_elastic_compliance_matrix(self):
        """Get elastic compliance in matrix form.

        Returns
        -------
        elastic_compliance_matrix : numpy.ndarray (2d)
            Elastic compliance in matrix form.
        """
        return copy.deepcopy(self._elastic_compliance_matrix)
    # -------------------------------------------------------------------------
    def init_scs_iter(self):
        """Initialize self-consistent scheme iteration counter."""
        self._scs_iter = 0
    # -------------------------------------------------------------------------
    def update_scs_iter(self):
        """Update self-consistent scheme iteration counter."""
        self._scs_iter += 1
    # -------------------------------------------------------------------------
    def get_scs_iter(self):
        """Get self-consistent scheme iteration counter.

        Returns
        -------
        scs_iter : int
            Self-consistent scheme iteration counter.
        """
        return self._scs_iter
    # -------------------------------------------------------------------------
    def get_norm_dE(self):
        """Get normalized iterative change of Young modulus.

        Returns
        -------
        norm_dE : float
            Normalized iterative change of Young modulus associated with the
            last self-consistent iteration convergence evaluation.
        """
        return self._norm_dE
    # -------------------------------------------------------------------------
    def get_norm_dv(self):
        """Get normalized iterative change of Poisson ratio.

        Returns
        -------
        norm_dv : float
            Normalized iterative change of Poisson ratio associated with the
            last self-consistent iteration convergence evaluation.
        """
        return self._norm_dv
    # -------------------------------------------------------------------------
    def self_consistent_update(self, strain_mf, strain_old_mf, stress_mf,
                               stress_old_mf, eff_tangent_mf):
        """Compute reference elastic properties through self-consistent scheme.

        Parameters
        ----------
        strain_mf : numpy.ndarray (1d)
            Homogenized strain (matricial form): infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains)
        strain_old_mf : numpy.ndarray (1d)
            Last converged homogenized strain (matricial form): infinitesimal
            strain tensor (infinitesimal strains) or deformation gradient
            (finite strains)
        stress_mf : numpy.ndarray (1d)
            Homogenized stress (matricial form): Cauchy stress tensor
            (infinitesimal strains) or first Piola-Kirchhoff stress tensor
            (finite strains).
        stress_old_mf : numpy.ndarray (1d)
            Last converged homogenized stress (matricial form): Cauchy stress
            tensor (infinitesimal strains) or first Piola-Kirchhoff stress
            tensor (finite strains).
        eff_tangent_mf : numpy.ndarray (2d)
            CRVE effective material tangent modulus (matricial form).

        Returns
        -------
        is_admissible : bool
            True if self-consistent scheme iterative solution is admissible,
            False otherwise.
        E : float
            Young modulus of elastic reference material.
        v : float
            Poisson ratio of elastic reference material.
        """
        # Compute incremental homogenized strain and stress tensors according
        # to problem strain formulation and self-consistent scheme
        if self._strain_formulation == 'infinitesimal':
            # Compute incremental homogenized infinitesimal strain tensor
            inc_strain_mf = strain_mf - strain_old_mf
            # Compute incremental homogenized Cauchy stress tensor
            inc_stress_mf = stress_mf - stress_old_mf
        else:
            raise RuntimeError('A suitable self-consistent scheme has not '
                               'been developed under finite strains yet.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic reference material properties based on the
        # regression-based self-consistent scheme
        if self._self_consistent_scheme in ('regression',):
            # Initialize elastic reference material properties regression-based
            # self-consistent scheme optimizer
            if self._strain_formulation == 'infinitesimal':
                # Initialize elastic reference material properties optimizer
                ref_optimizer = InfinitesimalRegressionSCS(
                    self._strain_formulation, self._problem_type,
                    copy.deepcopy(self._material_properties_old),
                    inc_strain_mf, inc_stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic reference material properties
        E, v = ref_optimizer.compute_reference_properties()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check admissibility of self-consistent scheme solution
        is_admissible = self._check_scs_solution(E, v)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return is_admissible, E, v
    # -------------------------------------------------------------------------
    def _update_elastic_tangent(self):
        """Update reference material elastic tangent modulus and compliance.

        *Infinitesimal strains:*

        .. math::

           \\boldsymbol{\\mathsf{D}}^{e,\\,0} = \\lambda^{0} \\boldsymbol{I}
           \\otimes \\boldsymbol{I} + 2 \\mu^{0}  \\boldsymbol{\\mathsf{I}}_{s}
           \\, ,

        where :math:`\\boldsymbol{\\mathsf{D}}^{e,\\,0}` is the reference
        material elastic tangent modulus, :math:`\\lambda^{0}` and
        :math:`\\mu^{0}` are the elastic Lamé parameters,
        :math:`\\boldsymbol{I}` is the second-order identity tensor, and
        :math:`\\boldsymbol{\\mathsf{I}}_{s}` is the fourth-order symmetric
        identity tensor.

        .. math::

           \\boldsymbol{\\mathsf{S}}^{e,\\,0} =
           - \\dfrac{\\lambda^{0}}{2 \\mu^{0} (3\\lambda^{0} + 2\\mu^{0})}
           \\boldsymbol{I} \\otimes \\boldsymbol{I} +
           \\dfrac{1}{2 \\mu^{0}}  \\boldsymbol{\\mathsf{I}}_{s} \\, ,

        where :math:`\\boldsymbol{\\mathsf{S}}^{e,\\,0}` is the reference
        material elastic compliance, :math:`\\lambda^{0}` and :math:`\\mu^{0}`
        are the elastic Lamé parameters, :math:`\\boldsymbol{I}` is the
        second-order identity tensor, and :math:`\\boldsymbol{\\mathsf{I}}_{s}`
        is the fourth-order symmetric identity tensor.

        ----

        *Finite strains:*

        .. math::

           \\boldsymbol{\\mathsf{A}}^{e,\\,0} = \\lambda^{0} \\boldsymbol{I}
           \\otimes \\boldsymbol{I} + 2 \\mu^{0}  \\boldsymbol{\\mathsf{I}}
           \\, ,

        where :math:`\\boldsymbol{\\mathsf{A}}^{e,\\,0}` is the reference
        material hyperelastic tangent modulus, :math:`\\lambda^{0}` and
        :math:`\\mu^{0}` are the elastic Lamé parameters,
        :math:`\\boldsymbol{I}` is the second-order identity tensor, and
        :math:`\\boldsymbol{\\mathsf{I}}` is the fourth-order identity tensor.

        .. math::

           \\boldsymbol{\\mathsf{S}}^{e,\\,0} =
           - \\dfrac{\\lambda^{0}}{2 \\mu^{0} (3\\lambda^{0} + 2\\mu^{0})}
           \\boldsymbol{I} \\otimes \\boldsymbol{I} +
           \\dfrac{1}{2 \\mu^{0}}  \\boldsymbol{\\mathsf{I}} \\, ,

        where :math:`\\boldsymbol{\\mathsf{S}}^{e,\\,0}` is the reference
        material hyperelastic compliance, :math:`\\lambda^{0}` and
        :math:`\\mu^{0}` are the elastic Lamé parameters,
        :math:`\\boldsymbol{I}` is the second-order identity tensor, and
        :math:`\\boldsymbol{\\mathsf{I}}` is the fourth-order symmetric
        identity tensor.
        """
        # Get Young's Modulus and Poisson's ratio
        E = self._material_properties['E']
        v = self._material_properties['v']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Lamé parameters
        lam, miu = type(self).lame_from_technical(E, v)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        _, foid, _, fosym, fodiagtrace, _, _ = \
            top.get_id_operators(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic tangent modulus and elastic compliance matrix
        if self._strain_formulation == 'infinitesimal':
            # Set symmetric strain/stress component order
            comp_order = self._comp_order_sym
            # Compute elastic tangent modulus and elastic compliance matrix
            # according to problem type
            if self._problem_type in [1, 4]:
                # Compute elastic tangent modulus
                elastic_tangent = lam*fodiagtrace + 2.0*miu*fosym
                # Compute elastic compliance
                elastic_compliance = -(lam/(2*miu*(3*lam + 2*miu))) \
                    * fodiagtrace + (1.0/(2.0*miu))*fosym
        else:
            # Set nonsymmetric strain/stress component order
            comp_order = self._comp_order_nsym
            # Compute elastic tangent modulus and elastic compliance matrix
            # according to problem type
            if self._problem_type in [1, 4]:
                # Compute elastic tangent modulus (2D problem (plane strain),
                # 3D problem)
                elastic_tangent = lam*fodiagtrace + 2.0*miu*foid
                # Compute elastic compliance
                elastic_compliance = -(lam/(2*miu*(3*lam + 2*miu))) \
                    * fodiagtrace + (1.0/(2.0*miu))*foid
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build elastic tangent modulus matricial form
        elastic_tangent_mf = mop.get_tensor_mf(elastic_tangent, self._n_dim,
                                               comp_order)
        # Build elastic compliance matricial form
        elastic_compliance_mf = mop.get_tensor_mf(elastic_compliance,
                                                  self._n_dim, comp_order)
        # Build elastic compliance matrix (without matricial form associated
        # coefficients)
        elastic_compliance_matrix = np.zeros(elastic_compliance_mf.shape)
        for j in range(len(comp_order)):
            for i in range(len(comp_order)):
                elastic_compliance_matrix[i, j] = \
                    (1.0/mop.kelvin_factor(i, comp_order)) \
                    * (1.0/mop.kelvin_factor(j, comp_order)) \
                    * elastic_compliance_mf[i, j]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update reference material elastic tangent modulus and compliance
        # matrix
        self._elastic_tangent_mf = elastic_tangent_mf
        self._elastic_compliance_matrix = elastic_compliance_matrix
    # -------------------------------------------------------------------------
    def _check_scs_solution(self, E, v):
        """Check admissibility of self-consistent scheme iterative solution.

        Parameters
        ----------
        E : float
            Young modulus of elastic reference material.
        v : float
            Poisson ratio of elastic reference material.

        Returns
        -------
        is_admissible : bool
            True if self-consistent scheme iterative solution is admissible,
            False otherwise.
        """
        # Set admissibility default value
        is_admissible = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate admissibility conditions:
        # Reference material Young modulus
        if self._material_properties_init is None:
            condition_1 = E > 0.0
        else:
            condition_1 = (E/self._material_properties_init['E']) >= 0.01
        # Reference material Poisson ratio
        condition_2 = v > 0.0 and (v/0.5) < 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set admissibility of self-consistent scheme iterative solution
        is_admissible = condition_1 and condition_2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_admissible
    # -------------------------------------------------------------------------
    def check_scs_convergence(self, E, v):
        """Check self-consistent scheme iterative solution convergence.

        Parameters
        ----------
        E : float
            Young modulus of elastic reference material.
        v : float
            Poisson ratio of elastic reference material.

        Returns
        -------
        is_converged : bool
            True if self-consistent scheme iterative solution converged, False
            otherwise.
        """
        # Compute iterative variation of the reference material Young modulus
        # and Poisson ratio
        dE = E - self._material_properties['E']
        dv = v - self._material_properties['v']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute normalized iterative change of the reference material Young
        # modulus and Poisson ratio
        norm_dE = abs(dE/E)
        norm_dv = abs(dv/v)
        # Store normalized iterative change of reference material properties
        self._norm_dE = norm_dE
        self._norm_dv = norm_dv
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check self-consistent scheme convergence
        is_converged = (norm_dE < self._conv_tol) \
            and (norm_dv < self._conv_tol)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return is_converged
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_scs(strain_formulation):
        """Get available self-consistent schemes.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.

        Returns
        -------
        available_scs : tuple[str]
            Available self-consistent schemes.
        """
        if strain_formulation == 'infinitesimal':
            available_scs = ('none', 'regression')
        elif strain_formulation == 'finite':
            available_scs = ('none',)
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return available_scs
    # -------------------------------------------------------------------------
    @staticmethod
    def lame_from_technical(E, v):
        """Get Lamé parameters from Young modulus and Poisson ratio.

        Parameters
        ----------
        E : float
            Young modulus.
        v : float
            Poisson ratio.

        Returns
        -------
        lam : float
            Lamé parameter.
        miu : float
            Lamé parameter.
        """
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return lam, miu
    # -------------------------------------------------------------------------
    @staticmethod
    def technical_from_lame(lam, miu):
        """Get Young modulus and Poisson ratio from Lamé parameters.

        Parameters
        ----------
        lam : float
            Lamé parameter.
        miu : float
            Lamé parameter.

        Returns
        -------
        E : float
            Young modulus.
        v : float
            Poisson ratio.
        """
        # Compute Young modulus and Poisson ratio
        E = (miu*(3.0*lam + 2.0*miu))/(lam + miu)
        v = lam/(2.0*(lam + miu))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return E, v
#
#                            Interface: Reference material properties optimizer
# =============================================================================
class ReferenceMaterialOptimizer(ABC):
    """Elastic reference material properties optimizer interface.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    """
    @abstractmethod
    def __init__(self, strain_formulation, problem_type):
        """Elastic reference material properties optimizer constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_reference_properties(self):
        """Compute elastic reference material properties.

        Returns
        -------
        young : float
            Young modulus of elastic reference material.
        poiss : float
            Poisson ratio of elastic reference material.
        """
        pass
#
#                                      Reference material properties optimizers
# =============================================================================
class InfinitesimalRegressionSCS(ReferenceMaterialOptimizer):
    """Infinitesimal strains format regression-based self-consistent scheme.

    *Mimization problem:*

    .. math::

       \\left\\{ \\lambda^{0}_{n+1}, \\, \\mu^{0}_{n+1} \\right\\} =
       \\underset{ \\left\\{ \\lambda ', \\, \\, \\mu ' \\right\\} }{
       \\mathrm{argmin}} \\,  || \\Delta \\boldsymbol{\\sigma}_{n+1} -
       \\boldsymbol{\\mathsf{D}}^{e, \\, 0}_{n+1}(\\lambda ',\\mu ') :
       \\Delta \\boldsymbol{\\varepsilon}_{n+1} ||^{2}

    where :math:`\\lambda^{0}` and :math:`\\mu^{0}` are the elastic Lamé
    parameters, :math:`\\Delta \\boldsymbol{\\sigma}` is the homogenized
    incremental Cauchy stress tensor,
    :math:`\\boldsymbol{\\mathsf{D}}^{e,\\,0}` is the reference material
    elastic tangent modulus, :math:`\\Delta \\boldsymbol{\\varepsilon}` is
    the homogenized incremental infinitesimal strain tensor, and
    :math:`n+1` denotes the current increment.

    ----

    *Solution:*

    .. math::

        \\begin{bmatrix}
                \\lambda^{0}_{m+1} \\\\[10pt] \\mu^{0}_{m+1}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
                \\text{tr} \\, \\left[ \\boldsymbol{I} \\right] \\,
                \\text{tr} \\, \\left[ \\Delta
                \\boldsymbol{\\varepsilon}_{m+1} \\right] & 2 \\,
                \\text{tr} \\, \\left[ \\Delta
                \\boldsymbol{\\varepsilon}_{m+1} \\right] \\\\[10pt]
                 \\text{tr} \\, \\left[ \\Delta
                 \\boldsymbol{\\varepsilon}_{m+1} \\right]^{2} & 2
                 \\Delta \\boldsymbol{\\varepsilon}_{m+1} :
                 \\Delta \\boldsymbol{\\varepsilon}_{m+1}
            \\end{bmatrix}^{-1}
            \\begin{bmatrix}
                 \\,  \\text{tr} \\, \\left[ \\Delta
                 \\boldsymbol{\\sigma}_{m+1} \\right] \\\\[10pt]
                 \\Delta \\boldsymbol{\\sigma}_{m+1}: \\Delta
                 \\boldsymbol{\\varepsilon}_{m+1}
            \\end{bmatrix} \\, .

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    """
    def __init__(self, strain_formulation, problem_type,
                 material_properties_old, inc_strain_mf, inc_stress_mf,
                 is_symmetrized=False):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        material_properties_old : dict
            Last loading increment converged elastic reference material
            properties (key, str) values (item, {int, float, bool}).
        inc_strain_mf : numpy.ndarray (1d)
            Incremental homogenized strain (matricial form).
        inc_stress_mf : numpy.ndarray (1d)
            Incremental homogenized stress (matricial form).
        is_symmetrized : bool, default=False
            True if a symmetric alternative stress-strain conjugate pair is
            adopted in the finite strains regression-based self-consistent
            scheme, False otherwise.
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_properties_old = copy.deepcopy(material_properties_old)
        self._inc_strain_mf = copy.deepcopy(inc_strain_mf)
        self._inc_stress_mf = copy.deepcopy(inc_stress_mf)
        self._is_symmetrized = is_symmetrized
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # -------------------------------------------------------------------------
    def compute_reference_properties(self):
        """Compute elastic reference material properties.

        Returns
        -------
        E : float
            Young modulus of elastic reference material.
        v : float
            Poisson ratio of elastic reference material.
        """
        # Set strain/stress components order
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            if self._is_symmetrized:
                comp_order = self._comp_order_sym
            else:
                comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if np.all([abs(self._inc_strain_mf[i]) < 1e-10
                   for i in range(self._inc_strain_mf.shape[0])]):
            # Get last loading increment converged elastic reference material
            # properties
            E = self._material_properties_old['E']
            v = self._material_properties_old['v']
            # Return
            return E, v
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set second-order identity tensor
        soid, _, _, _, _, _, _ = top.get_id_operators(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize self-consistent scheme system of linear equations
        # coefficient matrix and right-hand side
        scs_matrix = np.zeros((2, 2))
        scs_rhs = np.zeros(2)
        # Get incremental strain and stress tensors
        inc_strain = mop.get_tensor_from_mf(self._inc_strain_mf, self._n_dim,
                                            comp_order)
        inc_stress = mop.get_tensor_from_mf(self._inc_stress_mf, self._n_dim,
                                            comp_order)
        # Compute self-consistent scheme system of linear equations right-hand
        # side
        scs_rhs[0] = np.trace(inc_stress)
        scs_rhs[1] = top.ddot22_1(inc_stress, inc_strain)
        # Compute self-consistent scheme system of linear equations coefficient
        # matrix
        scs_matrix[0, 0] = np.trace(inc_strain)*np.trace(soid)
        scs_matrix[0, 1] = 2.0*np.trace(inc_strain)
        scs_matrix[1, 0] = np.trace(inc_strain)**2
        scs_matrix[1, 1] = 2.0*top.ddot22_1(inc_strain, inc_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Limitation 1: Under isochoric loading conditions the first equation
        # of the self-consistent scheme system of linear equations vanishes
        # (derivative with respect to lambda). In this case, adopt the previous
        # converged lambda and compute miu from the second equation of the
        # self-consistent scheme system of linear equations
        if (abs(np.trace(inc_strain))/np.linalg.norm(inc_strain)) < 1e-10 \
                or np.linalg.solve(scs_matrix, scs_rhs)[0] < 0:
            # Get previous converged reference material elastic properties
            E_old = self._material_properties_old['E']
            v_old = self._material_properties_old['v']
            # Compute previous converged lambda
            lam = (E_old*v_old)/((1.0 + v_old)*(1.0 - 2.0*v_old))
            # Compute miu
            miu = scs_rhs[1]/scs_matrix[1, 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Limitation 2: Under hydrostatic loading conditions both equations of
        # the self-consistent scheme system of linear equations become linearly
        # dependent. In this case, assume that the ratio between lambda and miu
        # is the same as in the previous converged values and solve the first
        # equation of self-consistent scheme system of linear equations
        elif np.all([abs(inc_strain[0, 0] - inc_strain[i, i])
                     / np.linalg.norm(inc_strain) < 1e-10
                     for i in range(self._n_dim)]) and \
                np.allclose(inc_strain, np.diag(np.diag(inc_strain)),
                            atol=1e-10):
            # Get previous converged reference material elastic properties
            E_old = self._material_properties_old['E']
            v_old = self._material_properties_old['v']
            # Compute previous converged reference material Lamé parameters
            lam_old = (E_old*v_old)/((1.0 + v_old)*(1.0 - 2.0*v_old))
            miu_old = E_old/(2.0*(1.0 + v_old))
            # Compute reference material Lamé parameters
            lam = (scs_rhs[0]/scs_matrix[0, 0])*(lam_old/(lam_old + miu_old))
            miu = (scs_rhs[0]/scs_matrix[0, 0])*(miu_old/(lam_old + miu_old))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve self-consistent scheme system of linear equations
        else:
            scs_solution = np.linalg.solve(scs_matrix, scs_rhs)
            # Get reference material Lamé parameters
            lam = scs_solution[0]
            miu = scs_solution[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute reference material Young modulus and Poisson ratio
        E, v = ElasticReferenceMaterial.technical_from_lame(lam, miu)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return E, v
