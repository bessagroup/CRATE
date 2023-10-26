"""Constitutive modeling of CRVE material clusters and homogenization.

This module includes the class that contains the data associated with the
CRVE's material clusters constitutive state and homogenized strain and stress
tensors, as well as the required methods to perform the material clusters
constitutive state update, the computation of the material clusters consistent
tangent modulus, and the computational homogenization of the material clusters
strain and stress tensors.

Classes
-------
MaterialState
    CRVE material constitutive state.

Functions
---------
get_available_material_models
    Get available material constitutive models.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import itertools as it
# Third-party
import numpy as np
# Local
import tensor.tensoroperations as top
import tensor.matrixoperations as mop
from material.materialoperations import cauchy_from_kirchhoff, \
                                        first_piola_from_kirchhoff, \
                                        material_from_spatial_tangent_modulus
from material.models.elastic import Elastic
from material.models.von_mises import VonMises
from material.models.stvenant_kirchhoff import StVenantKirchhoff
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                                Material state
# =============================================================================
class MaterialState:
    """CRVE material constitutive state.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _material_phases_models : dict
        Material constitutive model (item, ConstitutiveModel) associated to
        each material phase (key, str).
    _phase_clusters : dict
        Clusters labels (item, list[int]) associated to each material phase
        (key, str).
    _clusters_vf : dict
        Volume fraction (item, float) associated to each material cluster
        (key, str).
    _clusters_def_gradient_mf : dict
        Deformation gradient (item, numpy.darray (1d)) associated to each
        material cluster (key, str), stored in matricial form.
    _clusters_def_gradient_old_mf : dict
        Last converged deformation gradient (item, numpy.darray (1d))
        associated to each material cluster (key, str), stored in matricial
        form.
    _clusters_state : dict
        Material constitutive model state variables (item, dict) associated to
        each material cluster (key, str).
    _clusters_state_old : dict
        Last converged material constitutive model state variables (item, dict)
        associated to each material cluster (key, str).
    _clusters_tangent_mf : dict
        Material consistent tangent modulus (item, numpy.ndarray) associated to
        each material cluster (key, str), stored in matricial form.
    _hom_strain_mf : numpy.ndarray (1d)
        Homogenized strain tensor stored in matricial form: infinitesimal
        strain tensor (infinitesimal strains) or deformation gradient (finite
        strains).
    _hom_strain_33 : float
        Homogenized strain tensor out-of-plane component: infinitesimal strain
        tensor (infinitesimal strains) or deformation gradient (finite
        strains).
    _hom_stress_mf : numpy.ndarray (1d)
        Homogenized stress tensor stored in matricial form: Cauchy stress
        tensor (infinitesimal strains) or first Piola-Kirchhoff stress tensor
        (finite strains).
    _hom_stress_33 : float
        Homogenized stress tensor out-of-plane component: Cauchy stress tensor
        (infinitesimal strains) or first Piola-Kirchhoff stress tensor (finite
        strains).
    _hom_strain_old_mf : numpy.ndarray (1d)
        Last converged homogenized strain tensor stored in matricial form:
        infinitesimal strain tensor (infinitesimal strains) or deformation
        gradient (finite strains).
    _hom_stress_old_mf : numpy.ndarray (1d)
        Last converged homogenized stress tensor stored in matricial form:
        Cauchy stress tensor (infinitesimal strains) or first Piola-Kirchhoff
        stress tensor (finite strains).

    Methods
    -------
    init_constitutive_model(self, mat_phase, model_keyword, \
                            model_source='crate')
        Initialize material phase constitutive model.
    init_clusters_state(self)
        Initialize clusters state variables.
    set_phase_clusters(self, phase_clusters, clusters_vf)
        Set CRVE cluster labels and volume fractions of each material phase.
    get_clusters_inc_strain_mf(self, global_strain_mf)
        Get clusters incremental strain in matricial form.
    update_clusters_state(self, clusters_inc_strain_mf)
        Update clusters state variables and consistent tangent modulus.
    update_state_homogenization(self)
        Update homogenized strain and stress tensors.
    update_converged_state(self)
        Update last converged material state variables.
    set_rewind_state_updated_clustering(self, phase_clusters, clusters_vf, \
                                        clusters_state, \
                                        clusters_def_gradient_mf)
        Set rewind state variables according to updated clustering.
    get_hom_strain_mf(self)
        Get homogenized strain tensor (matricial form).
    get_hom_stress_mf(self)
        Get homogenized stress tensor (matricial form).
    get_oop_hom_comp(self)
        Get homogenized strain or stress tensor out-of-plane component.
    get_inc_hom_strain_mf(self)
        Get incremental homogenized strain tensor (matricial form).
    get_inc_hom_stress_mf(self)
        Get incremental homogenized stress tensor (matricial form).
    get_hom_strain_old_mf(self)
        Get last converged homogenized strain tensor (matricial form).
    get_hom_stress_old_mf(self)
        Get last converged homogenized stress tensor (matricial form).
    get_material_phases(self)
        Get RVE material phases.
    get_material_phases_properties(self)
        Get RVE material phases constitutive properties.
    get_material_phases_models(self)
        Get RVE material phases constitutive models.
    get_material_phases_vf(self)
        Get RVE material phases volume fraction.
    get_clusters_def_gradient_mf(self)
        Get deformation gradient of each material cluster.
    get_clusters_def_gradient_old_mf(self)
        Get last converged deformation gradient of each material cluster.
    get_clusters_state(self)
        Get material state variables of each material cluster.
    get_clusters_state_old(self)
        Get last converged material state variables of each material cluster.
    get_clusters_tangent_mf(self)
        Get material consistent tangent modulus of each material cluster.
    _material_su_interface(strain_formulation, problem_type, \
                           constitutive_model, def_gradient_old, inc_strain, \
                           state_variables_old)
        Material constitutive state update interface.
    compute_inc_log_strain(e_log_strain_old, inc_def_gradient)
        Compute incremental spatial logarithmic strain.
    compute_spatial_tangent_modulus(e_log_strain_old, def_gradient_old, \
                                    inc_def_gradient, cauchy_stress, \
                                    inf_consistent_tangent)
        Compute finite strain spatial consistent tangent modulus.
    clustering_adaptivity_update(self, phase_clusters, clusters_vf, \
                                 adaptive_clustering_map)
        Update cluster variables according to clustering adaptivity step.
    """
    def __init__(self, strain_formulation, problem_type, material_phases,
                 material_phases_properties, material_phases_vf):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        material_phases : list[str]
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to
            each material phase (key, str).
        material_phases_vf : dict
            Volume fraction (item, float) associated to each material phase
            (key, str).
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_phases = copy.deepcopy(material_phases)
        self._material_phases_properties = \
            copy.deepcopy(material_phases_properties)
        self._material_phases_vf = copy.deepcopy(material_phases_vf)
        self._phase_clusters = None
        self._clusters_vf = None
        self._material_phases_models = {mat_phase: None
                                        for mat_phase in material_phases}
        self._clusters_def_gradient_mf = None
        self._clusters_def_gradient_old_mf = None
        self._clusters_state = None
        self._clusters_state_old = None
        self._clusters_tangent_mf = None
        self._hom_strain_mf = None
        self._hom_strain_33 = None
        self._hom_stress_mf = None
        self._hom_stress_33 = None
        self._hom_strain_old_mf = None
        self._hom_stress_old_mf = None
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # -------------------------------------------------------------------------
    def init_constitutive_model(self, mat_phase, model_keyword,
                                model_source='crate'):
        """Initialize material phase constitutive model.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        model_keyword : str
            Material constitutive model input data file keyword.
        model_source : {'crate',}, default='crate'
            Material constitutive model source.
        """
        # Initialize material phase constitutive model
        if model_source == 'crate':
            if model_keyword == 'elastic':
                constitutive_model = Elastic(
                    self._strain_formulation, self._problem_type,
                    self._material_phases_properties[mat_phase])
            elif model_keyword == 'von_mises':
                constitutive_model = VonMises(
                    self._strain_formulation, self._problem_type,
                    self._material_phases_properties[mat_phase])
            elif model_keyword == 'stvenant_kirchhoff':
                constitutive_model = StVenantKirchhoff(
                    self._strain_formulation, self._problem_type,
                    self._material_phases_properties[mat_phase])
            else:
                raise RuntimeError('Unknown constitutive model from CRATE\'s '
                                   'source.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown material constitutive model source.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update material phases constitutive models
        self._material_phases_models[mat_phase] = constitutive_model
    # -------------------------------------------------------------------------
    def init_clusters_state(self):
        """Initialize clusters state variables."""
        # Initialize clusters state variables
        self._clusters_state = {}
        self._clusters_state_old = {}
        # Initialize clusters deformation gradient
        self._clusters_def_gradient_mf = {}
        self._clusters_def_gradient_old_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set second-order identity tensor matricial form
        soid_mf = mop.get_tensor_mf(np.eye(self._n_dim), self._n_dim,
                                    self._comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase constitutive model
            constitutive_model = self._material_phases_models[mat_phase]
            # Initialize material constitutive model state variables
            state_variables = constitutive_model.state_init()
            state_variables_old = copy.deepcopy(state_variables)
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Initialize clusters state variables
                self._clusters_state[str(cluster)] = state_variables
                self._clusters_state_old[str(cluster)] = state_variables_old
                # Initialize clusters deformation gradient
                self._clusters_def_gradient_mf[str(cluster)] = soid_mf
                self._clusters_def_gradient_old_mf[str(cluster)] = soid_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute initial state homogenized strain and stress tensors
        self.update_state_homogenization()
        self.update_converged_state()
    # -------------------------------------------------------------------------
    def set_phase_clusters(self, phase_clusters, clusters_vf):
        """Set CRVE cluster labels and volume fractions of each material phase.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list[int]) associated to each material
            phase (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster
            (key, str).
        """
        self._phase_clusters = copy.deepcopy(phase_clusters)
        self._clusters_vf = copy.deepcopy(clusters_vf)
    # -------------------------------------------------------------------------
    def get_clusters_inc_strain_mf(self, global_strain_mf):
        """Get clusters incremental strain in matricial form.

        *Infinitesimal strains*:

        .. math::

           \\Delta \\boldsymbol{\\varepsilon}_{\\mu, n + 1}^{(I)} =
               \\boldsymbol{\\varepsilon}_{\\mu, n + 1}^{(I)} -
               \\boldsymbol{\\varepsilon}_{\\mu, n}^{(I)} \\, ,
               \\quad I=1,\\dots, n_{c}

        where :math:`\\Delta \\boldsymbol{\\varepsilon}_{\\mu}^{(I)}` is the
        :math:`I` th material cluster incremental infinitesimal strain tensor,
        :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{(I)}` is the
        :math:`I` th material cluster infinitesimal strain tensor,
        :math:`n_{c}` is the number of material clusters, :math:`n+1` denotes
        the current increment, and :math:`n` denotes the last converged
        increment.

        ----

        *Finite strains*:

        .. math::

           (\\boldsymbol{F}_{\\Delta})_{\\mu, n + 1}^{(I)} =
               \\boldsymbol{F}_{\\mu, n + 1}^{(I)}
               ( \\boldsymbol{F}_{\\mu, n}^{(I)})^{-1} \\, ,
               \\quad I=1,\\dots, n_{c}

        where :math:`\\Delta \\boldsymbol{F}_{\\mu}^{(I)}` is the
        :math:`I` th material cluster incremental deformation gradient
        :math:`\\boldsymbol{F}_{\\mu}^{(I)}` is the :math:`I` th material
        cluster deformation gradient, :math:`n_{c}` is the number of material
        clusters, :math:`n+1` denotes the current increment, and :math:`n`
        denotes the last converged increment.

        ----

        Parameters
        ----------
        global_strain_mf : numpy.ndarray (1d)
            Global vector of clusters strains stored in matricial form.

        Returns
        -------
        clusters_inc_strain_mf : dict
            Incremental strain (item, dict) associated to each material cluster
            (key, str), stored in matricial form.
        """
        # Set strain components according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        else:
            comp_order = self._comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize dictionary of clusters incremental strain
        clusters_inc_strain_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material cluster strain range indexes
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster last converged infinitesimal strain
                # tensor (infinitesimal strains) or deformation gradient tensor
                # (finite strains)
                if self._strain_formulation == 'infinitesimal':
                    strain_old_mf = \
                        self._clusters_state_old[str(cluster)]['strain_mf']
                else:
                    def_gradient_old_mf = \
                        self._clusters_def_gradient_old_mf[str(cluster)]
                # Get material cluster infinitesimal strain tensor
                # (infinitesimal strains) or deformation gradient tensor
                # (finite strains) from global vector of clusters strains
                strain_mf = global_strain_mf[i_init:i_end]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute material cluster incremental infinitesimal strain
                # tensor (infinitesimal strains) or deformation gradient tensor
                # (finite strains)
                if self._strain_formulation == 'infinitesimal':
                    inc_strain_mf = strain_mf - strain_old_mf
                else:
                    # Build last converged deformation gradient tensor
                    def_gradient_old = mop.get_tensor_from_mf(
                        def_gradient_old_mf, self._n_dim, comp_order)
                    # Build deformation gradient tensor
                    def_gradient = mop.get_tensor_from_mf(
                        strain_mf, self._n_dim, comp_order)
                    # Compute material cluster incremental deformation gradient
                    # tensor
                    inc_def_gradient = np.matmul(
                        def_gradient, np.linalg.inv(def_gradient_old))
                    # Build material cluster incremental deformation gradient
                    # tensor (matricial form)
                    inc_strain_mf = mop.get_tensor_mf(inc_def_gradient,
                                                      self._n_dim, comp_order)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Store material cluster incremental strain (matricial form)
                clusters_inc_strain_mf[str(cluster)] = \
                    copy.deepcopy(inc_strain_mf)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update material cluster strain range indexes
                i_init += len(comp_order)
                i_end = i_init + len(comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return clusters_inc_strain_mf
    # -------------------------------------------------------------------------
    def update_clusters_state(self, clusters_inc_strain_mf):
        """Update clusters state variables and consistent tangent modulus.

        Parameters
        ----------
        clusters_inc_strain_mf : dict
            Incremental strain (item, numpy.ndarray) associated to each
            material cluster (key, str), stored in matricial form.

        Returns
        -------
        su_fail_state : dict
            State update failure state.
        """
        # Initialize state update failure state
        su_fail_state = {'is_su_fail': False, 'mat_phase': None,
                         'cluster': None}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clusters deformation gradient
        if self._strain_formulation == 'finite':
            self._clusters_def_gradient_mf = {}
        # Initialize clusters state variables and material consistent tangent
        # modulus
        self._clusters_state = {}
        self._clusters_tangent_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase constitutive model
            constitutive_model = self._material_phases_models[mat_phase]
            # Get material constitutive model source
            source = constitutive_model.get_source()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster incremental strain tensor (matricial
                # form)
                inc_strain_mf = clusters_inc_strain_mf[str(cluster)]
                if self._strain_formulation == 'infinitesimal':
                    # Infinitesimal strain tensor (symmetric)
                    comp_order = self._comp_order_sym
                else:
                    # Deformation gradient tensor (nonsymmetric)
                    comp_order = self._comp_order_nsym
                inc_strain = mop.get_tensor_from_mf(inc_strain_mf, self._n_dim,
                                                    comp_order)
                # Get material cluster last converged state variables
                state_variables_old = self._clusters_state_old[str(cluster)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material cluster last converged deformation gradient
                # tensor
                def_gradient_old_mf = \
                    self._clusters_def_gradient_old_mf[str(cluster)]
                def_gradient_old = mop.get_tensor_from_mf(
                    def_gradient_old_mf, self._n_dim, self._comp_order_nsym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform state update through the suitable material interface
                if source == 'crate':
                    state_variables, consistent_tangent_mf = \
                        self._material_su_interface(
                            self._strain_formulation, self._problem_type,
                            constitutive_model,
                            copy.deepcopy(def_gradient_old),
                            copy.deepcopy(inc_strain),
                            copy.deepcopy(state_variables_old))
                else:
                    state_variables, consistent_tangent_mf = \
                        constitutive_model.state_update(
                            copy.deepcopy(inc_strain),
                            copy.deepcopy(state_variables_old),
                            copy.deepcopy(def_gradient_old))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check state update failure status
                try:
                    if state_variables['is_su_fail']:
                        # Update state update failure status
                        su_fail_state = {'is_su_fail': True,
                                         'mat_phase': mat_phase,
                                         'cluster': cluster}
                        # Return
                        return su_fail_state
                except KeyError:
                    raise RuntimeError('Material constitutive model state '
                                       'variables must include the state '
                                       'update failure flag '
                                       '(\'is_su_fail\': bool).')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update cluster deformation gradient
                if self._strain_formulation == 'finite':
                    self._clusters_def_gradient_mf[str(cluster)] = \
                        mop.get_tensor_mf(np.matmul(inc_strain,
                                                    def_gradient_old),
                                          self._n_dim, self._comp_order_nsym)
                # Update cluster state variables and material consistent
                # tangent modulus
                self._clusters_state[str(cluster)] = state_variables
                self._clusters_tangent_mf[str(cluster)] = consistent_tangent_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return su_fail_state
    # -------------------------------------------------------------------------
    def update_state_homogenization(self):
        """Update homogenized strain and stress tensors.

        *Infinitesimal strains*:

        .. math::

           \\boldsymbol{\\varepsilon}_{n + 1} =
               \\sum_{I=1}^{n_{c}} f^{(I)}
               \\boldsymbol{\\varepsilon}_{\\mu, n + 1}^{(I)}

        where :math:`\\boldsymbol{\\varepsilon}` is the homogenized
        infinitesimal strain tensor, :math:`f^{(I)}` is the :math:`I` th
        material cluster volume fraction,
        :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{(I)}` is the :math:`I` th
        material cluster infinitesimal strain tensor, :math:`n_{c}` is the
        number of material clusters, and :math:`n+1` denotes the current
        increment.

        .. math::

           \\boldsymbol{\\sigma}_{n + 1} =
               \\sum_{I=1}^{n_{c}} f^{(I)}
               \\boldsymbol{\\sigma}_{\\mu, n + 1}^{(I)}

        where :math:`\\boldsymbol{\\sigma}` is the homogenized Cauchy stress
        tensor, :math:`f^{(I)}` is the :math:`I` th material cluster volume
        fraction, :math:`\\boldsymbol{\\sigma}_{\\mu}^{(I)}` is the
        :math:`I` th material cluster Cauchy stress tensor, :math:`n_{c}` is
        the number of material clusters, and :math:`n+1` denotes the current
        increment.

        ----

        *Finite strains*:

        .. math::

           \\boldsymbol{F}_{n + 1} =
               \\sum_{I=1}^{n_{c}} f^{(I)}
               \\boldsymbol{F}_{\\mu, n + 1}^{(I)}

        where :math:`\\boldsymbol{F}` is the homogenized deformation gradient,
        :math:`f^{(I)}` is the :math:`I` th material cluster volume fraction,
        :math:`\\boldsymbol{F}_{\\mu}^{(I)}` is the :math:`I` th material
        cluster deformation gradient, :math:`n_{c}` is the number of material
        clusters, and :math:`n+1` denotes the current increment.

        .. math::

           \\boldsymbol{P}_{n + 1} =
               \\sum_{I=1}^{n_{c}} f^{(I)}
               \\boldsymbol{P}_{\\mu, n + 1}^{(I)}

        where :math:`\\boldsymbol{P}` is the homogenized first Piola-Kirchhoff
        stress tensor, :math:`f^{(I)}` is the :math:`I` th material cluster
        volume fraction, :math:`\\boldsymbol{P}_{\\mu}^{(I)}` is the
        :math:`I` th material cluster first Piola-Kirchhoff stress tensor,
        :math:`n_{c}` is the number of material clusters, and :math:`n+1`
        denotes the current increment.
        """
        # Set strain components according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        else:
            comp_order = self._comp_order_nsym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize incremental homogenized strain and stress tensors
        # (matricial form)
        hom_strain_mf = np.zeros(len(comp_order))
        hom_stress_mf = np.zeros(len(comp_order))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster strain tensor (matricial form)
                if self._strain_formulation == 'infinitesimal':
                    strain_mf = self._clusters_state[str(cluster)]['strain_mf']
                else:
                    strain_mf = self._clusters_def_gradient_mf[str(cluster)]
                # Get material cluster stress tensor (matricial form)
                stress_mf = self._clusters_state[str(cluster)]['stress_mf']
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Add material cluster contribution to homogenized strain and
                # stress tensors (matricial form)
                hom_strain_mf = np.add(
                    hom_strain_mf, self._clusters_vf[str(cluster)]*strain_mf)
                hom_stress_mf = np.add(
                    hom_stress_mf, self._clusters_vf[str(cluster)]*stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update homogenized strain and stress tensors
        self._hom_strain_mf = copy.deepcopy(hom_strain_mf)
        self._hom_stress_mf = copy.deepcopy(hom_stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self._problem_type == 1:
            # Set out-of-plane stress component (2D plane strain problem)
            if self._problem_type == 1:
                comp_name = 'stress_33'
            else:
                raise RuntimeError('Unavailable plane problem type.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize homogenized out-of-plane component
            oop_hom_comp = 0.0
            # Loop over material phases
            for mat_phase in self._material_phases:
                # Loop over material phase clusters
                for cluster in self._phase_clusters[mat_phase]:
                    # Add material cluster contribution to the homogenized
                    # out-of-plane component component
                    oop_hom_comp = oop_hom_comp \
                        + self._clusters_vf[str(cluster)] \
                        * self._clusters_state[str(cluster)][comp_name]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update out-of-plane stress or strain component
            if self._strain_formulation == 'infinitesimal':
                self._hom_strain_33 = 0.0
            else:
                self._hom_strain_33 = 1.0
            self._hom_stress_33 = oop_hom_comp
    # -------------------------------------------------------------------------
    def update_converged_state(self):
        """Update last converged material state variables."""
        self._clusters_def_gradient_old_mf = \
            copy.deepcopy(self._clusters_def_gradient_mf)
        self._clusters_state_old = copy.deepcopy(self._clusters_state)
        self._hom_strain_old_mf = copy.deepcopy(self._hom_strain_mf)
        self._hom_stress_old_mf = copy.deepcopy(self._hom_stress_mf)
    # -------------------------------------------------------------------------
    def set_rewind_state_updated_clustering(self, phase_clusters, clusters_vf,
                                            clusters_state,
                                            clusters_def_gradient_mf):
        """Set rewind state variables according to updated clustering.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list[int]) associated to each material
            phase (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            to each material cluster (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.darray (1d)) associated to each
            material cluster (key, str), stored in matricial form.
        """
        self._phase_clusters = copy.deepcopy(phase_clusters)
        self._clusters_vf = copy.deepcopy(clusters_vf)
        self._clusters_state_old = copy.deepcopy(clusters_state)
        self._clusters_def_gradient_old_mf = \
            copy.deepcopy(clusters_def_gradient_mf)
    # -------------------------------------------------------------------------
    def get_hom_strain_mf(self):
        """Get homogenized strain tensor (matricial form).

        Returns
        -------
        hom_strain_mf : numpy.ndarray (1d)
            Homogenized strain tensor stored in matricial form.
        """
        return copy.deepcopy(self._hom_strain_mf)
    # -------------------------------------------------------------------------
    def get_hom_stress_mf(self):
        """Get homogenized stress tensor (matricial form).

        Returns
        -------
        hom_stress_mf : numpy.ndarray (1d)
            Homogenized stress tensor stored in matricial form.
        """
        return copy.deepcopy(self._hom_stress_mf)
    # -------------------------------------------------------------------------
    def get_oop_hom_comp(self):
        """Get homogenized strain or stress tensor out-of-plane component.

        Returns
        -------
        oop_hom_comp : float
            Homogenized strain or stress tensor out-of-plane component.
        """
        if self._problem_type == 1:
            oop_hom_comp = copy.deepcopy(self._hom_stress_33)
        elif self._problem_type == 2:
            oop_hom_comp = copy.deepcopy(self._hom_strain_33)
        else:
            oop_hom_comp = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return oop_hom_comp
    # -------------------------------------------------------------------------
    def get_inc_hom_strain_mf(self):
        """Get incremental homogenized strain tensor (matricial form).

        *Infinitesimal strains*:

        .. math::

           \\Delta \\boldsymbol{\\varepsilon}_{n + 1} =
               \\boldsymbol{\\varepsilon}_{n + 1} -
               \\boldsymbol{\\varepsilon}_{n}

        where :math:`\\Delta \\boldsymbol{\\varepsilon}` is the incremental
        homogenized infinitesimal strain tensor,
        :math:`\\boldsymbol{\\varepsilon}` is the homogenized infinitesimal
        strain tensor, :math:`n+1` denotes the current increment, and
        :math:`n` denotes the last converged increment.

        ----

        *Finite strains*:

        .. math::

           (\\boldsymbol{F}_{\\Delta})_{n + 1} =
               \\boldsymbol{F}_{n + 1}
               (\\boldsymbol{F}_{n})^{-1}

        where :math:`\\boldsymbol{F}_{\\Delta}` is the homogenized incremental
        deformation gradient, :math:`\\boldsymbol{F}` is the homogenized
        deformation gradient, :math:`n+1` denotes the current increment,
        and :math:`n` denotes the last converged increment.

        ----

        Returns
        -------
        inc_hom_strain_mf : numpy.ndarray (1d)
            Incremental homogenized strain tensor stored in matricial form:
            infinitesimal strain tensor (infinitesimal strains) or deformation
            gradient (finite strains).
        """
        # Compute incremental homogenized strain tensor
        if self._strain_formulation == 'infinitesimal':
            # Additive decomposition of infinitesimal strain tensor
            inc_hom_strain_mf = self._hom_strain_mf - self._hom_strain_old_mf
        else:
            # Build homogenized deformation gradient
            hom_strain = mop.get_tensor_from_mf(self._hom_strain_mf,
                                                self._n_dim,
                                                self._comp_order_nsym)
            hom_strain_old = mop.get_tensor_from_mf(self._hom_strain_old_mf,
                                                    self._n_dim,
                                                    self._comp_order_nsym)
            # Multiplicative decomposition of deformation gradient
            inc_hom_strain = np.matmul(hom_strain,
                                       np.linalg.inv(hom_strain_old))
            # Get deformation gradient (matricial form)
            inc_hom_strain_mf = mop.get_tensor_mf(inc_hom_strain, self._n_dim,
                                                  self._comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_hom_strain_mf
    # -------------------------------------------------------------------------
    def get_inc_hom_stress_mf(self):
        """Get incremental homogenized stress tensor (matricial form).

        *Infinitesimal strains*:

        .. math::

           \\Delta \\boldsymbol{\\sigma}_{n + 1} =
               \\boldsymbol{\\sigma}_{n + 1} -
               \\boldsymbol{\\sigma}_{n}

        where :math:`\\Delta \\boldsymbol{\\sigma}` is the incremental
        homogenized Cauchy stress tensor, :math:`\\boldsymbol{\\sigma}` is the
        homogenized Cauchy stress tensor, :math:`n+1` denotes the current
        increment, and :math:`n` denotes the last converged increment.

        ----

        *Finite strains*:

        .. math::

           \\Delta \\boldsymbol{P}_{n + 1} =
               \\boldsymbol{P}_{n + 1} -
               \\boldsymbol{P}_{n}

        where :math:`\\Delta \\boldsymbol{P}` is the incremental homogenized
        first Piola-Kirchhoff stress tensor, :math:`\\boldsymbol{P}` is the
        homogenized first Piola-Kirchhoff stress tensor, :math:`n+1` denotes
        the current increment, and :math:`n` denotes the last converged
        increment.

        ----

        Returns
        -------
        inc_hom_stress_mf : numpy.ndarray (1d)
            Incremental homogenized stress tensor stored in matricial form:
            Cauchy stress tensor (infinitesimal strains) or first
            Piola-Kirchhoff stress tensor (finite strains).
        """
        # Compute incremental homogenized stress tensor
        inc_hom_stress_mf = self._hom_stress_mf - self._hom_stress_old_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_hom_stress_mf
    # -------------------------------------------------------------------------
    def get_hom_strain_old_mf(self):
        """Get last converged homogenized strain tensor (matricial form).

        Returns
        -------
        hom_strain_old_mf : numpy.ndarray (1d)
            Last converged homogenized strain tensor stored in matricial form:
            infinitesimal strain tensor (infinitesimal strains) or deformation
            gradient (finite strains).
        """
        return copy.deepcopy(self._hom_strain_old_mf)
    # -------------------------------------------------------------------------
    def get_hom_stress_old_mf(self):
        """Get last converged homogenized stress tensor (matricial form).

        Returns
        -------
        hom_stress_old_mf : numpy.ndarray (1d)
            Last converged homogenized stress tensor stored in matricial form:
            Cauchy stress tensor (infinitesimal strains) or first
            Piola-Kirchhoff stress tensor (finite strains).
        """
        return copy.deepcopy(self._hom_stress_old_mf)
    # -------------------------------------------------------------------------
    def get_material_phases(self):
        """Get RVE material phases.

        Returns
        -------
        material_phases : list[str]
            RVE material phases labels (str).
        """
        return copy.deepcopy(self._material_phases)
    # -------------------------------------------------------------------------
    def get_material_phases_properties(self):
        """Get RVE material phases constitutive properties.

        Returns
        -------
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to
            each material phase (key, str).
        """
        return copy.deepcopy(self._material_phases_properties)
    # -------------------------------------------------------------------------
    def get_material_phases_models(self):
        """Get RVE material phases constitutive models.

        Returns
        -------
        _material_phases_models : dict
            Material constitutive model (item, ConstitutiveModel) associated to
            each material phase (key, str).
        """
        return copy.deepcopy(self._material_phases_models)
    # -------------------------------------------------------------------------
    def get_material_phases_vf(self):
        """Get RVE material phases volume fraction.

        Returns
        -------
        material_phases_vf : dict
            Volume fraction (item, float) associated to each material phase
            (key, str).
        """
        return copy.deepcopy(self._material_phases_vf)
    # -------------------------------------------------------------------------
    def get_clusters_def_gradient_mf(self):
        """Get deformation gradient of each material cluster.

        Returns
        -------
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.ndarray (1d)) associated to each
            material cluster (key, str), stored in matricial form.
        """
        return copy.deepcopy(self._clusters_def_gradient_mf)
    # -------------------------------------------------------------------------
    def get_clusters_def_gradient_old_mf(self):
        """Get last converged deformation gradient of each material cluster.

        Returns
        -------
        clusters_def_gradient_old_mf : dict
            Last converged deformation gradient (item, numpy.ndarray (1d))
            associated to each material cluster (key, str), stored in matricial
            form.
        """
        return copy.deepcopy(self._clusters_def_gradient_old_mf)
    # -------------------------------------------------------------------------
    def get_clusters_state(self):
        """Get material state variables of each material cluster.

        Returns
        -------
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            to each material cluster (key, str).
        """
        return copy.deepcopy(self._clusters_state)
    # -------------------------------------------------------------------------
    def get_clusters_state_old(self):
        """Get last converged material state variables of each mat. cluster.

        Returns
        -------
        clusters_state_old : dict
            Last converged material constitutive model state variables
            (item, dict) associated to each material cluster (key, str).
        """
        return copy.deepcopy(self._clusters_state_old)
    # -------------------------------------------------------------------------
    def get_clusters_tangent_mf(self):
        """Get material consistent tangent modulus of each material cluster.

        Returns
        -------
        clusters_tangent_mf : dict
            Material consistent tangent modulus (item, numpy.ndarray)
            associated to each material cluster (key, str), stored in matricial
            form.
        """
        return copy.deepcopy(self._clusters_tangent_mf)
    # -------------------------------------------------------------------------
    @staticmethod
    def _material_su_interface(strain_formulation, problem_type,
                               constitutive_model, def_gradient_old,
                               inc_strain, state_variables_old):
        """Material constitutive state update interface.

        This material constitutive state update interface contemplates three
        different families of constitutive models: (1) infinitesimal strains
        constitutive models, (2) finite strains constitutive models, and
        (3) isotropic hyperelastic-based finite strain constitutive models
        whose finite strain extension (from infinitesimal counterpart) is
        purely kinematical.

        This interface is schematically illustrated in Figure 5.3 of
        Ferreira (2022) [1]_, and the last family of constitutive models is
        described on Appendix F.

        .. [1] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        constitutive_model : ConstitutiveModel
            Material constitutive model.
        def_gradient_old : numpy.ndarray (2d)
            Last converged deformation gradient.
        inc_strain : numpy.ndarray (2d)
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged material constitutive model state variables.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : ndarray
            Material constitutive model material consistent tangent modulus in
            matricial form.
        """
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # Get material phase constitutive model strain type
        strain_type = constitutive_model.get_strain_type()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental spatial logarithmic strain tensor
        if strain_formulation == 'finite' and strain_type == 'finite-kinext':
            # Save incremental deformation gradient
            inc_def_gradient = copy.deepcopy(inc_strain)
            # Compute deformation gradient
            def_gradient = np.matmul(inc_def_gradient, def_gradient_old)
            # Get last converged elastic spatial logarithmic strain tensor
            e_log_strain_old_mf = state_variables_old['e_strain_mf']
            e_log_strain_old = mop.get_tensor_from_mf(e_log_strain_old_mf,
                                                      n_dim, comp_order_sym)
            # Compute incremental spatial logarithmic strain tensor
            inc_strain = MaterialState.compute_inc_log_strain(
                e_log_strain_old, inc_def_gradient=inc_def_gradient)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform state update and compute material consistent tangent modulus
        state_variables, consistent_tangent_mf = \
            constitutive_model.state_update(inc_strain, state_variables_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Cauchy stress tensor and material consistent tangent modulus
        if (not state_variables['is_su_fail']
                and strain_formulation == 'finite'
                and strain_type == 'finite-kinext'):
            # Get Kirchhoff stress tensor (matricial form)
            kirchhoff_stress_mf = state_variables['stress_mf']
            # Build Kirchhoff stress tensor
            kirchhoff_stress = mop.get_tensor_from_mf(kirchhoff_stress_mf,
                                                      n_dim, comp_order_sym)
            # Compute first Piola-Kirchhoff stress tensor
            first_piola_stress = first_piola_from_kirchhoff(def_gradient,
                                                            kirchhoff_stress)
            # Get first Piola-Kirchhoff stress tensor (matricial form)
            first_piola_stress_mf = mop.get_tensor_mf(first_piola_stress,
                                                      n_dim, comp_order_nsym)
            # Get first Piola-Kirchhoff stress tensor out-of-plane component
            if problem_type == 1:
                first_piola_stress_33 = state_variables['stress_33']
            # Update stress tensor
            state_variables['stress_mf'] = first_piola_stress_mf
            if problem_type == 1:
                state_variables['stress_33'] = first_piola_stress_33
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get last converged elastic spatial logarithmic strain tensor
            e_log_strain_old_mf = state_variables_old['e_strain_mf']
            e_log_strain_old = mop.get_tensor_from_mf(e_log_strain_old_mf,
                                                      n_dim, comp_order_sym)
            # Compute Cauchy stress tensor (matricial form)
            cauchy_stress = cauchy_from_kirchhoff(def_gradient,
                                                  kirchhoff_stress)
            # Get infinitesimal strains consistent tangent modulus
            inf_consistent_tangent = mop.get_tensor_from_mf(
                consistent_tangent_mf, n_dim, comp_order_sym)
            # Compute spatial consistent tangent modulus
            spatial_consistent_tangent = \
                MaterialState.compute_spatial_tangent_modulus(
                    e_log_strain_old, def_gradient_old, inc_def_gradient,
                    cauchy_stress, inf_consistent_tangent)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute material consistent tangent modulus (matricial form)
            material_consistent_tangent = \
                material_from_spatial_tangent_modulus(
                    spatial_consistent_tangent, def_gradient)
            consistent_tangent_mf = mop.get_tensor_mf(
                material_consistent_tangent, n_dim, comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables, consistent_tangent_mf
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_inc_log_strain(e_log_strain_old, inc_def_gradient):
        """Compute incremental spatial logarithmic strain.

        *Incremental spatial logarithmic strain*:

        .. math::

           \\Delta \\boldsymbol{\\varepsilon}_{n + 1} =
               \\boldsymbol{\\varepsilon}_{n + 1}^{e, \\, \\text{trial}} -
               \\boldsymbol{\\varepsilon}_{n}^{e}

        where :math:`\\Delta \\boldsymbol{\\varepsilon}` is the incremental
        spatial logarithmic strain tensor,
        :math:`\\boldsymbol{\\varepsilon}^{e, \\, \\text{trial}}` is the
        elastic trial spatial logarithmic strain tensor,
        :math:`\\boldsymbol{\\varepsilon}^{e}` is the elastic spatial
        logarithmic strain tensor, :math:`n+1` denotes
        the current increment, and :math:`n` denotes the last converged
        increment.

        ----

        *Elastic trial left Cauchy-Green strain tensor*:

        .. math::

           \\boldsymbol{\\varepsilon}_{n + 1}^{e, \\, \\text{trial}} =
               \\dfrac{1}{2} \\ln ( \\boldsymbol{B}^{e, \\,
               \\text{trial}}_{n+1} )
               = \\dfrac{1}{2} \\ln \\Big( (\\boldsymbol{F}_{\\Delta})_{n+1}
               \\boldsymbol{B}^{e}_{n} (\\boldsymbol{F}_{\\Delta})_{n+1}^{T}
               \\Big)

        where :math:`\\boldsymbol{\\varepsilon}^{e, \\, \\text{trial}}` is the
        elastic trial spatial logarithmic strain tensor,
        :math:`\\boldsymbol{B}^{e, \\, \\text{trial}}` is the elastic trial
        left Cauchy-Green strain tensor,
        :math:`\\boldsymbol{F}_{\\Delta}` is the incremental deformation
        gradient, :math:`\\boldsymbol{B}^{e}` is the elastic left Cauchy-Green
        strain tensor, :math:`n+1` denotes the current increment, and :math:`n`
        denotes the last converged increment.

        The definition of the elastic trial spatial logarithmic strain tensor
        can be found in Appendix F.4 of Ferreira (2022) [#]_ (see Equations
        (F.14) and (F.15)).

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        e_log_strain_old : numpy.ndarray (2d)
            Last converged elastic spatial logarithmic strain tensor.
        inc_def_gradient : numpy.ndarray (2d)
            Incremental deformation gradient.

        Returns
        -------
        inc_log_strain : numpy.ndarray (2d)
            Incremental spatial logarithmic strain.
        """
        # Compute last converged elastic left Cauchy-Green strain tensor
        e_cauchy_green_old = top.isotropic_tensor('exp', 2.0*e_log_strain_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic trial left Cauchy-Green strain tensor
        e_trial_cauchy_green = np.matmul(
            inc_def_gradient, np.matmul(e_cauchy_green_old,
                                        np.transpose(inc_def_gradient)))
        # Compute elastic trial spatial logarithmic strain
        e_trial_log_strain = 0.5*top.isotropic_tensor('log',
                                                      e_trial_cauchy_green)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental spatial logarithmic strain
        inc_log_strain = e_trial_log_strain - e_log_strain_old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_log_strain
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_spatial_tangent_modulus(e_log_strain_old, def_gradient_old,
                                        inc_def_gradient, cauchy_stress,
                                        inf_consistent_tangent):
        """Compute finite strain spatial consistent tangent modulus.

        .. math::

           \\mathsf{a}_{ijkl} = \\dfrac{1}{2 \\det (\\boldsymbol{F})} \\,
                                \\left[ \\mathsf{D} : \\mathsf{L} : \\mathsf{B}
                                \\right]_{ijkl} - \\sigma_{il} \\delta_{jk}

        where :math:`\\mathbf{\\mathsf{a}}` is the spatial consistent tangent
        modulus, :math:`\\mathbf{\\mathsf{D}}` is the derivative of the
        Kirchhoff stress tensor with respect to the spatial logarithmic strain
        tensor, :math:`\\mathbf{\\mathsf{L}}` is the derivative of the tensor
        logarithm function evaluated at the elastic trial left Cauchy-Green
        strain tensor, :math:`\\mathbf{\\mathsf{B}}` is computed from the
        elastic trial left Cauchy-Green strain tensor components,
        :math:`\\boldsymbol{\\sigma}` is the Cauchy stress tensor, and
        :math:`\\delta_{ij}` is the Kronecker delta.

        The detailed definition of the finite strain spatial consistent tangent
        modulus of isotropic hyperelastic-based finite strain elastoplastic
        constitutive models whose finite strain formalism is purely kinematical
        can be found in Appendix F.4 of Ferreira (2022) [#]_ (see Equation
        (F.18)).

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        e_log_strain_old : numpy.ndarray (2d)
            Last converged elastic spatial logarithmic strain tensor.
        def_gradient_old : numpy.ndarray (2d)
            Last converged deformation gradient.
        inc_def_gradient : numpy.ndarray (2d)
            Incremental deformation gradient.
        cauchy_stress : numpy.ndarray (2d)
            Cauchy stress tensor.
        inf_consistent_tangent : numpy.ndarray (4d)
            Infinitesimal consistent tangent modulus.

        Returns
        -------
        spatial_consistent_tangent : numpy.ndarray (4d)
            Spatial consistent tangent modulus.
        """
        # Get problem number of spatial dimensions
        n_dim = inc_def_gradient.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute deformation gradient
        def_gradient = np.matmul(inc_def_gradient, def_gradient_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute last converged elastic left Cauchy-Green strain tensor
        e_cauchy_green_old = top.isotropic_tensor('exp', 2.0*e_log_strain_old)
        # Compute elastic trial left Cauchy-Green strain tensor
        e_trial_cauchy_green = np.matmul(
            inc_def_gradient, np.matmul(e_cauchy_green_old,
                                        np.transpose(inc_def_gradient)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Cauchy-Green-related fourth-order tensor
        fo_cauchy_green = np.zeros(4*(n_dim,))
        # Compute Cauchy-Green-related fourth-order tensor
        for i, j, k, l in it.product(range(n_dim), repeat=4):
            fo_cauchy_green[i, j, k, l] = \
                top.dd(i, k)*e_trial_cauchy_green[j, l] \
                + top.dd(j, k)*e_trial_cauchy_green[i, l]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of tensor logarithm evaluated at elastic trial
        # left Cauchy-Green strain tensor
        fo_log_derivative = \
            top.derivative_isotropic_tensor('log', e_trial_cauchy_green)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize spatial consistent tangent modulus
        spatial_consistent_tangent = np.zeros(4*(n_dim,))
        # Compute spatial consistent tangent modulus
        spatial_consistent_tangent = (1.0/(2.0*np.linalg.det(def_gradient))) \
            * (top.ddot44_1(inf_consistent_tangent,
                            top.ddot44_1(fo_log_derivative, fo_cauchy_green)))
        for i, j, k, l in it.product(range(n_dim), repeat=4):
            spatial_consistent_tangent[i, j, k, l] += \
                -1.0*cauchy_stress[i, l]*top.dd(j, k)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return spatial_consistent_tangent
    # -------------------------------------------------------------------------
    def clustering_adaptivity_update(self, phase_clusters, clusters_vf,
                                     adaptive_clustering_map):
        """Update cluster variables according to clustering adaptivity step.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list[int]) associated to each material
            phase (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster
            (key, str).
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels
            (item, list[int]) resulting from the refinement of each target
            cluster (key, str)) for each material phase (key, str).
        """
        # Update CRVE material state clusters labels and volume fraction
        self.set_phase_clusters(phase_clusters, clusters_vf)
        # Group cluster-related dictionaries
        cluster_dicts = [self._clusters_def_gradient_mf,
                         self._clusters_def_gradient_old_mf,
                         self._clusters_state, self._clusters_state_old,
                         self._clusters_tangent_mf]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in adaptive_clustering_map.keys():
            # Loop over material phase target clusters
            for target_cluster in adaptive_clustering_map[mat_phase].keys():
                # Get list of target's child clusters
                child_clusters = \
                    adaptive_clustering_map[mat_phase][target_cluster]
                # Loop over cluster-keyd dictionaries
                for cluster_dict in cluster_dicts:
                    # Loop over child clusters and build their items
                    for child_cluster in child_clusters:
                        cluster_dict[str(child_cluster)] = \
                            copy.deepcopy(cluster_dict[target_cluster])
                    # Remove target cluster item
                    cluster_dict.pop(target_cluster)
#
#                                        Available material constitutive models
# =============================================================================
def get_available_material_models(model_source='crate'):
    """Get available material constitutive models.

    Parameters
    ----------
    model_source : {'crate',}, default='crate'
        Material constitutive model source.

    Returns
    -------
    available_mat_models : list[str]
        Available material constitutive models.
    """
    # Set the available material constitutive models from a given source
    if model_source == 'crate':
        # CRATE material constitutive models
        available_mat_models = ['elastic', 'von_mises', 'stvenant_kirchhoff']
    else:
        raise RuntimeError('Unknown material constitutive model source.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return available_mat_models
