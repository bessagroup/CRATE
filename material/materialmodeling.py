#
# Material State (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing the required procedures to perform the state update of the material
# clusters constitutive models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Refactoring and OOP implementation.
#                                 | Finite strain extension.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# Constitutive models
from material.models.linear_elastic import Elastic
from material.models.von_mises import VonMises
# Links constitutive models
from links.material.models.links_elastic import LinksElastic
from links.material.models.links_von_mises import LinksVonMises
#
#                                                                             Material state
# ==========================================================================================
class MaterialState:
    '''CRVE material constitutive state.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _material_phases_models : dict
        Material constitutive model (item, ConstitutiveModel) associated to each material
        phase (key, str).
    _phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase
        (key, str).
    _clusters_vf : dict
        Volume fraction (item, float) associated to each material cluster (key, str).
    _clusters_def_gradient_mf : dict
        Deformation gradient (item, 1darray) associated to each material cluster (key, str),
        stored in matricial form.
    _clusters_def_gradient_old_mf : dict
        Last converged deformation gradient (item, 1darray) associated to each material
        cluster (key, str), stored in matricial form.
    _clusters_state : dict
        Material constitutive model state variables (item, dict) associated to each
        material cluster (key, str).
    _clusters_state_old : dict
        Last converged material constitutive model state variables (item, dict) associated
        to each material cluster (key, str).
    _clusters_tangent_mf : dict
        Material consistent tangent modulus (item, ndarray) associated to each material
        cluster (key, str), stored in matricial form.
    _hom_strain_mf : 1darray
        Homogenized strain tensor stored in matricial form.
    _hom_strain_33 : float
        Homogenized strain tensor out-of-plane component.
    _hom_stress_mf : 1darray
        Homogenized stress tensor stored in matricial form.
    _hom_stress_33 : float
        Homogenized stress tensor out-of-plane component.
    _hom_strain_old_mf : 1darray
        Last converged homogenized strain tensor stored in matricial form.
    _hom_stress_old_mf : 1darray
        Last converged homogenized stress tensor stored in matricial form.
    '''
    def __init__(self, strain_formulation, problem_type, material_phases,
                 material_phases_properties, material_phases_vf):
        '''CRVE material constitutive state constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        material_phases : list
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to each material
            phase (key, str).
        material_phases_vf : dict
            Volume fraction (item, float) associated to each material phase (key, str).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_phases = copy.deepcopy(material_phases)
        self._material_phases_properties = copy.deepcopy(material_phases_properties)
        self._material_phases_vf = copy.deepcopy(material_phases_vf)
        self._phase_clusters = None
        self._clusters_vf = None
        self._material_phases_models = {mat_phase: None for mat_phase in material_phases}
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
    # --------------------------------------------------------------------------------------
    def init_constitutive_model(self, mat_phase, model_keyword, model_source='crate'):
        '''Initialize material phase constitutive model.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        model_keyword : str
            Material constitutive model input data file keyword.
        model_source : str, {'crate', }, default='crate'
            Material constitutive model source.
        '''
        # Initialize material phase constitutive model
        if model_source == 'crate':
            if model_keyword == 'elastic':
                constitutive_model = Elastic(self._strain_formulation, self._problem_type,
                                             self._material_phases_properties[mat_phase])
            elif model_keyword == 'von_mises':
                constitutive_model = VonMises(self._strain_formulation, self._problem_type,
                                              self._material_phases_properties[mat_phase])
            else:
                raise RuntimeError('Unknown constitutive model from CRATE\'s source.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_source == 'links':
            if model_keyword == 'ELASTIC':
                constitutive_model = \
                    LinksElastic(self._strain_formulation, self._problem_type,
                                 self._material_phases_properties[mat_phase])
            elif model_keyword == 'VON_MISES':
                constitutive_model = \
                    LinksVonMises(self._strain_formulation, self._problem_type,
                                  self._material_phases_properties[mat_phase])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown material constitutive model source.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update material phases constitutive models
        self._material_phases_models[mat_phase] = constitutive_model
    # --------------------------------------------------------------------------------------
    def init_clusters_state(self):
        '''Initialize clusters state variables.'''
        # Initialize clusters state variables
        self._clusters_state = {}
        self._clusters_state_old = {}
        # Initialize clusters deformation gradient
        self._clusters_def_gradient_mf = {}
        self._clusters_def_gradient_old_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set second-order identity tensor matricial form
        soid_mf = mop.get_tensor_mf(np.eye(self._n_dim), self._n_dim, self._comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute initial state homogenized strain and stress tensors
        self.perform_state_homogenization()
        self.set_last_state_homogenization(copy.deepcopy(self._hom_strain_mf),
                                           copy.deepcopy(self._hom_stress_mf))
    # --------------------------------------------------------------------------------------
    def set_phase_clusters(self, phase_clusters, clusters_vf):
        '''Set CRVE cluster labels and volume fractions associated to each material phase.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster (key, str).
        '''
        self._phase_clusters = copy.deepcopy(phase_clusters)
        self._clusters_vf = copy.deepcopy(clusters_vf)
    # --------------------------------------------------------------------------------------
    def get_clusters_inc_strain_mf(self, global_inc_strain_mf):
        '''Get clusters incremental strain in matricial form.

        Parameters
        ----------
        global_inc_strain_mf : 1darray
            Global vector of clusters incremental strains stored in matricial form.

        Returns
        -------
        clusters_inc_strain_mf : dict
            Incremental strain (item, dict) associated to each material cluster (key, str),
            stored in matricial form.
        '''
        # Set strain components according to problem strain formulation
        if self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            comp_order = self._comp_order_sym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize dictionary of clusters incremental strain
        global_inc_strain_mf = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material cluster strain range indexes
        i_init = 0
        i_end = i_init + len(comp_order)
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster incremental strain (matricial form)
                inc_strain_mf = global_inc_strain_mf[i_init:i_end]
                # Store material cluster incremental strain (matricial form)
                global_inc_strain_mf[str(cluster)] = inc_strain_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return global_inc_strain_mf
    # --------------------------------------------------------------------------------------
    def update_clusters_state(self, clusters_inc_strain_mf):
        '''Update clusters state variables and associated consistent tangent modulus.

        Parameters
        ----------
        clusters_inc_strain_mf : dict
            Incremental strain (item, ndarray) associated to each material cluster
            (key, str), stored in matricial form.

        Returns
        -------
        su_fail_state : dict
            State update failure state.
        '''
        # Initialize state update failure state
        su_fail_state = {'is_su_fail': False, 'mat_phase': None, 'cluster': None}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase constitutive model
            constitutive_model = self._material_phases_models[mat_phase]
            # Get material constitutive model source
            source = constitutive_model.get_source()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster incremental strain tensor (matricial form)
                inc_strain_mf = clusters_inc_strain_mf[str(cluster)]
                if self._strain_formulation == 'infinitesimal':
                    # Infinitesimal strain tensor (symmetric)
                    comp_order = self._comp_order_sym
                else:
                    # Deformation gradient tensor (nonsymmetric)
                    comp_order = self._comp_order_nsym
                inc_strain = mop.get_tensor_mf(inc_strain_mf, self._n_dim, comp_order)
                # Get material cluster last converged state variables
                state_variables_old = self._clusters_state_old[str(cluster)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material cluster last converged deformation gradient tensor
                def_gradient_old_mf = self._clusters_def_gradient_old_mf[str(cluster)]
                def_gradient_old = mop.get_tensor_mf(def_gradient_old_mf, self._n_dim,
                                                     self._comp_order_nsym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform state update through the suitable material interface
                if source == 'crate':
                    state_variables, consistent_tangent_mf = \
                        self._material_su_interface(constitutive_model, def_gradient_old,
                                                    copy.deepcopy(inc_strain),
                                                    copy.deepcopy(state_variables_old))
                else:
                    state_variables, consistent_tangent_mf = \
                        constitutive_model.state_update(copy.deepcopy(inc_strain),
                                                        copy.deepcopy(state_variables_old),
                                                        copy.deepcopy(def_gradient_old))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check state update failure status
                try:
                    if state_variables['is_su_fail']:
                        # Update state update failure status
                        su_fail_state = \
                            {'is_su_fail': True, 'mat_phase': mat_phase, 'cluster': cluster}
                        # Return
                        return su_fail_state
                except KeyError:
                    raise RuntimeError('Material constitutive model state variables must '
                                       'include the state update failure flag '
                                       '(\'is_su_fail\': bool)')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update cluster deformation gradient
                if self._strain_formulation == 'finite':
                    self._clusters_def_gradient_mf[str(cluster)] = \
                        np.matmul(inc_strain_mf, def_gradient_old_mf)
                # Update cluster state variables and material consistent tangent modulus
                self._clusters_state[str(cluster)] = state_variables
                self._clusters_tangent_mf[str(cluster)] = consistent_tangent_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return su_fail_state
    # --------------------------------------------------------------------------------------
    def update_state_homogenization(self):
        '''Update homogenized strain and stress tensors.'''
        # Set strain components according to problem strain formulation
        if self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            comp_order = self._comp_order_sym
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize incremental homogenized strain and stress tensors (matricial form)
        hom_strain_mf = np.zeros(len(comp_order))
        hom_stress_mf = np.zeros(len(comp_order))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Loop over material phase clusters
            for cluster in self._phase_clusters[mat_phase]:
                # Get material cluster strain and stress tensor (matricial form)
                strain_mf = self._clusters_state[str(cluster)]['strain_mf']
                stress_mf = self._clusters_state[str(cluster)]['stress_mf']
                # Add material cluster contribution to homogenized strain and stress tensors
                # (matricial form)
                hom_strain_mf = \
                    np.add(hom_strain_mf, self._clusters_vf[str(cluster)]*strain_mf)
                hom_stress_mf = \
                    np.add(hom_stress_mf, self._clusters_vf[str(cluster)]*stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update homogenized strain and stress tensors
        self._hom_strain_mf = copy.deepcopy(hom_strain_mf)
        self._hom_stress_mf = copy.deepcopy(hom_stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self._problem_type in [1, 2]:
            # Set out-of-plane stress component (2D plane strain problem) / strain component
            # (2D plane stress problem)
            if self._problem_type == 1:
                comp_name = 'stress_33'
            elif self._problem_type == 2:
                comp_name = 'strain_33'
            else:
                raise RuntimeError('Unknown plane problem type.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize homogenized out-of-plane component
            oop_hom_comp = 0.0
            # Loop over material phases
            for mat_phase in self._material_phases:
                # Loop over material phase clusters
                for cluster in self._phase_clusters[mat_phase]:
                    # Add material cluster contribution to the homogenized out-of-plane
                    # component component
                    oop_hom_comp = oop_hom_comp + self._clusters_vf[str(cluster)]*\
                        self._clusters_state[str(cluster)][comp_name]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update out-of-plane stress or strain component
            if self._problem_type == 1:
                self._hom_stress_33 = oop_hom_comp
            elif self._problem_type == 2:
                self._hom_strain_33 = oop_hom_comp
    # --------------------------------------------------------------------------------------
    def update_converged_state(self):
        '''Update last converged material state variables.'''
        self._clusters_def_gradient_old_mf = copy.deepcopy(self._clusters_def_gradient_mf)
        self._clusters_state_old = copy.deepcopy(self._clusters_state)
        self._hom_strain_old_mf = copy.deepcopy(self._hom_strain_mf)
        self._hom_stress_old_mf = copy.deepcopy(self._hom_stress_mf)
    # --------------------------------------------------------------------------------------
    def set_rewind_state_updated_clustering(self, phase_clusters, clusters_vf,
                                            clusters_state, clusters_def_gradient_mf):
        '''Set rewind state clustering-related variables according to updated clustering.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, 1darray) associated to each material cluster
            (key, str), stored in matricial form.
        '''
        self._phase_clusters = copy.deepcopy(phase_clusters)
        self._clusters_vf = copy.deepcopy(clusters_vf)
        self._clusters_state_old = copy.deepcopy(clusters_state)
        self._clusters_def_gradient_old_mf = copy.deepcopy(clusters_def_gradient_mf)
    # --------------------------------------------------------------------------------------
    def get_hom_strain_mf(self):
        '''Get homogenized strain tensor (matricial form).

        Returns
        -------
        hom_strain_mf : 1darray
            Homogenized strain tensor stored in matricial form.
        '''
        return copy.deepcopy(self._hom_strain_mf)
    # --------------------------------------------------------------------------------------
    def get_hom_stress_mf(self):
        '''Get homogenized stress tensor (matricial form).

        Returns
        -------
        hom_stress_mf : 1darray
            Homogenized stress tensor stored in matricial form.
        '''
        return copy.deepcopy(self._hom_stress_mf)
    # --------------------------------------------------------------------------------------
    def get_oop_hom_comp(self):
        '''Get homogenized strain or stress tensor out-of-plane component.

        Returns
        -------
        oop_hom_comp : float
            Homogenized strain or stress tensor out-of-plane component.
        '''
        if self._problem_type == 1:
            oop_hom_comp = copy.deepcopy(self._hom_stress_33)
        elif self._problem_type == 2:
            oop_hom_comp = copy.deepcopy(self._hom_strain_33)
        else:
            oop_hom_comp = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return oop_hom_comp
    # --------------------------------------------------------------------------------------
    def get_inc_hom_strain_mf(self):
        '''Get incremental homogenized strain tensor (matricial form).

        Returns
        -------
        inc_hom_strain_mf : 1darray
            Incremental homogenized strain tensor stored in matricial form.
        '''
        # Compute incremental homogenized strain tensor
        inc_hom_strain_mf = self._hom_strain_mf - self._hom_strain_old_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_hom_strain_mf
    # --------------------------------------------------------------------------------------
    def get_inc_hom_stress_mf(self):
        '''Get incremental homogenized stress tensor (matricial form).

        Returns
        -------
        inc_hom_stress_mf : 1darray
            Incremental homogenized stress tensor stored in matricial form.
        '''
        # Compute incremental homogenized stress tensor
        inc_hom_stress_mf = self._hom_stress_mf - self._hom_stress_old_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_hom_stress_mf
    # --------------------------------------------------------------------------------------
    def get_material_phases(self):
        '''Get RVE material phases.

        Returns
        -------
        material_phases : list
            RVE material phases labels (str).
        '''
        return copy.deepcopy(self._material_phases)
    # --------------------------------------------------------------------------------------
    def get_material_phases_properties(self):
        '''Get RVE material phases constitutive properties.

        Returns
        -------
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to each material
            phase (key, str).
        '''
        return copy.deepcopy(self._material_phases_properties)
    # --------------------------------------------------------------------------------------
    def get_material_phases_models(self):
        '''Get RVE material phases constitutive models.

        Returns
        -------
        _material_phases_models : dict
            Material constitutive model (item, ConstitutiveModel) associated to each
            material phase (key, str).
        '''
        return copy.deepcopy(self._material_phases_models)
    # --------------------------------------------------------------------------------------
    def get_material_phases_vf(self):
        '''Get RVE material phases volume fraction.

        Returns
        -------
        material_phases_vf : dict
            Volume fraction (item, float) associated to each material phase (key, str).
        '''
        return copy.deepcopy(self._material_phases_vf)
    # --------------------------------------------------------------------------------------
    def get_clusters_def_gradient_mf(self):
        '''Get deformation gradient (matricial form) associated to each material cluster.

        Returns
        -------
        clusters_def_gradient_mf : dict
            Deformation gradient (item, 1darray) associated to each material cluster
            (key, str), stored in matricial form.
        '''
        return copy.deepcopy(self._clusters_def_gradient_mf)
    # --------------------------------------------------------------------------------------
    def get_clusters_state(self):
        '''Get material state variables associated to each material cluster.

        Returns
        -------
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).
        '''
        return copy.deepcopy(self._clusters_state)
    # --------------------------------------------------------------------------------------
    def get_clusters_state_old(self):
        '''Get last converged material state variables associated to each material cluster.

        Returns
        -------
        clusters_state_old : dict
            Last converged material constitutive model state variables (item, dict)
            associated to each material cluster (key, str).
        '''
        return copy.deepcopy(self._clusters_state_old)
    # --------------------------------------------------------------------------------------
    def get_clusters_tangent_mf(self):
        '''Get material consistent tangent modulus associated to each material cluster.

        Returns
        -------
        clusters_tangent_mf : dict
            Material consistent tangent modulus (item, ndarray) associated to each material
            cluster (key, str), stored in matricial form.
        '''
        return copy.deepcopy(self._clusters_tangent_mf)
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _material_su_interface(problem_type, constitutive_model, def_gradient_old,
                               inc_strain, state_variables_old):
        '''Material constitutive state update interface.

        Parameters
        ----------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        constitutive_model : ConstitutiveModel
            Material constitutive model.
        def_gradient_old : 2darray
            Last converged deformation gradient.
        inc_strain : 2darray,
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged material constitutive model state variables.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : ndarray
            Material constitutive model material consistent tangent modulus in matricial
            form.
        '''
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # Get material phase constitutive model strain type
        strain_type = constitutive_model.get_strain_type()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental logarithmic strain tensor
        if strain_type == 'finite-kinext':
            # Save incremental deformation gradient
            inc_def_gradient = copy.deepcopy(inc_strain)
            # Compute deformation gradient
            def_gradient = def_gradient_old + inc_def_gradient
            # Get last converged elastic logarithmic strain tensor
            e_log_strain_old_mf = state_variables_old['e_strain_mf']
            e_log_strain_old = mop.get_tensor_from_mf(e_log_strain_old_mf, n_dim,
                                                      comp_order_sym)
            # Compute incremental logarithmic strain tensor
            inc_strain = MaterialState.compute_inc_log_strain(e_log_strain_old,
                                                              inc_def_gradient=inc_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform state update and compute material consistent tangent modulus
        state_variables, consistent_tangent_mf = \
            constitutive_model.state_update(inc_strain, state_variables_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Cauchy stress tensor and material consistent tangent modulus
        if strain_type == 'finite-kinext':
            # Get Kirchhoff stress tensor
            kirchhoff_stress_mf = state_variables['stress_mf']
            kirchhoff_stress = mop.get_tensor_from_mf(kirchhoff_stress_mf, n_dim,
                                                      comp_order_sym)
            # Compute Cauchy stress tensor (matricial form)
            cauchy_stress = MaterialState.cauchy_from_kirchhoff(def_gradient,
                                                                kirchhoff_stress)
            cauchy_stress_mf = mop.get_tensor_mf(cauchy_stress, n_dim, comp_order_sym)
            # Update stress tensor
            state_variables['stress_mf'] = cauchy_stress_mf
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get last converged elastic logarithmic strain tensor
            e_log_strain_old_mf = state_variables_old['e_strain_mf']
            e_log_strain_old = mop.get_tensor_from_mf(e_log_strain_old_mf, n_dim,
                                                      comp_order_sym)
            # Compute spatial consistent tangent modulus
            spatial_consistent_tangent = \
                MaterialState.compute_spatial_tangent_modulus(e_log_strain_old,
                    def_gradient_old, inc_def_gradient, cauchy_stress,
                        consistent_tangent_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute material consistent tangent modulus (matricial form)
            material_consistent_tangent = \
                MaterialState.material_from_spatial_tangent_modulus(
                    spatial_consistent_tangent, def_gradient)
            consistent_tangent_mf = mop.get_tensor_mf(material_consistent_tangent, n_dim,
                                                      comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables, consistent_tangent_mf
    # --------------------------------------------------------------------------------------
    @staticmethod
    def compute_inc_log_strain(e_log_strain_old, inc_def_gradient):
        '''Compute incremental logarithmic strain.

        Parameters
        ----------
        e_log_strain_old : 2darray
            Last converged elastic logarithmic strain tensor.
        inc_def_gradient : 2darray
            Incremental deformation gradient.

        Returns
        -------
        inc_log_strain : 2darray
            Incremental logarithmic strain.
        '''
        # Compute last converged elastic left Cauchy-Green strain tensor
        e_cauchy_green_old = top.isotropic_tensor('exp', 2.0*e_log_strain_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute elastic trial left Cauchy-Green strain tensor
        e_trial_cauchy_green = np.matmul(inc_def_gradient, np.matmul(e_cauchy_green_old,
            np.transpose(inc_def_gradient)))
        # Compute elastic trial logarithmic strain
        e_trial_log_strain = 0.5*top.isotropic_tensor('log', e_trial_cauchy_green)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental logarithmic strain
        inc_log_strain = e_trial_log_strain - e_log_strain_old
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return inc_log_strain
    # --------------------------------------------------------------------------------------
    @staticmethod
    def cauchy_from_kirchhoff(def_gradient, kirchhoff_stress):
        '''Compute Cauchy stress tensor from Kirchhoff stress tensor.

        Parameters
        ----------
        def_gradient : 2darray
            Deformation gradient.
        kirchhoff_stress : 2darray
            Kirchhoff stress tensor.

        Returns
        -------
        cauchy_stress : 2darray
            Cauchy stress tensor.
        '''
        # Compute Cauchy stress tensor
        cauchy_stress = (1.0/np.linalg.det(def_gradient))*kirchhoff_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return cauchy_stress
    # --------------------------------------------------------------------------------------
    @staticmethod
    def first_piola_from_kirchhoff(def_gradient, kirchhoff_stress):
        '''Compute First Piola-Kirchhoff stress tensor from Kirchhoff stress tensor.

        Parameters
        ----------
        def_gradient : 2darray
            Deformation gradient.
        kirchhoff_stress : 2darray
            Kirchhoff stress tensor.

        Returns
        -------
        first_piola_stress : 2darray
            First Piola-Kirchhoff stress tensor.
        '''
        # Compute First Piola-Kirchhoff stress tensor
        first_piola_stress = np.matmul(kirchhoff_stress,
                                       np.transpose(np.linalg.inv(def_gradient)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return first_piola_stress
    # --------------------------------------------------------------------------------------
    @staticmethod
    def compute_spatial_tangent_modulus(e_log_strain_old, def_gradient_old,
                                        inc_def_gradient, cauchy_stress,
                                        inf_consistent_tangent):
        '''Compute finite strains spatial consistent tangent modulus.

        Isotropic hyperelastic-based finite strain elastoplastic constitutive models whose
        finite strain formalism is purely kinematical.

        Parameters
        ----------
        e_log_strain_old : 2darray
            Last converged elastic logarithmic strain tensor.
        def_gradient_old : 2darray
            Last converged deformation gradient.
        inc_def_gradient : 2darray
            Incremental deformation gradient.
        cauchy_stress : 2darray
            Cauchy stress tensor.
        inf_consistent_tangent : 4darray
            Infinitesimal consistent tangent modulus.

        Returns
        -------
        spatial_consistent_tangent : 4darray
            Spatial consistent tangent modulus.
        '''
        # Get problem number of spatial dimensions
        n_dim = inc_def_gradient.shape[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute deformation gradient
        def_gradient = np.matmul(inc_def_gradient, def_gradient_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute last converged elastic left Cauchy-Green strain tensor
        e_cauchy_green_old = top.isotropic_tensor('exp', 2.0*e_log_strain_old)
        # Compute elastic trial left Cauchy-Green strain tensor
        e_trial_cauchy_green = np.matmul(inc_def_gradient, np.matmul(e_cauchy_green_old,
            np.transpose(inc_def_gradient)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Cauchy-Green-related fourth-order tensor
        fo_cauchy_green = np.zeros(4*(n_dim,))
        # Compute Cauchy-Green-related fourth-order tensor
        for i, j, k, l in it.product(range(n_dim), repeat=4):
            fo_cauchy_green[i, j, k, l] = top.dd(i, k)*e_trial_cauchy_green[j, l] + \
                                          top.dd(j, k)*e_trial_cauchy_green[i, l]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute derivative of tensor logarithm evaluated at elastic trial left
        # Cauchy-Green strain tensor
        fo_log_derivative = top.derivative_isotropic_tensor('log', e_trial_cauchy_green)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize spatial consistent tangent modulus
        spatial_consistent_tangent = np.zeros(4*(n_dim,))
        # Compute spatial consistent tangent modulus
        spatial_consistent_tangent = \
            (1.0/(2.0*np.linalg.det(def_gradient)))*(top.ddot44_1(inf_consistent_tangent,
                top.ddot44_1(fo_log_derivative, fo_cauchy_green)))
        for i, j, k, l in it.product(range(n_dim), repeat=4):
            spatial_consistent_tangent[i, j, k, l] += -1.0*cauchy_stress[i, l]*top.dd(j, k)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return spatial_consistent_tangent
    # --------------------------------------------------------------------------------------
    @staticmethod
    def material_from_spatial_tangent_modulus(spatial_consistent_tangent, def_gradient):
        '''Compute material consistent tangent modulus from spatial counterpart.

        Parameters
        ----------
        spatial_consistent_tangent : 4darray
            Spatial consistent tangent modulus.
        def_gradient : 2darray
            Deformation gradient.

        Returns
        -------
        material_consistent_tangent : 4darray
            Material consistent tangent modulus.
        '''
        # Compute inverse of deformation gradient
        def_gradient_inv = np.linalg.inv(def_gradient)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute material consistent tangent modulus
        material_consistent_tangent = np.linalg.det(def_gradient)*top.dot42_2(
            top.dot42_1(spatial_consistent_tangent, def_gradient_inv), def_gradient_inv)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return material_consistent_tangent
    # --------------------------------------------------------------------------------------
    def clustering_adaptivity_update(self, phase_clusters, clusters_vf,
                                     adaptive_clustering_map):
        '''Update cluster-related variables according to clustering adaptivity step.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_vf : dict
            Volume fraction (item, float) associated to each material cluster (key, str).
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str).
        '''
        # Update CRVE material state clusters labels and volume fraction
        self.set_phase_clusters(phase_clusters, clusters_vf)
        # Group cluster-related dictionaries
        cluster_dicts = [self._clusters_def_gradient_mf, self._clusters_def_gradient_old_mf,
                         self._clusters_state, self._clusters_state_old,
                         self._clusters_tangent_mf]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in adaptive_clustering_map.keys():
            # Loop over material phase target clusters
            for target_cluster in adaptive_clustering_map[mat_phase].keys():
                # Get list of target's child clusters
                child_clusters = adaptive_clustering_map[mat_phase][target_cluster]
                # Loop over cluster-keyd dictionaries
                for cluster_dict in cluster_dicts:
                    # Loop over child clusters and build their items
                    for child_cluster in child_clusters:
                        cluster_dict[str(child_cluster)] = \
                            copy.deepcopy(cluster_dict[target_cluster])
                    # Remove target cluster item
                    cluster_dict.pop(target_cluster)
    # --------------------------------------------------------------------------------------
    def constitutive_source_conversion(self):
        '''Convert external sources constitutive models to CRATE corresponding model.'''
        # Initialize available conversions
        available_conversions = {}
        # Set available Links-CRATE conversions
        links_conversions = {'links_elastic': 'elastic', 'links_von_mises': 'von_mises'}
        available_conversions['links'] = links_conversions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase constitutive model
            constitutive_model = self._material_phases_models[str(mat_phase)]
            # Get material constitutive model source
            model_source = constitutive_model.get_source()
            # Skip to next material phase if not external constitutive model
            if model_source != 'crate':
                continue
            # Get material constitutive material properties
            material_properties = constitutive_model.get_material_properties()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get CRATE corresponding constitutive model name
            new_model_name = available_conversions[str(model_source)]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get CRATE corresponding constitutive model
            if model_source == 'links':
                if new_model_name == 'elastic':
                    # Get CRATE constitutive model
                    new_model = Elastic
                elif new_model_name == 'von_mises':
                    # Get CRATE constitutive model
                    new_model = VonMises
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get CRATE constitutive model required material properties
                new_required_properties = new_model.get_required_properties()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get CRATE constitutive model material properties
                new_material_properties = {}
                for property in new_required_properties:
                    # Get CRATE constitutive model material property
                    if property not in material_properties.keys():
                        raise RuntimeError('Incompatible material properties to ' +
                                           'convert constitutive model.')
                    else:
                        new_material_properties[str(property)] = \
                            copy.deepcopy(material_properties[str(property)])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Initialize CRATE constitutive model
                    new_constitutive_model = new_model(self._strain_formulation,
                                                       self._problem_type,
                                                       new_material_properties)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update material phases constitutive models
                self._material_phases_models[str(mat_phase)] = \
                    new_constitutive_model
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            else:
                raise RuntimeError('Unknown material constitutive model source.')
#
#                                                     Available material constitutive models
# ==========================================================================================
def get_available_material_models(model_source='crate'):
    '''Get available material constitutive models.

    Parameters
    ----------
    model_source : str, {'crate', 'links'}, default='crate'
        Material constitutive model source.

    Returns
    -------
    available_mat_models : list
        Available material constitutive models (list of str).
    '''
    # Set the available material constitutive models from a given source
    if model_source == 'crate':
        # CRATE material constitutive models
        available_mat_models = ['elastic', 'von_mises']
    elif model_source == 'links':
        # Links material constitutive models
        available_mat_models = ['ELASTIC', 'VON_MISES']
    else:
        raise RuntimeError('Unknown material constitutive model source.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return available_mat_models
