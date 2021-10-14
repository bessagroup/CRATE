#
# Material Interface (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing the required procedures to perform the constitutive state update of the
# the material clusters. Includes proper interfaces to implement constitutive models from
# different sources.
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
# Defining abstract base classes
from abc import ABC, abstractmethod
# Generate efficient iterators
import itertools as it
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# Constitutive models
from material.models.linear_elastic import Elastic
from material.models.von_mises import VonMises
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
    '''
    def __init__(self, strain_formulation, problem_type, material_phases,
                 material_phases_properties, material_phases_f, phase_clusters):
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
        material_phases_f : dict
            Volume fraction (item, float) associated to each material phase (key, str).
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_phases = material_phases
        self._material_phases_properties = material_phases_properties
        self._material_phases_f = material_phases_f
        self._phase_clusters = phase_clusters
        self._material_phases_models = {mat_phase: None for mat_phase in material_phases}
        self._clusters_def_gradient_mf = None
        self._clusters_def_gradient_old_mf = None
        self._clusters_state = None
        self._clusters_state_old = None
        self._clusters_tangent_mf = None
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # --------------------------------------------------------------------------------------
    def init_constitutive_model(self, mat_phase, model_name, model_source='crate'):
        '''Initialize material phase constitutive model.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        model_name : str
            Material constitutive model name.
        model_source : str, {'crate', }, default='crate'
            Material constitutive model source.
        '''
        # Initialize material phase constitutive model
        if model_source == 'crate':
            if model_name == 'elastic':
                constitutive_model = Elastic(self._problem_type,
                                             self._material_phases_properties[mat_phase])
            elif model_name == 'von_mises':
                constitutive_model = VonMises(self._problem_type,
                                              self._material_phases_properties[mat_phase])
            else:
                raise RuntimeError('Unknown constitutive model from CRATE\'s source.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif model_source == 'links':
            raise RuntimeError('Links: Not implemented yet.')
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
        soid_mf = mop.gettensormf(np.eye(self._n_dim), self._n_dim, self._comp_order_nsym)
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
                inc_strain = mop.gettensormf(inc_strain_mf, self._n_dim, comp_order)
                # Get material cluster last converged state variables
                state_variables_old = self._clusters_state_old[str(cluster)]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get material cluster last converged deformation gradient tensor
                def_gradient_old_mf = self._clusters_def_gradient_old_mf[str(cluster)]
                def_gradient_old = \
                    mop.gettensormf(def_gradient_old_mf, self._n_dim, self._comp_order_nsym)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Perform state update through the suitable material interface
                if source == 'crate':
                    state_variables, consistent_tangent_mf = \
                        self._material_su_interface(constitutive_model, def_gradient_old,
                                                    copy.deepcopy(inc_strain),
                                                    copy.deepcopy(state_variables_old))
                else:
                    raise RuntimeError('Links: Not implemented yet.')
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
                self._clusters_def_gradient_mf[str(cluster)] = \
                    np.matmul(inc_strain_mf, def_gradient_old_mf)
                # Update cluster state variables and material consistent tangent modulus
                self._clusters_state[str(cluster)] = state_variables
                self._clusters_tangent_mf[str(cluster)] = consistent_tangent_mf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return su_fail_state
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
            e_log_strain_old = mop.gettensorfrommf(e_log_strain_old_mf, n_dim,
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
            kirchhoff_stress = mop.gettensorfrommf(kirchhoff_stress_mf, n_dim,
                                                   comp_order_sym)
            # Compute Cauchy stress tensor (matricial form)
            cauchy_stress = MaterialState.cauchy_from_kirchhoff(def_gradient,
                                                                kirchhoff_stress)
            cauchy_stress_mf = mop.gettensormf(cauchy_stress, n_dim, comp_order_sym)
            # Update stress tensor
            state_variables['stress_mf'] = cauchy_stress_mf
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get last converged elastic logarithmic strain tensor
            e_log_strain_old_mf = state_variables_old['e_strain_mf']
            e_log_strain_old = mop.gettensorfrommf(e_log_strain_old_mf, n_dim,
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
            consistent_tangent_mf = mop.gettensormf(material_consistent_tangent, n_dim,
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
    def clustering_adaptivity_update(self, adaptive_clustering_map):
        '''Update cluster-related dictionaries according to clustering adaptivity step.

        Parameters
        ----------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str).
        '''
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
#
#                                                               Constitutive model interface
# ==========================================================================================
class ConstitutiveModel(ABC):
    '''Constitutive model interface.

    Attributes
    ----------
    _strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite strain
        formulation through kinematic extension (infinitesimal constitutive formulation and
        purely finite strain kinematic extension - 'finite-kinext').
    _source : str, {'crate', }
        Material constitutive model source.
    '''
    @abstractmethod
    def __init__(self, problem_type, material_properties):
        '''Constitutive model constructor.

        Parameters
        ----------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        material_properties : dict
            Constitutive model material properties (key, str) values (item, int/float/bool).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    @staticmethod
    def get_required_properties():
        '''Get the material constitutive model required properties.

        Returns
        -------
        req_mat_properties : list
            List of constitutive model required material properties (str).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_strain_type(self):
        '''Get material constitutive model strain formulation.

        Returns
        -------
        strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
            Constitutive model strain formulation: infinitesimal strain formulation
            ('infinitesimal'), finite strain formulation ('finite') or finite strain
            formulation through kinematic extension (infinitesimal constitutive formulation
            and purely finite strain kinematic extension - 'finite-kinext').
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_source(self):
        '''Get material constitutive model source.

        Returns
        -------
        source : str, {'crate', }
            Material constitutive model source.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def state_init(self):
        '''Initialize material constitutive model state variables.

        Returns
        -------
        state_variables_init : dict
            Initial material constitutive model state variables.
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def state_update(self, inc_strain, state_variables_old, su_max_n_iterations=20,
                     su_conv_tol=1e-6):
        '''Perform material constitutive model state update.

        Parameters
        ----------
        inc_strain : 2darray
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged material constitutive model state variables.
        su_max_n_iterations : int, default=20
            State update maximum number of iterations.
        su_conv_tol : float, default=1e-6
            State update convergence tolerance.

        Returns
        -------
        state_variables : dict
            Material constitutive model state variables.
        consistent_tangent_mf : ndarray
            Material constitutive model material consistent tangent modulus in matricial
            form.
        '''
        pass
#
#                                                     Available material constitutive models
# ==========================================================================================
def get_available_material_models(model_source='crate'):
    '''Get available material constitutive models.

    Parameters
    ----------
    model_source : str, {'crate', }, default='crate'
        Material constitutive model source.

    Returns
    -------
    available_mat_models : list
        Available material constitutive models (list of str).
    '''
    # Set the available material constitutive models from a given source
    if model_source == 'crate':
        # CRATE material constitutive models
        available_mat_models = ['linear_elastic', 'von_mises']
    elif model_source == 'links':
        # Links material constitutive models
        available_mat_models = ['ELASTIC', 'VON_MISES']
    else:
        raise RuntimeError('Unknown material constitutive model source.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return available_mat_models
