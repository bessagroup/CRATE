#
# Linear Elastic Constitutive Model (CRATE Program)
# ==========================================================================================
# Summary:
# Infinitesimal strains isotropic linear elastic constitutive model.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2021 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Refactoring and OOP implementation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# Constitutive models
from material.models.interface import ConstitutiveModel
#
#                                                                 Elastic constitutive model
# ==========================================================================================
class Elastic(ConstitutiveModel):
    '''Elastic constitutive model.

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite strain
        formulation through kinematic extension (infinitesimal constitutive formulation and
        purely finite strain kinematic extension - 'finite-kinext').
    _source : str, {'crate', }
        Material constitutive model source.
    _ndim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    '''
    def __init__(self, strain_formulation, problem_type, material_properties):
        '''Constitutive model constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        material_properties : dict
            Constitutive model material properties (key, str) values (item, int/float/bool).
        '''
        self._name = 'elastic'
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_properties = material_properties
        # Set strain formulation
        self._strain_type = 'finite-kinext'
        # Set source
        self._source = 'crate'
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_required_properties():
        '''Get the constitutive model required material properties.

        Material properties:
        E - Young modulus
        v - Poisson's ratio

        Returns
        -------
        req_mat_properties : list
            List of constitutive model required material properties (str).
        '''
        # Set required material properties
        req_material_properties = ['E', 'v']
        # Return
        return req_material_properties
    # --------------------------------------------------------------------------------------
    def state_init(self):
        '''Get initialized material constitutive model state variables.

        Constitutive model state variables:
            e_strain_mf  | Infinitesimal strains: Elastic infinitesimal strain tensor
                         |                        (matricial form)
                         | Finite strains: Elastic spatial logarithmic strain tensor
                         |                 (matricial form)
            strain_mf    | Infinitesimal strains: Infinitesimal strain tensor
                         |                        (matricial form)
                         | Finite strains: Spatial logarithmic strain tensor
                         |                 (matricial form)
            stress_mf    | Infinitesimal strains: Cauchy stress tensor (matricial form)
                         | Finite strains: Kirchhoff stress tensor (matricial form) within
                         |                 state_update(), First-Piola Kirchhoff stress
                         |                 tensor (matricial form) otherwise.
            is_su_fail   | State update failure flag

        Returns
        -------
        state_variables_init : dict
            Initialized constitutive model material state variables.
        '''
        # Initialize constitutive model state variables
        state_variables_init = dict()
        state_variables_init['e_strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                              self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                              self._comp_order_sym)
        if self._strain_formulation == 'infinitesimal':
            # Cauchy stress tensor
            state_variables_init['stress_mf'] = \
                mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                                  self._comp_order_sym)
        else:
            # First Piola-Kirchhoff stress tensor
            state_variables_init['stress_mf'] = \
                mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                                  self._comp_order_nsym)
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
        # ----------------------------------------------------------------------------------
        # Return
        return state_variables_init
    # --------------------------------------------------------------------------------------
    def state_update(self, inc_strain, state_variables_old, su_max_n_iterations=20,
                     su_conv_tol=1e-6):
        '''Perform constitutive model state update.

        Parameters
        ----------
        inc_strain : 2darray
            Incremental strain second-order tensor.
        state_variables_old : dict
            Last converged constitutive model material state variables.
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
        # Get material properties
        E = self._material_properties['E']
        v = self._material_properties['v']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get last increment converged state variables
        e_strain_old_mf = state_variables_old['e_strain_mf']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update fail flag
        is_su_fail = False
        #
        #                                                         Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        _, _, _, fosym, fodiagtrace, _, _ = top.get_id_operators(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute consistent tangent modulus according to problem type
        if self._problem_type in [1, 4]:
            consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
        # Build consistent tangent modulus matricial form
        consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent, self._n_dim,
                                                  self._comp_order_sym)
        #
        #                                                                       State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental strain matricial form
        inc_strain_mf = mop.get_tensor_mf(inc_strain, self._n_dim, self._comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic strain
        e_strain_mf = e_strain_old_mf + inc_strain_mf
        # Update stress
        stress_mf = np.matmul(consistent_tangent_mf, e_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute out-of-plane stress component in a 2D plane strain problem (output purpose
        # only)
        if self._problem_type == 1:
            stress_33 = lam*(e_strain_mf[self._comp_order_sym.index('11')] + \
                             e_strain_mf[self._comp_order_sym.index('22')])
        #
        #                                                             Update state variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state variables dictionary
        state_variables = self.state_init()
        # Store updated state variables in matricial form
        state_variables['e_strain_mf'] = e_strain_mf
        state_variables['strain_mf'] = e_strain_mf
        state_variables['stress_mf'] = stress_mf
        state_variables['is_su_fail'] = is_su_fail
        if self._problem_type == 1:
            state_variables['stress_33'] = stress_33
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [state_variables, consistent_tangent_mf]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def elastic_tangent_modulus(problem_type, elastic_properties):
        '''Compute infinitesimal strains elasticity tensor.

        Parameters
        ----------
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float). Expecting Young
            modulus ('E') and Poisson's ratio ('v').

        Returns
        -------
        elastic_tangent_mf : ndarray
            Infinitesimal strains elasticity tensor in matricial form.
        '''
        # Get problem type parameters
        n_dim, comp_order_sym, _ = mop.get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get Young's Modulus and Poisson's ratio
        E = elastic_properties['E']
        v = elastic_properties['v']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Lamé parameters
        lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        miu = E/(2.0*(1.0 + v))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required fourth-order tensors
        _, _, _, fosym, fodiagtrace, _, _ = top.get_id_operators(n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute infinitesimal strains elasticity tensor according to problem type
        if problem_type in [1, 4]:
            # 2D problem (plane strain) / 3D problem
            elastic_tangent = lam*fodiagtrace + 2.0*miu*fosym
        # Build infinitesimal strains elasticity tensor matricial form
        elastic_tangent_mf = mop.get_tensor_mf(elastic_tangent, n_dim, comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_tangent_mf
