#
# Linear Elastic Constitutive Model (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the infinitesimal strain isotropic linear elastic constitutive
# model.
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
from material.materialmodeling import ConstitutiveModel
#
#                                                                 Elastic constitutive model
# ==========================================================================================
class Elastic(ConstitutiveModel):
    '''Elastic constitutive model.

    Attributes
    ----------
    _strain_type : str, {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite strain
        formulation through kinematic extension (infinitesimal constitutive formulation and
        purely finite strain kinematic extension - 'finite-kinext').
    _ndim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    '''
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
        self._problem_type = problem_type
        self._material_properties = material_properties
        self._strain_type = 'finite-kinext'
        # Get problem type parameters
        n_dim, comp_order_sym, _ = mop.getproblemtypeparam(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
    # --------------------------------------------------------------------------------------
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
        return self._strain_type
    # --------------------------------------------------------------------------------------
    def state_init(self):
        '''Initialize constitutive model material state variables.

        Constitutive model state variables:
            e_strain_mf | Elastic strain tensor (matricial form)
            strain_mf   | Total strain tensor (matricial form)
            stress_mf   | Cauchy stress tensor (matricial form)
            is_su_fail  | State update failure flag

        Returns
        -------
        state_variables_init : dict
            Initial constitutive model material state variables.
        '''
        # Initialize constitutive model state variables
        state_variables_init = dict()
        state_variables_init['e_strain_mf'] = \
            mop.gettensormf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                            self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            mop.gettensormf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                            self._comp_order_sym)
        state_variables_init['stress_mf'] = \
            mop.gettensormf(np.zeros((self._n_dim, self._n_dim)), self._n_dim,
                            self._comp_order_sym)
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
            Constitutive model material state variables.
        consistent_tangent_mf : ndarray
            Constitutive model consistent tangent modulus in matricial form.
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
        _, _, _, fosym, fodiagtrace, _, _ = top.getidoperators(self._n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute consistent tangent modulus according to problem type
        if self._problem_type in [1, 4]:
            consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
        # Build consistent tangent modulus matricial form
        consistent_tangent_mf = mop.gettensormf(consistent_tangent, self._n_dim,
                                                self._comp_order_sym)
        #
        #                                                                       State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental strain matricial form
        inc_strain_mf = mop.gettensormf(inc_strain, self._n_dim, self._comp_order_sym)
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
        state_variables = type(self).state_init()
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
        n_dim, comp_order_sym, _ = mop.getproblemtypeparam(problem_type)
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
        _, _, _, fosym, fodiagtrace, _, _ = top.getidoperators(n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute infinitesimal strains elasticity tensor according to problem type
        if problem_type in [1, 4]:
            # 2D problem (plane strain) / 3D problem
            elastic_tangent = lam*fodiagtrace + 2.0*miu*fosym
        # Build infinitesimal strains elasticity tensor matricial form
        elastic_tangent_mf = mop.gettensormf(elastic_tangent, n_dim, comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_tangent_mf
#
#                                                               Required material properties
#                                                                    (check input data file)
# ==========================================================================================
# Set the constitutive model required material properties
#
# Material properties meaning:
#
# E - Young modulus
# v - Poisson ratio
#
def getrequiredproperties():
    # Set required material properties
    req_material_properties = ['E', 'v']
    # Return
    return req_material_properties
#                                                                             Initialization
# ==========================================================================================
# Define material constitutive model state variables and build an initialized state
# variables dictionary
#
# List of constitutive model state variables:
#
#   e_strain_mf | Elastic strain tensor (matricial form)
#   strain_mf   | Total strain tensor (matricial form)
#   stress_mf   | Cauchy stress tensor (matricial form)
#   is_su_fail  | State update failure flag
#   acc_e_energy_dens | Accumulated elastic strain energy density  (post-process)
#
def init(problem_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    problem_type = problem_dict['problem_type']
    # Define constitutive model state variables (names and initialization)
    state_variables_init = dict()
    state_variables_init['e_strain_mf'] = mop.gettensormf(np.zeros((n_dim, n_dim)), n_dim,
                                                          comp_order)
    state_variables_init['strain_mf'] = mop.gettensormf(np.zeros((n_dim, n_dim)), n_dim,
                                                        comp_order)
    state_variables_init['stress_mf'] = mop.gettensormf(np.zeros((n_dim, n_dim)), n_dim,
                                                        comp_order)
    state_variables_init['is_plast'] = False
    state_variables_init['is_su_fail'] = False
    state_variables_init['acc_e_energy_dens'] = 0.0
    # Set additional out-of-plane strain and stress components
    if problem_type == 1:
        state_variables_init['e_strain_33'] = 0.0
        state_variables_init['stress_33'] = 0.0
    # Return initialized state variables dictionary
    return state_variables_init
#
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def suct(problem_dict, algpar_dict, material_properties, mat_phase, inc_strain,
         state_variables_old):
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material properties
    E = material_properties[mat_phase]['E']
    v = material_properties[mat_phase]['v']
    # Get last increment converged state variables
    e_strain_old_mf = state_variables_old['e_strain_mf']
    acc_e_energy_dens_old = state_variables_old['acc_e_energy_dens']
    # Set state update fail flag
    is_su_fail = False
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Lamé parameters
    lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
    miu = E/(2.0*(1.0 + v))
    # Set required fourth-order tensors
    _, _, _, fosym, fodiagtrace, _, _ = top.getidoperators(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type in [1, 4]:
        # 2D problem (plane strain) / 3D problem
        consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent, n_dim, comp_order)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build incremental strain matricial form
    inc_strain_mf = mop.gettensormf(inc_strain, n_dim, comp_order)
    # Update elastic strain
    e_strain_mf = e_strain_old_mf + inc_strain_mf
    # Update stress
    stress_mf = np.matmul(consistent_tangent_mf, e_strain_mf)
    # Compute out-of-plane stress component in a 2D plane strain problem (output purpose
    # only)
    if problem_type == 1:
        stress_33 = lam*(e_strain_mf[comp_order.index('11')] + \
                         e_strain_mf[comp_order.index('22')])
    # Initialize state variables dictionary
    state_variables = init(problem_dict)
    # Store updated state variables in matricial form
    state_variables['e_strain_mf'] = e_strain_mf
    state_variables['strain_mf'] = e_strain_mf
    state_variables['stress_mf'] = stress_mf
    state_variables['is_su_fail'] = is_su_fail
    if problem_type == 1:
        state_variables['stress_33'] = stress_33
    #
    #                                                      Accumulated strain energy density
    #                                                                         (post-process)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute incremental stress
    delta_stress_mf = np.matmul(consistent_tangent_mf, e_strain_mf - e_strain_old_mf)
    # Compute incremental elastic strain
    delta_e_strain_mf = e_strain_mf - e_strain_old_mf
    # Compute accumulated elastic strain energy density
    acc_e_energy_dens = acc_e_energy_dens_old + np.matmul(delta_stress_mf,
                                                          delta_e_strain_mf)
    # Store accumulated elastic strain energy density
    state_variables['acc_e_energy_dens'] = acc_e_energy_dens
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables, consistent_tangent_mf]
#
#                                                                 Consistent tangent modulus
# ==========================================================================================
# Compute the consistent tangent modulus
def ct(problem_dict, properties):
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
    miu = E/(2.0*(1.0 + v))
    # Set required fourth-order tensors
    _, _, _, fosym, fodiagtrace, _, _ = top.getidoperators(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type in [1, 4]:
        # 2D problem (plane strain) / 3D problem
        consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent, n_dim, comp_order)
    # Return
    return consistent_tangent_mf
