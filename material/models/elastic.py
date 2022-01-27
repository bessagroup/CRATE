#
# Linear Elastic Constitutive Model (CRATE Program)
# ==========================================================================================
# Summary:
# Infinitesimal strains isotropic linear elastic constitutive model.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2021 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Refactoring and OOP implementation.
# Bernardo P. Ferreira | Jan 2022 | Add method to compute general anisotropic elasticity
#                                 | tensor.
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
        E | Young modulus
        v | Poisson's ratio

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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            Material constitutive model consistent tangent modulus in matricial form.
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
        # Set state update failure flag
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
        # Store updated state variables
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
    def get_available_elastic_symmetries():
        '''Get available elastic symmetries under general anisotropic elasticity.

        Returns
        -------
        elastic_symmetries : dict
            Elastic modulii (tuple of str, item) required for each available elastic
            symmetry (str, key).
        '''
        # Set available elastic symmetries and required elastic modulii
        elastic_symmetries = {
            'isotropic':            ('E1111', 'E1122'),
            'transverse_isotropic': ('E1111', 'E3333', 'E1122', 'E1133', 'E2323'),
            'orthotropic':          ('E1111', 'E2222', 'E3333', 'E1122', 'E1133', 'E2233',
                                     'E1212', 'E2323', 'E1313'),
            'monoclinic':           ('E1111', 'E2222', 'E3333', 'E1122', 'E1133', 'E2233',
                                     'E1112', 'E2212', 'E3312', 'E1212', 'E2323', 'E1313',
                                     'E2313'),
            'triclinic':            ('E1111', 'E1122', 'E1133', 'E1112', 'E1123', 'E1113',
                                     'E2222', 'E2233', 'E2212', 'E2223', 'E2213', 'E3333',
                                     'E3312', 'E3323', 'E3313', 'E1212', 'E1223', 'E1213',
                                     'E2323', 'E2313', 'E1313')}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_symmetries
    # --------------------------------------------------------------------------------------
    @staticmethod
    def elastic_tangent_modulus(elastic_properties, elastic_symmetry='isotropic'):
        '''Compute 3D elasticity tensor under general anisotropic elasticity.

        Parameters
        ----------
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float). Expecting
            independent elastic modulii ('Eijkl') according to elastic symmetries.
            Young modulus ('E') and Poisson's ratio ('v') may alternatively provided under
            elastic isotropy.
        elastic_symmetry : str, {'isotropic', 'transverse_isotropic', 'orthotropic',
                                 'monoclinic', 'triclinic'}, default='isotropic'
            Elastic symmetries: 'triclinic' assumes no elastic symmetries, 'monoclinic'
            assumes plane of symmetry 12, 'orthotropic' assumes planes of symmetry 12 and
            13, 'transverse_isotropic' assumes axis of symmetry 3, 'isotropic' assumes
            complete symmetry.

        Returns
        -------
        elastic_tangent_mf : 2darray
            3D elasticity tensor in matricial form.
        '''
        # Get available elastic symmetries and required elastic modulii
        elastic_symmetries = Elastic.get_available_elastic_symmetries()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check elastic symmetry and required elastic modulii
        if elastic_symmetry not in elastic_symmetries.keys():
            raise RuntimeError('Unavailable elastic symmetry.')
        else:
            # Get required elastic modulii
            required_modulii = elastic_symmetries[elastic_symmetry]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Conversion from technical constants to elastic modulii
            if elastic_symmetry == 'isotropic' and \
                    {'E', 'v'}.issubset(set(elastic_properties.keys())):
                # Get Young modulus and Poisson coefficient
                E = elastic_properties['E']
                v = elastic_properties['v']
                # Compute elastic modulii
                elastic_properties['E1111'] = (E*(1.0 - v))/((1.0 + v)*(1.0 - 2.0*v))
                elastic_properties['E1122'] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize independent elastic modulii
            ind_modulii = {str(modulus) : 0.0 for modulus in \
                           elastic_symmetries['triclinic']}
            # Loop over required elastic modulii
            for modulus in required_modulii:
                # Set symmetric modulus
                sym_modulus = modulus[3:5] + modulus[1:3]
                # Check if requires elastic modulus has been provided
                if modulus in elastic_properties.keys():
                    ind_modulii[modulus] = elastic_properties[modulus]
                elif sym_modulus in elastic_properties.keys():
                    ind_modulii[modulus] = elastic_properties[sym_modulus]
                else:
                    raise RuntimeError('Missing elastic modulii for ' + elastic_symmetry +
                                       ' material.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set all (non-symmetric) elastic modulii according to elastic symmetry
        all_modulii = {str(modulus) : 0.0 for modulus in ind_modulii.keys()}
        if elastic_symmetry == 'isotropic':
            all_modulii['E1111'] = ind_modulii['E1111']
            all_modulii['E2222'] = all_modulii['E1111']
            all_modulii['E3333'] = all_modulii['E1111']
            all_modulii['E1122'] = ind_modulii['E1122']
            all_modulii['E1133'] = all_modulii['E1122']
            all_modulii['E2233'] = all_modulii['E1122']
            all_modulii['E1212'] = 0.5*(all_modulii['E1111'] - all_modulii['E1122'])
            all_modulii['E2323'] = all_modulii['E1212']
            all_modulii['E1313'] = all_modulii['E1212']
        elif elastic_symmetry == 'transverse_isotropic':
            all_modulii['E1111'] = ind_modulii['E1111']
            all_modulii['E2222'] = all_modulii['E1111']
            all_modulii['E3333'] = ind_modulii['E3333']
            all_modulii['E1122'] = ind_modulii['E1122']
            all_modulii['E1133'] = ind_modulii['E1133']
            all_modulii['E2233'] = all_modulii['E1133']
            all_modulii['E1212'] = 0.5*(all_modulii['E1111'] - all_modulii['E1122'])
            all_modulii['E2323'] = ind_modulii['E2323']
            all_modulii['E1313'] = all_modulii['E2323']
        elif elastic_symmetry == 'orthotropic':
            all_modulii['E1111'] = ind_modulii['E1111']
            all_modulii['E2222'] = ind_modulii['E2222']
            all_modulii['E3333'] = ind_modulii['E3333']
            all_modulii['E1122'] = ind_modulii['E1122']
            all_modulii['E1133'] = ind_modulii['E1133']
            all_modulii['E2233'] = ind_modulii['E2233']
            all_modulii['E1212'] = ind_modulii['E1212']
            all_modulii['E2323'] = ind_modulii['E2323']
            all_modulii['E1313'] = ind_modulii['E1313']
        elif elastic_symmetry == 'monoclinic':
            all_modulii['E1111'] = ind_modulii['E1111']
            all_modulii['E2222'] = ind_modulii['E2222']
            all_modulii['E3333'] = ind_modulii['E3333']
            all_modulii['E1122'] = ind_modulii['E1122']
            all_modulii['E1133'] = ind_modulii['E1133']
            all_modulii['E2233'] = ind_modulii['E2233']
            all_modulii['E1212'] = ind_modulii['E1212']
            all_modulii['E2323'] = ind_modulii['E2323']
            all_modulii['E1313'] = ind_modulii['E1313']
            all_modulii['E1112'] = ind_modulii['E1112']
            all_modulii['E2212'] = ind_modulii['E2212']
            all_modulii['E3312'] = ind_modulii['E3312']
            all_modulii['E2313'] = ind_modulii['E2313']
        elif elastic_symmetry == 'triclinic':
            all_modulii = ind_modulii
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem parameters
        _, comp_order_sym, _ = mop.get_problem_type_parameters(problem_type=4)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elasticity tensor
        elastic_tangent_mf = np.zeros(2*(len(comp_order_sym),))
        # Build elasticity tensor according with elastic symmetry
        for modulus in all_modulii.keys():
            # Get elastic modulus second-order indexes and associated kelvin factors
            idx_1 = comp_order_sym.index(modulus[1:3])
            kf_1 = mop.kelvin_factor(idx_1, comp_order_sym)
            idx_2 = comp_order_sym.index(modulus[3:5])
            kf_2 = mop.kelvin_factor(idx_2, comp_order_sym)
            # Assemble elastic modulus in elasticity tensor matricial form
            elastic_tangent_mf[idx_1, idx_2] = kf_1*kf_2*all_modulii[modulus]
            # Set symmetric component of elasticity tensor
            if idx_1 != idx_2:
                elastic_tangent_mf[idx_2, idx_1] = elastic_tangent_mf[idx_1, idx_2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_tangent_mf
