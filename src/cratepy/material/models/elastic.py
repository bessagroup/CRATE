"""Linear elastic constitutive model.

This module includes the implementation of the linear elastic constitutive
model under general anisotropic elasticity. The corresponding class also
includes several methods to handle elastic anisotropy, namely to process the
specification of the elastic moduli and the computation of the elasticity
tensor.

Classes
-------
Elastic
    Elastic constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
from material.models.interface import ConstitutiveModel
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class Elastic(ConstitutiveModel):
    """Linear elastic constitutive model.

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite
        strain formulation through kinematic extension (infinitesimal
        constitutive formulation and purely finite strain kinematic extension -
        'finite-kinext').
    _source : {'crate',}
        Material constitutive model source.
    _ndim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.

    Methods
    -------
    get_required_properties()
        Get constitutive model material properties and constitutive options.
    state_init(self)
        Get initialized material constitutive model state variables.
    state_update(self, inc_strain, state_variables_old, \
                 su_max_n_iterations=20, su_conv_tol=1e-6)
        Perform constitutive model state update.
    get_available_elastic_symmetries()
        Get available elastic symmetries under general anisotropy.
    elastic_tangent_modulus(elastic_properties, elastic_symmetry='isotropic')
        Compute 3D elasticity tensor under general anisotropic elasticity.
    get_technical_from_elastic_moduli(elastic_symmetry, elastic_properties)
        Get technical constants of elasticity from elastic moduli.
    """
    def __init__(self, strain_formulation, problem_type, material_properties):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        material_properties : dict
            Constitutive model material properties (key, str) values
            (item, {int, float, bool}).
        """
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
        # Get elastic symmetry
        elastic_symmetry = material_properties['elastic_symmetry']
        # Check finite strains formulation
        if self._strain_formulation == 'finite' and \
                elastic_symmetry != 'isotropic':
            raise RuntimeError('The elastic constitutive model is only '
                               'available under finite strains for the '
                               'elastic isotropic case.')
        # Compute technical constants of elasticity
        if elastic_symmetry == 'isotropic':
            # Compute technical constants of elasticity
            technical_constants = Elastic.get_technical_from_elastic_moduli(
                elastic_symmetry, material_properties)
            # Assemble technical constants of elasticity
            self._material_properties.update(technical_constants)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_required_properties():
        """Get constitutive model material properties and constitutive options.

        *Input data file syntax*:

        .. code-block:: text

           elastic_symmetry < option > < number_of_elastic_moduli >
               euler_angles < value > < value > < value >
               Eijkl < value >
               Eijkl < value >
               ...

        where

        - ``elastic_symmetry`` - Elastic symmetry and number of elastic
          moduli.
        - ``euler_angles`` - Euler angles (degrees) sorted according with Bunge
          convention. Not required if ``elastic_symmetry`` is set as
          `isotropic`.
        - ``Eijkl`` - Elastic moduli. Young's modulus (``E``) and Poisson's
          coefficient (``v``) may be alternatively provided if
          ``elastic_symmetry`` is set as `isotropic`.

        ----

        Returns
        -------
        material_properties : list[str]
            Constitutive model material properties names (str).
        constitutive_options : dict
            Constitutive options (key, str) and available specifications
            (item, tuple[str]).
        """
        # Get available elastic symmetries and required elastic moduli
        elastic_symmetries = Elastic.get_available_elastic_symmetries()
        # Set constitutive options and available specifications
        constitutive_options = {'elastic_symmetry':
                                tuple(elastic_symmetries.keys())}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material properties names
        material_properties = ()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return material_properties, constitutive_options
    # -------------------------------------------------------------------------
    def state_init(self):
        """Get initialized material constitutive model state variables.

        Constitutive model state variables:

        * ``e_strain_mf``

            * *Infinitesimal strains*: Elastic infinitesimal strain tensor
              (matricial form).

            * *Finite strains*: Elastic spatial logarithmic strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon^{e}}` /
              :math:`\\boldsymbol{\\varepsilon^{e}}`

        * ``strain_mf``

            * *Infinitesimal strains*: Infinitesimal strain tensor
              (matricial form).

            * *Finite strains*: Spatial logarithmic strain tensor
              (matricial form).

            * *Symbol*: :math:`\\boldsymbol{\\varepsilon}` /
              :math:`\\boldsymbol{\\varepsilon}`

        * ``stress_mf``

            * *Infinitesimal strains*: Cauchy stress tensor (matricial form).

            * *Finite strains*: Kirchhoff stress tensor (matricial form) within
              :py:meth:`state_update`, first Piola-Kirchhoff stress tensor
              (matricial form) otherwise.

            * *Symbol*: :math:`\\boldsymbol{\\sigma}` /
              (:math:`\\boldsymbol{\\tau}`, :math:`\\boldsymbol{P}`)

        * ``is_su_fail``

            * State update failure flag.

        ----

        Returns
        -------
        state_variables_init : dict
            Initialized constitutive model material state variables.
        """
        # Initialize constitutive model state variables
        state_variables_init = dict()
        state_variables_init['e_strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)),
                              self._n_dim, self._comp_order_sym)
        state_variables_init['strain_mf'] = \
            mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)),
                              self._n_dim, self._comp_order_sym)
        if self._strain_formulation == 'infinitesimal':
            # Cauchy stress tensor
            state_variables_init['stress_mf'] = \
                mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)),
                                  self._n_dim, self._comp_order_sym)
        else:
            # First Piola-Kirchhoff stress tensor
            state_variables_init['stress_mf'] = \
                mop.get_tensor_mf(np.zeros((self._n_dim, self._n_dim)),
                                  self._n_dim, self._comp_order_nsym)
        state_variables_init['is_plast'] = False
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 0.0
            state_variables_init['stress_33'] = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables_init
    # -------------------------------------------------------------------------
    def state_update(self, inc_strain, state_variables_old,
                     su_max_n_iterations=20, su_conv_tol=1e-6):
        """Perform constitutive model state update.

        Parameters
        ----------
        inc_strain : numpy.ndarray (2d)
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
        consistent_tangent_mf : numpy.ndarray (2d)
            Material constitutive model consistent tangent modulus in
            matricial form.
        """
        # Get last increment converged state variables
        e_strain_old_mf = state_variables_old['e_strain_mf']
        if self._problem_type == 1:
            e_strain_33_old = state_variables_old['e_strain_33']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set state update failure flag
        is_su_fail = False
        #
        #                             State update & Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental strain tensor matricial form
        inc_strain_mf = mop.get_tensor_mf(inc_strain, self._n_dim,
                                          self._comp_order_sym)
        #
        #                                                    2D > 3D conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, perform the state
        # update and consistent tangent computation as in the 3D case,
        # considering the appropriate out-of-plain strain and stress components
        if self._problem_type == 4:
            n_dim = self._n_dim
            comp_order_sym = self._comp_order_sym
            comp_order_nsym = self._comp_order_nsym
        else:
            # Set 3D problem parameters
            n_dim, comp_order_sym, comp_order_nsym = \
                mop.get_problem_type_parameters(4)
            # Build strain tensors (matricial form) by including the
            # appropriate out-of-plain components
            inc_strain_mf = mop.get_state_3Dmf_from_2Dmf(self._problem_type,
                                                         inc_strain_mf,
                                                         comp_33=0.0)
            e_strain_old_mf = mop.get_state_3Dmf_from_2Dmf(self._problem_type,
                                                           e_strain_old_mf,
                                                           e_strain_33_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update elastic strain
        e_strain_mf = e_strain_old_mf + inc_strain_mf
        # Compute 3D elasticity tensor (matricial form)
        consistent_tangent_mf = Elastic.elastic_tangent_modulus(
            self._material_properties,
            elastic_symmetry=self._material_properties['elastic_symmetry'])
        # Update stress
        stress_mf = np.matmul(consistent_tangent_mf, e_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get the out-of-plane strain and stress components
        if self._problem_type == 1:
            e_strain_33 = e_strain_mf[comp_order_sym.index('33')]
            stress_33 = stress_mf[comp_order_sym.index('33')]
        #
        #                                                    3D > 2D Conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the
        # 2D strain and stress tensors (matricial form) once the state update
        # has been performed
        if self._problem_type == 1:
            # Builds 2D strain and stress tensors (matricial form) from the
            # associated 3D counterparts
            e_strain_mf = mop.get_state_2Dmf_from_3Dmf(self._problem_type,
                                                       e_strain_mf)
            stress_mf = mop.get_state_2Dmf_from_3Dmf(self._problem_type,
                                                     stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the
        # 2D consistent tangent modulus (matricial form) from the
        # 3D counterpart
        if self._problem_type == 1:
            consistent_tangent_mf = mop.get_state_2Dmf_from_3Dmf(
                self._problem_type, consistent_tangent_mf)
        #
        #                                                Update state variables
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize state variables dictionary
        state_variables = self.state_init()
        # Store updated state variables
        state_variables['e_strain_mf'] = e_strain_mf
        state_variables['strain_mf'] = e_strain_mf
        state_variables['stress_mf'] = stress_mf
        state_variables['is_su_fail'] = is_su_fail
        if self._problem_type == 1:
            state_variables['e_strain_33'] = e_strain_33
            state_variables['stress_33'] = stress_33
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables, consistent_tangent_mf
    # -------------------------------------------------------------------------
    @staticmethod
    def get_available_elastic_symmetries():
        """Get available elastic symmetries under general anisotropy.

        Available elastic symmetries:

        * Isotropic:

            *Input data file syntax*:

            .. code-block:: text

               elastic_symmetry isotropic 2
                   E1111 < value >
                   E1122 < value >

            or

            .. code-block:: text

               elastic_symmetry isotropic 2
                   E < value >
                   v < value >

            where

            - ``elastic_symmetry`` - Elastic symmetry and number of elastic
              moduli.
            - ``Eijkl`` - Elastic moduli. Young's modulus (``E``) and Poisson's
              coefficient (``v``) may be alternatively provided if
              ``elastic_symmetry`` is set as `isotropic`.

        ----

        * Transverse isotropic (axis of symmetry 3):

            *Input data file syntax*:

            .. code-block:: text

               elastic_symmetry transverse_isotropic 6
                   euler_angles < value > < value > < value >
                   Eijkl < value >
                   ...

            where

            - ``elastic_symmetry`` - Elastic symmetry and number of elastic
              moduli.
            - ``euler_angles`` - Euler angles (degrees) sorted according with
              Bunge convention.
            - ``Eijkl`` - Elastic moduli.

        ----

        * Orthotropic (planes of symmetry 12 and 13):

            *Input data file syntax*:

            .. code-block:: text

               elastic_symmetry orthotropic 10
                   euler_angles < value > < value > < value >
                   Eijkl < value >
                   ...

            where

            - ``elastic_symmetry`` - Elastic symmetry and number of elastic
              moduli.
            - ``euler_angles`` - Euler angles (degrees) sorted according with
              Bunge convention.
            - ``Eijkl`` - Elastic moduli.

        ----

        * Monoclinic (plane of symmetry 12):

            *Input data file syntax*:

            .. code-block:: text

               elastic_symmetry monoclinic 14
                   euler_angles < value > < value > < value >
                   Eijkl < value >
                   ...

            where

            - ``elastic_symmetry`` - Elastic symmetry and number of elastic
              moduli.
            - ``euler_angles`` - Euler angles (degrees) sorted according with
              Bunge convention.
            - ``Eijkl`` - Elastic moduli.

        ----

        * Triclinic:

            *Input data file syntax*:

            .. code-block:: text

               elastic_symmetry triclinic 22
                   euler_angles < value > < value > < value >
                   Eijkl < value >
                   ...

            where

            - ``elastic_symmetry`` - Elastic symmetry and number of elastic
              moduli.
            - ``euler_angles`` - Euler angles (degrees) sorted according with
              Bunge convention.
            - ``Eijkl`` - Elastic moduli.

        ----

        Returns
        -------
        elastic_symmetries : dict
            Elastic moduli (item, tuple[str]) required for each available
            elastic symmetry (key, str).
        """
        # Set available elastic symmetries and required elastic moduli
        elastic_symmetries = {
            'isotropic': ('E1111', 'E1122'),
            'transverse_isotropic': ('E1111', 'E3333', 'E1122', 'E1133',
                                     'E2323'),
            'orthotropic': ('E1111', 'E2222', 'E3333', 'E1122', 'E1133',
                            'E2233', 'E1212', 'E2323', 'E1313'),
            'monoclinic': ('E1111', 'E2222', 'E3333', 'E1122', 'E1133',
                           'E2233', 'E1112', 'E2212', 'E3312', 'E1212',
                           'E2323', 'E1313', 'E2313'),
            'triclinic': ('E1111', 'E1122', 'E1133', 'E1112', 'E1123', 'E1113',
                          'E2222', 'E2233', 'E2212', 'E2223', 'E2213', 'E3333',
                          'E3312', 'E3323', 'E3313', 'E1212', 'E1223', 'E1213',
                          'E2323', 'E2313', 'E1313')
            }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_symmetries
    # -------------------------------------------------------------------------
    @staticmethod
    def elastic_tangent_modulus(elastic_properties,
                                elastic_symmetry='isotropic'):
        """Compute 3D elasticity tensor under general anisotropic elasticity.

        Parameters
        ----------
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float).
            Expecting independent elastic moduli ('Eijkl') according to
            elastic symmetries. Young modulus ('E') and Poisson's ratio ('v')
            may alternatively provided under elastic isotropy.
        elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                            'orthotropic', 'monoclinic', 'triclinic'}, \
                            default='isotropic'
            Elastic symmetries:

            * 'triclinic':  assumes no elastic symmetries.

            * 'monoclinic': assumes plane of symmetry 12.

            * 'orthotropic': assumes planes of symmetry 12 and 13.

            * 'transverse_isotropic': assumes axis of symmetry 3

            * 'isotropic': assumes complete symmetry.

        Returns
        -------
        elastic_tangent_mf : numpy.ndarray (2d)
            3D elasticity tensor in matricial form.
        """
        # Get available elastic symmetries and required elastic moduli
        elastic_symmetries = Elastic.get_available_elastic_symmetries()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check elastic symmetry and required elastic moduli
        if elastic_symmetry not in elastic_symmetries.keys():
            raise RuntimeError('Unavailable elastic symmetry.')
        else:
            # Get required elastic moduli
            required_moduli = elastic_symmetries[elastic_symmetry]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Conversion from technical constants to elastic moduli
            if elastic_symmetry == 'isotropic' and \
                    {'E', 'v'}.issubset(set(elastic_properties.keys())):
                # Get Young modulus and Poisson coefficient
                E = elastic_properties['E']
                v = elastic_properties['v']
                # Compute elastic moduli
                elastic_properties['E1111'] = \
                    (E*(1.0 - v))/((1.0 + v)*(1.0 - 2.0*v))
                elastic_properties['E1122'] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize independent elastic moduli
            ind_moduli = {str(modulus): 0.0 for modulus in
                          elastic_symmetries['triclinic']}
            # Loop over required elastic moduli
            for modulus in required_moduli:
                # Set symmetric modulus
                sym_modulus = modulus[3:5] + modulus[1:3]
                # Check if requires elastic modulus has been provided
                if modulus in elastic_properties.keys():
                    ind_moduli[modulus] = elastic_properties[modulus]
                elif sym_modulus in elastic_properties.keys():
                    ind_moduli[modulus] = elastic_properties[sym_modulus]
                else:
                    raise RuntimeError('Missing elastic moduli for '
                                       + elastic_symmetry + ' material.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set all (non-symmetric) elastic moduli according to elastic symmetry
        all_moduli = {str(modulus): 0.0 for modulus in ind_moduli.keys()}
        if elastic_symmetry == 'isotropic':
            all_moduli['E1111'] = ind_moduli['E1111']
            all_moduli['E2222'] = all_moduli['E1111']
            all_moduli['E3333'] = all_moduli['E1111']
            all_moduli['E1122'] = ind_moduli['E1122']
            all_moduli['E1133'] = all_moduli['E1122']
            all_moduli['E2233'] = all_moduli['E1122']
            all_moduli['E1212'] = 0.5*(all_moduli['E1111']
                                       - all_moduli['E1122'])
            all_moduli['E2323'] = all_moduli['E1212']
            all_moduli['E1313'] = all_moduli['E1212']
        elif elastic_symmetry == 'transverse_isotropic':
            all_moduli['E1111'] = ind_moduli['E1111']
            all_moduli['E2222'] = all_moduli['E1111']
            all_moduli['E3333'] = ind_moduli['E3333']
            all_moduli['E1122'] = ind_moduli['E1122']
            all_moduli['E1133'] = ind_moduli['E1133']
            all_moduli['E2233'] = all_moduli['E1133']
            all_moduli['E1212'] = 0.5*(all_moduli['E1111']
                                       - all_moduli['E1122'])
            all_moduli['E2323'] = ind_moduli['E2323']
            all_moduli['E1313'] = all_moduli['E2323']
        elif elastic_symmetry == 'orthotropic':
            all_moduli['E1111'] = ind_moduli['E1111']
            all_moduli['E2222'] = ind_moduli['E2222']
            all_moduli['E3333'] = ind_moduli['E3333']
            all_moduli['E1122'] = ind_moduli['E1122']
            all_moduli['E1133'] = ind_moduli['E1133']
            all_moduli['E2233'] = ind_moduli['E2233']
            all_moduli['E1212'] = ind_moduli['E1212']
            all_moduli['E2323'] = ind_moduli['E2323']
            all_moduli['E1313'] = ind_moduli['E1313']
        elif elastic_symmetry == 'monoclinic':
            all_moduli['E1111'] = ind_moduli['E1111']
            all_moduli['E2222'] = ind_moduli['E2222']
            all_moduli['E3333'] = ind_moduli['E3333']
            all_moduli['E1122'] = ind_moduli['E1122']
            all_moduli['E1133'] = ind_moduli['E1133']
            all_moduli['E2233'] = ind_moduli['E2233']
            all_moduli['E1212'] = ind_moduli['E1212']
            all_moduli['E2323'] = ind_moduli['E2323']
            all_moduli['E1313'] = ind_moduli['E1313']
            all_moduli['E1112'] = ind_moduli['E1112']
            all_moduli['E2212'] = ind_moduli['E2212']
            all_moduli['E3312'] = ind_moduli['E3312']
            all_moduli['E2313'] = ind_moduli['E2313']
        elif elastic_symmetry == 'triclinic':
            all_moduli = ind_moduli
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem parameters
        _, comp_order_sym, _ = mop.get_problem_type_parameters(problem_type=4)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize elasticity tensor
        elastic_tangent_mf = np.zeros(2*(len(comp_order_sym),))
        # Build elasticity tensor according with elastic symmetry
        for modulus in all_moduli.keys():
            # Get elastic modulus second-order indexes and associated kelvin
            # factors
            idx_1 = comp_order_sym.index(modulus[1:3])
            kf_1 = mop.kelvin_factor(idx_1, comp_order_sym)
            idx_2 = comp_order_sym.index(modulus[3:5])
            kf_2 = mop.kelvin_factor(idx_2, comp_order_sym)
            # Assemble elastic modulus in elasticity tensor matricial form
            elastic_tangent_mf[idx_1, idx_2] = kf_1*kf_2*all_moduli[modulus]
            # Set symmetric component of elasticity tensor
            if idx_1 != idx_2:
                elastic_tangent_mf[idx_2, idx_1] = elastic_tangent_mf[idx_1,
                                                                      idx_2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return elastic_tangent_mf
    # -------------------------------------------------------------------------
    def get_technical_from_elastic_moduli(elastic_symmetry,
                                          elastic_properties):
        """Get technical constants of elasticity from elastic moduli.

        Parameters
        ----------
        elastic_symmetry : {'isotropic', 'transverse_isotropic', \
                            'orthotropic', 'monoclinic', 'triclinic'}, \
                            default='isotropic'
            Elastic symmetries:

            * 'triclinic':  assumes no elastic symmetries.

            * 'monoclinic': assumes plane of symmetry 12.

            * 'orthotropic': assumes planes of symmetry 12 and 13.

            * 'transverse_isotropic': assumes axis of symmetry 3

            * 'isotropic': assumes complete symmetry.
        elastic_properties : dict
            Elastic material properties (key, str) values (item, float).
            Expecting independent elastic moduli ('Eijkl') according to
            elastic symmetries.

        Returns
        -------
        technical_constants : dict
            Technical constants of elasticity according with elastic
            symmetries.
        """
        # Get available elastic symmetries and required elastic moduli
        elastic_symmetries = Elastic.get_available_elastic_symmetries()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check elastic symmetry and required elastic moduli
        if elastic_symmetry not in elastic_symmetries.keys():
            raise RuntimeError('Unavailable elastic symmetry.')
        else:
            # Get required elastic moduli
            required_moduli = elastic_symmetries[elastic_symmetry]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize independent elastic moduli
            ind_moduli = {str(modulus): 0.0 for modulus in
                          elastic_symmetries['triclinic']}
            # Check elastic moduli
            if not (elastic_symmetry == 'isotropic'
                    and {'E', 'v'}.issubset(set(elastic_properties.keys()))):
                # Loop over required elastic moduli
                for modulus in required_moduli:
                    # Set symmetric modulus
                    sym_modulus = modulus[3:5] + modulus[1:3]
                    # Check if requires elastic modulus has been provided
                    if modulus in elastic_properties.keys():
                        ind_moduli[modulus] = elastic_properties[modulus]
                    elif sym_modulus in elastic_properties.keys():
                        ind_moduli[modulus] = elastic_properties[sym_modulus]
                    else:
                        raise RuntimeError('Missing elastic moduli for '
                                           + elastic_symmetry + ' material.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize technical constants of elasticity
        technical_constants = {}
        # Compute technical constants of elasticity according with elastic
        # symmetries
        if elastic_symmetry == 'isotropic':
            if {'E', 'v'}.issubset(set(elastic_properties.keys())):
                # Assemble technical constants of elasticity
                technical_constants['E'] = elastic_properties['E']
                technical_constants['v'] = elastic_properties['v']
            else:
                # Get required elastic moduli
                E1111 = ind_moduli['E1111']
                E1122 = ind_moduli['E1122']
                # Compute Young's modulus
                E = (1.0/(E1111 + E1122))*(E1111**2 + E1111*E1122
                                           - 2.0*E1122**2)
                # Compute Poisson's coefficient
                v = E1122/(E1111 + E1122)
                # Assemble technical constants of elasticity
                technical_constants['E'] = E
                technical_constants['v'] = v
        else:
            raise RuntimeError('Technical constants are not implemented for '
                               + elastic_symmetry + ' material.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return technical_constants
