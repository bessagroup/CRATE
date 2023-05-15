"""St.Venant-Kirchhoff hyperelastic constitutive model.

This module includes the implementation of the St.Venant-Kirchhoff hyperelastic
constitutive model under general anisotropic elasticity. The constitutive
formulation can be found in de Vieira de Carvalho et al. (2022) [#]_.

.. [#] Vieira de Carvalho, M., de Bortoli, D., and Andrade Pires, F.M. (2022).
       Consistent modeling of the coupling between crystallographic slip and
       martensitic phase transformation for mechanically induced loadings.
       Int J Numer Methods Eng. 123(14): 3179–3236 (see
       `here <https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6962>`_)

Classes
-------
StVenantKirchhoff
    St.Venant-Kirchhoff hyperelastic constitutive model.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
import tensor.tensoroperations as top
import tensor.matrixoperations as mop
from material.materialoperations import first_piola_from_second_piola, \
                                        cauchy_from_second_piola, \
                                        material_from_spatial_tangent_modulus
from material.models.interface import ConstitutiveModel
from material.models.elastic import Elastic
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class StVenantKirchhoff(ConstitutiveModel):
    """St.Venant-Kirchhoff hyperelastic constitutive model.

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
    _source : {'crate', }
        Material constitutive model source.
    _ndim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.

    Methods
    -------
    get_required_properties()
        Get constitutive model material properties and constitutive options.
    state_init(self)
        Get initialized material constitutive model state variables.
    state_update(self, inc_strain, state_variables_old, \
                 su_max_n_iterations=20, su_conv_tol=1e-6)
        Perform material constitutive model state update.
    """
    def __init__(self, strain_formulation, problem_type, material_properties):
        """Constitutive model constructor.

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
        self._name = 'stvenant_kirchhoff'
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._material_properties = material_properties
        # Set strain formulation
        self._strain_type = 'finite'
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
        # Compute technical constants of elasticity
        if elastic_symmetry == 'isotropic':
            # Compute technical constants of elasticity
            technical_constants = Elastic.get_technical_from_elastic_modulii(
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
        # Get available elastic symmetries and required elastic modulii
        elastic_symmetries = Elastic.get_available_elastic_symmetries()
        # Set constitutive options and available specifications
        constitutive_options = \
            {'elastic_symmetry': tuple(elastic_symmetries.keys())}
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

            * Elastic deformation gradient (matricial form).

            * *Symbol*: :math:`\\boldsymbol{F}^{e}`

        * ``strain_mf``

            * Deformation gradient (matricial form).

            * *Symbol*: :math:`\\boldsymbol{F}`

        * ``stress_mf``

            * First Piola-Kirchhoff stress tensor (matricial form).

            * *Symbol*: :math:`\\boldsymbol{P}`

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
        state_variables_init['e_strain_mf'] = mop.get_tensor_mf(
            np.eye(self._n_dim), self._n_dim, self._comp_order_nsym)
        state_variables_init['strain_mf'] = mop.get_tensor_mf(
            np.eye(self._n_dim), self._n_dim, self._comp_order_nsym)
        state_variables_init['stress_mf'] = mop.get_tensor_mf(
            np.zeros((self._n_dim, self._n_dim)), self._n_dim,
            self._comp_order_nsym)
        state_variables_init['is_su_fail'] = False
        # Set additional out-of-plane strain and stress components
        if self._problem_type == 1:
            state_variables_init['e_strain_33'] = 1.0
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
        #                                                          State update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build incremental deformation gradient matricial form
        inc_strain_mf = mop.get_tensor_mf(inc_strain, self._n_dim,
                                          self._comp_order_nsym)
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
            # Build deformation gradient tensors (matricial form) by including
            # the appropriate out-of-plain components
            inc_strain_mf = mop.get_state_3Dmf_from_2Dmf(self._problem_type,
                                                         inc_strain_mf,
                                                         comp_33=1.0)
            e_strain_old_mf = mop.get_state_3Dmf_from_2Dmf(self._problem_type,
                                                           e_strain_old_mf,
                                                           e_strain_33_old)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build last converged and incremental deformation gradient tensors
        e_strain_old = mop.get_tensor_from_mf(e_strain_old_mf, n_dim,
                                              comp_order_nsym)
        inc_strain = mop.get_tensor_from_mf(inc_strain_mf, n_dim,
                                            comp_order_nsym)
        # Update elastic deformation gradient tensor
        e_strain = np.matmul(inc_strain, e_strain_old)
        # Build elastic deformationg gradient tensor matricial form
        e_strain_mf = mop.get_tensor_mf(e_strain, n_dim, comp_order_nsym)
        # Compute right Cauchy-Green strain tensor
        right_cauchy_green = np.matmul(np.transpose(e_strain), e_strain)
        # Compute Green-Lagrange strain tensor
        green_lagrange = 0.5*(right_cauchy_green - np.eye(n_dim))
        # Compute 3D elasticity tensor (matricial form)
        elastic_tangent_mf = Elastic.elastic_tangent_modulus(
            self._material_properties,
            elastic_symmetry=self._material_properties['elastic_symmetry'])
        # Compute second Piola-Kirchhoff stress tensor (matricial form)
        second_piola_stress_mf = np.matmul(
            elastic_tangent_mf, mop.get_tensor_mf(green_lagrange, n_dim,
                                                  comp_order_sym))
        # Build second Piola-Kirchhoff stress tensor
        second_piola_stress = mop.get_tensor_from_mf(second_piola_stress_mf,
                                                     n_dim, comp_order_sym)
        # Compute first Piola-Kirchhoff stress tensor
        first_piola_stress = first_piola_from_second_piola(e_strain,
                                                           second_piola_stress)
        # Build first Piola-Kirchhoff stress tensor matricial form
        stress_mf = mop.get_tensor_mf(first_piola_stress, n_dim,
                                      comp_order_nsym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get the out-of-plane strain and stress components
        if self._problem_type == 1:
            e_strain_33 = e_strain_mf[comp_order_nsym.index('33')]
            stress_33 = stress_mf[comp_order_nsym.index('33')]
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
        #
        #                                            Consistent tangent modulus
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set required identity tensors
        soid, _, _, _, _, _, _ = top.get_id_operators(n_dim)
        # Get Cauchy stress tensor
        cauchy_stress = cauchy_from_second_piola(e_strain, second_piola_stress)
        # Get 3D elasticity tensor
        elastic_tangent = mop.get_tensor_from_mf(elastic_tangent_mf, n_dim,
                                                 comp_order_sym)
        # Compute derivative of Kirchhoff stress tensor in order to the elastic
        # deformation gradient tensor
        der_kirchhoff_e_strain = \
            top.dyad22_2(soid, np.matmul(e_strain, second_piola_stress)) \
            + top.dyad22_3(np.matmul(e_strain, second_piola_stress), soid) \
            + top.dot24_1(e_strain, top.dot24_2(
                e_strain, top.dot24_3(e_strain, elastic_tangent)))
        # Compute spatial consistent tangent modulus
        spatial_consistent_tangent = (1.0/np.linalg.det(e_strain)) \
            * top.dot42_3(der_kirchhoff_e_strain, np.transpose(e_strain)) \
            - top.dyad22_3(cauchy_stress, soid)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute material consistent tangent modulus (matricial form)
        material_consistent_tangent = material_from_spatial_tangent_modulus(
            spatial_consistent_tangent, e_strain)
        consistent_tangent_mf = mop.get_tensor_mf(material_consistent_tangent,
                                                  n_dim, comp_order_nsym)
        #
        #                                                    3D > 2D Conversion
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the
        # 2D consistent tangent modulus (matricial form) from the
        # 3D counterpart
        if self._problem_type == 1:
            consistent_tangent_mf = mop.get_state_2Dmf_from_3Dmf(
                self._problem_type, consistent_tangent_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return state_variables, consistent_tangent_mf
