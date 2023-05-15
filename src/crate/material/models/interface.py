"""Constitutive model interface.

This module includes the interface to implement any constitutive model.

Classes
-------
ConstitutiveModel
    Constitutive model interface.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class ConstitutiveModel(ABC):
    """Constitutive model interface.

    Attributes
    ----------
    _name : str
        Constitutive model name.
    _strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
        Constitutive model strain formulation: infinitesimal strain formulation
        ('infinitesimal'), finite strain formulation ('finite') or finite
        strain formulation through kinematic extension (infinitesimal
        constitutive formulation and purely finite strain kinematic
        extension - 'finite-kinext').
    _source : {'crate',}
        Material constitutive model source.

    Methods
    -------
    get_required_properties()
        *abstract*: Get constitutive model material properties and constitutive
        options.
    state_init(self)
        *abstract*: Get initialized material constitutive model state
        variables.
    state_update(self, inc_strain, state_variables_old, \
                 su_max_n_iterations=20, su_conv_tol=1e-6)
        *abstract*: Perform material constitutive model state update.
    get_name(self)
        Get constitutive model name.
    get_strain_type(self)
        Get material constitutive model strain formulation.
    get_source(self)
        Get material constitutive model source.
    get_material_properties(self)
        Constitutive model material properties.
    """
    @abstractmethod
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
        pass
    # -------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_required_properties():
        """Get constitutive model material properties and constitutive options.

        Returns
        -------
        material_properties : list[str]
            Constitutive model material properties names (str).
        constitutive_options : dict
            Constitutive options (key, str) and available specifications
            (item, tuple[str]).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def state_init(self):
        """Get initialized material constitutive model state variables.

        Returns
        -------
        state_variables_init : dict
            Initialized material constitutive model state variables.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def state_update(self, inc_strain, state_variables_old,
                     su_max_n_iterations=20, su_conv_tol=1e-6):
        """Perform material constitutive model state update.

        Parameters
        ----------
        inc_strain : numpy.ndarray (2d)
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
        consistent_tangent_mf : numpy.ndarray (2d)
            Material constitutive model consistent tangent modulus in matricial
            form.
        """
        pass
    # -------------------------------------------------------------------------
    def get_name(self):
        """Get constitutive model name.

        Returns
        -------
        name : str
            Constitutive model name.
        """
        return self._name
    # -------------------------------------------------------------------------
    def get_strain_type(self):
        """Get material constitutive model strain formulation.

        Returns
        -------
        strain_type : {'infinitesimal', 'finite', 'finite-kinext'}
            Constitutive model strain formulation: infinitesimal strain
            formulation ('infinitesimal'), finite strain formulation ('finite')
            or finite strain formulation through kinematic extension
            (infinitesimal constitutive formulation and purely finite strain
            kinematic extension - 'finite-kinext').
        """
        return self._strain_type
    # -------------------------------------------------------------------------
    def get_source(self):
        """Get material constitutive model source.

        Returns
        -------
        source : {'crate',}
            Material constitutive model source.
        """
        return self._source
    # -------------------------------------------------------------------------
    def get_material_properties(self):
        """Constitutive model material properties.

        Returns
        -------
        material_properties : dict
            Constitutive model material properties (key, str) values
            (item, {int, float, bool}).
        """
        return copy.deepcopy(self._material_properties)
