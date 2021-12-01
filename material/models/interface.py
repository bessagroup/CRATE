#
# Constitutive Modeling Interface (CRATE Program)
# ==========================================================================================
# Summary:
# Interface of material constitutive models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Nov 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Defining abstract base classes
from abc import ABC, abstractmethod
# Shallow and deep copy operations
import copy
#
#                                                               Constitutive model interface
# ==========================================================================================
class ConstitutiveModel(ABC):
    '''Constitutive model interface.

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
    '''
    @abstractmethod
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
        pass
    # --------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
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
    # --------------------------------------------------------------------------------------
    def get_name(self):
        '''Get constitutive model name.

        Returns
        -------
        name : str
            Constitutive model name.
        '''
        return self._name
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
    def get_source(self):
        '''Get material constitutive model source.

        Returns
        -------
        source : str, {'crate', }
            Material constitutive model source.
        '''
        return self._source
    # --------------------------------------------------------------------------------------
    def get_material_properties(self):
        '''Constitutive model material properties.

        Returns
        -------
        material_properties : dict
            Constitutive model material properties (key, str) values (item, int/float/bool).
        '''
        return copy.deepcopy(self._material_properties)
