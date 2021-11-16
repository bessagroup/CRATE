#
# Homogenization-based Multi-scale Methods Interface (CRATE Program)
# ==========================================================================================
# Summary:
# Interface of Direct Numerical Simulation homogenization-based multi-scale methods required
# to solve RVE elastic microscale equilibrium problems.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Defining abstract base classes
from abc import ABC, abstractmethod
#
#                                               Homogenization-based multi-scale DNS methods
# ==========================================================================================
class DNSHomogenizationMethod(ABC):
    '''Homogenization-based multi-scale DNS method interface.'''
    @abstractmethod
    def __init__(self):
        '''Homogenization-based multi-scale DNS method constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def compute_rve_local_response(self, mac_strain_id, mac_strain):
        '''Compute RVE local elastic strain response.

        Compute the local response of the material's representative volume element (RVE)
        subjected to a given macroscale strain loading: infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains). It is assumed that
        the RVE is spatially discretized in a regular grid of voxels.

        Parameters
        ----------
        mac_strain_id : int
            Macroscale strain second-order tensor identifier.
        mac_strain : 2darray
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str). Infinitesimal strain
            tensor (infinitesimal strains) or material logarithmic strain tensor (finite
            strains).
        '''
        pass
