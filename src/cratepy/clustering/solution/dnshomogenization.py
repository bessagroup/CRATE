"""DNS Homogenization-based multi-scale method interface.

This module includes the interface to implement any direct numerical simulation
(DNS) homogenization-based multi-scale method suitable to solve a microscale
equilibrium problem where the RVE is spatially discretized in a regular grid of
voxels.

Classes
-------
DNSHomogenizationMethod
    DNS homogenization-based multi-scale DNS method interface.
"""
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class DNSHomogenizationMethod(ABC):
    """DNS homogenization-based multi-scale method interface.

    Methods
    -------
    compute_rve_local_response(self, mac_strain_id, mac_strain)
        *abstract*: Compute RVE local strain response.
    get_hom_stress_strain(self)
        *abstract*: Get the homogenized strain-stress material response.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def compute_rve_local_response(self, mac_strain_id, mac_strain):
        """Compute RVE local strain response.

        Compute the RVE local strain response (solution of microscale
        equilibrium problem) when subjected to a given macroscale strain
        loading, namely a macroscale infinitesimal strain tensor (infinitesimal
        strains) or a macroscale deformation gradient (finite strains). It is
        assumed that the RVE is spatially discretized in a regular grid of
        voxels.

        ----

        Parameters
        ----------
        mac_strain_id : int
            Macroscale strain second-order tensor identifier.
        mac_strain : numpy.ndarray (2d)
            Macroscale strain second-order tensor. Infinitesimal strain tensor
            (infinitesimal strains) or deformation gradient (finite strains).

        Returns
        -------
        strain_vox: dict
            RVE local strain response (item, numpy.ndarray of shape equal to
            RVE regular grid discretization) for each strain component
            (key, str). Infinitesimal strain tensor (infinitesimal strains) or
            material logarithmic strain tensor (finite strains).
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_hom_stress_strain(self):
        """Get the homogenized strain-stress material response.

        Returns
        -------
        _hom_stress_strain : numpy.ndarray (2d)
            RVE homogenized stress-strain response (item, numpy.ndarray (2d))
            for each macroscale strain loading identifier (key, int). The
            homogenized strain and homogenized stress tensor components of the
            i-th loading increment are stored columnwise in the i-th row,
            sorted respectively. Infinitesimal strain tensor and Cauchy stress
            tensor (infinitesimal strains) or Deformation gradient and first
            Piola-Kirchhoff stress tensor (finite strains).
        """
        pass
