#
# Direct Numerical Simulation Multi-scale Methods Module (CRATE Program)
# ==========================================================================================
# Summary:
# Interface of Direct Numerical Simulation multi-scale methods required to solve RVE
# elastic microscale equilibrium problems.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Defining abstract base classes
from abc import ABC, abstractmethod
# Links related procedures
import links.ioput.genlinksinputdatafile as linksglif
import links.execution.linksexec as linksexec
import links.postprocess.linkspostprocess as linkspp
#
#                                               Homogenization-based multi-scale DNS methods
# ==========================================================================================
class DNSHomogenizationMethod(ABC):
    '''Homogenization-based multiscale DNS method interface.'''
    @abstractmethod
    def __init__(self):
        '''Homogenization-based multiscale DNS method constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def compute_rve_local_response(self, mac_strain):
        '''Compute RVE local elastic strain response.

        Compute the local response of the material's representative volume element (RVE)
        subjected to a given macroscale strain loading: infinitesimal strain tensor
        (infinitesimal strains) or deformation gradient (finite strains). It is assumed that
        the RVE is spatially discretized in a regular grid of voxels.

        Parameters
        ----------
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
# ------------------------------------------------------------------------------------------
class LinksFEMHomogenization(DNSHomogenizationMethod):
    '''Links FEM-based homogenization method.

    FEM-based first-order multiscale hierarchical model based on computational
    homogenization implemented in multiscale finite element code Links (Large Strain
    Implicit Nonlinear Analysis of Solids Linking Scales), developed by the CM2S research
    group at the Faculty of Engineering, University of Porto.
    '''
    def __init__(self, problem_dict, dirs_dict, rg_dict, mat_dict, clst_dict):
        '''Links FEM-based homogenization method constructor.'''
        self._problem_dict = problem_dict
        self._dirs_dict = dirs_dict
        self._rg_dict = rg_dict
        self._mat_dict = mat_dict
        self._clst_dict = clst_dict
    # --------------------------------------------------------------------------------------
    def compute_rve_local_response(self, mac_strain):
        '''Compute RVE local elastic strain response.'''
        # Set Links input file name
        Links_file_name = 'mac_strain_' + ''.join([str(int(x)) for x in
                                                  mac_strain.flatten(order='F')])
        # Generate RVE microscale problem's Links input data file
        links_file_path = linksglif.writelinksinputdatafile(Links_file_name, self._dirs_dict,
            self._problem_dict, self._mat_dict, self._rg_dict, self._clst_dict, mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve RVE microscale equilibrium problem
        links_bin_path = self._clst_dict['links_dict']['links_bin_path']
        linksexec.runlinks(links_bin_path, links_file_path)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE local elastic strain response from Links' output file
        n_dim = self._problem_dict['n_dim']
        comp_order_sym = self._problem_dict['comp_order_sym']
        n_voxels_dims = self._rg_dict['n_voxels_dims']
        strain_vox = linkspp.getlinksstrainvox(links_file_path, n_dim, comp_order_sym,
                                               n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_vox
