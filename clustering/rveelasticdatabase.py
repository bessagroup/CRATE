#
# RVE Elastic Response Database Module (CRATE Program)
# ==========================================================================================
# Summary:
# Computation of a RVE's local elastic strain response database through the solution of
# microscale equilibrium problems (each associated with a given macroscale strain
# loading) through a given homogenization-based multiscale method.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | October 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Defining abstract base classes
from abc import ABC, abstractmethod
# Display messages
import ioput.info as info
# FFT-based Homogenization Basic Scheme
import clustering.solution.ffthombasicscheme as ffthom
# Links related procedures
import links.ioput.genlinksinputdatafile as linksglif
import links.execution.linksexec as linksexec
import links.postprocess.linkspostprocess as linkspp
#
#                                                        RVE local elastic response database
# ==========================================================================================
class RVEElasticDatabase:
    '''RVE local elastic response database class.

    Attributes
    ----------
    rve_global_response: ndarray of shape (n_voxels, n_mac_strains*n_strain_comps)
        RVE local elastic response for a given set of macroscale loadings, where each
        macroscale loading is associated with a set of independent strain components.
    '''
    def __init__(self, homogenization_method_id, mac_strains, problem_dict, dirs_dict,
                 rg_dict, mat_dict, clst_dict):
        '''RVE DNS local response database constructor.

        Parameters
        ----------
        homogenization_method_id: int
            Multiscale homogenization method identifier.
        mac_strains: list
            List of macroscale strain loadings (ndarray, second-order strain tensor)
            to impose in turn on the RVE.
        '''
        self._homogenization_method_id = homogenization_method_id
        self.mac_strains = mac_strains
        self._problem_dict = problem_dict
        self._dirs_dict = dirs_dict
        self._rg_dict = rg_dict
        self._mat_dict = mat_dict
        self._clst_dict = clst_dict
        if self._problem_dict['strain_formulation'] == 1:
            self.comp_order = self._problem_dict['comp_order_sym']
        else:
            self.comp_order = self._problem_dict['comp_order_nsym']
        regular_grid = rg_dict['regular_grid']
        n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        self._n_voxels = np.prod(n_voxels_dims)
        self.rve_global_response = None
    # --------------------------------------------------------------------------------------
    def set_RVE_response_database(self):
        '''Compute RVE's local elastic strain response database.

        Build a RVE's local elastic strain response database by solving one or more
        microscale equilibrium problems (each associated with a given macroscale strain
        loading) through a given homogenization-based multiscale method.
        '''
        # Instatiate homogenization-based multiscale method
        if self._homogenization_method_id == 1:
            homogenization_method = FFTHomogenizationBasicScheme(self._problem_dict,
                self._rg_dict, self._mat_dict)
        elif self._homogenization_method_id == 2:
            homogenization_method = LinksFEMHomogenization(self._problem_dict,
                self._dirs_dict, self._rg_dict, self._mat_dict, self._clst_dict)
        else:
            raise RuntimeError('Unknown multiscale homogenization method.')
        # ----------------------------------------------------------------------------------
        # Compute RVE's local elastic strain response database
        n_mac_strains = len(self.mac_strains)
        n_strain_comps = len(self.comp_order)
        self.rve_global_response = np.zeros((self._n_voxels, n_mac_strains*n_strain_comps))
        # Loop over macroscale strain loadings
        for i in range(len(self.mac_strains)):
            info.displayinfo('5', 'Macroscale strain loading (' + str(i + 1) + ' of ' +
                             str(len(self.mac_strains)) + ')...', 2)
            # Get macroscale strain loading
            mac_strain = self.mac_strains[i]
            # Compute RVE's local elastic strain response
            strain_vox = homogenization_method.get_rve_local_response(mac_strain)
            # Assemble RVE's local elastic strain response to database
            for j in range(len(self.comp_order)):
                comp_j = self.comp_order[j]
                self.rve_global_response[:, i*len(self.comp_order) + j] = \
                    strain_vox[comp_j].flatten()
#
#                                                Homogenization-based multiscale DNS methods
# ==========================================================================================
class DNSHomogenizationMethod(ABC):
    '''Homogenization-based multiscale DNS method interface.'''

    @abstractmethod
    def __init__(self):
        '''Homogenization-based multiscale DNS method constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_rve_local_response(self, mac_strain):
        '''Compute RVE local elastic strain response.

        Compute the local response of a given material's representative volume element (RVE)
        subjected to a given macroscale strain loading. It is assumed that the RVE is
        spatially discretized in a regular grid of voxels and that all material phase's
        behavior is governed by the linear elastic constitutive model.

        Parameters
        ----------
        mac_strain: ndarray
            Second-order macroscale strain loading.

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str).
        '''
        pass
# ------------------------------------------------------------------------------------------
class FFTHomogenizationBasicScheme(DNSHomogenizationMethod):
    '''FFT-based homogenization basic scheme.

    FFT-based homogenization basic scheme proposed by H. Moulinec and P. Suquet
    ("A numerical method for computing the overall response of nonlinear composites with
    complex microstructure" Comp Methods Appl M 157 (1998):69-94) for the solution of
    microscale equilibrium problems of linear elastic heterogeneous materials.
    '''
    def __init__(self, problem_dict, rg_dict, mat_dict):
        '''FFT-based homogenization basic scheme constructor.'''
        self._problem_dict = problem_dict
        self._rg_dict = rg_dict
        self._mat_dict = mat_dict
    # --------------------------------------------------------------------------------------
    def get_rve_local_response(self, mac_strain):
        '''Compute RVE local elastic strain response.'''
        # Solve RVE microscale equilibrium problem and get RVE local elastic strain response
        strain_vox = ffthom.ffthombasicscheme(self._problem_dict, self._rg_dict,
                                              self._mat_dict, mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return strain_vox
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
    def get_rve_local_response(self, mac_strain):
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
