#
# RVE Elastic Response Database Module (CRATE Program)
# ==========================================================================================
# Summary:
# Computation of a RVE's local elastic strain response database through the solution of
# microscale equilibrium problems (each associated with a given macroscale strain
# loading) by a given homogenization-based multi-scale method.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Refactoring and OOP implementation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Defining abstract base classes
from abc import ABC, abstractmethod
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# FFT-based Homogenization Basic Scheme
from clustering.solution.ffthombasicscheme import FFTBasicScheme
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
    def __init__(self, strain_formulation, problem_type, rve_dims, n_voxels_dims,
                 regular_grid, material_phases, material_properties):
        '''RVE DNS local response database constructor.

        Parameters
        ----------
        homogenization_method_id: int
            Multiscale homogenization method identifier.
        mac_strains: list
            List of macroscale strain loadings (ndarray, second-order strain tensor)
            to impose in turn on the RVE.
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_properties = material_properties
        self.rve_global_response = None
    # --------------------------------------------------------------------------------------
    def compute_rve_response_database(self, method, mac_strains, is_strain_sym=True):
        '''Compute RVE's local elastic strain response database.

        Build a RVE's local elastic strain response database by solving one or more
        microscale equilibrium problems (each associated with a given macroscale strain
        loading) through a given homogenization-based multi-scale method.

        Parameters
        ----------
        method : str, {'fft-basic',}
            Homogenization-based multi-scale method.
        mac_strains : list
            List of macroscale strain loadings (2darray, strain second-order tensor).
        is_strain_sym : bool, default=True
            True if the macroscale strain second-order tensor is symmetric by definition,
            False otherwise.
        '''
        # Get problem type parameters
        _, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        # Set components order compatible with macroscale strain loading
        if is_strain_sym:
            comp_order = comp_order_sym
        else:
            comp_order = comp_order_nsym
        # Get total number of voxels
        n_voxels = self._n_voxels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instatiate homogenization-based multi-scale method
        if method == 'fft_basic':
            homogenization_method = \
                FFTBasicScheme(self._strain_formulation, self._problem_type, self._rve_dims,
                               self._n_voxels_dims, self._regular_grid,
                               self._material_phases, self._material_properties)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif method == 'fem_links':
            RuntimeError('Temporarily unavailable!')
            #homogenization_method = LinksFEMHomogenization(self._problem_dict,
            #    self._dirs_dict, self._rg_dict, self._mat_dict, self._clst_dict)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown homogenization-based multi-scale method.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize RVE's local elastic strain response database
        self.rve_global_response = np.zeros((n_voxels, len(mac_strains)*len(comp_order)))
        # Loop over macroscale strain loadings
        for i in range(len(mac_strains)):
            info.displayinfo('5', 'Macroscale strain loading (' + str(i + 1) + ' of ' +
                             str(len(mac_strains)) + ')...', 2)
            # Get macroscale strain tensor
            mac_strain = self.mac_strains[i]
            # Compute RVE's local elastic strain response
            strain_vox = homogenization_method.get_rve_local_response(mac_strain)
            # Assemble RVE's local elastic strain response to database
            for j in range(len(comp_order)):
                self.rve_global_response[:, i*len(comp_order) + j] = \
                    strain_vox[comp_order[j]].flatten()
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
        subjected to a given macroscale strain loading. It is assumed that the RVE is
        spatially discretized in a regular grid of voxels.

        Parameters
        ----------
        mac_strain : 2darray
            Macroscale strain second-order tensor.

        Returns
        -------
        strain_vox: dict
            Local strain response (item, ndarray of shape equal to RVE regular grid
            discretization) for each strain component (key, str).
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
