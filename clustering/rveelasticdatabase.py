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
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# FFT-based Homogenization Basic Scheme
from clustering.solution.ffthombasicscheme import FFTBasicScheme
# Links homogenization
from links.offlinestage import LinksFEMHomogenization
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
                 regular_grid, material_phases, material_phases_properties):
        '''RVE DNS local response database constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        rve_dims : list
            RVE size in each dimension.
        n_voxels_dims : list
            Number of voxels in each dimension of the regular grid (spatial discretization
            of the RVE).
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to each material
            phase (key, str).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_phases_properties = material_phases_properties
        self.rve_global_response = None
    # --------------------------------------------------------------------------------------
    def compute_rve_response_database(self, dns_method, dns_method_data, mac_strains,
                                      is_strain_sym):
        '''Compute RVE's local elastic strain response database.

        Build a RVE's local elastic strain response database by solving one or more
        microscale equilibrium problems (each associated with a given macroscale strain
        loading) through a given homogenization-based multi-scale method.

        Parameters
        ----------
        dns_method : str, {'fft-basic', 'fem_links'}
            DNS homogenization-based multi-scale method.
        dns_method_data : dict
            Parameters of DNS homogenization-based multi-scale method.
        mac_strains : list
            List of macroscale strain loadings (2darray, strain second-order tensor).
        is_strain_sym : bool
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
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instatiate homogenization-based multi-scale method
        if dns_method == 'fft_basic':
            homogenization_method = \
                FFTBasicScheme(self._strain_formulation, self._problem_type, self._rve_dims,
                               self._n_voxels_dims, self._regular_grid,
                               self._material_phases, self._material_phases_properties)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif dns_method == 'fem_links':
            homogenization_method = \
                LinksFEMHomogenization(self._strain_formulation, self._problem_type,
                                       self._rve_dims, self._n_voxels_dims,
                                       self._regular_grid, self._material_phases,
                                       self._material_phases_properties,
                                       dns_method_data['links_bin_path'],
                                       dns_method_data['links_offline_dir'],
                                       dns_method_data['fe_order'],
                                       dns_method_data['boundary_type'],
                                       dns_method_data['convergence_tolerance'],
                                       dns_method_data['element_avg_output_mode'])
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
            mac_strain_id = i + 1
            mac_strain = mac_strains[i]
            # Compute RVE's local elastic strain response
            strain_vox = homogenization_method.compute_rve_local_response(mac_strain_id,
                                                                          mac_strain)
            # Assemble RVE's local elastic strain response to database
            for j in range(len(comp_order)):
                self.rve_global_response[:, i*len(comp_order) + j] = \
                    strain_vox[comp_order[j]].flatten()
