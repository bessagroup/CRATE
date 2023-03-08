"""Compute RVE local elastic response database.

This module includes the class which embodies a RVE local elastic response
database. This physical-based database is required to compute the clustering
features and perform the RVE cluster analysis. This class also includes the
computation of the RVE elastic effective tangent modulus when certain
macroscale strain loadings are met.

Classes
-------
RVEElasticDatabase
    RVE local elastic response database class.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import numpy as np
# Local
import ioput.info as info
import tensor.matrixoperations as mop
from clustering.solution.ffthombasicscheme import FFTBasicScheme
from material.materialoperations import compute_rotation_tensor
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class RVEElasticDatabase:
    """RVE local elastic response database class.

    Attributes
    ----------
    _global_hom_stress_strain : dict
        RVE homogenized stress-strain response (item, numpy.ndarray (2d)) for
        each macroscale strain loading identifier (key, int). The homogenized
        strain and homogenized stress tensor components of the i-th loading
        increment are stored columnwise in the i-th row, sorted respectively.
        Infinitesimal strain tensor and Cauchy stress tensor (infinitesimal
        strains) or Deformation gradient and first Piola-Kirchhoff stress
        tensor (finite strains).
    _elastic_eff_modulus_matrix : numpy.ndarray (2d)
        RVE's elastic effective tangent modulus in matrix form, where each
        entry contains a given elastic modulus (similar to Voigt notation).
        The elastic effective tangent modulus is computed from the stress
        conjugate to the infinitesimal strain tensor (infinitesimal strains)
        or to the material logarithmic strain tensor (finite strains).
    _eff_elastic_properties : dict
        Elastic properties (key, str) and their values (item, float) estimated
        from the RVE's elastic effective tangent modulus.
    rve_global_response : numpy.ndarray (2d)
        RVE local elastic strain response for a given set of macroscale
        loadings, where each macroscale loading is associated with a set of
        independent strain components (numpy.ndarray of shape
        (n_voxels, n_mac_strains*n_strain_comps)). Each column is associated
        associated with a independent strain component of the infinitesimal
        strain tensor (infinitesimal strains) or material logarithmic strain
        tensor (finite strains).

    Methods
    -------
    compute_rve_response_database(self, dns_method, dns_method_data, \
                                  mac_strains, is_strain_sym)
        Compute RVE's local elastic strain response database.
    compute_rve_elastic_tangent_modulus(self, strain_magnitude_factor=1.0)
        Compute RVE's elastic effective tangent modulus.
    set_eff_isotropic_elastic_constants(self):
        Set isotropic elastic constants from effective tangent modulus.
    get_eff_isotropic_elastic_constants(self):
        Get isotropic elastic constants from effective tangent modulus.
    """
    def __init__(self, strain_formulation, problem_type, rve_dims,
                 n_voxels_dims, regular_grid, material_phases,
                 material_phases_properties):
        """Constructor.

        Parameters
        ----------
        strain_formulation : {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        regular_grid : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the material phase label (int) assigned to the
            corresponding voxel.
        material_phases : list[str]
            RVE material phases labels (str).
        material_phases_properties : dict
            Constitutive model material properties (item, dict) associated to
            each material phase (key, str).
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._regular_grid = regular_grid
        self._material_phases = material_phases
        self._material_phases_properties = material_phases_properties
        self._global_hom_stress_strain = {}
        self._elastic_eff_modulus_matrix = None
        self._eff_elastic_properties = None
        self.rve_global_response = None
    # -------------------------------------------------------------------------
    def compute_rve_response_database(self, dns_method, dns_method_data,
                                      mac_strains, is_strain_sym):
        """Compute RVE's local elastic strain response database.

        Build a RVE's local elastic strain response database by solving one or
        more microscale equilibrium problems (each associated with a given
        macroscale strain loading) through a given homogenization-based
        multi-scale method.

        ----

        Parameters
        ----------
        dns_method : {'fft-basic'}
            DNS homogenization-based multi-scale method.
        dns_method_data : dict
            Parameters of DNS homogenization-based multi-scale method.
        mac_strains : list[numpy.ndarray (2d)]
            List of macroscale strain loadings (numpy.ndarray (2d)).
            Infinitesimal strain tensor (infinitesimal strains) or deformation
            gradient (finite strains).
        is_strain_sym : bool
            True if the macroscale strain second-order tensor associated with
            the RVE's local elastic strain response database is symmetric by
            definition, False otherwise.
        """
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        # Set components order compatible with macroscale strain loading
        if is_strain_sym:
            comp_order = comp_order_sym
        else:
            comp_order = comp_order_nsym
        # Get total number of voxels
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instatiate homogenization-based multi-scale method
        if dns_method == 'fft_basic':
            homogenization_method = FFTBasicScheme(
                self._strain_formulation, self._problem_type, self._rve_dims,
                self._n_voxels_dims, self._regular_grid, self._material_phases,
                self._material_phases_properties)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown homogenization-based multi-scale '
                               'method.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize RVE's local elastic strain response database
        self.rve_global_response = \
            np.zeros((n_voxels, len(mac_strains)*len(comp_order)))
        # Loop over macroscale strain loadings
        for i in range(len(mac_strains)):
            info.displayinfo('5', 'Macroscale strain loading (' + str(i + 1)
                             + ' of ' + str(len(mac_strains)) + ')...', 2)
            # Get macroscale strain tensor
            mac_strain_id = i + 1
            mac_strain = mac_strains[i]
            # Compute RVE's local elastic strain response
            strain_vox = homogenization_method.compute_rve_local_response(
                mac_strain_id, mac_strain)
            # Assemble RVE's local elastic strain response to database
            for j in range(len(comp_order)):
                self.rve_global_response[:, i*len(comp_order) + j] = \
                    strain_vox[comp_order[j]].flatten()
            # Store RVE's homogenized stress-strain material response
            self._global_hom_stress_strain[mac_strain_id] = \
                copy.deepcopy(homogenization_method.get_hom_stress_strain())
    # -------------------------------------------------------------------------
    def compute_rve_elastic_tangent_modulus(self, strain_magnitude_factor=1.0):
        """Compute RVE's elastic effective tangent modulus.

        Parameters
        ----------
        strain_magnitude_factor : float, default=1.0
            Macroscale strain magnitude factor.
        """
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check number of macroscale strain loadings
        if len(self._global_hom_stress_strain.keys()) != len(comp_order_sym):
            raise RuntimeError('The computation of the RVE\'s elastic '
                               'effective tangent modulus requires the RVE '
                               'homogenized stress-strain response under a '
                               'suitable set of orthogonal macroscale '
                               'strain loadings.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize RVE's effective tangent modulus matrix
        elastic_eff_modulus_matrix = \
            np.zeros((len(comp_order_sym), len(comp_order_sym)))
        # Loop over orthogonal macroscale strain loadings
        for i in range(len(comp_order_sym)):
            # Get Kelvin factor associated with macroscale strain loading
            # strain component
            kf_i = mop.kelvin_factor(i, comp_order_sym)
            # Get macroscale strain loading identifier
            mac_strain_id = i + 1
            # Get RVE homogenized stress-strain material response
            hom_stress_strain = self._global_hom_stress_strain[mac_strain_id]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute stress conjugate
            if self._strain_formulation == 'infinitesimal':
                # Get homogenized Cauchy stress tensor
                stress_conjugate = \
                    hom_stress_strain[-1, len(comp_order_nsym):].reshape(
                        (n_dim, n_dim), order='F')
            else:
                # Get homogenized deformation gradient
                def_gradient = \
                    hom_stress_strain[-1, :len(comp_order_nsym)].reshape(
                        (n_dim, n_dim), order='F')
                # Get homogenized first Piola-Kirchhoff stress tensor
                first_piola_stress = \
                    hom_stress_strain[-1, len(comp_order_nsym):].reshape(
                        (n_dim, n_dim), order='F')
                # Compute rotation tensor
                rotation = compute_rotation_tensor(def_gradient)
                # Compute stress conjugate to material logarithmic strain
                stress_conjugate = np.matmul(
                    np.transpose(rotation), np.matmul(
                        first_piola_stress, np.matmul(
                            np.transpose(def_gradient), rotation)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble column of RVE's effective tangent modulus matrix
            for j in range(len(comp_order_sym)):
                # Get strain component and associated second-order indexes
                comp_j = comp_order_sym[j]
                so_idx = tuple([int(x) - 1 for x in list(comp_j)])
                # Get Kelvin factor associated with strain component
                kf_j = mop.kelvin_factor(j, comp_order_sym)
                # Assemble column of RVE's effective tangent modulus matrix
                elastic_eff_modulus_matrix[j, i] = \
                    (1.0/strain_magnitude_factor)*kf_j*stress_conjugate[so_idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove Kelvin coefficient
                elastic_eff_modulus_matrix[j, i] = \
                    (1.0/kf_i)*(1.0/kf_j)*elastic_eff_modulus_matrix[j, i]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store RVE's effective tangent modulus matrix
        self._elastic_eff_modulus_matrix = \
            copy.deepcopy(elastic_eff_modulus_matrix)
    # -------------------------------------------------------------------------
    def set_eff_isotropic_elastic_constants(self):
        """Set isotropic elastic constants from effective tangent modulus."""
        # Check if RVE's elastic effective tangent modulus is available
        if self._elastic_eff_modulus_matrix is None:
            raise RuntimeError('Unavailable RVE\'s elastic effective tangent '
                               'modulus.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get isotropic elastic modulii
        E1111 = self._elastic_eff_modulus_matrix[0, 0]
        E1122 = self._elastic_eff_modulus_matrix[0, 1]
        # Compute Young's modulus
        E = (1.0/(E1111 + E1122))*(E1111**2 + E1111*E1122 - 2.0*E1122**2)
        # Compute Poisson's coefficient
        v = E1122/(E1111 + E1122)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check isotropic elastic constants
        is_admissible = (E >= 0) and (v >= 0.0 and (v/0.5) <= 1.0)
        if not is_admissible:
            raise RuntimeError('Inadmissible isotropic elastic constants from '
                               'RVE\'s elastic effective tangent modulus.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update estimates of effective elastic properties
        self._eff_elastic_properties = {'E': E, 'v': v}
    # -------------------------------------------------------------------------
    def get_eff_isotropic_elastic_constants(self):
        """Get isotropic elastic constants from effective tangent modulus.

        Returns
        -------
        eff_elastic_properties : dict
            Elastic properties (key, str) and their values (item, float)
            estimated from the RVE's elastic effective tangent modulus.
        """
        return copy.deepcopy(self._eff_elastic_properties)
