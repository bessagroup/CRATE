"""Compute RVE clustering features data.

This module includes all the required tools to compute the global clustering
data matrix that is needed to perform the RVE cluster analysis. Besides the
overall high-level management of such a computation, it includes the interface
to implement different clustering features as well as an interface for data
standardization algorithms.

Classes
-------
ClusterAnalysisData
    Features data required to perform the RVE cluster analysis.
FeatureAlgorithm
    Feature computation algorithm interface.
StrainConcentrationTensor
    Fourth-order elastic strain concentration tensor.
Standardizer
    Data standardization algorithm interface.
MinMaxScaler:
    Min-Max scaling algorithm (wrapper).
StandardScaler
    Standard scaling algorithm (wrapper).

Functions
---------
set_clustering_data
    Compute the features data required to perform the RVE cluster analysis.
get_available_clustering_features
    Get available clustering features and corresponding descriptors.
def_gradient_from_log_strain
    Get deformation gradient from material logarithmic strain tensor.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
# Third-party
import numpy as np
import sklearn.preprocessing as skpp
# Local
import ioput.info as info
import tensor.tensoroperations as top
import tensor.matrixoperations as mop
from clustering.rveelasticdatabase import RVEElasticDatabase
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def set_clustering_data(strain_formulation, problem_type, rve_dims,
                        n_voxels_dims, regular_grid, material_phases,
                        material_phases_properties, dns_method,
                        dns_method_data, standardization_method,
                        base_clustering_scheme, adaptive_clustering_scheme):
    """Compute the features data required to perform the RVE cluster analysis.

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
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
        Regular grid of voxels (spatial discretization of the RVE), where each
        entry contains the material phase label (int) assigned to the
        corresponding voxel.
    material_phases : list[str]
        RVE material phases labels (str).
    material_phases_properties : dict
        Constitutive model material properties (item, dict) associated to each
        material phase (key, str).
    dns_method : str
        DNS homogenization-based multi-scale method.
    dns_method_data : dict
        Parameters of DNS homogenization-based multi-scale method.
    standardization_method : int
        Identifier of global cluster analysis data standardization algorithm.
    base_clustering_scheme : dict
        Prescribed base clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).
    adaptive_clustering_scheme : dict
        Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
        (n_clusterings, 3)) for each material phase (key, str). Each row is
        associated with a unique clustering characterized by a clustering
        algorithm (col 1, int), a list of features (col 2, list[int]) and a
        list of the features data matrix' indexes (col 3, list[int]).

    Returns
    -------
    clustering_data : ClusterAnalysisData
        Feature data required to perform the RVE cluster analysis.
    rve_elastic_database : RVEElasticDatabase
        RVE's local elastic response database.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Setting cluster analysis\' features...')
    # Get available clustering features descriptors
    feature_descriptors = get_available_clustering_features(strain_formulation,
                                                            problem_type)
    # Instatiante cluster analysis data
    clustering_data = ClusterAnalysisData(
        strain_formulation, problem_type, rve_dims, n_voxels_dims,
        base_clustering_scheme, adaptive_clustering_scheme,
        feature_descriptors)
    # Set prescribed clustering features
    clustering_data.set_prescribed_features()
    # Set prescribed clustering features' clustering global data matrix'
    # indexes
    clustering_data.set_feature_global_indexes()
    # Set required macroscale strain loadings to compute clustering features
    clustering_data.set_clustering_mac_strains()
    mac_strains = clustering_data.get_clustering_mac_strains()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing RVE local elastic strain response '
                     'database...')
    # Instatiate RVE's local elastic response database
    rve_elastic_database = RVEElasticDatabase(strain_formulation, problem_type,
                                              rve_dims, n_voxels_dims,
                                              regular_grid, material_phases,
                                              material_phases_properties)
    # Compute RVE's elastic response database
    rve_elastic_database.compute_rve_response_database(
        dns_method, dns_method_data, mac_strains, is_strain_sym=True)
    # Compute RVE's elastic effective tangent modulus if the elastic response
    # database contains a suitable set of orthogonal macroscale strain loadings
    if clustering_data.get_features() == {1}:
        # Get strain magnitude factor associated with orthogonal macroscale
        # strain loadings
        strain_magnitude_factor = feature_descriptors['1'][3]
        # Compute RVE's elastic effective tangent modulus
        rve_elastic_database.compute_rve_elastic_tangent_modulus(
            strain_magnitude_factor=strain_magnitude_factor)
        # Estimate isotropic elastic constants from RVE's elastic effective
        # tangent modulus
        rve_elastic_database.set_eff_isotropic_elastic_constants()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing cluster analysis global data matrix...')
    # Compute clustering global data matrix containing all clustering features
    clustering_data.set_global_data_matrix(
        rve_elastic_database.rve_global_response)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Standardizing cluster analysis global '
                     'data matrix...')
    # Instantiate standardization algorithm
    if standardization_method == 1:
        standardizer = MinMaxScaler()
    elif standardization_method == 2:
        standardizer = StandardScaler()
    else:
        raise RuntimeError('Unknown standardization method.')
    # Standardize clustering global data matrix
    clustering_data._global_data_matrix = \
        standardizer.get_standardized_data_matrix(
            clustering_data._global_data_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return clustering_data, rve_elastic_database
# =============================================================================
def get_available_clustering_features(strain_formulation, problem_type):
    """Get available clustering features and corresponding descriptors.

    Available clustering features identifiers:

    * Identifier: 1

        * *Infinitesimal strains*: Fourth-order local elastic strain
          concentration tensor based on the elastic infinitesimal strain
          tensor,

          .. math::

             \\boldsymbol{\\varepsilon}_{\\mu}^{e}(\\boldsymbol{Y}) =
             \\boldsymbol{\\mathsf{H}}^{e}(\\boldsymbol{Y}):
             \\boldsymbol{\\varepsilon}^{e} (\\boldsymbol{X}) \\, , \\quad
             \\forall \\boldsymbol{Y} \\in \\Omega_{\\mu,\\,0} \\, ,

          where :math:`\\boldsymbol{\\mathsf{H}}^{e}` is the
          fourth-order local elastic strain concentration tensor,
          :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{e}` is the
          microscale elastic infinitesimal strain tensor,
          :math:`\\boldsymbol{\\varepsilon}^{e}` is the
          macroscale elastic infinitesimal strain tensor,
          :math:`\\boldsymbol{Y}` is a point of the microscale reference
          configuration (:math:`\\Omega_{\\mu,\\,0}`), and
          :math:`\\boldsymbol{X}` is a point of the macroscale reference
          configuration (:math:`\\Omega_{0}`).

        * *Finite strains*: Fourth-order local elastic strain concentration
          tensor based on the elastic material logarithmic strain tensor,

          .. math::

             \\boldsymbol{E}_{\\mu}^{e}(\\boldsymbol{Y}) =
             \\boldsymbol{\\mathsf{H}}^{e}(\\boldsymbol{Y}):
             \\boldsymbol{E}^{e} (\\boldsymbol{X}) \\, , \\quad
             \\forall \\boldsymbol{Y} \\in \\Omega_{\\mu,\\,0} \\, ,

          where :math:`\\boldsymbol{\\mathsf{H}}^{e}` is the
          fourth-order local elastic strain concentration tensor,
          :math:`\\boldsymbol{E}_{\\mu}^{e}` is the
          microscale elastic material logarithmic strain tensor,
          :math:`\\boldsymbol{E}^{e}` is the
          macroscale elastic material logarithmic strain tensor,
          :math:`\\boldsymbol{Y}` is a point of the microscale reference
          configuration (:math:`\\Omega_{\\mu,\\,0}`), and
          :math:`\\boldsymbol{X}` is a point of the macroscale reference
          configuration (:math:`\\Omega_{0}`).

    ----

    * Identifier: 2

        * Spatial coordinates first-order tensor in the reference
          configuration, :math:`\\boldsymbol{Y}`.

    ----

    Parameters
    ----------
    strain_formulation: {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2),
        2D axisymmetric (3) and 3D (4).
    n_dim: int
        Number of spatial dimensions.
    comp_order_sym: list[str]
        Symmetric strain/stress components (str) order.
    comp_order_nsym: list[str]
        Nonsymmetric strain/stress components (str) order.

    Returns
    -------
    features_descriptors : dict
        Data (tuple structured as (number of feature dimensions (int), feature
        computation algorithm (function), list of macroscale strain loadings
        (list[numpy.ndarray (2d)]), strain magnitude factor (float)))
        associated to each feature (key, str). The macroscale strain loading
        is the infinitesimal strain tensor (infinitesimal strains) or the
        deformation gradient (finite strains).
    """
    features_descriptors = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = \
        mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fourth-order local elastic strain concentration tensor:
    # Set strain components order
    comp_order = comp_order_sym
    # Set number of feature dimensions
    n_feature_dim = len(comp_order)**2
    # Set feature computation algorithm
    feature_algorithm = StrainConcentrationTensor()
    # Set macroscale strain loadings required to compute feature
    mac_strains = []
    for i in range(len(comp_order)):
        # Get strain component and associated indexes
        comp = comp_order[i]
        so_idx = tuple([int(x) - 1 for x in list(comp_order[i])])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set macroscale strain magnitude factor
        if strain_formulation == 'finite':
            strain_magnitude_factor = 1.0e-6
        else:
            strain_magnitude_factor = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set orthogonal infinitesimal strain tensor (infinitesimal strains) or
        # material logarithmic strain tensor (finite strains) according with
        # Kelvin notation
        mac_strain = np.zeros((n_dim, n_dim))
        mac_strain[so_idx] = \
            strain_magnitude_factor*(1.0/mop.kelvin_factor(i, comp_order))*1.0
        if comp[0] != comp[1]:
            mac_strain[so_idx[::-1]] = mac_strain[so_idx]
        # Compute deformation gradient associated to the material logarithmic
        # strain tensor
        if strain_formulation == 'finite':
            mac_strain = def_gradient_from_log_strain(mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store macroscale strain loading
        mac_strains.append(mac_strain)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble to available clustering features
    features_descriptors['1'] = (n_feature_dim, feature_algorithm,
                                 mac_strains, strain_magnitude_factor)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spatial coordinates:
    # Set number of feature dimensions
    n_feature_dim = n_dim
    # Set feature computation algorithm
    feature_algorithm = SpatialCoordinates()
    # Set macroscale strain loadings required to compute feature
    mac_strains = []
    # Set macroscale strain magnitude factor
    strain_magnitude_factor = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble to available clustering features
    features_descriptors['2'] = (n_feature_dim, feature_algorithm,
                                 mac_strains, strain_magnitude_factor)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return features_descriptors
# =============================================================================
def def_gradient_from_log_strain(log_strain):
    """Get deformation gradient from material logarithmic strain tensor.

    Among the multitude of deformation gradients that may correspond to a given
    material logarithmic strain tensor, a particular choice stems from assuming
    that both tensors are coaxial, i.e., that the deformation gradient shares
    the eigenvectors with the material logarithmic strain tensor. In this case,
    the deformation gradient is symmetric and admits spectral decomposition as
    shown below.

    Given the spectral decomposition of the elastic material logarithmic strain
    tensor

    .. math::

       \\boldsymbol{E}^{e} = \\sum_{i=1}^{3} \\lambda_{i}^{\\boldsymbol{E}} \\,
       \\boldsymbol{l}_{i}^{\\boldsymbol{E}} \\otimes
       \\boldsymbol{l}_{i}^{\\boldsymbol{E}} \\, ,

    where :math:`\\boldsymbol{E}^{e}` is the elastic material logarithmic
    strain tensor, and :math:`\\lambda_{i}^{\\boldsymbol{E}}` and
    :math:`\\boldsymbol{l}_{i}^{\\boldsymbol{E}}`, :math:`i=1,2,3`, are the
    eigenvalues and eigenvectors of :math:`\\boldsymbol{E}^{e}`, the coaxial
    unique symmetric elastic deformation gradient comes

    .. math::

       \\boldsymbol{F}^{e} = \\sum_{i=1}^{3} \\lambda_{i}^{\\boldsymbol{F}} \\,
       \\boldsymbol{l}_{i}^{\\boldsymbol{E}} \\otimes
       \\boldsymbol{l}_{i}^{\\boldsymbol{E}} \\, ,

    where :math:`\\boldsymbol{F}^{e}` is the elastic deformation gradient and
    :math:`\\lambda_{i}^{\\boldsymbol{F}} =
    \\exp \\left[\\lambda_{i}^{\\boldsymbol{E}}\\right], \\, i=1,2,3`, are the
    corresponding eigenvalues.

    ----

    Parameters
    ----------
    log_strain : numpy.ndarray (2d)
        Material logarithmic strain tensor.

    Returns
    -------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    """
    # Perform spectral decomposition of material logarithmic strain tensor
    log_eigenvalues, log_eigenvectors, _, _ = \
        top.spectral_decomposition(log_strain)
    # Compute deformation gradient eigenvalues
    dg_eigenvalues = np.exp(log_eigenvalues)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute deformation gradient from spectral decomposition by assuming
    # coaxility with the material logarithmic strain tensor
    def_gradient = np.matmul(log_eigenvectors,
                             np.matmul(np.diag(dg_eigenvalues),
                                       np.transpose(log_eigenvectors)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return def_gradient
#
#                                                         Cluster analysis data
# =============================================================================
class ClusterAnalysisData:
    """Features data required to perform the RVE cluster analysis.

    Attributes
    ----------
    _features : set[int]
        Set of prescribed features identifiers (int).
    _features_idxs : dict
        Global data matrix's indexes (item, list) associated to each feature
        (key, str).
    _feature_loads_ids :  dict
        List of macroscale strain loadings identifiers (item, list[int])
        associated to each feature (key, str).
    _mac_strains : list[numpy.ndarray (2d)]
        List of macroscale strain loadings required to compute the clustering
        features. The macroscale strain loading is the infinitesimal strain
        tensor (infinitesimal strains) and the deformation gradient (finite
        strains).
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _global_data_matrix : numpy.ndarray (2d)
        Data matrix (numpy.ndarray of shape (n_voxels, n_features_dims))
        containing the required clustering features' data to perform all the
        prescribed RVE clusterings.

    Methods
    -------
    set_prescribed_features(self)
        Set prescribed clustering features.
    get_features(self)
        Get prescribed clustering features.
    set_feature_global_indexes(self)
        Set prescribed clustering features global data matrix indexes.
    set_clustering_mac_strains(self)
        Set macroscale strain loadings to compute clustering features.
    get_clustering_mac_strains(self)
        Set macroscale strain loadings to compute clustering features.
    set_global_data_matrix(self, rve_global_response)
        Compute global data matrix containing all clustering features.
    get_global_data_matrix(self)
        Get global data matrix containing all clustering features.
    """
    def __init__(self, strain_formulation, problem_type, rve_dims,
                 n_voxels_dims, base_clustering_scheme,
                 adaptive_clustering_scheme, feature_descriptors):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        features_descriptors : dict
            Data (tuple structured as (number of feature dimensions (int),
            feature computation algorithm (function), list of macroscale strain
            loadings (list[numpy.ndarray (2d)]),
            strain magnitude factor (float))) associated to each feature
            (key, str). The macroscale strain loading is the infinitesimal
            strain tensor (infinitesimal strains) or the deformation gradient
            (finite strains).
        n_voxels : int
            Total number of voxels of the regular grid (spatial discretization
            of the RVE).
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._rve_dims = rve_dims
        self._n_voxels_dims = n_voxels_dims
        self._base_clustering_scheme = base_clustering_scheme
        self._adaptive_clustering_scheme = adaptive_clustering_scheme
        self._feature_descriptors = feature_descriptors
        self._n_voxels = np.prod(n_voxels_dims)
        self._features = None
        self._n_features_dims = None
        self._features_idxs = None
        self._features_loads_ids = None
        self._mac_strains = None
        self._global_data_matrix = None
        # Get problem type parameters
        _, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(self._problem_type)
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
    # -------------------------------------------------------------------------
    def set_prescribed_features(self):
        """Set prescribed clustering features."""
        # Get material phases
        material_phases = self._base_clustering_scheme.keys()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering features
        self._features = []
        # Loop over material phases' prescribed clustering schemes and append
        # associated clustering features
        for mat_phase in material_phases:
            # Get material phase base clustering features
            for i in range(self._base_clustering_scheme[mat_phase].shape[0]):
                self._features += self._base_clustering_scheme[mat_phase][i, 1]
            # Get material phase adaptive clustering features
            if mat_phase in self._adaptive_clustering_scheme.keys():
                for i in range(
                        self._adaptive_clustering_scheme[mat_phase].shape[0]):
                    self._features += \
                        self._adaptive_clustering_scheme[mat_phase][i, 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get set of unique prescribed clustering features
        self._features = set(self._features)
    # -------------------------------------------------------------------------
    def get_features(self):
        """Get prescribed clustering features.

        Returns
        -------
        features : set[int]
            Set of prescribed features identifiers (int).
        """
        return copy.deepcopy(self._features)
    # -------------------------------------------------------------------------
    def set_feature_global_indexes(self):
        """Set prescribed clustering features global data matrix indexes.

        Assign a list of global data matrix indexes to each clustering feature
        according to the corresponding dimensionality. This list is essentialy
        a unitary-step slice, i.e., described by initial and ending delimitary
        indexes. The clustering scheme is also updated by assigning the global
        data matrix' indexes associated to each prescribed RVE clustering.

        Return
        ------
        features_idxs: dict
            Global data matrix indexes (item, list[range]) associated to each
            feature (key, str).
        """
        init = 0
        self._n_features_dims = 0
        self._features_idxs = {}
        # Loop over prescribed clustering features
        for feature in self._features:
            # Get feature dimensionality
            feature_dim = self._feature_descriptors[str(feature)][0]
            # Increment total number of features dimensions
            self._n_features_dims += feature_dim
            # Assign data matrix' indexes
            self._features_idxs[str(feature)] = \
                list(range(init, init + feature_dim))
            init += feature_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = self._base_clustering_scheme.keys()
        # Loop over material phases' prescribed clustering schemes and set
        # associated clustering features data matrix indexes
        for mat_phase in material_phases:
            # Loop over material phase base clustering scheme
            for i in range(self._base_clustering_scheme[mat_phase].shape[0]):
                indexes = []
                # Loop over prescribed clustering features
                for feature in self._base_clustering_scheme[mat_phase][i, 1]:
                    indexes += self._features_idxs[str(feature)]
                # Set clustering features data matrix' indexes
                self._base_clustering_scheme[mat_phase][i, 2] = \
                    copy.deepcopy(indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phase adaptive clustering scheme
            if mat_phase in self._adaptive_clustering_scheme.keys():
                for i in range(
                        self._adaptive_clustering_scheme[mat_phase].shape[0]):
                    indexes = []
                    # Loop over prescribed clustering features
                    for feature in \
                            self._adaptive_clustering_scheme[mat_phase][i, 1]:
                        indexes += self._features_idxs[str(feature)]
                    # Set clustering features data matrix' indexes
                    self._adaptive_clustering_scheme[mat_phase][i, 2] = \
                        copy.deepcopy(indexes)
    # -------------------------------------------------------------------------
    def set_clustering_mac_strains(self):
        """Set macroscale strain loadings to compute clustering features.

        List the required macroscale strain loadings to compute all the
        prescribed clustering features. The macroscale strain loading is the
        infinitesimal strain tensor (infinitesimal strains) or the deformation
        gradient (finite strains). In addition, assign the macroscale strain
        loadings identifiers (index in the previous list) associated to each
        clustering feature.
        """
        self._mac_strains = []
        self._features_loads_ids = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prescribed clustering features
        for feature in self._features:
            self._features_loads_ids[str(feature)] = []
            # Loop over clustering feature's required macroscale strain
            # loadings
            for mac_strain in self._feature_descriptors[str(feature)][2]:
                is_new_mac_strain = True
                # Loop over already prescribed macroscale strain loadings
                for i in range(len(self._mac_strains)):
                    array = self._mac_strains[i]
                    if np.allclose(mac_strain, array, rtol=1e-10, atol=1e-10):
                        # Append macroscale strain loading identifier to
                        # clustering feature
                        self._features_loads_ids[str(feature)].append(i)
                        is_new_mac_strain = False
                        break
                # Assemble new macroscale strain loading
                if is_new_mac_strain:
                    # Append macroscale strain loading identifier to clustering
                    # feature
                    self._features_loads_ids[str(feature)].append(
                        len(self._mac_strains))
                    # Append macroscale strain loading
                    self._mac_strains.append(mac_strain)
    # -------------------------------------------------------------------------
    def get_clustering_mac_strains(self):
        """Set macroscale strain loadings to compute clustering features.

        Returns
        -------
        mac_strains : list[numpy.ndarray (2d)]
            List of macroscale strain loadings required to compute the
            clustering features. The macroscale strain loading is the
            infinitesimal strain tensor (infinitesimal strains) and the
            deformation gradient (finite strains).
        """
        return copy.deepcopy(self._mac_strains)
    # -------------------------------------------------------------------------
    def set_global_data_matrix(self, rve_global_response):
        """Compute global data matrix containing all clustering features.

        Compute the data matrix required to perform all the RVE clusterings
        prescribed in the clustering scheme. This involves the computation of
        each clustering feature's data matrix (based on a RVE response
        database) and post assembly to the clustering global data matrix.

        ----

        Parameters
        ----------
        rve_global_response : numpy.ndarray (2d)
            RVE local elastic strain response for a given set of macroscale
            loadings, where each macroscale loading is associated with a set of
            independent strain components (numpy.ndarray of shape
            (n_voxels, n_mac_strains*n_strain_comps)). Each column is
            associated with a independent strain component of the infinitesimal
            strain tensor (infinitesimal strains) or material logarithmic
            strain tensor (finite strains).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_sym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering global data matrix
        self._global_data_matrix = np.zeros((self._n_voxels,
                                             self._n_features_dims))
        # Loop over prescribed clustering features
        for feature in self._features:
            # Get clustering feature macroscale strain loadings identifiers
            mac_loads_ids = self._features_loads_ids[str(feature)]
            # Get data from the RVE response database required to compute the
            # clustering feature (data associated to the response to one or
            # more macroscale strain loadings)
            rve_response = np.zeros((self._n_voxels, 0))
            for mac_load_id in mac_loads_ids:
                j_init = mac_load_id*len(comp_order)
                j_end = j_init + len(comp_order)
                rve_response = np.append(rve_response,
                                         rve_global_response[:, j_init:j_end],
                                         axis=1)
            # Get clustering feature's computation algorithm and compute
            # associated data matrix
            feature_algorithm = self._feature_descriptors[str(feature)][1]
            data_matrix = feature_algorithm.get_feature_data_matrix(
                self._strain_formulation, self._problem_type, self._rve_dims,
                self._n_voxels_dims, rve_response)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize data matrix according to strain magnitude factor
            if type(feature_algorithm) == StrainConcentrationTensor:
                # Get macroscale strain magnitude factor
                strain_magnitude_factor = \
                    self._feature_descriptors[str(feature)][3]
                # Normalize data matrix
                data_matrix = (1.0/strain_magnitude_factor)*data_matrix
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble clustering feature's data matrix to clustering global
            # data matrix
            j_init = self._features_idxs[str(feature)][0]
            j_end = self._features_idxs[str(feature)][-1] + 1
            self._global_data_matrix[:, j_init:j_end] = data_matrix
    # -------------------------------------------------------------------------
    def get_global_data_matrix(self):
        """Get global data matrix containing all clustering features.

        Returns
        -------
        global_data_matrix : numpy.ndarray (2d)
            Data matrix (numpy.ndarray of shape (n_voxels, n_features_dims))
            containing the required clustering features data to perform all the
            prescribed RVE clusterings.
        """
        return copy.deepcopy(self._global_data_matrix)
#
#                           Interface: Clustering feature computation algorithm
# =============================================================================
class FeatureAlgorithm(ABC):
    """Feature computation algorithm interface.

    Methods
    -------
    get_feature_data_matrix(self, strain_formulation, problem_type, rve_dims, \
                            n_voxels_dims, rve_response)
        *abstract*: Compute data matrix associated to clustering feature.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_feature_data_matrix(self, strain_formulation, problem_type,
                                rve_dims, n_voxels_dims, rve_response):
        """Compute data matrix associated to clustering feature.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        rve_response : numpy.ndarray (2d)
            RVE elastic response for one or more macroscale loadings
            (numpy.ndarray of shape (n_voxels, n_strain_comps)), where each
            macroscale loading is associated with a set of independent strain
            components.

        Returns
        -------
        data_matrix : ndarray of shape (n_voxels, n_feature_dim)
            Clustering features data matrix (numpy.ndarray of shape
            (n_voxels, n_feature_dim)).
        """
        pass
#
#                                    Clustering features computation algorithms
# =============================================================================
class StrainConcentrationTensor(FeatureAlgorithm):
    """Fourth-order elastic strain concentration tensor.

    The fourth-order elastic strain concentration tensor is defined in terms of
    the elastic infinitesimal strain tensor (infinitesimal strains) or the
    elastic material logarithmic strain tensor (finite strains) as shown below.
    Note that both strain tensors are symmetric and admit the reduced matricial
    form (Kelvin notation).

    * *Infinitesimal strains*: Fourth-order local elastic strain
      concentration tensor based on the elastic infinitesimal strain
      tensor.

      .. math::

         \\boldsymbol{\\varepsilon}_{\\mu}^{e}(\\boldsymbol{Y}) =
         \\boldsymbol{\\mathsf{H}}^{e}(\\boldsymbol{Y}):
         \\boldsymbol{\\varepsilon}^{e} (\\boldsymbol{X}) \\, , \\quad
         \\forall \\boldsymbol{Y} \\in \\Omega_{\\mu,\\,0} \\, ,

      where :math:`\\boldsymbol{\\mathsf{H}}^{e}` is the
      fourth-order local elastic strain concentration tensor,
      :math:`\\boldsymbol{\\varepsilon}_{\\mu}^{e}` is the
      microscale elastic infinitesimal strain tensor,
      :math:`\\boldsymbol{\\varepsilon}^{e}` is the
      macroscale elastic infinitesimal strain tensor,
      :math:`\\boldsymbol{Y}` is a point of the microscale reference
      configuration (:math:`\\Omega_{\\mu,\\,0}`), and
      :math:`\\boldsymbol{X}` is a point of the macroscale reference
      configuration (:math:`\\Omega_{0}`).

    * *Finite strains*: Fourth-order local elastic strain concentration
      tensor based on the elastic material logarithmic strain tensor.

      .. math::

         \\boldsymbol{E}_{\\mu}^{e}(\\boldsymbol{Y}) =
         \\boldsymbol{\\mathsf{H}}^{e}(\\boldsymbol{Y}):
         \\boldsymbol{E}^{e} (\\boldsymbol{X}) \\, , \\quad
         \\forall \\boldsymbol{Y} \\in \\Omega_{\\mu,\\,0} \\, ,

      where :math:`\\boldsymbol{\\mathsf{H}}^{e}` is the
      fourth-order local elastic strain concentration tensor,
      :math:`\\boldsymbol{E}_{\\mu}^{e}` is the
      microscale elastic material logarithmic strain tensor,
      :math:`\\boldsymbol{E}^{e}` is the
      macroscale elastic material logarithmic strain tensor,
      :math:`\\boldsymbol{Y}` is a point of the microscale reference
      configuration (:math:`\\Omega_{\\mu,\\,0}`), and
      :math:`\\boldsymbol{X}` is a point of the macroscale reference
      configuration (:math:`\\Omega_{0}`).

    Methods
    -------
    get_feature_data_matrix(self, strain_formulation, problem_type, rve_dims, \
                            n_voxels_dims, rve_response)
        Compute data matrix associated to clustering feature.
    """
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    def get_feature_data_matrix(self, strain_formulation, problem_type,
                                rve_dims, n_voxels_dims, rve_response):
        """Compute data matrix associated to clustering feature.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_dims : list[float]
            RVE size in each dimension.
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        rve_response : numpy.ndarray (2d)
            RVE elastic response for one or more macroscale loadings
            (numpy.ndarray of shape (n_voxels, n_strain_comps)), where each
            macroscale loading is associated with a set of independent strain
            components.

        Returns
        -------
        data_matrix : numpy.ndarray (2d)
            Clustering feature data matrix (numpy.ndarray of shape
            (n_voxels, n_feature_dim)).
        """
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering feature's data matrix
        data_matrix = np.zeros(rve_response.shape)
        # Initialize storage index
        idx = 0
        # Loop over macroscale loadings
        for i in range(len(comp_order_sym)):
            # Get Kelvin factor associated with macroscale strain loading
            # strain component
            kf_i = mop.kelvin_factor(i, comp_order_sym)
            # Loop over strain components
            for j in range(len(comp_order_sym)):
                # Get Kelvin factor associated with strain component
                kf_j = mop.kelvin_factor(j, comp_order_sym)
                # Assemble fourth-order elastic strain concentration tensor
                # component (accounting for the Kelvin notation coefficients)
                data_matrix[:, idx] = kf_j*rve_response[:, idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove Kelvin coefficient
                data_matrix[:, idx] = (1.0/kf_i)*(1.0/kf_j)*data_matrix[:, idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update storage index
                idx += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return data_matrix
# =============================================================================
class SpatialCoordinates(FeatureAlgorithm):
    """Spatial coordinates.

    Methods
    -------
    get_feature_data_matrix(self, strain_formulation, problem_type, rve_dims, \
                            n_voxels_dims, rve_response)
        Compute data matrix associated to clustering feature.
    """
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    def get_feature_data_matrix(self, strain_formulation, problem_type,
                                rve_dims, n_voxels_dims, rve_response):
        """Compute data matrix associated to clustering feature.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        rve_response : numpy.ndarray (2d)
            RVE elastic response for one or more macroscale loadings
            (numpy.ndarray of shape (n_voxels, n_strain_comps)), where each
            macroscale loading is associated with a set of independent strain
            components.

        Returns
        -------
        data_matrix : numpy.ndarray (2d)
            Clustering feature data matrix (numpy.ndarray of shape
            (n_voxels, n_feature_dim)).
        """
        # Get problem number of spatial dimensions
        n_dim, _, _ = mop.get_problem_type_parameters(problem_type)
        # Compute total number of voxels
        n_voxels = np.prod(n_voxels_dims)
        # Set sampling spatial periods
        sampling_periods = tuple([rve_dims[i]/n_voxels_dims[i]
                                  for i in range(n_dim)])
        # Set coordinate axes offset
        offsets = tuple([0.5*x for x in sampling_periods])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize voxels coordinates global data matrix
        data_matrix = np.zeros((n_voxels, n_dim))
        # Initialize voxel row index
        i_voxel = 0
        if n_dim == 2:
            # Loop over voxels
            for i in range(n_voxels_dims[0]):
                # Loop over voxels
                for j in range(n_voxels_dims[1]):
                    # Compute voxel coordinates
                    voxel_coord = (sampling_periods[0]*i + offsets[0],
                                   sampling_periods[1]*j + offsets[1])
                    # Store voxel coordinates in global data matrix
                    data_matrix[i_voxel, :] = np.array(voxel_coord)
                    # Update voxel row index
                    i_voxel += 1
        else:
            # Loop over voxels
            for i in range(n_voxels_dims[0]):
                # Loop over voxels
                for j in range(n_voxels_dims[1]):
                    # Loop over voxels
                    for k in range(n_voxels_dims[2]):
                        # Compute voxel coordinates
                        voxel_coord = (sampling_periods[0]*i + offsets[0],
                                       sampling_periods[1]*j + offsets[1],
                                       sampling_periods[2]*k + offsets[2])
                        # Store voxel coordinates in global data matrix
                        data_matrix[i_voxel, :] = np.array(voxel_coord)
                        # Update voxel row index
                        i_voxel += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return data_matrix
#
#                                     Interface: Data standardization algorithm
# =============================================================================
class Standardizer(ABC):
    """Data standardization algorithm interface.

    Methods
    -------
    get_standardized_data_matrix(self, data_matrix)
        *abstract*: Standardize provided data matrix.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def get_standardized_data_matrix(self, data_matrix):
        """Standardize provided data matrix.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix to be standardized (numpy.ndarray of shape
            (n_items, n_features)).

        Returns
        -------
        data_matrix : numpy.ndarray (2d)
            Transformed data matrix (numpy.ndarrayndarray of shape
            (n_items, n_features)).
        """
        pass
#
#                                               Data standardization algorithms
# =============================================================================
class MinMaxScaler(Standardizer):
    """Min-Max scaling algorithm (wrapper).

    Transform features by scaling each feature to a given min-max range.

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.
    MinMaxScaler>`_.

    Attributes
    ----------
    _feature_range : tuple[float], default=(0, 1)
        Desired range of transformed data (tuple(min, max)).

    Methods
    -------
    get_standardized_data_matrix(self, data_matrix)
        Standardize provided data matrix.
    """
    def __init__(self, feature_range=(0, 1)):
        """Standardization algorithm constructor.

        Parameters
        ----------
        feature_range : tuple[float], default=(0, 1)
            Desired range of transformed data (tuple(min, max)).
        """
        self._feature_range = feature_range
    # -------------------------------------------------------------------------
    def get_standardized_data_matrix(self, data_matrix):
        """Standardize provided data matrix.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix to be standardized (numpy.ndarray of shape
            (n_items, n_features)).

        Returns
        -------
        data_matrix : numpy.ndarray (2d)
            Transformed data matrix (numpy.ndarray of shape
            (n_items, n_features)).
        """
        # Instatiante standardizer
        standardizer = skpp.MinMaxScaler(feature_range=self._feature_range,
                                         copy=False)
        # Fit scaling parameters and transform data
        data_matrix = standardizer.fit_transform(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_matrix
# =============================================================================
class StandardScaler(Standardizer):
    """Standard scaling algorithm (wrapper).

    Transform features by removing the mean and scaling to unit variance
    (standard normal distribution).

    Documentation: see `here <https://scikit-learn.org/stable/modules/
    generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.
    StandardScaler>`_.

    Methods
    -------
    get_standardized_data_matrix(self, data_matrix)
        Standardize provided data matrix.
    """
    def __init__(self):
        """Standardization algorithm constructor."""
    # -------------------------------------------------------------------------
    def get_standardized_data_matrix(self, data_matrix):
        """Standardize provided data matrix.

        Parameters
        ----------
        data_matrix : numpy.ndarray (2d)
            Data matrix to be standardized (numpy.ndarray of shape
            (n_items, n_features)).

        Returns
        -------
        data_matrix : numpy.ndarray (2d)
            Transformed data matrix (numpy.ndarray of shape
            (n_items, n_features)).
        """
        # Instatiante standardizer
        standardizer = skpp.StandardScaler(with_mean=True, with_std=True,
                                           copy=False)
        # Fit scaling parameters and transform data
        data_matrix = standardizer.fit_transform(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_matrix
