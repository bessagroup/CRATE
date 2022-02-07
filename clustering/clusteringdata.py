#
# Cluster Analysis Features Data Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the computation of the physical-based features' data serving as a
# basis to perform the RVE clustering-based domain decomposition.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Jan 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2020 | Refactoring and OOP implementation.
# Bernardo P. Ferreira | Oct 2021 | Finite strains extension.
# Bernardo P. Ferreira | Feb 2022 | Updated fourth-order strain concentration tensors to
#                                 | properly account for the matricial form following the
#                                 | Kelvin notation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Defining abstract base classes
from abc import ABC, abstractmethod
# Data preprocessing tools
import sklearn.preprocessing as skpp
# Display messages
import ioput.info as info
# Tensorial operations
import tensor.tensoroperations as top
# Matricial operations
import tensor.matrixoperations as mop
# RVE response database
from clustering.rveelasticdatabase import RVEElasticDatabase
#
#                                                    Compute cluster analysis features' data
# ==========================================================================================
def set_clustering_data(strain_formulation, problem_type, rve_dims, n_voxels_dims,
                        regular_grid, material_phases, material_phases_properties,
                        dns_method, dns_method_data, standardization_method,
                        base_clustering_scheme, adaptive_clustering_scheme):
    '''Compute the physical-based data required to perform the RVE cluster analysis.

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
    dns_method : int
        DNS homogenization-based multi-scale method.
    dns_method_data : dict
        Parameters of DNS homogenization-based multi-scale method.
    standardization_method : int
        Identifier of global cluster analysis data standardization algorithm.
    base_clustering_scheme : dict
        Prescribed base clustering scheme (item, ndarray of shape (n_clusterings, 3)) for
        each material phase (key, str). Each row is associated with a unique clustering
        characterized by a clustering algorithm (col 1, int), a list of features
        (col 2, list of int) and a list of the features data matrix' indexes
        (col 3, list of int).
    adaptive_clustering_scheme : dict
        Prescribed adaptive clustering scheme (item, ndarray of shape (n_clusterings, 3))
        for each material phase (key, str). Each row is associated with a unique
        clustering characterized by a clustering algorithm (col 1, int), a list of
        features (col 2, list of int) and a list of the features data matrix' indexes
        (col 3, list of int).

    Returns
    -------
    clustering_data : ClusterAnalysisData
        Physical-based data required to perform the RVE cluster analyses.
    rve_elastic_database : RVEElasticDatabase
        RVE's local elastic response database.
    '''
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Setting cluster analysis\' features...')
    # Get available clustering features descriptors
    feature_descriptors = get_available_clustering_features(strain_formulation,
                                                            problem_type)
    # Instatiante cluster analysis data
    clustering_data = ClusterAnalysisData(strain_formulation, problem_type, n_voxels_dims,
                                          base_clustering_scheme,
                                          adaptive_clustering_scheme, feature_descriptors)
    # Set prescribed clustering features
    clustering_data.set_prescribed_features()
    # Set prescribed clustering features' clustering global data matrix' indexes
    clustering_data.set_feature_global_indexes()
    # Set required macroscale strain loadings to compute clustering features
    clustering_data.set_clustering_mac_strains()
    mac_strains = clustering_data.get_clustering_mac_strains()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing RVE local elastic strain response database...')
    # Instatiate RVE's local elastic response database
    rve_elastic_database = RVEElasticDatabase(strain_formulation, problem_type, rve_dims,
                                              n_voxels_dims, regular_grid, material_phases,
                                              material_phases_properties)
    # Compute RVE's elastic response database
    rve_elastic_database.compute_rve_response_database(dns_method, dns_method_data,
                                                       mac_strains, is_strain_sym=True)

    # Compute RVE's elastic effective tangent modulus if the elastic response database
    # contains a suitable set of orthogonal macroscale strain loadings
    if clustering_data.get_features() == {1}:
        # Get strain magnitude factor associated with orthogonal macroscale strain loadings
        strain_magnitude_factor = feature_descriptors['1'][3]
        # Compute RVE's elastic effective tangent modulus
        rve_elastic_database.compute_rve_elastic_tangent_modulus(
            strain_magnitude_factor=strain_magnitude_factor)
        # Estimate isotropic elastic constants from RVE's elastic effective tangent modulus
        rve_elastic_database.set_eff_isotropic_elastic_constants()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing cluster analysis global data matrix...')
    # Compute clustering global data matrix containing all clustering features
    clustering_data.set_global_data_matrix(rve_elastic_database.rve_global_response)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Standardizing cluster analysis global data matrix...')
    # Instantiate standardization algorithm
    if standardization_method == 1:
        standardizer = MinMaxScaler()
    elif standardization_method == 2:
        standardizer = StandardScaler()
    else:
        raise RuntimeError('Unknown standardization method.')
    # Standardize clustering global data matrix
    clustering_data._global_data_matrix = \
        standardizer.get_standardized_data_matrix(clustering_data._global_data_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return clustering_data, rve_elastic_database
#
#                                                              Available clustering features
# ==========================================================================================
def get_available_clustering_features(strain_formulation, problem_type):
    '''Get available clustering features.

    Available clustering features identifiers:
    1 - Fourth-order local elastic strain concentration tensor
        > Infinitesimal strains: Infinitesimal strain tensor;
        > Finite strains: Material logarithmic strain tensor;

    Parameters
    ----------
    strain_formulation: str, {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    n_dim: int
        Number of spatial dimensions.
    comp_order_sym: list
        Symmetric strain/stress components (str) order.
    comp_order_nsym: list
        Nonsymmetric strain/stress components (str) order.

    Returns
    -------
    features_descriptors : dict
        Data (tuple structured as (number of feature dimensions (int), feature
        computation algorithm (function), list of macroscale strain loadings (list of
        2darrays), strain magnitude factor (float))) associated to each feature
        (key, str). The macroscale strain loading is the infinitesimal strain tensor
        (infinitesimal strains) and the deformation gradient (finite strains).
    '''
    features_descriptors = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set macroscale strain magnitude factor
        if strain_formulation == 'finite':
            strain_magnitude_factor = 1.0e-6
        else:
            strain_magnitude_factor = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set orthogonal infinitesimal strain tensor (infinitesimal strains) or material
        # logarithmic strain tensor (finite strains) according with Kelvin notation
        mac_strain = np.zeros((n_dim, n_dim))
        mac_strain[so_idx] = \
            strain_magnitude_factor*(1.0/mop.kelvin_factor(i, comp_order))*1.0
        if comp[0] != comp[1]:
            mac_strain[so_idx[::-1]] = mac_strain[so_idx]
        # Compute deformation gradient associated to the material logarithmic strain tensor
        if strain_formulation == 'finite':
            mac_strain = def_gradient_from_log_strain(mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store macroscale strain loading
        mac_strains.append(mac_strain)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assemble to available clustering features
    features_descriptors['1'] = (n_feature_dim, feature_algorithm,
                                 mac_strains, strain_magnitude_factor)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return features_descriptors
# ------------------------------------------------------------------------------------------
def def_gradient_from_log_strain(log_strain):
    '''Get deformation gradient corresponding to material logarithmic strain tensor.

    Among the multitude of deformation gradients that may correspond to a given material
    logarithmic strain tensor, a particular choice stems from assuming that both tensors
    are coaxial, i.e., that the deformation gradient shares the eigenvectors with the
    material logarithmic strain tensor. In this case, the deformation gradient is symmetric
    and admits spectral decomposition.

    Parameters
    ----------
    log_strain : 2darray
        Material logarithmic strain tensor.

    Returns
    -------
    def_gradient : 2darray
        Deformation gradient.
    '''
    # Perform spectral decomposition of material logarithmic strain tensor
    log_eigenvalues, log_eigenvectors, _, _ = top.spectral_decomposition(log_strain)
    # Compute deformation gradient eigenvalues
    dg_eigenvalues = np.exp(log_eigenvalues)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute deformation gradient from spectral decomposition by assuming coaxility with
    # the material logarithmic strain tensor
    def_gradient = np.matmul(log_eigenvectors, np.matmul(np.diag(dg_eigenvalues),
                                                         np.transpose(log_eigenvectors)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return def_gradient
#
#                                                                      Cluster analysis data
# ==========================================================================================
class ClusterAnalysisData:
    '''Physical-based data required to perform the RVE cluster analyses.

    Attributes
    ----------
    _features : set
        Set of prescribed features identifiers (int).
    _features_idxs : dict
        Global data matrix's indexes (item, list) associated to each feature (key, str).
    _feature_loads_ids :  dict
        List of macroscale strain loadings identifiers (item, list of int) associated to
        each feature (key, str).
    _mac_strains : list
        List of macroscale strain loadings required to compute the clustering features.
        The macroscale strain loading is the infinitesimal strain tensor (infinitesimal
        strains) and the deformation gradient (finite strains).
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _global_data_matrix : ndarray of shape (n_voxels, n_features_dims)
        Data matrix containing the required clustering features' data to perform all the
        prescribed RVE clusterings.
    '''
    def __init__(self, strain_formulation, problem_type, n_voxels_dims,
                 base_clustering_scheme, adaptive_clustering_scheme, feature_descriptors):
        '''Clustering data constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        n_voxels_dims : list
            Number of voxels in each dimension of the regular grid (spatial discretization
            of the RVE).
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, ndarray of shape (n_clusterings, 3)) for
            each material phase (key, str). Each row is associated with a unique clustering
            characterized by a clustering algorithm (col 1, int), a list of features
            (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, ndarray of shape (n_clusterings, 3))
            for each material phase (key, str). Each row is associated with a unique
            clustering characterized by a clustering algorithm (col 1, int), a list of
            features (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        features_descriptors : dict
            Data (tuple structured as (number of feature dimensions (int), feature
            computation algorithm (function), list of macroscale strain loadings (list of
            2darrays), strain_magnitude_factor (float))) associated to each feature
            (key,str). The macroscale strain loading is the infinitesimal strain tensor
            (infinitesimal strains) and the deformation gradient (finite strains).
        n_voxels : int
            Total number of voxels of the regular grid (spatial discretization of the
            RVE).
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
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
    # --------------------------------------------------------------------------------------
    def set_prescribed_features(self):
        '''Set prescribed clustering features.'''
        # Get material phases
        material_phases = self._base_clustering_scheme.keys()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering features
        self._features = []
        # Loop over material phases' prescribed clustering schemes and append associated
        # clustering features
        for mat_phase in material_phases:
            # Get material phase base clustering features
            for i in range(self._base_clustering_scheme[mat_phase].shape[0]):
                self._features += self._base_clustering_scheme[mat_phase][i, 1]
            # Get material phase adaptive clustering features
            if mat_phase in self._adaptive_clustering_scheme.keys():
                for i in range(self._adaptive_clustering_scheme[mat_phase].shape[0]):
                    self._features += self._adaptive_clustering_scheme[mat_phase][i, 1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get set of unique prescribed clustering features
        self._features = set(self._features)
    # --------------------------------------------------------------------------------------
    def get_features(self):
        '''Get prescribed clustering features.

        Returns
        -------
        features : set
            Set of prescribed features identifiers (int).
        '''
        return copy.deepcopy(self._features)
    # --------------------------------------------------------------------------------------
    def set_feature_global_indexes(self):
        '''Set prescribed clustering features' clustering global data matrix' indexes.

        Assign a list of global data matrix' indexes to each clustering feature according
        to the corresponding dimensionality. This list is essentialy a unitary-step slice,
        i.e. described by initial and ending delimitary indexes. The clustering scheme
        is also updated by assigning the global data matrix' indexes associated to each
        prescribed RVE clustering.

        Return
        ------
        features_idxs: dict
            Global data matrix's indexes (item, list) associated to each feature (key, str).
        '''
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
            self._features_idxs[str(feature)] = list(range(init, init + feature_dim))
            init += feature_dim
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material phases
        material_phases = self._base_clustering_scheme.keys()
        # Loop over material phases' prescribed clustering schemes and set associated
        # clustering features data matrix' indexes
        for mat_phase in material_phases:
            # Loop over material phase base clustering scheme
            for i in range(self._base_clustering_scheme[mat_phase].shape[0]):
                indexes = []
                # Loop over prescribed clustering features
                for feature in self._base_clustering_scheme[mat_phase][i, 1]:
                    indexes += self._features_idxs[str(feature)]
                # Set clustering features data matrix' indexes
                self._base_clustering_scheme[mat_phase][i, 2] = copy.deepcopy(indexes)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phase adaptive clustering scheme
            if mat_phase in self._adaptive_clustering_scheme.keys():
                for i in range(self._adaptive_clustering_scheme[mat_phase].shape[0]):
                    indexes = []
                    # Loop over prescribed clustering features
                    for feature in self._adaptive_clustering_scheme[mat_phase][i, 1]:
                        indexes += self._features_idxs[str(feature)]
                    # Set clustering features data matrix' indexes
                    self._adaptive_clustering_scheme[mat_phase][i, 2] = \
                        copy.deepcopy(indexes)
    # --------------------------------------------------------------------------------------
    def set_clustering_mac_strains(self):
        '''Set macroscale strain loadings required to compute clustering features.

        List the required macroscale strain loadings to compute all the prescribed
        clustering features. The macroscale strain loading is the infinitesimal strain
        tensor (infinitesimal strains) and the deformation gradient (finite strains).
        In addition, assign the macroscale strain loadings identifiers (index in the
        previous list) associated to each clustering feature.
        '''
        self._mac_strains = []
        self._features_loads_ids = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over prescribed clustering features
        for feature in self._features:
            self._features_loads_ids[str(feature)] = []
            # Loop over clustering feature's required macroscale strain loadings
            for mac_strain in self._feature_descriptors[str(feature)][2]:
                is_new_mac_strain = True
                # Loop over already prescribed macroscale strain loadings
                for i in range(len(self._mac_strains)):
                    array = self._mac_strains[i]
                    if np.allclose(mac_strain, array, rtol=1e-10, atol=1e-10):
                        # Append macroscale strain loading identifier to clustering feature
                        self._features_loads_ids[str(feature)].append(i)
                        is_new_mac_strain = False
                        break
                # Assemble new macroscale strain loading
                if is_new_mac_strain:
                    # Append macroscale strain loading identifier to clustering feature
                    self._features_loads_ids[str(feature)].append(len(self._mac_strains))
                    # Append macroscale strain loading
                    self._mac_strains.append(mac_strain)
    # --------------------------------------------------------------------------------------
    def get_clustering_mac_strains(self):
        '''Set macroscale strain loadings required to compute clustering features.

        Returns
        -------
        mac_strains : list
            List of macroscale strain loadings required to compute the clustering features.
            The macroscale strain loading is the infinitesimal strain tensor (infinitesimal
            strains) and the deformation gradient (finite strains).
        '''
        return copy.deepcopy(self._mac_strains)
    # --------------------------------------------------------------------------------------
    def set_global_data_matrix(self, rve_global_response):
        '''Compute clustering global data matrix containing all clustering features.

        Compute the data matrix required to perform all the RVE clusterings prescribed in
        the clustering scheme. This involves the computation of each clustering feature's
        data matrix (based on a RVE response database) and post assembly to the clustering
        global data matrix.

        Parameters
        ----------
        rve_global_response : ndarray of shape (n_voxels, n_mac_strains*n_strain_comps)
            RVE local elastic strain response for a given set of macroscale loadings, where
            each macroscale loading is associated with a set of independent strain
            components. Each column is associated with a independent strain component of the
            infinitesimal strain tensor (infinitesimal strains) or material logarithmic
            strain tensor (finite strains).
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_sym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering global data matrix
        self._global_data_matrix = np.zeros((self._n_voxels, self._n_features_dims))
        # Loop over prescribed clustering features
        for feature in self._features:
            # Get clustering feature macroscale strain loadings identifiers
            mac_loads_ids = self._features_loads_ids[str(feature)]
            # Get data from the RVE response database required to compute the clustering
            # feature (data associated to the response to one or more macroscale strain
            # loadings)
            rve_response = np.zeros((self._n_voxels, 0))
            for mac_load_id in mac_loads_ids:
                j_init = mac_load_id*len(comp_order)
                j_end = j_init + len(comp_order)
                rve_response = np.append(rve_response, rve_global_response[:, j_init:j_end],
                                         axis=1)
            # Get clustering feature's computation algorithm and compute associated data
            # matrix
            feature_algorithm = self._feature_descriptors[str(feature)][1]
            data_matrix = \
                feature_algorithm.get_feature_data_matrix(self._strain_formulation,
                                                          self._problem_type, rve_response)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Normalize data matrix according to strain magnitude factor
            if type(feature_algorithm) == StrainConcentrationTensor:
                # Get macroscale strain magnitude factor
                strain_magnitude_factor = self._feature_descriptors[str(feature)][3]
                # Normalize data matrix
                data_matrix = (1.0/strain_magnitude_factor)*data_matrix
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble clustering feature's data matrix to clustering global data matrix
            j_init = self._features_idxs[str(feature)][0]
            j_end = self._features_idxs[str(feature)][-1] + 1
            self._global_data_matrix[:, j_init:j_end] = data_matrix
    # --------------------------------------------------------------------------------------
    def get_global_data_matrix(self):
        '''Get clustering global data matrix containing all clustering features.

        Returns
        -------
        global_data_matrix : ndarray of shape (n_voxels, n_features_dims)
            Data matrix containing the required clustering features' data to perform all the
            prescribed RVE clusterings.
        '''
        return copy.deepcopy(self._global_data_matrix)
#
#                                                             Clustering features algorithms
# ==========================================================================================
class FeatureAlgorithm(ABC):
    '''Feature computation algorithm interface.'''
    @abstractmethod
    def __init__(self):
        '''Feature computation algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_feature_data_matrix(self, strain_formulation, problem_type, rve_response):
        '''Compute data matrix associated to clustering feature.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        rve_response : ndarray of shape (n_voxels, n_strain_comps)
            RVE elastic response for one or more macroscale loadings, where each macroscale
            loading is associated with a set of independent strain components.

        Returns
        -------
        data_matrix : ndarray of shape (n_voxels, n_feature_dim)
            Clustering feature's data matrix.
        '''
        pass
# ------------------------------------------------------------------------------------------
class StrainConcentrationTensor(FeatureAlgorithm):
    '''Fourth-order elastic strain concentration tensor.

    The fourth-order elastic strain concentration tensor is defined in terms of the
    infinitesimal strain tensor (infinitesimal strains) and the material logarithmic strain
    tensor (finite strains). Note that both strain tensors are symmetric and admit the
    reduced matricial form (Kelvin notation).
    '''
    def __init__(self):
        '''Feature computation algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    def get_feature_data_matrix(self, strain_formulation, problem_type, rve_response):
        '''Compute data matrix associated to clustering feature.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        rve_response : ndarray of shape (n_voxels, n_strain_comps)
            RVE elastic response for one or more macroscale loadings, where each macroscale
            loading is associated with a set of independent strain components.

        Returns
        -------
        data_matrix : ndarray of shape (n_voxels, n_feature_dim)
            Clustering feature's data matrix.
        '''
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize clustering feature's data matrix
        data_matrix = np.zeros(rve_response.shape)
        # Initialize storage index
        idx = 0
        # Loop over macroscale loadings
        for i in range(len(comp_order_sym)):
            # Get Kelvin factor associated with macroscale strain loading strain component
            kf_i = mop.kelvin_factor(i, comp_order_sym)
            # Loop over strain components
            for j in range(len(comp_order_sym)):
                # Get Kelvin factor associated with strain component
                kf_j = mop.kelvin_factor(j, comp_order_sym)
                # Assemble fourth-order elastic strain concentration tensor component
                # (accounting for the Kelvin notation coefficients)
                data_matrix[:, idx] = kf_j*rve_response[:, idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Remove Kelvin coefficient
                data_matrix[:, idx] = (1.0/kf_i)*(1.0/kf_j)*data_matrix[:, idx]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update storage index
                idx += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return data_matrix
#
#                                                            Data standardization algorithms
# ==========================================================================================
class Standardizer(ABC):
    '''Data standardization algorithm interface.'''
    @abstractmethod
    def __init__(self):
        '''Standardization algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_standardized_data_matrix(self, data_matrix):
        '''Standardize provided data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_items, n_features)
            Data matrix to be standardized.

        Returns
        -------
        data_matrix : ndarray of shape (n_items, n_features)
            Transformed data matrix.
        '''
        pass
# ------------------------------------------------------------------------------------------
class MinMaxScaler(Standardizer):
    '''Transform features by scaling each feature to a given min-max range.

    Attributes
    ----------
    _feature_range : tuple(min, max), default=(0, 1)
        Desired range of transformed data.

    Notes
    -----
    The Min-Max scaling algorithm is taken from scikit-learn (https://scikit-learn.org).
    Further information can be found in there.
    '''
    def __init__(self, feature_range=(0, 1)):
        '''Standardization algorithm constructor.'''
        self._feature_range = feature_range
    # --------------------------------------------------------------------------------------
    def get_standardized_data_matrix(self, data_matrix):
        '''Standardize provided data matrix.'''
        # Instatiante standardizer
        standardizer = skpp.MinMaxScaler(feature_range=self._feature_range, copy=False)
        # Fit scaling parameters and transform data
        data_matrix = standardizer.fit_transform(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_matrix
# ------------------------------------------------------------------------------------------
class StandardScaler(Standardizer):
    '''Transform features by removing the mean and scaling to unit variance (standard
    normal distribution).

    Notes
    -----
    The Standard scaling algorithm is taken from scikit-learn (https://scikit-learn.org).
    Further information can be found in there.
    '''
    def __init__(self):
        '''Standardization algorithm constructor.'''
    # --------------------------------------------------------------------------------------
    def get_standardized_data_matrix(self, data_matrix):
        '''Standardize provided data matrix.'''
        # Instatiante standardizer
        standardizer = skpp.StandardScaler(with_mean=True, with_std=True, copy=False)
        # Fit scaling parameters and transform data
        data_matrix = standardizer.fit_transform(data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return data_matrix
