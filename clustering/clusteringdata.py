#
# Cluster Analysis Features Data Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the computation of the physical-based features' data serving as a
# basis to perform the RVE clustering-based domain decomposition.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# Bernardo P. Ferreira | October 2020 | Refactoring and OOP implementation.
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
# RVE response database
from clustering.rveelasticdatabase import RVEElasticDatabase
#
#                                                    Compute cluster analysis features' data
# ==========================================================================================
def set_clustering_data(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict):
    '''Compute the physical-based data required to perform the RVE cluster analyses.'''
    # Get problem data
    strain_formulation = problem_dict['strain_formulation']
    n_dim = problem_dict['n_dim']
    comp_order_sym = problem_dict['comp_order_sym']
    comp_order_nsym = problem_dict['comp_order_nsym']
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Get clustering data
    clustering_solution_method = clst_dict['clustering_solution_method']
    standardization_method = clst_dict['standardization_method']
    base_clustering_scheme = clst_dict['base_clustering_scheme']
    adaptive_clustering_scheme = clst_dict['adaptive_clustering_scheme']
    # Compute total number of voxels
    n_voxels = np.prod(n_voxels_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Setting cluster analysis\' features...')
    # Get available clustering features descriptors
    feature_descriptors = get_available_clustering_features(strain_formulation, n_dim,
                                                            comp_order_sym, comp_order_nsym)
    # Instatiante cluster analysis data
    clustering_data = ClusterAnalysisData(base_clustering_scheme,
                                          adaptive_clustering_scheme, feature_descriptors,
                                          strain_formulation, comp_order_sym,
                                          comp_order_nsym, n_voxels)
    # Set prescribed clustering features
    clustering_data.set_prescribed_features()
    # Set prescribed clustering features' clustering global data matrix' indexes
    clustering_data.set_feature_global_indexes()
    # Get required macroscale strain loadings to compute clustering features
    mac_strains = clustering_data.get_clustering_mac_strains()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    info.displayinfo('5', 'Computing RVE local elastic strain response database...')
    # Instatiate RVE's local elastic response database
    rve_elastic_database = RVEElasticDatabase(clustering_solution_method, mac_strains,
                                              problem_dict, dirs_dict, rg_dict, mat_dict,
                                              clst_dict)
    # Compute RVE's elastic response database
    rve_elastic_database.set_RVE_response_database()
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
    clustering_data.global_data_matrix = \
        standardizer.get_standardized_data_matrix(clustering_data.global_data_matrix)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store clustering global data matrix
    clst_dict['clst_quantities'] = clustering_data.global_data_matrix
#
#                                                              Available clustering features
# ==========================================================================================
def get_available_clustering_features(strain_formulation, n_dim, comp_order_sym,
                                      comp_order_nsym):
    '''Get available clustering features in CRATE.

    Available clustering features identifiers:
    1 - Fourth-order local elastic strain concentration tensor

    Parameters
    ----------
    strain_formulation: int
        Strain formulation: (1) infinitesimal strains, (2) finite strains.
    n_dim: int
        Number of spatial dimensions.
    comp_order_sym: list
        Symmetric strain/stress components (str) order.
    comp_order_nsym: list
        Nonsymmetric strain/stress components (str) order.

    Returns
    -------
    features_descriptors: dict
        Available clustering features identifiers (key, int) and descriptors (tuple
        structured as (number of feature dimensions (int), list of macroscale strain
        loadings (list of ndarrays), feature computation algorithm.
    '''
    features_descriptors = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if strain_formulation == 1:
        comp_order = comp_order_sym
    else:
        comp_order = comp_order_nsym
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fourth-order local elastic strain concentration tensor:
    # Set number of feature dimensions
    n_feature_dim = len(comp_order)**2
    # Set macroscale strain loadings required to compute feature
    mac_strains = []
    for i in range(len(comp_order)):
        comp_i = comp_order[i]
        so_idx = tuple([int(x) - 1 for x in list(comp_order[i])])
        # Set macroscopic strain loading
        mac_strain = np.zeros((n_dim, n_dim))
        mac_strain[so_idx] = 1.0
        if strain_formulation == 1 and comp_i[0] != comp_i[1]:
            mac_strain[so_idx[::-1]] = 1.0
        mac_strains.append(mac_strain)
    # Set feature computation algorithm
    feature_algorithm = StrainConcentrationTensor()
    # Assemble to available clustering features
    features_descriptors['1'] = (n_feature_dim, mac_strains, feature_algorithm)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return features_descriptors
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
        List of macroscale loadings identifiers (item, list of int) associated to each
        feature (key, str).
    global_data_matrix : ndarray of shape (n_voxels, n_features_dims)
        Data matrix containing the required clustering features' data to perform all the
        prescribed RVE clusterings.
    '''
    def __init__(self, base_clustering_scheme, adaptive_clustering_scheme,
                 feature_descriptors, strain_formulation, comp_order_sym, comp_order_nsym,
                 n_voxels):
        '''Clustering data constructor.

        Parameters
        ----------
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
            Available clustering features identifiers (key, int) and descriptors (tuple
            structured as (number of feature dimensions (int), list of macroscale strain
            loadings (list of ndarrays), feature computation algorithm (function)).
        strain_formulation : int
            Strain formulation: (1) infinitesimal strains, (2) finite strains.
        comp_order_sym : list
            Symmetric strain/stress components (str) order.
        comp_order_nsym : list
            Nonsymmetric strain/stress components (str) order.
        n_voxels : int
            Total number of voxels of the regular grid (spatial discretization of the
            RVE).
        '''
        self._base_clustering_scheme = base_clustering_scheme
        self._adaptive_clustering_scheme = adaptive_clustering_scheme
        self._feature_descriptors = feature_descriptors
        if strain_formulation == 1:
            self._comp_order = comp_order_sym
        else:
            self._comp_order = comp_order_nsym
        self._n_voxels = n_voxels
        self._features = None
        self._n_features_dims = None
        self._features_idxs = None
        self._features_loads_ids = None
        self.global_data_matrix = None
    # --------------------------------------------------------------------------------------
    def set_prescribed_features(self):
        '''Set prescribed clustering features.'''
        # Get material phases
        material_phases = self._base_clustering_scheme.keys()
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
        # Get set of unique prescribed clustering features
        self._features = set(self._features)
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Validation output:
        # print('\nbase_clustering_scheme: ')
        # print(self._base_clustering_scheme)
        # print('\nadaptive_clustering_scheme: ')
        # print(self._adaptive_clustering_scheme)
    # --------------------------------------------------------------------------------------
    def get_clustering_mac_strains(self):
        '''Get required macroscale strain loadings to compute clustering features.

        List the required macroscale strain loadings (second-order macroscale strain
        tensor) to compute all the prescribed clustering features. In addition, assign the
        macroscale strain loadings identifiers (index in the previous list) associated
        to each clustering feature.

        Returns
        -------
        mac_strains : list
            List of macroscale strain loadings (ndarray, second-order strain tensor)
            required to compute all the prescribed clustering features.
        '''
        mac_strains = []
        self._features_loads_ids = {}
        # Loop over prescribed clustering features
        for feature in self._features:
            self._features_loads_ids[str(feature)] = []
            # Loop over clustering feature's required macroscale strain loadings
            for mac_strain in self._feature_descriptors[str(feature)][1]:
                is_new_mac_strain = True
                # Loop over already prescribed macroscale strain loadings
                for i in range(len(mac_strains)):
                    array = mac_strains[i]
                    if np.allclose(mac_strain, array, atol=1e-10):
                        # Append macroscale strain loading identifier to clustering feature
                        self._features_loads_ids[str(feature)].append(i)
                        is_new_mac_strain = False
                        break
                # Assemble new macroscale strain loading
                if is_new_mac_strain:
                    # Append macroscale strain loading identifier to clustering feature
                    self._features_loads_ids[str(feature)].append(len(mac_strains))
                    # Append macroscale strain loading
                    mac_strains.append(mac_strain)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mac_strains
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
            RVE elastic response for the required set of macroscale loadings, where each
            macroscale loading is associated with a set of independent strain components.
        '''
        # Initialize clustering global data matrix
        self.global_data_matrix = np.zeros((self._n_voxels, self._n_features_dims))
        # Loop over prescribed clustering features
        for feature in self._features:
            # Get clustering feature macroscale strain loadings identifiers
            mac_loads_ids = self._features_loads_ids[str(feature)]
            # Get data from the RVE response database required to compute the clustering
            # feature (data associated to the response to one or more macroscale strain
            # loadings)
            rve_response = np.zeros((self._n_voxels, 0))
            for mac_load_id in mac_loads_ids:
                j_init = mac_load_id*len(self._comp_order)
                j_end = j_init + len(self._comp_order)
                rve_response = np.append(rve_response, rve_global_response[:, j_init:j_end],
                                         axis=1)
            # Get clustering feature's computation algorithm and compute associated data
            # matrix
            feature_algorithm = self._feature_descriptors[str(feature)][2]
            data_matrix = feature_algorithm.get_feature_data_matrix(rve_response)
            # Assemble clustering feature's data matrix to clustering global data matrix
            j_init = self._features_idxs[str(feature)][0]
            j_end = self._features_idxs[str(feature)][-1] + 1
            self.global_data_matrix[:, j_init:j_end] = data_matrix
#
#                                                             Clustering features algorithms
# ==========================================================================================
class FeatureAlgorithm(ABC):
    '''Feature computation algorithm interface.'''

    @abstractmethod
    def get_feature_data_matrix(self, rve_response):
        '''Compute data matrix associated to a given clustering feature.

        Parameters
        ----------
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
    '''Fourth-order elastic strain concentration tensor.'''
    def get_feature_data_matrix(self, rve_response):
        data_matrix = copy.deepcopy(rve_response)
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
