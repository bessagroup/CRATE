#
# CRVE Adaptivity Criterion Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the CRVE clustering adaptivity criterion.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | May 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Defining abstract base classes
from abc import ABC, abstractmethod
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                       CRVE clustering adaptivity criterion
# ==========================================================================================
class AdaptivityCriterion(ABC):
    '''Clustering adaptivity criterion interface.'''
    @abstractmethod
    def __init__(self):
        '''Clustering adaptivity criterion constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @staticmethod
    @abstractmethod
    def get_parameters():
        '''Get clustering adaptivity criterion mandatory and optional parameters.

        Besides returning the mandatory and optional adaptivity criterion parameters, this
        method establishes the default values for the optional parameters.

        Returns
        -------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def get_target_clusters(self):
        '''Get clustering adaptivity target clusters ans associated data.

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters required for the adaptive procedures.
        '''
        pass
#
#                                            Adaptivity criterion: Adaptive cluster grouping
# ==========================================================================================
class AdaptiveClusterGrouping(AdaptivityCriterion):
    '''Adaptive cluster grouping criterion.

    Class that provides all the required attributes and methods associated with the
    adaptive cluster grouping CRVE clustering adaptivity criterion.

    Attributes
    ----------
    _adapt_groups : dict
        Store the cluster labels (item, list of int) associated to each adaptive cluster
        group (key, int).
    _groups_adapt_level : dict
        Store adaptive level (item, int) of each adaptive cluster group (key, int).
    _target_groups_ids : list
        Target adaptive cluster groups (item, list of int) whose clusters are to adapted.
    '''
    def __init__(self, adapt_mat_phase, phase_clusters, adapt_trigger_ratio=None,
                 adapt_split_threshold=None, adapt_max_level=None):
        '''Adaptive cluster grouping criterion constructor.

        Parameters
        ----------
        adapt_mat_phase : str
            Adaptive material phase label.
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        adapt_trigger_ratio : float, default=None
            Threshold associated to the adaptivity trigger condition, defining the value of
            the relative ratio (max - avg)/avg above which the adaptive cluster group is set
            to be adapted, where max and avg are the maximum and average adaptivity feature
            values in the adaptive cluster group, respectively.
        adapt_split_threshold : float, default=None
            Threshold associated to the adaptivity selection criterion, defining the split
            boundary of each adaptive cluster group according to the associated position in
            terms of the adaptivity value range within the group. For instance, a
            adapt_split_threshold=0.2 means that the split boundary divides the clusters
            whose adaptivity feature value is above min + 0.8*(max - min) (top 20% of the
            value range) from the remaining clusters, resulting two child adaptive cluster
            groups.
        adapt_max_level : int, default=None
            Maximum adaptive cluster group adaptive level.

        '''
        # Set initial adaptive clusters group (labeled as group '0') and initialize
        # associated adaptive level
        self._adapt_groups = {}
        self._groups_adapt_level = {}
        self._adapt_groups['0'] = copy.deepcopy(phase_clusters[adapt_mat_phase])
        self._groups_adapt_level['0'] = 0
        # Get optional parameters
        optional_parameters = type(self).get_parameters()
        # Get adaptivity trigger ratio
        if adapt_trigger_ratio is None:
            self._adapt_trigger_ratio = optional_parameters['adapt_trigger_ratio']
        else:
            self._adapt_trigger_ratio = adapt_trigger_ratio
        # Get adaptivity split threshold
        if adapt_split_threshold is None:
            self._adapt_split_threshold = optional_parameters['adapt_split_threshold']
        else:
            self._adapt_split_threshold = adapt_split_threshold
        # Get maximum adaptive cluster group adaptive level
        if adapt_max_level is None:
            self._adapt_max_level = optional_parameters['adapt_max_level']
        else:
            self._adapt_max_level = adapt_max_level
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_parameters():
        '''Get clustering adaptivity criterion mandatory and optional parameters.

        Besides returning the mandatory and optional adaptivity criterion parameters, this
        method establishes the default values for the optional parameters.

        Returns
        -------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        # Set mandatory adaptivity criterion parameters
        mandatory_parameters = {}
        # Set optional adaptivity criterion parameters and associated default values
        optional_parameters = {'adapt_trigger_ratio': 0.1,
                               'adapt_split_threshold': 0.1,
                               'adapt_max_level': 15}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [mandatory_parameters, optional_parameters]
    # --------------------------------------------------------------------------------------
    def get_target_clusters(self, adapt_data_matrix):
        '''Get clustering adaptivity target clusters.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (adapt_phase_n_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the adaptive
            material phase, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters required for the adaptive procedures.
        '''
        # Initialize target clusters list
        target_clusters = []
        # Initialize target clusters data
        target_clusters_data = {}
        # Initialize target cluster groups
        self._target_groups_ids = []
        # Get adaptive material phase cluster groups
        adapt_groups_ids = list(self._adapt_groups.keys())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive cluster groups
        for group_id in adapt_groups_ids:
            # Check if adaptive cluster group can be further adapted and, if not, skip
            # to the next one
            if self._groups_adapt_level[group_id] >= self._adapt_max_level:
                continue
            # Get adaptive cluster group clusters and current adaptive level
            adapt_group = self._adapt_groups[group_id]
            base_adapt_level = self._groups_adapt_level[group_id]
            # Get adaptivity data matrix row indexes associated with the adaptive
            # cluster group clusters
            adapt_group_idxs = np.where(np.in1d(adapt_data_matrix[:, 0], adapt_group))[0]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check adaptive cluster group adaptivity trigger condition
            is_trigger = self._adaptivity_trigger_condition(
                adapt_data_matrix[adapt_group_idxs, :])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get adaptive cluster group target clusters and assemble them to the global
            # target clusters list
            if is_trigger:
                # Get target clusters according to adaptivity selection criterion
                group_target_clusters = self._adaptivity_selection_criterion(
                    adapt_data_matrix[adapt_group_idxs, :])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get maximum adaptive cluster group id
                max_group_id = np.max([int(x) for x in self._adapt_groups.keys()])
                # Set child adaptive cluster group to be adapted (set id, set clusters
                # and increment adaptive level relative to parent cluster group)
                child_group_1_id = str(max_group_id + 1)
                self._adapt_groups[child_group_1_id] = group_target_clusters
                self._groups_adapt_level[child_group_1_id] = base_adapt_level + 1
                self._target_groups_ids.append(child_group_1_id)
                # Set child adaptive cluster group to be left unchanged (set id, set
                # clusters and set adaptive level equal to the parent cluster group)
                child_group_2_id = str(max_group_id + 2)
                self._adapt_groups[child_group_2_id] = list(set(adapt_group) - \
                                                            set(group_target_clusters))
                self._groups_adapt_level[child_group_2_id] = base_adapt_level
                # Remove parent adaptive cluster group
                self._adapt_groups.pop(group_id)
                self._groups_adapt_level.pop(group_id)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Assemble target clusters
                target_clusters += group_target_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for cluster in target_clusters:
            # Build target cluster data dictionary
            target_clusters_data[str(cluster)] = {'is_dynamic_split_factor': False}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return target_clusters, target_clusters_data
    # --------------------------------------------------------------------------------------
    def update_group_clusters(self, adaptive_clustering_map):
        '''Update adaptive cluster groups after adaptive procedures.

        Parameters
        ----------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the adaptive
            procedures over each target cluster (key, str).
        '''
        # Loop over adaptive cluster groups
        for group_id in self._target_groups_ids:
            # Get parent adaptive cluster group clusters
            old_clusters = self._adapt_groups[group_id].copy()
            # Loop over adaptive cluster group clusters
            for old_cluster in old_clusters:
                # If cluster has been adapted, then update adaptive cluster group
                if str(old_cluster) in adaptive_clustering_map.keys():
                    # Remove parent cluster from group
                    self._adapt_groups[group_id].remove(old_cluster)
                    # Add child clusters to group
                    new_clusters = adaptive_clustering_map[str(old_cluster)]
                    self._adapt_groups[group_id] += new_clusters
    # --------------------------------------------------------------------------------------
    def get_adapt_groups(self):
        '''Get adaptive cluster groups.

        Returns
        -------
        adapt_groups : dict
            Store the cluster labels (item, list of int) associated to each adaptive cluster
            group (key, int).
        '''
        return copy.deepcopy(self._adapt_groups)
    # --------------------------------------------------------------------------------------
    def get_groups_adapt_level(self):
        '''Get adaptive cluster groups adaptive level.

        Returns
        -------
        groups_adapt_level : dict
            Store adaptive level (item, int) of each adaptive cluster group (key, int).
        '''
        return copy.deepcopy(self._groups_adapt_level)
    # --------------------------------------------------------------------------------------
    def _adaptivity_trigger_condition(self, adapt_data_matrix):
        '''Evaluate adaptive cluster group adaptivity trigger condition.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (adapt_group_n_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the adaptive
            cluster group, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].

        Returns
        -------
        bool
            True if clustering adaptivity is triggered in the adaptive cluster group, False
            otherwise.
        '''
        # Compute average and maximum values of adaptivity feature
        a = np.average(adapt_data_matrix[:, 1])
        b = np.max(adapt_data_matrix[:, 1])
        # Compute adaptivity ratio
        if abs(a) < 1e-10:
            adapt_ratio = abs(b - a)
        else:
            adapt_ratio = abs(b - a)/abs(a)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check adaptivity trigger condition
        if adapt_ratio > self._adapt_trigger_ratio:
            return True
        else:
            return False
   # ---------------------------------------------------------------------------------------
    def _adaptivity_selection_criterion(self, adapt_data_matrix):
        '''Select target clusters from adaptive cluster group.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (adapt_group_n_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the adaptive
            cluster group, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        '''
        # Check threshold
        if not ioutil.is_between(self._adapt_split_threshold, lower_bound=0, upper_bound=1):
            raise RuntimeError('Clustering adaptivity selection criterion\'s threshold ' +
                               'must be between 0 and 1 (included).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get split boundary adaptivity feature value
        adapt_boundary = min(adapt_data_matrix[:, 1]) + \
            (1.0 - self._adapt_split_threshold)*(max(adapt_data_matrix[:, 1]) -
                                                 min(adapt_data_matrix[:, 1]))
        # Get indexes of clusters whose adaptivity feature value is greater or equal than
        # the split boundary value
        idxs = adapt_data_matrix[:, 1] >= adapt_boundary
        # Get target clusters
        target_clusters = [int(x) for x in adapt_data_matrix[idxs, 0]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return target_clusters
#
#                                              Adaptivity criterion: Spatial discontinuities
# ==========================================================================================
class SpatialDiscontinuities(AdaptivityCriterion):
    '''Spatial discontinuities criterion.

    Class that provides all the required attributes and methods associated with the
    spatial discontinuities CRVE clustering adaptivity criterion.

    Attributes
    ----------
    _clusters_adapt_level : dict
        Adaptive level (item, int) of each cluster (key, str).

    '''
    def __init__(self, adapt_mat_phase, phase_clusters, adapt_trigger_ratio=None,
                 adapt_max_level=None, adapt_level_max_diff=None):
        '''Spatial discontinuities criterion constructor.

        Parameters
        ----------
        adapt_mat_phase : str
            Adaptive material phase label.
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        adapt_trigger_ratio : float, default=None
            Threshold associated to the adaptivity trigger condition.
        adapt_max_level : int, default=None
            Maximum cluster adaptive level.
        '''
        # Initialize clusters adaptive level
        self._clusters_adapt_level = {str(cluster) : 0 for cluster in
                                      phase_clusters[adapt_mat_phase]}
        # Get optional parameters
        optional_parameters = type(self).get_parameters()
        # Get adaptivity trigger ratio
        if adapt_trigger_ratio is None:
            self._adapt_trigger_ratio = optional_parameters['adapt_trigger_ratio']
        else:
            self._adapt_trigger_ratio = adapt_trigger_ratio
        # Get maximum cluster adaptive level
        if adapt_max_level is None:
            self._adapt_max_level = optional_parameters['adapt_max_level']
        else:
            self._adapt_max_level = adapt_max_level
        # Get cluster adaptive level maximum difference
        if adapt_max_level is None:
            self._adapt_level_max_diff = optional_parameters['adapt_level_max_diff']
        else:
            self._adapt_level_max_diff = adapt_level_max_diff
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_parameters():
        '''Get clustering adaptivity criterion mandatory and optional parameters.

        Besides returning the mandatory and optional adaptivity criterion parameters, this
        method establishes the default values for the optional parameters.

        Returns
        -------
        mandatory_parameters : dict
            Mandatory adaptivity type parameters (str) and associated type (item, type).
        optional_parameters : dict
            Optional adaptivity type parameters (key, str) and associated default value
            (item).
        '''
        # Set mandatory adaptivity criterion parameters
        mandatory_parameters = {}
        # Set optional adaptivity criterion parameters and associated default values
        optional_parameters = {'adapt_trigger_ratio': 0.1,
                               'adapt_max_level': 15,
                               'adapt_level_max_diff': 2}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [mandatory_parameters, optional_parameters]
    # --------------------------------------------------------------------------------------
    def get_target_clusters(self, adapt_data_matrix, voxels_clusters):
        '''Get clustering adaptivity target clusters.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (adapt_phase_n_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the adaptive
            material phase, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].
        voxels_clusters : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the cluster label (int) assigned to the corresponding pixel/voxel.
        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters required for the adaptive procedures.
        '''
        # Initialize target clusters list
        target_clusters = []
        # Initialize target clusters data
        target_clusters_data = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform required spatial evaluations, update list of target clusters and
        # associated data
        if len(voxels_clusters.shape) == 2:
            self._swipe_dimension(adapt_data_matrix, voxels_clusters, target_clusters,
                                  target_clusters_data, '12')
            self._swipe_dimension(adapt_data_matrix, voxels_clusters, target_clusters,
                                  target_clusters_data, '21')
        elif len(voxels_clusters.shape) == 3:
            self._swipe_dimension(adapt_data_matrix, voxels_clusters, target_clusters,
                                  target_clusters_data, '123')
            self._swipe_dimension(adapt_data_matrix, voxels_clusters, target_clusters,
                                  target_clusters_data, '213')
            self._swipe_dimension(adapt_data_matrix, voxels_clusters, target_clusters,
                                  target_clusters_data, '312')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return target_clusters, target_clusters_data
    # --------------------------------------------------------------------------------------
    def update_clusters_adapt_level(self, adaptive_clustering_map):
        '''Update clusters adaptive level after adaptive procedures.

        Parameters
        ----------
        adaptive_clustering_map : dict
            List of new cluster labels (item, list of int) resulting from the adaptive
            procedures over each target cluster (key, str).
        '''
        for old_cluster in adaptive_clustering_map.keys():
            # Get parent cluster adaptive level
            old_cluster_adapt_level = self._clusters_adapt_level[str(old_cluster)]
            # Update child clusters adaptive level
            for cluster in adaptive_clustering_map[str(old_cluster)]:
                self._clusters_adapt_level[str(cluster)] = old_cluster_adapt_level + 1
            # Remove parent cluster
            self._clusters_adapt_level.pop(old_cluster)
    # --------------------------------------------------------------------------------------
    def get_clusters_adapt_level(self):
        '''Get clusters adaptive level.

        Returns
        -------
        clusters_adapt_level : dict
            Adaptive level (item, int) of each cluster (key, str).
        '''
        return copy.deepcopy(self._clusters_adapt_level)
    # --------------------------------------------------------------------------------------
    def _swipe_dimension(self, adapt_data_matrix, voxels_clusters, target_clusters,
                         target_clusters_data, dim_loops):
        '''Evaluate spatial discontinuities along a given dimension.

        The spatial dimensions are cycled according to the code provided in 'dim_loops'.
        The dimension where the spatial discontinuities are evaluated is always the first
        dimension specified in this code (assumed dimension i), while the others cycle the
        remainder dimensions (assumed dimensions j and k). During this process, both the
        list of target clusters and the dictionary containing associated data are updated.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (adapt_phase_n_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the adaptive
            material phase, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].
        voxels_clusters : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the cluster label (int) assigned to the corresponding pixel/voxel.
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict) containing
            cluster associated parameters required for the adaptive procedures.
        dim_loops : string
            Ordered specification of dimension cycles, being the spatial discontinuities
            evaluated along dimension dim_loops[0].
        '''
        # Get material phase clusters
        phase_clusters = adapt_data_matrix[:, 0]
        # Get number of voxels in each dimension
        n_voxels_dims = [voxels_clusters.shape[i] for i in
                         range(len(voxels_clusters.shape))]
        # Set numbers of voxels associated to dimension cycles
        if len(n_voxels_dims) == 2:
            if dim_loops not in ('12', '21'):
                raise RuntimeError('Invalid dimension cycles code.')
            else:
                # Set cycling dimensions map (ordered as i, j)
                cycle_spatial_map = [int(dim_loops[0]) - 1,
                                     int(dim_loops[1]) - 1]
                # Set cycling dimensions numbers of voxels
                n_voxels_i = n_voxels_dims[0]
                n_voxels_j = n_voxels_dims[1]
                n_voxels_k = 1
        elif len(n_voxels_dims) == 3:
            if dim_loops not in ('123', '132', '213', '231', '312', '321'):
                raise RuntimeError('Invalid dimension cycles code.')
            else:
                # Set cycling dimensions map (ordered as i, j, k)
                cycle_spatial_map = [int(dim_loops[0]) - 1,
                                     int(dim_loops[1]) - 1,
                                     int(dim_loops[2]) - 1]
                # Set cycling dimensions numbers of voxels
                n_voxels_i = n_voxels_dims[0]
                n_voxels_j = n_voxels_dims[1]
                n_voxels_k = n_voxels_dims[2]
        else:
            raise RuntimeError('Invalid number of dimensions.')
        # Set spatial dimensions map (ordered as 1, 2 [, 3])
        spatial_cycle_map = [cycle_spatial_map.index(i) for i in range(len(n_voxels_dims))]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get minimum and maximum value of adaptivity feature
        min_feature_val = min(adapt_data_matrix[:, 1])
        max_feature_val = max(adapt_data_matrix[:, 1])
        # Get absolute value of maximum range of adaptivity feature
        norm_factor = abs(max_feature_val - min_feature_val)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over voxels of dimension k
        for voxel_k in range(n_voxels_k):
            # Loop over voxels of dimension j
            for voxel_j in range(n_voxels_j):
                # Loop over voxels of dimension i (evaluation of spatial discontinuities)
                for voxel_i in range(n_voxels_i):
                    # Get voxel (i) ordered spatial indexes
                    idxs = [(voxel_i, voxel_j, voxel_k)[i] for i in spatial_cycle_map]
                    # Get cluster label of voxel (i)
                    cluster = voxels_clusters[tuple(idxs)]
                    # Get next voxel (i+1) ordered spatial indexes. If voxel (i) is the last
                    # one, then the next voxel (i+1) is the first one
                    if voxel_i == n_voxels_i - 1:
                        idxs[cycle_spatial_map[0]] = 0
                    else:
                        idxs[cycle_spatial_map[0]] += 1
                    # Get cluster labels of next voxel (i+1)
                    cluster_next = voxels_clusters[tuple(idxs)]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip computations if at least one of the clusters does not belong to
                    # the adaptive material phase associated to the criterion instance
                    if cluster not in phase_clusters or cluster_next not in phase_clusters:
                        continue
                    # Skip computations if voxels belong to the same cluster
                    if cluster == cluster_next:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Evaluate clusters adaptive levels
                    is_cluster_targetable = self._clusters_adapt_level[str(cluster)] < \
                        self._adapt_max_level
                    is_cluster_next_targetable = \
                        self._clusters_adapt_level[str(cluster_next)] < \
                            self._adapt_max_level
                    # Skip computations if both clusters are untargetable
                    if not is_cluster_targetable and not is_cluster_next_targetable:
                        continue
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get cluster index and feature value
                    cluster_idx = list(adapt_data_matrix[:, 0]).index(cluster)
                    value = adapt_data_matrix[cluster_idx, 1]
                    # Get cluster of next voxel and feature value
                    cluster_next_idx = list(adapt_data_matrix[:, 0]).index(cluster_next)
                    value_next = adapt_data_matrix[cluster_next_idx, 1]
                    # Compute normalized spatial discontinuity along dimension i
                    ratio = abs(value_next - value)/norm_factor
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Skip computations if normalized spatial discontinuity is lower than
                    # prescribed threshold. Otherwise, compute associated magnitude
                    if ratio < self._adapt_trigger_ratio:
                        continue
                    else:
                        magnitude = abs(ratio - self._adapt_trigger_ratio)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Evaluate if clusters have already been targeted
                    is_cluster_targeted = cluster in target_clusters
                    is_cluster_next_targeted = cluster_next in target_clusters
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update previously targeted clusters data
                    if is_cluster_targeted:
                        max_magn = target_clusters_data[str(cluster)]['max_magnitude']
                        if magnitude > max_magn:
                            target_clusters_data[str(cluster)]['max_magnitude'] = magnitude
                    if is_cluster_next_targeted:
                        max_magn = target_clusters_data[str(cluster_next)]['max_magnitude']
                        if magnitude > max_magn:
                            target_clusters_data[str(cluster_next)]['max_magnitude'] = \
                                magnitude
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Update target clusters list and associated data
                    if not(is_cluster_targeted and is_cluster_next_targeted):
                        # Evaluate clusters adaptive level
                        cluster_adapt_level = self._clusters_adapt_level[str(cluster)]
                        cluster_next_adapt_level = \
                            self._clusters_adapt_level[str(cluster_next)]
                        # Compute differences of clusters adaptive level
                        diff_1 = cluster_next_adapt_level - cluster_adapt_level
                        diff_2 = cluster_adapt_level - cluster_next_adapt_level
                        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        # Update target clusters list according to associated adaptive
                        # and update data of new targeted clusters
                        if diff_1 > self._adapt_level_max_diff:
                            if not is_cluster_targeted and is_cluster_targetable:
                                target_clusters.append(cluster)
                                target_clusters_data[str(cluster)] = {}
                                target_clusters_data[str(cluster)]['max_magnitude'] = \
                                    magnitude
                        elif diff_2 > self._adapt_level_max_diff:
                            if not is_cluster_next_targeted and is_cluster_next_targetable:
                                target_clusters.append(cluster_next)
                                target_clusters_data[str(cluster_next)] = {}
                                target_clusters_data[str(cluster_next)]['max_magnitude'] = \
                                    magnitude
                        else:
                            if not is_cluster_targeted and is_cluster_targetable:
                                target_clusters.append(cluster)
                                target_clusters_data[str(cluster)] = {}
                                target_clusters_data[str(cluster)]['max_magnitude'] = \
                                    magnitude
                            if not is_cluster_next_targeted and is_cluster_next_targetable:
                                target_clusters.append(cluster_next)
                                target_clusters_data[str(cluster_next)] = {}
                                target_clusters_data[str(cluster_next)]['max_magnitude'] = \
                                    magnitude
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over target clusters
        for cluster in target_clusters:
            # Store adaptive trigger ratio for each target cluster
            target_clusters_data[str(cluster)]['adapt_trigger_ratio'] = \
                self._adapt_trigger_ratio
            # Set dynamic adaptive clustering split factor flag
            target_clusters_data[str(cluster)]['is_dynamic_split_factor'] = True
