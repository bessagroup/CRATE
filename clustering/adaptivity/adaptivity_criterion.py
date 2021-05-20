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
    def get_parameters(self):
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
        '''Get clustering adaptivity target clusters.

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
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
        '''
        # Initialize target clusters list
        target_clusters = []
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
        return target_clusters
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
