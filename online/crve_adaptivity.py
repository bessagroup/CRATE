#
# CRVE Online Adaptivity Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the CRVE adaptivity during the clustering-based reduced order model
# solution of the microscale equilibrium problem (online stage).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | October 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Date and time
import time
# Display messages
import ioput.info as info
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                            CRVE Online Adaptivity Criteria
# ==========================================================================================
class AdaptivityManager:
    '''Online CRVE clustering adaptivity manager.

    Attributes
    ----------
    _adapt_groups : dict
        Adaptive cluster groups (item, list of int) for each adaptive material phase
        (key, str).
    _groups_adapt_level : dict
        For each adaptive material phase (key, str), store a dictionary containing the
        adaptive level (item, int) of each associated adaptive cluster group (key, int).
    _target_groups_ids : dict
        Target adaptive cluster groups (item, list of int), whose clusters are to adapted,
        for each adaptive material phase (key, str).
    '''
    def __init__(self, adaptive_material_phases, adaptivity_features):
        '''Online CRVE clustering adaptivity manager constructor.

        Parameters
        ----------
        adaptive_material_phases : list
            RVE adaptive material phases labels (str).
        adaptive_features : dict
            Adaptive feature (item, str) to be considered for each adaptive material phase
            (key, str).
        adapt_trigger_ratio : float
            Threshold associated to the adaptivity trigger condition, defining the value of
            the relative ratio (max - avg)/avg above which the adaptive cluster group is set
            to be adapted, where max and avg are the maximum and average adaptivity feature
            values in the adaptive cluster group, respectively.
        adapt_split_threshold : float
            Threshold associated to the adaptivity selection criterion, defining the split
            boundary of each adaptive cluster group according to the associated position in
            terms of the adaptivity value range within the group. For instance, a
            adapt_split_threshold=0.2 means that the split boundary divides the clusters
            whose adaptivity feature value is above min + 0.8*(max - min) (top 20% of the
            value range) from the remaining clusters, resulting two child adaptive cluster
            groups.
        adapt_max_level : int
            Maximum adaptive cluster group adaptive level.

        Notes
        -----
        These hyperparameters are to be defined for each adaptive material phase and to be
        read in the user input data file!
        '''
        adapt_trigger_ratio = 0.1
        adapt_split_threshold = 0.1
        adapt_max_level = 15
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._adaptive_material_phases = adaptive_material_phases
        self._adaptivity_features = adaptivity_features
        self._adapt_trigger_ratio = adapt_trigger_ratio
        self._adapt_split_threshold = adapt_split_threshold
        self._adapt_max_level = adapt_max_level
        self._adapt_groups = {mat_phase: {} for mat_phase in adaptive_material_phases}
        self._groups_adapt_level = {mat_phase: {} for mat_phase in adaptive_material_phases}
        self._target_groups_ids = {mat_phase: None for mat_phase in
                                   adaptive_material_phases}
    # --------------------------------------------------------------------------------------
    def get_target_clusters(self, phase_clusters, clusters_state):
        '''Get online adaptive clustering target clusters.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        '''
        # Initialize target clusters list
        target_clusters = []
        # Loop over adaptive material phases
        for adapt_phase in self._adaptive_material_phases:
            # Get adaptivity feature
            adapt_feature = self._adaptivity_features[adapt_phase]
            # Build adaptivity feature data matrix
            adapt_data_matrix = self._get_adaptivity_data_matrix(adapt_phase, adapt_feature,
                                                                 phase_clusters,
                                                                 clusters_state)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set initial adaptive clusters group (labeled as group '0') and initialize
            # associated adaptive level
            if not self._adapt_groups[adapt_phase]:
                self._adapt_groups[adapt_phase]['0'] = phase_clusters[adapt_phase]
                self._groups_adapt_level[adapt_phase]['0'] = 0
            # Get adaptive material phase cluster groups
            adapt_groups_ids = list(self._adapt_groups[adapt_phase].keys())
            # Initialize adaptive material phase target cluster groups
            self._target_groups_ids[adapt_phase] = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            print('\n\nAdaptive material phase: ', adapt_phase, '\n' + 80*'-')
            print('\nCurrent cluster groups ids:\n\n', adapt_groups_ids)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over adaptive cluster groups
            for group_id in adapt_groups_ids:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                print('\n\nEvaluating adaptive cluster group: ', group_id, '\n' + 40*'-')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check if adaptive cluster group can be further adapted and, if not, skip
                # to the next one
                if self._groups_adapt_level[adapt_phase][group_id] >= self._adapt_max_level:
                    continue
                # Get adaptive cluster group clusters and current adaptive level
                adapt_group = self._adapt_groups[adapt_phase][group_id]
                base_adapt_level = self._groups_adapt_level[adapt_phase][group_id]
                # Get adaptivity data matrix row indexes associated with the adaptive
                # cluster group clusters
                adapt_group_idxs = np.where(np.in1d(adapt_data_matrix[:, 0],
                                                    adapt_group))[0]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check adaptive cluster group adaptivity trigger condition
                is_trigger = self._adaptivity_trigger_condition(
                    adapt_data_matrix[adapt_group_idxs, :],
                        trigger_ratio=self._adapt_trigger_ratio)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                print('\n  > is_trigger? ', is_trigger)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get adaptive cluster group target clusters and assemble them to the global
                # target clusters list
                if is_trigger:
                    # Get target clusters according to adaptivity selection criterion
                    group_target_clusters = self._adaptivity_selection_criterion(
                        adapt_data_matrix[adapt_group_idxs, :],
                            threshold=self._adapt_split_threshold)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get maximum adaptive cluster group id
                    max_group_id = np.max([int(x) for x in
                                           self._adapt_groups[adapt_phase].keys()])
                    # Set child adaptive cluster group to be adapted (set id, set clusters
                    # and increment adaptive level relative to parent cluster group)
                    child_group_1_id = str(max_group_id + 1)
                    self._adapt_groups[adapt_phase][child_group_1_id] = \
                        group_target_clusters
                    self._groups_adapt_level[adapt_phase][child_group_1_id] = \
                        base_adapt_level + 1
                    self._target_groups_ids[adapt_phase].append(child_group_1_id)
                    # Set child adaptive cluster group to be left unchanged (set id, set
                    # clusters and set adaptive level equal to the parent cluster group)
                    child_group_2_id = str(max_group_id + 2)
                    self._adapt_groups[adapt_phase][child_group_2_id] = \
                        list(set(adapt_group) - set(group_target_clusters))
                    self._groups_adapt_level[adapt_phase][child_group_2_id] = \
                        base_adapt_level
                    # Remove parent adaptive cluster group
                    self._adapt_groups[adapt_phase].pop(group_id)
                    self._groups_adapt_level[adapt_phase].pop(group_id)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble target clusters
                    target_clusters += group_target_clusters
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    print('\n\nGroup ' + group_id + ' was split in groups:')
                    print('\n  Group ' + child_group_1_id + ': ',
                          self._adapt_groups[adapt_phase][child_group_1_id])
                    print('\n  Group ' + child_group_2_id + ': ',
                          self._adapt_groups[adapt_phase][child_group_2_id])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set adaptivity trigger flag according to the list of target clusters
        if target_clusters:
            is_trigger = True
        else:
            is_trigger = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n\nAdaptive clustering refinement:\n')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_trigger, target_clusters
    # --------------------------------------------------------------------------------------
    def _get_adaptivity_data_matrix(self, target_phase, adapt_feature, phase_clusters,
                                    clusters_state):
        '''Build adaptivity feature data matrix for a given target adaptive material phase.

        Parameters
        ----------
        target_phase : str
            Target adaptive material phase whose clusters adaptive feature data is
            collected or computed.
        adapt_feature : str, {'acc_p_strain',}
            Adaptivity feature available directly or indirectly from clusters state
            variables.
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).

        Returns
        -------
        adapt_data_matrix : ndarray of shape (n_target_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the target adaptive
            material phase, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].
        '''
        # Get target adaptive material phase number of clusters
        n_clusters = len(phase_clusters[target_phase])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize adaptivity feature data matrix
        adapt_data_matrix = np.zeros((n_clusters, 2))
        adapt_data_matrix[:, 0] = phase_clusters[target_phase]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build adaptivity feature data matrix
        if adapt_feature == 'acc_p_strain':
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                adapt_data_matrix[i, 1] = clusters_state[str(cluster)][adapt_feature]
        else:
            raise RuntimeError('Unknown clustering adaptivity feature.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return adapt_data_matrix
    # --------------------------------------------------------------------------------------
    def _adaptivity_trigger_condition(self, adapt_data_matrix, trigger_ratio):
        '''Online CRVE clustering adaptivity trigger condition.

        Parameters
        ----------
        adapt_data_matrix : ndarray of shape (n_target_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the target adaptive
            material phase, contains the cluster label in adapt_data_matrix[i, 0] and the
            associated adaptive feature value in adapt_data_matrix[i, 1].
        trigger_ratio : float
            Threshold defining the value of the relative ratio (max - avg)/avg above which
            the adaptive cluster group is set to be adapted, where max and avg are the
            maximum and average adaptivity feature values in the adaptive cluster group,
            respectively.

        Returns
        -------
        bool
            True if online adaptive clustering procedure is triggered, False otherwise.
        '''
        # Compute average and maximum values of adaptivity feature
        a = np.average(adapt_data_matrix[:, 1])
        b = np.max(adapt_data_matrix[:, 1])
        # Compute adaptivity ratio
        adapt_ratio = abs(b - a)/abs(a)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n\nEvaluating trigger condition:')
        print('\n  > adaptive group adapt_data_matrix:\n')
        print(adapt_data_matrix)
        print('\n  > average = ', a)
        print('\n  > max     = ', b)
        print('\n  > adapt_ratio = ', adapt_ratio)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check adaptivity trigger condition
        if adapt_ratio > trigger_ratio:
            return True
        else:
            return False
   # ---------------------------------------------------------------------------------------
    def _adaptivity_selection_criterion(self, adapt_data_matrix, threshold):
        '''Online CRVE clustering adaptivity selection criterion.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_target_clusters, 2)
            Adaptivity feature data matrix that, for the i-th cluster of the target adaptive
            material phase, contains the cluster label in data_matrix[i, 0] and the
            associated adaptive feature value in data_matrix[i, 1].
        threshold : float
            Threshold that defines the split boundary of each adaptive cluster group
            according to the associated position in terms of the adaptivity value range
            within the group. For instance, a threshold=0.2 means that the split boundary
            divides the clusters whose adaptivity feature value is above 0.8*(max - min)
            (top 20% of the value range) from the remaining clusters, resulting two child
            adaptive cluster groups.

        Returns
        -------
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        '''
        # Check threshold
        if not ioutil.is_between(threshold, lower_bound=0, upper_bound=1):
            raise RuntimeError('Clustering adaptivity selection criterion\'s threshold ' +
                               'must be between 0 and 1 (included).')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get split boundary adaptivity feature value
        adapt_boundary = min(adapt_data_matrix[:, 1]) + \
            (1.0 - threshold)*(max(adapt_data_matrix[:, 1]) - min(adapt_data_matrix[:, 1]))
        # Get indexes of clusters whose adaptivity feature value is greater or equal than
        # the split boundary value
        idxs = adapt_data_matrix[:, 1] >= adapt_boundary
        # Get target clusters
        target_clusters = [int(x) for x in adapt_data_matrix[idxs, 0]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n\nAdaptivity selection:')
        print('\n  > Number of target clusters: ', len(target_clusters))
        print('\n  > Target clusters: ', target_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return target_clusters
    # --------------------------------------------------------------------------------------
    def adaptive_refinement(self, crve, target_clusters, cluster_dicts, verbose=False):
        '''Perform CRVE online adaptive clustering refinement.

        Parameters
        ----------
        crve : CRVE
            Cluster-reduced Representative Volume Element.
        target_clusters : list
            List containing the labels (int) of clusters to be refined.
        cluster_dicts : list
            List containing cluster-label-keyd dictionaries (key, str) that will be updated
            as the result of the CRVE online adaptive refinement.
        verbose : bool, default=False
            Enable verbose output.
        '''
        # Set initial time
        init_time = time.time()
        # Output execution data
        if verbose:
            info.displayinfo('5', 'A-CRVE clustering (adaptive step ' +
                                   str(crve._adaptive_step + 1) + '):')
            info.displayinfo('5', 'A - Building A-CRVE clustering...', 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform CRVE online adaptive clustering refinement
        if not target_clusters:
            return
        else:
            adaptive_clustering_map, _ = crve.perform_adaptive_clustering(target_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            a_time = time.time() - init_time
            info.displayinfo('5', 'B - Computing A-CRVE cluster interaction tensors...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute CRVE cluster interaction tensors
        crve.compute_cit(mode='adaptive', adaptive_clustering_map=adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            b_time = time.time() - ref_time
            info.displayinfo('5', 'C - Updating A-CRVE cluster-related quantities...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for adapt_phase in self._adaptive_material_phases:
            # Loop over material phase target clusters
            for target_cluster in adaptive_clustering_map[adapt_phase].keys():
                # Get list of target's child clusters
                child_clusters = adaptive_clustering_map[adapt_phase][target_cluster]
                # Loop over cluster-keyd dictionaries
                for cluster_dict in cluster_dicts:
                    # Loop over child clusters and build their items
                    for child_cluster in child_clusters:
                        cluster_dict[str(child_cluster)] = \
                            copy.deepcopy(cluster_dict[target_cluster])
                    # Remove target cluster item
                    cluster_dict.pop(target_cluster)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            c_time = time.time() - ref_time
            info.displayinfo('5', 'D - Updating A-CRVE cluster groups...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update adaptive cluster groups
        self._update_group_clusters(adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            d_time = time.time() - ref_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the A-CRVE adaptive procedures
        dtime = time.time() - init_time
        crve.adaptive_time += dtime
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            # Output adaptive phases execution time table
            indent = 10*' '
            info.displayinfo('5', 'Execution times (s):' + '\n\n' +
                             indent + 'Phase      Time(s)       %' + '\n' +
                             indent + 28*'-' + '\n' +
                             indent + '{:<5s}'.format('A') + '{:^18.4e}'.format(a_time) +
                             '{:>5.2f}'.format(a_time/dtime) + '\n' +
                             indent + '{:<5s}'.format('B') + '{:^18.4e}'.format(b_time) +
                             '{:>5.2f}'.format(b_time/dtime) + '\n' +
                             indent + '{:<5s}'.format('C') + '{:^18.4e}'.format(c_time) +
                             '{:>5.2f}'.format(c_time/dtime) + '\n' +
                             indent + '{:<5s}'.format('D') + '{:^18.4e}'.format(c_time) +
                             '{:>5.2f}'.format(d_time/dtime) + '\n' +
                             indent + 28*'-' + '\n' +
                             indent + '{:^5s}'.format('Total') + '{:^18.4e}'.format(dtime),
                             2)
            # Print adaptive clustering mapping
            print('\n\n' + 4*' ' + 'Adaptive cluster mapping: ')
            for mat_phase in crve._material_phases:
                print('\n' + 6*' ' +'Material phase ' + mat_phase + ':\n')
                for old_cluster in adaptive_clustering_map[mat_phase].keys():
                    print(8*' ' + 'Old cluster: ' + '{:>4s}'.format(old_cluster) +
                          '  ->  ' +
                          'New clusters: ',
                          adaptive_clustering_map[mat_phase][str(old_cluster)])
    # --------------------------------------------------------------------------------------
    def _update_group_clusters(self, adaptive_clustering_map):
        '''Update adaptive cluster groups.

        Parameters
        ----------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the adaptation of each target cluster (key, str))
            for each material phase (key, str).
        '''
        # Loop over adaptive material phases
        for adapt_phase in self._adaptive_material_phases:
            # Loop over adaptive cluster groups
            for group_id in self._target_groups_ids[adapt_phase]:
                # Get parent adaptive cluster group clusters
                old_clusters = self._adapt_groups[adapt_phase][group_id].copy()
                # Loop over adaptive cluster group clusters
                for old_cluster in old_clusters:
                    # If cluster has been adapted, then update adaptive cluster group
                    if str(old_cluster) in adaptive_clustering_map[adapt_phase].keys():
                        # Remove parent cluster from group
                        self._adapt_groups[adapt_phase][group_id].remove(old_cluster)
                        # Add child clusters to group
                        new_clusters = \
                            adaptive_clustering_map[adapt_phase][str(old_cluster)]
                        self._adapt_groups[adapt_phase][group_id] += new_clusters
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                print('\n\n    > Updated group ' + group_id + ' from\n')
                print('    ', old_clusters)
                print('\n    to\n')
                print('    ', self._adapt_groups[adapt_phase][group_id], '\n')
