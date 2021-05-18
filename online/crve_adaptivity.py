#
# CRVE Online Adaptivity Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the CRVE adaptivity during the clustering-based reduced order model
# solution of the microscale equilibrium problem (online stage).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2020 | Initial coding.
# Bernardo P. Ferreira | Dec 2020 | Reformulation with cluster-reduced material phases.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Date and time
import time
# Regular expressions
import re
# Display messages
import ioput.info as info
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                         CRVE clustering adaptivity manager
# ==========================================================================================
class AdaptivityManager:
    '''CRVE clustering adaptivity manager.

    Attributes
    ----------
    _adapt_groups : dict
        For each adaptive material phase (key, str), store a dictionary containing the
        cluster labels (item, list of int) associated to each adaptive cluster group
        (key, int).
    _groups_adapt_level : dict
        For each adaptive material phase (key, str), store a dictionary containing the
        adaptive level (item, int) of each associated adaptive cluster group (key, int).
    _target_groups_ids : dict
        Target adaptive cluster groups (item, list of int), whose clusters are to adapted,
        for each adaptive material phase (key, str).
    inc_adaptive_steps : dict
        For each macroscale loading increment (key, str), store the performed number of
        clustering adaptive steps (item, int).
    max_inc_adaptive_steps : int
        Maximum number of clustering adaptive steps per macroscale loading increment.
    adaptive_evaluation_time : float
        Total amount of time (s) spent in selecting target clusters for clustering
        adaptivity.
    adaptive_time : float
        Total amount of time (s) spent in clustering adaptivity procedures.
    '''
    def __init__(self, comp_order, adapt_material_phases, adaptivity_control_feature,
                 adaptivity_criterion, clust_adapt_freq):
        '''Online CRVE clustering adaptivity manager constructor.

        Parameters
        ----------
        comp_order : list
            Strain/Stress components (str) order.
        adapt_material_phases : list
            RVE adaptive material phases labels (str).
        adaptivity_control_feature : dict
            Clustering adaptivity control feature (item, str) associated to each material
            phase (key, str).
        adaptivity_criterion : dict
            Clustering adaptivity criterion (item, dict) associated to each material phase
            (key, str). This dictionary contains the adaptivity criterion to be used and the
            required parameters.
        clust_adapt_freq : dict
            Clustering adaptivity frequency (relative to the macroscale loading)
            (item, int, default=1) associated with each adaptive cluster-reduced
            material phase (key, str).

        Notes
        -----
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
        '''
        self._comp_order = comp_order
        self._adapt_material_phases = adapt_material_phases
        self._adaptivity_control_feature = adaptivity_control_feature
        self._adaptivity_criterion = adaptivity_criterion
        self._adapt_groups = {mat_phase: {} for mat_phase in adapt_material_phases}
        self._groups_adapt_level = {mat_phase: {} for mat_phase in adapt_material_phases}
        self._target_groups_ids = {mat_phase: None for mat_phase in
                                   adapt_material_phases}
        self._clust_adapt_freq = clust_adapt_freq
        self.inc_adaptive_steps = {}
        self.max_inc_adaptive_steps = 1
        self.adaptive_evaluation_time = 0
        self.adaptive_time = 0
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_criterions():
        '''Get available clustering adaptivity criterions.

        Returns
        -------
        available_adapt_criterions : dict
            Available clustering adaptivity criterions (item, str) and associated
            identifiers (key, str).
        '''
        # Set available clustering adaptivity criterions
        available_adapt_criterions = {'1': 'Non-spatial'}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_adapt_criterions
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_criterion_parameters():
        '''Get mandatory and optional adaptivity criterion parameters.

        Besides returning the mandatory and optional adaptivity criterion parameters, this
        method establishes the default values for the optional parameters.

        Returns
        ----------
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
    def get_target_clusters(self, phase_clusters, clusters_state, clusters_state_old,
                            clusters_sct_mf, clusters_sct_mf_old, clusters_residuals_mf,
                            inc=None, verbose=False):
        '''Get online adaptive clustering target clusters.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).
        clusters_state_old : dict
            Last increment converged material constitutive model state variables
            (item, dict) associated to each material cluster (key, str).
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).
        clusters_sct_mf_old : dict
            Last increment converged fourth-order strain concentration tensor
            (matricial form) (item, ndarray) associated to each material cluster (key, str).
        clusters_residuals_mf : dict
            Equilibrium residual second-order tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).
        inc : int, default=None
            Incremental counter serving as a reference basis for the clustering adaptivity
            frequency control.
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        is_trigger : bool
            True if clustering adaptivity is triggered, False otherwise.
        target_clusters : list
            List containing the labels (int) of clusters to be adapted.
        '''
        init_time = time.time()
        # Output execution data
        if verbose:
            info.displayinfo('5', 'Clustering adaptivity criterion:')
            info.displayinfo('5', 'Selecting target clusters...', 2)
        # Get activated adaptive material phases
        if inc != None:
            activated_adapt_phases = self._get_activated_adaptive_phases(inc)
        else:
            activated_adapt_phases = self._adapt_material_phases
        # Initialize target clusters list
        target_clusters = []
        # Loop over activated adaptive material phases
        for mat_phase in activated_adapt_phases:
            # Get adaptivity trigger ratio
            adapt_trigger_ratio = \
                self._adaptivity_criterion[mat_phase]['adapt_trigger_ratio']
            # Get adaptivity split threhsold
            adapt_split_threshold = \
                self._adaptivity_criterion[mat_phase]['adapt_split_threshold']
            # Get maximum adaptive cluster group adaptive level
            adapt_max_level = self._adaptivity_criterion[mat_phase]['adapt_max_level']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get adaptivity feature
            adapt_control_feature = self._adaptivity_control_feature[mat_phase]
            # Build adaptivity feature data matrix
            adapt_data_matrix = \
                self._get_adaptivity_data_matrix(mat_phase, adapt_control_feature,
                                                 phase_clusters, clusters_state,
                                                 clusters_state_old, clusters_sct_mf,
                                                 clusters_sct_mf_old, clusters_residuals_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set initial adaptive clusters group (labeled as group '0') and initialize
            # associated adaptive level
            if not self._adapt_groups[mat_phase]:
                self._adapt_groups[mat_phase]['0'] = phase_clusters[mat_phase]
                self._groups_adapt_level[mat_phase]['0'] = 0
            # Get adaptive material phase cluster groups
            adapt_groups_ids = list(self._adapt_groups[mat_phase].keys())
            # Initialize adaptive material phase target cluster groups
            self._target_groups_ids[mat_phase] = []
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over adaptive cluster groups
            for group_id in adapt_groups_ids:
                # Check if adaptive cluster group can be further adapted and, if not, skip
                # to the next one
                if self._groups_adapt_level[mat_phase][group_id] >= adapt_max_level:
                    continue
                # Get adaptive cluster group clusters and current adaptive level
                adapt_group = self._adapt_groups[mat_phase][group_id]
                base_adapt_level = self._groups_adapt_level[mat_phase][group_id]
                # Get adaptivity data matrix row indexes associated with the adaptive
                # cluster group clusters
                adapt_group_idxs = np.where(np.in1d(adapt_data_matrix[:, 0],
                                                    adapt_group))[0]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check adaptive cluster group adaptivity trigger condition
                is_trigger = self._adaptivity_trigger_condition(
                    adapt_data_matrix[adapt_group_idxs, :],
                        trigger_ratio=adapt_trigger_ratio)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get adaptive cluster group target clusters and assemble them to the global
                # target clusters list
                if is_trigger:
                    # Get target clusters according to adaptivity selection criterion
                    group_target_clusters = self._adaptivity_selection_criterion(
                        adapt_data_matrix[adapt_group_idxs, :],
                            threshold=adapt_split_threshold)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get maximum adaptive cluster group id
                    max_group_id = np.max([int(x) for x in
                                           self._adapt_groups[mat_phase].keys()])
                    # Set child adaptive cluster group to be adapted (set id, set clusters
                    # and increment adaptive level relative to parent cluster group)
                    child_group_1_id = str(max_group_id + 1)
                    self._adapt_groups[mat_phase][child_group_1_id] = \
                        group_target_clusters
                    self._groups_adapt_level[mat_phase][child_group_1_id] = \
                        base_adapt_level + 1
                    self._target_groups_ids[mat_phase].append(child_group_1_id)
                    # Set child adaptive cluster group to be left unchanged (set id, set
                    # clusters and set adaptive level equal to the parent cluster group)
                    child_group_2_id = str(max_group_id + 2)
                    self._adapt_groups[mat_phase][child_group_2_id] = \
                        list(set(adapt_group) - set(group_target_clusters))
                    self._groups_adapt_level[mat_phase][child_group_2_id] = \
                        base_adapt_level
                    # Remove parent adaptive cluster group
                    self._adapt_groups[mat_phase].pop(group_id)
                    self._groups_adapt_level[mat_phase].pop(group_id)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Assemble target clusters
                    target_clusters += group_target_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set adaptivity trigger flag according to the list of target clusters
        if target_clusters:
            is_trigger = True
        else:
            is_trigger = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in selecting target clusters for clustering
        # adaptivity
        self.adaptive_evaluation_time += time.time() - init_time
        # Update total amount of time spent in clustering adaptivity procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            # Build output data
            output_list = []
            output_total = 0
            for mat_phase in self._adapt_material_phases:
                phase_target_clusters = \
                    list(set(target_clusters).intersection(phase_clusters[mat_phase]))
                output_list += [mat_phase, len(phase_target_clusters)]
                output_total += len(phase_target_clusters)
            # Output adaptive phases target clusters summary table
            indent = 10*' '
            info.displayinfo('5', 'Summary:' + '\n\n' +
                             indent + 'Phase   Target Clusters' + '\n' +
                             indent + 23*'-' + '\n' +
                             ((indent + '{:^5s}{:>11d}\n')*
                             (len(self._adapt_material_phases))).format(*output_list) +
                             indent + 23*'-' + '\n' +
                             indent + '{:^5s}'.format('Total') +
                             '{:>11d}'.format(output_total) + '\n',
                             2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_trigger, target_clusters
    # --------------------------------------------------------------------------------------
    def _get_adaptivity_data_matrix(self, target_phase, adapt_control_feature,
                                    phase_clusters, clusters_state, clusters_state_old,
                                    clusters_sct_mf, clusters_sct_mf_old,
                                    clusters_residuals_mf):
        '''Build adaptivity feature data matrix for a given target adaptive material phase.

        Parameters
        ----------
        target_phase : str
            Target adaptive material phase whose clusters adaptive feature data is
            collected or computed.
        adapt_control_feature : str
            Scalar adaptivity feature available directly or indirectly from clusters state
            variables.
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).
        clusters_state_old : dict
            Last increment converged material constitutive model state variables
            (item, dict) associated to each material cluster (key, str).
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).
        clusters_sct_mf_old : dict
            Last increment converged fourth-order strain concentration tensor
            (matricial form) (item, ndarray) associated to each material cluster (key, str).
        clusters_residuals_mf : dict
            Equilibrium residual second-order tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).

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
        # Get target material phase state variables
        state_variables = clusters_state[str(int(adapt_data_matrix[0, 0]))].keys()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate nature of adaptivity feature
        is_state_variable = False
        for state_variable in state_variables:
            if state_variable in adapt_control_feature:
                is_state_variable = True
                break
        is_component = False
        is_norm = False
        is_incremental = False
        if re.search('_[1-3][1-3]$', adapt_control_feature):
            # If adaptivity feature is component of second order tensor
            is_component = True
            # Get second order tensor component and matricial form index
            feature_comp = adapt_control_feature[-2:]
            index = self._comp_order.index(feature_comp)
            # Trim adaptivity feature
            adapt_control_feature = adapt_control_feature[:-3]
        if re.search('_norm$', adapt_control_feature):
            # If considering the norm of the tensorial adaptivity feature
            is_norm = True
            # Trim adaptivity feature
            adapt_control_feature = adapt_control_feature[:-5]
        if re.search('^inc_', adapt_control_feature):
            # If considering the incremental value of the adaptivity feature
            is_incremental = True
            # Trim adaptivity feature
            adapt_control_feature = adapt_control_feature[4:]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is scalar state variable
        if is_state_variable and not is_component:
            # Check if adaptivity feature is scalar
            if not ioutil.checknumber(clusters_state[str(int(adapt_data_matrix[0, 0]))]
                                      [adapt_control_feature]) and not is_norm:
                raise RuntimeError('The clustering adaptivity feature (' +
                                   adapt_control_feature + ') prescribed for material ' +
                                   'material phase ' + target_phase + ' must be a scalar.')
            else:
                # Build adaptivity feature data matrix
                for i in range(n_clusters):
                    # Get cluster label
                    cluster = int(adapt_data_matrix[i, 0])
                    # Collect adaptivity feature data
                    if is_norm:
                        if is_incremental:
                            adapt_data_matrix[i, 1] = \
                                np.linalg.norm(clusters_state[str(cluster)]
                                               [adapt_control_feature]) - \
                                np.linalg.norm(clusters_state_old[str(cluster)]
                                               [adapt_control_feature])
                        else:
                            adapt_data_matrix[i, 1] = \
                                np.linalg.norm(clusters_state[str(cluster)]
                                               [adapt_control_feature])
                    else:
                        if is_incremental:
                            adapt_data_matrix[i, 1] = \
                                clusters_state[str(cluster)][adapt_control_feature] - \
                                clusters_state_old[str(cluster)][adapt_control_feature]
                        else:
                            adapt_data_matrix[i, 1] = \
                                clusters_state[str(cluster)][adapt_control_feature]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is component of strain/stress related second order
        # tensor (stored in matricial form) state variable
        elif is_state_variable and is_component:
            # Check if adaptivity feature is scalar
            if ioutil.checknumber(clusters_state[str(int(adapt_data_matrix[0, 0]))]
                                  [adapt_control_feature]):
                raise RuntimeError('The clustering adaptivity feature (' +
                                   adapt_control_feature + ') prescribed for material ' +
                                   'material phase ' + target_phase + ' is a scalar.')
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                if is_incremental:
                    adapt_data_matrix[i, 1] = \
                        clusters_state[str(cluster)][adapt_control_feature][index] - \
                        clusters_state_old[str(cluster)][adapt_control_feature][index]
                else:
                    adapt_data_matrix[i, 1] = \
                        clusters_state[str(cluster)][adapt_control_feature][index]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is norm of strain concentration tensor
        elif adapt_control_feature == 'strain_concentration_tensor' and is_norm:
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                if is_incremental:
                    adapt_data_matrix[i, 1] = \
                        np.linalg.norm(clusters_sct_mf[str(cluster)]) - \
                        np.linalg.norm(clusters_sct_mf_old[str(cluster)])
                else:
                    adapt_data_matrix[i, 1] = np.linalg.norm(clusters_sct_mf[str(cluster)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is norm of Lippmann-Schwinger equilibrium residual
        elif adapt_control_feature == 'equilibrium_residual' and is_norm:
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                adapt_data_matrix[i, 1] = \
                    np.linalg.norm(clusters_residuals_mf[str(cluster)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown or unavailable clustering adaptivity feature.')
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
        if abs(a) < 1e-10:
            adapt_ratio = abs(b - a)
        else:
            adapt_ratio = abs(b - a)/abs(a)
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
        return target_clusters
    # --------------------------------------------------------------------------------------
    def adaptive_refinement(self, crve, target_clusters, cluster_dicts, inc,
                            improved_init_guess=None, verbose=False):
        '''Perform CRVE adaptive clustering refinement.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        target_clusters : list
            List containing the labels (int) of clusters to be refined.
        cluster_dicts : list
            List containing cluster-label-keyd dictionaries (key, str) that will be updated
            as the result of the CRVE adaptive refinement.
        inc : int
            Macroscale loading increment.
        improved_init_guess : list, default=None
            List that allows an improved initial iterative guess for the clusters
            incremental strain global vector (matricial form) after the clustering
            adaptivity is carried out. Index 0 contains a flag which is True if an improved
            initial iterative guess is to be computed, False otherwise. Index 1 contains the
            improved incremental strain global vector (matricial form) if computation flag
            is True, otherwise is None.
        verbose : bool, default=False
            Enable verbose output.
        '''
        # Set initial time
        init_time = time.time()
        # Output execution data
        if verbose:
            info.displayinfo('5', 'Clustering adaptivity refinement:')
            info.displayinfo('5', 'A - Performing clustering adaptivity...', 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform CRVE adaptive clustering refinement
        if not target_clusters:
            return
        else:
            # Store previous clustering cluster labels and total number of clusters
            phase_clusters_old = copy.deepcopy(crve.phase_clusters)
            n_total_clusters_old = crve.get_n_total_clusters()
            # Perform CRVE adaptive clustering refinement
            adaptive_clustering_map = crve.perform_crve_adaptivity(target_clusters)
            # Increment current increment adaptive step
            if str(inc) in self.inc_adaptive_steps.keys():
                self.inc_adaptive_steps[str(inc)] += 1
            else:
                self.inc_adaptive_steps[str(inc)] = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            a_time = time.time() - init_time
            info.displayinfo('5', 'B - Computing cluster interaction tensors...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute CRVE cluster interaction tensors
        crve.compute_cit(mode='adaptive', adaptive_clustering_map=adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            b_time = time.time() - ref_time
            info.displayinfo('5', 'C - Updating cluster-related quantities...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Loop over material phase target clusters
            for target_cluster in adaptive_clustering_map[mat_phase].keys():
                # Get list of target's child clusters
                child_clusters = adaptive_clustering_map[mat_phase][target_cluster]
                # Loop over cluster-keyd dictionaries
                for cluster_dict in cluster_dicts:
                    # Loop over child clusters and build their items
                    for child_cluster in child_clusters:
                        cluster_dict[str(child_cluster)] = \
                            copy.deepcopy(cluster_dict[target_cluster])
                    # Remove target cluster item
                    cluster_dict.pop(target_cluster)
            # Check material phase adaptivity lock status. If material phase adaptivity is
            # deactivated, then set the associated clustering adaptivity frequency to zero
            if crve._cluster_phases[mat_phase].adaptivity_lock:
                self._clust_adapt_freq[mat_phase] = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get improved initial iterative guess flag
        if improved_init_guess is None:
            is_improved_init_guess = False
        else:
            is_improved_init_guess = improved_init_guess[0]
            if not is_improved_init_guess:
                improved_init_guess[1] = None
        # Compute improved initial iterative guess for the clusters incremental strain
        # global vector (matricial form)
        if is_improved_init_guess:
            # Get CRVE material phases and total number of clusters
            material_phases = crve.get_material_phases()
            n_total_clusters = crve.get_n_total_clusters()
            comp_order = crve.get_comp_order()
            # Get previous clustering clusters incremental strain global vector
            if len(improved_init_guess[1]) != n_total_clusters_old*len(comp_order):
                raise RuntimeError('Unexpected size of previous clustering clusters '
                                   'incremental strain global vector.')
            else:
                gbl_inc_strain_mf_old = copy.deepcopy(improved_init_guess[1])
            # Initialize new clusters incremental strain global vector
            gbl_inc_strain_mf_new = np.zeros((n_total_clusters*len(comp_order)))
            # Initialize material phase initial index in global vector
            np.set_printoptions(linewidth=50)
            mat_phase_init_idx_old = 0
            mat_phase_init_idx_new = 0
            # Loop over material phases
            for mat_phase in material_phases:
                # Loop over previous clustering cluster labels
                for cluster in phase_clusters_old[mat_phase]:
                    # Get previous clustering cluster initial index
                    init_idx_old = mat_phase_init_idx_old + \
                        phase_clusters_old[mat_phase].index(cluster)*len(comp_order)
                    # Build new clusters incremental strain global vector. If cluster
                    # remained unchanged after the clustering adaptive step, then simply
                    # transfer the associated values to the proper position in the new
                    # global vector. If cluster has been refined, then copy the associated
                    # values to its child clusters positions in the new global vector.
                    if cluster in crve.phase_clusters[mat_phase]:
                        # Get new clustering cluster initial index
                        init_idx_new = mat_phase_init_idx_new + \
                            crve.phase_clusters[mat_phase].index(cluster)*len(comp_order)
                        # Transfer cluster incremental strain
                        gbl_inc_strain_mf_new[init_idx_new:init_idx_new+len(comp_order)] = \
                            gbl_inc_strain_mf_old[init_idx_old:init_idx_old+len(comp_order)]
                    else:
                        # Get list of target's child clusters
                        child_clusters = adaptive_clustering_map[mat_phase][str(cluster)]
                        # Loop over child clusters
                        for child_cluster in child_clusters:
                            # Get new clustering child cluster initial index
                            init_idx_new = mat_phase_init_idx_new + \
                                crve.phase_clusters[mat_phase].index(
                                    child_cluster)*len(comp_order)
                            # Copy parent cluster incremental strain
                            gbl_inc_strain_mf_new[init_idx_new:
                                                  init_idx_new+len(comp_order)] = \
                                gbl_inc_strain_mf_old[init_idx_old:
                                                      init_idx_old+len(comp_order)]
                # Update material phase initial index in global vector
                mat_phase_init_idx_old += \
                    len(phase_clusters_old[mat_phase])*len(comp_order)
                mat_phase_init_idx_new += \
                    len(crve.phase_clusters[mat_phase])*len(comp_order)
            # Store improved initial iterative guess for the clusters incremental strain
            # global vector (matricial form)
            improved_init_guess[1] = gbl_inc_strain_mf_new
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            c_time = time.time() - ref_time
            info.displayinfo('5', 'D - Updating adaptive cluster groups...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update adaptive cluster groups
        self._update_group_clusters(adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            d_time = time.time() - ref_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the ACRVE adaptive procedures
        dtime = time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in clustering adaptivity procedures
        self.adaptive_time += time.time() - init_time
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            # Build adaptive material phase's target clusters list
            output_list = []
            output_total = [0, 0]
            # Loop over adaptive material phases
            for mat_phase in self._adapt_material_phases:
                # Loop over material phase target clusters
                n_new_clusters = 0
                for target_cluster in adaptive_clustering_map[mat_phase].keys():
                    n_new_clusters += \
                        len(adaptive_clustering_map[mat_phase][target_cluster])
                n_total_clusters = len(crve.phase_clusters[mat_phase])
                output_list += [mat_phase, n_new_clusters, n_total_clusters]
                output_total[0] += n_new_clusters
                output_total[1] += n_total_clusters
            # Output adaptive phases new clusters summary table
            indent = 10*' '
            info.displayinfo('5', 'Summary:' + '\n\n' +
                             indent + 'Phase   New Clusters   Total Clusters' + '\n' +
                             indent + 37*'-' + '\n' +
                             ((indent + '{:^5s}{:>11d}{:>15d}\n')*
                             (len(self._adapt_material_phases))).format(*output_list) +
                             indent + 37*'-' + '\n' +
                             indent + '{:^5s}'.format('Total') +
                             '{:>11d}{:>15d}'.format(*output_total),
                             2)
            # Output adaptive phases execution time table
            indent = 10*' '
            info.displayinfo('5', 'Execution times (s):' + '\n\n' +
                             indent + '           Time(s)        %' + '\n' +
                             indent + 28*'-' + '\n' +
                             indent + '{:^5s}'.format('A') + '{:^18.4e}'.format(a_time) +
                             '{:>5.2f}'.format((a_time/dtime)*100) + '\n' +
                             indent + '{:^5s}'.format('B') + '{:^18.4e}'.format(b_time) +
                             '{:>5.2f}'.format((b_time/dtime)*100) + '\n' +
                             indent + '{:^5s}'.format('C') + '{:^18.4e}'.format(c_time) +
                             '{:>5.2f}'.format((c_time/dtime)*100) + '\n' +
                             indent + '{:^5s}'.format('D') + '{:^18.4e}'.format(c_time) +
                             '{:>5.2f}'.format((d_time/dtime)*100) + '\n' +
                             indent + 28*'-' + '\n' +
                             indent + '{:^5s}'.format('Total') + '{:>14.4e}'.format(dtime),
                             2)
    # --------------------------------------------------------------------------------------
    def check_inc_adaptive_steps(self, inc):
        '''Check number of clustering adaptive steps performed in current loading increment.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.

        Returns
        -------
        bool
            True if maximum number of clustering adaptive steps has not been reached, False
            otherwise.
        '''
        if str(inc) in self.inc_adaptive_steps.keys():
            if self.inc_adaptive_steps[str(inc)] >= self.max_inc_adaptive_steps:
                return False
            else:
                return True
        else:
            self.inc_adaptive_steps[str(inc)] = 0
            return True
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
        for mat_phase in self._adapt_material_phases:
            # Loop over adaptive cluster groups
            for group_id in self._target_groups_ids[mat_phase]:
                # Get parent adaptive cluster group clusters
                old_clusters = self._adapt_groups[mat_phase][group_id].copy()
                # Loop over adaptive cluster group clusters
                for old_cluster in old_clusters:
                    # If cluster has been adapted, then update adaptive cluster group
                    if str(old_cluster) in adaptive_clustering_map[mat_phase].keys():
                        # Remove parent cluster from group
                        self._adapt_groups[mat_phase][group_id].remove(old_cluster)
                        # Add child clusters to group
                        new_clusters = \
                            adaptive_clustering_map[mat_phase][str(old_cluster)]
                        self._adapt_groups[mat_phase][group_id] += new_clusters
    # --------------------------------------------------------------------------------------
    def _get_activated_adaptive_phases(self, inc):
        '''Get activated adaptive material phases for a given incremental counter.

        Parameters
        ----------
        inc : int
            Incremental counter serving as a reference basis for the clustering adaptivity
            frequency control.

        Returns
        -------
        activated_adapt_phases : list
            List of adaptive cluster-reduced material phases (str) whose associated
            adaptivity procedures are to be performed in the current state of the
            incremental counter.
        '''
        # Check incremental counter
        if not ioutil.checkposint(inc):
            raise RuntimeError('Incremental counter must be positive integer.')
        # Initialize list of activated adaptive material phases
        activated_adapt_phases = []
        # Loop over adaptive material phases
        for mat_phase in self._clust_adapt_freq.keys():
            # Append activated material phase
            if self._clust_adapt_freq[mat_phase] == 0:
                continue
            elif inc % self._clust_adapt_freq[mat_phase] == 0:
                activated_adapt_phases.append(mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return activated_adapt_phases
    # --------------------------------------------------------------------------------------
    def get_adapt_vtk_array(self, voxels_clusters):
        '''Get regular grid array containing the adaptive level associated to each cluster.

        All the clusters belonging to the same adaptive cluster group share the same
        adaptive level and that the adaptive level 0 is associated to the base clustering.

        Parameters
        ----------
        voxels_clusters : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the cluster label (int) assigned to the corresponding pixel/voxel.

        Notes
        -----
        Such regular grid array is only being currently employed to write the
        post-processing VTK output files.
        '''
        # Initialize flattened regular grid array
        rg_array = np.zeros(voxels_clusters.shape).flatten('F')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get adaptive material phase cluster groups
            adapt_groups_ids = list(self._adapt_groups[mat_phase].keys())
            # Loop over adaptive cluster groups
            for group_id in adapt_groups_ids:
                # Get adaptive cluster group adaptive level
                adapt_level = self._groups_adapt_level[mat_phase][group_id]
                # Get adaptive cluster group clusters
                adapt_group = self._adapt_groups[mat_phase][group_id]
                # Get flat indexes associated with the adaptive cluster group clusters
                flat_idxs = np.in1d(voxels_clusters.flatten('F'), adapt_group)
                # Store adaptive cluster group adaptive level
                rg_array[flat_idxs] = adapt_level
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build regular grid array
        rg_array = rg_array.reshape(voxels_clusters.shape, order='F')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return rg_array
# ------------------------------------------------------------------------------------------
#
#                                               CRVE clustering adaptivity output file class
# ==========================================================================================
class ClusteringAdaptivityOutput:
    '''Clustering adaptivity output.

    Attributes
    ----------
    _header : list
        List containing the header of each column (str).
    _col_width : int
        Output file column width.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, adapt_file_path, adapt_material_phases):
        '''Clustering adaptivity output constructor.

        Parameters
        ----------
        adapt_file_path : str
            Path of clustering adaptivity output file.
        adapt_material_phases : list
            RVE adaptive material phases labels (str).
        '''
        self._adapt_file_path = adapt_file_path
        self._adapt_material_phases = adapt_material_phases
        # Set clustering adaptivity output file header
        self._header = ['Increment', 'total_adapt_time', 'eval_adapt_time',
                        'clust_adapt_time', 'cit_adapt_time']
        for mat_phase in self._adapt_material_phases:
            self._header += ['n_clusters_' + mat_phase, 'adapt_step_' + mat_phase,
                             'adapt_time_' + mat_phase]
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # --------------------------------------------------------------------------------------
    def write_adapt_file(self, inc, adaptivity_manager, crve, mode='increment'):
        '''Write clustering adaptivity output file.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.
        adaptivity_manager : AdaptivityManager
            CRVE clustering adaptivity manager.
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        mode : {'init', 'increment'}, default='increment'
            Clustering adaptivity output mode. Mode 'init' writes the file header and the
            increment 0 (base clustering), while 'increment' appends the clustering
            adaptivity data associated to macroscale loading increment.
        '''
        write_list = []
        if mode == 'init':
            # Open clustering adaptivity output file (write mode)
            adapt_file = open(self._adapt_file_path, 'w')
            # Set clustering adaptivity output file header format structure
            write_list += ['{:>9s}'.format(self._header[0]) +
                           ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                           for x in self._header[1:]]),]
        elif mode == 'increment':
            # Open clustering adaptivity output file (append mode)
            adapt_file = open(self._adapt_file_path, 'a')
        else:
            raise RuntimeError('Unknown clustering adaptivity output mode.')
        # Build adaptive material phases output list
        adaptivity_output = crve.get_adaptivity_output()
        output_list = []
        for mat_phase in self._adapt_material_phases:
            output_list += adaptivity_output[mat_phase]
        # Set clustering adaptivity output file increment format structure
        write_list += ['\n' + '{:>9d}'.format(inc) +
                       ('{:>' + str(self._col_width) + '.8e}').format(
                           adaptivity_manager.adaptive_time) +
                       ('{:>' + str(self._col_width) + '.8e}').format(
                            adaptivity_manager.adaptive_evaluation_time) +
                       ('{:>' + str(self._col_width) + '.8e}').format(
                           crve.adaptive_clustering_time) +
                       ('{:>' + str(self._col_width) + '.8e}').format(
                           crve.adaptive_cit_time) +
                       ''.join([''.join([('{:>' + str(self._col_width) + 'd}').format(x)
                                for x in output_list[3*i:3*i+2]] +
                               [('{:>' + str(self._col_width) + '.8e}').format(
                               output_list[3*i+2])])
                       for i in range(len(self._adapt_material_phases))])]
        # Write clustering adaptivity output file
        adapt_file.writelines(write_list)
        # Close clustering adaptivity output file
        adapt_file.close()
