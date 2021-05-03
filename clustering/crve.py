#
# Cluster-Reduced Representative Volume Element Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the generation of the Cluster-Reduced Representative Volume Element
# (CRVE), a key step in the so called clustering-based reduced order models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2020 | Merged cluster interaction tensors computation methods.
# Bernardo P. Ferreira | Dec 2020 | Reformulated CRVE class (cluster-reduced mat. phases).
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
# Date and time
import time
# Python object serialization
import pickle
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# Cluster interaction tensors operations
import clustering.citoperations as citop
# Cluster-Reduced material phases
from clustering.clusteringphase import SCRMP, GACRMP, HAACRMP
#
#                                                                                 CRVE class
# ==========================================================================================
class CRVE:
    '''Cluster-Reduced Representative Volume Element.

    Attributes
    ----------
    _n_dim : int
        Problem dimension.
    _n_voxels_dims : list
        Number of voxels in each dimension of the regular grid (spatial discretization of
        the RVE).
    _n_voxels : int
        Total number of voxels of the regular grid (spatial discretization of the RVE).
    _phase_voxel_flatidx : dict
        Flat (1D) voxels' indexes (item, list) associated to each material phase (key, str).
    _gop_X_dft_vox : list
        Green operator material independent terms. Each term is stored in a dictionary,
        where each pair of strain/stress components (key, str) is associated with the Green
        operator material independent term evaluated in all spatial discrete points
        (item, ndarray).
    _cluster_phases : dict
        Cluster-Reduced material phase instance (item, CRMP) associated to each material
        phase (key, str).
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base clustering.
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    cit_X_mf : list
        Cluster interaction tensors associated to the Green operator material independent
        terms. Each term is stored in a dictionary (item, dict) for each pair of material
        phases (key, str), which in turn contains the corresponding matricial form
        (item, ndarray) associated to each pair of clusters (key, str).
    adapt_material_phases : list
        RVE adaptive material phases labels (str).
    adaptive_clustering_time : float
        Total amount of time (s) spent in clustering adaptivity.
    adaptive_cit_time : float
        Total amount of time (s) spent in clustering adaptivity cluster interaction tensors
        computation procedures.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, rve_dims, regular_grid, material_phases, comp_order,
                 global_data_matrix, clustering_type, phase_n_clusters,
                 base_clustering_scheme, adaptive_clustering_scheme=None,
                 adaptivity_criterion=None, adaptivity_type=None,
                 adaptivity_control_feature=None):
        '''Cluster-Reduced Representative Volume Element constructor.

        Parameters
        ----------
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        comp_order : list
            Strain/Stress components (str) order.
        global_data_matrix: ndarray of shape (n_voxels, n_features_dims)
            Data matrix containing the required clustering features' data to perform all
            the prescribed cluster analyses.
        clustering_type : dict
            Clustering type (item, {'static', 'adaptive'}) of each material phase
            (key, str).
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, ndarry of shape (n_clusterings, 3)) for
            each material phase (key, str). Each row is associated with a unique clustering
            characterized by a clustering algorithm (col 1, int), a list of features
            (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, ndarry of shape (n_clusterings, 3))
            for each material phase (key, str). Each row is associated with a unique
            clustering characterized by a clustering algorithm (col 1, int), a list of
            features (col 2, list of int) and a list of the features data matrix' indexes
            (col 3, list of int).
        adaptivity_criterion : dict, default=None
            Clustering adaptivity criterion (item, dict) associated to each material phase
            (key, str). This dictionary contains the adaptivity criterion to be used and the
            required parameters.
        adaptivity_type : dict, default=None
            Clustering adaptivity type (item, dict) associated to each material phase
            (key, str). This dictionary contains the adaptivity type to be used and the
            required parameters.
        adaptivity_control_feature : dict, default=None
            Clustering adaptivity control feature (item, str) associated to each material
            phase (key, str).
        '''
        self._rve_dims = rve_dims
        self._material_phases = material_phases
        self._comp_order = comp_order
        self._global_data_matrix = global_data_matrix
        self._clustering_type = clustering_type
        self._phase_n_clusters = phase_n_clusters
        self._base_clustering_scheme = base_clustering_scheme
        self._adaptive_clustering_scheme = adaptive_clustering_scheme
        self._adaptivity_type = adaptivity_type
        self._gop_X_dft_vox = None
        self._cluster_phases = None
        self._adaptive_step = 0
        self.voxels_clusters = None
        self.phase_clusters = None
        self.clusters_f = None
        self.cit_X_mf = None
        self.adaptivity_control_feature = adaptivity_control_feature
        self.adaptivity_criterion = adaptivity_criterion
        self.adaptive_clustering_time = 0
        self.adaptive_cit_time = 0
        # Get number of dimensions
        self._n_dim = len(rve_dims)
        # Get number of voxels on each dimension and total number of voxels
        self._n_voxels_dims = \
            [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        self._n_voxels = np.prod(self._n_voxels_dims)
        # Get material phases' voxels' 1D flat indexes
        self._phase_voxel_flatidx = \
            type(self)._get_phase_idxs(regular_grid, material_phases)
        # Get adaptive material phases
        self.adapt_material_phases = [x for x in self._clustering_type.keys()
                                      if self._clustering_type[x] == 'adaptive']
    # --------------------------------------------------------------------------------------
    def perform_crve_base_clustering(self):
        '''Compute CRVE base clustering.'''
        info.displayinfo('5', 'Computing CRVE base clustering...')
        # Initialize base clustering labels
        labels = np.full(self._n_voxels, -1, dtype=int)
        # Initialize cluster-reduced material phases dictionary
        self._cluster_phases = {}
        # Initialize minimum cluster label
        min_label = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            info.displayinfo('5', 'Computing material phase ' + mat_phase +
                             ' base clustering...', 2)
            # Get material phase clustering type
            ctype = self._clustering_type[mat_phase]
            # Get material phase initial number of clusters
            n_phase_clusters = self._phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase's voxels' indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get base clustering features' indexes
            features_idxs = []
            features_idxs += \
                self._get_features_indexes(self._base_clustering_scheme[mat_phase])
            # Get adaptive clustering features' indexes
            if ctype == 'adaptive':
                features_idxs += \
                    self._get_features_indexes(self._adaptive_clustering_scheme[mat_phase])
            # Get unique clustering features' indexes
            features_idxs = list(set(features_idxs))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase cluster data matrix containing the required data to perform
            # all the prescribed cluster analyses
            cluster_data_matrix = mop.getcondmatrix(self._global_data_matrix,
                                                    voxels_idxs, features_idxs)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Instatiate cluster-reduced material phase
            if ctype == 'static':
                # Instantiate static cluster-reduced material phase
                crmp = SCRMP(mat_phase, cluster_data_matrix, n_phase_clusters)
            elif ctype == 'adaptive':
                # Get material phase adaptivity type
                atype = self._adaptivity_type[mat_phase]['adapt_type']
                # Instantiate adaptive cluster-reduced material phase
                if atype == GACRMP:
                    # Instantiate generalized adaptive cluster-reduced material phase
                    crmp = GACRMP(mat_phase, cluster_data_matrix, n_phase_clusters,
                                  self._adaptivity_type[mat_phase])
                elif atype == HAACRMP:
                    # Instantiate hierarchical-agglomerative adaptive cluster-reduced
                    # material phase
                    crmp = HAACRMP(mat_phase, cluster_data_matrix, n_phase_clusters,
                                   self._adaptivity_type[mat_phase])
                else:
                    raise RuntimeError('Unknown adaptivity type.')
            else:
                raise RuntimeError('Unknown material phase clustering type.')
            # Store cluster-reduced material phase
            self._cluster_phases[mat_phase] = crmp
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase base clustering scheme
            clustering_scheme = self._base_clustering_scheme[mat_phase]
            # Perform material phase base clustering
            crmp.perform_base_clustering(clustering_scheme, min_label)
            # Update minimum cluster label
            min_label = crmp.max_label + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check for duplicated cluster labels
            if not set(crmp.cluster_labels).isdisjoint(labels):
                raise RuntimeError('Duplicated cluster labels between different ' +
                                   'material phases.')
            # Assemble material phase cluster labels
            labels[voxels_idxs] = crmp.cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # Compute base CRVE descriptors
        info.displayinfo('5', 'Computing CRVE base clustering descriptors...', 2)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def perform_crve_adaptivity(self, target_clusters):
        '''Perform CRVE clustering adaptivity.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be adapted.

        Returns
        -------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str).
        '''
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        if not set(target_clusters).issubset(set(self.voxels_clusters.flatten())):
            raise RuntimeError('List of target clusters contains unexistent labels.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering refinement step counter
        self._adaptive_step += 1
        # Get CRVE current clustering
        labels = self.voxels_clusters.flatten()
        # Initialize adaptive material phase's target clusters
        phase_target_clusters = {mat_phase: [] for mat_phase in self.adapt_material_phases}
        # Initialize maximum cluster label
        max_label = -1
        # Get maximum cluster label and build adaptive material phase's target clusters list
        for mat_phase in self._material_phases:
            # Get adaptive cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Get maximum cluster label
            max_label = max(max_label, crmp.max_label)
            # Build adaptive material phase's target clusters lists
            if mat_phase in self.adapt_material_phases:
                phase_target_clusters[mat_phase] = \
                    list(set(target_clusters).intersection(self.phase_clusters[mat_phase]))
        # Set minimum cluster label
        min_label = max_label + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize adaptive clustering map
        adaptive_clustering_map = \
            {mat_phase: {} for mat_phase in self.adapt_material_phases}
        # Loop over adaptive material phases
        for mat_phase in self.adapt_material_phases:
            # If there are no target clusters, skip to next adaptive material phase
            if not bool(phase_target_clusters[mat_phase]):
                continue
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Get adaptive clustering scheme
            adaptive_clustering_scheme = self._adaptive_clustering_scheme[mat_phase]
            # Perform adaptive clustering
            adaptive_clustering_map[mat_phase] = \
                crmp.perform_adaptive_clustering(phase_target_clusters[mat_phase],
                                                 adaptive_clustering_scheme, min_label)
            # Update minimum cluster label
            min_label = crmp.max_label + 1
            # Update CRVE clustering
            labels[self._phase_voxel_flatidx[mat_phase]] = crmp.cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build adaptive CRVE
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in clustering adaptivity
        self.adaptive_clustering_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # --------------------------------------------------------------------------------------
    def get_adaptivity_output(self):
        '''Get required data for clustering adaptivity output file.

        Returns
        -------
        adaptivity_output : dict
            For each adaptive material phase (key, str), stores a list (item) containing the
            adaptivity metrics associated to the clustering adaptivity output file.
        '''
        # Initialize adaptivity output
        adaptivity_output = {}
        # Loop over adaptive material phases
        for mat_phase in self.adapt_material_phases:
            # Get adaptive cluster reduced material phase
            acrmp = self._cluster_phases[mat_phase]
            # Get material phase adaptivity metrics
            adaptivity_output[mat_phase] = \
                [*acrmp.get_adaptive_output()]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptivity_output
    # --------------------------------------------------------------------------------------
    def get_clustering_summary(self):
        '''Get summary of base and final number of clusters of each material phase.

        Returns
        -------
        clustering_summary : dict
            For each material phase (key, str), stores list (item) containing the associated
            type ('static' or 'adaptive'), the base number of clusters (int) and the final
            number of clusters (int).
        '''
        # Initialize clustering summary
        clustering_summary = {}
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Build material phase clustering summary
            clustering_summary[mat_phase] = \
                [self._clustering_type[mat_phase], self._phase_n_clusters[mat_phase],
                 self._cluster_phases[mat_phase].get_n_clusters()]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return clustering_summary
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_crmp_types():
        '''Get available cluster-reduced material phases types.

        Returns
        -------
        available_crmp_types : dict
            Available cluster-reduced material phase classes (item, CRMP) and associated
            identifiers (key, str).
        '''
        # Set available cluster-reduced material phase types
        available_crmp_types = {'0': SCRMP,
                                '1': GACRMP,
                                '2': HAACRMP}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_crmp_types
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_features_indexes(clustering_scheme):
        '''Get list of unique clustering features indexes from a given clustering scheme.

        Parameters
        ----------
        clustering_scheme : ndarry of shape (n_clusterings, 3)
            Clustering scheme. Each row is associated with a unique clustering characterized
            by a clustering algorithm (col 1, int), a list of features (col 2, list of int)
            and a list of the features data matrix' indexes (col 3, list of int).

        Returns
        -------
        indexes : list
            List of unique clustering features indexes.
        '''
        # Collect prescribed clusterings features' indexes
        indexes = []
        for i in range(clustering_scheme.shape[0]):
            indexes += clustering_scheme[i, 2]
        # Get list of unique clustering features' indexes
        indexes = list(set(indexes))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return indexes
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_phase_idxs(regular_grid, material_phases):
        '''Get flat indexes of each material phase's voxels.

        Parameters
        ----------
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).

        Returns
        -------
        phase_voxel_flat_idx : dict
            Flat voxels' indexes (item, list of int) associated to each material phase
            (key, str).
        '''
        phase_voxel_flat_idx = dict()
        # Loop over material phases
        for mat_phase in material_phases:
            # Build boolean 'belongs to material phase' list
            is_phase_list = regular_grid.flatten() == int(mat_phase)
            # Get material phase's voxels' indexes
            phase_voxel_flat_idx[mat_phase] = list(it.compress(range(len(is_phase_list)),
                                                              is_phase_list))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return phase_voxel_flat_idx
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_cluster_idxs(voxels_clusters, cluster_label):
        '''Get flat indexes of given cluster's voxels.

        Parameters
        ----------
        voxels_clusters : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the cluster label (int) assigned to the corresponding pixel/voxel.
        cluster_label : int
            Cluster label.

        Returns
        -------
        cluster_voxel_flat_idx : list
            Flat voxels' indexes (int) associated to given material cluster.
        '''
        # Build boolean 'belongs to cluster' list
        is_cluster_list = voxels_clusters.flatten() == int(cluster_label)
        # Get material phase's voxels' indexes
        cluster_voxel_flat_idx = \
            list(it.compress(range(len(is_cluster_list)), is_cluster_list))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_voxel_flat_idx
    # --------------------------------------------------------------------------------------
    def _set_phase_clusters(self):
        '''Set CRVE cluster labels associated to each material phase.'''
        self.phase_clusters = {}
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get cluster labels
            self.phase_clusters[mat_phase] = \
                np.unique(self.voxels_clusters.flatten()[
                          self._phase_voxel_flatidx[mat_phase]])
    # --------------------------------------------------------------------------------------
    def _set_clusters_vf(self):
        '''Set CRVE clusters' volume fractions.'''
        # Compute voxel volume
        voxel_vol = np.prod([float(self._rve_dims[i])/self._n_voxels_dims[i]
                             for i in range(len(self._rve_dims))])
        # Compute RVE volume
        rve_vol = np.prod(self._rve_dims)
        # Compute volume fraction associated to each material cluster
        self.clusters_f = {}
        for cluster in np.unique(self.voxels_clusters):
            n_voxels_cluster = np.sum(self.voxels_clusters == cluster)
            self.clusters_f[str(cluster)] = (n_voxels_cluster*voxel_vol)/rve_vol
    # --------------------------------------------------------------------------------------
    def _sort_cluster_labels(self):
        '''Reassign and sort CRVE cluster labels material phasewise.

        Reassign CRVE cluster labels in the range (0, n_clusters) and sort them in
        ascending order of material phase's labels.

        Notes
        -----
        This method is not being currently used.
        '''
        # Initialize material phase initial cluster label
        lbl_init = 0
        # Initialize old cluster labels
        old_clusters = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize mapping dictionary to sort the cluster labels in asceding order of
        # material phase
        sort_dict = dict()
        # Loop over material phases sorted in ascending order
        sorted_mat_phases = list(np.sort(list(self._material_phases)))
        for mat_phase in sorted_mat_phases:
            # Get material phase old cluster labels
            phase_old_clusters = np.unique(
                self.voxels_clusters.flatten()[self._phase_voxel_flatidx[mat_phase]])
            # Set material phase new cluster labels
            phase_new_clusters = list(range(lbl_init,
                                      lbl_init + self._phase_n_clusters[mat_phase]))
            # Build mapping dictionary to sort the cluster labels
            for i in range(self._phase_n_clusters[mat_phase]):
                if phase_old_clusters[i] in sort_dict.keys():
                    raise RuntimeError('Cluster label (key) already exists in cluster' +
                                       'labels mapping dictionary.')
                else:
                    sort_dict[phase_old_clusters[i]] = phase_new_clusters[i]
            # Set next material phase initial cluster label
            lbl_init = lbl_init + self._phase_n_clusters[mat_phase]
            # Append old cluster labels
            old_clusters += list(phase_old_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check cluster labels mapping dictionary
        new_clusters = [sort_dict[key] for key in sort_dict.keys()]
        if set(sort_dict.keys()) != set(old_clusters) or \
                len(set(new_clusters)) != len(set(old_clusters)):
            raise RuntimeError('Invalid cluster labels mapping dictionary.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort cluster labels in ascending order of material phase
        for voxel_idx in it.product(*[list(range(self._n_voxels_dims[i])) \
                for i in range(len(self._n_voxels_dims))]):
            self.voxels_clusters[voxel_idx] = sort_dict[self.voxels_clusters[voxel_idx]]
    # --------------------------------------------------------------------------------------
    def compute_cit(self, mode='full', adaptive_clustering_map=None):
        '''Compute CRVE cluster interaction tensors.

        Parameters
        ----------
        mode : str, {'full', 'adaptive'}, default='full'
            The default `full` mode performs the complete computation of all cluster
            interaction tensors. The 'adaptive' mode speeds up the computation of the new
            cluster interaction tensors resulting from an adaptive clustering characterized
            by `adaptive_clustering_map`.
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str). Required if `mode='adaptive'`, otherwise
            ignored.

        Notes
        -----
        The cluster interaction tensors \'adaptive\' computation mode can only be performed
        after at least one \'full\' computation has been performed.
        '''
        # Check parameters
        if mode not in ['full', 'adaptive']:
            raise RuntimeError('Unknown mode to compute cluster interaction tensors.')
        elif mode == 'adaptive' and adaptive_clustering_map is None:
            raise RuntimeError('Adaptive clustering map must be provided in \'adaptive\' ' +
                               'mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform mode-specific initialization procedures
        if mode == 'full':
            info.displayinfo('5', 'Computing CRVE cluster interaction tensors...')
            # Initialize cluster interaction tensors dictionary
            self.cit_X_mf = [{} for i in range(3)]
            for mat_phase_B in self._material_phases:
                for mat_phase_A in self._material_phases:
                    for i in range(len(self.cit_X_mf)):
                        self.cit_X_mf[i][mat_phase_A + '_' + mat_phase_B] = {}
            # Compute Green operator material independent terms
            self._gop_X_dft_vox = citop.gop_matindterms(self._n_dim, self._rve_dims,
                                                        self._comp_order,
                                                        self._n_voxels_dims)
        elif mode == 'adaptive':
            init_time = time.time()
            # Build lists with old (preexistent) clusters and new (adapted) clusters for
            # each material phase. Also get list of clusters that no longer exist due to
            # clustering adaptivity
            phase_old_clusters = {}
            phase_new_clusters = {}
            pop_clusters = []
            for mat_phase in self._material_phases:
                if mat_phase in adaptive_clustering_map.keys():
                    phase_new_clusters[mat_phase] = \
                        sum(adaptive_clustering_map[mat_phase].values(), [])
                    pop_clusters += list(adaptive_clustering_map[mat_phase].keys())
                else:
                    phase_new_clusters[mat_phase] = []
                phase_old_clusters[mat_phase] = list(set(self.phase_clusters[mat_phase]) -
                                                     set(phase_new_clusters[mat_phase]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase_B in self._material_phases:
            # Set material phase B clusters to be looped over
            if mode == 'full':
                clusters_J = self.phase_clusters[mat_phase_B]
            elif mode == 'adaptive':
                clusters_J = phase_new_clusters[mat_phase_B]
            # Loop over material phase B clusters
            for cluster_J in clusters_J:
                # Set material phase B cluster characteristic function
                _, cluster_J_filter_dft = self._cluster_filter(cluster_J)
                # Perform discrete convolution between the material phase B cluster
                # characteristic function and each of Green operator material independent
                # terms
                gop_X_filt_vox = self._gop_convolution(cluster_J_filter_dft,
                                                       *self._gop_X_dft_vox)
                # Loop over material phases
                for mat_phase_A in self._material_phases:
                    # Set material phase pair dictionary
                    mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                    # Loop over material phase A clusters
                    for cluster_I in self.phase_clusters[mat_phase_A]:
                        # Set material cluster pair
                        cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                        # Check if cluster-symmetric cluster interaction tensor
                        sym_cluster_pair = self._switch_pair(cluster_pair)
                        sym_mat_phase_pair = self._switch_pair(mat_phase_pair)
                        is_clst_sym = sym_cluster_pair in \
                            self.cit_X_mf[0][sym_mat_phase_pair].keys()
                        # Compute cluster interaction tensor between material phase A
                        # cluster and material phase B cluster (complete computation or
                        # cluster-symmetric computation)
                        if is_clst_sym:
                            # Set cluster volume fractions ratio
                            clst_f_ratio = self.clusters_f[str(cluster_J)]/ \
                                self.clusters_f[str(cluster_I)]
                            # Compute clustering interaction tensor between material phase A
                            # cluster and material phase B cluster through cluster-symmetry
                            for cit_mf in self.cit_X_mf:
                                cit_mf[mat_phase_pair][cluster_pair] = \
                                    np.multiply(clst_f_ratio,
                                        cit_mf[sym_mat_phase_pair][sym_cluster_pair])
                        else:
                            # Set material phase A cluster characteristic function
                            cluster_I_filter, _ = self._cluster_filter(cluster_I)
                            # Perform discrete integral over the spatial domain of material
                            # phase A cluster I
                            cit_X_integral_mf = \
                                self._discrete_cit_integral(cluster_I_filter,
                                                            *gop_X_filt_vox)
                            # Compute cluster interaction tensor between the material phase
                            # A cluster and the material phase B cluster
                            rve_vol = np.prod(self._rve_dims)
                            factor = 1.0/(self.clusters_f[str(cluster_I)]*rve_vol)
                            for i in range(len(self.cit_X_mf)):
                                self.cit_X_mf[i][mat_phase_pair][cluster_pair] = \
                                    np.multiply(factor, cit_X_integral_mf[i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute remaining adaptive cluster interaction tensors through cluster-symmetry
        # and remove vanished clustering interaction tensors
        if mode == 'adaptive':
            # Loop over material phases
            for mat_phase_B in self._material_phases:
                # Loop over material phase B old clusters
                for cluster_J in phase_old_clusters[mat_phase_B]:
                    # Loop over material phases
                    for mat_phase_A in self._material_phases:
                        # Set material phase pair dictionary
                        mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                        # Loop over material phase A new clusters
                        for cluster_I in phase_new_clusters[mat_phase_A]:
                            # Set material cluster pair
                            cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                            # Check if cluster-symmetric cluster interaction tensor
                            sym_cluster_pair = self._switch_pair(cluster_pair)
                            sym_mat_phase_pair = self._switch_pair(mat_phase_pair)
                            is_clst_sym = sym_cluster_pair in \
                                self.cit_X_mf[0][sym_mat_phase_pair].keys()
                            # Compute cluster interaction tensor between material phase A
                            # cluster and material phase B cluster through cluster-symmetry
                            if not is_clst_sym:
                                raise RuntimeError('All the remaining adaptive ' +
                                                   'clustering interaction tensors ' +
                                                   'should be cluster-symmetric.')
                            # Set cluster volume fractions ratio
                            clst_f_ratio = self.clusters_f[str(cluster_J)]/ \
                                self.clusters_f[str(cluster_I)]
                            # Compute clustering interaction tensor
                            for cit_mf in self.cit_X_mf:
                                cit_mf[mat_phase_pair][cluster_pair] = \
                                    np.multiply(clst_f_ratio,
                                        cit_mf[sym_mat_phase_pair][sym_cluster_pair])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phases
            for mat_phase_B in self._material_phases:
                # Loop over material phases
                for mat_phase_A in self._material_phases:
                    # Set material phase pair dictionary
                    mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                    # Set existent cluster interactions
                    cluster_pairs = [x for x in self.cit_X_mf[0][mat_phase_pair].keys()]
                    # Loop over cluster pairs
                    for cluster_pair in cluster_pairs:
                        cluster_I = cluster_pair.split('_')[0]
                        cluster_J = cluster_pair.split('_')[1]
                        # If any of the interacting clusters no longer exists, then remove
                        # the associated cluster interaction tensor
                        if cluster_I in pop_clusters or cluster_J in pop_clusters:
                            for i in range(len(self.cit_X_mf)):
                                self.cit_X_mf[i][mat_phase_pair].pop(cluster_pair)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update total amount of time spent in clustering adaptivity cluster interaction
            # tensors computation procedures
            self.adaptive_cit_time += time.time() - init_time
    # --------------------------------------------------------------------------------------
    def _cluster_filter(self, cluster):
        '''Compute cluster discrete characteristic function (spatial and frequency domains).

        Parameters
        ----------
        cluster : int
            Cluster label.

        Returns
        -------
        cluster_filter : ndarray
            Cluster discrete characteristic function in spatial domain.
        cluster_filter_dft : ndarray
            Cluster discrete characteristic function in frequency domain (discrete Fourier
            transform).
        '''
        # Check if valid cluster
        if not isinstance(cluster, int) and not isinstance(cluster, np.integer):
            raise RuntimeError('Cluster label must be an integer.')
        elif cluster not in self.voxels_clusters:
            raise RuntimeError('Cluster label does not exist in the CRVE.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build cluster filter (spatial domain)
        cluster_filter = self.voxels_clusters == cluster
        # Perform Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
        cluster_filter_dft = np.fft.fftn(cluster_filter)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return [cluster_filter, cluster_filter_dft]
    # --------------------------------------------------------------------------------------
    def _gop_convolution(self, cluster_filter_dft, gop_1_dft_vox, gop_2_dft_vox,
                           gop_0_freq_dft_vox):
        '''Compute convolution between cluster characteristic function and Green operator.

        Compute the discrete convolution required to compute the cluster interaction tensor
        between a material cluster I and a material cluster J. The convolution is performed
        in the frequency domain between the material J characteristic function and each of
        the Green operator material independent terms.

        Parameters
        ----------
        cluster_filter_dft : ndarray
            Cluster discrete characteristic function in frequency domain (discrete Fourier
            transform).
        gop_1_dft_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the first Green operator material
            independent term in the frequency domain (discrete Fourier transform).
        gop_2_dft_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the second Green operator material
            independent term in the frequency domain (discrete Fourier transform).
        gop_0_freq_dft_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the Green operator zero-frequency
            (material independent) term in the frequency domain (discrete Fourier
            transform).

        Returns
        -------
        gop_1_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the first Green operator material
            independent term in the spatial domain (inverse discrete Fourier transform).
        gop_2_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the second Green operator material
            independent term in the spatial domain (inverse discrete Fourier transform).
        gop_0_freq_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the zero-frequency Green operator (material
            independent) term in the spatial domain (inverse discrete Fourier transform).
        '''
        # Initialize discrete convolution (spatial and frequency domain)
        gop_1_filt_dft_vox = copy.deepcopy(gop_1_dft_vox)
        gop_2_filt_dft_vox = copy.deepcopy(gop_2_dft_vox)
        gop_0_freq_filt_dft_vox = copy.deepcopy(gop_0_freq_dft_vox)
        gop_1_filt_vox = copy.deepcopy(gop_1_dft_vox)
        gop_2_filt_vox = copy.deepcopy(gop_1_dft_vox)
        gop_0_freq_filt_vox = copy.deepcopy(gop_1_dft_vox)
        # Compute RVE volume and total number of voxels
        rve_vol = np.prod(self._rve_dims)
        n_voxels = np.prod(self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Green operator components
        for i in range(len(self._comp_order)):
            compi = self._comp_order[i]
            for j in range(len(self._comp_order)):
                compj = self._comp_order[j]
                # Perform discrete convolution in the frequency domain
                gop_1_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels), np.multiply(cluster_filter_dft,
                        gop_1_filt_dft_vox[compi + compj]))
                gop_2_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels), np.multiply(cluster_filter_dft,
                        gop_2_filt_dft_vox[compi + compj]))
                gop_0_freq_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels),np.multiply(cluster_filter_dft,
                        gop_0_freq_filt_dft_vox[compi + compj]))
                # Perform an Inverse Discrete Fourier Transform (IDFT) by means of Fast
                # Fourier Transform (FFT)
                gop_1_filt_vox[compi + compj] = \
                    np.real(np.fft.ifftn(gop_1_filt_dft_vox[compi + compj]))
                gop_2_filt_vox[compi + compj] = \
                    np.real(np.fft.ifftn(gop_2_filt_dft_vox[compi + compj]))
                gop_0_freq_filt_vox[compi + compj] = \
                    np.real(np.fft.ifftn(gop_0_freq_filt_dft_vox[compi + compj]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return [gop_1_filt_vox, gop_2_filt_vox, gop_0_freq_filt_vox]
    # --------------------------------------------------------------------------------------
    def _discrete_cit_integral(self, cluster_filter, gop_1_filt_vox, gop_2_filt_vox,
                               gop_0_freq_filt_vox):
        '''Compute discrete integral over the spatial domain of material cluster.

        In order to compute the cluster interaction tensor between material cluster I and
        material cluster J, compute the discrete integral over the spatial domain of
        material cluster I of the discrete convolution (characteristic function - Green
        operator) associated to material cluster J.

        Parameters
        ----------
        cluster_filter : ndarray
            Cluster discrete characteristic function in spatial domain.
        gop_1_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the first Green operator material
            independent term in the spatial domain (inverse discrete Fourier transform).
        gop_2_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the second Green operator material
            independent term in the spatial domain (inverse discrete Fourier transform).
        gop_0_freq_filt_vox : dict
            Regular grid shaped matrix (item, ndarray) containing each fourth-order
            matricial form component (key, str) of the convolution between the material
            cluster characteristic function and the zero-frequency Green operator (material
            independent) term in the spatial domain (inverse discrete Fourier transform).

        Returns
        -------
        cit_1_integral_mf : ndarray of shape (n_comps, n_comps)
            Discrete integral over the spatial domain of material cluster I of the discrete
            convolution between the material cluster J characteristic function and the first
            Green operator material independent term in the spatial domain.
        cit_2_integral_mf : ndarray of shape (n_comps, n_comps)
            Discrete integral over the spatial domain of material cluster I of the discrete
            convolution between the material cluster J characteristic function and the
            second Green operator material independent term in the spatial domain.
        cit_0_freq_integral_mf : ndarray of shape (n_comps, n_comps)
            Discrete integral over the spatial domain of material cluster I of the discrete
            convolution between the material cluster J characteristic function and the
            zero-frequency Green operator (material independent) term in the spatial domain.
        '''
        # Initialize discrete integral
        cit_1_integral_mf = np.zeros((len(self._comp_order), len(self._comp_order)))
        cit_2_integral_mf = np.zeros((len(self._comp_order), len(self._comp_order)))
        cit_0_freq_integral_mf = np.zeros((len(self._comp_order), len(self._comp_order)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over matricial form components
        for i in range(len(self._comp_order)):
            compi = self._comp_order[i]
            for j in range(len(self._comp_order)):
                compj = self._comp_order[j]
                # Perform discrete integral over the spatial domain of material cluster I
                cit_1_integral_mf[i, j] = mop.kelvinfactor(i, self._comp_order)* \
                    mop.kelvinfactor(j, self._comp_order)*\
                        np.sum(np.multiply(cluster_filter, gop_1_filt_vox[compi + compj]))
                cit_2_integral_mf[i, j] = mop.kelvinfactor(i, self._comp_order)* \
                    mop.kelvinfactor(j, self._comp_order)*\
                        np.sum(np.multiply(cluster_filter, gop_2_filt_vox[compi + compj]))
                cit_0_freq_integral_mf[i, j] = mop.kelvinfactor(i, self._comp_order)*\
                    mop.kelvinfactor(j, self._comp_order)*\
                        np.sum(np.multiply(cluster_filter,
                                           gop_0_freq_filt_vox[compi + compj]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return [cit_1_integral_mf, cit_2_integral_mf, cit_0_freq_integral_mf]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _switch_pair(x, delimiter='_'):
        '''Switch left and right sides of string with separating delimiter.

        Parameters
        ----------
        x : str
            Target string.
        delimiter : str, default='_'
            Separating delimiter between target's string left and right sides.

        Returns
        -------
        y : str
            Switched string.
        '''
        if not isinstance(x, str) or x.count(delimiter) != 1:
            raise RuntimeError('Input parameter must be a string and can only contain ' + \
                               'one delimiter.')
        return delimiter.join(x.split(delimiter)[::-1])
    # --------------------------------------------------------------------------------------
    @staticmethod
    def save_crve_file(crve, crve_file_path):
        '''Dump CRVE into file.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        crve_file_path : str
            Path of file where the CRVE's instance is dumped.
        '''
        # Dump CRVE instance into file
        with open(crve_file_path, 'wb') as crve_file:
            pickle.dump(crve, crve_file)
