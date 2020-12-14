#
# Cluster-reduced Representative Volume Element Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the generation of the Cluster-reduced Representative Volume Element
# (CRVE), a key step in the so called clustering-based reduced order models.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Oct 2020 | Initial coding.
# Bernardo P. Ferreira | Nov 2020 | Merged cluster interaction tensors computation methods.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Generate efficient iterators
import itertools as it
# Unsupervised clustering algorithms
import clustering.clusteringalgs as clstalgs
import scipy.cluster.hierarchy as sciclst
# Date and time
import time
# Display messages
import ioput.info as info
# Matricial operations
import tensor.matrixoperations as mop
# I/O utilities
import ioput.ioutilities as ioutil
# Cluster interaction tensors operations
import clustering.citoperations as citop
# Cluster-reduced material phases
from clustering.clusteringphase import SCRMP, HAACRMP
#
#                                                                                 CRVE class
# ==========================================================================================
class CRVE:
    '''Cluster-reduced Representative Volume Element.

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
        Cluster-reduced material phase instance (item, CRMP) associated to each material
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
    adaptive_time : float
        Total amount of time (s) spent in clustering adaptivity procedures.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, rve_dims, regular_grid, material_phases, comp_order,
                 global_data_matrix, clustering_type, phase_n_clusters,
                 base_clustering_scheme, adaptive_clustering_scheme=None,
                 adaptivity_criterion=None, adaptivity_type=None,
                 adaptivity_control_feature=None, clust_adapt_freq=None):
        '''Cluster-reduced Representative Volume Element constructor.

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
        clust_adapt_freq : dict, default=None
            Clustering adaptivity frequency (relative to the macroscale loading)
            (item, int, default=1) associated with each adaptive cluster-reduced
            material phase (key, str).
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
        self.clust_adapt_freq = clust_adapt_freq
        self.adaptive_time = 0
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
    def compute_base_crve(self):
        '''Compute CRVE base clustering.'''
        # Initialize base clustering labels
        labels = np.full(self._n_voxels, -1, dtype=int)
        # Initialize cluster-reduced material phases dictionary
        self._cluster_phases = {}
        # Initialize minimum cluster label
        min_label = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
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
                if atype == HAACRMP:
                    # Instantiate hierarchical-agglomerative cluster-reduced material phase
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

            print('\n\nBASE CLUSTERING:')
            print('mat_phase: ', mat_phase)
            print('n_phase_clusters: ', n_phase_clusters)
            print('crmp.cluster_labels: ', crmp.cluster_labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # Compute base CRVE descriptors
        info.displayinfo('5', 'Computing base CRVE descriptors...', 2)
        # Reassign and sort RVE clustering labels
        # self._sort_cluster_labels()
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


            print('\n\ncrve_adaptivity: ')
            print('\nmat_phase: ', mat_phase)
            print('\nphase_target_clusters[mat_phase]: ', phase_target_clusters[mat_phase])
            # If there are no target clusters, skip to next adaptive material phase
            if not bool(phase_target_clusters[mat_phase]):
                continue
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Perform adaptive clustering
            adaptive_clustering_map[mat_phase] = \
                crmp.perform_adaptive_clustering(phase_target_clusters[mat_phase],
                                                 min_label)
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
        # Update total amount of time spent in clustering adaptivity procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
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
                                '1': HAACRMP}
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
        Why is this required?
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
            print('\n\nPOP CLUSTERS:')
            print(pop_clusters)
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
                _, cluster_J_filter_dft = self.clusterfilter(cluster_J)
                # Perform discrete convolution between the material phase B cluster
                # characteristic function and each of Green operator material independent
                # terms
                gop_X_filt_vox = self.clstgopconvolution(cluster_J_filter_dft,
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
                        sym_cluster_pair = self.switch_pair(cluster_pair)
                        sym_mat_phase_pair = self.switch_pair(mat_phase_pair)
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
                            cluster_I_filter, _ = self.clusterfilter(cluster_I)
                            # Perform discrete integral over the spatial domain of material
                            # phase A cluster I
                            cit_X_integral_mf = self.discretecitintegral(cluster_I_filter,
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
                            sym_cluster_pair = self.switch_pair(cluster_pair)
                            sym_mat_phase_pair = self.switch_pair(mat_phase_pair)
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


                        #if cluster_I in adaptive_clustering_map[mat_phase_A].keys() or \
                        #    cluster_J in adaptive_clustering_map[mat_phase_B].keys():
                            # Remove cluster interaction tensor
                            #for i in range(len(self.cit_X_mf)):
                            #    self.cit_X_mf[i][mat_phase_pair].pop(cluster_pair)
                        print('\n\nREMOVING CLUSTERS:')
                        if cluster_I in pop_clusters or cluster_J in pop_clusters:
                            for i in range(len(self.cit_X_mf)):
                                self.cit_X_mf[i][mat_phase_pair].pop(cluster_pair)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update total amount of time spent in clustering adaptivity procedures
            self.adaptive_time += time.time() - init_time
    # --------------------------------------------------------------------------------------
    def clusterfilter(self, cluster):
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
    def clstgopconvolution(self, cluster_filter_dft, gop_1_dft_vox, gop_2_dft_vox,
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
    def discretecitintegral(self, cluster_filter, gop_1_filt_vox, gop_2_filt_vox,
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
    def switch_pair(x, delimiter='_'):
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





#
#                                                                                 CRVE class
# ==========================================================================================
class CRVEOLD:
    '''Cluster-reduced Representative Volume Element.

    Base class of a Cluster-reduced Representative Volume Element (CRVE) from which
    Static CRVE (S-CRVE) and Adaptive CRVE (XA-CRVE) are to be derived from.

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
    '''
    def __new__(cls, *args, **kwargs):
        if cls is CRVE:
            raise TypeError("CRVE base class may not be instantiated")
        return super().__new__(cls)
    # --------------------------------------------------------------------------------------
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 comp_order):
        '''Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        comp_order : list
            Strain/Stress components (str) order.
        '''
        self._material_phases = material_phases
        self._phase_n_clusters = phase_n_clusters
        self._rve_dims = rve_dims
        self._comp_order = comp_order
        self._n_dim = len(rve_dims)
        self._gop_X_dft_vox = None
        self.voxels_clusters = None
        self.phase_clusters = None
        self.clusters_f = None
        self.cit_X_mf = None
        # Get number of voxels on each dimension and total number of voxels
        self._n_voxels_dims = \
            [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        self._n_voxels = np.prod(self._n_voxels_dims)
        # Get material phases' voxels' 1D flat indexes
        self._phase_voxel_flatidx = \
            type(self)._get_phase_idxs(regular_grid, material_phases)
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
            Flat voxels' indexes associated to given material cluster.
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
        Why is this required?
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
            # Build list with old clusters and new clusters (adaptive) for each material
            # phase
            phase_old_clusters = {}
            phase_new_clusters = {}
            for mat_phase in self._material_phases:
                phase_new_clusters[mat_phase] = \
                    sum(adaptive_clustering_map[mat_phase].values(), [])
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
                _, cluster_J_filter_dft = self.clusterfilter(cluster_J)
                # Perform discrete convolution between the material phase B cluster
                # characteristic function and each of Green operator material independent
                # terms
                gop_X_filt_vox = self.clstgopconvolution(cluster_J_filter_dft,
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
                        sym_cluster_pair = self.switch_pair(cluster_pair)
                        sym_mat_phase_pair = self.switch_pair(mat_phase_pair)
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
                            cluster_I_filter, _ = self.clusterfilter(cluster_I)
                            # Perform discrete integral over the spatial domain of material
                            # phase A cluster I
                            cit_X_integral_mf = self.discretecitintegral(cluster_I_filter,
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
                            sym_cluster_pair = self.switch_pair(cluster_pair)
                            sym_mat_phase_pair = self.switch_pair(mat_phase_pair)
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
                        if cluster_I in adaptive_clustering_map[mat_phase_A].keys() or \
                            cluster_J in adaptive_clustering_map[mat_phase_B].keys():
                            # Remove cluster interaction tensor
                            for i in range(len(self.cit_X_mf)):
                                self.cit_X_mf[i][mat_phase_pair].pop(cluster_pair)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update total amount of time spent in the A-CRVE adaptive procedures
            self.adaptive_time += time.time() - init_time
    # --------------------------------------------------------------------------------------
    def clusterfilter(self, cluster):
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
    def clstgopconvolution(self, cluster_filter_dft, gop_1_dft_vox, gop_2_dft_vox,
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
    def discretecitintegral(self, cluster_filter, gop_1_filt_vox, gop_2_filt_vox,
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
    def switch_pair(x, delimiter='_'):
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
#
#                                                                               S-CRVE class
# ==========================================================================================
class SCRVE(CRVE):
    '''Static Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation of a Static Cluster-reduced Representative Volume Element (S-CRVE).

    Attributes
    ----------
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
    _clustering_solutions : list
        List containing one or more RVE clustering solutions (ndarray of shape
        (n_clusters,)).
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    cit_X_mf : list
        Cluster interaction tensors material independent terms. Each term is stored in a
        dictionary (item, dict) for each pair of material phases (key, str), which in turn
        contains the corresponding matricial form (item, ndarray) associated to each
        pair of clusters (key, str).
    '''
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 comp_order, clustering_scheme, clustering_ensemble_strategy):
        '''Static Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        comp_order : list
            Strain/Stress components (str) order.
        clustering_scheme : ndarray of shape (n_clusterings, 3)
            Prescribed global clustering scheme to generate the CRVE. Each row is associated
            with a unique RVE clustering, characterized by a clustering algorithm
            (col 1, int), a list of features (col 2, list of int) and a list of the feature
            data matrix' indexes (col 3, list of int).
        '''
        super().__init__(phase_n_clusters, rve_dims, regular_grid, material_phases,
                         comp_order)
        self._clustering_scheme = clustering_scheme
        self._clustering_ensemble_strategy = clustering_ensemble_strategy
        self._clustering_solutions = []
    # --------------------------------------------------------------------------------------
    def get_scrve(self, global_data_matrix):
        '''Generate S-CRVE from one or more RVE clustering solutions.

        Main method commanding the generation of the Static Cluster-Reduced Representative
        Volume Element (S-CRVE): (1) performs the prescribed clustering scheme on the
        provided global data matrix, acquiring one or more RVE clustering solutions;
        (2) obtains a unique clustering solution (consensus solution) that materializes the
        S-CRVE; (3) computes several descriptors of the S-CRVE.

        Parameters
        ----------
        global_data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform all the RVE clusterings
            prescribed in the clustering scheme.
        '''
        # Loop over prescribed RVE clustering solutions
        for i_clst in range(self._clustering_scheme.shape[0]):
            # Get clustering algorithm
            clustering_method = self._clustering_scheme[i_clst, 0]
            # Get clustering features' columns
            feature_cols = self._clustering_scheme[i_clst, 2]
            # Get RVE clustering data matrix
            rve_data_matrix = mop.getcondmatrix(global_data_matrix,
                                                list(range(global_data_matrix.shape[0])),
                                                feature_cols)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            info.displayinfo('5', 'RVE clustering (' + str(i_clst + 1) + ' of ' +
                             str(self._clustering_scheme.shape[0]) + ')...', 2)
            # Instantiate RVE clustering
            rve_clustering = RVEClustering(clustering_method, self._phase_n_clusters,
                                           self._n_voxels, self._material_phases,
                                           self._phase_voxel_flatidx)
            # Perform RVE clustering
            rve_clustering.perform_rve_clustering(rve_data_matrix)
            # Assemble RVE clustering
            self._add_new_clustering(rve_clustering.labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE consensus clustering solution
        info.displayinfo('5', 'Building S-CRVE (RVE consensus clustering)...', 2)
        self._set_consensus_clustering()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing S-CRVE descriptors...', 2)
        # Reassign and sort RVE clustering labels
        self._sort_cluster_labels()
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def _add_new_clustering(self, rve_clustering):
        '''Add new RVE clustering to collection of clustering solutions.

        Parameters
        ----------
        rve_clustering : ndarray of shape (n_clusters,)
            Cluster label (int) assigned to each RVE voxel.
        '''
        self._clustering_solutions.append(rve_clustering)
    # --------------------------------------------------------------------------------------
    def _set_consensus_clustering(self):
        '''Set a unique RVE clustering solution (consensus solution).

        Notes
        -----
        Even if the clustering scheme only accounts for a single RVE clustering solution,
        this method must be called in order to the set unique S-CRVE clustering solution.
        '''
        # Get RVE consensus clustering solution according to the prescribed clustering
        # ensemble strategy
        if self._clustering_ensemble_strategy == 0:
            # Build S-CRVE from the single RVE clustering solution
            self.voxels_clusters = np.reshape(np.array(self._clustering_solutions[0],
                                              dtype=int),self._n_voxels_dims)
#
#                                                                             RVE clustering
# ==========================================================================================
class RVEClustering:
    '''RVE clustering class.

    RVE clustering-based domain decomposition based on a given clustering algorithm.

    Atributes
    ---------
    labels : ndarray of shape (n_clusters,)
        Cluster label (int) assigned to each RVE voxel.
    '''
    def __init__(self, clustering_method, phase_n_clusters, n_voxels, material_phases,
                 phase_voxel_flat_idx):
        '''RVE clustering constructor.

        Parameters
        ----------
        _clustering_method : int
            Clustering algorithm identifier.
        _phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        _n_voxels : int
            Total number of voxels of the RVE spatial discretization.
        _material_phases : list
            RVE material phases labels (str).
        _phase_voxel_flatidx : dict
            Flat (1D) voxels' indexes (item, list) associated to each material phase
            (key, str).
        '''
        self._clustering_method = clustering_method
        self._phase_n_clusters = phase_n_clusters
        self._n_voxels = n_voxels
        self._material_phases = material_phases
        self._phase_voxel_flatidx = phase_voxel_flat_idx
        self.labels = None
    # --------------------------------------------------------------------------------------
    def perform_rve_clustering(self, data_matrix):
        '''Perform the RVE clustering-based domain decomposition.

        Instantiates a given clustering algorithm and performs the RVE clustering-based
        domain decomposition based on the provided data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform the RVE clustering.

        Notes
        -----
        The clustering is performed independently for each RVE's material phase. With the
        exception of the number of clusters, which is in general different for each material
        phase, the clustering algorithm and associated parameters remain the same for all
        material phases.
        '''
        # Initialize RVE clustering
        self.labels = np.full(self._n_voxels, -1, dtype=int)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate clustering algorithm
        if self._clustering_method == 1:
            # Set number of full batch K-Means clusterings (with different initializations)
            n_init = 10
            # Instantiate K-Means
            clst_alg = clstalgs.KMeansSK(init='k-means++', n_init=n_init, max_iter=300,
                                        tol=1e-4, random_state=None, algorithm='auto')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 2:
            # Instatiante K-Means
            clst_alg = clstalgs.KMeansPC(tolerance=1e-03, itermax=200)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 3:
            # Set size of the mini-batches
            batch_size = 100
            # Set number of random initializations
            n_init = 3
            # Intantiate Mini-Batch K-Means
            clst_alg = clstalgs.MiniBatchKMeansSK(init='k-means++', max_iter=100, tol=0.0,
                                                 random_state=None, batch_size=batch_size,
                                                 max_no_improvement=10, init_size=None,
                                                 n_init=n_init, reassignment_ratio=0.01)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 4:
            # Instantiate Agglomerative clustering
            clst_alg = clstalgs.AgglomerativeSK(n_clusters=None, affinity='euclidean',
                                               memory=None, connectivity=None,
                                               compute_full_tree='auto',
                                               linkage='ward', distance_threshold=None)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 5:
            # Instatiate Agglomerative clustering
            clst_alg = clstalgs.AgglomerativeSP(0, n_clusters=None, method='ward',
                                               metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 6:
            # Instatiate Agglomerative clustering
            clst_alg = clstalgs.AgglomerativeFC(0, n_clusters=None, method='ward',
                                               metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 7:
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clst_alg = clstalgs.BirchSK(threshold=threshold,
                                       branching_factor=branching_factor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 8:
            # Set merging radius threshold
            threshold = 0.1
            # Set maximum number of CF subclusters in each node
            branching_factor = 50
            # Instantiate Birch
            clst_alg = clstalgs.BirchPC(threshold=threshold,
                                       branching_factor=branching_factor)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 9:
            # Instantiate Cure
            clst_alg = clstalgs.CurePC(number_represent_points=5, compression=0.5)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif self._clustering_method == 10:
            # Instantiate X-Means
            clst_alg = clstalgs.XMeansPC(tolerance=2.5e-2, repeat=1)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown clustering algorithm.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize label offset (avoid that different material phases share the same
        # labels)
        label_offset = 0
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase's voxels indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # Set material phase number of clusters
            if hasattr(clst_alg, 'n_clusters'):
                clst_alg.n_clusters = self._phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase data matrix
            phase_data_matrix = mop.getcondmatrix(data_matrix, voxels_idxs,
                                                  list(range(data_matrix.shape[1])))
            # Perform material phase clustering
            cluster_labels = clst_alg.perform_clustering(phase_data_matrix)
            # Check number of clusters formed
            if len(set(cluster_labels)) != self._phase_n_clusters[mat_phase]:
                raise RuntimeError('The number of clusters (' +
                                   str(len(set(cluster_labels))) +
                                   ') obtained for material phase ' + mat_phase + ' is ' +
                                   'different from the prescribed number of clusters (' +
                                   str(self._phase_n_clusters[mat_phase]) + ').')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material phase cluster labels
            self.labels[voxels_idxs] = label_offset + cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update label offset
            if not ioutil.checkposint(clst_alg.n_clusters):
                raise RuntimeError('Invalid number of clusters.')
            else:
                label_offset = label_offset + clst_alg.n_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the dataset items have been labeled
        if np.any(self.labels == -1):
            raise RuntimeError('At least one RVE domain point has not been labeled' +
                               'during the cluster analysis.')
#
#                                                                              HA-CRVE class
# ==========================================================================================
class HACRVE(CRVE):
    '''Hierarchical Adaptive Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation and update of a Hierarchical Adaptive Cluster-reduced Representative Volume
    Element (HA-CRVE).

    Attributes
    ----------
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
    _phase_linkage_matrix : dict
        Linkage matrix (item, ndarray of shape (n_voxels-1, 4)) associated with the
        hierarchical agglomerative clustering of each material phase (key, str).
    _phase_map_cluster_node : dict
        Cluster-node mapping (item, dict with tree node id (item, int) associated to each
        cluster label (item, str)) for each material phase (key, str).
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base clustering.
    adaptive_time : float
        Total amount of time (s) spent in the A-CRVE adaptive procedures.
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    cit_X_mf : list
        Cluster interaction tensors material independent terms. Each term is stored in a
        dictionary (item, dict) for each pair of material phases (key, str), which in turn
        contains the corresponding matricial form (item, ndarray) associated to each
        pair of clusters (key, str).
    '''
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 comp_order, adaptive_split_factor):
        '''Hierarchical Adaptive Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        comp_order : list
            Strain/Stress components (str) order.
        adaptive_split_factor : dict
            Clustering adaptive split factor (item, float) for each material phase
            (key, str). The clustering adaptive split factor must be contained between
            0 and 1 (included). The lower bound (0) prevents any cluster to be split,
            while the upper bound (1) performs the maximum number splits of each cluster
            (single-voxel clusters).
        '''
        super().__init__(phase_n_clusters, rve_dims, regular_grid, material_phases,
                         comp_order)
        self._adaptive_split_factor = adaptive_split_factor
        self._phase_linkage_matrix = None
        self._phase_map_cluster_node = {}
        self._adaptive_step = 0
        self.adaptive_time = 0
    # --------------------------------------------------------------------------------------
    def get_base_clustering(self, data_matrix):
        '''Perform the RVE hierarchical agglomerative base clustering-based domain
        decomposition.

        Instantiates a given hierarchical agglomerative clustering algorithm and performs
        the RVE hierarchical agglomerative base clustering-based domain decomposition
        (hierarchical tree building and horizontal cut) based on the provided data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform the RVE clustering.

        Notes
        -----
        The clustering is performed independently for each RVE's material phase. With the
        exception of the number of clusters, which is in general different for each material
        phase, the clustering algorithm and associated parameters remain the same for all
        material phases.
        '''
        # Instatiate Agglomerative clustering
        clst_alg = clstalgs.AgglomerativeSP(0, n_clusters=None, method='ward',
                                            metric='euclidean', criterion='maxclust')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize linkage matrices dictionary
        self._phase_linkage_matrix = {}
        # Initialize RVE hierarchical agglomerative base clustering
        labels = np.full(self._n_voxels, -1, dtype=int)
        # Initialize label offset (avoid that different material phases share the same
        # labels)
        label_offset = 0
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase's voxels indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # Set material phase base number of clusters
            clst_alg.n_clusters = self._phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase data matrix
            phase_data_matrix = mop.getcondmatrix(data_matrix, voxels_idxs,
                                                  list(range(data_matrix.shape[1])))
            # Perform material phase hierarchical agglomerative base clustering
            # (hierarchical tree horizontal cut)
            cluster_labels = clst_alg.perform_clustering(phase_data_matrix)
            # Check number of clusters formed
            if len(set(cluster_labels)) != self._phase_n_clusters[mat_phase]:
                raise RuntimeError('The number of clusters (' +
                                   str(len(set(cluster_labels))) +
                                   ') obtained for material phase ' + mat_phase + ' is ' +
                                   'different from the prescribed number of clusters (' +
                                   str(self._phase_n_clusters[mat_phase]) + ').')
            # Store material phase hierarchical agglomerative base clustering linkage matrix
            self._phase_linkage_matrix[mat_phase] = clst_alg.get_linkage_matrix()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Assemble material phase cluster labels
            labels[voxels_idxs] = label_offset + cluster_labels
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update label offset
            if not ioutil.checkposint(clst_alg.n_clusters):
                raise RuntimeError('Invalid number of clusters.')
            else:
                label_offset = label_offset + clst_alg.n_clusters
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check if all the voxels have been labeled
        if np.any(labels == -1):
            raise RuntimeError('At least one RVE domain point has not been labeled' +
                               'during the cluster analysis.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE from the hierarchical agglomerative base clustering
        info.displayinfo('5', 'Building HA-CRVE base clustering...', 2)
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing HA-CRVE descriptors...', 2)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters):
        '''Perform a hierarchical adaptive clustering refinement step.

        Refine the provided target clusters by splitting them according to the hierarchical
        agglomerative tree, prioritizing child nodes by descending order of linkage
        distance.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be refined.

        Returns
        -------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str).
        adaptive_tree_node_map : dict
            Adaptive tree node map (item, dict with list of new cluster tree node ids
            (item, list of int) resulting from the split of each target cluster tree node id
            (key, str).) for each material phase' (key, str) hierarchical tree.
            Validation purposes only (not returned otherwise).

        Notes
        -----
        Given that the hierarchical agglomerative base clustering is performed independently
        for each RVE's material phase, so is the adaptive clustering refinement, i.e., each
        material phase has an associated hierarchical agglomerative tree.
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            is_exist = False
            for mat_phase in self._material_phases:
                if target_cluster in self.phase_clusters[mat_phase]:
                    is_exist = True
                    break
            if not is_exist:
                raise RuntimeError('Target cluster ' + str(target_cluster) + ' does not ' +
                                   'exist in the current CRVE clustering.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering refinement step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary and associated phase
        # dictionaries
        adaptive_clustering_map = {str(mat_phase): {}
                                   for mat_phase in self._material_phases}
        # Initialize adaptive tree node mapping dictionary (only validation purposes)
        adaptive_tree_node_map = {str(mat_phase): {}
                                   for mat_phase in self._material_phases}
        # Get RVE hierarchical agglomerative base clustering
        labels = self.voxels_clusters.flatten()
        # Get maximum cluster label in RVE base clustering (avoid that new cluster labels
        # override existing ones)
        new_cluster_label = max(labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get material phase cluster labels (conversion to int32 is required to avoid
            # raising a TypeError in scipy hierarchy leaders function)
            cluster_labels = labels[self._phase_voxel_flatidx[mat_phase]].astype('int32')
            # Convert hierarchical original clustering linkage matrix into tree object
            rootnode, nodelist = sciclst.to_tree(self._phase_linkage_matrix[mat_phase],
                                                 rd=True)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build initial cluster-node mapping between cluster labels and tree nodes
            # associated with the hierarchical agglomerative base clustering
            if not mat_phase in self._phase_map_cluster_node.keys():
                # Get root nodes of hierarchical clustering corresponding to an horizontal
                # cut defined by a flat clustering assignment vector. L contains the tree
                # nodes ids while M contains the corresponding cluster labels
                L, M = sciclst.leaders(self._phase_linkage_matrix[mat_phase],
                                       cluster_labels)
                # Build initial cluster-node mapping
                self._phase_map_cluster_node[mat_phase] = dict(zip([str(x) for x in M], L))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over target clusters
            for i in range(len(target_clusters)):
                # Get target cluster label
                target_cluster = target_clusters[i]
                # If target cluster label belongs to the current material phase, then get
                # target cluster tree node instance. Otherwise skip to the next target
                # cluster
                if str(target_cluster) in self._phase_map_cluster_node[mat_phase].keys():
                    target_node = nodelist[
                        self._phase_map_cluster_node[mat_phase][str(target_cluster)]]
                else:
                    continue
                # Get total number of leaf nodes associated to target node. If target node
                # is a leaf itself (not splitable), skip to the next target cluster
                if target_node.is_leaf():
                    continue
                else:
                    n_leaves = target_node.get_count()
                # Compute total number of tree node splits, enforcing at least one split
                n_splits = max(1,
                    int(np.round(self._adaptive_split_factor[mat_phase]*(n_leaves - 1))))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize child nodes list
                child_nodes = []
                # Initialize child target nodes list
                child_target_nodes = []
                # Split loop
                for i_split in range(n_splits):
                    # Set node to be splitted
                    if i_split == 0:
                        # In the first split operation, the node to be splitted is the
                        # target cluster tree node
                        node_to_split = target_node
                    else:
                        # Get maximum linkage distance child target node and remove it from
                        # the child target nodes list
                        node_to_split = child_target_nodes[0]
                        child_target_nodes.pop(0)
                    # Loop over child target node's left and right child nodes
                    for node in [node_to_split.get_left(), node_to_split.get_right()]:
                        if node.is_leaf():
                            # Append to child nodes list if leaf node
                            child_nodes.append(node)
                        else:
                            # Append to child target nodes list if non-leaf node
                            child_target_nodes = \
                                self.add_to_tree_node_list(child_target_nodes, node)
                # Add remaining child target nodes to child nodes list
                child_nodes += child_target_nodes
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize target cluster mapping
                adaptive_clustering_map[mat_phase][str(target_cluster)] = []
                # Remove target cluster from cluster node mapping
                self._phase_map_cluster_node[mat_phase].pop(str(target_cluster))
                # Update target cluster mapping and flat clustering labels
                for node in child_nodes:
                    # Increment new cluster label
                    new_cluster_label += 1
                    # Add new cluster to target cluster mapping
                    adaptive_clustering_map[mat_phase][str(target_cluster)].append(
                        new_cluster_label)
                    # Update flat clustering labels
                    cluster_labels[node.pre_order()] = new_cluster_label
                    # Update cluster-node mapping
                    self._phase_map_cluster_node[mat_phase][str(new_cluster_label)] = \
                        node.id
                # Update adaptive tree node mapping dictionary (only validation purposes)
                adaptive_tree_node_map[mat_phase][str(target_node.id)] = \
                    [x.id for x in child_nodes]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update RVE hierarchical agglomerative clustering
            labels[self._phase_voxel_flatidx[mat_phase]] = cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build hiearchical adaptive CRVE from the hierarchical agglomerative adaptive
        # clustering
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the A-CRVE adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.print_adaptive_clustering(adaptive_clustering_map, adaptive_tree_node_map)
        # Return
        return [adaptive_clustering_map, adaptive_tree_node_map]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def add_to_tree_node_list(node_list, node):
        '''Add node to tree node list and sort it by descending order of linkage distance.

        Parameters
        ----------
        node_list : list of ClusterNode
            List of ClusterNode instances.
        node : ClusterNode
            ClusterNode to be added to list of ClusterNode.
        '''
        # Check parameters
        if not isinstance(node, sciclst.ClusterNode):
            raise TypeError('Node must be of type ClusterNode, not ' + str(type(node)) +
                            '.')
        if any([not isinstance(node, sciclst.ClusterNode) for node in node_list]):
            raise TypeError('Node list can only contain elements of the type ClusterNode.')
        # Append tree node to node list
        node_list.append(node)
        # Sort tree node list by descending order of linkage distance
        node_list = sorted(node_list, reverse=True, key=lambda x: x.dist)
        # Return sorted tree node list
        return node_list
    # --------------------------------------------------------------------------------------
    def print_adaptive_clustering(self, adaptive_clustering_map, adaptive_tree_node_map):
        '''Print hierarchical adaptive clustering refinement descriptors (validation).'''
        # Print report header
        print(3*'\n' + 'Hierarchical adaptive clustering report\n' + 80*'-')
        # Print adaptive clustering refinement step
        print('\nAdaptive refinement step: ', self._adaptive_step)
        # Print hiearchical adaptive CRVE
        print('\n\n' + 'Adaptive clustering: ' + '(' +
              str(len(np.unique(self.voxels_clusters))) + ' clusters)' + '\n\n',
              self.voxels_clusters)
        # Print adaptive clustering mapping
        print('\n\n' + 'Adaptive cluster mapping: ')
        for mat_phase in self._material_phases:
            print('\n  Material phase ' + mat_phase + ':\n')
            for old_cluster in adaptive_clustering_map[mat_phase].keys():
                print('    Old cluster: ' + '{:>4s}'.format(old_cluster) +
                      '  ->  ' +
                      'New clusters: ',
                      adaptive_clustering_map[mat_phase][str(old_cluster)])
        # Print adaptive tree node mapping
        print('\n\n' + 'Adaptive tree node mapping (validation): ')
        for mat_phase in self._material_phases:
            print('\n  Material phase ' + mat_phase + ':\n')
            for old_node in adaptive_tree_node_map[mat_phase].keys():
                print('  Old node: ' + '{:>4s}'.format(old_node) +
                      '  ->  ' +
                      'New nodes: ', adaptive_tree_node_map[mat_phase][str(old_node)])
        # Print cluster-node mapping
        print('\n\n' + 'Cluster-Node mapping: ')
        for mat_phase in self._material_phases:
            print('\n  Material phase ' + mat_phase + ':\n')
            for new_cluster in self._phase_map_cluster_node[mat_phase].keys():
                print('    Cluster: ' + '{:>4s}'.format(new_cluster) +
                      '  ->  ' +
                      'Tree node: ',
                      self._phase_map_cluster_node[mat_phase][str(new_cluster)])
#
#                                                                              PA-CRVE class
# ==========================================================================================
class PACRVE(CRVE):
    '''Partitional Adaptive Cluster-reduced Representative Volume Element.

    This class provides all the required attributes and methods associated with the
    generation and update of a Partitional Adaptive Cluster-reduced Representative Volume
    Element (PA-CRVE).

    Attributes
    ----------
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
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base clustering.
    adaptive_time : float
        Total amount of time (s) spent in the A-CRVE adaptive procedures.
    voxels_clusters : ndarray
        Regular grid of voxels (spatial discretization of the RVE), where each entry
        contains the cluster label (int) assigned to the corresponding pixel/voxel.
    phase_clusters : dict
        Clusters labels (item, list of int) associated to each material phase (key, str).
    clusters_f : dict
        Clusters volume fraction (item, float) associated to each material cluster
        (key, str).
    cit_X_mf : list
        Cluster interaction tensors material independent terms. Each term is stored in a
        dictionary (item, dict) for each pair of material phases (key, str), which in turn
        contains the corresponding matricial form (item, ndarray) associated to each
        pair of clusters (key, str).
    '''
    def __init__(self, phase_n_clusters, rve_dims, regular_grid, material_phases,
                 comp_order, adaptive_split_factor):
        '''Partitional Adaptive Cluster-reduced Representative Volume Element constructor.

        Parameters
        ----------
        phase_n_clusters : dict
            Number of clusters (item, int) prescribed for each material phase (key, str).
        rve_dims : list
            RVE size in each dimension.
        regular_grid : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where each entry
            contains the material phase label (int) assigned to the corresponding voxel.
        material_phases : list
            RVE material phases labels (str).
        comp_order : list
            Strain/Stress components (str) order.
        adaptive_split_factor : dict
            Clustering adaptive split factor (item, float) for each material phase
            (key, str). The clustering adaptive split factor must be contained between
            0 and 1 (included). The lower bound (0) prevents any cluster to be split,
            while the upper bound (1) performs the maximum number splits of each cluster
            (single-voxel clusters).
        '''
        super().__init__(phase_n_clusters, rve_dims, regular_grid, material_phases,
                         comp_order)
        self._adaptive_split_factor = adaptive_split_factor
        self._adaptive_step = 0
        self.adaptive_time = 0
    # --------------------------------------------------------------------------------------
    def get_base_clustering(self, data_matrix):
        '''Perform the RVE partitional base clustering-based domain decomposition.

        Instantiates a given partitional clustering algorithm and performs the RVE
        partitional base clustering-based domain decomposition based on the provided data
        matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_voxels, n_features)
            Data matrix containing the required data to perform the RVE clustering.

        Notes
        -----
        The clustering is performed independently for each RVE's material phase. With the
        exception of the number of clusters, which is in general different for each material
        phase, the clustering algorithm and associated parameters remain the same for all
        material phases.
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get RVE clustering data matrix
        rve_data_matrix = mop.getcondmatrix(data_matrix,
                                            list(range(data_matrix.shape[0])),
                                            list(range(data_matrix.shape[1])))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set clustering algorithm
        clustering_method = 1
        # Instantiate RVE clustering
        rve_clustering = RVEClustering(clustering_method, self._phase_n_clusters,
                                       self._n_voxels, self._material_phases,
                                       self._phase_voxel_flatidx)
        # Perform RVE clustering
        rve_clustering.perform_rve_clustering(rve_data_matrix)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE from the partitional base clustering
        info.displayinfo('5', 'Building PA-CRVE base clustering...', 2)
        self.voxels_clusters = np.reshape(np.array(rve_clustering.labels, dtype=int),
                                          self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        info.displayinfo('5', 'Computing PA-CRVE descriptors...', 2)
        # Reassign and sort RVE clustering labels
        self._sort_cluster_labels()
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # --------------------------------------------------------------------------------------
    def perform_adaptive_clustering(self, target_clusters):
        '''Perform a partitional adaptive clustering refinement step.

        Refine the provided target clusters by further clustering the associated spatial
        domain voxels.

        Parameters
        ----------
        target_clusters : list
            List with the labels (int) of clusters to be refined.

        Returns
        -------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels (item,
            list of int) resulting from the refinement of each target cluster (key, str))
            for each material phase (key, str).
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated labels.')
        # Check for unexistent target clusters
        for target_cluster in target_clusters:
            is_exist = False
            for mat_phase in self._material_phases:
                if target_cluster in self.phase_clusters[mat_phase]:
                    is_exist = True
                    break
            if not is_exist:
                raise RuntimeError('Target cluster ' + str(target_cluster) + ' does not ' +
                                   'exist in the current CRVE clustering.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering refinement step counter
        self._adaptive_step += 1
        # Initialize adaptive clustering mapping dictionary and associated phase
        # dictionaries
        adaptive_clustering_map = {str(mat_phase): {}
                                   for mat_phase in self._material_phases}
        # Get RVE base clustering
        labels = self.voxels_clusters.flatten()
        # Get maximum cluster label in RVE base clustering (avoid that new cluster labels
        # override existing ones)
        new_cluster_label = max(labels)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of full batch K-Means clusterings (with different initializations)
        n_init = 10
        # Instantiate K-Means
        clst_alg = clstalgs.KMeansSK(init='k-means++', n_init=n_init, max_iter=300,
                                    tol=1e-4, random_state=None, algorithm='auto')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Loop over target clusters
            for i in range(len(target_clusters)):
                # Get target cluster label
                target_cluster = target_clusters[i]
                # If target cluster label belongs to the current material phase, then get
                # the number and flat indexes of associated spatial domain voxels. Otherwise
                # skip to the next target cluster
                if target_cluster in self.phase_clusters[mat_phase]:
                    n_voxels_cluster = np.sum(self.voxels_clusters == target_cluster)
                    target_cluster_idxs = self._get_cluster_idxs(self.voxels_clusters,
                                                                 target_cluster)
                else:
                    continue
                # If target cluster is already a single-voxel (cannot be further refined),
                # then skip to the next target cluster
                if n_voxels_cluster < 2:
                    continue
                # Compute total number of child clusters, enforcing at least two child
                # clusters
                n_child_clusters = max(2,
                    int(np.round(self._adaptive_split_factor[mat_phase]*n_voxels_cluster)))
                # Set material phase number of clusters
                if hasattr(clst_alg, 'n_clusters'):
                    clst_alg.n_clusters = n_child_clusters
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                # REMOVE
                data_matrix = self._cluster_data_matrix

                # Get cluster data matrix
                cluster_data_matrix = mop.getcondmatrix(data_matrix, target_cluster_idxs,
                                                        list(range(data_matrix.shape[1])))
                # Perform material phase clustering
                childs_labels = clst_alg.perform_clustering(cluster_data_matrix)
                # Check number of clusters formed
                if len(set(childs_labels)) != n_child_clusters:
                    raise RuntimeError('The number of child clusters (' +
                                       str(len(set(childs_labels))) +
                                       ') obtained for cluster ' + str(target_cluster) +
                                       ' is different from the prescribed number of ' +
                                       'clusters (' + str(n_child_clusters) + ').')
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Initialize target cluster mapping
                adaptive_clustering_map[mat_phase][str(target_cluster)] = []
                # Update child clusters labels
                for child_label in set(childs_labels):
                    # Increment new cluster label
                    new_cluster_label += 1
                    # Add new cluster to target cluster mapping
                    adaptive_clustering_map[mat_phase][str(target_cluster)].append(
                        new_cluster_label)
                    # Update child cluster labels
                    childs_labels = np.where(childs_labels == child_label,
                                             new_cluster_label, childs_labels)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Update RVE clustering
                labels[target_cluster_idxs] = childs_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build partitional adaptive CRVE from the partitional adaptive clustering
        self.voxels_clusters = np.reshape(np.array(labels, dtype=int), self._n_voxels_dims)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the A-CRVE adaptive procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Print adaptive clustering mapping
        print('\n\n' + 4*' ' + 'Adaptive cluster mapping: ')
        for mat_phase in self._material_phases:
            print('\n' + 6*' ' +'Material phase ' + mat_phase + ':\n')
            for old_cluster in adaptive_clustering_map[mat_phase].keys():
                print(8*' ' + 'Old cluster: ' + '{:>4s}'.format(old_cluster) +
                      '  ->  ' +
                      'New clusters: ',
                      adaptive_clustering_map[mat_phase][str(old_cluster)])

        # Return
        return adaptive_clustering_map
