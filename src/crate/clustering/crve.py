"""Cluster-Reduced Representative Volume Element.

This module includes the class that materializes a Cluster-Reduced
Representative Volume Element (CRVE) as an aggregation of Cluster-Reduced
Material Phases and a set of cluster interaction tensors. This class has the
high-level control of all clustering-related procedures, namely both base and
adaptive cluter analyses as well as the computation of the cluster interaction
tensors.

The concept of Cluster-Reduced Representative Volume Element arises in the
context of clustering-based reduced order modeling (see Chapter 4 of
Ferreira (2022) [#]_).

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <https://repositorio-aberto.up.pt/handle/10216/
       146900?locale=en>`_)

Classes
-------
CRVE
    Cluster-Reduced Representative Volume Element.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import time
import copy
import itertools as it
import pickle
# Third-party
import numpy as np
# Local
import ioput.info as info
import tensor.matrixoperations as mop
import clustering.citoperations as citop
from clustering.clusteringphase import SCRMP, GACRMP, HAACRMP
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class CRVE:
    """Cluster-Reduced Representative Volume Element.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _n_voxels_dims : list[int]
        Number of voxels in each dimension of the regular grid (spatial
        discretization of the RVE).
    _n_voxels : int
        Total number of voxels of the regular grid (spatial discretization of
        the RVE).
    _phase_voxel_flatidx : dict
        Flat (1D) voxels' indexes (item, list[int]) associated with each
        material phase (key, str).
    _gop_X_dft_vox : list[dict]
        Green operator material independent terms. Each term is stored in a
        dictionary, where each pair of strain/stress components (key, str) is
        associated with the Green operator material independent term evaluated
        in all spatial discrete points (item, numpy.ndarray (2d or 3d)).
    _cluster_phases : dict
        Cluster-Reduced material phase instance (item, CRMP) associated with
        each material phase (key, str).
    _base_phase_n_clusters : dict
        Number of clusters (item, int) prescribed (base clustering) for each
        material phase (key, str).
    _adaptive_step : int
        Counter of adaptive clustering steps, with 0 associated with the base
        clustering.
    _voxels_clusters : numpy.ndarray (2d or 3d)
        Regular grid of voxels (spatial discretization of the RVE), where each
        entry contains the cluster label (int) assigned to the corresponding
        voxel.
    _phase_clusters : dict
        Clusters labels (item, list[int]) associated with each material phase
        (key, str).
    _clusters_vf : dict
        Volume fraction (item, float) associated with each material cluster
        (key, str).
    _cit_x_mf : list[dict]
        Cluster interaction tensors associated with the Green operator material
        independent terms. Each term is stored in a dictionary (item, dict) for
        each pair of material phases (key, str), which in turn contains the
        corresponding matricial form (item, numpy.ndarray (2d)) associated with
        each pair of clusters (key, str).
    _adapt_material_phases : list[str]
        RVE adaptive material phases labels (str).
    _adaptive_clustering_time : float
        Total amount of time (s) spent in clustering adaptivity.
    _adaptive_cit_time : float
        Total amount of time (s) spent in clustering adaptivity cluster
        interaction tensors computation procedures.

    Methods
    -------
    perform_crve_base_clustering(self)
        Compute CRVE base clustering.
    perform_crve_adaptivity(self, target_clusters, target_clusters_data)
        Perform CRVE clustering adaptivity.
    get_rve_dims(self)
        Get RVE dimensions.
    get_material_phases(self)
        Get RVE material phases.
    get_n_voxels(self)
        Get number of voxels in each dimension and total number of voxels.
    get_regular_grid(self)
        Get regular grid of voxels with material phase labels.
    get_phase_n_clusters(self)
        Get number of clusters associated with each material phase.
    get_phase_clusters(self)
        Get clusters associated with each material phase.
    get_voxels_clusters(self)
        Get regular grid containing the cluster label of each voxel.
    get_n_total_clusters(self)
        Get current total number of clusters.
    get_cluster_phases(self)
        Get cluster-reduced material phase instance of each material phase.
    get_clusters_vf(self)
        Get clusters volume fraction.
    get_cit_x_mf(self)
        Get cluster interaction tensors (Green material independent terms).
    get_eff_isotropic_elastic_constants(self)
        Get isotropic elastic constants from elastic tangent modulus.
    get_adapt_material_phases(self)
        Get adaptive material phases labels.
    get_adaptivity_control_feature(self)
        Get clustering adaptivity control feature of each material phase.
    get_adapt_criterion_data(self)
        Get clustering adaptivity criterion data of each material phase.
    get_voxels_array_variables(self)
        Get required variables to build a clusters state based voxels array.
    get_adaptive_step(self)
        Get counter of adaptive clustering steps.
    get_adaptive_clustering_time(self)
        Get total amount of time spent in clustering adaptivity.
    get_adaptive_cit_time(self)
        Get total time spent in adaptivity cluster interaction tensors.
    get_clustering_type(self)
        Get clustering type of each material phase.
    get_adaptivity_output(self)
        Get required data for clustering adaptivity output file.
    get_clustering_summary(self)
        Get summary of number of clusters of each material phase.
    reset_adaptive_parameters(self)
        Reset CRVE adaptive progress parameters and set base clustering.
    update_adaptive_parameters(self, adaptive_clustering_scheme, \
                               adapt_criterion_data, adaptivity_type, \
                               adaptivity_control_feature)
        Update CRVE clustering adaptivity attributes.
    get_crmp_types()
        Get available cluster-reduced material phases types.
    _get_features_indexes(clustering_scheme)
        Get unique clustering features indexes from clustering scheme.
    _get_phase_idxs(regular_grid, material_phases)
        Get flat indexes of each material phase's voxels.
    _get_cluster_idxs(voxels_clusters, cluster_label)
        Get flat indexes of given cluster's voxels.
    _set_phase_clusters(self)
        Set CRVE cluster labels associated with each material phase.
    _set_clusters_vf(self)
        Set CRVE clusters' volume fractions.
    _get_clusters_max_label(self)
        Get CRVE maximum cluster label.
    _sort_cluster_labels(self)
        Reassign and sort CRVE cluster labels material phasewise.
    compute_cit(self, mode='full', adaptive_clustering_map=None)
        Compute CRVE cluster interaction tensors.
    _cluster_filter(self, cluster)
        Compute cluster discrete characteristic function.
    _gop_convolution(self, cluster_filter_dft, gop_1_dft_vox, \
                     gop_2_dft_vox, gop_0_freq_dft_vox)
        Convolution of cluster characteristic function and Green operator.
    _discrete_cit_integral(self, cluster_filter, gop_1_filt_vox, \
                           gop_2_filt_vox, gop_0_freq_filt_vox)
        Discrete integral over the spatial domain of material cluster.
    _switch_pair(x, delimiter='_')
        Switch left and right sides of string with separating delimiter.
    save_crve_file(crve, crve_file_path)
        Dump CRVE into file.
    """
    # -------------------------------------------------------------------------
    def __init__(self, rve_dims, regular_grid, material_phases,
                 strain_formulation, problem_type, global_data_matrix,
                 clustering_type, phase_n_clusters, base_clustering_scheme,
                 eff_elastic_properties=None, adaptive_clustering_scheme=None,
                 adapt_criterion_data=None, adaptivity_type=None,
                 adaptivity_control_feature=None):
        """Constructor.

        Parameters
        ----------
        rve_dims : list[float]
            RVE size in each dimension.
        regular_grid : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the material phase label (int) assigned to the
            corresponding voxel.
        material_phases : list[str]
            RVE material phases labels (str).
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        global_data_matrix: numpy.ndarray (2d)
            Data matrix (numpy.ndarray of shape (n_voxels, n_features_dims))
            containing the required clustering features' data to perform all
            the prescribed cluster analyses.
        clustering_type : dict
            Clustering type (item, {'static', 'adaptive'}) of each material
            phase (key, str).
        phase_n_clusters : dict
            Number of clusters (item, int) associated with each material phase
            (key, str).
        base_clustering_scheme : dict
            Prescribed base clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        eff_elastic_properties : dict, default=None
            Elastic properties (key, str) and their values (item, float)
            estimated from the RVE's elastic effective tangent modulus.
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        adapt_criterion_data : dict, default=None
            Clustering adaptivity criterion (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            criterion to be used and the required parameters.
        adaptivity_type : dict, default=None
            Clustering adaptivity type (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            type to be used and the required parameters.
        adaptivity_control_feature : dict, default=None
            Clustering adaptivity control feature (item, str) associated with
            each material phase (key, str).
        """
        self._rve_dims = copy.deepcopy(rve_dims)
        self._regular_grid = copy.deepcopy(regular_grid)
        self._material_phases = copy.deepcopy(material_phases)
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._global_data_matrix = copy.deepcopy(global_data_matrix)
        self._clustering_type = copy.deepcopy(clustering_type)
        self._phase_n_clusters = copy.deepcopy(phase_n_clusters)
        self._base_clustering_scheme = copy.deepcopy(base_clustering_scheme)
        self._eff_elastic_properties = copy.deepcopy(eff_elastic_properties)
        self._adaptive_clustering_scheme = \
            copy.deepcopy(adaptive_clustering_scheme)
        self._adaptivity_type = copy.deepcopy(adaptivity_type)
        self._gop_X_dft_vox = None
        self._cluster_phases = None
        self._adaptive_step = 0
        self._voxels_clusters = None
        self._phase_clusters = None
        self._clusters_vf = None
        self._cit_x_mf = None
        self._adaptivity_control_feature = \
            copy.deepcopy(adaptivity_control_feature)
        self._adapt_criterion_data = copy.deepcopy(adapt_criterion_data)
        self._adaptive_clustering_time = 0
        self._adaptive_cit_time = 0
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Get number of voxels on each dimension and total number of voxels
        self._n_voxels_dims = \
            [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
        self._n_voxels = np.prod(self._n_voxels_dims)
        # Get material phases' voxels' 1D flat indexes
        self._phase_voxel_flatidx = \
            type(self)._get_phase_idxs(regular_grid, self._material_phases)
        # Get adaptive material phases
        self._adapt_material_phases = \
            [x for x in self._clustering_type.keys()
             if self._clustering_type[x] == 'adaptive']
        # Set number of clusters prescribed (base clustering) for each material
        # phase
        self._base_phase_n_clusters = copy.deepcopy(self._phase_n_clusters)
    # -------------------------------------------------------------------------
    def perform_crve_base_clustering(self):
        """Compute CRVE base clustering."""
        info.displayinfo('5', 'Computing CRVE base clustering...')
        # Initialize base clustering labels
        labels = np.full(self._n_voxels, -1, dtype=int)
        # Initialize cluster-reduced material phases dictionary
        self._cluster_phases = {}
        # Initialize minimum cluster label
        min_label = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase in self._material_phases:
            info.displayinfo('5', 'Computing material phase ' + mat_phase
                             + ' base clustering...', 2)
            # Get material phase clustering type
            ctype = self._clustering_type[mat_phase]
            # Get material phase initial number of clusters
            n_phase_clusters = self._base_phase_n_clusters[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase's voxels' indexes
            voxels_idxs = self._phase_voxel_flatidx[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase cluster data matrix containing the required
            # data to perform all the prescribed cluster analyses (condensation
            # in terms of voxels only)
            cluster_data_matrix = mop.get_condensed_matrix(
                self._global_data_matrix, voxels_idxs,
                np.arange(self._global_data_matrix.shape[1]))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Instatiate cluster-reduced material phase
            if ctype == 'static':
                # Instantiate static cluster-reduced material phase
                crmp = SCRMP(mat_phase, cluster_data_matrix, n_phase_clusters)
            elif ctype == 'adaptive':
                # Get material phase adaptivity type
                atype = self._adaptivity_type[mat_phase]['adapt_type']
                # Instantiate adaptive cluster-reduced material phase
                if atype == GACRMP:
                    # Instantiate generalized adaptive cluster-reduced material
                    # phase
                    crmp = GACRMP(mat_phase, cluster_data_matrix,
                                  n_phase_clusters,
                                  self._adaptivity_type[mat_phase])
                elif atype == HAACRMP:
                    # Instantiate hierarchical-agglomerative adaptive
                    # cluster-reduced material phase
                    crmp = HAACRMP(mat_phase, cluster_data_matrix,
                                   n_phase_clusters,
                                   self._adaptivity_type[mat_phase])
                else:
                    raise RuntimeError('Unknown adaptivity type.')
            else:
                raise RuntimeError('Unknown material phase clustering type.')
            # Store cluster-reduced material phase
            self._cluster_phases[mat_phase] = crmp
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase base clustering scheme
            clustering_scheme = self._base_clustering_scheme[mat_phase]
            # Perform material phase base clustering
            crmp.perform_base_clustering(clustering_scheme, min_label)
            # Update minimum cluster label
            min_label = crmp.max_label + 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check for duplicated cluster labels
            if not set(crmp.cluster_labels).isdisjoint(labels):
                raise RuntimeError('Duplicated cluster labels between '
                                   'different material phases.')
            # Assemble material phase cluster labels
            labels[voxels_idxs] = crmp.cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build base CRVE
        self._voxels_clusters = np.reshape(np.array(labels, dtype=int),
                                           self._n_voxels_dims)
        # Compute base CRVE descriptors
        info.displayinfo('5', 'Computing CRVE base clustering descriptors...',
                         2)
        # Store cluster labels belonging to each material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
    # -------------------------------------------------------------------------
    def perform_crve_adaptivity(self, target_clusters, target_clusters_data):
        """Perform CRVE clustering adaptivity.

        Parameters
        ----------
        target_clusters : list[int]
            List with the labels (int) of clusters to be adapted.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict)
            containing cluster associated parameters relevant for the adaptive
            procedures.

        Returns
        -------
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels
            (item, list[int]) resulting from the refinement of each target
            cluster (key, str)) for each material phase (key, str).
        """
        # Check for duplicated target clusters
        if len(target_clusters) != len(np.unique(target_clusters)):
            raise RuntimeError('List of target clusters contains duplicated '
                               'labels.')
        # Check for unexistent target clusters
        if not set(target_clusters).issubset(
                set(self._voxels_clusters.flatten())):
            raise RuntimeError('List of target clusters contains unexistent '
                               'labels.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        init_time = time.time()
        # Increment adaptive clustering refinement step counter
        self._adaptive_step += 1
        # Get CRVE current clustering
        labels = self._voxels_clusters.flatten()
        # Initialize adaptive material phase's target clusters and associated
        # data
        phase_target_clusters = {mat_phase: []
                                 for mat_phase in self._adapt_material_phases}
        phase_target_clusters_data = \
            {mat_phase: [] for mat_phase in self._adapt_material_phases}
        # Build adaptive material phase's target clusters lists and associated
        # data
        for mat_phase in self._adapt_material_phases:
            phase_target_clusters[mat_phase] = \
                list(set(target_clusters).intersection(
                    self._phase_clusters[mat_phase]))
            phase_target_clusters_data[mat_phase] = \
                {str(cluster): target_clusters_data[str(cluster)]
                 for cluster in phase_target_clusters[mat_phase]}
        # Get CRVE maximum cluster label
        max_label = self._get_clusters_max_label()
        # Set minimum cluster label
        min_label = max_label + 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize adaptive clustering map
        adaptive_clustering_map = \
            {mat_phase: {} for mat_phase in self._adapt_material_phases}
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # If there are no target clusters, skip to next adaptive material
            # phase
            if not bool(phase_target_clusters[mat_phase]):
                continue
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Get adaptive clustering scheme
            adaptive_clustering_scheme = \
                self._adaptive_clustering_scheme[mat_phase]
            # Perform adaptive clustering
            adaptive_clustering_map[mat_phase] = \
                crmp.perform_adaptive_clustering(
                    phase_target_clusters[mat_phase],
                    phase_target_clusters_data[mat_phase],
                    adaptive_clustering_scheme, min_label)
            # Update minimum cluster label
            min_label = self._get_clusters_max_label() + 1
            # Update CRVE clustering
            labels[self._phase_voxel_flatidx[mat_phase]] = crmp.cluster_labels
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build adaptive CRVE
        self._voxels_clusters = np.reshape(np.array(labels, dtype=int),
                                           self._n_voxels_dims)
        # Store cluster labels and update number of clusters belonging to each
        # material phase
        self._set_phase_clusters()
        # Compute material clusters' volume fraction
        self._set_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in clustering adaptivity
        self._adaptive_clustering_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptive_clustering_map
    # -------------------------------------------------------------------------
    def get_rve_dims(self):
        """Get RVE dimensions.

        Returns
        -------
        rve_dims : list
            RVE size in each dimension.
        """
        return copy.deepcopy(self._rve_dims)
    # -------------------------------------------------------------------------
    def get_material_phases(self):
        """Get RVE material phases.

        Returns
        -------
        material_phases : list
            RVE material phases labels (str).
        """
        return copy.deepcopy(self._material_phases)
    # -------------------------------------------------------------------------
    def get_n_voxels(self):
        """Get number of voxels in each dimension and total number of voxels.

        Returns
        -------
        n_voxels_dims : list[int]
            Number of voxels in each dimension of the regular grid (spatial
            discretization of the RVE).
        n_voxels : int
            Total number of voxels of the regular grid (spatial discretization
            of the RVE).
        """
        return self._n_voxels_dims, self._n_voxels
    # -------------------------------------------------------------------------
    def get_regular_grid(self):
        """Get regular grid of voxels with material phase labels.

        Returns
        -------
        regular_grid : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the material phase label (int) assigned to the
            corresponding voxel.
        """
        return copy.deepcopy(self._regular_grid)
    # -------------------------------------------------------------------------
    def get_phase_n_clusters(self):
        """Get number of clusters associated with each material phase.

        Returns
        -------
        phase_n_clusters : dict
            Number of clusters (item, int) associated with each material phase
            (key, str).
        """
        return copy.deepcopy(self._phase_n_clusters)
    # -------------------------------------------------------------------------
    def get_phase_clusters(self):
        """Get clusters associated with each material phase.

        Returns
        -------
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        """
        return copy.deepcopy(self._phase_clusters)
    # -------------------------------------------------------------------------
    def get_voxels_clusters(self):
        """Get regular grid containing the cluster label of each voxel.

        Returns
        -------
        voxels_clusters : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the cluster label (int) assigned to the
            corresponding voxel.
        """
        return copy.deepcopy(self._voxels_clusters)
    # -------------------------------------------------------------------------
    def get_n_total_clusters(self):
        """Get current total number of clusters.

        Returns
        -------
        n_total_clusters : int
            Total number of clusters.
        """
        # Initialize total number of clusters
        n_total_clusters = 0
        # Loop over material phases
        for mat_phase in self._material_phases:
            n_total_clusters += len(self._phase_clusters[mat_phase])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return n_total_clusters
    # -------------------------------------------------------------------------
    def get_cluster_phases(self):
        """Get cluster-reduced material phase instance of each material phase.

        Returns
        -------
        cluster_phases : dict
            Cluster-Reduced material phase instance (item, CRMP) associated
            with each material phase (key, str).
        """
        return copy.deepcopy(self._cluster_phases)
    # -------------------------------------------------------------------------
    def get_clusters_vf(self):
        """Get clusters volume fraction.

        Returns
        -------
        clusters_vf : dict
            Volume fraction (item, float) associated with each material cluster
            (key, str).
        """
        return copy.deepcopy(self._clusters_vf)
    # -------------------------------------------------------------------------
    def get_cit_x_mf(self):
        """Get cluster interaction tensors (Green material independent terms).

        Returns
        -------
        cit_x_mf : list[dict]
            Cluster interaction tensors associated with the Green operator
            material independent terms. Each term is stored in a dictionary
            (item, dict) for each pair of material phases (key, str), which in
            turn contains the corresponding matricial form (item,
            numpy.ndarray (2d)) associated with each pair of clusters
            (key, str).
        """
        return self._cit_x_mf
    # -------------------------------------------------------------------------
    def get_eff_isotropic_elastic_constants(self):
        """Get isotropic elastic constants from elastic tangent modulus.

        Returns
        -------
        eff_elastic_properties : dict
            Elastic properties (key, str) and their values (item, float)
            estimated from the RVE's elastic effective tangent modulus.
        """
        return copy.deepcopy(self._eff_elastic_properties)
    # -------------------------------------------------------------------------
    def get_adapt_material_phases(self):
        """Get adaptive material phases labels.

        Returns
        -------
        adapt_material_phases : list[str]
            RVE adaptive material phases labels (str).
        """
        return copy.deepcopy(self._adapt_material_phases)
    # -------------------------------------------------------------------------
    def get_adaptivity_control_feature(self):
        """Get clustering adaptivity control feature of each material phase.

        Returns
        -------
        adaptivity_control_feature : dict, default=None
            Clustering adaptivity control feature (item, str) associated with
            each material phase (key, str).
        """
        return copy.deepcopy(self._adaptivity_control_feature)
    # -------------------------------------------------------------------------
    def get_adapt_criterion_data(self):
        """Get clustering adaptivity criterion data of each material phase.

        Returns
        -------
        adapt_criterion_data : dict, default=None
            Clustering adaptivity criterion (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            criterion to be used and the required parameters.
        """
        return copy.deepcopy(self._adapt_criterion_data)
    # -------------------------------------------------------------------------
    def get_voxels_array_variables(self):
        """Get required variables to build a clusters state based voxels array.

        Returns
        -------
        material_phases : list
            CRVE material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list of int) associated with each material
            phase (key, str).
        voxels_clusters : ndarray
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the cluster label (int) assigned to the
            corresponding voxel.
        """
        return self._material_phases, self._phase_clusters, \
            self._voxels_clusters
    # -------------------------------------------------------------------------
    def get_adaptive_step(self):
        """Get counter of adaptive clustering steps.

        Returns
        -------
        adaptive_step : int
            Counter of adaptive clustering steps, with 0 associated with the
            base clustering.
        """
        return self._adaptive_step
    # -------------------------------------------------------------------------
    def get_adaptive_clustering_time(self):
        """Get total amount of time spent in clustering adaptivity.

        Returns
        -------
        adaptive_clustering_time : float
            Total amount of time (s) spent in clustering adaptivity.
        """
        return self._adaptive_clustering_time
    # -------------------------------------------------------------------------
    def get_adaptive_cit_time(self):
        """Get total time spent in adaptivity cluster interaction tensors.

        Returns
        -------
        adaptive_cit_time : float
            Total amount of time (s) spent in clustering adaptivity cluster
            interaction tensors computation procedures.
        """
        return self._adaptive_cit_time
    # -------------------------------------------------------------------------
    def get_clustering_type(self):
        """Get clustering type of each material phase.

        Returns
        -------
        clustering_type : dict
            Clustering type (item, {'static', 'adaptive'}) of each material
            phase (key, str).
        """
        return copy.deepcopy(self._clustering_type)
    # -------------------------------------------------------------------------
    def get_adaptivity_output(self):
        """Get required data for clustering adaptivity output file.

        Returns
        -------
        adaptivity_output : dict
            For each adaptive material phase (key, str), stores a list (item)
            containing the adaptivity metrics associated with the clustering
            adaptivity output file.
        """
        # Initialize adaptivity output
        adaptivity_output = {}
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get adaptive cluster reduced material phase
            acrmp = self._cluster_phases[mat_phase]
            # Get material phase adaptivity metrics
            adaptivity_output[mat_phase] = \
                [*acrmp.get_adaptive_output()]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return adaptivity_output
    # -------------------------------------------------------------------------
    def get_clustering_summary(self):
        """Get summary of number of clusters of each material phase.

        Returns
        -------
        clustering_summary : dict
            For each material phase (key, str), stores list (item) containing
            the associated type ('static' or 'adaptive'), the base number of
            clusters (int) and the final number of clusters (int).
        """
        # Initialize clustering summary
        clustering_summary = {}
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Build material phase clustering summary
            clustering_summary[mat_phase] = \
                [self._clustering_type[mat_phase],
                 self._base_phase_n_clusters[mat_phase],
                 self._cluster_phases[mat_phase].get_n_clusters()]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return clustering_summary
    # -------------------------------------------------------------------------
    def reset_adaptive_parameters(self):
        """Reset CRVE adaptive progress parameters and set base clustering."""
        # Reset counter of adaptive clustering steps
        self._adaptive_step = 0
        # Set number of clusters prescribed (base clustering) for each material
        # phase
        self._base_phase_n_clusters = copy.deepcopy(self._phase_n_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset time spent in clustering adaptivity procedures
        self._adaptive_clustering_time = 0
        self._adaptive_cit_time = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Reset adaptive progress parameters
            crmp.reset_adaptive_parameters()
    # -------------------------------------------------------------------------
    def update_adaptive_parameters(self, adaptive_clustering_scheme,
                                   adapt_criterion_data, adaptivity_type,
                                   adaptivity_control_feature):
        """Update CRVE clustering adaptivity attributes.

        Parameters
        ----------
        adaptive_clustering_scheme : dict
            Prescribed adaptive clustering scheme (item, numpy.ndarray of shape
            (n_clusterings, 3)) for each material phase (key, str). Each row is
            associated with a unique clustering characterized by a clustering
            algorithm (col 1, int), a list of features (col 2, list[int]) and a
            list of the features data matrix' indexes (col 3, list[int]).
        adapt_criterion_data : dict
            Clustering adaptivity criterion (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            criterion to be used and the required parameters.
        adaptivity_type : dict
            Clustering adaptivity type (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            type to be used and the required parameters.
        adaptivity_control_feature : dict
            Clustering adaptivity control feature (item, str) associated with
            each material phase (key, str).
        """
        # Update prescribed adaptive clustering scheme
        self._adaptive_clustering_scheme = \
            copy.deepcopy(adaptive_clustering_scheme)
        # Update clustering adaptivity criterion data
        self._adapt_criterion_data = copy.deepcopy(adapt_criterion_data)
        # Update clustering adaptivity type data
        self._adaptivity_type = copy.deepcopy(adaptivity_type)
        # Update clustering adaptivity control feature data
        self._adaptivity_control_feature = \
            copy.deepcopy(adaptivity_control_feature)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get adaptive clustering scheme features indexes from the base
            # clustering scheme
            for i in range(
                    self._adaptive_clustering_scheme[mat_phase].shape[0]):
                self._adaptive_clustering_scheme[mat_phase][i, 2] = \
                    copy.deepcopy(self._base_clustering_scheme[
                        mat_phase][i, 2])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Update clustering adaptivity type parameters
            crmp.update_adaptivity_type(adaptivity_type[mat_phase])
    # -------------------------------------------------------------------------
    @staticmethod
    def get_crmp_types():
        """Get available cluster-reduced material phases types.

        Returns
        -------
        available_crmp_types : dict
            Available cluster-reduced material phase classes (item, CRMP) and
            associated identifiers (key, str).
        """
        # Set available cluster-reduced material phase types
        available_crmp_types = {'0': SCRMP,
                                '1': GACRMP,
                                '2': HAACRMP}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_crmp_types
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_features_indexes(clustering_scheme):
        """Get unique clustering features indexes from clustering scheme.

        Parameters
        ----------
        clustering_scheme : ndarry of shape (n_clusterings, 3)
            Clustering scheme (numpy.ndarray of shape (n_clusterings, 3)). Each
            row is associated with a unique clustering characterized by a
            clustering algorithm (col 1, int), a list of features
            (col 2, list[int]) and a list of the features data matrix' indexes
            (col 3, list[int]).

        Returns
        -------
        indexes : list[int]
            List of unique clustering features indexes.
        """
        # Collect prescribed clusterings features' indexes
        indexes = []
        for i in range(clustering_scheme.shape[0]):
            indexes += clustering_scheme[i, 2]
        # Get list of unique clustering features' indexes
        indexes = list(set(indexes))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return indexes
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_phase_idxs(regular_grid, material_phases):
        """Get flat indexes of each material phase's voxels.

        Parameters
        ----------
        regular_grid : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the material phase label (int) assigned to the
            corresponding voxel.
        material_phases : list[str]
            RVE material phases labels (str).

        Returns
        -------
        phase_voxel_flat_idx : dict
            Flat voxels' indexes (item, list[int]) associated with each
            material phase (key, str).
        """
        phase_voxel_flat_idx = dict()
        # Loop over material phases
        for mat_phase in material_phases:
            # Build boolean 'belongs to material phase' list
            is_phase_list = regular_grid.flatten() == int(mat_phase)
            # Get material phase's voxels' indexes
            phase_voxel_flat_idx[mat_phase] = list(
                it.compress(range(len(is_phase_list)), is_phase_list))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return phase_voxel_flat_idx
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_cluster_idxs(voxels_clusters, cluster_label):
        """Get flat indexes of given cluster's voxels.

        Parameters
        ----------
        voxels_clusters : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the cluster label (int) assigned to the
            corresponding voxel.
        cluster_label : int
            Cluster label.

        Returns
        -------
        cluster_voxel_flat_idx : list[int]
            Flat voxels' indexes (int) associated with given material cluster.
        """
        # Build boolean 'belongs to cluster' list
        is_cluster_list = voxels_clusters.flatten() == int(cluster_label)
        # Get material phase's voxels' indexes
        cluster_voxel_flat_idx = \
            list(it.compress(range(len(is_cluster_list)), is_cluster_list))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_voxel_flat_idx
    # -------------------------------------------------------------------------
    def _set_phase_clusters(self):
        """Set CRVE cluster labels associated with each material phase."""
        self._phase_clusters = {}
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get cluster labels
            self._phase_clusters[mat_phase] = \
                list(np.unique(self._voxels_clusters.flatten()[
                               self._phase_voxel_flatidx[mat_phase]]))
            # Update material phase number of clusters
            self._phase_n_clusters[mat_phase] = \
                len(self._phase_clusters[mat_phase])
    # -------------------------------------------------------------------------
    def _set_clusters_vf(self):
        """Set CRVE clusters' volume fractions."""
        # Compute voxel volume
        voxel_vol = np.prod([float(self._rve_dims[i])/self._n_voxels_dims[i]
                             for i in range(len(self._rve_dims))])
        # Compute RVE volume
        rve_vol = np.prod(self._rve_dims)
        # Compute volume fraction associated with each material cluster
        self._clusters_vf = {}
        for cluster in np.unique(self._voxels_clusters):
            n_voxels_cluster = np.sum(self._voxels_clusters == cluster)
            self._clusters_vf[str(cluster)] = \
                (n_voxels_cluster*voxel_vol)/rve_vol
    # -------------------------------------------------------------------------
    def _get_clusters_max_label(self):
        """Get CRVE maximum cluster label.

        Returns
        -------
        max_label : int
            CRVE maximum cluster label.
        """
        # Initialize maximum cluster label
        max_label = -1
        # Loop over material phases
        for mat_phase in self._material_phases:
            # Get cluster-reduced material phase
            crmp = self._cluster_phases[mat_phase]
            # Update maximum cluster label
            max_label = max(max_label, crmp.max_label)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return max_label
    # -------------------------------------------------------------------------
    def _sort_cluster_labels(self):
        """Reassign and sort CRVE cluster labels material phasewise.

        Reassign CRVE cluster labels in the range (0, n_clusters) and sort them
        in ascending order of material phase's labels.
        """
        # Initialize material phase initial cluster label
        lbl_init = 0
        # Initialize old cluster labels
        old_clusters = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize mapping dictionary to sort the cluster labels in asceding
        # order of material phase
        sort_dict = dict()
        # Loop over material phases sorted in ascending order
        sorted_mat_phases = list(np.sort(list(self._material_phases)))
        for mat_phase in sorted_mat_phases:
            # Get material phase old cluster labels
            phase_old_clusters = np.unique(self._voxels_clusters.flatten()[
                self._phase_voxel_flatidx[mat_phase]])
            # Set material phase new cluster labels
            phase_new_clusters = list(
                range(lbl_init, lbl_init + self._phase_n_clusters[mat_phase]))
            # Build mapping dictionary to sort the cluster labels
            for i in range(self._phase_n_clusters[mat_phase]):
                if phase_old_clusters[i] in sort_dict.keys():
                    raise RuntimeError('Cluster label (key) already exists in '
                                       'cluster labels mapping dictionary.')
                else:
                    sort_dict[phase_old_clusters[i]] = phase_new_clusters[i]
            # Set next material phase initial cluster label
            lbl_init = lbl_init + self._phase_n_clusters[mat_phase]
            # Append old cluster labels
            old_clusters += list(phase_old_clusters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check cluster labels mapping dictionary
        new_clusters = [sort_dict[key] for key in sort_dict.keys()]
        if set(sort_dict.keys()) != set(old_clusters) or \
                len(set(new_clusters)) != len(set(old_clusters)):
            raise RuntimeError('Invalid cluster labels mapping dictionary.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort cluster labels in ascending order of material phase
        for voxel_idx in it.product(*[list(range(self._n_voxels_dims[i]))
                                    for i in range(len(self._n_voxels_dims))]):
            self._voxels_clusters[voxel_idx] = \
                sort_dict[self._voxels_clusters[voxel_idx]]
    # -------------------------------------------------------------------------
    def compute_cit(self, mode='full', adaptive_clustering_map=None):
        """Compute CRVE cluster interaction tensors.

        *Cluster interaction tensors:*

        .. math::

           \\boldsymbol{\\mathsf{T}}^{(I)(J)} = \\dfrac{1}{f^{(I)} v_{\\mu}}
           \\int_{\\Omega_{\\mu, \\, 0}} \\int_{\\Omega_{\\mu, \\, 0}}
           \\chi^{(I)}(\\boldsymbol{Y}) \\, \\chi^{(J)} (\\boldsymbol{Y}') \\,
           \\boldsymbol{\\mathsf{\\Phi}}^{0} (\\boldsymbol{Y}-\\boldsymbol{Y}')
           \\, \\mathrm{d} v' \\mathrm{d} v \\, ,

        .. math::
           \\quad I,J = 1,2,\\dots, n_{\\mathrm{c}}

        where :math:`\\boldsymbol{\\mathsf{T}}^{(I)(J)}` is the cluster
        interaction tensor (fourth-order tensor) between the :math:`I` th
        and :math:`J` th material clusters, :math:`f^{(I)}` is the volume
        fraction of the :math:`I` th material cluster, :math:`v_{\\mu}` is
        the volume of the CRVE, :math:`\\boldsymbol{Y}` and
        :math:`\\boldsymbol{Y'}` are points of the microscale reference
        configuration (:math:`\\Omega_{\\mu,\\,0}`), :math:`\\chi^{(I)}` is the
        characteristic function of the :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{\\Phi}}^{0}` is the reference material
        Green operator (fourth-order tensor), and :math:`n_{c}` is the number
        of material clusters.

        The detailed description of the custer interaction tensors can
        be found in Section 4.3 of Ferreira (2022) [#]_.

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        *Adaptive update of cluster interaction matrix:*

        The cluster interaction tensors `adaptive` computation mode can only be
        performed after at least one `full` computation has been performed.

        A detailed description of the `adaptive` update of the cluster
        interaction matrix can be found in Section 2.3.3 of
        Ferreira et. al (2022) [#]_.

        .. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
               *Adaptivity for clustering-based reduced-order modeling of
               localized history-dependent phenomena.* Comp Methods Appl M, 393
               (see `here <https://www.sciencedirect.com/science/article/pii/
               S0045782522000895?via%3Dihub>`_)

        ----

        Parameters
        ----------
        mode : {'full', 'adaptive'}, default='full'
            The default `full` mode performs the complete computation of all
            cluster interaction tensors. The 'adaptive' mode speeds up the
            computation of the new cluster interaction tensors resulting from
            an adaptive clustering characterized by `adaptive_clustering_map`.
        adaptive_clustering_map : dict
            Adaptive clustering map (item, dict with list of new cluster labels
            (item, list[int]) resulting from the refinement of each target
            cluster (key, str)) for each material phase (key, str). Required in
            `adaptive` mode, otherwise ignored.
        """
        # Check parameters
        if mode not in ['full', 'adaptive']:
            raise RuntimeError('Unknown mode to compute cluster interaction '
                               'tensors.')
        elif mode == 'adaptive' and adaptive_clustering_map is None:
            raise RuntimeError('Adaptive clustering map must be provided in '
                               '`adaptive` mode.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform mode-specific initialization procedures
        if mode == 'full':
            info.displayinfo('5',
                             'Computing CRVE cluster interaction tensors...')
            # Initialize cluster interaction tensors dictionary
            self._cit_x_mf = [{} for i in range(3)]
            for mat_phase_B in self._material_phases:
                for mat_phase_A in self._material_phases:
                    for i in range(len(self._cit_x_mf)):
                        self._cit_x_mf[i][mat_phase_A + '_' + mat_phase_B] = {}
            # Compute Green operator material independent terms
            self._gop_X_dft_vox = citop.gop_material_independent_terms(
                self._strain_formulation, self._problem_type, self._rve_dims,
                self._n_voxels_dims)
        elif mode == 'adaptive':
            init_time = time.time()
            # Build lists with old (preexistent) clusters and new (adapted)
            # clusters for each material phase. Also get list of clusters that
            # no longer exist due to clustering adaptivity
            phase_old_clusters = {}
            phase_new_clusters = {}
            pop_clusters = []
            for mat_phase in self._material_phases:
                if mat_phase in adaptive_clustering_map.keys():
                    phase_new_clusters[mat_phase] = \
                        sum(adaptive_clustering_map[mat_phase].values(), [])
                    pop_clusters += list(
                        adaptive_clustering_map[mat_phase].keys())
                else:
                    phase_new_clusters[mat_phase] = []
                phase_old_clusters[mat_phase] = list(
                    set(self._phase_clusters[mat_phase])
                    - set(phase_new_clusters[mat_phase]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over material phases
        for mat_phase_B in self._material_phases:
            # Set material phase B clusters to be looped over
            if mode == 'full':
                clusters_J = self._phase_clusters[mat_phase_B]
            elif mode == 'adaptive':
                clusters_J = phase_new_clusters[mat_phase_B]
            # Loop over material phase B clusters
            for cluster_J in clusters_J:
                # Set material phase B cluster characteristic function
                _, cluster_J_filter_dft = self._cluster_filter(cluster_J)
                # Perform discrete convolution between the material phase B
                # cluster characteristic function and each of Green operator
                # material independent terms
                gop_X_filt_vox = self._gop_convolution(cluster_J_filter_dft,
                                                       *self._gop_X_dft_vox)
                # Loop over material phases
                for mat_phase_A in self._material_phases:
                    # Set material phase pair dictionary
                    mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                    # Loop over material phase A clusters
                    for cluster_I in self._phase_clusters[mat_phase_A]:
                        # Set material cluster pair
                        cluster_pair = str(cluster_I) + '_' + str(cluster_J)
                        # Check if cluster-symmetric cluster interaction tensor
                        sym_cluster_pair = self._switch_pair(cluster_pair)
                        sym_mat_phase_pair = self._switch_pair(mat_phase_pair)
                        is_clst_sym = sym_cluster_pair in \
                            self._cit_x_mf[0][sym_mat_phase_pair].keys()
                        # Compute cluster interaction tensor between material
                        # phase A cluster and material phase B cluster
                        # (complete computation or cluster-symmetric
                        # computation)
                        if is_clst_sym:
                            # Set cluster volume fractions ratio
                            clst_vf_ratio = (
                                self._clusters_vf[str(cluster_J)]
                                / self._clusters_vf[str(cluster_I)])
                            # Compute clustering interaction tensor between
                            # material phase A cluster and material phase B
                            # cluster through cluster-symmetry
                            for cit_mf in self._cit_x_mf:
                                cit_mf[mat_phase_pair][cluster_pair] = \
                                    np.multiply(clst_vf_ratio,
                                                cit_mf[sym_mat_phase_pair][
                                                    sym_cluster_pair])
                        else:
                            # Set material phase A cluster characteristic
                            # function
                            cluster_I_filter, _ = \
                                self._cluster_filter(cluster_I)
                            # Perform discrete integral over the spatial domain
                            # of material phase A cluster I
                            cit_X_integral_mf = self._discrete_cit_integral(
                                cluster_I_filter, *gop_X_filt_vox)
                            # Compute cluster interaction tensor between the
                            # material phase A cluster and the material phase B
                            # cluster
                            rve_vol = np.prod(self._rve_dims)
                            factor = 1.0/(self._clusters_vf[str(cluster_I)]
                                          * rve_vol)
                            for i in range(len(self._cit_x_mf)):
                                self._cit_x_mf[i][mat_phase_pair][
                                    cluster_pair] = np.multiply(
                                        factor, cit_X_integral_mf[i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute remaining adaptive cluster interaction tensors through
        # cluster-symmetry and remove vanished clustering interaction tensors
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
                            cluster_pair = str(cluster_I) + '_' \
                                + str(cluster_J)
                            # Check if cluster-symmetric cluster interaction
                            # tensor
                            sym_cluster_pair = self._switch_pair(cluster_pair)
                            sym_mat_phase_pair = \
                                self._switch_pair(mat_phase_pair)
                            is_clst_sym = sym_cluster_pair in \
                                self._cit_x_mf[0][sym_mat_phase_pair].keys()
                            # Compute cluster interaction tensor between
                            # material phase A cluster and material phase B
                            # cluster through cluster-symmetry
                            if not is_clst_sym:
                                raise RuntimeError(
                                    'All the remaining adaptive clustering '
                                    'interaction tensors should be '
                                    'cluster-symmetric.')
                            # Set cluster volume fractions ratio
                            clst_vf_ratio = (
                                self._clusters_vf[str(cluster_J)]
                                / self._clusters_vf[str(cluster_I)])
                            # Compute clustering interaction tensor
                            for cit_mf in self._cit_x_mf:
                                cit_mf[mat_phase_pair][cluster_pair] = \
                                    np.multiply(clst_vf_ratio,
                                                cit_mf[sym_mat_phase_pair][
                                                    sym_cluster_pair])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phases
            for mat_phase_B in self._material_phases:
                # Loop over material phases
                for mat_phase_A in self._material_phases:
                    # Set material phase pair dictionary
                    mat_phase_pair = mat_phase_A + '_' + mat_phase_B
                    # Set existent cluster interactions
                    cluster_pairs = [x for x in
                                     self._cit_x_mf[0][mat_phase_pair].keys()]
                    # Loop over cluster pairs
                    for cluster_pair in cluster_pairs:
                        cluster_I = cluster_pair.split('_')[0]
                        cluster_J = cluster_pair.split('_')[1]
                        # If any of the interacting clusters no longer exists,
                        # then remove the associated cluster interaction tensor
                        if cluster_I in pop_clusters \
                                or cluster_J in pop_clusters:
                            for i in range(len(self._cit_x_mf)):
                                self._cit_x_mf[i][mat_phase_pair].pop(
                                    cluster_pair)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update total amount of time spent in clustering adaptivity
            # cluster interaction tensors computation procedures
            self._adaptive_cit_time += time.time() - init_time
    # -------------------------------------------------------------------------
    def _cluster_filter(self, cluster):
        """Compute cluster discrete characteristic function.

        .. math::

           \\chi^{(I)} (\\boldsymbol{Y}) = \\begin{cases} 1 \\quad \\text{if}
           \\; \\; \\; \\boldsymbol{Y}\\in \\Omega^{(I)}_{\\mu, \\, 0} \\\\ 0
           \\quad \\text{otherwise} \\end{cases}

        where :math:`\\chi^{(I)}` is the
        characteristic function of the :math:`I` th material cluster and
        :math:`\\boldsymbol{Y}` is a point of the microscale reference
        configuration (:math:`\\Omega_{\\mu,\\,0}`).

        The detailed description of the cluster characteristic function can
        be found in Section 4.3.1 of Ferreira (2022) [#]_.

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        cluster : int
            Cluster label.

        Returns
        -------
        cluster_filter : numpy.ndarray[bool] (2d or 3d)
            Cluster discrete characteristic function in spatial domain.
        cluster_filter_dft : numpy.ndarray
            Cluster discrete characteristic function in frequency domain
            (discrete Fourier transform).
        """
        # Check if valid cluster
        if not isinstance(cluster, int) \
                and not isinstance(cluster, np.integer):
            raise RuntimeError('Cluster label must be an integer.')
        elif cluster not in self._voxels_clusters:
            raise RuntimeError('Cluster label does not exist in the CRVE.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build cluster filter (spatial domain)
        cluster_filter = self._voxels_clusters == cluster
        # Perform Discrete Fourier Transform (DFT) by means of Fast Fourier
        # Transform (FFT)
        cluster_filter_dft = np.fft.fftn(cluster_filter)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cluster_filter, cluster_filter_dft
    # -------------------------------------------------------------------------
    def _gop_convolution(self, cluster_filter_dft, gop_1_dft_vox,
                         gop_2_dft_vox, gop_0_freq_dft_vox):
        """Convolution of cluster characteristic function and Green operator.

        .. math::

           \\int_{\\Omega_{\\mu, \\, 0}} \\chi^{(J)} (\\boldsymbol{Y}') \\,
           \\boldsymbol{\\mathsf{\\Phi}}^{0} (\\boldsymbol{Y} -
           \\boldsymbol{Y}') \\, \\mathrm{d} v' = \\mathscr{F}^{-1} \\left(
           \\breve{\\chi}^{(J)}(\\boldsymbol{\\zeta}) \\,
           \\breve{\\boldsymbol{\\mathsf{\\Phi}}}^{0} (\\boldsymbol{\\zeta})
           \\right) \\, ,

        where :math:`\\chi^{(J)}` is the characteristic function of the
        :math:`J` th material cluster,
        :math:`\\boldsymbol{\\mathsf{\\Phi}}^{0}` is the reference material
        Green operator (fourth-order tensor), :math:`\\boldsymbol{Y}` and
        :math:`\\boldsymbol{Y'}` are points of the microscale reference
        configuration (:math:`\\Omega_{\\mu,\\,0}`), and
        :math:`\\boldsymbol{\\zeta}` is the frequency wave vector. The operator
        :math:`\\mathscr{F}^{-1}(\\cdot)` denotes the inverse Fourier transform
        and :math:`\\breve{(\\cdot)}` denotes a field defined in the frequency
        domain.

        Such a computation is required to compute the cluster interaction
        tensors. More information can be found in Ferreira (2022) [#]_
        (see Equations (4.111) and surrounding text).

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        cluster_filter_dft : numpy.ndarray
            Cluster discrete characteristic function in frequency domain
            (discrete Fourier transform).
        gop_1_dft_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the first Green
            operator material independent term in the frequency domain
            (discrete Fourier transform).
        gop_2_dft_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the second
            Green operator material independent term in the frequency domain
            (discrete Fourier transform).
        gop_0_freq_dft_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the Green
            operator zero-frequency (material independent) term in the
            frequency domain (discrete Fourier transform).

        Returns
        -------
        gop_1_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the
            first Green operator material independent term in the spatial
            domain (inverse discrete Fourier transform).
        gop_2_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the
            second Green operator materia independent term in the spatial
            domain (inverse discrete Fourier transform).
        gop_0_freq_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the
            zero-frequency Green operator (material independent) term in the
            spatial domain (inverse discrete Fourier transform).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over Green operator components
        for i in range(len(comp_order)):
            compi = comp_order[i]
            for j in range(len(comp_order)):
                compj = comp_order[j]
                # Perform discrete convolution in the frequency domain
                gop_1_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels),
                                np.multiply(cluster_filter_dft,
                                            gop_1_filt_dft_vox[compi + compj]))
                gop_2_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels),
                                np.multiply(cluster_filter_dft,
                                            gop_2_filt_dft_vox[compi + compj]))
                gop_0_freq_filt_dft_vox[compi + compj] = \
                    np.multiply((rve_vol/n_voxels), np.multiply(
                        cluster_filter_dft,
                        gop_0_freq_filt_dft_vox[compi + compj]))
                # Perform an Inverse Discrete Fourier Transform (IDFT) by means
                # of Fast Fourier Transform (FFT)
                gop_1_filt_vox[compi + compj] = \
                    np.real(np.fft.ifftn(gop_1_filt_dft_vox[compi + compj]))
                gop_2_filt_vox[compi + compj] = \
                    np.real(np.fft.ifftn(gop_2_filt_dft_vox[compi + compj]))
                gop_0_freq_filt_vox[compi + compj] = np.real(np.fft.ifftn(
                    gop_0_freq_filt_dft_vox[compi + compj]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return gop_1_filt_vox, gop_2_filt_vox, gop_0_freq_filt_vox
    # -------------------------------------------------------------------------
    def _discrete_cit_integral(self, cluster_filter, gop_1_filt_vox,
                               gop_2_filt_vox, gop_0_freq_filt_vox):
        """Discrete integral over the spatial domain of material cluster.

        .. math::

           \\int_{\\Omega_{\\mu, 0}} \\chi^{(I)}(\\boldsymbol{Y}) \\left(
           \\int_{\\Omega_{\\mu, 0}}  \\, \\chi^{(J)}(\\boldsymbol{Y}) \\,
           \\boldsymbol{\\Phi}^{0}(\\boldsymbol{Y}-\\boldsymbol{Y}') \\,
           \\mathrm{d}v' \\right) \\mathrm{d}v \\, ,

        where :math:`\\chi^{(I)}` is the characteristic function of the
        :math:`I` th material cluster,
        :math:`\\boldsymbol{\\mathsf{\\Phi}}^{0}` is the reference material
        Green operator (fourth-order tensor), and :math:`\\boldsymbol{Y}` and
        :math:`\\boldsymbol{Y'}` are points of the microscale reference
        configuration (:math:`\\Omega_{\\mu,\\,0}`).

        Such a computation is required to compute the cluster interaction
        tensors. More information can be found in Ferreira (2022) [#]_
        (see Equations (4.112) and surrounding text).

        .. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
               Optimization of Thermoplastic Blends: Microstructural
               Generation, Constitutive Development and Clustering-based
               Reduced-Order Modeling.* PhD Thesis, University of Porto
               (see `here <https://repositorio-aberto.up.pt/handle/10216/
               146900?locale=en>`_)

        ----

        Parameters
        ----------
        cluster_filter : numpy.ndarray
            Cluster discrete characteristic function in spatial domain.
        gop_1_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the first
            Green operator material independent term in the spatial domain
            (inverse discrete Fourier transform).
        gop_2_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the second
            Green operator material independent term in the spatial domain
            (inverse discrete Fourier transform).
        gop_0_freq_filt_vox : dict
            Regular grid shaped matrix (item, numpy.ndarray) containing each
            fourth-order matricial form component (key, str) of the convolution
            between the material cluster characteristic function and the
            zero-frequency Green operator (material independent) term in the
            spatial domain (inverse discrete Fourier transform).

        Returns
        -------
        cit_1_integral_mf : numpy.ndarray (2d)
            Discrete integral over the spatial domain of material cluster I of
            the discrete convolution between the material cluster J
            characteristic function and the first Green operator material
            independent term in the spatial domain (numpy.ndarray of shape
            (n_comps, n_comps)).
        cit_2_integral_mf : numpy.ndarray (2d)
            Discrete integral over the spatial domain of material cluster I of
            the discrete convolution between the material cluster J
            characteristic function and the second Green operator material
            independent term in the spatial domain (numpy.ndarray of shape
            (n_comps, n_comps)).
        cit_0_freq_integral_mf : numpy.ndarray (2d)
            Discrete integral over the spatial domain of material cluster I of
            the discrete convolution between the material cluster J
            characteristic function and the zero-frequency Green operator
            (material independent) term in the spatial domain.
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize discrete integral
        cit_1_integral_mf = np.zeros((len(comp_order), len(comp_order)))
        cit_2_integral_mf = np.zeros((len(comp_order), len(comp_order)))
        cit_0_freq_integral_mf = np.zeros((len(comp_order), len(comp_order)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over matricial form components
        for i in range(len(comp_order)):
            compi = comp_order[i]
            for j in range(len(comp_order)):
                compj = comp_order[j]
                # Perform discrete integral over the spatial domain of material
                # cluster I
                cit_1_integral_mf[i, j] = mop.kelvin_factor(i, comp_order) \
                    * mop.kelvin_factor(j, comp_order) \
                    * np.sum(np.multiply(cluster_filter,
                                         gop_1_filt_vox[compi + compj]))
                cit_2_integral_mf[i, j] = mop.kelvin_factor(i, comp_order) \
                    * mop.kelvin_factor(j, comp_order) \
                    * np.sum(np.multiply(cluster_filter,
                                         gop_2_filt_vox[compi + compj]))
                cit_0_freq_integral_mf[i, j] = \
                    mop.kelvin_factor(i, comp_order) \
                    * mop.kelvin_factor(j, comp_order) \
                    * np.sum(np.multiply(cluster_filter,
                                         gop_0_freq_filt_vox[compi + compj]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cit_1_integral_mf, cit_2_integral_mf, cit_0_freq_integral_mf
    # -------------------------------------------------------------------------
    @staticmethod
    def _switch_pair(x, delimiter='_'):
        """Switch left and right sides of string with separating delimiter.

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
        """
        if not isinstance(x, str) or x.count(delimiter) != 1:
            raise RuntimeError('Input parameter must be a string and can '
                               'only contain one delimiter.')
        return delimiter.join(x.split(delimiter)[::-1])
    # -------------------------------------------------------------------------
    @staticmethod
    def save_crve_file(crve, crve_file_path):
        """Dump CRVE into file.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        crve_file_path : str
            Path of file where the CRVE's instance is dumped.
        """
        # Dump CRVE instance into file
        with open(crve_file_path, 'wb') as crve_file:
            pickle.dump(crve, crve_file)
