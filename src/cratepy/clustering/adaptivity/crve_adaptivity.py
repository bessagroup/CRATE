"""Clustering adaptivity.

This module includes the class that has major control over the clustering
adaptivity procedures. It also includes a class that handles the clustering
adaptivity output file.

The so-called adaptive clustering-based reduced-order models were introduced
by Ferreira et. al (2022) [#]_.

.. [#] Ferreira, B.P., Andrade Pires, F.M. and Bessa, M.A. (2022).
       *Adaptivity for clustering-based reduced-order modeling of
       localized history-dependent phenomena.* Comp Methods Appl M, 393
       (see `here <https://www.sciencedirect.com/science/article/pii/
       S0045782522000895?via%3Dihub>`_)

Classes
-------
AdaptivityManager
    CRVE clustering adaptivity manager.
ClusteringAdaptivityOutput
    Clustering adaptivity output.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import time
import re
# Third-party
import numpy as np
# Local
import ioput.info as info
import tensor.matrixoperations as mop
import ioput.ioutilities as ioutil
from material.materialoperations import compute_spatial_log_strain, \
                                        cauchy_from_first_piola, \
                                        MaterialQuantitiesComputer
from clustering.adaptivity.adaptivity_criterion import \
    AdaptiveClusterGrouping, SpatialDiscontinuities
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class AdaptivityManager:
    """CRVE clustering adaptivity manager.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _adapt_phase_criterions : dict
        Clustering adaptivity criterion instance (item, AdaptivityCriterion)
        associated with each material phase (key, str).
    _inc_adaptive_steps : dict
        For each macroscale loading increment (key, str), store the performed
        number of clustering adaptive steps (item, int).
    _adapt_ref_init_inc : dict
        Clustering adaptivity reference initial increment (item, int)
        associated with each material phase (key, str).
    _adapt_feature_min_trigger : dict
        Clustering adaptivity feature minimum significant value surpassed
        status (item, bool) associated with each material phase (key, str).
    max_inc_adaptive_steps : int
        Maximum number of clustering adaptive steps per macroscale loading
        increment.
    adaptive_evaluation_time : float
        Total amount of time (s) spent in selecting target clusters for
        clustering adaptivity.
    adaptive_time : float
        Total amount of time (s) spent in clustering adaptivity procedures.

    Methods
    -------
    get_adaptivity_criterions()
        Get available clustering adaptivity criterions.
    get_target_clusters(self, phase_clusters, voxels_clusters, clusters_state,\
                        clusters_def_gradient_mf, \
                        clusters_def_gradient_old_mf, clusters_state_old, \
                        clusters_sct_mf, clusters_sct_old_mf, \
                        clusters_residuals_mf, inc=None, verbose=False)
        Get adaptive clustering target clusters.
    _get_adaptivity_data_matrix(self, target_phase, adapt_control_feature, \
                                phase_clusters, clusters_def_gradient_mf, \
                                clusters_def_gradient_old_mf, \
                                clusters_state, clusters_state_old, \
                                clusters_sct_mf, clusters_sct_old_mf, \
                                clusters_residuals_mf)
        Build adaptivity feature data matrix for target material phase.
    adaptive_refinement(self, crve, material_state, target_clusters, \
                        target_clusters_data, inc, improved_init_guess=None, \
                        verbose=False)
        Perform CRVE adaptive clustering refinement.
    check_inc_adaptive_steps(self, inc)
        Check number of adaptive steps performed in loading increment.
    clear_inc_adaptive_steps(self, inc_threshold)
        Reset number of adaptive steps performed after threshold increment.
    is_adapt_phase_activated(self, mat_phase, inc)
        Check if material phase adaptivity procedures are activated.
    reset_adapt_activation_parameters(self)
        Reset parameters associated with clustering adaptivity activation.
    get_adapt_vtk_array(self, voxels_clusters)
        Get regular grid array with the adaptive level of each cluster.
    """
    def __init__(self, strain_formulation, problem_type, adapt_material_phases,
                 phase_clusters, adaptivity_control_feature,
                 adapt_criterion_data, clust_adapt_freq):
        """Constructor.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type identifier (1 - Plain strain (2D), 4- Tridimensional)
        adapt_material_phases : list[str]
            RVE adaptive material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        adaptivity_control_feature : dict
            Clustering adaptivity control feature (item, str) associated with
            each material phase (key, str).
        adapt_criterion_data : dict
            Clustering adaptivity criterion (item, dict) associated with each
            material phase (key, str). This dictionary contains the adaptivity
            criterion to be used and the required parameters.
        clust_adapt_freq : dict
            Clustering adaptivity frequency (relative to the macroscale
            loading) (item, int, default=1) associated with each adaptive
            cluster-reduced material phase (key, str).
        """
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._adapt_material_phases = copy.deepcopy(adapt_material_phases)
        self._adaptivity_control_feature = \
            copy.deepcopy(adaptivity_control_feature)
        self._adapt_criterion_data = copy.deepcopy(adapt_criterion_data)
        self._clust_adapt_freq = copy.deepcopy(clust_adapt_freq)
        self._inc_adaptive_steps = {}
        self._adapt_ref_init_inc = {mat_phase: 0
                                    for mat_phase in adapt_material_phases}
        self._adapt_feature_min_trigger = {mat_phase: False for mat_phase
                                           in adapt_material_phases}
        self.max_inc_adaptive_steps = 1
        self.adaptive_evaluation_time = 0
        self.adaptive_time = 0
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Loop over adaptive material phases
        self._adapt_phase_criterions = {}
        for mat_phase in adapt_material_phases:
            # Get adaptive material phase clustering adaptivity criterion
            adapt_criterion = adapt_criterion_data[mat_phase]['criterion']
            # Initialize adaptive material phase clustering adaptivity
            # criterion
            if adapt_criterion is AdaptiveClusterGrouping:
                # Get clustering adaptivity criterion parameters
                adapt_trigger_ratio = \
                    adapt_criterion_data[mat_phase]['adapt_trigger_ratio']
                adapt_max_level = \
                    adapt_criterion_data[mat_phase]['adapt_max_level']
                adapt_min_voxels = \
                    adapt_criterion_data[mat_phase]['adapt_min_voxels']
                adapt_split_threshold = \
                    adapt_criterion_data[mat_phase]['adapt_split_threshold']
                is_merge_adapt_groups = bool(adapt_criterion_data[mat_phase][
                                             'is_merge_adapt_groups'])
                min_adapt_feature_val = \
                    adapt_criterion_data[mat_phase]['min_adapt_feature_val']
                # Initialize clustering adaptivity criterion
                self._adapt_phase_criterions[mat_phase] = \
                    AdaptiveClusterGrouping(
                        mat_phase, phase_clusters,
                        adapt_trigger_ratio=adapt_trigger_ratio,
                        adapt_split_threshold=adapt_split_threshold,
                        adapt_max_level=adapt_max_level,
                        adapt_min_voxels=adapt_min_voxels,
                        is_merge_adapt_groups=is_merge_adapt_groups,
                        min_adapt_feature_val=min_adapt_feature_val)
            elif adapt_criterion == SpatialDiscontinuities:
                # Get clustering adaptivity criterion parameters
                adapt_trigger_ratio = \
                    adapt_criterion_data[mat_phase]['adapt_trigger_ratio']
                adapt_max_level = \
                    adapt_criterion_data[mat_phase]['adapt_max_level']
                adapt_min_voxels = \
                    adapt_criterion_data[mat_phase]['adapt_min_voxels']
                adapt_level_max_diff = \
                    adapt_criterion_data[mat_phase]['adapt_level_max_diff']
                swipe_dim_1_every = \
                    adapt_criterion_data[mat_phase]['swipe_dim_1_every']
                swipe_dim_2_every = \
                    adapt_criterion_data[mat_phase]['swipe_dim_2_every']
                swipe_dim_3_every = \
                    adapt_criterion_data[mat_phase]['swipe_dim_3_every']
                min_adapt_feature_val = \
                    adapt_criterion_data[mat_phase]['min_adapt_feature_val']
                magnitude_lower_factor = \
                    adapt_criterion_data[mat_phase]['magnitude_lower_factor']
                # Initialize clustering adaptivity criterion
                self._adapt_phase_criterions[mat_phase] = \
                    SpatialDiscontinuities(
                        mat_phase, phase_clusters,
                        adapt_trigger_ratio=adapt_trigger_ratio,
                        adapt_max_level=adapt_max_level,
                        adapt_min_voxels=adapt_min_voxels,
                        adapt_level_max_diff=adapt_level_max_diff,
                        swipe_dim_1_every=swipe_dim_1_every,
                        swipe_dim_2_every=swipe_dim_2_every,
                        swipe_dim_3_every=swipe_dim_3_every,
                        min_adapt_feature_val=min_adapt_feature_val,
                        magnitude_lower_factor=magnitude_lower_factor)
            else:
                raise RuntimeError('Unknown clustering adaptivity criterion.')
    # -------------------------------------------------------------------------
    @staticmethod
    def get_adaptivity_criterions():
        """Get available clustering adaptivity criterions.

        Returns
        -------
        available_adapt_criterions : dict
            Available clustering adaptivity criterions
            (item, AdaptivityCriterion) and associated identifiers (key, str).
        """
        # Set available clustering adaptivity criterions
        available_adapt_criterions = {'1': AdaptiveClusterGrouping,
                                      '2': SpatialDiscontinuities}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_adapt_criterions
    # -------------------------------------------------------------------------
    def get_target_clusters(self, phase_clusters, voxels_clusters,
                            clusters_state, clusters_def_gradient_mf,
                            clusters_def_gradient_old_mf, clusters_state_old,
                            clusters_sct_mf, clusters_sct_old_mf,
                            clusters_residuals_mf, inc=None, verbose=False):
        """Get adaptive clustering target clusters.

        Parameters
        ----------
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.ndarray (1d)) associated with
            each material cluster (key, str), stored in matricial form.
        clusters_def_gradient_old_mf : dict
            Last converged deformation gradient (item, numpy.ndarray (1d))
            associated with each material cluster (key, str), stored in
            matricial form.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            with each material cluster (key, str).
        clusters_state_old : dict
            Last increment converged material constitutive model state
            variables (item, dict) associated with each material cluster
            (key, str).
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form)
            (item, numpy.ndarray) associated with each material cluster
            (key, str).
        clusters_sct_old_mf : dict
            Last increment converged fourth-order strain concentration tensor
            (matricial form) (item, numpy.ndarray) associated with each
            material cluster (key, str).
        clusters_residuals_mf : dict
            Equilibrium residual second-order tensor (matricial form)
            (item, numpy.ndarray) associated with each material cluster
            (key, str).
        inc : int, default=None
            Incremental counter serving as a reference basis for the clustering
            adaptivity frequency control.
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        is_trigger : bool
            True if clustering adaptivity is triggered, False otherwise.
        target_clusters : list[int]
            List containing the labels (int) of clusters to be adapted.
        """
        init_time = time.time()
        # Output execution data
        if verbose:
            info.displayinfo('5', 'Clustering adaptivity criterion:')
            info.displayinfo('5', 'Selecting target clusters...', 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize target clusters list
        target_clusters = []
        # Initialize target clusters data
        target_clusters_data = {}
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Check material phase adaptivity activation if clustering
            # adaptivity feature minimum significant value has already been
            # surpassed
            if self._adapt_feature_min_trigger[mat_phase]:
                if not self.is_adapt_phase_activated(mat_phase, inc):
                    continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get adaptivity feature
            adapt_control_feature = self._adaptivity_control_feature[mat_phase]
            # Build adaptivity feature data matrix
            adapt_data_matrix = self._get_adaptivity_data_matrix(
                mat_phase, adapt_control_feature, phase_clusters,
                clusters_def_gradient_mf, clusters_def_gradient_old_mf,
                clusters_state, clusters_state_old, clusters_sct_mf,
                clusters_sct_old_mf, clusters_residuals_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase clustering adaptivity criterion
            adapt_criterion = self._adapt_phase_criterions[mat_phase]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Check clustering adaptivity feature minimum significant value
            # surpassed status if hasn't been previously triggered
            if not self._adapt_feature_min_trigger[mat_phase]:
                # Get clustering adaptivity feature minimum significant value
                min_sign_value = adapt_criterion.get_min_adapt_feature_val()
                # Check clustering adaptivity feature minimum significant value
                if np.max(adapt_data_matrix[:, 1]) >= min_sign_value:
                    # Set clustering adaptivity reference initial increment
                    self._adapt_ref_init_inc[mat_phase] = inc
                    # Update clustering adaptivity feature minimum significant
                    # value surpassed status
                    self._adapt_feature_min_trigger[mat_phase] = True
                else:
                    continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phase clustering adaptivity target clusters
            if isinstance(adapt_criterion, AdaptiveClusterGrouping):
                phase_target_clusters, phase_target_clusters_data = \
                    adapt_criterion.get_target_clusters(adapt_data_matrix,
                                                        voxels_clusters)
            elif isinstance(adapt_criterion, SpatialDiscontinuities):
                phase_target_clusters, phase_target_clusters_data = \
                    adapt_criterion.get_target_clusters(adapt_data_matrix,
                                                        voxels_clusters)
            # Update list of target clusters and associated data
            target_clusters += phase_target_clusters
            target_clusters_data.update(phase_target_clusters_data)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set adaptivity trigger flag according to the list of target clusters
        if target_clusters:
            is_trigger = True
        else:
            is_trigger = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in selecting target clusters for
        # clustering adaptivity
        self.adaptive_evaluation_time += time.time() - init_time
        # Update total amount of time spent in clustering adaptivity procedures
        self.adaptive_time += time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            # Build output data
            output_list = []
            output_total = 0
            for mat_phase in self._adapt_material_phases:
                phase_target_clusters = list(set(target_clusters).intersection(
                    phase_clusters[mat_phase]))
                output_list += [mat_phase, len(phase_target_clusters)]
                output_total += len(phase_target_clusters)
            # Output adaptive phases target clusters summary table
            indent = 10*' '
            info.displayinfo(
                '5', 'Summary:' + '\n\n'
                + indent + 'Phase   Target Clusters' + '\n'
                + indent + 23*'-' + '\n'
                + ((indent + '{:^5s}{:>11d}\n')
                   * (len(self._adapt_material_phases))).format(*output_list)
                + indent + 23*'-' + '\n'
                + indent + '{:^5s}'.format('Total')
                + '{:>11d}'.format(output_total) + '\n',
                2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return is_trigger, target_clusters, target_clusters_data
    # -------------------------------------------------------------------------
    def _get_adaptivity_data_matrix(self, target_phase, adapt_control_feature,
                                    phase_clusters, clusters_def_gradient_mf,
                                    clusters_def_gradient_old_mf,
                                    clusters_state, clusters_state_old,
                                    clusters_sct_mf, clusters_sct_old_mf,
                                    clusters_residuals_mf):
        """Build adaptivity feature data matrix for target material phase.

        Parameters
        ----------
        target_phase : str
            Target adaptive material phase whose clusters adaptive feature data
            is collected or computed.
        adapt_control_feature : str
            Scalar adaptivity feature available directly or indirectly from
            clusters state variables.
        phase_clusters : dict
            Clusters labels (item, list[int]) associated with each material
            phase (key, str).
        clusters_def_gradient_mf : dict
            Deformation gradient (item, numpy.ndarray (1d)) associated with
            each material cluster (key, str), stored in matricial form.
        clusters_def_gradient_old_mf : dict
            Last converged deformation gradient (item, numpy.ndarray (1d))
            associated with each material cluster (key, str), stored in
            matricial form.
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated
            with each material cluster (key, str).
        clusters_state_old : dict
            Last increment converged material constitutive model state
            variables (item, dict) associated with each material cluster
            (key, str).
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form)
            (item, numpy.ndarray (2d)) associated with each material cluster
            (key, str).
        clusters_sct_old_mf : dict
            Last increment converged fourth-order strain concentration tensor
            (matricial form) (item, numpy.ndarray (2d)) associated with each
            material cluster (key, str).
        clusters_residuals_mf : dict
            Equilibrium residual second-order tensor (matricial form)
            (item, numpy.ndarray (1d)) associated with each material cluster
            (key, str).

        Returns
        -------
        adapt_data_matrix : numpy.ndarray (2d)
            Adaptivity feature data matrix (numpy.ndarray of shape
            (adapt_phase_n_clusters, 2)) that, for the i-th cluster of the
            adaptive material phase, contains the cluster label in
            adapt_data_matrix[i, 0] and the associated adaptive feature value
            in adapt_data_matrix[i, 1].
        """
        # Get target adaptive material phase number of clusters
        n_clusters = len(phase_clusters[target_phase])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize adaptivity feature data matrix
        adapt_data_matrix = np.zeros((n_clusters, 2))
        adapt_data_matrix[:, 0] = phase_clusters[target_phase]
        # Get target material phase state variables
        state_variables = \
            clusters_state[str(int(adapt_data_matrix[0, 0]))].keys()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is scalar state variable
        if is_state_variable and not is_component:
            # Check if adaptivity feature is scalar
            if not ioutil.checknumber(
                    clusters_state[str(int(adapt_data_matrix[0, 0]))][
                        adapt_control_feature]) and not is_norm:
                raise RuntimeError('The clustering adaptivity feature ('
                                   + adapt_control_feature + ') prescribed '
                                   + 'for material phase ' + target_phase
                                   + ' must be a scalar.')
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
                                clusters_state[str(cluster)][
                                adapt_control_feature] - clusters_state_old[
                                str(cluster)][adapt_control_feature]
                        else:
                            adapt_data_matrix[i, 1] = clusters_state[
                                str(cluster)][adapt_control_feature]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is component of strain/stress related
        # second order tensor (stored in matricial form) state variable
        elif is_state_variable and is_component:
            # Check if adaptivity feature is scalar
            if ioutil.checknumber(clusters_state[
                    str(int(adapt_data_matrix[0, 0]))][adapt_control_feature]):
                raise RuntimeError('The clustering adaptivity feature ('
                                   + adapt_control_feature + ') prescribed '
                                   + 'for material material phase '
                                   + target_phase + ' is a scalar.')
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Get second order tensor matricial form index
                if len(clusters_state[str(cluster)][
                        adapt_control_feature]) == self._comp_order_nsym:
                    index = self._comp_order_nsym.index(feature_comp)
                elif len(clusters_state[str(cluster)][
                        adapt_control_feature]) == self._comp_order_sym:
                    if feature_comp in self._comp_order_sym:
                        index = self._comp_order_sym.index(feature_comp)
                    elif feature_comp[::-1] in self._comp_order_sym:
                        index = self._comp_order_sym.index(feature_comp)
                    else:
                        raise RuntimeError('Invalid component specified for '
                                           'the clustering adaptivity '
                                           'feature '
                                           + adapt_control_feature + '.')
                else:
                    raise RuntimeError('Clustering adaptivity feature '
                                       'is not a second-order tensor stored '
                                       'in matricial form as expected.')
                # Collect adaptivity feature data
                if is_incremental:
                    adapt_data_matrix[i, 1] = \
                        clusters_state[str(cluster)][
                            adapt_control_feature][index] \
                        - clusters_state_old[str(cluster)][
                            adapt_control_feature][index]
                else:
                    adapt_data_matrix[i, 1] = clusters_state[
                        str(cluster)][adapt_control_feature][index]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is Von Mises equivalent stress
        elif adapt_control_feature == 'vm_stress':
            # Instantiate material state computations
            csbvar_computer = MaterialQuantitiesComputer()
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get Cauchy stress tensor
                if self._strain_formulation == 'infinitesimal':
                    # Get cluster stress tensor (matricial form)
                    stress_mf = clusters_state[str(cluster)]['stress_mf']
                else:
                    # Get cluster deformation gradient (matricial form)
                    def_gradient_mf = clusters_def_gradient_mf[str(cluster)]
                    # Build deformation gradient
                    def_gradient = mop.get_tensor_from_mf(
                        def_gradient_mf, self._n_dim, self._comp_order_nsym)
                    # Get first Piola-Kirchhoff stress tensor (matricial form)
                    first_piola_stress_mf = \
                        clusters_state[str(cluster)]['stress_mf']
                    # Build first Piola-Kirchhoff stress tensor
                    first_piola_stress = mop.get_tensor_from_mf(
                        first_piola_stress_mf, self._n_dim,
                        self._comp_order_sym)
                    # Compute Cauchy stress tensor
                    cauchy_stress = cauchy_from_first_piola(def_gradient,
                                                            first_piola_stress)
                    # Get Cauchy stress tensor (matricial form)
                    stress_mf = mop.get_tensor_mf(cauchy_stress, self._n_dim,
                                                  self._comp_order_sym)
                # Build 3D stress tensor (matricial form)
                if self._problem_type == 1:
                    # Get Cauchy stress tensor out-of-plain component
                    if self._strain_formulation == 'infinitesimal':
                        stress_33 = clusters_state[str(cluster)]['stress_33']
                    else:
                        # Get Cauchy stress tensor out-of-plain component from
                        # first Piola-Kirchhoff counterpart
                        stress_33 = (1.0/np.linalg.det(def_gradient))\
                            * clusters_state[str(cluster)]['stress_33']
                    # Build 3D stress tensor (matricial form)
                    stress_mf = mop.getstate3Dmffrom2Dmf(self._problem_type,
                                                         stress_mf, stress_33)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute von Mises equivalent stress
                vm_stress = csbvar_computer.get_vm_stress(stress_mf)
                # Assemble adaptivity feature data matrix
                if is_incremental:
                    # Get cluster previously converged stress tensor (matricial
                    # form)
                    if self._strain_formulation == 'infinitesimal':
                        # Get cluster stress tensor (matricial form)
                        stress_mf = \
                            clusters_state_old[str(cluster)]['stress_mf']
                    else:
                        # Get cluster deformation gradient (matricial form)
                        def_gradient_mf = \
                            clusters_def_gradient_old_mf[str(cluster)]
                        # Build deformation gradient
                        def_gradient = mop.get_tensor_from_mf(
                            def_gradient_mf, self._n_dim,
                            self._comp_order_nsym)
                        # Get first Piola-Kirchhoff stress tensor (matricial
                        # form)
                        first_piola_stress_mf = \
                            clusters_state_old[str(cluster)]['stress_mf']
                        # Build first Piola-Kirchhoff stress tensor
                        first_piola_stress = mop.get_tensor_from_mf(
                            first_piola_stress_mf, self._n_dim,
                            self._comp_order_sym)
                        # Compute Cauchy stress tensor
                        cauchy_stress = cauchy_from_first_piola(
                            def_gradient, first_piola_stress)
                        # Get Cauchy stress tensor (matricial form)
                        stress_mf = mop.get_tensor_mf(cauchy_stress,
                                                      self._n_dim,
                                                      self._comp_order_sym)
                    # Build 3D stress tensor (matricial form)
                    if self._problem_type == 1:
                        # Get Cauchy stress tensor out-of-plain component
                        if self._strain_formulation == 'infinitesimal':
                            stress_33 = \
                                clusters_state_old[str(cluster)]['stress_33']
                        else:
                            # Get Cauchy stress tensor out-of-plain component
                            # from first Piola-Kirchhoff counterpart
                            stress_33 = (1.0/np.linalg.det(def_gradient))\
                                * clusters_state_old[str(cluster)]['stress_33']
                        # Build 3D stress tensor (matricial form)
                        stress_mf = mop.getstate3Dmffrom2Dmf(
                            self._problem_type, stress_mf, stress_33)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute previously converged von Mises equivalent stress
                    vm_stress_old = csbvar_computer.get_vm_stress(stress_mf)
                    # Assemble incremental adaptivity feature data matrix
                    adapt_data_matrix[i, 1] = vm_stress - vm_stress_old
                else:
                    adapt_data_matrix[i, 1] = vm_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is Von Mises equivalent strain
        elif adapt_control_feature == 'vm_strain':
            # Instantiate material state computations
            csbvar_computer = MaterialQuantitiesComputer()
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Get infinitesimal strain tensor (infinitesimal strains) or
                # spatial logarithmic strain tensor (finite strains)
                if self._strain_formulation == 'infinitesimal':
                    # Get cluster strain tensor (matricial form)
                    strain_mf = clusters_state[str(cluster)]['strain_mf']
                else:
                    # Get cluster deformation gradient (matricial form)
                    def_gradient_mf = clusters_def_gradient_mf[str(cluster)]
                    # Build deformation gradient
                    def_gradient = mop.get_tensor_from_mf(
                        def_gradient_mf, self._n_dim, self._comp_order_nsym)
                    # Compute spatial logarithmic strain tensor
                    log_strain = compute_spatial_log_strain(def_gradient)
                    # Get spatial logarithmic strain tensor (matricial form)
                    strain_mf = mop.get_tensor_mf(log_strain, self._n_dim,
                                                  self._comp_order_sym)
                # Build 3D strain tensor (matricial form)
                if self._problem_type == 1:
                    # Get out-of-plain strain component
                    strain_33 = 0.0
                    # Build 3D strain tensor (matricial form)
                    strain_mf = mop.getstate3Dmffrom2Dmf(self._problem_type,
                                                         strain_mf, strain_33)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute von Mises equivalent strain
                vm_strain = csbvar_computer.get_vm_strain(strain_mf)
                # Assemble adaptivity feature data matrix
                if is_incremental:
                    # Get previously converged infinitesimal strain tensor
                    # (infinitesimal strains) or spatial logarithmic strain
                    # tensor (finite strains)
                    if self._strain_formulation == 'infinitesimal':
                        # Get cluster strain tensor (matricial form)
                        strain_mf = \
                            clusters_state_old[str(cluster)]['strain_mf']
                    else:
                        # Get cluster deformation gradient (matricial form)
                        def_gradient_mf = \
                            clusters_def_gradient_old_mf[str(cluster)]
                        # Build deformation gradient
                        def_gradient = mop.get_tensor_from_mf(
                            def_gradient_mf, self._n_dim,
                            self._comp_order_nsym)
                        # Compute spatial logarithmic strain tensor
                        log_strain = compute_spatial_log_strain(def_gradient)
                        # Get spatial logarithmic strain tensor (matricial
                        # form)
                        strain_mf = mop.get_tensor_mf(log_strain, self._n_dim,
                                                      self._comp_order_sym)
                    # Build 3D strain tensor (matricial form)
                    if self._problem_type == 1:
                        # Get out-of-plain strain component
                        strain_33 = 0.0
                        # Build 3D strain tensor (matricial form)
                        strain_mf = mop.getstate3Dmffrom2Dmf(
                            self._problem_type, strain_mf, strain_33)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute previously converged von Mises equivalent strain
                    vm_strain_old = csbvar_computer.get_vm_strain(strain_mf)
                    # Assemble incremental adaptivity feature data matrix
                    adapt_data_matrix[i, 1] = vm_strain - vm_strain_old
                else:
                    adapt_data_matrix[i, 1] = vm_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is norm of strain concentration tensor
        elif adapt_control_feature == 'strain_concentration_tensor' \
                and is_norm:
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                if is_incremental:
                    adapt_data_matrix[i, 1] = \
                        np.linalg.norm(clusters_sct_mf[str(cluster)]) - \
                        np.linalg.norm(clusters_sct_old_mf[str(cluster)])
                else:
                    adapt_data_matrix[i, 1] = \
                        np.linalg.norm(clusters_sct_mf[str(cluster)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Clustering adaptivity feature is norm of Lippmann-Schwinger
        # equilibrium residual
        elif adapt_control_feature == 'equilibrium_residual' and is_norm:
            # Build adaptivity feature data matrix
            for i in range(n_clusters):
                # Get cluster label
                cluster = int(adapt_data_matrix[i, 0])
                # Collect adaptivity feature data
                adapt_data_matrix[i, 1] = \
                    np.linalg.norm(clusters_residuals_mf[str(cluster)])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown or unavailable clustering adaptivity '
                               'feature.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return adapt_data_matrix
    # -------------------------------------------------------------------------
    def adaptive_refinement(self, crve, material_state, target_clusters,
                            target_clusters_data, inc,
                            improved_init_guess=None, verbose=False):
        """Perform CRVE adaptive clustering refinement.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        material_state : MaterialState
            CRVE material constitutive state at rewind state.
        target_clusters : list[int]
            List containing the labels (int) of clusters to be refined.
        target_clusters_data : dict
            For each target cluster (key, str), store dictionary (item, dict)
            containing cluster associated parameters required for the adaptive
            procedures.
        inc : int
            Macroscale loading increment.
        improved_init_guess : list, default=None
            List that allows an improved initial iterative guess for the
            clusters incremental strain global vector (matricial form) after
            the clustering adaptivity is carried out. Index 0 contains a flag
            which is True if an improved initial iterative guess is to be
            computed, False otherwise. Index 1 contains the improved
            incremental strain global vector (matricial form) if computation
            flag is True, otherwise is None.
        verbose : bool, default=False
            Enable verbose output.
        """
        # Set initial time
        init_time = time.time()
        # Output execution data
        if verbose:
            info.displayinfo('5', 'Clustering adaptivity refinement:')
            info.displayinfo('5', 'A - Performing clustering adaptivity...', 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform CRVE adaptive clustering refinement
        if not target_clusters:
            return
        else:
            # Store previous clustering cluster labels and total number of
            # clusters
            phase_clusters_old = copy.deepcopy(crve.get_phase_clusters())
            n_total_clusters_old = crve.get_n_total_clusters()
            # Perform CRVE adaptive clustering refinement
            adaptive_clustering_map = crve.perform_crve_adaptivity(
                target_clusters, target_clusters_data)
            # Increment current increment adaptive step
            if str(inc) in self._inc_adaptive_steps.keys():
                self._inc_adaptive_steps[str(inc)] += 1
            else:
                self._inc_adaptive_steps[str(inc)] = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            a_time = time.time() - init_time
            info.displayinfo('5',
                             'B - Computing cluster interaction tensors...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute CRVE cluster interaction tensors
        crve.compute_cit(mode='adaptive',
                         adaptive_clustering_map=adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            b_time = time.time() - ref_time
            info.displayinfo('5',
                             'C - Updating cluster-related quantities...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update cluster-related variables according to clustering adaptivity
        # step
        material_state.clustering_adaptivity_update(crve.get_phase_clusters(),
                                                    crve.get_clusters_vf(),
                                                    adaptive_clustering_map)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Check material phase adaptivity lock status. If material phase
            # adaptivity is deactivated, then set the associated clustering
            # adaptivity frequency to zero
            if crve.get_cluster_phases()[mat_phase].adaptivity_lock:
                self._clust_adapt_freq[mat_phase] = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get improved initial iterative guess flag
        if improved_init_guess is None:
            is_improved_init_guess = False
        else:
            is_improved_init_guess = improved_init_guess[0]
            if not is_improved_init_guess:
                improved_init_guess[1] = None
        # Compute improved initial iterative guess for the global vector of
        # clusters strain tensors (matricial form)
        if is_improved_init_guess:
            # Get CRVE material phases and total number of clusters
            material_phases = crve.get_material_phases()
            n_total_clusters = crve.get_n_total_clusters()
            # Set cluster strain component order
            if self._strain_formulation == 'infinitesimal':
                comp_order = self._comp_order_sym
            else:
                comp_order = self._comp_order_nsym
            # Get previous clustering global vector of clusters strain tensors
            if len(improved_init_guess[1]) \
                    != n_total_clusters_old*len(comp_order):
                raise RuntimeError('Unexpected size of previous clustering '
                                   'global vector of clusters strain tensors.')
            else:
                global_strain_mf_old = copy.deepcopy(improved_init_guess[1])
            # Initialize new clustering global vector of clusters strain
            # tensors
            global_strain_mf_new = np.zeros((n_total_clusters*len(comp_order)))
            # Initialize material phase initial index in global vector
            mat_phase_init_idx_old = 0
            mat_phase_init_idx_new = 0
            # Loop over material phases
            for mat_phase in material_phases:
                # Loop over previous clustering cluster labels
                for cluster in phase_clusters_old[mat_phase]:
                    # Get previous clustering cluster initial index
                    init_idx_old = mat_phase_init_idx_old \
                        + phase_clusters_old[mat_phase].index(cluster) \
                        * len(comp_order)
                    # Build new clustering global vector of clusters strain
                    # tensors. If cluster remained unchanged after the
                    # clustering adaptive step, then simply transfer the
                    # associated values to the proper position in the new
                    # global vector. If cluster has been refined, then copy the
                    # associated values to its child clusters positions in the
                    # new global vector.
                    if cluster in crve.get_phase_clusters()[mat_phase]:
                        # Get new clustering cluster initial index
                        init_idx_new = mat_phase_init_idx_new \
                            + crve.get_phase_clusters()[mat_phase].index(
                                cluster)*len(comp_order)
                        # Transfer cluster strain
                        global_strain_mf_new[
                            init_idx_new:init_idx_new
                            + len(comp_order)] = global_strain_mf_old[
                                init_idx_old:init_idx_old+len(comp_order)]
                    else:
                        # Get list of target's child clusters
                        child_clusters = adaptive_clustering_map[mat_phase][
                            str(cluster)]
                        # Loop over child clusters
                        for child_cluster in child_clusters:
                            # Get new clustering child cluster initial index
                            init_idx_new = mat_phase_init_idx_new \
                                + crve.get_phase_clusters()[mat_phase].index(
                                    child_cluster)*len(comp_order)
                            # Copy parent cluster strain
                            global_strain_mf_new[init_idx_new:init_idx_new
                                                 + len(comp_order)] = \
                                global_strain_mf_old[init_idx_old:
                                                     init_idx_old
                                                     + len(comp_order)]
                # Update material phase initial index in global vector
                mat_phase_init_idx_old += \
                    len(phase_clusters_old[mat_phase])*len(comp_order)
                mat_phase_init_idx_new += \
                    len(crve.get_phase_clusters()[mat_phase])*len(comp_order)
            # Store improved initial iterative guess for the global vector of
            # clusters strain tensors (matricial form)
            improved_init_guess[1] = global_strain_mf_new
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            c_time = time.time() - ref_time
            info.displayinfo('5',
                             'D - Perfoming clustering adaptivity criterion '
                             'post-processing computations...', 2)
            ref_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get adaptive material phase clustering adaptivity criterion
            adapt_criterion = self._adapt_phase_criterions[mat_phase]
            # Perform clustering adaptivity criterion post-processing
            # computations
            if isinstance(adapt_criterion, AdaptiveClusterGrouping):
                # Update adaptive cluster groups
                adapt_criterion.update_group_clusters(
                    adaptive_clustering_map[mat_phase])
            elif isinstance(adapt_criterion, SpatialDiscontinuities):
                # Update clusters adaptive level
                adapt_criterion.update_clusters_adapt_level(
                    adaptive_clustering_map[mat_phase])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            d_time = time.time() - ref_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in the ACRVE adaptive procedures
        dtime = time.time() - init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total amount of time spent in clustering adaptivity procedures
        self.adaptive_time += time.time() - init_time
        #
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Output execution data
        if verbose:
            # Build adaptive material phase's target clusters list
            output_list = []
            output_total = [0, 0]
            # Loop over adaptive material phases
            for mat_phase in self._adapt_material_phases:
                # Loop over material phase target clusters
                n_new_clusters = 0
                for target_cluster in \
                        adaptive_clustering_map[mat_phase].keys():
                    n_new_clusters += \
                        len(adaptive_clustering_map[mat_phase][target_cluster])
                n_total_clusters = len(crve.get_phase_clusters()[mat_phase])
                output_list += [mat_phase, n_new_clusters, n_total_clusters]
                output_total[0] += n_new_clusters
                output_total[1] += n_total_clusters
            # Output adaptive phases new clusters summary table
            indent = 10*' '
            info.displayinfo(
                '5',
                'Summary:' + '\n\n'
                + indent + 'Phase   New Clusters   Total Clusters' + '\n'
                + indent + 37*'-' + '\n'
                + ((indent + '{:^5s}{:>11d}{:>15d}\n')
                   * (len(self._adapt_material_phases))).format(*output_list)
                + indent + 37*'-' + '\n'
                + indent + '{:^5s}'.format('Total')
                + '{:>11d}{:>15d}'.format(*output_total),
                2)
            # Output adaptive phases execution time table
            indent = 10*' '
            info.displayinfo(
                '5', 'Execution times (s):' + '\n\n'
                + indent + '           Time(s)        %' + '\n'
                + indent + 28*'-' + '\n'
                + indent + '{:^5s}'.format('A') + '{:^18.4e}'.format(a_time)
                + '{:>5.2f}'.format((a_time/dtime)*100) + '\n'
                + indent + '{:^5s}'.format('B') + '{:^18.4e}'.format(b_time)
                + '{:>5.2f}'.format((b_time/dtime)*100) + '\n'
                + indent + '{:^5s}'.format('C') + '{:^18.4e}'.format(c_time)
                + '{:>5.2f}'.format((c_time/dtime)*100) + '\n'
                + indent + '{:^5s}'.format('D') + '{:^18.4e}'.format(c_time)
                + '{:>5.2f}'.format((d_time/dtime)*100) + '\n'
                + indent + 28*'-' + '\n'
                + indent + '{:^5s}'.format('Total')
                + '{:>14.4e}'.format(dtime),
                2)
    # -------------------------------------------------------------------------
    def check_inc_adaptive_steps(self, inc):
        """Check number of adaptive steps performed in loading increment.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.

        Returns
        -------
        bool
            True if maximum number of clustering adaptive steps has not been
            reached, False otherwise.
        """
        if str(inc) in self._inc_adaptive_steps.keys():
            if self._inc_adaptive_steps[str(inc)] \
                    >= self.max_inc_adaptive_steps:
                return False
            else:
                return True
        else:
            self._inc_adaptive_steps[str(inc)] = 0
            return True
    # -------------------------------------------------------------------------
    def clear_inc_adaptive_steps(self, inc_threshold):
        """Reset number of adaptive steps performed after threshold increment.

        Parameters
        ----------
        inc_threshold : int
            Macroscale loading increment.
        """
        # Get clustering adaptive steps associated increments
        adaptive_steps_incs = [int(x) for x in self._inc_adaptive_steps.keys()]
        # Loop over clustering adaptive steps associated increments
        for inc in adaptive_steps_incs:
            # Reset number of clustering adaptive steps if increment is greater
            # than provided threshold
            if inc > inc_threshold:
                self._inc_adaptive_steps[str(inc)] = 0
    # -------------------------------------------------------------------------
    def is_adapt_phase_activated(self, mat_phase, inc):
        """Check if material phase adaptivity procedures are activated.

        Parameters
        ----------
        mat_phase : str
            Material phase label.
        inc : int
            Incremental counter serving as a reference basis for the clustering
            adaptivity frequency control.

        Returns
        -------
        is_activated : bool
            True if material phase adaptivity procedures are activated, False
            otherwise.
        """
        # Get clustering adaptivity reference initial increment
        ref_inc = self._adapt_ref_init_inc[mat_phase]
        # Get clustering adaptivity frequency
        adapt_freq = self._clust_adapt_freq[mat_phase]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize adaptivity activation flag
        is_activated = False
        # Evaluate activation conditions
        if adapt_freq != 0:
            # Clustering adaptivity incremental frequency
            if inc > ref_inc and (inc - ref_inc) % adapt_freq == 0:
                is_activated = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return adaptivity activation flag
        return is_activated
    # -------------------------------------------------------------------------
    def reset_adapt_activation_parameters(self):
        """Reset parameters associated with clustering adaptivity activation.
        """
        self._adapt_ref_init_inc = \
            {mat_phase: 0 for mat_phase in self._adapt_material_phases}
        self._adapt_feature_min_trigger = \
            {mat_phase: False for mat_phase in self._adapt_material_phases}
    # -------------------------------------------------------------------------
    def get_adapt_vtk_array(self, voxels_clusters):
        """Get regular grid array with the adaptive level of each cluster.

        A cluster adaptive level of 0 is associated with the base clustering
        and is increased by one whenever the cluster is refined.

        ----

        Parameters
        ----------
        voxels_clusters : numpy.ndarray (2d or 3d)
            Regular grid of voxels (spatial discretization of the RVE), where
            each entry contains the cluster label (int) assigned to the
            corresponding pixel/voxel.
        """
        # Initialize flattened regular grid array
        rg_array = np.zeros(voxels_clusters.shape).flatten('F')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over adaptive material phases
        for mat_phase in self._adapt_material_phases:
            # Get adaptive material phase clustering adaptivity criterion
            adapt_criterion = self._adapt_phase_criterions[mat_phase]
            # Assemble flattened regular grid array
            if isinstance(adapt_criterion, AdaptiveClusterGrouping):
                # Get adaptive cluster groups and associated adaptive level
                adapt_groups = adapt_criterion.get_adapt_groups()
                groups_adapt_level = adapt_criterion.get_groups_adapt_level()
                # Get adaptive material phase cluster groups
                adapt_groups_ids = list(adapt_groups.keys())
                # Loop over adaptive cluster groups
                for group_id in adapt_groups_ids:
                    # Get adaptive cluster group adaptive level
                    adapt_level = groups_adapt_level[group_id]
                    # Get adaptive cluster group clusters
                    adapt_group = adapt_groups[group_id]
                    # Get flat indexes associated with the adaptive cluster
                    # group clusters
                    flat_idxs = np.in1d(voxels_clusters.flatten('F'),
                                        adapt_group)
                    # Store adaptive cluster group adaptive level
                    rg_array[flat_idxs] = adapt_level
            elif isinstance(adapt_criterion, SpatialDiscontinuities):
                # Get cluster adaptive level
                clusters_adapt_level = \
                    adapt_criterion.get_clusters_adapt_level()
                # Loop over clusters
                for cluster in clusters_adapt_level.keys():
                    # Get cluster flat indexes
                    flat_idxs = np.in1d(voxels_clusters.flatten('F'),
                                        int(cluster))
                    # Store cluster adaptive level
                    rg_array[flat_idxs] = clusters_adapt_level[str(cluster)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build regular grid array
        rg_array = rg_array.reshape(voxels_clusters.shape, order='F')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return rg_array
#
#                                             Clustering adaptivity output file
# =============================================================================
class ClusteringAdaptivityOutput:
    """Clustering adaptivity output.

    Attributes
    ----------
    _header : list[str]
        List containing the header of each column (str).
    _col_width : int
        Output file column width.

    Methods
    -------
    init_adapt_file(self, crve)
        Open clustering adaptivity output file and write file header.
    write_adapt_file(self, inc, adaptivity_manager, crve, mode='increment')
        Write clustering adaptivity output file.
    rewind_file(self, rewind_inc)
        Rewind clustering adaptivity output file.
    """
    # -------------------------------------------------------------------------
    def __init__(self, adapt_file_path, adapt_material_phases):
        """Constructor.

        Parameters
        ----------
        adapt_file_path : str
            Path of clustering adaptivity output file.
        adapt_material_phases : list[str]
            RVE adaptive material phases labels (str).
        """
        self._adapt_file_path = adapt_file_path
        self._adapt_material_phases = adapt_material_phases
        # Set clustering adaptivity output file header
        self._header = ['Increment', 'total_adapt_time', 'eval_adapt_time',
                        'clust_adapt_time', 'cit_adapt_time']
        for mat_phase in self._adapt_material_phases:
            self._header += ['n_clusters_' + mat_phase,
                             'adapt_step_' + mat_phase,
                             'adapt_time_' + mat_phase]
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # -------------------------------------------------------------------------
    def init_adapt_file(self, crve):
        """Open clustering adaptivity output file and write file header.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        """
        # Build adaptive material phases output list
        adaptivity_output = crve.get_adaptivity_output()
        output_list = []
        for mat_phase in self._adapt_material_phases:
            output_list += adaptivity_output[mat_phase]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open clustering adaptivity output file (write mode)
        adapt_file = open(self._adapt_file_path, 'w')
        # Set clustering adaptivity output file header format structure
        write_list = ['{:>9s}'.format(self._header[0])
                      + ''.join([('{:>' + str(self._col_width)
                                  + 's}').format(x)
                                for x in self._header[1:]]), ]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set clustering adaptivity output file increment format structure
        write_list += [
            '\n' + '{:>9d}'.format(0)
            + ('{:>' + str(self._col_width) + '.8e}').format(0)
            + ('{:>' + str(self._col_width) + '.8e}').format(0)
            + ('{:>' + str(self._col_width) + '.8e}').format(0)
            + ('{:>' + str(self._col_width) + '.8e}').format(0)
            + ''.join([''.join([('{:>' + str(self._col_width) + 'd}').format(x)
                                for x in output_list[3*i:3*i+2]]
                               + [('{:>' + str(self._col_width)
                                   + '.8e}').format(output_list[3*i+2])])
                       for i in range(len(self._adapt_material_phases))])]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write clustering adaptivity output file
        adapt_file.writelines(write_list)
        # Close clustering adaptivity output file
        adapt_file.close()
    # -------------------------------------------------------------------------
    def write_adapt_file(self, inc, adaptivity_manager, crve,
                         mode='increment'):
        """Write clustering adaptivity output file.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.
        adaptivity_manager : AdaptivityManager
            CRVE clustering adaptivity manager.
        crve : CRVE
            Cluster-Reduced Representative Volume Element.
        mode : {'init', 'increment'}, default='increment'
            Clustering adaptivity output mode. Mode `init` writes the file
            header and the increment 0 (base clustering), while `increment`
            appends the clustering adaptivity data associated with macroscale
            loading increment.
        """
        write_list = []
        if mode == 'init':
            # Open clustering adaptivity output file (write mode)
            adapt_file = open(self._adapt_file_path, 'w')
            # Set clustering adaptivity output file header format structure
            write_list += ['{:>9s}'.format(self._header[0])
                           + ''.join([('{:>' + str(self._col_width)
                                       + 's}').format(x)
                                      for x in self._header[1:]]), ]
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
        write_list += [
            '\n' + '{:>9d}'.format(inc)
            + ('{:>' + str(self._col_width) + '.8e}').format(
                adaptivity_manager.adaptive_time)
            + ('{:>' + str(self._col_width) + '.8e}').format(
                adaptivity_manager.adaptive_evaluation_time)
            + ('{:>' + str(self._col_width) + '.8e}').format(
                crve.get_adaptive_clustering_time())
            + ('{:>' + str(self._col_width) + '.8e}').format(
                crve.get_adaptive_cit_time())
            + ''.join([''.join([('{:>' + str(self._col_width) + 'd}').format(x)
                                for x in output_list[3*i:3*i+2]]
                               + [('{:>' + str(self._col_width)
                                   + '.8e}').format(output_list[3*i+2])])
                       for i in range(len(self._adapt_material_phases))])]
        # Write clustering adaptivity output file
        adapt_file.writelines(write_list)
        # Close clustering adaptivity output file
        adapt_file.close()
    # -------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        """Rewind clustering adaptivity output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated with the rewind state.
        """
        # Open clustering adaptivity output file and read lines (read)
        file_lines = open(self._adapt_file_path, 'r').readlines()
        # Set output file last line
        last_line = 1 + rewind_inc
        file_lines[last_line] = file_lines[last_line][:-1]
        # Open clustering adaptivity output file (write mode)
        open(self._adapt_file_path, 'w').writelines(
            file_lines[: last_line + 1])
