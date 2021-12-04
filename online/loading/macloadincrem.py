#
# Loading Incrementation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the enforcement of the loading constraints and the overall
# loading incrementation flow.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Jul 2020 | Implemented LoadingPath and LoadingSubpath classes.
# Bernardo P. Ferreira | Nov 2020 | Updated documentation.
# Bernardo P. Ferreira | Jul 2021 | Implemented RewindManager and IncrementRewinder classes.
# Bernardo P. Ferreira | Dec 2021 | LoadingSubpath loading incremental nature.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Shallow and deep copy operations
import copy
# Date and time
import time
# Tree data structure
from anytree.walker import Walker
from anytree.exporter import DotExporter
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                           Loading path and subpath classes
# ==========================================================================================
class LoadingPath:
    '''Class that controls the loading incrementation flow.

    This class contains a collection of loading subpaths, the current loading state and a
    set of methods to control the loading incrementation flow.

    Attributes
    ----------
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _n_load_subpaths : int
        Number of loading subpaths.
    _load_subpaths : list
        List of LoadingSubpath.
    _conv_hom_state : dict
        Converged homogenized state (item, ndarray of shape (n_comps,)) for key in
        {'strain', 'stress'}.
    _is_last_inc : bool
        Loading last increment flag.
    _n_cinc_cuts : int
        Consecutive loading increment cuts counter.
    _increm_state : dict
        Increment state: key 'inc' contains the current increment number, key 'subpath_id'
        contains the current loading subpath index.
    '''
    def __init__(self, strain_formulation, problem_type, mac_load, mac_load_presctype,
                 mac_load_increm, max_subinc_level=5, max_cinc_cuts=5):
        '''Loading path constructor.

        Parameters
        ----------
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        mac_load : dict
            For each loading nature type (key, {'strain', 'stress'}), stores the loading
            constraints for each loading subpath in a ndarray, where the i-th row is
            associated with the i-th strain/stress component and the j-th column is
            associated with the j-th loading subpath.
        mac_load_presctype : ndarray of shape (n_comps, n_load_subpaths)
            Loading nature type (str, {'strain', 'stress'}) associated with each loading
            constraint, where the i-th row is associated with the i-th strain/stress
            component and the j-th column is associated with the j-th loading subpath.
        mac_load_increm : dict
            For each loading subpath id (key, str), stores a ndarray of shape
            (n_load_increments, 2) where each row is associated with a prescribed loading
            increment, and the columns 0 and 1 contain the corresponding incremental load
            factor and incremental time, respectively.
        max_subinc_level : int, default=5
            Maximum level of loading subincrementation.
        max_cinc_cuts : int, default=5
            Maximum number of consecutive load increment cuts.
        '''
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._mac_load = mac_load
        self._mac_load_presctype = mac_load_presctype
        self._mac_load_increm = mac_load_increm
        self._max_subinc_level = max_subinc_level
        self._max_cinc_cuts = max_cinc_cuts
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Remove symmetric components under an infinitesimal strain formulation
        if strain_formulation == 'infinitesimal':
            self._remove_sym(comp_order_sym, comp_order_nsym)
        # Set total number of loading subpaths
        self._n_load_subpaths = len(mac_load_increm.keys())
        # Initialize list of loading subpaths
        self._load_subpaths = []
        # Initialize increment state
        self._increm_state = {'inc': 0, 'subpath_id': -1}
        # Initialize converged homogenized state
        self._conv_hom_state = {key: None for key in ['strain', 'stress']}
        # Initialize loading last increment flag
        self._is_last_inc = False
        # Initialize consecutive increment cuts counter
        self._n_cinc_cuts = 0
    # --------------------------------------------------------------------------------------
    def new_load_increment(self):
        '''Setup new loading increment and get associated data.

        Returns
        -------
        inc_mac_load_mf : dict
            For each loading nature type (key, {'strain', 'stress'}), stores the incremental
            loading constraint matricial form in a ndarray of shape (n_comps,).
        n_presc_strain : int
            Number of prescribed loading strain components.
        presc_strain_idxs : list
            Prescribed loading strain components indexes.
        n_presc_stress : int
            Number of prescribed loading stress components.
        presc_stress_idxs : list
            Prescribed loading stress components indexes.
        is_last_inc : bool
            Loading last increment flag.
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reset consecutive loading increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add a new loading subpath to the loading path if either first load increment or
        # current loading subpath is completed
        if self._increm_state['inc'] == 0 or self._get_load_subpath()._is_last_subpath_inc:
            # Add a new loading subpath
            self._new_subpath()
        # Get current loading subpath
        load_subpath = self._get_load_subpath()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update loading increment
        self._update_inc()
        # Check if last loading increment
        if load_subpath._id == self._n_load_subpaths - 1 and \
                load_subpath._is_last_subpath_inc:
            self._is_last_inc = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental loading
        inc_mac_load = self._get_increm_load()
        inc_mac_load_mf = {}
        for ltype in inc_mac_load.keys():
            inc_mac_load_mf[ltype] = type(self)._get_load_mf(self._n_dim, comp_order,
                                                             inc_mac_load[ltype])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [inc_mac_load_mf, load_subpath._n_presc_strain,
                load_subpath._presc_strain_idxs, load_subpath._n_presc_stress,
                load_subpath._presc_stress_idxs, self._is_last_inc]
    # --------------------------------------------------------------------------------------
    def increment_cut(self, n_dim, comp_order):
        '''Perform loading increment cut, setup new increment and get associated data.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order : list
            Strain/Stress components (str) order.

        Returns
        -------
        inc_mac_load_mf : dict
            For each loading nature type (key, {'strain', 'stress'}), stores the incremental
            macroscale loading constraint matricial form in a ndarray of shape (n_comps,).
        n_presc_strain : int
            Number of prescribed macroscale loading strain components.
        presc_strain_idxs : list
            Prescribed macroscale loading strain components indexes.
        n_presc_stress : int
            Number of prescribed macroscale loading stress components.
        presc_stress_idxs : list
            Prescribed macroscale loading stress components indexes.
        is_last_inc : bool
            Loading last increment flag.
        '''
        # Get current loading subpath
        load_subpath = self._get_load_subpath()
        # Perform loading increment
        load_subpath.increment_cut()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set last loading increment flag
        self._is_last_inc = False
        # Increment (+1) consecutive loading increment cuts counter
        self._n_cinc_cuts += 1
        # Check if maximum number of consecutive loading increment cuts is surpassed
        if self._n_cinc_cuts > self._max_cinc_cuts:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00096', location.filename, location.lineno + 1,
                                self._max_cinc_cuts)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental loading
        inc_mac_load = self._get_increm_load()
        inc_mac_load_mf = {}
        for ltype in inc_mac_load.keys():
            inc_mac_load_mf[ltype] = type(self)._get_load_mf(n_dim, comp_order,
                                                             inc_mac_load[ltype])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return [inc_mac_load_mf, load_subpath._n_presc_strain,
                load_subpath._presc_strain_idxs, load_subpath._n_presc_stress,
                load_subpath._presc_stress_idxs, self._is_last_inc]
    # --------------------------------------------------------------------------------------
    def update_hom_state(self, hom_strain_mf, hom_stress_mf):
        '''Update converged homogenized state.

        Parameters
        ----------
        hom_strain_mf : 1darray
            Homogenized strain tensor stored in matricial form.
        hom_stress_mf : 1darray
            Homogenized stress tensor stored in matricial form.
        '''
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build homogenized strain tensor
        hom_strain = mop.get_tensor_from_mf(hom_strain_mf, self._n_dim, comp_order)
        # Build homogenized stress tensor
        hom_stress = mop.get_tensor_from_mf(hom_stress_mf, self._n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize converged homogenized strain and stress tensors vector form
        self._conv_hom_state['strain'] = np.zeros(len(comp_order))
        self._conv_hom_state['stress'] = np.zeros(len(comp_order))
        # Loop over strain/stress components
        for k in range(len(comp_order)):
            # Get strain/stress component
            comp = comp_order[k]
            # Get component indexes
            i = int(comp[0]) - 1
            j = int(comp[1]) - 1
            # Build converged homogenized strain and stress tensors vector form
            self._conv_hom_state['strain'][k] = hom_strain[i, j]
            self._conv_hom_state['stress'][k] = hom_stress[i, j]
    # --------------------------------------------------------------------------------------
    def get_subpath_state(self):
        '''Get current loading subpath state.

        Returns
        -------
        id : int
            Loading subpath id.
        inc : int
            Current loading subpath increment counter.
        total_lfact : float
            Current loading subpath current total load factor.
        inc_lfact : float
            Current loading subpath current incremental load factor.
        total_time : float
            Current loading subpath current total time.
        inc_time : float
            Current loading subpath current incremental time.
        sub_inc_level : int
            Current loading subpath current subincrementation level.
        '''
        # Get current loading subpath
        load_subpath = self._get_load_subpath()
        # Return loading subpath state
        return load_subpath.get_state()
    # --------------------------------------------------------------------------------------
    def get_increm_state(self):
        '''Get incremental state.

        Returns
        -------
        increm_state : dict
            Increment state: key 'inc' contains the current increment number, key
            'subpath_id' contains the current loading subpath index.
        '''
        return copy.deepcopy(self._increm_state)
    # --------------------------------------------------------------------------------------
    def _new_subpath(self):
        '''Add a new loading subpath to the loading path.'''
        # Increment (+1) loading subpath id
        self._increm_state['subpath_id'] += 1
        subpath_id = self._increm_state['subpath_id']
        # Get load and prescription types of the current loading subpath
        presctype = self._mac_load_presctype[:, subpath_id]
        load = {key: np.zeros(self._mac_load[key].shape[0])
                for key in self._mac_load.keys() if key in presctype}
        for ltype in load.keys():
            load[ltype] = self._mac_load[ltype][:, 1 + subpath_id]
        # Get loading subpath incremental load factors and incremental times
        inc_lfacts = list(self._mac_load_increm[str(subpath_id)][:, 0])
        inc_times = list(self._mac_load_increm[str(subpath_id)][:, 1])
        # Add a new loading subpath
        self._load_subpaths.append(LoadingSubpath(subpath_id, self._conv_hom_state, load,
                                                  presctype, inc_lfacts, inc_times,
                                                  self._max_subinc_level))
    # --------------------------------------------------------------------------------------
    def _get_load_subpath(self):
        '''Get current macroscale loading subpath.

        Returns
        -------
        load_subpath : LoadingSubpath
            Current macroscale loading subpath.
        '''
        return self._load_subpaths[self._increm_state['subpath_id']]
    # --------------------------------------------------------------------------------------
    def _update_inc(self):
        '''Update loading increment counters.'''
        # Increment (+1) global increment counter
        self._increm_state['inc'] += 1
        # Increment (+1) loading subpath increment counter
        self._get_load_subpath().update_inc()
    # --------------------------------------------------------------------------------------
    def _get_increm_load(self):
        '''Compute incremental macroscale loading.

        Returns
        -------
        inc_mac_load : dict
            For each loading nature type (key, {'strain', 'stress'}), stores the incremental
            macroscale loading constraint in a ndarray of shape (n_comps,).
        '''
        # Get current incremental applied macroscale loading
        inc_mac_load = self._get_load_subpath().get_inc_applied_load()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return inc_mac_load
    # --------------------------------------------------------------------------------------
    def _remove_sym(self, comp_order_sym, comp_order_nsym):
        '''Remove the symmetric components of macroscale loading related objects.

        Under an infinitesimal strain formulation, remove the symmetric strain/stress
        components of macroscale loading related objects. In addition, the remaining
        independent components are sorted according to the problem strain/stress symmetric
        component order.

        Parameters
        ----------
        comp_order_sym : list
            Symmetric strain/stress components (str) order.
        comp_order_nsym : list
            Nonsymmetric strain/stress components (str) order.
        '''
        # Copy macroscale loading objects
        mac_load_cp = copy.deepcopy(self._mac_load)
        mac_load_presctype_cp = copy.deepcopy(self._mac_load_presctype)
        # Loop over symmetric components indexes
        for i in range(len(comp_order_sym)):
            # Get non-symmetric component index
            j = comp_order_nsym.index(comp_order_sym[i])
            # Assemble symmetric components
            for ltype in self._mac_load.keys():
                if ltype in self._mac_load_presctype:
                    self._mac_load[ltype][i, :] = mac_load_cp[ltype][j, :]
            self._mac_load_presctype[i, :] = mac_load_presctype_cp[j, :]
        # Remove (non-symmetric) additional components
        n_sym = len(comp_order_sym)
        for ltype in self._mac_load.keys():
            if ltype in self._mac_load_presctype:
                self._mac_load[ltype] = self._mac_load[ltype][0:n_sym, :]
        self._mac_load_presctype = self._mac_load_presctype[:n_sym, :]
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_load_mf(n_dim, comp_order, load_vector):
        '''Get matricial form of macroscale load tensor given in vector form.

        Parameters
        ----------
        comp_order : list
            Strain/Stress components (str) order.
        load_vector : ndarray of shape (n_comps,)
            Macroscale load tensor in vector form.

        Returns
        -------
        load_mf : ndarray  of shape (n_comps,)
            Macroscale load tensor matricial form.

        Notes
        -----
        The matricial form storage is perform according to the provided strain/stress
        components order.
        '''
        # Initialize incremental macroscale load tensor
        load_matrix = np.zeros((n_dim, n_dim))
        # Build incremental macroscale load tensor
        for j in range(n_dim):
            for i in range(0, j + 1):
                load_matrix[i, j] = load_vector[comp_order.index(str(i + 1) + str(j + 1))]
                if i != j:
                    if n_dim**2 == len(comp_order):
                        load_matrix[j, i] = load_vector[comp_order.index(str(j + 1) +
                                                                         str(i + 1))]
                    else:
                        load_matrix[j, i] = load_matrix[i, j]
        # Set incremental macroscopic load matricial form
        load_mf = mop.get_tensor_mf(load_matrix, n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return load_mf
# ------------------------------------------------------------------------------------------
class LoadingSubpath:
    '''Loading subpath.

    Attributes
    ----------
    _inc : int
        Loading subpath increment counter.
    _total_lfact : float
        Loading subpath total load factor.
    _total_time : float
        Loading subpath total time.
    _n_presc_strain : int
        Number of prescribed loading strain components.
    _n_presc_stress : int
        Number of prescribed loading stress components.
    _presc_strain_idxs : list
        Prescribed loading strain components indexes.
    _presc_stress_idxs : list
        Prescribed loading stress components indexes.
    _inc_applied_load : dict
        For each prescribed loading nature type (key, {'strain', 'stress'}), stores the
        current incremental applied loading constraints in a ndarray of shape (n_comps,).
    _is_last_subpath_inc : bool
        Loading subpath last increment flag.
    _sub_inc_levels : list
        History of subincrementation levels.
    '''
    def __init__(self, id, init_conv_hom_state, load, presctype, inc_lfacts, inc_times,
                 max_subinc_level):
        '''Loading subpath constructor.

        Parameters
        ----------
        id : int
            Loading subpath id.
        init_conv_hom_state : dict
            Converged homogenized state (item, ndarray of shape (n_comps,)) for key in
            {'strain', 'stress'} at the beginning of loading subpath.
        load : dict
            For each prescribed loading nature type (key, {'strain', 'stress'}), stores the
            loading constraints in a ndarray of shape (n_comps,).
        presctype : ndarray of shape (n_comps,)
            Loading nature type (str, {'strain', 'stress'}) associated with each loading
            constraint.
        inc_lfacts : ndarray of shape (n_increments,)
            Loading subpath incremental load factors.
        inc_times : ndarray of shape (n_increments,)
            Loading subpath incremental times.
        max_subinc_level : int
            Maximum level of loading subincrementation.
        '''
        self._id = id
        self._init_conv_hom_state = copy.deepcopy(init_conv_hom_state)
        self._load = copy.deepcopy(load)
        self._presctype = copy.deepcopy(presctype)
        self._inc_lfacts = copy.deepcopy(inc_lfacts)
        self._inc_times = copy.deepcopy(inc_times)
        self._max_subinc_level = max_subinc_level
        # Initialize loading subpath increment counter
        self._inc = 0
        # Initialize loading subpath total load factor
        self._total_lfact = 0
        # Initialize loading subpath total time
        self._total_time = 0
        # Set number of prescribed loading strain and stress components and associated
        # indexes
        self._n_presc_strain = sum([x == 'strain' for x in self._presctype])
        self._n_presc_stress = sum([x == 'stress' for x in self._presctype])
        self._presc_strain_idxs = []
        self._presc_stress_idxs = []
        for i in range(len(presctype)):
            if presctype[i] == 'strain':
                self._presc_strain_idxs.append(i)
            else:
                self._presc_stress_idxs.append(i)
        # Initialize current incremental applied loading
        self._inc_applied_load = {key: np.zeros(load[key].shape[0]) for key in load.keys()}
        # Initialize loading subpath last increment flag
        self._is_last_subpath_inc = False
        # Initialize subincrementation levels
        self._sub_inc_levels = [0]*len(self._inc_lfacts)
    # --------------------------------------------------------------------------------------
    def get_state(self):
        '''Get loading subpath state data.

        Returns
        -------
        id : int
            Loading subpath id.
        inc : int
            Loading subpath increment counter.
        total_lfact : float
            Loading subpath current total load factor.
        inc_lfact : float
            Loading subpath current incremental load factor.
        total_time : float
            Loading subpath current total time.
        inc_time : float
            Loading subpath current incremental time.
        sub_inc_level : int
            Loading subpath current subincrementation level.
        '''
        # Get loading subpath current increment index
        inc_idx = self._inc - 1
        # Return
        return [self._id, self._inc, self._total_lfact, self._inc_lfacts[inc_idx],
                self._total_time, self._inc_times[inc_idx], self._sub_inc_levels[inc_idx]]
    # --------------------------------------------------------------------------------------
    def update_inc(self):
        '''Update increment counter, total load factor and applied macroscale loading.'''
        # Increment (+1) loading subpath increment counter
        self._inc += 1
        # Get loading subpath current increment index
        inc_idx = self._inc - 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Procedure related with the loading subincrementation: upon convergence of a given
        # increment, guarantee that the following increment magnitude is at most one
        # (subincrementation) level above. The increment cut procedure is performed the
        # required number of times in order to ensure this progressive recovery towards the
        # prescribed incrementation
        if self._inc > 1:
            while self._sub_inc_levels[inc_idx - 1] - self._sub_inc_levels[inc_idx] >= 2:
                self.increment_cut()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        # Update total time
        self._total_time = sum(self._inc_times[0:self._inc])
        # Update current incremental applied loading
        self._update_inc_applied_load()
        # Check if last increment
        if self._inc == len(self._inc_lfacts):
            self._is_last_subpath_inc = True
    # --------------------------------------------------------------------------------------
    def increment_cut(self):
        '''Perform loading increment cut.'''
        # Get loading subpath current increment index
        inc_idx = self._inc - 1
        # Update subincrementation level
        self._sub_inc_levels[inc_idx] += 1
        self._sub_inc_levels.insert(inc_idx + 1, self._sub_inc_levels[inc_idx])
        # Check if maximum subincrementation level is surpassed
        if self._sub_inc_levels[inc_idx] > self._max_subinc_level:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00095', location.filename, location.lineno + 1,
                                self._max_subinc_level)
        # Get current incremental load factor and associated incremental time
        inc_lfact = self._inc_lfacts[inc_idx]
        inc_time = self._inc_times[inc_idx]
        # Cut load increment in half
        self._inc_lfacts[inc_idx] = inc_lfact/2.0
        self._inc_lfacts.insert(inc_idx + 1, self._inc_lfacts[inc_idx])
        self._inc_times[inc_idx] = inc_time/2.0
        self._inc_times.insert(inc_idx + 1, self._inc_times[inc_idx])
        # Update total load factor and total time
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        self._total_time = sum(self._inc_times[0:self._inc])
        # Update current incremental applied loading
        self._update_inc_applied_load()
        # Set loading subpath last increment flag
        self._is_last_subpath_inc = False
    # --------------------------------------------------------------------------------------
    def get_inc_applied_load(self):
        '''Get current incremental applied loading.

        Returns
        -------
        _inc_applied_load : dict
            For each prescribed loading nature type (key, {'strain', 'stress'}), stores the
            current incremental applied loading constraints in a ndarray of shape
            (n_comps,).
        '''
        return copy.deepcopy(self._inc_applied_load)
    # --------------------------------------------------------------------------------------
    def _update_inc_applied_load(self):
        '''Update current incremental applied loading.'''
        # Get loading subpath current increment index
        inc_idx = self._inc - 1
        # Get current incremental load factor and associated incremental time
        inc_lfact = self._inc_lfacts[inc_idx]
        # Update current incremental applied loading
        for ltype in self._inc_applied_load.keys():
            for i in range(len(self._inc_applied_load[ltype])):
                if self._presctype[i] == ltype:
                    self._inc_applied_load[ltype][i] = \
                        inc_lfact*(self._load[ltype][i] -
                                   self._init_conv_hom_state[ltype][i])
# ------------------------------------------------------------------------------------------
class IncrementRewinder:
    '''Rewind analysis to initial instant (rewind state) of past macroscale loading
    increment.

    Attributes
    ----------
    _rewind_inc : int
        Increment associated to the rewind state.
    _loading_path : LoadingPath
        Loading path instance rewind state.
    _material_state : MaterialState
        CRVE material constitutive state at rewind state.
    _clusters_sct_mf : dict
        Fourth-order strain concentration tensor (matricial form) (item, ndarray)
        associated to each material cluster (key, str).
    _ref_material : ElasticReferenceMaterial
        Elastic reference material at rewind state.
    '''
    def __init__(self, rewind_inc, phase_clusters):
        '''Increment rewinder constructor.

        Parameters
        ----------
        rewind_inc : int
            Increment associated to the rewind state.
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        '''
        self._rewind_inc = rewind_inc
        self._phase_clusters = copy.deepcopy(phase_clusters)
        # Initialize loading path at rewind state
        self._loading_path = None
        # Initialize material constitutive state at rewind state
        self._material_state = None
        # Initialize elastic reference material at rewind state
        self._ref_material = None
        # Initialize clusters strain concentration tensors at rewind state
        self._clusters_sct_mf = None
    # --------------------------------------------------------------------------------------
    def get_rewind_inc(self):
        '''Get increment associated to the rewind state.

        Parameters
        ----------
        rewind_inc : int
            Increment associated to the rewind state.
        '''
        return self._rewind_inc
    # --------------------------------------------------------------------------------------
    def save_loading_path(self, loading_path):
        '''Save loading path rewind state.

        Parameters
        ----------
        loading_path : LoadingPath
            LoadingPath instance.
        '''
        self._loading_path = copy.deepcopy(loading_path)
    # --------------------------------------------------------------------------------------
    def get_loading_path(self):
        '''Get loading path at rewind state.

        Returns
        -------
        loading_path : LoadingPath
            Loading path instance rewind state.
        '''
        return copy.deepcopy(self._loading_path)
    # --------------------------------------------------------------------------------------
    def save_material_state(self, material_state):
        '''Save material constitutive state at rewind state.

        Parameters
        ----------
        material_state : MaterialState
            CRVE material constitutive state at rewind state.
        '''
        self._material_state = copy.deepcopy(material_state)
    # --------------------------------------------------------------------------------------
    def get_material_state(self, crve):
        '''Get material constitutive state at rewind state.

        Parameters
        ----------
        crve : CRVE
            Cluster-Reduced Representative Volume Element.

        Returns
        -------
        material_state : MaterialState
            CRVE material constitutive state at rewind state.
        '''
        # If the current CRVE clustering is coincident with the CRVE clustering at the
        # rewind state, simply return the material constitutive state stored at rewind
        # state. Otherwise, perform a suitable transfer of state variables between the
        # rewind state CRVE clustering and the current CRVE clustering
        if self._phase_clusters == crve.get_cluster_phases():
            # Return material constitutive state stored at rewind state
            return copy.deepcopy(self._material_state)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Get clusters state variables at rewind state
            clusters_state_rew = self._material_state.get_clusters_state()
            # Get clusters deformation gradient at rewind state
            clusters_def_gradient_rew_mf = \
                self._material_state.get_clusters_def_gradient_mf()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize clusters state variables
            clusters_state = {}
            # Initialize clusters deformation gradient
            clusters_def_gradient_mf = {}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Get material phases
            material_phases = crve.get_material_phases()
            # Get cluster-reduced material phases
            cluster_phases = crve.get_cluster_phases()
            # Get clusters associated to each material phase
            phase_clusters = crve.get_phase_clusters()
            # Get clusters volume fraction
            clusters_vf = crve.get_clusters_vf()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over material phases
            for mat_phase in material_phases:
                # Get cluster-reduced material phase
                crmp = cluster_phases[mat_phase]
                # Get clustering type
                clustering_type = crmp.get_clustering_type()
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Proceed according to clustering type
                if clustering_type == 'static':
                    # Loop over material phase clusters
                    for cluster in phase_clusters[mat_phase]:
                        # Set cluster state variables
                        clusters_state[str(cluster)] = \
                            copy.deepcopy(clusters_state_rew[str(cluster)])
                        # Set cluster deformation gradient
                        clusters_def_gradient_mf[str(cluster)] = \
                            copy.deepcopy(clusters_def_gradient_rew_mf[str(cluster)])
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                elif clustering_type == 'adaptive':
                    # Get cluster-reduced material phase clustering tree nodes
                    clustering_tree_nodes, root_cluster_node = \
                        crmp.get_clustering_tree_nodes()
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Get rewind state cluster nodes
                    rewind_clusters_nodes = []
                    for cluster in self._phase_clusters[mat_phase]:
                        rewind_clusters_nodes.append(clustering_tree_nodes[str(cluster)])
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Initialize node walker
                    node_walker = Walker()
                    # Loop over material phase clusters
                    for cluster in phase_clusters[mat_phase]:
                        # Get cluster node
                        cluster_node = clustering_tree_nodes[str(cluster)]
                        # Build walk from cluster node up to the root node
                        node_walk_to_root = node_walker.walk(cluster_node,
                                                             root_cluster_node)
                        # Loop over walk nodes
                        for node in node_walk_to_root[0]:
                            # Find hierarchicaly closest rewind state cluster node
                            if node in rewind_clusters_nodes:
                                # Get node cluster
                                parent_cluster = int(node.name)
                                # Set cluster state variables
                                clusters_state[str(cluster)] = \
                                    copy.deepcopy(clusters_state_rew[str(parent_cluster)])
                                # Set cluster deformation gradient
                                clusters_def_gradient_mf[str(cluster)] = \
                                    copy.deepcopy(clusters_def_gradient_rew_mf[
                                        str(parent_cluster)])
                                # Skip to the following cluster
                                break
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                else:
                    raise RuntimeError('Unknown material phase clustering type.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material constitutive state at rewind state according to the current
            # clustering
            self._material_state.set_rewind_state_updated_clustering(
                phase_clusters, clusters_vf, clusters_state, clusters_def_gradient_mf)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return material constitutive state stored at rewind state according to the
            # update clustering
            return copy.deepcopy(self._material_state)
    # --------------------------------------------------------------------------------------
    def save_reference_material(self, ref_material):
        '''Save elastic reference material at rewind state.

        Parameters
        ----------
        ref_material : ElasticReferenceMaterial
            Elastic reference material at rewind state.
        '''
        # Save elastic reference material
        self._ref_material = copy.deepcopy(ref_material)
    # --------------------------------------------------------------------------------------
    def get_reference_material(self):
        '''Get elastic reference material at rewind state.

        Returns
        -------
        ref_material : ElasticReferenceMaterial
            Elastic reference material at rewind state.
        '''
        return copy.deepcopy(self._ref_material)
    # --------------------------------------------------------------------------------------
    def save_clusters_sct(self, clusters_sct_mf):
        '''Save clusters strain concentration tensors at rewind state.

        Parameters
        ----------
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).
        '''
        # Save clusters state variables
        self._clusters_sct_mf = copy.deepcopy(clusters_sct_mf)
    # --------------------------------------------------------------------------------------
    def get_clusters_sct(self):
        '''Get clusters strain concentration tensors at rewind state.

        Parameters
        ----------
        clusters_sct_mf : dict
            Fourth-order strain concentration tensor (matricial form) (item, ndarray)
            associated to each material cluster (key, str).
        '''
        # Save clusters state variables
        return copy.deepcopy(self._clusters_sct_mf)
    # --------------------------------------------------------------------------------------
    def rewind_output_files(self, hres_output=None, ref_mat_output=None, voxels_output=None,
                            adapt_output=None, vtk_output=None):
        '''Rewind output files to the rewind state.

        Parameters
        ----------
        hres_output : HomResOutput
            Output associated to the homogenized results.
        ref_mat_output : RefMatOutput
            Output associated to the reference material.
        voxels_output : VoxelsOutput
            Output associated to voxels material-related quantities.
        adapt_output : ClusteringAdaptivityOutput
            Output associated to the clustering adaptivity procedures.
        vtk_output : VTKOutput
            Output associated to the VTK files.
        '''
        # Rewind output files
        if hres_output != None:
            hres_output.rewind_file(self._rewind_inc)
        if ref_mat_output != None:
            ref_mat_output.rewind_file(self._rewind_inc)
        if voxels_output != None:
            voxels_output.rewind_file(self._rewind_inc)
        if adapt_output != None:
            adapt_output.rewind_file(self._rewind_inc)
        if vtk_output != None:
            vtk_output.rewind_files(self._rewind_inc)
# ------------------------------------------------------------------------------------------
class RewindManager:
    '''Manage analysis rewind operations and evaluate analysis rewind criteria.

    Attributes
    ----------
    _n_rewinds : int
        Number of rewind operations.
    _rewind_time : float
        Total time spent in rewind operations and in deleted analysis increments.
    _init_time : float
        Reference time.
    '''
    def __init__(self, rewind_state_criterion, rewinding_criterion, max_n_rewinds=1):
        '''Analysis rewind manager constructor.

        Parameters
        ----------
        rewind_state_criterion : tuple
            Rewind state storage criterion [0] and associated parameter [1].
        rewinding_criterion : tuple
            Rewinding criterion [0] and associated parameter [1].
        max_n_rewinds : int, default=1
            Maximum number of rewind operations.
        '''
        self._rewind_state_criterion = rewind_state_criterion
        self._rewinding_criterion = rewinding_criterion
        self._max_n_rewinds = max_n_rewinds
        # Initialize number of rewind operations
        self._n_rewinds = 0
        # Initialize total rewind time
        self._rewind_time = 0
    # --------------------------------------------------------------------------------------
    def get_rewind_time(self):
        '''Get total time spent in rewind operations and deleted analysis increments.

        Parameters
        ----------
        rewind_time : float
            Total time spent in rewind operations and in deleted analysis increments.
        '''
        return self._rewind_time
    # --------------------------------------------------------------------------------------
    def update_rewind_time(self, mode='init'):
        '''Update total rewind time.

        Parameters
        ----------
        mode : str, {'init', 'update'}, default='init'
        '''
        if mode == 'init':
            # Set reference initial time
            self._init_time = time.time()
        elif mode == 'update':
            # Update total rewind time
            self._rewind_time += time.time() - self._init_time
            # Set reference initial time
            self._init_time = time.time()
        else:
            raise RuntimeError('Unknown mode.')
    # --------------------------------------------------------------------------------------
    def is_rewind_available(self):
        '''Evaluate if rewind operations are available.

        Returns
        -------
        is_available : bool
            True if rewind operations are available, False otherwise.
        '''
        # Initialize rewind operations availability
        is_available = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate maximum number of rewind operations
        if self._n_rewinds >= self._max_n_rewinds:
            is_available = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return rewind operations availability
        return is_available
    # --------------------------------------------------------------------------------------
    def is_save_rewind_state(self, inc):
        '''Evaluate conditions to save rewind state.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.

        Returns
        -------
        is_save_state : bool
            True if conditions to save rewind state are satisfied, False otherwise.
        '''
        # Initialize save rewind state flag
        is_save_state = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get rewind state criterion
        criterion = self._rewind_state_criterion[0]
        # Evaluate rewind state criterion
        if criterion == 'increment_number':
            # Evaluate increment number
            if inc == self._rewind_state_criterion[1]:
                is_save_state = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown rewind state criterion.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return save rewind state flag
        return is_save_state
    # --------------------------------------------------------------------------------------
    def is_rewinding_criteria(self, inc, material_phases, phase_clusters, clusters_state):
        '''Check analysis rewinding criteria.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.
        material_phases : list
            CRVE material phases labels (str).
        phase_clusters : dict
            Clusters labels (item, list of int) associated to each material phase
            (key, str).
        clusters_state : dict
            Material constitutive model state variables (item, dict) associated to each
            material cluster (key, str).

        Returns
        -------
        is_rewind : bool
            True if analysis rewinding criteria are satisfied, False otherwise.
        '''
        # Initialize analysis rewind flag
        is_rewind = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get rewinding criterion
        criterion = self._rewinding_criterion[0]
        # Evaluate analysis rewinding criterion
        if criterion == 'increment_number':
            # Evaluate increment number
            is_rewind = inc == self._rewinding_criterion[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif criterion == 'max_acc_p_strain':
            # Evaluate accumulated plastic strain threshold
            for mat_phase in material_phases:
                # Loop over material phase clusters
                for cluster in phase_clusters[mat_phase]:
                    # Get cluster state variables
                    state_variables = clusters_state[str(cluster)]
                    # Check if accumulated plastic strain is cluster state variable
                    if not 'acc_p_strain' in state_variables:
                        continue
                    # Evaluate accumulated plastic strain
                    if state_variables['acc_p_strain'] > self._rewinding_criterion[1]:
                        is_rewind = True
                        break
                if is_rewind:
                    break
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            raise RuntimeError('Unknown rewinding criterion.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Increment number of rewind operations
        if is_rewind:
            self._n_rewinds += 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return analysis rewinding flag
        return is_rewind
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_save_rewind_state_criteria():
        '''Get available rewind state storage criteria and default parameters.

        Returns
        -------
        available_save_rewind_state_criteria : dict
            Available rewind state storage criteria (key, str) and associated default
            parameters (item).
        '''
        # Set available rewind state storage criteria
        available_save_rewind_state_criteria = {'increment_number': 0,}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_save_rewind_state_criteria
    # --------------------------------------------------------------------------------------
    @staticmethod
    def get_rewinding_criteria():
        '''Get rewinding criteria and default parameters.

        Returns
        -------
        available_rewinding_criteria : dict
            Available rewinding criteria (key, str) and associated default
            parameters (item).
        '''
        # Set available rewinding criteria
        available_rewinding_criteria = {'increment_number': 0,
                                        'max_acc_p_strain': 1.0e-10}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return available_rewinding_criteria
