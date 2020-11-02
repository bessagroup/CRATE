#
# Macroscale Loading Incrementation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the enforcement of the macroscale loading constraints and the
# overall macroscalse loading incrementation flow..
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2020 | Initial coding.
# Bernardo P. Ferreira | Jul 2020 | Implemented LoadingPath and LoadingSubpath classes.
# Bernardo P. Ferreira | Nov 2020 | Updated documentation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Shallow and deep copy operations
import copy
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                           Loading path and subpath classes
# ==========================================================================================
class LoadingPath:
    '''Class that controls the macroscale loading incrementation flow.

    This class contains a collection of macroscale loading subpaths, the current macroscale
    loading state and a set of methods to control the macroscale loading incrementation
    flow.

    Attributes
    ----------
    _n_load_subpaths : int
        Number of loading subpaths.
    _load_subpaths : list
        List of LoadingSubpath.
    _conv_hom_state : dict
        Converged macroscale homogenized state (item, ndarray of shape (n_comps,)) for
        key in {'strain', 'stress'}.
    _is_last_inc : bool
        Loading last increment flag.
    _n_cinc_cuts : int
        Consecutive macroscale loading increment cuts counter
    increm_state : dict
        Increment state: key 'inc' contains the current increment number, key 'subpath_id'
        contains the current loading subpath index.
    '''
    def __init__(self, strain_formulation, comp_order_sym, comp_order_nsym, mac_load,
                 mac_load_presctype, mac_load_increm, max_subinc_level=5, max_cinc_cuts=5):
        '''Macroscale loading path constructor.

        Parameters
        ----------
        strain_formulation : int
            Strain formulation: (1) infinitesimal strains, (2) finite strains.
        comp_order_sym : list
            Symmetric strain/stress components (str) order.
        comp_order_nsym : list
            Nonsymmetric strain/stress components (str) order.
        mac_load : dict
            For each loading nature type (key, {'strain', 'stress'}), stores the macroscale
            loading constraints for each loading subpath in a ndarray, where the i-th row
            is associated with the component mac_load[i, 0] (str) and the j-th column is
            associated with the j-th loading subpath.
        mac_load_presctype : ndarray of shape (n_comps, n_load_subpaths)
            Loading nature type (str, {'strain', 'stress'}) associated with each macroscale
            loading constraint, where the i-th row is associated with the component
            mac_load[i, 0] (str) and the j-th column is associated with the (j+1)-th loading
            subpath.
        mac_load_increm : dict
            For each loading subpath id (key, str), stores a ndarray of shape
            (n_load_increments, 2) where each row is associated with a prescribed loading
            increment, and the columns 0 and 1 contain the corresponding incremental load
            factor and incremental time, respectively.
        max_subinc_level : int, default=5
            Maximum level of macroscale loading subincrementation.
        max_cinc_cuts : int, default=5
            Maximum number of consecutive macroscale load increment cuts.
        '''
        self._strain_formulation = strain_formulation
        self._mac_load = mac_load
        self._mac_load_presctype = mac_load_presctype
        self._mac_load_increm = mac_load_increm
        self._max_subinc_level = max_subinc_level
        self._max_cinc_cuts = max_cinc_cuts
        # Remove symmetric components under an infinitesimal strain formulation
        if strain_formulation == 1:
            self._remove_sym(comp_order_sym, comp_order_nsym)
        # Set total number of loading subpaths
        self._n_load_subpaths = len(mac_load_increm.keys())
        # Initialize list of macroscale loading subpaths
        self._load_subpaths = []
        # Initialize increment state
        self.increm_state = {'inc': 0, 'subpath_id': -1}
        # Initialize converged macroscale (homogenized) state
        self._conv_hom_state = {key: None for key in ['strain', 'stress']}
        # Initialize loading last increment flag
        self._is_last_inc = False
        # Initialize consecutive increment cuts counter
        self._n_cinc_cuts = 0
    # --------------------------------------------------------------------------------------
    def new_load_increment(self, n_dim, comp_order):
        '''Setup new macroscale loading increment and get associated data.

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
        # Reset consecutive macroscale loading increment cuts counter
        self._n_cinc_cuts = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add a new loading subpath to the loading path if either first load increment or
        # current loading subpath is completed
        if self.increm_state['inc'] == 0 or self._get_load_subpath()._is_last_subpath_inc:
            # Add a new loading subpath
            self._new_subpath()
        # Get current loading subpath
        load_subpath = self._get_load_subpath()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update macroscale loading increment
        self._update_inc()
        # Check if last macroscale loading increment
        if load_subpath._id == self._n_load_subpaths - 1 and \
                load_subpath._is_last_subpath_inc:
            self._is_last_inc = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental macroscale loading
        inc_mac_load = self._get_increm_load()
        inc_mac_load_mf = {}
        for ltype in inc_mac_load.keys():
            inc_mac_load_mf[ltype] = type(self)._get_load_mf(n_dim, comp_order,
                                                             inc_mac_load[ltype])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [inc_mac_load_mf, load_subpath._n_presc_strain,
                load_subpath._presc_strain_idxs, load_subpath._n_presc_stress,
                load_subpath._presc_stress_idxs, self._is_last_inc]
    # --------------------------------------------------------------------------------------
    def increment_cut(self, n_dim, comp_order):
        '''Perform macroscale loading increment cut, setup the resulting increment and get
        associated data.

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
        # Perform macroscale loading increment
        load_subpath.increment_cut()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set last macroscale loading increment flag
        self._is_last_inc = False
        # Increment (+1) consecutive macroscale loading increment cuts counter
        self._n_cinc_cuts += 1
        # Check if maximum number of consecutive macroscale loading increment cuts is
        # surpassed
        if self._n_cinc_cuts > self._max_cinc_cuts:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00096', location.filename, location.lineno + 1,
                                self._max_cinc_cuts)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental macroscale loading
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
    def update_hom_state(self, n_dim, comp_order, hom_strain, hom_stress):
        '''Update converged macroscale (homogenized) state.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order : list
            Strain/Stress components (str) order.
        hom_strain : ndarray
            Macroscale homogenized strain tensor.
        hom_stress : ndarray
            Macroscale homogenized stress tensor.
        '''
        # Initialize converged macroscale strain and stress tensors vector form
        self._conv_hom_state['strain'] = np.zeros(len(comp_order))
        self._conv_hom_state['stress'] = np.zeros(len(comp_order))
        # Loop over strain/stress components
        for k in range(len(comp_order)):
            # Get strain/stress component
            comp = comp_order[k]
            # Get component indexes
            i = int(comp[0]) - 1
            j = int(comp[1]) - 1
            # Build converged macroscale strain and stress tensors vector form
            self._conv_hom_state['strain'][k] = hom_strain[i, j]
            self._conv_hom_state['stress'][k] = hom_stress[i, j]
    # --------------------------------------------------------------------------------------
    def get_subpath_state(self):
        '''Get current loading subpath state.

        Returns
        -------
        id : int
            Macroscale loading subpath id.
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
    def _new_subpath(self):
        '''Add a new macroscale loading subpath to the loading path.'''
        # Increment (+1) loading subpath id
        self.increm_state['subpath_id'] += 1
        subpath_id = self.increm_state['subpath_id']
        # Get macroscale load and prescription types of the current loading subpath
        presctype = self._mac_load_presctype[:, subpath_id]
        load = {key: np.zeros(self._mac_load[key].shape[0])
                for key in self._mac_load.keys() if key in presctype}
        for ltype in load.keys():
            load[ltype] = self._mac_load[ltype][:, 1 + subpath_id]
        # Get loading subpath incremental load factors and incremental times
        inc_lfacts = list(self._mac_load_increm[str(subpath_id)][:, 0])
        inc_times = list(self._mac_load_increm[str(subpath_id)][:, 1])
        # Get maximum macroscale loading subincrementation level
        _max_subinc_level = self._max_subinc_level
        # Add a new macroscale loading subpath
        self._load_subpaths.append(LoadingSubpath(subpath_id, load, presctype, inc_lfacts,
                                                  inc_times, _max_subinc_level))
    # --------------------------------------------------------------------------------------
    def _get_load_subpath(self):
        '''Get current macroscale loading subpath.

        Returns
        -------
        load_subpath : LoadingSubpath
            Current macroscale loading subpath.
        '''
        return self._load_subpaths[self.increm_state['subpath_id']]
    # --------------------------------------------------------------------------------------
    def _update_inc(self):
        '''Update macroscale loading increment counters.'''
        # Increment (+1) global increment counter
        self.increm_state['inc'] += 1
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
        # Get total macroscale loading applied in the current increment
        applied_load = self._get_load_subpath()._applied_load
        # Compute incremental macroscale loading
        inc_mac_load = {key: np.zeros(len(applied_load[key]))
                        for key in applied_load.keys()}
        for ltype in inc_mac_load.keys():
            inc_mac_load[ltype] = applied_load[ltype] - self._conv_hom_state[ltype]
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
        load_mf = mop.gettensormf(load_matrix, n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return load_mf
# ------------------------------------------------------------------------------------------
class LoadingSubpath:
    '''Macroscale loading subpath.

    Attributes
    ----------
    _inc : int
        Loading subpath increment counter.
    _total_lfact : float
        Loading subpath total load factor.
    _total_time : float
        Loading subpath total time.
    _n_presc_strain : int
        Number of prescribed macroscale loading strain components.
    _n_presc_stress : int
        Number of prescribed macroscale loading stress components.
    _presc_strain_idxs : list
        Prescribed macroscale loading strain components indexes.
    _presc_stress_idxs : list
        Prescribed macroscale loading stress components indexes.
    _applied_load : dict
        For each prescribed loading nature type (key, {'strain', 'stress'}), stores the
        total applied macroscale loading constraints in a ndarray of shape (n_comps,).
    _is_last_subpath_inc : bool
        Loading subpath last increment flag.
    _sub_inc_levels : list
        History of subincrementation levels.
    '''
    def __init__(self, id, load, presctype, inc_lfacts, inc_times, max_subinc_level):
        '''Macroscale loading subpath constructor.

        Parameters
        ----------
        id : int
            Macroscale loading subpath id.
        load : dict
            For each prescribed loading nature type (key, {'strain', 'stress'}), stores the
            macroscale loading constraints in a ndarray of shape (n_comps,).
        presctype : ndarray of shape (n_comps,)
            Loading nature type (str, {'strain', 'stress'}) associated with each
            macroscale loading constraint.
        inc_lfacts : ndarray of shape (n_increments,)
            Loading subpath incremental load factors.
        inc_times : ndarray of shape (n_increments,)
            Loading subpath incremental times.
        max_subinc_level : int
            Maximum level of macroscale loading subincrementation.
        '''
        self._id = id
        self._load = load
        self._presctype = presctype
        self._inc_lfacts = inc_lfacts
        self._inc_times = inc_times
        self._max_subinc_level = max_subinc_level
        # Initialize loading subpath increment counter
        self._inc = 0
        # Initialize loading subpath total load factor
        self._total_lfact = 0
        # Initialize loading subpath total time
        self._total_time = 0
        # Set number of prescribed macroscale loading strain and stress components and
        # associated indexes
        self._n_presc_strain = sum([x == 'strain' for x in self._presctype])
        self._n_presc_stress = sum([x == 'stress' for x in self._presctype])
        self._presc_strain_idxs = []
        self._presc_stress_idxs = []
        for i in range(len(presctype)):
            if presctype[i] == 'strain':
                self._presc_strain_idxs.append(i)
            else:
                self._presc_stress_idxs.append(i)
        # Initialize total applied macroscale loading
        self._applied_load = {key: np.zeros(load[key].shape[0]) for key in load.keys()}
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
            Macroscale loading subpath id.
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
        # Procedure related with the macroscale loading subincrementation: upon convergence
        # of a given increment, guarantee that the following increment magnitude is at most
        # one (subincrementation) level above. The increment cut procedure is performed the
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
        # Update total applied macroscale loading
        self._update_applied_load()
        # Check if last increment
        if self._inc == len(self._inc_lfacts):
            self._is_last_subpath_inc = True
    # --------------------------------------------------------------------------------------
    def increment_cut(self):
        '''Perform macroscale loading increment cut.'''
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
        # Cut the macroscale load increment in half
        self._inc_lfacts[inc_idx] = inc_lfact/2.0
        self._inc_lfacts.insert(inc_idx + 1, self._inc_lfacts[inc_idx])
        self._inc_times[inc_idx] = inc_time/2.0
        self._inc_times.insert(inc_idx + 1, self._inc_times[inc_idx])
        # Update total load factor and total time
        self._total_lfact = sum(self._inc_lfacts[0:self._inc])
        self._total_time = sum(self._inc_times[0:self._inc])
        # Update total applied macroscale loading
        self._update_applied_load()
        # Set loading subpath last increment flag
        self._is_last_subpath_inc = False
    # --------------------------------------------------------------------------------------
    def _update_applied_load(self):
        '''Update total applied macroscale loading.'''
        for ltype in self._applied_load.keys():
            for i in range(len(self._applied_load[ltype])):
                if self._presctype[i] == ltype:
                    self._applied_load[ltype][i] = self._total_lfact*self._load[ltype][i]
