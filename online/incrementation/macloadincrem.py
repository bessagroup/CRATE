#
# Macroscale Loading Incrementation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the enforcement of the macroscale loading constraints.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                           Loading path and subpath classes
# ==========================================================================================
class LoadingPath:
    '''Collection of macroscale loading subpaths, macroscale loading state and tools to
    control the macroscale loading incrementation flow.'''

    def __init__(self, strain_formulation, mac_load, mac_load_presctype, mac_load_increm):
        '''Macroscale loading path constructor.'''

        self._strain_formulation = strain_formulation
        self._mac_load = mac_load
        self._mac_load_presctype = mac_load_presctype
        self._mac_load_increm = mac_load_increm
        # Set total number of loading subpaths
        self._n_load_subpaths = len(mac_load_increm.keys())
        # Initialize list of macroscale loading subpaths
        self._load_subpaths = []
        # Initialize increment state
        self._increm_state = {'inc': 0, 'subpath_id': -1}
        # Initialize converged macroscale load
        self._conv_mac_load = {key: np.zeros(mac_load.shape[0]) for key in mac_load.keys()}
        # Initialize loading last increment flag
        self._is_last_inc = False
    # --------------------------------------------------------------------------------------
    def new_load_increment(self, n_dim, comp_order):
        '''Setup new macroscale loading increment and get associated data.'''

        # Add a new loading subpath to the loading path if either first load increment or
        # current loading subpath is completed
        if self._increm_state['inc'] == 0 or self._get_load_subpath()._is_last_subpath_inc:
            # Add a new loading subpath
            self._new_subpath()
        # Get current loading subpath
        load_subpath = self._get_load_subpath()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update macroscale loading increment
        self._update_inc()
        # Check if last macroscale loading increment
        if load_subpath._id == self._n_load_subpaths and load_subpath._is_last_subpath_inc:
            self._is_last_inc = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute incremental macroscale loading
        inc_mac_load = self._get_increm_load()
        inc_mac_load_mf = {}
        for ltype in inc_mac_load_mf.keys():
            inc_mac_load_mf[ltype] = type(self)._get_load_mf(n_dim, comp_order,
                                                             inc_mac_load[ltype])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return [inc_mac_load_mf, load_subpath._n_presc_strain,
                load_subpath._presc_strain_idx, load_subpath._n_presc_stress,
                load_subpath._presc_stress_idx, self._is_last_inc]
    # --------------------------------------------------------------------------------------
    def update_conv_load(self):
        '''Update converged macroscale load.'''

        self._conv_mac_load = copy.deepcopy(self._get_load_subpath._applied_load)
    # --------------------------------------------------------------------------------------
    def _new_subpath(self):
        '''Add a new macroscale loading subpath to the loading path.'''

        # Increment (+1) loading subpath id
        self._increm_state['subpath_id'] += 1
        subpath_id = self._increm_state['subpath_id']
        # Get macroscale load and prescription types of the current loading subpath
        load = {key: np.zeros(self.mac_load.shape[0]) for key in self.mac_load.keys()}
        for ltype in self._mac_load.keys():
            load[ltype] = self._mac_load[ltype][:, 1 + subpath_id]
        presctype = self._mac_load_presctype[:, subpath_id]
        # Get loading subpath incremental load factors and incremental times
        inc_lfacts = self._mac_load_increm[str(subpath_id)][:, 0]
        inc_times = self._mac_load_increm[str(subpath_id)][:, 1]
        # Add a new macroscale loading subpath
        self._load_subpaths.append(LoadingSubpath(subpath_id, load, presctype, inc_lfacts,
                                                  inc_times))
    # --------------------------------------------------------------------------------------
    def _get_load_subpath(self):
        '''Get current loading subpath.'''

        return self._load_subpaths[str(self._increm_state['subpath_id'])]
    # --------------------------------------------------------------------------------------
    def _update_inc(self):
        '''Update macroscale loading increment counters.'''

        # Increment (+1) global increment counter
        self._increm_state['inc'] += 1
        # Increment (+1) loading subpath increment counter
        self._get_load_subpath._update_inc()
    # --------------------------------------------------------------------------------------
    def _get_increm_load(self):
        '''Compute incremental macroscale loading.'''

        # Get total macroscale loading applied in the current increment
        applied_load = self._get_load_subpath._applied_load
        # Compute incremental macroscale loading
        inc_mac_load = {key: np.zeros(applied_load) for key in applied_load.keys()}
        for ltype in inc_mac_load.keys():
            inc_mac_load[ltype] = applied_load[ltype] - self._conv_mac_load[ltype]
        return inc_mac_load
    # --------------------------------------------------------------------------------------
    @staticmethod
    def _get_load_mf(n_dim, comp_order, load_vector):
        '''Get matricial form of macroscale load tensor given in vector form.'''

        # Initialize incremental macroscale load tensor
        load_matrix = np.zeros((n_dim, n_dim))
        # Build incremental macroscale load tensor
        k = 0
        for j in range(n_dim):
            for i in range(n_dim):
                load_matrix[i, j] = load_vector[k]
                k = k + 1
        # Set incremental macroscopic load matricial form
        load_mf = mop.gettensormf(load_matrix, n_dim, comp_order)
        # Return
        return load_mf
# ------------------------------------------------------------------------------------------
class LoadingSubpath:
    '''Macroscale loading subpath.'''

    def __init__(self, id, load, presctype, inc_lfacts, inc_times):
        '''Macroscale loading subpath constructor.'''

        self._id = id
        self._load = load
        self._presctype = presctype
        self._inc_lfacts = inc_lfacts
        self._inc_times = inc_times
        # Initialize loading subpath increment counter
        self._inc = 0
        # Initialize loading subpath total load factor
        self._total_lfact = 0
        # Set number of prescribed macroscale loading strain and stress components and
        # associated indexes
        self._n_presc_strain = sum([x == 'strain' for x in self._presctype])
        self._n_presc_stress = sum([x == 'stress' for x in self._presctype])
        for i in range(len(presctype)):
            if presctype[i] == 'strain':
                self._presc_strain_idx.append(i)
            else:
                self._presc_stress_idx.append(i)
        # Initialize total applied macroscale loading
        self._applied_load = {key: np.zeros(load.shape[0]) for key in load.keys()}
        # Initialize loading subpath last increment flag
        self._is_last_subpath_inc = False
    # --------------------------------------------------------------------------------------
    def _update_inc(self):
        '''Update increment counter and total load factor accordingly.'''

        # Increment (+1) loading subpath increment counter
        self._inc += 1
        # Update total load factor
        self._total_lfact = sum(self._inc_lfacts[0:self._inc - 1])
        # Update total applied macroscale loading
        self._update_applied_load()
        # Check if last increment
        if self._inc == len(self._inc_lfacts):
            self._is_last_subpath_inc = True
    # --------------------------------------------------------------------------------------
    def _update_applied_load(self):
        '''Update total applied macroscale loading.'''

        for ltype in self._applied_load.keys():
            for i in range(len(self._applied_load[type])):
                if self._presctype[i] == ltype:
                    self._applied_load[ltype][i] = self.total_lfact*self._load[ltype][i]



#
#                                                          Macroscale loading incrementation
# ==========================================================================================
# Set the incremental macroscale load data
def macloadincrem(problem_dict, macload_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get macroscale loading data
    mac_load_type = macload_dict['mac_load_type']
    mac_load = macload_dict['mac_load']
    mac_load_presctype = macload_dict['mac_load_presctype']
    n_load_increments = macload_dict['n_load_increments']
    # Set incremental macroscale loading
    inc_mac_load_mf = dict()
    load_types = {1: ['strain',], 2: ['stress',], 3: ['strain','stress']}
    for load_type in load_types[mac_load_type]:
        inc_mac_load_mf[load_type] = getincmacloadmf(
            n_dim, comp_order, mac_load[load_type][:,1])/n_load_increments
    # Compute number of prescribed macroscale strain and stress components
    n_presc_mac_strain = sum([mac_load_presctype[comp] == 'strain' for comp in comp_order])
    n_presc_mac_stress = sum([mac_load_presctype[comp] == 'stress' for comp in comp_order])
    # Set macroscale strain and stress prescribed components indexes
    presc_strain_idxs = list()
    presc_stress_idxs = list()
    for i in range(len(comp_order)):
        comp = comp_order[i]
        if mac_load_presctype[comp] == 'strain':
            presc_strain_idxs.append(i)
        else:
            presc_stress_idxs.append(i)
    # Return
    return [inc_mac_load_mf, n_presc_mac_strain, n_presc_mac_stress, presc_strain_idxs,
            presc_stress_idxs]
# ------------------------------------------------------------------------------------------
# Under an infinitesimal strain formulation, set the incremental macroscopic load strain or
# stress tensor matricial form according to Kelvin notation
def getincmacloadmf(n_dim, comp_order, inc_mac_load_vector):
    # Initialize incremental macroscale load tensor
    inc_mac_load = np.zeros((n_dim, n_dim))
    # Build incremental macroscale load tensor
    k = 0
    for j in range(n_dim):
        for i in range(n_dim):
            inc_mac_load[i, j] = inc_mac_load_vector[k]
            k = k + 1
    # Set incremental macroscopic load matricial form
    inc_mac_load_mf = mop.gettensormf(inc_mac_load, n_dim, comp_order)
    # Return
    return inc_mac_load_mf
