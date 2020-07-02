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
# Matricial operations
import tensor.matrixoperations as mop
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
