#
# Linear Elastic Constitutive Model (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Tensorial operations
import tensorOperations as top
#
#
#                                                                             Initialization
# ==========================================================================================
# Define material constitutive model state variables and build an initialized state
# variables dictionary
#
# List of constitutive model state variables:
#
#   e_strain_mf | Elastic strain tensor (matricial form)
#   strain_mf   | Total strain tensor (matricial form)
#   is_su_fail  | State update failure flag
#
def init(problem_dict):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    problem_type = problem_dict['problem_type']
    # Define constitutive model state variables (names and initialization)
    state_variables_init = dict()
    state_variables_init['e_strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['strain_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['stress_mf'] = \
                        top.setTensorMatricialForm(np.zeros((n_dim,n_dim)),n_dim,comp_order)
    state_variables_init['is_su_fail'] = False
    # Set additional out-of-plane stress component in a 2D plane strain problem (output
    # purpose only)
    if problem_type == 1:
        state_variables_init['stress_33'] = 0.0
    # Return initialized state variables dictionary
    return state_variables_init
#
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def suct(problem_dict,material_properties,mat_phase,inc_strain,state_variables_old):
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get material properties
    E = material_properties[mat_phase]['E']
    v = material_properties[mat_phase]['v']
    # Get last increment converged state variables
    e_strain_old_mf = state_variables_old['e_strain_mf']
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = top.setIdentityTensors(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type in [1,4]:
        # 2D problem (plane strain) / 3D problem
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = top.setTensorMatricialForm(consistent_tangent,n_dim,comp_order)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build incremental strain matricial form
    inc_strain_mf = top.setTensorMatricialForm(inc_strain,n_dim,comp_order)
    # Update elastic strain
    e_strain_mf = e_strain_old_mf + inc_strain_mf
    # Update stress
    stress_mf = np.matmul(consistent_tangent_mf,e_strain_mf)
    # Compute out-of-plane stress component in a 2D plane strain problem (output purpose
    # only)
    if problem_type == 1:
        stress_33 = \
                   lam*(stress_mf[comp_order.index('11')]*stress_mf[comp_order.index('22')])
    # Set state update fail flag
    is_su_fail = False
    # Initialize state variables dictionary
    state_variables = init(problem_dict)
    # Store updated state variables in matricial form
    state_variables['e_strain_mf'] = e_strain_mf
    state_variables['strain_mf'] = e_strain_mf
    state_variables['stress_mf'] = stress_mf
    state_variables['is_su_fail'] = is_su_fail
    if problem_type == 1:
        state_variables['stress_33'] = stress_33
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables,consistent_tangent_mf]
#
#                                                                 Consistent tangent modulus
# ==========================================================================================
# Compute the consistent tangent modulus
def ct(problem_dict,properties):
    # Get problem data
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = top.setIdentityTensors(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type in [1,4]:
        # 2D problem (plane strain) / 3D problem
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = top.setTensorMatricialForm(consistent_tangent,n_dim,comp_order)
    # Return
    return consistent_tangent_mf
#
#                                                               Required material properties
#                                                                    (check input data file)
# ==========================================================================================
# Set the constitutive model required material properties
#
# Material properties meaning:
#
# E - Young modulus
# v - Poisson ratio
#
def setRequiredProperties():
    # Set required material properties
    req_material_properties = ['E','v']
    # Return
    return req_material_properties
