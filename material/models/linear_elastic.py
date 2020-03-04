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
#                                                State update and consistent tangent modulus
# ==========================================================================================
# For a given increment of strain, perform the update of the material state variables and
# compute the associated consistent tangent modulus
def linear_elastic_suct(problem_type,n_dim,comp_order,material_properties,mat_phase,
                                                            inc_strain,state_variables_old):
    # Get material properties
    E = material_properties[mat_phase]['E']
    v = material_properties[mat_phase]['v']
    # Get last increment converged state variables
    e_strain_mf_old = state_variables_old['e_strain_mf']
    #
    #                                                             Consistent tangent modulus
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = top.setIdentityTensors(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type == 1:
        # 2D problem (plane strain)
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    elif problem_type == 4:
        # 3D problem
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = top.setTensorMatricialForm(consistent_tangent,n_dim,comp_order)
    #
    #                                                                           State update
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build incremental strain matricial form
    inc_strain_mf = top.setTensorMatricialForm(inc_strain,n_dim,comp_order)
    # Update elastic strain
    e_strain_mf = e_strain_mf_old + inc_strain_mf
    # Update stress
    stress_mf = np.matmul(De_tensor_mf,e_strain_mf)
    # Set state update fail flag
    is_su_fail = False
    # Store updated state variables in matricial form
    state_variables['e_strain_mf'] = e_strain_mf
    state_variables['stress_mf'] = stress_mf
    state_variables['is_su_fail'] = is_su_fail
    #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return updated state variables and consistent tangent modulus
    return [state_variables,consistent_tangent_mf]
# ------------------------------------------------------------------------------------------
#                                                                 Consistent tangent modulus
# ==========================================================================================
# Compute the consistent tangent modulus (the so called elasticity tensor in the case of
# the linear elastic constitutive model)
def linear_elastic_ct(problem_type,n_dim,comp_order,properties):
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = top.setIdentityTensors(n_dim)
    # Compute consistent tangent modulus according to problem type
    if problem_type == 1:
        # 2D problem (plane strain)
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    elif problem_type == 4:
        # 3D problem
        consistent_tangent = lam*FODiagTrace + 2.0*miu*FOSym
    # Build consistent tangent modulus matricial form
    consistent_tangent_mf = top.setTensorMatricialForm(consistent_tangent,n_dim,comp_order)
    # Return
    return consistent_tangent_mf
