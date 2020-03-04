#
# Material Interface (UNNAMED Program)
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
#
#                                                                         Material Interface
# ==========================================================================================
# For a given material cluster subjected to a given increment of strain, perform the update
# of the associated material state variables and compute the associated consistent tangent
# modulus. The required material constitutive model procedures may be requested from
# different sources:
#
# source = | 0 - UNNAMED program material procedures (default)
#          | 1 - Links material procedures
#          | 2 - Abaqus material procedures#
#
def materialInterface(problem_dict,mat_dict,mat_phase,inc_strain,state_variables_old):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    problem_type = problem_dict['problem_type']
    # Get material data
    material_phases = mat_dict['material_phases']
    material_properties = mat_dict['material_properties']
    material_phases_models = mat_dict['material_phases_models']
    # Set consistutive model procedures source
    model_source = material_phases_models[str(mat_phase)]['source']
    #
    #                                                    UNNAMED program material procedures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if model_source == 1:
        # Set required arguments to perform the state update procedure and to compute the
        # consistent tangent modulus
        suct_args = (problem_type,n_dim,comp_order,material_properties,mat_phase,
                                                             inc_strain,state_variables_old)
        # Call constitutive model function to perform the state update procedure and to
        # compute the consistent tangent modulus
        state_variables,consistent_tangent_mf = \
                          material_phases_models[str(mat_phase)]['suct_function'](suct_args)
        # Return updated state variables and consistent tangent modulus
        return [state_variables,consistent_tangent_mf]
    #
    #                                                              Links material procedures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_source == 2:
        pass
    #
    #                                                             Abaqus material procedures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_source == 3:
        pass
