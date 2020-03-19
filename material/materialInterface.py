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
#
#                                                                         Material Interface
# ==========================================================================================
# Perform the required constitutive model procedures:
#
#   1. Initialization
#   --------------------------------------
#   Define material constitutive model state variables and build an initialized state
#   variables dictionary
#
#   2. State update and consistent tangent
#   --------------------------------------
#   For a given material state subjected to a given increment of strain, perform the update
#   of the associated material state variables and compute the associated consistent tangent
#   modulus.
#
# Note: The required constitutive model procedures are requested from the source specified
#       for the associated material phase in the input data file
#
def materialInterface(procedure,problem_dict,mat_dict,mat_phase,*args):
    # Get material data
    material_properties = mat_dict['material_properties']
    material_phases_models = mat_dict['material_phases_models']
    # Set constitutive model procedures source
    model_source = material_phases_models[str(mat_phase)]['source']
    #
    #                                                    UNNAMED program material procedures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if model_source == 1:
        #                                                                     Initialization
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if procedure == 'init':
            # Call constitutive model function to perform initialization procedure
            state_variables = material_phases_models[str(mat_phase)]['init'](problem_dict)
            # Return initialized state variables dictionary
            return state_variables
        #
        #                                                State update and Consistent tangent
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        elif procedure == 'suct':
            # Set required arguments to perform the state update procedure and to compute
            # the consistent tangent modulus
            inc_strain = args[0]
            state_variables_old = args[1]
            suct_args = \
                 (problem_dict,material_properties,mat_phase,inc_strain,state_variables_old)
            # Call constitutive model function to perform the state update procedure and to
            # compute the consistent tangent modulus
            state_variables,consistent_tangent_mf = \
                                  material_phases_models[str(mat_phase)]['suct'](*suct_args)
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
#
#                                                     Available material constitutive models
#                                                                    (check input data file)
# ==========================================================================================
# Set the available material constitutive models from a given source
def getAvailableConstitutiveModels(model_source):
    if model_source == 1:
        # UNNAMED program material constitutive models
        available_mat_models = ['linear_elastic',]
    elif model_source == 2:
        # Links material constitutive models
        available_mat_models = ['ELASTIC',]
    elif model_source == 3:
        # Abaqus material constitutive models
        pass
    # Return
    return available_mat_models
