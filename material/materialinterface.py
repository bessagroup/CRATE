#
# Material Interface (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing CRATE's material constitutive models interface, allowing to perform the
# state update and the computation of the consistent tangent modulus of constitutive models
# from different sources.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Import from string
import importlib
# Shallow and deep copy operations
import copy
# Links related procedures
import links.material.linkssuct as linkssuct
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
#   2. State update and consistent tangent modulus
#   ----------------------------------------------
#   For a given material state subjected to a given increment of strain, perform the update
#   of the associated material state variables and compute the associated consistent tangent
#   modulus.
#
# Note: The required constitutive model procedures are requested from the source specified
#       for the associated material phase in the input data file
#
def materialinterface(procedure, problem_dict, mat_dict, clst_dict, algpar_dict, mat_phase,
                      *args):
    # Get material data
    material_properties = mat_dict['material_properties']
    material_phases_models = mat_dict['material_phases_models']
    # Set constitutive model procedures source
    model_source = material_phases_models[str(mat_phase)]['source']
    #
    #                                                              CRATE material procedures
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
            suct_args = (problem_dict, algpar_dict, material_properties, mat_phase,
                         inc_strain, state_variables_old)
            # Call constitutive model function to perform the state update procedure and to
            # compute the consistent tangent modulus
            state_variables, consistent_tangent_mf = \
                material_phases_models[str(mat_phase)]['suct'](*suct_args)
            # Return updated state variables and consistent tangent modulus
            return [state_variables, consistent_tangent_mf]
    #
    #                                                              Links material procedures
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif model_source == 2:
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
            suct_args = (problem_dict, clst_dict, material_properties,
                         material_phases_models, mat_phase, inc_strain, state_variables_old)
            # Call constitutive model function to perform the state update procedure and to
            # compute the consistent tangent modulus
            state_variables, consistent_tangent_mf = linkssuct.suct(*suct_args)
            # Return updated state variables and consistent tangent modulus
            return [state_variables, consistent_tangent_mf]
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
def getavailablematmodels(model_source):
    if model_source == 1:
        # CRATE material constitutive models
        available_mat_models = ['linear_elastic', 'von_mises']
    elif model_source == 2:
        # Links material constitutive models
        available_mat_models = ['ELASTIC', 'VON_MISES']
    elif model_source == 3:
        # Abaqus material constitutive models
        pass
    # Return
    return available_mat_models
#
#                                              Material constitutive model source conversion
#                                                      (Links (source 2) > Crate (source 1))
# ==========================================================================================
def material_source_conversion(n_material_phases, material_phase_models,
                               material_properties):
    '''Convert material-related objects from Links source to CRATE source.

    The material-related objects associated with the material phase constitutive models and
    properties are converted from the Links source (as specified in the input data file) to
    the CRATE source. This allows that the offline stage is performed with Links (FEM
    first-order homogenization) but the following online stage with CRATE material modules.

    Parameters
    ----------
    n_material_phases : int
        Number of material phases.
    material_phase_models : dict
        Material phases constitutive models (item, dict) associated to each material phase
        (key, str).
    material_properties : dict
        Material phases properties (item, dict) associated to each material phase
        (key, str).

    Returns
    -------
    new_material_phase_models : dict
        Material phases constitutive models (item, dict) associated to each material phase
        (key, str) converted to CRATE source.
    new_material_properties : dict
        Material phases properties (item, dict) associated to each material phase
        (key, str) converted to CRATE source.

    Notes
    -----
    The input data file is not changed! This function can be seen as an utility tool to
    perform the offline stage with FEM without compromising efficiency in the online stage.
    '''
    # Initialize material constitutive model and properties dictionaries
    new_material_phase_models = dict()
    new_material_properties = dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get material phases
    mat_phases = [mat_phase for mat_phase in material_phase_models.keys()]
    # Loop over material phases
    for i in range(n_material_phases):
        # Get material phase
        mat_phase = mat_phases[i]
        # Get CRATE corresponding material phase model
        if material_phase_models[mat_phase]['name'] == 'ELASTIC':
            model_name = 'linear_elastic'
        elif material_phase_models[mat_phase]['name'] == 'VON_MISES':
            model_name = 'von_mises'
        # Initialize material phase dictionary
        new_material_phase_models[mat_phase] = dict()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material phase source as CRATE
        new_material_phase_models[mat_phase]['source'] = 1
        # Set material phase name
        new_material_phase_models[mat_phase]['name'] = model_name
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get material constitutive model module
        model_module = importlib.import_module('material.models.' + str(model_name))
        # Set material constitutive model required procedures
        req_procedures = ['getrequiredproperties', 'init', 'suct']
        # Check if the material constitutive model required procedures are available
        for procedure in req_procedures:
            if hasattr(model_module, procedure):
                # Get material constitutive model procedures
                new_material_phase_models[mat_phase][procedure] = getattr(model_module,
                                                                          procedure)
            else:
                raise RuntimeError(
                    'The required {} function is not implemented for '.format(procedure) + \
                    'the constitutive model \'{}\'.'.format(model_module))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize material phase properties dictionary
        new_material_properties[mat_phase] = dict()
        # Get required material properties
        req_material_properties = \
            new_material_phase_models[mat_phase]['getrequiredproperties']()
        # Set new material phase properties
        for property in req_material_properties:
            if property == 'IHL':
                new_material_properties[mat_phase]['hardeningLaw'] = \
                    copy.deepcopy(material_properties[mat_phase]['hardeningLaw'])
                new_material_properties[mat_phase]['hardening_parameters'] = \
                    copy.deepcopy(material_properties[mat_phase]['hardening_parameters'])
            elif property not in material_properties[mat_phase].keys():
                raise RuntimeError('Incompatible material properties between sources.')
            else:
                new_material_properties[mat_phase][property] = \
                    material_properties[mat_phase][property]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [new_material_phase_models, new_material_properties]
