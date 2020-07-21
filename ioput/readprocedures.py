#
# Reading Procedures Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the read and extraction of data from the user provided input data
# file.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | June 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Import from string
import importlib
# Working with arrays
import numpy as np
# Shallow and deep copy operations
import copy
# Read specific lines from file
import linecache
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Regular expressions
import re
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Links related procedures
import links.material.linksmaterialmodels as LinksMat
# Material interface
import material.materialinterface
# Isotropic hardening laws
import material.isotropichardlaw
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                                           Search functions
# ==========================================================================================
# Find the line number where a given mandatory keyword is specified in a file. Return line
# number if found, otherwise raise error
def searchkeywordline(file, keyword):
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line.split() and line.strip()[0] != '#':
            return line_number
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayerror('E00003', location.filename, location.lineno + 1, keyword)
# ------------------------------------------------------------------------------------------
# Search for a given optional keyword in a file and return the associated line number if
# found
def searchoptkeywordline(file, keyword):
    is_found = False
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line.split() and line.strip()[0] != '#':
            is_found = True
            return [is_found,line_number]
    return [is_found,line_number]
#
#                                                                             Keyword type A
# ==========================================================================================
# Read a keyword of type A specification, characterized as follows:
#
# < keyword > < int >
#
def readtypeAkeyword(file, file_path, keyword, max):
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number).split()
    if len(line) == 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00007', location.filename, location.lineno + 1, keyword)
    elif not ioutil.checkposint(line[1]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00007', location.filename, location.lineno + 1, keyword)
    elif isinstance(max, int) or isinstance(max, np.integer):
        if int(line[1]) > max:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00007',location.filename,location.lineno + 1, keyword)
    return int(line[1])
#
#                                                                             Keyword type B
# ==========================================================================================
# Read a keyword of type B specification, characterized as follows:
#
# < keyword >
# < float >
#
def readtypeBkeyword(file, file_path, keyword):
    keyword_line_number = searchkeywordline(file,keyword)
    line = linecache.getline(file_path,keyword_line_number+1).split()
    if line == '':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00004', location.filename, location.lineno + 1, keyword)
    elif len(line) != 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00004', location.filename, location.lineno + 1, keyword)
    elif not ioutil.checknumber(line[0]) or float(line[0]) <= 0:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00004', location.filename, location.lineno + 1, keyword)
    return float(line[0])
#
#                                                                        Material properties
# ==========================================================================================
# Read the number of material phases and associated properties of the general heterogeneous
# material specified as follows:
#
# Material_Phases < n_material_phases >
# < phase_id > < model_name > < n_properties > [ < model_source > ]
# < property1_name > < value >
# < property2_name > < value >
# < phase_id > < model_name > < n_properties > [ < model_source > ]
# < property1_name > < value >
# < property2_name > < value >
#
# In the case of a constitutive model that involves a given strain hardening law, then one
# of the properties specifies the type of hardening law and the associated parameters. In
# the particular case of a piecewise linear isotropic hardening law, a list of the
# hardening curve points must be provided below the property main specification.
#
def readmaterialproperties(file, file_path, keyword):
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number).split()
    if len(line) == 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00052', location.filename, location.lineno + 1, keyword)
    elif not ioutil.checkposint(line[1]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00052', location.filename, location.lineno + 1, keyword)
    # Set number of material phases
    n_material_phases = int(line[1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize material phases properties and constitutive models dictionaries
    material_properties = dict()
    material_phases_models = dict()
    # Loop over material phases
    line_number = keyword_line_number + 1
    for i in range(n_material_phases):
        phase_header = linecache.getline(file_path, line_number).split()
        if phase_header[0] == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00005', location.filename, location.lineno + 1, keyword,
                                i + 1)
        elif len(phase_header) not in [3, 4]:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00005', location.filename, location.lineno + 1, keyword,
                                i + 1)
        elif not ioutil.checkposint(phase_header[0]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00005', location.filename, location.lineno + 1, keyword,
                                i + 1)
        elif phase_header[0] in material_properties.keys():
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00005', location.filename, location.lineno + 1, keyword,
                                i + 1)
        # Set material phase
        mat_phase = str(phase_header[0])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material phase constitutive model source
        material_phases_models[mat_phase] = dict()
        if len(phase_header) == 3:
            # If the material phase constitutive model source has not been specified, then
            # assume CRATE material procedures by default
            model_source = 1
            material_phases_models[mat_phase]['source'] = model_source
        elif len(phase_header) == 4:
            # Set constitutive model source
            if not ioutil.checkposint(phase_header[3]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00053', location.filename, location.lineno + 1,
                                    keyword, mat_phase)
            elif int(phase_header[3]) not in [1, 2, 3]:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00053', location.filename, location.lineno + 1,
                                    keyword, mat_phase)
            model_source = int(phase_header[3])
            material_phases_models[mat_phase]['source'] = model_source
            # Model source 3 is not implemented yet...
            if model_source in [3,]:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00054', location.filename, location.lineno + 1,
                                    mat_phase)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material phase constitutive model and associated procedures
        available_mat_models = \
            material.materialinterface.getavailablematmodels(model_source)
        if phase_header[1] not in available_mat_models:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00055', location.filename, location.lineno + 1,
                                 mat_phase, model_source)
        model_name = phase_header[1]
        material_phases_models[mat_phase]['name'] = model_name
        if model_source == 1:
            model_module = importlib.import_module('material.models.' + str(model_name))
            # Set material constitutive model required procedures
            req_procedures = ['getrequiredproperties', 'init', 'suct']
            # Check if the material constitutive model required procedures are available
            for procedure in req_procedures:
                if hasattr(model_module, procedure):
                    # Get material constitutive model procedures
                    material_phases_models[mat_phase][procedure] = getattr(model_module,
                                                                           procedure)
                else:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00056', location.filename, location.lineno + 1,
                                        model_name,procedure)
        elif model_source == 2:
            # Get material constitutive model procedures
            getrequiredproperties, init, writematproperties, linksxprops, \
                linksxxxxva = LinksMat.getlinksmodel(model_name)
            material_phases_models[mat_phase]['getrequiredproperties'] = \
                getrequiredproperties
            material_phases_models[mat_phase]['init'] = init
            material_phases_models[mat_phase]['writematproperties'] = \
                writematproperties
            material_phases_models[mat_phase]['linksxprops'] = linksxprops
            material_phases_models[mat_phase]['linksxxxxva'] = linksxxxxva
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set number of material properties
        required_properties = material_phases_models[mat_phase]['getrequiredproperties']()
        n_required_properties = len(required_properties)
        if not ioutil.checkposint(phase_header[2]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00005', location.filename, location.lineno + 1, keyword,
                                i + 1)
        elif int(phase_header[2]) != n_required_properties:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00058', location.filename, location.lineno + 1,
                                int(phase_header[2]), mat_phase, n_required_properties)
        n_properties = int(phase_header[2])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set material properties
        material_properties[mat_phase] = dict()
        property_header_line = line_number
        for j in range(n_properties):
            property_header_line = property_header_line + 1
            property_line = linecache.getline(file_path, property_header_line).split()
            if property_line[0] == '':
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00006', location.filename, location.lineno + 1,
                                    keyword, j + 1, mat_phase)
            elif not ioutil.checkvalidname(property_line[0]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00006', location.filename, location.lineno + 1,
                                    keyword, j + 1, mat_phase)
            elif property_line[0] not in required_properties:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00059', location.filename, location.lineno + 1,
                                    property_line[0], mat_phase)
            elif property_line[0] in material_properties[mat_phase].keys():
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00006', location.filename, location.lineno + 1,
                                    keyword, j + 1, mat_phase)
            # Read material property or hardening law
            if str(property_line[0]) == 'IHL':
                # Get available isotropic hardening types
                if model_source == 1:
                    available_hardening_types = \
                        material.isotropichardlaw.getavailabletypes()
                elif model_source == 2:
                    available_hardening_types = ['piecewise_linear']
                # Check if specified isotropic hardening type is available
                if property_line[1] not in available_hardening_types:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00074', location.filename, location.lineno + 1,
                                        property_line[1], mat_phase,
                                        available_hardening_types)
                else:
                    hardening_type = str(property_line[1])
                # Get parameters required by isotropic hardening type
                req_hardening_parameters = \
                    material.isotropichardlaw.setrequiredparam(hardening_type)
                # Check if the required number of hardening parameters is specified
                if len(property_line[2:]) != len(req_hardening_parameters):
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00075', location.filename, location.lineno + 1,
                                        len(property_line[2:]), mat_phase,hardening_type,
                                        len(req_hardening_parameters),
                                        req_hardening_parameters)
                # Initialize material phase hardening parameters dictionary
                hardening_parameters = dict()
                # Read hardening parameters
                if hardening_type == 'piecewise_linear':
                    # Read number of hardening curve points
                    if not ioutil.checkposint(property_line[2]) and \
                        int(property_line[2]) < 2:
                        location = inspect.getframeinfo(inspect.currentframe())
                        errors.displayerror('E00076', location.filename,
                                            location.lineno + 1, mat_phase, hardening_type)
                    else:
                        n_hardening_points = int(property_line[2])
                    # Read hardening curve points
                    hardening_points = np.zeros((n_hardening_points, 2))
                    for k in range(n_hardening_points):
                        hardening_point_line = linecache.getline(
                            file_path, property_header_line + 1 + k).split()
                        if hardening_point_line[0] == '':
                            location = inspect.getframeinfo(inspect.currentframe())
                            errors.displayerror('E00077', location.filename,
                                                location.lineno + 1, k + 1, mat_phase)
                        elif len(hardening_point_line) != 2:
                            location = inspect.getframeinfo(inspect.currentframe())
                            errors.displayerror('E00077', location.filename,
                                                location.lineno + 1, k + 1, mat_phase)
                        elif not ioutil.checknumber(hardening_point_line[0]) or \
                                not ioutil.checknumber(hardening_point_line[1]):
                            location = inspect.getframeinfo(inspect.currentframe())
                            errors.displayerror('E00077', location.filename,
                                                location.lineno + 1, k + 1, mat_phase)
                        hardening_points[k, 0] = float(hardening_point_line[0])
                        hardening_points[k, 1] = float(hardening_point_line[1])
                    # Assemble hardening parameters
                    hardening_parameters['n_hardening_points'] = n_hardening_points
                    hardening_parameters['hardening_points'] = hardening_points
                    # Fix line numbers by adding the number of hardening curve points
                    property_header_line = property_header_line + n_hardening_points
                    line_number = line_number + n_hardening_points
                else:
                    # Loop over and assemble required hardening parameters
                    for k in range(len(req_hardening_parameters)):
                        if not ioutil.checknumber(property_line[2 + k]):
                            location = inspect.getframeinfo(inspect.currentframe())
                            errors.displayerror('E00078', location.filename,
                                                location.lineno + 1, k + 1, mat_phase)
                        else:
                            hardening_parameters[str(req_hardening_parameters[k])] = \
                                float(property_line[2 + k])
                # Get material phase hardening law
                hardeningLaw = material.isotropichardlaw.gethardeninglaw(
                    hardening_type)
                # Store material phase hardening law and parameters
                material_properties[mat_phase]['hardeningLaw'] = hardeningLaw
                material_properties[mat_phase]['hardening_parameters'] = \
                    hardening_parameters
            else:
                if len(property_line) != 2:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00006', location.filename, location.lineno + 1,
                                        keyword, j + 1, mat_phase)
                elif not ioutil.checknumber(property_line[1]):
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00006', location.filename, location.lineno + 1,
                                        keyword, j + 1, mat_phase)
                prop_name = str(property_line[0])
                prop_value = float(property_line[1])
                material_properties[mat_phase][prop_name] = prop_value
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set Links constitutive model required integer and real material properties arrays
        if model_source == 2:
            iprops,rprops = linksxprops(material_properties[mat_phase])
            material_properties[mat_phase]['iprops'] = iprops
            material_properties[mat_phase]['rprops'] = rprops
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Skip to the next material phase
        line_number = line_number + n_properties + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [n_material_phases, material_phases_models, material_properties]
#
#                                                             Macroscale loading constraints
# ==========================================================================================
# Read the macroscale loading constraints, specified as
#
#   2D Problem
#   -------------------------------------------------------------------------------------
#   Macroscale_Strain or Macroscale_Stress < int >     |     Mixed_Prescription_Index (*)
#   < component_name_11 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_21 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_12 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_22 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#
#   3D Problem
#   -------------------------------------------------------------------------------------
#   Macroscale_Strain or Macroscale_Stress < int >     |     Mixed_Prescription_Index (*)
#   < component_name_11 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_21 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_31 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_12 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_22 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_32 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_13 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_23 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#   < component_name_33 > < float > < float >  ...     |     < 0 or 1 > < 0 or 1 >  ...
#
#   (*) Only required if prescribed macroscale strains and stresses
#
# and store them in a dictionary as
#                               _                                      _
#                              |'component_name_11', float, float, ... |
#            dictionary[key] = |'component_name_21', float, float, ... | ,
#                              |_       ...        , float, float, ..._|
#
# where key in ['strain','stress'], and in an array as
#                               _                           _
#                              | < 0 or 1 >, < 0 or 1 >, ... |
#                      array = | < 0 or 1 >, < 0 or 1 >, ... |
#                              |_   ...    ,    ...    , ..._|
#
# Note: The macroscale strain or stress tensor is always assumed to be nonsymmetric and
#       all components must be specified in columnwise order
#
# Note: The symmetry of the macroscale strain and stress tensors is verified under an
#       infinitesimal strain formulation
#
# Note: Both strain/stress tensor dictionaries and prescription array are then reordered
#       according to the program assumed nonsymmetric component order
#
def readmacroscaleloading(file, file_path, mac_load_type, strain_formulation, n_dim,
                          comp_order_nsym):
    # Set macroscale loading keywords according to loading type
    if mac_load_type == 1:
        loading_keywords = {'Macroscale_Strain': 'strain'}
    elif mac_load_type == 2:
        loading_keywords = {'Macroscale_Stress': 'stress'}
    elif mac_load_type == 3:
        loading_keywords = {'Macroscale_Strain': 'strain', 'Macroscale_Stress': 'stress'}
        presc_keyword = 'Mixed_Prescription_Index'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize macroscale loading dictionary
    mac_load = {key: None for key in ['strain', 'stress']}
    # Initialize number of macroscale loading subpaths dictionary
    n_load_subpaths = {key: 0 for key in ['strain', 'stress']}
    # Loop over macroscale loading keywords
    for load_key in loading_keywords.keys():
        # Get load nature type
        ltype = loading_keywords[load_key]
        # Get macroscale loading keyword line number
        load_keyword_line_number = searchkeywordline(file, load_key)
        # Check number of loading subpaths
        keyword_line = linecache.getline(file_path, load_keyword_line_number).split()
        if len(keyword_line) > 2:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00088', location.filename, location.lineno + 1, load_key)
        elif len(keyword_line) == 2:
            if ioutil.checkposint(keyword_line[1]):
                n_load_subpaths[ltype] = int(keyword_line[1])
            else:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00088', location.filename, location.lineno + 1,
                                    load_key)
        else:
            n_load_subpaths[ltype] = 1
        # Initialize macroscale loading array
        mac_load[ltype] = np.full((n_dim**2, 1 + n_load_subpaths[ltype]), 0.0, dtype=object)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale loading
    for load_key in loading_keywords:
        # Get load nature type
        ltype = loading_keywords[load_key]
        load_keyword_line_number = searchkeywordline(file, load_key)
        # Loop over macroscale loading components
        for i_comp in range(n_dim**2):
            component_line = linecache.getline(
                file_path, load_keyword_line_number + i_comp + 1).split()
            if not component_line:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00008', location.filename, location.lineno + 1,
                                    load_key, i_comp + 1)
            elif len(component_line) != 1 + n_load_subpaths[ltype]:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00008', location.filename, location.lineno + 1,
                                    load_key, i_comp + 1)
            elif not ioutil.checkvalidname(component_line[0]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00008', location.filename, location.lineno + 1,
                                    load_key, i_comp + 1)
            # Set component name
            mac_load[ltype][i_comp, 0] = component_line[0]
            # Set component values for each loading subpath
            for j in range(n_load_subpaths[ltype]):
                presc_val = component_line[1 + j]
                if not ioutil.checknumber(presc_val):
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00008', location.filename, location.lineno + 1,
                                        load_key, i_comp + 1)
                else:
                    mac_load[ltype][i_comp, 1 + j] = float(presc_val)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale prescription nature indexes
    if mac_load_type == 1:
        ltype = loading_keywords['Macroscale_Strain']
        mac_load_presctype = np.full((n_dim**2, n_load_subpaths[ltype]), 'strain',
                                     dtype=object)
    elif mac_load_type == 2:
        ltype = loading_keywords['Macroscale_Stress']
        mac_load_presctype = np.full((n_dim**2, n_load_subpaths[ltype]), 'stress',
                                     dtype=object)
    elif mac_load_type == 3:
        mac_load_presctype = np.full((n_dim**2, max(n_load_subpaths.values())), 'ND',
                                      dtype=object)
        presc_keyword = 'Mixed_Prescription_Index'
        presc_keyword_line_number = searchkeywordline(file, presc_keyword)
        # Loop over macroscale loading components
        for i_comp in range(n_dim**2):
            component_line = linecache.getline(
                file_path, presc_keyword_line_number + i_comp + 1).split()
            if not component_line:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00011', location.filename, location.lineno + 1,
                                    presc_keyword, i_comp + 1)
            # Set prescription nature indexes for each loading subpath
            for j in range(max(n_load_subpaths.values())):
                presc_val = int(component_line[j])
                if presc_val not in [0, 1]:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00011', location.filename, location.lineno + 1,
                                        presc_keyword, i_comp + 1)
                else:
                    ltype = 'strain' if presc_val == 0 else 'stress'
                    if j >= n_load_subpaths[ltype]:
                        location = inspect.getframeinfo(inspect.currentframe())
                        errors.displayerror('E00011', location.filename,
                                            location.lineno + 1, presc_keyword, i_comp + 1)
                    else:
                        mac_load_presctype[i_comp, j] = ltype
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check small strain formulation symmetry
    if strain_formulation == 1:
        # Set symmetric indexes (columnwise)
        if n_dim**2 == 4:
            symmetric_indexes = np.array([[2], [1]])
        elif n_dim**2 == 9:
            symmetric_indexes = np.array([[3, 6, 7], [1, 2, 5]])
        # Loop over symmetric indexes
        for i in range(symmetric_indexes.shape[1]):
            # Loop over loading subpaths
            for j in range(max(n_load_subpaths.values())):
                # Get load nature type
                ltype = mac_load_presctype[symmetric_indexes[0, i], j]
                if mac_load_type == 3 and \
                        mac_load_presctype[symmetric_indexes[0, i] , j] != \
                        mac_load_presctype[symmetric_indexes[1, i], j]:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayerror('E00012', location.filename, location.lineno + 1, i)
                # Check symmetry
                isEqual = np.allclose(
                      mac_load[ltype][symmetric_indexes[0, i], j + 1],
                      mac_load[ltype][symmetric_indexes[1, i], j + 1], atol=1e-10)
                if not isEqual:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displaywarning('W00001', location.filename,
                                          location.lineno + 1, ltype)
                    # Adopt symmetric component with the lowest first index
                    mac_load[ltype][symmetric_indexes[1, i], j + 1] = \
                        mac_load[ltype][symmetric_indexes[0, i], j + 1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort macroscale strain and stress tensors according to the defined problem
    # nonsymmetric component order
    if n_dim == 2:
        aux = {'11': 0, '21': 1, '12': 2, '22': 3}
    else:
        aux = {'11': 0, '21': 1, '31': 2, '12': 3, '22': 4, '32': 5, '13': 6, '23': 7,
               '33': 8}
    mac_load_copy = copy.deepcopy(mac_load)
    mac_load_presctype_copy = copy.deepcopy(mac_load_presctype)
    for i in range(n_dim**2):
        if mac_load_type == 1:
            mac_load['strain'][i, :] = mac_load_copy['strain'][aux[comp_order_nsym[i]], :]
        elif mac_load_type == 2:
            mac_load['stress'][i, :] = mac_load_copy['stress'][aux[comp_order_nsym[i]], :]
        elif mac_load_type == 3:
            mac_load['strain'][i, :] = mac_load_copy['strain'][aux[comp_order_nsym[i]], :]
            mac_load['stress'][i, :] = mac_load_copy['stress'][aux[comp_order_nsym[i]], :]
            mac_load_presctype[i, :] = mac_load_presctype_copy[aux[comp_order_nsym[i]], :]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return mac_load, mac_load_presctype
#
#                                                          Macroscale loading incrementation
# ==========================================================================================
# Read the macroscale loading incrementation, specified as
#
# Number_of_Load_Increments < int >
#
# or as
#
# Increment_List
# [< int >:] < float > [_< float >] | [< int >:] < float > [_< float >] | ...
# [< int >:] < float > [_< float >] | [< int >:] < float > [_< float >] | ...
#                                   | [< int >:] < float > [_< float >] | ...
#
# and store it in a dictionary as
#
#                              incremental
#                                 time
#                         _         v  _
#                        | float, float |
#      dictionary[key] = | float, float | , where key is the loading subpath index.
#                        |_float, float_|
#                            ^
#                       incremental
#                       load factor
#
# The optional keyword associated to the loading time factor may also be specified as
#
# Loading_Time_Factor
# < float >
#
def readmacloadincrem(file, file_path, keyword, n_load_subpaths):
    # Initialize macroscale loading incrementation dictionary
    mac_load_increm = dict()
    # Set load time factor
    keyword_time = 'Loading_Time_Factor'
    is_found, _ = searchoptkeywordline(file, keyword_time)
    if is_found:
        load_time_factor = readtypeBkeyword(file, file_path, keyword_time)
    else:
        load_time_factor = 1.0
    # Set macroscale loading incrementation
    if keyword == 'Number_of_Load_Increments':
        max = '~'
        n_load_increments = readtypeAkeyword(file, file_path, keyword, max)
        # Build macroscale loading incrementation dictionary
        for i in range(n_load_subpaths):
            # Set loading subpath default total load factor
            total_lfact = 1.0
            # Build macroscale loading subpath
            load_subpath = np.zeros((n_load_increments, 2))
            load_subpath[:, 0] = total_lfact/n_load_increments
            load_subpath[:, 1] = load_time_factor*load_subpath[:, 0]
            # Store macroscale loading subpath
            mac_load_increm[str(i)] = load_subpath
    elif keyword == 'Increment_List':
        # Find keyword line number
        keyword_line_number = searchkeywordline(file, keyword)
        # Initialize macroscale loading increment array
        increm_list = np.full((0, n_load_subpaths), '', dtype=object)
        # Read increment specification line
        line = linecache.getline(file_path, keyword_line_number + 1)
        increm_line = [x.strip() for x in line.split('|')]
        # At least one increment specification line must be provided for each macroscale
        # loading subpath
        is_empty_line = not bool(line.split())
        if is_empty_line or len(increm_line) != n_load_subpaths:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00092', location.filename, location.lineno + 1)
        i = 0
        # Build macroscale loading increment array
        while not is_empty_line:
            increm_list = np.append(increm_list,
                                    np.full((1, n_load_subpaths), '', dtype=object), axis=0)
            # Assemble macroscale increment specification line
            increm_list[i, 0:len(increm_line)] = increm_line
            i += 1
            # Read increment specification line
            line = linecache.getline(file_path, keyword_line_number + 1 + i)
            is_empty_line = not bool(line.split())
            increm_line = [x.strip() for x in line.split('|')]
        # Build macroscale loading incrementation dictionary
        for j in range(n_load_subpaths):
            # Initialize macroscale loading subpath
            load_subpath = np.zeros((0, 2))
            # Loop over increment specifications
            for i in range(increm_list.shape[0]):
                # Get increment specification
                spec = increm_list[i, j]
                # Decode increment specification
                if spec == '':
                    break
                else:
                    rep, inc_lfact, inc_time = decodeincremspec(spec, load_time_factor)
                # Build macroscale loading subpath
                load_subpath = np.append(load_subpath,
                                         np.tile([inc_lfact, inc_time], (rep, 1)), axis=0)
            # Store macroscale loading subpath
            mac_load_increm[str(j)] = load_subpath
    else:
        # Unknown macroscale loading keyword
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00093', location.filename, location.lineno + 1)
    # Return
    return mac_load_increm
# ------------------------------------------------------------------------------------------
# Decode macroscale loading increment specification
def decodeincremspec(spec, load_time_factor):
    # Split specifications based on multiple delimiters
    code = re.split('[:_]', spec)
    # Check if the repetition and incremental time have been specified
    has_rep = ':' in re.findall('[:_]', spec)
    has_time = '_' in re.findall('[:_]', spec)
    if not code or len(code) > 3:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00091', location.filename, location.lineno + 1)
    # Set macroscale loading increment parameters
    try:
        rep = int(code[0]) if has_rep else 1
        inc_lfact = float(code[int(has_rep)])
        inc_time = float(code[-1]) if has_time else load_time_factor*inc_lfact
    except:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00091', location.filename, location.lineno + 1)
    else:
        if any([x < 0 for x in [rep, inc_lfact, inc_time]]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00094', location.filename, location.lineno + 1)
    # Return
    return [rep, inc_lfact, inc_time]
#
#                                                                         Number of clusters
# ==========================================================================================
# Read the number of clusters associated to each material phase, specified as
#
# Number_of_Clusters
# < phase_id > < number_of_clusters >
# < phase_id > < number_of_clusters >
#
def readphaseclustering(file, file_path, keyword, n_material_phases, material_properties):
    phase_n_clusters = dict()
    line_number = searchkeywordline(file, keyword) + 1
    for iphase in range(n_material_phases):
        line = linecache.getline(file_path, line_number + iphase).split()
        if line[0] == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00013', location.filename, location.lineno + 1, keyword,
                                iphase + 1)
        elif len(line) != 2:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00013', location.filename, location.lineno + 1,
                                keyword, iphase + 1)
        elif not ioutil.checkposint(line[0]) or not ioutil.checkposint(line[1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00013', location.filename, location.lineno + 1, keyword,
                                iphase + 1)
        elif str(int(line[0])) not in material_properties.keys():
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00049', location.filename, location.lineno + 1, keyword,
                                int(line[0]), material_properties.keys())
        elif str(int(line[0])) in phase_n_clusters.keys():
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00050', location.filename, location.lineno + 1, keyword,
                                int(line[0]))
        phase_n_clusters[str(int(line[0]))] = int(line[1])
    return phase_n_clusters
#
#                                                                   Discretization file path
# ==========================================================================================
# Read the spatial discretization file path
#
# Discretization_File
# < path >
#
def readdiscretizationfilepath(file, file_path, keyword, valid_exts):
    line_number = searchkeywordline(file, keyword) + 1
    discret_file_path = linecache.getline(file_path, line_number).strip()
    if not os.path.isfile(discret_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00014', location.filename, location.lineno + 1, keyword,
                            discret_file_path)
    format_exts = ['.npy']
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] in format_exts:
        if not ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[0])[-1] \
                in valid_exts:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00015', location.filename, location.lineno + 1, keyword,
                                valid_exts)
    else:
        if not ntpath.splitext(ntpath.basename(discret_file_path))[-1] in valid_exts:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00015', location.filename, location.lineno + 1, keyword,
                                valid_exts)
    return os.path.abspath(discret_file_path)
#
#                                                                             RVE dimensions
# ==========================================================================================
# Read the RVE dimensions specified as follows:
#
# 2D Problems:
#
# RVE_Dimensions
# < dim1_size > < dim2_size >
#
# 3D Problems:
#
# RVE_Dimensions
# < dim1_size > < dim2_size > < dim3_size >
#
def readrvedimensions(file, file_path, keyword, n_dim):
    keyword_line_number = searchkeywordline(file, keyword)
    line = linecache.getline(file_path, keyword_line_number + 1).split()
    if line == '':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00031', location.filename, location.lineno + 1, keyword)
    elif len(line) != n_dim:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00031', location.filename, location.lineno + 1, keyword)
    for i in range(n_dim):
        if not ioutil.checknumber(line[i]) or float(line[i]) <= 0:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00031', location.filename, location.lineno + 1, keyword)
    rve_dims = list()
    for i in range(n_dim):
        rve_dims.append(float(line[i]))
    return rve_dims
#
#                                                                                 VTK output
# ==========================================================================================
# Read the VTK output options specified as follows:
#
# VTK_Output [ < options > ]
#
# where the options are | format:          ascii (default) or binary
#                       | increments:      all (default) or every < positive_integer >
#                       | state variables: all_variables (default) or common_variables
#
def readvtkoptions(file, file_path, keyword, keyword_line_number):
    line = linecache.getline(file_path, keyword_line_number).split()
    line = [x.lower() if not ioutil.checknumber(x) else x for x in line]
    if 'binary' in line:
        vtk_format = 'binary'
    else:
        vtk_format = 'ascii'
    if 'every' in line:
        if not ioutil.checkposint(line[line.index('every') + 1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00057', location.filename, location.lineno + 1, keyword)
        vtk_inc_div = int(line[line.index('every') + 1])
    else:
        vtk_inc_div = 1
    if 'common_variables' in line:
        vtk_vars = 'common_variables'
    else:
        vtk_vars = 'all'
    return [vtk_format, vtk_inc_div, vtk_vars]
