#
# VTK (XML format) Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the output of the data associated to snapshots
# (macroscale loading increments) of the microscale equilibrium problem solution into VTK
# files (XML format). This files contain the data associated to the microstructure (material
# phases, clusters, ...) and to the problem local fields (strain, stress, internal
# variables, ...), allowing the spatial visualization through suitable VTK-reading software
# (e.g. Paraview).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | February 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Parse command-line options and arguments
import sys
# Import from string
import importlib
# Working with arrays
import numpy as np
# Shallow and deep copies
import copy
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Matricial operations
import tensor.matrixoperations as mop
# Links related procedures
import links.material.linksmaterialmodels as LinksMat
#
#                                                                           Global variables
# ==========================================================================================
# Set VTK file indentation
indent = '  '
#
#                                                                  Write clustering VTK file
# ==========================================================================================
# Write VTK file with the clustering discretization
def writevtkclusterfile(vtk_dict, dirs_dict, rg_dict, clst_dict):
    # Get VTK output parameters
    vtk_format = vtk_dict['vtk_format']
    # Get input data file name
    input_file_name = dirs_dict['input_file_name']
    # Get offline stage directory
    offline_stage_dir = dirs_dict['offline_stage_dir']
    # Get the spatial discretization file (regular grid of pixels/voxels)
    regular_grid = rg_dict['regular_grid']
    # Get number of pixels/voxels in each dimension and total number of pixels/voxels
    n_voxels_dims = rg_dict['n_voxels_dims']
    # Get RVE dimensions
    rve_dims = rg_dict['rve_dims']
    # Get RVE clustering discretization
    voxels_clusters = clst_dict['voxels_clusters']
    # Set clustering VTK file name and path
    vtk_cluster_file_name = input_file_name + '_clusters.vti'
    vtk_cluster_file_path = offline_stage_dir + vtk_cluster_file_name
    # Open clustering VTK file (append mode)
    if os.path.isfile(vtk_cluster_file_path):
        os.remove(vtk_cluster_file_path)
    vtk_file = open(vtk_cluster_file_path, 'a')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file header
    vtk_dict['type'] = 'ImageData'
    vtk_dict['version'] = '1.0'
    if sys.byteorder == 'little':
        vtk_dict['byte_order'] = 'LittleEndian'
    else:
        vtk_dict['byte_order'] = 'BigEndian'
    vtk_dict['header_type'] = 'UInt64'
    writevtk_fileheader(vtk_file, vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    dataset_parameters, piece_parameters = setimagedataparam(n_voxels_dims, rve_dims)
    writevtk_opendatasetelem(vtk_file, vtk_dict, dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    writevtk_openpiece(vtk_file, piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    writevtk_opencelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Material phases
    data_list = list(regular_grid.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name': 'Material phase', 'format': vtk_format, 'RangeMin': min_val,
                       'RangeMax': max_val}
    writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Clusters
    data_list = list(voxels_clusters.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name': 'Cluster', 'format': vtk_format, 'RangeMin': min_val,
                       'RangeMax': max_val}
    writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    writevtk_closecelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    writevtk_closepiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    writevtk_closedatasetelem(vtk_file, vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    writevtk_filefooter(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close clustering VTK file
    vtk_file.close()
#
#                                                Write macroscale loading increment VTK file
# ==========================================================================================
# Write VTK file associated to a given macroscale loading increment
def writevtkmacincrement(vtk_dict, dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict,
                         inc, clusters_state):
    # Get VTK output parameters
    vtk_format = vtk_dict['vtk_format']
    vtk_vars = vtk_dict['vtk_vars']
    # Get input data file name
    input_file_name = dirs_dict['input_file_name']
    # Get post processing directory
    postprocess_dir = dirs_dict['postprocess_dir']
    # Set VTK macroscale loading increment file name and path
    vtk_inc_file_name = input_file_name + '_' + str(inc) + '.vti'
    vtk_inc_file_path = postprocess_dir + 'VTK/' + vtk_inc_file_name
    # Open VTK macroscale loading increment file (append mode)
    vtk_file = open(vtk_inc_file_path, 'a')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file header
    vtk_dict['type'] = 'ImageData'
    vtk_dict['version'] = '1.0'
    if sys.byteorder == 'little':
        vtk_dict['byte_order'] = 'LittleEndian'
    else:
        vtk_dict['byte_order'] = 'BigEndian'
    vtk_dict['header_type'] = 'UInt64'
    writevtk_fileheader(vtk_file, vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    if vtk_dict['type'] == 'ImageData':
        # Get regular grid data
        n_voxels_dims = rg_dict['n_voxels_dims']
        rve_dims = rg_dict['rve_dims']
        # Get number of pixels/voxels in each dimension and total number of pixels/voxels
        n_voxels_dims = rg_dict['n_voxels_dims']
        # Get RVE dimensions
        rve_dims = rg_dict['rve_dims']
        # Set VTK dataset element parameters
        dataset_parameters, piece_parameters = setimagedataparam(n_voxels_dims,
                                                                      rve_dims)
    elif vtk_dict['type'] == 'UnstructuredGrid':
        print('Error: VTK UnstructuredGrid dataset element is not implemeted yet!')
    writevtk_opendatasetelem(vtk_file, vtk_dict, dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    writevtk_openpiece(vtk_file, piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    writevtk_opencelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem data
    problem_type = problem_dict['problem_type']
    # Get material data
    material_phases = mat_dict['material_phases']
    material_phases_models = mat_dict['material_phases_models']
    # Get regular grid data
    regular_grid = rg_dict['regular_grid']
    # Get clustering data
    phase_clusters = clst_dict['phase_clusters']
    voxels_clusters = clst_dict['voxels_clusters']
    # Get material constitutive models associated to existent material phases in the
    # microstructure
    material_models = list(np.unique([material_phases_models[mat_phase]['name'] \
                           for mat_phase in material_phases]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Material phases
    data_list = list(regular_grid.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name': 'Material phase', 'format': vtk_format, 'RangeMin': min_val,
                       'RangeMax':max_val}
    writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Clusters
    data_list = list(voxels_clusters.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name': 'Cluster', 'format': vtk_format, 'RangeMin': min_val,
                       'RangeMax': max_val}
    writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set state variables common to all material constitutive models
    if problem_type == 1:
        common_var_list = ['stress_mf', 'stress_33']
    else:
        common_var_list = ['stress_mf']
    # Loop over common state variables
    for var_name in common_var_list:
        # Loop over material constitutive models
        for model_name in material_models:
            # Loop over material phases
            for mat_phase in material_phases:
                if material_phases_models[mat_phase]['name'] == model_name:
                    # Get constitutive model source
                    model_source = material_phases_models[mat_phase]['source']
                    break
            # Get constitutive model initialize procedure according to source
            if model_source == 1:
                # Get constitutive model module
                model_module = importlib.import_module('material.models.' + str(model_name))
                # Get constitutive model initialize procedure
                init_procedure = getattr(model_module, 'init')
            elif model_source == 2:
                # Get constitutive model initialize procedure
                _, init_procedure, _, _, _ = LinksMat.getlinksmodel(model_name)
            # Get constitutive model state variables dictionary
            model_state_variables = init_procedure(problem_dict)
            # Skip state variable output if it is not defined for all constitutive models
            if var_name not in model_state_variables.keys():
                break
        # Get state variable descriptors
        var, var_type, var_n_comp = setoutputvariable(0, problem_dict, var_name,
                                                      model_state_variables)
        # Loop over state variable components
        for icomp in range(var_n_comp):
            # Loop over material constitutive models
            for model_name in material_models:
                # Build state variable cell data array
                rg_array = buildvarcelldataarray(model_name, var_name, var_type, icomp,
                                                 problem_dict, material_phases,
                                                 material_phases_models, phase_clusters,
                                                 voxels_clusters, clusters_state)
            # Set output variable data name
            data_name = setdataname(problem_dict, var_name, var_type, icomp, True)
            # Write VTK cell data array - State variable
            data_list = list(rg_array.flatten('F'))
            min_val = min(data_list)
            max_val = max(data_list)
            data_parameters = {'Name': data_name, 'format': vtk_format, 'RangeMin': min_val,
                               'RangeMax': max_val}
            writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if vtk_vars == 'all':
        # Loop over material constitutive models
        for model_name in material_models:
            # Loop over material phases
            for mat_phase in material_phases:
                if material_phases_models[mat_phase]['name'] == model_name:
                    # Get constitutive model source
                    model_source = material_phases_models[mat_phase]['source']
                    break
            # Get constitutive model initialize procedure according to source
            if model_source == 1:
                # Get constitutive model module
                model_module = importlib.import_module('material.models.' + str(model_name))
                # Get constitutive model initialize procedure
                init_procedure = getattr(model_module, 'init')
            elif model_source == 2:
                # Get constitutive model initialize procedure
                _, init_procedure, _, _, _ = LinksMat.getlinksmodel(model_name)
            # Get constitutive model state variables dictionary
            model_state_variables = init_procedure(problem_dict)
            # Loop over constitutive model state variables
            for var_name in list(set(model_state_variables.keys()) - set(common_var_list)):
                # Get state variable descriptors
                var, var_type, var_n_comp = setoutputvariable(0, problem_dict, var_name,
                                                              model_state_variables)
                # Loop over state variable components
                for icomp in range(var_n_comp):
                    # Build state variable cell data array
                    rg_array = buildvarcelldataarray(model_name, var_name, var_type, icomp,
                                                     problem_dict, material_phases,
                                                     material_phases_models, phase_clusters,
                                                     voxels_clusters, clusters_state)
                    # Set output variable data name
                    data_name = setdataname(problem_dict, var_name, var_type, icomp, False,
                                            model_name)
                    # Write VTK cell data array - State variable
                    data_list = list(rg_array.flatten('F'))
                    min_val = min(data_list)
                    max_val = max(data_list)
                    data_parameters = {'Name': data_name, 'format': vtk_format,
                                       'RangeMin': min_val, 'RangeMax': max_val}
                    writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    writevtk_closecelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    writevtk_closepiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    writevtk_closedatasetelem(vtk_file, vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    writevtk_filefooter(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK file
    vtk_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Add macroscale loading increment VTK file to VTK collection file
    writevtkcollectionfile(input_file_name, postprocess_dir, inc, vtk_inc_file_path)
#
#                                                                  Write VTK collection file
# ==========================================================================================
# Open VTK collection file
def openvtkcollectionfile(input_file_name, postprocess_dir):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file (append mode)
    vtk_file = open(vtk_pvd_file_path, 'a')
    # Set endianness
    if sys.byteorder == 'little':
        byte_order = 'LittleEndian'
    else:
        byte_order = 'BigEndian'
    # Write VTK collection file header
    vtk_file.write('<' + '?xml version="1.0"?' + '>' + '\n')
    vtk_file.write('<' + 'VTKFile type=' + enclose('Collection') + ' ' + \
                   'version=' + enclose('0.1') + ' ' + \
                   'byte_order=' + enclose(byte_order) + '>' + '\n')
    # Open VTK collection element
    vtk_file.write(indent + '<' + 'Collection' + '>' + '\n')
    # Close VTK collection element
    vtk_file.write(indent + '<' + '/Collection' + '>' + '\n')
    # Close VTK collection file
    vtk_file.write('<' + '/VTKFile' + '>' + '\n')
    # Close VTK collection file
    vtk_file.close()
# ------------------------------------------------------------------------------------------
# Add time step VTK file
def writevtkcollectionfile(input_file_name, postprocess_dir, time_step,
                           time_step_file_path):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file and read lines (read)
    file_lines = open(vtk_pvd_file_path, 'r').readlines()
    # Add time step VTK file
    file_lines.insert(-2, 2*indent + '<' + 'DataSet' + ' ' + 'timestep=' +
                      enclose(time_step) + ' ' + 'file=' + enclose(time_step_file_path) +
                      '/>' + '\n')
    # Open VTK collection file (write mode)
    vtk_file = open(vtk_pvd_file_path, 'w')
    # Write updated VTK collection file
    vtk_file.writelines(file_lines)
    # Close VTK collection file
    vtk_file.close()
# ------------------------------------------------------------------------------------------
# Close VTK collection file
def closevtkcollectionfile(input_file_name, postprocess_dir):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file (append mode)
    vtk_file = open(vtk_pvd_file_path,'a')
    # Close VTK collection element
    vtk_file.write(indent + '<' + '/Collection' + '>' + '\n')
    # Close VTK collection file
    vtk_file.write('<' + '/VTKFile' + '>' + '\n')
    # Close VTK collection file
    vtk_file.close()
#
#                                                                    Complementary functions
# ==========================================================================================
# Set ImageData dataset element parameters
def setimagedataparam(n_voxels_dims, rve_dims):
    # Set WholeExtent parameter
    wholeextent = list(copy.deepcopy(n_voxels_dims))
    for i in range(len(wholeextent)):
        wholeextent.insert(2*i, 0)
    # Set Origin parameter
    origin = [0, 0, 0]
    # Set Spacing parameter
    spacing = [rve_dims[i]/n_voxels_dims[i] for i in range(len(rve_dims))]
    # Set null third dimension in 2D problem
    if len(wholeextent) == 4:
        wholeextent = wholeextent + [0, 1]
        spacing.append(0.0)
    # Build ImageData dataset parameters
    dataset_parameters = {'WholeExtent': wholeextent, 'Origin': origin, 'Spacing': spacing}
    # Build ImageData dataset piece parameters
    piece_parameters = {'Extent': wholeextent}
    # Return
    return [dataset_parameters, piece_parameters]
# ------------------------------------------------------------------------------------------
# Enclose input in literal quotation marks
def enclose(x):
    if isinstance(x, str):
        return '\'' + x + '\''
    elif isinstance(x, list):
        return '\'' + ' '.join(str(i) for i in x) + '\''
    else:
        return '\'' + str(x) + '\''
#
#                                                                     Write VTK XML elements
# ==========================================================================================
# Write VTK file header and footer
def writevtk_fileheader(vtk_file, vtk_dict):
    vtk_file.write('<' + '?xml version="1.0"?' + '>' + '\n')
    vtk_file.write('<' + 'VTKFile type=' + enclose(vtk_dict['type']) + ' ' + \
                   'version=' + enclose(vtk_dict['version']) + ' ' + \
                   'byte_order=' + enclose(vtk_dict['byte_order']) + ' ' + \
                   'header_type=' + enclose(vtk_dict['header_type']) + '>' + '\n')
def writevtk_filefooter(vtk_file):
    vtk_file.write('<' + '/VTKFile' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK dataset element
def writevtk_opendatasetelem(vtk_file, vtk_dict, dataset_parameters):
    parameters = copy.deepcopy(dataset_parameters)
    vtk_file.write(indent + '<' + vtk_dict['type'] + ' ' + \
                   ' '.join([ key + '=' + enclose(parameters[key]) \
                   for key in parameters]) + '>' + '\n')
def writevtk_closedatasetelem(vtk_file, vtk_dict):
    vtk_file.write(indent + '<' + '/' + vtk_dict['type'] + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK dataset element piece
def writevtk_openpiece(vtk_file, piece_parameters):
    parameters = copy.deepcopy(piece_parameters)
    vtk_file.write(2*indent + '<' + 'Piece' + ' ' + \
                   ' '.join([ key + '=' + enclose(parameters[key]) \
                   for key in parameters]) + '>' + '\n')
def writevtk_closepiece(vtk_file):
    vtk_file.write(2*indent + '</Piece>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK point data element
def writevtk_openpointdata(vtk_file):
    vtk_file.write(3*indent + '<' + 'PointData' + '>' + '\n')
def writevtk_closepointdata(vtk_file):
    vtk_file.write(3*indent + '<' + '/PointData' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK cell data element
def writevtk_opencelldata(vtk_file):
    vtk_file.write(3*indent + '<' + 'CellData' + '>' + '\n')
def writevtk_closecelldata(vtk_file):
    vtk_file.write(3*indent + '<' + '/CellData' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK cell data array
def writevtk_celldataarray(vtk_file, vtk_dict, data_list, data_parameters):
    # Set cell data array data type and associated ascii format
    if all(isinstance(x, int) or isinstance(x, np.integer) for x in data_list):
        data_type = 'Int32'
        frmt = 'd'
    elif all('bool' in str(type(x)).lower() for x in data_list):
        data_type = 'Int32'
        frmt = 'd'
    else:
        if vtk_dict['vtk_precision'] == 'SinglePrecision':
            data_type = 'Float32'
            frmt = '16.8e'
        elif vtk_dict['vtk_precision'] == 'DoublePrecision':
            data_type = 'Float64'
            frmt = '25.16e'
    # Set cell data array parameters
    parameters = dict()
    values = data_list
    if 'Name' in data_parameters.keys():
        parameters['Name'] = data_parameters['Name']
    else:
        parameters['Name'] = '?'
    parameters['type'] = data_type
    if 'NumberofComponents' in data_parameters.keys():
        parameters['NumberofComponents'] = data_parameters['NumberofComponents']
    if 'format' in data_parameters.keys():
        parameters['format'] = data_parameters['format']
    else:
        print('error: missing format')
    if 'RangeMin' in data_parameters.keys():
        parameters['RangeMin'] = data_parameters['RangeMin']
        min_val = data_parameters['RangeMin']
        # Mask data values according to specified range lower bound
        values = list(np.where(np.array(values) < min_val, min_val, np.array(values)))
    if 'RangeMax' in data_parameters.keys():
        parameters['RangeMax'] = data_parameters['RangeMax']
        max_val = data_parameters['RangeMax']
        # Mask data values according to specified range upper bound
        values = list(np.where(np.array(values) > max_val, max_val, np.array(values)))
    # Write VTK data array header
    vtk_file.write(4*indent + '<' + 'DataArray' + ' ' + \
                   ' '.join([ key + '=' + enclose(parameters[key]) \
                   for key in parameters]) + '>' + '\n')
    # Write VTK data array values
    n_line_vals = 6
    template1 = 5*indent + n_line_vals*('{: ' + frmt + '}') + '\n'
    template2 = 5*indent + (len(values) % n_line_vals)*('{: ' + frmt + '}') + '\n'
    aux_list = [values[i:i+n_line_vals] for i in range(0, len(values),n_line_vals)]
    for i in range(len(aux_list)):
        if i == len(aux_list) - 1 and len(values) % n_line_vals != 0:
            vtk_file.write(template2.format(*aux_list[i]))
        else:
            vtk_file.write(template1.format(*aux_list[i]))
    # Write VTK data array footer
    vtk_file.write(4*indent + '<' + '/DataArray' + '>' + '\n')
#
#                                                                    Complementary functions
# ==========================================================================================
# Set state variable descriptors and format required to output associated cell data array
def setoutputvariable(mode, problem_dict, var_name, *args):
    # Get problem data
    n_dim = problem_dict['n_dim']
    comp_order_sym = problem_dict['comp_order_sym']
    comp_order_nsym = problem_dict['comp_order_nsym']
    # Get stored state variable for output
    if mode == 0:
        model_state_variables = args[0]
        stored_var = model_state_variables[var_name]
    else:
        cluster = args[0]
        clusters_state = args[1]
        stored_var = clusters_state[str(cluster)][var_name]
    # Set output state variable descriptors
    if isinstance(stored_var, int) or isinstance(stored_var, np.integer):
        var_type = 'int'
        var_n_comp = 1
        var = stored_var
    elif isinstance(stored_var, float) or isinstance(stored_var, np.float):
        var_type = 'float'
        var_n_comp = 1
        var = stored_var
    elif isinstance(stored_var, bool):
        var_type = 'bool'
        var_n_comp = 1
        var = stored_var
    elif isinstance(stored_var, np.ndarray) and len(stored_var.shape) == 1:
        if var_name.split('_')[-1] == 'mf':
            if len(stored_var) == len(comp_order_sym):
                var_type = 'sym_matrix_mf'
                var_n_comp = len(comp_order_sym)
                var = mop.gettensorfrommf(stored_var, n_dim, comp_order_sym)
            else:
                var_type = 'nsym_matrix_mf'
                var_n_comp = len(comp_order_nsym)
                var = mop.gettensorfrommf(stored_var, n_dim, comp_order_nsym)
    else:
        var_type = 'matrix'
        var_n_comp = len(comp_order_nsym)
        var = stored_var
    # Return
    return [var, var_type, var_n_comp]
# ------------------------------------------------------------------------------------------
# Build given constitutive model' state variable cell data array
def buildvarcelldataarray(model_name, var_name, var_type, icomp, problem_dict,
                          material_phases, material_phases_models, phase_clusters,
                          voxels_clusters, clusters_state):
    # Get problem data
    comp_order_sym = problem_dict['comp_order_sym']
    comp_order_nsym = problem_dict['comp_order_nsym']
    # Initialize regular grid shape array
    rg_array = copy.deepcopy(voxels_clusters)
    rg_array = rg_array.astype(str)
    rg_array = rg_array.astype(object)
    # Loop over material phases
    for mat_phase in material_phases:
        if material_phases_models[mat_phase]['name'] == model_name:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Get material cluster state variable
                cluster_var, _, _ = setoutputvariable(1, problem_dict, var_name, cluster,
                                                      clusters_state)
                # Assemble material cluster state variable
                if var_type in ['int', 'bool', 'float']:
                    rg_array = np.where(rg_array == str(cluster), cluster_var, rg_array)
                elif var_type == 'vector':
                    rg_array = np.where(rg_array == str(cluster), cluster_var[icomp],
                                        rg_array)
                elif var_type == 'sym_matrix_mf':
                    idx = tuple([int(x) - 1 for x in comp_order_sym[icomp]])
                    rg_array = np.where(rg_array == str(cluster), cluster_var[idx],
                                        rg_array)
                elif var_type == 'nsym_matrix_mf':
                    idx = tuple([int(x) - 1 for x in comp_order_nsym[icomp]])
                    rg_array = np.where(rg_array == str(cluster), cluster_var[idx],
                                        rg_array)
                else:
                    idx = tuple([int(x) - 1 for x in comp_order_nsym[icomp]])
                    rg_array = np.where(rg_array == str(cluster), cluster_var[idx],
                                        rg_array)
        else:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Assemble a default state variable value for all clusters for which the
                # associated material phase is not governed by the given material
                # constitutive model (required for valid output cell data array)
                if var_type in ['int']:
                    rg_array = np.where(rg_array == str(cluster), int(0), rg_array)
                elif var_type in ['bool']:
                    rg_array = np.where(rg_array == str(cluster), False, rg_array)
                else:
                    rg_array = np.where(rg_array == str(cluster), float(0), rg_array)
    # Check if the state variable has been specified for every pixels (2D) / voxels (3D)
    # in order to build a valid output cell data array
    if any(isinstance(x, str) for x in list(rg_array.flatten('F'))):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00060', location.filename, location.lineno + 1, var_name,
                            model_name)
    # Return
    return rg_array
# ------------------------------------------------------------------------------------------
# Set state variable cell data array name
def setdataname(problem_dict, var_name, var_type, index, is_common_var, *args):
    # Set cell data array name prefix
    if is_common_var:
        prefix = ''
    else:
        model_name = args[0]
        prefix = model_name + ': '
    # Get problem data
    comp_order_sym = problem_dict['comp_order_sym']
    comp_order_nsym = problem_dict['comp_order_nsym']
    # Set output variable name
    if var_type in ['int', 'bool', 'float']:
        data_name = prefix + var_name
    elif var_type == 'vector':
        data_name = prefix + var_name + '_' + str(index)
    elif var_type == 'sym_matrix_mf':
        data_name = prefix + var_name[:-3] + '_' + comp_order_sym[index]
    elif var_type == 'nsym_matrix_mf':
        data_name = prefix + var_name[:-3] + '_' + comp_order_nsym[index]
    else:
        data_name = prefix + var_name + '_' + comp_order_nsym[index]
    # Return
    return data_name
