#
# Write VTK (XML format) Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
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
# Working with arrays
import numpy as np
# Shallow and deep copies
import copy
# Display messages
import info
#
#                                                                           Global variables
# ==========================================================================================
# Set VTK file indentation
indent = '  '
#
#                                                                  Write clustering VTK file
# ==========================================================================================
# Write VTK file with the clustering discretization
def writeVTKClusterFile(vtk_dict,dirs_dict,rg_dict,clst_dict):
    #info.displayInfo('5','Writing cluster VTK file...') # Revert
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
    vtk_file = open(vtk_cluster_file_path,'a')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file header
    vtk_dict['type'] = 'ImageData'
    vtk_dict['version'] = '1.0'
    if sys.byteorder == 'little':
        vtk_dict['byte_order'] = 'LittleEndian'
    else:
        vtk_dict['byte_order'] = 'BigEndian'
    vtk_dict['header_type'] = 'UInt64'
    writeVTKFileHeader(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    dataset_parameters,piece_parameters = setImageDataParameters(n_voxels_dims,rve_dims)
    writeVTKOpenDatasetElement(vtk_file,vtk_dict,dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    writeVTKOpenPiece(vtk_file,piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    writeVTKOpenCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Material phases
    data_list = list(regular_grid.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Material phase','format':'ascii','RangeMin':min_val,
                                                                         'RangeMax':max_val}
    writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # Write VTK cell data array - Clusters
    data_list = list(voxels_clusters.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Cluster','format':'ascii','RangeMin':min_val,
                                                                         'RangeMax':max_val}
    writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    writeVTKCloseCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    writeVTKClosePiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    writeVTKCloseDatasetElement(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    writeVTKFileFooter(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close clustering VTK file
    vtk_file.close()
#
#                                                Write macroscale loading increment VTK file
# ==========================================================================================
# Write VTK file associated to a given macroscale loading increment
def writeVTKMacroLoadIncrement(vtk_dict,dirs_dict,problem_dict,rg_dict,clst_dict):
    #info.displayInfo('5','Writing macroscale loading increment VTK file...') # Revert
    # Get input data file name
    input_file_name = dirs_dict['input_file_name']
    # Get post processing directory
    postprocess_dir = dirs_dict['postprocess_dir']
    # Set VTK macroscale loading increment file name and path
    vtk_inc_file_name = input_file_name + '_' + str(increment) + '.vti'
    vtk_inc_file_path = postprocess_dir + 'VTK/' + vtk_inc_file_name
    # Open VTK macroscale loading increment file (append mode)
    if os.path.isfile(vtk_inc_file_path):
        os.remove(vtk_inc_file_path)
    vtk_file = open(vtk_inc_file_path,'a')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file header
    vtk_dict['type'] = 'ImageData'
    vtk_dict['version'] = '1.0'
    if sys.byteorder == 'little':
        vtk_dict['byte_order'] = 'LittleEndian'
    else:
        vtk_dict['byte_order'] = 'BigEndian'
    vtk_dict['header_type'] = 'UInt64'
    writeVTKFileHeader(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    if vtk_dict['type'] == 'ImageData':
        # Get number of pixels/voxels in each dimension and total number of pixels/voxels
        n_voxels_dims = rg_dict['n_voxels_dims']
        # Get RVE dimensions
        rve_dims = rg_dict['rve_dims']
        # Set VTK dataset element parameters
        dataset_parameters,piece_parameters = setImageDataParameters(n_voxels_dims,rve_dims)
    elif vtk_dict['type'] == 'UnstructuredGrid':
        print('Error: VTK UnstructuredGrid dataset element is not implemeted yet!')
    writeVTKOpenDatasetElement(vtk_file,vtk_dict,dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    writeVTKOpenPiece(vtk_file,piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    writeVTKOpenCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get problem data
    comp_order_sym = problem_dict['comp_order_sym']
    comp_order_nsym = problem_dict['comp_order_nsym']

    # Set output variables common to all material constitutive models
    common_var_list = ['e_strain_mf','strain_mf','stress_mf']
    # Loop over common state variables
    for var_name in common_var_list:
        # Initialize regular grid shape array
        rg_array = copy.deepcopy(voxels_clusters)
        # Loop over material phases
        for mat_phase in material_phases:
            # Loop over material phase clusters
            for cluster in phase_clusters[mat_phase]:
                # Get
                var,var_type = processVar(var_name,cluster,clusters_state,comp_order_sym,comp_order_nsym)
                if var_type in ['int','bool']:
                    np.where(rg_array == cluster,var,rg_array)
                elif var_type == 'vector':
                    for i in range(len(var)):
                        np.where(rg_array == cluster,var[i],rg_array)
                elif var_type == 'sym_matrix':
                    for comp in comp_order_sym:
                        np.where(rg_array == cluster,var[i],rg_array)


        # Write VTK cell data array
        data_list = list(rg_array.flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name':'','format':'ascii','RangeMin':min_val,'RangeMax':max_val}

    def processVar(var_name,cluster,clusters_state,comp_order_sym,comp_order_nsym):
        # Get stored state variable
        stored_var = clusters_state[str(cluster)][var_name]
        # Build
        if isinstance(stored_var,int) or isinstance(stored_var,np.integer):
            var_type = 'int'
            var = stored_var
        elif isinstance(stored_var,bool):
            var_type = 'bool'
            var = stored_var
        elif isinstance(stored_var,numpy.ndarray) and len(stored_var.shape) == 1:
            if var_name.split('_')[-1] == 'mf':
                if len(stored_var) == comp_order_sym:
                    var_type = 'sym_matrix'
                    var = top.getTensorFromMatricialForm(stored_var,n_dim,comp_order_sym)
                else:
                    var_type = 'nsym_matrix'
                    var = top.getTensorFromMatricialForm(stored_var,n_dim,comp_order_nsym)
        else:
            var_type = 'matrix'
            var = stored_var
        # Return
        return [var,var_type]

    kelvinFactor(idx,comp_order)


    # Loop over common output variables


    for var in common_var_list:

        if isinstance()
    # Write VTK cell


    # Write VTK cell data array
    #data_list = list(...)
    #min_val = min(data_list)
    #max_val = max(data_list)
    #data_parameters = {'Name':'?','format':'ascii','NumberofComponents':1,
    #                                                 'RangeMin':min_val,'RangeMax':max_val}
    #writeVTKOpenCellData(vtk_file)
    #writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    #writeVTKCloseCellData(vtk_file)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    writeVTKCloseCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    writeVTKClosePiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    writeVTKCloseDatasetElement(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    writeVTKFileFooter(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close clustering VTK file
    vtk_file.close()
#
#                                                                  Write VTK collection file
# ==========================================================================================
# Open VTK collection file
def openVTKCollectionFile(input_file_name,postprocess_dir):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file (append mode)
    if os.path.isfile(vtk_pvd_file_path):
        os.remove(vtk_pvd_file_path)
    vtk_file = open(vtk_pvd_file_path,'a')
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
# ------------------------------------------------------------------------------------------
def writeVTKCollectionFile(input_file_name,postprocess_dir,time_step,time_step_file_path):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file (append mode)
    vtk_file = open(vtk_pvd_file_path,'a')
    # Add time step VTK file
    vtk_file.write(2*indent + '<' + 'Dataset' + 'time_step=' + enclose(time_step) + ' ' + \
                              'file=' + enclose(time_step_file_path) + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Close VTK collection file
def closeVTKCollectionFile(input_file_name,postprocess_dir):
    # Set VTK collection file name and path
    vtk_pvd_file_name = input_file_name + '.pvd'
    vtk_pvd_file_path = postprocess_dir + vtk_pvd_file_name
    # Open VTK collection file (append mode)
    vtk_file = open(vtk_pvd_file_path,'a')
    # Close VTK collection element
    vtk_file.write(indent + '<' + '/Collection' + '>' + '\n')
    # Close VTK collection file
    vtk_file.write('<' + '/VTKFile' + '>' + '\n')
#
#                                                                    Complementary functions
# ==========================================================================================
# Set ImageData dataset element parameters
def setImageDataParameters(n_voxels_dims,rve_dims):
    # Set WholeExtent parameter
    WholeExtent = list(copy.deepcopy(n_voxels_dims))
    for i in range(len(WholeExtent)):
        WholeExtent.insert(2*i,0)
    # Set Origin parameter
    Origin = [0,0,0]
    # Set Spacing parameter
    Spacing = [rve_dims[i]/n_voxels_dims[i] for i in range(len(rve_dims))]
    # Set null third dimension in 2D problem
    if len(WholeExtent) == 4:
        WholeExtent = WholeExtent + [0,1]
        Spacing.append(0.0)
    # Build ImageData dataset parameters
    dataset_parameters = {'WholeExtent':WholeExtent, 'Origin':Origin, 'Spacing':Spacing}
    # Build ImageData dataset piece parameters
    piece_parameters = {'Extent':WholeExtent}
    # Return
    return [dataset_parameters,piece_parameters]
# ------------------------------------------------------------------------------------------
# Enclose input in literal quotation marks
def enclose(x):
    if isinstance(x,str):
        return '\'' + x + '\''
    elif isinstance(x,list):
        return '\'' + ' '.join(str(i) for i in x) + '\''
    else:
        return '\'' + str(x) + '\''
#
#                                                                     Write VTK XML elements
# ==========================================================================================
# Write VTK file header and footer
def writeVTKFileHeader(vtk_file,vtk_dict):
    vtk_file.write('<' + '?xml version="1.0"?' + '>' + '\n')
    vtk_file.write('<' + 'VTKFile type=' + enclose(vtk_dict['type']) + ' ' + \
                   'version=' + enclose(vtk_dict['version']) + ' ' + \
                   'byte_order=' + enclose(vtk_dict['byte_order']) + ' ' + \
                   'header_type=' + enclose(vtk_dict['header_type']) + '>' + '\n')
def writeVTKFileFooter(vtk_file):
    vtk_file.write('<' + '/VTKFile' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK dataset element
def writeVTKOpenDatasetElement(vtk_file,vtk_dict,dataset_parameters):
    parameters = copy.deepcopy(dataset_parameters)
    vtk_file.write(indent + '<' + vtk_dict['type'] + ' ' + \
                            ' '.join([ key + '=' + enclose(parameters[key]) \
                                                                 for key in parameters]) + \
                            '>' + '\n')
def writeVTKCloseDatasetElement(vtk_file,vtk_dict):
        vtk_file.write(indent + '<' + '/' + vtk_dict['type'] + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK dataset element piece
def writeVTKOpenPiece(vtk_file,piece_parameters):
    parameters = copy.deepcopy(piece_parameters)
    vtk_file.write(2*indent + '<' + 'Piece' + ' ' + \
                              ' '.join([ key + '=' + enclose(parameters[key]) \
                                                                 for key in parameters]) + \
                              '>' + '\n')
def writeVTKClosePiece(vtk_file):
    vtk_file.write(2*indent + '</Piece>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK point data element
def writeVTKOpenPointData(vtk_file):
    vtk_file.write(3*indent + '<' + 'PointData' + '>' + '\n')
def writeVTKClosePointData(vtk_file):
    vtk_file.write(3*indent + '<' + '/PointData' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK cell data element
def writeVTKOpenCellData(vtk_file):
    vtk_file.write(3*indent + '<' + 'CellData' + '>' + '\n')
def writeVTKCloseCellData(vtk_file):
    vtk_file.write(3*indent + '<' + '/CellData' + '>' + '\n')
# ------------------------------------------------------------------------------------------
# Write VTK cell data array
def writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters):
    # Set cell data array data type and associated ascii format
    if all(isinstance(x,int) or isinstance(x,np.integer) for x in data_list):
        data_type = 'Int32'
        frmt = 'd'
    elif all('bool' in str(type(x)).lower() for x in data_list):
        data_type = 'Int32'
        frmt = 'd'
    else:
        if vtk_dict['precision'] == 'SinglePrecision':
            data_type = 'Float32'
            frmt = '16.8e'
        elif vtk_dict['precision'] == 'DoublePrecision':
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
        values = list(np.where(np.array(values) < min_val,min_val,np.array(values)))
    if 'RangeMax' in data_parameters.keys():
        parameters['RangeMax'] = data_parameters['RangeMax']
        max_val = data_parameters['RangeMax']
        # Mask data values according to specified range upper bound
        values = list(np.where(np.array(values) > max_val,max_val,np.array(values)))
    # Write VTK data array header
    vtk_file.write(4*indent + '<' + 'DataArray' + ' ' + \
                              ' '.join([ key + '=' + enclose(parameters[key]) \
                                                       for key in parameters]) + '>' + '\n')
    # Write VTK data array values
    n_line_vals = 6
    template1 = 5*indent + n_line_vals*('{: ' + frmt + '}') + '\n'
    template2 = 5*indent + (len(values) % n_line_vals)*('{: ' + frmt + '}') + '\n'
    aux_list = [values[i:i+n_line_vals] for i in range(0,len(values),n_line_vals)]
    for i in range(len(aux_list)):
        if i == len(aux_list) - 1 and len(values) % n_line_vals != 0:
            vtk_file.write(template2.format(*aux_list[i]))
        else:
            vtk_file.write(template1.format(*aux_list[i]))
    # Write VTK data array footer
    vtk_file.write(4*indent + '<' + '/DataArray' + '>' + '\n')
