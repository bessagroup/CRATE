#
# Links Input Data File Generator Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the generation of a finite element code Links'
# input data file, including the discretization of the RVE in a regular quadrilateral (2D) /
# hexahedral (3D) finite element mesh.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Display errors, warnings and built-in exceptions
import errors
# Manage files and directories
import fileOperations
#
#                                                           Links input data file generation
# ==========================================================================================
# Write Links input data file for a given macroscale strain loading
def writeLinksInputDataFile(file_name,dirs_dict,problem_dict,mat_dict,rg_dict,clst_dict,
                                                                                mac_strain):
    # Get directories data
    offline_stage_dir = dirs_dict['offline_stage_dir']
    # Get problem data
    n_dim = problem_dict['n_dim']
    # Get material data
    n_material_phases = mat_dict['n_material_phases']
    material_phases = mat_dict['material_phases']
    material_properties = mat_dict['material_properties']
    material_phases_models = mat_dict['material_phases_models']
    # Get regular grid data
    n_voxels_dims = rg_dict['n_voxels_dims']
    rve_dims = rg_dict['rve_dims']
    regular_grid = rg_dict['regular_grid']
    rg_file_name = rg_dict['rg_file_name']
    # Get Links input data file parameters
    Links_dict = clst_dict['Links_dict']
    fe_order = Links_dict['fe_order']
    analysis_type = Links_dict['analysis_type']
    boundary_type = Links_dict['boundary_type']
    convergence_tolerance = Links_dict['convergence_tolerance']
    element_avg_output_mode = Links_dict['element_avg_output_mode']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set and create offline stage Links directory if it does not exist
    os_Links_dir = offline_stage_dir + 'Links' + '/'
    if not os.path.exists(os_Links_dir):
        fileOperations.makeDirectory(os_Links_dir)
    # Set Links input data file path
    Links_file_path = os_Links_dir + file_name + '.rve'
    # Abort if attempting to overwrite an existing Links input data file
    if os.path.isfile(Links_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00066',location.filename,location.lineno+1,
                                                           ntpath.basename(Links_file_path))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set additional Links input data file parameters (fixed)
    title = 'Links input data file generated automatically by CRATE program'
    large_strain_formulation = 'OFF'
    number_of_increments = 1
    solver = 'PARDISO'
    parallel_solver = 4
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate Links finite element mesh
    coords,connectivities,element_phases = generateFEMesh(n_dim,rve_dims,n_voxels_dims,
                                                                      regular_grid,fe_order)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open Links input data file
    Links_file = open(Links_file_path,'w')
    # Format file structure
    write_list = ['\n' + 'TITLE ' + '\n' + title + '\n'] + \
                 ['\n' + 'ANALYSIS_TYPE ' + str(analysis_type) + '\n'] + \
                 ['\n' + 'LARGE_STRAIN_FORMULATION ' + large_strain_formulation + '\n'] + \
                 ['\n' + 'Boundary_Type ' + boundary_type + '\n'] + \
                 ['\n' + 'Prescribed_Epsilon' + '\n'] + \
                 [' '.join([str('{:16.8e}'.format(mac_strain[i,j]))
                          for j in range(n_dim)]) + '\n' for i in range(n_dim)] + ['\n'] + \
                 ['Number_of_Increments ' + str(number_of_increments) + '\n'] + \
                 ['\n' + 'CONVERGENCE_TOLERANCE' + '\n' + str(convergence_tolerance) +
                                                                                   '\n'] + \
                 ['\n' + 'SOLVER ' + solver + '\n'] + \
                 ['\n' + 'PARALLEL_SOLVER ' + str(parallel_solver) + '\n'] + \
                 ['\n' + 'VTK_OUTPUT' + '\n'] + \
                 ['\n' + 'Element_Average_Output ' + str(element_avg_output_mode) + '\n']
    # Write Links input data file
    Links_file.writelines(write_list)
    # Close Links input data file
    Links_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append finite element mesh to Links input data file
    writeLinksFEMesh(Links_file_path,n_dim,n_material_phases,material_phases,
                  material_properties,material_phases_models,fe_order,coords,connectivities,
                                                                             element_phases)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create a file containing solely the Links finite element mesh data if it does not
    # exist yet
    mesh_path = os_Links_dir + rg_file_name + '.femsh'
    if not os.path.isfile(mesh_path):
        writeLinksFEMesh(mesh_path,n_dim,n_material_phases,material_phases,
                                 material_properties,material_phases_models,fe_order,coords,
                                                              connectivities,element_phases)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return Links_file_path
#
#                                                                  Links finite element mesh
# ==========================================================================================
# Generate regular mesh of quadrilateral (2D) / hexahedral (3D) finite linear or quadratic
# elements
def generateFEMesh(n_dim,rve_dims,n_voxels_dims,regular_grid,fe_order):
    # Initialize array with finite element mesh nodes
    if fe_order == 'linear':
        nodes_grid = np.zeros(np.array(n_voxels_dims)+1,dtype=int)
    else:
        nodes_grid = np.zeros(2*np.array(n_voxels_dims)+1,dtype=int)
    # Initialize coordinates dictionary
    coords = dict()
    # Initialize connectivities dictionary
    connectivities = dict()
    # Initialize element phases dictionary
    element_phases = dict()
    # Set sampling periods in each dimension
    sampling_period = [rve_dims[i]/n_voxels_dims[i] for i in range(n_dim)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set nodes coordinates
    node = 1
    if n_dim == 2:
        # Set nodes for linear (QUAD4) or quadratic (QUAD8) finite element mesh
        if fe_order == 'linear':
            # Loop over nodes
            for j in range(n_voxels_dims[1]+1):
                for i in range(n_voxels_dims[0]+1):
                    nodes_grid[i,j] = node
                    # Set node coordinates
                    coords[str(node)] = [i*sampling_period[0],j*sampling_period[1]]
                    # Increment node counter
                    node = node + 1
        elif fe_order == 'quadratic':
            # Loop over nodes
            for j in range(2*n_voxels_dims[1]+1):
                for i in range(2*n_voxels_dims[0]+1):
                    if j % 2 != 0 and i % 2 != 0:
                        # Skip inexistent node
                        nodes_grid[i,j] = -1
                    else:
                        nodes_grid[i,j] = node
                        # Set node coordinates
                        coords[str(node)] = \
                                         [i*0.5*sampling_period[0],j*0.5*sampling_period[1]]
                        # Increment node counter
                        node = node + 1
    elif n_dim == 3:
        # Set nodes for linear (HEXA8) or quadratic (HEXA20) finite element mesh
        if fe_order == 'linear':
            # Loop over nodes
            for k in range(n_voxels_dims[2]+1):
                for j in range(n_voxels_dims[1]+1):
                    for i in range(n_voxels_dims[0]+1):
                        nodes_grid[i,j,k] = node
                        # Set node coordinates
                        coords[str(node)] = \
                            [i*sampling_period[0],j*sampling_period[1],k*sampling_period[2]]
                        # Increment node counter
                        node = node + 1
        if fe_order == 'quadratic':
            # Loop over nodes
            for k in range(2*n_voxels_dims[2]+1):
                for j in range(2*n_voxels_dims[1]+1):
                    for i in range(2*n_voxels_dims[0]+1):
                        # Skip inexistent node
                        if (j % 2 != 0 and i % 2 != 0) or \
                                                (k % 2 != 0 and (j % 2 != 0 or i % 2 != 0)):
                            nodes_grid[i,j,k] = -1
                        else:
                            # Set node coordinates
                            nodes_grid[i,j,k] = node
                            coords[str(node)] = [i*0.5*sampling_period[0], \
                                          j*0.5*sampling_period[1],k*0.5*sampling_period[2]]
                            # Increment node counter
                            node = node + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set element connectivities and material phases
    elem = 1
    if n_dim == 2:
        # Set linear (QUAD4) or quadratic (QUAD8) finite element mesh connectivities
        if fe_order == 'linear':
            # Loop over elements
            for j in range(n_voxels_dims[1]):
                for i in range(n_voxels_dims[0]):
                    # Set element connectivities
                    connectivities[str(elem)] = [nodes_grid[i,j],nodes_grid[i+1,j],
                                                      nodes_grid[i+1,j+1],nodes_grid[i,j+1]]
                    # Set element material phase
                    element_phases[str(elem)] = regular_grid[i,j]
                    # Increment element counter
                    elem = elem + 1
        elif fe_order == 'quadratic':
            # Loop over elements
            for j in range(n_voxels_dims[1]):
                for i in range(n_voxels_dims[0]):
                    # Set element connectivities
                    connectivities[str(elem)] = [nodes_grid[2*i,2*j],nodes_grid[2*i+1,2*j],
                                              nodes_grid[2*i+2,2*j],nodes_grid[2*i+2,2*j+1],
                                           nodes_grid[2*i+2,2*j+2], nodes_grid[2*i+1,2*j+2],
                                                nodes_grid[2*i,2*j+2],nodes_grid[2*i,2*j+1]]
                    # Set element material phase
                    element_phases[str(elem)] = regular_grid[i,j]
                    # Increment element counter
                    elem = elem + 1
    elif n_dim == 3:
        # Set linear (HEXA8) or quadratic (HEXA20) finite element mesh connectivities
        if fe_order == 'linear':
            # Loop over elements
            for k in range(n_voxels_dims[2]):
                for j in range(n_voxels_dims[1]):
                    for i in range(n_voxels_dims[0]):
                        # Set element connectivities
                        connectivities[str(elem)] = [nodes_grid[i,j,k],nodes_grid[i,j,k+1],
                                                  nodes_grid[i+1,j,k+1],nodes_grid[i+1,j,k],
                                                  nodes_grid[i,j+1,k],nodes_grid[i,j+1,k+1],
                                              nodes_grid[i+1,j+1,k+1],nodes_grid[i+1,j+1,k]]
                        # Set element material phase
                        element_phases[str(elem)] = regular_grid[i,j,k]
                        # Increment element counter
                        elem = elem + 1
        elif fe_order == 'quadratic':
            # Loop over elements
            for k in range(n_voxels_dims[2]):
                for j in range(n_voxels_dims[1]):
                    for i in range(n_voxels_dims[0]):
                        # Set element connectivities
                        connectivities[str(elem)] = [nodes_grid[2*i,2*j,2*k],
                                      nodes_grid[2*i,2*j,2*k+2],nodes_grid[2*i+2,2*j,2*k+2],
                                        nodes_grid[2*i+2,2*j,2*k],nodes_grid[2*i,2*j+2,2*k],
                                  nodes_grid[2*i,2*j+2,2*k+2],nodes_grid[2*i+2,2*j+2,2*k+2],
                                      nodes_grid[2*i+2,2*j+2,2*k],nodes_grid[2*i,2*j,2*k+1],
                                    nodes_grid[2*i+1,2*j,2*k+2],nodes_grid[2*i+2,2*j,2*k+1],
                                        nodes_grid[2*i+1,2*j,2*k],nodes_grid[2*i,2*j+1,2*k],
                                  nodes_grid[2*i,2*j+1,2*k+2],nodes_grid[2*i+2,2*j+1,2*k+2],
                                    nodes_grid[2*i+2,2*j+1,2*k],nodes_grid[2*i,2*j+2,2*k+1],
                                nodes_grid[2*i+1,2*j+2,2*k+2],nodes_grid[2*i+2,2*j+2,2*k+1],
                                                                nodes_grid[2*i+1,2*j+2,2*k]]
                        # Set element material phase
                        element_phases[str(elem)] = regular_grid[i,j,k]
                        # Increment element counter
                        elem = elem + 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [coords,connectivities,element_phases]
# ------------------------------------------------------------------------------------------
# Append Links finite element mesh (groups, elements, materials, element connectivities and
# nodal coordinates) to a given data file
def writeLinksFEMesh(file_path,n_dim,n_material_phases,material_phases,material_properties,
                      material_phases_models,fe_order,coords,connectivities,element_phases):
    # Set element designation and number of Gauss integration points
    if n_dim == 2:
        if fe_order == 'linear':
            elem_type = 'QUAD4'
            n_gp = 4
        else:
            elem_type = 'QUAD8'
            n_gp = 4
    else:
        if fe_order == 'linear':
            elem_type = 'HEXA8'
            n_gp = 8
        else:
            elem_type = 'HEXA20'
            n_gp = 8
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open data file to append Links finite element mesh
    data_file = open(file_path,'a')
    # Format file structure
    write_list = ['\n' + 'ELEMENT_GROUPS ' + str(n_material_phases) + '\n'] + \
                 [str(mat+1) + ' 1 ' + str(mat+1) + '\n' \
                                                    for mat in range(n_material_phases)] + \
                 ['\n' + 'ELEMENT_TYPES 1' + '\n', '1 ' + elem_type + '\n', '  ' + \
                                                               str(n_gp) + ' GP' + '\n'] + \
                 ['\n' + 'MATERIALS ' + str(n_material_phases) + '\n']
    # Append first part of the Links finite element mesh to data file
    data_file.writelines(write_list)
    # Close data file
    data_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Append material phases Links constitutive models and associated properties
    for mat_phase in material_phases:
        material_phases_models[mat_phase]['writeMaterialProperties'](file_path,
                                                   mat_phase,material_properties[mat_phase])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open data file to append Links finite element mesh
    data_file = open(file_path,'a')
    # Format file structure
    write_list = ['\n' + 'ELEMENTS ' + str(len(connectivities.keys())) + '\n'] + \
                 ['{:>3s}'.format(str(elem)) + \
                          '{:^5d}'.format(element_phases[str(elem)]) + ' '.join([str(node) \
                                            for node in connectivities[str(elem)]]) + '\n' \
                       for elem in np.sort([int(key) for key in connectivities.keys()])] + \
                 ['\n' + 'NODE_COORDINATES ' + str(len(coords.keys())) + \
                                                                    ' CARTESIAN' + '\n'] + \
                 ['{:>3s}'.format(str(node)) + ' ' + \
                                                   ' '.join([str('{:16.8e}'.format(coord)) \
                                                   for coord in coords[str(node)]]) + '\n' \
                                   for node in np.sort([int(key) for key in coords.keys()])]
    # Append last part of the Links finite element mesh to data file
    data_file.writelines(write_list)
    # Close data file
    data_file.close()
