#
# Links Interface (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | March 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Subprocess management
import subprocess
# Working with arrays
import numpy as np
# Extract information from path
import ntpath
# Read specific lines from file
import linecache
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import errors
# Manage files and directories
import fileOperations
#
#                                                                    Links parameters reader
#                                                                          (input data file)
# ==========================================================================================
# Read the parameters from the input data file required to generate the Links input data
# file and solve the microscale equilibrium problem
def readLinksParameters(file,file_path,problem_type,checkNumber,checkPositiveInteger,
                                               searchKeywordLine,searchOptionalKeywordLine):
    # Initialize Links parameters dictionary
    Links_dict = dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links problem type
    problem_type_converter = {'1':2, '2':1, '3':3, '4':6}
    Links_dict['analysis_type'] = problem_type_converter[str(problem_type)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the Links binary absolute path
    keyword = 'Links_bin'
    line_number = searchKeywordLine(file,keyword) + 1
    Links_bin_path = linecache.getline(file_path,line_number).strip()
    if not os.path.isabs(Links_bin_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00068',location.filename,location.lineno+1,keyword, \
                                                                             Links_bin_path)
    elif not os.path.isfile(Links_bin_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00068',location.filename,location.lineno+1,keyword, \
                                                                             Links_bin_path)
    # Store Links binary absolute path
    Links_dict['Links_bin_path'] = Links_bin_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the finite element order (linear or quadratic). If the associated keyword is not
    # found, then a default specification is assumed
    keyword = 'Links_FE_order'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number).split()
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00063',location.filename,location.lineno+1,keyword)
        elif line[1] not in ['linear','quadratic']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00063',location.filename,location.lineno+1,keyword)
        fe_order = line[1]
    else:
        fe_order = 'quadratic'
    # Store finite element order
    Links_dict['fe_order'] = fe_order
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read microscale boundary condition. If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Links_boundary_type'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number).split()
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00064',location.filename,location.lineno+1,keyword)
        elif line[1] not in ['Taylor_Condition','Linear_Condition','Periodic_Condition',
                               'Uniform_Traction_Condition','Uniform_Traction_Condition_II',
                                'Mortar_Periodic_Condition','Mortar_Periodic_Condition_LM']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00064',location.filename,location.lineno+1,keyword)
        boundary_type = line[1]
    else:
        boundary_type = 'Periodic_Condition'
    # Store microscale boundary condition
    Links_dict['boundary_type'] = boundary_type
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read convergence tolerance. If the associated keyword is not found, then a default
    # specification is assumed
    keyword = 'Links_convergence_tolerance'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number+1).split()
        if line == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        elif len(line) != 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        elif not checkNumber(line[0]) or float(line[0]) <= 0:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        convergence_tolerance = float(line[0])
    else:
        convergence_tolerance = 1e-6
    # Store convergence tolerance
    Links_dict['convergence_tolerance'] = convergence_tolerance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read elemental average output mode. If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Links_Element_Average_Output_Mode'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00069',location.filename,location.lineno+1,keyword)
        elif not checkPositiveInteger(line[1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00069',location.filename,location.lineno+1,keyword)
        element_avg_output_mode = int(line[1])
    else:
        element_avg_output_mode = 1
    # Store element average output mode
    Links_dict['element_avg_output_mode'] = element_avg_output_mode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return Links_dict
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
    title = 'Links input data file generated automatically by UNNAMED program'
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
#
#                                                         Links material constitutive models
# ==========================================================================================
# Set material procedures for a given Links constitutive model
def LinksMaterialProcedures(model_name):
    if model_name == 'ELASTIC':
        # Set the constitutive model required material properties
        def setRequiredProperties():
            # Set required material properties
            req_material_properties = ['density','E','v']
            # Return
            return req_material_properties
        # Append Links constitutive model properties specification to a given data file
        def writeMaterialProperties(file_path,mat_phase,properties):
            # Open data file to append Links constitutive model properties
            data_file = open(file_path,'a')
            # Format file structure
            write_list = [mat_phase + ' ' + 'ELASTIC' + '\n'] + \
                         [(len(mat_phase) + 1)*' ' + \
                                   str('{:16.8e}'.format(properties['density'])) + '\n'] + \
                         [(len(mat_phase) + 1)*' ' + \
                                             str('{:16.8e}'.format(properties['E'])) +
                                             str('{:16.8e}'.format(properties['v'])) + '\n']
            # Append Links constitutive model properties
            data_file.writelines(write_list)
            # Close data file
            data_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [setRequiredProperties,writeMaterialProperties]
#
#                                                                         Links program call
# ==========================================================================================
# Solve a given microscale equilibrium problem with program Links
def runLinks(Links_bin_path,Links_file_path):
    # Call program Links
    subprocess.run([Links_bin_path,Links_file_path],stdout=subprocess.PIPE,\
                                                                     stderr=subprocess.PIPE)
    # Check if the microscale equilibrium problem was successfully solved
    screen_file_name = ntpath.splitext(ntpath.basename(Links_file_path))[0]
    screen_file_path = ntpath.dirname(Links_file_path) + '/' + \
                                       screen_file_name + '/' + screen_file_name + '.screen'
    if not os.path.isfile(screen_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00071',location.filename,location.lineno+1,screen_file_path)
    else:
        is_solved = False
        screen_file = open(screen_file_path,'r')
        screen_file.seek(0)
        line_number = 0
        for line in screen_file:
            line_number = line_number + 1
            if 'Program L I N K S successfully completed.' in line:
                is_solved = True
                break
        if not is_solved:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00072',location.filename,location.lineno+1,
                                                           ntpath.basename(Links_file_path))
#
#                                                                 Post process Links results
# ==========================================================================================
# Get the elementwise average strain tensor components
def getStrainVox(Links_file_path,n_dim,comp_order,n_voxels_dims):
    # Initialize strain tensor
    strain_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    # Set elementwise average output file path and check file existence
    elagv_file_name = ntpath.splitext(ntpath.basename(Links_file_path))[0]
    elagv_file_path = ntpath.dirname(Links_file_path) + '/' + \
                                          elagv_file_name + '/' + elagv_file_name + '.elavg'
    if not os.path.isfile(elagv_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00070',location.filename,location.lineno+1,elagv_file_path)
    # Load elementwise average strain tensor components
    elagv_array = np.genfromtxt(elagv_file_path,autostrip=True)
    # Get Links strain components order
    Links_comp_order_sym,_ = getLinksCompOrder(n_dim)
    # Loop over Links strain components
    for i in range(len(Links_comp_order_sym)):
        # Get Links strain component
        Links_comp = Links_comp_order_sym[i]
        # Set Links Voigt notation factor
        Voigt_factor = 2.0 if Links_comp[0] != Links_comp[1] else 1.0
        # Store elementwise average strain component
        strain_vox[Links_comp] = \
                        (1.0/Voigt_factor)*elagv_array[i,:].reshape(n_voxels_dims,order='F')
    # Return
    return strain_vox
# ------------------------------------------------------------------------------------------
# Set Links strain/stress components order in symmetric and nonsymmetric cases
def getLinksCompOrder(n_dim):
    if n_dim == 2:
        Links_comp_order_sym = ['11','22','12']
        Links_comp_order_nsym = ['11','21','12','22']
    else:
        Links_comp_order_sym = ['11','22','33','12','23','13']
        Links_comp_order_nsym = ['11','21','31','12','22','32','13','23','33']
    return [Links_comp_order_sym,Links_comp_order_nsym]
