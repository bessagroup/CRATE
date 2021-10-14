#
# Utilities Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module which serves only the purpose of implementing utilities functions which have easy
# 'access' to the main code and that are used in order to validate or perform numerical
# studies.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#
#                                                                      FFT vs FEM Comparison
# ==========================================================================================
# Solve microscale static equilibrium problem with the FFT-based homogenization method
# proposed in "A numerical method for computing the overall response of nonlinear composites
#  with complex microstructure. Comp Methods Appl M 157 (1998):69-94 (Moulinec, H. and
# Suquet, P.". The RVE is subjected to a given macroscale strain and constrained by periodic
# boundary conditions.
#
# Notes:
#
#   1. Add stress_vox, phase_names and phase_times to the output of
#      FFTHomogenizationBasicScheme()
#   2. Choose the appropriate convergence tolerance
#   3. Choose the appropriate convergence criterion
#
def utility1():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import modules
    import numpy as np
    import itertools as it
    import ntpath
    import time
    import ioput.info as info
    import ioput.fileoperations as filop
    import tensor.matrixoperations as mop
    import FFTHomogenizationBasicScheme as FFT
    import ioput.ioutilities as ioutil
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem type
    problem_type = 1
    # Set problem parameters
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # Set problem data
    problem_dict = dict()
    problem_dict['problem_type'] = problem_type
    problem_dict['n_dim'] = n_dim
    problem_dict['comp_order_sym'] = comp_order_sym
    problem_dict['comp_order_nsym'] = comp_order_nsym
    # Set spatial discretization file absolute path (regular grid file)
    if problem_type == 1:
        rve_dims = [1.0,1.0]
        discret_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                            'microstructures/2D/main/regular_grids/' + \
                            'Disk_50_0.3_800_800.rgmsh.npy'
    else:
        rve_dims = [1.0,1.0,1.0]
        discret_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                            'microstructures/3D/main/regular_grids/' + \
                            'Sphere_20_0.2_80_80_80.rgmsh.npy'
    # Read spatial discretization file and set regular grid data
    regular_grid = np.load(discret_file_path)
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    rg_dict = dict()
    rg_dict['rve_dims'] = rve_dims
    rg_dict['regular_grid'] = regular_grid
    rg_dict['n_voxels_dims'] = n_voxels_dims
    # Set material properties
    material_properties = dict()
    material_properties['1'] = dict()
    material_properties['1']['E'] = 100
    material_properties['1']['v'] = 0.3
    material_properties['2'] = dict()
    material_properties['2']['E'] = 500
    material_properties['2']['v'] = 0.19
    mat_dict = dict()
    mat_dict['material_properties'] = material_properties
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    mat_dict['material_phases'] = material_phases
    # Set macroscale strain loading
    loading = 'pureshear'
    if n_dim == 2:
        if loading == 'uniaxial':
            mac_strain = np.array([[ 5.0e-3 , 0.0e-3 ],
                                   [ 0.0e-3 , 0.0e-3 ]])
        elif loading == 'pureshear':
            mac_strain = np.array([[ 0.0e-3 , 5.0e-3 ],
                                   [ 5.0e-3 , 0.0e-3 ]])
    else:
        if loading == 'uniaxial':
            mac_strain = np.array([[ 5.0e-3 , 0.0e-3 , 0.0e-3 ],
                                   [ 0.0e-3 , 0.0e-3 , 0.0e-3 ],
                                   [ 0.0e-3 , 0.0e-3 , 0.0e-3 ]])
        elif loading == 'pureshear':
            mac_strain = np.array([[ 0.0e-3 , 5.0e-3 , 5.0e-3 ],
                                   [ 5.0e-3 , 0.0e-3 , 5.0e-3 ],
                                   [ 5.0e-3 , 5.0e-3 , 0.0e-3 ]])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set output directory
    discret_file_basename = \
        ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[-2])[-2]
    if problem_type == 1:
        output_dir = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                     'offline_stage/main/2D/FFT_NEW/'
    else:
        output_dir = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                     'offline_stage/main/3D/FFT_NEW/'
    output_dir = output_dir + discret_file_basename + '_' + loading + '/'
    filop.makedirectory(output_dir,option='overwrite')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set absolute path of the file where the error for the diferent convergence criteria
    # is written at every iteration
    conv_file_path = output_dir + 'convergence_table.dat'
    FFT.writeIterationConvergence(conv_file_path,'header')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start timer
    time_init = time.time()
    # Compute FFT solution
    strain_vox, stress_vox, phase_names, phase_times = \
        FFT.FFTHomogenizationBasicScheme(problem_dict,rg_dict,mat_dict,mac_strain)
    # Stop timer
    time_end = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute voxel volume
    voxel_vol = np.prod([float(rve_dims[i])/n_voxels_dims[i] for i in range(len(rve_dims))])
    # Compute RVE volume
    rve_vol = np.prod(rve_dims)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize homogenized strain and stress tensors
    hom_strain = np.zeros((3,3))
    hom_stress = np.zeros((3,3))
    # Loop over voxels
    for voxel in it.product(*[list(range(n)) for n in n_voxels_dims]):
        # Loop over strain/stress components
        for comp in comp_order_sym:
            # 2nd order index
            idx = tuple([int(i) - 1 for i in comp])
            # Add voxel contribution to homogenized strain tensor
            hom_strain[idx] = hom_strain[idx] + (voxel_vol/rve_vol)*strain_vox[comp][voxel]
            # Add voxel contribution to homogenized stress tensor
            hom_stress[idx] = hom_stress[idx] + (voxel_vol/rve_vol)*stress_vox[comp][voxel]
        # Compute out-of-plane stress component in a 2D plane strain problem
        if problem_type == 1:
            # Get voxel material phase
            mat_phase = str(regular_grid[voxel])
            # Get material phase Lamé parameter
            E = material_properties[mat_phase]['E']
            v = material_properties[mat_phase]['v']
            lam = (E*v)/((1.0+v)*(1.0-2.0*v))
            # Compute out-of-plane stress component
            stress_33 = lam*(strain_vox['11'][voxel] + strain_vox['22'][voxel])
            # Assemble out-of-plane stress component
            hom_stress[2,2] = hom_stress[2,2] + (voxel_vol/rve_vol)*stress_33
    # Assemble symmetric strain and stress components
    for comp in comp_order_sym:
        if comp[0] != comp[1]:
            idx = tuple([int(i) - 1 for i in comp])
            hom_strain[idx[::-1]] = hom_strain[idx]
            hom_stress[idx[::-1]] = hom_stress[idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set homogenized results output format
    display_features = ioutil.setdisplayfeatures()
    output_width = display_features[0]
    indent = display_features[2]
    equal_line = display_features[5]
    arguments = list()
    for i in range(3):
        for j in range(3):
            arguments.append(hom_strain[i,j])
        for j in range(3):
            arguments.append(hom_stress[i,j])
    arguments = arguments + [time_end - time_init,]
    info = tuple(arguments)
    space1 = (output_width - 84)*' '
    space2 = (output_width - (len('Homogenized strain tensor') + 48))*' '
    template = '\n' + \
               indent + 'FFT-Based Homogenization Method Results' + '\n' + \
               indent + equal_line[:-len(indent)] + '\n' + \
               indent + 7*' ' + 'Homogenized strain tensor (\u03B5)' + space2 + \
                                   'Homogenized stress tensor (\u03C3)' + '\n\n' + \
               indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                                              '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
               indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                                              '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
               indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                                              '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
               '\n' + indent + equal_line[:-len(indent)] + '\n' + \
               indent + 'Total run time (s): {:>11.4e}' + '\n\n'
    # Output homogenized results to default stdout
    print(template.format(*info,width=output_width))
    # Output homogenized results to results file
    results_file_path = output_dir + 'results.dat'
    open(results_file_path, 'w').close()
    results_file = open(results_file_path,'a')
    print(template.format(*info,width=output_width),file = results_file)
    results_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute time profile quantities
    total_time = phase_times[0,1] - phase_times[0,0]
    number_of_phases = len(phase_names)
    phase_durations = [ phase_times[i,1] - phase_times[i,0] \
                                                         for i in range(0,number_of_phases)]
    for i in range(0,number_of_phases):
        phase_durations.insert(3*i,phase_names[i])
        phase_durations.insert(3*i+2,(phase_durations[3*i+1]/total_time)*100)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set time profile output format
    arguments = ['Phase','Duration (s)','%'] + phase_durations[3:]
    info = tuple(arguments)
    template = indent + 'Execution times (1 iteration): \n\n' + \
               2*indent + '{:50}{:^20}{:^5}' + '\n' + \
               2*indent + 75*'-' + '\n' + \
               (2*indent + '{:50}{:^20.2e}{:>5.2f} \n')*(number_of_phases-1) + \
               2*indent + 75*'-' + '\n'
    # Output time profile to default stdout
    print(template.format(*info,width=output_width))
    # Output time profile to results file
    results_file = open(results_file_path,'a')
    print(template.format(*info,width=output_width),file = results_file)
    results_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file with material phases
    import ntpath
    dirs_dict = dict()
    dirs_dict['input_file_name'] = discret_file_basename
    dirs_dict['offline_stage_dir'] = output_dir
    clst_dict = dict()
    clst_dict['voxels_clusters'] = np.full(n_voxels_dims,-1,dtype=int)
    vtk_dict = dict()
    vtk_dict['vtk_format'] = 'ascii'
    vtk_dict['vtk_precision'] = 'SinglePrecision'
    utility2(vtk_dict,dirs_dict,rg_dict,clst_dict,comp_order_sym,strain_vox,stress_vox)
#
#                                                                        Image data VTK file
# ==========================================================================================
# Write VTK file with required data defined in a regular grid (image data)
def utility2(vtk_dict,dirs_dict,rg_dict,clst_dict,comp_order_sym,strain_vox,stress_vox):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import modules
    import os
    import sys
    import ioput.vtkoutput as vtkoutput
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    vtk_cluster_file_name = input_file_name + '_FFT.vti'
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
    vtkoutput.writevtk_fileheader(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    dataset_parameters,piece_parameters = \
        vtkoutput.setimagedataparam(n_voxels_dims,rve_dims)
    vtkoutput.writevtk_opendatasetelem(vtk_file,vtk_dict,dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    vtkoutput.writevtk_openpiece(vtk_file,piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    vtkoutput.writevtk_opencelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Material phases
    data_list = list(regular_grid.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Material phase','format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
    vtkoutput.writevtk_celldataarray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Clusters
    data_list = list(voxels_clusters.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Cluster','format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
    vtkoutput.writevtk_celldataarray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Strain tensor
    for comp in comp_order_sym:
        data_list = list(strain_vox[comp].flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name':'e_strain_' + comp,'format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
        vtkoutput.writevtk_celldataarray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Stress tensor
    for comp in comp_order_sym:
        data_list = list(stress_vox[comp].flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name':'stress_' + comp,'format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
        vtkoutput.writevtk_celldataarray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    vtkoutput.writevtk_closecelldata(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    vtkoutput.writevtk_closepiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    vtkoutput.writevtk_closedatasetelem(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    vtkoutput.writevtk_filefooter(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close clustering VTK file
    vtk_file.close()
#
#                                                               Links regular grid mesh file
# ==========================================================================================
# Given a regular grid mesh file (.rgmsh), generate a regular mesh of quadrilateral (2D) /
# hexahedral (3D) finite linear or quadratic elements and store it in a .fmsh file
# compatible with Links
def utility3():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import modules
    import numpy as np
    import ntpath
    import links.ioput.genlinksinputdatafile
    #
    #                                                                            Data import
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discretization file path (.rgmsh file)
    discret_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/microstructures/3D/main/regular_grids/Sphere_20_0.2_100_100_100.rgmsh.npy'
    # Read the spatial discretization file (regular grid of pixels/voxels)
    if ntpath.splitext(ntpath.basename(discret_file_path))[-1] == '.npy':
        regular_grid = np.load(discret_file_path)
        rg_file_name = \
                ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[-2])[-2]
    else:
        regular_grid = np.loadtxt(discret_file_path)
        rg_file_name = ntpath.splitext(ntpath.basename(discret_file_path))[-2]
    # Set number of pixels/voxels in each dimension
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    #
    #                                                                  Regular grid file fix
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if True:
        # Change required material phases labels
        regular_grid = np.where(regular_grid==0,1,regular_grid)
        # Write updated spatial discretization file (numpy binary format)
        np.save(discret_file_path,regular_grid)
    #
    #                                                                  Links input data file
    #                                         (finite element mesh nodes and connectivities)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if True:
        # Get problem dimension
        n_dim = len(n_voxels_dims)
        # Set RVE dimensions
        rve_dims = n_dim*[1.0,]
        # Set finite element order
        fe_order = 'quadratic'
        # Set Links input data file path
        if fe_order == 'linear':
            links_file_path = ntpath.dirname(discret_file_path) + '/' + \
                              rg_file_name + '_L' + '.femsh'
        else:
            links_file_path = ntpath.dirname(discret_file_path) + '/' + \
                              rg_file_name + '_Q' + '.femsh'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Links finite element mesh
        coords,connectivities,element_phases = \
            Links.ioput.genlinksinputdatafile.genlinksfemesh(n_dim,rve_dims,n_voxels_dims,
                                                         regular_grid,fe_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # Open data file to append Links finite element mesh
        data_file = open(links_file_path,'a')
        # Format file structure
        write_list = ['\n' + 'ELEMENT_TYPES 1' + '\n', '1 ' + elem_type + '\n', '  ' + \
                                                               str(n_gp) + ' GP' + '\n'] + \
                     ['\n' + 'ELEMENTS ' + str(len(connectivities.keys())) + '\n'] + \
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
# ------------------------------------------------------------------------------------------
#
#                                                                 Discrete frequencies loops
# ==========================================================================================
# Assessment of efficient implementations of computations perform over discrete frequencies
def utility4():
    # Import modules
    import numpy as np
    import numpy.matlib
    import numpy.linalg
    import time
    import itertools as it
    import copy
    import tensor.tensoroperations as top
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem dimension
    n_dim = 2
    # Set strain/stress component order
    if n_dim == 2:
        comp_order_sym = ['11','22','12']
    else:
        comp_order_sym = ['11','22','33','12','23','13']
    comp_order = comp_order_sym
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # Set number of voxels on each dimension
    n = [10,12,13]
    n_voxels_dims = [i for i in n[0:n_dim]]
    # Set material properties
    mat_prop_ref = dict()
    mat_prop_ref['E'] = 100
    mat_prop_ref['v'] = 0.3
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize time arrays
    phase_names = ['']
    phase_times = np.zeros((1,2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #                                                      Reference material Green operator
    #                           (original version - standard loop over discrete frequencies)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # The fourth-order Green operator for the reference material is computed for every
    # pixel/voxel and stored as follows:
    #
    # A. 2D problem (plane strain):
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |      form ('1111', '2211', '1211', '1122', ...)
    #
    # B. 3D problem:
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2,d3),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |     form ('1111', '2211', '3311', '1211', ...)
    #
    # Get reference material Young modulus and Poisson coeficient
    E_ref = mat_prop_ref['E']
    v_ref = mat_prop_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator reference material related constants
    c1 = 1.0/(4.0*miu_ref)
    c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    # Set Green operator matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                             [comp_order.index(comps[i][0]),comp_order.index(comps[i][1])]])
    # Initialize Green operator
    Green_operator_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get discrete frequency index
            freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) for \
                                                                         x in range(n_dim)])
            # Skip zero-frequency computation (prescribed macroscale strain)
            if freq_idx == n_dim*(0,):
                continue
            # Compute frequency vector norm
            freq_norm = np.linalg.norm(freq_coord)
            # Compute first material independent term of Green operator
            first_term = (1.0/freq_norm**2)*(
                   top.dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[3]] +
                   top.dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                   top.dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                   top.dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
            # Compute second material independent term of Green operator
            second_term = -(1.0/freq_norm**4)*(freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                               freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
            # Compute Green operator matricial form component for current voxel
            Green_operator_DFT_vox[comp][freq_idx] = c1*first_term + c2*second_term
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_end_time = time.time()
    phase_names.append('Original version')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    # Copy Green_operator_DFT_vox (comparison)
    GOP_old = copy.deepcopy(Green_operator_DFT_vox)
    #
    #                                                      Reference material Green operator
    #                                                                          (new version)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # The fourth-order Green operator for the reference material is computed for every
    # pixel/voxel and stored as follows:
    #
    # A. 2D problem (plane strain):
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |      form ('1111', '2211', '1211', '1122', ...)
    #
    # B. 3D problem:
    #
    #   Green_operator_DFT_vox[comp] = array(d1,d2,d3),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |     form ('1111', '2211', '3311', '1211', ...)
    #
    # Get reference material Young modulus and Poisson coeficient
    E_ref = mat_prop_ref['E']
    v_ref = mat_prop_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator reference material related constants
    c1 = 1.0/(4.0*miu_ref)
    c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    # Set Green operator matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                             [comp_order.index(comps[i][0]),comp_order.index(comps[i][1])]])
    # Set optimized variables
    var1 = [*np.meshgrid(*freqs_dims,indexing = 'ij')]
    var2 = dict()
    for fo_idx in fo_indexes:
        if str(fo_idx[1]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[1]],var1[fo_idx[3]])
        if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[1]],var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[0]],var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[0]],var1[fo_idx[3]])
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]],var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]],var1[fo_idx[3]]))
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]),np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]),np.square(var1[1])),
                              np.square(var1[2])))
    # Initialize Green operator
    Gop_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Set optimized variables
        var4 = [fo_idx[0] == fo_idx[2],fo_idx[0] == fo_idx[3],
                fo_idx[1] == fo_idx[3],fo_idx[1] == fo_idx[2]]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]),str(fo_idx[1]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[2]),str(fo_idx[0]) + str(fo_idx[3])]
        # Compute first material independent term of Green operator
        first_term = np.zeros(tuple(n_voxels_dims))
        for j in range(len(var4)):
            if var4[j]:
                first_term = np.add(first_term,var2[var5[j]])
        first_term = np.divide(first_term,np.square(var3),where = abs(var3) > 1e-10)
        # Compute second material independent term of Green operator
        second_term = -1.0*np.divide(var2[''.join([str(x) for x in fo_idx])],
                                     np.square(np.square(var3)),where = abs(var3) > 1e-10)
        # Compute Green operator matricial form component
        Gop_DFT_vox[comp] = c1*first_term + c2*second_term
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_end_time = time.time()
    phase_names.append('New version')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    # Copy Green_operator_DFT_vox
    GOP_new = copy.deepcopy(Gop_DFT_vox)
    #
    #                                                                             Comparison
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all the Green operator has the same value between the original version
    # (reference) and the new implementation
    print('\n' + 'Green operator computation')
    print('--------------------------')
    print('\n' + 'Same values as the original implementation?' + '\n')
    for fo_idx in fo_indexes:
        comp = ''.join([str(x+1) for x in fo_idx])
        print(comp + ': ' + str(np.allclose(GOP_old[comp],GOP_new[comp],atol=1e-10)))
    # Compare computation times
    number_of_phases = len(phase_names)
    phase_durations = [phase_times[i,1] - phase_times[i,0] \
                                                         for i in range(0,number_of_phases)]
    print('\n' + 'Performance comparison - time (s):')
    print('\n' + ' time (ref)   time (new)   speedup' + '\n' + 35*'-')
    print('{:11.4e}  {:11.4e} {:>9.1f}'.format(phase_durations[1],phase_durations[2],
                                               phase_durations[1]/phase_durations[2]))
    print('')
# ------------------------------------------------------------------------------------------
def utility5():
    # Import modules
    import sys
    import numpy as np
    import time
    import copy
    import itertools as it
    import tensor.tensoroperations as top
    import tensor.matrixoperations as mop
    import FFTHomogenizationBasicScheme as FFT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem type
    problem_type = 4
    # Set problem dimension, strain/stress component order and spatial discretization file
    # path
    if problem_type == 1:
        n_dim = 2
        comp_order_sym = ['11','22','12']
    else:
        n_dim = 3
        comp_order_sym = ['11','22','33','12','23','13']
    comp_order = comp_order_sym
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # Set material properties
    material_properties = dict()
    material_properties['1'] = dict()
    material_properties['1']['E'] = 100
    material_properties['1']['v'] = 0.30
    material_properties['2'] = dict()
    material_properties['2']['E'] = 500
    material_properties['2']['v'] = 0.19
    material_phases = list(material_properties.keys())
    n_material_phases = len(material_phases)
    # Set number of voxels on each dimension
    n = [10,10,10]
    n_voxels_dims = [i for i in n[0:n_dim]]
    # Set random regular grid
    regular_grid = \
                  np.random.randint(1,high=n_material_phases+1,size=n_voxels_dims,dtype=int)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_tensors_mf = dict()
    for mat_phase in material_phases:
        # Set required elastic properties according to material phase constitutive model
        req_props = ['E','v']
        for iprop in range(len(req_props)):
            if req_props[iprop] not in material_properties[mat_phase]:
                values = tuple([req_props[iprop],mat_phase])
                template = '\nAbort: The elastic property - {} - of material phase {} ' + \
                           'hasn\'t been specified in ' + '\n' + \
                           'the input data file (FFTHomogenizationBasicScheme.py).\n'
                print(template.format(*values))
                sys.exit(1)
        # Compute elasticity tensor (matricial form) for current material phase
        De_tensor_mf = np.zeros((len(comp_order),len(comp_order)))
        De_tensor_mf = FFT.getElasticityTensor(problem_type,n_dim,comp_order,\
                                                             material_properties[mat_phase])
        # Store material phase elasticity tensor (matricial form)
        De_tensors_mf[mat_phase] = De_tensor_mf
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain and stress tensors
    strain_vox = {comp: np.ones(tuple(n_voxels_dims)) for comp in comp_order}
    stress_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize time arrays
    phase_names = ['']
    phase_times = np.zeros((1,2))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #                                                                          Update stress
    #                           (original version - standard loop over discrete frequencies)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Loop over discrete frequencies
    for freq_coord in it.product(*freqs_dims):
        # Get voxel material phase
        voxel_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)])
        mat_phase = str(regular_grid[voxel_idx])
        # Get material phase elasticity tensor (matricial form)
        De_tensor_mf = De_tensors_mf[mat_phase]
        # Get strain vector for current discrete frequency
        strain_mf = np.zeros(len(comp_order))
        for i in range(len(comp_order)):
            comp = comp_order[i]
            strain_mf[i] = mop.kelvinfactor(i,comp_order)*strain_vox[comp][voxel_idx]
        # Update stress for current discrete frequency
        stress_mf = np.zeros(len(comp_order))
        stress_mf = top.dot21_1(De_tensor_mf,strain_mf)
        for i in range(len(comp_order)):
            comp = comp_order[i]
            stress_vox[comp][voxel_idx] = (1.0/mop.kelvinfactor(i,comp_order))*stress_mf[i]
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_end_time = time.time()
    phase_names.append('Original version')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    # Copy stress_vox (comparison)
    stress_vox_old = copy.deepcopy(stress_vox)
    #
    #                                                                          Update stress
    #                                                                          (new version)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set optimized variables
    var6 = np.zeros(tuple(n_voxels_dims))
    var7 = np.zeros(tuple(n_voxels_dims))
    for mat_phase in material_phases:
        E = material_properties[mat_phase]['E']
        v = material_properties[mat_phase]['v']
        var6[regular_grid == int(mat_phase)] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
        var7[regular_grid == int(mat_phase)] = np.multiply(2,E/(2.0*(1.0 + v)))
    var8 = np.add(var6,var7)
    # Update stress
    if problem_type == 1:
        stress_vox['11'] = np.add(np.multiply(var8,strain_vox['11']),
                                  np.multiply(var6,strain_vox['22']))
        stress_vox['22'] = np.add(np.multiply(var8,strain_vox['22']),
                                  np.multiply(var6,strain_vox['11']))
        stress_vox['12'] = np.multiply(var7,strain_vox['12'])
    else:
        stress_vox['11'] = np.add(np.multiply(var8,strain_vox['11']),
                                  np.multiply(var6,np.add(strain_vox['22'],
                                                          strain_vox['33'])))
        stress_vox['22'] = np.add(np.multiply(var8,strain_vox['22']),
                                  np.multiply(var6,np.add(strain_vox['11'],
                                                          strain_vox['33'])))
        stress_vox['33'] = np.add(np.multiply(var8,strain_vox['33']),
                                  np.multiply(var6,np.add(strain_vox['11'],
                                                          strain_vox['22'])))
        stress_vox['12'] = np.multiply(var7,strain_vox['12'])
        stress_vox['23'] = np.multiply(var7,strain_vox['23'])
        stress_vox['13'] = np.multiply(var7,strain_vox['13'])
    # --------------------------------------------------------------------------------------
    # Time profile
    phase_end_time = time.time()
    phase_names.append('New version')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    # Copy stress_vox (comparison)
    stress_vox_new = copy.deepcopy(stress_vox)
    #
    #                                                                             Comparison
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all the Green operator has the same value between the original version
    # (reference) and the new implementation
    print('\n' + 'Stress update')
    print('-------------')
    print('\n' + 'Same values as the original implementation?' + '\n')
    for comp in comp_order:
        print(comp + ': ' + str(np.allclose(stress_vox_old[comp],stress_vox_new[comp],
                                            atol=1e-10)))
    # Compare computation times
    number_of_phases = len(phase_names)
    phase_durations = [phase_times[i,1] - phase_times[i,0] \
                                                         for i in range(0,number_of_phases)]
    print('\n' + 'Performance comparison - time (s):')
    print('\n' + ' time (ref)   time (new)   speedup' + '\n' + 35*'-')
    print('{:11.4e}  {:11.4e} {:>9.1f}'.format(phase_durations[1],phase_durations[2],
                                               phase_durations[1]/phase_durations[2]))
    print('')
# ------------------------------------------------------------------------------------------
def utility7():
    # Import modules
    import numpy as np
    import numpy.matlib
    import numpy.linalg
    import itertools as it
    import copy
    import tensor.tensoroperations as top
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem dimension
    n_dim = 3
    # Set strain/stress component order
    if n_dim == 2:
        comp_order_sym = ['11','22','12']
    else:
        comp_order_sym = ['11','22','33','12','23','13']
    comp_order = comp_order_sym
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # Set number of voxels on each dimension
    n = [10,10,10]
    n_voxels_dims = [i for i in n[0:n_dim]]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    #
    #                                    Reference material Green operator (CIT computation)
    #                           (original version - standard loop over discrete frequencies)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Green operator matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                             [comp_order.index(comps[i][0]),comp_order.index(comps[i][1])]])
    # Initialize Green operator material independent terms
    Gop_1_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_2_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_0_freq_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get discrete frequency index
            freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) for \
                                                                         x in range(n_dim)])
            # Set Green operator zero-frequency term to unit at zero-frequency. Skip the
            # zero-frequency computation for the remaining Green operator terms
            if freq_idx == n_dim*(0,):
                Gop_0_freq_DFT_vox[comp][freq_idx] = 1.0
                continue
            # Compute frequency vector norm
            freq_norm = np.linalg.norm(freq_coord)
            # Compute first material independent term of Green operator
            Gop_1_DFT_vox[comp][freq_idx] = (1.0/freq_norm**2)*(
                   top.dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[3]] +
                   top.dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                   top.dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                   top.dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
            # Compute second material independent term of Green operator
            Gop_2_DFT_vox[comp][freq_idx] = -(1.0/freq_norm**4)*\
                                               (freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                               freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
    # --------------------------------------------------------------------------------------
    # Copy Gop_X_DFT_vox
    Gop_1_DFT_vox_old = copy.deepcopy(Gop_1_DFT_vox)
    Gop_2_DFT_vox_old = copy.deepcopy(Gop_2_DFT_vox)
    Gop_0_freq_DFT_vox_old = copy.deepcopy(Gop_0_freq_DFT_vox)
    #
    #                                    Reference material Green operator (CIT computation)
    #                                                                          (new version)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Green operator matricial form components
    comps = list(it.product(comp_order,comp_order))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_order)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                             [comp_order.index(comps[i][0]),comp_order.index(comps[i][1])]])
    # Set optimized variables
    var1 = [*np.meshgrid(*freqs_dims,indexing = 'ij')]
    var2 = dict()
    for fo_idx in fo_indexes:
        if str(fo_idx[1]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[1]],var1[fo_idx[3]])
        if str(fo_idx[1]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[1]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[1]],var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[2]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[2])] = \
                np.multiply(var1[fo_idx[0]],var1[fo_idx[2]])
        if str(fo_idx[0]) + str(fo_idx[3]) not in var2.keys():
            var2[str(fo_idx[0]) + str(fo_idx[3])] = \
                np.multiply(var1[fo_idx[0]],var1[fo_idx[3]])
        if ''.join([str(x) for x in fo_idx]) not in var2.keys():
            var2[''.join([str(x) for x in fo_idx])] = \
                np.multiply(np.multiply(var1[fo_idx[0]],var1[fo_idx[1]]),
                            np.multiply(var1[fo_idx[2]],var1[fo_idx[3]]))
    if n_dim == 2:
        var3 = np.sqrt(np.add(np.square(var1[0]),np.square(var1[1])))
    else:
        var3 = np.sqrt(np.add(np.add(np.square(var1[0]),np.square(var1[1])),
                              np.square(var1[2])))
    # Initialize Green operator material independent terms
    Gop_1_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_2_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    Gop_0_freq_DFT_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator component
        comp = ''.join([str(x+1) for x in fo_idx])
        # Set optimized variables
        var4 = [fo_idx[0] == fo_idx[2],fo_idx[0] == fo_idx[3],
                fo_idx[1] == fo_idx[3],fo_idx[1] == fo_idx[2]]
        var5 = [str(fo_idx[1]) + str(fo_idx[3]),str(fo_idx[1]) + str(fo_idx[2]),
                str(fo_idx[0]) + str(fo_idx[2]),str(fo_idx[0]) + str(fo_idx[3])]

        # Compute first material independent term of Green operator
        first_term = np.zeros(tuple(n_voxels_dims))
        for j in range(len(var4)):
            if var4[j]:
                first_term = np.add(first_term,var2[var5[j]])
        first_term = np.divide(first_term,np.square(var3),where = abs(var3) > 1e-10)
        Gop_1_DFT_vox[comp] = copy.copy(first_term)
        # Compute second material independent term of Green operator
        Gop_2_DFT_vox[comp] = -1.0*np.divide(var2[''.join([str(x) for x in fo_idx])],
                                     np.square(np.square(var3)),where = abs(var3) > 1e-10)
        # Compute Green operator zero-frequency term
        Gop_0_freq_DFT_vox[comp][tuple(n_dim*(0,))] = 1.0
    # --------------------------------------------------------------------------------------
    # Copy Gop_X_DFT_vox
    Gop_1_DFT_vox_new = copy.deepcopy(Gop_1_DFT_vox)
    Gop_2_DFT_vox_new = copy.deepcopy(Gop_2_DFT_vox)
    Gop_0_freq_DFT_vox_new = copy.deepcopy(Gop_0_freq_DFT_vox)
    #
    #                                                                             Comparison
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if all the Green operator has the same value between the original version
    # (reference) and the new implementation
    print('\n' + 'Green operator computation')
    print('--------------------------')
    print('\n' + 'Same values as the original implementation?' + '\n')
    print('\n' + 'Gop_1_DFT_vox:' + '\n')
    for fo_idx in fo_indexes:
        comp = ''.join([str(x+1) for x in fo_idx])
        print(comp + ': ' + str(np.allclose(Gop_1_DFT_vox_old[comp],
                                            Gop_1_DFT_vox_new[comp],atol=1e-10)))
    print('\n' + 'Gop_2_DFT_vox:' + '\n')
    for fo_idx in fo_indexes:
        comp = ''.join([str(x+1) for x in fo_idx])
        print(comp + ': ' + str(np.allclose(Gop_2_DFT_vox_old[comp],
                                            Gop_2_DFT_vox_new[comp],atol=1e-10)))
    print('\n' + 'Gop_0_freq_DFT_vox:' + '\n')
    for fo_idx in fo_indexes:
        comp = ''.join([str(x+1) for x in fo_idx])
        print(comp + ': ' + str(np.allclose(Gop_0_freq_DFT_vox_old[comp],
                                            Gop_0_freq_DFT_vox_new[comp],atol=1e-10)))
    print('')
#
#                                                                              2D elasticity
# ==========================================================================================
def utility6():
    # Import modules
    import numpy as np
    import tensor.tensoroperations as top
    import tensor.matrixoperations as mop
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validation header
    print('\n\n' + 'Clarify questions about 2D plane strain elasticity')
    print(         '--------------------------------------------------')
    np.set_printoptions(linewidth = np.inf)
    np.set_printoptions(formatter={'float':'{: 11.4e}'.format})
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Young modulus and Poisson ratio
    E = 210e3
    v = 0.3
    # Compute Bulk modulus and shear modulus
    K = E/(3.0*(1.0 - 2.0*v))
    G = E/(2.0*(1.0+v))
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Print assumed elastic constants
    print('\n' + 'Assumed elastic properties:' + '\n')
    print('E =   ', '{: 11.4e}'.format(E))
    print('v =   ', '{: 11.4e}'.format(v))
    print('K =   ', '{: 11.4e}'.format(K))
    print('G =   ', '{: 11.4e}'.format(G))
    print('lam = ', '{: 11.4e}'.format(lam))
    print('miu = ', '{: 11.4e}'.format(miu))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + 92*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print header
    print('\n' + '3D standard elasticity (stored with Kelvin notation):' + '\n')
    # Set dimensions and components order
    n_dim = 3
    comp_order = ['11','22','33','12','23','13']
    # Set required fourth-order tensors
    soid,foid,fotransp,fosym,fodiagtrace,fodevproj,fodevprojsym = \
                                                               top.getidoperators(n_dim)
    # Compute consistent tangent modulus (Lamé parameters)
    consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus (Lamé parameters) matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (Lamé parameters)
    print('\n' + '>> consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym')
    print('\n' + 'consistent_tangent_mf (Lamé parameters):' + '\n')
    print(consistent_tangent_mf)
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (bulk and shear modulii)
    print('\n\n' + '>> consistent_tangent = K*fodiagtrace + 2.0*G*(fosym - ' + \
                                                                   '(1.0/3.0)*fodiagtrace)')
    print('\n' + 'consistent_tangent_mf (bulk and shear modulii):' + '\n')
    print(consistent_tangent_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + 92*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print header
    print('\n' + '2D plane strain elasticity (stored with Kelvin notation):' + '\n')
    # Set dimensions and components order
    n_dim = 2
    comp_order = ['11','22','12']
    # Set required fourth-order tensors
    soid,foid,fotransp,fosym,fodiagtrace,fodevproj,fodevprojsym = \
                                                               top.getidoperators(n_dim)
    # Compute consistent tangent modulus (Lamé parameters)
    consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus (Lamé parameters) matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (Lamé parameters)
    print('\n' + '>> consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym')
    print('\n' + 'consistent_tangent_mf (Lamé parameters):' + '\n')
    print(consistent_tangent_mf)
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (bulk and shear modulii)
    print('\n\n' + '>> consistent_tangent = K*fodiagtrace + 2.0*G*(fosym - ' + \
                                                                   '(1.0/3.0)*fodiagtrace)')
    print('\n' + 'consistent_tangent_mf (bulk and shear modulii):' + '\n')
    print(consistent_tangent_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + 92*'-')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print header
    print('\n' + '2D "bulk modulus" plane strain elasticity ' + \
                                                    '(stored with Kelvin notation):' + '\n')
    # Set dimensions and components order
    n_dim = 2
    comp_order = ['11','22','12']
    # Set required fourth-order tensors
    soid,foid,fotransp,fosym,fodiagtrace,_,_ = top.getidoperators(n_dim)
    # Set fourth-order deviatoric projection tensor (second order symmetric tensors)
    fodevprojsym = fosym - (1.0/2.0)*fodiagtrace
    # Compute 2D bulk modulus
    K2d = K + (1.0/3.0)*miu
    print('\n' + '>> K2d = K + (1.0/3.0)*miu =', '{:11.4e}'.format(K2d))
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K2d*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.gettensormf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (bulk and shear modulii)
    print('\n' + '>> consistent_tangent = K2d*fodiagtrace + 2.0*G*(fosym - ' + \
                                                                   '(1.0/2.0)*fodiagtrace)')
    print('\n' + 'consistent_tangent_mf (2d bulk modulus and shear modulus):' + '\n')
    print(consistent_tangent_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + 92*'-' + '\n')
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    utility7()
