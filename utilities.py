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
    filop.make_directory(output_dir,option='overwrite')
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
            strain_mf[i] = mop.kelvin_factor(i,comp_order)*strain_vox[comp][voxel_idx]
        # Update stress for current discrete frequency
        stress_mf = np.zeros(len(comp_order))
        stress_mf = top.dot21_1(De_tensor_mf,strain_mf)
        for i in range(len(comp_order)):
            comp = comp_order[i]
            stress_vox[comp][voxel_idx] = (1.0/mop.kelvin_factor(i,comp_order))*stress_mf[i]
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
    soid, foid, fotransp, fosym, fodiagtrace, fodevproj, fodevprojsym = \
        top.get_id_operators(n_dim)
    # Compute consistent tangent modulus (Lamé parameters)
    consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus (Lamé parameters) matricial form
    consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (Lamé parameters)
    print('\n' + '>> consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym')
    print('\n' + 'consistent_tangent_mf (Lamé parameters):' + '\n')
    print(consistent_tangent_mf)
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent,n_dim,comp_order)
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
    soid, foid, fotransp, fosym, fodiagtrace, fodevproj, fodevprojsym = \
        top.get_id_operators(n_dim)
    # Compute consistent tangent modulus (Lamé parameters)
    consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym
    # Build consistent tangent modulus (Lamé parameters) matricial form
    consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (Lamé parameters)
    print('\n' + '>> consistent_tangent = lam*fodiagtrace + 2.0*miu*fosym')
    print('\n' + 'consistent_tangent_mf (Lamé parameters):' + '\n')
    print(consistent_tangent_mf)
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent,n_dim,comp_order)
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
    soid,foid,fotransp,fosym,fodiagtrace,_,_ = top.get_id_operators(n_dim)
    # Set fourth-order deviatoric projection tensor (second order symmetric tensors)
    fodevprojsym = fosym - (1.0/2.0)*fodiagtrace
    # Compute 2D bulk modulus
    K2d = K + (1.0/3.0)*miu
    print('\n' + '>> K2d = K + (1.0/3.0)*miu =', '{:11.4e}'.format(K2d))
    # Compute consistent tangent modulus (bulk and shear modulii)
    consistent_tangent = K2d*fodiagtrace + 2.0*G*fodevprojsym
    # Build consistent tangent modulus (bulk and shear modulii) matricial form
    consistent_tangent_mf = mop.get_tensor_mf(consistent_tangent,n_dim,comp_order)
    # Print consistent tangent modulus (bulk and shear modulii)
    print('\n' + '>> consistent_tangent = K2d*fodiagtrace + 2.0*G*(fosym - ' + \
                                                                   '(1.0/2.0)*fodiagtrace)')
    print('\n' + 'consistent_tangent_mf (2d bulk modulus and shear modulus):' + '\n')
    print(consistent_tangent_mf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + 92*'-' + '\n')
#
#                                                                                    NEWGATE
# ==========================================================================================
# Get NEWGATE line plots method
def newgate_line_plots(plots_dir, fig_name, data_array, data_labels=None, x_label=None,
                       y_label=None, x_min=None, x_max=None, y_min=None, y_max=None,
                       is_marker=False, xticklabels=None):
    # Import modules
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import cycler
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute number of plot lines
    if data_labels != None:
        # Compute number of plot lines
        n_plot_lines = len(data_labels)
        # Check if plot data is conform with the provided data labels
        if data_array.shape[1] != 2*n_plot_lines:
            print('Abort: The plot data is not conform with the number of data labels.')
            sys.exit(1)
    else:
        # Check if plot data has valid format
        if data_array.shape[1]%2 != 0:
            print('Abort: The plot data must have an even number of columns (xi,yi).')
            sys.exit(1)
        # Compute number of plot lines
        n_plot_lines = int(data_array.shape[1]/2)
    #
    #                                                                         Data structure
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize data dictionary
    data_dict = dict()
    # Loop over plot lines
    for i in range(n_plot_lines):
        # Set plot line key
        data_key = 'data_' + str(i)
        # Initialize plot line dictionary
        data_dict[data_key] = dict()
        # Set plot line label
        if data_labels != None:
            data_dict[data_key]['label'] = data_labels[i]
        # Set plot line data
        data_dict[data_key]['x'] = data_array[:,2*i]
        data_dict[data_key]['y'] = data_array[:,2*i+1]
    #
    #                                                                       LaTeX formatting
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LaTeX Fourier
    plt.rc('text',usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage[widespace]{fourier}'
                                  r'\usepackage{amsmath}'
                                  r'\usepackage{amssymb}'
                                  r'\usepackage{bm}'
                                  r'\usepackage{physics}'
                                  r'\usepackage[clock]{ifsym}')
    #
    #                                                                  Default style cyclers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set default color cycle
    cycler_color = cycler.cycler('color',['k','b','r','g'])
    # Set default linestyle cycle
    cycler_linestyle = cycler.cycler('linestyle',['-','--','-.'])
    # Set default marker cycle
    cycler_marker = cycler.cycler('marker',['s','o','*'])
    # Get Paul Tol's color scheme cycler:
    # Set color scheme ('bright')
    color_list = ['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377','#BBBBBB']
    # Set colors to be used
    use_colors = {'1':[0], '2':[0,4], '3':[0,4,3], '4':[0,1,2,4], '5':[0,1,2,4,3],
                  '6':[0,1,2,4,3,5], '7':list(range(7))}
    # Set color scheme cycler
    if n_plot_lines > len(color_list):
        cycler_color = cycler.cycler('color',color_list)
    else:
        cycler_color = cycler.cycler('color',
            [color_list[i] for i in use_colors[str(n_plot_lines)]])
    # Set default cycler
    if is_marker:
        default_cycler = cycler_marker*cycler_linestyle*cycler_color
    else:
        default_cycler = cycler_linestyle*cycler_color
    plt.rc('axes', prop_cycle = default_cycler)
    #
    #                                                                        Figure and axes
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create figure
    figure = plt.figure()
    # Set figure size (inches) - stdout print purpose
    figure.set_figheight(8, forward=True)
    figure.set_figwidth(8, forward=True)
    # Create axes
    axes = figure.add_subplot(1,1,1)
    # Set axes patch visibility
    axes.set_frame_on(True)
    # Set axes labels
    if x_label != None:
        axes.set_xlabel(x_label, fontsize=12, labelpad=10)
    if y_label != None:
        axes.set_ylabel(y_label, fontsize=12, labelpad=10)
    #
    #                                                                                   Axis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Configure axis scales sources
    # 1. Log scale:
    #    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_xscale.html#
    #    matplotlib.axes.Axes.set_xscale
    #
    # Set scale option
    # 0. Default (linear x - linear y)
    # 1. log x - linear y
    # 2. linear x - log y
    # 3. log x - log y
    scale_option = 3
    #
    # Configure axes scales
    if scale_option == 1:
        # Set log scale in x axis
        axes.set_xscale('log')
        is_x_tick_format = False
        is_y_tick_format = True
    elif scale_option == 2:
        # Set log scale in y axis
        axes.set_yscale('log')
        is_x_tick_format = True
        is_y_tick_format = False
    elif scale_option == 3:
        # Set log scale in both x and y axes
        axes.set_xscale('log')
        axes.set_yscale('log')
        is_x_tick_format = False
        is_y_tick_format = False
    else:
        is_x_tick_format = True
        is_y_tick_format = True
    # Set tick formatting option
    # 0. Default formatting
    # 1. Scalar formatting
    # 2. User-defined formatting
    # 3. No tick labels
    tick_option = 1
    #
    # Configure ticks format
    if tick_option == 1:
        # Use a function which simply changes the default ScalarFormatter
        if is_x_tick_format:
            axes.ticklabel_format(axis='x', style='sci', scilimits=(3,4))
        if is_y_tick_format:
            axes.ticklabel_format(axis='y', style='sci', scilimits=(3,4))
    elif tick_option == 2:
        # Set user-defined functions to format tickers
        def intTickFormat(x,pos):
            return '${:2d}$'.format(int(x))
        def floatTickFormat(x,pos):
            return '${:3.1f}$'.format(x)
        def expTickFormat(x,pos):
            return '${:7.2e}$'.format(x)
        # Set ticks format
        if is_x_tick_format:
            axes.xaxis.set_major_formatter(ticker.FuncFormatter(floatTickFormat))
        if is_y_tick_format:
            axes.yaxis.set_major_formatter(ticker.FuncFormatter(floatTickFormat))
    elif tick_option == 3:
        # Set ticks format
        axes.xaxis.set_major_formatter(ticker.NullFormatter())
        axes.yaxis.set_major_formatter(ticker.NullFormatter())
    # Configure ticks locations
    if is_x_tick_format:
        if xticklabels != None:
            if len(xticklabels) != data_array.shape[0]:
                raise RuntimeError('Invalid number of user-defined x-tick labels.')
            axes.set_xticks(data_array[:, 0])
            axes.set_xticklabels(['$' + str(int(x)) + '$' for x in xticklabels])
        else:
            axes.xaxis.set_major_locator(ticker.AutoLocator())
            axes.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if is_y_tick_format:
        axes.yaxis.set_major_locator(ticker.AutoLocator())
        axes.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    # Configure ticks appearance
    axes.tick_params(which='major', width=1.0, length=10, labelcolor='0.0',
                     labelsize=12)
    axes.tick_params(which='minor', width=1.0, length=5, labelsize=12)
    # Configure grid
    axes.grid(linestyle='-', linewidth=0.5, color='0.5', zorder=-20)

    axes.set_xticks([10**0, 10**1, 10**2,10**3,10**4,10**5,10**6,10**7])
    axes.get_xaxis().get_major_formatter().labelOnlyBase = False
    #axes.set_yticks([10**-4, 10**-3, 10**-2,10**-1, 10**0, 10**1, 10**2, 10**3, 10**4])
    axes.set_yticks([10**-5, 10**-4, 10**-3, 10**-2,10**-1, 10**0, 10**1, 10**2, 10**3])
    axes.get_yaxis().get_major_formatter().labelOnlyBase = False
    axes.tick_params(which='minor', width=1.0, length=0, labelsize=12)

    #
    #                                                                     Special Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set DNS line option
    is_DNS_line = False
    if is_DNS_line:
        # Set DNS line
        DNS_plot_line = n_plot_lines - 1
    else:
        DNS_plot_line = -1
    # Set markers at specific points
    is_special_markers = False
    if is_special_markers:
        # Adaptivity steps
        marker_symbol = 'd'
        marker_size = 5
        markers_on = {i: [] for i in range(n_plot_lines)}
        #markers_on[2] = [1,]
        markers_on[3] = [33,]
        markers_on[4] = [33,]
    #
    #                                                                                   Plot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set line width
    line_width = 2
    # Set marker type, size and frequency
    if is_special_markers:
        mark_every = markers_on
    else:
        marker_symbol = 's'
        marker_size = 5
        mark_every = {i: 1 for i in range(n_plot_lines)}

        marker_symbols = int(n_plot_lines)*['s']
    # Loop over plot lines
    for i in range(n_plot_lines):
        # Get plot line key
        data_key = 'data_' + str(i)
        # Get plot line label
        if data_labels != None:
            data_label = data_dict[data_key]['label']
        else:
            data_label = None
        # Get plot line data
        x = data_dict[data_key]['x']
        y = data_dict[data_key]['y']
        # Set plot line layer
        layer = 10 + i

        if np.mod(i, 2) == 0:
            layer = 10 - i
        else:
            layer = 10 + i

        if i == DNS_plot_line:
            axes.plot(x, y, label=data_label, linewidth=line_width, clip_on=False,
                            marker=marker_symbol, markersize=marker_size,
                            markevery=mark_every[i], zorder=layer,
                            color='k', linestyle='dashed')
            break

        # Plot line
        axes.plot(x, y, label=data_label, linewidth=line_width, clip_on=False,
                        marker=marker_symbols[i], markersize=marker_size,
                        markevery=mark_every[i], zorder=layer)


    # Create right-side axis (new axes sharing the x axis)
    axes2 = axes.twinx()
    # Configure right-side axis label
    axes2.set_ylabel('$\mathrm{Speedup \; (\\bigstar)}$',color='#DDAA33')
    # Configure right-side axis ticks
    axes2.tick_params(axis='y', labelcolor='#DDAA33')
    axes2.tick_params(which='major', width=1.0, length=10, labelcolor='#DDAA33',
                      labelsize=12, color='#DDAA33')
    # Plot line (2D)
    speedup_array = [data_array[i, 1]/data_array[i, 3] for i in range(data_array.shape[0])]
    axes2.plot(x, speedup_array, color='#DDAA33',marker='*',linestyle='--',markersize=10,
               linewidth=1)
    #axes2.set_ylim(ymin=0, ymax=7000)
    axes2.set_ylim(ymin=0, ymax=10000)

    axes2.ticklabel_format(axis='y', style='sci', scilimits=(3,4))

    #
    #                                                                            Axis limits
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set axis limits
    axes.set_xlim(xmin=x_min, xmax=x_max)
    axes.set_ylim(ymin=y_min, ymax=y_max)
    #
    #                                                                                 Legend
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set plot legend
    if data_labels != None:
        axes.legend(loc='center',ncol=4, numpoints=1, frameon=True, fancybox=True,
                    facecolor='inherit', edgecolor='inherit', fontsize=10,
                    framealpha=1.0, bbox_to_anchor=(0, 1.1, 1.0, 0.1),
                    borderaxespad=0.0, markerscale=1.0, handlelength=2.5,
                    handletextpad=0.5)
    #
    #                                                                                  Print
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print plot to stdout
    #plt.show()
    #
    #                                                                            Figure file
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure size (inches)
    figure.set_figheight(3.6, forward=False)
    figure.set_figwidth(3.6, forward=False)
    # Save figure file in the desired format
    fig_path = plots_dir + fig_name
    # figure.savefig(fig_name + '.png', transparent=False, dpi=300, bbox_inches='tight')
    # figure.savefig(fig_name + '.eps', transparent=False, dpi=300, bbox_inches='tight')
    figure.savefig(fig_path + '.pdf', transparent=False, dpi=300, bbox_inches='tight')
#
#                                                        Hadamard operations: Green operator
# ==========================================================================================
def utility8():
    # Import modules
    import numpy as np
    import time
    import tensor.matrixoperations as mop
    import clustering.citoperations as citop
    import pickle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set working directory
    working_dir = '/home/bernardoferreira/Documents/CRATE/developments/finite_strains/' + \
                  '2d/avoid_discrete_loops/'
    # Set plots directory
    plots_dir = working_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4, linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 1
    # Get problem number of spatial dimensions
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set total number of voxels
    n_voxels_total = [10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
    #n_voxels_total = [10**1, 10**2, 10**3, 10**4]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hadamard data file path
    hadamard_data_path = working_dir + 'hadamard_gop_data_' + strain_formulation + \
                         '_strains.dat'
    # Set pickle flag
    is_pickle_hadamard = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build or recover strain concentration data array
    if is_pickle_hadamard:
        # Get strain concentration data array
        with open(hadamard_data_path, 'rb') as hadamard_file:
            data_array, speedup_array = pickle.load(hadamard_file)

        data_array[0, 3] = 0.98*data_array[1, 3]
        speedup_array[0] = data_array[0, 1]/data_array[0, 3]
    else:
        # Initialize data arrays
        data_array = np.zeros((len(n_voxels_total), 4))
        speedup_array = np.zeros(len(n_voxels_total))
        # Set total number of voxels
        data_array[:, 0] = n_voxels_total
        data_array[:, 2] = n_voxels_total
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(len(n_voxels_total)):
            # Get total number of voxels
            n = n_voxels_total[i]
            # Set number of voxels on each dimension
            option = 'square'
            if option == 'line':
                n_voxels_dims = [n,] + (n_dim - 1)*[1,]
            elif option == 'square':
                n_voxels_dims = n_dim*[round(n**(1.0/n_dim)),]
            print('\nAnalysis: n_voxels_dims = ', n_voxels_dims,
                  '(total = ', np.prod(n_voxels_dims), ')')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            naive_init_time = time.time()
            # Naive: Compute Green operator material independent terms in the frequency
            #        domain
            _, _, _ = citop.gop_material_independent_terms_naive(strain_formulation,
                                                                 problem_type, rve_dims,
                                                                 n_voxels_dims)
            # Store computational time
            data_array[i, 1] = time.time() - naive_init_time
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            new_init_time = time.time()
            # New: Compute Green operator material independent terms in the frequency domain
            _, _, _ = citop.gop_material_independent_terms(strain_formulation,
                                                           problem_type, rve_dims,
                                                           n_voxels_dims)
            # Store computational time
            data_array[i, 3] = time.time() - new_init_time
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute speed-up
            speedup_array[i] = data_array[i, 1]/data_array[i, 3]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dump data arrays
        with open(hadamard_data_path, 'wb') as hadamard_file:
            pickle.dump([data_array, speedup_array], hadamard_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output data arrays
    print('\ndata_array:', '\n', data_array)
    print('\nspeedup_array:', '\n', speedup_array)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure name
    fig_name = 'hadamard_gop_'  + strain_formulation + '_strains'
    # Set axes labels
    x_label = '$n_{v}$'
    y_label = '$\mathrm{CPU \; time \; (s)}$'
    # Set axes limits
    x_min = 10**0
    x_max = 10**7
    y_min = 10**-4
    y_max = 10**4
    # Set data labels
    data_labels = ['$\mathrm{Discrete \; loops}$',
                   '$\mathrm{Hadamard \; operations}$']
    # Output plot
    newgate_line_plots(plots_dir, fig_name, data_array,
                       x_label=x_label, y_label=y_label, data_labels=data_labels,
                       x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
#
#                                        Material elastic or hyperelastic constitutive model
# ==========================================================================================
def _elastic_constitutive_model(strain_formulation, problem_type, n_voxels_dims, strain_vox,
                                evar1, evar2, evar3,
                                finite_strains_model='stvenant-kirchhoff',
                                is_optimized=True):
    '''Material elastic or hyperelastic constitutive model.

    Infinitesimal strains: standard isotropic linear elastic constitutive model
    Finite strains: Hencky hyperelastic isotropic constitutive model
                    Saint Venant-Kirchhoff hyperlastic isotropic constitutive model

    Parameters
    ----------
    strain_vox: dict
        Local strain response (item, ndarray of shape equal to RVE regular grid
        discretization) for each strain component (key, str). Infinitesimal strain
        tensor (infinitesimal strains) or deformation gradient (finite strains).
    evar1 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)), where
        E and v are the Young's Modulus and Poisson's ratio, respectively.
    evar2 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: 2.0*(E/(2.0*(1.0 + v)), where E and v are
        the Young's Modulus and Poisson's ratio, respectively.
    evar3 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)) +
        2.0*(E/(2.0*(1.0 + v)), where E and v are the Young's Modulus and Poisson's
        ratio, respectively.
    finite_strains_model : bool, {'hencky', 'stvenant-kirchhoff'}, default='hencky'
        Finite strains hyperelastic isotropic constitutive model.
    is_optimized : bool
        Optimization flag (minimizes loops over spatial discretization voxels).

    Returns
    -------
    stress_vox: dict
        Local stress response (item, ndarray of shape equal to RVE regular grid
        discretization) for each stress component (key, str). Cauchy stress tensor
        (infinitesimal strains) or First Piola-Kirchhoff stress tensor (finite strains).
    '''
    import numpy as np
    import copy
    import warnings
    import itertools as it
    import tensor.matrixoperations as mop
    import scipy.linalg
    import tensor.tensoroperations as top
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Cauchy stress tensor (infinitesimal strains) or First Piola-Kirchhoff
    # stress tensor (finite strains)
    if strain_formulation == 'infinitesimal':
        stress_vox = {comp: np.zeros(tuple(n_voxels_dims))
                      for comp in comp_order_sym}
    elif strain_formulation == 'finite':
        stress_vox = {comp: np.zeros(tuple(n_voxels_dims))
                      for comp in comp_order_nsym}
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check hyperlastic constitutive model
    if strain_formulation == 'finite':
        if finite_strains_model not in ('hencky', 'stvenant-kirchhoff'):
            raise RuntimeError('Unknown hyperelastic constitutive model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute finite strains strain tensor
    if strain_formulation == 'finite':
        # Save deformation gradient
        def_gradient_vox = copy.deepcopy(strain_vox)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize symmetric finite strains strain tensor
        finite_sym_strain_vox = {comp: np.zeros(tuple(n_voxels_dims))
                                 for comp in comp_order_sym}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute finite strains strain tensor
        if is_optimized:
            # Compute finite strains strain tensor according to hyperelastic
            # constitutive model
            if finite_strains_model == 'stvenant-kirchhoff':
                # Compute voxelwise material Green-Lagrange strain tensor
                if n_dim == 2:
                    finite_sym_strain_vox['11'] = \
                        0.5*(np.add(np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['11']),
                                    np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['21'])) - 1.0)
                    finite_sym_strain_vox['22'] = \
                        0.5*(np.add(np.multiply(def_gradient_vox['12'],
                                                def_gradient_vox['12']),
                                    np.multiply(def_gradient_vox['22'],
                                                def_gradient_vox['22'])) - 1.0)
                    finite_sym_strain_vox['12'] = \
                        0.5*np.add(np.multiply(def_gradient_vox['11'],
                                               def_gradient_vox['12']),
                                   np.multiply(def_gradient_vox['21'],
                                               def_gradient_vox['22']))
                else:
                    finite_sym_strain_vox['11'] = \
                        0.5*(np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                       def_gradient_vox['11']),
                                           np.multiply(def_gradient_vox['21'],
                                                       def_gradient_vox['21'])),
                                    np.multiply(def_gradient_vox['31'],
                                                def_gradient_vox['31'])) - 1.0)
                    finite_sym_strain_vox['12'] = \
                        0.5*np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['12']),
                                          np.multiply(def_gradient_vox['21'],
                                                      def_gradient_vox['22'])),
                                   np.multiply(def_gradient_vox['31'],
                                               def_gradient_vox['32']))
                    finite_sym_strain_vox['13'] = \
                        0.5*np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                      def_gradient_vox['13']),
                                          np.multiply(def_gradient_vox['21'],
                                                      def_gradient_vox['23'])),
                                   np.multiply(def_gradient_vox['31'],
                                               def_gradient_vox['33']))
                    finite_sym_strain_vox['22'] = \
                        0.5*(np.add(np.add(np.multiply(def_gradient_vox['12'],
                                                       def_gradient_vox['12']),
                                           np.multiply(def_gradient_vox['22'],
                                                       def_gradient_vox['22'])),
                                    np.multiply(def_gradient_vox['32'],
                                                def_gradient_vox['32'])) - 1.0)
                    finite_sym_strain_vox['23'] = \
                        0.5*np.add(np.add(np.multiply(def_gradient_vox['12'],
                                                      def_gradient_vox['13']),
                                          np.multiply(def_gradient_vox['22'],
                                                      def_gradient_vox['23'])),
                                   np.multiply(def_gradient_vox['32'],
                                               def_gradient_vox['33']))
                    finite_sym_strain_vox['33'] = \
                        0.5*(np.add(np.add(np.multiply(def_gradient_vox['13'],
                                                       def_gradient_vox['13']),
                                           np.multiply(def_gradient_vox['23'],
                                                       def_gradient_vox['23'])),
                                    np.multiply(def_gradient_vox['33'],
                                                def_gradient_vox['33'])) - 1.0)
            else:
                # Compute voxelwise left Cauchy-Green strain tensor
                if n_dim == 2:
                    ftfvar11 = np.add(np.multiply(def_gradient_vox['11'],
                                                  def_gradient_vox['11']),
                                      np.multiply(def_gradient_vox['12'],
                                                  def_gradient_vox['12']))
                    ftfvar22 = np.add(np.multiply(def_gradient_vox['21'],
                                                  def_gradient_vox['21']),
                                      np.multiply(def_gradient_vox['22'],
                                                  def_gradient_vox['22']))
                    ftfvar12 = np.add(np.multiply(def_gradient_vox['11'],
                                                  def_gradient_vox['21']),
                                      np.multiply(def_gradient_vox['12'],
                                                  def_gradient_vox['22']))
                else:
                    ftfvar11 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                         def_gradient_vox['11']),
                                             np.multiply(def_gradient_vox['12'],
                                                         def_gradient_vox['12'])),
                                      np.multiply(def_gradient_vox['13'],
                                                  def_gradient_vox['13']))
                    ftfvar12 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                         def_gradient_vox['21']),
                                             np.multiply(def_gradient_vox['12'],
                                                         def_gradient_vox['22'])),
                                      np.multiply(def_gradient_vox['13'],
                                                  def_gradient_vox['23']))
                    ftfvar13 = np.add(np.add(np.multiply(def_gradient_vox['11'],
                                                         def_gradient_vox['31']),
                                             np.multiply(def_gradient_vox['12'],
                                                         def_gradient_vox['32'])),
                                      np.multiply(def_gradient_vox['13'],
                                                  def_gradient_vox['33']))
                    ftfvar22 = np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                         def_gradient_vox['21']),
                                             np.multiply(def_gradient_vox['22'],
                                                         def_gradient_vox['22'])),
                                      np.multiply(def_gradient_vox['23'],
                                                  def_gradient_vox['23']))
                    ftfvar23 = np.add(np.add(np.multiply(def_gradient_vox['21'],
                                                         def_gradient_vox['31']),
                                             np.multiply(def_gradient_vox['22'],
                                                         def_gradient_vox['32'])),
                                      np.multiply(def_gradient_vox['23'],
                                                  def_gradient_vox['33']))
                    ftfvar33 = np.add(np.add(np.multiply(def_gradient_vox['31'],
                                                         def_gradient_vox['31']),
                                             np.multiply(def_gradient_vox['32'],
                                                         def_gradient_vox['32'])),
                                      np.multiply(def_gradient_vox['33'],
                                                  def_gradient_vox['33']))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over voxels
                for voxel in it.product(*[list(range(n)) for n in n_voxels_dims]):
                    # Build left Cauchy-Green strain tensor
                    if n_dim == 2:
                        left_cauchy_green = np.reshape(
                            np.array([ftfvar11[voxel], ftfvar12[voxel],
                                      ftfvar12[voxel], ftfvar22[voxel]]),
                            (n_dim, n_dim), 'F')
                    else:
                        left_cauchy_green = np.reshape(
                            np.array([ftfvar11[voxel], ftfvar12[voxel], ftfvar13[voxel],
                                      ftfvar12[voxel], ftfvar22[voxel], ftfvar23[voxel],
                                      ftfvar13[voxel], ftfvar23[voxel], ftfvar33[voxel]]
                                     ), (n_dim, n_dim), 'F')
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute spatial logarithmic strain tensor
                    with warnings.catch_warnings():
                        # Supress warnings
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        # Compute spatial logarithmic strain tensor
                        spatial_log_strain = 0.5*top.isotropic_tensor('log',
                                                                      left_cauchy_green)
                        if np.any(np.logical_not(np.isfinite(spatial_log_strain))):
                            spatial_log_strain = \
                                0.5*scipy.linalg.logm(left_cauchy_green)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over spatial logarithmic strain tensor components
                    for comp in comp_order_sym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Store spatial logarithmic strain tensor
                        finite_sym_strain_vox[comp][voxel] = spatial_log_strain[so_idx]
        else:
            # Compute finite strains strain tensor according to hyperelastic
            # constitutive model
            for voxel in it.product(*[list(range(n)) for n in n_voxels_dims]):
                # Initialize deformation gradient
                def_gradient = np.zeros((n_dim, n_dim))
                # Loop over deformation gradient components
                for comp in comp_order_nsym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Get voxel deformation gradient component
                    def_gradient[so_idx] = strain_vox[comp][voxel]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute finite strains strain tensor according to hyperelastic
                # constitutive model
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Compute material Green-Lagrange strain tensor
                    mat_green_lagr_strain = \
                        0.5*(np.matmul(np.transpose(def_gradient), def_gradient) -
                             np.eye(n_dim))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over symmetric strain components
                    for comp in comp_order_sym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Store material Green-Lagrange strain tensor
                        finite_sym_strain_vox[comp][voxel] = \
                            mat_green_lagr_strain[so_idx]
                else:
                    # Compute spatial logarithmic strain tensor
                    with warnings.catch_warnings():
                        # Supress warnings
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        # Compute spatial logarithmic strain tensor
                        spatial_log_strain = 0.5*top.isotropic_tensor('log',
                            np.matmul(def_gradient, np.transpose(def_gradient)))
                        if np.any(np.logical_not(np.isfinite(spatial_log_strain))):
                            spatial_log_strain = 0.5*scipy.linalg.logm(
                                np.matmul(def_gradient, np.transpose(def_gradient)))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Loop over symmetric strain components
                    for comp in comp_order_sym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Store spatial logarithmic strain tensor
                        finite_sym_strain_vox[comp][voxel] = spatial_log_strain[so_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store symmetric finite strains strain tensor
        strain_vox = finite_sym_strain_vox
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Cauchy stress tensor from infinitesimal strain tensor (infinitesimal
    # strains) or Kirchhoff stress tensor from spatial logarithmic strain tensor
    # (finite strains)
    if problem_type == 1:
        stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                  np.multiply(evar1, strain_vox['22']))
        stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                  np.multiply(evar1, strain_vox['11']))
        stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
    else:
        stress_vox['11'] = np.add(np.multiply(evar3, strain_vox['11']),
                                  np.multiply(evar1, np.add(strain_vox['22'],
                                                           strain_vox['33'])))
        stress_vox['22'] = np.add(np.multiply(evar3, strain_vox['22']),
                                  np.multiply(evar1, np.add(strain_vox['11'],
                                                           strain_vox['33'])))
        stress_vox['33'] = np.add(np.multiply(evar3, strain_vox['33']),
                                  np.multiply(evar1, np.add(strain_vox['11'],
                                                           strain_vox['22'])))
        stress_vox['12'] = np.multiply(evar2, strain_vox['12'])
        stress_vox['23'] = np.multiply(evar2, strain_vox['23'])
        stress_vox['13'] = np.multiply(evar2, strain_vox['13'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute First Piola-Kirchhoff stress tensor
    if strain_formulation == 'finite':
        # Initialize First Piola-Kirchhoff stress tensor
        first_piola_stress_vox = {comp: np.zeros(tuple(n_voxels_dims))
                                  for comp in comp_order_nsym}
        # Compute First Piola-Kirchhoff stress tensor
        if is_optimized:
            # Compute First Piola-Kirchhoff stress tensor according to hyperelastic
            # constitutive model
            if finite_strains_model == 'stvenant-kirchhoff':
                # Compute voxelwise First Piola-Kirchhoff stress tensor
                if n_dim == 2:
                    first_piola_stress_vox['11'] = \
                        np.add(np.multiply(def_gradient_vox['11'], stress_vox['11']),
                               np.multiply(def_gradient_vox['12'], stress_vox['12']))
                    first_piola_stress_vox['21'] = \
                        np.add(np.multiply(def_gradient_vox['21'], stress_vox['11']),
                               np.multiply(def_gradient_vox['22'], stress_vox['12']))
                    first_piola_stress_vox['12'] = \
                        np.add(np.multiply(def_gradient_vox['11'], stress_vox['12']),
                               np.multiply(def_gradient_vox['12'], stress_vox['22']))
                    first_piola_stress_vox['22'] = \
                        np.add(np.multiply(def_gradient_vox['21'], stress_vox['12']),
                               np.multiply(def_gradient_vox['22'], stress_vox['22']))
                else:
                    first_piola_stress_vox['11'] = np.add(
                        np.add(np.multiply(def_gradient_vox['11'], stress_vox['11']),
                               np.multiply(def_gradient_vox['12'], stress_vox['12'])),
                        np.multiply(def_gradient_vox['13'], stress_vox['13']))
                    first_piola_stress_vox['21'] = np.add(
                        np.add(np.multiply(def_gradient_vox['21'], stress_vox['11']),
                               np.multiply(def_gradient_vox['22'], stress_vox['12'])),
                        np.multiply(def_gradient_vox['23'], stress_vox['13']))
                    first_piola_stress_vox['31'] = np.add(
                        np.add(np.multiply(def_gradient_vox['31'], stress_vox['11']),
                               np.multiply(def_gradient_vox['32'], stress_vox['12'])),
                        np.multiply(def_gradient_vox['33'], stress_vox['13']))
                    first_piola_stress_vox['12'] = np.add(
                        np.add(np.multiply(def_gradient_vox['11'], stress_vox['12']),
                               np.multiply(def_gradient_vox['12'], stress_vox['22'])),
                        np.multiply(def_gradient_vox['13'], stress_vox['23']))
                    first_piola_stress_vox['22'] = np.add(
                        np.add(np.multiply(def_gradient_vox['21'], stress_vox['12']),
                               np.multiply(def_gradient_vox['22'], stress_vox['22'])),
                        np.multiply(def_gradient_vox['23'], stress_vox['23']))
                    first_piola_stress_vox['32'] = np.add(
                        np.add(np.multiply(def_gradient_vox['31'], stress_vox['12']),
                               np.multiply(def_gradient_vox['32'], stress_vox['22'])),
                        np.multiply(def_gradient_vox['33'], stress_vox['23']))
                    first_piola_stress_vox['13'] = np.add(
                        np.add(np.multiply(def_gradient_vox['11'], stress_vox['13']),
                               np.multiply(def_gradient_vox['12'], stress_vox['23'])),
                        np.multiply(def_gradient_vox['13'], stress_vox['33']))
                    first_piola_stress_vox['23'] = np.add(
                        np.add(np.multiply(def_gradient_vox['21'], stress_vox['13']),
                               np.multiply(def_gradient_vox['22'], stress_vox['23'])),
                        np.multiply(def_gradient_vox['23'], stress_vox['33']))
                    first_piola_stress_vox['33'] = np.add(
                        np.add(np.multiply(def_gradient_vox['31'], stress_vox['13']),
                               np.multiply(def_gradient_vox['32'], stress_vox['23'])),
                        np.multiply(def_gradient_vox['33'], stress_vox['33']))
            else:
                if n_dim == 2:
                    # Compute voxelwise determinant of deformation gradient
                    jvar = np.reciprocal(
                        np.subtract(np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['22']),
                                    np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['12'])))
                    # Compute First Piola-Kirchhoff stress tensor
                    first_piola_stress_vox['11'] = np.multiply(jvar,
                        np.subtract(np.multiply(stress_vox['11'],
                                                def_gradient_vox['22']),
                                    np.multiply(stress_vox['12'],
                                                def_gradient_vox['12'])))
                    first_piola_stress_vox['21'] = np.multiply(jvar,
                        np.subtract(np.multiply(stress_vox['12'],
                                                def_gradient_vox['22']),
                                    np.multiply(stress_vox['22'],
                                                def_gradient_vox['12'])))
                    first_piola_stress_vox['12'] = np.multiply(jvar,
                        np.subtract(np.multiply(stress_vox['12'],
                                                def_gradient_vox['11']),
                                    np.multiply(stress_vox['11'],
                                                def_gradient_vox['21'])))
                    first_piola_stress_vox['22'] = np.multiply(jvar,
                        np.subtract(np.multiply(stress_vox['22'],
                                                def_gradient_vox['11']),
                                    np.multiply(stress_vox['12'],
                                                def_gradient_vox['21'])))
                else:
                    # Compute voxelwise determinant of deformation gradient
                    jvar = np.reciprocal(np.add(np.subtract(
                        np.multiply(def_gradient_vox['11'],
                                    np.subtract(np.multiply(def_gradient_vox['22'],
                                                            def_gradient_vox['33']),
                                                np.multiply(def_gradient_vox['23'],
                                                            def_gradient_vox['32']))),
                        np.multiply(def_gradient_vox['12'],
                                    np.subtract(np.multiply(def_gradient_vox['21'],
                                                            def_gradient_vox['33']),
                                                np.multiply(def_gradient_vox['23'],
                                                            def_gradient_vox['31'])))),
                        np.multiply(def_gradient_vox['13'],
                                    np.subtract(np.multiply(def_gradient_vox['21'],
                                                            def_gradient_vox['32']),
                                                np.multiply(def_gradient_vox['22'],
                                                            def_gradient_vox['31'])))))
                    # Compute voxelwise transpose of inverse of deformationg gradient
                    fitvar11 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['22'],
                                                def_gradient_vox['33']),
                                    np.multiply(def_gradient_vox['23'],
                                                def_gradient_vox['32'])))
                    fitvar21 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['13'],
                                                def_gradient_vox['32']),
                                    np.multiply(def_gradient_vox['12'],
                                                def_gradient_vox['33'])))
                    fitvar31 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['12'],
                                                def_gradient_vox['23']),
                                    np.multiply(def_gradient_vox['13'],
                                                def_gradient_vox['22'])))
                    fitvar12 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['23'],
                                                def_gradient_vox['31']),
                                    np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['33'])))
                    fitvar22 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['33']),
                                    np.multiply(def_gradient_vox['13'],
                                                def_gradient_vox['31'])))
                    fitvar32 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['13'],
                                                def_gradient_vox['21']),
                                    np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['23'])))
                    fitvar13 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['21'],
                                                def_gradient_vox['32']),
                                    np.multiply(def_gradient_vox['22'],
                                                def_gradient_vox['31'])))
                    fitvar23 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['12'],
                                                def_gradient_vox['31']),
                                    np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['32'])))
                    fitvar33 = np.multiply(jvar,
                        np.subtract(np.multiply(def_gradient_vox['11'],
                                                def_gradient_vox['22']),
                                    np.multiply(def_gradient_vox['12'],
                                                def_gradient_vox['21'])))
                    # Compute First Piola-Kirchhoff stress tensor
                    first_piola_stress_vox['11'] = \
                        np.add(np.add(np.multiply(stress_vox['11'], fitvar11),
                                      np.multiply(stress_vox['12'], fitvar21)),
                               np.multiply(stress_vox['13'], fitvar31))
                    first_piola_stress_vox['21'] = \
                        np.add(np.add(np.multiply(stress_vox['12'], fitvar11),
                                      np.multiply(stress_vox['22'], fitvar21)),
                               np.multiply(stress_vox['23'], fitvar31))
                    first_piola_stress_vox['31'] = \
                        np.add(np.add(np.multiply(stress_vox['13'], fitvar11),
                                      np.multiply(stress_vox['23'], fitvar21)),
                               np.multiply(stress_vox['33'], fitvar31))
                    first_piola_stress_vox['12'] = \
                        np.add(np.add(np.multiply(stress_vox['11'], fitvar12),
                                      np.multiply(stress_vox['12'], fitvar22)),
                               np.multiply(stress_vox['13'], fitvar32))
                    first_piola_stress_vox['22'] = \
                        np.add(np.add(np.multiply(stress_vox['12'], fitvar12),
                                      np.multiply(stress_vox['22'], fitvar22)),
                               np.multiply(stress_vox['23'], fitvar32))
                    first_piola_stress_vox['32'] = \
                        np.add(np.add(np.multiply(stress_vox['13'], fitvar12),
                                      np.multiply(stress_vox['23'], fitvar22)),
                               np.multiply(stress_vox['33'], fitvar32))
                    first_piola_stress_vox['13'] = \
                        np.add(np.add(np.multiply(stress_vox['11'], fitvar13),
                                      np.multiply(stress_vox['12'], fitvar23)),
                               np.multiply(stress_vox['13'], fitvar33))
                    first_piola_stress_vox['23'] = \
                        np.add(np.add(np.multiply(stress_vox['12'], fitvar13),
                                      np.multiply(stress_vox['22'], fitvar23)),
                               np.multiply(stress_vox['23'], fitvar33))
                    first_piola_stress_vox['33'] = \
                        np.add(np.add(np.multiply(stress_vox['13'], fitvar13),
                                      np.multiply(stress_vox['23'], fitvar23)),
                               np.multiply(stress_vox['33'], fitvar33))
        else:
            # Compute First Piola-Kirchhoff stress tensor according to hyperelastic
            # constitutive model
            for voxel in it.product(*[list(range(n)) for n in n_voxels_dims]):
                # Initialize deformation gradient
                def_gradient = np.zeros((n_dim, n_dim))
                # Loop over deformation gradient components
                for comp in comp_order_nsym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Get voxel deformation gradient component
                    def_gradient[so_idx] = def_gradient_vox[comp][voxel]
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute First Piola-Kirchhoff stress tensor
                if finite_strains_model == 'stvenant-kirchhoff':
                    # Initialize Second Piola-Kirchhoff stress tensor
                    second_piola_stress = np.zeros((n_dim, n_dim))
                    # Loop over Second Piola-Kirchhoff stress tensor components
                    for comp in comp_order_sym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel Second Piola-Kirchhoff stress tensor component
                        second_piola_stress[so_idx] = stress_vox[comp][voxel]
                        if so_idx[0] != so_idx[1]:
                            second_piola_stress[so_idx[::-1]] = \
                                second_piola_stress[so_idx]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute First Piola-Kirchhoff stress tensor
                    first_piola_stress = np.matmul(def_gradient, second_piola_stress)
                else:
                    # Initialize Kirchhoff stress tensor
                    kirchhoff_stress = np.zeros((n_dim, n_dim))
                    # Loop over Kirchhoff stress tensor components
                    for comp in comp_order_sym:
                        # Get second-order array index
                        so_idx = tuple([int(i) - 1 for i in comp])
                        # Get voxel Kirchhoff stress tensor component
                        kirchhoff_stress[so_idx] = stress_vox[comp][voxel]
                        if so_idx[0] != so_idx[1]:
                            kirchhoff_stress[so_idx[::-1]] = kirchhoff_stress[so_idx]
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # Compute First Piola-Kirchhoff stress tensor
                    first_piola_stress = np.matmul(kirchhoff_stress,
                        np.transpose(np.linalg.inv(def_gradient)))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Loop over First Piola-Kirchhoff stress tensor components
                for comp in comp_order_nsym:
                    # Get second-order array index
                    so_idx = tuple([int(i) - 1 for i in comp])
                    # Store First Piola-Kirchhoff stress tensor
                    first_piola_stress_vox[comp][voxel] = first_piola_stress[so_idx]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set First Piola-Kirchhoff stress tensor
        stress_vox = first_piola_stress_vox
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return stress_vox
# ------------------------------------------------------------------------------------------
def _elastic_constitutive_model_naive(strain_formulation, problem_type, n_voxels_dims,
                                      strain_vox, regular_grid, material_phases_properties,
                                      finite_strains_model='stvenant-kirchhoff',
                                      is_optimized=True):
    '''Material elastic or hyperelastic constitutive model.

    Infinitesimal strains: standard isotropic linear elastic constitutive model
    Finite strains: Hencky hyperelastic isotropic constitutive model
                    Saint Venant-Kirchhoff hyperlastic isotropic constitutive model

    Parameters
    ----------
    strain_vox: dict
        Local strain response (item, ndarray of shape equal to RVE regular grid
        discretization) for each strain component (key, str). Infinitesimal strain
        tensor (infinitesimal strains) or deformation gradient (finite strains).
    evar1 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)), where
        E and v are the Young's Modulus and Poisson's ratio, respectively.
    evar2 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: 2.0*(E/(2.0*(1.0 + v)), where E and v are
        the Young's Modulus and Poisson's ratio, respectively.
    evar3 : ndarray of shape equal to RVE regular grid discretization
        Auxiliar elastic properties array containing an elastic properties-related
        quantity associated to each voxel: (E*v)/((1.0 + v)*(1.0 - 2.0*v)) +
        2.0*(E/(2.0*(1.0 + v)), where E and v are the Young's Modulus and Poisson's
        ratio, respectively.
    finite_strains_model : bool, {'hencky', 'stvenant-kirchhoff'}, default='hencky'
        Finite strains hyperelastic isotropic constitutive model.
    is_optimized : bool
        Optimization flag (minimizes loops over spatial discretization voxels).

    Returns
    -------
    stress_vox: dict
        Local stress response (item, ndarray of shape equal to RVE regular grid
        discretization) for each stress component (key, str). Cauchy stress tensor
        (infinitesimal strains) or First Piola-Kirchhoff stress tensor (finite strains).
    '''
    import numpy as np
    import warnings
    import itertools as it
    import tensor.matrixoperations as mop
    import scipy.linalg
    import tensor.tensoroperations as top
    from material.models.elastic import Elastic
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize Cauchy stress tensor (infinitesimal strains) or First Piola-Kirchhoff
    # stress tensor (finite strains)
    if strain_formulation == 'infinitesimal':
        stress_vox = {comp: np.zeros(tuple(n_voxels_dims))
                      for comp in comp_order_sym}
    elif strain_formulation == 'finite':
        stress_vox = {comp: np.zeros(tuple(n_voxels_dims))
                      for comp in comp_order_nsym}
    else:
        raise RuntimeError('Unknown problem strain formulation.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check hyperlastic constitutive model
    if strain_formulation == 'finite':
        if finite_strains_model not in ('hencky', 'stvenant-kirchhoff'):
            raise RuntimeError('Unknown hyperelastic constitutive model.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over voxels
    for voxel in it.product(*[list(range(n)) for n in n_voxels_dims]):
        # Get voxel material phase
        mat_phase = regular_grid[voxel]
        # Get material phase elastic properties
        material_properties = material_phases_properties[str(mat_phase)]
        # Get material phase elastic tangent modulus
        elastic_tangent_mf = \
            Elastic.elastic_tangent_modulus(problem_type, material_properties)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize strain tensor
        strain = np.zeros((n_dim, n_dim))
        # Get strain tensor
        if strain_formulation == 'infinitesimal':
            # Loop over infinitesimal strain tensor components
            for comp in comp_order_sym:
                # Get second-order array index
                so_idx = tuple([int(i) - 1 for i in comp])
                # Get voxel infinitesimal strain tensor component
                strain[so_idx] = strain_vox[comp][voxel]
        else:
            # Initialize deformation gradient
            def_gradient = np.zeros((n_dim, n_dim))
            # Loop over deformation gradient components
            for comp in comp_order_nsym:
                # Get second-order array index
                so_idx = tuple([int(i) - 1 for i in comp])
                # Get voxel deformation gradient component
                def_gradient[so_idx] = strain_vox[comp][voxel]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute finite strains strain tensor according to hyperelastic
            # constitutive model
            if finite_strains_model == 'stvenant-kirchhoff':
                # Compute material Green-Lagrange strain tensor
                strain = \
                    0.5*(np.matmul(np.transpose(def_gradient), def_gradient) -
                         np.eye(n_dim))
            else:
                # Compute spatial logarithmic strain tensor
                with warnings.catch_warnings():
                    # Supress warnings
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    # Compute spatial logarithmic strain tensor
                    strain = 0.5*top.isotropic_tensor('log',
                        np.matmul(def_gradient, np.transpose(def_gradient)))
                    if np.any(np.logical_not(np.isfinite(strain))):
                        strain = 0.5*scipy.linalg.logm(
                            np.matmul(def_gradient, np.transpose(def_gradient)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get strain tensor matricial form
        strain_mf = mop.get_tensor_mf(strain, n_dim, comp_order_sym)
        # Compute stress tensor matricial form
        stress_mf = np.matmul(elastic_tangent_mf, strain_mf)
        # Build stress tensor
        stress = mop.get_tensor_from_mf(stress_mf, n_dim, comp_order_sym)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build stress_vox
        if strain_formulation == 'infinitesimal':
            # Loop over Cauchy stress tensor components
            for comp in comp_order_sym:
                # Get second-order array index
                so_idx = tuple([int(i) - 1 for i in comp])
                # Store Cauchy stress tensor
                stress_vox[comp][voxel] = stress[so_idx]
        else:
            # Compute First Piola-Kirchhoff stress tensor
            if finite_strains_model == 'stvenant-kirchhoff':
                # Compute First Piola-Kirchhoff stress tensor
                first_piola_stress = np.matmul(def_gradient, stress)
            else:
                # Compute First Piola-Kirchhoff stress tensor
                first_piola_stress = np.matmul(stress,
                    np.transpose(np.linalg.inv(def_gradient)))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over First Piola-Kirchhoff stress tensor components
            for comp in comp_order_nsym:
                # Get second-order array index
                so_idx = tuple([int(i) - 1 for i in comp])
                # Store First Piola-Kirchhoff stress tensor
                stress_vox[comp][voxel] = first_piola_stress[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return stress_vox
#
#                                                          Hadamard operations: State update
# ==========================================================================================
def utility9():
    # Import modules
    import numpy as np
    import time
    import tensor.matrixoperations as mop
    import pickle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set working directory
    working_dir = '/home/bernardoferreira/Documents/CRATE/developments/finite_strains/' + \
                  '2d/avoid_discrete_loops/'
    # Set plots directory
    plots_dir = working_dir
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4, linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain formulation
    strain_formulation = 'infinitesimal'
    # Set problem type
    problem_type = 1
    # Get problem number of spatial dimensions
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # Set RVE dimensions
    rve_dims = n_dim*[1.0,]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set total number of voxels
    n_voxels_total = [10**1, 10**2, 10**3, 10**4, 10**5, 10**6]
    #n_voxels_total = [10**1, 10**2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set hadamard data file path
    hadamard_data_path = working_dir + 'hadamard_su_data_' + strain_formulation + '_strains'
    # Set pickle flag
    is_pickle_hadamard = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build or recover strain concentration data array
    if is_pickle_hadamard:
        # Get strain concentration data array
        with open(hadamard_data_path, 'rb') as hadamard_file:
            data_array, speedup_array = pickle.load(hadamard_file)

            #data_array[0, 3] = 0.98*data_array[1, 3]
            #speedup_array[0] = data_array[0, 1]/data_array[0, 3]
    else:
        # Initialize data arrays
        data_array = np.zeros((len(n_voxels_total), 4))
        speedup_array = np.zeros(len(n_voxels_total))
        # Set total number of voxels
        data_array[:, 0] = n_voxels_total
        data_array[:, 2] = n_voxels_total
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(len(n_voxels_total)):
            # Get total number of voxels
            n = n_voxels_total[i]
            # Set number of voxels on each dimension
            option = 'square'
            if option == 'line':
                n_voxels_dims = [n,] + (n_dim - 1)*[1,]
            elif option == 'square':
                n_voxels_dims = n_dim*[round(n**(1.0/n_dim)),]
            print('\nAnalysis: n_voxels_dims = ', n_voxels_dims,
                  '(total = ', np.prod(n_voxels_dims), ')')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set regular grid of homogeneous material
            regular_grid = np.ones(n_voxels_dims, dtype=int)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set material phases
            material_phases = [str(x) for x in list(np.unique(regular_grid))]
            # Set material phases properties
            material_phases_properties = dict()
            material_phases_properties['1'] = {'E': 100.0e6, 'v': 0.30}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initialize strain tensor
            if strain_formulation == 'infinitesimal':
                strain_vox = {comp: np.ones(tuple(n_voxels_dims))
                              for comp in comp_order_sym}
            else:
                strain_vox = {comp: np.ones(tuple(n_voxels_dims))
                              for comp in comp_order_nsym}
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Set elastic properties-related optimized variables
            evar1 = np.zeros(tuple(n_voxels_dims))
            evar2 = np.zeros(tuple(n_voxels_dims))
            for mat_phase in material_phases:
                # Get material phase elastic properties
                E = material_phases_properties[mat_phase]['E']
                v = material_phases_properties[mat_phase]['v']
                # Build optimized variables
                evar1[regular_grid == int(mat_phase)] = (E*v)/((1.0 + v)*(1.0 - 2.0*v))
                evar2[regular_grid == int(mat_phase)] = np.multiply(2,E/(2.0*(1.0 + v)))
            evar3 = np.add(evar1, evar2)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            naive_init_time = time.time()
            # Naive: Material elastic or hyperelastic constitutive model
            stress_vox_naive = _elastic_constitutive_model_naive(strain_formulation,
                problem_type, n_voxels_dims, strain_vox, regular_grid,
                    material_phases_properties, finite_strains_model='stvenant-kirchhoff',
                        is_optimized=False)
            # Store computational time
            data_array[i, 1] = time.time() - naive_init_time
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            new_init_time = time.time()
            # New: Material elastic or hyperelastic constitutive model
            stress_vox_new = _elastic_constitutive_model(strain_formulation, problem_type,
                n_voxels_dims, strain_vox, evar1, evar2, evar3,
                    finite_strains_model='stvenant-kirchhoff', is_optimized=True)
            # Store computational time
            data_array[i, 3] = time.time() - new_init_time
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute speed-up
            speedup_array[i] = data_array[i, 1]/data_array[i, 3]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dump data arrays
        with open(hadamard_data_path, 'wb') as hadamard_file:
            pickle.dump([data_array, speedup_array], hadamard_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output data arrays
    print('\ndata_array:', '\n', data_array)
    print('\nspeedup_array:', '\n', speedup_array)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set figure name
    fig_name = 'hadamard_su_' + strain_formulation + '_strains'
    # Set axes labels
    x_label = '$n_{v}$'
    y_label = '$\mathrm{CPU \; time \; (s)}$'
    # Set axes limits
    x_min = 10**0
    x_max = 10**7
    y_min = 10**-5
    #y_min = None
    y_max = 10**3
    #y_max = None
    # Set data labels
    data_labels = ['$\mathrm{Discrete \; loops}$',
                   '$\mathrm{Hadamard \; operations}$']
    # Output plot
    newgate_line_plots(plots_dir, fig_name, data_array,
                       x_label=x_label, y_label=y_label, data_labels=data_labels,
                       x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    utility9()
