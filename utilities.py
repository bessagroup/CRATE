#
# Utilities Module (UNNAMED Program)
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
#   1. Add stress_vox to the output of FFTHomogenizationBasicScheme()
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
    import info
    import fileOperations
    import readInputData as rid
    import FFTHomogenizationBasicScheme as FFT
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem type
    problem_type = 1
    # Set problem parameters
    n_dim, comp_order_sym, comp_order_nsym = rid.setProblemTypeParameters(problem_type)
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
                            'Disk_50_0.3_700_700.rgmsh.npy'
    else:
        rve_dims = [1.0,1.0,1.0]
        discret_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                            'microstructures/3D/main/regular_grids/' + \
                            'Sphere_20_0.2_30_30_30.rgmsh.npy'
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
    if n_dim == 2:
        loading = 'uniaxial'
        mac_strain = np.array([[ 5.0e-3 , 0.0e-3 ],
                               [ 0.0e-3 , 0.0e-3 ]])
    else:
        loading = 'uniaxial'
        mac_strain = np.array([[ 5.0e-3 , 0.0e-3 , 0.0e-3 ],
                               [ 0.0e-3 , 0.0e-3 , 0.0e-3 ],
                               [ 0.0e-3 , 0.0e-3 , 0.0e-3 ]])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set output directory
    discret_file_basename = \
        ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[-2])[-2]
    if problem_type == 1:
        output_dir = \
           '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/offline_stage/main/2D/FFT/'
    else:
        output_dir = \
           '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/offline_stage/main/3D/FFT/'
    output_dir = output_dir + '/' + discret_file_basename + '_' + loading + '/'
    fileOperations.makeDirectory(output_dir,option='overwrite')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set absolute path of the file where the error for the diferent convergence criteria
    # is written at every iteration
    conv_file_path = output_dir + 'convergence_table.dat'
    FFT.writeIterationConvergence(conv_file_path,'header')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start timer
    time_init = time.time()
    # Compute FFT solution
    strain_vox, stress_vox = FFT.FFTHomogenizationBasicScheme(problem_dict,rg_dict,
                                                              mat_dict,mac_strain)
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
    # Set results output format
    displayFeatures = info.setDisplayFeatures()
    output_width = displayFeatures[0]
    indent = displayFeatures[2]
    equal_line = displayFeatures[5]
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
    # Output results to default stdout
    print(template.format(*info,width=output_width))
    # Output results to results file
    results_file_path = output_dir + 'results.dat'
    open(results_file_path, 'w').close()
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
    import VTKOutput
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
    VTKOutput.writeVTKFileHeader(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK dataset element
    dataset_parameters,piece_parameters = \
        VTKOutput.setImageDataParameters(n_voxels_dims,rve_dims)
    VTKOutput.writeVTKOpenDatasetElement(vtk_file,vtk_dict,dataset_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece
    VTKOutput.writeVTKOpenPiece(vtk_file,piece_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open VTK dataset element piece cell data
    VTKOutput.writeVTKOpenCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Material phases
    data_list = list(regular_grid.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Material phase','format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
    VTKOutput.writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Clusters
    data_list = list(voxels_clusters.flatten('F'))
    min_val = min(data_list)
    max_val = max(data_list)
    data_parameters = {'Name':'Cluster','format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
    VTKOutput.writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Strain tensor
    for comp in comp_order_sym:
        data_list = list(strain_vox[comp].flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name':'e_strain_' + comp,'format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
        VTKOutput.writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK cell data array - Stress tensor
    for comp in comp_order_sym:
        data_list = list(stress_vox[comp].flatten('F'))
        min_val = min(data_list)
        max_val = max(data_list)
        data_parameters = {'Name':'stress_' + comp,'format':vtk_format,'RangeMin':min_val,
                                                                         'RangeMax':max_val}
        VTKOutput.writeVTKCellDataArray(vtk_file,vtk_dict,data_list,data_parameters)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece cell data
    VTKOutput.writeVTKCloseCellData(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element piece
    VTKOutput.writeVTKClosePiece(vtk_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Close VTK dataset element
    VTKOutput.writeVTKCloseDatasetElement(vtk_file,vtk_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write VTK file footer
    VTKOutput.writeVTKFileFooter(vtk_file)
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
    import Links.ioput.genLinksInputFile
    #
    #                                                                            Data import
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discretization file path (.rgmsh file)
    discret_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/microstructures/3D/main/regular_grids/Sphere_20_0.2_70_70_70.rgmsh.npy'
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
        # Set Links input data file path
        Links_file_path = ntpath.dirname(discret_file_path) + '/' + rg_file_name + '.femsh'
        # Get problem dimension
        n_dim = len(n_voxels_dims)
        # Set RVE dimensions
        rve_dims = n_dim*[1.0,]
        # Set finite element order
        fe_order = 'quadratic'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate Links finite element mesh
        coords,connectivities,element_phases = \
            Links.ioput.genLinksInputFile.generateFEMesh(n_dim,rve_dims,n_voxels_dims,
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
        data_file = open(Links_file_path,'a')
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
if __name__ == '__main__':
    utility1()
    #utility3()
