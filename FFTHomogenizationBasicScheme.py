#
# Moulinec and Suquet FFT-Based Homogenization Method Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Tensorial operations
import tensorOperations as top
#
#                                                      FFT-Based Homogenization Basic Scheme
# ==========================================================================================
# This function is the implementation of the FFT-based homogenization method proposed in
# "A numerical method for computing the overall response of nonlinear composites with
# complex microstructure. Comp Methods Appl M 157 (1998):69-94 (Moulinec, H. and
# Suquet, P.)". For a given RVE discretized in a regular grid of pixels (2D) / voxels (3D)
# (with a total of n_voxels pixels/voxels) and where each pixel/voxel is associated to a
# given material phase, this method solves the microscale static equilibrium problem when
# the RVE is subjected to a given macroscale strain and constrained by periodic boundary
# conditions.
#
# Method scope: | Small strains
#
# Implementation scope: | 2D problems (plain strain) and 3D problems
#                       | Linear elastic constitutive behavior
#                       | Material isotropy
#
# Note 1: At the end of article's 2.4 section, Moulinec and Suquet propose a modification of
#         the Green operator at the higher frequencies in order to overcome limitations of
#         the FFT packages (1998...) for low spatial resolutions. Such modification is not
#         implemented here.
#
# Note 2: Besides the original convergence criterion proposed by Moulinec and Suquet, a
#         second convergence criterion based on the average stress tensor norm has been
#         implemented.
#
# The function returns the strain tensor in every pixel/voxel stored componentwise as:
#
# A. 2D problem (plane strain):
#
#   strain_vox[comp] = array(d1,d2), where | di is the number of pixels in dimension i
#                                          | comp is the strain component that would be
#                                          | stored in matricial form ('11', '22', '12')
#
# B. 3D problem:
#
#   strain_vox[comp] = array(d1,d2,d3), where | di is the number of pixels in dimension i
#                                             | comp is the strain component that would be
#                                             |     stored in matricial form ('11', '22',
#                                             |     '33', '12', '23', '13')
#
# Note 1: All the strain or stress related tensors stored componentwise in dictionaries (for
#         every pixels/voxels) do not follow the Kelvin notation, i.e. the stored component
#         values are the real ones.
#
# Note 2: The suffix '_mf' is employed to denote tensorial quantities stored in matricial
#         form. The matricial form follows the Kelvin notation when symmetry conditions
#         exist and is performed columnwise otherwise.
#
def FFTHomogenizationBasicScheme(problem_dict,rg_dict,mat_dict,mac_strain):
    # --------------------------------------------------------------------------------------
    # Time profile
    is_time_profile = False
    is_conv_output = False
    is_validation_return = False
    if is_time_profile:
        # Date and time
        import time
        # Initialize time profile variables
        start_time_s = time.time()
        phase_names = ['']
        phase_times = np.zeros((1,2))
        phase_names[0] = 'Total'
        phase_times[0,:] = [start_time_s,0.0]
        # Set validation return flag
        is_validation_return = True
        # Set convergence output flag
        is_conv_output = True
    # --------------------------------------------------------------------------------------
    #
    #                                                                             Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set convergence criterion
    # 1. Original convergence criterion (Moulinec & Suquet, 1998) - Non-optimized!
    # 2. Average stress norm based convergence criterion
    conv_criterion = 2
    # Set maximum number of iterations
    max_n_iterations = 100
    # Set convergence tolerance
    conv_tol = 1e-6
    #
    #                                                                        Input arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Get problem type and dimensions
    problem_type = problem_dict['problem_type']
    n_dim = problem_dict['n_dim']
    comp_order = problem_dict['comp_order_sym']
    # Get the spatial discretization file (regular grid of pixels/voxels)
    regular_grid = rg_dict['regular_grid']
    # Get number of pixels/voxels in each dimension and total number of pixels/voxels
    n_voxels_dims = rg_dict['n_voxels_dims']
    n_voxels = np.prod(n_voxels_dims)
    # Get RVE dimensions
    rve_dims = rg_dict['rve_dims']
    # Get material phases
    material_phases = mat_dict['material_phases']
    # Get material properties
    material_properties = mat_dict['material_properties']
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Input arguments')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
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
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Material phases elasticity tensors')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                  Reference material elastic properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set reference material elastic properties as the mean between the minimum and maximum
    # values existent in the microstructure material phases (this is reported in
    # Moulinec, H. and Suquet, P., 1998 as the choice that leads to the best rate of
    # convergence)
    material_properties_ref = dict()
    material_properties_ref['E'] = \
                0.5*(min([material_properties[phase]['E'] for phase in material_phases]) + \
                     max([material_properties[phase]['E'] for phase in material_phases]))
    material_properties_ref['v'] = \
                0.5*(min([material_properties[phase]['v'] for phase in material_phases]) + \
                     max([material_properties[phase]['v'] for phase in material_phases]))
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Reference material elastic properties')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                               Frequency discretization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*np.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Frequency discretization')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                      Reference material Green operator
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
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
    E_ref = material_properties_ref['E']
    v_ref = material_properties_ref['v']
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
        first_term = np.divide(first_term,np.square(var3),where = var3 != 0)
        # Compute second material independent term of Green operator
        second_term = -1.0*np.divide(var2[''.join([str(x) for x in fo_idx])],
                                     np.square(np.square(var3)),where = var3 != 0)
        # Compute Green operator matricial form component
        Gop_DFT_vox[comp] = c1*first_term + c2*second_term
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Reference material Green operator')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Initialize strain and stress tensors
    strain_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    stress_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_order}
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Strain and stress tensors initialization')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                Initial iterative guess
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Set strain initial iterative guess
    for comp in comp_order:
        so_idx = tuple([int(x) - 1 for x in comp])
        strain_vox[comp] = np.full(regular_grid.shape,mac_strain[so_idx])
    # Compute stress initial iterative guess
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
    # Compute average stress norm (convergence criterion)
    avg_stress_norm = 0
    for i in range(len(comp_order)):
        comp = comp_order[i]
        if comp[0] == comp[1]:
            avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
        else:
            avg_stress_norm = avg_stress_norm + 2.0*np.square(stress_vox[comp])
    avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
    avg_stress_norm_old = 0
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Initial iterative guess')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                Strain Discrete Fourier Transform (DFT)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_init_time = time.time()
    # --------------------------------------------------------------------------------------
    # Compute strain Discrete Fourier Transform (DFT)
    strain_DFT_vox = \
                 {comp: np.zeros(tuple(n_voxels_dims),dtype=complex) for comp in comp_order}
    for comp in comp_order:
        # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
        strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
    # Store macroscale strain DFT at the zero-frequency
    freq_0_idx = n_dim*(0,)
    mac_strain_DFT_0 = np.zeros((n_dim,n_dim))
    mac_strain_DFT_0 = np.array([strain_DFT_vox[comp][freq_0_idx] for comp in comp_order])
    # --------------------------------------------------------------------------------------
    # Time profile
    if is_time_profile:
        phase_end_time = time.time()
        phase_names.append('Strain Discrete Fourier Transform (DFT)')
        phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    # --------------------------------------------------------------------------------------
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize iteration counter:
    iter = 0
    # Start iterative loop
    while True:
        #
        #                                            Stress Discrete Fourier Transform (DFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Compute stress Discrete Fourier Transform (DFT)
        stress_DFT_vox = \
                 {comp: np.zeros(tuple(n_voxels_dims),dtype=complex) for comp in comp_order}
        for comp in comp_order:
            # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
            stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Stress Discrete Fourier Transform (DFT)')
            phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                             Convergence evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Convergence criterion 1:
        if conv_criterion == 1:
            # Discrete error as proposed in (Moulinec, H. and Suquet, P., 1998)
            error_sum = 0
            stress_DFT_0_mf = np.zeros(len(comp_order),dtype=complex)
            div_stress_DFT = {str(comp+1): np.zeros(tuple(n_voxels_dims),dtype=complex) \
                                                                   for comp in range(n_dim)}
            # Loop over discrete frequencies
            for freq_coord in it.product(*freqs_dims):
                # Get discrete frequency index
                freq_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) \
                                                                     for x in range(n_dim)])
                # Initialize stress tensor Discrete Fourier Transform (DFT) matricial form
                stress_DFT_mf = np.zeros(len(comp_order),dtype=complex)
                for i in range(len(comp_order)):
                    comp = comp_order[i]
                    # Build stress tensor Discrete Fourier Transform (DFT) matricial form
                    stress_DFT_mf[i] = \
                        top.kelvinFactor(i,comp_order)*stress_DFT_vox[comp][freq_idx]
                    # Store stress tensor Discrete Fourier Transform (DFT) matricial form
                    # for zero-frequency
                    if freq_idx == n_dim*(0,):
                        stress_DFT_0_mf[i] = \
                            top.kelvinFactor(i,comp_order)*stress_DFT_vox[comp][freq_idx]
                # Build stress tensor Discrete Fourier Transform (DFT)
                stress_DFT = np.zeros((n_dim,n_dim),dtype=complex)
                stress_DFT = top.getTensorFromMatricialForm(stress_DFT_mf,n_dim,comp_order)
                # Add discrete frequency contribution to discrete error required sum
                error_sum = error_sum + \
                    np.linalg.norm(top.dot12_1(1j*np.asarray(freq_coord),stress_DFT))**2
                # Compute stress divergence Discrete Fourier Transform (DFT)
                for i in range(n_dim):
                    div_stress_DFT[str(i+1)][freq_idx] = \
                        top.dot12_1(1j*np.asarray(freq_coord),stress_DFT)[i]
            # Compute discrete error serving to check convergence
            discrete_error_1 = np.sqrt(error_sum/n_voxels)/np.linalg.norm(stress_DFT_0_mf)
            # Compute stress divergence Inverse Discrete Fourier Transform (IDFT)
            div_stress = {str(comp+1): np.zeros(tuple(n_voxels_dims)) \
                                                                   for comp in range(n_dim)}
            for i in range(n_dim):
                # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
                # Transform (FFT)
                div_stress[str(i+1)] = np.real(np.fft.ifftn(div_stress_DFT[str(i+1)]))
        # ----------------------------------------------------------------------------------
        # Convergence criterion 2:
        if conv_criterion == 2:
            # Discrete error based on the average stress norm
            discrete_error_2 = abs(avg_stress_norm-avg_stress_norm_old)/avg_stress_norm
        # ----------------------------------------------------------------------------------
        # Validation:
        if is_conv_output:
            print('\nIteration', iter, '- Convergence evaluation:\n')
            print('Average stress norm     = ', '{:>11.4e}'.format(avg_stress_norm))
            print('Average stress norm old = ', '{:>11.4e}'.format(avg_stress_norm_old))
            if conv_criterion == 1:
                print('Discrete error          = ', '{:>11.4e}'.format(discrete_error_1))
            else:
                print('Discrete error          = ', '{:>11.4e}'.format(discrete_error_2))
            # Print iterative stress convergence for file
            if n_dim == 2:
                conv_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                                 'offline_stage/main/2D/FFT_NEW/' + \
                                 'Disk_50_0.3_100_100_uniaxial/convergence_table.dat'
            else:
                conv_file_path = '/media/bernardoferreira/HDD/FEUP PhD/Studies/seminar/' + \
                                 'offline_stage/main/3D/FFT_NEW/' + \
                                 'Sphere_20_0.2_30_30_30_uniaxial/convergence_table.dat'
            if conv_criterion == 1:
                writeIterationConvergence(conv_file_path,'iteration',\
                                          iter,discrete_error_1,0.0)
            elif conv_criterion == 2:
                writeIterationConvergence(conv_file_path,'iteration',\
                                          iter,0.0,discrete_error_2)
            else:
                writeIterationConvergence(conv_file_path,'iteration',\
                                          iter,discrete_error_1,discrete_error_2)
        # ----------------------------------------------------------------------------------
        # Check if the solution converged (return) and if the maximum number of iterations
        # was reached (stop execution)
        if conv_criterion == 1 and discrete_error_1 < conv_tol:
            # ------------------------------------------------------------------------------
            # Time profile
            if is_validation_return:
                return [strain_vox, stress_vox, phase_names, phase_times]
            # ------------------------------------------------------------------------------
            # Return strain
            return strain_vox
        elif conv_criterion == 2 and discrete_error_2 < conv_tol:
            # ------------------------------------------------------------------------------
            # Time profile
            if is_validation_return:
                return [strain_vox, stress_vox, phase_names, phase_times]
            # ------------------------------------------------------------------------------
            # Return strain
            return strain_vox
        if iter == max_n_iterations:
            # Stop execution
            print('\nAbort: The maximum number of iterations was reached before ' + \
                  'solution convergence \n(FFTHomogenizationBasicScheme.py).')
            sys.exit(1)
        # Increment iteration counter
        iter = iter + 1
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Convergence evaluation')
            phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                                      Strain update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        for i in range(len(comp_order)):
            compi = comp_order[i]
            # Update strain
            aux = 0
            for j in range(len(comp_order)):
                compj = comp_order[j]
                idx1 = [comp_order.index(compi),comp_order.index(compj)]
                idx2 = comp_order.index(compj)
                aux = np.add(aux,np.multiply(
                                 top.kelvinFactor(idx1,comp_order)*Gop_DFT_vox[compi+compj],
                                 top.kelvinFactor(idx2,comp_order)*stress_DFT_vox[compj]))
            strain_DFT_vox[compi] = np.subtract(strain_DFT_vox[compi],
                                                (1.0/top.kelvinFactor(i,comp_order))*aux)
            # Enforce macroscopic strain at the zero-frequency strain component
            freq_0_idx = n_dim*(0,)
            strain_DFT_vox[compi][freq_0_idx] = mac_strain_DFT_0[i]
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Update strain')
            phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                   Strain Inverse Discrete Fourier Transform (IDFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
        # Compute strain Inverse Discrete Fourier Transform (IDFT)
        for comp in comp_order:
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            strain_vox[comp] = np.real(np.fft.ifftn(strain_DFT_vox[comp]))
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Strain Inverse Discrete Fourier Transform (IDFT)')
            phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
        # ----------------------------------------------------------------------------------
        #
        #                                                                      Stress update
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_init_time = time.time()
        # ----------------------------------------------------------------------------------
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
        # Compute average stress norm (convergence criterion)
        avg_stress_norm_old = avg_stress_norm
        avg_stress_norm = 0
        for i in range(len(comp_order)):
            comp = comp_order[i]
            if comp[0] == comp[1]:
                avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
            else:
                avg_stress_norm = avg_stress_norm + 2.0*np.square(stress_vox[comp])
        avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
        # ----------------------------------------------------------------------------------
        # Time profile
        if is_time_profile:
            phase_end_time = time.time()
            phase_names.append('Update stress')
            phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
            end_time_s = time.time()
            phase_times[0,1] = end_time_s
            is_time_profile = False
        # ----------------------------------------------------------------------------------
#
#                                                                    Complementary functions
# ==========================================================================================
# Compute the small strain elasticity tensor according to the problem type and material
# constitutive model. Then store it in matricial form following Kelvin notation.
#
# The elasticity tensor is described as follows:
#
# A. Small strain:
#
#   A.1 Isotropic material:
#
#      General Hooke's Law: De = lambda*(IdyadI) + 2*miu*IIsym
#
#                           where | lambda, miu denote the Lamé parameters
#                                 | I denotes the second-order identity tensor
#                                 | IIsym denotes the fourth-order symmetric project. tensor
#
#      A.1.1 2D problem (plane strain) - Kelvin notation:
#                            _                _
#                           | 1-v   v      0   | (11)
#          De =  E/(2*(1+v))|  v   1-v     0   | (22)
#                           |_ 0    0   (1-2v)_| (12)
#
#      A.1.2 2D problem (plane stress) - Kelvin notation:
#                           _              _
#                          |  1   v     0   | (11)
#          De =  E/(1-v**2)|  v   1     0   | (22)
#                          |_ 0   0   (1-v)_| (12)
#
#     A.1.3 2D problem (axisymmetric) - Kelvin notation:
#                            _                     _
#                           | 1-v   v      0     v  | (11)
#          De =  E/(2*(1+v))|  v   1-v     0     v  | (22)
#                           |  0    0   (1-2v)   0  | (12)
#                           |_ v    v      0    1-v_| (33)
#
#     A.1.4 3D problem - Kelvin notation:
#                            _                                    _
#                           | 1-v   v    v     0       0       0   | (11)
#                           |  v   1-v   v     0       0       0   | (22)
#          De =  E/(2*(1+v))|  v    v  1-v     0       0       0   | (33)
#                           |  0    0    0  (1-2v)     0       0   | (12)
#                           |  0    0    0     0    (1-2v)     0   | (23)
#                           |_ 0    0    0     0       0    (1-2v)_| (13)
#
#    Note: E and v denote the Young modulus and the Poisson ratio respectively.
#
def getElasticityTensor(problem_type,n_dim,comp_order,properties):
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = top.setIdentityTensors(n_dim)
    # 2D problem (plane strain)
    if problem_type == 1:
        De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
        De_tensor_mf = top.setTensorMatricialForm(De_tensor,n_dim,comp_order)
    # 3D problem
    elif problem_type == 4:
        De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
        De_tensor_mf = top.setTensorMatricialForm(De_tensor,n_dim,comp_order)
    # Return
    return De_tensor_mf
#
#                                                                     Validation (temporary)
# ==========================================================================================
def writeVoxelStressDivergence(file_path,mode,voxel_idx,*args):
    if mode == 'header':
        # Open file where the stress divergence tensor will be written
        open(file_path,'w').close()
        file = open(file_path,'a')
        # Write file header
        print('\nStress divergence tensor evaluation',file=file)
        print('-----------------------------------',file=file)
        # Write voxel indexes
        print('\nVoxel indexes:',voxel_idx,file=file)
        # Write stress divergence tensor header
        print('\nIter'+ 9*' ' + 'x' + 12*' ' + 'y' + 12*' ' + 'z', file=file)
        # Close file where the stress divergence tensor are written
        file.close()
    else:
        # Open file where the stress divergence tensor are written
        file = open(file_path,'a')
        # Get iteration and stress divergence tensor
        iter = args[0]
        div_stress = args[1]
        # Write iterative value of stress divergence tensor
        print('{:>4d}  '.format(iter),\
              (n_dim*'{:>11.4e}  ').format(*[div_stress[str(i+1)][voxel_idx] \
                                                          for i in range(n_dim)]),file=file)
# ------------------------------------------------------------------------------------------
def writeIterationConvergence(file_path,mode,*args):
    if mode == 'header':
        # Open file where the iteration convergence metrics are written
        open(file_path,'w').close()
        file = open(file_path,'a')
        # Write file header
        print('\nFFT Homogenization Basic Scheme Convergence',file=file)
        print('-------------------------------------------',file=file)
        # Write stress divergence tensor header
        print('\nIter' + 5*' ' + 'error 1' +  7*' ' + 'error 2',file=file)
        # Close file where the stress divergence tensor will be written
        file.close()
    else:
        # Open file where the stress divergence tensor are written
        file = open(file_path,'a')
        # Get iteration and stress divergence tensor
        iter = args[0]
        error1 = args[1]
        error2 = args[2]
        # Write iterative value of stress divergence tensor
        print('{:>4d} '.format(iter), \
              '{:>11.4e}'.format(error1), '  {:>11.4e}'.format(error2),file=file)
# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['FFTHomogenizationBasicScheme()','getElasticityTensor()']
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set functions arguments:
    # Set problem type
    problem_type = 1
    import readInputData as rid
    # Set problem parameters (number of dimensions and components order)
    n_dim, comp_order_sym, comp_order_nsym = rid.setProblemTypeParameters(problem_type)
    # Set problem data
    problem_dict = dict()
    problem_dict['problem_type'] = problem_type
    problem_dict['n_dim'] = n_dim
    problem_dict['comp_order_sym'] = comp_order_sym
    problem_dict['comp_order_nsym'] = comp_order_nsym
    # Set spatial discretization file absolute path
    if problem_type == 1:
        rve_dims = [1.0,1.0]
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
                            'debug/FFT_Homogenization_Method/validation/' + \
                            'RVE_2D_2Phases_1025x1025_Circular_Fiber.rgmsh.npy'
    else:
        rve_dims = [1.0,1.0,1.0]
        discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
                            'debug/FFT_Homogenization_Method/issues/stress_divergence/' + \
                            'examples/RVE_3D_2Phases_50x50_Homogeneous.rgmsh.npy'
    # Read spatial discretization file and set regular grid data
    regular_grid = np.load(discret_file_path)
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    rg_dict = dict()
    rg_dict['rve_dims'] = rve_dims
    rg_dict['regular_grid'] = regular_grid
    rg_dict['n_voxels_dims'] = n_voxels_dims
    # Set material properties
    material_properties = dict()
    material_properties['2'] = dict()
    material_properties['2']['E'] = 68.9e3
    material_properties['2']['v'] = 0.35
    material_properties['1'] = dict()
    material_properties['1']['E'] = 400e3
    material_properties['1']['v'] = 0.23
    mat_dict = dict()
    mat_dict['material_properties'] = material_properties
    material_phases = [str(x) for x in list(np.unique(regular_grid))]
    mat_dict['material_phases'] = material_phases
    # Set macroscale strain loading
    if n_dim == 2:
        mac_strain = np.array([[0,0.005],[0.005,0]])
    else:
        mac_strain = np.array([[2,0.5,0.5],[0.5,1,1],[0.5,1,3]])
    # Set numpy default print options
    np.set_printoptions(precision=4,linewidth=np.inf)
    # Set absolute path of the file where the stress divergence tensor for a given voxel
    # is written at every iteration
    div_file_path = '/home/bernardoferreira/Documents/SCA/validation/' + \
                    'FFT_Homogenization_Method/validation/' + \
                    'StressDivergenceEvolution.dat'
    if n_dim == 2:
        val_voxel_idx = (2,1)
    else:
        val_voxel_idx = (2,1,3)
    writeVoxelStressDivergence(div_file_path,'header',val_voxel_idx)
    # Set absolute path of the file where the error for the diferent convergence criteria
    # is written at every iteration
    conv_file_path = '/home/bernardoferreira/Documents/SCA/validation/' + \
                     'FFT_Homogenization_Method/validation/' + \
                     'ConvergenceEvolution.dat'
    writeIterationConvergence(conv_file_path,'header')
    # Call function
    strain_vox = FFTHomogenizationBasicScheme(problem_dict,rg_dict,mat_dict,mac_strain)
    # Write VTK file with material phases
    import VTKOutput
    import copy
    import ntpath
    dirs_dict = dict()
    dirs_dict['input_file_name'] = \
                ntpath.splitext(ntpath.splitext(ntpath.basename(discret_file_path))[-2])[-2]
    dirs_dict['offline_stage_dir'] = ntpath.dirname(discret_file_path) + '/'
    clst_dict = dict()
    clst_dict['voxels_clusters'] = np.full(n_voxels_dims,-1,dtype=int)
    vtk_dict = dict()
    vtk_dict['format'] = 'ascii'
    vtk_dict['precision'] = 'SinglePrecision'
    VTKOutput.writeVTKClusterFile(vtk_dict,copy.deepcopy(dirs_dict),copy.deepcopy(rg_dict),
                                                                   copy.deepcopy(clst_dict))
    # 2D matplotlib plot
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    matplotlib.rc('text',usetex=True)
    matplotlib.rc('font',**{'family':'serif'})
    figure1 = plt.figure()
    axes1 = figure1.add_subplot(111)
    axes1.set_title('Validation Example - Circular Fiber')
    figure1.set_figheight(8, forward=True)
    figure1.set_figwidth(8, forward=True)
    axes1.set_frame_on(True)
    axes1.set_xlabel('$x_{1}$', fontsize=12, labelpad=10)
    axes1.set_ylabel('$\\varepsilon_{12}$', fontsize=12, labelpad=8)
    axes1.ticklabel_format(axis='x', style='plain', scilimits=(3,4))
    axes1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes1.xaxis.set_minor_formatter(ticker.NullFormatter())
    axes1.yaxis.set_major_locator(ticker.AutoLocator())
    axes1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    axes1.tick_params(which='major', width=1.0, length=10, labelcolor='0.0', labelsize=12)
    axes1.tick_params(which='minor', width=1.0, length=5, labelsize=12)
    axes1.grid(linestyle='-',linewidth=0.5, color='0.5',zorder=15)
    axes1.set_xlim(xmin=0.4,xmax=0.6)
    axes1.set_ylim(ymin=0,ymax=0.008)
    aux1 = np.array(range(n_voxels_dims[0]))*(rve_dims[0]/n_voxels_dims[0])
    x = aux1[int(np.nonzero(aux1>=0.4)[0][0]):int(np.nonzero(aux1>=0.6)[0][0])+1]
    y = [strain_vox['12'][(i,int(n_voxels_dims[0]/2))] \
        for i in range(int(np.nonzero(aux1>=0.4)[0][0]),int(np.nonzero(aux1>=0.6)[0][0])+1)]
    axes1.plot(x,y,color='k',linewidth=2,linestyle='-',clip_on=False)
    plt.show()
    figure1.set_figheight(3.6, forward=False)
    figure1.set_figwidth(3.6, forward=False)
    figure1.savefig('Validation'+'.pdf', transparent=False, dpi=300, bbox_inches='tight')
    # Display validation footer
    print('\n' + 92*'-' + '\n')
