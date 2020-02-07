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
# Working with arrays
import numpy as np
# Mathematics
import math
# Inspect file name and line
import inspect
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
# Suquet, P.". For a given RVE discretized in a regular grid of pixels (2D) / voxels (3D)
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
#         values are the real ones
#
# Note 2: The suffix '_mf' is employed to denote tensorial quantities stored in matricial
#         form. The matricial form follows the Kelvin notation when symmetry conditions
#         exist and is performed columnwise otherwise.
#
def FFTHomogenizationBasicScheme(problem_type,rve_dims,regular_grid,material_properties,
                                                                      comp_list,mac_strain):
    #
    #                                                                             Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set maximum number of iterations
    max_n_iterations = 30
    # Set convergence tolerance
    conv_tol = 1e-2
    #
    #                                                                        Input arguments
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem dimensions
    n_dim = len(regular_grid.shape)
    # Set number of pixels/voxels in each dimension and total number of pixels/voxels
    n_voxels_dims = [regular_grid.shape[i] for i in range(len(regular_grid.shape))]
    n_voxels = np.prod(n_voxels_dims)
    # Set number of material phases
    n_material_phases = material_properties.shape[2]
    # Set macroscale strain matricial form
    mac_strain_mf = top.setTensorMatricialForm(mac_strain,n_dim,comp_list)
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_tensors_mf = list()
    for iphase in range(n_material_phases):
        # Set required elastic properties according to material phase constitutive model
        req_props = ['E','v']
        req_props_vals = {prop: None for prop in req_props}
        for iprop in range(len(req_props)):
            match = np.where(material_properties[:,0,iphase]==req_props[iprop])
            if len(match[0]) != 1:
                values = tuple(req_props[iprop],iphase+1)
                template = 'The elastic property - {} - of material phase {} hasn\'t ' + \
                           'been specified or has been ' + '\n' + \
                           'specified more than once in the input data file.'
                print(template.format(*values))
            else:
                req_props_vals[req_props[iprop]] = material_properties[match[0][0],1,iphase]
        # Compute elasticity tensor (matricial form) for current material phase
        De_tensor_mf = np.zeros((len(comp_list),len(comp_list)))
        De_tensor_mf = getElasticityTensor(problem_type,n_dim,comp_list,req_props_vals)
        # Store material phase elasticity tensor (matricial form)
        De_tensors_mf.append(De_tensor_mf)
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        for iphase in range(n_material_phases):
            print('\nElasticity tensor (material phase ', iphase + 1, ') - ' + \
                                                                       'Kelvin notation:\n')
            np.set_printoptions(precision=2)
            print(De_tensors_mf[iphase])
    # --------------------------------------------------------------------------------------
    #
    #                                                  Reference material elastic properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required elastic properties
    req_props_ref = ['E','v']
    req_props_vals_ref = {prop: None for prop in req_props}
    # Set reference material elastic properties
    req_props_vals_ref['E'] = 100e6
    req_props_vals_ref['v'] = 0.3
    # Compute compliance tensor (matricial form)
    Se_tensor_mf_ref = np.zeros((len(comp_list),len(comp_list)))
    Se_tensor_mf_ref = \
         np.linalg.inv(getElasticityTensor(problem_type,n_dim,comp_list,req_props_vals_ref))
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        print('\nCompliance tensor (reference material) - Kelvin notation:\n')
        print(Se_tensor_mf_ref)
    # --------------------------------------------------------------------------------------
    #
    #                                                               Frequency discretization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (rad/m) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_dims[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*math.pi*np.fft.fftfreq(n_voxels_dims[i],sampling_period))
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        print('\nDiscrete frequencies in each dimension:\n')
        for i in range(n_dim):
            print('  Dimension ', i, ': ', freqs_dims[i],'\n')
    # --------------------------------------------------------------------------------------
    #
    #                                                      Reference material Green operator
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The fourth-order Green operator for the reference material is computed for every
    # pixel/voxel and stored as follows:
    #
    # A. 2D problem (plane strain):
    #
    #   Green_operator_vox[comp] = array(d1,d2),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |      form ('1111', '2211', '1211', '1122', ...)
    #
    # B. 3D problem:
    #
    #   Green_operator_vox[comp] = array(d1,d2,d3),
    #
    #                        where | di is the number of pixels in dimension i
    #                              | comp is the component that would be stored in matricial
    #                              |     form ('1111', '2211', '3311', '1211', ...)
    #
    # Get reference material Young modulus and Poisson coeficient
    E_ref = req_props_vals_ref['E']
    v_ref = req_props_vals_ref['v']
    # Compute reference material Lamé parameters
    lam_ref = (E_ref*v_ref)/((1.0 + v_ref)*(1.0 - 2.0*v_ref))
    miu_ref = E_ref/(2.0*(1.0 + v_ref))
    # Compute Green operator reference material related constants
    c1 = 1.0/(4.0*miu_ref)
    c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
    # Set Green operator matricial form components
    comps = list(it.product(comp_list,comp_list))
    # Set mapping between Green operator fourth-order tensor and matricial form components
    fo_indexes = list()
    mf_indexes = list()
    for i in range(len(comp_list)**2):
        fo_indexes.append([int(x)-1 for x in list(comps[i][0]+comps[i][1])])
        mf_indexes.append([x for x in \
                               [comp_list.index(comps[i][0]),comp_list.index(comps[i][1])]])
    # Initialize Green operator
    Green_operator_vox = {''.join([str(x+1) for x in idx]): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in fo_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get matrix index
        mf_idx = mf_indexes[i]
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
                   top.Dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[3]] +
                   top.Dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                   top.Dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                   top.Dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
            # Compute second material independent term of Green operator
            second_term = -(1.0/freq_norm**4)*(freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                               freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
            # Compute Green operator matricial form component for current voxel
            Green_operator_vox[comp][freq_idx] = c1*first_term + c2*second_term
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        if n_dim == 2:
            val_voxel_idx = (2,1)
        else:
            val_voxel_idx = (2,1,3)
        val_voxel_freqs = [freqs_dims[i][val_voxel_idx[i]] for i in range(n_dim)]
        val_voxel_freqs_norm = np.linalg.norm(val_voxel_freqs)
        print('\nGreen operator components (freq_idx = ' + str(val_voxel_idx) + '):\n')
        print('  Frequency point = ', val_voxel_freqs)
        print('  Norm            = ', '{:>11.4e}'.format(val_voxel_freqs_norm))
        print('\n  Material-dependent constants:')
        print('  c1 = ', '{:>11.4e}'.format(c1))
        print('  c2 = ', '{:>11.4e}\n'.format(c2))
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            comp = ''.join([str(x+1) for x in fo_indexes[i]])
            print('  Component ' + comp + ': ', \
                                '{:>11.4e}'.format(Green_operator_vox[comp][val_voxel_idx]))
    # --------------------------------------------------------------------------------------
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize strain and stress tensors
    strain_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_list}
    stress_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_list}
    #
    #                                                                Initial iterative guess
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loop over discrete frequencies
    for freq_coord in it.product(*freqs_dims):
        # Get voxel material phase
        voxel_idx = tuple([list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)])
        phase_idx = regular_grid[voxel_idx] - 1
        # Get material phase elasticity tensor (matricial form)
        De_tensor_mf = De_tensors_mf[phase_idx]
        # Set strain initial iterative guess
        strain_mf = np.zeros(len(comp_list))
        strain_mf = mac_strain_mf
        for i in range(len(comp_list)):
            comp = comp_list[i]
            strain_vox[comp][voxel_idx] = (1.0/top.kelvinFactor(i,comp_list))*strain_mf[i]
        # Set stress initial iterative guess
        stress_mf = np.zeros(len(comp_list))
        stress_mf = top.dot21_1(De_tensor_mf,strain_mf) #(confirmar dot21_1)
        #stress_mf = np.matmul(De_tensor_mf,strain_mf)
        for i in range(len(comp_list)):
            comp = comp_list[i]
            stress_vox[comp][voxel_idx] = (1.0/top.kelvinFactor(i,comp_list))*stress_mf[i]
    # Compute average stress norm (convergence criterion)
    avg_stress_norm = 0
    for i in range(len(comp_list)):
        comp = comp_list[i]
        if comp[0] == comp[1]:
            avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
        else:
            avg_stress_norm = avg_stress_norm + 1.0*np.square(stress_vox[comp])
    avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
    avg_stress_norm_Old = 0
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        print('\nStrain initial iterative guess (voxel_idx = ' + str(val_voxel_idx) + \
                                                                                     '):\n')
        for i in range(len(comp_list)):
            comp = comp_list[i]
            print('Component ' + comp + ': ', \
                                        '{:>11.4e}'.format(strain_vox[comp][val_voxel_idx]))
        print('\nStress initial iterative guess (voxel_idx = ' + str(val_voxel_idx) + \
                                                                                     '):\n')
        for i in range(len(comp_list)):
            comp = comp_list[i]
            print('Component ' + comp + ': ', \
                                        '{:>11.4e}'.format(stress_vox[comp][val_voxel_idx]))
    # --------------------------------------------------------------------------------------
    #
    #                                                Strain Discrete Fourier Transform (DFT)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute strain Discrete Fourier Transform (DFT)
    strain_DFT_vox = \
                  {comp: np.zeros(tuple(n_voxels_dims),dtype=complex) for comp in comp_list}
    for comp in comp_list:
        # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
        strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
    # Store macroscale strain DFT at the zero-frequency
    freq_0_idx = n_dim*(0,)
    mac_strain_DFT_0 = np.zeros((n_dim,n_dim))
    mac_strain_DFT_0 = np.array([strain_DFT_vox[comp][freq_0_idx] for comp in comp_list])
    # --------------------------------------------------------------------------------------
    # Validation:
    if __name__ == '__main__':
        print('\nStrain DFT (freq_idx = ' + str(val_voxel_idx) + '):\n')
        for i in range(len(comp_list)):
            comp = comp_list[i]
            print('Component ' + comp + ': ', \
                                    '{:>23.4e}'.format(strain_DFT_vox[comp][val_voxel_idx]))
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
        # Compute stress Discrete Fourier Transform (DFT)
        stress_DFT_vox = \
                  {comp: np.zeros(tuple(n_voxels_dims),dtype=complex) for comp in comp_list}
        for comp in comp_list:
            # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
            stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
        # ----------------------------------------------------------------------------------
        # Validation:
        if __name__ == '__main__':
            print('\nStress DFT (freq_idx = ' + str(val_voxel_idx) + '):\n')
            for i in range(len(comp_list)):
                comp = comp_list[i]
                print('Component ' + comp + ': ', \
                                    '{:>23.4e}'.format(stress_DFT_vox[comp][val_voxel_idx]))
        # ----------------------------------------------------------------------------------
        #
        #                                                             Convergence evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Convergence criterion 1:
        # Discrete error as proposed in (Moulinec, H. and Suquet, P., 1998):
        # Compute sum of stress divergence norm for all discrete frequencies and store
        # zero-frequency stress
        error_sum = 0
        stress_DFT_0_mf = np.zeros(len(comp_list),dtype=complex)
        for freq_coord in it.product(*freqs_dims):
            # Get voxel index
            freq_idx = \
                     tuple([list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)])
            # Initialize stress auxiliary vector
            stress_DFT = np.zeros((n_dim,n_dim),dtype=complex)
            stress_DFT_mf = np.zeros(len(comp_list),dtype=complex)
            for i in range(len(comp_list)):
                comp = comp_list[i]
                # Get stress vector for current discrete frequency
                stress_DFT_mf[i] = kelvinFactor(i,comp_list)*stress_DFT_vox[comp][freq_idx]
                # Store stress vector for zero-frequency
                if freq_idx == n_dim*(0,):
                    stress_DFT_0_mf[i] = kelvinFactor(i,comp_list)*\
                                                              stress_DFT_vox[comp][freq_idx]
            # Build stress tensor (frequency domain)
            stress_DFT = top.getTensorFromMatricialForm(stress_DFT_mf,n_dim,comp_list)
            # Add discrete frequency contribution to discrete error required sum
            error_sum = error_sum + \
                        np.linalg.norm(top.dot12_1(1j*np.asarray(freq_coord),stress_DFT))**2

            print('\nVerify original convergence criterion:')
            print('\nvoxel_idx / freq_idx = ', freq_idx)
            print('\nfreq_coord           = ', freq_coord)
            print('\nnorm(divergence_DFT[freq_idx]) = ', '{:>11.4e}'.format(np.linalg.norm(top.dot12_1(1j*np.asarray(freq_coord),stress_DFT))))
            print('divergence_DFT[freq_idx]  = ', top.dot12_1(1j*np.asarray(freq_coord),stress_DFT))
            print('divergence[voxel_idx]     = ', np.fft.ifftn(top.dot12_1(np.asarray(freq_coord),stress_DFT)))

        # Compute discrete error serving to check convergence
        n_voxels = np.prod(n_voxels_dims)
        discrete_error_2 = math.sqrt(error_sum/n_voxels)/np.linalg.norm(stress_DFT_0_mf)
        # ----------------------------------------------------------------------------------
        # Convergence criterion 2:
        # Discrete error based on the average stress norm
        discrete_error = abs(avg_stress_norm-avg_stress_norm_Old)/avg_stress_norm
        # ----------------------------------------------------------------------------------
        # Validation:
        if __name__ == '__main__':
            print('\nIteration', iter, '- Convergence evaluation:\n')
            print('Average stress norm     = ', '{:>11.4e}'.format(avg_stress_norm))
            print('Average stress norm old = ', '{:>11.4e}'.format(avg_stress_norm_Old))
            print('Discrete error          = ', '{:>11.4e}'.format(discrete_error))
        # ----------------------------------------------------------------------------------
        # Check if the solution converged (return) and if the maximum number of iterations
        # was reached (stop execution)
        if discrete_error < conv_tol:
            # Return strain
            return strain_vox
        elif iter == max_n_iterations:
            # Stop execution
            print('\nAbort: The maximum number of iterations was reached before ' + \
                  'solution convergence.\n')
            sys.exit(1)
        # Increment iteration counter
        iter = iter + 1
        #
        #                                                                      Update strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(len(comp_list)):
            compi = comp_list[i]
            # Update strain
            aux = 0
            for j in range(len(comp_list)):
                compj = comp_list[j]
                idx1 = [comp_list.index(compi),comp_list.index(compj)]
                idx2 = comp_list.index(compj)
                aux = aux + \
                          top.kelvinFactor(idx1,comp_list)*Green_operator_vox[compi+compj]*\
                                      top.kelvinFactor(idx2,comp_list)*stress_DFT_vox[compj]
            strain_DFT_vox[compi] = strain_DFT_vox[compi] - \
                                                     (1.0/top.kelvinFactor(i,comp_list))*aux
            # Enforce macroscopic strain at the zero-frequency strain component
            freq_0_idx = n_dim*(0,)
            strain_DFT_vox[compi][freq_0_idx] = mac_strain_DFT_0[i]
        # ----------------------------------------------------------------------------------
        # Validation:
        if __name__ == '__main__':
            print('\nStrain DFT - Update (freq_idx = ' + str(val_voxel_idx) + '):\n')
            for i in range(len(comp_list)):
                comp = comp_list[i]
                print('Component ' + comp + ': ', \
                                    '{:>23.4e}'.format(strain_DFT_vox[comp][val_voxel_idx]))
        # ----------------------------------------------------------------------------------
        #
        #                                   Strain Inverse Discrete Fourier Transform (IDFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute strain Inverse Discrete Fourier Transform (IDFT)
        for comp in comp_list:
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            strain_vox[comp] = np.real(np.fft.ifftn(strain_DFT_vox[comp]))
        # ----------------------------------------------------------------------------------
        # Validation:
        if __name__ == '__main__':
            print('\nStrain (voxel_idx = ' + str(val_voxel_idx) + '):\n')
            for i in range(len(comp_list)):
                comp = comp_list[i]
                print('Component ' + comp + ': ', \
                                        '{:>11.4e}'.format(strain_vox[comp][val_voxel_idx]))
        # ----------------------------------------------------------------------------------
        #
        #                                                                      Update stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get voxel material phase
            voxel_idx = \
                     tuple([list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)])
            phase_idx = regular_grid[voxel_idx] - 1
            # Get material phase elasticity tensor (matricial form)
            De_tensor_mf = De_tensors_mf[phase_idx]
            # Get strain vector for current discrete frequency
            strain_mf = np.zeros(len(comp_list))
            for i in range(len(comp_list)):
                comp = comp_list[i]
                strain_mf[i] = top.kelvinFactor(i,comp_list)*strain_vox[comp][voxel_idx]
            # Update stress for current discrete frequency
            stress_mf = np.zeros(len(comp_list))
            stress_mf = top.dot21_1(De_tensor_mf,strain_mf)
            for i in range(len(comp_list)):
                comp = comp_list[i]
                stress_vox[comp][voxel_idx] = \
                                            (1.0/top.kelvinFactor(i,comp_list))*stress_mf[i]
        # Compute average stress norm (convergence criterion)
        avg_stress_norm_Old = avg_stress_norm
        avg_stress_norm = 0
        for i in range(len(comp_list)):
            comp = comp_list[i]
            if comp[0] == comp[1]:
                avg_stress_norm = avg_stress_norm + np.square(stress_vox[comp])
            else:
                avg_stress_norm = avg_stress_norm + 1.0*np.square(stress_vox[comp])
        avg_stress_norm = np.sum(np.sqrt(avg_stress_norm))/n_voxels
        # ----------------------------------------------------------------------------------
        # Validation:
        if __name__ == '__main__':
            print('\nStress (voxel_idx = ' + str(val_voxel_idx) + '):\n')
            for i in range(len(comp_list)):
                comp = comp_list[i]
                print('Component ' + comp + ': ', \
                                        '{:>11.4e}'.format(stress_vox[comp][val_voxel_idx]))
        # ----------------------------------------------------------------------------------
#
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
def getElasticityTensor(problem_type,n_dim,comp_list,properties):
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
        De_tensor_mf = top.setTensorMatricialForm(De_tensor,n_dim,comp_list)
    # 3D problem
    elif problem_type == 4:
        De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
        De_tensor_mf = top.setTensorMatricialForm(De_tensor,n_dim,comp_list)
    # Return
    return De_tensor_mf
#
#                                                                     Validation (temporary)
# ==========================================================================================
if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['FFTHomogenizationBasicScheme()','getElasticityTensor()']
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set functions arguments
    problem_type = 1
    rve_dims = [1.0,1.0,1.0]
    discret_file_path = '/home/bernardoferreira/Documents/SCA/' + \
    'debug/FFT_Homogenization_Method/RVE_2D_2Phases_100x100.rgmsh.npy'
    regular_grid = np.load(discret_file_path)
    n_dim = len(regular_grid.shape)
    material_properties = np.zeros((2,2,2),dtype=object)
    material_properties[0,0,0] = 'E' ; material_properties[0,1,0] = 210e6
    material_properties[1,0,0] = 'v' ; material_properties[1,1,0] = 0.3
    material_properties[0,0,1] = 'E' ; material_properties[0,1,1] = 70e6
    material_properties[1,0,1] = 'v' ; material_properties[1,1,1] = 0.33
    if problem_type == 1:
        comp_list = ['11','22','12']
    elif problem_type == 4:
        comp_list = ['11','22','33','12','23','13']
    if n_dim == 2:
        mac_strain = np.array([[2,0.5],[0.5,1]])
    else:
        mac_strain = np.array([[2,0.5,0.5],[0.5,1,1],[0.5,1,3]])
    # Call function
    FFTHomogenizationBasicScheme(problem_type,rve_dims,regular_grid,material_properties,
                                                                       comp_list,mac_strain)
    # Display validation footer
    print('\n' + 92*'-' + '\n')
