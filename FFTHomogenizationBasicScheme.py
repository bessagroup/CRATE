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
import tensorOperations
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
def FFTHomogenizationBasicScheme(problem_type,n_dim,n_voxels_dims,regular_grid,
                                                     n_material_phases,material_properties):
    #
    #                                                                             Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set macroscale strain
    mac_strain_mf = [ 1.0 , 0 , 0 , 0 , 0 , 0]
    # Set RVE dimensions (this must be argument)
    rve_dims = [1.0,1.0,1.0]
    # Set strain components according to problem type (this must be argument)
    if problem_type == 1:
        comp_list = ['11','22','12']
    elif problem_type == 4:
        comp_list = ['11','22','33','12','23','13']
    # Set maximum number of iterations
    max_n_iterations = 100
    # Set convergence tolerance
    conv_tol = 1e-6
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_matrices_mf = list()
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
        De_tensor_mf = getElasticityTensor(problem_type,n_dim,comp_list,req_props_vals)
        # Store material phase elasticity tensor (matricial form)
        De_tensors_mf.append(De_tensor_mf)
    #
    #                                                  Reference material elastic properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required elastic properties
    req_props_ref = ['E','v']
    req_props_vals_ref = {prop: None for prop in req_props}
    # Set reference material elastic properties
    req_props_vals_ref['E'] = 1.0
    req_props_vals_ref['v'] = 0.3
    # Compute compliance tensor (matricial form)
    Se_tensor_mf_ref = \
                      np.linalg.inv(getElasticityTensor(problem_type,n_dim,required_values))
    #
    #                                                               Frequency discretization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set discrete frequencies (Hz) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_size[i]/n_voxels_dims[i]
        # Set discrete frequencies
        freqs_dims.append(2*math.pi*np.fft.fftfreq(n_voxels_dim[i],sampling_period))
        # >>>> António
        # Set sampling angular frequency
        # sampling_freq = 2*math.pi/sampling_period
        # Set discrete frequencies
        # freqs_dims.append(sampling_freq*np.linspace(0,1,num=n_voxels_dims[i],endpoint = False))
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
    Green_operator_vox = {str(idx[0]+1)+str(idx[1]+1): \
                                       np.zeros(tuple(n_voxels_dims)) for idx in mf_indexes}
    # Compute Green operator matricial form components
    for i in range(len(mf_indexes)):
        # Get matrix index
        mf_idx = mf_indexes[i]
        # Get fourth-order tensor indexes
        fo_idx = fo_indexes[i]
        # Get Green operator matricial form component
        comp = str(mf_idx[0]+1)+str(mf_idx[1]+1)
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get voxel index
            voxel_idx = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
            # Compute frequency vector norm
            freq_norm = np.linalg.norm(freq_coord)
            # Compute first material independent term of Green operator
            first_term = (1.0/freq_norm**2)*(
                       Dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[1]] +
                       Dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                       Dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                       Dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
            # Compute second material independent term of Green operator
            second_term = -(1.0/freq_norm**4)*(freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                               freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
            # Compute Green operator matricial form component for current voxel
            Green_operator_vox[comp][voxel_idx] = c1*first_term + c2*second_term
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
        voxel_idx = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
        phase_idx = regular_grid[voxel_idx] - 1
        # Get material phase elasticity tensor (matricial form)
        De_tensor_mf = De_tensors_mf[phase_idx]
        # Set strain initial iterative guess
        strain_mf = mac_strain_mf
        for i in range(len(comp_list)):
            comp = comp_list[i]
            strain_vox[comp][voxel_idx] = (1.0/kelvinFactor(i,comp_list))*strain_mf[i]
        # Set stress initial iterative guess
        stress_mf = De_tensor_mf*strain_mf
        for i in range(len(comp_list)):
            comp = comp_list[i]
            stress_vox[comp][voxel_idx] = (1.0/kelvinFactor(i,comp_list))*stress_mf[i]
    #
    #                                                Strain Discrete Fourier Transform (DFT)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute strain Discrete Fourier Transform (DFT)
    strain_DFT_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_list}
    for comp in comp_list:
        # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
        strain_DFT_vox[comp] = np.fft.fftn(strain_vox[comp])
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
        stress_DFT_vox = {comp: np.zeros(tuple(n_voxels_dims)) for comp in comp_list}
        for comp in comp_list:
            # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
            stress_DFT_vox[comp] = np.fft.fftn(stress_vox[comp])
        #
        #                                                             Convergence evaluation
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute sum of stress divergence norm for all discrete frequencies and store
        # zero-frequency stress
        sum = 0
        for freq_coord in it.product(*freqs_dims):
            # Get voxel index
            voxel_idx = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
            # Initialize stress auxiliary vectors
            stress_DFT_mf = np.array(len(comp_list))
            stress_DFT_0_mf = np.array(len(comp_list))
            for i in range(len(comp_list)):
                comp = comp_list[i]
                # Get stress vector for current discrete frequency
                stress_DFT_mf[i] = kelvinFactor(i,comp_list)*stress_DFT_vox[comp][voxel_idx]
                # Store stress vector for zero-frequency
                if all([voxel_idx[x] == 0 for x in range(n_dim)]):
                    stress_DFT_0_mf[i] = kelvinFactor(i,comp_list)*\
                                                             stress_DFT_vox[comp][voxel_idx]
            # Build stress tensor (frequency domain)
            stress_DFT = \
                  tensorOperations.getTensorFromMatricialForm(stress_DFT_mf,n_dim,comp_list)
            # Add discrete frequency contribution to discrete error required sum
            sum = sum + \
                     np.norm(tensorOperations.dot21_1(stress_DFT,np.asarray(freq_coord)))**2
        # Compute discrete error serving to check convergence
        n_voxels = np.prod(n_voxels_dir)
        discrete_error = math.sqrt(sum/n_voxels)/np.norm(stress_DFT_0_mf)
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
        #
        #                                                                      Update strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~----~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for i in range(len(comp_list)):
            comp = comp_list[i]
            # Update strain
            strain_DFT_vox[comp] = strain_DFT_vox[comp] - \
                    sum([np.multiply(Green_operator_vox[comp+dummy],stress_DFT_vox[dummy]) \
                                                                    for dummy in comp_list])
            # Enforce macroscopic strain at the zero-frequency strain component
            freq_0_idx = n_dim*(0,)
            strain_DFT_vox[comp][freq_0_idx] = (1.0/kelvinFactor(i,comp_list))*mac_strain[i]
        #
        #                                   Strain Inverse Discrete Fourier Transform (IDFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute strain Inverse Discrete Fourier Transform (IDFT)
        for comp in comp_list:
            # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
            # Transform (FFT)
            strain_vox[comp] = np.fft.ifftn(strain_DFT_vox[comp])
        #
        #                                                                      Update stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get voxel material phase
            voxel_idx = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
            phase_idx = regular_grid[voxel_idx] - 1
            # Get material phase elasticity tensor (matricial form)
            De_tensor_mf = De_tensors_mf[phase_idx]
            # Get strain vector for current discrete frequency
            for i in range(len(comp_list)):
                comp = comp_list[i]
                strain_mf[i] = (1.0/kelvinFactor(i,comp_list))*strain[comp][voxel_idx]
            # Update stress for current discrete frequency
            stress_vector = De_tensor_mf*strain_mf
            for i in range(len(comp_list)):
                comp = comp_list[i]
                stress[comp][voxel_idx] = (1.0/kelvinFactor(i,comp_list))*stress_vector[i]
#
#                                                                    Complementary functions
# ==========================================================================================
# Set the coefficient associated to the Kelvin notation when storing a symmetric
# second-order tensor or a minor simmetric fourth-order tensor in matrix form.
# For a given component index in a given component list, this function returns the
# component's associated Kelvin notation factor.
#
# For instance, assuming that the component list is ['11','22','12'] (2D problem) or
# ['11','22','33','12','23','13'] (3D problem), the Kelvin notation matricial form is
# described as follows:
#
# A. Symmetric second-order tensor Aij (Aij=Aji):
#          _       _
#     A = | A11 A12 |      stored as  A = [ A11 A22 sr(2)*A12 ]
#         |_A21 A22_|
#          _           _
#         | A11 A12 A13 |
#     A = | A21 A22 A23 |  stored as  A = [ A11 A22 A33 sr(2)*A12 sr(2)*A23 sr(2)*A13 ]
#         |_A31 A32 A33_|
#
# B. Minor simmetric fourth-order tensor Aijkl (Aijkl=Ajikl=Aijlk=Ajilk):
#                                          _                                     _
#                                         |    A1111        A1122     sr(2)*A1112 |
#     A[i,j,k,l] = Aijkl,  stored as  A = |    A2211        A2222     sr(2)*A2212 |
#      i,j,k,l in [1,2]                   |_sr(2)*A1211  sr(2)*A1222    2*A1212  _|
#
#
#     A[i,j,k,l] = Aijkl, i,j,k,l in [1,2,3]  stored as
#            _                                                                            _
#           |    A1111        A1122        A1133     sr(2)*A1112  sr(2)*A1123  sr(2)*A1113 |
#           |    A2211        A2222        A2233     sr(2)*A2212  sr(2)*A2223  sr(2)*A2213 |
#       A = |    A3311        A3322        A3333     sr(2)*A3312  sr(2)*A3323  sr(2)*A3313 |
#           | sr(2)*A1211  sr(2)*A1222  sr(2)*A1233    2*A1212      2*A1223      2*A1213   |
#           | sr(2)*A2311  sr(2)*A2322  sr(2)*A2333    2*A2312      2*A2323      2*A2313   |
#           |_sr(2)*A1311  sr(2)*A1322  sr(2)*A1333    2*A1312      2*A1323      2*A1313  _|
#
# Note: The sr() stands for square-root of ().
#
def kelvinFactor(idx,comp_list):
    if isinstance(idx,int):
        if int(list(comp_list[idx])[0]) == int(list(comp_list[idx])[1]):
            factor = 1.0
        else:
            factor = math.sqrt(2)
    else:
        factor = 1.0
        for i in idx:
            if int(list(comp_list[i])[0]) != int(list(comp_list[i])[1]):
                factor = factor*math.sqrt(2)
    return factor
# ------------------------------------------------------------------------------------------
# Discrete Dirac's delta function (dij = 1 if i=j, dij = 0 if i!=j).
def Dd(i,j):
    if not isinstance(i,int) or not isinstance(j,int):
          print('\nAbort: The discrete Dirac\'s delta function only accepts two ' + \
                'integer indexes as arguments.\n')
          sys.exit(1)
    value = 1 if i == j else 0
    return value
# ------------------------------------------------------------------------------------------
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
def getElasticityTensor(strain_formulation,problem_type,n_dim,comp_list,properties):
    # Get Young's Modulus and Poisson ratio
    E = properties['E']
    v = properties['v']
    # Compute Lamé parameters
    lam = (E*v)/((1.0+v)*(1.0-2.0*v))
    miu = E/(2.0*(1.0+v))
    # Set required fourth-order tensors
    _,_,_,FOSym,FODiagTrace,_,_ = tensorOperations.setIdentityTensors(n_dim)
    # 2D problem (plane strain)
    if problem_type == 1:
        De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
        De_tensor_mf = tensorOperations.setTensorMatricialForm(De_tensor,n_dim,comp_list)
    # 3D problem
    elif problem_type == 4:
        De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
        De_tensor_mf = tensorOperations.setTensorMatricialForm(De_tensor,n_dim,comp_list)
    # Return
    return De_tensor_mf
#
#                                                                     Validation (temporary)
# ==========================================================================================
if True:
    # Set functions being validated
    val_functions = ['FFTHomogenizationBasicScheme()','getElasticityTensor()']
    # Set functions arguments
    problem_type = 4
    n_dim = 3
    n = [5,5,5]
    n_voxels_dims = tuple([n[i] for i in range(n_dim)])
    discrete_file_path = '/home/bernardoferreira/Documents/SCA/' + \
                         'debug/FFT_Homogenization_Method/RVE_2D_2Phases.rgmsh.npy'
    regular_grid = np.load(discret_file_path)
    n_material_phases = 2
    material_properties = np.zeros((2,2,2),dtype=object)
    material_properties[0,0,0] = 'E' ; material_properties[0,1,0] = 210e9
    material_properties[1,0,0] = 'v' ; material_properties[1,1,0] = 0.3
    material_properties[0,0,1] = 'E' ; material_properties[0,1,1] = 70e9
    material_properties[1,0,1] = 'v' ; material_properties[1,1,1] = 0.33
    # Call function
    FFTHomogenizationBasicScheme(problem_type,n_dim,n_voxels_dims,regular_grid,
                                                      n_material_phases,material_properties)
