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
# Display errors, warnings and built-in exceptions
import errors
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
#   strain[key] = array(d1,d2), where | di is the number of pixels in dimension i
#                                     | key is the strain component defined as '11', '22' or
#                                     |     '12'
#
# B. 3D problem:
#
#   strain[key] = array(d1,d2,d3), where | di is the number of pixels in dimension i
#                                        | key is the strain component defined as '11', '22'
#                                        |     '33', '12', '23' or '13'
#
def FFTHomogenizationBasicScheme(problem_type,n_dim,n_voxels_dims,regular_grid,
                                                     n_material_phases,material_properties):
    #
    #                                                                             Parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set strain components according to problem type
    if problem_type == 1:
        strain_list = ['11','22','12']
    elif problem_type == 4:
        strain_list = ['11','22','33','12','23','13']
    # Set maximum number of iterations
    max_n_iterations = 100
    # Set convergence tolerance
    conv_tol = 1e-6
    #
    #                                                     Material phases elasticity tensors
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_matrices = list()
    for iphase in range(n_material_phases):
        # Set required elastic properties according to material phase constitutive model
        required_props = ['E','v']
        required_values = {prop: None for prop in required_props}
        for iprop in range(len(required_props)):
            match = np.where(material_properties[:,0,iphase]==required_props[iprop])
            if len(match[0]) != 1:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00022',location.filename,location.lineno+1,
                                                         required_props[iprop],iphase+1)
            else:
                required_values[required_props[iprop]] = \
                                               material_properties[match[0][0],1,iphase]
        # Compute elasticity tensor (matricial form)
        De_matrix = getElasticityTensor(strain_formulation,problem_type,n_dim,
                                                                            required_values)
        De_matrices.append(De_matrix)
    #
    #                                                  Reference material elastic properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute reference material compliance tensor (matricial form) according to reference
    # material constitutive assumptions
    if strain_formulation == 1:
        # Set required elastic properties
        required_props = ['E','v']
        required_values = {prop: None for prop in required_props}
        # Set homogeneous reference material elastic properties
        E_ref = 1.0
        v_ref = 0.3
        required_values['E'] = E_ref
        required_values['v'] = v_ref
        # Compute compliance tensor (matricial form)
        Se_matrix_ref = np.linalg.inv(getElasticityTensor(strain_formulation,problem_type,
                                                                     n_dim,required_values))
    #
    #                                                               Frequency discretization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rve_sizes = [1.0,1.0,1.0]
    # Set discrete frequencies (Hz) for each dimension
    freqs_dims = list()
    for i in range(n_dim):
        # Set sampling spatial period
        sampling_period = rve_size[i]/n_voxels_dims[i]
        # >>>> António
        # Set sampling angular frequency
        sampling_freq = 2*math.pi/sampling_period
        # Set discrete frequencies
        freqs_dims.append(sampling_freq*np.linspace(0,1,num=n_voxels_dims[i],endpoint = False))

        # >>>> fft package
        # Set discrete frequencies
        # freqs_dim.append(2*math.pi*np.fft.fftfreq(n_voxels_dim[i],sampling_period)) Better?
        # >>>>
    #
    #                                                      Reference material Green operator
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Storage description:
    # ...
    # Compute Green operator according to strain formulation
    if strain_formulation == 1:
        # Compute reference material Lamé parameters
        lam_ref = (E_ref*v_ref)/((1.0+v_ref)*(1.0-2.0*v_ref))
        miu_ref = E_ref/(2.0*(1.0+v_ref))
        # Compute Green operator reference material related constants
        c1 = 1.0/(4.0*miu_ref)
        c2 = (miu_ref + lam_ref)/(miu_ref*(lam_ref + 2.0*miu_ref))
        # Set mapping between Green operator fourth-order tensor and storing matrix
        aux = list(it.product(strain_list,strain_list))
        fo_indexes = list()
        mat_indexes = list()
        for i in range(len(strain_list)**2):
            fo_indexes.append([int(x)-1 for x in list(aux[i][0]+aux[i][1])])
            mat_indexes.append([x for x in \
                               [strain_list.index(aux[i][0]),strain_list.index(aux[i][1])]])
        # Initialize Green operator (matricial form)
        Green_operator = {str(comp[0]+1)+str(comp[1]+1): np.zeros(tuple(n_voxels_dims)) for comp in mat_indexes}
        # Compute Green operator (matricial form) components
        for i in range(len(mat_indexes)):
            # Get matrix index
            mat_idx = mat_indexes[i]
            # Get fourth-order tensor indexes
            fo_idx = fo_indexes[i]
            # Get Green operator component key
            key = str(mat_idx[0]+1)+str(mat_idx[1]+1)
            # Loop over discrete frequencies
            for freq_coord in it.product(*freqs_dims):
                # Compute frequency vector norm
                freq_norm = np.linalg.norm(freq_coord)
                # Compute first term of Green operator
                first_term = (1.0/freq_norm**2)*(
                       Dd(fo_idx[0],fo_idx[2])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[1]] +
                       Dd(fo_idx[0],fo_idx[3])*freq_coord[fo_idx[1]]*freq_coord[fo_idx[2]] +
                       Dd(fo_idx[1],fo_idx[3])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[2]] +
                       Dd(fo_idx[1],fo_idx[2])*freq_coord[fo_idx[0]]*freq_coord[fo_idx[3]])
                # Compute second term of Green operator
                second_term = -(1.0/freq_norm**4)*(
                                                freq_coord[fo_idx[0]]*freq_coord[fo_idx[1]]*
                                                freq_coord[fo_idx[2]]*freq_coord[fo_idx[3]])
                # Compute Green operator (matricial form) component
                Green_operator[key][freq_coord] = kelvinFactor(mat_idx,mat_indexes)*(
                                                             c1*first_term + c2*second_term)
    #
    #                                                                       Iterative scheme
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if strain_formulation == 1:
        # Initialize strain and stress tensors (matricial form)
        strain = {comp: np.zeros(tuple(n_voxels_dims)) for comp in strain_list}
        stress = {comp: np.zeros(tuple(n_voxels_dims)) for comp in strain_list}
        #
        #                                                            Initial iterative guess
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Loop over discrete frequencies
        for freq_coord in it.product(*freqs_dims):
            # Get voxel material phase
            voxel_id = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
            phase = regular_grid[voxel_id]
            # Get material phase elasticity tensor (matricial form)
            De_matrix = De_matrices[phase-1]
            # Set strain initial iterative guess
            strain_vector = mac_strain
            for i in range(len(strain_list)):
                key = strain_list[i]
                strain[key][freq_coord] = strain_vector[i]
            # Set stress initial iterative guess
            stress_vector = De_matrix*strain_vector
            for i in range(len(strain_list)):
                key = strain_list[i]
                stress[key][freq_coord] = stress_vector[i]
        #
        #                                            Strain Discrete Fourier Transform (DFT)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute strain Discrete Fourier Transform (DFT) (matricial form)
        strain_DFT = {comp: np.zeros(tuple(n_voxels_dims)) for comp in strain_list}
        for i in range(len(strain_list)):
            key = strain_list[i]
            # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
            strain_DFT[key] = np.fft.fftn(strain[key])
        #
        #                                                                   Iterative scheme
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set maximum number of iterations
        max_n_iterations = 100
        # Start iterative loop
        for iter in range(max_n_iterations):
            #
            #                                        Stress Discrete Fourier Transform (DFT)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute stress Discrete Fourier Transform (DFT)
            stress_DFT = {comp: np.zeros(tuple(n_voxels_dims)) for comp in strain_list}
            for i in range(len(strain_list)):
                key = strain_list[i]
                # Discrete Fourier Transform (DFT) by means of Fast Fourier Transform (FFT)
                stress_DFT[key] = np.fft.fftn(stress[key])
            #
            #                                                         Convergence evaluation
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute sum of stress divergence norm for all discrete frequencies and store
            # store zero-frequency stress
            sum = 0
            for freq_coord in it.product(*freqs_dims):
                # Initialize stress auxiliary vectors
                stress_DFT_vector = np.array(len(strain_list))
                stress_DFT_vector_0 = np.array(len(strain_list))
                for i in range(len(strain_list)):
                    key = strain_list[i]
                    # Get stress vector for current discrete frequency
                    stress_DFT_vector[i] = stress_DFT[key][freq_coord]
                    # Store stress vector for zero-frequency
                    if all([list(freqs_dims[x]).index(freq_coord[x]) == 0 for x in range(n_dim)]):
                        stress_DFT_vector_0[i] = stress_DFT[key][freq_coord]
                # Build stress tensor (frequency domain)
                stress_DFT_tensor = tensorOperations.setTensorToMatrix(stress_DFT_vector,True)
                # Add discrete frequency contribution to discrete error required sum
                sum = sum + np.norm(tensorOperations.dot21_1(stress_DFT_tensor,np.asarray(freq_coord)))**2
            # Compute discrete error serving to check convergence
            n_voxels = np.prod(n_voxels_dir)
            discrete_error = math.sqrt(sum/n_voxels)/np.norm(stress_DFT_vector_0)
            # Check if the solution converged
            if discrete_error < conv_tol:
                # Return strain
                return strain
            #
            #                                                                  Update strain
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in range(len(strain_list)):
                key = strain_list[i]
                # Update strain
                strain_DFT[key] = strain_DFT[key] - sum([np.multiply(Green_operator[key+key2],stress_DFT[key2] for key2 in strain_list))
                # Enforce macroscopic strain at the zero-frequency strain component
                freq_0_idx = n_dim*(0,)
                strain_DFT[key][freq_0_idx] = mac_strain[i]
            #
            #                               Strain Inverse Discrete Fourier Transform (IDFT)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute strain Inverse Discrete Fourier Transform (IDFT)
            for i in range(len(strain_list)):
                key = strain_list[i]
                # Inverse Discrete Fourier Transform (IDFT) by means of Fast Fourier
                # Transform (FFT)
                strain[key] = np.fft.ifftn(strain_DFT[key])
            #
            #                                                                  Update stress
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Loop over discrete frequencies
            for freq_coord in it.product(*freqs_dims):
                # Get voxel material phase
                voxel_id = [list(freqs_dims[x]).index(freq_coord[x]) for x in range(n_dim)]
                phase = regular_grid[voxel_id]
                # Get material phase elasticity tensor (matricial form)
                De_matrix = De_matrices[phase-1]
                # Get strain vector for current discrete frequency
                for i in range(len(strain_list)):
                    key = strain_list[i]
                    strain_vector[i] = strain[key][freq_coord]
                # Update stress for current discrete frequency
                stress_vector = De_matrix*strain_vector
                for i in range(len(strain_list)):
                    key = strain_list[i]
                    stress[key][freq_coord] = stress_vector[i]









# ------------------------------------------------------------------------------------------
# Set the coefficient associated to the Kelvin notation when storing a symmetric
# second-order tensor or a minor simmetric fourth-order tensor in matrix form. The storage
# is performed as follows:
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
#
#            _                                                                            _
#           |    A1111        A1122        A1133     sr(2)*A1112  sr(2)*A1123  sr(2)*A1113 |
#           |    A2211        A2222        A2233     sr(2)*A2212  sr(2)*A2223  sr(2)*A2213 |
#       A = |    A3311        A3322        A3333     sr(2)*A3312  sr(2)*A3323  sr(2)*A3313 |
#           | sr(2)*A1211  sr(2)*A1222  sr(2)*A1233    2*A1212      2*A1223      2*A1213   |
#           | sr(2)*A2311  sr(2)*A2322  sr(2)*A2333    2*A2312      2*A2323      2*A2313   |
#           |_sr(2)*A1311  sr(2)*A1322  sr(2)*A1333    2*A1312      2*A1323      2*A1313  _|
#
# Note: The sr() stands for square-root of ().
def kelvinFactor(mat_idx,mat_indexes):
    ref_index = math.ceil(math.sqrt(len(mat_indexes))/2)
    if isinstance(mat_idx,int):
        factor = math.sqrt(2) if mat_idx >= ref_index else 1.0
    else:
        if mat_idx[0] < ref_index and mat_idx[1] < ref_index:
            factor = 1.0
        elif mat_idx[0] >= ref_index and mat_idx[1] >= ref_index:
            factor = 2.0
        else:
            factor = math.sqrt(2)
    return factor
# ------------------------------------------------------------------------------------------
# Discrete Dirac's delta function (dij = 1 if i=j, dij = 0 if i!=j)
def Dd(i,j):
    if not isinstance(i,int) or not isinstance(j,int):
        print('error:not integer')
    value = 1 if i == j else 0
    return value
# ------------------------------------------------------------------------------------------
# Compute the elasticity tensor according to the strain formulation, problem type and
# material constitutive model, storing it in matricial form afterwards. The elasticity
# tensor is described as follows:
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
#      A.1.1 2D problem (plane strain):
#                            _                   _
#                           | 1-v   v      0      | (11)
#          De =  E/(2*(1+v))|  v   1-v     0      | (22)
#                           |_ 0    0   0.5(1-2v)_| (12)
#
#      A.1.2 2D problem (plane stress):
#                           _                _
#                          |  1   v     0     | (11)
#          De =  E/(1-v**2)|  v   1     0     | (22)
#                          |_ 0   0  0.5(1-v)_| (12)
#
#     A.1.3 2D problem (axisymmetric):
#
#                            _                        _
#                           | 1-v   v      0        v  | (11)
#          De =  E/(2*(1+v))|  v   1-v     0        v  | (22)
#                           |  0    0   0.5(1-2v)   0  | (12)
#                           |_ v    v      0       1-v_| (33)
#
#
#     A.1.4 3D problem:
#                            _                                               _
#                           | 1-v   v    v       0          0          0      | (11)
#                           |  v   1-v   v       0          0          0      | (22)
#          De =  E/(2*(1+v))|  v    v  1-v       0          0          0      | (33)
#                           |  0    0    0   0.5(1-2v)      0          0      | (12)
#                           |  0    0    0       0      0.5(1-2v)      0      | (23)
#                           |_ 0    0    0       0          0      0.5(1-2v) _| (13)
#
#    Note: E and v denote the Young's Modulus and the Poisson ratio respectively.
#
def getElasticityTensor(strain_formulation,problem_type,n_dim,properties):
    # Compute elasticity tensor according to the strain formulation, problem type and
    # material constitutive model
    if strain_formulation == 1:
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
            De_matrix = tensorOperations.setTensorToMatrix(De_tensor,True)
        # 3D problem
        elif problem_type == 4:
            De_tensor = lam*FODiagTrace + 2.0*miu*FOSym
            De_matrix = tensorOperations.setTensorToMatrix(De_tensor,True)
    # Return
    return De_matrix

#                                                                                    Testing
# ==========================================================================================
n_material_phases = 2
material_properties = np.zeros((2,2,2),dtype=object)
material_properties[0,0,0] = 'E' ; material_properties[0,1,0] = 210e9
material_properties[1,0,0] = 'v' ; material_properties[1,1,0] = 0.3
material_properties[0,0,1] = 'E' ; material_properties[0,1,1] = 70e9
material_properties[1,0,1] = 'v' ; material_properties[1,1,1] = 0.33
strain_formulation = 1
problem_type = 1
n_dim = 2

n_voxels = 0
n_voxels_dir = 0
regular_grid = 0

#function(strain_formulation,problem_type,n_dim,n_voxels,n_voxels_dir,regular_grid,n_material_phases,material_properties)


# Set strain components according to problem type (move to readInputData?)
if problem_type == 1:
    strain_list = ['11','22','12']
elif problem_type == 4:
    strain_list = ['11','22','33','12','23','13']
# Set mapping between Green operator fourth-order tensor and storing matrix
aux = list(it.product(strain_list,strain_list))
fo_indexes = list()
mat_indexes = list()
for i in range(len(strain_list)**2):
    fo_indexes.append([int(x)-1 for x in list(aux[i][0]+aux[i][1])])
    mat_indexes.append([x for x in \
                       [strain_list.index(aux[i][0]),strain_list.index(aux[i][1])]])
# Initialize Green operator
GreenOperator = {str(comp[0]+1)+str(comp[1]+1): None for comp in mat_indexes}

for i in range(len(strain_list)**2):
    print(mat_indexes[i],'>',fo_indexes[i], ' factor:', kelvinFactor(mat_indexes[i],mat_indexes))
print(GreenOperator)
