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
# Inspect file name and line
import inspect
# Generate efficient iterators
import itertools as it
# Display errors, warnings and built-in exceptions
import errors
# Tensorial operations
import tensorOperations
#
#

#                                                        Compute cluster-defining quantities
# ==========================================================================================
def function(strain_formulation,problem_type,n_dim,n_voxels,n_voxels_dims,regular_grid,n_material_phases,material_properties):

    #                                                     Material phases elasticity tensors
    # ======================================================================================
    # Compute the elasticity tensors (matricial form) associated to each material phase
    De_matrices = list()
    for iphase in range(n_material_phases):
        # Set required elastic properties according to material phase constitutive model
        if strain_formulation == 1:
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
    # ======================================================================================
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
    #                                                           Set frequency discretization
    # ======================================================================================
    rve_sizes = [1.0,1.0,1.0]
    # Set discrete frequencies (Hz) for each dimension
    frequency = list()
    for i in range(n_dim):
        # Set sampling period
        sampling_period = rve_size[i]/n_voxels_dims[i]
        # Set sampling angular frequency
        sampling_freq = 2*math.pi/sampling_period
        # Set discrete frequencies
        freqs_dims[i] = sampling_freq*np.linspace(0,1,num=n_voxels_dims[i],endpoint = False)
    #
    #                                                      Reference material Green operator
    # ======================================================================================
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
            fo_indexes.append([int(x) for x in list(aux[i][0]+aux[i][1])])
            mat_indexes.append([x for x in \
                           [strain_list.index(aux[i][0]),strain_list.index(aux[i][1])]])

        # Initialize Green operator
        GreenOperator = {str(comp[0]+1)+str(comp[1]+1): None for comp in mat_indexes}
        # Compute Green operator component (matricial form)
        for index in range(len(strain_list)):
            key = strain_list[index]
            # Initialize Green operator component
            GreenOperator[key] = np.zeros(tuple(n_voxels_dims))
            # Loop over discrete frequencies
            for freq_coords in it.product(*freqs_dims):
                # Compute frequency vector norm
                freq_norm = np.linalg.norm(freq_coords)

        #




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
problem_type = 4
n_dim = 3

n_voxels = 0
n_voxels_dir = 0
regular_grid = 0

#function(strain_formulation,problem_type,n_dim,n_voxels,n_voxels_dir,regular_grid,n_material_phases,material_properties)


strain_list = ['11','22','12']
aux = list(it.product(strain_list,strain_list))
indexation = list()
location = list()
for i in range(len(strain_list)**2):
    indexation.append([int(x) for x in list(aux[i][0]+aux[i][1])])
    location.append([x for x in [strain_list.index(aux[i][0]),strain_list.index(aux[i][1])]])

GreenOperator = {str(comp[0]+1)+str(comp[1]+1): None for comp in location}

for i in range(len(strain_list)**2):
    print(location[i],'>',indexation[i])
print(GreenOperator)
