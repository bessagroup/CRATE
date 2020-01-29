#
# Tensor Operations Module (UNNAMED Program)
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
# Display errors, warnings and built-in exceptions
import errors
#
#                                                                       Tensorial operations
# ==========================================================================================
# Tensorial products
dyad22 = lambda A2,B2 : np.einsum('ij,kl -> ijkl',A2,B2)
# Tensorial contractions
ddot44_1 = lambda A4,B4 : np.einsum('ijmn,mnkl -> ijkl',A4,B4)
#
#                                                                       Set identity tensors
# ==========================================================================================
# Set the following common identity tensors:
#
#   Second-order identity tensor              > Tij = dii
#   Fourth-order identity tensor              > Tijkl = dik*djl
#   Fourth-order symmetric projection tensor  > Tijkl = 0.5*(dik*djl+dil*djk)
#   Fourth-order 'diagonal trace' tensor      > Tijkl = dij*dkl
#   Fourth-order deviatoric projection tensor > Tijkl = dik*djl-(1/3)*dij*dkl
#   Fourth-order deviatoric projection tensor > Tijkl = 0.5*(dik*djl+dil*djk)-(1/3)*dij*dkl
#            (second order symmetric tensors)
#
#   where 'd' represents the discrete Dirac delta.
#
def setIdentityTensors(n_dim):
    # Set second-order identity tensor
    SOId = np.eye(n_dim)
    # Set fourth-order identity tensor and fourth-order transpose tensor
    FOId = np.zeros((n_dim,n_dim,n_dim,n_dim))
    FOTransp = np.zeros((n_dim,n_dim,n_dim,n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            FOId[i,j,i,j] = 1.0
            FOTransp[i,j,j,i] = 1.0
    # Set fourth-order symmetric projection tensor
    FOSym = 0.5*(FOId + FOTransp)
    # Set fourth-order 'diagonal trace' tensor
    FODiagTrace = dyad22(SOId,SOId)
    # Set fourth-order deviatoric projection tensor
    FODevProj = FOId - (1.0/3.0)*FODiagTrace
    # Set fourth-order deviatoric projection tensor (second order symmetric tensors)
    FODevProjSym = FOSym - (1.0/3.0)*FODiagTrace
    # Return
    return [SOId,FOId,FOTransp,FOSym,FODiagTrace,FODevProj,FODevProjSym]
#
#                                                                Tensor - Matrix conversions
# ==========================================================================================
# Store a given second-order or fourth-order tensor in matricial form. If the second-order
# tensor is symmetric or the fourth-order tensor has minor symmetry, then the Kelvin
# notation is employed to perform the storage. The storage is described as follows:
#
# A. Second-order tensor Aij:
#
#   A.1 Symmetric (Aij=Aji) - Kelvin notation:
#          _       _
#     A = | A11 A12 |      stored as  A = [ A11 A22 sr(2)*A12 ]
#         |_A21 A22_|
#          _           _
#         | A11 A12 A13 |
#     A = | A21 A22 A23 |  stored as  A = [ A11 A22 A33 sr(2)*A12 sr(2)*A23 sr(2)*A13 ]
#         |_A31 A32 A33_|
#
#   A.2 General - Columnwise:
#          _       _
#     A = | A11 A12 |      stored as  A = [ A11 A21 A12 A22 ]
#         |_A21 A22_|
#          _           _
#         | A11 A12 A13 |
#     A = | A21 A22 A23 |  stored as  A = [ A11 A21 A31 A12 A22 A32 A13 A23 A33 ]
#         |_A31 A32 A33_|
#
# B. Fourth-order tensor Aijkl:
#
#   B.1 Minor symmetry (Aijkl=Ajikl=Aijlk=Ajilk) - Kelvin notation:
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
#   B.2 General - Columnwise:
#                                                           _                          _
#                                                          | A1111  A1121  A1112  A1122 |
#     A[i,j,k,l] = Aijkl, i,j,k,l in [1,2]  stored as  A = | A2111  A1221  A1212  A1222 |
#                                                          | A1211  A2121  A2112  A2122 |
#                                                          |_A2211  A2221  A2212  A2222_|
#
#     A[i,j,k,l] = Aijkl, i,j,k,l in [1,2,3]  stored as
#                  _                                                            _
#                 | A1111  A1121  A1131  A1112  A1122  A1132  A1113  A1123  A1133 |
#                 | A2111  A2121  A2131  A2112  A2122  A2132  A2113  A2123  A2133 |
#                 | A3111  A3121  A3131  A3112  A3122  A3132  A3113  A3123  A3133 |
#                 | A1211  A1221  A1231  A1212  A1222  A1232  A1213  A1223  A1233 |
#             A = | A2211  A2221  A2231  A2212  A2222  A2232  A2213  A2223  A2233 |
#                 | A3211  A3221  A3231  A3212  A3222  A3232  A3213  A3223  A3233 |
#                 | A1311  A1321  A1331  A1312  A1322  A1332  A1313  A1323  A1333 |
#                 | A2311  A2321  A2331  A2312  A2322  A2332  A2313  A2323  A2333 |
#                 |_A3311  A3321  A3331  A3312  A3322  A3332  A3313  A3323  A3333_|
#
# Note: The sr() stands for square-root of ().
#
def setTensorToMatrix(tensor,isSym):
    # Check if valid second-order or fourth-order tensor
    if np.ndim(tensor) not in [2,4]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00018',location.filename,location.lineno+1)
    elif not all(x == tensor.shape[0] for x in tensor.shape) or \
                                                               tensor.shape[0] not in [2,3]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00019',location.filename,location.lineno+1)
    # Set dimension
    n_dim = tensor.shape[0]
    # Set matrix storage index order (and weights associated to the Kelvin notation)
    if n_dim == 2:
        if isSym:
            indexes = np.array([[1,1],[2,2],[1,2]])
            weights = np.array([1,1,math.sqrt(2)])
        else:
            indexes = np.array([[1,1],[2,1],[1,2],[2,2]])
    elif n_dim == 3:
        if isSym:
            indexes = np.array([[1,1],[2,2],[3,3],[1,2],[2,3],[1,3]])
            weights = np.array([1,1,1,math.sqrt(2),math.sqrt(2),math.sqrt(2)])
        else:
            indexes = np.array([[1,1],[2,1],[3,1],[1,2],[2,2],[3,2],[1,3],[2,3],[3,3]])
    # Store tensor in matricial form
    n_index = indexes.shape[0]
    if np.ndim(tensor) == 2:
        matrix = np.zeros(n_index)
        for iStore in range(n_index):
            i = indexes[iStore,0] - 1
            j = indexes[iStore,1] - 1
            matrix[iStore] = tensor[i,j]
            if isSym:
                matrix[iStore] = weights[iStore]*matrix[iStore]
    elif np.ndim(tensor) ==4:
        matrix = np.zeros((n_index,n_index))
        for iStore in range(n_index):
            for jStore in range(n_index):
                i = indexes[iStore,0] - 1
                j = indexes[iStore,1] - 1
                k = indexes[jStore,0] - 1
                l = indexes[jStore,1] - 1
                matrix[iStore,jStore] = tensor[i,j,k,l]
                if isSym:
                    matrix[iStore,jStore] = \
                                       weights[iStore]*weights[jStore]*matrix[iStore,jStore]
    # Return
    return matrix
# ------------------------------------------------------------------------------------------
# Recover a given second-order or fourth-order tensor from its matricial form. If the
# second-order tensor is symmetric or the fourth-order tensor has minor symmetry, then the
# associated matricial form follows the Kelvin notation.
def getTensorFromMatrix(matrix):
    # Check if valid matrix
    if np.ndim(matrix) not in [1,2]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00020',location.filename,location.lineno+1)
    elif np.ndim(matrix) == 2 and matrix.shape[0] != matrix.shape[1]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00020',location.filename,location.lineno+1)
    elif matrix.shape[0] not in [3,4,6,9]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00021',location.filename,location.lineno+1)
    # Set dimension
    if matrix.shape[0] in [3,4]:
        n_dim = 2
    else:
        n_dim = 3
    # Set matrix storage index order (and weights associated to the Kelvin notation)
    if n_dim == 2:
        if matrix.shape[0] == 3:
            indexes = np.array([[1,1],[2,2],[1,2]])
            weights = np.array([1,1,math.sqrt(2)])
            isSym = True
        else:
            indexes = np.array([[1,1],[2,1],[1,2],[2,2]])
            isSym = False
    elif n_dim == 3:
        if matrix.shape[0] == 6:
            indexes = np.array([[1,1],[2,2],[3,3],[1,2],[2,3],[1,3]])
            weights = np.array([1,1,1,math.sqrt(2),math.sqrt(2),math.sqrt(2)])
            isSym = True
        else:
            indexes = np.array([[1,1],[2,1],[3,1],[1,2],[2,2],[3,2],[1,3],[2,3],[3,3]])
            isSym = False
    # Initialize tensor
    if np.ndim(matrix) == 1:
        if matrix.shape[0] in [3,4]:
            tensor = np.zeros((2,2))
        else:
            tensor = np.zeros((3,3))
    else:
        if matrix.shape[0] in [3,4]:
            tensor = np.zeros((2,2,2,2))
        else:
            tensor = np.zeros((3,3,3,3))
    # Recover tensor from matrix
    n_index = indexes.shape[0]
    if np.ndim(tensor) == 2:
        for iStore in range(n_index):
            i = indexes[iStore,0] - 1
            j = indexes[iStore,1] - 1
            tensor[i,j] = matrix[iStore]
            if isSym and iStore > n_dim - 1:
                tensor[i,j] = (1.0/weights[iStore])*tensor[i,j]
                tensor[j,i] = tensor[i,j]
    else:
        for iStore in range(n_index):
            for jStore in range(n_index):
                i = indexes[iStore,0] - 1
                j = indexes[iStore,1] - 1
                k = indexes[jStore,0] - 1
                l = indexes[jStore,1] - 1
                tensor[i,j,k,l] = matrix[iStore,jStore]
                if isSym and (iStore > n_dim - 1 or jStore > n_dim - 1):
                    tensor[i,j,k,l] = \
                                 (1.0/weights[iStore])*(1.0/weights[jStore])*tensor[i,j,k,l]
                    tensor[j,i,k,l] = tensor[i,j,k,l]
                    tensor[i,j,l,k] = tensor[i,j,k,l]
                    tensor[j,i,l,k] = tensor[i,j,k,l]
    # Return
    return tensor
#
#                                       Check identity tensors and tensor-matrix conversions
#                                                                                (temporary)
# ==========================================================================================
if False:
    # Set identity tensors
    SOId,FOId,FOTransp,FOSym,FODiagTrace,FODevProj,FODevProjSym = setIdentityTensors(3)
    # Print identity tensors in matricial form
    print('\nCheck identity tensors:')
    print('\nSOId (matricial form):')
    print(setTensorToMatrix(SOId,False))
    print('\nFOId (matricial form):')
    print(setTensorToMatrix(FOId,False))
    print('\nFOTransp (matricial form):')
    print(setTensorToMatrix(FOTransp,False))
    print('\nFOSym (matricial form):')
    print(setTensorToMatrix(FOSym,True))
    print('\nFODiagTrace (matricial form):')
    print(setTensorToMatrix(FODiagTrace,True))
    print('\nFODevProj (matricial form):')
    print(setTensorToMatrix(FODevProj,False))
    # Check tensor-matrix conversions
    print('\nCheck tensor-matrix conversions:\n')
    print('SOId:       ', np.all(getTensorFromMatrix(setTensorToMatrix(SOId,True))==SOId))
    print('FOId:       ', np.all(getTensorFromMatrix(setTensorToMatrix(FOId,False))==FOId))
    print('FOTransp:   ', \
                   np.all(getTensorFromMatrix(setTensorToMatrix(FOTransp,False))==FOTransp))
    print('FOSym:      ', np.all(getTensorFromMatrix(setTensorToMatrix(FOSym,True))==FOSym))
    print('FODiagTrace:', \
              np.all(getTensorFromMatrix(setTensorToMatrix(FODiagTrace,True))==FODiagTrace))
    print('FODevProj:  ', \
                 np.all(getTensorFromMatrix(setTensorToMatrix(FODevProj,False))==FODevProj))
