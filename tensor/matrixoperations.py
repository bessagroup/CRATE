#
# Matrix Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Matrix operations (e.g. matricial form storage, condensation).
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
import ioput.errors as errors
# Tensorial operations
import tensor.tensoroperations as top
#
#                                                                    Problem type parameters
# ==========================================================================================
# Get parameters dependent on the problem type
def getproblemtypeparam(problem_type):
    # Set problem dimension and strain/stress components order in symmetric and nonsymmetric
    # cases
    if problem_type == 1:
        n_dim = 2
        comp_order_sym = ['11', '22', '12']
        comp_order_nsym = ['11', '21', '12', '22']
    elif problem_type == 2:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00017', location.filename, location.lineno + 1, problem_type)
    elif problem_type == 3:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00017', location.filename, location.lineno + 1, problem_type)
    elif problem_type == 4:
        n_dim = 3
        comp_order_sym = ['11', '22', '33', '12', '23', '13']
        comp_order_nsym = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
    return [n_dim, comp_order_sym, comp_order_nsym]
#
#                                                    Tensorial < > Matricial form conversion
# ==========================================================================================
# Store a given second-order or fourth-order tensor in matricial form for a given number of
# dimensions and given ordered component list. If the second-order tensor is symmetric or
# the fourth-order tensor has minor symmetry (component list only contains independent
# components), then the Kelvin notation is employed to perform the storage. The tensor
# recovery from the associated matricial form follows a precisely inverse procedure.
#
# For instance, assume the following four examples:
#
# 	(a) n_dim = 2, comp_order = ['11','22','12'] (symmetry)
# 	(b) n_dim = 3, comp_order = ['11','22','33','12','23','13'] (symmetry)
# 	(c) n_dim = 2, comp_order = ['11','21','12','22']
# 	(d) n_dim = 3, comp_order = ['11','21','31','21','22','32','13','23','33']
#
# For a given second-order or fourth-order tensor, the storage is performed as described
# below:
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
def gettensormf(tensor, n_dim, comp_order):
    # Set tensor order
    tensor_order = len(tensor.shape)
    # Check input arguments validity
    if tensor_order not in [2, 4]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00023', location.filename, location.lineno + 1)
    elif any([ int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif any([tensor.shape[i] != n_dim for i in range(len(tensor.shape))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00025', location.filename, location.lineno + 1)
    elif any([len(comp) != 2 for comp in comp_order]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00026', location.filename, location.lineno + 1)
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        isKelvinNotation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        isKelvinNotation = True
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00027', location.filename, location.lineno + 1)
    # Store tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # Initialize tensor matricial form
        if tensor.dtype == 'complex':
            tensor_mf = np.zeros(len(comp_order), dtype=complex)
        else:
            tensor_mf = np.zeros(len(comp_order))
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if isKelvinNotation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
            tensor_mf[mf_idx] = factor*tensor[so_idx]
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0]+comps[i][1])])
            mf_indexes.append([x for x in \
                [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
        # Initialize tensor matricial form
        if tensor.dtype == 'complex':
            tensor_mf = np.zeros((len(comp_order), len(comp_order)), dtype=complex)
        else:
            tensor_mf = np.zeros((len(comp_order), len(comp_order)))
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if isKelvinNotation and not (fo_idx[0] == fo_idx[1] and fo_idx[2] == fo_idx[3]):
                factor = factor*np.sqrt(2) if fo_idx[0] != fo_idx[1] else factor
                factor = factor*np.sqrt(2) if fo_idx[2] != fo_idx[3] else factor
            tensor_mf[mf_idx] = factor*tensor[fo_idx]
    # Return
    return tensor_mf
# ------------------------------------------------------------------------------------------
def gettensorfrommf(tensor_mf, n_dim, comp_order):
    # Set tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
        if tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim + 1)):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00028', location.filename, location.lineno + 1)
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
        if tensor_mf.shape[0] != tensor_mf.shape[1]:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00029', location.filename, location.lineno + 1)
        elif tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim + 1)):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00028', location.filename, location.lineno + 1)
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00030', location.filename, location.lineno + 1)
    # Check input arguments validity
    if any([int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif any([len(comp) != 2 for comp in comp_order]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00026', location.filename, location.lineno + 1)
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        isKelvinNotation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        isKelvinNotation = True
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00027', location.filename, location.lineno + 1)
    # Get tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # Initialize tensor
        if tensor_mf.dtype == 'complex':
            tensor = np.zeros(tensor_order*(n_dim,), dtype=complex)
        else:
            tensor = np.zeros(tensor_order*(n_dim,))
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if isKelvinNotation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
                tensor[so_idx[::-1]] = (1.0/factor)*tensor_mf[mf_idx]
            tensor[so_idx] = (1.0/factor)*tensor_mf[mf_idx]
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        mf_indexes = list()
        fo_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in \
                [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
        # Initialize tensor
        if tensor_mf.dtype == 'complex':
            tensor = np.zeros(tensor_order*(n_dim ,), dtype=complex)
        else:
            tensor = np.zeros(tensor_order*(n_dim ,))
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if isKelvinNotation and not (fo_idx[0] == fo_idx[1] and fo_idx[2] == fo_idx[3]):
                factor = factor*np.sqrt(2) if fo_idx[0] != fo_idx[1] else factor
                factor = factor*np.sqrt(2) if fo_idx[2] != fo_idx[3] else factor
                if fo_idx[0] != fo_idx[1] and fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[1::-1]+fo_idx[2:])] = (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[:2]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[1::-1]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[0] != fo_idx[1]:
                    tensor[tuple(fo_idx[1::-1]+fo_idx[2:])] = (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[:2]+fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
            tensor[fo_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # Return
    return tensor
# ------------------------------------------------------------------------------------------
# Set the coefficient associated to the Kelvin notation when storing a symmetric
# second-order tensor or a minor simmetric fourth-order tensor in matricial form. For a
# given component index in a given component list, this function returns the component's
# associated Kelvin notation factor.
def kelvinfactor(idx, comp_order):
    if isinstance(idx, int) or isinstance(idx, np.integer):
        if int(list(comp_order[idx])[0]) == int(list(comp_order[idx])[1]):
            factor = 1.0
        else:
            factor = np.sqrt(2)
    else:
        factor = 1.0
        for i in idx:
            if int(list(comp_order[i])[0]) != int(list(comp_order[i])[1]):
                factor = factor*np.sqrt(2)
    return factor
#
#                                                                Matricial form condensation
# ==========================================================================================
# Perform the condensation of a given matrix A (n x m), returning a matrix B (p,q) with the
# matrix A elements specified by a given list of p rows indexes and q columns indexes. The
# following example suffices to understand the procedure:
#
#         rows = [0,2,3] , cols = [0,2]
#                      _       _                                    _   _
#                     | 1 2 3 4 |                                  | 1 3 |
#            matrix = | 5 6 7 8 |              >      matrix_cnd = | 9 7 |
#                     | 9 8 7 6 |                                  |_5 3_|
#                     |_5 4 3 2_|
#
# Note: Lists of rows and columns cannot contain duplicated indexes
#
def getcondmatrix(matrix, rows, cols):
    # Check validity of rows and columns indexes to perform the condensation
    if not np.all([isinstance(rows[i], int) for i in range(len(rows))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00032', location.filename, location.lineno + 1)
    elif not np.all([isinstance(cols[i], int) for i in range(len(cols))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00032', location.filename, location.lineno + 1)
    elif len(list(dict.fromkeys(rows))) != len(rows) or \
            len(list(dict.fromkeys(cols))) != len(cols):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00033', location.filename, location.lineno + 1)
    elif np.any([rows[i] not in range(matrix.shape[0]) for i in range(len(rows))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00034', location.filename, location.lineno + 1)
    elif np.any([cols[i] not in range(matrix.shape[1]) for i in range(len(cols))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00035', location.filename, location.lineno + 1)
    # Build auxiliary matrices with rows and columns condensation indexes
    rows_matrix = np.zeros((len(rows), len(cols)), dtype=int)
    cols_matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for j in range(len(cols)):
        rows_matrix[:, j] = rows
    for i in range(len(rows)):
        cols_matrix[i, :] = cols
    # Build condensed matrix
    matrix_cnd = matrix[rows_matrix, cols_matrix]
    # Return condensed matrix
    return matrix_cnd
#
#                                         Strain/Stress 2D < > 3D matricial form conversions
# ==========================================================================================
# Given a 2D strain/stress tensor (matricial form) associated to a given 2D problem type,
# build the corresponding 3D counterpart by including the appropriate out-of-plain strain
# components
def getstate3Dmffrom2Dmf(problem_dict, mf_2d, comp_33):
    # Get problem type
    problem_type = problem_dict['problem_type']
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = getproblemtypeparam(problem_type)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = getproblemtypeparam(4)
    # Set required strain/stress component order according to strain tensor symmetry
    if len(mf_2d) == len(comp_order_sym_2d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # Build 3D strain tensor (matricial form)
    mf_3d = np.zeros(len(comp_order_3d))
    if problem_type == 1:
        for i in range(len(comp_order_2d)):
            comp = comp_order_2d[i]
            mf_3d[comp_order_3d.index(comp)] = mf_2d[i]
        mf_3d[comp_order_3d.index('33')] = comp_33
    # Return
    return mf_3d
# ------------------------------------------------------------------------------------------
# Given a 3D strain/stress second-order tensor (matricial form) or a 3D strain/stress
# related fourth-order tensor associated to a given 2D problem type, build the reduced 2D
# counterpart including only the in-plain strain/stress components
def getstate2Dmffrom3Dmf(problem_dict, mf_3d):
    # Get problem type
    problem_type = problem_dict['problem_type']
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = getproblemtypeparam(problem_type)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = getproblemtypeparam(4)
    # Set required strain/stress component order according to strain tensor symmetry
    if len(mf_3d) == len(comp_order_sym_3d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # Build 2D tensor (matricial form)
    mf_2d = np.zeros(len(mf_3d.shape)*(len(comp_order_2d),))
    if len(mf_3d.shape) == 1:
        for i in range(len(comp_order_2d)):
            comp = comp_order_2d[i]
            mf_2d[i] = mf_3d[comp_order_3d.index(comp)]
    elif len(mf_3d.shape) == 2:
        for j in range(len(comp_order_2d)):
            comp_j = comp_order_2d[j]
            for i in range(len(comp_order_2d)):
                comp_i = comp_order_2d[i]
                mf_2d[i,j] = mf_3d[comp_order_3d.index(comp_i), comp_order_3d.index(comp_j)]
    return mf_2d
#
#                                                                     Validation (temporary)
# ==========================================================================================
if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['gettensormf()','gettensorfrommf()']
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set functions arguments
    tensor = np.ones((3,3,3,3))
    n_dim = 3
    comp_order = ['11','22','33','12','23','13']
    # Save original tensor
    original_tensor = tensor
    # Set tensor matricial form
    tensor_mf = gettensormf(tensor,n_dim,comp_order)
    # Get tensor back from matricial form
    tensor = gettensorfrommf(tensor_mf,n_dim,comp_order)
    # Display validation
    print('\nNumber of dimensions: ',n_dim)
    print('Component list      : ',comp_order)
    print('\nTensor:'+'\n\n',original_tensor)
    print('\nMatricial form:'+'\n\n',tensor_mf)
    # Display validation footer
    print('\n' + 92*'-' + '\n')

if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['getidoperators()',]
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set function arguments
    n_dim = 3
    # Set identity tensors
    soid,foid,fotransp,fosym,fodiagtrace,fodevproj,fodevprojsym = top.getidoperators(n_dim)
    # Display validation
    print('\nCheck identity tensors:')
    print('\nSOId (matricial form):')
    comp_order = ['11','22','33','12','23','13']
    print(gettensormf(soid,n_dim,comp_order))
    print('\nFOId (matricial form):')
    comp_order = ['11','21','31','12','22','32','13','23','33']
    print(gettensormf(foid,n_dim,comp_order))
    print('\nFOTransp (matricial form):')
    comp_order = ['11','21','31','12','22','32','13','23','33']
    print(gettensormf(fotransp,n_dim,comp_order))
    print('\nFOSym (matricial form):')
    comp_order = ['11','22','33','12','23','13']
    print(gettensormf(fosym,n_dim,comp_order))
    print('\nFODiagTrace (matricial form):')
    comp_order = ['11','22','33','12','23','13']
    print(gettensormf(fodiagtrace,n_dim,comp_order))
    print('\nFODevProj (matricial form):')
    comp_order = ['11','21','31','12','22','32','13','23','33']
    print(gettensormf(fodevproj,n_dim,comp_order))
    # Check tensor-matrix conversions
    print('\nCheck tensor-matrix conversions:\n')
    print('soid:       ', np.all(gettensorfrommf(gettensormf(soid,\
                                                 n_dim,comp_order),n_dim,comp_order)==soid))
    print('foid:       ', np.all(gettensorfrommf(gettensormf(foid,\
                                                 n_dim,comp_order),n_dim,comp_order)==foid))
    print('fotransp:   ', \
                   np.all(gettensorfrommf(gettensormf(fotransp,\
                                             n_dim,comp_order),n_dim,comp_order)==fotransp))
    print('fosym:      ', np.all(gettensorfrommf(gettensormf(fosym,\
                                                n_dim,comp_order),n_dim,comp_order)==fosym))
    print('fodiagtrace:', \
              np.all(gettensorfrommf(gettensormf(fodiagtrace,n_dim,\
                                                comp_order),n_dim,comp_order)==fodiagtrace))
    print('fodevproj:  ', \
                 np.all(gettensorfrommf(gettensormf(fodevproj,n_dim,\
                                                  comp_order),n_dim,comp_order)==fodevproj))
    # Display validation footer
    print('\n' + 92*'-' + '\n')

if __name__ == '__main__':
    # Set functions being validated
    val_functions = ['getcondmatrix()',]
    # Display validation header
    print('\nValidation: ',(len(val_functions)*'{}, ').format(*val_functions), 3*'\b', ' ')
    print(92*'-')
    # Set function arguments
    matrix = np.array([[0,1,2,3],[3,4,5,1],[6,7,8,2],[9,10,11,0]])
    rows = [1,3]
    cols = [0,2,3]
    # Call function
    matrix_cnd = getcondmatrix(matrix,rows,cols)
    # Display validation
    print('\nrows:', rows)
    print('\ncols:', cols)
    print('\nmatrix:')
    print(matrix)
    print('\ncondensed matrix:')
    print(matrix_cnd)
    # Display validation footer
    print('\n' + 92*'-' + '\n')
