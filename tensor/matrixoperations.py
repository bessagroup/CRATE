#
# Matrix Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Matrix operations (e.g. matricial form storage, matrix condensation).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Jan 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Updated documentation.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
#
#                                                                    Problem type parameters
# ==========================================================================================
def get_problem_type_parameters(problem_type):
    '''Get parameters dependent on the problem type.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).

    Returns
    -------
    n_dim : int
        Problem number of spatial dimensions.
    comp_order_sym : list
        Strain/Stress components symmetric order.
    comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    '''
    # Set problem number of spatial dimensions and strain/stress components symmetric and
    # nonsymmetric order
    if problem_type == 1:
        n_dim = 2
        comp_order_sym = ['11', '22', '12']
        comp_order_nsym = ['11', '21', '12', '22']
    elif problem_type == 4:
        n_dim = 3
        comp_order_sym = ['11', '22', '33', '12', '23', '13']
        comp_order_nsym = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
    else:
        raise RuntimeError('Unavailable problem type.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return [n_dim, comp_order_sym, comp_order_nsym]
#
#                                                    Tensorial < > Matricial form conversion
# ==========================================================================================
def gettensormf(tensor, n_dim, comp_order):
    '''Get tensor matricial form.

    Store a given second-order or fourth-order tensor in matricial form for a given number
    of problem spatial dimensions and given ordered strain/stress components list. If the
    second-order tensor is symmetric or the fourth-order tensor has minor symmetry
    (component list only contains independent components), then the Kelvin notation is
    employed to perform the storage. Otherwise, matricial form is built columnwise.

    Parameters
    ----------
    tensor : ndarray
        Tensor to be stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : list
        Strain/Stress components order associated to matricial form.

    Returns
    -------
    tensor_mf : ndarray
        Matricial form of input tensor.
    '''
    # Get tensor order
    tensor_order = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if tensor_order not in [2, 4]:
        raise RuntimeError('Matricial form storage is only available for second-order '
                           'or fourth-order tensors.')
    elif any([int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif any([tensor.shape[i] != n_dim for i in range(len(tensor.shape))]):
        raise RuntimeError('Invalid tensor dimensions.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        is_kelvin_notation = True
    else:
        raise RuntimeError('Invalid number of components in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            if is_kelvin_notation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
            tensor_mf[mf_idx] = factor*tensor[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            if is_kelvin_notation and not (fo_idx[0] == fo_idx[1] and fo_idx[2] == fo_idx[3]):
                factor = factor*np.sqrt(2) if fo_idx[0] != fo_idx[1] else factor
                factor = factor*np.sqrt(2) if fo_idx[2] != fo_idx[3] else factor
            tensor_mf[mf_idx] = factor*tensor[fo_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return tensor_mf
# ------------------------------------------------------------------------------------------
def gettensorfrommf(tensor_mf, n_dim, comp_order):
    '''Recover tensor from associated matricial form.

    Recover a given second-order or fourth-order tensor from the associated matricial form,
    given the problem number of spatial dimensions and given a (compatible) ordered
    strain/stress components list. If the second-order tensor is symmetric or the
    fourth-order tensor has minor symmetry (component list only contains independent
    components), then matricial form is assumed to follow the Kelvin notation. Otherwise,
    a columnwise matricial form is assumed.

    Parameters
    ----------
    tensor_mf : ndarray
        Tensor stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : list
        Strain/Stress components order associated to matricial form.

    Returns
    -------
    tensor : ndarray
        Tensor recovered from matricial form.
    '''
    # Set tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
        if tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim + 1)):
            raise RuntimeError('Invalid number of components in tensor matricial form.')
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
        if tensor_mf.shape[0] != tensor_mf.shape[1]:
            raise RuntimeError('Fourth-order tensor matricial form must be a square '
                               'matrix.')
        elif tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim + 1)):
            raise RuntimeError('Invalid number of components in tensor matricial form.')
    else:
        raise RuntimeError('Tensor matricial form must be a vector or a matrix.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if any([int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Kelvin notation flag
    if len(comp_order) == n_dim**2:
        is_kelvin_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        is_kelvin_notation = True
    else:
        raise RuntimeError('Invalid number of components in strain/stress components '
                           'order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            if is_kelvin_notation and not so_idx[0] == so_idx[1]:
                factor = np.sqrt(2)
                tensor[so_idx[::-1]] = (1.0/factor)*tensor_mf[mf_idx]
            tensor[so_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            if is_kelvin_notation and not (fo_idx[0] == fo_idx[1] and
                                           fo_idx[2] == fo_idx[3]):
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return tensor
# ------------------------------------------------------------------------------------------
def kelvinfactor(idx, comp_order):
    '''Set Kelvin notation coefficient associated to given strain/stress component.

    Parameters
    ----------
    idx : int (or list of int)
        Index of strain/stress component. Alternatively, a pair of strain/stress components
        indexes (associated to a given fourth-order tensor matricial form element) can also
        be provided.
    comp_order : list
        Strain/Stress components order associated to matricial form.

    Returns
    -------
    factor : float
        Kelvin notation coefficient.
    '''
    if isinstance(idx, int) or isinstance(idx, np.integer):
        # Set Kelvin coefficient associated with single strain/stress component
        if int(list(comp_order[idx])[0]) == int(list(comp_order[idx])[1]):
            factor = 1.0
        else:
            factor = np.sqrt(2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif isinstance(idx, list) and len(idx) == 2:
        # Set Kelvin coefficient associated with pair of strain/stress components
        factor = 1.0
        for i in idx:
            if int(list(comp_order[i])[0]) != int(list(comp_order[i])[1]):
                factor = factor*np.sqrt(2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        raise RuntimeError('Invalid strain/stress component(s) index(es).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return factor
#
#                                                                Matricial form condensation
# ==========================================================================================
def getcondmatrix(matrix, rows, cols):
    '''Perform condensation of matrix given a set of rows and columns.

    Parameters
    ----------
    matrix : ndarray
        Matrix to be condensed.
    rows : 1darray
        Indexes of rows to keep in condensed matrix.
    cols : 1darray
        Indexes of columns to keep in condensed matrix.

    Returns
    -------
    matrix_condensed : ndarray
        Condensed matrix.
    '''
    # Check validity of rows and columns indexes to perform the condensation
    if not np.all([isinstance(rows[i], int) or isinstance(rows[i], np.integer)
            for i in range(len(rows))]):
        raise RuntimeError('All the indexes specified to perform a matrix condensation '
                           'must be non-negative integers.')
    elif not np.all([isinstance(cols[i], int) or isinstance(cols[i], np.integer)
            for i in range(len(cols))]):
        raise RuntimeError('All the indexes specified to perform a matrix condensation '
                           'must be non-negative integers.')
    elif len(list(dict.fromkeys(rows))) != len(rows) or \
            len(list(dict.fromkeys(cols))) != len(cols):
        raise RuntimeError('Duplicated rows or columns indexes.')
    elif np.any([rows[i] not in range(matrix.shape[0]) for i in range(len(rows))]):
        raise RuntimeError('Out-of-bounds row index.')
    elif np.any([cols[i] not in range(matrix.shape[1]) for i in range(len(cols))]):
        raise RuntimeError('Out-of-bounds column index.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build auxiliary matrices with rows and columns condensation indexes
    rows_matrix = np.zeros((len(rows), len(cols)), dtype=int)
    cols_matrix = np.zeros((len(rows), len(cols)), dtype=int)
    for j in range(len(cols)):
        rows_matrix[:, j] = rows
    for i in range(len(rows)):
        cols_matrix[i, :] = cols
    # Build condensed matrix
    matrix_condensed = matrix[rows_matrix, cols_matrix]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return condensed matrix
    return matrix_condensed
#
#                                         Strain/Stress 2D < > 3D matricial form conversions
# ==========================================================================================
def getstate3Dmffrom2Dmf(problem_type, mf_2d, comp_33):
    '''Build 3D strain/stress second-order tensor from 2D strain/stress second-order tensor.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    mf_2d : 1darray
        Matricial form of 2D strain/stress second-order tensor.
    comp_33 : float
        Out-of-plane strain/stress component.

    Returns
    -------
    mf_3d : 1darray
        Matricial form of 3D strain/stress second-order tensor.
    '''
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor symmetry
    if len(mf_2d) == len(comp_order_sym_2d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 3D strain/stress second-order tensor (matricial form)
    mf_3d = np.zeros(len(comp_order_3d))
    if problem_type in [3, 4]:
        raise RuntimeError('Unexpected problem type.')
    else:
        # Include out-of-plane strain/stress component under 2D plane strain/stress
        # conditions
        for i in range(len(comp_order_2d)):
            comp = comp_order_2d[i]
            mf_3d[comp_order_3d.index(comp)] = mf_2d[i]
        mf_3d[comp_order_3d.index('33')] = comp_33
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return mf_3d
# ------------------------------------------------------------------------------------------
def getstate2Dmffrom3Dmf(problem_type, mf_3d):
    '''Build 2D counterpart of 3D strain/stress related second- or fourth-order tensor.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    mf_3d : 1darray or 2darray
        Matricial form of 3D strain/stress related tensor.

    Returns
    -------
    mf_2d : 1darray or 2darray
        Matricial form of 2D strain/stress related tensor.
    '''
    # Get 2D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_2d, comp_order_nsym_2d = get_problem_type_parameters(problem_type=1)
    # Get 3D strain/stress components order in symmetric and nonsymmetric cases
    _, comp_order_sym_3d, comp_order_nsym_3d = get_problem_type_parameters(problem_type=4)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required strain/stress component order according to strain tensor symmetry
    if len(mf_3d) == len(comp_order_sym_3d):
        comp_order_2d = comp_order_sym_2d
        comp_order_3d = comp_order_sym_3d
    else:
        comp_order_2d = comp_order_nsym_2d
        comp_order_3d = comp_order_nsym_3d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 2D strain/stress related tensor (matricial form)
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return mf_2d
