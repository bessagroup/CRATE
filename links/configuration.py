#
# Links Configuration (CRATE Program)
# ==========================================================================================
# Summary:
# Configuration of the Links settings and associated procedures.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Nov 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                                             Links settings
# ==========================================================================================
def get_links_analysis_type(problem_type):
    '''Get Links analysis type corresponding to CRATE problem type.

    Parameters
    ----------
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).

    Returns
    -------
    analysis_type : int
        Links analysis type: 2D plane stress (1), 2D plane strains (2), 2D axisymmetric (3)
        and 3D (6).
    '''
    # Set CRATE (key) - Links (item) problem type conversion
    problem_type_converter = {'1': 2, '2': 1, '3': 3, '4': 6}
    # Get Links analysis type
    analysis_type = problem_type_converter[str(problem_type)]
    # Return
    return analysis_type
# ------------------------------------------------------------------------------------------
# Set Links strain/stress components order in symmetric and nonsymmetric cases
def get_links_comp_order(n_dim):
    '''Get Links strain/stress components order.

    Parameters
    ----------
    n_dim : int
        Problem number of spatial dimensions.

    Returns
    -------
    links_comp_order_sym : list
        Links strain/stress components symmetric order.
    links_comp_order_nsym : list
        Links strain/stress components nonsymmetric order.
    '''
    # Set Links strain/stress components order
    if n_dim == 2:
        links_comp_order_sym = ['11', '22', '12']
        links_comp_order_nsym = ['11', '21', '12', '22']
    else:
        links_comp_order_sym = ['11', '22', '33', '12', '23', '13']
        links_comp_order_nsym = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
    # Return
    return links_comp_order_sym, links_comp_order_nsym
# ------------------------------------------------------------------------------------------
def get_links_dims(strain_formulation, problem_type, iprops, rprops, rstava, lalgva,
                   ralgva):
    '''Get Links dimensions.

    Parameters
    ----------
    strain_formulation: str, {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    iprops : 1darray
        Integer material properties.
    rprops : 1darray
        Real material properties.
    rstava : 1darray
        Real state variables array.
    lalgva : 1darray
        Logical algorithmic variables array.
    ralgva : 1darray
        Real algorithmic variables array.

    Returns
    -------
    nxxxx : int
        Links dimensions.
    '''
    # Set Links strain formulation flag
    if strain_formulation == 'infinitesimal':
        nlarge = 0
    else:
        nlarge = 1
    # Set Links problem type
    ntype = get_links_analysis_type(str(problem_type))
    # Set Links general dimensions
    if ntype in [1, 2, 3]:
        ndim, nstre, nstra, nddim, nadim = 2, 4, 4, 4, 5
    else:
        ndim, nstre, nstra, nddim, nadim = 3, 6, 6, 6, 9
    # Set Links dimensions associated to material properties arrays
    niprop = len(iprops)
    nrprop = len(rprops)
    # Set Links dimensions associated to state and algorithmic variables arrays
    nrstav = len(rstava)
    nlalgv = len(lalgva)
    nralgv = len(ralgva)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return nlarge, ntype, ndim, nstre, nstra, nddim, nadim, niprop, nrprop, nrstav, \
           nlalgv, nralgv
# ------------------------------------------------------------------------------------------
def build_xmatx(nddim, nadim):
    '''Initialize Links consistent tangent modulus.

    Parameters
    ----------
    nddim : int
        Dimension of infinitesimal strains consistent tangent modulus.
    nadim : int
        Dimension of finite strains spatial consistent tangent modulus.

    Returns
    -------
    dmatx : 2darray
        Infinitesimal strains consistent tangent modulus.
    amatx : 2darray
        Finite strains spatial consistent tangent modulus.
    '''
    # Initialize consistent tangent modulii
    dmatx = np.zeros((nddim, nddim))
    amatx = np.zeros((nadim, nadim))
    # Return
    return dmatx, amatx
# ------------------------------------------------------------------------------------------
def get_consistent_tangent_from_xmatx(strain_formulation, problem_type, dmatx, amatx):
    '''Get consistent tangent modulus from Links counterpart.

    Parameters
    ----------
    strain_formulation: str, {'infinitesimal', 'finite'}
        Problem strain formulation.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).
    dmatx : 2darray
        Infinitesimal strains consistent tangent modulus.
    amatx : 2darray
        Finite strains spatial consistent tangent modulus.

    Returns
    -------
    consistent_tangent_mf : ndarray
        Material constitutive model material consistent tangent modulus in matricial form.
    '''
    # Get problem type parameters
    n_dim, comp_order_sym, comp_order_nsym = mop.get_problem_type_parameters(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get consistent tangent modulus (matricial form) according to strain formulation
    if strain_formulation == 'infinitesimal':
        # Get infinitesimal strains consistent tangent modulus
        if problem_type == 1:
            consistent_tangent_mf = \
                mop.get_tensor_mf(get_tensor_from_mf_links(dmatx[0:3, 0:3], n_dim,
                    comp_order_sym, 'elasticity'), n_dim, comp_order_sym)
        else:
            consistent_tangent_mf = \
                mop.get_tensor_mf(get_tensor_from_mf_links(dmatx, n_dim, comp_order_sym,
                    'elasticity'), n_dim, comp_order_sym)
    else:
        # Get finite strains spatial consistent tangent modulus
        if problem_type == 1:
            consistent_tangent_mf = \
                mop.get_tensor_mf(get_tensor_from_mf_links(amatx[0:4, 0:4], n_dim,
                    comp_order_sym, 'elasticity'),n_dim, comp_order_nsym)
        else:
            consistent_tangent_mf = \
                mop.get_tensor_mf(get_tensor_from_mf_links(amatx, n_dim, comp_order_nsym,
                    'elasticity'), n_dim, comp_order_sym)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return consistent_tangent_mf
#
#                                                         Links Tensorial < > Matricial Form
# ==========================================================================================
def get_tensor_mf_links(tensor, n_dim, comp_order, nature):
    '''Get tensor matricial form.

    Store a given second-order or fourth-order tensor in matricial form for a given number
    of problem spatial dimensions and given ordered strain/stress components list. If the
    second-order tensor is symmetric or the fourth-order tensor has minor symmetry
    (component list only contains independent components), then the Voigt notation is
    employed to perform the storage. Otherwise, matricial form is built columnwise.

    Parameters
    ----------
    tensor : ndarray
        Tensor to be stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : list
        Strain/Stress components order associated to matricial form.
    nature : str, {'strain', 'stress', 'elasticity', 'compliance'}
        Nature of tensor to be stored in matricial form: 'strain' or 'stress' second-order
        tensor, or 'elasticity' or 'compliance' fourth-order tensor.

    Returns
    -------
    tensor_mf : ndarray
        Matricial form of input tensor.
    '''
    # Set tensor order
    tensor_order = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if tensor_order not in [2, 4]:
        raise RuntimeError('Matricial form storage is only available for second- and ' +
                           'fourth-order tensors.')
    elif any([int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif any([tensor.shape[i] != n_dim for i in range(len(tensor.shape))]):
        raise RuntimeError('Invalid tensor dimensions.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Voigt notation flag
    if len(comp_order) == n_dim**2:
        is_voigt_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        if nature not in ['strain', 'stress', 'elasticity', 'compliance']:
            raise RuntimeError('Unknown tensor nature to be stored in matricial form ' +
                               'following Voigt notation.')
        if nature in ['strain', 'compliance']:
            is_voigt_notation = True
        else:
            is_voigt_notation = False
    else:
        raise RuntimeError('Invalid number of components in component order list.')
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
            if is_voigt_notation and not so_idx[0] == so_idx[1]:
                factor = 2.0
            tensor_mf[mf_idx] = factor*tensor[so_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
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
            if is_voigt_notation and not (fo_idx[0] == fo_idx[1] and \
                    fo_idx[2] == fo_idx[3]):
                factor = factor*2.0 if fo_idx[0] != fo_idx[1] else factor
                factor = factor*2.0 if fo_idx[2] != fo_idx[3] else factor
            tensor_mf[mf_idx] = factor*tensor[fo_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return tensor_mf
# ------------------------------------------------------------------------------------------
def get_tensor_from_mf_links(tensor_mf, n_dim, comp_order, nature):
    '''Recover tensor from associated matricial form.

    Recover a given second-order or fourth-order tensor from the associated matricial form,
    given the problem number of spatial dimensions and given a (compatible) ordered
    strain/stress components list. If the second-order tensor is symmetric or the
    fourth-order tensor has minor symmetry (component list only contains independent
    components), then matricial form is assumed to follow the Voigt notation. Otherwise,
    a columnwise matricial form is assumed.

    Parameters
    ----------
    tensor_mf : ndarray
        Tensor stored in matricial form.
    n_dim : int
        Problem number of spatial dimensions.
    comp_order : list
        Strain/Stress components order associated to matricial form.
    nature : str, {'strain', 'stress', 'elasticity', 'compliance'}
        Nature of tensor to be stored in matricial form: 'strain' or 'stress' second-order
        tensor, or 'elasticity' or 'compliance' fourth-order tensor.

    Returns
    -------
    tensor : ndarray
        Tensor recovered from matricial form.
    '''
    # Set tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
        if tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim+1)):
            raise RuntimeError('Invalid number of components in tensor matricial form.')
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
        if tensor_mf.shape[0] != tensor_mf.shape[1]:
            raise RuntimeError('Fourth-order tensor matricial form must be a square '
                               'matrix.')
        elif tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim+1)):
            raise RuntimeError('Invalid number of components in tensor matricial form.')
    else:
        raise RuntimeError('Tensor matricial form must be a vector or a matrix.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check input arguments validity
    if any([ int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif any([len(comp) != 2 for comp in comp_order]):
        raise RuntimeError('Invalid component in strain/stress components order.')
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        raise RuntimeError('Duplicated component in strain/stress components order.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Voigt notation flag
    if len(comp_order) == n_dim**2:
        is_voigt_notation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        if nature not in ['strain', 'stress', 'elasticity', 'compliance']:
            raise RuntimeError('Unknown tensor nature to be stored in matricial form ' +
                               'following Voigt notation.')
        if nature in ['strain', 'compliance']:
            is_voigt_notation = True
        else:
            is_voigt_notation = False
    else:
        raise RuntimeError('Invalid number of components in component order list.')
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
            if is_voigt_notation and not so_idx[0] == so_idx[1]:
                factor = 2.0
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
            tensor = np.zeros(tensor_order*(n_dim,), dtype=complex)
        else:
            tensor = np.zeros(tensor_order*(n_dim,))
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if is_voigt_notation and not (fo_idx[0] == fo_idx[1] and
                                          fo_idx[2] == fo_idx[3]):
                factor = factor*2.0 if fo_idx[0] != fo_idx[1] else factor
                factor = factor*2.0 if fo_idx[2] != fo_idx[3] else factor
                if fo_idx[0] != fo_idx[1] and fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[1::-1] + fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[:2] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[1::-1] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[0] != fo_idx[1]:
                    tensor[tuple(fo_idx[1::-1] + fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[:2] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
            tensor[fo_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return tensor
