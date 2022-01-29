#
# Tensor Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Algebraic tensorial operations and standard tensorial operators.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Jan 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Spectral decomposition of second-order symmetric
#                                 | tensors.
#                                 | Isotropic tensor-valued functions and associated
#                                 | derivatives.
# Bernardo P. Ferreira | Jan 2022 | Rotation of n-dimensional tensor.
#                                 | Rotation tensor from euler angles (Bunge convention).
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Regular expressions
import re
#                                                                       Tensorial operations
# ==========================================================================================
# Tensorial products
dyad11 = lambda a1, b1 : np.einsum('i,j -> ij', a1, b1)
dyad22 = lambda a2, b2 : np.einsum('ij,kl -> ijkl', a2, b2)
dyad22_2 = lambda a2, b2 : np.einsum('ik,jl -> ijkl', a2, b2)
dyad22_3 = lambda a2, b2 : np.einsum('il,jk -> ijkl', a2, b2)
# Tensorial single contractions
dot21_1 = lambda a2, b1 : np.einsum('ij,j -> i', a2, b1)
dot12_1 = lambda a1, b2 : np.einsum('i,ij -> j', a1, b2)
dot42_1 = lambda a4, b2 : np.einsum('ijkm,lm -> ijkl', a4, b2)
dot42_2 = lambda a4, b2 : np.einsum('ipkl,jp -> ijkl', a4, b2)
dot42_3 = lambda a4, b2 : np.einsum('ijkm,ml -> ijkl', a4, b2)
dot24_1 = lambda a2, b4 : np.einsum('im,mjkl -> ijkl', a2, b4)
dot24_2 = lambda a2, b4 : np.einsum('jm,imkl -> ijkl', a2, b4)
dot24_3 = lambda a2, b4 : np.einsum('km,ijml -> ijkl', a2, b4)
dot24_4 = lambda a2, b4 : np.einsum('lm,ijkm -> ijkl', a2, b4)
# Tensorial double contractions
ddot22_1 = lambda a2, b2 : np.einsum('ij,ij', a2, b2)
ddot42_1 = lambda a4, b2 : np.einsum('ijkl,kl -> ij', a4, b2)
ddot44_1 = lambda a4, b4 : np.einsum('ijmn,mnkl -> ijkl', a4, b4)
#
#                                                                                  Operators
# ==========================================================================================
def dd(i, j):
    '''Discrete Dirac's delta function (d_ij = 1 if i == j, d_ij = 0 if i != j).

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    value : 0 or 1
        Discrete Dirac's delta.
    '''
    if (not isinstance(i, int) and not isinstance(i, np.integer)) or \
            (not isinstance(j, int) and not isinstance(j, np.integer)):
        raise RuntimeError('The discrete Dirac\'s delta function only accepts two ' +
                           'integer indexes as arguments.')
    value = 1 if i == j else 0
    return value
# ------------------------------------------------------------------------------------------
def get_id_operators(n_dim):
    '''Set common identity operators.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.

    Returns
    -------
    soid : 2darray
        Second-order identity tensor (I_ij = d_ii).
    foid : 4darray
        Fourth-order identity tensor (I_ijkl = d_ik*d_jl).
    fotransp : 4darray
        Fourth-order transposition tensor (I_ijkl = d_il*d_jk).
    fosym : 4darray
        Fourth-order symmetric projection tensor (I_ijkl = 0.5*(d_ik*d_jl+d_il*d_jk)).
    fodiagtrace : 4darray
        Fourth-order 'diagonal trace' tensor (I_ijkl = d_ij*d_kl).
    fodevproj : 4darray
        Fourth-order deviatoric projection tensor (I_ijkl = d_ik*d_jl - (1/3)*d_ij*d_kl).
    fodevprojsym : 4darray
        Fourth-order deviatoric projection tensor (second-order symmetric tensors,
        I_ijkl = 0.5*(d_ik*d_jl+d_il*d_jk) - (1/3)*d_ij*d_kl).
    '''
    # Set second-order identity tensor
    soid = np.eye(n_dim)
    # Set fourth-order identity tensor and fourth-order transposition tensor
    foid = np.zeros((n_dim, n_dim, n_dim, n_dim))
    fotransp = np.zeros((n_dim, n_dim, n_dim, n_dim))
    for i in range(n_dim):
        for j in range(n_dim):
            foid[i, j, i, j] = 1.0
            fotransp[i, j, j, i] = 1.0
    # Set fourth-order symmetric projection tensor
    fosym = 0.5*(foid + fotransp)
    # Set fourth-order 'diagonal trace' tensor
    fodiagtrace = dyad22(soid,soid)
    # Set fourth-order deviatoric projection tensor
    fodevproj = foid - (1.0/3.0)*fodiagtrace
    # Set fourth-order deviatoric projection tensor (second order symmetric tensors)
    fodevprojsym = fosym - (1.0/3.0)*fodiagtrace
    # Return
    return [soid, foid, fotransp, fosym, fodiagtrace, fodevproj, fodevprojsym]
#
#                                                                     Spectral decomposition
# ==========================================================================================
def spectral_decomposition(x):
    '''Perform spectral decomposition of symmetric second-order tensor (square array).

    Parameters
    ----------
    x : 2darray
        Second-order tensor (square array) whose eigenvalues and eigenvectors are computed.

    Returns
    -------
    eigenvals : 1darray
        Eigenvalues of second-order tensor sorted in descending order.
    eigenvectors : 2darray
        Eigenvectors of second-order tensor stored columnwise according with eigenvalues.
    eig_multiplicity : dict
        Multiplicity (item, int) of the eigenvalue stored at given index (key, str).
    eigenprojections : list of tuple
        Eigenprojections of second-order tensor stored as tuples (item) as
        (eigenvalue, eigenprojection) and sorted in descending order of eigenvalues. Only
        available for 2x2 and 3x3 second-order tensors, otherwise an empty list is returned.
    '''
    # Check if second-order tensor is symmetric
    if not np.allclose(x, np.transpose(x), rtol=1e-5, atol=1e-10):
        raise RuntimeError('Second-order tensor must be symmetric to perform spectral '
                           'decomposition.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eig(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get eigenvalues sorted in descending order
    sort_idxs = np.argsort(eigenvalues)[::-1]
    # Sort eigenvalues in descending order and eigenvectors accordingly
    eigenvalues = eigenvalues[sort_idxs]
    eigenvectors = eigenvectors[:,sort_idxs]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get square array dimensions
    n_dim = x.shape[0]
    # Set eigenvalue multiplicity tolerance
    eig_toler = 1e-10
    # Compute eigenprojections
    if n_dim == 2:
        # Set eigenvalue normalization factor
        eig_norm = np.max(np.abs(eigenvalues))
        # Check eigenvalues multiplicity
        if eig_norm < eig_toler:
            eig_mult = [(eigenvalues[0] - eigenvalues[1]) < eig_toler,]
        else:
            eig_mult = [(eigenvalues[0] - eigenvalues[1])/eig_norm < eig_toler,]
        # Get distinct eigenvalues
        if np.sum(eig_mult) == 0:
            n_eig_distinct = 2
            eig_multiplicity = {'0': 1, '1': 1}
        else:
            n_eig_distinct = 1
            eig_multiplicity = {'0': 2, '1': 0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize eigenprojections
        eigenprojections = []
        # Compute eigenprojections according with eigenvalues multiplicity
        if n_eig_distinct == 1:
            eig = eigenvalues[0]
            eigenprojections = [(eig, np.eye(n_dim)),]
        else:
            # Compute first principal invariant of second-order tensor
            pinv_1 = np.trace(x)
            # Compute eigenprojections
            for i in range(2):
                # Get eigenvalue
                eig = eigenvalues[i]
                # Compute eigenprojection
                eigenprojections.append((eig, (1.0/(2.0*eig - pinv_1))*(x +
                    (eig - pinv_1)*np.eye(n_dim))))
    elif n_dim == 3:
        # Set eigenvalue normalization factor
        eig_norm = np.max(np.abs(eigenvalues))
        # Check eigenvalues multiplicity
        if eig_norm < eig_toler:
            eig_mult = [(eigenvalues[0] - eigenvalues[1]) < eig_toler,
                        (eigenvalues[0] - eigenvalues[2]) < eig_toler,
                        (eigenvalues[1] - eigenvalues[2]) < eig_toler]
        else:
            eig_mult = [(eigenvalues[0] - eigenvalues[1])/eig_norm < eig_toler,
                        (eigenvalues[0] - eigenvalues[2])/eig_norm < eig_toler,
                        (eigenvalues[1] - eigenvalues[2])/eig_norm < eig_toler]
        # Get distinct eigenvalues
        if np.sum(eig_mult) == 0:
            n_eig_distinct = 3
            eig_multiplicity = {'0': 1, '1': 1, '2': 1}
        elif np.sum(eig_mult) == 1:
            n_eig_distinct = 2
            if eig_mult[0]:
                eig_multiplicity = {'0': 2, '1': 0, '2': 1}
            elif eig_mult[1]:
                eig_multiplicity = {'0': 2, '1': 1, '2': 0}
            else:
                eig_multiplicity = {'0': 1, '1': 2, '2': 0}
        else:
            n_eig_distinct = 1
            eig_multiplicity = {'0': 3, '1': 0, '2': 0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize eigenprojections
        eigenprojections = []
        # Compute eigenprojections according with eigenvalues multiplicity
        if n_eig_distinct == 1:
            eig = eigenvalues[0]
            eigenprojections = [(eig, np.eye(n_dim)),]
        else:
            # Compute principal invariants of second-order tensor
            pinv_1 = np.trace(x)
            pinv_3 = np.linalg.det(x)
            # Compute eigenprojections
            if n_eig_distinct == 2:
                # Compute first eigenprojection
                idxa = [int(key) for key, val in eig_multiplicity.items() if val == 1][0]
                eig = eigenvalues[idxa]
                eigenprojections.append((eig, (eig/(2*eig**3 - pinv_1*eig**2 + pinv_3))*(
                    np.linalg.matrix_power(x, 2) - (pinv_1 - eig)*x +
                        (pinv_3/eig)*np.eye(n_dim))))
                # Compute second eigenprojection
                idxc = [int(key) for key, val in eig_multiplicity.items() if val == 2][0]
                eig = eigenvalues[idxc]
                eigenprojections.append((eig, np.eye(n_dim) - eigenprojections[0][1]))
            else:
                # Compute eigenprojections
                for i in range(3):
                    # Get eigenvalue
                    eig = eigenvalues[i]
                    # Compute eigenprojection
                    eigenprojections.append((eig, (eig/(2*eig**3 - pinv_1*eig**2 + pinv_3))*(
                        np.linalg.matrix_power(x, 2) - (pinv_1 - eig)*x +
                            (pinv_3/eig)*np.eye(n_dim))))
    else:
        eigenprojections = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return eigenvalues, eigenvectors, eig_multiplicity, eigenprojections
#
#                                                          Isotropic tensor-valued functions
# ==========================================================================================
def isotropic_tensor(mode, x):
    '''Compute isotropic symmetric tensor-valued function of symmetric tensor.

    Parameters
    ----------
    mode : {'log', 'exp'}
        Scalar function with single argument associated to the symmetric tensor-valued
        function (particular classe of isotropic tensor-valued functions).
    x : 2darray
        Second-order tensor at which isotropic tensor-valued function is evaluated.

    Returns
    -------
    y : 2darray
        Isotropic symmetric tensor-valued function evaluated at x.
    '''
    # Set scalar function with single argument
    if mode == 'log':
        fun = lambda x : np.log(x)
    elif mode == 'exp':
        fun = lambda x : np.exp(x)
    else:
        raise RuntimeError('Unknown scalar function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors, eig_multiplicity, eigenprojections = \
        spectral_decomposition(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize isotropic symmetric tensor-valued function
    y = np.zeros(x.shape)
    # Compute isotropic symmetric tensor-valued function
    for i in range(len(eigenprojections)):
        y += fun(eigenprojections[i][0])*eigenprojections[i][1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return y
# ------------------------------------------------------------------------------------------
def derivative_isotropic_tensor(mode, x):
    '''Compute derivative of isotropic symmetric tensor-valued function of symmetric tensor.

    Parameters
    ----------
    mode : {'log', 'exp'}
        Scalar function with single argument associated to the symmetric tensor-valued
        function (particular classe of isotropic tensor-valued functions).
    x : 2darray
        Second-order tensor at which isotropic tensor-valued function is evaluated.

    Returns
    -------
    y : 4darray
        Derivative of isotropic symmetric tensor-valued function evaluated at x.
    '''
    # Get square array dimensions
    n_dim = x.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required second-order and fourth-order identity tensors
    soid, foid, _, fosym, _, _, _ = get_id_operators(n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set scalar function with single argument and associated derivative
    if mode == 'log':
        fun = lambda x : np.log(x)
        fund = lambda x : 1.0/x
    elif mode == 'exp':
        fun = lambda x : np.exp(x)
        fund = lambda x : np.exp(x)
    else:
        raise RuntimeError('Unknown scalar function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors, eig_multiplicity, eigenprojections = \
        spectral_decomposition(x)
    # Compute number of distinct eigenvalues
    n_eig_distinct = n_dim - \
        np.sum([1 for key, val in eig_multiplicity.items() if val == 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize derivative of isotropic symmetric tensor-valued function
    y = np.zeros(2*x.shape)
    # Compute derivative of isotropic symmetric tensor-valued function
    if n_eig_distinct == 1:
        y = fund(eigenvalues[0])*fosym
    else:
        if n_dim == 2:
            # Set eigenvalues and eigenprojections
            eig1 = eigenprojections[0][0]
            eigproj1 = eigenprojections[0][1]
            eig2 = eigenprojections[1][0]
            eigproj2 = eigenprojections[1][1]
            # Evaluate scalar function and derivative
            fun1 = fun(eig1)
            fund1 = fund(eig1)
            fun2 = fun(eig2)
            fund2 = fund(eig2)
            # Compute derivative of isotropic symmetric tensor-valued function
            y = ((fun1 - fun2)/(eig1 - eig2))*(fosym - dyad22(eigproj1, eigproj1) -
                dyad22(eigproj2, eigproj2)) + fund1*dyad22(eigproj1, eigproj1) + \
                    fund2*dyad22(eigproj2, eigproj2)
        elif n_dim == 3:
            # Compute derivative of square symmetric tensor
            dx2dx = np.zeros(4*(n_dim,))
            for i, j, k, l in it.product(range(n_dim), repeat=4):
                dx2dx[i, j, k, l] = 0.5*(dd(i, k)*x[l, j] + dd(i, l)*x[k, j] +
                                         dd(j, l)*x[i, k] + dd(k, j)*x[i, l])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if n_eig_distinct == 2:
                # Set eigenvalues evaluation triplet (a != b = c)
                idxa = [int(key) for key, val in eig_multiplicity.items() if val == 1][0]
                idxb = [int(key) for key, val in eig_multiplicity.items() if val == 0][0]
                idxc = [int(key) for key, val in eig_multiplicity.items() if val == 2][0]
                eig_abc = str(idxa) + str(idxb) + str(idxc)
                # Compute convenient scalars
                s1, s2, s3, s4, s5, s6 = diso_scalars(eig_abc, eigenvalues, fun, fund)
                # Compute derivative of isotropic symmetric tensor-valued function
                y = s1*dx2dx - s2*fosym - s3*dyad22(x, x) + s4*dyad22(x, soid) + \
                    s5*dyad22(soid, x) - s6*dyad22(soid, soid)
            else:
                # Initialize derivative of isotropic symmetric tensor-valued function
                y = np.zeros(4*(n_dim,))
                # Set eigenvalues cyclic permutations
                eig_cyclic = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
                # Compute derivative of isotropic symmetric tensor-valued function
                for a in range(n_dim):
                    # Set eigenvalues evaluation order
                    eiga = eigenprojections[eig_cyclic[a][0]][0]
                    eigproja = eigenprojections[eig_cyclic[a][0]][1]
                    eigb = eigenprojections[eig_cyclic[a][1]][0]
                    eigprojb = eigenprojections[eig_cyclic[a][1]][1]
                    eigc = eigenprojections[eig_cyclic[a][2]][0]
                    eigprojc = eigenprojections[eig_cyclic[a][2]][1]
                    # Evaluate scalar function and derivative
                    funa = fun(eiga)
                    funda = fund(eiga)
                    # Assemble derivative of isotropic symmetric tensor-valued function
                    y += (funa/((eiga - eigb)*(eiga - eigc)))*(dx2dx - (eigb + eigc)*fosym -
                        ((eiga - eigb) + (eiga - eigc))*dyad22(eigproja, eigproja) -
                            (eigb - eigc)*(dyad22(eigprojb, eigprojb) -
                                dyad22(eigprojc, eigprojc))) + \
                                    funda*dyad22(eigproja, eigproja)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return y
# ------------------------------------------------------------------------------------------
def diso_scalars(abc, eigenvalues, fun, fund):
    '''Compute scalars associated to the derivative of isotropic tensor-valued function.

    Parameters
    ----------
    abc : str, {'012', '210', '102', '021', '102', '120', '201'}
        Triplet containing the eigenvalues evaluation order. The first and last are
        assumed to be the distinct eigenvalues.
    eigenvals : 1darray
        Eigenvalues of second-order tensor sorted in descending order.
    fun : function
        Scalar function with single argument associated to the symmetric tensor-valued
        function.
    fund: function
        Derivative of scalar function with single argument associated to the symmetric
        tensor-valued function.

    Returns
    -------
    s : list
        List of convenient scalars (float).
    '''
    # Check eigenvalues order triplet validity
    if not re.match('^[0-2]{3}$', str(abc)):
        raise RuntimeError('Invalid triplet.')
    elif set([int(x) for x in abc]) != {0,1,2}:
        raise RuntimeError('Invalid triplet.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set eigenvalues according to triplet order
    eiga = eigenvalues[int(abc[0])]
    eigc = eigenvalues[int(abc[2])]
    # Evaluate scalar function and derivative
    funa = fun(eiga)
    funda = fund(eiga)
    func = fun(eigc)
    fundc = fund(eigc)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute convenient scalars
    s = []
    s.append((funa - func)/(eiga - eigc)**2 - fundc/(eiga - eigc))
    s.append(2*eigc*((funa - func)/(eiga - eigc)**2) - ((eiga + eigc)/(eiga - eigc))*fundc)
    s.append(2*((funa - func)/(eiga - eigc)**3) - ((funda + fundc)/(eiga - eigc)**2))
    s.append(s[2]*eigc)
    s.append(s[2]*eigc)
    s.append(s[2]*eigc**2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return s
# ------------------------------------------------------------------------------------------
def rotate_tensor(tensor, r):
    '''Rotation of n-dimensional tensor.

    Parameters
    ----------
    tensor : ndarray
        Tensor.
    r : 2darray
        Rotation tensor (for given rotation angle theta, active transformation (+ theta) and
        passive transformation (- theta)).

    Returns
    -------
    rtensor : ndarray
        Rotated tensor.
    '''
    # Get number of spatial dimensions
    n_dim = tensor.shape[0]
    # Get number of tensor dimensions
    tensor_dim = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize rotated tensor
    rtensor = np.zeros(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if tensor_dim == 1:
        # Compute rotation of first-order tensor
        for i, p in it.product(range(n_dim), repeat=2):
            rtensor[i] = rtensor[i] + r[i, p]*tensor[p]
    elif tensor_dim == 2:
        # Compute rotation of second-order tensor
        for i, j, p, q in it.product(range(n_dim), repeat=4):
            rtensor[i, j] = rtensor[i, j] + r[i, p]*r[j, q]*tensor[p, q]
    elif tensor_dim == 3:
        # Compute rotation of third-order tensor
        for i, j, k, p, q, r in it.product(range(n_dim), repeat=6):
            rtensor[i, j, k] = rtensor[i, j, k] + r[i, p]*r[j, q]*r[k, r]*tensor[p, q, r]
    elif tensor_dim == 4:
        # Compute rotation of fourth-order tensor
        for i, j, k, l, p, q, r, s in it.product(range(n_dim), repeat=8):
            rtensor[i, j, k, l] = \
                rtensor[i, j, k, l] + r[i, p]*r[j, q]*r[k, r]*r[l, s]*tensor[p, q, r, s]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return rtensor
# ------------------------------------------------------------------------------------------
def rotation_tensor_from_euler_angles(n_dim, euler_deg):
    '''Set rotation tensor from euler angles (Bunge convention).

    Parameters
    ----------
    n_dim : int
        Number of spatial dimensions.
    euler_deg : tuple
        Euler angles (degrees) sorted according to Bunge convention (Z1-X2-Z3).

    Returns
    -------
    r : 2darray
        Rotation tensor (for given rotation angle theta, active transformation (+ theta) and
        passive transformation (- theta)).
    '''
    # Convert euler angles to radians
    euler_rad = tuple(np.radians(x) for x in euler_deg)
    # Compute convenient sins and cosines
    s1 = np.sin(euler_rad[0])
    s2 = np.sin(euler_rad[1])
    s3 = np.sin(euler_rad[2])
    c1 = np.cos(euler_rad[0])
    c2 = np.cos(euler_rad[1])
    c3 = np.cos(euler_rad[2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize rotation tensor
    r = np.zeros((n_dim, n_dim))
    # Build rotation tensor
    r[0, 0] = c1*c3 - c2*s1*s3
    r[1, 0] = c3*s1 + c1*c2*s3
    r[2, 0] = s2*s3
    r[0, 1] = -c1*s3 - c2*c3*s1
    r[1, 1] = c1*c2*c3 - s1*s3
    r[2, 1] = c3*s2
    r[0, 2] = s1*s2
    r[1, 2] = -c1*s2
    r[2, 2] = c2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return r
