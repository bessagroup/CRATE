"""Algebraic tensorial operations and standard tensorial operators.

This module is essentially a toolkit containing the definition of several
standard tensorial operators (e.g., Kronecker delta, second- and fourth-order
identity tensors, rotation tensor) and tensorial operations (e.g., tensorial
product, tensorial contraction, spectral decomposition) arising in
computational mechanics.

Functions
---------
dyad11
    Dyadic product: :math:`i \\otimes j \\rightarrow ij`.
dyad22_1
    Dyadic product: :math:`ij \\otimes kl \\rightarrow ijkl`.
dyad22_2
    Dyadic product: :math:`ik \\otimes jl \\rightarrow ijkl`.
dyad22_3
    Dyadic product: :math:`il \\otimes jk \\rightarrow ijkl`.
dot21_1
    Single contraction: :math:`ij \\cdot j \\rightarrow i`.
dot12_1
    Single contraction: :math:`i \\cdot ij \\rightarrow j`.
dot42_1
    Single contraction: :math:`ijkm \\cdot lm \\rightarrow ijkl`.
dot42_2
    Single contraction: :math:`ipkl \\cdot jp \\rightarrow ijkl`.
dot42_3
    Single contraction: :math:`ijkm \\cdot ml \\rightarrow ijkl`.
dot24_1
    Single contraction: :math:`im \\cdot mjkl \\rightarrow ijkl`.
dot24_2
    Single contraction: :math:`jm \\cdot imkl \\rightarrow ijkl`.
dot24_3
    Single contraction: :math:`km \\cdot ijml \\rightarrow ijkl`.
dot24_4
    Single contraction: :math:`lm \\cdot ijkm \\rightarrow ijkl`.
ddot22_1
    Double contraction: :math:`ij : ij \\rightarrow \\text{scalar}`.
ddot42_1
    Double contraction: :math:`ijkl : kl \\rightarrow ij`.
ddot44_1
    Double contraction: :math:`ijmn : mnkl \\rightarrow ijkl`.
dd
    Kronecker delta function.
get_id_operators
    Set common second- and fourth-order identity operators.
spectral_decomposition
    Perform spectral decomposition of symmetric second-order tensor.
isotropic_tensor
    Isotropic symmetric tensor-valued function of symmetric tensor.
derivative_isotropic_tensor
    Derivative of isotropic tensor-valued function of symmetric tensor.
diso_scalars
    Auxiliar scalars of derivative of isotropic tensor-valued function.
rotate_tensor
    Rotation of :math:`n`-dimensional tensor.
rotation_tensor_from_euler_angles
    Set rotation tensor from Euler angles (Bunge convention).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import re
import itertools as it
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                          Tensorial operations
# =============================================================================
# Tensorial products
dyad11 = lambda a1, b1: np.einsum('i,j -> ij', a1, b1)
dyad22_1 = lambda a2, b2: np.einsum('ij,kl -> ijkl', a2, b2)
dyad22_2 = lambda a2, b2: np.einsum('ik,jl -> ijkl', a2, b2)
dyad22_3 = lambda a2, b2: np.einsum('il,jk -> ijkl', a2, b2)
# Tensorial single contractions
dot21_1 = lambda a2, b1: np.einsum('ij,j -> i', a2, b1)
dot12_1 = lambda a1, b2: np.einsum('i,ij -> j', a1, b2)
dot42_1 = lambda a4, b2: np.einsum('ijkm,lm -> ijkl', a4, b2)
dot42_2 = lambda a4, b2: np.einsum('ipkl,jp -> ijkl', a4, b2)
dot42_3 = lambda a4, b2: np.einsum('ijkm,ml -> ijkl', a4, b2)
dot24_1 = lambda a2, b4: np.einsum('im,mjkl -> ijkl', a2, b4)
dot24_2 = lambda a2, b4: np.einsum('jm,imkl -> ijkl', a2, b4)
dot24_3 = lambda a2, b4: np.einsum('km,ijml -> ijkl', a2, b4)
dot24_4 = lambda a2, b4: np.einsum('lm,ijkm -> ijkl', a2, b4)
# Tensorial double contractions
ddot22_1 = lambda a2, b2: np.einsum('ij,ij', a2, b2)
ddot42_1 = lambda a4, b2: np.einsum('ijkl,kl -> ij', a4, b2)
ddot44_1 = lambda a4, b4: np.einsum('ijmn,mnkl -> ijkl', a4, b4)
#
#                                                                     Operators
# =============================================================================
def dd(i, j):
    """Kronecker delta function.

    .. math::

       \\delta_{ij} =
           \\begin{cases}
                   1, &         \\text{if } i=j, \\\\
                   0, &         \\text{if } i\\neq j.
           \\end{cases}

    ----

    Parameters
    ----------
    i : int
        First index.
    j : int
        Second index.

    Returns
    -------
    value : int (0 or 1)
        Kronecker delta.
    """
    if (not isinstance(i, int) and not isinstance(i, np.integer)) or \
            (not isinstance(j, int) and not isinstance(j, np.integer)):
        raise RuntimeError('The Kronecker delta function only accepts two '
                           + 'integer indexes as arguments.')
    value = 1 if i == j else 0
    return value
# =============================================================================
def get_id_operators(n_dim):
    """Set common second- and fourth-order identity operators.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.

    Returns
    -------
    soid : numpy.ndarray (2d)
        Second-order identity tensor:

        .. math::

           I_{ij} = \\delta_{ij}
    foid : numpy.ndarray (4d)
        Fourth-order identity tensor:

        .. math::
           I_{ijkl} = \\delta_{ik}\\delta_{jl}
    fotransp : numpy.ndarray (4d)
        Fourth-order transposition tensor:

        .. math::

           I_{ijkl} = \\delta_{il}\\delta_{jk}
    fosym : numpy.ndarray (4d)
        Fourth-order symmetric projection tensor:

        .. math::

           I_{ij} = 0.5(\\delta_{ik}\\delta_{jl} +
                    \\delta_{il}\\delta_{jk})
    fodiagtrace : numpy.ndarray (4d)
        Fourth-order 'diagonal trace' tensor:

        .. math::

           I_{ijkl} = \\delta_{ij}\\delta_{kl}
    fodevproj : numpy.ndarray (4d)
        Fourth-order deviatoric projection tensor:

        .. math::

           I_{ijkl} = \\delta_{ik}\\delta_{jl}
                      - \\dfrac{1}{3} \\delta_{ij}\\delta_{kl}
    fodevprojsym : numpy.ndarray (4d)
        Fourth-order deviatoric projection tensor (second-order symmetric
        tensors):

        .. math::

           I_{ijkl} = 0.5(\\delta_{ik}\\delta_{jl}
                      + \\delta_{il}\\delta_{jk})
                      - \\dfrac{1}{3} \\delta_{ij}\\delta_{kl}
    """
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
    fodiagtrace = dyad22_1(soid, soid)
    # Set fourth-order deviatoric projection tensor
    fodevproj = foid - (1.0/3.0)*fodiagtrace
    # Set fourth-order deviatoric projection tensor (second order symmetric
    # tensors)
    fodevprojsym = fosym - (1.0/3.0)*fodiagtrace
    # Return
    return soid, foid, fotransp, fosym, fodiagtrace, fodevproj, fodevprojsym
#
#                                                        Spectral decomposition
# =============================================================================
def spectral_decomposition(x):
    """Perform spectral decomposition of symmetric second-order tensor.

    The computational implementation of the spectral decomposition follows the
    Appendix A of Computational Methods for Plasticity [#]_.

    .. [#] de Souza Neto, E. A., Peri, D., and Owen, D. R. J. (2008).
           Computational Methods for Plasticity. John Wiley & Sons, Ltd,
           Chichester, UK (see `here <https://onlinelibrary.wiley.com/doi/
           book/10.1002/9780470694626>`_)

    ----

    Parameters
    ----------
    x : numpy.ndarray (2d)
        Second-order tensor (square array) whose eigenvalues and eigenvectors
        are computed.

    Returns
    -------
    eigenvals : numpy.ndarray (1d)
        Eigenvalues of second-order tensor sorted in descending order.
    eigenvectors : numpy.ndarray (2d)
        Eigenvectors of second-order tensor stored columnwise according with
        eigenvalues.
    eig_multiplicity : dict
        Multiplicity (item, int) of the eigenvalue stored at given index
        (key, str).
    eigenprojections : list[tuple]
        Eigenprojections of second-order tensor stored as tuples (item) as
        (eigenvalue, eigenprojection) and sorted in descending order of
        eigenvalues. Only available for 2x2 and 3x3 second-order tensors,
        otherwise an empty list is returned.
    """
    # Check if second-order tensor is symmetric
    if not np.allclose(x, np.transpose(x), rtol=1e-5, atol=1e-10):
        raise RuntimeError('Second-order tensor must be symmetric to perform '
                           'spectral decomposition.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors = np.linalg.eig(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get eigenvalues sorted in descending order
    sort_idxs = np.argsort(eigenvalues)[::-1]
    # Sort eigenvalues in descending order and eigenvectors accordingly
    eigenvalues = eigenvalues[sort_idxs]
    eigenvectors = eigenvectors[:, sort_idxs]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            eig_mult = [(eigenvalues[0] - eigenvalues[1]) < eig_toler, ]
        else:
            eig_mult = \
                [(eigenvalues[0] - eigenvalues[1])/eig_norm < eig_toler, ]
        # Get distinct eigenvalues
        if np.sum(eig_mult) == 0:
            n_eig_distinct = 2
            eig_multiplicity = {'0': 1, '1': 1}
        else:
            n_eig_distinct = 1
            eig_multiplicity = {'0': 2, '1': 0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize eigenprojections
        eigenprojections = []
        # Compute eigenprojections according with eigenvalues multiplicity
        if n_eig_distinct == 1:
            eig = eigenvalues[0]
            eigenprojections = [(eig, np.eye(n_dim)), ]
        else:
            # Compute first principal invariant of second-order tensor
            pinv_1 = np.trace(x)
            # Compute eigenprojections
            for i in range(2):
                # Get eigenvalue
                eig = eigenvalues[i]
                # Compute eigenprojection
                eigenprojections.append((eig, (1.0/(2.0*eig - pinv_1))*(x
                                         + (eig - pinv_1)*np.eye(n_dim))))
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
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize eigenprojections
        eigenprojections = []
        # Compute eigenprojections according with eigenvalues multiplicity
        if n_eig_distinct == 1:
            eig = eigenvalues[0]
            eigenprojections = [(eig, np.eye(n_dim)), ]
        else:
            # Compute principal invariants of second-order tensor
            pinv_1 = np.trace(x)
            pinv_3 = np.linalg.det(x)
            # Compute eigenprojections
            if n_eig_distinct == 2:
                # Compute first eigenprojection
                idxa = [int(key) for key, val in
                        eig_multiplicity.items() if val == 1][0]
                eig = eigenvalues[idxa]
                eigenprojections.append(
                    (eig, (eig/(2*eig**3 - pinv_1*eig**2 + pinv_3))*(
                        np.linalg.matrix_power(x, 2) - (pinv_1 - eig)*x
                        + (pinv_3/eig)*np.eye(n_dim))))
                # Compute second eigenprojection
                idxc = [int(key) for key, val in
                        eig_multiplicity.items() if val == 2][0]
                eig = eigenvalues[idxc]
                eigenprojections.append(
                    (eig, np.eye(n_dim) - eigenprojections[0][1]))
            else:
                # Compute eigenprojections
                for i in range(3):
                    # Get eigenvalue
                    eig = eigenvalues[i]
                    # Compute eigenprojection
                    eigenprojections.append(
                        (eig, (eig/(2*eig**3 - pinv_1*eig**2 + pinv_3))*(
                            np.linalg.matrix_power(x, 2) - (pinv_1 - eig)*x
                            + (pinv_3/eig)*np.eye(n_dim))))
    else:
        eigenprojections = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return eigenvalues, eigenvectors, eig_multiplicity, eigenprojections
#
#                                             Isotropic tensor-valued functions
# =============================================================================
def isotropic_tensor(mode, x):
    """Isotropic symmetric tensor-valued function of symmetric tensor.

    The computational implementation to compute the isotropic symmetric
    tensor-valued function of a symmetric tensor follows the Appendix A of
    Computational Methods for Plasticity [#]_.

    .. [#] de Souza Neto, E. A., Peri, D., and Owen, D. R. J. (2008).
           Computational Methods for Plasticity. John Wiley & Sons, Ltd,
           Chichester, UK (see `here <https://onlinelibrary.wiley.com/doi/
           book/10.1002/9780470694626>`_)

    ----

    Parameters
    ----------
    mode : {'log', 'exp'}
        Scalar function with single argument associated to the symmetric
        tensor-valued function (particular classe of isotropic tensor-valued
        functions).
    x : numpy.ndarray (2d)
        Second-order tensor at which isotropic tensor-valued function is
        evaluated.

    Returns
    -------
    y : numpy.ndarray (2d)
        Isotropic symmetric tensor-valued function evaluated at x.
    """
    # Set scalar function with single argument
    if mode == 'log':
        fun = lambda x: np.log(x)
    elif mode == 'exp':
        fun = lambda x: np.exp(x)
    else:
        raise RuntimeError('Unknown scalar function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors, eig_multiplicity, eigenprojections = \
        spectral_decomposition(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize isotropic symmetric tensor-valued function
    y = np.zeros(x.shape)
    # Compute isotropic symmetric tensor-valued function
    for i in range(len(eigenprojections)):
        y += fun(eigenprojections[i][0])*eigenprojections[i][1]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return y
# =============================================================================
def derivative_isotropic_tensor(mode, x):
    """Derivative of isotropic tensor-valued function of symmetric tensor.

    The computational implementation to compute the derivative of an isotropic
    tensor-valued function of a symmetric tensor follows the Appendix A of
    Computational Methods for Plasticity [#]_.

    .. [#] de Souza Neto, E. A., Peri, D., and Owen, D. R. J. (2008).
           Computational Methods for Plasticity. John Wiley & Sons, Ltd,
           Chichester, UK (see `here <https://onlinelibrary.wiley.com/doi/
           book/10.1002/9780470694626>`_)

    ----

    Parameters
    ----------
    mode : {'log', 'exp'}
        Scalar function with single argument associated to the symmetric
        tensor-valued function (particular classe of isotropic tensor-valued
        functions).
    x : numpy.ndarray (2d)
        Second-order tensor at which isotropic tensor-valued function is
        evaluated.

    Returns
    -------
    y : numpy.ndarray (4d)
        Derivative of isotropic symmetric tensor-valued function evaluated
        at x.
    """
    # Get square array dimensions
    n_dim = x.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set required second-order and fourth-order identity tensors
    soid, foid, _, fosym, _, _, _ = get_id_operators(n_dim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set scalar function with single argument and associated derivative
    if mode == 'log':
        fun = lambda x: np.log(x)
        fund = lambda x: 1.0/x
    elif mode == 'exp':
        fun = lambda x: np.exp(x)
        fund = lambda x: np.exp(x)
    else:
        raise RuntimeError('Unknown scalar function.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform spectral decomposition
    eigenvalues, eigenvectors, eig_multiplicity, eigenprojections = \
        spectral_decomposition(x)
    # Compute number of distinct eigenvalues
    n_eig_distinct = n_dim - \
        np.sum([1 for key, val in eig_multiplicity.items() if val == 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            y = ((fun1 - fun2)/(eig1 - eig2))*(
                fosym - dyad22_1(eigproj1, eigproj1)
                - dyad22_1(eigproj2, eigproj2)) \
                + fund1*dyad22_1(eigproj1, eigproj1) \
                + fund2*dyad22_1(eigproj2, eigproj2)
        elif n_dim == 3:
            # Compute derivative of square symmetric tensor
            dx2dx = np.zeros(4*(n_dim,))
            for i, j, k, l in it.product(range(n_dim), repeat=4):
                dx2dx[i, j, k, l] = 0.5*(dd(i, k)*x[l, j] + dd(i, l)*x[k, j]
                                         + dd(j, l)*x[i, k] + dd(k, j)*x[i, l])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if n_eig_distinct == 2:
                # Set eigenvalues evaluation triplet (a != b = c)
                idxa = [int(key) for key, val in
                        eig_multiplicity.items() if val == 1][0]
                idxb = [int(key) for key, val in
                        eig_multiplicity.items() if val == 0][0]
                idxc = [int(key) for key, val in
                        eig_multiplicity.items() if val == 2][0]
                eig_abc = str(idxa) + str(idxb) + str(idxc)
                # Compute convenient scalars
                s1, s2, s3, s4, s5, s6 = diso_scalars(eig_abc, eigenvalues,
                                                      fun, fund)
                # Compute derivative of isotropic symmetric tensor-valued
                # function
                y = s1*dx2dx - s2*fosym - s3*dyad22_1(x, x) \
                    + s4*dyad22_1(x, soid) + s5*dyad22_1(soid, x) \
                    - s6*dyad22_1(soid, soid)
            else:
                # Initialize derivative of isotropic symmetric tensor-valued
                # function
                y = np.zeros(4*(n_dim,))
                # Set eigenvalues cyclic permutations
                eig_cyclic = [(0, 1, 2), (2, 0, 1), (1, 2, 0)]
                # Compute derivative of isotropic symmetric tensor-valued
                # function
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
                    # Assemble derivative of isotropic symmetric tensor-valued
                    # function
                    y += (funa/((eiga - eigb)*(eiga - eigc)))*(
                        dx2dx
                        - (eigb + eigc)*fosym
                        - ((eiga - eigb) + (eiga - eigc))
                        * dyad22_1(eigproja, eigproja)
                        - (eigb - eigc)*(dyad22_1(eigprojb, eigprojb)
                                         - dyad22_1(eigprojc, eigprojc))) \
                        + funda*dyad22_1(eigproja, eigproja)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return y
# =============================================================================
def diso_scalars(abc, eigenvalues, fun, fund):
    """Auxiliar scalars of derivative of isotropic tensor-valued function.

    The auxilar scalars to compute the derivative of an isotropic
    tensor-valued function of a symmetric tensor are defined in Equation (A.47)
    of Computational Methods for Plasticity [#]_.

    .. [#] de Souza Neto, E. A., Peri, D., and Owen, D. R. J. (2008).
           Computational Methods for Plasticity. John Wiley & Sons, Ltd,
           Chichester, UK (see `here <https://onlinelibrary.wiley.com/doi/
           book/10.1002/9780470694626>`_)

    ----

    Parameters
    ----------
    abc : {'012', '210', '102', '021', '102', '120', '201'}
        Triplet containing the eigenvalues evaluation order. The first and last
        are assumed to be the distinct eigenvalues.
    eigenvals : numpy.ndarray (1d)
        Eigenvalues of second-order tensor sorted in descending order.
    fun : function
        Scalar function with single argument associated to the symmetric
        tensor-valued function.
    fund: function
        Derivative of scalar function with single argument associated to the
        symmetric tensor-valued function.

    Returns
    -------
    s : list
        List of convenient scalars (float).
    """
    # Check eigenvalues order triplet validity
    if not re.match('^[0-2]{3}$', str(abc)):
        raise RuntimeError('Invalid triplet.')
    elif set([int(x) for x in abc]) != {0, 1, 2}:
        raise RuntimeError('Invalid triplet.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set eigenvalues according to triplet order
    eiga = eigenvalues[int(abc[0])]
    eigc = eigenvalues[int(abc[2])]
    # Evaluate scalar function and derivative
    funa = fun(eiga)
    funda = fund(eiga)
    func = fun(eigc)
    fundc = fund(eigc)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute convenient scalars
    s = []
    s.append((funa - func)/(eiga - eigc)**2 - fundc/(eiga - eigc))
    s.append(2*eigc*((funa - func)/(eiga - eigc)**2)
             - ((eiga + eigc)/(eiga - eigc))*fundc)
    s.append(2*((funa - func)/(eiga - eigc)**3)
             - ((funda + fundc)/(eiga - eigc)**2))
    s.append(s[2]*eigc)
    s.append(s[2]*eigc)
    s.append(s[2]*eigc**2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return s
# =============================================================================
def rotate_tensor(tensor, r):
    """Rotation of :math:`n`-dimensional tensor.

    The rotation of the :math:`n`-dimensional tensor is here defined as

    .. math::

       A^{r}_{i_{1}i_{2}\\dots i_{n}} = R_{i_{1}j_{1}} R_{i_{2}j_{2}} \\dots
       R_{i_{n}j_{n}} A_{j_{1}j_{2} \\dots j_{n}}

    where :math:`\\mathbf{R}` denotes the rotation tensor, :math:`\\mathbf{A}`
    denotes the original tensor, and :math:`\\mathbf{A}^{r}` denotes the
    rotated tensor.

    ----

    Parameters
    ----------
    tensor : numpy.ndarray
        Tensor.
    r : numpy.ndarray (2d)
        Rotation tensor (for given rotation angle theta, active transformation
        (+ theta) and passive transformation (- theta)).

    Returns
    -------
    rtensor : numpy.ndarray
        Rotated tensor.
    """
    # Get number of spatial dimensions
    n_dim = tensor.shape[0]
    # Get number of tensor dimensions
    tensor_dim = len(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize rotated tensor
    rtensor = np.zeros(tensor.shape)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            rtensor[i, j, k] = rtensor[i, j, k] \
                               + r[i, p]*r[j, q]*r[k, r]*tensor[p, q, r]
    elif tensor_dim == 4:
        # Compute rotation of fourth-order tensor
        for i, j, k, l, p, q, r, s in it.product(range(n_dim), repeat=8):
            rtensor[i, j, k, l] = rtensor[i, j, k, l] \
                + r[i, p]*r[j, q]*r[k, r]*r[l, s]*tensor[p, q, r, s]
    else:
        raise RuntimeError('The rotation tensor is not available for '
                           + 'tensor order greater than 4.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return rtensor
# =============================================================================
def rotation_tensor_from_euler_angles(euler_deg):
    """Set rotation tensor from Euler angles (Bunge convention).

    The rotation tensor is defined as

    .. math::

       \\mathbf{R} =
           \\begin{bmatrix}
               c_1 c_3 - c_2 s_1 s_3 & -c_1 s_3 - c_2 c_3 s_1 & s_1 s_2 \\\\
               c_3 s_1 + c_1 c_2 s_3 & c_1 c_2 c_3 - s_1 s_3 & - c_1 s_2 \\\\
               s_2 s_3 & c_3 s_2 & c_2
           \\end{bmatrix}

    where

    .. math::

       \\begin{align}
           c_1 = \\cos(\\alpha) \\qquad s_1 = \\sin(\\alpha) \\\\
           c_2 = \\cos(\\beta) \\qquad s_2 = \\sin(\\beta) \\\\
           c_3 = \\cos(\\gamma) \\qquad s_3 = \\sin(\\gamma)
        \\end{align}

    and :math:`(\\alpha, \\beta, \\gamma)` are the Euler angles corresponding
    to the Bunge convention (Z1-X2-Z3).

    ----

    Parameters
    ----------
    euler_deg : tuple
        Euler angles (degrees) sorted according to Bunge convention (Z1-X2-Z3).

    Returns
    -------
    r : numpy.ndarray (2d)
        Rotation tensor (for given rotation angle theta, active transformation
        (+ theta) and passive transformation (- theta)).
    """
    # Convert euler angles to radians
    euler_rad = tuple(np.radians(x) for x in euler_deg)
    # Compute convenient sins and cosines
    s1 = np.sin(euler_rad[0])
    s2 = np.sin(euler_rad[1])
    s3 = np.sin(euler_rad[2])
    c1 = np.cos(euler_rad[0])
    c2 = np.cos(euler_rad[1])
    c3 = np.cos(euler_rad[2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize rotation tensor
    r = np.zeros((3, 3))
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return r
