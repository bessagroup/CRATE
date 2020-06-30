#
# Tensor Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Algebraic tensorial operations and standard tensorial operators.
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
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
#
#                                                                       Tensorial operations
# ==========================================================================================
# Tensorial products
dyad22 = lambda a2, b2 : np.einsum('ij,kl -> ijkl', a2, b2)
# Tensorial single contractions
dot21_1 = lambda a2, b1 : np.einsum('ij,j -> i', a2, b1)
dot12_1 = lambda a1, b2 : np.einsum('i,ij -> j', a1, b2)
# Tensorial double contractions
ddot22_1 = lambda a2, b2 : np.einsum('ij,ij', a2, b2)
ddot42_1 = lambda a4, b2 : np.einsum('ijkl,kl -> ij', a4, b2)
ddot44_1 = lambda a4, b4 : np.einsum('ijmn,mnkl -> ijkl', a4, b4)
#
#                                                                                  Operators
# ==========================================================================================
# Discrete Dirac's delta function (dij = 1 if i=j, dij = 0 if i!=j).
def dd(i, j):
    if (not isinstance(i, int) and not isinstance(i, np.integer)) or \
            (not isinstance(j, int) and not isinstance(j, np.integer)):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00086', location.filename, location.lineno + 1)
    value = 1 if i == j else 0
    return value
# ------------------------------------------------------------------------------------------
# Set the following common identity operators:
#
# Second-order identity tensor              > Tij = dii
# Fourth-order identity tensor              > Tijkl = dik*djl
# Fourth-order symmetric projection tensor  > Tijkl = 0.5*(dik*djl+dil*djk)
# Fourth-order 'diagonal trace' tensor      > Tijkl = dij*dkl
# Fourth-order deviatoric projection tensor > Tijkl = dik*djl-(1/3)*dij*dkl
# Fourth-order deviatoric projection tensor > Tijkl = 0.5*(dik*djl+dil*djk)-(1/3)*dij*dkl
# (second order symmetric tensors)
#
# where 'd' represents the discrete Dirac delta.
#
def getidoperators(n_dim):
    # Set second-order identity tensor
    soid = np.eye(n_dim)
    # Set fourth-order identity tensor and fourth-order transpose tensor
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
