#
# Material-related Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Material-related operations such as strain/stress tensors conversions and computation of
# general material-related metrics based on state variables.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | May 2021 | Initial coding.
# Bernardo P. Ferreira | Jan 2022 | Strain/stress tensors conversions.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Matricial operations
import tensor.matrixoperations as mop
# Tensorial operations
import tensor.tensoroperations as top
#
#                                                                 Strain tensors conversions
# ==========================================================================================
def compute_spatial_log_strain(def_gradient):
    '''Compute spatial logarithmic strain.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.

    Returns
    -------
    log_strain : 2darray
        Spatial logarithmic strain.
    '''
    # Compute spatial logarithmic strain tensor
    log_strain = 0.5*top.isotropic_tensor('log', np.matmul(def_gradient,
                                                           np.transpose(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return log_strain
#
#                                                                 Stress tensors conversions
# ==========================================================================================
def cauchy_from_kirchhoff(def_gradient, kirchhoff_stress):
    '''Compute Cauchy stress tensor from Kirchhoff stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    kirchhoff_stress : 2darray
        Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : 2darray
        Cauchy stress tensor.
    '''
    # Compute Cauchy stress tensor
    cauchy_stress = (1.0/np.linalg.det(def_gradient))*kirchhoff_stress
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# ------------------------------------------------------------------------------------------
def first_piola_from_kirchhoff(def_gradient, kirchhoff_stress):
    '''Compute First Piola-Kirchhoff stress tensor from Kirchhoff stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    kirchhoff_stress : 2darray
        Kirchhoff stress tensor.

    Returns
    -------
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.
    '''
    # Compute First Piola-Kirchhoff stress tensor
    first_piola_stress = np.matmul(kirchhoff_stress,
                                   np.transpose(np.linalg.inv(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
# ------------------------------------------------------------------------------------------
def cauchy_from_first_piola(def_gradient, first_piola_stress):
    '''Compute Cauchy stress tensor from first Piola_Kirchhoff stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : 2darray
        Cauchy stress tensor.
    '''
    # Compute Cauchy stress tensor
    cauchy_stress = \
        (1.0/np.linalg.det(def_gradient))*np.matmul(first_piola_stress,
                                                    np.transpose(def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# ------------------------------------------------------------------------------------------
def first_piola_from_cauchy(def_gradient, cauchy_stress):
    '''Compute first Piola_Kirchhoff stress tensor from Cauchy stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    cauchy_stress : 2darray
        Cauchy stress tensor.

    Returns
    -------
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.
    '''
    # Compute first Piola_Kirchhoff stress tensor
    first_piola_stress = np.linalg.det(def_gradient)*np.matmul(cauchy_stress,
        np.transpose(np.linalg.inv(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
#
#                                                 Computation of material-related quantities
# ==========================================================================================
class MaterialQuantitiesComputer:
    '''Computation of material-related quantities based on material state variables.

    Attributes
    ----------
    _n_dim : int
        Problem dimension.
    _comp_order : list
        Strain/Stress components (str) order.
    _fodevprojsym_mf : ndarray
        Fourth-order deviatoric projection tensor (second order symmetric tensors)
        (matricial form).

    Notes
    -----
    Material-related quantities computations are always performed considering the 3D
    strain and/or stress state.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self):
        '''Material-related quantities computer constructor.'''
        # Get 3D problem parameters
        n_dim, comp_order_sym, _ = mop.get_problem_type_parameters(problem_type=4)
        # Set material-related quantities computer parameters
        self._n_dim = n_dim
        self._comp_order = comp_order_sym
        # Get fourth-order tensors
        _, _, _, _, _, _, fodevprojsym = top.get_id_operators(self._n_dim)
        # Get fourth-order tensors matricial form
        fodevprojsym_mf = mop.get_tensor_mf(fodevprojsym, self._n_dim, self._comp_order)
        self._fodevprojsym_mf = fodevprojsym_mf
    # --------------------------------------------------------------------------------------
    def get_vm_stress(self, stress_mf):
        '''Compute von Mises equivalent stress.

        Parameters
        ----------
        stress_mf : ndarray
            Cauchy stress tensor (matricial form).
        '''
        # Compute deviatoric stress tensor (matricial form)
        dev_stress_mf = np.matmul(self._fodevprojsym_mf, stress_mf)
        # Compute von Mises equivalent stress
        vm_stress = np.sqrt(3.0/2.0)*np.linalg.norm(dev_stress_mf)
        # Return
        return vm_stress
    # --------------------------------------------------------------------------------------
    def get_vm_strain(self, strain_mf):
        '''Compute von Mises equivalent strain.

        Parameters
        ----------
        strain_mf : ndarray
            Strain tensor (matricial form): Infinitesimal strain tensor (infinitesimal
            strains) or Spatial logarithmic strain tensor (finite strains).
        '''
        # Compute deviatoric strain tensor (matricial form)
        dev_strain_mf = np.matmul(self._fodevprojsym_mf, strain_mf)
        # Compute von Mises equivalent strain
        vm_strain = np.sqrt(2.0/3.0)*np.linalg.norm(dev_strain_mf)
        # Return
        return vm_strain
