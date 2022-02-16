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
#                                 | Spatial to material consistent tangent modulus.
# Bernardo P. Ferreira | Feb 2022 | Stress-strain conjugate pairs.
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
#                                                                            Rotation tensor
# ==========================================================================================
def compute_rotation_tensor(def_gradient):
    '''Compute rotation tensor from the polar decomposition of the deformation gradient.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.

    Returns
    -------
    r : 2darray
        Rotation tensor.
    '''
    # Compute right stretch tensor
    right_stretch = \
        mop.matrix_root(np.matmul(np.transpose(def_gradient), def_gradient), p=0.5)
    # Compute rotation tensor
    r = np.matmul(def_gradient, np.linalg.inv(right_stretch))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return r
#
#                                                                 Strain tensors conversions
# ==========================================================================================
def compute_material_log_strain(def_gradient):
    '''Compute material logarithmic strain.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.

    Returns
    -------
    material_log_strain : 2darray
        Material logarithmic strain.
    '''
    # Compute material logarithmic strain tensor
    material_log_strain = 0.5*top.isotropic_tensor('log',
        np.matmul(np.transpose(def_gradient), def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_log_strain
# ------------------------------------------------------------------------------------------
def compute_spatial_log_strain(def_gradient):
    '''Compute spatial logarithmic strain.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.

    Returns
    -------
    spatial_log_strain : 2darray
        Spatial logarithmic strain.
    '''
    # Compute spatial logarithmic strain tensor
    spatial_log_strain = 0.5*top.isotropic_tensor('log', np.matmul(def_gradient,
                                                           np.transpose(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return spatial_log_strain
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
    '''Compute first Piola-Kirchhoff stress tensor from Kirchhoff stress tensor.

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
    '''Compute Cauchy stress tensor from first Piola-Kirchhoff stress tensor.

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
def cauchy_from_second_piola(def_gradient, second_piola_stress):
    '''Compute Cauchy stress tensor from second Piola-Kirchhoff stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    second_piola_stress : 2darray
        Second Piola-Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : 2darray
        Cauchy stress tensor.
    '''
    # Compute Cauchy stress tensor
    cauchy_stress = \
        (1.0/np.linalg.det(def_gradient))*np.matmul(def_gradient,
            np.matmul(second_piola_stress, np.transpose(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# ------------------------------------------------------------------------------------------
def first_piola_from_cauchy(def_gradient, cauchy_stress):
    '''Compute first Piola-Kirchhoff stress tensor from Cauchy stress tensor.

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
# ------------------------------------------------------------------------------------------
def first_piola_from_second_piola(def_gradient, second_piola_stress):
    '''Compute first Piola-Kirchhoff stress tensor from second Piola-Kirchhoff stress
    tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    second_piola_stress : 2darray
        Second Piola-Kirchhoff stress tensor.

    Returns
    -------
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.
    '''
    # Compute first Piola_Kirchhoff stress tensor
    first_piola_stress = np.matmul(def_gradient, second_piola_stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
# ------------------------------------------------------------------------------------------
def kirchhoff_from_first_piola(def_gradient, first_piola_stress):
    '''Compute Kirchhoff stress tensor from first Piola-Kirchhoff stress tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    kirchhoff_stress : 2darray
        Kirchhoff stress tensor.
    '''
    # Compute Kirchhoff stress tensor
    kirchhoff_stress = np.matmul(first_piola_stress, np.transpose(def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return kirchhoff_stress
# ------------------------------------------------------------------------------------------
def material_from_spatial_tangent_modulus(spatial_consistent_tangent, def_gradient):
    '''Compute material consistent tangent modulus from spatial counterpart.

    Parameters
    ----------
    spatial_consistent_tangent : 4darray
        Spatial consistent tangent modulus.
    def_gradient : 2darray
        Deformation gradient.

    Returns
    -------
    material_consistent_tangent : 4darray
        Material consistent tangent modulus.
    '''
    # Compute inverse of deformation gradient
    def_gradient_inv = np.linalg.inv(def_gradient)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute material consistent tangent modulus
    material_consistent_tangent = np.linalg.det(def_gradient)*top.dot42_2(
        top.dot42_1(spatial_consistent_tangent, def_gradient_inv), def_gradient_inv)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_consistent_tangent
#
#                                                              Stress-strain conjugate pairs
# ==========================================================================================
def conjugate_material_log_strain(def_gradient, first_piola_stress):
    '''Compute stress conjugate of material logarithmic strain tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    material_log_strain : 2darray
        Material logarithmic strain.
    stress_conjugate : 2darray
        Stress conjugate to material logarithmic strain.
    '''
    # Compute material logarithmic strain tensor
    material_log_strain = compute_material_log_strain(def_gradient)
    # Compute rotation tensor (polar decomposition of deformation gradient)
    rotation = compute_rotation_tensor(def_gradient)
    # Compute stress conjugate to material logarithmic strain tensor
    stress_conjugate = np.matmul(np.transpose(rotation),
                                 np.matmul(first_piola_stress,
                                           np.matmul(np.transpose(def_gradient), rotation)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_log_strain, stress_conjugate
# ------------------------------------------------------------------------------------------
def conjugate_spatial_log_strain(def_gradient, first_piola_stress):
    '''Compute stress conjugate of spatial logarithmic strain tensor.

    Parameters
    ----------
    def_gradient : 2darray
        Deformation gradient.
    first_piola_stress : 2darray
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    spatial_log_strain : 2darray
        Spatial logarithmic strain.
    stress_conjugate : 2darray
        Stress conjugate to spatial logarithmic strain.
    '''
    # Compute spatial logarithmic strain tensor
    spatial_log_strain = compute_spatial_log_strain(def_gradient)
    # Compute stress conjugate to spatial logarithmic strain tensor
    stress_conjugate = kirchhoff_from_first_piola(def_gradient, first_piola_stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return spatial_log_strain, stress_conjugate
#
#                                                 Computation of material-related quantities
# ==========================================================================================
class MaterialQuantitiesComputer:
    '''Computation of material-related quantities based on material state variables.

    Attributes
    ----------
    _n_dim : int
        Problem dimension.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
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
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type=4)
        # Set material-related quantities computer parameters
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Get fourth-order tensors
        _, _, _, _, _, _, fodevprojsym = top.get_id_operators(self._n_dim)
        # Get fourth-order tensors matricial form
        fodevprojsym_mf = mop.get_tensor_mf(fodevprojsym, self._n_dim, self._comp_order_sym)
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
