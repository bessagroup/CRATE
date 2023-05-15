"""Computational solid mechanics standard computations.

This module includes several functions to compute common tensorial quantities
arising in computational solid mechanics.

Classes
-------
MaterialQuantitiesComputer:
    Computation of quantities based on material state variables.

Functions
---------
compute_rotation_tensor
    Rotation tensor from polar decomposition of deformation gradient.
compute_material_log_strain
    Material logarithmic strain from deformation gradient.
compute_spatial_log_strain
    Spatial logarithmic strain from deformation gradient.
cauchy_from_kirchhoff
    Cauchy stress tensor from Kirchhoff stress tensor.
first_piola_from_kirchhoff
    First Piola-Kirchhoff stress tensor from Kirchhoff stress tensor.
cauchy_from_first_piola
    Cauchy stress tensor from first Piola-Kirchhoff stress tensor.
cauchy_from_second_piola
    Cauchy stress tensor from second Piola-Kirchhoff stress tensor.
first_piola_from_cauchy
    First Piola-Kirchhoff stress tensor from Cauchy stress tensor.
first_piola_from_second_piola
    First Piola-Kirchhoff from second Piola-Kirchhoff stress tensor.
kirchhoff_from_first_piola
    Kirchhoff stress tensor from first Piola-Kirchhoff stress tensor.
material_from_spatial_tangent_modulus
    Material consistent tangent modulus from spatial counterpart.
conjugate_material_log_strain
    Stress conjugate of material logarithmic strain tensor.
conjugate_spatial_log_strain
    Stress conjugate of spatial logarithmic strain tensor.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
import tensor.tensoroperations as top
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
#                                                               Rotation tensor
# =============================================================================
def compute_rotation_tensor(def_gradient):
    """Rotation tensor from polar decomposition of deformation gradient.

    .. math::

       \\boldsymbol{R} = \\boldsymbol{F}(\\boldsymbol{F}^{T}
                         \\boldsymbol{F})^{\\frac{1}{2}}

    where :math:`\\boldsymbol{R}` is the rotation tensor and
    :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.

    Returns
    -------
    r : numpy.ndarray (2d)
        Rotation tensor.
    """
    # Compute right stretch tensor
    right_stretch = mop.matrix_root(np.matmul(
        np.transpose(def_gradient), def_gradient), p=0.5)
    # Compute rotation tensor
    r = np.matmul(def_gradient, np.linalg.inv(right_stretch))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return r
#
#                                                    Strain tensors conversions
# =============================================================================
def compute_material_log_strain(def_gradient):
    """Material logarithmic strain from deformation gradient.

    .. math::

       \\boldsymbol{E} = \\frac{1}{2} \\ln (\\boldsymbol{F}^{T}
                         \\boldsymbol{F})

    where :math:`\\boldsymbol{E}` is the material logarithmic strain tensor and
    :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.

    Returns
    -------
    material_log_strain : numpy.ndarray (2d)
        Material logarithmic strain.
    """
    # Compute material logarithmic strain tensor
    material_log_strain = 0.5*top.isotropic_tensor(
        'log', np.matmul(np.transpose(def_gradient), def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_log_strain
# =============================================================================
def compute_spatial_log_strain(def_gradient):
    """Spatial logarithmic strain from deformation gradient.

    .. math::

       \\boldsymbol{\\varepsilon} = \\frac{1}{2} \\ln (\\boldsymbol{F}
                                    \\boldsymbol{F}^{T})

    where :math:`\\boldsymbol{\\varepsilon}` is the spatial logarithmic strain
    tensor and :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.

    Returns
    -------
    spatial_log_strain : numpy.ndarray (2d)
        Spatial logarithmic strain.
    """
    # Compute spatial logarithmic strain tensor
    spatial_log_strain = 0.5*top.isotropic_tensor(
        'log', np.matmul(def_gradient, np.transpose(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return spatial_log_strain
#
#                                                    Stress tensors conversions
# =============================================================================
def cauchy_from_kirchhoff(def_gradient, kirchhoff_stress):
    """Cauchy stress tensor from Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{\\sigma} = \\frac{1}{\\det (\\boldsymbol{F})} \\,
                               \\boldsymbol{\\tau}

    where :math:`\\boldsymbol{\\sigma}` is the Cauchy stress tensor,
    :math:`\\boldsymbol{F}` is the deformation gradient, and
    :math:`\\boldsymbol{\\tau}` is the Kirchhoff stress tensor.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    kirchhoff_stress : numpy.ndarray (2d)
        Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : numpy.ndarray (2d)
        Cauchy stress tensor.
    """
    # Compute Cauchy stress tensor
    cauchy_stress = (1.0/np.linalg.det(def_gradient))*kirchhoff_stress
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# =============================================================================
def first_piola_from_kirchhoff(def_gradient, kirchhoff_stress):
    """First Piola-Kirchhoff stress tensor from Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{P} = \\boldsymbol{\\tau} \\boldsymbol{F}^{-T}

    where :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor,
    :math:`\\boldsymbol{\\tau}` is the Kirchhoff stress tensor, and
    :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    kirchhoff_stress : numpy.ndarray (2d)
        Kirchhoff stress tensor.

    Returns
    -------
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.
    """
    # Compute First Piola-Kirchhoff stress tensor
    first_piola_stress = np.matmul(kirchhoff_stress,
                                   np.transpose(np.linalg.inv(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
# =============================================================================
def cauchy_from_first_piola(def_gradient, first_piola_stress):
    """Cauchy stress tensor from first Piola-Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{\\sigma} = \\frac{1}{\\det (\\boldsymbol{F})} \\,
                               \\boldsymbol{P} \\boldsymbol{F}^{T}

    where :math:`\\boldsymbol{\\sigma}` is the Cauchy stress tensor,
    :math:`\\boldsymbol{F}` is the deformation gradient, and
    :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : numpy.ndarray (2d)
        Cauchy stress tensor.
    """
    # Compute Cauchy stress tensor
    cauchy_stress = (1.0/np.linalg.det(def_gradient))*np.matmul(
        first_piola_stress, np.transpose(def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# =============================================================================
def cauchy_from_second_piola(def_gradient, second_piola_stress):
    """Cauchy stress tensor from second Piola-Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{\\sigma} = \\frac{1}{\\det (\\boldsymbol{F})} \\,
                               \\boldsymbol{F}  \\boldsymbol{S}
                               \\boldsymbol{F}^{T}

    where :math:`\\boldsymbol{\\sigma}` is the Cauchy stress tensor,
    :math:`\\boldsymbol{F}` is the deformation gradient, and
    :math:`\\boldsymbol{S}` is the second Piola-Kirchhoff stress tensor.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    second_piola_stress : numpy.ndarray (2d)
        Second Piola-Kirchhoff stress tensor.

    Returns
    -------
    cauchy_stress : numpy.ndarray (2d)
        Cauchy stress tensor.
    """
    # Compute Cauchy stress tensor
    cauchy_stress = (1.0/np.linalg.det(def_gradient))*np.matmul(
        def_gradient, np.matmul(second_piola_stress,
                                np.transpose(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return cauchy_stress
# =============================================================================
def first_piola_from_cauchy(def_gradient, cauchy_stress):
    """First Piola-Kirchhoff stress tensor from Cauchy stress tensor.

    .. math::

       \\boldsymbol{P} = \\det (\\boldsymbol{F}) \\,
                         \\boldsymbol{\\sigma} \\boldsymbol{F}^{-T}

    where :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor,
    :math:`\\boldsymbol{F}` is the deformation gradient, and
    :math:`\\boldsymbol{\\sigma}` is the Cauchy stress tensor.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    cauchy_stress : numpy.ndarray (2d)
        Cauchy stress tensor.

    Returns
    -------
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.
    """
    # Compute first Piola_Kirchhoff stress tensor
    first_piola_stress = np.linalg.det(def_gradient)*np.matmul(
        cauchy_stress, np.transpose(np.linalg.inv(def_gradient)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
# =============================================================================
def first_piola_from_second_piola(def_gradient, second_piola_stress):
    """First Piola-Kirchhoff from second Piola-Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{P} = \\boldsymbol{F} \\boldsymbol{S}

    where :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor
    and :math:`\\boldsymbol{S}` is the second Piola-Kirchhoff stress tensor.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    second_piola_stress : numpy.ndarray (2d)
        Second Piola-Kirchhoff stress tensor.

    Returns
    -------
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.
    """
    # Compute first Piola_Kirchhoff stress tensor
    first_piola_stress = np.matmul(def_gradient, second_piola_stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return first_piola_stress
# =============================================================================
def kirchhoff_from_first_piola(def_gradient, first_piola_stress):
    """Kirchhoff stress tensor from first Piola-Kirchhoff stress tensor.

    .. math::

       \\boldsymbol{\\tau} = \\boldsymbol{P} \\boldsymbol{F}^{T}

    where :math:`\\boldsymbol{\\tau}` is the Kirchhoff stress tensor,
    :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor,
    and :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    kirchhoff_stress : numpy.ndarray (2d)
        Kirchhoff stress tensor.
    """
    # Compute Kirchhoff stress tensor
    kirchhoff_stress = np.matmul(first_piola_stress,
                                 np.transpose(def_gradient))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return kirchhoff_stress
# =============================================================================
def material_from_spatial_tangent_modulus(spatial_consistent_tangent,
                                          def_gradient):
    """Material consistent tangent modulus from spatial counterpart.

    .. math::

       \\mathsf{A}_{ijkl} = \\det (\\boldsymbol{F}) \\, \\mathsf{a}_{ipkq} \\,
                            F_{lq}^{-1} F_{jp}^{-1}

    where :math:`\\mathbf{\\mathsf{A}}` is the material consistent tangent
    modulus, :math:`\\boldsymbol{F}` is the deformation gradient, and
    :math:`\\mathbf{\\mathsf{a}}` is the spatial consistent tangent modulus.

    ----

    Parameters
    ----------
    spatial_consistent_tangent : numpy.ndarray (4d)
        Spatial consistent tangent modulus.
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.

    Returns
    -------
    material_consistent_tangent : numpy.ndarray (4d)
        Material consistent tangent modulus.
    """
    # Compute inverse of deformation gradient
    def_gradient_inv = np.linalg.inv(def_gradient)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute material consistent tangent modulus
    material_consistent_tangent = np.linalg.det(def_gradient)*top.dot42_2(
        top.dot42_1(spatial_consistent_tangent, def_gradient_inv),
        def_gradient_inv)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_consistent_tangent
#
#                                                 Stress-strain conjugate pairs
# =============================================================================
def conjugate_material_log_strain(def_gradient, first_piola_stress):
    """Stress conjugate of material logarithmic strain tensor.

    *Material logarithmic strain tensor*:

    .. math::

       \\boldsymbol{E} = \\frac{1}{2} \\ln (\\boldsymbol{F}^{T}
                         \\boldsymbol{F})

    where :math:`\\boldsymbol{E}` is the material logarithmic strain tensor and
    :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    *Stress conjugate of material logarithmic strain tensor*:

    .. math::

       \\boldsymbol{T} = \\boldsymbol{R}^{T} \\boldsymbol{P}
                         \\boldsymbol{F}^{T} \\boldsymbol{R}

    where :math:`\\boldsymbol{T}` is the stress conjugate of the material
    logarithmic strain tensor, :math:`\\boldsymbol{R}` is the rotation tensor,
    :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor,
    and :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    material_log_strain : numpy.ndarray (2d)
        Material logarithmic strain.
    stress_conjugate : numpy.ndarray (2d)
        Stress conjugate to material logarithmic strain.
    """
    # Compute material logarithmic strain tensor
    material_log_strain = compute_material_log_strain(def_gradient)
    # Compute rotation tensor (polar decomposition of deformation gradient)
    rotation = compute_rotation_tensor(def_gradient)
    # Compute stress conjugate to material logarithmic strain tensor
    stress_conjugate = np.matmul(
        np.transpose(rotation), np.matmul(first_piola_stress,
                                          np.matmul(np.transpose(def_gradient),
                                                    rotation)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return material_log_strain, stress_conjugate
# =============================================================================
def conjugate_spatial_log_strain(def_gradient, first_piola_stress):
    """Stress conjugate of spatial logarithmic strain tensor.

    *Spatial logarithmic strain tensor*:

    .. math::

       \\boldsymbol{\\varepsilon} = \\frac{1}{2} \\ln (\\boldsymbol{F}
                                    \\boldsymbol{F}^{T})

    where :math:`\\boldsymbol{\\varepsilon}` is the spatial logarithmic strain
    tensor and :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    *Stress conjugate of spatial logarithmic strain tensor*:

    .. math::

       \\boldsymbol{\\tau} = \\boldsymbol{P} \\boldsymbol{F}^{T}

    where :math:`\\boldsymbol{\\tau}` is the Kirchhoff stress tensor,
    :math:`\\boldsymbol{P}` is the first Piola-Kirchhoff stress tensor,
    and :math:`\\boldsymbol{F}` is the deformation gradient.

    ----

    Parameters
    ----------
    def_gradient : numpy.ndarray (2d)
        Deformation gradient.
    first_piola_stress : numpy.ndarray (2d)
        First Piola-Kirchhoff stress tensor.

    Returns
    -------
    spatial_log_strain : numpy.ndarray (2d)
        Spatial logarithmic strain.
    stress_conjugate : numpy.ndarray (2d)
        Stress conjugate to spatial logarithmic strain.
    """
    # Compute spatial logarithmic strain tensor
    spatial_log_strain = compute_spatial_log_strain(def_gradient)
    # Compute stress conjugate to spatial logarithmic strain tensor
    stress_conjugate = kirchhoff_from_first_piola(def_gradient,
                                                  first_piola_stress)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return spatial_log_strain, stress_conjugate
#
#                                    Computation of material-related quantities
# =============================================================================
class MaterialQuantitiesComputer:
    """Computation of quantities based on material state variables.

    Material-related quantities computations are always performed assuming the
    three-dimensional strain and/or stress state.

    Attributes
    ----------
    _n_dim : int
        Problem dimension.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.
    _fodevprojsym_mf : numpy.ndarray (2d)
        Fourth-order deviatoric projection tensor (second order symmetric
        tensors) (matricial form).

    Methods
    -------
    get_vm_stress(self, stress_mf)
        Compute von Mises equivalent stress.
    get_vm_strain(self, strain_mf)
        Compute von Mises equivalent strain.
    """
    # -------------------------------------------------------------------------
    def __init__(self):
        """Constructor."""
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
        fodevprojsym_mf = mop.get_tensor_mf(fodevprojsym, self._n_dim,
                                            self._comp_order_sym)
        self._fodevprojsym_mf = fodevprojsym_mf
    # -------------------------------------------------------------------------
    def get_vm_stress(self, stress_mf):
        """Compute von Mises equivalent stress.

        .. math::

           \\sigma_{\\text{VM}} = \\frac{3}{2}
                                  || \\boldsymbol{\\sigma_{d}} ||

        where :math:`\\sigma_{\\text{VM}}` is the von Mises equivalent stress
        and :math:`\\boldsymbol{\\sigma_{d}}` is the deviatoric Cauchy stress
        tensor.

        ----

        Parameters
        ----------
        stress_mf : numpy.ndarray (1d)
            Cauchy stress tensor (matricial form).
        """
        # Compute deviatoric stress tensor (matricial form)
        dev_stress_mf = np.matmul(self._fodevprojsym_mf, stress_mf)
        # Compute von Mises equivalent stress
        vm_stress = np.sqrt(3.0/2.0)*np.linalg.norm(dev_stress_mf)
        # Return
        return vm_stress
    # -------------------------------------------------------------------------
    def get_vm_strain(self, strain_mf):
        """Compute von Mises equivalent strain.

        .. math::

           \\varepsilon_{\\text{VM}} = \\frac{2}{3}
                                  || \\boldsymbol{\\varepsilon_{d}} ||

        where :math:`\\varepsilon_{\\text{VM}}` is the von Mises equivalent
        strain and :math:`\\boldsymbol{\\varepsilon_{d}}` is either the
        deviatoric infinitesimal strain tensor (infinitesimal strains) or the
        spatial logarithmic strain tensor (finite strains).

        ----

        Parameters
        ----------
        strain_mf : numpy.ndarray (1d)
            Strain tensor (matricial form): infinitesimal strain tensor
            (infinitesimal strains) or spatial logarithmic strain tensor
            (finite strains).
        """
        # Compute deviatoric strain tensor (matricial form)
        dev_strain_mf = np.matmul(self._fodevprojsym_mf, strain_mf)
        # Compute von Mises equivalent strain
        vm_strain = np.sqrt(2.0/3.0)*np.linalg.norm(dev_strain_mf)
        # Return
        return vm_strain
