#
# Material-related Quantities Computation Module (CRATE Program)
# ==========================================================================================
# Summary:
# Computation of material-related quantities based on material state variables.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | May 2021 | Initial coding.
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
            Stress tensor (matricial form).
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
            Strain tensor (matricial form).
        '''
        # Compute deviatoric strain tensor (matricial form)
        dev_strain_mf = np.matmul(self._fodevprojsym_mf, strain_mf)
        # Compute von Mises equivalent strain
        vm_strain = np.sqrt(2.0/3.0)*np.linalg.norm(dev_strain_mf)
        # Return
        return vm_strain
