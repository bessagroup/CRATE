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
    _fodevprojsym_mf : ndarray
        Fourth-order deviatoric projection tensor (second order symmetric tensors)
        (matricial form).
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, n_dim, comp_order):
        '''Material-related quantities computer constructor.

        Parameters
        ----------
        n_dim : int
            Problem dimension.
        comp_order : list
            Strain/Stress components (str) order.
        '''
        self._n_dim = n_dim
        self._comp_order = comp_order
        # Get fourth-order tensors
        _, _, _, _, _, _, fodevprojsym = top.getidoperators(n_dim)
        # Get fourth-order tensors matricial form
        fodevprojsym_mf = mop.gettensormf(fodevprojsym, n_dim, comp_order)
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
