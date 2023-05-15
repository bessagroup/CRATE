"""Output file: Homogenized strain/stress results.

This module includes the class associated with the output file where the
homogenized strain/stress results are stored.

Classes
-------
HomResOutput
    Output file: Homogenized strain/stress results.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
import tensor.tensoroperations as top
from ioput.incoutputfiles.interface import IncrementalOutputFile
from material.materialoperations import compute_spatial_log_strain, \
                                        cauchy_from_first_piola, \
                                        MaterialQuantitiesComputer
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class HomResOutput(IncrementalOutputFile):
    """Output file: Homogenized strain/stress results.

    Attributes
    ----------
    _file_path : str
        Output file path.
    _header : list[str]
        List containing the header of each column (str).
    _col_width : int
        Output file column width.

    Methods
    -------
    init_file(self, strain_formulation)
        Open output file and write file header.
    write_file(self, strain_formulation, problem_type, mac_load_path, \
               hom_results, effective_time)
        Write output file.
    """
    def __init__(self, file_path):
        """Constructor.

        Parameters
        ----------
        file_path : str
            Output file path.
        """
        self._file_path = file_path
        # Set output file header
        self._header = ['Increment', 'RunEffectTime', 'LoadSubpath',
                        'LoadFactor', 'Time', 'SubincLevel',
                        'strain_11', 'strain_21', 'strain_31',
                        'strain_12', 'strain_22', 'strain_32',
                        'strain_13', 'strain_23', 'strain_33',
                        'stress_11', 'stress_21', 'stress_31',
                        'stress_12', 'stress_22', 'stress_32',
                        'stress_13', 'stress_23', 'stress_33',
                        'vm_strain', 'vm_stress']
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # -------------------------------------------------------------------------
    def init_file(self, strain_formulation):
        """Open output file and write file header.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        """
        # Set loading initial output
        load_init = (0, 0.0, 0, 0.0, 0.0, 0)
        # Set strain and stress initial output
        strain_init = 9*[0.0, ]
        stress_init = 9*[0.0, ]
        if strain_formulation == 'finite':
            strain_init[0] = 1.0
            strain_init[4] = 1.0
            strain_init[8] = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open output file (write mode)
        output_file = open(self._file_path, 'w')
        # Set output file header format structure
        write_list = [
            '{:>9s}'.format(self._header[0])
            + ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                       for x in self._header[1:]]),
            '\n' + '{:>9d}'.format(load_init[0])
            + ('{:>' + str(self._col_width) + '.8e}').format(load_init[1])
            + ('{:>' + str(self._col_width) + 'd}').format(load_init[2])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in load_init[3:5]])
            + ('{:>' + str(self._col_width) + 'd}').format(load_init[5])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in strain_init])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in stress_init])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(0.0)
                       for x in range(2)])]
        # Write output file header
        output_file.writelines(write_list)
        # Close output file
        output_file.close()
    # -------------------------------------------------------------------------
    def write_file(self, strain_formulation, problem_type, mac_load_path,
                   hom_results, effective_time):
        """Write output file.

        Parameters
        ----------
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        mac_load_path : LoadingPath
            Macroscale loading path.
        hom_results : dict
            Homogenized strain/stress results stored as:

            * 'strain' : homogenized strain tensor (numpy.ndarray (2d))
            * 'stress' : homogenized stress tensor (numpy.ndarray (2d))
            * 'hom_stress_33' : homogenized out-of-plain stress (float)

            Infinitesimal strain tensor and Cauchy stress tensor (infinitesimal
            strains) or Deformation gradient and first Piola-Kirchhoff stress
            tensor (finite strains).
        effective_time : float
            Current time (s) associated with the solution of the equilibrium
            problem.
        """
        # Get loading path data
        sp_id, sp_inc, sp_total_lfact, _, sp_total_time, _, subinc_level = \
            mac_load_path.get_subpath_state()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get homogenized strain/stress data
        hom_strain = hom_results['hom_strain']
        hom_stress = hom_results['hom_stress']
        if problem_type == 1:
            hom_stress_33 = hom_results['hom_stress_33']
        # When the problem type corresponds to a 2D analysis, build the 3D
        # homogenized strain and stress tensors by considering the appropriate
        # out-of-plane strain and stress components
        out_hom_strain = np.zeros((3, 3))
        out_hom_stress = np.zeros((3, 3))
        if problem_type == 1:
            out_hom_strain[0:2, 0:2] = hom_strain
            if strain_formulation == 'finite':
                out_hom_strain[2, 2] = 1.0
            out_hom_stress[0:2, 0:2] = hom_stress
            out_hom_stress[2, 2] = hom_stress_33
        else:
            out_hom_strain[:, :] = hom_strain
            out_hom_stress[:, :] = hom_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem parameters
        n_dim, comp_order_sym, _ = \
            mop.get_problem_type_parameters(problem_type=4)
        # Get fourth-order tensors
        _, _, _, _, _, _, fodevprojsym = top.get_id_operators(n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate material state computations
        csbvar_computer = MaterialQuantitiesComputer()
        # Compute spatial logarithmic strain tensor and Cauchy stress tensor
        if strain_formulation == 'infinitesimal':
            strain = copy.deepcopy(out_hom_strain)
            cauchy_stress = copy.deepcopy(out_hom_stress)
        else:
            # Compute spatial logarithmic strain tensor
            strain = compute_spatial_log_strain(out_hom_strain)
            # Get Cauchy stress tensor from first Piola-Kirchhoff stress tensor
            cauchy_stress = cauchy_from_first_piola(out_hom_strain,
                                                    out_hom_stress)
        # Get spatial logarithmic strain tensor (matricial form)
        strain_mf = mop.get_tensor_mf(strain, n_dim, comp_order_sym)
        # Get Cauchy stress tensor (matricial form)
        cauchy_stress_mf = mop.get_tensor_mf(cauchy_stress, n_dim,
                                             comp_order_sym)
        # Compute von Mises equivalent strain
        vm_strain = csbvar_computer.get_vm_strain(strain_mf)
        # Compute von Mises equivalent stress
        vm_stress = csbvar_computer.get_vm_stress(cauchy_stress_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open homogenized results output file (append mode)
        output_file = open(self._file_path, 'a')
        # Set increment homogenized results format structure
        inc_data = [sp_inc, effective_time, sp_id, sp_total_lfact,
                    sp_total_time, subinc_level,
                    out_hom_strain[0, 0], out_hom_strain[1, 0],
                    out_hom_strain[2, 0], out_hom_strain[0, 1],
                    out_hom_strain[1, 1], out_hom_strain[2, 1],
                    out_hom_strain[0, 2], out_hom_strain[1, 2],
                    out_hom_strain[2, 2], out_hom_stress[0, 0],
                    out_hom_stress[1, 0], out_hom_stress[2, 0],
                    out_hom_stress[0, 1], out_hom_stress[1, 1],
                    out_hom_stress[2, 1], out_hom_stress[0, 2],
                    out_hom_stress[1, 2], out_hom_stress[2, 2],
                    vm_strain, vm_stress]
        write_list = \
            ['\n' + ('{:>9d}').format(inc_data[0])
             + ('{:>' + str(self._col_width) + '.8e}').format(inc_data[1])
             + ('{:>' + str(self._col_width) + 'd}').format(inc_data[2])
             + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                        for x in inc_data[3:5]])
             + ('{:>' + str(self._col_width) + 'd}').format(inc_data[5])
             + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                        for x in inc_data[6:]])]
        # Write increment homogenized results
        output_file.writelines(write_list)
        # Close output file
        output_file.close()
