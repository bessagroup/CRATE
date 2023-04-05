"""Output file: Homogeneous (fictitious) reference material.

This module includes the class associated with the output file where data
associated with the homogeneous (fictitious) reference material is stored.

Classes
-------
RefMatOutput
    Output file: Homogeneous (fictitious) reference material data
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from ioput.incoutputfiles.interface import IncrementalOutputFile
import tensor.matrixoperations as mop
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class RefMatOutput(IncrementalOutputFile):
    """Output file: Homogeneous (fictitious) reference material data.

    Attributes
    ----------
    _file_path : str
        Output file path.
    _header : list[str]
        List containing the header of each column (str).
    _col_width : int
        Output file column width.
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list[str]
        Strain/Stress components symmetric order.
    _comp_order_nsym : list[str]
        Strain/Stress components nonsymmetric order.

    Methods
    -------
    init_file(self)
        Open reference material output file and write file header.
    write_mat(self, inc, ref_material, hom_strain_mf, hom_stress_mf, \
              eff_tangent_mf=None, farfield_strain_mf=None, \
              applied_mac_load_strain_mf=None)
        Write reference material output file.
    rewind_file(self, rewind_inc)
        Rewind reference material output file.
    """
    def __init__(self, file_path, strain_formulation, problem_type,
                 self_consistent_scheme='regression',
                 ref_output_mode='converged'):
        """Constructor.

        Parameters
        ----------
        file_path : str
            Output file path.
        strain_formulation: {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2),
            2D axisymmetric (3) and 3D (4).
        self_consistent_scheme : {'regression',}, default='regression'
            Self-consistent scheme to update the elastic reference material
            properties.
        ref_output_mode : {'iterative', 'converged'}, default='converged'
            Output mode:

            * 'iterative' : outputs the reference material quantities at \
                            every self-consistent scheme iteration
            * 'converged' : outputs the reference material quantities that \
                            converged at each loading increment
        """
        self._file_path = file_path
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._self_consistent_scheme = self_consistent_scheme
        self._ref_output_mode = ref_output_mode
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        self._n_dim = n_dim
        self._comp_order_sym = comp_order_sym
        self._comp_order_nsym = comp_order_nsym
        # Set reference material output file header
        self._header = ['Increment', 'SCS Iteration',
                        'E_ref', 'v_ref',
                        'strain0_11', 'strain0_21', 'strain0_31',
                        'strain0_12', 'strain0_22', 'strain0_32',
                        'strain0_13', 'strain0_23', 'strain0_33',
                        'strain_0_rdiff', 'rel_scs_cost', 'tangent_rdiff']
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # -------------------------------------------------------------------------
    def init_file(self):
        """Open output file and write file header."""
        # Set reference material elastic properties initial output
        properties_init = (0.0, 0.0)
        # Set far-field strain initial output
        strain_init = 9*[0.0, ]
        if self._strain_formulation == 'finite':
            strain_init[0] = 1.0
            strain_init[4] = 1.0
            strain_init[8] = 1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open output file (write mode)
        refm_file = open(self._file_path, 'w')
        # Set output file header format structure
        write_list = [
            '{:>9s}'.format(self._header[0]) + '  '
            + '{:>13s}'.format(self._header[1])
            + ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                       for x in self._header[2:]]),
            '\n' + '{:>9d}'.format(0) + '  ' + '{:>13d}'.format(0)
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in properties_init])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in strain_init])
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(0)
                       for x in self._header[13:]]), ]
        # Write output file header
        refm_file.writelines(write_list)
        # Close output file
        refm_file.close()
    # -------------------------------------------------------------------------
    def write_file(self, inc, ref_material, hom_strain_mf, hom_stress_mf,
                   farfield_strain_mf, applied_mac_load_strain_mf,
                   eff_tangent_mf=None):
        """Write output file.

        Parameters
        ----------
        inc : int
            Loading increment.
        ref_material : ElasticReferenceMaterial
            Elastic reference material.
        hom_strain_mf : numpy.ndarray (1d)
            Homogenized strain tensor (matricial form): infinitesimal strain
            tensor (infinitesimal strains) or deformation gradient (finite
            strains).
        hom_stress_mf : numpy.ndarray (1d)
            Homogenized stress tensor (matricial form): Cauchy stress tensor
            (infinitesimal strains) or first Piola-Kirchhoff stress tensor
            (finite strains).
        farfield_strain_mf : numpy.ndarray (1d)
            Far-field strain tensor (matricial form).
        applied_mac_load_strain_mf : numpy.ndarray (1d)
            Prescribed loading strain tensor (matricial form).
        eff_tangent_mf : numpy.ndarray (2d), default=None
            CRVE effective material tangent modulus (matricial form).
        """
        # Set strain/stress components order according to problem strain
        # formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get self-consistent scheme iteration counter
        scs_iter = ref_material.get_scs_iter()
        # Get elastic reference material properties
        ref_material_properties = ref_material.get_material_properties()
        # Get elastic reference material tangent modulus (matricial form)
        ref_elastic_tangent_mf = ref_material.get_elastic_tangent_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the 3D
        # far-field strain tensor considering the appropriate out-of-plane
        # strain component (output purpose only).
        farfield_strain = mop.get_tensor_from_mf(farfield_strain_mf,
                                                 self._n_dim, comp_order)
        out_farfield_strain = np.zeros((3, 3))
        if self._problem_type == 1:
            out_farfield_strain[0:2, 0:2] = farfield_strain
            if self._strain_formulation == 'finite':
                out_farfield_strain[2, 2] = 1.0
        else:
            out_farfield_strain[:, :] = farfield_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute norm of difference between the far-field strain tensor and
        # prescribed loading strain tensor and then normalize it to obtain
        # relative measure
        diff_norm = np.linalg.norm(farfield_strain_mf
                                   - applied_mac_load_strain_mf)
        rel_diff_farfield = \
            diff_norm/np.linalg.norm(applied_mac_load_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute self-consistent scheme normalized cost function
        if self._self_consistent_scheme == 'regression':
            # Compute regression-based scheme cost function
            scs_cost = np.linalg.norm(hom_stress_mf
                                      - np.matmul(ref_elastic_tangent_mf,
                                                  hom_strain_mf))**2
            # Normalize cost function
            rel_scs_cost = scs_cost/(np.linalg.norm(hom_stress_mf)**2)
        else:
            # If self-consistent scheme cost function computation is not
            # implemented, output normalized cost function value as infinite
            rel_scs_cost = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute norm of difference between the effective tangent modulus and
        # the reference material tangent modulus and then normalize it to
        # obtain a relative measure
        if eff_tangent_mf is not None:
            diff_norm = np.linalg.norm(ref_elastic_tangent_mf - eff_tangent_mf)
            rel_diff_tangent = diff_norm/np.linalg.norm(eff_tangent_mf)
        else:
            rel_diff_tangent = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material format structure
        inc_data = [inc, scs_iter,
                    ref_material_properties['E'], ref_material_properties['v'],
                    out_farfield_strain[0, 0], out_farfield_strain[1, 0],
                    out_farfield_strain[2, 0], out_farfield_strain[0, 1],
                    out_farfield_strain[1, 1], out_farfield_strain[2, 1],
                    out_farfield_strain[0, 2], out_farfield_strain[1, 2],
                    out_farfield_strain[2, 2],
                    rel_diff_farfield, rel_scs_cost, rel_diff_tangent]
        write_list = [
            '\n' + '{:>9d}'.format(inc) + '  ' + '{:>13d}'.format(scs_iter)
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in inc_data[2:]])]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open reference material output file and read lines (read)
        file_lines = open(self._file_path, 'r').readlines()
        # Update reference material output file according to output mode
        if self._ref_output_mode == 'iterative':
            # If loading increment is being repeated for some reason, then
            # clear the associated data and append the first iteration.
            # Otherwise append iteration belonging to the current loading
            # increment
            if len(file_lines) > 1 and int(file_lines[-1].split()[0]) == inc \
                    and not scs_iter > int(file_lines[-1].split()[1]):
                # If loading increment is being repeated for some reason, then
                # clear the associated data
                del file_lines[-(int(file_lines[-1].split()[1]) + 1):]
                file_lines += write_list[0][1:]
            else:
                file_lines += write_list[0]
        else:
            # If is the same loading increment, then replace the last
            # iteration. Otherwise append iteration belonging to new loading
            # increment
            if len(file_lines) > 1 and int(file_lines[-1].split()[0]) == inc:
                file_lines[-1] = write_list[0][1:]
            else:
                file_lines += write_list[0]
        # Open output file (write mode) and write data
        open(self._file_path, 'w').writelines(file_lines)
    # -------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        """Rewind output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated with the rewind state.
        """
        # Open reference material output file and read lines (read)
        file_lines = open(self._file_path, 'r').readlines()
        # Rewind reference material output file according to output mode
        if self._ref_output_mode == 'iterative':
            # Loop over file lines
            for i in range(1, len(file_lines)):
                # Get file line
                line = file_lines[i]
                # Check for increment after increment associated with rewind
                # state
                if int(line.split()[0]) == rewind_inc + 1:
                    # Set output file last line
                    last_line = i - 1
                    break
        else:
            # Set output file last line
            last_line = 1 + rewind_inc
        # Remove next line character
        file_lines[last_line] = file_lines[last_line][:-1]
        # Open output file (write mode) and write data
        open(self._file_path, 'w').writelines(file_lines[: last_line + 1])
