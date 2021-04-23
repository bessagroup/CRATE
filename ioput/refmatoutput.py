#
# Reference Material Output Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the output file where reference material associated quantities are
# stored, namely properties and the far-field strain tensor.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2021 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Matricial operations
import tensor.matrixoperations as mop
#
#                                                       Reference material output file class
# ==========================================================================================
class RefMatOutput:
    '''Reference material output.

    Attributes
    ----------
    _header : list
        List containing the header of each column (str).
    _col_width : int
        Output file column width.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, refm_file_path, self_consistent_scheme, ref_output_mode='converged'):
        '''Reference material output constructor.

        Parameters
        ----------
        refm_file_path : str
            Path of reference material output file.
        self_consistent_scheme : int
            Self-consistent scheme (1-Regression-based)
        ref_output_mode : {'iterative', 'converged'}, default='converged'
            Output mode: 'iterative' outputs the reference material quantities at every
            self-consistent scheme iteration; 'converged' outputs the reference material
            quantities that converged at each macroscale loading increment.
        '''
        self._refm_file_path = refm_file_path
        self._self_consistent_scheme = self_consistent_scheme
        self._ref_output_mode = ref_output_mode
        # Set reference material output file header
        self._header = ['Increment', 'SCS Iteration',
                        'E_ref', 'v_ref',
                        'inc_strain0_11', 'inc_strain0_22', 'inc_strain0_33',
                        'inc_strain0_12', 'inc_strain0_23', 'inc_strain0_13',
                        'strain_0_rdiff', 'rel_scs_cost']
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # --------------------------------------------------------------------------------------
    def init_ref_mat_file(self):
        '''Open reference material output file and write file header.'''
        # Open reference material output file (write mode)
        refm_file = open(self._refm_file_path, 'w')
        # Set reference material output file header format structure
        write_list = ['{:>9s}'.format(self._header[0]) + '  ' +
                      '{:>13s}'.format(self._header[1]) +
                      ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                      for x in self._header[2:]]),
                      '\n' + '{:>9d}'.format(0) + '  ' + '{:>13d}'.format(0) +
                      ''.join([('{:>' + str(self._col_width) + '.8e}').format(0)
                      for x in self._header[2:]]),]
        # Write reference material output file header
        refm_file.writelines(write_list)
        # Close homogenized results output file
        refm_file.close()
    # --------------------------------------------------------------------------------------
    def write_ref_mat(self, problem_type, n_dim, comp_order, inc, scs_iter, mat_prop_ref,
                            De_ref_mf, inc_farfield_strain_mf, inc_mac_load_strain_mf,
                            inc_hom_strain_mf, inc_hom_stress_mf):
        '''Write reference material output file.

        Parameters
        ----------
        problem_type : int
            Problem type (1-Plane strain, 4-Tridimensional)
        n_dim : int
            Problem dimension.
        comp_order : list
            Strain/Stress components (str) order.
        inc : int
            Macroscale loading increment.
        scs_iter : int
            Self-consistent scheme iteration.
        mat_prop_ref : dict
            Isotropic elastic reference material properties.
        De_ref_mf : ndarray
            Isotropic elastic reference material tangent modulus (matricial form).
        inc_farfield_strain_mf : ndarray
            Incremental farfield strain tensor (matricial form).
        inc_mac_load_strain_mf : ndarray
            Incremental prescribed macroscale loading strain tensor (matricial form).
        inc_hom_strain_mf : ndarray
            Incremental homogenized strain tensor (matricial form).
        inc_hom_stress_mf : ndarray
            Incremental homogenized stress tensor (matricial form).
        '''
        # When the problem type corresponds to a 2D analysis, build the 3D incremental
        # farfield strain tensor considering the appropriate out-of-plane strain component
        # (output purpose only).
        inc_farfield_strain = mop.gettensorfrommf(inc_farfield_strain_mf, n_dim, comp_order)
        out_inc_farfield_strain = np.zeros((3, 3))
        if problem_type == 1:
            out_inc_farfield_strain[0:2, 0:2] = inc_farfield_strain
        else:
            out_inc_farfield_strain[:, :] = inc_farfield_strain
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute norm of difference between the incremental farfield strain tensor and
        # prescribed macroscale loading strain tensor and then normalize it to obtain
        # relative measure
        diff_norm = np.linalg.norm(inc_farfield_strain_mf - inc_mac_load_strain_mf)
        rel_diff = diff_norm/np.linalg.norm(inc_mac_load_strain_mf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute self-consistent scheme normalized cost function
        if self._self_consistent_scheme == 1:
            # Compute regression-based scheme cost function
            scs_cost = np.linalg.norm(inc_hom_stress_mf -
                                      np.matmul(De_ref_mf, inc_hom_strain_mf))**2
            # Normalize cost function
            rel_scs_cost = scs_cost/(np.linalg.norm(inc_hom_stress_mf)**2)
        else:
            # If self-consistent scheme cost function computation is not implemented, output
            # normalized cost function value as infinite
            rel_scs_cost = np.inf
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material format structure
        inc_data = [inc, scs_iter,
                    mat_prop_ref['E'], mat_prop_ref['v'],
                    out_inc_farfield_strain[0, 0], out_inc_farfield_strain[1, 1],
                    out_inc_farfield_strain[2, 2], out_inc_farfield_strain[0, 1],
                    out_inc_farfield_strain[1, 2], out_inc_farfield_strain[0, 2],
                    rel_diff, rel_scs_cost]
        write_list = ['\n' + '{:>9d}'.format(inc) + '  ' + '{:>13d}'.format(scs_iter) +
                      ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                      for x in inc_data[2:]])]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update reference material output file according to output mode
        if self._ref_output_mode == 'iterative':
            # Open reference material output file (append mode)
            refm_file = open(self._refm_file_path, 'a')
            # Write reference material related quantities
            refm_file.writelines(write_list)
        else:
            # Open reference material output file and read lines (read)
            file_lines = open(self._refm_file_path, 'r').readlines()
            # If is the same macroscale loading increment, then replace the last iteration.
            # Otherwise append iteration belonging to new macroscale loading increment
            if len(file_lines) > 1 and int(file_lines[-1].split()[0]) == inc:
                file_lines[-1] = write_list[0][1:]
            else:
                file_lines += write_list[0]
            # Open reference material output file (write mode)
            refm_file = open(self._refm_file_path, 'w')
            # Write reference material related quantities
            refm_file.writelines(file_lines)
        # Close reference material output file
        refm_file.close()
