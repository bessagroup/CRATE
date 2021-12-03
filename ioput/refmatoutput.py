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
    _n_dim : int
        Problem number of spatial dimensions.
    _comp_order_sym : list
        Strain/Stress components symmetric order.
    _comp_order_nsym : list
        Strain/Stress components nonsymmetric order.
    _header : list
        List containing the header of each column (str).
    _col_width : int
        Output file column width.
    '''
    # --------------------------------------------------------------------------------------
    def __init__(self, refm_file_path, strain_formulation, problem_type,
                 self_consistent_scheme, is_farfield_formulation=True,
                 ref_output_mode='converged'):
        '''Reference material output constructor.

        Parameters
        ----------
        refm_file_path : str
            Path of reference material output file.
        strain_formulation: str, {'infinitesimal', 'finite'}
            Problem strain formulation.
        problem_type : int
            Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
            3D (4).
        self_consistent_scheme : int
            Self-consistent scheme (1-Regression-based)
        is_farfield_formulation : bool, default=True
            True if SCA farfield formulation, False otherwise.
        ref_output_mode : {'iterative', 'converged'}, default='converged'
            Output mode: 'iterative' outputs the reference material quantities at every
            self-consistent scheme iteration; 'converged' outputs the reference material
            quantities that converged at each macroscale loading increment.
        '''
        self._refm_file_path = refm_file_path
        self._strain_formulation = strain_formulation
        self._problem_type = problem_type
        self._self_consistent_scheme = self_consistent_scheme
        self._is_farfield_formulation = is_farfield_formulation
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
                        'inc_strain0_11', 'inc_strain0_22', 'inc_strain0_33',
                        'inc_strain0_12', 'inc_strain0_23', 'inc_strain0_13',
                        'strain_0_rdiff', 'rel_scs_cost', 'tangent_rdiff']
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
    def write_ref_mat(self, inc, ref_material, inc_hom_strain_mf, inc_hom_stress_mf,
                      eff_tangent_mf=None, inc_farfield_strain_mf=None,
                      inc_mac_load_strain_mf=None):
        '''Write reference material output file.

        Parameters
        ----------
        inc : int
            Macroscale loading increment.
        ref_material : ElasticReferenceMaterial
            Elastic reference material.
        inc_hom_strain_mf : ndarray
            Incremental homogenized strain tensor (matricial form).
        inc_hom_stress_mf : ndarray
            Incremental homogenized stress tensor (matricial form).
        eff_tangent_mf : ndarray, default=None
            CRVE effective (homogenized) tangent modulus (matricial form).
        inc_farfield_strain_mf : ndarray, default=None
            Incremental farfield strain tensor (matricial form).
        inc_mac_load_strain_mf : ndarray, default=None
            Incremental prescribed macroscale loading strain tensor (matricial form).
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check required parameters if SCA farfield formulation is being used
        if self._is_farfield_formulation:
            if inc_farfield_strain_mf is None or inc_mac_load_strain_mf is None:
                raise RuntimeError('Required parameters for SCA farfield formulation '
                                   'output are missing.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set strain/stress components order according to problem strain formulation
        if self._strain_formulation == 'infinitesimal':
            comp_order = self._comp_order_sym
        elif self._strain_formulation == 'finite':
            comp_order = self._comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get self-consistent scheme iteration counter
        scs_iter = ref_material.get_scs_iter()
        # Get elastic reference material properties
        ref_material_properties = ref_material.get_material_properties()
        # Get elastic reference material tangent modulus (matricial form)
        ref_elastic_tangent_mf = ref_material.get_elastic_tangent_mf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # When the problem type corresponds to a 2D analysis, build the 3D incremental
        # farfield strain tensor considering the appropriate out-of-plane strain component
        # (output purpose only).
        if self._is_farfield_formulation:
            inc_farfield_strain = mop.get_tensor_from_mf(inc_farfield_strain_mf,
                                                         self._n_dim, comp_order)
            out_inc_farfield_strain = np.zeros((3, 3))
            if self._problem_type == 1:
                out_inc_farfield_strain[0:2, 0:2] = inc_farfield_strain
            else:
                out_inc_farfield_strain[:, :] = inc_farfield_strain
        else:
            out_inc_farfield_strain = np.zeros((3, 3))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute norm of difference between the incremental farfield strain tensor and
        # prescribed macroscale loading strain tensor and then normalize it to obtain
        # relative measure
        if self._is_farfield_formulation:
            diff_norm = np.linalg.norm(inc_farfield_strain_mf - inc_mac_load_strain_mf)
            rel_diff_farfield = diff_norm/np.linalg.norm(inc_mac_load_strain_mf)
        else:
            rel_diff_farfield = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute self-consistent scheme normalized cost function
        if self._self_consistent_scheme == 1:
            # Compute regression-based scheme cost function
            scs_cost = np.linalg.norm(inc_hom_stress_mf -
                                      np.matmul(ref_elastic_tangent_mf,
                                                inc_hom_strain_mf))**2
            # Normalize cost function
            rel_scs_cost = scs_cost/(np.linalg.norm(inc_hom_stress_mf)**2)
        else:
            # If self-consistent scheme cost function computation is not implemented, output
            # normalized cost function value as infinite
            rel_scs_cost = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute norm of difference between the effective tangent modulus and the reference
        # material tangent modulus and then normalize it to obtain a relative measure
        if not eff_tangent_mf is None:
            diff_norm = np.linalg.norm(ref_elastic_tangent_mf - eff_tangent_mf)
            rel_diff_tangent = diff_norm/np.linalg.norm(eff_tangent_mf)
        else:
            rel_diff_tangent = 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set reference material format structure
        inc_data = [inc, scs_iter,
                    ref_material_properties['E'], ref_material_properties['v'],
                    out_inc_farfield_strain[0, 0], out_inc_farfield_strain[1, 1],
                    out_inc_farfield_strain[2, 2], out_inc_farfield_strain[0, 1],
                    out_inc_farfield_strain[1, 2], out_inc_farfield_strain[0, 2],
                    rel_diff_farfield, rel_scs_cost, rel_diff_tangent]
        write_list = ['\n' + '{:>9d}'.format(inc) + '  ' + '{:>13d}'.format(scs_iter) +
                      ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                      for x in inc_data[2:]])]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open reference material output file and read lines (read)
        file_lines = open(self._refm_file_path, 'r').readlines()
        # Update reference material output file according to output mode
        if self._ref_output_mode == 'iterative':
            # If macroscale loading increment is being repeated for some reason, then
            # clear associated data and append the first iteration. Otherwise append
            # iteration belonging to the current macroscale loading increment
            if len(file_lines) > 1 and int(file_lines[-1].split()[0]) == inc and \
                not scs_iter > int(file_lines[-1].split()[1]):
                # If macroscale loading increment is being repeated for some reason, then
                # clear associated data
                del file_lines[-(int(file_lines[-1].split()[1]) + 1):]
                file_lines += write_list[0][1:]
            else:
                file_lines += write_list[0]
        else:
            # If is the same macroscale loading increment, then replace the last iteration.
            # Otherwise append iteration belonging to new macroscale loading increment
            if len(file_lines) > 1 and int(file_lines[-1].split()[0]) == inc:
                file_lines[-1] = write_list[0][1:]
            else:
                file_lines += write_list[0]
        # Open reference material output file (write mode)
        open(self._refm_file_path, 'w').writelines(file_lines)
    # --------------------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        '''Rewind reference material output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated to the rewind state.
        '''
        # Open reference material output file and read lines (read)
        file_lines = open(self._refm_file_path, 'r').readlines()
        # Rewind reference material output file according to output mode
        if self._ref_output_mode == 'iterative':
            # Loop over file lines
            for i in range(1, len(file_lines)):
                # Get file line
                line = file_lines[i]
                # Check for increment after increment associated to rewind state
                if int(line.split()[0]) == rewind_inc + 1:
                    # Set output file last line
                    last_line = i - 1
                    break
        else:
            # Set output file last line
            last_line = 1 + rewind_inc
        # Remove next line character
        file_lines[last_line] = file_lines[last_line][:-1]
        # Open reference material output file (write mode)
        open(self._refm_file_path, 'w').writelines(file_lines[: last_line + 1])
