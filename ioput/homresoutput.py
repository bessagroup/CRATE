#
# Homogenized Results Output Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the output file where the homogenized results are stored. Some
# post-process computations are also performed here, essentially derived from the
# homogenized strain and stress tensors.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Mar 2020 | Initial coding.
# Bernardo P. Ferreira | Jul 2020 | OOP refactorization.
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
#                                                      Homogenized results output file class
# ==========================================================================================
class HomResOutput:
    '''Homogenized results output.

    Attributes
    ----------
    _header : list
        List containing the header of each column (str).
    _col_width : int
        Output file column width.
    '''
    def __init__(self, hres_file_path):
        '''Homogenized results output constructor.

        Parameters
        ----------
        hres_file_path : str
            Path of homogenized results output file.
        '''
        self._hres_file_path = hres_file_path
        # Set homogenized results output file header
        self._header = ['Increment',
                        'strain_11', 'strain_22', 'strain_33',
                        'strain_12', 'strain_23', 'strain_13',
                        'stress_11', 'stress_22', 'stress_33',
                        'stress_12', 'stress_23', 'stress_13',
                        'vm_strain', 'vm_stress',
                        'strain_1', 'strain_2', 'strain_3',
                        'stress_1', 'stress_2', 'stress_3']
        # Set column width
        self._col_width = max(16, max([len(x) for x in self._header]) + 2)
    # --------------------------------------------------------------------------------------
    def init_hres_file(self):
        '''Open homogenized results output file and write file header.'''
        # Open homogenized results output file (write mode)
        hres_file = open(self._hres_file_path, 'w')
        # Set homogenized results output file header format structure
        write_list = ['{:>9s}'.format(self._header[0]) +
                      ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                      for x in self._header[1:]]),
                      '\n' + '{:>9d}'.format(0) +
                      ''.join([('{:>' + str(self._col_width) + '.8e}').format(0)
                                    for x in range(20)])]
        # Write homogenized results output file header
        hres_file.writelines(write_list)
        # Close homogenized results output file
        hres_file.close()
    # --------------------------------------------------------------------------------------
    def write_hres_file(self, problem_type, inc, hom_results):
        '''Write homogenized results output file.

        Parameters
        ----------
        problem_type : int
            Problem type identifier (1 - Plain strain (2D), 4- Tridimensional)
        inc : int
            Macroscale loading increment.
        hom_results : dict
            Homogenized results: homogenized strain tensor (key = 'strain'), homogenized
            stress tensor (key = 'stress'), homogenized out-of-plain stress component
            (key = 'hom_stress_33').
        '''
        # Get homogenized data
        hom_strain = hom_results['hom_strain']
        hom_stress = hom_results['hom_stress']
        if problem_type == 1:
            hom_stress_33 = hom_results['hom_stress_33']
        # When the problem type corresponds to a 2D analysis, build the 3D homogenized
        # strain and stress tensors by considering the appropriate out-of-plane strain and
        # stress components
        out_hom_strain = np.zeros((3, 3))
        out_hom_stress = np.zeros((3, 3))
        if problem_type == 1:
            out_hom_strain[0:2, 0:2] = hom_strain
            out_hom_stress[0:2, 0:2] = hom_stress
            out_hom_stress[2, 2] = hom_stress_33
        else:
            out_hom_strain[:, :] = hom_strain
            out_hom_stress[:, :] = hom_stress
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get 3D problem parameters
        n_dim, comp_order_sym, _ = mop.get_problem_type_parameters(problem_type=4)
        # Get fourth-order tensors
        _, _, _, _, _, _, fodevprojsym = top.get_id_operators(n_dim)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the von Mises equivalent strain
        vm_strain = np.sqrt(2.0/3.0)*np.linalg.norm(top.ddot42_1(fodevprojsym,
                                                                 out_hom_strain))
        # Compute the von Mises equivalent stress
        vm_stress = np.sqrt(3.0/2.0)*np.linalg.norm(top.ddot42_1(fodevprojsym,
                                                                 out_hom_stress))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute the eigenstrains (strain_1, strain_2, strain_3)
        eigenstrains = np.sort(np.linalg.eig(out_hom_strain)[0])[::-1]
        # Compute the eigenstresses (stress_1, stress_2, stress_3)
        eigenstresses = np.sort(np.linalg.eig(out_hom_stress)[0])[::-1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open homogenized results output file (append mode)
        hres_file = open(self._hres_file_path, 'a')
        # Set increment homogenized results format structure
        inc_data = [inc,
                    out_hom_strain[0, 0], out_hom_strain[1, 1], out_hom_strain[2, 2],
                    out_hom_strain[0, 1], out_hom_strain[1, 2], out_hom_strain[0, 2],
                    out_hom_stress[0, 0], out_hom_stress[1, 1], out_hom_stress[2, 2],
                    out_hom_stress[0, 1], out_hom_stress[1, 2], out_hom_stress[0, 2],
                    vm_strain, vm_stress,
                    eigenstrains[0], eigenstrains[1], eigenstrains[2],
                    eigenstresses[0], eigenstresses[1], eigenstresses[2]]
        write_list = ['\n' + '{:>9d}'.format(inc) +
                      ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                      for x in inc_data[1:]])]
        # Write increment homogenized results
        hres_file.writelines(write_list)
        # Close homogenized results output file
        hres_file.close()
    # --------------------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        '''Rewind homogenized results output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated to the rewind state.
        '''
        # Open homogenized results output file and read lines (read)
        file_lines = open(self._hres_file_path, 'r').readlines()
        # Set output file last line
        last_line = 1 + rewind_inc
        file_lines[last_line] = file_lines[last_line][:-1]
        # Open homogenized results output file (write mode)
        open(self._hres_file_path, 'w').writelines(file_lines[: last_line + 1])
