"""Output file: CRVE effective material consistent tangent modulus.

This module includes the class associated with the output file where the
CRVE effective material consistent tangent modulus is stored.

Classes
-------
EffTanOutput
    Output file: CRVE effective material consistent tangent modulus.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools as it
# Third-party
import numpy as np
# Local
import tensor.matrixoperations as mop
from ioput.incoutputfiles.interface import IncrementalOutputFile
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class EffTanOutput(IncrementalOutputFile):
    """Output file: CRVE effective material consistent tangent modulus.

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
               eff_tangent_mf)
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
        # Set output strain/stress components order
        out_comp_order = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
        # Set output file header
        self._header = ['Increment', ] \
            + [''.join((y, x)) for (x, y) in  it.product(out_comp_order,
                                                         out_comp_order)]
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
        # Set increment and CRVE effective material consistent tangent modulus
        # initial outputs
        inc_init = 0
        eff_tan_init = (9**2)*[0.0, ]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open output file (write mode)
        output_file = open(self._file_path, 'w')
        # Set output file header format structure
        write_list = [
            '{:>9s}'.format(self._header[0])
            + ''.join([('{:>' + str(self._col_width) + 's}').format(x)
                       for x in self._header[1:]]),
            '\n' + '{:>9d}'.format(inc_init)
            + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                       for x in eff_tan_init])]
        # Write output file header
        output_file.writelines(write_list)
        # Close output file
        output_file.close()
    # -------------------------------------------------------------------------
    def write_file(self, strain_formulation, problem_type, mac_load_path,
                   eff_tangent_mf):
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
        eff_tangent_mf : numpy.ndarray (2d)
            CRVE effective material tangent modulus (matricial form).
        """
        # Get loading path data
        _, sp_inc, _, _, _, _, _ = mac_load_path.get_subpath_state()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get problem type parameters
        n_dim, comp_order_sym, comp_order_nsym = \
            mop.get_problem_type_parameters(problem_type)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set default strain/stress components order according to problem
        # strain formulation
        if strain_formulation == 'infinitesimal':
            comp_order = comp_order_sym
        elif strain_formulation == 'finite':
            comp_order = comp_order_nsym
        else:
            raise RuntimeError('Unknown problem strain formulation.')
        # Get CRVE effective material consistent tangent modulus tensor from
        # matricial form
        eff_tangent = mop.get_tensor_from_mf(eff_tangent_mf, n_dim, comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize output CRVE effective material consistent tangent modulus
        # tensor
        out_eff_tangent = np.zeros((3, 3, 3, 3))
        # Build output CRVE effective material consistent tangent modulus
        # tensor
        for i, j, k, l in it.product(range(n_dim), repeat=4):
            out_eff_tangent[i, j, k, l] = eff_tangent[i, j, k, l]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set output strain/stress components order
        out_comp_order = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
        # Store output CRVE effective material consistent tangent modulus
        # tensor in matricial form
        out_eff_tangent_mf = mop.get_tensor_mf(out_eff_tangent, n_dim=3,
                                               comp_order=out_comp_order)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Open output file (append mode)
        output_file = open(self._file_path, 'a')
        # Set CRVE effective material consistent tangent modulus format
        # structure
        inc_data = [sp_inc, ] + list(out_eff_tangent_mf.flatten(order='F'))
        write_list = \
            ['\n' + ('{:>9d}').format(inc_data[0])
             + ''.join([('{:>' + str(self._col_width) + '.8e}').format(x)
                        for x in inc_data[1:]])]
        # Write CRVE effective material consistent tangent modulus
        output_file.writelines(write_list)
        # Close output file
        output_file.close()
