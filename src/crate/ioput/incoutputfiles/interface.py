"""Incremental output file interface.

This module includes the interface to implement any incremental output file.

Classes
-------
IncrementalOutputFile
    Incremental output file interface.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
class IncrementalOutputFile(ABC):
    """Incremental output file interface.

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
    init_file(self)
        *abstract*: Open output file and write file header.
    write_file(self)
        *abstract*: Write output file.
    rewind_file(self, rewind_inc)
        Rewind output file.
    """
    @abstractmethod
    def __init__(self, file_path):
        """Constructor.

        Parameters
        ----------
        file_path : str
            Output file path.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def init_file(self):
        """Open output file and write file header."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def write_file(self):
        """Write output file."""
        pass
    # -------------------------------------------------------------------------
    def rewind_file(self, rewind_inc):
        """Rewind output file.

        Parameters
        ----------
        rewind_inc : int
            Increment associated with the rewind state.
        """
        # Open output file and read lines (read)
        file_lines = open(self._file_path, 'r').readlines()
        # Set output file last line
        last_line = 1 + rewind_inc
        file_lines[last_line] = file_lines[last_line][:-1]
        # Open output file (write mode) and write data
        open(self._file_path, 'w').writelines(file_lines[: last_line + 1])
