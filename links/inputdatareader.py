#
# Input Links Data Reader (CRATE Program)
# ==========================================================================================
# Summary:
# Reading procedures of Links parameters from CRATE input data file.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Nov 2021 | Initial coding.
# ==========================================================================================
# Operating system related functions
import os
# Read specific lines from file
import linecache
# I/O utilities
from ioput.ioutilities import checknumber, checkposint
# Read procedures
from ioput.readprocedures import searchkeywordline, searchoptkeywordline
# Links related procedures
from links.configuration import get_links_analysis_type
#
#                                                                    Links parameters reader
# ==========================================================================================
def read_links_input_parameters(file, file_path, problem_type):
    '''Read Links parameters from CRATE input data file.

    Parameters
    ----------
    file : file
        CRATE input data file.
    file_path : str
        CRATE input data file path.
    problem_type : int
        Problem type: 2D plane strain (1), 2D plane stress (2), 2D axisymmetric (3) and
        3D (4).

    Returns
    -------
    links_data : dict
        Dictionary containing Links parameters.
    '''
    # Initialize Links parameters dictionary
    links_data = dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links analysis type
    links_data['analysis_type'] = get_links_analysis_type(problem_type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the Links binary absolute path
    keyword = 'Links_bin'
    line_number = searchkeywordline(file, keyword) + 1
    links_bin_path = linecache.getline(file_path, line_number).strip()
    if not os.path.isabs(links_bin_path):
        raise RuntimeError('Input data error: Links binary path must be absolute.')
    elif not os.path.isfile(links_bin_path):
        raise RuntimeError('Input data error: Links binary file has not been found.')
    # Store Links binary absolute path
    links_data['links_bin_path'] = links_bin_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read Links finite element order (linear or quadratic)
    keyword = 'Links_FE_Order'
    is_found, keyword_line_number = searchoptkeywordline(file, keyword)
    if is_found:
        line = linecache.getline(file_path, keyword_line_number).split()
        if len(line) == 1:
            raise RuntimeError('Input data error: Missing Links finite element order ' +
                               'specification.')
        elif line[1] not in ['linear', 'quadratic']:
            raise RuntimeError('Input data error: Links finite element order must be' +
                               'either \'linear\' or \'quadratic\'.')
        fe_order = line[1]
    else:
        fe_order = 'quadratic'
    # Store Links finite element order
    links_data['fe_order'] = fe_order
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read Links microscale boundary condition
    keyword = 'Links_Boundary_Type'
    is_found, keyword_line_number = searchoptkeywordline(file, keyword)
    if is_found:
        line = linecache.getline(file_path, keyword_line_number).split()
        if len(line) == 1:
            raise RuntimeError('Input data error: Missing Links microscale boundary ' +
                               'condition specification.')
        elif line[1] not in ['Taylor_Condition', 'Linear_Condition', 'Periodic_Condition',
                             'Uniform_Traction_Condition', 'Uniform_Traction_Condition_II',
                             'Mortar_Periodic_Condition', 'Mortar_Periodic_Condition_LM']:
            raise RuntimeError('Input data error: Unknown Links microscale boundary ' +
                               'condition.')
        boundary_type = line[1]
    else:
        boundary_type = 'Periodic_Condition'
    # Store Links microscale boundary condition
    links_data['boundary_type'] = boundary_type
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read Links convergence tolerance
    keyword = 'Links_Convergence_Tolerance'
    is_found, keyword_line_number = searchoptkeywordline(file, keyword)
    if is_found:
        line = linecache.getline(file_path, keyword_line_number + 1).split()
        if line == '':
            raise RuntimeError('Input data error: Missing Links convergence tolerance ' +
                               'specification.')
        elif len(line) != 1:
            raise RuntimeError('Input data error: Invalid Links convergence tolerance ' +
                               'specification.')
        elif not checknumber(line[0]) or float(line[0]) <= 0:
            raise RuntimeError('Input data error: Links convergence tolerance must be' +
                               'positive floating-point number.')
        convergence_tolerance = float(line[0])
    else:
        convergence_tolerance = 1e-6
    # Store Links convergence tolerance
    links_data['convergence_tolerance'] = convergence_tolerance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read Links elementwise average output mode
    keyword = 'Links_Element_Average_Output_Mode'
    is_found, keyword_line_number = searchoptkeywordline(file, keyword)
    if is_found:
        if len(line) == 1:
            raise RuntimeError('Input data error: Missing Links elementwise average ' +
                               'output mode specification.')
        elif not checkposint(line[1]):
            raise RuntimeError('Input data error: Unknown Links elementwise average ' +
                               'output mode.')
        element_avg_output_mode = int(line[1])
    else:
        element_avg_output_mode = 1
    # Store Links elementwise average output mode
    links_data['element_avg_output_mode'] = element_avg_output_mode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return links_data
