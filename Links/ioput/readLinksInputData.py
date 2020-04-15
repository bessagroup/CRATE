#
# Links Input Data Reader Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Read specific lines from file
import linecache
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import errors
#
#                                                                Links input data parameters
# ==========================================================================================
# Read all the Links related parameters from the input data file required to generate a
# Links input data file, to solve a microscale equilibrium problem with Links or to use the
# Links state update and consistent tangent modulus interfaces
def readLinksInputData(file,file_path,problem_type,checkNumber,checkPositiveInteger,
                                               searchKeywordLine,searchOptionalKeywordLine):
    # Initialize Links parameters dictionary
    Links_dict = dict()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set Links problem type
    problem_type_converter = {'1':2, '2':1, '3':3, '4':6}
    Links_dict['analysis_type'] = problem_type_converter[str(problem_type)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the Links binary absolute path
    keyword = 'Links_bin'
    line_number = searchKeywordLine(file,keyword) + 1
    Links_bin_path = linecache.getline(file_path,line_number).strip()
    if not os.path.isabs(Links_bin_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00068',location.filename,location.lineno+1,keyword, \
                                                                             Links_bin_path)
    elif not os.path.isfile(Links_bin_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00068',location.filename,location.lineno+1,keyword, \
                                                                             Links_bin_path)
    # Store Links binary absolute path
    Links_dict['Links_bin_path'] = Links_bin_path
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the finite element order (linear or quadratic). If the associated keyword is not
    # found, then a default specification is assumed
    keyword = 'Links_FE_Order'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number).split()
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00063',location.filename,location.lineno+1,keyword)
        elif line[1] not in ['linear','quadratic']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00063',location.filename,location.lineno+1,keyword)
        fe_order = line[1]
    else:
        fe_order = 'quadratic'
    # Store finite element order
    Links_dict['fe_order'] = fe_order
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read microscale boundary condition. If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Links_Boundary_Type'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number).split()
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00064',location.filename,location.lineno+1,keyword)
        elif line[1] not in ['Taylor_Condition','Linear_Condition','Periodic_Condition',
                               'Uniform_Traction_Condition','Uniform_Traction_Condition_II',
                                'Mortar_Periodic_Condition','Mortar_Periodic_Condition_LM']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00064',location.filename,location.lineno+1,keyword)
        boundary_type = line[1]
    else:
        boundary_type = 'Periodic_Condition'
    # Store microscale boundary condition
    Links_dict['boundary_type'] = boundary_type
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read convergence tolerance. If the associated keyword is not found, then a default
    # specification is assumed
    keyword = 'Links_Convergence_Tolerance'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        line = linecache.getline(file_path,keyword_line_number+1).split()
        if line == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        elif len(line) != 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        elif not checkNumber(line[0]) or float(line[0]) <= 0:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00065',location.filename,location.lineno+1,keyword)
        convergence_tolerance = float(line[0])
    else:
        convergence_tolerance = 1e-6
    # Store convergence tolerance
    Links_dict['convergence_tolerance'] = convergence_tolerance
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read elemental average output mode. If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Links_Element_Average_Output_Mode'
    isFound,keyword_line_number = searchOptionalKeywordLine(file,keyword)
    if isFound:
        if len(line) == 1:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00069',location.filename,location.lineno+1,keyword)
        elif not checkPositiveInteger(line[1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00069',location.filename,location.lineno+1,keyword)
        element_avg_output_mode = int(line[1])
    else:
        element_avg_output_mode = 1
    # Store element average output mode
    Links_dict['element_avg_output_mode'] = element_avg_output_mode
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return Links_dict
