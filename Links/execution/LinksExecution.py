#
# Links Execution Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures required to call the finite element code Links in order to
# solve a microscale equilibrium problem.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Subprocess management
import subprocess
# Extract information from path
import ntpath
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
#
#                                                                    Links execution process
# ==========================================================================================
# Solve a given microscale equilibrium problem with Links
def runLinks(Links_bin_path,Links_file_path):
    # Call Links
    subprocess.run([Links_bin_path,Links_file_path],stdout=subprocess.PIPE,\
                                                                     stderr=subprocess.PIPE)
    # Check if the microscale equilibrium problem was successfully solved
    screen_file_name = ntpath.splitext(ntpath.basename(Links_file_path))[0]
    screen_file_path = ntpath.dirname(Links_file_path) + '/' + \
                                       screen_file_name + '/' + screen_file_name + '.screen'
    if not os.path.isfile(screen_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00071',location.filename,location.lineno+1,screen_file_path)
    else:
        is_solved = False
        screen_file = open(screen_file_path,'r')
        screen_file.seek(0)
        line_number = 0
        for line in screen_file:
            line_number = line_number + 1
            if 'Program L I N K S successfully completed.' in line:
                is_solved = True
                break
        if not is_solved:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00072',location.filename,location.lineno+1,
                                                           ntpath.basename(Links_file_path))
