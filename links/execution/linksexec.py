#
# Links Execution Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures required to call the finite element code Links in order to solve a microscale
# equilibrium problem.
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
#
#                                                                    Links execution process
# ==========================================================================================
# Solve a given microscale equilibrium problem with Links
def runlinks(links_bin_path, links_file_path):
    # Call Links
    subprocess.run([links_bin_path, links_file_path], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
    # Check if the microscale equilibrium problem was successfully solved
    screen_file_name = ntpath.splitext(ntpath.basename(links_file_path))[0]
    screen_file_path = ntpath.dirname(links_file_path) + '/' + \
                                      screen_file_name + '/' + screen_file_name + '.screen'
    if not os.path.isfile(screen_file_path):
        raise RuntimeError('The following Links screen output file has '
                           'not been found. Most probably the file has '
                           'not been written by the Links program.' \
                           '\n\n' + screen_file_path)
    else:
        is_solved = False
        screen_file = open(screen_file_path, 'r')
        screen_file.seek(0)
        line_number = 0
        for line in screen_file:
            line_number = line_number + 1
            if 'Program L I N K S successfully completed.' in line:
                is_solved = True
                break
        if not is_solved:
            raise RuntimeError('The program Links could not successfully '
                               'solve the following microscale equilibrium '
                               'problem:' + '\n\n' + screen_file_path)
