#
# File Operations Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to operations on files, directories and associated paths. Among these
# operations is the definition of the results folder associated to a given problem input
# data file, directory where all the related output is stored.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Jan 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2020 | Replaced '.clusters' and '.cit' files by '.crve' file.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Operations on files and directories
import shutil
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Display messages
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                                       Directory operations
# ==========================================================================================
# Make directory (both overwrite and no overwrite (default) options available)
def makedirectory(dir, option='no_overwrite'):
    try:
        if option == 'no_overwrite':
            os.mkdir(dir)
        elif option == 'overwrite':
            if not os.path.exists(dir):
                os.mkdir(dir)
            else:
                shutil.rmtree(dir)
                os.mkdir(dir)
    except OSError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1, message)
# Remove all the unrequired directories and files in a target directory
def rmunreqdirs(target_dir, required_dirnames):
    dirnames = os.listdir(target_dir)
    for dir in dirnames:
        if dir not in required_dirnames:
            path = target_dir + dir
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
#
#                                                    User input data file and problem folder
# ==========================================================================================
# Set input data file name, absolute path and directory
def setinputdatafilepath(path):
    # Set input data file path, directory, name and extension
    input_file_path = os.path.abspath(path)
    input_file_dir = ntpath.dirname(input_file_path) + '/'
    input_file_name = ntpath.splitext(ntpath.basename(input_file_path))[-2]
    input_file_ext = ntpath.splitext(ntpath.basename(input_file_path))[-1]
    # Check if the input data file has the required '.dat' extension
    if input_file_ext != '.dat':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00010', location.filename, location.lineno + 1)
    # Check if the input data file name only contains numbers, letters or underscores
    if not ioutil.checkvalidname(input_file_name):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00002', location.filename, location.lineno + 1)
    # Return
    return [input_file_name, input_file_path, input_file_dir]
# ------------------------------------------------------------------------------------------
# Set the problem folder (directories and files) as described below:
#
# example.dat
# shinyregulargrid.rgmsh
# example/ ------- example.screen
#            |---- shinyregulargrid.rgmsh
#            |---- Offline_Stage/ ------- example.crve
#            |                      |---- example_clusters.vti
#            |---- example.hres
#            |---- example.refm
#            |---- example.adapt
#            |---- Post_Process/  ------- example.pvd
#                                   |---- VTK/
#                                   |---- example.voxout
#
# Folders and files meaning:
# example.dat            - User input data file
# shinyregulargrid.rgmsh - Example of spatial discretization file (doesn't need to be in the
#                          same directory as the user input data file)
# example/               - Problem folder
# example.screen         - File where all the data printed to the default standard output
#                          device is stored
# shinyregulargrid.rgmsh - Copy of the spatial discretization file contained in the problem
#                          folder
# Offline_Stage/         - Offline stage folder
# example.crve           - File which contains an offline state CRVE instance (binary
#                          format)
# example_clusters.vti   - VTK XML file with data related to the material phases and
#                          material clusters
# example.hres           - File where the homogenized results are stored
# example.refm           - File where the reference material properties and associated
#                          farfield strain are stored
# example.adapt          - File where the adaptive material phases number of clusters and
#                          adaptivity steps are stored (only if clustering adaptivity is
#                          activated)
# Post_Process/          - Post processing folder
# example.pvd            - VTK XML file which contains the references to all the snapshots
#                          of the problem solution
# VTK/                   - Folder which constains all the snapshots of the problem solution
#                          (VTK XML format)
# example.voxout         - File where voxels material-related quantities are stored
#
def setproblemdirs(input_file_name, input_file_dir):
    # Set problem name, directory
    problem_name = input_file_name
    problem_dir = input_file_dir + problem_name + '/'
    # Set offline stage and post processing subdirectories
    offline_stage_dir = problem_dir + 'Offline_Stage' + '/'
    postprocess_dir = problem_dir + 'Post_Process' + '/'
    # Set '.screen' path and delete the file it it already exists
    ioutil.screen_file_path = problem_dir + input_file_name + '.screen'
    if os.path.isfile(ioutil.screen_file_path):
        os.remove(ioutil.screen_file_path)
    # Set '.crve' path
    crve_file_path = offline_stage_dir + input_file_name + '.crve'
    # Set '.hres' path
    hres_file_path = problem_dir + input_file_name + '.hres'
    # Set '.refm' path
    refm_file_path = problem_dir + input_file_name + '.refm'
    # Set '.adapt' path
    adapt_file_path = problem_dir + input_file_name + '.adapt'
    # Check if the problem directory already exists or not
    if not os.path.exists(problem_dir):
        status = 0
        # Create problem directory and main subdirectories
        makedirectory(problem_dir)
        for dir in [offline_stage_dir, postprocess_dir]:
            makedirectory(dir)
        # There is no previously computed offline stage data
        is_same_offstage = False
    else:
        ioutil.print2('\nWarning: The problem directory for the specified input data ' + \
                      'file already exists.')
        # Ask user if the purpose is to consider a previously computed offline state
        # CRVE data file
        is_same_offstage = ioutil.query_yn('\nDo you wish to consider the already ' + \
                                           'existent offline state \'.crve\' datafile?',
                                           'no')
        if is_same_offstage:
            status = 1
            # If the offline stage subdirectory does not exist raise error
            if not os.path.exists(offline_stage_dir):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00009', location.filename, location.lineno + 1,
                                    problem_name, offline_stage_dir)
            # Remove all the existent directories and files except the existent offline
            # stage subdirectory and '.screen' file
            required_dirnames = [input_file_name + '.screen', 'Offline_Stage']
            rmunreqdirs(problem_dir, required_dirnames)
            # Create post processing subdirectory
            makedirectory(postprocess_dir, 'overwrite')
            # Warn user to potential compatibility issues between the problem input data
            # file and the already existent offline state '.crve' data file
            ioutil.useraction('\n\nWarning: Please make sure that the problem input ' + \
                              'data file is consistent with the already ' + '\n' + \
                              len('Warning: ')*' ' + 'existent offline stage \'.crve\' ' + \
                              'data file (stored in Offline_Stage/) to avoid ' + \
                              '\n' + \
                              len('Warning: ')*' ' + 'unexpected errors or ' + \
                              'misleading conclusions.' + '\n\n' + \
                              'Press any key to continue or type \'exit\' to leave: ')
        else:
            # Ask user if the existent problem directory is to be overwritten
            is_overwrite = ioutil.query_yn('\nDo you wish to overwrite the problem ' + \
                                           'directory?')
            if is_overwrite:
                status = 2
                # Remove all the existent directories and files
                required_dirnames = [input_file_name + '.screen']
                rmunreqdirs(problem_dir, required_dirnames)
                # Create problem directory and main subdirectories
                for dir in [offline_stage_dir, postprocess_dir]:
                    makedirectory(dir, 'overwrite')
            else:
                status = 3
        # Display information about the problem directory and status
        info.displayinfo('-1', problem_dir, status)
    # Return
    return [problem_name, problem_dir, offline_stage_dir, postprocess_dir, is_same_offstage,
            crve_file_path, hres_file_path, refm_file_path, adapt_file_path]
