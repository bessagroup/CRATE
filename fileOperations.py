#
# File Operations Module (UNNAMED Program)
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Parse command-line options and arguments
import sys
# Operations on files and directories
import shutil
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Display messages
import info
# Display errors, warnings and built-in exceptions
import errors
# Read user input data file
import readInputData as rid
#
#                                                                           Global variables
# ==========================================================================================
# Set '.screen' file path as a global variable
screen_file_path = ''
#
#                                                                      Prompt user functions
# ==========================================================================================
# Prompt the user to answer a 'yes' or 'no' question
def query_yn(question, default_answer = 'yes'):
    answer = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default_answer is None:
        prompt = ' [y/n] '
    elif default_answer == 'yes':
        prompt = ' [Y/n] '
    elif default_answer == 'no':
        prompt = ' [y/N] '
    n_invalid_ans = 0
    while True:
        option = input(question + prompt).lower()
        screen_file = open(screen_file_path,'a')
        print(question+prompt+option,file = screen_file)
        screen_file.close()
        if default_answer is not None and option == '':
            return answer[default_answer]
        elif option in answer:
            return answer[option]
        else:
            n_invalid_ans = n_invalid_ans + 1
            if n_invalid_ans > 3 or str(option).lower() == 'exit':
                info.print2('\nProgram aborted.\n')
                sys.exit(1)
            info.print2('Please answer with \'yes\' or \'no\' (or \'exit\' to quit).')
# ------------------------------------------------------------------------------------------
# Prompt the user to perform some action before proceeding the program execution
def userAction(message):
    option = input(message)
    screen_file = open(screen_file_path,'a')
    print(message+option,file = screen_file)
    screen_file.close()
    if str(option).lower() == 'exit':
        info.print2('\nProgram aborted.\n')
        sys.exit(1)
#
#                                                                       Directory operations
# ==========================================================================================
# Make directory (both overwrite and no overwrite (default) options available)
def makeDirectory(dir, default = 'no_overwrite'):
    try:
        if default is None or default == 'no_overwrite':
            os.mkdir(dir)
        elif default == 'overwrite':
            if not os.path.exists(dir):
                os.mkdir(dir)
            else:
                shutil.rmtree(dir)
                os.mkdir(dir)
    except OSError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
# Remove all the unrequired directories and files in a target directory
def rmUnrequiredDirs(target_dir,required_dirnames):
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
def setInputDataFilePath(path):
    # Set input data file path, directory, name and extension
    input_file_path = path
    input_file_dir = ntpath.dirname(input_file_path) + '/'
    input_file_name = ntpath.splitext(ntpath.basename(input_file_path))[-2]
    input_file_ext = ntpath.splitext(ntpath.basename(input_file_path))[-1]
    # Check if the input data file has the required '.dat' extension
    if input_file_ext != '.dat':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00010',location.filename,location.lineno+1)
    # Check if the input data file name only contains numbers, letters or underscores
    if not rid.checkValidName(input_file_name):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00002',location.filename,location.lineno+1)
    # Return
    return [input_file_name,input_file_path,input_file_dir]
# ------------------------------------------------------------------------------------------
# Set the problem folder (directories and files) as described below:
#
# example.dat
# shinyregulargrid.rgmsh
# example/ ------- example.screen
#            |---- shinyregulargrid.rgmsh
#            |---- Offline_Stage/ ------- example.info
#            |                      |---- example.cluster
#            |                      |---- example.cit
#            |---- example.hres
#            |---- Post_Process/  ------- example.pvd
#                                   |---- VTK/
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
# example.info           - File which contains some parameters associated to the offline
#                          stage (serving to check if previously computed offline stage can
#                          be considered)
# example.cluster        - File which contains all the required information associated to
#                          the clustering discretization (binary format)
# example.cit            - File which contains the cluster interaction tensors
# example.hres           - File where standard homogenized results are stored
# Post_Process/          - Post processing folder
# example.pvd            - VTK XML file which contains the references to all the snapshots
#                          of the problem solution
# VTK/                   - Folder which constains all the snapshots of the problem solution
#                          (VTK XML format)
#
def setProblemDirs(input_file_name,input_file_dir):
    # Set problem name, directory
    problem_name = input_file_name
    problem_dir = input_file_dir + problem_name + '/'
    # Set offline stage and post processing subdirectories
    offline_stage_dir = problem_dir + 'Offline_Stage' + '/'
    postprocess_dir = problem_dir + 'Post_Process' + '/'
    # Set '.screen' path and delete the file it it already exists
    global screen_file_path
    screen_file_path = problem_dir + input_file_name + '.screen'
    if os.path.isfile(screen_file_path):
        os.remove(screen_file_path)
    # Set '.cluster' path
    cluster_file_path = offline_stage_dir + input_file_name + '.clusters'
    # Set '.cit' path
    cit_file_path = offline_stage_dir + input_file_name + '.cit'
    # Set '.hres' path
    hres_file_path = problem_dir + input_file_name + '.hres'
    # Check if the problem directory already exists or not
    if not os.path.exists(problem_dir):
        status = 0
        # Create problem directory and main subdirectories
        makeDirectory(problem_dir)
        for dir in [offline_stage_dir,postprocess_dir]:
            makeDirectory(dir)
        # There is no previously computed offline stage data
        is_same_offstage = False
    else:
        info.print2('\nWarning: The problem directory for the specified input data ' + \
                                                                     'file already exists.')
        # Ask user if the purpose is to consider the previously computed offline stage data
        # files
        is_same_offstage = query_yn('\nDo you wish to consider the already existent ' + \
                                      'offline stage datafiles?','no')
        if is_same_offstage:
            status = 1
            # If the offline stage subdirectory does not exist raise error
            if not os.path.exists(offline_stage_dir):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00009',location.filename,location.lineno+1,
                                                             problem_name,offline_stage_dir)
            # Remove all the existent directories and files except the existent offline
            # stage subdirectory and '.screen' file
            required_dirnames = [input_file_name + '.screen','Offline_Stage']
            rmUnrequiredDirs(problem_dir,required_dirnames)
            # Create post processing subdirectory
            makeDirectory(postprocess_dir,'overwrite')
            # Warn user to potential compatibility issues between the problem input data
            # file and the already existent offline stage data files
            userAction('\n\nWarning: Please make sure that the problem input data ' + \
                       'file is consistent with the already ' + '\n' + \
                       len('Warning: ')*' ' + 'existent offline stage data files ' + \
                       '(stored in Offline_Stage/) to avoid unexpected ' + '\n' + \
                       len('Warning: ')*' ' + 'errors or misleading conclusions.' + \
                       '\n\n' + \
                       'Press any key to continue or type \'exit\' to leave: ')
        else:
            # Ask user if the existent problem directory is to be overwritten
            is_overwrite = query_yn('\nDo you wish to overwrite the problem directory?')
            if is_overwrite:
                status = 2
                # Remove all the existent directories and files
                required_dirnames = [input_file_name + '.screen']
                rmUnrequiredDirs(problem_dir,required_dirnames)
                # Create problem directory and main subdirectories
                for dir in [offline_stage_dir,postprocess_dir]:
                    makeDirectory(dir,'overwrite')
            else:
                status = 3
        # Display information about the problem directory and status
        info.displayInfo('-1',problem_dir,status)
    # Return
    return [problem_name,problem_dir,offline_stage_dir,postprocess_dir,is_same_offstage,
            cluster_file_path,cit_file_path,hres_file_path]
