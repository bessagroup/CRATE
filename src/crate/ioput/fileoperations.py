"""File and directory operations.

This module includes several functions to handle files and directories, as well
as the functions that process the problem input data file and set the problem
output directory.

Functions
---------
make_directory
    Create new directory.
remove_dirs
    Remove unrequired directories and files in target directory.
set_input_datafile_path
    Process input data file path.
set_problem_dirs
    Set output directory structure and output files paths.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
# Local
import ioput.info as info
import ioput.ioutilities as ioutil
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
#
#                                                          Directory operations
# =============================================================================
def make_directory(dir, option='no_overwrite'):
    """Create new directory.

    Parameters
    ----------
    dir : str
        Path of new directory.
    option : {'no_overwrite', 'overwrite'}, default='no_overwrite'
        Either to overwrite or not an existing path.
    """
    if option == 'no_overwrite':
        os.mkdir(dir)
    elif option == 'overwrite':
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            shutil.rmtree(dir)
            os.mkdir(dir)
    else:
        raise RuntimeError('Unknown option.')
# =============================================================================
def remove_dirs(target_dir, required_dirnames):
    """Remove unrequired directories and files in target directory.

    Parameters
    ----------
    target_dir : str
        Path of target directory.
    required_dirnames : list[str]
        Paths that are to be preserved in target directory.
    """
    dirnames = os.listdir(target_dir)
    for dir in dirnames:
        if dir not in required_dirnames:
            path = target_dir + dir
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
#
#                                                               Input data file
# =============================================================================
def set_input_datafile_path(path):
    """Process input data file path.

    Parameters
    ----------
    path : str
        Path of input data file.

    Returns
    -------
    input_file_name : str
        Input data file name.
    input_file_path : str
        Input data file path.
    input_file_dir : str
        Input data file directory path.
    """
    # Set input data file path, directory, name and extension
    input_file_path = os.path.abspath(path)
    input_file_dir = os.path.dirname(input_file_path) + '/'
    input_file_name = os.path.splitext(os.path.basename(input_file_path))[-2]
    input_file_ext = os.path.splitext(os.path.basename(input_file_path))[-1]
    # Check if the input data file has the required '.dat' extension
    if input_file_ext != '.dat':
        summary = 'Input data file extension'
        description = 'The input data file must have \'.dat\' extension.'
        info.displayinfo('4', summary, description)
    # Check if the input data file name only contains numbers, letters or
    # underscores
    if not ioutil.checkvalidname(input_file_name):
        summary = 'Input data file name'
        description = 'The input data file name can only contain letters, ' \
            + 'numbers or underscores.'
        info.displayinfo('4', summary, description)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return input_file_name, input_file_path, input_file_dir
#
#                                                             Results directory
# =============================================================================
def set_problem_dirs(input_file_name, input_file_dir,
                     is_minimize_output=False, is_null_stdout=False):
    """Set output directory structure.

    *Output directory structure:*

    .. code-block :: text

       example.dat
       microstructure_regular_grid.rgmsh
       example/
            |-----example.screen
            |---- microstructure_regular_grid.rgmsh
            |---- example.hres
            |---- example.efftan
            |---- example.refm
            |---- example.adapt
            |---- offline_stage/
            |           |---- example.crve
            |           |---- example_clusters.vti
            |
            |---- post_process/
                        |---- example.pvd
                        |---- VTK/
                        |---- example.voxout

    ----

    *Glossary: Files and directories*

    * **example.dat**
        Input data file.

    * **regular_grid_mesh.rgmsh**
        Spatial discretization file (regular grid of voxels, where each voxel \
        is associated with a given material phase ID).

    * **example/**
        Output directory.

        * **example.screen**
            File where all the data printed to the default standard output \
            device is stored.

        * **regular_grid_mesh.rgmsh**
            Copy of spatial discretization file.

        * **example.hres**
            File where the homogenized strain/stress results are stored.

        * **example.efftan**
            File where the CRVE effective material consistent tangent modulus
            is stored.

        * **example.refm**
            File where data associated with the homogeneous (fictitious)
            reference material is stored.

        * **example.adapt**
            File where the adaptive material phases' number of clusters and \
            adaptivity steps are stored (only if clustering adaptivity is \
            activated).

        * **offline_stage/**
            Clustering-based model reduction (offline-stage) directory.

            * **example.crve**
                File that contains the CRVE instance obtained from the \
                clustering-based model reduction (offline-stage), accounting \
                for the base clustering and the cluster interaction tensors.

            * **example_clusters.vti**
                VTK XML file containing spatial (voxelwise) data \
                characterizing the CRVE after the clustering-based model \
                reduction (offline-stage), namely the material phases and the \
                base material clusters.

        * **post_processing/**
            Post-processing directory.

            * **example.pvd**
                VTK XML collection file containing the paths to all the VTK \
                snapshots taken during the problem solution.

            * **VTK/**
                VTK XML files directory containing all the snapshots taken \
                during the problem solution.

            * **example.voxout**
                File where material-related quantities defined at the voxel \
                level are stored.

    ----

    Parameters
    ----------
    input_file_name : str
        Problem input data file name.
    input_file_dir : str
        Problem input data file directory path.
    is_minimize_output : bool, default=False
        Output minimization flag.
    is_null_stdout : bool, default=False
        Suppress execution output to stdout.

    Returns
    -------
    problem_name : str
        Problem name.
    problem_dir : str
        Problem output directory path.
    offline_stage_dir : str
        Problem output offline-stage subdirectory path.
    postprocess_dir : str
        Problem output post-processing subdirectory path.
    is_same_offstage : bool
        `True` if an already existing offline-stage data file is to be
        considered, `False` otherwise.
    crve_file_path : str
        Problem '.crve' output file path.
    """
    # Get display features
    indent = ioutil.setdisplayfeatures()[2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set problem name and output directory
    problem_name = input_file_name
    problem_dir = input_file_dir + problem_name + '/'
    # Set offline-stage and post-processing subdirectories
    offline_stage_dir = problem_dir + 'offline_stage' + '/'
    postprocess_dir = problem_dir + 'post_processing' + '/'
    # Set '.screen' output file path (delete existing file)
    ioutil.screen_file_path = problem_dir + input_file_name + '.screen'
    if os.path.isfile(ioutil.screen_file_path):
        os.remove(ioutil.screen_file_path)
    # Set '.crve' output file path
    crve_file_path = offline_stage_dir + input_file_name + '.crve'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Disable user prompts
    is_user_prompts = True
    if is_null_stdout or is_minimize_output:
        is_user_prompts = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Check if the problem output directory exists
    if not os.path.exists(problem_dir):
        status = 0
        # Create problem output directory and subdirectories
        make_directory(problem_dir)
        for dir in [offline_stage_dir, postprocess_dir]:
            make_directory(dir)
        # No previously computed offline-stage data available
        is_same_offstage = False
    else:
        ioutil.print2('\nWarning: The problem output directory for the '
                      'specified input data file already exists.')
        # Ask user if the purpose is to consider previously computed
        # offline-stage data ('.crve' output file)
        if is_user_prompts:
            if os.path.exists(crve_file_path):
                is_same_offstage = ioutil.query_yn(
                    '\nDo you wish to consider the already existent '
                    'offline-stage \'.crve\' data file?', 'no')
            else:
                is_same_offstage = False
        else:
            # Consider existent offline-stage data by default
            if os.path.exists(offline_stage_dir) \
                    and os.path.exists(crve_file_path):
                is_same_offstage = True
            else:
                is_same_offstage = False
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if is_same_offstage:
            status = 1
            # Raise error if offline-stage subdirectory does not exist
            if not os.path.exists(offline_stage_dir):
                summary = 'Missing offline-stage subdirectory'
                description = 'The subdirectory with the previously computed '\
                    + 'offline-stage data files could not be found:' + '\n\n' \
                    + indent + '{}'
                info.displayinfo('4', summary, description, offline_stage_dir)
            # Remove all the existent subdirectories and files except the
            # offline-stage subdirectory and the '.screen' output file
            required_dirnames = [input_file_name + '.screen', 'offline_stage']
            remove_dirs(problem_dir, required_dirnames)
            # Create post-processing subdirectory
            make_directory(postprocess_dir, 'overwrite')
            # Warn user to potential compatibility issues between the problem
            # input data file and the existent offline-stage '.crve' data file
            if is_user_prompts:
                ioutil.useraction(
                    '\n\nWarning: Please make sure that the problem '
                    'input data file is consistent with the already'
                    '\n' + len('Warning: ')*' '
                    + 'existent offline-stage \'.crve\' data file '
                    '(stored in offline_stage/) to avoid ' + '\n'
                    + len('Warning: ')*' '
                    + 'unexpected errors or misleading conclusions.'
                    + '\n\n' + 'Press any key to continue or type '
                    '\'exit\' to leave: ')
        else:
            # Ask user if existent problem output directory should be
            # overwritten
            if is_user_prompts:
                is_overwrite = ioutil.query_yn('\nDo you wish to overwrite '
                                               'the existing problem output '
                                               'directory?')
            else:
                # Overwrite by default
                is_overwrite = True
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if is_overwrite:
                status = 2
                # Remove all existent subdirectories and files except the
                # '.screen' output file
                required_dirnames = [input_file_name + '.screen']
                remove_dirs(problem_dir, required_dirnames)
                # Create problem output directory subdirectories
                for dir in [offline_stage_dir, postprocess_dir]:
                    make_directory(dir, 'overwrite')
            else:
                status = 3
        # Display information about the problem directory and status
        info.displayinfo('-1', problem_dir, status)
    # Return
    return problem_name, problem_dir, offline_stage_dir, postprocess_dir, \
        is_same_offstage, crve_file_path
