"""File and directory operations.

This module includes several functions to handle files and directories, as well
as the functions that process the problem input data file and set the problem
output directory.

Functions
---------
make_directory
    Create new directory.
rmunreqdirs
    Remove unrequired directories and files in target directory.
setinputdatafilepath
    Process input data file path.
setproblemdirs
    Set output directory structure and output files paths.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
import inspect
import ntpath
# Local
import ioput.info as info
import ioput.errors as errors
import ioput.ioutilities as ioutil
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira',]
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
    try:
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
    except OSError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayexception(location.filename, location.lineno + 1,
                                message)
# =============================================================================
# Remove all the unrequired directories and files in a target directory
def rmunreqdirs(target_dir, required_dirnames):
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
def setinputdatafilepath(path):
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
    input_file_dir = ntpath.dirname(input_file_path) + '/'
    input_file_name = ntpath.splitext(ntpath.basename(input_file_path))[-2]
    input_file_ext = ntpath.splitext(ntpath.basename(input_file_path))[-1]
    # Check if the input data file has the required '.dat' extension
    if input_file_ext != '.dat':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00010', location.filename, location.lineno + 1)
    # Check if the input data file name only contains numbers, letters or
    # underscores
    if not ioutil.checkvalidname(input_file_name):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00002', location.filename, location.lineno + 1)
    # Return
    return input_file_name, input_file_path, input_file_dir
#
#                                                             Results directory
# =============================================================================
def setproblemdirs(input_file_name, input_file_dir):
    """Set output directory structure and output files paths.

    *Output directory structure:*

    .. code-block :: text

       example.dat
       microstructure_regular_grid.rgmsh
       example/
            |-----example.screen
            |---- microstructure_regular_grid.rgmsh
            |---- example.hres
            |---- example.refm
            |---- example.adapt
            |---- Offline_Stage/
            |           |---- example.crve
            |           |---- example_clusters.vti
            |
            |---- Post_Process/
                        |---- example.pvd
                        |---- VTK/
                        |---- example.voxout


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

        * **example.refm**
            File where data associated with the homogeneous (fictitious)
            reference material is stored.

        * **example.adapt**
            File where the adaptive material phases' number of clusters and \
            adaptivity steps are stored (only if clustering adaptivity is \
            activated).

        * **Offline_Stage/**
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

        * **Post_Process/**
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

    Parameters
    ----------
    input_file_name : str
        Problem input data file name.
    input_file_dir : str
        Problem input data file directory path.

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
    hres_file_path : str
        Problem '.hres' output file path.
    refm_file_path : str
        Problem '.refm' output file path.
    adapt_file_path : str
        Problem '.adapt' output file path.
    """
    # Set problem name and output directory
    problem_name = input_file_name
    problem_dir = input_file_dir + problem_name + '/'
    # Set offline-stage and post-processing subdirectories
    offline_stage_dir = problem_dir + 'Offline_Stage' + '/'
    postprocess_dir = problem_dir + 'Post_Process' + '/'
    # Set '.screen' output file path (delete existing file)
    ioutil.screen_file_path = problem_dir + input_file_name + '.screen'
    if os.path.isfile(ioutil.screen_file_path):
        os.remove(ioutil.screen_file_path)
    # Set '.crve' output file path
    crve_file_path = offline_stage_dir + input_file_name + '.crve'
    # Set '.hres' output file path
    hres_file_path = problem_dir + input_file_name + '.hres'
    # Set '.refm' output file path
    refm_file_path = problem_dir + input_file_name + '.refm'
    # Set '.adapt' output file path
    adapt_file_path = problem_dir + input_file_name + '.adapt'
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
        is_same_offstage = \
            ioutil.query_yn('\nDo you wish to consider the already existent '
                            'offline-stage \'.crve\' data file?', 'no')
        if is_same_offstage:
            status = 1
            # Raise error if offline-stage subdirectory does not exist
            if not os.path.exists(offline_stage_dir):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00009', location.filename,
                                    location.lineno + 1, problem_name,
                                    offline_stage_dir)
            # Remove all the existent subdirectories and files except the
            # offline-stage subdirectory and the '.screen' output file
            required_dirnames = [input_file_name + '.screen', 'Offline_Stage']
            rmunreqdirs(problem_dir, required_dirnames)
            # Create post-processing subdirectory
            make_directory(postprocess_dir, 'overwrite')
            # Warn user to potential compatibility issues between the problem
            # input data file and the existent offline-stage '.crve' data file
            ioutil.useraction('\n\nWarning: Please make sure that the problem '
                              'input data file is consistent with the already'
                              '\n' + len('Warning: ')*' '
                              + 'existent offline-stage \'.crve\' data file '
                              '(stored in Offline_Stage/) to avoid ' + '\n'
                              + len('Warning: ')*' '
                              + 'unexpected errors or misleading conclusions.'
                              + '\n\n' + 'Press any key to continue or type '
                              '\'exit\' to leave: ')
        else:
            # Ask user if existent problem output directory should be
            # overwritten
            is_overwrite = ioutil.query_yn('\nDo you wish to overwrite the '
                                           'existing problem output '
                                           'directory?')
            if is_overwrite:
                status = 2
                # Remove all existent subdirectories and files except the
                # '.screen' output file
                required_dirnames = [input_file_name + '.screen']
                rmunreqdirs(problem_dir, required_dirnames)
                # Create problem output directory subdirectories
                for dir in [offline_stage_dir, postprocess_dir]:
                    make_directory(dir, 'overwrite')
            else:
                status = 3
        # Display information about the problem directory and status
        info.displayinfo('-1', problem_dir, status)
    # Return
    return problem_name, problem_dir, offline_stage_dir, postprocess_dir, \
           is_same_offstage, crve_file_path, hres_file_path, refm_file_path, \
           adapt_file_path
