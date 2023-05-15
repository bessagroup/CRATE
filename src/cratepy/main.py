"""CRATE (Clustering-based Nonlinear Analysis of Materials).

CRATE was originally developed by Bernardo P. Ferreira in the context of his
PhD Thesis (see Ferreira (2022) [#]_). CRATE is devised to aid the design and
development of new materials by performing multi-scale nonlinear analyses of
heterogeneous materials through a suitable coupling between first-order
computational homogenization and clustering-based reduced-order modeling.

.. [#] Ferreira, B.P. (2022). *Towards Data-driven Multi-scale
       Optimization of Thermoplastic Blends: Microstructural
       Generation, Constitutive Development and Clustering-based
       Reduced-Order Modeling.* PhD Thesis, University of Porto
       (see `here <http://dx.doi.org/10.13140/RG.2.2.33940.17289>`_)

Functions
---------
crate_simulation
    Perform CRATE simulation.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pickle
import time
import copy
# Third-party
import numpy as np
# Local
import ioput.info as info
import ioput.readinputdata as rid
import ioput.fileoperations as filop
import ioput.packager as packager
from clustering.crve import CRVE
from clustering.clusteringdata import set_clustering_data
from online.crom.asca import ASCA
from ioput.miscoutputfiles.vtkoutput import VTKOutput
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
def crate_simulation(arg_input_file_path, arg_discret_file_dir=None,
                     is_null_stdout=False):
    """Perform CRATE simulation.

    Parameters
    ----------
    arg_input_file_path : str
        Input data file path provided as input.
    arg_discret_file_dir : str, default=None
        Spatial discretization file directory path provided as input.
    is_null_stdout : bool, default=False
        Suppress execution output to stdout.
    """
    # Suppress execution output to stdout
    if is_null_stdout:
        null_file = open(os.devnull, 'w')
        sys.stdout = null_file
    #
    #                                             Read user input data file and
    #                                 create problem output directory structure
    # =========================================================================
    # Check input data file path
    if not os.path.isfile(str(arg_input_file_path)):
        summary = 'Missing input data file'
        description = 'The input data file could not be found.'
        info.displayinfo('4', summary, description)
    # Check spatial discretization file directory
    discret_file_dir = None
    if arg_discret_file_dir is not None and \
            not os.path.exists(str(arg_discret_file_dir)):
        summary = 'Missing spatial discretization file directory'
        description = 'The spatial discretization file directory could ' \
            + 'not be found.'
        info.displayinfo('4', summary, description)
    else:
        discret_file_dir = os.path.normpath(str(arg_discret_file_dir)) + '/'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process input data file path
    input_file_name, input_file_path, input_file_dir = \
        filop.set_input_datafile_path(arg_input_file_path)
    # Check if data-driven simulation mode
    is_minimize_output = rid.read_output_minimization_option(input_file_path)
    # Set output directory structure and output files paths
    problem_name, problem_dir, offline_stage_dir, postprocess_dir, \
        is_same_offstage, crve_file_path = filop.set_problem_dirs(
            input_file_name, input_file_dir, is_minimize_output,
            is_null_stdout)
    # Store problem directories and files paths
    dirs_dict = packager.store_paths_data(
        input_file_name, input_file_path, input_file_dir, problem_name,
        problem_dir, offline_stage_dir, postprocess_dir, crve_file_path,
        discret_file_dir=discret_file_dir)
    #
    #                                                             Start program
    # =========================================================================
    # Get current time and date
    start_date = time.strftime("%d/%b/%Y")
    start_time = time.strftime("%Hh%Mm%Ss")
    start_time_s = time.time()
    phase_names = ['']
    phase_times = np.zeros((1, 2))
    phase_names[0] = 'Total'
    phase_times[0, :] = [start_time_s, 0.0]
    # Display starting program header
    info.displayinfo('0', problem_name, start_time, start_date)
    #
    #                                                 Read user input data file
    # =========================================================================
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Read input data file')
    phase_init_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Open user input data file
    input_file = open(input_file_path, 'r')
    # Read input data file and store data in convenient containers
    info.displayinfo('5', 'Reading the input data file...')
    problem_dict, mat_dict, macload_dict, rg_dict, clst_dict, scs_dict, \
        algpar_dict, vtk_dict, output_dict, material_state = \
        rid.read_input_data_file(input_file, dirs_dict, is_minimize_output)
    # Close user input data file
    input_file.close()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Read input data')
    phase_times = np.append(
        phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Read input data file',
                     phase_times[phase_times.shape[0] - 1, 1]
                     - phase_times[phase_times.shape[0] - 1, 0])
    #
    #           Offline-stage - Step 1: Compute clustering features data matrix
    # =========================================================================
    # Initialize offline stage post-processing time
    ofs_post_process_time = 0.0
    # Compute clustering features data matrix
    if not is_same_offstage:
        # Display starting phase information and set phase initial time
        info.displayinfo('2', 'Compute clustering features data matrix')
        phase_init_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set DNS homogenization-based multi-scale method and associated
        # parameters
        dns_method_id = clst_dict['clustering_solution_method']
        if dns_method_id == 1:
            dns_method = 'fft_basic'
            dns_method_data = None
        else:
            raise RuntimeError('Unknown DNS solution method.')
        # Compute the physical-based data required to perform the RVE
        # clustering-based domain decomposition
        clustering_data, rve_elastic_database = set_clustering_data(
            problem_dict['strain_formulation'], problem_dict['problem_type'],
            rg_dict['rve_dims'], rg_dict['n_voxels_dims'],
            rg_dict['regular_grid'], mat_dict['material_phases'],
            mat_dict['material_phases_properties'],
            dns_method, dns_method_data, clst_dict['standardization_method'],
            clst_dict['base_clustering_scheme'],
            clst_dict['adaptive_clustering_scheme'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set phase ending time and display finishing phase information
        phase_end_time = time.time()
        phase_names.append('Compute cluster analysis data matrix')
        phase_times = np.append(
            phase_times, [[phase_init_time, phase_end_time]], axis=0)
        info.displayinfo('3', 'Compute cluster analysis data matrix',
                         phase_times[phase_times.shape[0] - 1, 1]
                         - phase_times[phase_times.shape[0] - 1, 0])
    #
    #                     Offline-stage - Steps 2 & 3: Generate Cluster-Reduced
    #                                      Representative Volume Element (CRVE)
    # =========================================================================
    if is_same_offstage:
        # Display starting phase information and set phase initial time
        info.displayinfo('2', 'Import offline state CRVE instance')
        phase_init_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get CRVE file path
        crve_file_path = dirs_dict['crve_file_path']
        # Load CRVE instance from file
        info.displayinfo('5', 'Importing Cluster-Reduced Representative '
                         'Volume Element (.crve file)...')
        with open(crve_file_path, 'rb') as crve_file:
            crve = pickle.load(crve_file)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update CRVE material state clusters labels and volume fraction
        material_state.set_phase_clusters(crve.get_phase_clusters(),
                                          crve.get_clusters_vf())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update clustering dictionary
        clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
        clst_dict['phase_n_clusters'] = crve.get_phase_n_clusters()
        clst_dict['phase_clusters'] = copy.deepcopy(crve.get_phase_clusters())
        clst_dict['clusters_vf'] = copy.deepcopy(crve.get_clusters_vf())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update CRVE clustering adaptivity attributes
        if 'adaptive' in crve.get_clustering_type().values():
            crve.update_adaptive_parameters(
                copy.deepcopy(clst_dict['adaptive_clustering_scheme']),
                copy.deepcopy(clst_dict['adapt_criterion_data']),
                copy.deepcopy(clst_dict['adaptivity_type']),
                copy.deepcopy(clst_dict['adaptivity_control_feature']))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set phase ending time and display finishing phase information
        phase_end_time = time.time()
        phase_names.append('Import offline state CRVE instance')
        phase_times = np.append(
            phase_times, [[phase_init_time, phase_end_time]], axis=0)
        info.displayinfo('3', 'Import offline state CRVE instance',
                         phase_times[phase_times.shape[0] - 1, 1]
                         - phase_times[phase_times.shape[0] - 1, 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        # Display starting phase information and set phase initial time
        info.displayinfo('2', 'Perform RVE cluster analysis')
        phase_init_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instatiate Cluster-Reduced Representative Volume Element (CRVE)
        crve = CRVE(
            rg_dict['rve_dims'], rg_dict['regular_grid'],
            mat_dict['material_phases'], problem_dict['strain_formulation'],
            problem_dict['problem_type'],
            clustering_data.get_global_data_matrix(),
            clst_dict['clustering_type'], clst_dict['phase_n_clusters'],
            clst_dict['base_clustering_scheme'],
            rve_elastic_database.get_eff_isotropic_elastic_constants(),
            clst_dict['adaptive_clustering_scheme'],
            clst_dict['adapt_criterion_data'], clst_dict['adaptivity_type'],
            clst_dict['adaptivity_control_feature'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute Cluster-Reduced Representative Volume Element (CRVE)
        crve.perform_crve_base_clustering()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update CRVE material state clusters labels and volume fraction
        material_state.set_phase_clusters(crve.get_phase_clusters(),
                                          crve.get_clusters_vf())
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update clustering dictionary
        clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
        clst_dict['phase_clusters'] = crve.get_phase_clusters()
        clst_dict['clusters_vf'] = crve.get_clusters_vf()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Write clustering VTK file
        if vtk_dict['is_vtk_output']:
            # Set post-processing procedure initial time
            procedure_init_time = time.time()
            # Set VTK output files parameters
            vtk_byte_order = vtk_dict['vtk_byte_order']
            vtk_format = vtk_dict['vtk_format']
            vtk_precision = vtk_dict['vtk_precision']
            # Instantiante VTK output
            vtk_output = VTKOutput(
                type='ImageData', version='1.0', byte_order=vtk_byte_order,
                format=vtk_format, precision=vtk_precision,
                header_type='UInt64', base_name=input_file_name,
                vtk_dir=offline_stage_dir)
            # Write clustering VTK file
            info.displayinfo('5', 'Writing clustering VTK file...')
            vtk_output.write_vtk_file_clustering(crve=crve)
            # Increment post-processing time
            ofs_post_process_time += time.time() - procedure_init_time
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set phase ending time and display finishing phase information
        phase_end_time = time.time() - ofs_post_process_time
        phase_names.append('Perform RVE cluster analysis')
        phase_times = np.append(
            phase_times, [[phase_init_time, phase_end_time]], axis=0)
        info.displayinfo('3', 'Perform RVE cluster analysis',
                         phase_times[phase_times.shape[0] - 1, 1]
                         - phase_times[phase_times.shape[0] - 1, 0])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Display starting phase information and set phase initial time
        info.displayinfo('2', 'Compute CRVE cluster interaction tensors')
        phase_init_time = time.time()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute CRVE's cluster interaction tensors
        crve.compute_cit()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set phase ending time and display finishing phase information
        phase_end_time = time.time()
        phase_names.append('Compute cluster interaction tensors')
        phase_times = np.append(
            phase_times, [[phase_init_time, phase_end_time]], axis=0)
        info.displayinfo('3', 'Compute cluster interaction tensors',
                         phase_times[phase_times.shape[0] - 1, 1]
                         - phase_times[phase_times.shape[0] - 1, 0])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Dump CRVE into file
        crve_file_path = dirs_dict['crve_file_path']
        crve.save_crve_file(crve, crve_file_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output minimization
    if is_minimize_output:
        # Get offline-stage directory path
        offline_stage_dir = dirs_dict['offline_stage_dir']
        # Get path of spatial discretization file (within problem directory)
        discret_file_path = dirs_dict['discret_file_path']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filop.remove_dirs(problem_dir, [offline_stage_dir, discret_file_path])
    #
    #                   Online-stage: Solve CRVE mechanical equilibrium problem
    # =========================================================================
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Solve reduced microscale equilibrium problem')
    phase_init_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Adaptive Self-Consistent Clustering Analysis (ASCA)
    asca = ASCA(problem_dict['strain_formulation'],
                problem_dict['problem_type'],
                self_consistent_scheme=scs_dict['self_consistent_scheme'],
                scs_parameters=scs_dict['scs_parameters'],
                scs_max_n_iterations=scs_dict['scs_max_n_iterations'],
                scs_conv_tol=scs_dict['scs_conv_tol'],
                max_n_iterations=algpar_dict['max_n_iterations'],
                conv_tol=algpar_dict['conv_tol'],
                max_subinc_level=algpar_dict['max_subinc_level'],
                max_cinc_cuts=algpar_dict['max_cinc_cuts'])
    # Solve clustering-based reduced-order equilibrium problem
    asca.solve_equilibrium_problem(
        crve, material_state, macload_dict['mac_load'],
        macload_dict['mac_load_presctype'], macload_dict['mac_load_increm'],
        dirs_dict['problem_dir'], problem_name=dirs_dict['problem_name'],
        clust_adapt_freq=clst_dict['clust_adapt_freq'],
        is_solution_rewinding=macload_dict['is_solution_rewinding'],
        rewind_state_criterion=macload_dict['rewind_state_criterion'],
        rewinding_criterion=macload_dict['rewinding_criterion'],
        max_n_rewinds=macload_dict['max_n_rewinds'],
        is_clust_adapt_output=clst_dict['is_clust_adapt_output'],
        is_ref_material_output=output_dict['is_ref_material_output'],
        is_vtk_output=vtk_dict['is_vtk_output'], vtk_data=vtk_dict,
        is_voxels_output=output_dict['is_voxels_output'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = phase_init_time + asca.get_time_profile()[1]
    phase_names.append('Solve reduced microscale equilibrium problem')
    phase_times = np.append(
        phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Solve reduced microscale equilibrium problem',
                     phase_times[phase_times.shape[0] - 1, 1]
                     - phase_times[phase_times.shape[0] - 1, 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Store CRVE final clustering state into file
    if clst_dict['is_store_final_clustering']:
        # Reset CRVE adaptive progress parameters and set base clustering
        crve.reset_adaptive_parameters()
        # Dump CRVE into file
        crve_file_path = dirs_dict['crve_file_path']
        crve.save_crve_file(crve, crve_file_path)
    #
    #                       Compute post-processing operations accumulated time
    # =========================================================================
    # Get online-stage post-processing time
    ons_post_process_time = asca.get_time_profile()[2]
    # Set (fictitious) phase initial time
    phase_init_time = phase_times[-1, 1]
    # Set (fictitious) phase ending time
    phase_end_time = phase_init_time + ofs_post_process_time \
        + ons_post_process_time
    phase_names.append('Accumulated post-processing operations')
    phase_times = np.append(
        phase_times, [[phase_init_time, phase_end_time]], axis=0)
    #
    #                                                               End program
    # =========================================================================
    # Get current time and date
    end_date = time.strftime("%d/%b/%Y")
    end_time = time.strftime("%Hh%Mm%Ss")
    end_time_s = time.time()
    phase_times[0, 1] = end_time_s
    # Display ending program message
    info.displayinfo('1', end_time, end_date, problem_name, phase_names,
                     phase_times)
# =============================================================================
# A CRATE simulation can be performed directly by executing this script with
# the following command
#
# python main.py < input_data_file_path > [< discret_file_dir >]
#
# where input_data_file_path is the input data file path (mandatory) and
# discret_file_dir is the spatial discretization file directory path (optional)
if __name__ == '__main__':
    # Set input data file path
    if len(sys.argv[1:]) == 0:
        summary = 'Missing input data file'
        description = 'The input data file was not provided.'
        info.displayinfo('4', summary, description)
    else:
        input_file_path = str(sys.argv[1])
    # Set spatial discretization file directory
    discret_file_dir = None
    if len(sys.argv[1:]) == 2:
        discret_file_dir = os.path.normpath(str(sys.argv[2])) + '/'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform CRATE simulation
    crate_simulation(input_file_path, arg_discret_file_dir=discret_file_dir,
                     is_null_stdout=False)
