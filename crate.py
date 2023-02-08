#
# CRATE - Clustering-based Nonlinear Analysis of Materials
# ==========================================================================================
# Summary:
# Program suitable to perform accurate and efficient multi-scale nonlinear analyses of
# materials mainly relying on a clustering-based reduced order modeling approach coupled
# with computational homogenization.
# ------------------------------------------------------------------------------------------
# Description:
# CRATE has been designed with the main purpose of performing accurate and efficient
# multi-scale analyses of heterogeneous materials, a crucial task in the development of
# new materials with innovative and enhanced properties. This goal is achieved through the
# coupling between first-order computational homogenization and a clustering-based reduced
# order modeling approach, allowing the solution of a given microscale equilibrium problem
# formulated in the standard way: define the representative volume element (RVE) of the
# heterogeneous material under analyses, enforce first-order macroscale strain and/or stress
# loading constraints, solve the microscale equilibrium problem and compute the
# heterogeneous material first-order homogenized response. The clustering-based reduced
# order modeling approach comes into play by compressing the RVE into a cluster-reduced
# representative volume element (CRVE), aiming to reduce the overall computational cost of
# the analysis at the expense of an acceptable decrease of accuracy.
# ------------------------------------------------------------------------------------------
# Author(s):
# This program initial version was designed and fully coded by Bernardo P. Ferreira
# (bpferreira@fe.up.pt) and developed in colaboration with Dr. Miguel A. Bessa and
# Dr. Francisco M. Andrade Pires.
# ------------------------------------------------------------------------------------------
# Licensing and Copyrights:
# ...
# ------------------------------------------------------------------------------------------
#
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Operating system related functions
import os
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
# Python object serialization
import pickle
# Date and time
import time
# Inspect file name and line
import inspect
# Shallow and deep copy operations
import copy
# Display messages
import ioput.info as info
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
# Read user input data file
import ioput.readinputdata as rid
# Manage files and directories
import ioput.fileoperations as filop
# Packager
import ioput.packager as packager
# Clustering data computation
from clustering.clusteringdata import set_clustering_data
# Clustering-based Reduced-Order Models
from online.crom.asca import ASCA
# VTK output
from ioput.vtkoutput import VTKOutput
# CRVE generation
from clustering.crve import CRVE
#
#                             Check user input data file and create problem main directories
# ==========================================================================================
# Check if the input data file path was provided
if len(sys.argv[1:]) == 0:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayerror('E00001', location.filename, location.lineno + 1)
elif not os.path.isfile(str(sys.argv[1])):
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayerror('E00001', location.filename, location.lineno + 1)
# Set input data file name, path and directory
input_file_name, input_file_path, input_file_dir = \
    filop.set_input_datafile_path(str(sys.argv[1]))
# Set problem name, directory and main subdirectories
problem_name, problem_dir, offline_stage_dir, postprocess_dir, is_same_offstage, \
    crve_file_path, hres_file_path, refm_file_path, adapt_file_path = \
        filop.set_problem_dirs(input_file_name, input_file_dir)
# Package data associated to directories and paths
dirs_dict = packager.store_paths_data(input_file_name, input_file_path, input_file_dir,
                                  problem_name, problem_dir, offline_stage_dir,
                                  postprocess_dir, crve_file_path, hres_file_path,
                                  refm_file_path, adapt_file_path)
# Open user input data file
try:
    input_file = open(input_file_path, 'r')
except FileNotFoundError as message:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayexception(location.filename, location.lineno + 1, message)
#
#                                                                              Start program
# ==========================================================================================
# Get current time and date
start_date = time.strftime("%d/%b/%Y")
start_time = time.strftime("%Hh%Mm%Ss")
start_time_s = time.time()
phase_names = ['']
phase_times = np.zeros((1, 2))
phase_names[0] = 'Total'
phase_times[0,:] = [start_time_s, 0.0]
# Display starting program header
info.displayinfo('0', problem_name, start_time, start_date)
#
#                                                                  Read user input data file
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayinfo('2', 'Read input data file')
phase_init_time = time.time()
# Open user input data file
try:
    input_file = open(input_file_path, 'r')
except FileNotFoundError as message:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayexception(location.filename, location.lineno + 1, message)
# Read input data according to analysis type
info.displayinfo('5', 'Reading the input data file...')
problem_dict, mat_dict, macload_dict, rg_dict, clst_dict, scs_dict, algpar_dict, vtk_dict, \
    output_dict, material_state = rid.read_input_data_file(input_file, dirs_dict)
# Close user input data file
input_file.close()
# Save copy of clustering dictionary for compatibility check procedure (loading previously
# computed offline stage)
if is_same_offstage:
    clst_dict_read = copy.deepcopy(clst_dict)
# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Read input data')
phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
info.displayinfo('3', 'Read input data file',
                 phase_times[phase_times.shape[0] - 1, 1] -
                 phase_times[phase_times.shape[0] - 1, 0])
#
#                                        Offline stage: Compute cluster analysis data matrix
# ==========================================================================================
# Initialize offline stage post-processing time
ofs_post_process_time = 0.0
# Compute clustering analysis data matrix
if not is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Compute cluster analysis data matrix')
    phase_init_time = time.time()
    # Set DNS homogenization-based multi-scale method and associated parameters
    dns_method_id = clst_dict['clustering_solution_method']
    if dns_method_id == 1:
        dns_method = 'fft_basic'
        dns_method_data = None
    elif dns_method_id == 2:
        dns_method = 'fem_links'
        dns_method_data = clst_dict['links_data']
    else:
        raise RuntimeError('Unknown DNS solution method.')
    # Compute the physical-based data required to perform the RVE cluster analysis
    clustering_data, rve_elastic_database = set_clustering_data(
        problem_dict['strain_formulation'], problem_dict['problem_type'],
            rg_dict['rve_dims'], rg_dict['n_voxels_dims'], rg_dict['regular_grid'],
            mat_dict['material_phases'], mat_dict['material_phases_properties'], dns_method,
            dns_method_data, clst_dict['standardization_method'],
            clst_dict['base_clustering_scheme'], clst_dict['adaptive_clustering_scheme'])
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Compute cluster analysis data matrix')
    phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Compute cluster analysis data matrix',
                     phase_times[phase_times.shape[0] - 1, 1] -
                     phase_times[phase_times.shape[0] - 1, 0])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert external sources constitutive models to CRATE corresponding model whenever
# possible
material_state.constitutive_source_conversion()
#
#               Offline stage: Generate Cluster-Reduced Representative Volume Element (CRVE)
# ==========================================================================================
if is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Import offline state CRVE instance')
    phase_init_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get CRVE file path
    crve_file_path = dirs_dict['crve_file_path']
    # Load CRVE instance from file
    info.displayinfo('5', 'Importing Cluster-Reduced Representative Volume Element ' +
                          '(.crve file)...')
    with open(crve_file_path, 'rb') as crve_file:
        crve = pickle.load(crve_file)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update CRVE material state clusters labels and volume fraction
    material_state.set_phase_clusters(crve.get_phase_clusters(), crve.get_clusters_vf())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update clustering dictionary
    clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
    clst_dict['phase_n_clusters'] = crve.get_phase_n_clusters()
    clst_dict['phase_clusters'] = copy.deepcopy(crve.get_phase_clusters())
    clst_dict['clusters_vf'] = copy.deepcopy(crve.get_clusters_vf())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update CRVE clustering adaptivity attributes
    if 'adaptive' in crve.get_clustering_type().values():
        crve.update_adaptive_parameters(
            copy.deepcopy(clst_dict['adaptive_clustering_scheme']),
            copy.deepcopy(clst_dict['adapt_criterion_data']),
            copy.deepcopy(clst_dict['adaptivity_type']),
            copy.deepcopy(clst_dict['adaptivity_control_feature']))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Import offline state CRVE instance')
    phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Import offline state CRVE instance',
                     phase_times[phase_times.shape[0] - 1, 1] -
                     phase_times[phase_times.shape[0] - 1, 0])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
else:
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Perform RVE cluster analysis')
    phase_init_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instatiate Cluster-Reduced Representative Volume Element (CRVE)
    crve = CRVE(rg_dict['rve_dims'],
                rg_dict['regular_grid'],
                mat_dict['material_phases'],
                problem_dict['strain_formulation'],
                problem_dict['problem_type'],
                clustering_data.get_global_data_matrix(),
                clst_dict['clustering_type'],
                clst_dict['phase_n_clusters'],
                clst_dict['base_clustering_scheme'],
                rve_elastic_database.get_eff_isotropic_elastic_constants(),
                clst_dict['adaptive_clustering_scheme'],
                clst_dict['adapt_criterion_data'],
                clst_dict['adaptivity_type'],
                clst_dict['adaptivity_control_feature'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Cluster-Reduced Representative Volume Element (CRVE)
    crve.perform_crve_base_clustering()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update CRVE material state clusters labels and volume fraction
    material_state.set_phase_clusters(crve.get_phase_clusters(), crve.get_clusters_vf())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update clustering dictionary
    clst_dict['voxels_clusters'] = crve.get_voxels_clusters()
    clst_dict['phase_clusters'] = crve.get_phase_clusters()
    clst_dict['clusters_vf'] = crve.get_clusters_vf()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write clustering VTK file
    if vtk_dict['is_vtk_output']:
        # Set post-processing procedure initial time
        procedure_init_time = time.time()
        # Set VTK output files parameters
        vtk_byte_order = vtk_dict['vtk_byte_order']
        vtk_format = vtk_dict['vtk_format']
        vtk_precision = vtk_dict['vtk_precision']
        vtk_vars = vtk_dict['vtk_vars']
        vtk_inc_div = vtk_dict['vtk_inc_div']
        # Instantiante VTK output
        vtk_output = VTKOutput(type='ImageData', version='1.0', byte_order=vtk_byte_order,
                               format=vtk_format, precision=vtk_precision,
                               header_type='UInt64', base_name=input_file_name,
                               vtk_dir=offline_stage_dir)
        # Write clustering VTK file
        info.displayinfo('5', 'Writing clustering VTK file...')
        vtk_output.write_vtk_file_clustering(crve=crve)
        # Increment post-processing time
        ofs_post_process_time += time.time() - procedure_init_time
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time() - ofs_post_process_time
    phase_names.append('Perform RVE cluster analysis')
    phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Perform RVE cluster analysis',
                     phase_times[phase_times.shape[0] - 1, 1] -
                     phase_times[phase_times.shape[0] - 1, 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Compute CRVE cluster interaction tensors')
    phase_init_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute CRVE's cluster interaction tensors
    crve.compute_cit()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Compute cluster interaction tensors')
    phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Compute cluster interaction tensors',
                     phase_times[phase_times.shape[0] - 1, 1] -
                     phase_times[phase_times.shape[0] - 1, 0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dump CRVE into file
    crve_file_path = dirs_dict['crve_file_path']
    crve.save_crve_file(crve, crve_file_path)
#
#                     Online stage: Solve clustering-based reduced order equilibrium problem
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayinfo('2', 'Solve reduced microscale equilibrium problem')
phase_init_time = time.time()
# Adaptive Self-Consistent Clustering Analysis (ASCA)
asca = ASCA(problem_dict['strain_formulation'],
            problem_dict['problem_type'],
            self_consistent_scheme=scs_dict['self_consistent_scheme'],
            scs_init_properties=crve.get_eff_isotropic_elastic_constants(),
            scs_max_n_iterations=scs_dict['scs_max_n_iterations'],
            scs_conv_tol=scs_dict['scs_conv_tol'],
            max_n_iterations=algpar_dict['max_n_iterations'],
            conv_tol=algpar_dict['conv_tol'],
            max_subinc_level=algpar_dict['max_subinc_level'],
            max_cinc_cuts=algpar_dict['max_cinc_cuts'])
# Solve clustering-based reduced-order equilibrium problem
asca.solve_equilibrium_problem(
    crve, material_state, macload_dict['mac_load'], macload_dict['mac_load_presctype'],
        macload_dict['mac_load_increm'], dirs_dict['problem_dir'],
        problem_name=dirs_dict['problem_name'],
        clust_adapt_freq=clst_dict['clust_adapt_freq'],
        is_solution_rewinding=macload_dict['is_solution_rewinding'],
        rewind_state_criterion=macload_dict['rewind_state_criterion'],
        rewinding_criterion=macload_dict['rewinding_criterion'],
        max_n_rewinds=macload_dict['max_n_rewinds'],
        is_clust_adapt_output=clst_dict['is_clust_adapt_output'],
        is_ref_material_output=output_dict['is_ref_material_output'],
        is_vtk_output=vtk_dict['is_vtk_output'], vtk_data=vtk_dict,
        is_voxels_output=output_dict['is_voxels_output'])
# Set phase ending time and display finishing phase information
phase_end_time = phase_init_time + asca.get_time_profile()[1]
phase_names.append('Solve reduced microscale equilibrium problem')
phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
info.displayinfo('3', 'Solve reduced microscale equilibrium problem',
                 phase_times[phase_times.shape[0] - 1, 1] -
                 phase_times[phase_times.shape[0] - 1, 0])
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Store CRVE final clustering state into file
if clst_dict['is_store_final_clustering']:
    # Reset CRVE adaptive progress parameters and set base clustering
    crve.reset_adaptive_parameters()
    # Dump CRVE into file
    crve_file_path = dirs_dict['crve_file_path']
    crve.save_crve_file(crve, crve_file_path)
#
#                                                Post-processing operations accumulated time
# ==========================================================================================
# Get online-stage post-processing time
ons_post_process_time = asca.get_time_profile()[2]
# Set (fictitious) phase initial time
phase_init_time = phase_times[-1, 1]
# Set (fictitious) phase ending time
phase_end_time = phase_init_time + ofs_post_process_time + ons_post_process_time
phase_names.append('Accumulated post-processing operations')
phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
#
#                                                                                End program
# ==========================================================================================
# Get current time and date
end_date = time.strftime("%d/%b/%Y")
end_time = time.strftime("%Hh%Mm%Ss")
end_time_s = time.time()
phase_times[0, 1] = end_time_s
# Display ending program message
info.displayinfo('1', end_time, end_date, problem_name, phase_names, phase_times)
