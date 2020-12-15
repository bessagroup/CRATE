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
# This program initial version was coded by Bernardo P. Ferreira (bpferreira@fe.up.pt,
# CM2S research group, Department of Mechanical Engineering, Faculty of Engineering,
# University of Porto) and developed in colaboration with Dr. Miguel A. Bessa
# (m.a.bessa@tudelft.nl, Faculty of Mechanical, Maritime and Materials Engineering,
# Delft University of Technology) and Dr. Francisco M. Andrade Pires (fpires@fe.up.pt,
# CM2S research group, Department of Mechanical Engineering, Faculty of Engineering,
# University of Porto).
# ------------------------------------------------------------------------------------------
# Licensing and Copyrights:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
#
# Release v0.1.0 - Bernardo P. Ferreira (June 2020)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   > Initial coding.
#   > Implementation of the FFT-based homogenization basic scheme proposed by H. Moulinec
#     and P. Suquet (1994).
#   > Implementation of the Self-Consistent Clustering Analysis (SCA) proposed by Z. Liu,
#     M. A. Bessa and W. K. Liu (2016)
#   > Implementation of suitable interfaces with the multi-scale finite element code Links
#     (CM2S research group, Faculty of Engineering, University of Porto)
#   > All the details about this release can be found in the PhD seminar of Bernardo P.
#     Ferreira ("Accurate and efficient multi-scale analyses of nonlinear heterogeneous
#     materials based on clustering-based reduced order models", University of Porto, 2020)
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
# Clustering quantities computation
import clustering.clusteringdata as clstdata
# Online stage
import online.sca as sca
# VTK output
import ioput.vtkoutput as vtkoutput
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
    filop.setinputdatafilepath(str(sys.argv[1]))
# Set problem name, directory and main subdirectories
problem_name, problem_dir, offline_stage_dir, postprocess_dir, is_same_offstage, \
    crve_file_path, hres_file_path = filop.setproblemdirs(input_file_name, input_file_dir)
# Package data associated to directories and paths
dirs_dict = packager.packdirpaths(input_file_name, input_file_path, input_file_dir,
                                  problem_name, problem_dir, offline_stage_dir,
                                  postprocess_dir, crve_file_path, hres_file_path)
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
problem_dict, mat_dict, macload_dict, rg_dict, clst_dict, scs_dict, algpar_dict, vtk_dict =\
    rid.readinputdatafile(input_file, dirs_dict)
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
if not is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayinfo('2', 'Compute cluster analysis data matrix')
    phase_init_time = time.time()
    # Compute the data required to perform the clustering according to the adopted strategy
    clstdata.set_clustering_data(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict)
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Compute cluster analysis data matrix')
    phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
    info.displayinfo('3', 'Compute cluster analysis data matrix',
                     phase_times[phase_times.shape[0]-1, 1] -
                     phase_times[phase_times.shape[0]-1, 0])
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
    # Update clustering dictionary
    clst_dict['voxels_clusters'] = crve.voxels_clusters
    clst_dict['phase_clusters'] = crve.phase_clusters
    clst_dict['clusters_f'] = crve.clusters_f
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
                problem_dict['comp_order_sym'],
                clst_dict['global_data_matrix'],
                clst_dict['clustering_type'],
                clst_dict['phase_n_clusters'],
                clst_dict['base_clustering_scheme'],
                clst_dict['adaptive_clustering_scheme'],
                clst_dict['adaptivity_criterion'],
                clst_dict['adaptivity_type'],
                clst_dict['adaptivity_control_feature'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute Cluster-Reduced Representative Volume Element (CRVE)
    crve.perform_crve_base_clustering()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Update clustering dictionary
    clst_dict['voxels_clusters'] = crve.voxels_clusters
    clst_dict['phase_clusters'] = crve.phase_clusters
    clst_dict['clusters_f'] = crve.clusters_f
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write clustering VTK file
    if vtk_dict['is_VTK_output']:
        info.displayinfo('5', 'Writing clustering VTK file...')
        vtkoutput.writevtkclusterfile(vtk_dict, dirs_dict, rg_dict, clst_dict)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
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
#                                 Online stage: Solve reduced microscale equilibrium problem
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayinfo('2', 'Solve reduced microscale equilibrium problem')
phase_init_time = time.time()
# Solve the reduced microscale equilibrium problem through solution of the clusterwise
# discretized Lippmann-Schwinger system of equilibrium equations
sca.sca(dirs_dict, problem_dict, mat_dict, rg_dict, clst_dict, macload_dict, scs_dict,
    algpar_dict, vtk_dict, crve)
# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Solve reduced microscale equilibrium problem')
phase_times = np.append(phase_times, [[phase_init_time, phase_end_time]], axis=0)
info.displayinfo('3', 'Solve reduced microscale equilibrium problem',
                 phase_times[phase_times.shape[0] - 1, 1] -
                 phase_times[phase_times.shape[0] - 1, 0])
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
