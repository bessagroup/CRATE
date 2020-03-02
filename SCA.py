#
# Self-Consistent Clustering Analysis (SCA) - Clustering Reduced Order Model
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Description:
# ...
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
# Credits:
# This program structure was inspired on the original Self-Consistent Clustering Analysis
# Matlab code implemented and developed by Dr. Zeliang Liu and Dr. Miguel A. Bessa in the
# course of the research published in "Self-consistent clustering analysis: an efficient
# multi-scale scheme for inelastic heterogeneous materials. Comp Methods Appl M 305 (2016):
# 319-341 (Liu, Z., Bessa, M.A. and Liu, W.K.)".
#
# ------------------------------------------------------------------------------------------
# Licensing and Copyrights:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Z.Liu & M.A.Bessa    |     2016     | Original SCA Matlab code (see credits)
# Bernardo P. Ferreira | January 2020 | Initial coding.
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
import info
# Display errors, warnings and built-in exceptions
import errors
# Read user input data file
import readInputData as rid
# Manage files and directories
import fileOperations
# Packager
import packager
# Clustering quantities computation
import clusteringQuantities
# Perform clustering
import clusteringMethods
# Cluster interaction tensors computation
import clusterInteractionTensors
# VTK output
import VTKOutput
#
#                             Check user input data file and create problem main directories
# ==========================================================================================
# Check if the input data file path was provided
if len(sys.argv[1:]) == 0:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayError('E00001',location.filename,location.lineno+1)
elif not os.path.isfile(str(sys.argv[1])):
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayError('E00001',location.filename,location.lineno+1)
# Set input data file name, path and directory
input_file_name,input_file_path,input_file_dir = \
                                       fileOperations.setInputDataFilePath(str(sys.argv[1]))
# Set problem name, directory and main subdirectories
problem_name,problem_dir,offline_stage_dir,postprocess_dir,is_same_offstage,\
cluster_file_path,cit_file_path,hres_file_path \
                             = fileOperations.setProblemDirs(input_file_name,input_file_dir)
# Package data associated to directories and paths
dirs_dict = packager.packageDirsPaths(input_file_name,input_file_path,input_file_dir,
                                 problem_name,problem_dir,offline_stage_dir,postprocess_dir,
                                 cluster_file_path,cit_file_path,hres_file_path)
# Open user input data file
try:
    input_file = open(input_file_path,'r')
except FileNotFoundError as message:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayException(location.filename,location.lineno+1,message)
#
#                                                                              Start program
# ==========================================================================================
# Get current time and date
start_date = time.strftime("%d/%b/%Y")
start_time = time.strftime("%Hh%Mm%Ss")
start_time_s = time.time()
phase_names = ['']
phase_times = np.zeros((1,2))
phase_names[0] = 'Total'
phase_times[0,:] = [start_time_s,0.0]
# Display starting program header
info.displayInfo('0',problem_name,start_time,start_date)
#
#                                                                  Read user input data file
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayInfo('2','Read input data file')
phase_init_time = time.time()
# Open user input data file
try:
    input_file = open(input_file_path,'r')
except FileNotFoundError as message:
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayException(location.filename,location.lineno+1,message)
# Read input data according to analysis type
info.displayInfo('5','Reading the input data file...')
strain_formulation,problem_type,n_dim,comp_order_sym,comp_order_nsym,n_material_phases,\
material_properties, mac_load_type,mac_load,mac_load_presctype,self_consistent_scheme, \
scs_max_n_iterations,scs_conv_tol,clustering_method,clustering_strategy, \
clustering_solution_method,phase_n_clusters,n_load_increments,max_n_iterations,conv_tol, \
max_subincrem_level,su_max_n_iterations,su_conv_tol,discret_file_path,rve_dims = \
                                                     rid.readInputData(input_file,dirs_dict)
# Close user input data file
input_file.close()
# Package data associated to problem general parameters
info.displayInfo('5','Packaging problem general data...')
problem_dict = packager.packageProblem(strain_formulation,problem_type,n_dim,comp_order_sym,
                                                                            comp_order_nsym)
# Package data associated to the material phases
info.displayInfo('5','Packaging material data...')
mat_dict = packager.packageMaterialPhases(n_material_phases,material_properties)
# Package data associated to the macroscale loading
info.displayInfo('5','Packaging macroscale loading data...')
macload_dict = packager.packageMacroscaleLoading(mac_load_type,mac_load,mac_load_presctype,
                                                                          n_load_increments)
# Package data associated to the spatial discretization file(s)
info.displayInfo('5','Packaging regular grid data...')
rg_dict = packager.packageRegularGrid(discret_file_path,rve_dims,mat_dict,
                                                                copy.deepcopy(problem_dict))
# Package data associated to the clustering
info.displayInfo('5','Packaging clustering data...')
clst_dict = packager.packageRGClustering(clustering_method,clustering_strategy,\
                         clustering_solution_method,phase_n_clusters,copy.deepcopy(rg_dict))
if is_same_offstage:
    # Save copy of clustering dictionary for compatibility check procedure (loading
    # previously computed offline stage)
    clst_dict_read = copy.deepcopy(clst_dict)
# Package data associated to the VTK output
info.displayInfo('5','Packaging VTK output data...')
vtk_dict = packager.packageVTK()
# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Read input data')
phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
info.displayInfo('3','Read input data file',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
#
#                                      Offline stage: Compute clustering-defining quantities
# ==========================================================================================
if not is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayInfo('2','Compute cluster-defining quantities')
    phase_init_time = time.time()
    # Compute the quantities required to perform the clustering according to the strategy
    # adopted
    clusteringQuantities.computeClusteringQuantities(copy.deepcopy(problem_dict),
                                   copy.deepcopy(mat_dict),copy.deepcopy(rg_dict),clst_dict)
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Compute cluster-defining quantities')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    info.displayInfo('3','Compute cluster-defining quantities',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
#
#                                                          Offline stage: Perform clustering
# ==========================================================================================
if not is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayInfo('2','Perform clustering')
    phase_init_time = time.time()
    # Perform the clustering according to the selected method and adopted strategy
    clusteringMethods.performClustering(copy.deepcopy(dirs_dict),copy.deepcopy(mat_dict),
                                                           copy.deepcopy(rg_dict),clst_dict)
    # Write clustering VTK file
    VTKOutput.writeVTKClusterFile(vtk_dict,copy.deepcopy(dirs_dict),copy.deepcopy(rg_dict),
                                                                   copy.deepcopy(clst_dict))
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Perform clustering')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    info.displayInfo('3','Perform clustering',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
else:
    # Display starting phase information and set phase initial time
    info.displayInfo('2','Import known clustering data')
    phase_init_time = time.time()
    # Get clusters data file path
    info.displayInfo('5','Loading data from clusters data file (.clusters)...')
    cluster_file_path = dirs_dict['cluster_file_path']
    # Open clusters data file
    try:
        cluster_file = open(cluster_file_path,'rb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
    # Load cluster data from file
    clst_dict = pickle.load(cluster_file)
    # Close clusters data file
    cluster_file.close()
    # Check compatibility between the loaded clusters data and the input data file
    info.displayInfo('5','Performing compatibility check on loaded data...')
    clusteringMethods.checkClstCompatibility(copy.deepcopy(problem_dict),
              copy.deepcopy(rg_dict),copy.deepcopy(clst_dict_read),copy.deepcopy(clst_dict))
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Import known clustering data')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    info.displayInfo('3','Import known clustering data',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
#
#                                         Offline stage: Compute cluster interaction tensors
# ==========================================================================================
if not is_same_offstage:
    # Display starting phase information and set phase initial time
    info.displayInfo('2','Compute cluster interaction tensors')
    phase_init_time = time.time()
    # Compute the cluster interaction tensors
    clusterInteractionTensors.computeClusterInteractionTensors(copy.deepcopy(dirs_dict),
                                        copy.deepcopy(problem_dict),copy.deepcopy(mat_dict),
                                                           copy.deepcopy(rg_dict),clst_dict)
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Compute cluster interaction tensors')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    info.displayInfo('3','Compute cluster interaction tensors',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
else:
    # Display starting phase information and set phase initial time
    info.displayInfo('2','Import cluster interaction tensors')
    phase_init_time = time.time()
    # Get cluster interaction tensors file path
    info.displayInfo('5','Loading clustering interaction tensors (.cit)...')
    cit_file_path = dirs_dict['cit_file_path']
    # Open clustering interaction tensors file
    try:
        cit_file = open(cit_file_path,'rb')
    except Exception as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
    # Load clustering interaction tensors
    [clst_dict['cit_1'],clst_dict['cit_2'],clst_dict['cit_0_freq']] = pickle.load(cit_file)
    # Close clustering interaction tensors file
    cit_file.close()
    # Check compatibility between the loaded cluster interaction tensors and the material
    # phases existent in the spatial discretization file
    info.displayInfo('5','Performing compatibility check on loaded data...')
    clusterInteractionTensors.checkCITCompatibility(copy.deepcopy(mat_dict),
                                                                   copy.deepcopy(clst_dict))
    # Set phase ending time and display finishing phase information
    phase_end_time = time.time()
    phase_names.append('Import cluster interaction tensors')
    phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
    info.displayInfo('3','Import cluster interaction tensors',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
#
#                               Online stage: Solve discretized Lippmann-Schwinger equations
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayInfo('2','Solve discretized Lippmann-Schwinger equations')
phase_init_time = time.time()





# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Solve discretized Lippmann-Schwinger equations')
phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
info.displayInfo('3','Solve discretized Lippmann-Schwinger equations',
                phase_times[phase_times.shape[0]-1,1]-phase_times[phase_times.shape[0]-1,0])
#
#                                                                                End program
# ==========================================================================================
# Get current time and date
end_date = time.strftime("%d/%b/%Y")
end_time = time.strftime("%Hh%Mm%Ss")
end_time_s = time.time()
phase_times[0,1] = end_time_s
# Display ending program message
info.displayInfo('1',end_time,end_date,problem_name,phase_names,phase_times)
