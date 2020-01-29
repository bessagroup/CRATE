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
# Date and time
import time
# Inspect file name and line
import inspect
# Display messages
import info
# Display errors, warnings and built-in exceptions
import errors
# Read user input data file
import readInputData as rid
# Manage files and directories
import fileOperations
# Clustering quantities computation
import clusteringQuantities as clsq
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
problem_name,problem_dir,offline_stage_dir,postprocess_dir,is_same_offstage,hres_file_path \
                             = fileOperations.setProblemDirs(input_file_name,input_file_dir)
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
strain_formulation,problem_type,n_dim,n_strain,n_material_phases,material_properties, \
macroscale_loading_type,macroscale_loading,macroscale_load_indexes,self_consistent_scheme, \
scs_max_n_iterations,scs_conv_tol,clustering_method,clustering_strategy, \
clustering_solution_method,phase_clustering,n_load_increments,max_n_iterations,conv_tol, \
max_subincrem_level,max_n_iterations,su_conv_tol,discret_file_path = \
                      rid.readInputData(input_file,input_file_path,problem_name,problem_dir)
# Close user input data file
input_file.close()
# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Read input data')
phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
info.displayInfo('3','Read input data file',phase_times[1,1]-phase_times[1,0])
#
#
#                                      Offline stage: Compute clustering-defining quantities
# ==========================================================================================
# Display starting phase information and set phase initial time
info.displayInfo('2','Compute cluster-defining quantities')
phase_init_time = time.time()
# Compute the quantities required to perform the clustering according to the strategy
# adopted
clsq.computeClusteringQuantities(clustering_strategy,clustering_solution_method,discret_file_path)

# Set phase ending time and display finishing phase information
phase_end_time = time.time()
phase_names.append('Compute cluster-defining quantities')
phase_times = np.append(phase_times,[[phase_init_time,phase_end_time]],axis=0)
info.displayInfo('3','Compute cluster-defining quantities', \
                                                          phase_times[1,1]-phase_times[1,0])
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
