#
# Input Data Reader Module (UNNAMED Program)
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
# Working with arrays
import numpy as np
# Mathematics
import math
# Regular expressions
import re
# Read specific lines from file
import linecache
# Inspect file name and line
import inspect
# Extract information from path
import ntpath
# Display errors, warnings and built-in exceptions
import errors
#
#                                                             Check input validity functions
# ==========================================================================================
# Check if a given input is or represents a number (either integer or floating-point)
def checkNumber(x):
    isNumber = True
    try:
        float(x)
        return isNumber
    except ValueError:
        isNumber = False
        return isNumber
# ------------------------------------------------------------------------------------------
# Check if a given input is a positive integer
def checkPositiveInteger(x):
    isPositiveInteger = True
    if isinstance(x,int):
        if x <= 0:
            isPositiveInteger = False
    elif not re.match('^[1-9][0-9]*$',str(x)):
        isPositiveInteger = False
    return isPositiveInteger
# ------------------------------------------------------------------------------------------
# Check if a given input contains only letters, numbers or underscores
def checkValidName(x):
    isValid = True
    if not re.match('^[A-Za-z0-9_]+$',str(x)):
        isValid = False
    return isValid
#
#                                                                           Search functions
# ==========================================================================================
# Find the line number where a given keyword is specified in a file
def searchKeywordLine(file,keyword):
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line and line.strip()[0]!='#':
            return line_number
    location = inspect.getframeinfo(inspect.currentframe())
    errors.displayError('E00003',location.filename,location.lineno+1,keyword)
# ------------------------------------------------------------------------------------------
# Search for a given keyword in a file. If the keyword is found, the line number is returned
def searchOptionalKeywordLine(file,keyword):
    isFound = False
    file.seek(0)
    line_number = 0
    for line in file:
        line_number = line_number + 1
        if keyword in line and line.strip()[0]!='#':
            isFound = True
            return [isFound,line_number]
    return [isFound,line_number]
# ------------------------------------------------------------------------------------------
# Find the maximum number of specified material phase properties
def findMaxNumberProperties(file,file_path,keyword,keyword_line_number,n_material_phases):
    max_n_properties = 0
    line_number = keyword_line_number + 1
    for iphase in range(1,n_material_phases+1):
        phase_header = linecache.getline(file_path,line_number).strip().split(' ')
        if phase_header[0] == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00005',location.filename,location.lineno+1,keyword,iphase)
        elif phase_header[0] != str(iphase) or not checkPositiveInteger(phase_header[1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00005',location.filename,location.lineno+1,keyword,iphase)
        if int(phase_header[1]) > max_n_properties:
            max_n_properties = int(phase_header[1])
        line_number = line_number + int(phase_header[1]) + 1
    return max_n_properties
#
#                                                                  Read input data functions
# ==========================================================================================
# Read the input data file
def readInputData(input_file,input_file_path,problem_name,problem_dir):
    # Read strain formulation
    keyword = 'Strain_Formulation'
    max = 2
    strain_formulation = readTypeAKeyword(input_file,input_file_path,keyword,max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read problem type and set problem dimension
    keyword = 'Problem_Type'
    max = 4
    problem_type = readTypeAKeyword(input_file,input_file_path,keyword,max)
    if problem_type in [1,2,3]:
        problem_dimension = 4
    elif problem_type == 4:
        problem_dimension = 9
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of material phases
    keyword = 'Number_of_Material_Phases'
    max = '~'
    n_material_phases = readTypeAKeyword(input_file,input_file_path,keyword,max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material properties
    keyword = 'Material_Properties'
    material_properties = readMaterialProperties(input_file,input_file_path,keyword,
                                                                          n_material_phases)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read macroscale loading
    keyword = 'Macroscale_Loading'
    max = 3
    macroscale_loading_type = readTypeAKeyword(input_file,input_file_path,keyword,max)
    macroscale_loading, macroscale_load_indexes = \
                   readMacroscaleLoading(input_file,input_file_path,macroscale_loading_type,
                                                       problem_dimension,strain_formulation)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme (optional). If the associated keyword is not found, then
    # a default specification is assumed
    keyword = 'Self_Consistent_Scheme'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = 2
        self_consistent_scheme = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        self_consistent_scheme = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme maximum number of iterations (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SCS_Max_Number_of_Iterations'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = '~'
        scs_max_n_iterations = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        scs_max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read self consistent scheme convergence tolerance (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SCS_Convergence_Tolerance'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = '~'
        scs_conv_tol = readTypeBKeyword(input_file,input_file_path,keyword)
    else:
        scs_conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering method
    keyword = 'Clustering_Method'
    max = 1
    clustering_method = readTypeAKeyword(input_file,input_file_path,keyword,max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering strategy (optional). If the associated keyword is not found, then a
    # default specification is assumed
    keyword = 'Clustering_Strategy'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = 1
        clustering_strategy = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        clustering_strategy = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read clustering solution method (optional). If the associated keyword is not found,
    # then a default specification is assumed
    keyword = 'Clustering_Solution_Method'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = 1
        clustering_solution_method = \
                                    readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        clustering_solution_method = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of cluster associated to each material phase
    keyword = 'Number_of_Clusters'
    phase_clustering = readPhaseClustering(input_file,input_file_path,keyword,
                                                                          n_material_phases)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read number of load increments
    keyword = 'Number_of_Load_Increments'
    max = '~'
    n_load_increments = readTypeAKeyword(input_file,input_file_path,keyword,max)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum number of iterations to solve each load increment (optional). If the
    # associated keyword is not found, then a default specification is assumed
    keyword = 'Max_Number_of_Iterations'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = '~'
        max_n_iterations = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read convergence tolerance to solve each load increment (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'Convergence_Tolerance'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        conv_tol = readTypeBKeyword(input_file,input_file_path,keyword)
    else:
        conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read maximum level of subincrementation allowed (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'Max_SubIncrem_Level'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = '~'
        max_subincrem_level = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        max_subincrem_level = 5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update maximum number of iterations (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SU_Max_Number_of_Iterations'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        max = '~'
        max_n_iterations = readTypeAKeyword(input_file,input_file_path,keyword,max)
    else:
        max_n_iterations = 20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read material state update convergence tolerance (optional). If the associated
    # keyword is not found, then a default specification is assumed
    keyword = 'SU_Convergence_Tolerance'
    isFound, keyword_line_number = searchOptionalKeywordLine(input_file,keyword)
    if isFound:
        su_conv_tol = readTypeBKeyword(input_file,input_file_path,keyword)
    else:
        su_conv_tol = 1e-6
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the spatial discretization file absolute path
    # [WIP - Implement the case of several spatial discretization files]
    keyword = 'Discretization_File'
    valid_exts = ['.rgmsh']
    discret_file_path, discret_file_ext = \
                   readDiscretizationFilePath(input_file,input_file_path,keyword,valid_exts)
    # Copy the spatial discretization file to the problem directory and update the absolute
    # path to the copied file
    try:
        shutil.copy2(discret_file_path,problem_dir+problem_name+discret_file_ext)
        discret_file_path = problem_dir+problem_name+discret_file_ext
    except IOError as message:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayException(location.filename,location.lineno+1,message)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return [strain_formulation,problem_type,problem_dimension,n_material_phases,
            material_properties,macroscale_loading_type,macroscale_loading,
            macroscale_load_indexes,self_consistent_scheme,scs_max_n_iterations,
            scs_conv_tol,clustering_method,clustering_strategy,clustering_solution_method,
            phase_clustering,n_load_increments,max_n_iterations,conv_tol,
            max_subincrem_level,max_n_iterations,su_conv_tol,discret_file_path]
#
# ------------------------------------------------------------------------------------------
# Read a keyword of type A specification, characterized as follows:
#
# < keyword > < positive integer >
#
def readTypeAKeyword(file,file_path,keyword,max):
    keyword_line_number = searchKeywordLine(file,keyword)
    line = linecache.getline(file_path,keyword_line_number).strip().split(' ')
    if len(line) == 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00007',location.filename,location.lineno+1,keyword)
    elif not checkPositiveInteger(line[1]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00007',location.filename,location.lineno+1,keyword)
    elif isinstance(max,int):
        if int(line[1]) > max:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00007',location.filename,location.lineno+1,keyword)
    return int(line[1])
# ------------------------------------------------------------------------------------------
# Read a keyword of type B specification, characterized as follows:
#
# < keyword >
# < non-negative floating-point number >
#
def readTypeBKeyword(file,file_path,keyword):
    keyword_line_number = searchKeywordLine(file,keyword)
    line = linecache.getline(file_path,keyword_line_number+1).strip().split(' ')
    if line == '':
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00004',location.filename,location.lineno+1,keyword)
    elif len(line) != 1:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00004',location.filename,location.lineno+1,keyword)
    elif not checkNumber(line[0]) and float(line[0]) <= 0:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00004',location.filename,location.lineno+1,keyword)
    return float(line[0])
# ------------------------------------------------------------------------------------------
# Read the material properties of the general heterogeneous material specified as follows:
#
# Material_Properties
# < phase_id > < number of properties >
# < property1_name > < value >
# < property2_name > < value >
# < phase_id > < number of properties >
# < property1_name > < value >
# ...
#
# and store it in a array(max_n_properties,2,n_material_phases) as
#                                      _                       _
#                                     |'property1_name' , value |
#               array[:,:,phase_id] = |'property2_name' , value |
#                                     |_     ...        ,  ... _|
#
def readMaterialProperties(file,file_path,keyword,n_material_phases):
    keyword_line_number = searchKeywordLine(file,keyword)
    max_n_properties = findMaxNumberProperties(file,file_path,keyword,
                                                      keyword_line_number,n_material_phases)
    material_properties = np.full((max_n_properties,2,n_material_phases),'ND',dtype=object)
    line_number = keyword_line_number + 1
    for iphase in range(1,n_material_phases+1):
        phase_header = linecache.getline(file_path,line_number).strip().split(' ')
        for iproperty in range(1,int(phase_header[1])+1):
            property_line = linecache.getline(file_path,
                                                 line_number+iproperty).strip().split(' ')
            if property_line[0] == '':
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00006',location.filename,location.lineno+1,
                                                                 keyword,iproperty,iphase)
            elif len(property_line) != 2:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00006',location.filename,location.lineno+1,
                                                                 keyword,iproperty,iphase)
            elif not checkValidName(property_line[0]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00006',location.filename,location.lineno+1,
                                                                 keyword,iproperty,iphase)
            elif not checkNumber(property_line[1]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00006',location.filename,location.lineno+1,
                                                                 keyword,iproperty,iphase)
            material_properties[iproperty-1,0,iphase-1] = property_line[0]
            material_properties[iproperty-1,1,iphase-1] = float(property_line[1])
        line_number = line_number + int(phase_header[1]) + 1
    return material_properties
# ------------------------------------------------------------------------------------------
# Read the macroscale loading conditions, specified as
#
# Macroscale_Strain or Macroscale_Stress
# < component_name_11 > < value >
# < component_name_21 > < value >
#
# Mixed_Prescription_Index (only if prescribed macroscale strains and stresses)
# < 0 or 1 > < 0 or 1 > ...
#
# and store it in a array(problem_dimension,2,2) as
#                                    _                          _
#                                   |'component_name_11' , value |
#                    array[:,:,0] = |'component_name_21' , value |
#                                   |_       ...        ,   ... _|
#
# and a array(problem_dimension) as
#
#                    array = [ 0 , 1 , 1 , 0 , ... ]
#
# Note: The symmetry of the macroscale strain and stress tensors is verified under a small
#       strain formulation
#
def readMacroscaleLoading(file,file_path,macroscale_loading_type,
                                                      problem_dimension,strain_formulation):
    if macroscale_loading_type == 1:
        loading_keywords = ['Macroscale_Strain']
    elif macroscale_loading_type == 2:
        loading_keywords = ['Macroscale_Stress']
    elif macroscale_loading_type == 3:
        loading_keywords = ['Macroscale_Strain','Macroscale_Stress']
        presc_keyword = 'Mixed_Prescription_Index'
    macroscale_loading = np.full((problem_dimension,2,2),'ND',dtype=object)
    for load_keyword in loading_keywords:
        load_keyword_line_number = searchKeywordLine(file,load_keyword)
        for icomponent in range(1,problem_dimension+1):
            component_line = linecache.getline(file_path,
                                     load_keyword_line_number+icomponent).strip().split(' ')
            if component_line[0] == '':
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00008',location.filename,location.lineno+1,
                                                                    load_keyword,icomponent)
            elif len(component_line) != 2:
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00008',location.filename,location.lineno+1,
                                                                    load_keyword,icomponent)
            elif not checkValidName(component_line[0]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00008',location.filename,location.lineno+1,
                                                                    load_keyword,icomponent)
            elif not checkNumber(component_line[1]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayError('E00008',location.filename,location.lineno+1,
                                                                    load_keyword,icomponent)
            store_index = (0,1,loading_keywords.index(load_keyword))
            macroscale_loading[icomponent-1,0,store_index[macroscale_loading_type-1]] = \
                                                                           component_line[0]
            macroscale_loading[icomponent-1,1,store_index[macroscale_loading_type-1]] = \
                                                                    float(component_line[1])
    if macroscale_loading_type == 1:
        macroscale_load_indexes = np.zeros((problem_dimension),dtype=int)
    elif macroscale_loading_type == 2:
        macroscale_load_indexes = np.ones((problem_dimension),dtype=int)
    elif macroscale_loading_type == 3:
        macroscale_load_indexes = np.zeros((problem_dimension),dtype=int)
        presc_keyword_line_number = searchKeywordLine(file,presc_keyword)
        presc_line = \
                 linecache.getline(file_path,presc_keyword_line_number+1).strip().split(' ')
        if presc_line[0] == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00011',location.filename,location.lineno+1,presc_keyword)
        elif len(presc_line) != problem_dimension:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00011',location.filename,location.lineno+1,presc_keyword)
        elif not all(presc_line[i] == '0' or presc_line[i] == '1' \
                                                           for i in range(len(presc_line))):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00011',location.filename,location.lineno+1,presc_keyword)
        else:
            macroscale_load_indexes = \
                              np.array([int(presc_line[i]) for i in range(len(presc_line))])
    # Check small strain formulation symmetry
    if strain_formulation == 1:
        if problem_dimension == 4:
            symmetric_indexes = np.array([[3],[1]])
        elif problem_dimension == 9:
            symmetric_indexes = np.array([[3,6,7],[1,2,5]])
        for i in range(symmetric_indexes.shape[1]):
            if macroscale_loading_type == 1:
                isEqual = np.allclose(
                                  macroscale_loading[symmetric_indexes[0,i],1,0],
                                  macroscale_loading[symmetric_indexes[1,i],1,0],atol=1e-10)
                if not isEqual:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayWarning('W00001',location.filename,location.lineno+1,
                                                                    macroscale_loading_type)
                    macroscale_loading[symmetric_indexes[1,i],1,0] = \
                                              macroscale_loading[symmetric_indexes[0,i],1,0]
            if macroscale_loading_type == 2:
                isEqual = np.allclose(
                                  macroscale_loading[symmetric_indexes[0,i],1,1],
                                  macroscale_loading[symmetric_indexes[1,i],1,1],atol=1e-10)
                if not isEqual:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayWarning('W00001',location.filename,location.lineno+1,
                                                                    macroscale_loading_type)
                    macroscale_loading[symmetric_indexes[1,i],1,1] = \
                                              macroscale_loading[symmetric_indexes[0,i],1,1]
            elif macroscale_loading_type == 3:
                if macroscale_load_indexes[symmetric_indexes[0,i]] != \
                                            macroscale_load_indexes[symmetric_indexes[1,i]]:
                    location = inspect.getframeinfo(inspect.currentframe())
                    errors.displayError('E00012',location.filename,location.lineno+1,i)
                aux = macroscale_load_indexes[symmetric_indexes[0,i]]
                isEqual = np.allclose(
                                macroscale_loading[symmetric_indexes[0,i],1,aux],
                                macroscale_loading[symmetric_indexes[1,i],1,aux],atol=1e-10)
                if not isEqual:
                    if aux == 0:
                        location = inspect.getframeinfo(inspect.currentframe())
                        errors.displayWarning('W00001',location.filename,location.lineno+1,
                                                                macroscale_loading_type,aux)
                    elif aux == 1:
                        location = inspect.getframeinfo(inspect.currentframe())
                        errors.displayWarning('W00001',location.filename,location.lineno+1,
                                                                macroscale_loading_type,aux)
                    macroscale_loading[symmetric_indexes[1,i],1,aux] = \
                                            macroscale_loading[symmetric_indexes[0,i],1,aux]
    return macroscale_loading, macroscale_load_indexes
# ------------------------------------------------------------------------------------------
# Read the number of clusters associated to each material phase, specified as
#
# Number_of_Clusters
# < phase_id > < number_of_clusters >
# < phase_id > < number_of_clusters >
#
# and store it in a array(n_material_phases,2) as
#                                    _                                     _
#                                   | < phase_id > , < number_of_clusters > |
#                      array[:,2] = | < phase_id > , < number_of_clusters > |
#                                   |_     ...     ,          ...          _|
#
def readPhaseClustering(file,file_path,keyword,n_material_phases):
    phase_clustering = np.zeros((n_material_phases,2),dtype=int)
    line_number = searchKeywordLine(file,keyword) + 1
    for iphase in range(1,n_material_phases+1):
        line = linecache.getline(file_path,line_number+iphase-1).strip().split(' ')
        if line[0] == '':
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00013',location.filename,location.lineno+1,keyword,iphase)
        elif len(line) != 2:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00013',location.filename,location.lineno+1,keyword,iphase)
        elif not checkPositiveInteger(line[0]) or not checkPositiveInteger(line[1]):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00013',location.filename,location.lineno+1,keyword,iphase)
        elif int(line[0]) != iphase:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayError('E00013',location.filename,location.lineno+1,keyword,iphase)
        phase_clustering[iphase-1,:] = [int(i) for i in line[0:2]]
    return phase_clustering
# ------------------------------------------------------------------------------------------
# Read the spatial discretization file absolute path
#
# Discretization_File
# < absolute_path >
#
def readDiscretizationFilePath(file,file_path,keyword,valid_exts):
    line_number = searchKeywordLine(file,keyword) + 1
    discret_file_path = linecache.getline(file_path,line_number).strip()
    if not os.path.isabs(discret_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00014',location.filename,location.lineno+1,keyword,\
                                                                          discret_file_path)
    elif not os.path.isfile(discret_file_path):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00014',location.filename,location.lineno+1,keyword,\
                                                                          discret_file_path)
    elif not ntpath.splitext(ntpath.basename(discret_file_path))[-1] in valid_exts:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayError('E00015',location.filename,location.lineno+1,keyword,valid_exts)
    discret_file_ext = ntpath.splitext(ntpath.basename(discret_file_path))[-1]
    return [discret_file_path,discret_file_ext]
