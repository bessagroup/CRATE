#
# Information Module
# ==========================================================================================
# Summary:
# ...
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | January 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
# Mathematics
import math
# Date and time
import time
# Manage files and directories
import fileOperations
#
#                                                Display features and manipulation functions
# ==========================================================================================
# Set output features in a consistent way throughout the program
def setDisplayFeatures():
    output_width = 92
    dashed_line = '-'*output_width
    indent = '  '
    asterisk_line = '*'*output_width
    tilde_line = '~'*output_width
    displayFeatures = (output_width,dashed_line,indent,asterisk_line,tilde_line)
    return displayFeatures
# ------------------------------------------------------------------------------------------
# Convert iterable to list
def convertIterableToList(iterable):
    list = [ element for element in iterable ]
    return list
#
#                                                                         Set print function
# ==========================================================================================
def print2(*objects):
    # Print to default sys.stdout
    print(*objects)
    # Print to '.screen file'
    screen_file = open(fileOperations.screen_file_path,'a')
    print(*objects,file = screen_file)
    screen_file.close()
#
#                                                                           Display function
# ==========================================================================================
# Set and display information about the program start, execution phases and finish
def displayInfo(code,*args,**kwargs):
    # Get current date and time
    current_date = time.strftime("%d/%b/%Y")
    current_time = time.strftime("%Hh%Mm%Ss")
    current_time_s = time.time()
    # Get display features
    displayFeatures = setDisplayFeatures()
    output_width, dashed_line, indent = displayFeatures[0:3]
    tilde_line = displayFeatures[4]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set informations and formats to display
    if code == '-1':
        status = args[1]
        arguments = convertIterableToList([args[0],])
        info = tuple(arguments)
        if status == 0:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' + \
                                'Status: New problem'
        elif status == 1:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' + \
                                'Status: Repeating problem (considering existent ' + \
                                'offline stage)'
        elif status == 2:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' + \
                                'Status: New problem (overwriting existing directory)'
        elif status == 3:
            info.print2('Please rerun the program and provide a different problem name.' + \
                        '\n')
            sys.exit(1)
    elif code == '0':
        arguments = ['Brand New Shiny Unnamed Program','v1.0.0'] + \
                    convertIterableToList(2*[args[0],]) + convertIterableToList(args[1:3])
        info = tuple(arguments)
        template = '\n' + tilde_line + '\n{:^{width}}\n' + '{:^{width}}\n\n' + \
                   'Problem under analysis: {}' + '\n\n' + \
                   'Input data file: {}.dat' + '\n\n' + \
                   'Starting program execution at: {} ({})\n' + tilde_line + '\n\n' + \
                    dashed_line
    elif code == '1':
        phase_names = args[3]
        phase_times = args[4]
        total_time = phase_times[0,1] - phase_times[0,0]
        number_of_phases = len(phase_names)
        phase_durations = [ phase_times[i,1] - phase_times[i,0] \
                                                         for i in range(0,number_of_phases)]
        for i in range(0,number_of_phases):
            phase_durations.insert(3*i,phase_names[i])
            phase_durations.insert(3*i+2,(phase_durations[3*i+1]/total_time)*100)
        arguments = convertIterableToList(args[0:3]) + \
                    [total_time,math.floor(total_time/3600),(total_time%3600)/60] + \
                    ['Phase','Duration (s)','%'] + phase_durations[3:] + \
                    ['Program Completed']
        info = tuple(arguments)
        template = '\n' + tilde_line + '\n' + \
                   'Ending program execution at: {} ({})\n\n' + \
                   'Problem analysed: {}\n\n' + \
                   'Total execution time: {:.2e}s (~{:.0f}h{:.0f}m)\n\n' + \
                   'Execution times: \n\n' + \
                   2*indent + '{:50}{:^20}{:^5}' + '\n' + \
                   2*indent + 75*'-' + '\n' + \
                   (2*indent + '{:50}{:^20.2e}{:>5.2f} \n')*(number_of_phases-1) + \
                   2*indent + 75*'-' + '\n\n\n' + \
                   '{:^{width}}' + '\n'
    elif code == '2':
        arguments = convertIterableToList((args[0],))
        info = tuple(arguments)
        template = 'Start phase: {} \n'
    elif code == '3':
        arguments = convertIterableToList(args[0:2])
        info = tuple(arguments)
        template = '\n\nEnd phase: {} (phase duration time = {:.2e}s)\n' + dashed_line
    elif code == '5':
        if len(args) == 2:
            n_indents = args[1]
        else:
            n_indents = 1
        arguments = convertIterableToList((args[0],))
        info = tuple(arguments)
        template = '\n' + n_indents*indent + '> {}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display information
    print2(template.format(*info,width=output_width))
