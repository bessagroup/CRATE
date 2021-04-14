#
# Information Module (CRATE Program)
# ==========================================================================================
# Summary:
# Procedures related to the print of informative data to the default standard output device
# (usually the terminal). In general, this data is associated to the program launch, to the
# progress of the main execution phases and to the program end.
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
# Shallow and deep copy operations
import copy
# Terminal colors
import colorama
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                                           Display function
# ==========================================================================================
# Set and display information about the program start, execution phases and finish
def displayinfo(code, *args, **kwargs):
    # Get display features
    display_features = ioutil.setdisplayfeatures()
    output_width, dashed_line, indent, asterisk_line = display_features[0:4]
    tilde_line, equal_line = display_features[4:6]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set informations and formats to display
    if code == '-1':
        status = args[1]
        arguments = [args[0],]
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
            ioutil.print2('Please rerun the program and provide a different problem ' + \
                          'name.' + '\n')
            sys.exit(1)
    elif code == '0':
        arguments = ['CRATE - Clustering-based Nonlinear Analysis of Materials',
                     'Created by Bernardo P. Ferreira', 'Release 0.8.0 (Dec 2020)'] + \
                     2*[args[0],] + list(args[1:3])
        info = tuple(arguments)
        template = '\n' + colorama.Fore.WHITE + tilde_line + colorama.Style.RESET_ALL + \
                   colorama.Fore.WHITE + '\n{:^{width}}\n\n' + '{:^{width}}\n' + \
                   '\n{:^{width}}\n\n' + \
                   colorama.Fore.YELLOW + 'Problem under analysis: ' + \
                   colorama.Style.RESET_ALL + '{}' + '\n\n' + \
                   colorama.Fore.YELLOW + 'Input data file: ' + \
                   colorama.Style.RESET_ALL + '{}.dat' + '\n\n' + \
                   colorama.Fore.YELLOW + 'Starting program execution at: ' + \
                   colorama.Style.RESET_ALL + '{} ({})\n' + colorama.Fore.WHITE + \
                   tilde_line + colorama.Style.RESET_ALL + '\n\n' + \
                   colorama.Fore.WHITE + dashed_line + colorama.Style.RESET_ALL
    elif code == '1':
        phase_names = args[3]
        phase_times = args[4]
        total_time = phase_times[0, 1] - phase_times[0, 0]
        number_of_phases = len(phase_names)
        phase_durations = [phase_times[i, 1] - phase_times[i, 0] \
                           for i in range(0, number_of_phases)]
        for i in range(0, number_of_phases):
            phase_durations.insert(3*i, phase_names[i])
            phase_durations.insert(3*i + 2, (phase_durations[3*i + 1]/total_time)*100)
        arguments = list(args[0:3]) + \
                    [total_time, np.floor(total_time/3600), (total_time%3600)/60] + \
                    ['Phase','Duration (s)','%'] + phase_durations[3:] + \
                    [colorama.Fore.GREEN + 'Program Completed' + colorama.Style.RESET_ALL]
        info = tuple(arguments)
        template = '\n' + colorama.Fore.WHITE + tilde_line + colorama.Style.RESET_ALL + \
                   '\n' + \
                   colorama.Fore.YELLOW + 'Ending program execution at: ' + \
                   colorama.Style.RESET_ALL + '{} ({})\n\n' + \
                   colorama.Fore.YELLOW + 'Problem analysed: ' + \
                   colorama.Style.RESET_ALL + '{}\n\n' + \
                   colorama.Fore.YELLOW + 'Total execution time: ' + \
                   colorama.Style.RESET_ALL + '{:.2e}s (~{:.0f}h{:.0f}m)\n\n' + \
                   colorama.Fore.YELLOW + 'Execution times: \n\n' + \
                   colorama.Style.RESET_ALL + \
                   2*indent + '{:50}{:^20}{:^5}' + '\n' + \
                   2*indent + 75*'-' + '\n' + \
                   (2*indent + '{:50}{:^20.2e}{:>5.2f} \n')*(number_of_phases-1) + \
                   2*indent + 75*'-' + '\n\n\n' + \
                   '{:^{width}}' + '\n'
    elif code == '2':
        arguments = [args[0],]
        info = tuple(arguments)
        template = colorama.Fore.GREEN + 'Start phase: ' + colorama.Fore.WHITE + \
            '{} \n' + colorama.Style.RESET_ALL
    elif code == '3':
        arguments = args[0:2]
        info = tuple(arguments)
        template = colorama.Fore.GREEN + '\n\nEnd phase: ' + colorama.Fore.WHITE + \
            '{} (phase duration time = {:.2e}s)\n' + dashed_line + colorama.Style.RESET_ALL
    elif code == '5':
        if len(args) == 2:
            n_indents = args[1]
        else:
            n_indents = 1
        arguments = [args[0],]
        info = tuple(arguments)
        template = '\n' + n_indents*indent + '> {}'
    elif code == '6':
        mode = args[0]
        if mode == 'progress':
            arguments = args[1:3]
            if args[1] == 1:
                print(' '.format(width=output_width))
            info = tuple(arguments)
            template = indent + '> Performing clustering process {} of {}...'
            print(template.format(*info, width=output_width), end='\r')
            if args[1] == args[2]:
                print(' '.format(width=output_width))
            return
        elif mode == 'completed':
            arguments = ['',]
            info = tuple(arguments)
            template = '\n' + indent + '> Completed all clustering processes!'
    elif code == '7':
        mode = args[0]
        if mode == 'init':
            subinc_level = args[2]
            if subinc_level == 0:
                arguments = list([args[1], ]) + list(args[3:])
                info = tuple(arguments)
                template = colorama.Fore.CYAN + '\n' + \
                           indent + 'Increment number: {:3d}' + '\n' + \
                           indent + equal_line[:-len(indent)] + '\n' + \
                           indent + 'Loading subpath: {:4d} |' + 6*' ' + \
                           'Load factor | Total = {:8.1e}' + 7*' ' + \
                           'Time | Total = {:8.1e}' + '\n' + \
                           indent + 6*' ' + 'Increment: {:4d} |' + 18*' ' + \
                           '| Incr. = {:8.1e}' + 12*' ' + '| Incr. = {:8.1e}' + \
                           colorama.Style.RESET_ALL
            else:
                arguments = args[1:]
                info = tuple(arguments)
                template = colorama.Fore.CYAN + '\n' + \
                           indent + 'Increment number: {:3d}' + 3*' ' + \
                           '(Sub-inc. level: {:3d})' + '\n' + \
                           indent + equal_line[:-len(indent)] + '\n' + \
                           indent + 'Loading subpath: {:4d} |' + 6*' ' + \
                           'Load factor | Total = {:8.1e}' + 7*' ' + \
                           'Time | Total = {:8.1e}' + '\n' + \
                           indent + 6*' ' + 'Increment: {:4d} |' + 18*' ' + \
                           '| Incr. = {:8.1e}' + 12*' ' + '| Incr. = {:8.1e}' + \
                           colorama.Style.RESET_ALL
        elif mode == 'end':
            space1 = (output_width - 84)*' '
            space2 = (output_width - (len('Homogenized strain tensor') + 48))*' '
            space3 = (output_width - (len('Increment run time (s): ') + 44))*' '
            problem_type = args[1]
            hom_strain = args[2]
            hom_stress = args[3]
            hom_strain_out = np.zeros((3, 3))
            hom_stress_out = np.zeros((3, 3))
            if problem_type == 1:
                hom_strain_out[0:2,0:2] = hom_strain
                hom_stress_out[0:2,0:2] = hom_stress
                hom_stress_out[2, 2] = args[6]
            else:
                hom_strain_out = copy.deepcopy(hom_strain)
                hom_stress_out = copy.deepcopy(hom_stress)
            arguments = list()
            for i in range(3):
                for j in range(3):
                    arguments.append(hom_strain_out[i, j])
                for j in range(3):
                    arguments.append(hom_stress_out[i, j])
            arguments = arguments + [args[4], args[5]]
            info = tuple(arguments)
            template = '\n\n' + \
                       indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 7*' ' + 'Homogenized strain tensor (\u03B5)' + space2 + \
                       'Homogenized stress tensor (\u03C3)' + '\n\n' + \
                       indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                       '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                       '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
                       indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 + \
                       '[' + 3*'{:>12.4e}' + '  ]' + '\n' + \
                       '\n' + indent + equal_line[:-len(indent)] + '\n' + \
                       indent + 'Increment run time (s): {:>11.4e}' + space3 + \
                       'Total run time (s): {:>11.4e}' + '\n\n'
    elif code == '8':
        mode = args[0]
        if mode == 'init':
            if args[1] == 0:
                format_1 = '{:.4e}'
                format_2 = '-'
            else:
                format_1 = '{:.4e}'
                format_2 = '{:.4e}'
            arguments = args[1:]
            info = tuple(arguments)
            template = colorama.Fore.YELLOW + \
                       '\n\n' + indent + 'Self-consistent scheme iteration: {:3d}' + \
                       '\n' + \
                       indent + tilde_line[:-len(indent)] + '\n' + \
                       indent + 'Young modulus (E): ' + format_1 + \
                       '  (norm. change: ' + format_2 + ')' + \
                       '\n' + \
                       indent + 'Poisson ratio (\u03BD): ' + format_1 + \
                       '  (norm. change: ' + format_2 + ')' + \
                       '\n' + \
                       indent + tilde_line[:-len(indent)] + '\n' + \
                       colorama.Style.RESET_ALL
        elif mode == 'end':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + dashed_line[:-len(indent)] + \
                       '\n\n' + indent + tilde_line[:-len(indent)] + '\n' + \
                       indent + 'Iteration run time (s): {:>11.4e}'
    elif code == '9':
        mode = args[0]
        space1 = (output_width - 48)*' '
        space2 = (output_width - 67)*' '
        if mode == 'init':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + 5*' ' + 'Iteration' + space1 + \
                       'Normalized residuals' + '\n' + \
                       indent + ' Number    Run time (s)' + space2 + \
                       'Equilibrium    Mac. strain    Mac. stress' + '\n' + \
                       indent + dashed_line[:-len(indent)]
        elif mode == 'iter':
            if not isinstance(args[4], float):
                arguments = list(args[1:4]) + [args[5],]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 + \
                           '{:>11.4e}         -          {:^11.4e}'
            elif not isinstance(args[5], float):
                arguments = args[1:5]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 + \
                           '{:>11.4e}     {:^11.4e}        -'
            else:
                arguments = args[1:]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 + \
                           '{:>11.4e}     {:^11.4e}    {:^11.4e}'
    elif code == '10':
        mode = args[0]
        space1 = (output_width - 55)*' '
        space2 = (output_width - 67)*' '
        if mode == 'init':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + 5*' ' + 'Iteration' + space1 + \
                       'Normalized residuals       Norm. error' + '\n' + \
                       indent + ' Number    Run time (s)' + space2 + \
                       'Equilibrium    Mac. stress    Hom. Strain' + '\n' + \
                       indent + dashed_line[:-len(indent)]
        elif mode == 'iter':
            if not isinstance(args[4], float):
                arguments = list(args[1:4]) + [args[5],]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 + \
                           '{:>11.4e}         -          {:^11.4e}'
            else:
                arguments = args[1:]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 + \
                           '{:>11.4e}     {:^11.4e}    {:^11.4e}'
    elif code == '11':
        mode = args[0]
        if mode == 'max_iter':
            arguments = [args[1],]
            cut_msg = 'Maximum number of iterations ({}) reached without convergence.'
        elif mode == 'su_fail':
            arguments = [args[1]['cluster'], args[1]['mat_phase']]
            cut_msg = 'State update failure in cluster {} (material phase {}).'
        elif mode == 'max_scs_iter':
            arguments = [args[1],]
            cut_msg = 'Maximum number of self-consistent iterations ({}) reached ' + \
                      'without' + '\n' + \
                      indent + len('Increment cut: ')*' ' + 'convergence.'
        else:
            cut_msg = 'Undefined increment cut message.'
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.RED + indent + asterisk_line[:-len(indent)] + \
                   '\n' + \
                   indent + 'Increment cut: ' + colorama.Style.RESET_ALL + cut_msg + \
                   '\n' + colorama.Fore.RED + indent + asterisk_line[:-len(indent)] + \
                   colorama.Style.RESET_ALL + '\n'
    elif code == '12':
        arguments = [args[0],]
        info = tuple(arguments)
        template = '\n\n' + \
                   indent + 'Adaptive clustering step: {:3d}' + '\n' + \
                   indent + tilde_line[:-len(indent)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display information
    ioutil.print2(template.format(*info, width=output_width))
