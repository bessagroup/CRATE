"""Display information during program execution.

This module includes a function that allows the output of information to both
default standard output device (e.g., terminal) and to the '.screen' output
file in a formatted and consistent manner. Some output codes may precede a
program abortion.

Functions
---------
displayinfo
    Display information during program execution.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import copy
# Third-party
import numpy as np
import colorama
# Local
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
def displayinfo(code, *args, **kwargs):
    """Display information during program execution.

    Information is output to both default standard output device
    (e.g., terminal) and to the '.screen' output file.

    ----

    Parameters
    ----------
    code : str
        Code associated with the output information:

        * -1 : Setting problem directory
        *  0 : Program execution launched
        *  1 : Program execution completed
        *  2 : Execution phase started
        *  3 : Execution phase completed
        *  4 : Program aborted
        *  5 : Execution phase task
        *  6 : Clustering analysis and completion
        *  7 : Loading increment header and footer
        *  8 : Self-consistent scheme iteration header and footer
        *  9 : Newton-Raphson header and iteration
        * 11 : Loading increment cut
        * 12 : Adaptive clustering step header
        * 13 : Self-consistent scheme iteration header and footer \
               (locked reference material properties)
        * 14 : Locking reference material properties (self-consistent \
               scheme convergence failure)
        * 15 : Clustering adaptivity step summary
        * 16 : Clustering adaptivity step triggered
        * 17 : Adaptive clustering solution rewinding
        * 18 : Writing increment VTK output file
    """
    # Get display features
    display_features = ioutil.setdisplayfeatures()
    output_width, dashed_line, indent, asterisk_line = display_features[0:4]
    tilde_line, equal_line = display_features[4:6]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set informations and formats to display
    if code == '-1':
        status = args[1]
        arguments = [args[0], ]
        info = tuple(arguments)
        if status == 0:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' \
                + 'Status: New problem'
        elif status == 1:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' \
                + 'Status: Repeating problem (considering existent ' \
                + 'offline stage)'
        elif status == 2:
            template = 4*'\n' + 'Problem directory: {}' + '\n\n' \
                + 'Status: New problem (overwriting existing directory)'
        elif status == 3:
            ioutil.print2('Please rerun the program and provide a different '
                          'problem name.' + '\n')
            sys.exit(1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '0':
        arguments = \
            ['CRATE - Clustering-based Nonlinear Analysis of Materials',
             'Created by Bernardo P. Ferreira', 'Release 1.0.4 (Oct 2023)'] \
            + 2*[args[0], ] + list(args[1:3])
        info = tuple(arguments)
        template = '\n' + colorama.Fore.WHITE + tilde_line \
                   + colorama.Style.RESET_ALL \
                   + colorama.Fore.WHITE + '\n{:^{width}}\n\n' \
                   + '{:^{width}}\n' + '\n{:^{width}}\n\n' \
                   + colorama.Fore.YELLOW + 'Problem under analysis: ' \
                   + colorama.Style.RESET_ALL + '{}' + '\n\n' \
                   + colorama.Fore.YELLOW + 'Input data file: ' \
                   + colorama.Style.RESET_ALL + '{}.dat' + '\n\n' \
                   + colorama.Fore.YELLOW + 'Starting program execution at: ' \
                   + colorama.Style.RESET_ALL + '{} ({})\n' \
                   + colorama.Fore.WHITE + tilde_line \
                   + colorama.Style.RESET_ALL + '\n\n' + colorama.Fore.WHITE \
                   + dashed_line + colorama.Style.RESET_ALL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '1':
        phase_names = args[3]
        phase_times = args[4]
        total_time = phase_times[0, 1] - phase_times[0, 0]
        number_of_phases = len(phase_names)
        phase_durations = [phase_times[i, 1] - phase_times[i, 0]
                           for i in range(0, number_of_phases)]
        for i in range(0, number_of_phases):
            phase_durations.insert(3*i, phase_names[i])
            phase_durations.insert(3*i + 2,
                                   (phase_durations[3*i + 1]/total_time)*100)
        arguments = list(args[0:3]) \
            + [total_time, np.floor(total_time/3600),
               (total_time % 3600)/60] + ['Phase', 'Duration (s)', '%'] \
            + phase_durations[3:] \
            + [colorama.Fore.GREEN + 'Program Completed'
               + colorama.Style.RESET_ALL]
        info = tuple(arguments)
        template = '\n' + colorama.Fore.WHITE + tilde_line \
            + colorama.Style.RESET_ALL + '\n' \
            + colorama.Fore.YELLOW + 'Ending program execution at: ' \
            + colorama.Style.RESET_ALL + '{} ({})\n\n' \
            + colorama.Fore.YELLOW + 'Problem analysed: ' \
            + colorama.Style.RESET_ALL + '{}\n\n' \
            + colorama.Fore.YELLOW + 'Total execution time: ' \
            + colorama.Style.RESET_ALL \
            + '{:.2e}s (~{:.0f}h{:.0f}m)\n\n' \
            + colorama.Fore.YELLOW + 'Execution times: \n\n' \
            + colorama.Style.RESET_ALL \
            + 2*indent + '{:50}{:^20}{:^5}' + '\n' \
            + 2*indent + 75*'-' + '\n' \
            + (2*indent + '{:50}{:^20.2e}{:>5.2f} \n')*(number_of_phases - 1) \
            + 2*indent + 75*'-' + '\n\n\n' + '{:^{width}}' + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '2':
        arguments = [args[0], ]
        info = tuple(arguments)
        template = colorama.Fore.GREEN + 'Start phase: ' \
            + colorama.Fore.WHITE \
            + '{} \n' + colorama.Style.RESET_ALL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '3':
        arguments = args[0:2]
        info = tuple(arguments)
        template = colorama.Fore.GREEN + '\n\nEnd phase: ' \
            + colorama.Fore.WHITE + '{} (phase duration time = {:.2e}s)\n' \
            + dashed_line + colorama.Style.RESET_ALL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '4':
        summary = args[0]
        description = args[1]
        arguments = list(args[2:])
        arguments.append('Program Aborted')
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.RED \
            + indent + asterisk_line[:-len(indent)] + '\n' \
            + indent + summary \
            + '\n\n' + colorama.Style.RESET_ALL \
            + indent + description \
            + '\n' + colorama.Fore.RED \
            + indent + asterisk_line[:-len(indent)] + '\n\n' \
            + '{:^{width}}' + colorama.Style.RESET_ALL + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '5':
        if len(args) == 2:
            n_indents = args[1]
        else:
            n_indents = 1
        arguments = [args[0], ]
        info = tuple(arguments)
        template = '\n' + n_indents*indent + '> {}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '6':
        mode = args[0]
        if mode == 'progress':
            arguments = args[1:3]
            if args[1] == 1:
                print(' ')
            info = tuple(arguments)
            template = indent + '> Performing clustering process {} of {}...'
            print(template.format(*info, width=output_width), end='\r')
            if args[1] == args[2]:
                print(' ')
            return
        elif mode == 'completed':
            arguments = ['', ]
            info = tuple(arguments)
            template = '\n' + indent + '> Completed all clustering processes!'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '7':
        mode = args[0]
        if mode == 'init':
            subinc_level = args[2]
            if subinc_level == 0:
                arguments = list([args[1], ]) + list(args[3:])
                info = tuple(arguments)
                template = colorama.Fore.CYAN + '\n' \
                    + indent + 'Increment number: {:3d}' + '\n' \
                    + indent + equal_line[:-len(indent)] + '\n' \
                    + indent + 'Loading subpath: {:4d} |' + 6*' ' \
                    + 'Load factor | Total = {:8.1e}' + 7*' ' \
                    + 'Time | Total = {:8.1e}' + '\n' \
                    + indent + 6*' ' + 'Increment: {:4d} |' + 18*' ' \
                    + '| Incr. = {:8.1e}' + 12*' ' + '| Incr. = {:8.1e}' \
                    + colorama.Style.RESET_ALL
            else:
                arguments = args[1:]
                info = tuple(arguments)
                template = colorama.Fore.CYAN + '\n' \
                    + indent + 'Increment number: {:3d}' + 3*' ' \
                    + '(Sub-inc. level: {:3d})' + '\n' \
                    + indent + equal_line[:-len(indent)] + '\n' \
                    + indent + 'Loading subpath: {:4d} |' + 6*' ' \
                    + 'Load factor | Total = {:8.1e}' + 7*' ' \
                    + 'Time | Total = {:8.1e}' + '\n' \
                    + indent + 6*' ' + 'Increment: {:4d} |' + 18*' ' \
                    + '| Incr. = {:8.1e}' + 12*' ' + '| Incr. = {:8.1e}' \
                    + colorama.Style.RESET_ALL
        elif mode == 'end':
            space1 = (output_width - 84)*' '
            space2 = (output_width
                      - (len('Homogenized strain tensor') + 48))*' '
            space3 = (output_width
                      - (len('Increment run time (s): ') + 44))*' '
            strain_formulation = args[1]
            problem_type = args[2]
            hom_strain = args[3]
            hom_stress = args[4]
            hom_strain_out = np.zeros((3, 3))
            hom_stress_out = np.zeros((3, 3))
            if problem_type == 1:
                hom_strain_out[0:2, 0:2] = hom_strain
                if strain_formulation == 'infinitesimal':
                    hom_strain_out[2, 2] = 0.0
                else:
                    hom_strain_out[2, 2] = 1.0
                hom_stress_out[0:2, 0:2] = hom_stress
                hom_stress_out[2, 2] = args[7]
            else:
                hom_strain_out = copy.deepcopy(hom_strain)
                hom_stress_out = copy.deepcopy(hom_stress)
            if strain_formulation == 'infinitesimal':
                strain_symbol = '\u03B5'
                stress_symbol = '\u03C3'
            else:
                strain_symbol = 'F'
                stress_symbol = 'P'
            arguments = list()
            for i in range(3):
                for j in range(3):
                    arguments.append(hom_strain_out[i, j])
                for j in range(3):
                    arguments.append(hom_stress_out[i, j])
            arguments = arguments + [args[5], args[6]]
            info = tuple(arguments)
            template = '\n\n' \
                + indent + equal_line[:-len(indent)] + '\n' \
                + indent + 7*' ' + 'Homogenized strain tensor (' \
                + strain_symbol + ')' + space2 \
                + 'Homogenized stress tensor (' + stress_symbol + ')' \
                + '\n\n' \
                + indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 \
                + '[' + 3*'{:>12.4e}' + '  ]' + '\n' \
                + indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 \
                + '[' + 3*'{:>12.4e}' + '  ]' + '\n' \
                + indent + ' [' + 3*'{:>12.4e}' + '  ]' + space1 \
                + '[' + 3*'{:>12.4e}' + '  ]' + '\n' \
                + '\n' + indent + equal_line[:-len(indent)] + '\n' \
                + indent + 'Increment run time (s): {:>11.4e}' + space3 \
                + 'Total run time (s): {:>11.4e}' + '\n\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            template = colorama.Fore.YELLOW + '\n\n' + indent \
                + 'Self-consistent scheme iteration: {:3d}' + '\n' \
                + indent + tilde_line[:-len(indent)] + '\n' \
                + indent + 'Young modulus (E): ' + format_1 \
                + '  (norm. change: ' + format_2 + ')' + '\n' \
                + indent + 'Poisson ratio (\u03BD): ' + format_1 \
                + '  (norm. change: ' + format_2 + ')' + '\n' \
                + indent + tilde_line[:-len(indent)] + '\n' \
                + colorama.Style.RESET_ALL
        elif mode == 'end':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + dashed_line[:-len(indent)] + '\n\n' \
                + indent + tilde_line[:-len(indent)] + '\n' \
                + indent + 'Iteration run time (s): {:>11.4e}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '9':
        mode = args[0]
        space1 = (output_width - 48)*' '
        space2 = (output_width - 67)*' '
        if mode == 'init':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + 5*' ' + 'Iteration' + space1 \
                + 'Normalized residuals' + '\n' \
                + indent + ' Number    Run time (s)' + space2 \
                + 'Equilibrium    Mac. strain    Mac. stress' + '\n' \
                + indent + dashed_line[:-len(indent)]
        elif mode == 'iter':
            if not isinstance(args[4], float):
                arguments = list(args[1:4]) + [args[5], ]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 \
                    + '{:>11.4e}         -          {:^11.4e}'
            elif not isinstance(args[5], float):
                arguments = args[1:5]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 \
                    + '{:>11.4e}     {:^11.4e}        -'
            else:
                arguments = args[1:]
                info = tuple(arguments)
                template = indent + ' {:^6d}    {:^12.4e}' + space2 \
                    + '{:>11.4e}     {:^11.4e}    {:^11.4e}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '10':
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '11':
        mode = args[0]
        if mode == 'max_iter':
            arguments = [args[1], ]
            cut_msg = ('Maximum number of iterations ({}) reached without '
                       'convergence.')
        elif mode == 'su_fail':
            arguments = [args[1]['cluster'], args[1]['mat_phase']]
            cut_msg = 'State update failure in cluster {} (material phase {}).'
        elif mode == 'max_scs_iter':
            arguments = [args[1], ]
            cut_msg = 'Maximum number of self-consistent iterations ({}) ' \
                + 'reached without' + '\n' \
                + indent + len('Increment cut: ')*' ' + 'convergence.'
        else:
            cut_msg = 'Undefined increment cut message.'
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.RED \
            + indent + asterisk_line[:-len(indent)] + '\n' \
            + indent + 'Increment cut: ' + colorama.Style.RESET_ALL + cut_msg \
            + '\n' + colorama.Fore.RED \
            + indent + asterisk_line[:-len(indent)] \
            + colorama.Style.RESET_ALL + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '12':
        arguments = [args[0], ]
        info = tuple(arguments)
        template = '\n\n' \
            + indent + 'Adaptive clustering step: {:3d}' + '\n' \
            + indent + tilde_line[:-len(indent)]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '13':
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
            template = colorama.Fore.YELLOW \
                + '\n\n' + indent + 'Self-consistent scheme iteration: {:3d}' \
                + '\n' + indent + tilde_line[:-len(indent)] + '\n' \
                + indent + 'Young modulus (E): ' + format_1 \
                + '  (norm. change: ' + format_2 + ')' + ' \U0001F512' + '\n' \
                + indent + 'Poisson ratio (\u03BD): ' + format_1 \
                + '  (norm. change: ' + format_2 + ')' + ' \U0001F512' + '\n' \
                + indent + tilde_line[:-len(indent)] + '\n' \
                + colorama.Style.RESET_ALL
        elif mode == 'end':
            arguments = args[1:]
            info = tuple(arguments)
            template = indent + dashed_line[:-len(indent)] \
                + '\n\n' + indent + tilde_line[:-len(indent)] + '\n' \
                + indent + 'Iteration run time (s): {:>11.4e}' + ' \U0001F512'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '14':
        mode = args[0]
        if mode == 'max_scs_iter':
            arguments = [args[1], ]
            lock_msg = 'Maximum number of self-consistent iterations ({}) ' \
                + 'reached' + '\n' \
                + indent + len('Locking reference properties: ')*' ' \
                + 'without convergence.' + '\n' \
                + indent + len('Locking reference properties: ')*' ' \
                + 'Performing one last self-consistent scheme iteration ' \
                + 'with ' + '\n' + indent \
                + len('Locking reference properties: ')*' ' \
                + 'the last converged increment reference material elastic' \
                + '\n' + indent + len('Locking reference properties: ')*' ' \
                + 'properties.'
        elif mode == 'inadmissible_scs_solution':
            arguments = []
            lock_msg = 'Inadmissible self-consistent scheme iterative ' \
                'solution.' + '\n' \
                + indent + len('Locking reference properties: ')*' ' \
                + 'Performing one last self-consistent scheme iteration ' \
                + 'with ' + '\n' \
                + indent + len('Locking reference properties: ')*' ' \
                + 'the last converged increment reference material elastic ' \
                + '\n' + indent + len('Locking reference properties: ')*' ' \
                + 'properties.'
        elif mode == 'locked_scs_solution':
            arguments = []
            lock_msg = 'Inadmissible self-consistent scheme iterative ' \
                + 'solution.' + '\n' \
                + indent + len('Locking reference properties: ')*' ' \
                + 'Accepting solution with the last converged increment ' \
                + '\n' + indent + len('Locking reference properties: ')*' ' \
                + 'reference material elastic properties.'
        else:
            lock_msg = 'Undefined locking message.'
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.YELLOW \
            + indent + asterisk_line[:-len(indent)] + '\n' \
            + indent + 'Locking reference properties: ' \
            + colorama.Style.RESET_ALL + lock_msg + '\n' \
            + colorama.Fore.YELLOW + indent + asterisk_line[:-len(indent)] \
            + colorama.Style.RESET_ALL + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '15':
        # Get adaptivity manager and CRVE
        adaptivity_manager = args[0]
        crve = args[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get CRVE clustering summary
        clustering_summary = crve.get_clustering_summary()
        # Sort and get number of material phases
        mat_phases = list(clustering_summary.keys())
        mat_phases.sort()
        n_mat_phases = len(clustering_summary.keys())
        # Build output material phases clusters list and get the toal number of
        # base and final clusters
        output_clusters = []
        base_n_clusters = 0
        final_n_clusters = 0
        for mat_phase in mat_phases:
            output_clusters += [mat_phase, ] + clustering_summary[mat_phase]
            base_n_clusters += clustering_summary[mat_phase][1]
            final_n_clusters += clustering_summary[mat_phase][2]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get total clustering adaptivity procedures time and total online
        # stage time
        total_time_adapt = adaptivity_manager.adaptive_time
        total_time_os = args[2]
        # Set output phases designations
        time_phases = ['Select clustering adaptivity target clusters',
                       'Perform CRVE clustering adaptivity',
                       'Compute CRVE cluster interaction tensors',
                       'Other']
        n_time_phases = len(time_phases)
        # Set output phases times
        adapt_times = [adaptivity_manager.adaptive_evaluation_time,
                       crve.get_adaptive_clustering_time(),
                       crve.get_adaptive_cit_time()]
        adapt_times.append(total_time_adapt - sum(adapt_times))
        # Set output phases relative times
        if total_time_adapt > 1e-10:
            adapt_times_rel_1 = [(time/total_time_adapt)*100
                                 for time in adapt_times]
        else:
            adapt_times_rel_1 = [0.0 for time in adapt_times]
        if total_time_os > 1e-10:
            adapt_times_rel_2 = [(time/total_time_os)*100
                                 for time in adapt_times]
        else:
            adapt_times_rel_2 = [0.0 for time in adapt_times]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build output phases times list
        output_times = []
        for i in range(len(time_phases)):
            output_times += [time_phases[i], adapt_times[i],
                             adapt_times_rel_1[i], adapt_times_rel_2[i]]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Build output structure
        arguments = ['Mat. phase', 'Type', 'Base clusters', 'Final clusters',
                     *output_clusters,
                     'Total', base_n_clusters, final_n_clusters,
                     'Phase', 'Duration (s)', '% adapt', '% total',
                     *output_times, 'Total', total_time_adapt,
                     ]
        info = tuple(arguments)
        template = '\n\n' \
            + indent + 'Clustering adaptivity summary:' + '\n\n' \
            + 2*indent + '{:<10s}{:^21s}{:>20s}{:>20s}' + '\n' \
            + 2*indent + dashed_line[:-10*len(indent)] + '\n' \
            + (2*indent + '{:^10s}{:^21s}{:>20d}{:>20d}' + '\n')*n_mat_phases \
            + 2*indent + dashed_line[:-10*len(indent)] + '\n' \
            + 2*indent + '{:<31s}{:>20d}{:>20d}' '\n\n\n' \
            + 2*indent + '{:50s}{:^20s}{:>7s}{:>10s}' + '\n' \
            + 2*indent + dashed_line[:-2*len(indent)] + '\n' \
            + (2*indent + '{:50s}{:^20.2e}{:>7.2f}{:>10.2f}'
               + '\n')*n_time_phases \
            + 2*indent + dashed_line[:-2*len(indent)] + '\n' \
            + 2*indent + '{:50s}{:^20.2e}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '16':
        mode = args[0]
        inc = args[1]
        spacing = indent + len('Clustering adaptivity: ')*' '
        if mode == 'repeat':
            msg = 'Adaptivity condition(s) have been triggered and ' \
                'clustering' + '\n' \
                + spacing + 'adaptivity will be performed.' + '\n' \
                + spacing + 'Current macroscale loading increment ({}) ' \
                + 'will be repeated' + '\n' \
                + spacing + 'considering the new clustering.'
        elif mode == 'new':
            msg = 'Adaptivity condition(s) have been triggered and ' \
                + 'clustering' + '\n' \
                + spacing + 'adaptivity will be performed.' + '\n' \
                + spacing + 'Performing the new macroscale loading ' \
                + 'increment ({})' + '\n' + spacing \
                + 'considering the new clustering.'
        arguments = [inc, ]
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.CYAN + indent \
            + asterisk_line[:-len(indent)] + '\n' \
            + indent + 'Clustering adaptivity: ' + colorama.Style.RESET_ALL \
            + msg + '\n' + colorama.Fore.CYAN + indent \
            + asterisk_line[:-len(indent)] + colorama.Style.RESET_ALL + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '17':
        rewind_inc = args[0]
        spacing = indent + len('Analysis rewind: ')*' '
        msg = 'Rewind condition(s) have been triggered and the analysis ' \
            + 'will be' + '\n' + spacing \
            + 'rewound back to the end of macroscale loading increment {}.' \
            + '\n' \
            + spacing + 'The current CRVE clustering is considered, being ' \
            + 'a suitable' + '\n' \
            + spacing + 'transference of cluster-related variables performed.'
        arguments = [rewind_inc, ]
        info = tuple(arguments)
        template = '\n\n' + colorama.Fore.CYAN \
            + indent + asterisk_line[:-len(indent)] + '\n' \
            + indent + 'Analysis rewind: ' + colorama.Style.RESET_ALL + msg \
            + '\n' + colorama.Fore.CYAN \
            + indent + asterisk_line[:-len(indent)] \
            + colorama.Style.RESET_ALL + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif code == '18':
        msg = 'Writing increment VTK output file...'
        arguments = []
        info = tuple(arguments)
        template = '\n' + colorama.Fore.MAGENTA \
            + indent + asterisk_line[:-len(indent)] + '\n' \
            + indent + 'VTK Output: ' + colorama.Style.RESET_ALL + msg \
            + '\n' + colorama.Fore.MAGENTA \
            + indent + asterisk_line[:-len(indent)] \
            + colorama.Style.RESET_ALL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display information
    ioutil.print2(template.format(*info, width=output_width))
    # Program abortion codes
    if code in ('4',):
        sys.exit(1)
