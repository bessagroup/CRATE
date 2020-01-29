#
# Errors and Warnings Module (UNNAMED Program)
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
# Display messages
import info
#
#                                                                    Display errors function
# ==========================================================================================
# Set and display runtime errors
def displayError(code,*args):
    # Get display features
    displayFeatures = info.setDisplayFeatures()
    output_width, dashed_line, indent, asterisk_line = displayFeatures[0:4]
    # Set error display header and footer
    header = tuple(info.convertIterableToList(['!! Error !!',code]) + \
                  info.convertIterableToList(args[0:2]))
    template_header = '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + 'Code: {}' + '\n\n' + \
                      'Traceback: {} (at line {})' + '\n\n'
    footer = tuple(info.convertIterableToList(('Program Aborted',)))
    template_footer = '\n' + asterisk_line + '\n\n' + \
                      '{:^{width}}' + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set errors to display
    if code == 'E00001':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file hasn\'t been specified or found.' + \
                   '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'Specify the input data file (with mandatory \'.dat\'' + \
                   'extension) by calling the program' + '\n' + \
                   indent + 'with the following command line' + '\n\n' + \
                   indent + 'pythonX.X SCA.py < input_data_file_path >'
    elif code == 'E00002':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file name must only contain letters, ' + \
                   'numbers or underscores.'
    elif code == 'E00003':
        arguments = info.convertIterableToList([args[2],])
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been found in the input data file.'
    elif code == 'E00004':
        arguments = 3*[args[2],]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input ' + '\n' + \
                   indent + 'data file. The keyword specification is either missing, ' + \
                   'provided in a wrong format or ' + '\n' + \
                   indent + 'is not a non-negative floating-point number.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified as ' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '< value >'
    elif code == 'E00005':
        arguments = info.convertIterableToList(args[2:4]) + \
                    info.convertIterableToList(2*(args[2],))
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the header of the material ' + \
                   'phase {} is missing, misplaced or \n' + \
                   indent + 'the number of associated properties ' + \
                   'hasn\'t been specified as a positive integer.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '1 2' + '\n' + \
                   indent + '< property1_name > < value >' + '\n' + \
                   indent + '< property2_name > < value >' + \
                   '\n' + \
                   indent + '2 3' + '\n' + \
                   indent + '< property1_name > < value >' + \
                   '\n' + \
                   indent + '< property2_name > < value >' + \
                   '\n' + \
                   indent + '< property3_name > < value >' + '\n'
    elif code == 'E00006':
        arguments = info.convertIterableToList(args[2:5]) + \
                    info.convertIterableToList(2*(args[2],))
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the {}th property of material ' + \
                   'phase {} is not properly ' + '\n' + \
                   indent + 'specified potentially due to one of the following reasons:' + \
                   '\n\n' + \
                   indent + '1. Missing property specification;' + '\n' + \
                   indent + '2. Property name can only contain letters, numbers or ' + \
                   'underscores;' + '\n' + \
                   indent + '3. Invalid property value (e.g must be integer or ' + \
                   'floating-point number);' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '1 2' + '\n' + \
                   indent + '< property1_name > < value >' + '\n' + \
                   indent + '< property2_name > < value >' + \
                   '\n' + \
                   indent + '2 3' + '\n' + \
                   indent + '< property1_name > < value >' + \
                   '\n' + \
                   indent + '< property2_name > < value >' + \
                   '\n' + \
                   indent + '< property3_name > < value >' + '\n'
    elif code == 'E00007':
        arguments = info.convertIterableToList(3*(args[2],))
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the option or value ' + \
                   'specification is missing, the option ' + '\n' + \
                   indent + 'does not exist or the value is not a positive integer.' + \
                   '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified as' + '\n\n' + \
                   indent + '{} < option or value >'
    elif code == 'E00008':
        arguments = info.convertIterableToList(args[2:4]) + \
                    info.convertIterableToList(2*(args[2],))
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the {}th component is not ' + \
                   'properly specified potentially' + '\n' + \
                   indent + 'due to one of the following reasons: \n\n' + \
                   indent + '1. Missing descriptor specification;' + '\n' + \
                   indent + '2. Component name can only contain letters, numbers or ' + \
                   'underscores;' + '\n' + \
                   indent + '3. Invalid component value (e.g must be integers or ' + \
                   'floating-point number);' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. in a 2D ' + \
                   'problem as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '< descriptor_name_11 > < value >' + '\n' + \
                   indent + '< descriptor_name_21 > < value >' + '\n' + \
                   indent + '< descriptor_name_12 > < value >' + '\n' + \
                   indent + '< descriptor_name_22 > < value > < value >'
    elif code == 'E00009':
        arguments = info.convertIterableToList(args[2:4])
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'It was requested the consideration of the previously ' + \
                   'computed offline stage data files ' + '\n' + \
                   indent + 'for the problem \'{}\', but the associated subdirectory ' + \
                   'is missing:' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00010':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file must have \'.dat\' extension.'
    elif code == 'E00011':
        arguments = 3*[args[2],]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file, potentially due to one of the following reasons:' +\
                   '\n\n' + \
                   indent + '1. Missing specification;' + '\n' + \
                   indent + '2. Number of prescribed components is either insufficient ' + \
                   'or excessive;' + '\n' + \
                   indent + '3. Prescription must be either 0 (strain component) or 1 ' + \
                   '(stress component);' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. in a 2D ' + \
                   'problem as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '0 0 1 0'
    elif code == 'E00012':
        component = ['12','13','23']
        arguments = [component[args[2]],component[args[2]][::-1]]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
        indent + 'A different kind of macroscale prescription (strain or stress) has ' + \
        'been specified for ' + '\n' + \
        indent + 'the two dependent components {} and {} under a small strain ' + \
        'formulation.' + '\n' + \
        indent + 'Please prescribe both components as either macroscale strain or ' + \
        'macroscale stress.'
    elif code == 'E00013':
        arguments = info.convertIterableToList(args[2:4] + 2*(args[2],))
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the clustering discretization of the ' + \
                   'material phase {} is missing, ' + '\n' + \
                   indent + 'misplaced or has an invalid format ' + \
                   '(< phase_id > < number_of_clusters >).' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as ' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '1 10' + '\n' + \
                   indent + '2 15'
    elif code == 'E00014':
        arguments = args[2:4]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the spatial ' + \
                   'discretization file is not an absolute ' + '\n' + \
                   indent + 'path (mandatory) or does not exist.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00015':
        n_valid_exts = len(args[3])
        arguments = [args[2],] + info.convertIterableToList(args[3])
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the spatial ' + \
                   'discretization file has an invalid' + '\n' + \
                   indent + 'extension.' + '\n\n' + \
                   indent + 'Valid extensions:' + n_valid_exts*' \'{}\''
    elif code == 'E00016':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The large strain formulation option has not been ' + \
                   'implemented yet!'
    elif code == 'E00017':
        if args[2] == 2:
            type = 'plane stress'
        elif args[2] == 3:
            type = 'axisymmetric'
        arguments = [type,]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The program cannot deal with {} problems yet!'
    elif code == 'E00018':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Tensor to matrix conversions are only available for ' + \
                   'second-order and fourth-order ' + '\n' + \
                   indent + 'tensors.'
    elif code == 'E00019':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Tensor to matrix conversions are only available for ' + \
                   'second-order and fourth-order tensors ' + '\n' + \
                   indent + 'with the same size in each dimension, which must be 2 or 3.'
    elif code == 'E00020':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Matrix to tensor conversions are only available for ' + \
                   'vector or square matrix matricial ' + '\n' + \
                   indent + 'forms.'
    elif code == 'E00021':
        arguments = ['',]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Matrix to tensor conversions are only available for ' + \
                   'vector or square matrix with size 3, ' + '\n' + \
                   indent + '4, 6 or 9 in each dimension.'
    elif code == 'E00022':
        arguments = args[2:4]
        error = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The elastic property - {} - of material phase {} hasn\'t ' + \
                   'been specified or has been ' + '\n' + \
                   indent + 'specified more than once in the input data file.'
    #print(template_header.format(*header,width=output_width))
    #print(template.format(*error,width=output_width))
    #print(template_footer.format(*footer,width=output_width))
    #sys.exit(1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display error
    info.print2(template_header.format(*header,width=output_width))
    info.print2(template.format(*error,width=output_width))
    info.print2(template_footer.format(*footer,width=output_width))
    # Abort program
    sys.exit(1)
#
#                                                                  Display warnings function
# ==========================================================================================
# Set and display runtime warnings
def displayWarning(code,*args):
    # Get display features
    displayFeatures = info.setDisplayFeatures()
    output_width, dashed_line, indent, asterisk_line = displayFeatures[0:4]
    # Set error display header and footer
    header = tuple(info.convertIterableToList(['!! Warning !!',code]) + \
                  info.convertIterableToList(args[0:2]))
    template_header = '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + 'Code: {}' + '\n\n' + \
                      'Traceback: {} (at line {})' + '\n\n'
    footer = tuple([' '])
    template_footer = '\n' + asterisk_line + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set warnings to display
    if code == 'W00001':
        if args[2] == 1:
            arguments = ['strain',]
        elif args[2] == 2:
            arguments = ['stress',]
        elif args[2] == 3:
            if args[3] == 0:
                arguments = ['strain',]
            elif args[3] == 1:
                arguments = ['stress',]
        warning = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'A non-symmetric macroscale {} tensor was prescribed under ' + \
                   'a small strain formulation.' + '\n' + \
                   indent + 'The symmetric value with the lowest first index is ' + \
                   'enforced (p.e. if descriptor_12 is ' + '\n' + \
                   indent + 'different from descriptor_21, then descriptor_12 is enforced).'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display warning
    info.print2(template_header.format(*header,width=output_width))
    info.print2(template.format(*warning,width=output_width))
    info.print2(template_footer.format(*footer,width=output_width))
#
#                                                       Display built-in exceptions function
# ==========================================================================================
# Display runtime built-in exceptions
def displayException(*args):
    # Get display features
    displayFeatures = info.setDisplayFeatures()
    output_width, dashed_line, indent, asterisk_line = displayFeatures[0:4]
    # Set built-in exception display header and footer
    header = tuple(['!! Built-In Exception !!'] + \
                  info.convertIterableToList(args[0:2]))
    template_header = '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + \
                      'Traceback: {} (at line {})' + '\n\n'
    footer = tuple(info.convertIterableToList(('Program Aborted',)))
    template_footer = '\n' + asterisk_line + '\n\n' + \
                      '{:^{width}}' + '\n'
    # Set built-in exception to display
    arguments = info.convertIterableToList([args[2],])
    exception = tuple(arguments)
    template = 'Details:' + '\n\n' + \
                indent + '{}'
    # Display built-in exception
    info.print2(template_header.format(*header,width=output_width))
    info.print2(template.format(*exception,width=output_width))
    info.print2(template_footer.format(*footer,width=output_width))
    # Abort program
    sys.exit(1)
