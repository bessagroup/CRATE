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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display error
    print(template_header.format(*header,width=output_width))
    print(template.format(*error,width=output_width))
    print(template_footer.format(*footer,width=output_width))
    # Abort program
    sys.exit(1)

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
    print(template_header.format(*header,width=output_width))
    print(template.format(*exception,width=output_width))
    print(template_footer.format(*footer,width=output_width))
    # Abort program
    sys.exit(1)
