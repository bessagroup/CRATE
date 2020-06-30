#
# I/O Utilities Module (CRATE Program)
# ==========================================================================================
# Summary:
# Utility procedures related to input and output operations.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | June 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Parse command-line options and arguments
import sys
# Working with arrays
import numpy as np
# Regular expressions
import re
#
#                                                                           Global variables
# ==========================================================================================
# Set '.screen' file path as a global variable
screen_file_path = ''
#
#                                                             Multiple output print function
# ==========================================================================================
# Print to both default standard output device and '.screen' file
def print2(*objects):
    # Print to default sys.stdout
    print(*objects)
    # Print to '.screen' file
    global screen_file_path
    screen_file = open(screen_file_path, 'a')
    objects_esc = list()
    for i in range(len(objects)):
        objects_esc.append(escapeANSI(objects[i]))
    print(*objects_esc, file = screen_file)
    screen_file.close()
#                                                Display features and manipulation functions
# ==========================================================================================
# Set output features
def setdisplayfeatures():
    output_width = 92
    dashed_line = '-'*output_width
    indent = '  '
    asterisk_line = '*'*output_width
    tilde_line = '~'*output_width
    equal_line = '='*output_width
    display_features = (output_width, dashed_line, indent, asterisk_line, tilde_line,
                        equal_line)
    return display_features
#
#                                                                           Format functions
# ==========================================================================================
# Convert iterable to list
def convertiterabletolist(iterable):
    list = [element for element in iterable]
    return list
# ------------------------------------------------------------------------------------------
# Remove ANSI escape sequences from string
def escapeANSI(string):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', string)
#
#                                                                            Check instances
# ==========================================================================================
# Check if a given instance is or represents a number (either integer or floating-point)
def checknumber(x):
    is_number = True
    try:
        float(x)
        return is_number
    except ValueError:
        is_number = False
        return is_number
# ------------------------------------------------------------------------------------------
# Check if a given instance is a positive integer
def checkposint(x):
    is_posint = True
    if isinstance(x, int) or isinstance(x, np.integer):
        if x <= 0:
            is_posint = False
    elif not re.match('^[1-9][0-9]*$', str(x)):
        is_posint = False
    return is_posint
# ------------------------------------------------------------------------------------------
# Check if a given instance contains only letters, numbers or underscores
def checkvalidname(x):
    is_valid = True
    if not re.match('^[A-Za-z0-9_]+$', str(x)):
        is_valid = False
    return is_valid
#
#                                                                      Prompt user functions
# ==========================================================================================
# Prompt the user to answer a 'yes' or 'no' question
def query_yn(question, default_answer='yes'):
    answer = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False}
    if default_answer is None:
        prompt = ' [y/n] '
    elif default_answer == 'yes':
        prompt = ' [Y/n] '
    elif default_answer == 'no':
        prompt = ' [y/N] '
    n_invalid_ans = 0
    while True:
        option = input(question + prompt).lower()
        screen_file = open(screen_file_path, 'a')
        print(question + prompt + option, file=screen_file)
        screen_file.close()
        if default_answer is not None and option == '':
            return answer[default_answer]
        elif option in answer:
            return answer[option]
        else:
            n_invalid_ans = n_invalid_ans + 1
            if n_invalid_ans > 3 or str(option).lower() == 'exit':
                print2('\nProgram aborted.\n')
                sys.exit(1)
            print2('Please answer with \'yes\' or \'no\' (or \'exit\' to quit).')
# ------------------------------------------------------------------------------------------
# Prompt the user to perform some action before proceeding the program execution
def useraction(message):
    option = input(message)
    screen_file = open(screen_file_path, 'a')
    print(message + option, file=screen_file)
    screen_file.close()
    if str(option).lower() == 'exit':
        print2('\nProgram aborted.\n')
        sys.exit(1)
