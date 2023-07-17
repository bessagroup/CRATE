"""I/O utility tools.

This module includes the '.screen' file path global definition as well as
several useful tools associated with input/output procedures.

Functions
---------
print2(*objects)
    Double output printer.
setdisplayfeatures()
    Set output display features.
escapeANSI(string)
    Remove ANSI escape sequences from string.
checknumber(x)
    Check if instance is or represents a number.
checkposint(x)
    Check if instance is a positive integer.
checkvalidname(x)
    Check if string contains only letters, numbers or underscores.
is_between(x, lower_bound=0, upper_bound=1)
    Check if numeric instance is between lower and upper values (included).
query_yn(question, default_answer='yes')
    Prompt the user to answer yes/no question.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
import re
# Third-party
import numpy as np
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
# Set '.screen' file path as a global variable
screen_file_path = None
# =============================================================================
def print2(*objects):
    """Double output printer.

    Output to both default standard output device (e.g., terminal) and to the
    '.screen' output file.

    Parameters
    ----------
    objects : list
        Objects to print.
    """
    # Print to default sys.stdout
    print(*objects)
    # Print to '.screen' file
    global screen_file_path
    if screen_file_path is not None:
        screen_file = open(screen_file_path, 'a', encoding='utf-8')
        objects_esc = list()
        for i in range(len(objects)):
            objects_esc.append(escapeANSI(objects[i]))
        print(*objects_esc, file=screen_file)
        screen_file.close()
# =============================================================================
def setdisplayfeatures():
    """Set output display features.

    Returns
    -------
    display_features : tuple
        Output display features:

        * output_width (int) : \
            Maximum line length of '.screen' output file.
        * dashed_line (str) : \
            Dashed line of length `output_width`.
        * indent (str) : \
            Indentation spacing.
        * asterisk_line (str) : \
            Asterisks line of length `output_width`.
        * tilde_line (str) : \
            Tildes line of length `output_width`.
        * equal_line (str) : \
            Tildes line of length `output_width`.
    """
    # Set display features
    output_width = 92
    dashed_line = '-'*output_width
    indent = '  '
    asterisk_line = '*'*output_width
    tilde_line = '~'*output_width
    equal_line = '='*output_width
    # Build display features
    display_features = (output_width, dashed_line, indent, asterisk_line,
                        tilde_line, equal_line)
    # Return
    return display_features
# =============================================================================
def escapeANSI(string):
    """Remove ANSI escape sequences from string.

    Parameters
    ----------
    string : str
        String.
    """
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', string)
# =============================================================================
def checknumber(x):
    """Check if instance is or represents a number.

    Parameters
    ----------
    x
        Object.

    Returns
    -------
    is_number : bool
        `True` if `x` is or represents a number, `False` otherwise.
    """
    is_number = True
    try:
        float(x)
        return is_number
    except Exception:
        is_number = False
        return is_number
# =============================================================================
# Check if a given instance is a positive integer
def checkposint(x):
    """Check if instance is a positive integer.

    Parameters
    ----------
    x
        Object.

    Returns
    -------
    is_posint : bool
        `True` if `x` is a positive integer, `False` otherwise.
    """
    is_posint = True
    if isinstance(x, int) or isinstance(x, np.integer):
        if x <= 0:
            is_posint = False
    elif not re.match('^[1-9][0-9]*$', str(x)):
        is_posint = False
    return is_posint
# =============================================================================
def checkvalidname(x):
    """Check if string contains only letters, numbers or underscores.

    Parameters
    ----------
    x : str
        String.

    Returns
    -------
    is_valid : bool
        `True` if `x` contains only letters, numbers or underscores, `False`
        otherwise.
    """
    is_valid = True
    if not re.match('^[A-Za-z0-9_]+$', str(x)):
        is_valid = False
    return is_valid
# =============================================================================
def is_between(x, lower_bound=0, upper_bound=1):
    """Check if numeric instance is between lower and upper values (included).

    Parameters
    ----------
    x : {int, float}
        Numerical type instance.
    lower_bound : {int, float}, default=0
        Lower boundary value (included).
    upper_bound : {int, float}, default=1
        Upper boundary value (included).

    Returns
    -------
    bool : bool
        `True` if numeric instance is between lower and upper values, `False`
        otherwise.
    """
    try:
        x = float(x)
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
    except ValueError:
        print('Instance and bounds must be of numeric type.')
        raise
    if lower_bound > upper_bound:
        raise RuntimeError('Lower boundary value (' + str(lower_bound)
                           + ') must be greater or equal than the upper '
                           'boundary value (' + str(upper_bound) + ').')
    if x >= lower_bound and x <= upper_bound:
        return True
    else:
        return False
# =============================================================================
def query_yn(question, default_answer='yes'):
    """Prompt the user to answer yes/no question.

    Parameters
    ----------
    question : str
        Yes/No question.
    default_answer : {'yes', 'no'}, default='yes'
        Default answer to yes/no question.

    Returns
    -------
    bool : bool
        `True` if answer if 'yes', `False` if anwers is 'no'.
    """
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
            print2('Please answer with \'yes\' or \'no\' (or \'exit\' to '
                   'quit).')
# =============================================================================
def useraction(message):
    """Prompt user to perform action before proceeding execution.

    Parameters
    ----------
    message : str
        Prompt message.
    """
    option = input(message)
    screen_file = open(screen_file_path, 'a')
    print(message + option, file=screen_file)
    screen_file.close()
    if str(option).lower() == 'exit':
        print2('\nProgram aborted.\n')
        sys.exit(1)
