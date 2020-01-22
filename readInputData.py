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
