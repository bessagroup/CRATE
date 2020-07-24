#
# Errors and Warnings Module (CRATE Program)
# ==========================================================================================
# Summary:
# Display execution errors, warnings and built-in exceptions.
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
# Inspect file name and line
import inspect
# Terminal colors
import colorama
# I/O utilities
import ioput.ioutilities as ioutil
#
#                                                                    Display errors function
# ==========================================================================================
# Set and display runtime errors
def displayerror(code, *args):
    # Get display features
    display_features = ioutil.setdisplayfeatures()
    output_width, dashed_line, indent, asterisk_line = display_features[0:4]
    # Set error display header and footer
    error_color = colorama.Fore.RED
    header = tuple(['!! Error !!', code] + list(args[0:2]))
    template_header = error_color + '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + 'Code: ' + \
                      colorama.Style.RESET_ALL + '{}' + '\n\n' + \
                      error_color + 'Traceback: ' + \
                      colorama.Style.RESET_ALL + '{} (at line {})' + '\n\n'
    footer = tuple(['Program Aborted',])
    template_footer = error_color + '\n' + asterisk_line + '\n\n' + \
                      '{:^{width}}' + '\n' + colorama.Style.RESET_ALL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set errors to display
    if code == 'E00001':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file hasn\'t been specified or found.' + \
                   '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'Specify the input data file (with mandatory \'.dat\' ' + \
                   'extension) by calling the program' + '\n' + \
                   indent + 'with the following command line' + '\n\n' + \
                   indent + 'pythonX.X SCA.py < input_data_file_path >'
    elif code == 'E00002':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file name must only contain letters, ' + \
                   'numbers or underscores.'
    elif code == 'E00003':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been found in the input data file.'
    elif code == 'E00004':
        arguments = 3*[args[2],]
        values = tuple(arguments)
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
        suffix = getordinalnumber(args[3])
        arguments = list(args[2:4]) + list(2*[args[2],])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data\n' + \
                   indent + 'file. In particular, the header of the {}' + suffix + \
                   ' material phase is not properly specified ' + '\n' + \
                   indent + 'potentially due to one of the following reasons: \n\n' + \
                   indent + '1. Missing material phase header specification;' + '\n' + \
                   indent + '2. Material phase header specification wrong format;' + \
                   '\n' + \
                   indent + '3. Material phase label must be a positive integer;' + '\n' + \
                   indent + '4. Duplicated material phase label;' + '\n' + \
                   indent + '5. Material phase number of properties must be a positive ' + \
                   'integer;' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + ' < n_material_phases >' + '\n' + \
                   indent + '2 model1_name 2 [ < model_source > ]' + '\n' + \
                   indent + 'property1_name < value >' + '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + '5 model1_name 3 [ < model_source > ]' + '\n' + \
                   indent + 'property1_name < value >' + \
                   '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + 'property3_name < value >'
    elif code == 'E00006':
        suffix = getordinalnumber(args[3])
        arguments = list(args[2:5]) + list(2*[args[2],])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data\n' + \
                   indent + 'file. In particular, the {}' + suffix + ' property of ' + \
                   'material phase {} is not properly ' + '\n' + \
                   indent + 'specified potentially due to one of the following reasons:' + \
                   '\n\n' + \
                   indent + '1. Missing property specification;' + '\n' + \
                   indent + '2. Property specification wrong format;' + '\n' + \
                   indent + '3. Property name can only contain letters, numbers or ' + \
                   'underscores;' + '\n' + \
                   indent + '4. Duplicated material property name;' + '\n' + \
                   indent + '5. Invalid property value (e.g must be integer or ' + \
                   'floating-point number);' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + ' < n_material_phases >' + '\n' + \
                   indent + '2 model1_name 2 [ < model_source > ]' + '\n' + \
                   indent + 'property1_name < value >' + '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + '5 model1_name 3 [ < model_source > ]' + '\n' + \
                   indent + 'property1_name < value >' + \
                   '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + 'property3_name < value >'
    elif code == 'E00007':
        arguments = list(3*[args[2],])
        values = tuple(arguments)
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
        suffix = getordinalnumber(args[3])
        arguments = list(args[2:4]) + list(2*[args[2],])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the {}' + suffix + ' component ' + \
                   'is not properly specified potentially' + '\n' + \
                   indent + 'due to one of the following reasons: \n\n' + \
                   indent + '1. Missing descriptor specification;' + '\n' + \
                   indent + '2. Component name can only contain letters, numbers or ' + \
                   'underscores;' + '\n' + \
                   indent + '3. Number of prescriptions does not match the specified ' + \
                   'number of loading subpaths;' + '\n' + \
                   indent + '4. Invalid component value (must be integer or ' + \
                   'floating-point number);' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. in a 2D ' + \
                   'problem with 2 loading' + '\n' + \
                   indent + 'subpaths as' + '\n\n' + \
                   indent + '{} 2' + '\n' + \
                   indent + 'descriptor_name_11 < value > < value >' + '\n' + \
                   indent + 'descriptor_name_21 < value > < value >' + '\n' + \
                   indent + 'descriptor_name_12 < value > < value >' + '\n' + \
                   indent + 'descriptor_name_22 < value > < value >'
    elif code == 'E00009':
        arguments = list(args[2:4])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'It was requested the consideration of the previously ' + \
                   'computed offline stage data files ' + '\n' + \
                   indent + 'for the problem \'{}\', but the associated subdirectory ' + \
                   'is missing:' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00010':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The input data file must have \'.dat\' extension.'
    elif code == 'E00011':
        suffix = getordinalnumber(args[3])
        arguments = list(args[2:4]) + list(2*[args[2],])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input \n' + \
                   indent + 'data file. In particular, the {}' + suffix + ' component ' + \
                   'is not properly specified potentially' + '\n' + \
                   indent + 'due to one of the following reasons: \n\n' + \
                   indent + '1. Missing specification;' + '\n' + \
                   indent + '2. Prescription must be either 0 (strain component) or 1 ' + \
                   '(stress component);' + '\n' + \
                   indent + '3. Number of prescriptions exceeded the specified number ' + \
                   'of loading subpaths;' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. in a 2D ' + \
                   'problem with 3' + '\n' + \
                   indent + 'loading subpaths as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '0 1 0' + '\n' + \
                   indent + '0 1 1' + '\n' + \
                   indent + '0 1 1' + '\n' + \
                   indent + '0 1 0'
    elif code == 'E00012':
        component = ['12', '13', '23']
        arguments = [component[args[2]], component[args[2]][::-1]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
        indent + 'A different kind of macroscale prescription (strain or stress) has ' + \
        'been specified for ' + '\n' + \
        indent + 'the two dependent components {} and {} under a small strain ' + \
        'formulation.' + '\n' + \
        indent + 'Please prescribe both components as either macroscale strain or ' + \
        'macroscale stress.'
    elif code == 'E00013':
        suffix = getordinalnumber(args[3])
        arguments = list(args[2:4]) + list(2*[args[2],])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the clustering discretization of the {}' + \
                   suffix + ' material phase is missing, ' + '\n' + \
                   indent + 'misplaced or has an invalid format ' + \
                   '(< phase_id > < number_of_clusters >).' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. in a ' + \
                   'material with two' + '\n' + \
                   indent + 'phases (3 and 5) as ' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '3 10' + '\n' + \
                   indent + '5 15'
    elif code == 'E00014':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the spatial ' + \
                   'discretization file does not exist or' + '\n' + \
                   indent + 'cannot be found.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00015':
        n_valid_exts = len(args[3])
        arguments = [args[2],] + list(args[3])
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the spatial ' + \
                   'discretization file has an invalid' + '\n' + \
                   indent + 'extension (python format extensions are ignored ' + \
                   'when checking the spatial discretization ' + '\n' + \
                   indent + 'file extension).' + '\n\n' + \
                   indent + 'Valid extensions:' + n_valid_exts*' \'{}\''
    elif code == 'E00016':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The large strain formulation option has not been ' + \
                   'implemented yet!'
    elif code == 'E00017':
        if args[2] == 2:
            type = 'plane stress'
        elif args[2] == 3:
            type = 'axisymmetric'
        arguments = [type,]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The program cannot deal with {} problems yet!'
    elif code == 'E00018':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Tensor to matrix conversions are only available for ' + \
                   'second-order and fourth-order ' + '\n' + \
                   indent + 'tensors.'
    elif code == 'E00019':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Tensor to matrix conversions are only available for ' + \
                   'second-order and fourth-order tensors ' + '\n' + \
                   indent + 'with the same size in each dimension, which must be 2 or 3.'
    elif code == 'E00020':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Matrix to tensor conversions are only available for ' + \
                   'vector or square matrix matricial ' + '\n' + \
                   indent + 'forms.'
    elif code == 'E00021':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Matrix to tensor conversions are only available for ' + \
                   'vector or square matrix with size 3, ' + '\n' + \
                   indent + '4, 6 or 9 in each dimension.'
    elif code == 'E00022':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The elastic property - {} - of material phase {} hasn\'t ' + \
                   'been specified or has been ' + '\n' + \
                   indent + 'specified more than once in the input data file.'
    elif code == 'E00023':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid tensor order.'
    elif code == 'E00024':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid component in component order list.'
    elif code == 'E00025':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid tensor dimensions.'
    elif code == 'E00026':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Duplicated component in component order list.'
    elif code == 'E00027':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid number of components in component order list.'
    elif code == 'E00028':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid number of components in tensor matricial form.'
    elif code == 'E00029':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Fourth-order tensor matricial form must be a square matrix.'
    elif code == 'E00030':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Tensor matricial form must be a vector or a matrix.'
    elif code == 'E00031':
        arguments = 5*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in ' + \
                   'the input data file.' + '\n' + \
                   indent + 'The keyword specification is either missing, provided ' + \
                   'in a wrong format or has at least' + '\n' + \
                   indent + 'one specified dimension as a non-positive floating-point ' + \
                   'number.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified in a 2D problem as' + \
                   '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '< dim1_size > < dim2_size >' + '\n\n\n' + \
                   indent + 'The keyword - {} - should be specified in a 3D problem as' + \
                   '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '< dim1_size > < dim2_size > < dim3_size >'
    elif code == 'E00032':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'All the indexes specified to perform a matrix ' + \
                   'condensation must be non-negative integers.'
    elif code == 'E00033':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'There cannot be duplicated row or column indexes when ' + \
                   'performing a matrix condensation.'
    elif code == 'E00034':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'At least one specified row index is out of the matrix ' + \
                   'bounds when performing the ' + '\n' + \
                   indent + 'corresponding condensation.'
    elif code == 'E00035':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'At least one specified column index is out of the matrix ' + \
                   'bounds when performing the ' + '\n' + \
                   indent + 'corresponding condensation.'
    elif code == 'E00036':
        suffix = getordinalnumber(args[2])
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'At least one dataset point has not been labeled during ' + \
                   'the {}' + suffix + ' clustering process.'
    elif code == 'E00037':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid cardinal number.'
    elif code == 'E00038':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The cluster label already exists in the mapping ' + \
                   'dictionary to sort cluster labels.'
    elif code == 'E00039':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid mapping dictionary to sort cluster labels.'
    elif code == 'E00040':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Cluster label must be an integer.'
    elif code == 'E00041':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Unexistent cluster label.'
    elif code == 'E00042':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The number of dimensions of the regular grid of ' + \
                   'pixels/voxels (read from the \'.rgmsh\'' + '\n' + \
                   indent + 'spatial discretization file) must be either 2 ' + \
                   '(2D problem) or 3 (3D problem).'
    elif code == 'E00043':
        arguments = args[2] + args[3]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'There is at least one material phase in the RVE regular ' + \
                   'grid spatial discretization' + '\n' + \
                   indent + 'file (\'.rgmsh\') that has not been specified in the ' + \
                   'input data file.' + '\n\n' + \
                   indent + 'Material phases (input data file):      ' + \
                   len(args[2])*('{}, ') + '\b\b ' + '\n\n' + \
                   indent + 'Material phases (regular grid file):    ' + \
                   len(args[3])*('{}, ') + '\b\b '
    elif code == 'E00044':
        arguments = [' '.join(args[2].split('_')), ' '.join(args[2].split('_')),
                     args[3], ' '.join(args[2].split('_')), args[4]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The {} read in the input data file does not match the one ' + \
                   'loaded' + '\n' + \
                   indent + 'from the clusters data file (.clusters).' + '\n\n' + \
                   indent + '{} (input data file):    {}' + '\n\n' + \
                   indent + '{} (clusters data file): {}'
    elif code == 'E00045':
        list1 = ['\'' + key + '\': ' + '{:2s}'.format(str(args[2][key]))
                 for key in np.sort(list(args[2].keys()))]
        list2 = ['\'' + key + '\': ' + '{:2s}'.format(str(args[3][key]))
                 for key in np.sort(list(args[3].keys()))]
        arguments = list1 + list2
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The number of clusters of each material phase read in the ' + \
                   'input data file does not match' + '\n' + \
                   indent + 'the one loaded from the clusters data file (\'.clusters\').' +\
                   '\n\n' + \
                   indent + 'phase: n_clusters (input data file):    ' + \
                   len(list1)*'{} ' + '\n\n' + \
                   indent + 'phase: n_clusters (clusters data file): ' + len(list2)*'{} '
    elif code == 'E00046':
        arguments = [args[2], args[3]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The regular grid read from the spatial discretization ' + \
                   'file does not match the clustering' + '\n' + \
                   indent + 'grid loaded from the clusters data file (\'.clusters\').' + \
                   '\n\n' + \
                   indent + 'regular grid dimensions (input data file):       {}' + \
                   '\n\n' + \
                   indent + 'clustering grid dimensions (clusters data file): {}'
    elif code == 'E00047':
        arguments = [args[2], args[3], args[3]**2]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                    indent + 'The number of cluster interaction tensors read from the ' + \
                    'cluster interaction tensors data' + '\n' + \
                    indent + 'file (\'.cit\') - {} - is not consistent with the number ' + \
                    'of material phases read in the' + '\n' + \
                    indent + 'input data file - {} (leading to {} cluster interaction ' + \
                    'tensors).'
    elif code == 'E00048':
        arguments = [args[2], ] + args[3]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                    indent + 'The cluster interaction tensors read from the ' + \
                    'cluster interaction tensors data file' + '\n' + \
                    indent + '(\'.cit\') are not consistent with the material phases ' + \
                    'read in the input data file. The' + '\n' + \
                    indent + 'cluster interaction tensor \'{}\' has not been found.' + \
                    '\n\n' + \
                    indent + 'Material phases (input data file): ' + \
                    len(args[3])*('{}, ') + '\b\b '
    elif code == 'E00049':
        arguments = [args[2], args[3]] + list(np.sort(list(args[4])))
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the clustering discretization of an ' + \
                   'unknown material phase - {} - has' + '\n' + \
                   indent + 'been specified.' + '\n\n' + \
                   indent + 'Material phases (input data file): ' + \
                   len(args[4])*('{}, ') + '\b\b '
    elif code == 'E00050':
        arguments = [args[2], args[3]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the clustering discretization of the ' + \
                   'material phase {} has been specified' + '\n' + \
                   indent + 'more than once.'
    elif code == 'E00051':
        arguments = [args[2], args[3]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The elastic property - {} - of material phase {} ' + \
                   'hasn\'t been specified in ' + '\n' + \
                   indent + 'the input data file and is needed to compute the ' + \
                   'associated elasticity tensor.'
    elif code == 'E00052':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file.' + '\n' + \
                   indent + 'In particular, the number of material phases ' + \
                   'specification is missing or is not a' + '\n' + \
                   indent + 'positive integer.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + ' < n_material_phases >' + '\n' + \
                   indent + '2 model1_name 2 [ < model_source > ]' + '\n' + \
                   indent + 'property1_name < value >' + '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + '5 model1_name 3' + '\n' + \
                   indent + 'property1_name < value >' + \
                   '\n' + \
                   indent + 'property2_name < value >' + \
                   '\n' + \
                   indent + 'property3_name < value >'
    elif code == 'E00053':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file\n' + \
                   indent + 'In particular, the constitutive model source of the ' + \
                   'material phase {} is not available' + '\n' + \
                   indent + 'or has not been specified as a positive integer.'
    elif code == 'E00054':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The constitutive model source of the material phase {} ' + \
                   'hasn\'t been implemented yet!'
    elif code == 'E00055':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The constitutive model of the material phase {} is not ' + \
                   'available in the constitutive' + '\n' + \
                   indent + 'model source {}.'
    elif code == 'E00056':
        arguments = [args[2], args[3]]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The required {} function is not implemented for the ' + \
                   'constitutive model \'{}\''
    elif code == 'E00057':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The option \'every\' of the keyword - {} - must be ' + \
                   'must be followed by a positive' + '\n' + \
                   indent + 'integer.'
    elif code == 'E00058':
        arguments = args[2:5]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specified number of material properties ({}) for the ' + \
                   'material phase {} does not' + '\n' + \
                   indent + 'match the required number of properties of the associated ' + \
                   'constitutive model ({}).'
    elif code == 'E00059':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specified property \'{}\' for the material phase {} ' + \
                   'does not belong to the required' + '\n' + \
                   indent + 'properties of the associated constitutive model.'
    elif code == 'E00060':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The VTK output cell data array associated to the state ' + \
                   'variable \'{}\' of the' + '\n' + \
                   indent + 'constitutive model \'{}\' was not specified for all ' + \
                   'pixels (2D) / voxels (3D).'
    elif code == 'E00061':
        arguments = args[2:7]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The maximum number of iterations ({}) has been reached ' + \
                   'when solving the' + '\n' + \
                   indent + 'Lippmann-Schwinger nonlinear system of equilibrium ' + \
                   'equations associated to the' + '\n' + \
                   indent + 'macroscale load increment {}.' + '\n\n' + \
                   indent + 'The normalized errors associated to the ' + \
                   'residuals finished the iterative process with' + '\n' + \
                   indent + 'the following values:' + '\n\n' + \
                   2*indent + 'Clusters equilibrium residuals error: {:16.8e}'
        if args[5] != None:
            template = template + '\n\n' + \
                   2*indent + 'Macroscale strain residuals error   : {:16.8e}'
        if args[6] != None:
            template = template + '\n\n' + \
                   2*indent + 'Macroscale stress residuals error   : {:16.8e}'
    elif code == 'E00062':
        arguments = args[2:7]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The self-consistent scheme maximum number of iterations ' + \
                   '({}) has been reached when' + '\n' + \
                   indent + 'solving the macroscale load increment {}.' + '\n\n' + \
                   indent + 'The normalized iterative changes of the reference ' + \
                   'material elastic properties finished' + '\n' + \
                   indent + 'the iterative process with the following values:' + '\n\n' + \
                   2*indent + 'Normalized iterative change - E : {:16.8e}' + '\n\n' + \
                   2*indent + 'Normalized iterative change - v : {:16.8e}'
    elif code == 'E00063':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specification of the Links finite element order ' + \
                   'keyword - {} - is either' + '\n' + \
                   indent + 'missing or is invalid (must be \'linear\' or ' + \
                   '\'quadratic\').'  + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + ' ' + 'quadratic'
    elif code == 'E00064':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specification of the Links microscale boundary ' + \
                   'condition keyword - ' + '\n' + \
                   indent + '{} - is either missing or is invalid.' + '\n\n' + \
                   indent + 'Available boundary conditions: \'Taylor_Condition\'' + '\n' + \
                   indent + 31*' ' + '\'Linear_Condition\'' + '\n' + \
                   indent + 31*' ' + '\'Periodic_Condition\'' + '\n' + \
                   indent + 31*' ' + '\'Uniform_Traction_Condition\'' + '\n' + \
                   indent + 31*' ' + '\'Uniform_Traction_Condition_II\'' + '\n' + \
                   indent + 31*' ' + '\'Mortar_Periodic_Condition\'' + '\n' + \
                   indent + 31*' ' + '\'Mortar_Periodic_Condition_LM\'' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + ' ' + 'Periodic_Condition'
    elif code == 'E00065':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specification of the Links convergence tolerance ' + \
                   'keyword -' + '\n' + \
                   indent + '{} - is either missing or is invalid (must be ' + \
                   'a non-negative' + '\n' + \
                   indent + 'floating-point number).'  + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{}' + '\n' + \
                   indent + '1e-6'
    elif code == 'E00066':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Attempt to overwrite an existing Links input data file ' + \
                   'in the \'Offline_State/Links/\'' + '\n' + \
                   indent + 'directory.' + '\n\n' + \
                   indent + 'Existing file: {}'
    elif code == 'E00067':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'In order to solve the offline stage (computation of the ' + \
                   'clustering quantities) microscale' + '\n' + \
                   indent + 'equilibrium problems with the program Links it is ' + \
                   'mandatory that all the specified' + '\n' + \
                   indent + 'material constitutive models have the corresponding ' + \
                   'source (2).'
    elif code == 'E00068':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the Links binary ' + \
                   'is not an absolute path (mandatory)' + '\n' + \
                   indent + 'or does not exist.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00069':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specification of the Links element average output ' + \
                   'mode keyword -' + '\n' + \
                   indent + '{} - is either missing or is invalid (must be ' + \
                   'a positive' + '\n' + \
                   indent + 'integer).'  + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The keyword - {} - should be specified p.e. as' + '\n\n' + \
                   indent + '{} 1'
    elif code == 'E00070':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The following Links elementwise average output file has ' + \
                   'not been found. Most probably the' + '\n' + \
                   indent + 'file has not been written by the Links program.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00071':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The following Links screen output file has ' + \
                   'not been found. Most probably the file has' + '\n' + \
                   indent + 'not been written by the Links program.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00072':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The program Links could not successfully solve the ' + \
                   'following microscale equilibrium' + '\n' + \
                   indent + 'problem:' + '\n\n' + \
                   indent + '{}' + '\n\n' + \
                   indent + 'Check the associated \'.screen\' file for more details.'
    elif code == 'E00073':
        arguments = args[2:6]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The maximum number of iterations ({}) has been reached ' + \
                   'when solving the' + '\n' + \
                   indent + 'Lippmann-Schwinger nonlinear system of equilibrium ' + \
                   'equations associated to the' + '\n' + \
                   indent + 'macroscale load increment {}.' + '\n\n' + \
                   indent + 'The normalized errors associated to the ' + \
                   'residuals finished the iterative process with' + '\n' + \
                   indent + 'the following values:' + '\n\n' + \
                   2*indent + 'Clusters equilibrium residuals error: {:16.8e}'
        if args[5] != None:
            template = template + '\n\n' + \
                   2*indent + 'Macroscale stress residuals error   : {:16.8e}'
    elif code == 'E00074':
        arguments = [args[2], args[3]] + args[4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specified isotropic hardening type ({}) for the ' + \
                   'material phase {} is not available.'+ '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'Select one of the available types of isotropic hardening:' + \
                   '\n\n' + indent + len(args[4])*'{}, ' + '\b\b '
    elif code == 'E00075':
        arguments = list(args[2:6]) + args[6]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The specified number of parameters ({}) for the isotropic ' + \
                   'hardening law of material' + '\n' + \
                   indent + 'phase {} ({}) does not match the number of ' + \
                   'required parameters ({}).' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'Required isotropic hardening law parameters (sorted): ' + \
                   len(args[6])*'{}, ' + '\b\b '
    elif code == 'E00076':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The number of hardening curve points specified for the ' + \
                   'material phase {} isotropic' + '\n' + \
                   indent + 'hardening law ({}) must be at least 2.'
    elif code == 'E00077':
        suffix = getordinalnumber(args[2])
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The {}' + suffix + ' point of the hardening curve of the ' + \
                   'material phase {} is either missing or not' + '\n' + \
                   indent + 'properly specified.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The hardening curve must be specified as follows:' + '\n\n' + \
                   indent + 'IHL piecewise_linear 3' + '\n' + \
                   indent + '< acc_p_strain_0 > < yield_stress_0 >' + '\n' + \
                   indent + '< acc_p_strain_1 > < yield_stress_1 >' + '\n' + \
                   indent + '< acc_p_strain_2 > < yield_stress_2 >'
    elif code == 'E00078':
        suffix = getordinalnumber(args[2])
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The {}' + suffix + ' parameter of the isotropic hardening ' + \
                   'law of material phase {} is invalid' + '\n' + \
                   indent + '(must be a integer or floating-point number).'
    elif code == 'E00079':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Unknown type of isotropic hardening ({}).'
    elif code == 'E00080':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The points of the piecewise linear isotropic hardening ' + \
                   'law must be specified in asceding' + '\n' + \
                   indent + 'order of accumulated plastic strain.'
    elif code == 'E00081':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'At least one point of the piecewise linear isotropic ' + \
                   'hardening law has been specified' + '\n' + \
                   indent + 'with an invalid accumulated plastic strain (must be ' + \
                   'non-negative).'
    elif code == 'E00082':
        arguments = args[2:5]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The maximum number of iterations ({}) has been achieved ' + \
                   'without reaching convergence' + '\n' + \
                   indent + 'while performing the state update of the material phase {}.' +\
                   '\n\n' + \
                   indent + 'The normalized error associated to the return-mapping ' + \
                   'residuals finished the iterative' + '\n' + \
                   indent + 'process with the following value:' + '\n\n' + \
                   2*indent + 'Return-mapping residual error: {:16.8e}'
    elif code == 'E00083':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Unknown tensor\'s nature to be stored in matricial form ' + \
                   'following the Voigt notation.'
    elif code == 'E00084':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'In order to be used in the CRATE program online stage, ' + \
                   'the Links constitutive model must' + '\n' + \
                   indent + 'return the total strain tensor or the required data to ' + \
                   'compute it.'
    elif code == 'E00085':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The projection-based self-consistent scheme is only ' + \
                   'implemented for the 3D case.'
    elif code == 'E00086':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The discrete Dirac\'s delta function only accepts two ' + \
                   'integer indexes as arguments.'
    elif code == 'E00087':
        arguments = args[2:4]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file. ' + '\n' + \
                   indent + 'In particular, the specified path to the Links python ' + \
                   'binary is not an absolute path' + '\n' + \
                   indent + '(mandatory) or does not exist.' + '\n\n' + \
                   indent + '{}'
    elif code == 'E00088':
        arguments = 3*[args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The keyword - {} - hasn\'t been properly defined in the ' + \
                   'input data file.\n' + \
                   indent + 'In particular, the number of loading subpaths ' + \
                   'has not been properly specified.' + '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The number of loading subpaths should be specified as a ' + \
                   'positive integer after the' + '\n' + \
                   indent + 'keyword - {} - as' + '\n\n' + \
                   indent + '{} < value >'
    elif code == 'E00089':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'One of the keywords associated to the macroscale loading ' + \
                   'incrementation' + '\n' + \
                   indent + '(Number_of_Load_Increments or Increment_List) must be ' + \
                   'specified and none has been found' + '\n' + \
                   indent + 'in the input data file.'
    elif code == 'E00090':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Only one of the keywords associated to the macroscale ' + \
                   'loading incrementation' + '\n' + \
                   indent + '(Number_of_Load_Increments or Increment_List) can be ' + \
                   'specified and both have been found' + '\n' + \
                   indent + 'in the input data file.'
    elif code == 'E00091':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Invalid macroscale loading increment specification.' + \
                   '\n\n' + \
                   'Suggestion:' + '\n\n' + \
                   indent + 'The macroscale loading increment specification within the ' + \
                   'Increment_List option must have' + '\n' + \
                   indent + 'the format' + '\n\n' + \
                   indent + '[x:]y[_z]' + '\n\n' + \
                   indent + 'where x is the number of repeated increments ' + \
                   '(int, optional), y is the incremental load' + '\n' + \
                   indent + 'factor (float) and z is the incremental time (float, ' + \
                   'optional).'
    elif code == 'E00092':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The first line of the increment list must contain only ' + \
                   'one macroscale loading increment' + '\n' + \
                   indent + 'specification for each macroscale loading subpath.'
    elif code == 'E00093':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Unknown macroscale loading incrementation keyword.'
    elif code == 'E00094':
        arguments = ['',]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The number of repetitions and incremental time optional ' + \
                   'parameters associated to a' + '\n' + \
                   indent + 'macroscale loading increment specification must be positive.'
    elif code == 'E00095':
        arguments = [args[2], ]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The maximum macroscale loading subincrementation level ' + \
                   '({}) has been reached without' + '\n' + \
                   'solution convergence.'
    elif code == 'E00096':
        arguments = [args[2], ]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'The maximum number of macroscale loading consecutive ' + \
                   'increment cuts ({}) has been' + '\n' + \
                   'reached without solution convergence.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display error
    if code in ['E00001', 'E00002', 'E00010']:
        print(template_header.format(*header, width=output_width))
        print(template.format(*values, width=output_width))
        print(template_footer.format(*footer, width=output_width))
    else:
        ioutil.print2(template_header.format(*header, width=output_width))
        ioutil.print2(template.format(*values, width=output_width))
        ioutil.print2(template_footer.format(*footer, width=output_width))
    # Abort program
    sys.exit(1)
#
#                                                                  Display warnings function
# ==========================================================================================
# Set and display runtime warnings
def displaywarning(code, *args):
    # Get display features
    display_features = ioutil.setdisplayfeatures()
    output_width, dashed_line, indent, asterisk_line = display_features[0:4]
    # Set error display header and footer
    warning_color = colorama.Fore.YELLOW
    header = tuple(['!! Warning !!', code] + list(args[0:2]))
    template_header = warning_color + '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + 'Code: ' + colorama.Style.RESET_ALL + \
                      '{}' + '\n\n' + \
                      warning_color + 'Traceback: ' + colorama.Style.RESET_ALL + \
                      '{} (at line {})' + '\n\n'
    footer = tuple([' '])
    template_footer = '\n' + asterisk_line + '\n'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set warnings to display
    if code == 'W00001':
        arguments = [args[2],]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'A non-symmetric macroscale {} tensor was prescribed under ' + \
                   'a small strain formulation.' + '\n' + \
                   indent + 'The symmetric value with the lowest first index is ' + \
                   'enforced (p.e. if descriptor_12 is ' + '\n' + \
                   indent + 'different from descriptor_21, then descriptor_12 is enforced).'
    elif code == 'W00002':
        arguments = args[2] + args[3]
        values = tuple(arguments)
        template = 'Details:' + '\n\n' + \
                   indent + 'Not all the material phases specified in the input data ' + \
                   'file are present in the RVE ' + '\n' + \
                   indent + 'regular grid spatial discretization file (\'.rgmsh\').' + \
                   '\n\n' + \
                   indent + 'Material phases (input data file): ' + \
                   len(args[2])*('{}, ') + '\b\b ' + '\n\n' + \
                   indent + 'Material phases (regular grid):    ' + \
                   len(args[3])*('{}, ') + '\b\b ' + '\n\n' + \
                   indent + 'The number of material phases is going to be updated and ' + \
                   'the non-present material phases ' + '\n' + \
                   indent + 'will be ignored.'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Display warning
    warning_color = colorama.Fore.YELLOW
    template_header = warning_color + template_header + colorama.Style.RESET_ALL
    template_footer = warning_color + template_footer + colorama.Style.RESET_ALL
    ioutil.print2(template_header.format(*header, width=output_width))
    ioutil.print2(template.format(*values, width=output_width))
    ioutil.print2(template_footer.format(*footer, width=output_width))
#
#                                                       Display built-in exceptions function
# ==========================================================================================
# Display runtime built-in exceptions
def displayexception(*args):
    # Get display features
    display_features = ioutil.setdisplayfeatures()
    output_width, dashed_line, indent, asterisk_line = display_features[0:4]
    # Set built-in exception display header and footer
    exception_color = colorama.Fore.RED
    header = tuple(['!! Built-In Exception !!',] + list(args[0:2]))
    template_header = exception_color + '\n' + asterisk_line + '\n' + \
                      '{:^{width}}' + '\n\n' + \
                      'Traceback: ' + colorama.Style.RESET_ALL + \
                      '{} (at line {})' + '\n\n'
    footer = tuple([exception_color + 'Program Aborted' + colorama.Style.RESET_ALL,])
    template_footer = '\n' + asterisk_line + '\n\n' + \
                      '{:^{width}}' + '\n'
    # Set built-in exception to display
    arguments = [args[2],]
    values = tuple(arguments)
    template = 'Details:' + '\n\n' + \
               indent + '{}'
    # Display built-in exception
    exception_color = colorama.Fore.RED
    template_header = exception_color + template_header + colorama.Style.RESET_ALL
    template_footer = exception_color + template_footer + colorama.Style.RESET_ALL
    ioutil.print2(template_header.format(*header, width=output_width))
    ioutil.print2(template.format(*values, width=output_width))
    ioutil.print2(template_footer.format(*footer, width=output_width))
    # Abort program
    sys.exit(1)
#
#                                                                              Miscellaneous
# ==========================================================================================
# Set ordinal number suffix
def getordinalnumber(x):
    suffix_list = ['st', 'nd', 'rd', 'th']
    if (not isinstance(x, int) and not isinstance(x, np.integer)) or x < 0:
        location = inspect.getframeinfo(inspect.currentframe())
        displayerror('E00037', location.filename, location.lineno + 1)
    else:
        if int(str(x)[-1]) in range(1,4):
            suffix = suffix_list[int(str(x)[-1])-1]
        else:
            suffix = suffix_list[3]
    return suffix
