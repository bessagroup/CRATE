#
# Links Utilities Module (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to consistency operations required to implement a
# suitable interface with the multi-scale finite element code Links (e.g. strain/stress
# component order, tensorial matricial storage convention, ...).
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | April 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Generate efficient iterators
import itertools as it
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
#
#                                                       Links strain/stress components order
# ==========================================================================================
# Set Links strain/stress components order in symmetric and nonsymmetric cases
def getlinkscomporder(n_dim):
    # Set Links strain/stress components order
    if n_dim == 2:
        links_comp_order_sym = ['11', '22', '12']
        links_comp_order_nsym = ['11', '21', '12', '22']
    else:
        links_comp_order_sym = ['11', '22', '33', '12', '23', '13']
        links_comp_order_nsym = ['11', '21', '31', '12', '22', '32', '13', '23', '33']
    # Return
    return [links_comp_order_sym, links_comp_order_nsym]
#
#                                                         Links Tensorial < > Matricial Form
# ==========================================================================================
# Store a given second-order or fourth-order tensor in matricial form for a given number of
# dimensions and given ordered component list. If the second-order tensor is symmetric or
# the fourth-order tensor has minor symmetry (component list only contains independent
# components), then the Voigt notation is employed to perform the storage. The tensor
# recovery from the associated matricial form follows a precisely inverse procedure.
#
# Note: The Voigt notation depends on the nature of the tensor. Four tensor natures are
#       covered in this function, namely: 'strain'/'stress' associated to a given
#       second-order tensor; 'elasticity'/'compliance' associated to a given fourth-order
#       tensor. If the symmetries are to be ignored, then the provided nature specification
#       is ignored as well.
#
def gettensormflinks(tensor, n_dim, comp_order, nature):
    # Set tensor order
    tensor_order = len(tensor.shape)
    # Check input arguments validity
    if tensor_order not in [2, 4]:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00023', location.filename, location.lineno + 1)
    elif any([int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif any([tensor.shape[i] != n_dim for i in range(len(tensor.shape))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00025', location.filename, location.lineno + 1)
    elif any([len(comp) != 2 for comp in comp_order]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00026', location.filename, location.lineno + 1)
    # Set Voigt notation flag
    if len(comp_order) == n_dim**2:
        isVoigtNotation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        if nature not in ['strain', 'stress', 'elasticity', 'compliance']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00083', location.filename, location.lineno + 1)
        if nature in ['strain', 'compliance']:
            isVoigtNotation = True
        else:
            isVoigtNotation = False
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00027', location.filename, location.lineno + 1)
    # Store tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # Initialize tensor matricial form
        if tensor.dtype == 'complex':
            tensor_mf = np.zeros(len(comp_order), dtype=complex)
        else:
            tensor_mf = np.zeros(len(comp_order))
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if isVoigtNotation and not so_idx[0] == so_idx[1]:
                factor = 2.0
            tensor_mf[mf_idx] = factor*tensor[so_idx]
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        fo_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in \
                [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
        # Initialize tensor matricial form
        if tensor.dtype == 'complex':
            tensor_mf = np.zeros((len(comp_order), len(comp_order)), dtype=complex)
        else:
            tensor_mf = np.zeros((len(comp_order), len(comp_order)))
        # Store tensor in matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if isVoigtNotation and not (fo_idx[0] == fo_idx[1] and fo_idx[2] == fo_idx[3]):
                factor = factor*2.0 if fo_idx[0] != fo_idx[1] else factor
                factor = factor*2.0 if fo_idx[2] != fo_idx[3] else factor
            tensor_mf[mf_idx] = factor*tensor[fo_idx]
    # Return
    return tensor_mf
# ------------------------------------------------------------------------------------------
def gettensorfrommflinks(tensor_mf, n_dim, comp_order, nature):
    # Set tensor order
    if len(tensor_mf.shape) == 1:
        tensor_order = 2
        if tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim+1)):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00028', location.filename, location.lineno + 1)
    elif len(tensor_mf.shape) == 2:
        tensor_order = 4
        if tensor_mf.shape[0] != tensor_mf.shape[1]:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00029', location.filename, location.lineno + 1)
        elif tensor_mf.shape[0] != n_dim**2 and tensor_mf.shape[0] != sum(range(n_dim+1)):
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00028', location.filename, location.lineno + 1)
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00030', location.filename, location.lineno + 1)
    # Check input arguments validity
    if any([ int(x) not in range(1, n_dim + 1) for x in list(''.join(comp_order))]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif any([len(comp) != 2 for comp in comp_order]):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00024', location.filename, location.lineno + 1)
    elif len(list(dict.fromkeys(comp_order))) != len(comp_order):
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00026', location.filename, location.lineno + 1)
    # Set Voigt notation flag
    if len(comp_order) == n_dim**2:
        isVoigtNotation = False
    elif len(comp_order) == sum(range(n_dim + 1)):
        if nature not in ['strain', 'stress', 'elasticity', 'compliance']:
            location = inspect.getframeinfo(inspect.currentframe())
            errors.displayerror('E00083', location.filename, location.lineno + 1)
        if nature in ['strain', 'compliance']:
            isVoigtNotation = True
        else:
            isVoigtNotation = False
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00027', location.filename, location.lineno + 1)
    # Get tensor according to tensor order
    if tensor_order == 2:
        # Set second-order and matricial form indexes
        so_indexes = list()
        mf_indexes = list()
        for i in range(len(comp_order)):
            so_indexes.append([int(x) - 1 for x in list(comp_order[i])])
            mf_indexes.append(comp_order.index(comp_order[i]))
        # Initialize tensor
        if tensor_mf.dtype == 'complex':
            tensor = np.zeros(tensor_order*(n_dim,), dtype=complex)
        else:
            tensor = np.zeros(tensor_order*(n_dim,))
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = mf_indexes[i]
            so_idx = tuple(so_indexes[i])
            factor = 1.0
            if isVoigtNotation and not so_idx[0] == so_idx[1]:
                factor = 2.0
                tensor[so_idx[::-1]] = (1.0/factor)*tensor_mf[mf_idx]
            tensor[so_idx] = (1.0/factor)*tensor_mf[mf_idx]
    elif tensor_order == 4:
        # Set cartesian product of component list
        comps = list(it.product(comp_order, comp_order))
        # Set fourth-order and matricial form indexes
        mf_indexes = list()
        fo_indexes = list()
        for i in range(len(comp_order)**2):
            fo_indexes.append([int(x) - 1 for x in list(comps[i][0] + comps[i][1])])
            mf_indexes.append([x for x in \
                [comp_order.index(comps[i][0]), comp_order.index(comps[i][1])]])
        # Initialize tensor
        if tensor_mf.dtype == 'complex':
            tensor = np.zeros(tensor_order*(n_dim,), dtype=complex)
        else:
            tensor = np.zeros(tensor_order*(n_dim,))
        # Get tensor from matricial form
        for i in range(len(mf_indexes)):
            mf_idx = tuple(mf_indexes[i])
            fo_idx = tuple(fo_indexes[i])
            factor = 1.0
            if isVoigtNotation and not (fo_idx[0] == fo_idx[1] and fo_idx[2] == fo_idx[3]):
                factor = factor*2.0 if fo_idx[0] != fo_idx[1] else factor
                factor = factor*2.0 if fo_idx[2] != fo_idx[3] else factor
                if fo_idx[0] != fo_idx[1] and fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[1::-1] + fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[:2] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                    tensor[tuple(fo_idx[1::-1] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[0] != fo_idx[1]:
                    tensor[tuple(fo_idx[1::-1] + fo_idx[2:])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
                elif fo_idx[2] != fo_idx[3]:
                    tensor[tuple(fo_idx[:2] + fo_idx[3:1:-1])] = \
                        (1.0/factor)*tensor_mf[mf_idx]
            tensor[fo_idx] = (1.0/factor)*tensor_mf[mf_idx]
    # Return
    return tensor
