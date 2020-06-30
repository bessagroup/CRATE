#
# Isotropic hardening laws (CRATE Program)
# ==========================================================================================
# Summary:
# Module containing procedures related to the definition of material isotropic strain
# hardening laws.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | March 2020 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Inspect file name and line
import inspect
# Display errors, warnings and built-in exceptions
import ioput.errors as errors
#
#                                                                   Isotropic hardening laws
# ==========================================================================================
# For a given type of isotropic hardening, set the required parameters and the isotropic
# hardening law. The available types are described below:
#
#   type                 Hardening law                          Functional format
#   --------------------------------------------------------------------------------------
#   piecewise_linear  >  Piecewise linear isotropic hardening : {ep_i,s_i}, i = 1,n_points
#   linear            >  Linear isotropic hardening           : s = s0 + a*ep
#   swift             >  Swift isotropic hardening            : s = s0 + a*ep**b
#   ramberg_osgood    >  Ramberg-Osgood isotropic hardening   : s = s0*(1 + a*ep)**(1/b)
#
def getAvailableTypes():
    # Set available isotropic hardening types
    available_hardening_types = ['piecewise_linear','linear','swift','ramberg_osgood']
    # Return
    return available_hardening_types
# ------------------------------------------------------------------------------------------
def setRequiredParameters(type):
    # Piecewise linear isotropic hardening
    if type == 'piecewise_linear':
        req_hardening_parameters = ['n_points']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linear isotropic hardening
    elif type == 'linear':
        req_hardening_parameters = ['s0','a']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Swift isotropic hardening
    elif type == 'swift':
        req_hardening_parameters = ['s0','a','b']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ramberg-Osgood isotropic hardening
    elif type == 'ramberg_osgood':
        req_hardening_parameters = ['s0','a','b']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unknown isotropic hardening type
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00078',location.filename,location.lineno+1,type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return req_hardening_parameters
# ------------------------------------------------------------------------------------------
def setHardeningLaw(type):
    # Piecewise linear isotropic hardening
    if type == 'piecewise_linear':
        # Set piecewise linear isotropic hardening law
        def hardeningLaw(hardening_parameters,acc_p_strain):
            # Get hardening curve points array
            hardening_points = hardening_parameters['hardening_points']
            # Build lists with the accumulated plastic strain points and associated yield
            # stress values
            a = list(hardening_points[:,0])
            b = list(hardening_points[:,1])
            # Check if the accumulated plastic strain list is correctly sorted in asceding
            # order
            if not np.all(np.diff(a) > 0):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00080',location.filename,location.lineno+1,type)
            elif not np.all([i >= 0 for i in a]):
                location = inspect.getframeinfo(inspect.currentframe())
                errors.displayerror('E00081',location.filename,location.lineno+1,type)
            # If the value of the accumulated plastic strain is either below or above the
            # provided hardening curve points, then simply assume the yield stress
            # associated with the first or last provided points, respectively. Otherwise,
            # perform linear interpolation to compute the yield stress
            yield_stress = np.interp(acc_p_strain,a,b,b[0],b[-1])
            # Get hardening slope
            if acc_p_strain < a[0] or acc_p_strain >= a[-1]:
                H = 0
            else:
                # Get hardening curve interval
                x0 = list(filter(lambda i: i <= acc_p_strain, a[::-1]))[0]
                y0 = b[a.index(x0)]
                x1 = list(filter(lambda i: i > acc_p_strain, a))[0]
                y1 = b[a.index(x1)]
                # Compute hardening slope
                H = (y1 - y0)/(x1 - x0)
            # Return
            return [yield_stress,H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linear isotropic hardening
    elif type == 'linear':
        # Set linear isotropic hardening law
        def hardeningLaw(hardening_parameters,acc_p_strain):
            # Get initial yield stress and hardening slope
            yield_stress_init = float(hardening_parameters['s0'])
            H = float(hardening_parameters['a'])
            # Compute yield stress
            yield_stress = yield_stress_init + H*acc_p_strain
            # Return
            return [yield_stress,H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Swift isotropic hardening
    elif type == 'swift':
        # Set Swift isotropic hardening law
        def hardeningLaw(hardening_parameters,acc_p_strain):
            # Get initial yield stress and parameters
            yield_stress_init = float(hardening_parameters['s0'])
            a = float(hardening_parameters['a'])
            b = float(hardening_parameters['b'])
            # Compute yield stress
            yield_stress = yield_stress_init + a*(acc_p_strain**b)
            # Compute hardening slope
            if abs(acc_p_strain) < 10e-8:
                # Compute hardening slope at the initial point (null accumulated plastic
                # strain) with a forward finite difference
                H = (a*((acc_p_strain + 10e-8)**b) - a*(acc_p_strain**b))/10e-8
            else:
                # Compute analytical hardening slope
                H = a*b*(acc_p_strain**(b - 1))
            # Return
            return [yield_stress,H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ramberg-Osgood isotropic hardening
    elif type == 'ramberg_osgood':
        # Set Ramberg-Osgood isotropic hardening law
        def hardeningLaw(hardening_parameters,acc_p_strain):
            # Get initial yield stress and parameters
            yield_stress_init = float(hardening_parameters['s0'])
            a = float(hardening_parameters['a'])
            b = float(hardening_parameters['b'])
            # Compute yield stress
            yield_stress = yield_stress_init*(1.0 + a*acc_p_strain)**(1/b)
            # Compute hardening slope
            H = yield_stress_init*(a/b)*(1.0 + a*acc_p_strain)**((1 - b)/b)
            # Return
            return [yield_stress,H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unknown isotropic hardening type
    else:
        location = inspect.getframeinfo(inspect.currentframe())
        errors.displayerror('E00078',location.filename,location.lineno+1,type)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return hardeningLaw
