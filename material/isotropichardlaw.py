#
# Isotropic hardening laws (CRATE Program)
# ==========================================================================================
# Summary:
# Definition of material isotropic strain hardening laws.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Mar 2020 | Initial coding.
# Bernardo P. Ferreira | Oct 2021 | Updated documentation.
# Bernardo P. Ferreira | Jan 2022 | Build hardening parameters.
#
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
#
#                                                                   Isotropic hardening laws
# ==========================================================================================
def get_available_hardening_types():
    '''Get available isotropic hardening laws.

    Available isotropic hardening laws:

        1. Piecewise linear hardening ({ep_i, s_i}, i = 1, n_points)
            isotropic_hardening | Isotropic hardening type.
            hp_i                | Hardening point with coordinates
                                | (accumulated plastic strain, yield stress).
                                | Stored in a 2darray hardening_points.

            Input data file syntax:
            isotropic_hardening piecewise_linear < number of hardening points >
                hp_1 < value > < value >
                hp_2 < value > < value >
                ...

        2. Linear hardening (s = s0 + a*ep)
            isotropic_hardening | Isotropic hardening type.
            s0                  | Initial yield stress.
            a                   | Hardening law parameter.

            Input data file syntax:
            isotropic_hardening linear 2
                s0 < value >
                a  < value > < value >

        3. Swift hardening (s = s0 + a*ep**b)
            isotropic_hardening | Isotropic hardening type.
            s0                  | Initial yield stress.
            a                   | Hardening law parameter.
            b                   | Hardening law parameter.

            Input data file syntax:
            isotropic_hardening swift 3
                s0 < value >
                a  < value >
                b  < value >

        4. Ramberg-Osgood hardening (s = s0*(1 + a*ep)**(1/b))
            isotropic_hardening | Isotropic hardening type.
            s0                  | Initial yield stress.
            a                   | Hardening law parameter.
            b                   | Hardening law parameter.

            Input data file syntax:
            isotropic_hardening ramberg_osgood 3
                s0 < value >
                a  < value >
                b  < value >

    Returns
    -------
    available_hardening_types : tuple
        List of available isotropic hardening laws (str).
    '''
    # Set available isotropic hardening types
    available_hardening_types = ('piecewise_linear', 'linear', 'swift', 'ramberg_osgood')
    # Return
    return available_hardening_types
# ------------------------------------------------------------------------------------------
def build_hardening_parameters(type, material_properties):
    '''Build hardening parameters according with isotropic hardening type.

    Parameters
    ----------
    type : str
        Isotropic hardening law.
    material_properties : dict
        Constitutive model material properties (key, str) values (item, int/float/bool).

    Returns
    -------
    hardening_parameters : dict
        Isotropic hardening law parameters (key, str) values (item, float).
    '''
    # Initialize hardening parameters
    hardening_parameters = {}
    # Build hardening parameters according with isotropic hardening type
    try:
        if type == 'piecewise_linear':
            hardening_parameters['hardening_points'] = \
                material_properties['hardening_points']
        elif type == 'linear':
            hardening_parameters['s0'] = material_properties['s0']
            hardening_parameters['a'] = material_properties['a']
        elif type == 'swift' or type == 'ramberg_osgood':
            hardening_parameters['s0'] = material_properties['s0']
            hardening_parameters['a'] = material_properties['a']
            hardening_parameters['b'] = material_properties['b']
        else:
            raise RuntimeError('Unknown isotropic hardening type.')
    except KeyError as err:
        raise KeyError('Missing hardening parameter: ' + str(err.args[0]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return hardening_parameters
# ------------------------------------------------------------------------------------------
def get_hardening_law(type):
    '''Set isotropic hardening law to evaluate hardening modulus and hardening slope.

    Parameters
    ----------
    type : str
        Isotropic hardening law.

    Returns
    -------
    hardening_law : function
        Isotropic hardening law.
    '''
    # Piecewise linear isotropic hardening
    if type == 'piecewise_linear':
        # Set piecewise linear isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            '''Piecewise linear isotropic hardening law.

            Provided the required isotropic hardening law parameters and a given value of
            accumulated plastic strain, return the corresponding material yield stress and
            hardening slope.

            Parameters
            ----------
            hardening_parameters : dict
                Isotropic hardening law required parameters: hardening points (2darray),
                where each row (i, 0:2) sets a hardening point characterized by an
                accumulated plastic strain value (i, 0) and the associated yield stress
                (i, 1).

            Returns
            -------
            yield_stress : float
                Material yield stress.
            H : float
                Material hardening slope.
            '''
            # Get hardening curve points array
            hardening_points = hardening_parameters['hardening_points']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build lists with the accumulated plastic strain points and associated yield
            # stress values
            a = list(hardening_points[:, 0])
            b = list(hardening_points[:, 1])
            # Check if the accumulated plastic strain list is correctly sorted in ascending
            # order
            if not np.all(np.diff(a) > 0):
                raise RuntimeError('Points of piecewise linear isotropic hardening law ' +
                                   'must be specified in ascending order of accumulated ' +
                                   'plastic strain.')
            elif not np.all([i >= 0 for i in a]):
                raise RuntimeError('Points of piecewise linear isotropic hardening law ' +
                                   'must be associated with non-negative accumulated ' +
                                   'plastic strain values.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If the value of the accumulated plastic strain is either below or above the
            # provided hardening curve points, then simply assume the yield stress
            # associated with the first or last provided points, respectively. Otherwise,
            # perform linear interpolation to compute the yield stress
            yield_stress = np.interp(acc_p_strain, a, b, b[0], b[-1])
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return [yield_stress, H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linear isotropic hardening
    elif type == 'linear':
        '''Linear isotropic hardening law.

        Provided the required isotropic hardening law parameters and a given value of
        accumulated plastic strain, return the corresponding material yield stress and
        hardening slope.

        Parameters
        ----------
        hardening_parameters : dict
            Isotropic hardening law required parameters: s0 (float), initial yield stress;
            H (float), hardening slope.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        H : float
            Material hardening slope.
        '''
        # Set linear isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            # Get initial yield stress and hardening slope
            yield_stress_init = float(hardening_parameters['s0'])
            H = float(hardening_parameters['a'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute yield stress
            yield_stress = yield_stress_init + H*acc_p_strain
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return [yield_stress, H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Swift isotropic hardening
    elif type == 'swift':
        '''Swift isotropic hardening law.

        Provided the required isotropic hardening law parameters and a given value of
        accumulated plastic strain, return the corresponding material yield stress and
        hardening slope.

        Parameters
        ----------
        hardening_parameters : dict
            Isotropic hardening law required parameters: s0 (float), initial yield stress;
            a (float), function parameter; b (float), function parameter.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        H : float
            Material hardening slope.
        '''
        # Set Swift isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            # Get initial yield stress and parameters
            yield_stress_init = float(hardening_parameters['s0'])
            a = float(hardening_parameters['a'])
            b = float(hardening_parameters['b'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return [yield_stress, H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ramberg-Osgood isotropic hardening
    elif type == 'ramberg_osgood':
        '''Ramberg-Osgood isotropic hardening law.

        Provided the required isotropic hardening law parameters and a given value of
        accumulated plastic strain, return the corresponding material yield stress and
        hardening slope.

        Parameters
        ----------
        hardening_parameters : dict
            Isotropic hardening law required parameters: s0 (float), initial yield stress;
            a (float), function parameter; b (float), function parameter.

        Returns
        -------
        yield_stress : float
            Material yield stress.
        H : float
            Material hardening slope.
        '''
        # Set Ramberg-Osgood isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            # Get initial yield stress and parameters
            yield_stress_init = float(hardening_parameters['s0'])
            a = float(hardening_parameters['a'])
            b = float(hardening_parameters['b'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute yield stress
            yield_stress = yield_stress_init*(1.0 + a*acc_p_strain)**(1/b)
            # Compute hardening slope
            H = yield_stress_init*(a/b)*(1.0 + a*acc_p_strain)**((1 - b)/b)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return [yield_stress, H]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unknown isotropic hardening type
    else:
        raise RuntimeError('Unknown type of isotropic hardening law.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return hardening_law
