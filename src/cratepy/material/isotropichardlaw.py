"""Isotropic strain hardening laws.

This module includes the definition of several types of isotropic strain
hardening laws and the suitable processing of the associated parameters.

Functions
---------
get_available_hardening_types
    Get available isotropic hardening laws.
build_hardening_parameters
    Build hardening parameters according with isotropic hardening type.
get_hardening_law
    Get isotropic hardening law to compute yield stress and hardening slope.
"""
#
#                                                                       Modules
# =============================================================================
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
def get_available_hardening_types():
    """Get available isotropic hardening laws.

    The isotropic hardening law is specified in the input data file as a
    material property of the constitutive model.

    Available isotropic hardening laws:

    * Piecewise linear hardening

        .. math::

           \\{\\bar{\\varepsilon}^{p}_{i}, \\, \\sigma_{y, i}\\},
           \\quad i = 1, \\dots, n_{\\text{points}}

        *Input data file syntax*:

        .. code-block:: text

           isotropic_hardening piecewise_linear < n_points >
               hp_1 < value > < value >
               hp_2 < value > < value >
               ...

        where

        - ``isotropic_hardening`` - Isotropic strain hardening type and number
          of parameters.
        - ``hp_i`` - Hardening point with coordinates
          (:math:`\\bar{\\varepsilon}^{p}`, :math:`\\sigma_{y}`).

    ----

    * Linear hardening

        .. math::

           \\sigma_{y}(\\bar{\\varepsilon}^{p}) = \\sigma_{y, 0}
           + a \\bar{\\varepsilon}^{p}

        *Input data file syntax*:

        .. code-block:: text

           isotropic_hardening linear 2
               s0 < value >
               a  < value >

        where

        - ``isotropic_hardening`` - Isotropic strain hardening type and number
          of parameters.
        - ``s0`` - Initial yielding stress (:math:`\\sigma_{y, 0}`).
        - ``a`` - Hardening law parameter (:math:`a`).

    ----

    * Nadai-Ludwik hardening:

        .. math::

           \\sigma_{y}(\\bar{\\varepsilon}^{p}) = \\sigma_{y, 0}
           + a (\\bar{\\varepsilon}^{p}_{0} + \\bar{\\varepsilon}^{p})^{b}

        *Input data file syntax*:

        .. code-block:: text

           isotropic_hardening nadai_ludwik 4
               s0  < value >
               a   < value >
               b   < value >
               ep0 < value >

        where

        - ``isotropic_hardening`` - Isotropic strain hardening type and number
          of parameters.
        - ``s0`` - Initial yielding stress (:math:`\\sigma_{y, 0}`).
        - ``a`` - Hardening law parameter (:math:`a`).
        - ``b`` - Hardening law parameter (:math:`b`).
        - ``ep0`` - Accumulated plastic strain corresponding to initial \
                    yielding stress (:math:`\\bar{\\varepsilon}^{p}_{0}`)

    ----

    Returns
    -------
    available_hardening_types : tuple[str]
        List of available isotropic hardening laws (str).
    """
    # Set available isotropic hardening types
    available_hardening_types = ('piecewise_linear', 'linear', 'nadai_ludwik')
    # Return
    return available_hardening_types
# =============================================================================
def build_hardening_parameters(type, material_properties):
    """Build hardening parameters according with isotropic hardening type.

    Parameters
    ----------
    type : str
        Isotropic hardening law.
    material_properties : dict
        Constitutive model material properties (key, str) values
        (item, {int, float, bool}).

    Returns
    -------
    hardening_parameters : dict
        Isotropic hardening law parameters (key, str) values (item, float).
    """
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
        elif type == 'nadai_ludwik':
            hardening_parameters['s0'] = material_properties['s0']
            hardening_parameters['a'] = material_properties['a']
            hardening_parameters['b'] = material_properties['b']
            # Tolerance to avoid overflow when computing hardening slope at
            # null accumulated plastic strain
            ep0_tol = 1e-8
            hardening_parameters['ep0'] = \
                np.maximum(material_properties['ep0'], ep0_tol)
        else:
            raise RuntimeError('Unknown isotropic hardening type.')
    except KeyError as err:
        raise KeyError('Missing hardening parameter: ' + str(err.args[0]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return hardening_parameters
# =============================================================================
def get_hardening_law(type):
    """Get isotropic hardening law to compute yield stress and hardening slope.

    Parameters
    ----------
    type : str
        Isotropic hardening law.

    Returns
    -------
    hardening_law : function
        Isotropic hardening law.
    """
    # Piecewise linear isotropic hardening
    if type == 'piecewise_linear':
        # Set piecewise linear isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            """Piecewise linear isotropic hardening law.

            Provided the required isotropic hardening law parameters and a
            given value of accumulated plastic strain, return the corresponding
            material yield stress and hardening slope.

            ----

            Parameters
            ----------
            hardening_parameters : dict
                Isotropic hardening law required parameters:

                hardening_points : numpy.ndarray (2d), where each row (i, 0:2)
                sets a hardening point characterized by an accumulated plastic
                strain value (i, 0) and the associated yield stress (i, 1).

            Returns
            -------
            yield_stress : float
                Material yield stress.
            H : float
                Material hardening slope.
            """
            # Get hardening curve points array
            hardening_points = hardening_parameters['hardening_points']
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Build lists with the accumulated plastic strain points and
            # associated yield stress values
            a = list(hardening_points[:, 0])
            b = list(hardening_points[:, 1])
            # Check if the accumulated plastic strain list is correctly sorted
            # in ascending order
            if not np.all(np.diff(a) > 0):
                raise RuntimeError('Points of piecewise linear isotropic '
                                   'hardening law must be specified in '
                                   'ascending order of accumulated '
                                   'plastic strain.')
            elif not np.all([i >= 0 for i in a]):
                raise RuntimeError('Points of piecewise linear isotropic '
                                   'hardening law must be associated with '
                                   'non-negative accumulated plastic strain '
                                   'values.')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # If the value of the accumulated plastic strain is either below or
            # above the provided hardening curve points, then simply assume the
            # yield stress associated with the first or last provided points,
            # respectively. Otherwise, perform linear interpolation to compute
            # the yield stress
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
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return yield_stress, H
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linear isotropic hardening
    elif type == 'linear':
        # Set linear isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            """Linear isotropic hardening law.

            Provided the required isotropic hardening law parameters and a
            given value of accumulated plastic strain, return the corresponding
            material yield stress and hardening slope.

            ----

            Parameters
            ----------
            hardening_parameters : dict
                Isotropic hardening law required parameters:

                s0 : float, initial yield stress

                a : float, hardening law parameter

            Returns
            -------
            yield_stress : float
                Material yield stress.
            H : float
                Material hardening slope.
            """
            # Get initial yield stress and hardening slope
            yield_stress_init = float(hardening_parameters['s0'])
            H = float(hardening_parameters['a'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Compute yield stress
            yield_stress = yield_stress_init + H*acc_p_strain
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return yield_stress, H
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Nadai-Ludwik isotropic hardening
    elif type == 'nadai_ludwik':
        # Set Nadai-Ludwik isotropic hardening law
        def hardening_law(hardening_parameters, acc_p_strain):
            """Nadai-Ludwik isotropic hardening law.

            Provided the required isotropic hardening law parameters and a
            given value of accumulated plastic strain, return the corresponding
            material yield stress and hardening slope.

            ----

            Parameters
            ----------
            hardening_parameters : dict
                Isotropic hardening law required parameters:

                s0 : float, initial yield stress

                a : float, hardening law parameter

                b : float, hardening law parameter

                ep0 : float, accumulated plastic strain corresponding to \\
                      initial yielding stress

            Returns
            -------
            yield_stress : float
                Material yield stress.
            H : float
                Material hardening slope.
            """
            # Get initial yield stress and parameters
            yield_stress_init = float(hardening_parameters['s0'])
            a = float(hardening_parameters['a'])
            b = float(hardening_parameters['b'])
            ep0 = float(hardening_parameters['ep0'])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Consider minimum non-negative accumulated plastic strain
            acc_p_strain = np.maximum(0.0, acc_p_strain)
            # Compute yield stress
            yield_stress = yield_stress_init + a*((acc_p_strain + ep0)**b)
            # Compute hardening slope
            H = a*b*(acc_p_strain + ep0)**(b - 1)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Return
            return yield_stress, H
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Unknown isotropic hardening type
    else:
        raise RuntimeError('Unknown type of isotropic hardening law.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    return hardening_law
