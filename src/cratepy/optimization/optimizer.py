"""Optimization algorithms.

This module includes the interface to implement any optimization algorithm as
well as several wrappers over optimization algorithms available on open-source
libraries (e.g., `SciPy <https://docs.scipy.org/doc/scipy/reference/generated/
scipy.optimize.minimize.html>`_) and others.

Classes
-------
Optimizer
    Optimization algorithm interface.
SciPyMinimizer
    SciPy minimization optimizer (wrapper).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
# Third-party
import numpy as np
import scipy.optimize
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Bernardo Ferreira (bernardo_ferreira@brown.edu)'
__credits__ = ['Bernardo Ferreira', ]
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================
#
#                                             Interface: Optimization algorithm
# =============================================================================
class Optimizer(ABC):
    """Optimization algorithm interface.

    Methods
    -------
    solve_optimization(self, optimization_function, max_n_iter=None, \
                       verbose=False):
        *abstract*: Solve optimization problem.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def solve_optimization(self, optimization_function, max_n_iter=None,
                           verbose=False):
        """Solve optimization problem.

        Parameters
        ----------
        optimization_function : OptimizationFunction
            Instance of OptimizationFunction class.
        max_n_iter : int, default=None
            Maximum number of iterations.
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        parameters : dict
            Optimization parameters names (key, str) and values (item, float).
        """
        pass
#
#                                                       Optimization algorithms
# =============================================================================
class SciPyMinimizer(Optimizer):
    """SciPy minimization optimizer (wrapper).

    Documentation: see `here <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.optimize.minimize.html>`_.

    Methods
    -------
    solve_optimization(self, optimization_function, max_n_iter=None, \
                       verbose=False)
        Solve optimization problem.
    """
    def __init__(self, method='Powell'):
        """Constructor.

        Parameters
        ----------
        method : {'Nelder-Mead', 'Powell', 'CG', 'BFGS'}, default='Powell'
            Optimization method.
        """
        self._method = method
    # -------------------------------------------------------------------------
    def solve_optimization(self, optimization_function, max_n_iter=1,
                           verbose=False):
        """Solve optimization problem.

        Parameters
        ----------
        optimization_function : OptimizationFunction
            Instance of OptimizationFunction class.
        max_n_iter : int, default=1
            Maximum number of iterations.
        verbose : bool, default=False
            Enable verbose output.

        Returns
        -------
        parameters : dict
            Optimization parameters names (key, str) and values (item, float).
        """
        # Get optimization function with normalized parameters provided as
        # sequence
        norm_opt_function_seq = optimization_function.norm_opt_function_seq
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization function parameters names
        parameters_names = optimization_function.get_parameters_names()
        # Set number of optimization parameters
        dimension = len(parameters_names)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters initial guess
        init_shot = optimization_function.get_init_shot(is_normalized=True)
        x0 = np.array([init_shot[str(param)] for param in parameters_names])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters lower and upper bounds
        lower_bounds, upper_bounds = \
            optimization_function.get_bounds(is_normalized=True)
        # Build optimization parameters bounds array
        bounds = np.array([(lower_bounds[str(param)], upper_bounds[str(param)])
                           for param in parameters_names]).reshape(dimension,
                                                                   2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set algorithmic parameters
        options = {'maxiter': max_n_iter, }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve optimization problem
        result = scipy.optimize.minimize(fun=norm_opt_function_seq,
                                         x0=x0,
                                         method=self._method,
                                         bounds=bounds,
                                         options=options)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization solution
        parameters = {str(param): result.x[i]
                      for i, param in enumerate(parameters_names)}
        # optimum = result.fun
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
