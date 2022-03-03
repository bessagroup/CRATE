#
# Optimizers Module (CRATE Program)
# ==========================================================================================
# Summary:
# Optimization algorithms interface.
# ------------------------------------------------------------------------------------------
# Development history:
# Bernardo P. Ferreira | Feb 2022 | Initial coding.
# ==========================================================================================
#                                                                             Import modules
# ==========================================================================================
# Working with arrays
import numpy as np
# Defining abstract base classes
from abc import ABC, abstractmethod
# LIPO optimizer (NOT AVAILABLE UBUNTU 16.04LTS)
#import lipo
# Genetic algorithm optimizer
import geneticalgorithm
# SciPy optimizers
import scipy.optimize
# Scikit optimizers
import skopt
#
#                                                          Optimization algorithms interface
# ==========================================================================================
class Optimizer(ABC):
    '''Optimization algorithm interface.'''
    @abstractmethod
    def __init__(self):
        '''Optimization algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    @abstractmethod
    def solve_optimization(self, optimization_function, max_n_iter=None, verbose=False):
        '''Solve optimization problem.

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
            Optimization parameters.
        '''
        pass
# ------------------------------------------------------------------------------------------
class LIPO(Optimizer):
    '''LIPO optimizer.

    LIPO optimizer documentation: https://pypi.org/project/lipo/
    '''
    def __init__(self):
        '''Optimization algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    def solve_optimization(self, optimization_function, max_n_iter=1, verbose=False):
        '''Solve optimization problem.

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
            Optimization parameters.
        '''
        '''
        # Set type of optimization problem as maximization
        is_maximize = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization function with normalized parameters
        norm_opt_function = optimization_function.get_opt_function(is_normalized=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters lower and upper bounds
        lower_bounds, upper_bounds = optimization_function.get_bounds(is_normalized=True)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute optimization function at optimization parameters initial guess
        init_shot = optimization_function.get_init_shot(is_normalized=True)
        if not init_shot is None:
            # Initialize prior evaluations of optimization function
            evaluations = []
            # Append evaluation at optimization parameters intial guess
            evaluations.append((init_shot, norm_opt_function(init_shot)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate LIPO optimizer
        optimizer = lipo.GlobalOptimizer(
            func=norm_opt_function, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                evaluations=evaluations, maximize=is_maximize)
        # Solve optimization problem
        optimizer.run(num_function_calls=max_n_iter)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization solution
        parameters = optimizer.optimum[0]
        optimum = optimizer.optimum[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
        '''
# ------------------------------------------------------------------------------------------
class GeneticAlgorithm(Optimizer):
    '''Genetic algorithm optimizer.

    Genetic algorithm optimizer documentation: https://pypi.org/project/geneticalgorithm/
    '''
    def __init__(self):
        '''Optimization algorithm constructor.'''
        pass
    # --------------------------------------------------------------------------------------
    def solve_optimization(self, optimization_function, max_n_iter=1, verbose=False):
        '''Solve optimization problem.

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
            Optimization parameters.
        '''
        # Get optimization function with normalized parameters provided as sequence
        norm_opt_function_seq = optimization_function.norm_opt_function_seq
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization function parameters names
        parameters_names = optimization_function.get_parameters_names()
        # Set number of optimization parameters
        dimension = len(parameters_names)
        # Set optimization parameters type
        variable_type = 'real'
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters lower and upper bounds
        lower_bounds, upper_bounds = optimization_function.get_bounds(is_normalized=True)
        # Build optimization parameters bounds array
        variable_boundaries = \
            np.array([(lower_bounds[str(param)], upper_bounds[str(param)])
                      for param in parameters_names]).reshape(dimension, 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set algorithmic parameters
        algorithm_parameters = {'max_num_iteration': max_n_iter,
                                'population_size': 100,
                                'mutation_probability': 0.1,
                                'elit_ratio': 0.01,
                                'crossover_probability': 0.5,
                                'parents_portion': 0.3,
                                'crossover_type': 'uniform',
                                'max_iteration_without_improv': None}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Instantiate Genetic Algorithm optimizer
        optimizer = \
            geneticalgorithm.geneticalgorithm(function=norm_opt_function_seq,
                                              dimension=dimension,
                                              variable_type=variable_type,
                                              variable_boundaries=variable_boundaries,
                                              algorithm_parameters=algorithm_parameters,
                                              convergence_curve=False,
                                              progress_bar=False)
        # Solve optimization problem
        optimizer.run()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization solution
        parameters = {str(param): optimizer.best_variable[i]
                      for i, param in enumerate(parameters_names)}
        optimum = optimizer.best_function
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
# ------------------------------------------------------------------------------------------
class SciPyMinimizer(Optimizer):
    '''SciPy minimization optimizer.

    SciPy minization optimizer documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    '''
    def __init__(self, method='Powell'):
        '''Optimization algorithm constructor.

        Parameters
        ----------
        method : str, {'Nelder-Mead', 'Powell', 'CG', 'BFGS'}, default='Powell'
            Optimization method.
        '''
        self._method = method
    # --------------------------------------------------------------------------------------
    def solve_optimization(self, optimization_function, max_n_iter=1, verbose=False):
        '''Solve optimization problem.

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
            Optimization parameters.
        '''
        # Get optimization function with normalized parameters provided as sequence
        norm_opt_function_seq = optimization_function.norm_opt_function_seq
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization function parameters names
        parameters_names = optimization_function.get_parameters_names()
        # Set number of optimization parameters
        dimension = len(parameters_names)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters initial guess
        init_shot = optimization_function.get_init_shot(is_normalized=True)
        x0 = np.array([init_shot[str(param)] for param in parameters_names])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters lower and upper bounds
        lower_bounds, upper_bounds = optimization_function.get_bounds(is_normalized=True)
        # Build optimization parameters bounds array
        bounds = np.array([(lower_bounds[str(param)], upper_bounds[str(param)])
                           for param in parameters_names]).reshape(dimension, 2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set algorithmic parameters
        options = {'maxiter': max_n_iter,}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve optimization problem
        result = scipy.optimize.minimize(fun=norm_opt_function_seq,
                                         x0=x0,
                                         method=self._method,
                                         bounds=bounds,
                                         options=options)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization solution
        parameters = {str(param): result.x[i] for i, param in enumerate(parameters_names)}
        optimum = result.fun
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
# ------------------------------------------------------------------------------------------
class ScikitOptimizer(Optimizer):
    '''Scikit optimizer.

    Scikit optimizer documentation:
    https://scikit-optimize.github.io/stable/modules/minimize_functions.html
    '''
    def __init__(self, method='dummy_minimize'):
        '''Optimization algorithm constructor.

        Parameters
        ----------
        method : str, {'dummy_minimize', 'forest_minimize', 'gbrt_minimize', 'gp_minimize'},
                 default='dummy_minimize'
            Optimization method.
        '''
        self._method = method
    # --------------------------------------------------------------------------------------
    def solve_optimization(self, optimization_function, max_n_iter=1, verbose=False):
        '''Solve optimization problem.

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
            Optimization parameters.
        '''
        # Get optimization function with normalized parameters provided as sequence
        norm_opt_function_seq = optimization_function.norm_opt_function_seq
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization function parameters names
        parameters_names = optimization_function.get_parameters_names()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters initial guess
        init_shot = optimization_function.get_init_shot(is_normalized=True)
        x0 = [init_shot[str(param)] for param in parameters_names]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters lower and upper bounds
        lower_bounds, upper_bounds = optimization_function.get_bounds(is_normalized=True)
        # Build optimization parameters bounds array
        dimensions = [(lower_bounds[str(param)], upper_bounds[str(param)])
                      for param in parameters_names]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Set algorithmic parameters
        n_calls = max_n_iter
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization callable
        if self._method == 'forest_minimize':
            skopt_minimize = skopt.forest_minimize
        elif self._method == 'gbrt_minimize':
            skopt_minimize = skopt.gbrt_minimize
        elif self._method == 'gp_minimize':
            skopt_minimize = skopt.gp_minimize
        else:
            skopt_minimize = skopt.dummy_minimize
        # Solve optimization problem
        result = skopt_minimize(func=norm_opt_function_seq,
                                dimensions=dimensions,
                                n_calls=n_calls,
                                x0=x0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization solution
        parameters = {str(param): result.x[i] for i, param in enumerate(parameters_names)}
        optimum = result.fun
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
