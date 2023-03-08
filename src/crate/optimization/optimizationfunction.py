"""Parametric optimization function and loss function interfaces.

This module includes the interface to implement any parameterized function for
optimization purposes. This interface offers convenient methods to handle
optimization parameters, namely their specification (named or sequential),
their bounds, and their normalization/denormalization.

This module also includes the interface of a general loss function defined
between a given set of values of parametric and reference solutions, allowing
the implementation of common loss functions such as the Relative Root Mean
Squared Error (RRMSE).

Classes
-------
OptimizationFunction
    Optimization function interface.
Loss
    Loss function interface.
RelativeRootMeanSquaredError
    Relative Root Mean Squared Error (RRMSE).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from abc import ABC, abstractmethod
import copy
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
#
#                                              Interface: Optimization function
# =============================================================================
class OptimizationFunction(ABC):
    """Optimization function interface.

    Methods
    -------
    opt_function(self, parameters)
        *abstract*: Optimization function.
    get_parameters_names(self)
        Get optimization parameters names.
    get_bounds(self, is_normalized=False)
        Get optimization parameters lower and upper bounds.
    get_init_shot(self, is_normalized=False)
        Get optimization parameters initial guess.
    set_norm_bounds(self, norm_min=-1.0, norm_max=1.0)
        Set optimization parameters normalization bounds.
    get_norm_bounds(self)
        Get optimization parameters normalization bounds.
    normalize(self, parameters)
        Normalize optimization parameters between min and max values.
    denormalize(self, norm_parameters)
        Recover optimization parameters from normalized values.
    norm_opt_function(self, norm_parameters)
        Wrapper of optimization function with normalized parameters.
    opt_function_seq(self, parameters_seq)
        Wrapper of optimization function with sequential parameters.
    norm_opt_function_seq(self, parameters_seq)
        Wrapper of optimization function with norm. sequential parameters.
    """
    @abstractmethod
    def __init__(self, lower_bounds, upper_bounds, init_shot=None,
                 weights=None):
        """Constructor.

        Parameters
        ----------
        lower_bounds : dict
            Optimization parameters (key, str) lower bounds (item, float).
        upper_bounds : dict
            Optimization parameters (key, str) upper bounds (item, float).
        init_shot : dict, default=None
            Optimization parameters (key, str) initial guess (item, float).
        weights : tuple, default=None
            Weights attributed to each data point.
        """
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def opt_function(self, parameters):
        """Optimization function.

        Parameters
        ----------
        parameters : dict
            Optimization parameters names (key, str) and values (item, float).

        Returns
        -------
        value : float
            Optimization function value.
        """
        pass
    # -------------------------------------------------------------------------
    def get_parameters_names(self):
        """Get optimization parameters names.

        Returns
        -------
        parameters_names : tuple[str]
            Optimization parameters names (str).
        """
        return copy.deepcopy(self._parameters_names)
    # -------------------------------------------------------------------------
    def get_bounds(self, is_normalized=False):
        """Get optimization parameters lower and upper bounds.

        Parameters
        ----------
        is_normalized : bool, default=False
            Whether optimization parameters are normalized or not.

        Returns
        -------
        lower_bounds : dict
            Optimization parameters (key, str) lower bounds (item, float).
        upper_bounds : dict
            Optimization parameters (key, str) upper bounds (item, float).
        """
        # Get optimization function parameters lower and upper bounds
        if is_normalized:
            # Get normalized lower and upper bounds
            lower_bounds = {str(param): self._norm_bounds[0]
                            for param in self._parameters_names}
            upper_bounds = {str(param): self._norm_bounds[1]
                            for param in self._parameters_names}
        else:
            # Get lower and upper bounds
            lower_bounds = copy.deepcopy(self._lower_bounds)
            upper_bounds = copy.deepcopy(self._upper_bounds)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return lower_bounds, upper_bounds
    # -------------------------------------------------------------------------
    def get_init_shot(self, is_normalized=False):
        """Get optimization parameters initial guess.

        Parameters
        ----------
        is_normalized : bool, default=False
            Whether optimization parameters are normalized or not.

        Returns
        -------
        init_shot : dict
            Optimization parameters (key, str) initial guess (item, float).
        """
        # Check if optimization parameters initial guess is defined
        if self._init_shot is None:
            raise RuntimeError('Optimization parameters initial guess is not '
                               'defined.')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Get optimization parameters initial guess
        if is_normalized:
            init_shot = self.normalize(self._init_shot)
        else:
            init_shot = copy.deepcopy(self._init_shot)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return init_shot
    # -------------------------------------------------------------------------
    def set_norm_bounds(self, norm_min=-1.0, norm_max=1.0):
        """Set optimization parameters normalization bounds.

        Parameters
        ----------
        norm_min : float, default=-1.0
            Normalized optimization parameter lower bound.
        norm_max : float, default=1.0
            Normalized optimization parameter upper bound.
        """
        # Set normalization bounds
        self._norm_bounds = (norm_min, norm_max)
    # -------------------------------------------------------------------------
    def get_norm_bounds(self):
        """Get optimization parameters normalization bounds.

        Returns
        -------
        norm_bounds : tuple
            Normalization bounds (lower, upper) used to perform the
            normalization of the optimization parameters.
        """
        return copy.deepcopy(self._norm_bounds)
    # -------------------------------------------------------------------------
    def normalize(self, parameters):
        """Normalize optimization parameters between min and max values.

        Parameters
        ----------
        parameters : dict
            Optimization parameters names (key, str) and values (item, float).
        norm_min : float, default=-1.0
            Normalized optimization parameter lower bound.
        norm_max : float, default=1.0
            Normalized optimization parameter upper bound.

        Returns
        -------
        norm_parameters : dict
            Normalized optimization parameters names (key, str) and values
            (item, float).
        """
        # Get optimization parameters normalization bounds
        norm_min = self._norm_bounds[0]
        norm_max = self._norm_bounds[1]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize normalized optimization parameters
        norm_parameters = {}
        # Loop over optimization parameters
        for param in self._parameters_names:
            # Get optimization parameter value
            value = parameters[str(param)]
            # Get optimization parameter lower and upper bounds
            lbound = self._lower_bounds[str(param)]
            ubound = self._upper_bounds[str(param)]
            # Normalize optimization parameter
            norm_parameters[str(param)] = norm_min \
                + ((value - lbound)/(ubound - lbound))*(norm_max - norm_min)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return norm_parameters
    # -------------------------------------------------------------------------
    def denormalize(self, norm_parameters):
        """Recover optimization parameters from normalized values.

        Parameters
        ----------
        norm_parameters : dict
            Normalized optimization parameters names (key, str) and values
            (item, float).

        Returns
        -------
        parameters : dict
            Optimization parameters names (key, str) and values (item, float).
        """
        # Get optimization parameters normalization bounds
        norm_min, norm_max = self._norm_bounds
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize optimization function parameters
        parameters = {}
        # Loop over optimization parameters
        for param in self._parameters_names:
            # Get optimization parameter value
            norm_value = norm_parameters[str(param)]
            # Get optimization parameter lower and upper bounds
            lbound = self._lower_bounds[str(param)]
            ubound = self._upper_bounds[str(param)]
            # Recover optimization parameter
            parameters[str(param)] = lbound \
                + ((norm_value - norm_min)/(norm_max - norm_min))*(ubound
                                                                   - lbound)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return parameters
    # -------------------------------------------------------------------------
    def norm_opt_function(self, norm_parameters):
        """Wrapper of optimization function with normalized parameters.

        Parameters
        ----------
        norm_parameters : dict
            Normalized optimization parameters names (key, str) and values
            (item, float).

        Returns
        -------
        value : float
            Optimization function value.
        """
        # Recover optimization parameters from normalized values
        parameters = self.denormalize(norm_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute optimization function
        value = self.opt_function(parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return value
    # -------------------------------------------------------------------------
    def opt_function_seq(self, parameters_seq):
        """Wrapper of optimization function with sequential parameters.

        Parameters
        ----------
        parameters_seq : tuple[float]
            Optimization parameters values.

        Returns
        -------
        value : float
            Optimization function value.
        """
        # Build optimization parameters dictionary
        parameters = {str(param): parameters_seq[i]
                      for i, param in enumerate(self._parameters_names)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute optimization function
        value = self.opt_function(parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return value
    # -------------------------------------------------------------------------
    def norm_opt_function_seq(self, parameters_seq):
        """Wrapper of optimization function with norm. sequential parameters.

        Parameters
        ----------
        parameters_seq : tuple[float]
            Optimization parameters values.

        Returns
        -------
        value : float
            Optimization function value.
        """
        # Build normalized optimization parameters dictionary
        norm_parameters = {str(param): parameters_seq[i]
                           for i, param in enumerate(self._parameters_names)}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Recover optimization parameters from normalized values
        parameters = self.denormalize(norm_parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute optimization function
        value = self.opt_function(parameters)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return value
#
#                                                      Interface: Loss function
# =============================================================================
class Loss(ABC):
    """Loss function interface.

    Methods
    -------
    loss(self, y, y_ref, type='minimization')
        *abstract*: Loss function.
    """
    @abstractmethod
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    @abstractmethod
    def loss(self, y, y_ref, type='minimization'):
        """Loss function.

        Parameters
        ----------
        y : tuple[float]
            Values of parametric solution.
        y_ref : tuple[float]
            Values of reference solution.
        type : {'minimization', 'maximization'}, default='minimization'
            Type of optimization problem. The option 'maximization' negates the
            loss function evaluation.

        Returns
        -------
        loss : float
            Loss function value.
        """
        pass
#
#                                                                Loss functions
# =============================================================================
class RelativeRootMeanSquaredError(Loss):
    """Relative Root Mean Squared Error (RRMSE).

    Methods
    -------
    loss(self, y, y_ref, type='minimization')
        Loss function.
    """
    def __init__(self):
        """Constructor."""
        pass
    # -------------------------------------------------------------------------
    def loss(self, y, y_ref, type='minimization'):
        """Loss function.

        The Relative Root Mean Squared Error (RRMSE) is defined as

        .. math::

           \\text{RRMSE} (\\boldsymbol{y}, \\hat{\\boldsymbol{y}}) =
           \\sqrt{\\dfrac{\\dfrac{1}{n} \\sum_{i=1}^{n}(y_{i} -
           \\hat{y}_{i})^{2}}{ \\sum_{i=1}^{n} \\hat{y}_{i}^{2}}}

        where :math:`\\boldsymbol{y}` is the vector of predicted values,
        :math:`\\hat{\\boldsymbol{y}}` is the vector of reference values,
        and :math:`n` is the number of data points.

        ----

        Parameters
        ----------
        y : tuple[float]
            Values of parametric solution.
        y_ref : tuple[float]
            Values of reference solution.
        type : {'minimization', 'maximization'}, default='minimization'
            Type of optimization problem. The option 'maximization' negates the
            loss function evaluation.

        Returns
        -------
        loss : float
            Loss function value.
        """
        # Get number of data points
        n = len(y)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute required summations
        sum_1 = sum([(y[i] - y_ref[i])**2 for i in range(n)])
        sum_2 = sum([y_ref[i]**2 for i in range(n)])
        # Compute relative root mean squared error
        loss = np.sqrt(((1.0/n)*sum_1)/sum_2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Negate loss function if maximization optimization
        if type == 'maximization':
            loss = -loss
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Return
        return loss
