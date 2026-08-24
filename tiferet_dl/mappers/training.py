"""Tiferet DL training mappers."""

# *** imports

# ** core
from typing import List

# ** app
from tiferet.mappers import Aggregate

from ..domain.training import LinearRegressionModel
from ..utils.gradient_descent import GradientDescent

# *** mappers

# ** mapper: linear_regression_model_aggregate
class LinearRegressionModelAggregate(LinearRegressionModel, Aggregate):
    '''
    Mutable aggregate for the LinearRegressionModel domain object, owning
    the gradient-descent parameter update so numerical training state never
    mutates directly through the event layer.
    '''

    # * method: apply_epoch
    def apply_epoch(
            self,
            features: List[float],
            targets: List[float],
            learning_rate: float,
        ) -> None:
        '''
        Apply one gradient-descent epoch's parameter update in place,
        delegating the loss/gradient math to the stateless GradientDescent
        utility and translating its primitive results back into validated
        attribute mutations.

        :param features: The scalar input values for each observation.
        :type features: List[float]
        :param targets: The scalar target values for each observation.
        :type targets: List[float]
        :param learning_rate: The gradient-descent update magnitude.
        :type learning_rate: float
        :return: None
        :rtype: None
        '''

        # Delegate loss and gradient computation to the stateless utility.
        result = GradientDescent.compute_loss_and_gradients(
            features=features,
            targets=targets,
            weight=self.weight,
            bias=self.bias,
        )

        # Apply the parameter update using the computed loss and gradients.
        self.set_attribute('weight', self.weight - learning_rate * result['weight_gradient'])
        self.set_attribute('bias', self.bias - learning_rate * result['bias_gradient'])
        self.set_attribute('loss', result['loss'])
        self.set_attribute('epochs_trained', self.epochs_trained + 1)
