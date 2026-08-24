"""Tiferet DL gradient-descent computational utility."""

# *** imports

# ** core
from typing import Dict, List

# *** utils

# ** util: gradient_descent
class GradientDescent:
    '''
    Isolates the scalar linear-regression loss/gradient math as a stateless
    computation over primitives, so the numerical training algorithm can be
    swapped or optimized without touching the domain or event layers.
    '''

    # * method: compute_loss_and_gradients (static)
    @staticmethod
    def compute_loss_and_gradients(
            features: List[float],
            targets: List[float],
            weight: float,
            bias: float,
        ) -> Dict[str, float]:
        '''
        Compute the mean squared error loss and its gradients for one
        linear-regression epoch over the given scalar observations and
        current parameters.

        :param features: The scalar input values for each observation.
        :type features: List[float]
        :param targets: The scalar target values for each observation.
        :type targets: List[float]
        :param weight: The current linear model weight.
        :type weight: float
        :param bias: The current linear model bias.
        :type bias: float
        :return: A dict with ``loss``, ``weight_gradient``, and ``bias_gradient`` keys.
        :rtype: Dict[str, float]
        '''

        # Calculate predictions and residuals from the current parameters.
        predictions = [weight * feature + bias for feature in features]
        residuals = [
            prediction - target
            for prediction, target in zip(predictions, targets)
        ]

        # Compute the mean squared loss and its gradients.
        observation_count = len(features)
        loss = sum(residual ** 2 for residual in residuals) / observation_count
        weight_gradient = (
            2 / observation_count
            * sum(
                residual * feature
                for residual, feature in zip(residuals, features)
            )
        )
        bias_gradient = 2 / observation_count * sum(residuals)

        # Return the loss and gradients as plain primitives.
        return {
            'loss': loss,
            'weight_gradient': weight_gradient,
            'bias_gradient': bias_gradient,
        }
