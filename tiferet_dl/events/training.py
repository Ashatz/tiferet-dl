"""Tiferet DL training events."""

# *** imports

# ** app
from tiferet.events import DomainEvent

from ..domain import (
    LinearRegressionModel,
    RegressionDataset,
)

# *** events

# ** event: load_dataset
class LoadDataset(DomainEvent):
    '''
    Converts request observations into a regression dataset so later pipeline
    steps consume one named, validated training input.
    '''

    # * method: execute
    @DomainEvent.parameters_required([
        'features',
        'targets',
    ])
    def execute(
            self,
            features: list[float],
            targets: list[float],
            **kwargs,
        ) -> RegressionDataset:
        '''
        Load scalar regression observations into a domain dataset.

        :param features: The scalar input values.
        :type features: list[float]
        :param targets: The scalar target values.
        :type targets: list[float]
        :param kwargs: Additional pipeline values.
        :type kwargs: dict
        :return: The validated regression dataset.
        :rtype: RegressionDataset
        '''

        # Construct and return the validated dataset result.
        return RegressionDataset(
            features=features,
            targets=targets,
        )


# ** event: train_epoch
class TrainEpoch(DomainEvent):
    '''
    Executes exactly one gradient-descent epoch for scalar linear regression,
    keeping the pipeline's training granularity explicit rather than looping
    internally over an opaque training run.
    '''

    # * method: execute
    @DomainEvent.parameters_required([
        'dataset',
    ])
    def execute(
            self,
            dataset: RegressionDataset,
            learning_rate: float | str = 0.1,
            model: LinearRegressionModel | None = None,
            **kwargs,
        ) -> LinearRegressionModel:
        '''
        Train a scalar linear-regression model for one full dataset epoch.

        :param dataset: The validated observations to train on.
        :type dataset: RegressionDataset
        :param learning_rate: The gradient-descent update magnitude.
        :type learning_rate: float | str
        :param model: Optional parameters from a prior epoch.
        :type model: LinearRegressionModel | None
        :param kwargs: Additional pipeline values.
        :type kwargs: dict
        :return: The updated linear-regression model.
        :rtype: LinearRegressionModel
        '''

        # Use zero-valued parameters when this is the first epoch.
        current_model = model or LinearRegressionModel()

        # Normalize declarative YAML parameter values before numerical training.
        resolved_learning_rate = float(learning_rate)

        # Calculate predictions and residuals from the current parameters.
        predictions = [
            current_model.predict(feature)
            for feature in dataset.features
        ]
        residuals = [
            prediction - target
            for prediction, target in zip(predictions, dataset.targets)
        ]

        # Compute the mean squared loss and its gradients.
        observation_count = len(dataset.features)
        loss = sum(residual ** 2 for residual in residuals) / observation_count
        weight_gradient = (
            2 / observation_count
            * sum(
                residual * feature
                for residual, feature in zip(residuals, dataset.features)
            )
        )
        bias_gradient = 2 / observation_count * sum(residuals)

        # Return new immutable parameters after the single epoch update.
        return LinearRegressionModel(
            weight=current_model.weight - resolved_learning_rate * weight_gradient,
            bias=current_model.bias - resolved_learning_rate * bias_gradient,
            loss=loss,
            epochs_trained=current_model.epochs_trained + 1,
        )
