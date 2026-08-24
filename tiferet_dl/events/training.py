"""Tiferet DL training events."""

# *** imports

# ** app
from tiferet.events import DomainEvent

from ..domain import (
    LinearRegressionModel,
    RegressionDataset,
)
from ..mappers import LinearRegressionModelAggregate

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

        # Load the current parameters into a mutable aggregate for the update.
        aggregate = LinearRegressionModelAggregate(**current_model.model_dump())

        # Delegate the gradient-descent parameter update to the aggregate.
        aggregate.apply_epoch(
            features=dataset.features,
            targets=dataset.targets,
            learning_rate=resolved_learning_rate,
        )

        # Return the mutated aggregate as this epoch's updated model.
        return aggregate
