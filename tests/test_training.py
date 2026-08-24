"""Tiferet DL training pipeline tests."""

# *** imports

# ** core
from pathlib import Path

# ** infra
import pytest
from pydantic import ValidationError

# ** app
from tiferet.events import DomainEvent

from tiferet_dl.blueprints import run_training
from tiferet_dl.domain import (
    LinearRegressionModel,
    RegressionDataset,
)
from tiferet_dl.events import TrainEpoch

# *** constants

# ** constant: training_config
TRAINING_CONFIG = (
    Path(__file__).parents[1]
    / 'config'
    / 'training.yml'
)


# ** constant: training_data
TRAINING_DATA = {
    'features': [
        1.0,
        2.0,
        3.0,
    ],
    'targets': [
        2.0,
        4.0,
        6.0,
    ],
}

# *** tests

# ** test: train_epoch_returns_updated_model
def test_train_epoch_returns_updated_model():
    '''
    TrainEpoch returns one updated, immutable model for supplied observations.
    '''

    # Execute the event through the standard event test entry point.
    model = DomainEvent.handle(
        TrainEpoch,
        dataset=RegressionDataset(**TRAINING_DATA),
        learning_rate=0.1,
    )

    # Verify the result captures exactly one training epoch and updated parameters.
    assert model.epochs_trained == 1
    assert model.weight > 0
    assert model.bias > 0
    assert model.loss > 0


# ** test: regression_dataset_rejects_misaligned_observations
def test_regression_dataset_rejects_misaligned_observations():
    '''
    RegressionDataset rejects observations that cannot form feature-target pairs.
    '''

    # Verify mismatched observations fail domain validation.
    with pytest.raises(ValidationError):
        RegressionDataset(
            features=[
                1.0,
                2.0,
            ],
            targets=[
                2.0,
            ],
        )


# ** test: config_declared_pipeline_runs_through_di
def test_config_declared_pipeline_runs_through_di():
    '''
    The declared Feature workflow resolves both events through DI and returns
    its terminal linear-regression model.
    '''

    # Execute the config-selected feature through the application blueprint.
    model = run_training(
        config_file=TRAINING_CONFIG,
        feature_id='training.linear_regression',
        data=TRAINING_DATA,
    )

    # Verify the terminal pipeline result is the trained model from the config.
    assert isinstance(model, LinearRegressionModel)
    assert model.epochs_trained == 1
    assert model.weight > 0
