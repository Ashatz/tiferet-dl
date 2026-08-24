"""Tiferet DL training domain models."""

# *** imports

# ** infra
from pydantic import Field, model_validator

# ** app
from tiferet.domain import DomainObject, ModelError

# *** constants

# ** constant: empty_regression_dataset_id
EMPTY_REGRESSION_DATASET_ID = 'EMPTY_REGRESSION_DATASET'

# ** constant: misaligned_regression_observations_id
MISALIGNED_REGRESSION_OBSERVATIONS_ID = 'MISALIGNED_REGRESSION_OBSERVATIONS'

# *** models

# ** model: regression_dataset
class RegressionDataset(DomainObject):
    '''
    Holds the scalar observations consumed by the linear-regression training
    step, preserving the data as an explicit result in the declared pipeline.
    '''

    # * attribute: features
    features: list[float] = Field(
        ...,
        description='The scalar input values for each training observation.',
    )

    # * attribute: targets
    targets: list[float] = Field(
        ...,
        description='The scalar target values for each training observation.',
    )

    # * method: validate_observations (validator)
    @model_validator(mode='after')
    def validate_observations(self) -> 'RegressionDataset':
        '''
        Require non-empty, aligned feature and target observations.

        :return: The validated dataset.
        :rtype: RegressionDataset
        '''

        # Require at least one training observation.
        if not self.features:
            ModelError.raise_error(
                EMPTY_REGRESSION_DATASET_ID,
                message='A regression dataset requires at least one feature.',
                model=self,
            )

        # Require a target for every feature observation.
        if len(self.features) != len(self.targets):
            ModelError.raise_error(
                MISALIGNED_REGRESSION_OBSERVATIONS_ID,
                message='Regression features and targets must have equal lengths.',
                model=self,
                feature_count=len(self.features),
                target_count=len(self.targets),
            )

        # Return the validated dataset.
        return self


# ** model: linear_regression_model
class LinearRegressionModel(DomainObject):
    '''
    Captures the immutable parameters and loss emitted by one linear-regression
    training epoch, making the pipeline output directly inspectable.
    '''

    # * attribute: weight
    weight: float = Field(
        default=0.0,
        description='The coefficient applied to the scalar input feature.',
    )

    # * attribute: bias
    bias: float = Field(
        default=0.0,
        description='The intercept added to each model prediction.',
    )

    # * attribute: loss
    loss: float = Field(
        default=0.0,
        description='The mean squared error measured before the parameter update.',
    )

    # * attribute: epochs_trained
    epochs_trained: int = Field(
        default=0,
        description='The number of completed gradient-descent epochs.',
    )

    # * method: predict
    def predict(self, feature: float) -> float:
        '''
        Predict a target value for a single scalar feature.

        :param feature: The scalar feature value.
        :type feature: float
        :return: The linear-model prediction.
        :rtype: float
        '''

        # Apply the linear prediction equation.
        return self.weight * feature + self.bias
