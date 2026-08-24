"""Tiferet DL training domain models."""

# *** imports

# ** infra
from pydantic import Field, model_validator

# ** app
from tiferet.domain import DomainObject

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
            raise ValueError('A regression dataset requires at least one feature.')

        # Require a target for every feature observation.
        if len(self.features) != len(self.targets):
            raise ValueError('Regression features and targets must have equal lengths.')

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
