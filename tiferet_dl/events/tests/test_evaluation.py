"""Tiferet DL Evaluation Event Tests"""

# *** imports

# ** core
from typing import Sequence

# ** infra
import pytest
from tiferet.events import DomainEvent

# ** app
from tiferet_dl.domain.evaluation import EvaluationResult
from tiferet_dl.events.evaluation import EvaluateModel
from tiferet_dl.interfaces.evaluation import EvaluationService

# *** classes

# ** class: mean_squared_error_evaluation_service
class MeanSquaredErrorEvaluationService(EvaluationService):
    '''
    Test implementation that scores the linear-regression anchor technique
    using mean squared error.
    '''

    # * method: evaluate
    def evaluate(
            self,
            predictions: Sequence[float],
            targets: Sequence[float],
            role: str,
        ) -> EvaluationResult:
        '''
        Compute mean squared error for one data role.

        :param predictions: The numeric model predictions.
        :type predictions: Sequence[float]
        :param targets: The numeric targets.
        :type targets: Sequence[float]
        :param role: The evaluated data role.
        :type role: str
        :return: The structured mean-squared-error result.
        :rtype: EvaluationResult
        '''

        # Compute squared errors across corresponding prediction-target pairs.
        squared_errors = [
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets)
        ]

        # Return the role-specific structured risk result.
        return EvaluationResult(
            role=role,
            risk=sum(squared_errors) / len(squared_errors),
            sample_count=len(squared_errors),
            metric='mean_squared_error',
        )

# *** fixtures

# ** fixture: evaluation_service
@pytest.fixture
def evaluation_service() -> EvaluationService:
    '''
    Provide the concrete anchor-technique evaluation service.

    :return: The mean-squared-error evaluation service.
    :rtype: EvaluationService
    '''

    # Return the scorer used by the evaluation event.
    return MeanSquaredErrorEvaluationService()

# *** tests

# ** test: evaluate_model_returns_role_specific_evaluation_results
@pytest.mark.parametrize(
    ('predictions', 'targets', 'role', 'expected_risk'),
    [
        ([1.0, 2.0], [1.0, 3.0], 'train', 0.5),
        ([1.0, 2.0], [2.0, 4.0], 'held_out', 2.5),
    ],
)
def test_evaluate_model_returns_role_specific_evaluation_results(
        evaluation_service: EvaluationService,
        predictions: Sequence[float],
        targets: Sequence[float],
        role: str,
        expected_risk: float,
    ) -> None:
    '''
    Score train and held-out predictions through the same event contract.

    :param evaluation_service: The injected evaluation service.
    :type evaluation_service: EvaluationService
    :param predictions: The trained model's predictions.
    :type predictions: Sequence[float]
    :param targets: The target values.
    :type targets: Sequence[float]
    :param role: The evaluated data role.
    :type role: str
    :param expected_risk: The expected mean squared error.
    :type expected_risk: float
    '''

    # Evaluate the supplied model predictions via constructor injection.
    result = DomainEvent.handle(
        EvaluateModel,
        dependencies={'evaluation_service': evaluation_service},
        predictions=predictions,
        targets=targets,
        role=role,
    )

    # Preserve both the risk value and the dataset role in a domain object.
    assert isinstance(result, EvaluationResult)
    assert result.role == role
    assert result.risk == expected_risk
    assert result.sample_count == len(predictions)
    assert result.metric == 'mean_squared_error'
