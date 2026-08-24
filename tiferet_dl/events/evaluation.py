"""Tiferet DL Evaluation Domain Events"""

# *** imports

# ** core
from typing import Sequence

# ** app
from tiferet.events import DomainEvent

from ..domain.evaluation import EvaluationResult
from ..interfaces.evaluation import EvaluationService

# *** events

# ** event: evaluate_model
class EvaluateModel(DomainEvent):
    '''
    Makes model evaluation an ordinary injected pipeline operation so empirical
    and held-out risk are returned through the same result path as other steps.
    '''

    # * attribute: evaluation_service
    evaluation_service: EvaluationService

    # * init
    def __init__(self, evaluation_service: EvaluationService) -> None:
        '''
        Initialize the event with its evaluation service dependency.

        :param evaluation_service: The service used to score predictions.
        :type evaluation_service: EvaluationService
        '''

        # Store the injected evaluation service.
        self.evaluation_service = evaluation_service

    # * method: execute
    @DomainEvent.parameters_required([
        'predictions',
        'targets',
        'role',
    ])
    def execute(
            self,
            predictions: Sequence[float],
            targets: Sequence[float],
            role: str,
            **kwargs,
        ) -> EvaluationResult:
        '''
        Score a trained model's predictions for a named dataset role.

        :param predictions: The trained model's numeric predictions.
        :type predictions: Sequence[float]
        :param targets: The numeric target values paired with predictions.
        :type targets: Sequence[float]
        :param role: The data role, either train or held_out.
        :type role: str
        :param kwargs: Additional pipeline data ignored by this event.
        :type kwargs: dict
        :return: The structured result returned by the evaluation service.
        :rtype: EvaluationResult
        '''

        # Delegate the scoring operation to the injected contract.
        result = self.evaluation_service.evaluate(
            predictions=predictions,
            targets=targets,
            role=role,
        )

        # Return the structured evaluation result for pipeline storage.
        return result
