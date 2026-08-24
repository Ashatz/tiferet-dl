"""Tiferet DL Evaluation Service Interface"""

# *** imports

# ** core
from abc import abstractmethod
from typing import Sequence

# ** app
from tiferet.interfaces import Service

from ..domain.evaluation import EvaluationResult

# *** interfaces

# ** interface: evaluation_service
class EvaluationService(Service):
    '''
    Defines the swappable boundary that scores model predictions while
    preserving whether the resulting risk came from training or held-out data.
    '''

    # * method: evaluate
    @abstractmethod
    def evaluate(
            self,
            predictions: Sequence[float],
            targets: Sequence[float],
            role: str,
        ) -> EvaluationResult:
        '''
        Score predictions against targets for one dataset role.

        :param predictions: The trained model's numeric predictions.
        :type predictions: Sequence[float]
        :param targets: The numeric target values paired with predictions.
        :type targets: Sequence[float]
        :param role: The data role, either train or held_out.
        :type role: str
        :return: The structured evaluation result.
        :rtype: EvaluationResult
        '''

        raise NotImplementedError()
