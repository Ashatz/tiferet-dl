"""Tiferet DL Evaluation Domain Objects"""

# *** imports

# ** infra
from pydantic import Field

# ** app
from tiferet.domain import DomainObject

# *** models

# ** model: evaluation_result
class EvaluationResult(DomainObject):
    '''
    Preserves a computed risk with the data role and metric that make its
    generalization meaning inspectable outside a training loop.
    '''

    # * attribute: role
    role: str = Field(
        ...,
        pattern='^(train|held_out)$',
        description='The dataset role evaluated: train or held_out.',
    )

    # * attribute: risk
    risk: float = Field(
        ...,
        ge=0,
        description='The non-negative risk computed for the dataset role.',
    )

    # * attribute: sample_count
    sample_count: int = Field(
        ...,
        gt=0,
        description='The number of prediction-target pairs evaluated.',
    )

    # * attribute: metric
    metric: str = Field(
        ...,
        min_length=1,
        description='The named loss metric used to compute risk.',
    )
