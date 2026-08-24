"""Tiferet DL training blueprint."""

# *** imports

# ** core
from pathlib import Path
from typing import Any

# ** app
from tiferet.contexts.feature import FeatureContext
from tiferet.contexts.request import RequestContext
from tiferet.di import DIDynamicServiceResolver
from tiferet.repos.di import DIConfigRepository
from tiferet.repos.feature import FeatureConfigRepository

from ..domain import LinearRegressionModel

# *** blueprints

# ** blueprint: run_training
def run_training(
        config_file: str | Path,
        feature_id: str,
        data: dict[str, Any],
    ) -> LinearRegressionModel:
    '''
    Run a training feature declared in configuration through the framework's
    FeatureContext and dynamic DI resolver.

    The configuration, rather than this blueprint, selects and orders each
    training event. A final step without ``data_key`` becomes the return value.

    :param config_file: The YAML file holding feature and service declarations.
    :type config_file: str | Path
    :param feature_id: The fully qualified feature identifier to execute.
    :type feature_id: str
    :param data: Runtime request data passed to the configured steps.
    :type data: dict[str, Any]
    :return: The model returned by the terminal configured training step.
    :rtype: LinearRegressionModel
    '''

    # Normalize the configuration path for both framework repositories.
    config_path = str(config_file)

    # Load the declared feature workflow and its DI service registrations.
    feature = FeatureConfigRepository(config_path).get(feature_id)
    resolver = DIDynamicServiceResolver(DIConfigRepository(config_path))

    # Bind the framework context to the declared feature and DI handler.
    context = FeatureContext.from_domain(
        feature,
        get_dependency=resolver.get_dependency,
    )
    request = RequestContext(
        feature_id=feature.id,
        data=data,
    )

    # Execute the declared workflow and return its terminal result.
    context.execute_feature(request)
    return request.handle_response()
