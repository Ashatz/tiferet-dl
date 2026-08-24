"""Tiferet DL public exports."""

# *** imports

# ** app
from .blueprints.training import run_training

# *** exports

__all__ = [
    'run_training',
]
