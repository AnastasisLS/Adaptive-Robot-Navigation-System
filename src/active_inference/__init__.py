"""
Active inference implementation for adaptive robot navigation.

Provides the core components of active inference:
- Generative model for world representation
- Recognition model for belief updates
- Policy model for action selection
- Variational inference algorithms
"""

from .generative_model import GenerativeModel
from .recognition_model import RecognitionModel
from .policy_model import PolicyModel
from .variational_inference import VariationalInference

__all__ = [
    "GenerativeModel",
    "RecognitionModel", 
    "PolicyModel",
    "VariationalInference"
] 