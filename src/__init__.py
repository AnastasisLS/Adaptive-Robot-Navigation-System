"""
Adaptive Robot Navigation System Using Active Inference

A comprehensive implementation of active inference for autonomous robot navigation.
"""

__version__ = "0.1.0"
__author__ = "AI Assistant"

from .environment import NavigationEnvironment
from .agents import ActiveInferenceAgent
from .active_inference import GenerativeModel, RecognitionModel, PolicyModel

__all__ = [
    "NavigationEnvironment",
    "ActiveInferenceAgent", 
    "GenerativeModel",
    "RecognitionModel",
    "PolicyModel"
] 