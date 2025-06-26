"""
Environment package for adaptive robot navigation.

Provides simulation environments for testing active inference agents.
"""

from .navigation_env import NavigationEnvironment
from .sensors import LIDARSensor, CameraSensor, IMUSensor

__all__ = [
    "NavigationEnvironment",
    "LIDARSensor", 
    "CameraSensor",
    "IMUSensor"
] 