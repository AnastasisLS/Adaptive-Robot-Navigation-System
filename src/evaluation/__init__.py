"""
Evaluation module for adaptive robot navigation system.

Provides metrics calculation, agent comparison, and performance analysis
for evaluating active inference agents against traditional RL methods.
"""

from .metrics import NavigationMetrics
from .comparison import AgentComparison

__all__ = ['NavigationMetrics', 'AgentComparison'] 