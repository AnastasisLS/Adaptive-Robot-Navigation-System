"""
Test suite for adaptive robot navigation system.

Contains unit tests for all major components including:
- Environment and sensor systems
- Active inference framework
- Agent implementations
- Integration tests
"""

from .test_environment import TestNavigationEnvironment, TestSensors
from .test_active_inference import (
    TestGenerativeModel, TestRecognitionModel, 
    TestPolicyModel, TestVariationalInference
)
from .test_agents import (
    TestActiveInferenceAgent, TestPPOAgent, 
    TestDQNAgent, TestAgentIntegration
)

__all__ = [
    'TestNavigationEnvironment',
    'TestSensors', 
    'TestGenerativeModel',
    'TestRecognitionModel',
    'TestPolicyModel',
    'TestVariationalInference',
    'TestActiveInferenceAgent',
    'TestPPOAgent',
    'TestDQNAgent',
    'TestAgentIntegration'
] 