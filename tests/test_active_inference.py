"""
Unit tests for active inference components.

Tests the core active inference framework including generative model,
recognition model, policy model, and variational inference.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import yaml

from src.active_inference import (
    GenerativeModel, RecognitionModel, PolicyModel, VariationalInference
)


class TestGenerativeModel(unittest.TestCase):
    """Test cases for GenerativeModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'active_inference': {
                    'state_dim': 32,
                    'observation_dim': 4142,
                    'action_dim': 4,
                    'precision': 1.0,
                    'learning_rate': 0.01,
                    'temperature': 1.0,
                    'num_samples': 5,
                    'kl_weight': 1.0
                },
                'neural_networks': {
                    'generative': {
                        'hidden_dims': [64, 32],
                        'activation': 'relu',
                        'dropout': 0.1
                    }
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create model
        self.model = GenerativeModel(self.temp_config.name)
        
        # Test data
        self.batch_size = 8
        self.states = torch.randn(self.batch_size, 32)
        self.actions = torch.randint(0, 4, (self.batch_size,))
        self.observations = torch.randn(self.batch_size, 4142)
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.state_dim, 32)
        self.assertEqual(self.model.observation_dim, 4142)
        self.assertEqual(self.model.action_dim, 4)
        
        # Check that all sub-models exist
        self.assertIsNotNone(self.model.observation_model)
        self.assertIsNotNone(self.model.transition_model)
        self.assertIsNotNone(self.model.prior_model)
        
        # Check precision parameters
        self.assertIsNotNone(self.model.observation_precision)
        self.assertIsNotNone(self.model.transition_precision)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Test with states only
        outputs = self.model.forward(self.states)
        
        # Check output structure
        required_keys = [
            'observation_mean', 'observation_logvar',
            'prior_mean', 'prior_logvar'
        ]
        for key in required_keys:
            self.assertIn(key, outputs)
            self.assertEqual(outputs[key].shape[0], self.batch_size)
        
        # Test with states and actions
        outputs = self.model.forward(self.states, self.actions)
        
        # Check additional outputs
        additional_keys = ['next_state_mean', 'next_state_logvar']
        for key in additional_keys:
            self.assertIn(key, outputs)
            self.assertEqual(outputs[key].shape[0], self.batch_size)
    
    def test_observation_sampling(self):
        """Test observation sampling."""
        observations = self.model.sample_observations(self.states)
        
        # Check output shape
        self.assertEqual(observations.shape, (self.batch_size, 4142))
        self.assertTrue(torch.all(torch.isfinite(observations)))
    
    def test_next_state_sampling(self):
        """Test next state sampling."""
        next_states = self.model.sample_next_states(self.states, self.actions)
        
        # Check output shape
        self.assertEqual(next_states.shape, (self.batch_size, 32))
        self.assertTrue(torch.all(torch.isfinite(next_states)))
    
    def test_log_probability_computation(self):
        """Test log probability computation."""
        # Observation log probability
        log_prob = self.model.compute_observation_log_prob(self.states, self.observations)
        self.assertEqual(log_prob.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(log_prob)))
        
        # Transition log probability
        next_states = torch.randn(self.batch_size, 32)
        log_prob = self.model.compute_transition_log_prob(self.states, self.actions, next_states)
        self.assertEqual(log_prob.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(log_prob)))
        
        # Prior log probability
        log_prob = self.model.compute_prior_log_prob(self.states)
        self.assertEqual(log_prob.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(log_prob)))


class TestRecognitionModel(unittest.TestCase):
    """Test cases for RecognitionModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'active_inference': {
                    'state_dim': 32,
                    'observation_dim': 4142,
                    'action_dim': 4,
                    'precision': 1.0,
                    'learning_rate': 0.01,
                    'temperature': 1.0,
                    'num_samples': 5,
                    'kl_weight': 1.0
                },
                'neural_networks': {
                    'recognition': {
                        'hidden_dims': [64, 32],
                        'activation': 'relu',
                        'dropout': 0.1
                    }
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create model
        self.model = RecognitionModel(self.temp_config.name)
        
        # Test data
        self.batch_size = 8
        self.observations = torch.randn(self.batch_size, 4142)
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.state_dim, 32)
        self.assertEqual(self.model.observation_dim, 4142)
        
        # Check that the network exists
        self.assertIsNotNone(self.model.recognition_network)
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        outputs = self.model.forward(self.observations)
        
        # Check output structure
        required_keys = ['state_mean', 'state_logvar']
        for key in required_keys:
            self.assertIn(key, outputs)
            self.assertEqual(outputs[key].shape, (self.batch_size, 32))
            self.assertTrue(torch.all(torch.isfinite(outputs[key])))
    
    def test_state_sampling(self):
        """Test state sampling."""
        states = self.model.sample_states(self.observations)
        
        # Check output shape
        self.assertEqual(states.shape, (self.batch_size, 32))
        self.assertTrue(torch.all(torch.isfinite(states)))
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        # Create prior parameters
        prior_mean = torch.randn(self.batch_size, 32)
        prior_logvar = torch.randn(self.batch_size, 32)
        
        # Get recognition parameters
        outputs = self.model.forward(self.observations)
        recog_mean = outputs['state_mean']
        recog_logvar = outputs['state_logvar']
        
        # Compute KL divergence
        kl_div = self.model.compute_kl_divergence(
            recog_mean, recog_logvar, prior_mean, prior_logvar
        )
        
        # Check output
        self.assertEqual(kl_div.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(kl_div)))
        self.assertTrue(torch.all(kl_div >= 0))  # KL divergence should be non-negative


class TestPolicyModel(unittest.TestCase):
    """Test cases for PolicyModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'active_inference': {
                    'state_dim': 32,
                    'observation_dim': 4142,
                    'action_dim': 4,
                    'precision': 1.0,
                    'learning_rate': 0.01,
                    'temperature': 1.0,
                    'num_samples': 5,
                    'kl_weight': 1.0
                },
                'policy': {
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995,
                    'min_exploration_rate': 0.01,
                    'efe_weight': 1.0,
                    'information_gain_weight': 0.5,
                    'utility_weight': 0.5
                },
                'neural_networks': {
                    'policy': {
                        'hidden_dims': [32, 16],
                        'activation': 'relu'
                    }
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create model
        self.model = PolicyModel(self.temp_config.name)
        
        # Test data
        self.batch_size = 8
        self.states = torch.randn(self.batch_size, 32)
        
        # Create mock generative and recognition models
        self.mock_generative = MockGenerativeModel()
        self.mock_recognition = MockRecognitionModel()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.state_dim, 32)
        self.assertEqual(self.model.action_dim, 4)
        
        # Check policy parameters
        self.assertEqual(self.model.efe_weight, 1.0)
        self.assertEqual(self.model.information_gain_weight, 0.5)
        self.assertEqual(self.model.utility_weight, 0.5)
    
    def test_expected_free_energy_computation(self):
        """Test expected free energy computation."""
        actions = torch.randint(0, 4, (self.batch_size,))
        efe = self.model.compute_expected_free_energy(
            self.states, actions, self.mock_generative, self.mock_recognition
        )
        
        # Check output shape
        self.assertEqual(efe.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(efe)))
    
    def test_policy_efe_computation(self):
        """Test policy expected free energy computation."""
        efe_matrix = self.model.compute_policy_efe(
            self.states, self.mock_generative, self.mock_recognition
        )
        
        # Check output shape
        self.assertEqual(efe_matrix.shape, (self.batch_size, 4))  # 4 actions
        self.assertTrue(torch.all(torch.isfinite(efe_matrix)))
    
    def test_action_selection(self):
        """Test action selection."""
        action = self.model.select_action(self.states, exploration_rate=0.1)
        
        # Check output
        self.assertEqual(action.shape, (self.batch_size,))
        self.assertTrue(torch.all((action >= 0) & (action < 4)))


class TestVariationalInference(unittest.TestCase):
    """Test cases for VariationalInference."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'active_inference': {
                    'state_dim': 32,
                    'observation_dim': 4142,
                    'action_dim': 4,
                    'precision': 1.0,
                    'learning_rate': 0.01,
                    'temperature': 1.0,
                    'num_samples': 5,
                    'kl_weight': 1.0
                },
                'optimization': {
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'weight_decay': 1e-5,
                    'gradient_clip': 1.0
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create model
        self.model = VariationalInference(self.temp_config.name)
        
        # Test data
        self.batch_size = 8
        self.observations = torch.randn(self.batch_size, 4142)
        self.states = torch.randn(self.batch_size, 32)
        self.actions = torch.randint(0, 4, (self.batch_size,))
        self.rewards = torch.randn(self.batch_size)
        
        # Create mock models
        self.mock_generative = MockGenerativeModel()
        self.mock_recognition = MockRecognitionModel()
        self.mock_policy = MockPolicyModel()
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.config['learning_rate'], 0.01)
        self.assertEqual(self.model.config['precision'], 1.0)
        self.assertEqual(self.model.config['temperature'], 1.0)
        self.assertEqual(self.model.config['num_samples'], 5)
        self.assertEqual(self.model.config['kl_weight'], 1.0)
    
    def test_free_energy_computation(self):
        """Test free energy computation."""
        fe = self.model.compute_variational_free_energy(
            self.observations, self.states, self.mock_generative, self.mock_recognition
        )
        
        # Check output
        self.assertEqual(fe.shape, (self.batch_size,))
        self.assertTrue(torch.all(torch.isfinite(fe)))
    
    def test_model_optimization(self):
        """Test model optimization."""
        losses = self.model.optimize_models(
            self.observations, self.states, self.actions, self.rewards,
            self.mock_generative, self.mock_recognition, self.mock_policy
        )
        
        # Check output structure
        required_keys = ['generative_loss', 'recognition_loss', 'policy_loss']
        for key in required_keys:
            self.assertIn(key, losses)
            self.assertIsInstance(losses[key], float)
            self.assertTrue(np.isfinite(losses[key]))
    
    def test_belief_update(self):
        """Test belief update."""
        belief_update = self.model.update_beliefs(
            self.observations, self.mock_generative, self.mock_recognition, num_iterations=3
        )
        
        # Check output structure
        self.assertIn('belief_mean', belief_update)
        self.assertEqual(belief_update['belief_mean'].shape, (self.batch_size, 32))
        self.assertTrue(torch.all(torch.isfinite(belief_update['belief_mean'])))


class MockGenerativeModel:
    """Mock generative model for testing."""
    
    def __init__(self):
        # Create a dummy parameter for the optimizer
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
    
    def __call__(self, states, actions=None):
        return self.forward(states, actions)
    
    def forward(self, states, actions=None):
        batch_size = states.shape[0]
        return {
            'observation_mean': torch.randn(batch_size, 4142, requires_grad=True),
            'observation_logvar': torch.randn(batch_size, 4142, requires_grad=True),
            'next_state_mean': torch.randn(batch_size, 32, requires_grad=True),
            'next_state_logvar': torch.randn(batch_size, 32, requires_grad=True),
            'prior_mean': torch.randn(batch_size, 32, requires_grad=True),
            'prior_logvar': torch.randn(batch_size, 32, requires_grad=True)
        }
    
    def sample_observations(self, states):
        return torch.randn(states.shape[0], 4142, requires_grad=True)
    
    def sample_next_states(self, states, actions):
        return torch.randn(states.shape[0], 32, requires_grad=True)
    
    def compute_observation_log_prob(self, states, observations):
        return torch.randn(states.shape[0], requires_grad=True)
    
    def parameters(self):
        return iter([self.dummy_param])


class MockRecognitionModel:
    """Mock recognition model for testing."""
    
    def __init__(self):
        # Create a dummy parameter for the optimizer
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
    
    def __call__(self, observations):
        return self.forward(observations)
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        return {
            'state_mean': torch.randn(batch_size, 32, requires_grad=True),
            'state_logvar': torch.randn(batch_size, 32, requires_grad=True)
        }
    
    def sample_states(self, observations):
        return torch.randn(observations.shape[0], 32, requires_grad=True)
    
    def compute_kl_divergence(self, recog_mean, recog_logvar, prior_mean, prior_logvar):
        return torch.randn(recog_mean.shape[0], requires_grad=True)
    
    def parameters(self):
        return iter([self.dummy_param])


class MockPolicyModel:
    """Mock policy model for testing."""
    
    def __init__(self):
        # Create a dummy parameter for the optimizer
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        self.action_dim = 4
    
    def compute_policy_efe(self, states, generative_model, recognition_model):
        return torch.randn(states.shape[0], 4, requires_grad=True)
    
    def update_policy(self, states, actions, rewards, next_states):
        return torch.tensor(0.1, requires_grad=True)  # Mock loss
    
    def parameters(self):
        return iter([self.dummy_param])


if __name__ == '__main__':
    unittest.main() 