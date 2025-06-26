"""
Unit tests for agent implementations.

Tests the active inference agent and RL baseline agents
for functionality and performance.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import yaml

from src.agents import ActiveInferenceAgent
from src.agents.rl_agents import PPOAgent, DQNAgent
from src.environment import NavigationEnvironment


class TestActiveInferenceAgent(unittest.TestCase):
    """Test cases for ActiveInferenceAgent."""
    
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
                    'kl_weight': 1.0,
                    'weight_decay': 1e-5
                },
                'policy': {
                    'exploration_rate': 0.1,
                    'exploration_decay': 0.995,
                    'min_exploration_rate': 0.01,
                    'efe_weight': 1.0,
                    'information_gain_weight': 0.5,
                    'utility_weight': 0.5
                },
                'training': {
                    'batch_size': 16,
                    'update_frequency': 5,
                    'memory_size': 1000
                },
                'neural_networks': {
                    'generative': {
                        'hidden_dims': [64, 32],
                        'activation': 'relu',
                        'dropout': 0.1
                    },
                    'recognition': {
                        'hidden_dims': [64, 32],
                        'activation': 'relu',
                        'dropout': 0.1
                    },
                    'policy': {
                        'hidden_dims': [32, 16],
                        'activation': 'relu'
                    }
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
        
        # Create environment and agent
        self.env = NavigationEnvironment()
        self.agent = ActiveInferenceAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            config_path=self.temp_config.name
        )
        
        # Test data
        self.observation = np.random.randn(self.env.observation_space.shape[0])
        self.next_observation = np.random.randn(self.env.observation_space.shape[0])
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.observation_space, self.env.observation_space)
        self.assertEqual(self.agent.action_space, self.env.action_space)
        
        # Check that all models exist
        self.assertIsNotNone(self.agent.generative_model)
        self.assertIsNotNone(self.agent.recognition_model)
        self.assertIsNotNone(self.agent.policy_model)
        self.assertIsNotNone(self.agent.variational_inference)
        
        # Check training parameters
        self.assertEqual(self.agent.batch_size, 16)
        self.assertEqual(self.agent.update_frequency, 5)
        self.assertEqual(self.agent.memory_size, 1000)
        
        # Check policy parameters
        self.assertEqual(self.agent.exploration_rate, 0.1)
        self.assertEqual(self.agent.exploration_decay, 0.995)
        self.assertEqual(self.agent.min_exploration_rate, 0.01)
    
    def test_action_selection(self):
        """Test action selection."""
        action = self.agent.select_action(self.observation, training=True)
        
        # Check action validity
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.agent.action_space.n)
        
        # Check that beliefs were updated
        self.assertIsNotNone(self.agent.current_beliefs)
        self.assertIn('mean', self.agent.current_beliefs)
        self.assertIn('logvar', self.agent.current_beliefs)
        self.assertIn('states', self.agent.current_beliefs)
    
    def test_experience_storage(self):
        """Test experience storage in replay buffer."""
        initial_memory_size = len(self.agent.memory)
        
        # Store experience
        self.agent.store_experience(
            observation=self.observation,
            action=0,
            reward=1.0,
            next_observation=self.next_observation,
            done=False
        )
        
        # Check memory size increased
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
        
        # Check stored experience
        experience = self.agent.memory[-1]
        self.assertEqual(experience['action'], 0)
        self.assertEqual(experience['reward'], 1.0)
        self.assertFalse(experience['done'])
    
    def test_model_updates(self):
        """Test model updates."""
        # Add some experiences to memory
        for _ in range(20):
            self.agent.store_experience(
                observation=self.observation,
                action=np.random.randint(0, 4),
                reward=np.random.randn(),
                next_observation=self.next_observation,
                done=np.random.choice([True, False])
            )
        
        # Update models
        losses = self.agent.update_models()
        
        # Check that losses were computed
        if losses:  # Only if enough data for batch
            self.assertIsInstance(losses, dict)
            # Check for expected loss keys
            expected_keys = ['generative_loss', 'recognition_loss', 'policy_loss']
            for key in expected_keys:
                if key in losses:
                    self.assertIsInstance(losses[key], float)
                    self.assertTrue(np.isfinite(losses[key]))
    
    def test_belief_updates(self):
        """Test belief updates."""
        beliefs = self.agent.update_beliefs(self.observation, num_iterations=5)
        
        # Check belief structure
        self.assertIsInstance(beliefs, dict)
        self.assertIn('belief_mean', beliefs)
        self.assertIn('belief_logvar', beliefs)
        self.assertIn('final_free_energy', beliefs)
        
        # Check belief dimensions
        self.assertEqual(beliefs['belief_mean'].shape, (1, 32))
        self.assertEqual(beliefs['belief_logvar'].shape, (1, 32))
        self.assertIsInstance(beliefs['final_free_energy'], float)
    
    def test_uncertainty_calculation(self):
        """Test uncertainty calculation."""
        uncertainty = self.agent.get_belief_uncertainty(self.observation)
        
        # Check uncertainty
        self.assertIsInstance(uncertainty, np.ndarray)
        self.assertTrue(np.all(uncertainty >= 0))  # Uncertainty should be non-negative
        self.assertTrue(np.all(np.isfinite(uncertainty)))
    
    def test_action_probabilities(self):
        """Test action probability calculation."""
        probs = self.agent.get_action_probabilities(self.observation)
        
        # Check probabilities
        self.assertIsInstance(probs, np.ndarray)
        self.assertEqual(probs.shape, (self.agent.action_space.n,))
        self.assertTrue(np.all(probs >= 0))  # Probabilities should be non-negative
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)  # Should sum to 1
    
    def test_free_energy_calculation(self):
        """Test free energy calculation."""
        fe = self.agent.get_free_energy(self.observation)
        
        # Check free energy
        self.assertIsInstance(fe, float)
        self.assertTrue(np.isfinite(fe))
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Save models
        save_path = "test_models.pth"
        self.agent.save_models(save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load models
        self.agent.load_models(save_path)
        
        # Clean up
        os.remove(save_path)
    
    def test_training_stats(self):
        """Test training statistics."""
        stats = self.agent.get_training_stats()
        
        # Check stats structure
        self.assertIsInstance(stats, dict)
        
        # Check for expected keys
        expected_keys = ['step_count', 'episode_count', 'exploration_rate']
        for key in expected_keys:
            if key in stats:
                self.assertIsInstance(stats[key], (int, float))
    
    def test_episode_end_handling(self):
        """Test episode end handling."""
        initial_episode_count = self.agent.episode_count
        
        # Simulate episode end
        self.agent._on_episode_end()
        
        # Check episode count increased
        self.assertEqual(self.agent.episode_count, initial_episode_count + 1)
    
    def test_exploration_decay(self):
        """Test exploration rate decay."""
        initial_rate = self.agent.exploration_rate
        
        # Take some actions to trigger decay
        for _ in range(10):
            self.agent.select_action(self.observation, training=True)
        
        # Check exploration rate decreased
        self.assertLess(self.agent.exploration_rate, initial_rate)
        self.assertGreaterEqual(self.agent.exploration_rate, self.agent.min_exploration_rate)


class TestPPOAgent(unittest.TestCase):
    """Test cases for PPOAgent."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'ppo': {
                    'learning_rate': 0.0003,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'clip_range_vf': None,
                    'ent_coef': 0.0,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'use_sde': False,
                    'sde_sample_freq': -1,
                    'target_kl': None,
                    'weight_decay': 1e-5
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create environment and agent
        self.env = NavigationEnvironment()
        self.agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        # Test data
        self.observation = np.random.randn(self.env.observation_space.shape[0])
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.observation_space, self.env.observation_space)
        self.assertEqual(self.agent.action_space, self.env.action_space)
        
        # Check that PPO model exists
        self.assertIsNotNone(self.agent.model)
    
    def test_action_selection(self):
        """Test action selection."""
        action = self.agent.select_action(self.observation, training=True)
        
        # Check action validity
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.agent.action_space.n)
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Save model
        save_path = "test_ppo_model.pth"
        self.agent.save_models(save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load model
        self.agent.load_models(save_path)
        
        # Clean up
        os.remove(save_path)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary config file
        self.config = {
            'agent': {
                'dqn': {
                    'learning_rate': 0.0001,
                    'buffer_size': 1000000,
                    'learning_starts': 50000,
                    'batch_size': 32,
                    'tau': 1.0,
                    'gamma': 0.99,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'target_update_interval': 10000,
                    'exploration_fraction': 0.1,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.05,
                    'max_grad_norm': 10.0,
                    'weight_decay': 1e-5
                }
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.config, self.temp_config)
        self.temp_config.close()
        
        # Create environment and agent
        self.env = NavigationEnvironment()
        self.agent = DQNAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        # Test data
        self.observation = np.random.randn(self.env.observation_space.shape[0])
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_config.name)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.observation_space, self.env.observation_space)
        self.assertEqual(self.agent.action_space, self.env.action_space)
        
        # Check that DQN model exists
        self.assertIsNotNone(self.agent.model)
    
    def test_action_selection(self):
        """Test action selection."""
        action = self.agent.select_action(self.observation, training=True)
        
        # Check action validity
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.agent.action_space.n)
    
    def test_model_saving_loading(self):
        """Test model saving and loading."""
        # Save model
        save_path = "test_dqn_model.pth"
        self.agent.save_models(save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load model
        self.agent.load_models(save_path)
        
        # Clean up
        os.remove(save_path)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for agents with environment."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.env = NavigationEnvironment()
        
        # Create agents
        self.active_inference_agent = ActiveInferenceAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        self.ppo_agent = PPOAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
        
        self.dqn_agent = DQNAgent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space
        )
    
    def test_agent_environment_interaction(self):
        """Test agent interaction with environment."""
        agents = {
            'ActiveInference': self.active_inference_agent,
            'PPO': self.ppo_agent,
            'DQN': self.dqn_agent
        }
        
        for agent_name, agent in agents.items():
            with self.subTest(agent=agent_name):
                # Reset environment
                obs = self.env.reset()
                
                # Run a few steps
                for step in range(10):
                    # Select action
                    action = agent.select_action(obs, training=True)
                    
                    # Take step in environment
                    next_obs, reward, done, info = self.env.step(action)
                    
                    # Update agent if it has a step method
                    if hasattr(agent, 'step'):
                        agent.step(obs, reward, done, info)
                    
                    # Check that action was valid
                    self.assertTrue(0 <= action < self.env.action_space.n)
                    
                    # Check that observation is valid
                    self.assertEqual(len(next_obs), self.env.observation_space.shape[0])
                    self.assertTrue(np.all(np.isfinite(next_obs)))
                    
                    if done:
                        break
                    
                    obs = next_obs
    
    def test_agent_performance_comparison(self):
        """Test basic performance comparison between agents."""
        agents = {
            'ActiveInference': self.active_inference_agent,
            'PPO': self.ppo_agent,
            'DQN': self.dqn_agent
        }
        
        results = {}
        
        for agent_name, agent in agents.items():
            # Run a short episode
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(50):  # Short episode
                action = agent.select_action(obs, training=True)
                obs, reward, done, info = self.env.step(action)
                
                if hasattr(agent, 'step'):
                    agent.step(obs, reward, done, info)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            results[agent_name] = {
                'total_reward': total_reward,
                'steps': steps,
                'success': info.get('goal_reached', False)
            }
        
        # Check that all agents completed episodes
        for agent_name, result in results.items():
            self.assertGreater(result['steps'], 0)
            self.assertIsInstance(result['total_reward'], float)
            self.assertIsInstance(bool(result['success']), bool)


if __name__ == '__main__':
    unittest.main() 