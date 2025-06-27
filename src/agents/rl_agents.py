"""
Reinforcement learning baseline agents for comparison.

Implements PPO and DQN agents to compare against active inference
performance in navigation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gym


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for comparison.
    
    Uses stable-baselines3 implementation of PPO for navigation tasks.
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 learning_rate: float = 3e-4, n_steps: int = 2048):
        """
        Initialize PPO agent.
        
        Args:
            observation_space: Dimension of observation space
            action_space: Number of possible actions
            learning_rate: Learning rate for PPO
            n_steps: Number of steps per update
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create a dummy environment for stable-baselines3
        self.dummy_env = self._create_dummy_env()
        
        # Initialize PPO agent
        self.agent = PPO(
            "MlpPolicy",
            self.dummy_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_rewards = []
        
    def _build_network(self):
        """Build policy network."""
        if isinstance(self.observation_space, gym.spaces.Box):
            input_dim = self.observation_space.shape[0]
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            input_dim = 1
        else:
            raise NotImplementedError("Unsupported observation space type")
        if isinstance(self.action_space, gym.spaces.Discrete):
            output_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            output_dim = self.action_space.shape[0]
        else:
            raise NotImplementedError("Unsupported action space type")
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def _create_dummy_env(self):
        """Create a dummy environment for stable-baselines3."""
        obs_space = self.observation_space
        act_space = self.action_space
        class DummyNavigationEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self):
                if isinstance(self.observation_space, gym.spaces.Box):
                    return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                elif isinstance(self.observation_space, gym.spaces.Discrete):
                    return 0
                else:
                    raise NotImplementedError("Unsupported observation space type")
            def step(self, action):
                if isinstance(self.observation_space, gym.spaces.Box):
                    obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                elif isinstance(self.observation_space, gym.spaces.Discrete):
                    obs = 0
                else:
                    raise NotImplementedError("Unsupported observation space type")
                return obs, 0.0, False, {}
        return DummyVecEnv([lambda: DummyNavigationEnv()])
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action based on current observation.
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Use PPO to select action
        action, _ = self.agent.predict(observation, deterministic=not training)
        return int(action)
    
    def store_experience(self, observation: np.ndarray, action: int, 
                        reward: float, next_observation: np.ndarray, done: bool):
        """
        Store experience (not used for PPO as it learns online).
        
        Args:
            observation: Current observation
            action: Taken action
            reward: Received reward
            next_observation: Next observation
            done: Whether episode is done
        """
        # PPO learns online, so we don't need to store experiences
        pass
    
    def update_models(self) -> Dict[str, float]:
        """
        Update models (PPO updates automatically).
        
        Returns:
            Empty dictionary (no manual updates needed)
        """
        return {}
    
    def step(self, observation: np.ndarray, reward: float, done: bool, 
             info: Dict[str, Any]) -> int:
        """
        Take a step in the environment.
        
        Args:
            observation: Current observation
            reward: Received reward
            done: Whether episode is done
            info: Additional information
            
        Returns:
            Selected action
        """
        # Select action
        action = self.select_action(observation, training=True)
        
        # PPO learns online, so we don't need to manually update
        self.step_count += 1
        
        # Handle episode end
        if done:
            self.episode_count += 1
            self.training_rewards.append(reward)
        
        return action
    
    def save_models(self, path: str):
        """Save PPO model to disk."""
        self.agent.save(path)
    
    def load_models(self, path: str):
        """Load PPO model from disk."""
        self.agent = PPO.load(path, env=self.dummy_env)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        if self.training_rewards:
            stats['avg_reward'] = np.mean(self.training_rewards[-100:])
        
        return stats

    @property
    def model(self):
        return self.agent

    def save_model(self, path):
        self.save_models(path)


class DQNAgent:
    """
    Deep Q-Network (DQN) agent for comparison.
    
    Uses stable-baselines3 implementation of DQN for navigation tasks.
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 learning_rate: float = 1e-4, buffer_size: int = 100000):
        """
        Initialize DQN agent.
        
        Args:
            observation_space: Dimension of observation space
            action_space: Number of possible actions
            learning_rate: Learning rate for DQN
            buffer_size: Size of replay buffer
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create a dummy environment for stable-baselines3
        self.dummy_env = self._create_dummy_env()
        
        # Initialize DQN agent
        self.agent = DQN(
            "MlpPolicy",
            self.dummy_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0
        )
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_rewards = []
        
    def _build_network(self):
        """Build policy network."""
        if isinstance(self.observation_space, gym.spaces.Box):
            input_dim = self.observation_space.shape[0]
        elif isinstance(self.observation_space, gym.spaces.Discrete):
            input_dim = 1
        else:
            raise NotImplementedError("Unsupported observation space type")
        if isinstance(self.action_space, gym.spaces.Discrete):
            output_dim = self.action_space.n
        elif isinstance(self.action_space, gym.spaces.Box):
            output_dim = self.action_space.shape[0]
        else:
            raise NotImplementedError("Unsupported action space type")
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def _create_dummy_env(self):
        """Create a dummy environment for stable-baselines3."""
        obs_space = self.observation_space
        act_space = self.action_space
        class DummyNavigationEnv(gym.Env):
            def __init__(self):
                self.observation_space = obs_space
                self.action_space = act_space
            def reset(self):
                if isinstance(self.observation_space, gym.spaces.Box):
                    return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                elif isinstance(self.observation_space, gym.spaces.Discrete):
                    return 0
                else:
                    raise NotImplementedError("Unsupported observation space type")
            def step(self, action):
                if isinstance(self.observation_space, gym.spaces.Box):
                    obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                elif isinstance(self.observation_space, gym.spaces.Discrete):
                    obs = 0
                else:
                    raise NotImplementedError("Unsupported observation space type")
                return obs, 0.0, False, {}
        return DummyVecEnv([lambda: DummyNavigationEnv()])
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action based on current observation.
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Use DQN to select action
        action, _ = self.agent.predict(observation, deterministic=not training)
        return int(action)
    
    def store_experience(self, observation: np.ndarray, action: int, 
                        reward: float, next_observation: np.ndarray, done: bool):
        """
        Store experience (DQN uses replay buffer internally).
        
        Args:
            observation: Current observation
            action: Taken action
            reward: Received reward
            next_observation: Next observation
            done: Whether episode is done
        """
        # DQN handles experience storage internally
        pass
    
    def update_models(self) -> Dict[str, float]:
        """
        Update models (DQN updates automatically).
        
        Returns:
            Empty dictionary (no manual updates needed)
        """
        return {}
    
    def step(self, observation: np.ndarray, reward: float, done: bool, 
             info: Dict[str, Any]) -> int:
        """
        Take a step in the environment.
        
        Args:
            observation: Current observation
            reward: Received reward
            done: Whether episode is done
            info: Additional information
            
        Returns:
            Selected action
        """
        # Select action
        action = self.select_action(observation, training=True)
        
        # DQN learns online, so we don't need to manually update
        self.step_count += 1
        
        # Handle episode end
        if done:
            self.episode_count += 1
            self.training_rewards.append(reward)
        
        return action
    
    def save_models(self, path: str):
        """Save DQN model to disk."""
        self.agent.save(path)
    
    def load_models(self, path: str):
        """Load DQN model from disk."""
        self.agent = DQN.load(path, env=self.dummy_env)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count
        }
        
        if self.training_rewards:
            stats['avg_reward'] = np.mean(self.training_rewards[-100:])
        
        return stats

    @property
    def model(self):
        return self.agent

    def save_model(self, path):
        self.save_models(path)


class CustomDQNAgent:
    """
    Custom DQN implementation for more detailed control and comparison.
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 learning_rate: float = 1e-4, buffer_size: int = 100000):
        """
        Initialize custom DQN agent.
        
        Args:
            observation_space: Dimension of observation space
            action_space: Number of possible actions
            learning_rate: Learning rate for DQN
            buffer_size: Size of replay buffer
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Neural network
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 1000
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network.to(self.device)
        self.target_network.to(self.device)
    
    def _build_q_network(self) -> nn.Module:
        """Build Q-network."""
        return nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space.n)
        )
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_space.n - 1)
        else:
            # Exploitation: greedy action
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()
            return action
    
    def store_experience(self, observation: np.ndarray, action: int, 
                        reward: float, next_observation: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        """
        # Validate experience
        if (observation is None or next_observation is None or
            not isinstance(action, int) or
            not isinstance(reward, (float, int)) or
            not isinstance(done, (bool, np.bool_))):
            print(f"[WARNING] Invalid experience not stored: obs={observation}, action={action}, reward={reward}, next_obs={next_observation}, done={done}")
            return
        self.memory.append({
            'observation': observation,
            'action': action,
            'reward': float(reward),
            'next_observation': next_observation,
            'done': bool(done)
        })
    
    def update_models(self) -> Dict[str, float]:
        """
        Update Q-network using experience replay.
        
        Returns:
            Dictionary containing training loss
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            if not isinstance(batch, list):
                batch = [batch]
            if not batch or not isinstance(batch[0], dict):
                print(f"[DEBUG] Malformed batch in DQN update_models: {batch}")
                return {}
            
            # Prepare batch data
            observations = torch.FloatTensor([exp['observation'] for exp in batch]).to(self.device)
            actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
            next_observations = torch.FloatTensor([exp['next_observation'] for exp in batch]).to(self.device)
            dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
            
            # Compute current Q-values
            current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1))
            
            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_observations).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Store loss
            self.training_losses.append(loss.item())
            
            return {'dqn_loss': loss.item()}
            
        except Exception as e:
            print(f"[ERROR] Exception in DQN update_models batch processing: {e}")
            print(f"Batch: {batch}")
            return {}
    
    def step(self, observation: np.ndarray, reward: float, done: bool, 
             info: Dict[str, Any]) -> int:
        """
        Take a step in the environment.
        
        Args:
            observation: Current observation
            reward: Received reward
            done: Whether episode is done
            info: Additional information
            
        Returns:
            Selected action
        """
        # Store previous experience if available
        if hasattr(self, 'last_observation') and hasattr(self, 'last_action'):
            self.store_experience(
                self.last_observation, self.last_action, reward, observation, done
            )
        
        # Select action
        action = self.select_action(observation, training=True)
        
        # Update models
        self.update_models()
        
        # Store current state
        self.last_observation = observation.copy()
        self.last_action = action
        self.step_count += 1
        
        # Handle episode end
        if done:
            self.episode_count += 1
            self.last_observation = None
            self.last_action = None
        
        return action
    
    def save_models(self, path: str):
        """Save DQN model to disk."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'training_losses': self.training_losses
        }, path)
    
    def load_models(self, path: str):
        """Load DQN model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint['training_losses']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        
        if self.training_losses:
            stats['avg_loss'] = np.mean(self.training_losses[-100:])
        
        return stats 