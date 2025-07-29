"""
Active inference agent for adaptive robot navigation.

Integrates generative model, recognition model, policy model, and
variational inference to create a complete active inference agent.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import yaml
from collections import deque
import random

from ..active_inference import (
    GenerativeModel, RecognitionModel, PolicyModel, VariationalInference
)


class ActiveInferenceAgent:
    """
    Active inference agent for adaptive robot navigation.
    
    This agent implements the complete active inference framework:
    1. Maintains a generative model of the world
    2. Uses recognition model to infer hidden states
    3. Selects actions to minimize expected free energy
    4. Continuously updates beliefs through variational inference
    """
    
    def __init__(self, observation_space: int, action_space: int, 
                 config_path: str = "config/agent_config.yaml"):
        """
        Initialize the active inference agent.
        
        Args:
            observation_space: Dimension of observation space
            action_space: Number of possible actions
            config_path: Path to configuration file
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['agent']
        
        # Initialize models
        self.generative_model = GenerativeModel(config_path)
        self.recognition_model = RecognitionModel(config_path)
        self.policy_model = PolicyModel(config_path)
        self.variational_inference = VariationalInference(config_path)
        
        # Training parameters
        self.training_config = self.config['training']
        self.batch_size = self.training_config['batch_size']
        self.update_frequency = self.training_config['update_frequency']
        self.memory_size = self.training_config['memory_size']
        
        # Policy parameters
        self.policy_config = self.config['policy']
        self.exploration_rate = self.policy_config['exploration_rate']
        self.exploration_decay = self.policy_config['exploration_decay']
        self.min_exploration_rate = self.policy_config['min_exploration_rate']
        
        # Experience replay buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_losses = []
        
        # Belief state tracking
        self.current_beliefs = None
        self.belief_history = []
        self.free_energy_history = []
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._move_models_to_device()
    
    def _move_models_to_device(self):
        """Move all models to the specified device."""
        self.generative_model.to(self.device)
        self.recognition_model.to(self.device)
        self.policy_model.to(self.device)
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select an action based on current observation.
        
        Args:
            observation: Current observation
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Infer hidden states using recognition model
        with torch.no_grad():
            belief_outputs = self.recognition_model(obs_tensor)
            current_states = self.recognition_model.sample_states(obs_tensor)
        
        # Store current beliefs
        self.current_beliefs = {
            'mean': belief_outputs['state_mean'].cpu().numpy(),
            'logvar': belief_outputs['state_logvar'].cpu().numpy(),
            'states': current_states.cpu().numpy()
        }
        
        # Select action based on policy
        if training and random.random() < self.exploration_rate:
            # Exploration: random action
            action = random.randint(0, self.action_space.n - 1)
        else:
            # Exploitation: select action by minimizing EFE
            action = self._select_action_by_efe(current_states)
        
        # Update exploration rate
        if training:
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay
            )
        
        return action
    
    def _select_action_by_efe(self, states: torch.Tensor) -> int:
        """
        Select action by minimizing Expected Free Energy.
        
        Args:
            states: Current hidden states
            
        Returns:
            Selected action
        """
        # Compute EFE for all actions
        efe_matrix = self.policy_model.compute_policy_efe(
            states, self.generative_model, self.recognition_model
        )
        
        # Select action with minimum EFE
        action = torch.argmin(efe_matrix, dim=1).item()
        
        return action
    
    def store_experience(self, observation: np.ndarray, action: int, 
                        reward: float, next_observation: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        """
        # Validate experience - prevent None values from being stored
        if (observation is None or next_observation is None or
            not isinstance(action, int) or
            not isinstance(reward, (float, int)) or
            not isinstance(done, (bool, np.bool_))):
            print(f"[WARNING] Invalid experience not stored: obs={observation}, action={action}, reward={reward}, next_obs={next_observation}, done={done}, step={getattr(self, 'step_count', 'N/A')}, episode={getattr(self, 'episode_count', 'N/A')}")
            return
        
        # Ensure observations are numpy arrays
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        if not isinstance(next_observation, np.ndarray):
            next_observation = np.array(next_observation)
            
        self.memory.append({
            'observation': observation,
            'action': action,
            'reward': float(reward),
            'next_observation': next_observation,
            'done': bool(done)
        })
        
        # Memory size tracking (removed debug print)
    
    def update_models(self) -> Dict[str, float]:
        """
        Update all models using stored experiences.
        
        Returns:
            Dictionary containing training losses
        """
        if len(self.memory) < self.batch_size:
            return {}
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            if not isinstance(batch, list):
                batch = [batch]
            if not batch or not isinstance(batch, list) or not isinstance(batch[0], dict):
                print(f"[DEBUG] Malformed batch in update_models: {batch}")
                return {}
            
            # Optimize tensor creation by converting to numpy arrays first
            obs_list = [exp['observation'] for exp in batch]
            next_obs_list = [exp['next_observation'] for exp in batch]
            actions_list = [exp['action'] for exp in batch]
            rewards_list = [exp['reward'] for exp in batch]
            
            # Convert to numpy arrays first, then to tensors
            observations = torch.FloatTensor(np.array(obs_list)).to(self.device)
            next_observations = torch.FloatTensor(np.array(next_obs_list)).to(self.device)
            actions = torch.LongTensor(np.array(actions_list)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards_list)).to(self.device)
            
        except Exception as e:
            print(f"[ERROR] Exception in update_models batch processing: {e}\nBatch: {batch if 'batch' in locals() else 'N/A'}")
            return {}
        
        # Infer states using recognition model
        with torch.no_grad():
            current_states = self.recognition_model.sample_states(observations)
            next_states = self.recognition_model.sample_states(next_observations)
        
        # Update models using variational inference
        losses = self.variational_inference.optimize_models(
            observations, current_states, actions, rewards,
            self.generative_model, self.recognition_model, self.policy_model
        )
        
        # Store losses
        self.training_losses.append(losses)
        return losses
    
    def update_beliefs(self, observation: np.ndarray, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Update beliefs about hidden states.
        
        Args:
            observation: Current observation
            num_iterations: Number of belief update iterations
            
        Returns:
            Dictionary containing updated beliefs and metrics
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Update beliefs using variational inference
        belief_update = self.variational_inference.update_beliefs(
            obs_tensor, self.generative_model, self.recognition_model, num_iterations
        )
        
        # Store belief history
        self.belief_history.append({
            'mean': belief_update['belief_mean'].cpu().numpy(),
            'logvar': belief_update['belief_logvar'].cpu().numpy(),
            'free_energy': belief_update['final_free_energy']
        })
        
        # Store free energy history
        self.free_energy_history.extend(belief_update['free_energy_history'])
        
        return belief_update
    
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
        # Store previous experience if available and valid
        if (hasattr(self, 'last_observation') and hasattr(self, 'last_action') and
            self.last_observation is not None and self.last_action is not None):
            self.store_experience(
                self.last_observation, self.last_action, reward, observation, done
            )
        elif hasattr(self, 'last_observation') and hasattr(self, 'last_action'):
            # Debug: Track when we skip storing due to None values
            if self.step_count % 1000 == 0:  # Changed from 100 to 1000
                print(f"[DEBUG] Skipping experience storage: last_obs={self.last_observation is None}, last_action={self.last_action is None}")
        
        # Select action
        action = self.select_action(observation, training=True)
        
        # Update models periodically
        if self.step_count % self.update_frequency == 0:
            self.update_models()
        
        # Update beliefs
        belief_update = self.update_beliefs(observation)
        
        # Store current state
        self.last_observation = observation.copy()
        self.last_action = action
        self.step_count += 1
        
        # Handle episode end
        if done:
            self._on_episode_end()
        
        return action
    
    def _on_episode_end(self):
        """Handle episode end."""
        # Increment episode count
        self.episode_count += 1
        
        # Reset episode-specific variables
        self.last_observation = None
        self.last_action = None
        
        # Update exploration rate more gradually
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
    
    def get_belief_uncertainty(self, observation: np.ndarray) -> np.ndarray:
        """
        Get uncertainty in beliefs about hidden states.
        
        Args:
            observation: Current observation
            
        Returns:
            Uncertainty estimates
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            uncertainty = self.variational_inference.compute_belief_uncertainty(
                obs_tensor, self.recognition_model
            )
        
        return uncertainty.cpu().numpy().squeeze()
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities based on current beliefs.
        
        Args:
            observation: Current observation
            
        Returns:
            Action probabilities
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_states = self.recognition_model.sample_states(obs_tensor)
            action_probs = self.policy_model.compute_action_probabilities(
                current_states, self.generative_model, self.recognition_model
            )
        
        return action_probs.cpu().numpy().squeeze()
    
    def get_free_energy(self, observation: np.ndarray) -> float:
        """
        Get current variational free energy.
        
        Args:
            observation: Current observation
            
        Returns:
            Variational free energy
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            current_states = self.recognition_model.sample_states(obs_tensor)
            free_energy = self.variational_inference.compute_variational_free_energy(
                obs_tensor, current_states, self.generative_model, self.recognition_model
            )
        
        return free_energy.cpu().numpy().item()
    
    def save_models(self, path: str):
        """Save all models to disk."""
        torch.save({
            'generative_model': self.generative_model.state_dict(),
            'recognition_model': self.recognition_model.state_dict(),
            'policy_model': self.policy_model.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'training_losses': self.training_losses
        }, path)
    
    def load_models(self, path: str):
        """Load all models from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generative_model.load_state_dict(checkpoint['generative_model'])
        self.recognition_model.load_state_dict(checkpoint['recognition_model'])
        self.policy_model.load_state_dict(checkpoint['policy_model'])
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.exploration_rate = checkpoint['exploration_rate']
        self.training_losses = checkpoint['training_losses']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_losses:
            return {}
        
        recent_losses = self.training_losses[-100:]  # Last 100 updates
        
        stats = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'exploration_rate': self.exploration_rate,
            'memory_size': len(self.memory),
            'avg_generative_loss': np.mean([l['generative_loss'] for l in recent_losses]),
            'avg_recognition_loss': np.mean([l['recognition_loss'] for l in recent_losses]),
            'avg_policy_loss': np.mean([l['policy_loss'] for l in recent_losses])
        }
        
        if self.free_energy_history:
            stats['avg_free_energy'] = np.mean(self.free_energy_history[-100:])
        
        return stats 