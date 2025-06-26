"""
Generative model for active inference.

Implements the probabilistic generative model that represents how
observations are generated from hidden states and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import yaml


class GenerativeModel(nn.Module):
    """
    Generative model for active inference.
    
    This model represents p(o|s) and p(s'|s,a) where:
    - o: observations
    - s: hidden states
    - a: actions
    
    The model learns to predict observations from hidden states
    and state transitions from current states and actions.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize the generative model.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['agent']['active_inference']
        
        # Model dimensions
        self.state_dim = self.config['state_dim']
        self.observation_dim = self.config['observation_dim']
        self.action_dim = self.config['action_dim']
        
        # Get neural network configuration
        nn_config = yaml.safe_load(open(config_path, 'r'))['agent']['neural_networks']['generative']
        
        # Observation model: p(o|s)
        self.observation_model = self._build_observation_model(nn_config)
        
        # Transition model: p(s'|s,a)
        self.transition_model = self._build_transition_model(nn_config)
        
        # Prior model: p(s) (initial state distribution)
        self.prior_model = self._build_prior_model(nn_config)
        
        # Precision parameters
        self.observation_precision = nn.Parameter(torch.ones(1) * self.config['precision'])
        self.transition_precision = nn.Parameter(torch.ones(1) * self.config['precision'])
    
    def _build_observation_model(self, nn_config: Dict[str, Any]) -> nn.Module:
        """Build the observation model p(o|s)."""
        layers = []
        input_dim = self.state_dim
        
        # Hidden layers
        for hidden_dim in nn_config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if nn_config['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(nn_config['dropout'])
            ])
            input_dim = hidden_dim
        
        # Output layer (mean and log variance)
        layers.append(nn.Linear(input_dim, self.observation_dim * 2))
        
        return nn.Sequential(*layers)
    
    def _build_transition_model(self, nn_config: Dict[str, Any]) -> nn.Module:
        """Build the transition model p(s'|s,a)."""
        layers = []
        input_dim = self.state_dim + self.action_dim  # Concatenate state and action
        
        # Hidden layers
        for hidden_dim in nn_config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if nn_config['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(nn_config['dropout'])
            ])
            input_dim = hidden_dim
        
        # Output layer (mean and log variance)
        layers.append(nn.Linear(input_dim, self.state_dim * 2))
        
        return nn.Sequential(*layers)
    
    def _build_prior_model(self, nn_config: Dict[str, Any]) -> nn.Module:
        """Build the prior model p(s)."""
        layers = []
        input_dim = 1  # Simple prior, could be made more complex
        
        # Hidden layers
        for hidden_dim in nn_config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if nn_config['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(nn_config['dropout'])
            ])
            input_dim = hidden_dim
        
        # Output layer (mean and log variance)
        layers.append(nn.Linear(input_dim, self.state_dim * 2))
        
        return nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the generative model.
        
        Args:
            states: Hidden states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim] (optional)
            
        Returns:
            Dictionary containing:
            - observation_mean: Predicted observation means
            - observation_logvar: Predicted observation log variances
            - next_state_mean: Predicted next state means
            - next_state_logvar: Predicted next state log variances
            - prior_mean: Prior state means
            - prior_logvar: Prior state log variances
        """
        batch_size = states.shape[0]
        
        # Observation model: p(o|s)
        obs_output = self.observation_model(states)
        observation_mean = obs_output[:, :self.observation_dim]
        observation_logvar = obs_output[:, self.observation_dim:]
        
        # Transition model: p(s'|s,a)
        if actions is not None:
            # Convert 1D actions to one-hot if needed
            if actions.dim() == 1:
                actions = F.one_hot(actions, num_classes=self.action_dim).float()
            state_action = torch.cat([states, actions], dim=1)
            trans_output = self.transition_model(state_action)
            next_state_mean = trans_output[:, :self.state_dim]
            next_state_logvar = trans_output[:, self.state_dim:]
        else:
            next_state_mean = torch.zeros(batch_size, self.state_dim, device=states.device)
            next_state_logvar = torch.zeros(batch_size, self.state_dim, device=states.device)
        
        # Prior model: p(s)
        prior_input = torch.ones(batch_size, 1, device=states.device)
        prior_output = self.prior_model(prior_input)
        prior_mean = prior_output[:, :self.state_dim]
        prior_logvar = prior_output[:, self.state_dim:]
        
        return {
            'observation_mean': observation_mean,
            'observation_logvar': observation_logvar,
            'next_state_mean': next_state_mean,
            'next_state_logvar': next_state_logvar,
            'prior_mean': prior_mean,
            'prior_logvar': prior_logvar
        }
    
    def sample_observations(self, states: torch.Tensor) -> torch.Tensor:
        """
        Sample observations from the observation model.
        
        Args:
            states: Hidden states [batch_size, state_dim]
            
        Returns:
            Sampled observations [batch_size, observation_dim]
        """
        outputs = self.forward(states)
        mean = outputs['observation_mean']
        logvar = outputs['observation_logvar']
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        observations = mean + eps * std
        
        return observations
    
    def sample_next_states(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Sample next states from the transition model.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            
        Returns:
            Sampled next states [batch_size, state_dim]
        """
        outputs = self.forward(states, actions)
        mean = outputs['next_state_mean']
        logvar = outputs['next_state_logvar']
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        next_states = mean + eps * std
        
        return next_states
    
    def compute_observation_log_prob(self, states: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of observations given states.
        
        Args:
            states: Hidden states [batch_size, state_dim]
            observations: Observations [batch_size, observation_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = self.forward(states)
        mean = outputs['observation_mean']
        logvar = outputs['observation_logvar']
        
        # Compute log probability under Gaussian assumption
        precision = torch.exp(self.observation_precision)
        log_prob = -0.5 * (
            precision * torch.sum((observations - mean) ** 2, dim=1) +
            torch.sum(logvar, dim=1) +
            self.observation_dim * torch.log(2 * torch.pi / precision)
        )
        
        return log_prob
    
    def compute_transition_log_prob(self, states: torch.Tensor, actions: torch.Tensor, 
                                  next_states: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of next states given current states and actions.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = self.forward(states, actions)
        mean = outputs['next_state_mean']
        logvar = outputs['next_state_logvar']
        
        # Compute log probability under Gaussian assumption
        precision = torch.exp(self.transition_precision)
        log_prob = -0.5 * (
            precision * torch.sum((next_states - mean) ** 2, dim=1) +
            torch.sum(logvar, dim=1) +
            self.state_dim * torch.log(2 * torch.pi / precision)
        )
        
        return log_prob
    
    def compute_prior_log_prob(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of states under the prior.
        
        Args:
            states: States [batch_size, state_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = self.forward(states)
        mean = outputs['prior_mean']
        logvar = outputs['prior_logvar']
        
        # Compute log probability under Gaussian assumption
        log_prob = -0.5 * (
            torch.sum((states - mean) ** 2, dim=1) +
            torch.sum(logvar, dim=1) +
            self.state_dim * torch.log(torch.tensor(2 * torch.pi))
        )
        
        return log_prob
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all model parameters for optimization."""
        return {
            'observation_precision': self.observation_precision,
            'transition_precision': self.transition_precision,
            **{name: param for name, param in self.named_parameters() 
               if 'precision' not in name}
        } 