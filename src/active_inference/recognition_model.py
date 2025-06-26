"""
Recognition model for active inference.

Implements the approximate posterior q(s|o) that infers hidden states
from observations using variational inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
import yaml


class RecognitionModel(nn.Module):
    """
    Recognition model for active inference.
    
    This model represents the approximate posterior q(s|o) where:
    - s: hidden states
    - o: observations
    
    The model learns to infer hidden states from observations using
    variational inference techniques.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize the recognition model.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['agent']['active_inference']
        
        # Model dimensions
        self.state_dim = self.config['state_dim']
        self.observation_dim = self.config['observation_dim']
        
        # Get neural network configuration
        nn_config = yaml.safe_load(open(config_path, 'r'))['agent']['neural_networks']['recognition']
        
        # Recognition network: q(s|o)
        self.recognition_network = self._build_recognition_network(nn_config)
        
        # Precision parameter for the recognition model
        self.recognition_precision = nn.Parameter(torch.ones(1) * self.config['precision'])
    
    def _build_recognition_network(self, nn_config: Dict[str, Any]) -> nn.Module:
        """Build the recognition network q(s|o)."""
        layers = []
        input_dim = self.observation_dim
        
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
    
    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the recognition model.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            
        Returns:
            Dictionary containing:
            - state_mean: Inferred state means
            - state_logvar: Inferred state log variances
        """
        outputs = self.recognition_network(observations)
        state_mean = outputs[:, :self.state_dim]
        state_logvar = outputs[:, self.state_dim:]
        
        return {
            'state_mean': state_mean,
            'state_logvar': state_logvar
        }
    
    def sample_states(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden states from the recognition model.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            
        Returns:
            Sampled hidden states [batch_size, state_dim]
        """
        outputs = self.forward(observations)
        mean = outputs['state_mean']
        logvar = outputs['state_logvar']
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        states = mean + eps * std
        
        return states
    
    def compute_log_prob(self, observations: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of states given observations.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            states: Hidden states [batch_size, state_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        outputs = self.forward(observations)
        mean = outputs['state_mean']
        logvar = outputs['state_logvar']
        
        # Compute log probability under Gaussian assumption
        precision = torch.exp(self.recognition_precision)
        log_prob = -0.5 * (
            precision * torch.sum((states - mean) ** 2, dim=1) +
            torch.sum(logvar, dim=1) +
            self.state_dim * torch.log(2 * torch.pi / precision)
        )
        
        return log_prob
    
    def compute_kl_divergence(self, recog_mean: torch.Tensor, recog_logvar: torch.Tensor,
                            prior_mean: torch.Tensor, prior_logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between recognition model and prior.
        
        Args:
            recog_mean: Recognition model state means [batch_size, state_dim]
            recog_logvar: Recognition model state log variances [batch_size, state_dim]
            prior_mean: Prior state means [batch_size, state_dim]
            prior_logvar: Prior state log variances [batch_size, state_dim]
            
        Returns:
            KL divergence [batch_size]
        """
        # KL divergence between two Gaussians
        kl_div = 0.5 * torch.sum(
            prior_logvar - recog_logvar + 
            (torch.exp(recog_logvar) + (recog_mean - prior_mean) ** 2) / torch.exp(prior_logvar) - 1,
            dim=1
        )
        
        return kl_div
    
    def update_beliefs(self, observations: torch.Tensor, learning_rate: float = None) -> torch.Tensor:
        """
        Update beliefs about hidden states given observations.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            learning_rate: Learning rate for belief updates
            
        Returns:
            Updated state means [batch_size, state_dim]
        """
        if learning_rate is None:
            learning_rate = self.config['learning_rate']
        
        outputs = self.forward(observations)
        state_mean = outputs['state_mean']
        
        # Simple belief update (could be made more sophisticated)
        return state_mean
    
    def get_uncertainty(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Get uncertainty estimates for the inferred states.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            
        Returns:
            Uncertainty estimates [batch_size, state_dim]
        """
        outputs = self.forward(observations)
        logvar = outputs['state_logvar']
        
        # Uncertainty is the variance
        uncertainty = torch.exp(logvar)
        
        return uncertainty 