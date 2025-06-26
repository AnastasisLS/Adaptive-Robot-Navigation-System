"""
Policy model for active inference.

Implements action selection based on Expected Free Energy (EFE)
minimization, which balances information gain and utility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List
import yaml


class PolicyModel(nn.Module):
    """
    Policy model for active inference.
    
    This model implements action selection based on Expected Free Energy (EFE):
    G(π) = E_q(s,o|π)[log q(s|π) - log p(s,o)]
    
    The policy selects actions that minimize expected free energy,
    balancing information gain and utility.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize the policy model.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            self.config = config_data['agent']['active_inference']
            self.policy_config = config_data['agent']['policy']
        
        # Model dimensions
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.num_samples = self.config['num_samples']
        
        # Get neural network configuration
        nn_config = config_data['agent']['neural_networks']['policy']
        
        # Policy network for action selection
        self.policy_network = self._build_policy_network(nn_config)
        
        # EFE computation parameters
        self.efe_weight = self.policy_config['efe_weight']
        self.information_gain_weight = self.policy_config['information_gain_weight']
        self.utility_weight = self.policy_config['utility_weight']
        self.temperature = self.config['temperature']
    
    def _build_policy_network(self, nn_config: Dict[str, Any]) -> nn.Module:
        """Build the policy network for action selection."""
        layers = []
        input_dim = self.state_dim
        
        # Hidden layers
        for hidden_dim in nn_config['hidden_dims']:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if nn_config['activation'] == 'relu' else nn.Tanh()
            ])
            input_dim = hidden_dim
        
        # Output layer (action logits)
        layers.append(nn.Linear(input_dim, self.action_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            states: Hidden states [batch_size, state_dim]
            
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.policy_network(states)
    
    def select_action(self, states: torch.Tensor, exploration_rate: float = None) -> torch.Tensor:
        """
        Select actions based on current states.
        
        Args:
            states: Hidden states [batch_size, state_dim]
            exploration_rate: Exploration rate for epsilon-greedy
            
        Returns:
            Selected actions [batch_size]
        """
        if exploration_rate is None:
            exploration_rate = self.policy_config['exploration_rate']
        
        # Get action logits
        logits = self.forward(states)
        
        # Epsilon-greedy exploration
        if np.random.random() < exploration_rate:
            # Random action
            actions = torch.randint(0, self.action_dim, (states.shape[0],), device=states.device)
        else:
            # Greedy action
            actions = torch.argmax(logits, dim=1)
        
        return actions
    
    def compute_expected_free_energy(self, states: torch.Tensor, actions: torch.Tensor,
                                   generative_model, recognition_model) -> torch.Tensor:
        """
        Compute Expected Free Energy (EFE) for given states and actions.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            generative_model: Generative model for predictions
            recognition_model: Recognition model for inference
            
        Returns:
            Expected free energy [batch_size]
        """
        batch_size = states.shape[0]
        
        # Sample next states from transition model
        next_states = generative_model.sample_next_states(states, actions)
        
        # Sample observations from observation model
        observations = generative_model.sample_observations(next_states)
        
        # Compute recognition model posterior for next states
        recog_outputs = recognition_model(observations)
        recog_mean = recog_outputs['state_mean']
        recog_logvar = recog_outputs['state_logvar']
        
        # Compute generative model prior for next states
        gen_outputs = generative_model(next_states)
        prior_mean = gen_outputs['prior_mean']
        prior_logvar = gen_outputs['prior_logvar']
        
        # Compute EFE components
        # 1. Information gain: KL[q(s'|o')||p(s')]
        kl_div = recognition_model.compute_kl_divergence(recog_mean, recog_logvar, prior_mean, prior_logvar)
        
        # 2. Utility: -log p(o'|s')
        utility = -generative_model.compute_observation_log_prob(next_states, observations)
        
        # 3. Expected free energy
        efe = self.information_gain_weight * kl_div + self.utility_weight * utility
        
        return efe
    
    def compute_policy_efe(self, states: torch.Tensor, generative_model, 
                          recognition_model, num_samples: int = None) -> torch.Tensor:
        """
        Compute Expected Free Energy for all possible actions.
        
        Args:
            states: Current states [batch_size, state_dim]
            generative_model: Generative model for predictions
            recognition_model: Recognition model for inference
            num_samples: Number of samples for EFE computation
            
        Returns:
            EFE for each action [batch_size, action_dim]
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        batch_size = states.shape[0]
        efe_matrix = torch.zeros(batch_size, self.action_dim, device=states.device)
        
        # Compute EFE for each action
        for action_idx in range(self.action_dim):
            # Create one-hot action encoding
            actions = torch.zeros(batch_size, self.action_dim, device=states.device)
            actions[:, action_idx] = 1.0
            
            # Sample multiple times for robust EFE estimation
            efe_samples = []
            for _ in range(num_samples):
                efe = self.compute_expected_free_energy(states, actions, generative_model, recognition_model)
                efe_samples.append(efe)
            
            # Average over samples
            efe_matrix[:, action_idx] = torch.stack(efe_samples).mean(dim=0)
        
        return efe_matrix
    
    def select_action_by_efe(self, states: torch.Tensor, generative_model, 
                           recognition_model) -> torch.Tensor:
        """
        Select actions by minimizing Expected Free Energy.
        
        Args:
            states: Current states [batch_size, state_dim]
            generative_model: Generative model for predictions
            recognition_model: Recognition model for inference
            
        Returns:
            Selected actions [batch_size]
        """
        # Compute EFE for all actions
        efe_matrix = self.compute_policy_efe(states, generative_model, recognition_model)
        
        # Convert EFE to probabilities (lower EFE = higher probability)
        # Use softmax with temperature
        logits = -efe_matrix / self.temperature
        probs = F.softmax(logits, dim=1)
        
        # Sample actions from probability distribution
        actions = torch.multinomial(probs, 1).squeeze(1)
        
        return actions
    
    def compute_action_probabilities(self, states: torch.Tensor, generative_model, 
                                   recognition_model) -> torch.Tensor:
        """
        Compute action probabilities based on EFE.
        
        Args:
            states: Current states [batch_size, state_dim]
            generative_model: Generative model for predictions
            recognition_model: Recognition model for inference
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        # Compute EFE for all actions
        efe_matrix = self.compute_policy_efe(states, generative_model, recognition_model)
        
        # Convert EFE to probabilities
        logits = -efe_matrix / self.temperature
        probs = F.softmax(logits, dim=1)
        
        return probs
    
    def compute_information_gain(self, states: torch.Tensor, actions: torch.Tensor,
                               generative_model, recognition_model) -> torch.Tensor:
        """
        Compute information gain for given states and actions.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            generative_model: Generative model for predictions
            recognition_model: Recognition model for inference
            
        Returns:
            Information gain [batch_size]
        """
        batch_size = states.shape[0]
        
        # Sample next states and observations
        next_states = generative_model.sample_next_states(states, actions)
        observations = generative_model.sample_observations(next_states)
        
        # Compute recognition model posterior
        recog_outputs = recognition_model(observations)
        recog_mean = recog_outputs['state_mean']
        recog_logvar = recog_outputs['state_logvar']
        
        # Compute generative model prior
        gen_outputs = generative_model(next_states)
        prior_mean = gen_outputs['prior_mean']
        prior_logvar = gen_outputs['prior_logvar']
        
        # Information gain is KL divergence
        info_gain = recognition_model.compute_kl_divergence(recog_mean, recog_logvar, prior_mean, prior_logvar)
        
        return info_gain
    
    def compute_utility(self, states: torch.Tensor, actions: torch.Tensor,
                       generative_model) -> torch.Tensor:
        """
        Compute utility for given states and actions.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            generative_model: Generative model for predictions
            
        Returns:
            Utility [batch_size]
        """
        # Sample next states and observations
        next_states = generative_model.sample_next_states(states, actions)
        observations = generative_model.sample_observations(next_states)
        
        # Utility is negative log likelihood of observations
        utility = -generative_model.compute_observation_log_prob(next_states, observations)
        
        return utility
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        """
        Update policy based on experience.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Taken actions [batch_size]
            rewards: Received rewards [batch_size]
            next_states: Next states [batch_size, state_dim]
            
        Returns:
            Policy loss
        """
        # Get action logits
        logits = self.forward(states)
        
        # Convert actions to one-hot
        action_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        
        # Compute policy loss (cross-entropy with reward weighting)
        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = torch.sum(log_probs * action_onehot, dim=1)
        
        # Weight by rewards
        policy_loss = -torch.mean(action_log_probs * rewards)
        
        return policy_loss 