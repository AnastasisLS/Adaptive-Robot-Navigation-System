"""
Variational inference for active inference.

Implements variational inference algorithms for belief updates
and variational free energy minimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, List
import yaml


class VariationalInference:
    """
    Variational inference for active inference.
    
    Implements belief updates and variational free energy minimization
    for the active inference framework.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize variational inference.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            self.config = config_data['agent']['active_inference']
            self.opt_config = config_data['agent']['optimization']
        
        # Inference parameters
        self.learning_rate = self.config['learning_rate']
        self.kl_weight = self.config['kl_weight']
        self.num_samples = self.config['num_samples']
        
        # Optimization parameters
        self.optimizer_type = self.opt_config['optimizer']
        self.opt_learning_rate = self.opt_config['learning_rate']
        self.weight_decay = self.opt_config['weight_decay']
        self.gradient_clip = self.opt_config['gradient_clip']
    
    def compute_variational_free_energy(self, observations: torch.Tensor,
                                      states: torch.Tensor,
                                      generative_model,
                                      recognition_model) -> torch.Tensor:
        """
        Compute variational free energy.
        
        F = E_q(s)[log q(s) - log p(s,o)]
        = E_q(s)[log q(s) - log p(s) - log p(o|s)]
        = KL[q(s)||p(s)] - E_q(s)[log p(o|s)]
        
        Args:
            observations: Observations [batch_size, observation_dim]
            states: Sampled states [batch_size, state_dim]
            generative_model: Generative model
            recognition_model: Recognition model
            
        Returns:
            Variational free energy [batch_size]
        """
        # Get recognition model outputs
        recog_outputs = recognition_model(observations)
        recog_mean = recog_outputs['state_mean']
        recog_logvar = recog_outputs['state_logvar']
        
        # Get generative model outputs
        gen_outputs = generative_model(states)
        prior_mean = gen_outputs['prior_mean']
        prior_logvar = gen_outputs['prior_logvar']
        
        # Compute KL divergence: KL[q(s)||p(s)]
        kl_div = recognition_model.compute_kl_divergence(recog_mean, recog_logvar, prior_mean, prior_logvar)
        
        # Compute expected log likelihood: E_q(s)[log p(o|s)]
        expected_log_likelihood = generative_model.compute_observation_log_prob(states, observations)
        
        # Variational free energy
        free_energy = kl_div - expected_log_likelihood
        
        return free_energy
    
    def update_beliefs(self, observations: torch.Tensor,
                      generative_model,
                      recognition_model,
                      num_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """
        Update beliefs by minimizing variational free energy.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            generative_model: Generative model
            recognition_model: Recognition model
            num_iterations: Number of belief update iterations
            
        Returns:
            Dictionary containing updated beliefs and metrics
        """
        # Initialize optimizer for recognition model parameters
        optimizer = self._get_optimizer(recognition_model)
        
        # Belief update history
        free_energy_history = []
        kl_div_history = []
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Sample states from recognition model
            states = recognition_model.sample_states(observations)
            
            # Compute variational free energy
            free_energy = self.compute_variational_free_energy(
                observations, states, generative_model, recognition_model
            )
            
            # Compute loss (average free energy)
            loss = torch.mean(free_energy)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(recognition_model.parameters(), self.gradient_clip)
            
            # Update parameters
            optimizer.step()
            
            # Record metrics
            free_energy_history.append(loss.item())
            
            # Compute KL divergence for monitoring
            with torch.no_grad():
                recog_outputs = recognition_model(observations)
                gen_outputs = generative_model(states)
                kl_div = recognition_model.compute_kl_divergence(
                    recog_outputs['state_mean'], recog_outputs['state_logvar'],
                    gen_outputs['prior_mean'], gen_outputs['prior_logvar']
                )
                kl_div_history.append(torch.mean(kl_div).item())
        
        # Get final beliefs
        with torch.no_grad():
            final_outputs = recognition_model(observations)
            final_states = recognition_model.sample_states(observations)
        
        return {
            'belief_mean': final_outputs['state_mean'],
            'belief_logvar': final_outputs['state_logvar'],
            'belief_states': final_states,
            'free_energy_history': free_energy_history,
            'kl_div_history': kl_div_history,
            'final_free_energy': free_energy_history[-1],
            'final_kl_div': kl_div_history[-1]
        }
    
    def compute_expected_free_energy(self, states: torch.Tensor,
                                   actions: torch.Tensor,
                                   generative_model,
                                   recognition_model,
                                   num_samples: int = None) -> torch.Tensor:
        """
        Compute Expected Free Energy (EFE).
        
        G(π) = E_q(s,o|π)[log q(s|π) - log p(s,o)]
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions [batch_size, action_dim]
            generative_model: Generative model
            recognition_model: Recognition model
            num_samples: Number of samples for estimation
            
        Returns:
            Expected free energy [batch_size]
        """
        if num_samples is None:
            num_samples = self.num_samples
        
        batch_size = states.shape[0]
        efe_samples = []
        
        for _ in range(num_samples):
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
            
            # Compute KL divergence
            kl_div = recognition_model.compute_kl_divergence(observations, prior_mean, prior_logvar)
            
            # Compute expected log likelihood
            expected_log_likelihood = generative_model.compute_observation_log_prob(next_states, observations)
            
            # Expected free energy
            efe = kl_div - expected_log_likelihood
            efe_samples.append(efe)
        
        # Average over samples
        efe = torch.stack(efe_samples).mean(dim=0)
        
        return efe
    
    def optimize_models(self, observations: torch.Tensor,
                       states: torch.Tensor,
                       actions: torch.Tensor,
                       rewards: torch.Tensor,
                       generative_model,
                       recognition_model,
                       policy_model) -> Dict[str, float]:
        """
        Optimize all models using variational inference.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            states: States [batch_size, state_dim]
            actions: Actions [batch_size]
            rewards: Rewards [batch_size]
            generative_model: Generative model
            recognition_model: Recognition model
            policy_model: Policy model
            
        Returns:
            Dictionary containing losses
        """
        # Create optimizers
        gen_optimizer = self._get_optimizer(generative_model)
        recog_optimizer = self._get_optimizer(recognition_model)
        policy_optimizer = self._get_optimizer(policy_model)
        
        # Convert actions to one-hot
        action_onehot = torch.zeros(actions.shape[0], policy_model.action_dim, device=actions.device)
        action_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
        
        # Optimize generative model
        gen_optimizer.zero_grad()
        gen_loss = self._compute_generative_loss(observations, states, action_onehot, generative_model)
        gen_loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(generative_model.parameters(), self.gradient_clip)
        gen_optimizer.step()
        
        # Optimize recognition model
        recog_optimizer.zero_grad()
        recog_loss = self._compute_recognition_loss(observations, states, generative_model, recognition_model)
        recog_loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(recognition_model.parameters(), self.gradient_clip)
        recog_optimizer.step()
        
        # Optimize policy model
        policy_optimizer.zero_grad()
        policy_loss = policy_model.update_policy(states, actions, rewards, states)  # Simplified
        policy_loss.backward()
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), self.gradient_clip)
        policy_optimizer.step()
        
        return {
            'generative_loss': gen_loss.item(),
            'recognition_loss': recog_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def _compute_generative_loss(self, observations: torch.Tensor,
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               generative_model) -> torch.Tensor:
        """Compute generative model loss."""
        # Reconstruction loss
        gen_outputs = generative_model(states, actions)
        obs_mean = gen_outputs['observation_mean']
        obs_logvar = gen_outputs['observation_logvar']
        
        # MSE loss for observations
        recon_loss = torch.mean((observations - obs_mean) ** 2)
        
        # For generative model, we focus on reconstruction loss
        # KL divergence is handled in recognition loss
        return recon_loss
    
    def _compute_recognition_loss(self, observations: torch.Tensor,
                                states: torch.Tensor,
                                generative_model,
                                recognition_model) -> torch.Tensor:
        """Compute recognition model loss."""
        # Recognition model outputs
        recog_outputs = recognition_model(observations)
        recog_mean = recog_outputs['state_mean']
        recog_logvar = recog_outputs['state_logvar']
        
        # Generative model prior
        gen_outputs = generative_model(states)
        prior_mean = gen_outputs['prior_mean']
        prior_logvar = gen_outputs['prior_logvar']
        
        # KL divergence
        kl_loss = torch.mean(0.5 * torch.sum(
            prior_logvar - recog_logvar + 
            (torch.exp(recog_logvar) + (recog_mean - prior_mean) ** 2) / torch.exp(prior_logvar) - 1,
            dim=1
        ))
        
        return kl_loss
    
    def _get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Get optimizer for a model."""
        if self.optimizer_type.lower() == 'adam':
            return optim.Adam(
                model.parameters(),
                lr=self.opt_learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                model.parameters(),
                lr=self.opt_learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def compute_belief_uncertainty(self, observations: torch.Tensor,
                                 recognition_model) -> torch.Tensor:
        """
        Compute uncertainty in beliefs.
        
        Args:
            observations: Observations [batch_size, observation_dim]
            recognition_model: Recognition model
            
        Returns:
            Uncertainty estimates [batch_size, state_dim]
        """
        with torch.no_grad():
            outputs = recognition_model(observations)
            uncertainty = torch.exp(outputs['state_logvar'])
        
        return uncertainty
    
    def compute_belief_convergence(self, free_energy_history: List[float],
                                 threshold: float = 1e-4) -> bool:
        """
        Check if beliefs have converged.
        
        Args:
            free_energy_history: History of free energy values
            threshold: Convergence threshold
            
        Returns:
            True if converged, False otherwise
        """
        if len(free_energy_history) < 2:
            return False
        
        # Check if free energy has stabilized
        recent_changes = np.diff(free_energy_history[-5:])
        return np.all(np.abs(recent_changes) < threshold) 