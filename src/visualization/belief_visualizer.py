"""
Belief visualizer for active inference.

Provides visualization of internal belief states, uncertainty,
and free energy dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import torch


class BeliefVisualizer:
    """
    Belief state visualizer for active inference.
    
    Provides visualization of:
    - Belief state distributions
    - Uncertainty estimates
    - Free energy dynamics
    - Belief convergence
    """
    
    def __init__(self, figsize: tuple = (15, 10)):
        """
        Initialize belief visualizer.
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.belief_history = []
        self.free_energy_history = []
        self.uncertainty_history = []
        
    def update_beliefs(self, belief_data: Dict[str, Any], free_energy: float = None):
        """
        Update belief history with new data.
        
        Args:
            belief_data: Current belief state data
            free_energy: Current free energy value
        """
        self.belief_history.append(belief_data)
        
        if free_energy is not None:
            self.free_energy_history.append(free_energy)
        
        # Extract uncertainty if available
        if 'logvar' in belief_data:
            uncertainty = np.exp(belief_data['logvar'])
            self.uncertainty_history.append(uncertainty)
    
    def plot_belief_evolution(self, save_path: Optional[str] = None):
        """
        Plot evolution of belief states over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.belief_history:
            print("No belief history available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Extract belief means and uncertainties
        belief_means = [b['mean'].flatten() for b in self.belief_history]
        belief_means = np.array(belief_means)
        
        # Plot belief means over time
        for i in range(min(belief_means.shape[1], 5)):  # Plot first 5 dimensions
            axes[0, 0].plot(belief_means[:, i], label=f'Dimension {i+1}')
        
        axes[0, 0].set_title('Belief State Evolution')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Belief Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot uncertainty evolution
        if self.uncertainty_history:
            uncertainties = np.array(self.uncertainty_history)
            for i in range(min(uncertainties.shape[1], 5)):
                axes[0, 1].plot(uncertainties[:, i], label=f'Dimension {i+1}')
            
            axes[0, 1].set_title('Uncertainty Evolution')
            axes[0, 1].set_xlabel('Time Step')
            axes[0, 1].set_ylabel('Uncertainty (Variance)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot free energy
        if self.free_energy_history:
            axes[1, 0].plot(self.free_energy_history)
            axes[1, 0].set_title('Variational Free Energy')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Free Energy')
            axes[1, 0].grid(True)
        
        # Plot belief correlation matrix
        if len(belief_means) > 1:
            corr_matrix = np.corrcoef(belief_means.T)
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('Belief State Correlations')
            axes[1, 1].set_xlabel('Dimension')
            axes[1, 1].set_ylabel('Dimension')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_belief_distribution(self, belief_data: Dict[str, Any], 
                               save_path: Optional[str] = None):
        """
        Plot current belief distribution.
        
        Args:
            belief_data: Current belief state data
            save_path: Optional path to save the plot
        """
        if 'mean' not in belief_data or 'logvar' not in belief_data:
            print("Incomplete belief data for distribution plot.")
            return
        
        mean = belief_data['mean'].flatten()
        logvar = belief_data['logvar'].flatten()
        std = np.exp(0.5 * logvar)
        
        # Create subplots for each dimension
        n_dims = len(mean)
        n_cols = min(4, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_dims):
            row = i // n_cols
            col = i % n_cols
            
            # Generate samples from the belief distribution
            samples = np.random.normal(mean[i], std[i], 1000)
            
            # Plot histogram
            axes[row, col].hist(samples, bins=30, alpha=0.7, density=True)
            axes[row, col].axvline(mean[i], color='red', linestyle='--', 
                                 label=f'Mean: {mean[i]:.3f}')
            axes[row, col].set_title(f'Dimension {i+1}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_dims, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Belief State Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_heatmap(self, save_path: Optional[str] = None):
        """
        Plot uncertainty heatmap over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.uncertainty_history:
            print("No uncertainty history available for heatmap.")
            return
        
        uncertainties = np.array(self.uncertainty_history)
        
        plt.figure(figsize=(12, 6))
        
        # Create heatmap
        sns.heatmap(uncertainties.T, cmap='viridis', 
                   xticklabels=50, yticklabels=range(1, uncertainties.shape[1]+1))
        
        plt.title('Uncertainty Evolution Heatmap')
        plt.xlabel('Time Step')
        plt.ylabel('Belief Dimension')
        plt.colorbar(label='Uncertainty (Variance)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_free_energy_convergence(self, window_size: int = 10, 
                                   save_path: Optional[str] = None):
        """
        Plot free energy convergence analysis.
        
        Args:
            window_size: Window size for moving average
            save_path: Optional path to save the plot
        """
        if not self.free_energy_history:
            print("No free energy history available for convergence analysis.")
            return
        
        free_energy = np.array(self.free_energy_history)
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Raw free energy
        axes[0].plot(free_energy, alpha=0.7, label='Raw Free Energy')
        
        # Moving average
        if len(free_energy) >= window_size:
            moving_avg = np.convolve(free_energy, np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(range(window_size-1, len(free_energy)), moving_avg, 
                        linewidth=2, label=f'Moving Average (window={window_size})')
        
        axes[0].set_title('Free Energy Evolution')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Free Energy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Convergence analysis
        if len(free_energy) > 1:
            # Compute rate of change
            rate_of_change = np.diff(free_energy)
            axes[1].plot(rate_of_change, alpha=0.7)
            axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_title('Free Energy Rate of Change')
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('Δ Free Energy')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_belief_comparison(self, belief_data1: Dict[str, Any], 
                             belief_data2: Dict[str, Any],
                             labels: List[str] = None,
                             save_path: Optional[str] = None):
        """
        Compare two belief states.
        
        Args:
            belief_data1: First belief state
            belief_data2: Second belief state
            labels: Labels for the two belief states
            save_path: Optional path to save the plot
        """
        if labels is None:
            labels = ['Belief 1', 'Belief 2']
        
        if 'mean' not in belief_data1 or 'mean' not in belief_data2:
            print("Incomplete belief data for comparison.")
            return
        
        mean1 = belief_data1['mean'].flatten()
        mean2 = belief_data2['mean'].flatten()
        
        # Ensure same dimensions
        min_dims = min(len(mean1), len(mean2))
        mean1 = mean1[:min_dims]
        mean2 = mean2[:min_dims]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(mean1, mean2, alpha=0.7)
        ax1.plot([mean1.min(), mean1.max()], [mean1.min(), mean1.max()], 
                'r--', alpha=0.5, label='Identity line')
        ax1.set_xlabel(labels[0])
        ax1.set_ylabel(labels[1])
        ax1.set_title('Belief State Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of differences
        differences = mean2 - mean1
        ax2.bar(range(len(differences)), differences, alpha=0.7)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Difference')
        ax2.set_title('Belief State Differences')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def clear_history(self):
        """Clear belief history."""
        self.belief_history = []
        self.free_energy_history = []
        self.uncertainty_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about belief evolution."""
        if not self.belief_history:
            return {}
        
        stats = {
            'num_belief_updates': len(self.belief_history),
            'belief_dimensions': len(self.belief_history[0]['mean'].flatten())
        }
        
        if self.free_energy_history:
            fe_array = np.array(self.free_energy_history)
            stats.update({
                'free_energy_mean': float(np.mean(fe_array)),
                'free_energy_std': float(np.std(fe_array)),
                'free_energy_min': float(np.min(fe_array)),
                'free_energy_max': float(np.max(fe_array))
            })
        
        if self.uncertainty_history:
            unc_array = np.array(self.uncertainty_history)
            stats.update({
                'uncertainty_mean': float(np.mean(unc_array)),
                'uncertainty_std': float(np.std(unc_array))
            })
        
        return stats 