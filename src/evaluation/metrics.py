"""
Performance metrics for evaluating active inference agents.

Implements various metrics for comparing active inference agents
with traditional reinforcement learning methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class NavigationMetrics:
    """
    Metrics for evaluating navigation performance.
    
    Includes success rate, efficiency, adaptability, and
    comparison metrics for different agent types.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.metrics_history = []
    
    def calculate_episode_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate metrics for a single episode.
        
        Args:
            episode_data: Dictionary containing episode information
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Basic episode metrics
        metrics['episode_length'] = episode_data.get('episode_length', 0)
        metrics['total_reward'] = episode_data.get('total_reward', 0.0)
        metrics['success'] = episode_data.get('success', False)
        metrics['collision_count'] = episode_data.get('collision_count', 0)
        
        # Efficiency metrics
        if 'path_length' in episode_data and 'optimal_path_length' in episode_data:
            metrics['path_efficiency'] = (
                episode_data['optimal_path_length'] / episode_data['path_length']
            )
        
        # Time efficiency
        if 'episode_time' in episode_data:
            metrics['time_efficiency'] = episode_data.get('episode_time', 0)
        
        # Energy efficiency (if battery info available)
        if 'battery_consumed' in episode_data:
            metrics['energy_efficiency'] = episode_data.get('battery_consumed', 0)
        
        # Adaptability metrics
        if 'environment_changes' in episode_data:
            metrics['adaptability_score'] = self._calculate_adaptability(
                episode_data['environment_changes'],
                episode_data.get('response_times', [])
            )
        
        # Uncertainty metrics (for active inference)
        if 'belief_uncertainty' in episode_data:
            metrics['avg_uncertainty'] = np.mean(episode_data['belief_uncertainty'])
            metrics['uncertainty_reduction'] = self._calculate_uncertainty_reduction(
                episode_data['belief_uncertainty']
            )
        
        # Free energy metrics (for active inference)
        if 'free_energy_history' in episode_data:
            metrics['avg_free_energy'] = np.mean(episode_data['free_energy_history'])
            metrics['free_energy_convergence'] = self._calculate_free_energy_convergence(
                episode_data['free_energy_history']
            )
        
        return metrics
    
    def _calculate_adaptability(self, changes: List[Dict], response_times: List[float]) -> float:
        """
        Calculate adaptability score based on response to environmental changes.
        
        Args:
            changes: List of environmental changes
            response_times: Time taken to respond to each change
            
        Returns:
            Adaptability score (higher is better)
        """
        if not changes or not response_times:
            return 0.0
        
        # Normalize response times (shorter is better)
        max_expected_time = 50.0  # Maximum expected response time
        normalized_times = [min(rt / max_expected_time, 1.0) for rt in response_times]
        
        # Adaptability score is inverse of average normalized response time
        avg_response_time = np.mean(normalized_times)
        adaptability = 1.0 - avg_response_time
        
        return max(0.0, adaptability)
    
    def _calculate_uncertainty_reduction(self, uncertainty_history: List[float]) -> float:
        """
        Calculate how much uncertainty was reduced during the episode.
        
        Args:
            uncertainty_history: History of belief uncertainty
            
        Returns:
            Uncertainty reduction score
        """
        if len(uncertainty_history) < 2:
            return 0.0
        
        initial_uncertainty = uncertainty_history[0]
        final_uncertainty = uncertainty_history[-1]
        
        if initial_uncertainty == 0:
            return 0.0
        
        reduction = (initial_uncertainty - final_uncertainty) / initial_uncertainty
        return max(0.0, reduction)
    
    def _calculate_free_energy_convergence(self, free_energy_history: List[float]) -> float:
        """
        Calculate how well free energy converged during the episode.
        
        Args:
            free_energy_history: History of variational free energy
            
        Returns:
            Convergence score (lower is better)
        """
        if len(free_energy_history) < 10:
            return 0.0
        
        # Calculate stability in the last 20% of the episode
        start_idx = int(0.8 * len(free_energy_history))
        recent_fe = free_energy_history[start_idx:]
        
        if len(recent_fe) < 2:
            return 0.0
        
        # Calculate coefficient of variation (std/mean)
        mean_fe = np.mean(recent_fe)
        std_fe = np.std(recent_fe)
        
        if mean_fe == 0:
            return 0.0
        
        cv = std_fe / mean_fe
        # Convert to convergence score (lower CV = better convergence)
        convergence = 1.0 / (1.0 + cv)
        
        return convergence
    
    def calculate_aggregate_metrics(self, episodes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across multiple episodes.
        
        Args:
            episodes_data: List of episode data dictionaries
            
        Returns:
            Dictionary of aggregate metrics
        """
        if not episodes_data:
            return {}
        
        # Calculate individual episode metrics
        episode_metrics = []
        for episode_data in episodes_data:
            metrics = self.calculate_episode_metrics(episode_data)
            episode_metrics.append(metrics)
        
        # Aggregate metrics
        aggregate = {}
        
        # Success rate
        successes = [m['success'] for m in episode_metrics]
        aggregate['success_rate'] = np.mean(successes)
        aggregate['success_count'] = sum(successes)
        aggregate['total_episodes'] = len(episodes_data)
        
        # Average metrics
        numeric_metrics = ['episode_length', 'total_reward', 'path_efficiency', 
                          'time_efficiency', 'energy_efficiency', 'adaptability_score',
                          'avg_uncertainty', 'uncertainty_reduction', 'avg_free_energy',
                          'free_energy_convergence']
        
        for metric in numeric_metrics:
            values = [m.get(metric, 0) for m in episode_metrics if metric in m]
            if values:
                aggregate[f'avg_{metric}'] = np.mean(values)
                aggregate[f'std_{metric}'] = np.std(values)
                aggregate[f'min_{metric}'] = np.min(values)
                aggregate[f'max_{metric}'] = np.max(values)
        
        # Learning curve metrics
        if len(episodes_data) > 10:
            # Split into early and late episodes
            mid_point = len(episodes_data) // 2
            early_metrics = episode_metrics[:mid_point]
            late_metrics = episode_metrics[mid_point:]
            
            # Calculate improvement
            for metric in numeric_metrics:
                early_values = [m.get(metric, 0) for m in early_metrics if metric in m]
                late_values = [m.get(metric, 0) for m in late_metrics if metric in m]
                
                if early_values and late_values:
                    early_avg = np.mean(early_values)
                    late_avg = np.mean(late_values)
                    if early_avg != 0:
                        improvement = (late_avg - early_avg) / abs(early_avg)
                        aggregate[f'{metric}_improvement'] = improvement
        
        return aggregate
    
    def compare_agents(self, agent_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare performance between different agents.
        
        Args:
            agent_results: Dictionary mapping agent names to episode data
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        # Calculate aggregate metrics for each agent
        agent_metrics = {}
        for agent_name, episodes_data in agent_results.items():
            agent_metrics[agent_name] = self.calculate_aggregate_metrics(episodes_data)
        
        # Statistical comparisons
        for metric in ['success_rate', 'avg_episode_length', 'avg_total_reward', 
                      'avg_path_efficiency', 'avg_adaptability_score']:
            values = []
            agent_names = []
            
            for agent_name, metrics in agent_metrics.items():
                if metric in metrics:
                    values.append(metrics[metric])
                    agent_names.append(agent_name)
            
            if len(values) >= 2:
                # Perform statistical test (t-test for now)
                if len(values) == 2:
                    stat, p_value = stats.ttest_ind(
                        agent_results[agent_names[0]], 
                        agent_results[agent_names[1]]
                    )
                    comparison[f'{metric}_p_value'] = p_value
                    comparison[f'{metric}_significant'] = p_value < 0.05
        
        # Ranking
        for metric in ['success_rate', 'avg_total_reward', 'avg_path_efficiency']:
            if all(metric in metrics for metrics in agent_metrics.values()):
                rankings = sorted(
                    [(name, metrics[metric]) for name, metrics in agent_metrics.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                comparison[f'{metric}_ranking'] = rankings
        
        return comparison
    
    def generate_report(self, agent_results: Dict[str, List[Dict[str, Any]]], 
                       save_path: str = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            agent_results: Dictionary mapping agent names to episode data
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("ADAPTIVE ROBOT NAVIGATION SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Individual agent results
        for agent_name, episodes_data in agent_results.items():
            report.append(f"AGENT: {agent_name.upper()}")
            report.append("-" * 40)
            
            metrics = self.calculate_aggregate_metrics(episodes_data)
            
            report.append(f"Total Episodes: {metrics.get('total_episodes', 0)}")
            report.append(f"Success Rate: {metrics.get('success_rate', 0):.3f} ({metrics.get('success_count', 0)}/{metrics.get('total_episodes', 0)})")
            report.append(f"Average Episode Length: {metrics.get('avg_episode_length', 0):.1f} ± {metrics.get('std_episode_length', 0):.1f}")
            report.append(f"Average Total Reward: {metrics.get('avg_total_reward', 0):.2f} ± {metrics.get('std_total_reward', 0):.2f}")
            
            if 'avg_path_efficiency' in metrics:
                report.append(f"Path Efficiency: {metrics['avg_path_efficiency']:.3f} ± {metrics.get('std_path_efficiency', 0):.3f}")
            
            if 'avg_adaptability_score' in metrics:
                report.append(f"Adaptability Score: {metrics['avg_adaptability_score']:.3f} ± {metrics.get('std_adaptability_score', 0):.3f}")
            
            if 'avg_free_energy' in metrics:
                report.append(f"Average Free Energy: {metrics['avg_free_energy']:.4f} ± {metrics.get('std_avg_free_energy', 0):.4f}")
                report.append(f"Free Energy Convergence: {metrics.get('free_energy_convergence', 0):.3f}")
            
            report.append("")
        
        # Comparison results
        comparison = self.compare_agents(agent_results)
        if comparison:
            report.append("AGENT COMPARISON")
            report.append("-" * 40)
            
            for metric in ['success_rate', 'avg_total_reward', 'avg_path_efficiency']:
                ranking_key = f'{metric}_ranking'
                if ranking_key in comparison:
                    report.append(f"{metric.replace('_', ' ').title()} Ranking:")
                    for i, (agent, value) in enumerate(comparison[ranking_key]):
                        report.append(f"  {i+1}. {agent}: {value:.3f}")
                    report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_comparison(self, agent_results: Dict[str, List[Dict[str, Any]]], 
                       save_path: str = None):
        """
        Create comparison plots for different agents.
        
        Args:
            agent_results: Dictionary mapping agent names to episode data
            save_path: Path to save the plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Prepare data
        metrics_data = {}
        for agent_name, episodes_data in agent_results.items():
            episode_metrics = []
            for episode_data in episodes_data:
                metrics = self.calculate_episode_metrics(episode_data)
                episode_metrics.append(metrics)
            metrics_data[agent_name] = episode_metrics
        
        # Plot 1: Success Rate
        success_rates = []
        agent_names = []
        for agent_name, metrics in metrics_data.items():
            success_rate = np.mean([m['success'] for m in metrics])
            success_rates.append(success_rate)
            agent_names.append(agent_name)
        
        axes[0, 0].bar(agent_names, success_rates)
        axes[0, 0].set_title('Success Rate')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # Plot 2: Average Episode Length
        episode_lengths = []
        for agent_name, metrics in metrics_data.items():
            avg_length = np.mean([m['episode_length'] for m in metrics])
            episode_lengths.append(avg_length)
        
        axes[0, 1].bar(agent_names, episode_lengths)
        axes[0, 1].set_title('Average Episode Length')
        axes[0, 1].set_ylabel('Steps')
        
        # Plot 3: Average Total Reward
        total_rewards = []
        for agent_name, metrics in metrics_data.items():
            avg_reward = np.mean([m['total_reward'] for m in metrics])
            total_rewards.append(avg_reward)
        
        axes[0, 2].bar(agent_names, total_rewards)
        axes[0, 2].set_title('Average Total Reward')
        axes[0, 2].set_ylabel('Reward')
        
        # Plot 4: Learning Curves (Episode Rewards)
        for agent_name, metrics in metrics_data.items():
            rewards = [m['total_reward'] for m in metrics]
            axes[1, 0].plot(rewards, label=agent_name, alpha=0.7)
        
        axes[1, 0].set_title('Learning Curves')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Path Efficiency (if available)
        efficiency_data = {}
        for agent_name, metrics in metrics_data.items():
            efficiencies = [m.get('path_efficiency', 0) for m in metrics if 'path_efficiency' in m]
            if efficiencies:
                efficiency_data[agent_name] = efficiencies
        
        if efficiency_data:
            axes[1, 1].boxplot(efficiency_data.values(), labels=efficiency_data.keys())
            axes[1, 1].set_title('Path Efficiency Distribution')
            axes[1, 1].set_ylabel('Efficiency')
        
        # Plot 6: Free Energy (for active inference agents)
        fe_data = {}
        for agent_name, metrics in metrics_data.items():
            if 'active_inference' in agent_name.lower():
                fe_values = [m.get('avg_free_energy', 0) for m in metrics if 'avg_free_energy' in m]
                if fe_values:
                    fe_data[agent_name] = fe_values
        
        if fe_data:
            axes[1, 2].boxplot(fe_data.values(), labels=fe_data.keys())
            axes[1, 2].set_title('Free Energy Distribution')
            axes[1, 2].set_ylabel('Free Energy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 