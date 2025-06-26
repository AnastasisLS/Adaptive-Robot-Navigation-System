"""
Comparison framework for active inference vs reinforcement learning.

Provides systematic evaluation and comparison between active inference
agents and traditional reinforcement learning methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os

from ..environment import NavigationEnvironment
from ..agents import ActiveInferenceAgent
from ..agents.rl_agents import PPOAgent, DQNAgent
from .metrics import NavigationMetrics


class AgentComparison:
    """
    Comprehensive comparison between different agent types.
    
    Compares active inference agents with traditional RL methods
    across various metrics and scenarios.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """
        Initialize the comparison framework.
        
        Args:
            config_path: Path to agent configuration file
        """
        self.config_path = config_path
        self.metrics = NavigationMetrics()
        self.results = {}
        
    def run_comparison_experiment(self, 
                                agents: Dict[str, Any],
                                num_episodes: int = 50,
                                max_steps: int = 1000,
                                scenarios: List[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive comparison experiment.
        
        Args:
            agents: Dictionary mapping agent names to agent instances
            num_episodes: Number of episodes per agent
            max_steps: Maximum steps per episode
            scenarios: List of scenario names to test
            
        Returns:
            Dictionary containing all results
        """
        if scenarios is None:
            scenarios = ['basic', 'dynamic_obstacles', 'changing_goals']
        
        print(f"Starting comparison experiment with {len(agents)} agents")
        print(f"Testing scenarios: {scenarios}")
        print(f"Episodes per agent: {num_episodes}")
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n=== Testing Scenario: {scenario} ===")
            scenario_results = self._run_scenario(
                agents, scenario, num_episodes, max_steps
            )
            all_results[scenario] = scenario_results
        
        # Aggregate results across scenarios
        aggregated_results = self._aggregate_scenario_results(all_results)
        
        # Generate comparison report
        self._generate_comparison_report(aggregated_results)
        
        return aggregated_results
    
    def _run_scenario(self, agents: Dict[str, Any], scenario: str,
                     num_episodes: int, max_steps: int) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run a specific scenario for all agents.
        
        Args:
            agents: Dictionary of agents to test
            scenario: Scenario name
            num_episodes: Number of episodes per agent
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary mapping agent names to episode results
        """
        scenario_results = {}
        
        for agent_name, agent in agents.items():
            print(f"\nTesting agent: {agent_name}")
            
            # Create environment for this scenario
            env = self._create_scenario_environment(scenario)
            
            # Run episodes
            agent_results = []
            for episode in tqdm(range(num_episodes), desc=f"Episodes for {agent_name}"):
                episode_result = self._run_single_episode(
                    agent, env, max_steps, scenario
                )
                agent_results.append(episode_result)
            
            scenario_results[agent_name] = agent_results
            
            # Reset agent for next scenario
            if hasattr(agent, 'reset'):
                agent.reset()
        
        return scenario_results
    
    def _create_scenario_environment(self, scenario: str) -> NavigationEnvironment:
        """
        Create environment configured for specific scenario.
        
        Args:
            scenario: Scenario name
            
        Returns:
            Configured navigation environment
        """
        env = NavigationEnvironment()
        
        if scenario == 'basic':
            # Basic navigation with static obstacles
            pass  # Use default configuration
        
        elif scenario == 'dynamic_obstacles':
            # Increase dynamic obstacles
            env.config['obstacles']['num_dynamic'] = 10
            env.config['obstacles']['dynamic_speed'] = 0.8
        
        elif scenario == 'changing_goals':
            # Enable goal changes
            env.config['dynamics']['goal_changes'] = True
            env.config['dynamics']['change_frequency'] = 50
        
        elif scenario == 'complex':
            # Complex scenario with multiple challenges
            env.config['obstacles']['num_static'] = 25
            env.config['obstacles']['num_dynamic'] = 8
            env.config['obstacles']['dynamic_speed'] = 0.6
            env.config['dynamics']['goal_changes'] = True
            env.config['dynamics']['change_frequency'] = 75
        
        return env
    
    def _run_single_episode(self, agent: Any, env: NavigationEnvironment,
                           max_steps: int, scenario: str) -> Dict[str, Any]:
        """
        Run a single episode with detailed tracking.
        
        Args:
            agent: Agent to test
            env: Environment
            max_steps: Maximum steps
            scenario: Scenario name
            
        Returns:
            Episode result dictionary
        """
        observation = env.reset()
        episode_data = {
            'episode_length': 0,
            'total_reward': 0.0,
            'success': False,
            'collision_count': 0,
            'path_length': 0.0,
            'belief_uncertainty': [],
            'free_energy_history': [],
            'environment_changes': [],
            'response_times': [],
            'actions_taken': [],
            'rewards': []
        }
        
        start_time = time.time()
        last_change_time = 0
        
        for step in range(max_steps):
            # Track environment changes
            if hasattr(env, 'step_count') and env.step_count % 100 == 0:
                if env.step_count > last_change_time:
                    episode_data['environment_changes'].append({
                        'step': env.step_count,
                        'type': 'obstacle_movement' if env.config['dynamics']['obstacle_movement'] else 'none'
                    })
                    last_change_time = env.step_count
            
            # Select action
            action_start = time.time()
            action = agent.select_action(observation, training=True)
            action_time = time.time() - action_start
            
            # Take step
            next_observation, reward, done, info = env.step(action)
            
            # Update agent
            if hasattr(agent, 'step'):
                agent.step(observation, reward, done, info)
            
            # Track metrics
            episode_data['episode_length'] += 1
            episode_data['total_reward'] += reward
            episode_data['actions_taken'].append(action)
            episode_data['rewards'].append(reward)
            
            # Track collisions
            if info.get('collision', False):
                episode_data['collision_count'] += 1
            
            # Track path length
            if hasattr(env, 'robot'):
                if step > 0:
                    prev_pos = env.robot.position
                    curr_pos = env.robot.position
                    episode_data['path_length'] += np.linalg.norm(curr_pos - prev_pos)
            
            # Track active inference specific metrics
            if hasattr(agent, 'current_beliefs') and agent.current_beliefs is not None:
                # Belief uncertainty
                uncertainty = np.mean(np.exp(agent.current_beliefs['logvar']))
                episode_data['belief_uncertainty'].append(uncertainty)
                
                # Free energy
                if hasattr(agent, 'get_free_energy'):
                    fe = agent.get_free_energy(observation)
                    episode_data['free_energy_history'].append(fe)
            
            # Track response times to changes
            if episode_data['environment_changes'] and action_time > 0.01:
                episode_data['response_times'].append(action_time)
            
            # Check if done
            if done:
                episode_data['success'] = info.get('goal_reached', False)
                break
            
            observation = next_observation
        
        # Calculate episode time
        episode_data['episode_time'] = time.time() - start_time
        
        # Calculate optimal path length (straight line to goal)
        if hasattr(env, 'robot') and hasattr(env, 'goals') and env.goals:
            start_pos = np.array(env.robot_config['initial_position'])
            goal_pos = env.goals[0].position
            episode_data['optimal_path_length'] = np.linalg.norm(goal_pos - start_pos)
        
        return episode_data
    
    def _aggregate_scenario_results(self, all_results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Aggregate results across all scenarios.
        
        Args:
            all_results: Results from all scenarios
            
        Returns:
            Aggregated results
        """
        aggregated = {}
        
        # Get all agent names
        all_agents = set()
        for scenario_results in all_results.values():
            all_agents.update(scenario_results.keys())
        
        # Aggregate across scenarios for each agent
        for agent_name in all_agents:
            agent_episodes = []
            for scenario, scenario_results in all_results.items():
                if agent_name in scenario_results:
                    agent_episodes.extend(scenario_results[agent_name])
            
            aggregated[agent_name] = agent_episodes
        
        return aggregated
    
    def _generate_comparison_report(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Generate comprehensive comparison report.
        
        Args:
            results: Aggregated results from all agents
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*60)
        
        # Generate text report
        report_text = self.metrics.generate_report(
            results, 
            save_path="data/experiments/comparison_report.txt"
        )
        print(report_text)
        
        # Generate comparison plots
        self.metrics.plot_comparison(
            results,
            save_path="data/experiments/comparison_plots.png"
        )
        
        # Generate detailed analysis
        self._generate_detailed_analysis(results)
    
    def _generate_detailed_analysis(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Generate detailed statistical analysis.
        
        Args:
            results: Aggregated results from all agents
        """
        print("\n=== DETAILED STATISTICAL ANALYSIS ===")
        
        # Calculate metrics for each agent
        agent_metrics = {}
        for agent_name, episodes_data in results.items():
            agent_metrics[agent_name] = self.metrics.calculate_aggregate_metrics(episodes_data)
        
        # Statistical significance tests
        print("\nStatistical Significance Tests:")
        print("-" * 40)
        
        # Compare success rates
        success_rates = []
        agent_names = []
        for agent_name, metrics in agent_metrics.items():
            if 'success_rate' in metrics:
                success_rates.append(metrics['success_rate'])
                agent_names.append(agent_name)
        
        if len(success_rates) >= 2:
            from scipy import stats
            # Perform ANOVA if more than 2 agents, otherwise t-test
            if len(success_rates) > 2:
                f_stat, p_value = stats.f_oneway(*[
                    [1 if ep['success'] else 0 for ep in results[name]]
                    for name in agent_names
                ])
                print(f"Success Rate ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
                print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            else:
                agent1_successes = [1 if ep['success'] else 0 for ep in results[agent_names[0]]]
                agent2_successes = [1 if ep['success'] else 0 for ep in results[agent_names[1]]]
                stat, p_value = stats.ttest_ind(agent1_successes, agent2_successes)
                print(f"Success Rate t-test: t={stat:.3f}, p={p_value:.4f}")
                print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Learning curve analysis
        print("\nLearning Curve Analysis:")
        print("-" * 40)
        
        for agent_name, episodes_data in results.items():
            if len(episodes_data) >= 20:
                # Split into early and late episodes
                mid_point = len(episodes_data) // 2
                early_rewards = [ep['total_reward'] for ep in episodes_data[:mid_point]]
                late_rewards = [ep['total_reward'] for ep in episodes_data[mid_point:]]
                
                early_avg = np.mean(early_rewards)
                late_avg = np.mean(late_rewards)
                improvement = (late_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0
                
                print(f"{agent_name}:")
                print(f"  Early episodes avg reward: {early_avg:.2f}")
                print(f"  Late episodes avg reward: {late_avg:.2f}")
                print(f"  Improvement: {improvement:.1%}")
        
        # Active inference specific analysis
        print("\nActive Inference Specific Analysis:")
        print("-" * 40)
        
        for agent_name, episodes_data in results.items():
            if 'active_inference' in agent_name.lower():
                # Free energy analysis
                fe_values = []
                for ep in episodes_data:
                    if 'free_energy_history' in ep and ep['free_energy_history']:
                        fe_values.extend(ep['free_energy_history'])
                
                if fe_values:
                    print(f"{agent_name} Free Energy:")
                    print(f"  Average: {np.mean(fe_values):.4f}")
                    print(f"  Std: {np.std(fe_values):.4f}")
                    print(f"  Min: {np.min(fe_values):.4f}")
                    print(f"  Max: {np.max(fe_values):.4f}")
                
                # Uncertainty analysis
                uncertainty_values = []
                for ep in episodes_data:
                    if 'belief_uncertainty' in ep and ep['belief_uncertainty']:
                        uncertainty_values.extend(ep['belief_uncertainty'])
                
                if uncertainty_values:
                    print(f"{agent_name} Belief Uncertainty:")
                    print(f"  Average: {np.mean(uncertainty_values):.4f}")
                    print(f"  Std: {np.std(uncertainty_values):.4f}")
    
    def create_agent_ensemble(self) -> Dict[str, Any]:
        """
        Create a standard set of agents for comparison.
        
        Returns:
            Dictionary of agents for comparison
        """
        # Create environment to get observation/action spaces
        env = NavigationEnvironment()
        
        agents = {}
        
        # Active Inference Agent
        agents['ActiveInference'] = ActiveInferenceAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config_path=self.config_path
        )
        
        # PPO Agent
        agents['PPO'] = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config_path=self.config_path
        )
        
        # DQN Agent
        agents['DQN'] = DQNAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config_path=self.config_path
        )
        
        return agents
    
    def run_quick_comparison(self, num_episodes: int = 20) -> Dict[str, Any]:
        """
        Run a quick comparison with default agents.
        
        Args:
            num_episodes: Number of episodes per agent
            
        Returns:
            Comparison results
        """
        print("Running quick comparison experiment...")
        
        # Create agents
        agents = self.create_agent_ensemble()
        
        # Run comparison
        results = self.run_comparison_experiment(
            agents=agents,
            num_episodes=num_episodes,
            max_steps=500,
            scenarios=['basic']
        )
        
        return results


def run_comparison_experiment():
    """Run a complete comparison experiment."""
    # Create comparison framework
    comparison = AgentComparison()
    
    # Run quick comparison
    results = comparison.run_quick_comparison(num_episodes=30)
    
    print("\nExperiment completed!")
    print("Results saved to data/experiments/")
    
    return results


if __name__ == "__main__":
    run_comparison_experiment() 