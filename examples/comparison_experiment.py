#!/usr/bin/env python3
"""
Comparison experiment between active inference and RL methods.

Compares the performance of active inference agents against
PPO and DQN baselines in navigation tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import time
import os
from typing import Dict, List, Any, Tuple
import csv

from src.environment import NavigationEnvironment
from src.agents import ActiveInferenceAgent, PPOAgent, DQNAgent, CustomDQNAgent


class ComparisonExperiment:
    """
    Comprehensive comparison experiment between active inference and RL methods.
    """
    
    def __init__(self, num_episodes: int = 500, max_steps: int = 1000):
        """
        Initialize comparison experiment.
        
        Args:
            num_episodes: Number of episodes per agent
            max_steps: Maximum steps per episode
        """
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        # Create environment
        self.env = NavigationEnvironment()
        
        # Initialize agents
        self.agents = {
            'Active Inference': ActiveInferenceAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            ),
            'PPO': PPOAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            ),
            'DQN': CustomDQNAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
        }
        
        # Results storage
        self.results = {name: [] for name in self.agents.keys()}
        
    def run_experiment(self) -> Dict[str, List[Dict]]:
        """
        Run the comparison experiment.
        
        Returns:
            Dictionary containing results for each agent
        """
        print("Starting comparison experiment...")
        print(f"Episodes per agent: {self.num_episodes}")
        print(f"Max steps per episode: {self.max_steps}")
        
        for agent_name, agent in self.agents.items():
            print(f"\n=== Training {agent_name} ===")
            
            agent_results = self._train_agent(agent, agent_name)
            self.results[agent_name] = agent_results
            
            # Save agent model
            model_path = f"data/models/{agent_name.lower().replace(' ', '_')}.pth"
            agent.save_models(model_path)
            print(f"Model saved to {model_path}")
        
        return self.results
    
    def _train_agent(self, agent, agent_name: str) -> List[Dict]:
        """
        Train a single agent and collect results.
        
        Args:
            agent: Agent to train
            agent_name: Name of the agent
            
        Returns:
            List of episode results
        """
        results = []
        
        for episode in tqdm(range(self.num_episodes), desc=f"Training {agent_name}"):
            # === Curriculum learning: adjust difficulty ===
            if episode < 100:
                self.env.config['obstacles']['num_static'] = 2
                self.env.config['obstacles']['num_dynamic'] = 0
                self.env.config['goals']['goal_radius'] = 5.0
            elif episode < 200:
                self.env.config['obstacles']['num_static'] = 4
                self.env.config['obstacles']['num_dynamic'] = 1
                self.env.config['goals']['goal_radius'] = 4.0
            elif episode < 300:
                self.env.config['obstacles']['num_static'] = 6
                self.env.config['obstacles']['num_dynamic'] = 2
                self.env.config['goals']['goal_radius'] = 3.0
            elif episode < 400:
                self.env.config['obstacles']['num_static'] = 8
                self.env.config['obstacles']['num_dynamic'] = 3
                self.env.config['goals']['goal_radius'] = 2.0
            else:
                self.env.config['obstacles']['num_static'] = 10
                self.env.config['obstacles']['num_dynamic'] = 4
                self.env.config['goals']['goal_radius'] = 1.5
            print(f"\n{agent_name} Episode {episode + 1}/{self.num_episodes} (Static: {self.env.config['obstacles']['num_static']}, Dynamic: {self.env.config['obstacles']['num_dynamic']}, Goal radius: {self.env.config['goals']['goal_radius']})")
            
            # Reset environment
            observation = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            episode_collision = False
            trajectory = []
            
            # Run episode
            for step in range(self.max_steps):
                trajectory.append(self.env.robot.position.copy())
                action = agent.select_action(observation, training=True)
                next_observation, reward, done, info = self.env.step(action)
                agent.step(observation, reward, done, info)
                episode_reward += reward
                episode_length += 1
                
                # Check episode end conditions
                if done:
                    episode_success = info.get('goal_reached', False)
                    episode_collision = info.get('collision', False)
                    break
                
                observation = next_observation
            
            # Store episode results
            episode_result = {
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'success': episode_success,
                'collision': episode_collision,
                'step_count': agent.step_count,
                'exploration_rate': getattr(agent, 'exploration_rate', None)
            }
            
            # Add agent-specific metrics
            if hasattr(agent, 'get_training_stats'):
                stats = agent.get_training_stats()
                episode_result.update(stats)
            
            results.append(episode_result)
            
            # Save trajectory for this episode
            np.save(f"data/experiments/{agent_name.lower().replace(' ', '_')}_trajectory_episode_{episode+1}.npy", np.array(trajectory))
            
            # Print progress
            if (episode + 1) % 10 == 0:
                recent_rewards = [r['reward'] for r in results[-10:]]
                recent_successes = [r['success'] for r in results[-10:]]
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes)
                print(f"  Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Success Rate = {success_rate:.2f}")
        
            # Save all metrics to CSV for this agent
            csv_path = f"data/experiments/{agent_name.lower().replace(' ', '_')}_metrics.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = list(results[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for m in results:
                    writer.writerow(m)
            print(f"Metrics saved to {csv_path}")
        
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze the experimental results.
        
        Returns:
            Dictionary containing analysis results
        """
        print("\n=== Analyzing Results ===")
        
        analysis = {}
        
        for agent_name, results in self.results.items():
            print(f"\n{agent_name} Analysis:")
            
            # Extract metrics
            rewards = [r['reward'] for r in results]
            lengths = [r['length'] for r in results]
            successes = [r['success'] for r in results]
            collisions = [r['collision'] for r in results]
            
            # Compute statistics
            agent_analysis = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'success_rate': np.mean(successes),
                'collision_rate': np.mean(collisions),
                'total_episodes': len(results),
                'successful_episodes': sum(successes),
                'collision_episodes': sum(collisions)
            }
            
            # Learning curve analysis
            if len(rewards) > 1:
                # Split into early and late performance
                mid_point = len(rewards) // 2
                early_rewards = rewards[:mid_point]
                late_rewards = rewards[mid_point:]
                
                agent_analysis.update({
                    'early_mean_reward': np.mean(early_rewards),
                    'late_mean_reward': np.mean(late_rewards),
                    'improvement': np.mean(late_rewards) - np.mean(early_rewards)
                })
            
            analysis[agent_name] = agent_analysis
            
            # Print summary
            print(f"  Success Rate: {agent_analysis['success_rate']:.3f}")
            print(f"  Mean Reward: {agent_analysis['mean_reward']:.2f} ± {agent_analysis['std_reward']:.2f}")
            print(f"  Mean Length: {agent_analysis['mean_length']:.1f} ± {agent_analysis['std_length']:.1f}")
            print(f"  Collision Rate: {agent_analysis['collision_rate']:.3f}")
            if 'improvement' in agent_analysis:
                print(f"  Improvement: {agent_analysis['improvement']:.2f}")
        
        return analysis
    
    def plot_comparison(self, save_path: str = "data/experiments/comparison_results.png"):
        """
        Plot comparison results.
        
        Args:
            save_path: Path to save the comparison plot
        """
        print(f"\nGenerating comparison plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Colors for different agents
        colors = ['blue', 'red', 'green']
        agent_names = list(self.results.keys())
        
        # 1. Learning curves (rewards)
        for i, agent_name in enumerate(agent_names):
            rewards = [r['reward'] for r in self.results[agent_name]]
            axes[0, 0].plot(rewards, color=colors[i], alpha=0.7, label=agent_name)
        
        axes[0, 0].set_title('Learning Curves (Rewards)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Success rates over time
        for i, agent_name in enumerate(agent_names):
            successes = [r['success'] for r in self.results[agent_name]]
            # Compute running success rate
            running_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
            axes[0, 1].plot(running_success, color=colors[i], alpha=0.7, label=agent_name)
        
        axes[0, 1].set_title('Success Rate Over Time')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode lengths
        for i, agent_name in enumerate(agent_names):
            lengths = [r['length'] for r in self.results[agent_name]]
            axes[0, 2].plot(lengths, color=colors[i], alpha=0.7, label=agent_name)
        
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot of rewards
        reward_data = []
        labels = []
        for agent_name in agent_names:
            rewards = [r['reward'] for r in self.results[agent_name]]
            reward_data.append(rewards)
            labels.extend([agent_name] * len(rewards))
        
        axes[1, 0].boxplot(reward_data, labels=agent_names)
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Success rate comparison
        success_rates = []
        for agent_name in agent_names:
            successes = [r['success'] for r in self.results[agent_name]]
            success_rates.append(np.mean(successes))
        
        bars = axes[1, 1].bar(agent_names, success_rates, color=colors[:len(agent_names)])
        axes[1, 1].set_title('Overall Success Rate')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.3f}', ha='center', va='bottom')
        
        # 6. Collision rate comparison
        collision_rates = []
        for agent_name in agent_names:
            collisions = [r['collision'] for r in self.results[agent_name]]
            collision_rates.append(np.mean(collisions))
        
        bars = axes[1, 2].bar(agent_names, collision_rates, color=colors[:len(agent_names)])
        axes[1, 2].set_title('Collision Rate')
        axes[1, 2].set_ylabel('Collision Rate')
        axes[1, 2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, collision_rates):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plots saved to {save_path}")
    
    def generate_report(self, analysis: Dict[str, Any], 
                       save_path: str = "data/experiments/comparison_report.txt"):
        """
        Generate a detailed comparison report.
        
        Args:
            analysis: Analysis results
            save_path: Path to save the report
        """
        print(f"\nGenerating detailed report...")
        
        with open(save_path, 'w') as f:
            f.write("ACTIVE INFERENCE vs REINFORCEMENT LEARNING COMPARISON REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Experiment Configuration:\n")
            f.write(f"- Episodes per agent: {self.num_episodes}\n")
            f.write(f"- Max steps per episode: {self.max_steps}\n")
            f.write(f"- Environment: Navigation with dynamic obstacles\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 20 + "\n\n")
            
            # Create summary table
            f.write(f"{'Agent':<20} {'Success Rate':<12} {'Mean Reward':<12} {'Mean Length':<12} {'Collision Rate':<12}\n")
            f.write("-" * 80 + "\n")
            
            for agent_name, agent_analysis in analysis.items():
                f.write(f"{agent_name:<20} {agent_analysis['success_rate']:<12.3f} "
                       f"{agent_analysis['mean_reward']:<12.2f} {agent_analysis['mean_length']:<12.1f} "
                       f"{agent_analysis['collision_rate']:<12.3f}\n")
            
            f.write("\nDETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n\n")
            
            for agent_name, agent_analysis in analysis.items():
                f.write(f"{agent_name.upper()}:\n")
                f.write(f"  Success Rate: {agent_analysis['success_rate']:.3f} "
                       f"({agent_analysis['successful_episodes']}/{agent_analysis['total_episodes']})\n")
                f.write(f"  Mean Reward: {agent_analysis['mean_reward']:.2f} ± {agent_analysis['std_reward']:.2f}\n")
                f.write(f"  Mean Length: {agent_analysis['mean_length']:.1f} ± {agent_analysis['std_length']:.1f}\n")
                f.write(f"  Collision Rate: {agent_analysis['collision_rate']:.3f}\n")
                
                if 'improvement' in agent_analysis:
                    f.write(f"  Learning Improvement: {agent_analysis['improvement']:.2f}\n")
                
                f.write("\n")
            
            # Statistical significance test
            f.write("STATISTICAL ANALYSIS\n")
            f.write("-" * 20 + "\n\n")
            
            # Compare success rates
            success_rates = [analysis[name]['success_rate'] for name in analysis.keys()]
            best_agent = max(analysis.keys(), key=lambda x: analysis[x]['success_rate'])
            f.write(f"Best performing agent: {best_agent} (Success Rate: {analysis[best_agent]['success_rate']:.3f})\n\n")
            
            f.write("CONCLUSIONS\n")
            f.write("-" * 20 + "\n\n")
            
            # Generate conclusions based on results
            ai_success = analysis.get('Active Inference', {}).get('success_rate', 0)
            ppo_success = analysis.get('PPO', {}).get('success_rate', 0)
            dqn_success = analysis.get('DQN', {}).get('success_rate', 0)
            
            if ai_success > max(ppo_success, dqn_success):
                f.write("Active Inference demonstrates superior performance compared to traditional RL methods.\n")
                f.write("This suggests that the probabilistic world model and belief updating mechanism\n")
                f.write("provide advantages in dynamic navigation tasks.\n\n")
            elif ai_success < min(ppo_success, dqn_success):
                f.write("Traditional RL methods outperform Active Inference in this task.\n")
                f.write("This may indicate that the current implementation requires further optimization\n")
                f.write("or that the task is better suited to direct policy optimization.\n\n")
            else:
                f.write("Active Inference performs competitively with traditional RL methods.\n")
                f.write("This demonstrates the viability of active inference as an alternative\n")
                f.write("approach to robot navigation.\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Further investigate the belief state dynamics and their relationship to performance\n")
            f.write("2. Experiment with different generative model architectures\n")
            f.write("3. Analyze the trade-off between exploration and exploitation in active inference\n")
            f.write("4. Test on more complex environments with varying dynamics\n")
            f.write("5. Compare computational efficiency between methods\n")
        
        print(f"Detailed report saved to {save_path}")
    
    def save_results(self, save_path: str = "data/experiments/comparison_results.csv"):
        """
        Save results to CSV file.
        
        Args:
            save_path: Path to save the CSV file
        """
        print(f"\nSaving results to CSV...")
        
        # Prepare data for CSV
        csv_data = []
        for agent_name, results in self.results.items():
            for result in results:
                row = {'agent': agent_name}
                row.update(result)
                csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(save_path, index=False)
        
        print(f"Results saved to {save_path}")


def main():
    """Run the comparison experiment."""
    # Create data directories
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/experiments", exist_ok=True)
    
    # Run experiment
    experiment = ComparisonExperiment(num_episodes=500, max_steps=1000)
    results = experiment.run_experiment()
    
    # Analyze results
    analysis = experiment.analyze_results()
    
    # Generate visualizations and reports
    experiment.plot_comparison()
    experiment.generate_report(analysis)
    experiment.save_results()
    
    print("\n=== Comparison Experiment Completed ===")
    print("Check the following files for results:")
    print("- data/experiments/comparison_results.png (plots)")
    print("- data/experiments/comparison_report.txt (detailed report)")
    print("- data/experiments/comparison_results.csv (raw data)")
    print("- data/models/ (trained models)")


if __name__ == "__main__":
    main() 