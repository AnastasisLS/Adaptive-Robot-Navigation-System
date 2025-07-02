#!/usr/bin/env python3
"""
Headless experiment runner for cloud deployment.
This version runs without GUI requirements and saves results to files.
"""

import os
import sys
import time
import json
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import NavigationEnvironment
from src.agents import ActiveInferenceAgent
from src.evaluation import AgentComparison


def run_headless_basic_navigation(num_episodes=500, max_steps=1000):
    """Run basic navigation experiment in headless mode."""
    print(f"Starting headless basic navigation experiment: {num_episodes} episodes")
    
    # Create environment and agent
    env = NavigationEnvironment()
    agent = ActiveInferenceAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    episode_collisions = []
    metrics = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        observation = env.reset()
        episode_reward = 0
        episode_length = 0
        collisions = 0
        
        # Get curriculum info
        curriculum_info = env.get_curriculum_info()
        
        # Run episode
        for step in range(max_steps):
            # Select action
            action = agent.select_action(observation, training=True)
            
            # Take step in environment
            next_observation, reward, done, info = env.step(action)
            
            # Update agent
            agent.step(observation, reward, done, info)
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            
            if info.get('collision', False):
                collisions += 1
            
            # Check if episode is done
            if done:
                if info.get('goal_reached', False):
                    success_count += 1
                break
            
            observation = next_observation
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_collisions.append(collisions)
        
        episode_success = int(info.get('goal_reached', False))
        metrics.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'length': episode_length,
            'collisions': collisions,
            'success': episode_success,
            'curriculum_info': curriculum_info
        })
        
        # Print progress every 50 episodes
        if (episode + 1) % 50 == 0:
            elapsed = time.time() - start_time
            success_rate = 100 * success_count / (episode + 1)
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Success: {success_rate:.1f}% - "
                  f"Avg Reward: {avg_reward:.2f} - "
                  f"Time: {elapsed:.1f}s")
    
    # Save results
    results = {
        'experiment_type': 'basic_navigation',
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'total_time': time.time() - start_time,
        'final_success_rate': 100 * success_count / num_episodes,
        'average_reward': np.mean(episode_rewards),
        'average_length': np.mean(episode_lengths),
        'average_collisions': np.mean(episode_collisions),
        'episode_metrics': metrics,
        'training_stats': agent.get_training_stats()
    }
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/experiments/headless_basic_navigation_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create training curves plot
    create_training_curves(episode_rewards, episode_lengths, results_file.replace('.json', '_plot.png'))
    
    print(f"Experiment completed! Results saved to {results_file}")
    print(f"Final Success Rate: {results['final_success_rate']:.1f}%")
    print(f"Average Reward: {results['average_reward']:.2f}")
    
    return results


def run_headless_comparison(num_episodes=100, max_steps=1000):
    """Run comparison experiment in headless mode."""
    print(f"Starting headless comparison experiment: {num_episodes} episodes")
    
    # Create evaluator
    evaluator = AgentComparison()
    
    # Run comparison
    results = evaluator.run_comparison(
        num_episodes=num_episodes,
        max_steps=max_steps,
        headless=True
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/experiments/headless_comparison_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Comparison completed! Results saved to {results_file}")
    
    return results


def create_training_curves(rewards, lengths, save_path):
    """Create training curves plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reward curve
    ax1.plot(rewards, alpha=0.6)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Length curve
    ax2.plot(lengths, alpha=0.6)
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run headless experiments')
    parser.add_argument('--experiment', choices=['basic', 'comparison'], 
                       default='basic', help='Experiment type')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=1000, 
                       help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    if args.experiment == 'basic':
        run_headless_basic_navigation(args.episodes, args.max_steps)
    else:
        run_headless_comparison(args.episodes, args.max_steps) 